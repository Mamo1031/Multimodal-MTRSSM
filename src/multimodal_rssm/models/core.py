"""Base RSSM core module for RSSM variants."""

from abc import ABC, abstractmethod

from distribution_extension import kl_divergence
from lightning import LightningModule
from torch import Tensor, nn

from multimodal_rssm.models.networks import Representation, Transition
from multimodal_rssm.models.state import State, stack_states


class BaseRSSM(LightningModule, ABC):
    """Base RSSM module for RSSM variants."""

    def __init__(
        self,
        *,
        representation: Representation,
        transition: Transition,
        init_proj: nn.Module,
        kl_coeff: float,
        use_kl_balancing: bool,
    ) -> None:
        """Initialize BaseRSSM."""
        super().__init__()
        self.representation = representation
        self.transition = transition
        self.init_proj = init_proj
        self.kl_coeff = kl_coeff
        self.use_kl_balancing = use_kl_balancing

    @abstractmethod
    def encode_observation(self, observation: Tensor | tuple[Tensor, ...]) -> Tensor:
        """Encode observation(s) into embedding.

        Args:
            observation: Observation(s) to encode.
                For single modality: Tensor
                For multimodal: tuple[Tensor, Tensor]

        Returns
        -------
        Tensor: Observation embedding.
        """

    @abstractmethod
    def decode_state(self, state: State) -> dict[str, Tensor]:
        """Decode state into reconstruction(s).

        Args:
            state: State to decode.

        Returns
        -------
        dict[str, Tensor]: Dictionary of reconstructions.
            For single modality: {"recon": Tensor}
            For multimodal: {"recon/audio": Tensor, "recon/vision": Tensor}
        """

    @abstractmethod
    def compute_reconstruction_loss(
        self,
        reconstructions: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute reconstruction loss(es).

        Args:
            reconstructions: Dictionary of reconstructions.
            targets: Dictionary of targets.

        Returns
        -------
        dict[str, Tensor]: Dictionary of losses.
            Must include "recon" key.
        """

    @abstractmethod
    def get_observations_from_batch(self, batch: tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...]:
        """Extract observation(s) from batch (full sequence).

        Args:
            batch: Batch tuple.

        Returns
        -------
        tuple: Observation(s) sequence.
            For single modality: Tensor (shape: [B, T, ...])
            For multimodal: tuple[Tensor, Tensor] (each shape: [B, T, ...])
        """

    @abstractmethod
    def get_initial_observation(self, observations: Tensor | tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...]:
        """Extract initial observation(s) from observation sequence.

        Args:
            observations: Observation(s) sequence.

        Returns
        -------
        tuple: Initial observation(s).
            For single modality: Tensor (shape: [B, ...])
            For multimodal: tuple[Tensor, Tensor] (each shape: [B, ...])
        """

    @abstractmethod
    def get_targets_from_batch(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Extract target(s) from batch.

        Args:
            batch: Batch tuple.

        Returns
        -------
        dict[str, Tensor]: Dictionary of targets.
            For single modality: {"recon": Tensor}
            For multimodal: {"recon/audio": Tensor, "recon/vision": Tensor}
        """

    def initial_state(self, observation: Tensor | tuple[Tensor, Tensor]) -> State:
        """Initialize the latent state.

        Args:
            observation: Observation(s) to initialize the latent state.

        Returns
        -------
        State: The initial latent state.
        """
        obs_embed = self.encode_observation(observation)
        deter = self.init_proj(obs_embed)
        stoch = self.transition.rnn_to_prior_projector(deter)
        distribution = self.representation.distribution_factory(stoch)
        return State(deter=deter, distribution=distribution).to(self.device)

    def rollout_representation(
        self,
        *,
        actions: Tensor,
        observations: Tensor | tuple[Tensor, ...],
        prev_state: State,
    ) -> tuple[State, State]:
        """Rollout the representation.

        Args:
            actions: The actions to rollout the representation.
            observations: The observations to rollout the representation.
            prev_state: The previous state to rollout the representation.

        Returns
        -------
        tuple[State, State]
            (posterior, prior) states.
        """
        obs_embed = self.encode_observation(observations)
        priors: list[State] = []
        posteriors: list[State] = []
        for t in range(obs_embed.shape[1]):
            prior = self.transition.forward(actions[:, t], prev_state)
            posterior = self.representation.forward(obs_embed[:, t], prior)
            priors.append(prior)
            posteriors.append(posterior)
            prev_state = posterior

        prior = stack_states(priors, dim=1)
        posterior = stack_states(posteriors, dim=1)
        return posterior, prior

    def rollout_transition(self, *, actions: Tensor, prev_state: State) -> State:
        """Rollout the transition model.

        Args:
            actions: The actions to rollout the transition model.
            prev_state: The previous state to rollout the transition model.

        Returns
        -------
        State: The prior states.
        """
        priors: list[State] = []
        for t in range(actions.shape[1]):
            prev_state = self.transition.forward(actions[:, t], prev_state)
            priors.append(prev_state)
        return stack_states(priors, dim=1)

    def shared_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Perform a shared training/validation step.

        Args:
            batch: The batch to shared step.

        Returns
        -------
        dict[str, Tensor]: The loss dictionary.
        """
        action_input = batch[0]  # action_input is always first
        observations = self.get_observations_from_batch(batch)
        initial_observation = self.get_initial_observation(observations)

        posterior, prior = self.rollout_representation(
            actions=action_input,
            observations=observations,
            prev_state=self.initial_state(initial_observation),  # type: ignore[arg-type]
        )

        reconstructions = self.decode_state(posterior)
        targets = self.get_targets_from_batch(batch)

        loss_dict = self.compute_reconstruction_loss(reconstructions, targets)

        kl_div = kl_divergence(
            q=posterior.distribution.independent(1),
            p=prior.distribution.independent(1),
            use_balancing=self.use_kl_balancing,
        ).mul(self.kl_coeff)

        loss_dict["kl"] = kl_div
        loss_dict["loss"] = loss_dict["recon"] + kl_div

        return loss_dict

    def training_step(self, batch: tuple[Tensor, ...], _: int) -> dict[str, Tensor]:
        """Train the model on a batch.

        Args:
            batch: The batch to train the model.
            _: The batch index.

        Returns
        -------
        dict[str, Tensor]: The loss dictionary.
        """
        loss_dict = self.shared_step(batch)
        renamed_dict = {
            "loss": loss_dict["loss"],  # Required by Lightning for automatic optimization
            "train/loss": loss_dict["loss"],
        }
        # Add all keys with train/ prefix
        for key, value in loss_dict.items():
            if key != "loss":
                renamed_dict[f"train/{key}"] = value
        self.log_dict(renamed_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return renamed_dict

    def validation_step(self, batch: tuple[Tensor, ...], _batch_index: int) -> dict[str, Tensor]:
        """Validate the model on a batch.

        Args:
            batch: The batch to validate the model.
            _batch_index: The batch index.

        Returns
        -------
        dict[str, Tensor]: The loss dictionary.
        """
        loss_dict = self.shared_step(batch)
        renamed_dict = {
            "val/loss": loss_dict["loss"],
        }
        # Add all keys with val/ prefix
        for key, value in loss_dict.items():
            if key != "loss":
                renamed_dict[f"val/{key}"] = value
        self.log_dict(renamed_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return renamed_dict
