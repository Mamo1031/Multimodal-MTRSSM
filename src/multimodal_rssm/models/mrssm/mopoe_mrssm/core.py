"""MoPoE-MRSSM module for multimodal recurrent state-space model."""

import torch
from torch import Tensor, nn

from multimodal_rssm.models.core import BaseRSSM
from multimodal_rssm.models.networks import Representation, Transition
from multimodal_rssm.models.objective import likelihood
from multimodal_rssm.models.state import State, stack_states


class MoPoE_MRSSM(BaseRSSM):  # noqa: N801
    """Multimodal Recurrent State-Space Model with MoPoE-style posteriors (MoPoE-MRSSM).

    This model combines Product of Experts (PoE) and Mixture of Experts (MoE):
    1. Fused posterior is created by PoE fusion of audio and vision posteriors
    2. Mixed posterior is created by MoE fusion of {audio, vision, fused} posteriors
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        audio_representation: Representation,
        vision_representation: Representation,
        transition: Transition,
        audio_encoder: nn.Module,
        vision_encoder: nn.Module,
        audio_decoder: nn.Module,
        vision_decoder: nn.Module,
        init_proj: nn.Module,
        kl_coeff: float,
        use_kl_balancing: bool,
    ) -> None:
        """Initialize MoPoE-MRSSM.

        Args:
            audio_representation: Representation network for audio modality.
            vision_representation: Representation network for vision modality.
            transition: Transition network (RSSM prior).
            audio_encoder: Audio encoder network.
            vision_encoder: Vision encoder network.
            audio_decoder: Audio decoder network.
            vision_decoder: Vision decoder network.
            init_proj: Initial projection network.
            kl_coeff: KL divergence coefficient.
            use_kl_balancing: Whether to use KL balancing.
        """
        super().__init__(
            representation=audio_representation,
            transition=transition,
            init_proj=init_proj,
            kl_coeff=kl_coeff,
            use_kl_balancing=use_kl_balancing,
        )
        self.audio_representation = audio_representation
        self.vision_representation = vision_representation
        self.audio_encoder = audio_encoder
        self.vision_encoder = vision_encoder
        self.audio_decoder = audio_decoder
        self.vision_decoder = vision_decoder

    @staticmethod
    def _compute_posterior_with_logits(
        obs_embed: Tensor,
        prior_state: State,
        representation: Representation,
    ) -> tuple[State, Tensor]:
        """Compute posterior state and return logits for fusion.

        Args:
            obs_embed: Observation embedding. Shape: [*B, obs_embed_size]
            prior_state: Prior state.
            representation: Representation network to use.

        Returns
        -------
        tuple[State, Tensor]
            (posterior_state, logits) where logits shape is [*B, class_size * category_size]
        """
        projector_input = torch.cat([prior_state.deter, obs_embed], -1)
        stoch_source = representation.rnn_to_post_projector(projector_input)
        distribution = representation.distribution_factory.forward(stoch_source)
        posterior = State(deter=prior_state.deter, distribution=distribution)
        return posterior, stoch_source

    def _poe_fusion_categorical(self, audio_logits: Tensor, vision_logits: Tensor, prior_state: State) -> State:
        """Fuse categorical distributions using Product of Experts (PoE).

        PoE for categorical distributions is computed by summing log-probabilities.

        Args:
            audio_logits: Audio posterior logits. Shape: [*B, class_size * category_size]
            vision_logits: Vision posterior logits. Shape: [*B, class_size * category_size]
            prior_state: Prior state (for deterministic part).

        Returns
        -------
        State: Fused posterior state.
        """
        # Convert each expert's logits to log-probabilities
        audio_log_probs = torch.nn.functional.log_softmax(audio_logits, dim=-1)
        vision_log_probs = torch.nn.functional.log_softmax(vision_logits, dim=-1)

        # Sum log-probabilities: log p_fused ∝ log p_audio + log p_vision
        fused_log_probs = audio_log_probs + vision_log_probs

        # Create categorical distribution
        fused_distribution = self.audio_representation.distribution_factory.forward(fused_log_probs)

        return State(deter=prior_state.deter, distribution=fused_distribution)

    def _moe_fusion_categorical(
        self,
        audio_logits: Tensor,
        vision_logits: Tensor,
        fused_log_probs: Tensor,
        prior_state: State,
    ) -> State:
        """Fuse categorical distributions using Mixture of Experts (MoE).

        MoE fusion is performed by weighted averaging of probability distributions
        from all experts {A, V, A+V} with equal weights (1/3 each).

        Args:
            audio_logits: Audio posterior logits. Shape: [*B, class_size * category_size]
            vision_logits: Vision posterior logits. Shape: [*B, class_size * category_size]
            fused_log_probs: Fused (A+V) posterior log-probabilities from PoE.
                Shape: [*B, class_size * category_size]
            prior_state: Prior state (for deterministic part).

        Returns
        -------
        State: Mixed posterior state.
        """
        # Convert each expert's logits to log-probabilities (for numerical stability)
        audio_log_probs = torch.nn.functional.log_softmax(audio_logits, dim=-1)
        vision_log_probs = torch.nn.functional.log_softmax(vision_logits, dim=-1)

        # Weighted average in log-probability space for numerical stability:
        weight = 1.0 / 3.0
        log_weight = torch.log(torch.tensor(weight, device=audio_log_probs.device, dtype=audio_log_probs.dtype))

        # Stack log-probabilities with log-weight: [*B, 3, class_size * category_size]
        weighted_log_probs = torch.stack(
            [
                log_weight + audio_log_probs,
                log_weight + vision_log_probs,
                log_weight + fused_log_probs,
            ],
            dim=-2,
        )  # [*B, 3, class_size * category_size]

        # Log-sum-exp for numerical stability: log(sum(exp(x))) = logsumexp(x)
        mixed_log_probs = torch.logsumexp(weighted_log_probs, dim=-2)  # [*B, class_size * category_size]

        # Ensure mixed_log_probs is 2D: [B, class_size * category_size]
        min_dim_for_2d = 2
        if mixed_log_probs.dim() > min_dim_for_2d:
            mixed_log_probs = mixed_log_probs.view(mixed_log_probs.shape[0], -1)

        mixed_distribution = self.audio_representation.distribution_factory.forward(mixed_log_probs)

        return State(deter=prior_state.deter, distribution=mixed_distribution)

    def encode_observation(self, observation: tuple[Tensor, Tensor] | Tensor) -> Tensor:  # type: ignore[override]
        """Encode observation(s) into fused embedding.

        Args:
            observation: Observation(s) to encode.
                For tuple: (audio_obs, vision_obs) each shape: [*B, T, ...] or [*B, ...]
                For Tensor: single observation (for backward compatibility)

        Returns
        -------
        Tensor: Observation embedding. Shape: [*B, T, obs_embed_size] or [*B, obs_embed_size]
        """
        if isinstance(observation, tuple):
            audio_obs, vision_obs = observation
            audio_embed: Tensor = self.audio_encoder(audio_obs)  # type: ignore[no-any-return]
            vision_embed: Tensor = self.vision_encoder(vision_obs)  # type: ignore[no-any-return]
            return (audio_embed + vision_embed) / 2.0
        return observation

    def rollout_representation(  # noqa: PLR0914
        self,
        *,
        actions: Tensor,
        observations: Tensor | tuple[Tensor, ...],
        prev_state: State,
    ) -> tuple[State, State]:
        """Rollout representation and compute posteriors with PoE and MoE fusion.

        Args:
            actions: The actions to rollout the representation. Shape: [B, T, action_size]
            observations: The observations to rollout the representation.
                For tuple: (audio_obs, vision_obs) each shape: [B, T, ...]
            prev_state: The previous state to rollout the representation.

        Returns
        -------
        tuple[State, State]
            (mixed posterior, prior) states.

        Raises
        ------
        TypeError
            If observations is not a tuple.
        """
        if not isinstance(observations, tuple):
            msg = "MoPoE-MRSSM requires tuple of (audio_obs, vision_obs)"
            raise TypeError(msg)

        audio_obs, vision_obs = observations

        audio_embed: Tensor = self.audio_encoder(audio_obs)  # type: ignore[no-any-return]
        vision_embed: Tensor = self.vision_encoder(vision_obs)  # type: ignore[no-any-return]

        priors: list[State] = []
        mixed_posteriors: list[State] = []

        for t in range(audio_embed.shape[1]):
            prior = self.transition.forward(actions[:, t], prev_state)

            # Audio-only posterior q_A (with logits for fusion)
            audio_obs_embed = audio_embed[:, t]
            _, audio_logits = self._compute_posterior_with_logits(
                audio_obs_embed,
                prior,
                self.audio_representation,
            )

            # Vision-only posterior q_V (with logits for fusion)
            vision_obs_embed = vision_embed[:, t]
            _, vision_logits = self._compute_posterior_with_logits(
                vision_obs_embed,
                prior,
                self.vision_representation,
            )

            # Get fused log-probabilities for MoE fusion (PoE: sum of log-probabilities)
            audio_log_probs = torch.nn.functional.log_softmax(audio_logits, dim=-1)
            vision_log_probs = torch.nn.functional.log_softmax(vision_logits, dim=-1)
            fused_log_probs = audio_log_probs + vision_log_probs

            # Create mixed posterior by MoE fusion of {A, V, A+V}
            mixed_posterior = self._moe_fusion_categorical(
                audio_logits,
                vision_logits,
                fused_log_probs,
                prior,
            )

            priors.append(prior)
            mixed_posteriors.append(mixed_posterior)

            prev_state = mixed_posterior

        prior = stack_states(priors, dim=1)
        posterior_mixed = stack_states(mixed_posteriors, dim=1)
        return posterior_mixed, prior

    def decode_state(self, state: State) -> dict[str, Tensor]:
        """Decode mixed state into reconstructions for each modality.

        Args:
            state: The state to decode.

        Returns
        -------
        dict[str, Tensor]: Dictionary with reconstructions for "recon/audio" and "recon/vision".
        """
        audio_recon = self.audio_decoder(state.feature)
        vision_recon = self.vision_decoder(state.feature)
        return {
            "recon/audio": audio_recon,
            "recon/vision": vision_recon,
        }

    @staticmethod
    def compute_reconstruction_loss(
        reconstructions: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute reconstruction losses.

        Args:
            reconstructions: Dictionary with "recon/audio" and "recon/vision" keys.
            targets: Dictionary with "recon/audio" and "recon/vision" keys.

        Returns
        -------
        dict[str, Tensor]: Dictionary with "recon", "recon/audio", and "recon/vision" keys.
        """
        audio_recon_loss = likelihood(
            prediction=reconstructions["recon/audio"],
            target=targets["recon/audio"],
            event_ndims=3,
        )
        vision_recon_loss = likelihood(
            prediction=reconstructions["recon/vision"],
            target=targets["recon/vision"],
            event_ndims=3,
        )
        return {
            "recon": audio_recon_loss + vision_recon_loss,
            "recon/audio": audio_recon_loss,
            "recon/vision": vision_recon_loss,
        }

    @staticmethod
    def get_observations_from_batch(batch: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        """Extract observation sequences from batch.

        Args:
            batch: Batch tuple.
                Format: (action_input, audio_obs_input, vision_obs_input,
                        action_target, audio_obs_target, vision_obs_target)

        Returns
        -------
        tuple[Tensor, Tensor]: (audio_obs_input, vision_obs_input) each shape: [B, T, ...]
        """
        return batch[1], batch[2]

    @staticmethod
    def get_initial_observation(observations: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Extract initial observations from observation sequences.

        Args:
            observations: Tuple of (audio_obs, vision_obs) each shape: [B, T, ...]

        Returns
        -------
        tuple[Tensor, Tensor]: Initial observations each shape: [B, ...]
        """
        audio_obs, vision_obs = observations
        return audio_obs[:, 0], vision_obs[:, 0]

    @staticmethod
    def get_targets_from_batch(batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Extract reconstruction targets from batch.

        Args:
            batch: Batch tuple.
                Format: (action_input, audio_obs_input, vision_obs_input,
                        action_target, audio_obs_target, vision_obs_target)

        Returns
        -------
        dict[str, Tensor]: Dictionary with "recon/audio" and "recon/vision" keys.
        """
        return {
            "recon/audio": batch[4],
            "recon/vision": batch[5],
        }
