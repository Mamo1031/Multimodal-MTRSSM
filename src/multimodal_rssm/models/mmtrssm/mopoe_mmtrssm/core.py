"""MoPoE-MMTRSSM module for multimodal temporal recurrent state-space model."""

import torch
from distribution_extension import Distribution, MultiOneHotFactory, kl_divergence
from torch import Tensor, nn

from multimodal_rssm.models.mmtrssm.state import MTState, stack_mtstates
from multimodal_rssm.models.mrssm.mopoe_mrssm.core import MoPoE_MRSSM
from multimodal_rssm.models.networks import Representation


class MTRNN(nn.Module):
    """Multi-Timescale RNN cell."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = True,
        tau: float = 2.0,
    ) -> None:
        """Initialize MTRNN.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
            bias: Whether to use bias.
            tau: Time constant (must be > 1.0).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.tau = tau
        assert tau > 1.0, "tau must be greater than 1.0"

        self._d2h = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self._input2h = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.hidden: Tensor | None = None

    def _compute_mtrnn(self, input: Tensor, prev_d: Tensor) -> Tensor:
        """Compute MTRNN update.

        Args:
            input: Input tensor.
            prev_d: Previous deterministic state.

        Returns
        -------
        Tensor: Updated deterministic state.
        """
        if self.hidden is None:
            self.hidden = torch.zeros(
                input.shape[0],
                self.hidden_dim,
                device=input.device,
                dtype=input.dtype,
            )

        self.hidden = (1 - 1 / self.tau) * self.hidden + (self._d2h(prev_d) + self._input2h(input)) / self.tau
        d = torch.tanh(self.hidden)
        return d

    def forward(self, inputs: Tensor, prev_d: Tensor) -> Tensor:
        """Forward pass of MTRNN.

        Args:
            inputs: Input tensor.
            prev_d: Previous deterministic state.

        Returns
        -------
        Tensor: Updated deterministic state.
        """
        return self._compute_mtrnn(inputs, prev_d)


class MoPoE_MMTRSSM(MoPoE_MRSSM):  # noqa: N801
    """Multimodal Temporal Recurrent State-Space Model with MoPoE (MoPoE-MMTRSSM).

    This model extends MoPoE-MRSSM with MTRSSM's hierarchical structure:
    - 2-layer RNN: l_rnn (lower) and h_rnn (higher)
    - Hierarchical stochastic states: l_stoch (lower) and h_stoch (higher)
    - MoPoE fusion is applied to the lower layer, and higher layer depends on lower layer states.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        audio_representation: Representation,
        vision_representation: Representation,
        audio_encoder: nn.Module,
        vision_encoder: nn.Module,
        audio_decoder: nn.Module,
        vision_decoder: nn.Module,
        init_proj: nn.Module,
        kl_coeff: float,
        use_kl_balancing: bool,
        # MTRSSM specific parameters
        action_size: int,
        hd_dim: int,
        hs_dim: int,
        ld_dim: int,
        ls_dim: int,
        l_tau: float,
        h_tau: float,
        l_prior: nn.Module,
        l_posterior: nn.Module,
        h_prior: nn.Module,
        h_posterior: nn.Module,
        l_dist: MultiOneHotFactory,
        h_dist: MultiOneHotFactory,
        w_kl_h: float = 1.0,
    ) -> None:
        """Initialize MoPoE-MMTRSSM.

        Args:
            audio_representation: Representation network for audio modality.
            vision_representation: Representation network for vision modality.
            audio_encoder: Audio encoder network.
            vision_encoder: Vision encoder network.
            audio_decoder: Audio decoder network.
            vision_decoder: Vision decoder network.
            init_proj: Initial projection network.
            kl_coeff: KL divergence coefficient.
            use_kl_balancing: Whether to use KL balancing.
            hd_dim: Higher layer deterministic dimension.
            hs_dim: Higher layer stochastic dimension.
            ld_dim: Lower layer deterministic dimension.
            ls_dim: Lower layer stochastic dimension.
            l_tau: Lower layer RNN time constant.
            h_tau: Higher layer RNN time constant.
            l_prior: Lower layer prior network.
            l_posterior: Lower layer posterior network.
            h_prior: Higher layer prior network.
            h_posterior: Higher layer posterior network.
            l_dist: Lower layer distribution factory.
            h_dist: Higher layer distribution factory.
            w_kl_h: Weight for higher layer KL divergence.
            action_size: Action dimension.
        """
        # Create a dummy transition for BaseRSSM initialization
        # We'll use MTRNN instead of Transition
        from multimodal_rssm.models.networks import Transition

        dummy_transition = Transition(
            deterministic_size=ld_dim,
            hidden_size=ld_dim,
            action_size=1,  # Dummy
            distribution_config=[1, 1],  # Dummy
            activation_name="ELU",
        )

        super().__init__(
            audio_representation=audio_representation,
            vision_representation=vision_representation,
            transition=dummy_transition,
            audio_encoder=audio_encoder,
            vision_encoder=vision_encoder,
            audio_decoder=audio_decoder,
            vision_decoder=vision_decoder,
            init_proj=init_proj,
            kl_coeff=kl_coeff,
            use_kl_balancing=use_kl_balancing,
        )

        # MTRSSM specific attributes
        self.action_dim = action_size
        self.hd_dim = hd_dim
        self.hs_dim = hs_dim
        self.ld_dim = ld_dim
        self.ls_dim = ls_dim
        self.w_kl_h = w_kl_h

        # MTRNN layers
        self.l_rnn = MTRNN(
            input_dim=action_size + ls_dim + hs_dim,
            hidden_dim=ld_dim,
            tau=l_tau,
        )
        self.h_rnn = MTRNN(
            input_dim=hs_dim,
            hidden_dim=hd_dim,
            tau=h_tau,
        )

        # Prior and posterior networks
        self.l_prior = l_prior
        self.l_posterior = l_posterior
        self.h_prior = h_prior
        self.h_posterior = h_posterior

        # Distribution factories
        self.l_dist = l_dist
        self.h_dist = h_dist

    @property
    def feature_dim(self) -> int:
        """Get feature dimension.

        Returns
        -------
        int: Feature dimension (hd_dim + hs_dim + ld_dim + ls_dim).
        """
        return self.hd_dim + self.hs_dim + self.ld_dim + self.ls_dim

    def get_observations_from_batch(self, batch: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
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

    def get_initial_observation(self, observations: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Extract initial observations from observation sequences.

        Args:
            observations: Tuple of (audio_obs, vision_obs) each shape: [B, T, ...]

        Returns
        -------
        tuple[Tensor, Tensor]: Initial observations each shape: [B, ...]
        """
        audio_obs, vision_obs = observations
        return audio_obs[:, 0], vision_obs[:, 0]

    def get_targets_from_batch(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
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

    def _set_prev_hiddens(
        self,
        batch_size: int,
        init_obs: Tensor | None = None,
        prev_state: MTState | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set previous hidden states for MTRNN.

        Args:
            batch_size: Batch size.
            init_obs: Initial observation embedding.
            prev_state: Previous MTState.
            device: Device to use.
        """
        if device is None:
            device = next(self.parameters()).device

        if prev_state is not None:
            self.l_rnn.hidden = prev_state.hidden_l
            self.h_rnn.hidden = prev_state.hidden_h
        elif init_obs is not None:
            # Initialize from observation
            h = self.init_proj(init_obs)
            higher_h = h[..., : self.hd_dim]
            lower_h = h[..., self.hd_dim :]
            self.h_rnn.hidden = higher_h
            self.l_rnn.hidden = lower_h
        else:
            # Zero initialization
            higher_h = torch.zeros(batch_size, self.hd_dim, device=device)
            lower_h = torch.zeros(batch_size, self.ld_dim, device=device)
            self.h_rnn.hidden = higher_h
            self.l_rnn.hidden = lower_h

    def _compute_lower_posterior_with_logits(
        self,
        obs_embed: Tensor,
        prior_l_deter: Tensor,
        representation: Representation,
    ) -> tuple[Tensor, Tensor]:
        """Compute lower layer posterior logits for MoPoE fusion.

        Args:
            obs_embed: Observation embedding. Shape: [B, obs_embed_size]
            prior_l_deter: Lower layer prior deterministic state. Shape: [B, ld_dim]
            representation: Representation network to use.

        Returns
        -------
        tuple[Tensor, Tensor]
            (posterior_logits, logits) where logits shape is [B, class_size * category_size]
        """
        projector_input = torch.cat([prior_l_deter, obs_embed], -1)
        stoch_source = representation.rnn_to_post_projector(projector_input)
        return stoch_source, stoch_source

    def _compute_lower_prior(
        self, action: Tensor, prev_l_stoch: Tensor, prev_h_stoch: Tensor, prev_l_deter: Tensor
    ) -> tuple[Tensor, Distribution]:
        """Compute lower layer prior.

        Args:
            action: Action tensor. Shape: [B, action_size]
            prev_l_stoch: Previous lower stochastic state. Shape: [B, ls_dim]
            prev_h_stoch: Previous higher stochastic state. Shape: [B, hs_dim]
            prev_l_deter: Previous lower deterministic state. Shape: [B, ld_dim]

        Returns
        -------
        tuple[Tensor, Distribution]
            (l_deter, l_prior_distribution)
        """
        # Flatten stoch states if needed
        prev_l_stoch_flat = prev_l_stoch.flatten(start_dim=1) if prev_l_stoch.dim() > 2 else prev_l_stoch
        prev_h_stoch_flat = prev_h_stoch.flatten(start_dim=1) if prev_h_stoch.dim() > 2 else prev_h_stoch

        l_input = torch.cat([action, prev_l_stoch_flat, prev_h_stoch_flat], dim=-1)
        l_deter = self.l_rnn(l_input, prev_l_deter)
        l_prior_logits = self.l_prior(l_deter)
        l_prior_dist = self.l_dist(l_prior_logits)
        return l_deter, l_prior_dist

    def _compute_higher_prior_posterior(
        self,
        l_deter: Tensor,
        prev_h_deter: Tensor,
        prev_h_stoch: Tensor,
    ) -> tuple[Tensor, Distribution, Distribution]:
        """Compute higher layer prior and posterior.

        Args:
            l_deter: Lower layer deterministic state. Shape: [B, ld_dim]
            prev_h_deter: Previous higher deterministic state. Shape: [B, hd_dim]
            prev_h_stoch: Previous higher stochastic state. Shape: [B, hs_dim]

        Returns
        -------
        tuple[Tensor, Distribution, Distribution]
            (h_deter, h_prior_dist, h_posterior_dist)
        """
        # Flatten h_stoch if needed
        prev_h_stoch_flat = prev_h_stoch.flatten(start_dim=1) if prev_h_stoch.dim() > 2 else prev_h_stoch

        h_deter = self.h_rnn(prev_h_stoch_flat, prev_h_deter)
        h_prior_logits = self.h_prior(h_deter)
        h_prior_dist = self.h_dist(h_prior_logits)

        # Posterior depends on both l_deter and h_deter
        h_posterior_input = torch.cat([l_deter, h_deter], dim=-1)
        h_posterior_logits = self.h_posterior(h_posterior_input)
        h_posterior_dist = self.h_dist(h_posterior_logits)

        return h_deter, h_prior_dist, h_posterior_dist

    def initial_state(self, observation: tuple[Tensor, Tensor] | Tensor) -> MTState:  # type: ignore[override]
        """Initialize the hierarchical latent state.

        Args:
            observation: Observation(s) to initialize the latent state.
                For tuple: (audio_obs, vision_obs) each shape: [B, ...]

        Returns
        -------
        MTState: The initial hierarchical latent state.
        """
        if isinstance(observation, tuple):
            obs_embed = self.encode_observation(observation)
        else:
            obs_embed = observation

        batch_size = obs_embed.shape[0]
        device = obs_embed.device

        # Project observation to hierarchical state
        h = self.init_proj(obs_embed)  # [B, hd_dim + ld_dim]
        higher_h = h[..., : self.hd_dim]
        lower_h = h[..., self.hd_dim :]

        # Initialize RNN hidden states
        self.h_rnn.hidden = higher_h
        self.l_rnn.hidden = lower_h

        # Initialize stochastic states from priors
        h_prior_logits = self.h_prior(higher_h)
        l_prior_logits = self.l_prior(lower_h)
        h_prior_dist = self.h_dist(h_prior_logits)
        l_prior_dist = self.l_dist(l_prior_logits)

        return MTState(
            deter_h=higher_h,
            deter_l=lower_h,
            distribution_h=h_prior_dist,
            distribution_l=l_prior_dist,
            hidden_h=higher_h,
            hidden_l=lower_h,
        ).to(device)

    def rollout_representation(  # noqa: PLR0914
        self,
        *,
        actions: Tensor,
        observations: Tensor | tuple[Tensor, ...],
        prev_state: MTState,
    ) -> tuple[MTState, MTState]:
        """Rollout representation and compute posteriors with MoPoE fusion on hierarchical states.

        Args:
            actions: The actions to rollout the representation. Shape: [B, T, action_size]
            observations: The observations to rollout the representation.
                For tuple: (audio_obs, vision_obs) each shape: [B, T, ...]
            prev_state: The previous MTState to rollout the representation.

        Returns
        -------
        tuple[MTState, MTState]
            (mixed posterior, prior) states.

        Raises
        ------
        TypeError
            If observations is not a tuple.
        """
        if not isinstance(observations, tuple):
            msg = "MoPoE-MMTRSSM requires tuple of (audio_obs, vision_obs)"
            raise TypeError(msg)

        audio_obs, vision_obs = observations

        audio_embed: Tensor = self.audio_encoder(audio_obs)  # type: ignore[no-any-return]
        vision_embed: Tensor = self.vision_encoder(vision_obs)  # type: ignore[no-any-return]

        # Initialize RNN hidden states
        batch_size = audio_embed.shape[0]
        self._set_prev_hiddens(batch_size, prev_state=prev_state)

        priors: list[MTState] = []
        mixed_posteriors: list[MTState] = []

        for t in range(audio_embed.shape[1]):
            # Get previous stochastic states
            prev_l_stoch = prev_state.stoch_l
            prev_h_stoch = prev_state.stoch_h
            prev_l_deter = prev_state.deter_l
            prev_h_deter = prev_state.deter_h

            # Compute lower layer prior
            l_deter, l_prior_dist = self._compute_lower_prior(
                actions[:, t],
                prev_l_stoch,
                prev_h_stoch,
                prev_l_deter,
            )

            # Compute audio and vision posterior logits for MoPoE fusion
            audio_obs_embed = audio_embed[:, t]
            _, audio_logits = self._compute_lower_posterior_with_logits(
                audio_obs_embed,
                l_deter,
                self.audio_representation,
            )

            vision_obs_embed = vision_embed[:, t]
            _, vision_logits = self._compute_lower_posterior_with_logits(
                vision_obs_embed,
                l_deter,
                self.vision_representation,
            )

            # MoPoE fusion for lower layer
            audio_log_probs = torch.nn.functional.log_softmax(audio_logits, dim=-1)
            vision_log_probs = torch.nn.functional.log_softmax(vision_logits, dim=-1)
            fused_log_probs = audio_log_probs + vision_log_probs

            # MoE fusion for lower layer mixed posterior
            weight = 1.0 / 3.0
            log_weight = torch.log(torch.tensor(weight, device=audio_log_probs.device, dtype=audio_log_probs.dtype))
            weighted_log_probs = torch.stack(
                [
                    log_weight + audio_log_probs,
                    log_weight + vision_log_probs,
                    log_weight + fused_log_probs,
                ],
                dim=-2,
            )
            mixed_log_probs = torch.logsumexp(weighted_log_probs, dim=-2)
            if mixed_log_probs.dim() > 2:
                mixed_log_probs = mixed_log_probs.view(mixed_log_probs.shape[0], -1)

            l_posterior_dist = self.l_dist(mixed_log_probs)
            l_stoch = l_posterior_dist.rsample()

            # Compute higher layer prior and posterior
            h_deter, h_prior_dist, h_posterior_dist = self._compute_higher_prior_posterior(
                l_deter,
                prev_h_deter,
                prev_h_stoch,
            )
            h_stoch = h_posterior_dist.rsample()

            # Create MTState
            prior_state = MTState(
                deter_h=h_deter,
                deter_l=l_deter,
                distribution_h=h_prior_dist,
                distribution_l=l_prior_dist,
                hidden_h=self.h_rnn.hidden,
                hidden_l=self.l_rnn.hidden,
            )

            posterior_state = MTState(
                deter_h=h_deter,
                deter_l=l_deter,
                distribution_h=h_posterior_dist,
                distribution_l=l_posterior_dist,
                hidden_h=self.h_rnn.hidden,
                hidden_l=self.l_rnn.hidden,
                stoch_h=h_stoch,
                stoch_l=l_stoch,
            )

            priors.append(prior_state)
            mixed_posteriors.append(posterior_state)

            prev_state = posterior_state

        prior = stack_mtstates(priors, dim=1)
        posterior_mixed = stack_mtstates(mixed_posteriors, dim=1)
        return posterior_mixed, prior

    def rollout_transition(self, *, actions: Tensor, prev_state: MTState) -> MTState:  # type: ignore[override]
        """Rollout the transition model using 2-layer RNN.

        Args:
            actions: The actions to rollout the transition model. Shape: [B, T, action_size]
            prev_state: The previous MTState to rollout the transition model.

        Returns
        -------
        MTState: The prior states.
        """
        batch_size = actions.shape[0]
        self._set_prev_hiddens(batch_size, prev_state=prev_state)

        priors: list[MTState] = []

        for t in range(actions.shape[1]):
            prev_l_stoch = prev_state.stoch_l
            prev_h_stoch = prev_state.stoch_h
            prev_l_deter = prev_state.deter_l
            prev_h_deter = prev_state.deter_h

            # Compute lower layer prior
            l_deter, l_prior_dist = self._compute_lower_prior(
                actions[:, t],
                prev_l_stoch,
                prev_h_stoch,
                prev_l_deter,
            )

            # Compute higher layer prior
            prev_h_stoch_flat = prev_h_stoch.flatten(start_dim=1) if prev_h_stoch.dim() > 2 else prev_h_stoch
            h_deter = self.h_rnn(prev_h_stoch_flat, prev_h_deter)
            h_prior_logits = self.h_prior(h_deter)
            h_prior_dist = self.h_dist(h_prior_logits)

            prior_state = MTState(
                deter_h=h_deter,
                deter_l=l_deter,
                distribution_h=h_prior_dist,
                distribution_l=l_prior_dist,
                hidden_h=self.h_rnn.hidden,
                hidden_l=self.l_rnn.hidden,
            )

            priors.append(prior_state)
            prev_state = prior_state

        return stack_mtstates(priors, dim=1)

    def decode_state(self, state: MTState) -> dict[str, Tensor]:  # type: ignore[override]
        """Decode hierarchical state into reconstructions for each modality.

        Args:
            state: The MTState to decode.

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

    def shared_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:  # type: ignore[override]
        """Perform a shared training/validation step with hierarchical KL divergence.

        Args:
            batch: The batch to shared step.

        Returns
        -------
        dict[str, Tensor]: The loss dictionary.
        """
        action_input = batch[0]
        observations = self.get_observations_from_batch(batch)
        initial_observation = self.get_initial_observation(observations)

        posterior, prior = self.rollout_representation(
            actions=action_input,
            observations=observations,
            prev_state=self.initial_state(initial_observation),  # type: ignore[arg-type]
        )

        reconstructions = self.decode_state(posterior)
        targets = self.get_targets_from_batch(batch)

        loss_dict = MoPoE_MRSSM.compute_reconstruction_loss(reconstructions, targets)

        # Lower layer KL divergence
        kl_div_l = kl_divergence(
            q=posterior.distribution_l.independent(1),
            p=prior.distribution_l.independent(1),
            use_balancing=self.use_kl_balancing,
        ).mul(self.kl_coeff)

        # Higher layer KL divergence
        kl_div_h = kl_divergence(
            q=posterior.distribution_h.independent(1),
            p=prior.distribution_h.independent(1),
            use_balancing=self.use_kl_balancing,
        ).mul(self.kl_coeff * self.w_kl_h)

        loss_dict["kl"] = kl_div_l
        loss_dict["kl_h"] = kl_div_h
        loss_dict["loss"] = loss_dict["recon"] + kl_div_l + kl_div_h

        return loss_dict

    # Inherit other methods from MoPoE_MRSSM
    compute_reconstruction_loss = MoPoE_MRSSM.compute_reconstruction_loss
    encode_observation = MoPoE_MRSSM.encode_observation
