"""Callbacks for MoPoE-MMTRSSM."""

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger

from multimodal_rssm.models.mrssm.callback import LogMultimodalMRSSMOutput
from multimodal_rssm.models.mmtrssm.mopoe_mmtrssm.core import MoPoE_MMTRSSM
from multimodal_rssm.models.mmtrssm.state import MTState, cat_mtstates


class LogMoPoEMMTRSSMOutput(LogMultimodalMRSSMOutput):
    """Log MoPoE-MMTRSSM output (multimodal: audio + vision with hierarchical states)."""

    def __init__(
        self,
        *,
        every_n_epochs: int,
        indices: list[int],
        query_length: int,
        fps: float,
    ) -> None:
        """Initialize LogMoPoEMMTRSSMOutput.

        Args:
            every_n_epochs: Log every N epochs.
            indices: Episode indices to log.
            query_length: Query length for rollout.
            fps: Frames per second for video logging.
        """
        super().__init__(
            every_n_epochs=every_n_epochs,
            indices=indices,
            query_length=query_length,
            fps=fps,
            model_types=(MoPoE_MMTRSSM,),
        )

    def _compute_reconstructions(
        self,
        model: LightningModule,
        action_input: torch.Tensor,
        observation_input: tuple[torch.Tensor, torch.Tensor],
        audio_obs_input: torch.Tensor,
        vision_obs_input: torch.Tensor,
    ) -> tuple[MTState, MTState]:
        """Compute posterior and prior reconstructions for MTState.

        Args:
            model: Model instance
            action_input: Action input tensor
            observation_input: Observation input tuple
            audio_obs_input: Audio observation input
            vision_obs_input: Vision observation input

        Returns
        -------
        tuple[MTState, MTState]: (posterior, prior) states
        """
        rssm_model: MoPoE_MMTRSSM = model  # type: ignore[assignment]
        posterior, _ = rssm_model.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=rssm_model.initial_state((audio_obs_input[:, 0], vision_obs_input[:, 0])),
        )

        prior = rssm_model.rollout_transition(
            actions=action_input[:, self.query_length :],
            prev_state=posterior[:, self.query_length - 1],
        )
        prior = cat_mtstates([posterior[:, : self.query_length], prior], dim=1)
        return posterior, prior

    @staticmethod
    def _denormalize_reconstructions(
        model: LightningModule,
        prior: MTState,
        posterior: MTState,
        observation_info: dict[str, torch.Tensor | bool],
    ) -> dict[str, torch.Tensor]:
        """Denormalize reconstructions for visualization with MTState.

        Args:
            model: Model instance
            prior: Prior MTState
            posterior: Posterior MTState
            observation_info: Dictionary containing:
                - audio_target: Audio observation target
                - vision_target: Vision observation target
                - audio_missing: Whether audio modality was missing in the input
                - vision_missing: Whether vision modality was missing in the input

        Returns
        -------
        dict[str, Tensor]: Dictionary with denormalized video data
        """
        from typing import cast

        # Unpack observation information
        audio_obs_target = cast("torch.Tensor", observation_info["audio_target"])
        vision_obs_target = cast("torch.Tensor", observation_info["vision_target"])
        audio_missing = bool(observation_info["audio_missing"])
        vision_missing = bool(observation_info["vision_missing"])

        # Compute reconstructions using feature from MTState
        decoder_model = model  # type: ignore[assignment]
        posterior_audio_recon = decoder_model.audio_decoder.forward(posterior.feature)  # type: ignore[attr-defined, union-attr]
        posterior_vision_recon = decoder_model.vision_decoder.forward(posterior.feature)  # type: ignore[attr-defined, union-attr]
        prior_audio_recon = decoder_model.audio_decoder.forward(prior.feature)  # type: ignore[attr-defined, union-attr]
        prior_vision_recon = decoder_model.vision_decoder.forward(prior.feature)  # type: ignore[attr-defined, union-attr]

        # Denormalize: from [-1, 1] to [0, 1]
        prior_audio_recon = (prior_audio_recon + 1.0) / 2.0
        audio_obs_denorm: torch.Tensor = (audio_obs_target + 1.0) / 2.0
        posterior_audio_recon = (posterior_audio_recon + 1.0) / 2.0
        prior_vision_recon = (prior_vision_recon + 1.0) / 2.0
        vision_obs_denorm: torch.Tensor = (vision_obs_target + 1.0) / 2.0
        posterior_vision_recon = (posterior_vision_recon + 1.0) / 2.0

        # For missing modalities, visualize observation as pure black
        if audio_missing:
            audio_obs_denorm = torch.zeros_like(audio_obs_denorm)
        if vision_missing:
            vision_obs_denorm = torch.zeros_like(vision_obs_denorm)

        return {
            "prior_vision": prior_vision_recon,
            "observation_vision": vision_obs_denorm,
            "posterior_vision": posterior_vision_recon,
            "prior_audio": prior_audio_recon,
            "observation_audio": audio_obs_denorm,
            "posterior_audio": posterior_audio_recon,
        }
