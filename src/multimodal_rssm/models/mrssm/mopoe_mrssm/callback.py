"""Unified callbacks for MoPoE-MRSSM."""

import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger

from multimodal_rssm.models.mrssm.callback import LogMultimodalMRSSMOutput
from multimodal_rssm.models.mrssm.mopoe_mrssm.core import MoPoE_MRSSM


class LogMoPoEMRSSMOutput(LogMultimodalMRSSMOutput):
    """Log MoPoE-MRSSM output (multimodal: audio + vision)."""

    def __init__(
        self,
        *,
        every_n_epochs: int,
        indices: list[int],
        query_length: int,
        fps: float,
    ) -> None:
        """Initialize LogMoPoEMRSSMOutput.

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
            model_types=(MoPoE_MRSSM,),
        )


class LogWeightedMoPoEWeights(Callback):
    """Log Weighted MoPoE-MRSSM weights over time series to WandB."""

    def __init__(
        self,
        *,
        every_n_epochs: int = 10,
    ) -> None:
        """Initialize LogWeightedMoPoEWeights.

        Args:
            every_n_epochs: Log every N epochs. Defaults to 10.
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log weights at the end of validation epoch."""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not isinstance(logger := trainer.logger, WandbLogger):
            return
        # Note: WeightedMoPoE_MRSSM is not available in this implementation
        # This callback will not work without weighted_core.py
        # if not isinstance(pl_module, WeightedMoPoE_MRSSM):
        #     return

        # Log weights over time series by collecting data from validation episodes
        # self._log_weights_timeseries(trainer, pl_module, logger)
        pass

    @staticmethod
    def _log_weights_timeseries(  # noqa: PLR0914
        trainer: Trainer,
        pl_module: LightningModule,  # type: ignore[type-arg]
        logger: WandbLogger,
    ) -> None:
        """Log weights as time series data from all validation episodes."""
        if not trainer.datamodule:  # type: ignore[attr-defined]
            return
        try:
            val_dataloader = trainer.datamodule.val_dataloader()  # type: ignore[attr-defined]
        except AttributeError:
            return

        device = pl_module.device
        episode_idx = 0

        for raw_batch in val_dataloader:
            # Move batch to model device
            processed_batch = tuple(tensor.to(device) for tensor in raw_batch)

            # Get observations and actions from batch
            # Batch format: (action_input, audio_obs_input, vision_obs_input,
            #               action_target, audio_obs_target, vision_obs_target)
            actions = processed_batch[0]  # [B, T, action_size]
            observations = pl_module.get_observations_from_batch(processed_batch)
            audio_obs, _ = observations
            batch_size = audio_obs.shape[0]

            # Get initial state
            initial_obs = pl_module.get_initial_observation(observations)
            initial_state = pl_module.initial_state(observation=initial_obs)

            with torch.no_grad():
                # Rollout to compute weights
                _, _ = pl_module.rollout_representation(
                    actions=actions,
                    observations=observations,
                    prev_state=initial_state,
                )

                # Get computed weights [B, T, 3]
                weights = pl_module.weights_timeseries
                if weights is None:
                    continue

                weights_cpu = weights.detach().cpu()  # [B, T, 3]
                seq_len = weights_cpu.shape[1]

                # For each episode in the batch
                for b in range(batch_size):
                    episode_weights = weights_cpu[b]  # [T, 3]
                    w_audio = episode_weights[:, 0].tolist()
                    w_vision = episode_weights[:, 1].tolist()
                    w_fused = episode_weights[:, 2].tolist()
                    timesteps = list(range(seq_len))

                    # Create line series plot for this specific episode
                    chart = wandb.plot.line_series(
                        xs=[timesteps, timesteps, timesteps],
                        ys=[w_audio, w_vision, w_fused],
                        keys=["audio", "vision", "fused"],
                        title=f"MoPoE Weights - Episode {episode_idx}",
                        xname="timestep",
                    )
                    logger.experiment.log(
                        {
                            f"weights/episode_{episode_idx}": chart,
                            "epoch": trainer.current_epoch,
                        },
                        step=trainer.global_step,
                    )

                    # Log average weights for this episode
                    logger.experiment.log(
                        {
                            f"weights/episode_{episode_idx}/avg_audio": sum(w_audio) / len(w_audio),
                            f"weights/episode_{episode_idx}/avg_vision": sum(w_vision) / len(w_vision),
                            f"weights/episode_{episode_idx}/avg_fused": sum(w_fused) / len(w_fused),
                            "epoch": trainer.current_epoch,
                        },
                        step=trainer.global_step,
                    )
                    episode_idx += 1

        # Log total number of episodes processed
        logger.experiment.log(
            {
                "weights/num_episodes": episode_idx,
                "epoch": trainer.current_epoch,
            },
            step=trainer.global_step,
        )
