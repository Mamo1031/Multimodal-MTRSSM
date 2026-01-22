"""Common callbacks for RSSM variants."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

MAX_EPISODES = 60
RGB_CHANNELS = 3


class WandBMetricOrganizer(Callback):
    """Organize WandB metrics for better visualization.

    This callback defines metric groups and ordering in WandB:
    - Groups train/val metrics together in the same chart (loss, recon, kl)
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_metrics: dict[str, list[tuple[int, float]]] = {
            "recon": [],
            "kl": [],
            "loss": [],
        }
        self.val_metrics: dict[str, list[tuple[int, float]]] = {
            "recon": [],
            "kl": [],
            "loss": [],
        }

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        """Define WandB metrics organization at training start."""
        if not isinstance(logger := trainer.logger, WandbLogger):
            return

        # Get the wandb run object
        if hasattr(logger, "experiment"):
            wandb_run = logger.experiment
            # Define metrics with grouping and ordering
            # Use same step_metric="epoch" to ensure train/val are on same x-axis
            # Group 1: loss (train and val in same chart)
            wandb_run.define_metric("train/loss", step_metric="epoch", summary="min")
            wandb_run.define_metric("val/loss", step_metric="epoch", summary="min")

            # Group 2: recon (train and val in same chart)
            wandb_run.define_metric("train/recon", step_metric="epoch", summary="min")
            wandb_run.define_metric("val/recon", step_metric="epoch", summary="min")

            # Group 3: kl (train and val in same chart)
            wandb_run.define_metric("train/kl", step_metric="epoch", summary="min")
            wandb_run.define_metric("val/kl", step_metric="epoch", summary="min")

            # Group 4: epoch
            wandb_run.define_metric("epoch", summary="max")

            # Group 5: learning rate (lr-AdamW or similar)
            wandb_run.define_metric("lr-AdamW", summary="max")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        """Collect training metrics at the end of each epoch."""
        current_epoch = trainer.current_epoch
        logged_metrics = trainer.logged_metrics

        # Collect training metrics
        for metric_name in ["loss", "recon", "kl"]:
            key = f"train/{metric_name}"
            if key in logged_metrics:
                value = float(logged_metrics[key].cpu().item())
                self.train_metrics[metric_name].append((current_epoch, value))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        """Create combined charts for train/val metrics at the end of validation epoch."""
        if not isinstance(logger := trainer.logger, WandbLogger):
            return

        current_epoch = trainer.current_epoch
        logged_metrics = trainer.logged_metrics

        # Collect validation metrics
        for metric_name in ["loss", "recon", "kl"]:
            key = f"val/{metric_name}"
            if key in logged_metrics:
                value = float(logged_metrics[key].cpu().item())
                self.val_metrics[metric_name].append((current_epoch, value))

        # Create combined charts using wandb.plot.line_series
        try:
            wandb_run = logger.experiment
            for metric_name in ["loss", "recon", "kl"]:
                if len(self.train_metrics[metric_name]) > 0 and len(self.val_metrics[metric_name]) > 0:
                    # Prepare data for line_series
                    train_data = self.train_metrics[metric_name]
                    val_data = self.val_metrics[metric_name]

                    # Get x and y values
                    train_xs = [x for x, _ in train_data]
                    train_ys = [y for _, y in train_data]
                    val_xs = [x for x, _ in val_data]
                    val_ys = [y for _, y in val_data]

                    # Create combined chart
                    chart = wandb.plot.line_series(
                        xs=[train_xs, val_xs],
                        ys=[train_ys, val_ys],
                        keys=["train", "val"],
                        title=f"{metric_name} (train vs val)",
                        xname="epoch",
                    )
                    wandb_run.log({f"{metric_name}_combined": chart})
        except (ImportError, AttributeError) as e:
            # If plotting fails, continue without combined charts
            # This can happen if wandb is not properly initialized
            if isinstance(logger, WandbLogger):
                logger.experiment.log({})  # Log empty dict to ensure wandb is initialized
            # Silently continue - combined charts are optional
            _ = e  # Suppress unused variable warning


class BaseLogRSSMOutput(Callback, ABC):
    """Base class for logging RSSM output."""

    def __init__(
        self,
        *,
        every_n_epochs: int,
        indices: list[int],
        query_length: int,
        fps: float,
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.indices = indices
        self.query_length = query_length
        self.fps = fps

    @abstractmethod
    def _extract_episodes_from_batch(self, batch: tuple[Tensor, ...], device: torch.device) -> list[tuple[Tensor, ...]]:
        """Extract episodes from a batch.

        Args:
            batch: Batch tuple from dataloader
            device: Device to move tensors to

        Returns
        -------
        list: List of episode tuples
        """

    def _collect_episodes(self, trainer: Trainer, model: LightningModule, stage: str) -> list[tuple[Tensor, ...]]:
        """Collect episodes from dataloader.

        Args:
            trainer: PyTorch Lightning trainer
            model: The model instance
            stage: "train" or "val"

        Returns
        -------
        list: List of episodes
        """
        dataloader = getattr(trainer.datamodule, f"{stage}_dataloader")()  # type: ignore[attr-defined]
        all_episodes = []
        for batch in dataloader:
            episodes = self._extract_episodes_from_batch(batch, model.device)
            all_episodes.extend(episodes)
            # Limit to MAX_EPISODES episodes
            if len(all_episodes) >= MAX_EPISODES:
                break
        return all_episodes

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log RSSM output at the end of the validation epoch."""
        if trainer.current_epoch % self.every_n_epochs != 0 or trainer.current_epoch == 0:
            return
        if not isinstance(logger := trainer.logger, WandbLogger):
            return
        if not self._is_valid_model(pl_module):
            return

        for stage in ("train", "val"):
            all_episodes = self._collect_episodes(trainer, pl_module, stage)
            # Limit to 7 episodes per stage for visualization
            limited_episodes = all_episodes[:7]
            for episode_idx, episode in enumerate(limited_episodes):
                self._process_episode(episode, pl_module, stage, episode_idx, logger)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log RSSM output with the best model at the end of training."""
        if not isinstance(logger := trainer.logger, WandbLogger):
            return
        if not self._is_valid_model(pl_module):
            return

        # Load best model
        best_model = load_best_model_checkpoint(trainer, pl_module)

        # Visualize with the best model
        for stage in ("train", "val"):
            all_episodes = self._collect_episodes(trainer, best_model, stage)
            # Limit to 7 episodes per stage for visualization
            limited_episodes = all_episodes[:7]
            for episode_idx, episode in enumerate(limited_episodes):
                self._process_episode(episode, best_model, stage, episode_idx, logger)

    @abstractmethod
    def _process_episode(
        self,
        episode: tuple[Tensor, ...],
        pl_module: LightningModule,
        stage: str,
        episode_idx: int,
        logger: WandbLogger,
    ) -> None:
        """Process a single episode for visualization.

        Args:
            episode: Episode data tuple.
            pl_module: Lightning module.
            stage: Stage name ("train" or "val").
            episode_idx: Episode index.
            logger: WandB logger.
        """

    @abstractmethod
    def _is_valid_model(self, pl_module: LightningModule) -> bool:
        """Check if the module is a valid model type.

        Args:
            pl_module: PyTorch Lightning module

        Returns
        -------
        bool: True if valid, False otherwise
        """


def create_combined_video(prior: Tensor, observation: Tensor, posterior: Tensor) -> Tensor:
    """Create a combined video with prior, observation, and posterior side by side.

    Args:
        prior: Prior reconstruction video tensor [batch, time, channels, height, width]
        observation: Observation video tensor [batch, time, channels, height, width]
        posterior: Posterior reconstruction video tensor [batch, time, channels, height, width]

    Returns
    -------
    Tensor
        Combined video tensor [batch, time, channels, height, width*3]

    Raises
    ------
    ValueError
        If the shapes of prior, observation, and posterior do not match.
    """
    # Ensure all videos have the same shape
    if not (prior.shape == observation.shape == posterior.shape):
        msg = (
            f"Video shapes must match: prior={prior.shape}, "
            f"observation={observation.shape}, posterior={posterior.shape}"
        )
        raise ValueError(msg)

    # Concatenate videos horizontally: [batch, time, channels, height, width*3]
    return torch.cat([prior, observation, posterior], dim=4)


def add_timestep_labels(video: Tensor) -> Tensor:
    """Add timestep labels and captions to each frame of the video.

    Args:
        video: Video tensor [batch, time, channels, height, width]

    Returns
    -------
    Tensor
        Video tensor with timestep labels and captions [batch, time, channels, height+padding, width]
    """
    batch_size, time_steps, channels, height, width = video.shape
    result = []

    # Padding sizes
    top_padding = 20  # For timestep label
    bottom_padding = 20  # For captions
    side_padding = 10  # For side margins

    # Calculate new dimensions
    new_height = height + top_padding + bottom_padding
    new_width = width + 2 * side_padding

    # Load font for timestep
    timestep_font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
    try:
        timestep_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except OSError:
        try:
            timestep_font = ImageFont.load_default()
        except (OSError, AttributeError):
            timestep_font = None

    # Load font for captions
    caption_font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
    try:
        caption_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except OSError:
        try:
            caption_font = ImageFont.load_default()
        except (OSError, AttributeError):
            caption_font = None

    # Calculate individual GIF width (3 GIFs side by side)
    gif_width = width // 3

    for b in range(batch_size):
        batch_frames = []
        for t in range(time_steps):
            frame = video[b, t]  # [channels, height, width]

            # Convert to numpy for PIL processing
            if channels == RGB_CHANNELS:
                # RGB: [C, H, W] -> [H, W, C]
                frame_np = frame.cpu().detach().permute(1, 2, 0).numpy()
                frame_np = (frame_np * 255).astype(np.uint8).clip(0, 255)
            elif channels == 1:
                # Grayscale: [C, H, W] -> [H, W]
                frame_np = frame.cpu().detach().squeeze(0).numpy()
                frame_np = (frame_np * 255).astype(np.uint8).clip(0, 255)
                # Convert to RGB for text rendering
                frame_np = np.stack([frame_np, frame_np, frame_np], axis=2)
            else:
                # For other channel counts, use first 3 channels
                frame_np = frame.cpu().detach()[:RGB_CHANNELS].permute(1, 2, 0).numpy()
                frame_np = (frame_np * 255).astype(np.uint8).clip(0, 255)

            # Create new image with padding
            new_frame_np = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            # Place original frame in the center (with side padding)
            new_frame_np[top_padding : top_padding + height, side_padding : side_padding + width] = frame_np

            # Create PIL Image
            pil_image = Image.fromarray(new_frame_np)
            draw = ImageDraw.Draw(pil_image)

            # Draw timestep label in top-left corner (outside the frame)
            # Show as current/total (1-based indexing for readability)
            timestep_label = f"t={t + 1}/{time_steps}"
            timestep_x = side_padding
            timestep_y = 5

            if timestep_font:
                timestep_bbox = draw.textbbox((timestep_x, timestep_y), timestep_label, font=timestep_font)
            else:
                timestep_bbox = draw.textbbox((timestep_x, timestep_y), timestep_label)

            # Draw background rectangle for timestep
            timestep_padding = 3
            draw.rectangle(
                [
                    timestep_bbox[0] - timestep_padding,
                    timestep_bbox[1] - timestep_padding,
                    timestep_bbox[2] + timestep_padding,
                    timestep_bbox[3] + timestep_padding,
                ],
                fill=(0, 0, 0),
            )
            draw.text((timestep_x, timestep_y), timestep_label, fill=(255, 255, 255), font=timestep_font)

            # Draw captions below each GIF
            captions = ["prior", "observation", "posterior"]
            caption_y = top_padding + height + 5

            for i, caption in enumerate(captions):
                # Calculate center position for each GIF
                gif_center_x = side_padding + gif_width * i + gif_width // 2

                if caption_font:
                    caption_bbox = draw.textbbox((0, 0), caption, font=caption_font)
                else:
                    caption_bbox = draw.textbbox((0, 0), caption)

                caption_width = caption_bbox[2] - caption_bbox[0]
                caption_x = gif_center_x - caption_width // 2

                # Draw background rectangle for caption
                caption_padding = 3
                draw.rectangle(
                    [
                        caption_x - caption_padding,
                        caption_y - caption_padding,
                        caption_x + caption_width + caption_padding,
                        caption_y + (caption_bbox[3] - caption_bbox[1]) + caption_padding,
                    ],
                    fill=(0, 0, 0),
                )
                draw.text((caption_x, caption_y), caption, fill=(255, 255, 255), font=caption_font)

            # Convert back to tensor
            frame_array = np.array(pil_image)
            # Convert RGB back to grayscale or transpose
            frame_array = frame_array[:, :, 0:1].transpose(2, 0, 1) if channels == 1 else frame_array.transpose(2, 0, 1)

            # Normalize to [0, 1]
            frame_tensor = torch.from_numpy(frame_array).float() / 255.0
            if frame_tensor.shape[0] != channels:
                # If we had to convert channels, pad or slice
                if frame_tensor.shape[0] > channels:
                    frame_tensor = frame_tensor[:channels]
                else:
                    # Pad with zeros
                    padding_tensor = torch.zeros(channels - frame_tensor.shape[0], new_height, new_width)
                    frame_tensor = torch.cat([frame_tensor, padding_tensor], dim=0)

            batch_frames.append(frame_tensor)

        result.append(torch.stack(batch_frames, dim=0))

    return torch.stack(result, dim=0)


def log_video(batch_video: Tensor, key: str, logger: WandbLogger, fps: float, visualization_mode: str = "rgb") -> None:
    """Log video.

    Args:
        batch_video: Video tensor [batch, time, channels, height, width]
        key: Key for logging
        logger: WandB logger
        fps: Frames per second
        visualization_mode: "rgb" or "magma"
    """
    if visualization_mode == "magma":
        batch = batch_video.cpu().detach().numpy()
        magma = colormaps["magma"]
        b, t, c, h, w = batch.shape
        # For multi-channel, use first channel or convert to grayscale
        batch = batch.squeeze(2) if c == 1 else batch[:, :, 0, :, :]  # [b, t, h, w]

        batch_normalized = batch * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        batch_db = (batch_normalized + 1.0) / 2.0 * 80.0 - 80.0  # [-1, 1] -> [-80, 0] dB

        # Check if this is a combined video (3 GIFs side by side)
        is_combined = w % 3 == 0 and w > h

        if is_combined:
            # Split into 3 parts (prior, observation, posterior)
            part_width = w // 3
            rgb = np.zeros((b, t, 3, h, w), dtype=np.uint8)

            for i in range(b):
                video_db = batch_db[i]  # [t, h, w] in decibel scale [-80, 0] dB

                vmin = -80.0
                vmax = 0.0

                for part_idx in range(3):
                    start_w = part_idx * part_width
                    end_w = (part_idx + 1) * part_width
                    part_db = video_db[:, :, start_w:end_w]  # [t, h, part_width] in decibel scale

                    # Scale from [-80, 0] dB to [0, 1] for colormap
                    part_clipped = np.clip(part_db, vmin, vmax)
                    part_scaled = (part_clipped - vmin) / (vmax - vmin)
                    part_scaled = part_scaled.clip(0, 1)

                    # Apply colormap to all frames of this part
                    for j in range(t):
                        frame_scaled = part_scaled[j]  # [h, part_width]
                        rgba = magma(frame_scaled)[:, :, :3]
                        rgb[i, j, :, :, start_w:end_w] = (rgba * 255).astype(np.uint8).transpose(2, 0, 1)
        else:
            # Single video (not combined) - scale as a whole
            rgb = np.zeros((b, t, 3, h, w), dtype=np.uint8)
            for i in range(b):
                video_db = batch_db[i]  # [t, h, w] in decibel scale [-80, 0] dB

                vmin = -80.0
                vmax = 0.0

                # Scale from [-80, 0] dB to [0, 1] for colormap
                video_clipped = np.clip(video_db, vmin, vmax)
                video_scaled = (video_clipped - vmin) / (vmax - vmin)
                video_scaled = video_scaled.clip(0, 1)

                # Apply colormap to all frames
                for j in range(t):
                    frame_scaled = video_scaled[j]  # [h, w]
                    rgba = magma(frame_scaled)[:, :, :3]
                    rgb[i, j] = (rgba * 255).astype(np.uint8).transpose(2, 0, 1)

        videos = [torch.from_numpy(rgb[i]) for i in range(b)]
        logger.log_video(key=key, videos=videos, fps=[fps] * len(videos), format=["gif"] * len(videos))
        return

    # Convert to uint8 and ensure values are in [0, 255] range
    batch_video = batch_video.clamp(0, 1)
    videos = list(batch_video.cpu().mul(255).to(torch.uint8))
    logger.log_video(key=key, videos=videos, fps=[fps] * len(videos), format=["gif"] * len(videos))


def load_best_model_checkpoint(trainer: Trainer, model: LightningModule) -> LightningModule:
    """Load the best model checkpoint from trainer.

    Args:
        trainer: PyTorch Lightning trainer
        model: Current model instance

    Returns
    -------
    LightningModule
        Best model (or current model if checkpoint loading fails)
    """
    # Find the best model checkpoint
    checkpoint_callback = None
    for callback in trainer.callbacks:  # type: ignore[attr-defined]
        if hasattr(callback, "best_model_path"):
            checkpoint_callback = callback
            break

    if checkpoint_callback is None or not hasattr(checkpoint_callback, "best_model_path"):
        # If no checkpoint callback, use current model
        return model

    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        return model

    try:
        checkpoint = torch.load(best_model_path, map_location=model.device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        return model
    except (OSError, KeyError, RuntimeError):
        # If loading fails, use current model
        return model
