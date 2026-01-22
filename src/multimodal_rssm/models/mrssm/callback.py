"""Common callbacks for multimodal MRSSM variants."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from multimodal_rssm.models.callback import RGB_CHANNELS, BaseLogRSSMOutput
from multimodal_rssm.models.state import State, cat_states

if TYPE_CHECKING:
    from lightning import LightningModule
    from lightning.pytorch.loggers import WandbLogger
    from matplotlib.colors import Colormap

    from multimodal_rssm.models.core import BaseRSSM

# Constants
GIF_PARTS = 3  # Number of parts in combined video (prior, observation, posterior)
GRAYSCALE_CHANNELS = 1


class LogMultimodalMRSSMOutput(BaseLogRSSMOutput):
    """Log multimodal MRSSM output (audio + vision) for NN-MRSSM and PoE-MRSSM."""

    def __init__(
        self,
        *,
        every_n_epochs: int,
        indices: list[int],
        query_length: int,
        fps: float,
        model_types: tuple[type, ...],
    ) -> None:
        """Initialize LogMultimodalMRSSMOutput.

        Args:
            every_n_epochs: Log every N epochs.
            indices: Episode indices to log.
            query_length: Query length for rollout.
            fps: Frames per second for video logging.
            model_types: Tuple of model types to accept (e.g., (NN_MRSSM, PoE_MRSSM)).
        """
        super().__init__(
            every_n_epochs=every_n_epochs,
            indices=indices,
            query_length=query_length,
            fps=fps,
        )
        self.model_types = model_types

    def _is_valid_model(self, pl_module: LightningModule) -> bool:
        """Check if the module is a valid multimodal MRSSM model.

        Args:
            pl_module: PyTorch Lightning module

        Returns
        -------
        bool: True if valid, False otherwise
        """
        return isinstance(pl_module, self.model_types)

    @staticmethod
    def _extract_episodes_from_batch(batch: tuple[Tensor, ...], device: torch.device) -> list[tuple[Tensor, ...]]:
        """Extract episodes from a batch for multimodal MRSSM.

        Args:
            batch: Batch tuple from dataloader
            device: Device to move tensors to

        Returns
        -------
        list: List of episode tuples
        """
        (
            action_input,
            audio_obs_input,
            vision_obs_input,
            _,
            audio_obs_target,
            vision_obs_target,
        ) = (tensor.to(device) for tensor in batch)
        batch_size = action_input.shape[0]
        return [
            (
                action_input[i : i + 1],
                audio_obs_input[i : i + 1],
                vision_obs_input[i : i + 1],
                audio_obs_target[i : i + 1],
                vision_obs_target[i : i + 1],
            )
            for i in range(batch_size)
        ]

    def _process_episode(
        self,
        episode: tuple[Tensor, ...],
        model: LightningModule,
        stage: str,
        episode_idx: int,
        logger: WandbLogger,
    ) -> None:
        """Process a single episode and log the output for multimodal MRSSM."""
        if not isinstance(model, self.model_types):
            return

        (
            action_input,
            audio_obs_input,
            vision_obs_input,
            audio_obs_target,
            vision_obs_target,
        ) = episode
        observation_input = (audio_obs_input, vision_obs_input)

        # Detect missing modalities based on ZeroOut transform (-1 in normalized space)
        # Batch size is 1 at this point, so we can check the whole tensor.
        audio_missing = bool(torch.all(audio_obs_input == -1.0))
        vision_missing = bool(torch.all(vision_obs_input == -1.0))

        observation_info: dict[str, Tensor | bool] = {
            "audio_target": audio_obs_target,
            "vision_target": vision_obs_target,
            "audio_missing": audio_missing,
            "vision_missing": vision_missing,
        }

        # Compute posterior and prior reconstructions
        posterior, prior = self._compute_reconstructions(
            model,
            action_input,
            observation_input,
            audio_obs_input,
            vision_obs_input,
        )

        # Denormalize for visualization
        video_data = LogMultimodalMRSSMOutput._denormalize_reconstructions(
            model,
            prior,
            posterior,
            observation_info,
        )

        # Create and log combined multimodal video
        combined_multimodal_video = self.create_multimodal_combined_video(video_data)
        key = f"{stage}/episode_{episode_idx}"
        self.log_multimodal_video(combined_multimodal_video, key, logger)

    def _compute_reconstructions(
        self,
        model: LightningModule,
        action_input: Tensor,
        observation_input: tuple[Tensor, Tensor],
        audio_obs_input: Tensor,
        vision_obs_input: Tensor,
    ) -> tuple[State, State]:
        """Compute posterior and prior reconstructions.

        Args:
            model: Model instance
            action_input: Action input tensor
            observation_input: Observation input tuple
            audio_obs_input: Audio observation input
            vision_obs_input: Vision observation input

        Returns
        -------
        tuple[State, State]: (posterior, prior) states
        """
        rssm_model: BaseRSSM = model  # type: ignore[assignment]
        posterior, _ = rssm_model.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=rssm_model.initial_state((audio_obs_input[:, 0], vision_obs_input[:, 0])),
        )

        prior = rssm_model.rollout_transition(
            actions=action_input[:, self.query_length :],
            prev_state=posterior[:, self.query_length - 1],
        )
        prior = cat_states([posterior[:, : self.query_length], prior], dim=1)
        return posterior, prior

    @staticmethod
    def _denormalize_reconstructions(
        model: LightningModule,
        prior: State,
        posterior: State,
        observation_info: dict[str, Tensor | bool],
    ) -> dict[str, Tensor]:
        """Denormalize reconstructions for visualization.

        Args:
            model: Model instance
            prior: Prior state
            posterior: Posterior state
            observation_info: Dictionary containing:
                - audio_target: Audio observation target
                - vision_target: Vision observation target
                - audio_missing: Whether audio modality was missing in the input
                - vision_missing: Whether vision modality was missing in the input

        Returns
        -------
        dict[str, Tensor]: Dictionary with denormalized video data
        """
        # Unpack observation information
        audio_obs_target = cast("Tensor", observation_info["audio_target"])
        vision_obs_target = cast("Tensor", observation_info["vision_target"])
        audio_missing = bool(observation_info["audio_missing"])
        vision_missing = bool(observation_info["vision_missing"])

        # Compute reconstructions
        decoder_model = model  # type: ignore[assignment]
        posterior_audio_recon = decoder_model.audio_decoder.forward(posterior.feature)  # type: ignore[attr-defined, union-attr]
        posterior_vision_recon = decoder_model.vision_decoder.forward(posterior.feature)  # type: ignore[attr-defined, union-attr]
        prior_audio_recon = decoder_model.audio_decoder.forward(prior.feature)  # type: ignore[attr-defined, union-attr]
        prior_vision_recon = decoder_model.vision_decoder.forward(prior.feature)  # type: ignore[attr-defined, union-attr]

        # Denormalize: from [-1, 1] to [0, 1]
        prior_audio_recon = (prior_audio_recon + 1.0) / 2.0
        audio_obs_denorm: Tensor = (audio_obs_target + 1.0) / 2.0
        posterior_audio_recon = (posterior_audio_recon + 1.0) / 2.0
        prior_vision_recon = (prior_vision_recon + 1.0) / 2.0
        vision_obs_denorm: Tensor = (vision_obs_target + 1.0) / 2.0
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

    def create_multimodal_combined_video(self, video_data: dict[str, Tensor]) -> Tensor:
        """Create a combined multimodal video with vision on top row and audio on bottom row.

        Layout:
        - Top row: vision prior, vision observation, vision posterior
        - Bottom row: audio prior, audio observation, audio posterior

        Args:
            video_data: Dictionary containing:
                - prior_vision: Vision prior reconstruction [batch, time, channels, height, width]
                - observation_vision: Vision observation [batch, time, channels, height, width]
                - posterior_vision: Vision posterior reconstruction [batch, time, channels, height, width]
                - prior_audio: Audio prior reconstruction [batch, time, channels, height, width]
                - observation_audio: Audio observation [batch, time, channels, height, width]
                - posterior_audio: Audio posterior reconstruction [batch, time, channels, height, width]

        Returns
        -------
        Tensor
            Combined multimodal video [batch, time, channels, height*2, width*3]
        """
        prior_vision = video_data["prior_vision"]
        observation_vision = video_data["observation_vision"]
        posterior_vision = video_data["posterior_vision"]
        prior_audio = video_data["prior_audio"]
        observation_audio = video_data["observation_audio"]
        posterior_audio = video_data["posterior_audio"]

        # Create horizontal combined videos for vision and audio
        vision_row = torch.cat([prior_vision, observation_vision, posterior_vision], dim=4)  # [B, T, C, H, W*3]
        audio_row = torch.cat([prior_audio, observation_audio, posterior_audio], dim=4)  # [B, T, C, H, W*3]

        # Ensure vision and audio have the same width (use the larger width)
        max_width = max(vision_row.shape[4], audio_row.shape[4])
        if vision_row.shape[4] < max_width:
            padding = torch.zeros(
                vision_row.shape[0],
                vision_row.shape[1],
                vision_row.shape[2],
                vision_row.shape[3],
                max_width - vision_row.shape[4],
            ).to(vision_row.device)
            vision_row = torch.cat([vision_row, padding], dim=4)
        if audio_row.shape[4] < max_width:
            padding = torch.zeros(
                audio_row.shape[0],
                audio_row.shape[1],
                audio_row.shape[2],
                audio_row.shape[3],
                max_width - audio_row.shape[4],
            ).to(audio_row.device)
            audio_row = torch.cat([audio_row, padding], dim=4)

        # Ensure vision and audio have the same number of channels
        vision_channels = vision_row.shape[2]
        audio_channels = audio_row.shape[2]

        if vision_channels != audio_channels:
            if audio_channels == GRAYSCALE_CHANNELS and vision_channels == RGB_CHANNELS:
                # Expand audio from 1 channel to 3 channels (grayscale to RGB)
                audio_row = audio_row.repeat(1, 1, RGB_CHANNELS, 1, 1)  # [B, T, 3, H, W*3]
            elif audio_channels == RGB_CHANNELS and vision_channels == GRAYSCALE_CHANNELS:
                # Convert vision from 3 channels to 1 channel (RGB to grayscale)
                vision_row = vision_row.mean(dim=2, keepdim=True)  # [B, T, 1, H, W*3]
            else:
                # Use the larger channel count
                max_channels = max(vision_channels, audio_channels)
                if vision_channels < max_channels:
                    padding = torch.zeros(
                        vision_row.shape[0],
                        vision_row.shape[1],
                        max_channels - vision_channels,
                        vision_row.shape[3],
                        vision_row.shape[4],
                    ).to(vision_row.device)
                    vision_row = torch.cat([vision_row, padding], dim=2)
                if audio_channels < max_channels:
                    padding = torch.zeros(
                        audio_row.shape[0],
                        audio_row.shape[1],
                        max_channels - audio_channels,
                        audio_row.shape[3],
                        audio_row.shape[4],
                    ).to(audio_row.device)
                    audio_row = torch.cat([audio_row, padding], dim=2)

        # Concatenate vertically: [batch, time, channels, height*2, width*3]
        combined = torch.cat([vision_row, audio_row], dim=3)

        # Add timestep labels and captions
        return self.add_multimodal_timestep_labels(combined)

    def add_multimodal_timestep_labels(self, video: Tensor) -> Tensor:  # noqa: PLR0914
        """Add timestep labels and captions to multimodal video (2 rows: vision top, audio bottom).

        Args:
            video: Video tensor [batch, time, channels, height*2, width*3]

        Returns
        -------
        Tensor
            Video tensor with timestep labels and captions [batch, time, channels, height*2+padding, width*3+padding]
        """
        batch_size, time_steps, channels, height, width = video.shape
        # height is actually height*2 (vision + audio)
        vision_height = height // 2
        audio_height = height // 2
        result = []

        # Padding sizes
        top_padding = 20  # For timestep label
        bottom_padding = 20  # For captions
        side_padding = 10  # For side margins
        row_label_padding = 10  # For row labels (Vision/Audio)

        # Calculate new dimensions
        new_height = height + top_padding + bottom_padding + row_label_padding
        new_width = width + 2 * side_padding

        # Load fonts
        timestep_font, row_font, caption_font = self._load_fonts()

        # Calculate individual GIF width (3 GIFs side by side)
        gif_width = width // GIF_PARTS

        frame_params = {
            "time_steps": time_steps,
            "channels": channels,
            "vision_height": vision_height,
            "audio_height": audio_height,
            "width": width,
            "new_height": new_height,
            "new_width": new_width,
            "top_padding": top_padding,
            "row_label_padding": row_label_padding,
            "side_padding": side_padding,
            "gif_width": gif_width,
            "timestep_font": timestep_font,
            "row_font": row_font,
            "caption_font": caption_font,
        }
        for b in range(batch_size):
            batch_frames = self._process_batch_frames(video[b], frame_params)
            result.append(torch.stack(batch_frames, dim=0))

        return torch.stack(result, dim=0)

    @staticmethod
    def _process_batch_frames(batch_video: Tensor, params: dict) -> list[Tensor]:
        """Process frames for a single batch.

        Args:
            batch_video: Video tensor for single batch [time, channels, height*2, width*3]
            params: Dictionary with frame processing parameters

        Returns
        -------
        list[Tensor]: List of frame tensors
        """
        batch_frames = []
        for t in range(params["time_steps"]):
            frame = batch_video[t]  # [channels, height*2, width*3]
            frame_tensor = LogMultimodalMRSSMOutput._process_single_frame(frame, t, params)
            batch_frames.append(frame_tensor)
        return batch_frames

    @staticmethod
    def _process_single_frame(frame: Tensor, timestep: int, params: dict) -> Tensor:  # noqa: PLR0914
        """Process a single frame.

        Args:
            frame: Frame tensor [channels, height*2, width*3]
            timestep: Current timestep
            params: Dictionary with frame processing parameters

        Returns
        -------
        Tensor: Processed frame tensor
        """
        channels = params["channels"]
        vision_height = params["vision_height"]
        audio_height = params["audio_height"]
        width = params["width"]
        new_height = params["new_height"]
        new_width = params["new_width"]
        top_padding = params["top_padding"]
        row_label_padding = params["row_label_padding"]
        side_padding = params["side_padding"]
        gif_width = params["gif_width"]
        timestep_font = params["timestep_font"]
        row_font = params["row_font"]
        caption_font = params["caption_font"]
        time_steps = params["time_steps"]

        # Split into vision (top) and audio (bottom) rows
        vision_frame = frame[:, :vision_height, :]  # [channels, vision_height, width*3]
        audio_frame = frame[:, vision_height:, :]  # [channels, audio_height, width*3]

        # Convert frames to numpy for PIL processing
        vision_frame_np = LogMultimodalMRSSMOutput._convert_frame_to_numpy(vision_frame, channels)
        audio_frame_np = LogMultimodalMRSSMOutput._convert_frame_to_numpy(audio_frame, channels)

        # Create new image with padding
        new_frame_np = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Place vision frame in top row
        vision_y_start = top_padding + row_label_padding
        vision_y_end = vision_y_start + vision_height
        new_frame_np[vision_y_start:vision_y_end, side_padding : side_padding + width] = vision_frame_np

        # Place audio frame in bottom row
        audio_y_start = vision_y_end
        audio_y_end = audio_y_start + audio_height
        new_frame_np[audio_y_start:audio_y_end, side_padding : side_padding + width] = audio_frame_np

        # Create PIL Image
        pil_image = Image.fromarray(new_frame_np)
        draw = ImageDraw.Draw(pil_image)

        # Draw labels and captions
        LogMultimodalMRSSMOutput._draw_timestep_label(draw, timestep, time_steps, side_padding, timestep_font)
        LogMultimodalMRSSMOutput._draw_row_labels(draw, side_padding, top_padding, vision_y_end, row_font)
        LogMultimodalMRSSMOutput._draw_captions(draw, side_padding, gif_width, audio_y_end, caption_font)

        # Convert back to tensor
        return LogMultimodalMRSSMOutput._convert_pil_to_tensor(pil_image, channels, new_height, new_width)

    @staticmethod
    def _load_fonts() -> tuple[
        ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
        ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
        ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
    ]:
        """Load fonts for timestep labels, row labels, and captions.

        Returns
        -------
        tuple: (timestep_font, row_font, caption_font)
        """
        timestep_font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
        row_font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
        try:
            timestep_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
            row_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except OSError:
            try:
                timestep_font = ImageFont.load_default()
                row_font = ImageFont.load_default()
            except (OSError, AttributeError):
                timestep_font = None
                row_font = None

        caption_font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
        try:
            caption_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        except OSError:
            try:
                caption_font = ImageFont.load_default()
            except (OSError, AttributeError):
                caption_font = None

        return timestep_font, row_font, caption_font

    @staticmethod
    def _convert_frame_to_numpy(frame: Tensor, channels: int) -> np.ndarray:
        """Convert frame tensor to numpy array for PIL processing.

        Args:
            frame: Frame tensor [channels, height, width]
            channels: Number of channels

        Returns
        -------
        np.ndarray: Frame as numpy array [height, width, 3]
        """
        if channels == RGB_CHANNELS:
            frame_np = frame.cpu().detach().permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8).clip(0, 255)
        elif channels == GRAYSCALE_CHANNELS:
            frame_np = frame.cpu().detach().squeeze(0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8).clip(0, 255)
            frame_np = np.stack([frame_np, frame_np, frame_np], axis=2)
        else:
            frame_np = frame.cpu().detach()[:RGB_CHANNELS].permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8).clip(0, 255)
        return frame_np

    @staticmethod
    def _draw_timestep_label(
        draw: ImageDraw.ImageDraw,
        timestep: int,
        total_steps: int,
        side_padding: int,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
    ) -> None:
        """Draw timestep label in top-left corner.

        Args:
            draw: PIL ImageDraw object
            timestep: Current timestep (0-indexed)
            total_steps: Total number of timesteps
            side_padding: Side padding size
            font: Font to use
        """
        timestep_label = f"t={timestep + 1}/{total_steps}"
        timestep_x = side_padding
        timestep_y = 5

        if font:
            timestep_bbox = draw.textbbox((timestep_x, timestep_y), timestep_label, font=font)
        else:
            timestep_bbox = draw.textbbox((timestep_x, timestep_y), timestep_label)

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
        draw.text((timestep_x, timestep_y), timestep_label, fill=(255, 255, 255), font=font)

    @staticmethod
    def _draw_row_labels(
        draw: ImageDraw.ImageDraw,
        side_padding: int,
        top_padding: int,
        vision_y_end: int,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
    ) -> None:
        """Draw row labels (Vision and Audio).

        Args:
            draw: PIL ImageDraw object
            side_padding: Side padding size
            top_padding: Top padding size
            vision_y_end: Y position where vision row ends
            font: Font to use
        """
        row_label_x = side_padding
        vision_label_y = top_padding + 5
        audio_label_y = vision_y_end + 5

        for row_name, row_y in [("Vision", vision_label_y), ("Audio", audio_label_y)]:
            if font:
                row_bbox = draw.textbbox((row_label_x, row_y), row_name, font=font)
            else:
                row_bbox = draw.textbbox((row_label_x, row_y), row_name)

            row_padding = 3
            draw.rectangle(
                [
                    row_bbox[0] - row_padding,
                    row_bbox[1] - row_padding,
                    row_bbox[2] + row_padding,
                    row_bbox[3] + row_padding,
                ],
                fill=(0, 0, 0),
            )
            draw.text((row_label_x, row_y), row_name, fill=(255, 255, 255), font=font)

    @staticmethod
    def _draw_captions(
        draw: ImageDraw.ImageDraw,
        side_padding: int,
        gif_width: int,
        audio_y_end: int,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
    ) -> None:
        """Draw captions below each GIF (below audio row).

        Args:
            draw: PIL ImageDraw object
            side_padding: Side padding size
            gif_width: Width of individual GIF
            audio_y_end: Y position where audio row ends
            font: Font to use
        """
        captions = ["prior", "observation", "posterior"]
        caption_y = audio_y_end + 5

        for i, caption in enumerate(captions):
            gif_center_x = side_padding + gif_width * i + gif_width // 2

            caption_bbox = draw.textbbox((0, 0), caption, font=font) if font else draw.textbbox((0, 0), caption)

            caption_width = caption_bbox[2] - caption_bbox[0]
            caption_x = gif_center_x - caption_width // 2

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
            draw.text((caption_x, caption_y), caption, fill=(255, 255, 255), font=font)

    @staticmethod
    def _convert_pil_to_tensor(
        pil_image: Image.Image,
        channels: int,
        new_height: int,
        new_width: int,
    ) -> Tensor:
        """Convert PIL image back to tensor.

        Args:
            pil_image: PIL Image object
            channels: Expected number of channels
            new_height: New height dimension
            new_width: New width dimension

        Returns
        -------
        Tensor: Frame tensor [channels, height, width]
        """
        frame_array = np.array(pil_image)
        if channels == GRAYSCALE_CHANNELS:
            frame_array = frame_array[:, :, 0:1].transpose(2, 0, 1)
        else:
            frame_array = frame_array.transpose(2, 0, 1)

        frame_tensor = torch.from_numpy(frame_array).float() / 255.0
        if frame_tensor.shape[0] != channels:
            if frame_tensor.shape[0] > channels:
                frame_tensor = frame_tensor[:channels]
            else:
                padding_tensor = torch.zeros(channels - frame_tensor.shape[0], new_height, new_width)
                frame_tensor = torch.cat([frame_tensor, padding_tensor], dim=0)

        return frame_tensor

    def log_multimodal_video(self, batch_video: Tensor, key: str, logger: WandbLogger) -> None:
        """Log multimodal video (vision top row, audio bottom row).

        Args:
            batch_video: Video tensor [batch, time, channels, height*2, width*3]
            key: Key for logging
            logger: WandB logger
        """
        batch = batch_video.cpu().detach().numpy()
        b, t, c, h, w = batch.shape
        # h is actually height*2 (vision + audio)
        vision_h = h // 2
        audio_h = h // 2

        # Split into vision and audio parts
        vision_batch = batch[:, :, :, :vision_h, :]  # [b, t, c, vision_h, w]
        audio_batch = batch[:, :, :, vision_h:, :]  # [b, t, c, audio_h, w]

        # Process vision and audio
        batch_params = {"b": b, "t": t, "c": c, "vision_h": vision_h, "audio_h": audio_h, "w": w}
        combined_rgb = self._process_multimodal_batch(vision_batch, audio_batch, batch_params)

        videos = [torch.from_numpy(combined_rgb[i]) for i in range(b)]
        logger.log_video(key=key, videos=videos, fps=[self.fps] * len(videos), format=["gif"] * len(videos))

    @staticmethod
    def _process_multimodal_batch(
        vision_batch: np.ndarray,
        audio_batch: np.ndarray,
        params: dict[str, int],
    ) -> np.ndarray:
        """Process multimodal batch (vision + audio).

        Args:
            vision_batch: Vision batch [b, t, c, vision_h, w]
            audio_batch: Audio batch [b, t, c, audio_h, w]
            params: Dictionary with b, t, c, vision_h, audio_h, w

        Returns
        -------
        np.ndarray: Combined RGB array [b, t, 3, h, w]
        """
        b, t, c, vision_h, audio_h, w = (
            params["b"],
            params["t"],
            params["c"],
            params["vision_h"],
            params["audio_h"],
            params["w"],
        )

        # Process vision with RGB
        vision_params = {"b": b, "t": t, "c": c, "vision_h": vision_h, "w": w}
        vision_rgb = LogMultimodalMRSSMOutput._process_vision_batch(vision_batch, vision_params)

        # Process audio with magma colormap
        audio_params = {"b": b, "t": t, "c": c, "audio_h": audio_h, "w": w}
        audio_rgb = LogMultimodalMRSSMOutput._process_audio_batch(audio_batch, audio_params)

        # Combine vision and audio vertically
        return np.concatenate([vision_rgb, audio_rgb], axis=3)  # [b, t, 3, h, w]

    @staticmethod
    def _process_vision_batch(vision_batch: np.ndarray, params: dict[str, int]) -> np.ndarray:
        """Process vision batch with RGB conversion.

        Args:
            vision_batch: Vision batch [b, t, c, vision_h, w]
            params: Dictionary with b, t, c, vision_h, w

        Returns
        -------
        np.ndarray: Vision RGB array [b, t, 3, vision_h, w]
        """
        b, t, c, vision_h, w = params["b"], params["t"], params["c"], params["vision_h"], params["w"]
        vision_rgb = np.zeros((b, t, RGB_CHANNELS, vision_h, w), dtype=np.uint8)
        for i in range(b):
            for j in range(t):
                frame = vision_batch[i, j]  # [c, vision_h, w]
                frame_rgb = LogMultimodalMRSSMOutput._convert_frame_to_rgb(frame, c)
                vision_rgb[i, j] = (frame_rgb * 255).astype(np.uint8).clip(0, 255).transpose(2, 0, 1)
        return vision_rgb

    @staticmethod
    def _convert_frame_to_rgb(frame: np.ndarray, c: int) -> np.ndarray:
        """Convert frame to RGB format.

        Args:
            frame: Frame array [c, h, w]
            c: Number of channels

        Returns
        -------
        np.ndarray: RGB frame [h, w, 3]
        """
        if c == RGB_CHANNELS:
            return frame[:RGB_CHANNELS].transpose(1, 2, 0)  # [h, w, 3]
        if c == GRAYSCALE_CHANNELS:
            return np.stack([frame[0], frame[0], frame[0]], axis=2)  # [h, w, 3]
        return frame[:RGB_CHANNELS].transpose(1, 2, 0)  # [h, w, 3]

    @staticmethod
    def _process_audio_batch(audio_batch: np.ndarray, params: dict[str, int]) -> np.ndarray:
        """Process audio batch with magma colormap.

        Args:
            audio_batch: Audio batch [b, t, c, audio_h, w]
            params: Dictionary with b, t, c, audio_h, w

        Returns
        -------
        np.ndarray: Audio RGB array [b, t, 3, audio_h, w]
        """
        b, t, c, audio_h, w = params["b"], params["t"], params["c"], params["audio_h"], params["w"]
        magma = colormaps["magma"]
        audio_rgb = np.zeros((b, t, RGB_CHANNELS, audio_h, w), dtype=np.uint8)
        for i in range(b):
            audio_video = audio_batch[i]  # [t, c, audio_h, w]
            audio_video = (
                audio_video.squeeze(1) if c == GRAYSCALE_CHANNELS else audio_video[:, 0, :, :]
            )  # [t, audio_h, w]

            audio_normalized = audio_video * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            audio_db = (audio_normalized + 1.0) / 2.0 * 80.0 - 80.0  # [-1, 1] -> [-80, 0] dB

            # Check if combined (3 parts side by side)
            is_combined = w % GIF_PARTS == 0 and w > audio_h
            if is_combined:
                audio_rgb[i] = LogMultimodalMRSSMOutput._process_combined_audio(
                    audio_db,
                    t,
                    audio_h,
                    w,
                    magma,
                )
            else:
                audio_rgb[i] = LogMultimodalMRSSMOutput._process_single_audio(
                    audio_db,
                    t,
                    audio_h,
                    w,
                    magma,
                )
        return audio_rgb

    @staticmethod
    def _process_combined_audio(
        audio_db: np.ndarray,
        t: int,
        audio_h: int,
        w: int,
        magma: Colormap,
    ) -> np.ndarray:
        """Process combined audio (3 parts side by side).

        Args:
            audio_db: Audio in dB [t, audio_h, w]
            t: Time steps
            audio_h: Audio height
            w: Width
            magma: Magma colormap

        Returns
        -------
        np.ndarray: Audio RGB array [t, 3, audio_h, w]
        """
        part_width = w // GIF_PARTS
        audio_rgb = np.zeros((t, RGB_CHANNELS, audio_h, w), dtype=np.uint8)
        vmin, vmax = -80.0, 0.0

        for part_idx in range(GIF_PARTS):
            start_w = part_idx * part_width
            end_w = (part_idx + 1) * part_width
            part_db = audio_db[:, :, start_w:end_w]  # [t, audio_h, part_width]

            part_clipped = np.clip(part_db, vmin, vmax)
            part_scaled = (part_clipped - vmin) / (vmax - vmin)
            part_scaled = part_scaled.clip(0, 1)

            for j in range(t):
                frame_scaled = part_scaled[j]  # [audio_h, part_width]
                rgba = magma(frame_scaled)[:, :, :RGB_CHANNELS]
                audio_rgb[j, :, :, start_w:end_w] = (rgba * 255).astype(np.uint8).transpose(2, 0, 1)
        return audio_rgb

    @staticmethod
    def _process_single_audio(
        audio_db: np.ndarray,
        t: int,
        audio_h: int,
        w: int,
        magma: Colormap,
    ) -> np.ndarray:
        """Process single audio (not combined).

        Args:
            audio_db: Audio in dB [t, audio_h, w]
            t: Time steps
            audio_h: Audio height
            w: Width
            magma: Magma colormap

        Returns
        -------
        np.ndarray: Audio RGB array [t, 3, audio_h, w]
        """
        vmin, vmax = -80.0, 0.0
        audio_clipped = np.clip(audio_db, vmin, vmax)
        audio_scaled = (audio_clipped - vmin) / (vmax - vmin)
        audio_scaled = audio_scaled.clip(0, 1)

        audio_rgb = np.zeros((t, RGB_CHANNELS, audio_h, w), dtype=np.uint8)
        for j in range(t):
            frame_scaled = audio_scaled[j]  # [audio_h, w]
            rgba = magma(frame_scaled)[:, :, :RGB_CHANNELS]
            audio_rgb[j] = (rgba * 255).astype(np.uint8).transpose(2, 0, 1)
        return audio_rgb
