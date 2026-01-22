"""Common DataModule for multimodal MRSSM variants."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import StackDataset
from tqdm import tqdm

from multimodal_rssm.models.dataset import (
    BaseEpisodeDataModule,
    BaseEpisodeDataModuleConfig,
    EpisodeDataset,
    Transform,
    load_tensor,
    split_path_list,
)


@dataclass
class EpisodeDataModuleConfig(BaseEpisodeDataModuleConfig):
    """Configuration for multimodal MRSSM EpisodeDataModule."""

    audio_observation_file_name: str
    vision_observation_file_name: str
    audio_observation_preprocess: Transform
    vision_observation_preprocess: Transform
    audio_observation_input_transform: Transform
    audio_observation_target_transform: Transform
    vision_observation_input_transform: Transform
    vision_observation_target_transform: Transform

    def get_observation_file_names(self) -> list[str]:
        """Get the list of observation file names.

        Returns
        -------
        list[str]: The list of observation file names.
        """
        return [self.audio_observation_file_name, self.vision_observation_file_name]

    @staticmethod
    def get_observation_glob_patterns() -> list[str]:
        """Get the list of glob patterns for processed observation files.

        Returns
        -------
        list[str]: The list of glob patterns.
        """
        return ["audio_obs*", "vision_obs*"]


class EpisodeDataModule(BaseEpisodeDataModule):
    """DataModule for multimodal MRSSM EpisodeDataset."""

    def __init__(self, config: EpisodeDataModuleConfig) -> None:
        """Initialize the EpisodeDataModule.

        Args:
            config: The configuration for the EpisodeDataModule.
        """
        super().__init__(config)

    def _find_data_paths(self) -> tuple[Path, Path, Path, bool]:
        """Find the paths to audio observation, vision observation, and action data.

        Returns
        -------
        tuple[Path, Path, Path, bool]
            The audio observation path, vision observation path, action path, and whether local data exists.
        """
        config = cast("EpisodeDataModuleConfig", self.config)
        data_root = Path("data")
        audio_obs_root = data_root / config.audio_observation_file_name
        vision_obs_root = data_root / config.vision_observation_file_name
        act_root = data_root / "joint_states.npy"

        audio_obs_dd = config.data_dir / config.audio_observation_file_name
        vision_obs_dd = config.data_dir / config.vision_observation_file_name
        act_dd = config.data_dir / "joint_states.npy"

        if audio_obs_root.exists() and vision_obs_root.exists() and act_root.exists():
            return audio_obs_root, vision_obs_root, act_root, True
        if audio_obs_dd.exists() and vision_obs_dd.exists() and act_dd.exists():
            return audio_obs_dd, vision_obs_dd, act_dd, True
        return audio_obs_dd, vision_obs_dd, act_dd, False

    def _is_processed_data_ready(self) -> bool:
        """Check if processed data already exists.

        Returns
        -------
        bool
            True if processed data exists, False otherwise.
        """
        effective_dir = self.config.get_effective_processed_data_dir(self.config.get_observation_glob_patterns())
        if not effective_dir.exists():
            return False
        has_actions = bool(list(effective_dir.glob("act*")))
        has_audio_observations = bool(list(effective_dir.glob("audio_obs*")))
        has_vision_observations = bool(list(effective_dir.glob("vision_obs*")))
        return has_actions and has_audio_observations and has_vision_observations

    def _process_episode_data(self, audio_obs_path: Path, vision_obs_path: Path, act_path: Path) -> None:  # type: ignore[override]
        """Process episode data from audio observation, vision observation, and action files.

        Args:
            audio_obs_path: The path to audio observation data.
            vision_obs_path: The path to vision observation data.
            act_path: The path to action data.
        """
        audio_observations = load_tensor(audio_obs_path)  # (N,T,H,W)
        vision_observations = load_tensor(vision_obs_path)  # (N,T,H,W,C)
        actions = load_tensor(act_path)  # (N,T,A)

        audio_observations = self._normalize_observation_shape(audio_observations)
        vision_observations = self._normalize_observation_shape(vision_observations)
        num_episodes = audio_observations.shape[0]

        config = cast("EpisodeDataModuleConfig", self.config)
        for i in tqdm(range(num_episodes)):
            action = config.action_preprocess(actions[i])
            audio_observation = config.audio_observation_preprocess(audio_observations[i])  # type: ignore[attr-defined,assignment]
            vision_observation = config.vision_observation_preprocess(vision_observations[i])  # type: ignore[attr-defined,assignment]
            torch.save(action.detach().clone(), config.processed_data_dir / f"act_{i:03d}.pt")
            torch.save(
                audio_observation.detach().clone(),
                config.processed_data_dir / f"audio_obs_{i:03d}.pt",
            )
            torch.save(
                vision_observation.detach().clone(),
                config.processed_data_dir / f"vision_obs_{i:03d}.pt",
            )

    def _process_individual_files(self) -> None:
        """Process individual action and observation files."""
        config = cast("EpisodeDataModuleConfig", self.config)
        for action_path in tqdm(sorted(config.data_dir.glob("act*"))):
            action = config.action_preprocess(load_tensor(action_path))
            torch.save(action.detach().clone(), config.processed_data_dir / f"{action_path.stem}.pt")
        for audio_observation_path in tqdm(sorted(config.data_dir.glob("audio_obs*"))):
            audio_observation = config.audio_observation_preprocess(load_tensor(audio_observation_path))  # type: ignore[attr-defined,assignment]
            torch.save(
                audio_observation.detach().clone(),
                config.processed_data_dir / f"{audio_observation_path.stem}.pt",
            )
        for vision_observation_path in tqdm(sorted(config.data_dir.glob("vision_obs*"))):
            vision_observation = config.vision_observation_preprocess(load_tensor(vision_observation_path))  # type: ignore[attr-defined,assignment]
            torch.save(
                vision_observation.detach().clone(),
                config.processed_data_dir / f"{vision_observation_path.stem}.pt",
            )

    def setup(self, stage: str = "fit") -> None:
        """Set up the data."""
        config = cast("EpisodeDataModuleConfig", self.config)
        effective_dir = config.get_effective_processed_data_dir(config.get_observation_glob_patterns())
        action_path_list = sorted(effective_dir.glob("act*"))
        audio_observation_path_list = sorted(effective_dir.glob("audio_obs*"))
        vision_observation_path_list = sorted(effective_dir.glob("vision_obs*"))

        train_action_list, val_action_list = split_path_list(action_path_list, 0.8)
        train_audio_observation_list, val_audio_observation_list = split_path_list(audio_observation_path_list, 0.8)
        train_vision_observation_list, val_vision_observation_list = split_path_list(vision_observation_path_list, 0.8)

        if stage == "fit":
            self.train_dataset = StackDataset(
                EpisodeDataset(train_action_list, config.action_input_transform),
                EpisodeDataset(train_audio_observation_list, config.audio_observation_input_transform),
                EpisodeDataset(train_vision_observation_list, config.vision_observation_input_transform),
                EpisodeDataset(train_action_list, config.action_target_transform),
                EpisodeDataset(train_audio_observation_list, config.audio_observation_target_transform),
                EpisodeDataset(train_vision_observation_list, config.vision_observation_target_transform),
            )
        self.val_dataset = StackDataset(
            EpisodeDataset(val_action_list, config.action_input_transform),
            EpisodeDataset(val_audio_observation_list, config.audio_observation_input_transform),
            EpisodeDataset(val_vision_observation_list, config.vision_observation_input_transform),
            EpisodeDataset(val_action_list, config.action_target_transform),
            EpisodeDataset(val_audio_observation_list, config.audio_observation_target_transform),
            EpisodeDataset(val_vision_observation_list, config.vision_observation_target_transform),
        )
