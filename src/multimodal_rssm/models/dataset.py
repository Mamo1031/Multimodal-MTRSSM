"""Unified DataModule base classes for RSSM variants."""

import re
import sys
import tarfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import gdown
import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

Transform: TypeAlias = Callable[[Tensor], Tensor]

# Tensor dimension constants
OBSERVATION_DIM_5D = 5  # (N,T,H,W,C)
OBSERVATION_DIM_4D = 4  # (N,T,H,W)


def convert_gdrive_url(url: str) -> str:
    """Convert a Google Drive URL to a direct download URL.

    Args:
        url: The Google Drive URL to convert.

    Returns
    -------
    str: The direct download URL.
    """
    if url.startswith("https://drive.google.com/uc?id="):
        return url
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?id={file_id}"
    return url


def load_tensor(path: Path) -> Tensor:
    """Load a tensor from a file.

    Args:
        path: The path to the file.

    Returns
    -------
    Tensor
        The loaded tensor.

    Raises
    ------
    ValueError
        If the file extension is not `.npy` or `.pt`.
    """
    if path.suffix == ".npy":
        return torch.Tensor(np.load(path))
    if path.suffix == ".pt" and isinstance(tensor := torch.load(path, weights_only=False), Tensor):
        return tensor
    msg = f"Unknown file extension: {path.suffix}"
    raise ValueError(msg)


def split_path_list(path_list: list[Path], train_ratio: float) -> tuple[list[Path], list[Path]]:
    """Split a list of paths into train and validation paths.

    Args:
        path_list: The list of paths to split.
        train_ratio: The ratio of train paths.

    Returns
    -------
    tuple[list[Path], list[Path]]: The train and validation paths.
    """
    split_point = int(len(path_list) * train_ratio)
    return path_list[:split_point], path_list[split_point:]


class EpisodeDataset(Dataset[Tensor]):
    """Dataset for the EpisodeDataModule."""

    def __init__(self, path_list: list[Path], transform: Transform) -> None:
        super().__init__()
        self.path_list = path_list
        self.transform = transform

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int: The length of the dataset.
        """
        return len(self.path_list)

    def __getitem__(self, idx: int) -> Tensor:
        """Get the item at the given index.

        Args:
            idx: The index of the item.

        Returns
        -------
        Tensor: The item at the given index.
        """
        return self.transform(load_tensor(self.path_list[idx]))


@dataclass
class BaseEpisodeDataModuleConfig(ABC):
    """Base configuration for the EpisodeDataModule."""

    data_name: str
    batch_size: int
    num_workers: int
    gdrive_url: str
    action_preprocess: Transform
    action_input_transform: Transform
    action_target_transform: Transform

    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return Path("data") / self.data_name

    @property
    def processed_data_dir(self) -> Path:
        """Get the processed data directory."""
        return Path("data") / f"processed_{self.data_name}"

    def get_effective_processed_data_dir(self, observation_patterns: list[str]) -> Path:
        """Get the effective processed data directory, checking for processed_data first.

        If processed_data exists and contains the required files, use it.
        Otherwise, use the default processed_data_dir.

        Args:
            observation_patterns: List of glob patterns for observation files
                (e.g., ['obs*'] or ['audio_obs*', 'vision_obs*'])

        Returns
        -------
        Path: The effective processed data directory to use
        """
        # Check if processed_data exists and has required files
        common_processed_dir = Path("data") / "processed_data"
        if common_processed_dir.exists():
            has_actions = bool(list(common_processed_dir.glob("act*")))
            has_all_observations = all(
                bool(list(common_processed_dir.glob(pattern))) for pattern in observation_patterns
            )
            if has_actions and has_all_observations:
                return common_processed_dir

        # Fall back to default processed_data_dir
        return self.processed_data_dir

    def load_from_gdrive(self) -> None:
        """Load the data from Google Drive."""
        url = convert_gdrive_url(self.gdrive_url)
        filename = gdown.download(url, quiet=False, fuzzy=True)
        with tarfile.open(filename, "r:gz") as f:
            f.extractall(path=Path("data"), filter="data")
        Path(filename).unlink(missing_ok=False)

    @abstractmethod
    def get_observation_file_names(self) -> list[str]:
        """Get the list of observation file names.

        Returns
        -------
        list[str]: The list of observation file names.
        """

    @classmethod
    def get_observation_glob_patterns(cls) -> list[str]:
        """Get the list of glob patterns for processed observation files.

        Subclasses must implement this method.
        """
        msg = "Subclasses must implement get_observation_glob_patterns"
        raise NotImplementedError(msg)


class BaseEpisodeDataModule(LightningDataModule):
    """Base DataModule for the EpisodeDataset."""

    train_dataset: Dataset[tuple[Tensor, ...]] | None = None
    val_dataset: Dataset[tuple[Tensor, ...]] | None = None

    def __init__(self, config: BaseEpisodeDataModuleConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def _find_data_paths(self) -> tuple[Path, ...] | tuple[Path, Path, bool] | tuple[Path, Path, Path, bool]:
        """Find the paths to observation and action data.

        Returns
        -------
        tuple: The observation path(s), action path, and whether local data exists.
            For single observation: (obs_path, act_path, bool)
            For multimodal: (audio_obs_path, vision_obs_path, act_path, bool)
        """

    def _ensure_data_exists(self, *, has_local: bool) -> None:
        """Ensure that data files exist, downloading if necessary.

        Args:
            has_local: Whether local data exists.
        """
        if not has_local and not self.config.data_dir.exists():
            self.config.load_from_gdrive()

    @abstractmethod
    def _is_processed_data_ready(self) -> bool:
        """Check if processed data already exists.

        Returns
        -------
        bool
            True if processed data exists, False otherwise.
        """

    @staticmethod
    def _normalize_observation_shape(observations: Tensor) -> Tensor:
        """Normalize observation tensor shape to (N,T,C,H,W).

        Args:
            observations: The observation tensor.

        Returns
        -------
        Tensor
            The normalized observation tensor.
        """
        if observations.dim() == OBSERVATION_DIM_5D:
            # (N,T,H,W,C) -> (N,T,C,H,W)
            return observations.permute(0, 1, 4, 2, 3)
        if observations.dim() == OBSERVATION_DIM_4D:
            # (N,T,H,W) -> (N,T,1,H,W)
            return observations.unsqueeze(2)
        return observations

    @abstractmethod
    def _process_episode_data(self, *args: Path) -> None:
        """Process episode data from observation and action files.

        Args:
            *args: The paths to observation and action data.
                For single observation: (obs_path, act_path)
                For multimodal: (audio_obs_path, vision_obs_path, act_path)
        """

    @abstractmethod
    def _process_individual_files(self) -> None:
        """Process individual action and observation files."""

    def prepare_data(self) -> None:
        """Prepare the data."""
        if self._is_processed_data_ready():
            return

        paths_result = self._find_data_paths()
        has_local = paths_result[-1]  # Last element is always the boolean

        # Only try to download if we don't have local data and processed data doesn't exist
        if not has_local and not self.config.data_dir.exists():
            try:
                self.config.load_from_gdrive()
            except Exception:
                # If download fails, check if processed data exists anyway
                if self._is_processed_data_ready():
                    return
                # Check if raw data exists in data directory
                if not self.config.data_dir.exists():
                    # No data available, raise a more informative error
                    observation_files = ", ".join(self.config.get_observation_file_names())
                    error_msg = (
                        "\n"
                        + "=" * 80
                        + "\nERROR: Failed to download data from Google Drive."
                        + "\n"
                        + "=" * 80
                        + "\nPossible solutions:"
                        + "\n1. Check Google Drive URL and permissions in config file:"
                        + f"\n   {self.config.gdrive_url}"
                        + "\n2. Manually download data and place it in:"
                        + f"\n   {self.config.data_dir}"
                        + "\n   Required files:"
                        + f"\n   - {observation_files}"
                        + "\n   - joint_states.npy"
                        + "\n3. If processed data already exists, ensure it's in:"
                        + f"\n   {self.config.processed_data_dir}"
                        + "\n"
                        + "=" * 80
                        + "\n"
                    )
                    print(error_msg, file=sys.stderr)  # noqa: T201
                    raise

        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Check if episode files exist (all paths except the last boolean)
        paths = paths_result[:-1]

        if all(p.exists() for p in paths):
            self._process_episode_data(*paths)
        else:
            self._process_individual_files()

    @abstractmethod
    def setup(self, stage: str = "fit") -> None:
        """Set up the data."""

    def train_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """Get the train dataloader.

        Returns
        -------
        DataLoader[tuple[Tensor, ...]]: The train dataloader.

        Raises
        ------
        RuntimeError: If train_dataset is not set.
        """
        if self.train_dataset is None:
            msg = "train_dataset is not set. Call setup() first."
            raise RuntimeError(msg)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader[tuple[Tensor, ...]]: The validation dataloader.

        Raises
        ------
        RuntimeError: If val_dataset is not set.
        """
        if self.val_dataset is None:
            msg = "val_dataset is not set. Call setup() first."
            raise RuntimeError(msg)
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )
