"""Data transforms shared by RSSM variants."""

import torch
from numpy.random import MT19937, Generator
from torch import Tensor


class RemoveDim:
    """Remove dimension transform."""

    def __init__(self, axis: int, indices_to_remove: list[int]) -> None:
        """Initialize the remove dimension transform."""
        self.axis = axis
        self.remove = indices_to_remove

    def __call__(self, data: Tensor) -> Tensor:
        """Remove dimension transform.

        Args:
            data: The action tensor.

        Returns
        -------
        Tensor: The removed dimension action tensor.
        """
        all_indices = list(range(data.size(self.axis)))
        keep = [i for i in all_indices if i not in self.remove]
        return torch.index_select(data, self.axis, torch.tensor(keep))


class TakeFirstN:
    """Take first N timesteps transform."""

    def __init__(self, n: int) -> None:
        """Initialize the take first N transform.

        Args:
            n: Number of timesteps to take from the beginning.
        """
        self.n = n

    def __call__(self, data: Tensor) -> Tensor:
        """Take first N timesteps transform.

        Args:
            data: The tensor with time dimension as first dimension.

        Returns
        -------
        Tensor: The tensor with first N timesteps.
        """
        return data[: self.n]


class GaussianNoise:
    """Gaussian noise transform."""

    def __init__(self, std: float = 0.1) -> None:
        """Initialize the gaussian noise transform."""
        self.std = std

    def __call__(self, data: Tensor) -> Tensor:
        """Gaussian noise transform.

        Args:
            data: The action tensor.

        Returns
        -------
        Tensor: The gaussian noise action tensor.
        """
        return data + torch.randn_like(data) * self.std


class NormalizeVisionImage:
    """Normalize vision image from [0, 255] to [-1, 1].

    This transform normalizes vision image data in two steps:
    1. Divide by 255: [0, 255] -> [0, 1]
    2. Scale to [-1, 1]: [0, 1] -> [-1, 1] using (x * 2.0 - 1.0)
    """

    def __call__(self, data: Tensor) -> Tensor:
        """Normalize vision image transform.

        Args:
            data: The vision image tensor in [0, 255] range.

        Returns
        -------
        Tensor: The normalized vision image tensor in [-1, 1] range.
        """
        copy_data = data.detach().clone()
        # Step 1: [0, 255] -> [0, 1]
        copy_data /= 255.0
        # Step 2: [0, 1] -> [-1, 1]
        return copy_data * 2.0 - 1.0


class NormalizeAudioMelSpectrogram:
    """Normalize audio mel-spectrogram from [min, max] to [-1, 1].

    This transform normalizes audio mel-spectrogram data using Min-Max normalization:
    1. Shift by min: data = data - min
    2. Scale to [-1, 1]: data = (data / (max - min) * 2.0 - 1.0)

    Args:
        min_value: Minimum value of the original data range (default: -80.0).
        max_value: Maximum value of the original data range (default: 0.1).
    """

    def __init__(self, min_value: float = -80.0, max_value: float = 0.1) -> None:
        """Initialize the normalize audio mel-spectrogram transform."""
        self.min_value = min_value
        self.max_value = max_value
        self.range = max_value - min_value

    def __call__(self, data: Tensor) -> Tensor:
        """Normalize audio mel-spectrogram transform.

        Args:
            data: The audio mel-spectrogram tensor.

        Returns
        -------
        Tensor: The normalized audio mel-spectrogram tensor in [-1, 1] range.
        """
        copy_data = data.detach().clone()
        # Shift by min: [min, max] -> [0, max-min]
        copy_data -= self.min_value
        # Scale to [-1, 1]: [0, max-min] -> [-1, 1]
        return (copy_data / self.range) * 2.0 - 1.0
