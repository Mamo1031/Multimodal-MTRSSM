#!/usr/bin/env python3
"""Convert audio-mnist-generator .npz files to MoPoE-MRSSM format."""

import argparse
from pathlib import Path

import numpy as np
import torch


def convert_npz_to_pt(
    source_dir: Path,
    output_dir: Path,
    start_idx: int = 0,
) -> None:
    """Convert .npz files to .pt files for MoPoE-MRSSM.

    Args:
        source_dir: Directory containing .npz files
        output_dir: Output directory for .pt files
        start_idx: Starting index for file numbering
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(source_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} .npz files in {source_dir}")

    for local_idx, npz_path in enumerate(npz_files):
        # Load .npz file
        data = np.load(npz_path)

        # Extract data
        audio = data["audio"]  # (180, 32, 32)
        image = data["image"]  # (180, 1, 32, 32)
        speaker = data["speaker"]  # (180, 6)

        # Convert audio: (180, 32, 32) -> (180, 1, 32, 32)
        audio = audio[:, np.newaxis, :, :]  # Add channel dimension

        # Convert to torch tensors
        audio_tensor = torch.from_numpy(audio).float()
        image_tensor = torch.from_numpy(image).float()
        speaker_tensor = torch.from_numpy(speaker).float()

        # Save as .pt files (use global index)
        file_idx = start_idx + local_idx
        file_name = f"{file_idx:04d}"
        torch.save(speaker_tensor, output_dir / f"act_{file_name}.pt")
        torch.save(audio_tensor, output_dir / f"audio_obs_{file_name}.pt")
        torch.save(image_tensor, output_dir / f"vision_obs_{file_name}.pt")

        if (local_idx + 1) % 100 == 0:
            print(f"Processed {local_idx + 1}/{len(npz_files)} files...")

    print(f"Converted {len(npz_files)} files to {output_dir}")
    return len(npz_files)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert audio-mnist-generator .npz files to MoPoE-MRSSM format")
    parser.add_argument(
        "--source-train",
        type=Path,
        default=Path("../../audio-mnist-generator/data/train"),
        help="Source directory for training .npz files",
    )
    parser.add_argument(
        "--source-test",
        type=Path,
        default=Path("../../audio-mnist-generator/data/test"),
        help="Source directory for test .npz files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../data/audio_mnist"),
        help="Output directory for converted .pt files",
    )
    args = parser.parse_args()

    # Convert train data
    print(f"Converting train data from {args.source_train} to {args.output_dir}")
    train_count = convert_npz_to_pt(args.source_train, args.output_dir, start_idx=0)

    # Convert test data (continue numbering from train)
    print(f"Converting test data from {args.source_test} to {args.output_dir}")
    convert_npz_to_pt(args.source_test, args.output_dir, start_idx=train_count)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
