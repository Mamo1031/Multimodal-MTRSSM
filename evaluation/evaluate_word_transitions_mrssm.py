#!/usr/bin/env python3
"""Evaluate word-to-word transitions using MoPoE-MRSSM."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.mnist_classifier import load_pretrained_classifier, recognize_digit
from multimodal_rssm.models.mrssm.mopoe_mrssm.core import MoPoE_MRSSM
from multimodal_rssm.models.transform import NormalizeAudioMelSpectrogram, NormalizeVisionImage


def load_test_data_with_labels(
    test_data_dir: Path,
    use_pt_files: bool = False,
    npz_dir_for_labels: Path | None = None,
) -> list[dict]:
    """Load test data with labels from .npz files or .pt files.

    Args:
        test_data_dir: Directory containing test data files (.npz or .pt).
        use_pt_files: If True, load from .pt files and get labels from .npz files.
        npz_dir_for_labels: Directory containing .npz files for labels (if use_pt_files=True).

    Returns:
        List of dictionaries containing:
        - 'audio': audio data (180, 32, 32) or (180, 1, 32, 32)
        - 'image': image data (180, 1, 32, 32)
        - 'label': label data (180,)
        - 'speaker': speaker data (180, 6)
        - 'file_path': path to the data file
    """
    test_data = []

    # Resolve path
    test_data_dir = Path(test_data_dir).resolve()

    if not test_data_dir.exists():
        print(f"Warning: Test data directory does not exist: {test_data_dir}")
        return test_data

    if use_pt_files:
        # Load from .pt files
        act_files = sorted(test_data_dir.glob("act_*.pt"))

        if len(act_files) == 0:
            print(f"Warning: No act_*.pt files found in {test_data_dir}")
            return test_data

        # Get corresponding npz directory for labels
        if npz_dir_for_labels is None:
            # Try to find npz files in the same directory structure
            # Assume original npz files are in audio-mnist-generator
            npz_dir_for_labels = Path(__file__).parent.parent.parent / "audio-mnist-generator" / "data" / "test"

        npz_dir_for_labels = Path(npz_dir_for_labels).resolve()

        print(f"Loading .pt files from {test_data_dir}")
        print(f"Loading labels from .npz files in {npz_dir_for_labels}")

        for act_path in act_files:
            try:
                # Extract index from filename (e.g., act_0000.pt -> 0)
                file_idx = int(act_path.stem.split("_")[1])

                # Load .pt files
                audio_path = test_data_dir / f"audio_obs_{file_idx:04d}.pt"
                vision_path = test_data_dir / f"vision_obs_{file_idx:04d}.pt"

                if not audio_path.exists() or not vision_path.exists():
                    continue

                audio_tensor = torch.load(audio_path, weights_only=False)  # (180, 1, 32, 32)
                image_tensor = torch.load(vision_path, weights_only=False)  # (180, 1, 32, 32)
                speaker_tensor = torch.load(act_path, weights_only=False)  # (180, 6)

                # Convert to numpy
                # Remove channel dimension from audio if present: (180, 1, 32, 32) -> (180, 32, 32)
                if audio_tensor.dim() == 4 and audio_tensor.shape[1] == 1:
                    audio = audio_tensor.squeeze(1).numpy()  # (180, 32, 32)
                else:
                    audio = audio_tensor.numpy()

                image = image_tensor.numpy()  # (180, 1, 32, 32)
                speaker = speaker_tensor.numpy()  # (180, 6)

                # Load labels from corresponding .npz file
                # Map file index to original npz file index
                # Since train data comes first, we need to find the corresponding npz file
                # Try both train and test directories
                npz_paths = []
                if (npz_dir_for_labels.parent / "train").exists():
                    npz_paths.append(npz_dir_for_labels.parent / "train" / f"sample_{file_idx:04d}.npz")
                npz_paths.append(npz_dir_for_labels / f"sample_{file_idx:04d}.npz")

                label = None
                for npz_path in npz_paths:
                    if npz_path.exists():
                        npz_data = np.load(npz_path)
                        label = npz_data["label"]  # (180,)
                        break

                if label is None:
                    # If label not found, skip this file
                    continue

                test_data.append({
                    "audio": audio,  # (180, 32, 32)
                    "image": image,  # (180, 1, 32, 32)
                    "label": label,  # (180,)
                    "speaker": speaker,  # (180, 6)
                    "file_path": act_path,
                })
            except Exception as e:
                print(f"Warning: Failed to load {act_path}: {e}")
                continue
    else:
        # Load from .npz files (original method)
        npz_files = sorted(test_data_dir.glob("*.npz"))

        if len(npz_files) == 0:
            print(f"Warning: No .npz files found in {test_data_dir}")
            return test_data

        for npz_path in npz_files:
            try:
                data = np.load(npz_path)
                test_data.append({
                    "audio": data["audio"],  # (180, 32, 32)
                    "image": data["image"],  # (180, 1, 32, 32)
                    "label": data["label"],  # (180,)
                    "speaker": data["speaker"],  # (180, 6)
                    "file_path": npz_path,
                })
            except Exception as e:
                print(f"Warning: Failed to load {npz_path}: {e}")
                continue

    return test_data


def get_speaker_index(speaker_onehot: np.ndarray) -> int:
    """Get speaker index from one-hot vector.

    Args:
        speaker_onehot: One-hot vector of shape (6,).

    Returns:
        Speaker index (0-5).
    """
    return int(np.argmax(speaker_onehot[0]))


def select_intervals_for_word(
    word: int,
    test_data: list[dict],
    n_intervals: int = 6,
    query_length: int = 30,
) -> list[dict]:
    """Select observation intervals containing a specific word.

    Args:
        word: Word to search for (0-9).
        test_data: List of test data dictionaries.
        n_intervals: Number of intervals to select.
        query_length: Length of observation interval to use.

    Returns:
        List of selected intervals, each containing:
        - 'audio': audio observation (query_length, 32, 32)
        - 'image': image observation (query_length, 1, 32, 32)
        - 'speaker': speaker action (query_length, 6)
        - 'label': label sequence (query_length,)
        - 'speaker_idx': speaker index
        - 'file_idx': file index in test_data
    """
    selected_intervals = []
    speaker_used = set()

    for file_idx, data in enumerate(test_data):
        labels = data["label"]  # Use all 180 frames

        # Find positions where word appears (use np.where for numpy arrays)
        word_positions = np.where(labels == word)[0]
        if len(word_positions) == 0:
            continue

        # Use the first occurrence that allows us to extract a full query_length interval
        # We want the interval to end at or after the word position
        word_pos = word_positions[0]

        # Get speaker index
        speaker_idx = get_speaker_index(data["speaker"])

        # Skip if we already have an interval from this speaker
        if speaker_idx in speaker_used:
            continue

        # Extract interval ending at word position (or use first query_length frames)
        # Ensure we have enough frames before the word
        start_idx = max(0, word_pos - query_length + 1)
        end_idx = start_idx + query_length

        # If we don't have enough frames, use the first query_length frames
        if end_idx > len(labels):
            start_idx = 0
            end_idx = query_length

        interval = {
            "audio": data["audio"][start_idx:end_idx],  # (query_length, 32, 32)
            "image": data["image"][start_idx:end_idx],  # (query_length, 1, 32, 32)
            "speaker": data["speaker"][start_idx:end_idx],  # (query_length, 6)
            "label": data["label"][start_idx:end_idx],  # (query_length,)
            "speaker_idx": speaker_idx,
            "file_idx": file_idx,
        }

        selected_intervals.append(interval)
        speaker_used.add(speaker_idx)

        if len(selected_intervals) >= n_intervals:
            break

    return selected_intervals


def apply_transforms(
    audio: np.ndarray,
    image: np.ndarray,
    audio_transform: NormalizeAudioMelSpectrogram,
    vision_transform: NormalizeVisionImage,
) -> tuple[Tensor, Tensor]:
    """Apply transforms to audio and image data.

    Args:
        audio: Audio data of shape (T, 1, 32, 32) or (T, 32, 32).
        image: Image data of shape (T, 1, 32, 32).
        audio_transform: Audio normalization transform.
        vision_transform: Vision normalization transform.

    Returns:
        Tuple of (audio_tensor, image_tensor) with transforms applied.
    """
    # Convert to tensors
    audio_tensor = torch.from_numpy(audio).float()
    image_tensor = torch.from_numpy(image).float()

    # Apply transforms per timestep
    # Transform expects (H, W) or (C, H, W) input
    T = audio_tensor.shape[0]
    audio_transformed_list = []
    for t in range(T):
        audio_t = audio_tensor[t]  # (C, H, W) or (H, W)
        if audio_t.dim() == 3:
            # (C, H, W) -> apply to each channel or flatten
            audio_t_flat = audio_t.view(-1, *audio_t.shape[1:])  # (C, H, W)
            audio_t_transformed = audio_transform(audio_t_flat)
            audio_transformed_list.append(audio_t_transformed.view(*audio_t.shape))
        else:
            audio_transformed_list.append(audio_transform(audio_t))
    audio_tensor = torch.stack(audio_transformed_list, dim=0)

    image_transformed_list = []
    for t in range(T):
        image_t = image_tensor[t]  # (C, H, W)
        if image_t.dim() == 3:
            image_t_flat = image_t.view(-1, *image_t.shape[1:])  # (C, H, W)
            image_t_transformed = vision_transform(image_t_flat)
            image_transformed_list.append(image_t_transformed.view(*image_t.shape))
        else:
            image_transformed_list.append(vision_transform(image_t))
    image_tensor = torch.stack(image_transformed_list, dim=0)

    return audio_tensor, image_tensor


def generate_predictions_with_classifier(
    model: MoPoE_MRSSM,
    classifier: torch.nn.Module,
    observation_interval: dict,
    n_predictions: int = 10,
    n_frames: int = 10,
    audio_transform: NormalizeAudioMelSpectrogram | None = None,
    vision_transform: NormalizeVisionImage | None = None,
    device: str = "cpu",
) -> list[int]:
    """Generate predictions from an observation interval with digit recognition.

    Args:
        model: Trained MoPoE-MRSSM model.
        classifier: Trained MNIST classifier.
        observation_interval: Dictionary containing observation data.
        n_predictions: Number of predictions to generate.
        n_frames: Number of future frames to predict.
        audio_transform: Audio normalization transform.
        vision_transform: Vision normalization transform.
        device: Device to run inference on.

    Returns:
        List of predicted digits (0-9) for each prediction.
    """
    model.eval()
    model = model.to(device)

    # Prepare observation data
    audio_obs = observation_interval["audio"]  # (query_length, 32, 32)
    image_obs = observation_interval["image"]  # (query_length, 1, 32, 32)
    speaker_act = observation_interval["speaker"]  # (query_length, 6)

    # Add channel dimension to audio if needed: (T, 32, 32) -> (T, 1, 32, 32)
    if audio_obs.ndim == 3:
        audio_obs = audio_obs[:, np.newaxis, :, :]

    # Apply transforms if provided
    if audio_transform is not None and vision_transform is not None:
        audio_obs_tensor, image_obs_tensor = apply_transforms(audio_obs, image_obs, audio_transform, vision_transform)
    else:
        audio_obs_tensor = torch.from_numpy(audio_obs).float()
        image_obs_tensor = torch.from_numpy(image_obs).float()

    # Add batch dimension and move to device
    audio_obs_tensor = audio_obs_tensor.unsqueeze(0).to(device)  # (1, query_length, 1, 32, 32)
    image_obs_tensor = image_obs_tensor.unsqueeze(0).to(device)  # (1, query_length, 1, 32, 32)
    speaker_act_tensor = torch.from_numpy(speaker_act).float().unsqueeze(0).to(device)  # (1, query_length, 6)

    # Get initial state from observation
    initial_obs = (audio_obs_tensor[:, 0], image_obs_tensor[:, 0])
    initial_state = model.initial_state(initial_obs)

    # Prepare future actions (repeat last speaker action)
    last_speaker = speaker_act_tensor[:, -1:]  # (1, 1, 6)
    future_actions = last_speaker.repeat(1, n_frames, 1)  # (1, n_frames, 6)

    predicted_digits = []

    with torch.no_grad():
        for _ in range(n_predictions):
            # Rollout transition to predict future
            # rollout_transition returns State with shape (batch, time)
            predicted_states = model.rollout_transition(
                actions=future_actions,
                prev_state=initial_state,
            )

            # Decode predicted images
            # decode_state expects State with shape (batch, time) and returns (batch, time, ...)
            predicted_reconstructions = model.decode_state(predicted_states)
            predicted_images = predicted_reconstructions["recon/vision"]  # (1, n_frames, 1, 32, 32)

            # Denormalize: [-1, 1] -> [0, 1]
            predicted_images = (predicted_images + 1.0) / 2.0
            predicted_images = torch.clamp(predicted_images, 0.0, 1.0)

            # Use first frame for digit recognition
            # predicted_images shape: (1, n_frames, 1, 32, 32)
            # Get first batch, first time step, first channel: (32, 32)
            first_image = predicted_images[0, 0, 0]  # (32, 32)

            # Recognize digit using MNIST classifier
            digit = recognize_digit(classifier, first_image, device=device)
            predicted_digits.append(digit)

    return predicted_digits


def compute_prediction_distribution(predicted_words: list[int], word_set: list[int]) -> dict[int, float]:
    """Compute prediction distribution q(w|wa) from predicted words.

    Args:
        predicted_words: List of predicted words (0-9).
        word_set: Set of all possible words.

    Returns:
        Dictionary mapping word to probability.
    """
    total = len(predicted_words)
    if total == 0:
        return {w: 0.0 for w in word_set}

    counts = defaultdict(int)
    for word in predicted_words:
        if word in word_set:
            counts[word] += 1
        # If word not in word_set, it's treated as wf (failure)

    # Normalize
    distribution = {w: counts.get(w, 0) / total for w in word_set}
    # Add failure probability
    failure_count = total - sum(counts.values())
    distribution["wf"] = failure_count / total

    return distribution


def compute_true_distribution(
    word: int,
    test_data: list[dict],
    word_set: list[int],
    query_length: int = 30,
) -> dict[int, float]:
    """Compute true distribution p(w|wa) from test data.

    Args:
        word: Current word wa.
        test_data: List of test data dictionaries.
        word_set: Set of all possible words.
        query_length: Not used (kept for compatibility). Use all frames.

    Returns:
        Dictionary mapping word to probability.
    """
    next_word_counts = defaultdict(int)
    total_transitions = 0

    for data in test_data:
        labels = data["label"]  # Use all 180 frames

        # Extract unique digit sequence (remove consecutive duplicates and silence)
        digit_sequence = []
        prev_digit = None
        for label in labels:
            digit = int(label)
            if digit == -1:  # Skip silence
                continue
            if digit != prev_digit:  # Only add when digit changes
                digit_sequence.append(digit)
                prev_digit = digit

        # Find transitions from word to next word
        for i in range(len(digit_sequence) - 1):
            if digit_sequence[i] == word:
                next_word = digit_sequence[i + 1]
                if next_word in word_set:
                    next_word_counts[next_word] += 1
                total_transitions += 1

    # Normalize
    if total_transitions == 0:
        print(f"  Warning: No transitions found for word {word}")
        print(f"    Checked {len(test_data)} test samples")
        return {w: 0.0 for w in word_set} | {"wf": 0.0}

    distribution = {w: next_word_counts.get(w, 0) / total_transitions for w in word_set}
    distribution["wf"] = 0.0  # No failures in true distribution

    print(f"  True distribution stats: total_transitions={total_transitions}, counts={dict(next_word_counts)}")
    print(f"  Distribution sum: {sum(distribution.values()):.4f}")

    return distribution


def compute_matching_rate(
    q_dist: dict[int | str, float],
    p_dist: dict[int | str, float],
    word_set: list[int],
) -> float:
    """Compute Matching Rate (MR) between prediction and true distributions.

    Args:
        q_dist: Prediction distribution q(w|wa).
        p_dist: True distribution p(w|wa).
        word_set: Set of all possible words.

    Returns:
        Matching Rate value.
    """
    mr = 0.0

    # Sum over all words in W
    for word in word_set:
        q_w = q_dist.get(word, 0.0)
        p_w = p_dist.get(word, 0.0)
        mr += min(q_w, p_w)

    # Add failure term
    q_wf = q_dist.get("wf", 0.0)
    p_wf = p_dist.get("wf", 0.0)
    mr += min(q_wf, p_wf)

    return mr


def compute_baselines(
    p_dist: dict[int | str, float],
    word_set: list[int],
    n_random_trials: int = 100,
) -> dict[str, float]:
    """Compute baseline MR values.

    Args:
        p_dist: True distribution p(w|wa).
        word_set: Set of all possible words.
        n_random_trials: Number of trials for random one-hot baseline.

    Returns:
        Dictionary with MR values for each baseline method.
    """
    import random

    n_words = len(word_set)

    # Uniform baseline
    uniform_dist = {w: 1.0 / n_words for w in word_set}
    uniform_dist["wf"] = 0.0
    uniform_mr = compute_matching_rate(uniform_dist, p_dist, word_set)

    # Peak one-hot baseline
    peak_word = max(word_set, key=lambda w: p_dist.get(w, 0.0))
    peak_dist = {w: 0.0 for w in word_set}
    peak_dist[peak_word] = 1.0
    peak_dist["wf"] = 0.0
    peak_mr = compute_matching_rate(peak_dist, p_dist, word_set)

    # Random one-hot baseline (average over multiple trials)
    random_mrs = []
    for _ in range(n_random_trials):
        random_word = random.choice(word_set)
        random_dist = {w: 0.0 for w in word_set}
        random_dist[random_word] = 1.0
        random_dist["wf"] = 0.0
        random_mr = compute_matching_rate(random_dist, p_dist, word_set)
        random_mrs.append(random_mr)
    random_mr = sum(random_mrs) / len(random_mrs)

    return {
        "uniform": uniform_mr,
        "peak_onehot": peak_mr,
        "random_onehot": random_mr,
    }


def format_results_table(
    results: dict[int, dict[str, float]],
    word_set: list[int],
) -> str:
    """Format results as a Markdown table.

    Args:
        results: Dictionary mapping word to result dictionary with MR values.
        word_set: Set of all possible words.

    Returns:
        Formatted Markdown table string.
    """
    # Header
    header = "| Word | MoPoE-MRSSM | Uniform | Peak one-hot | Random one-hot |\n"
    separator = "|-----|---------------|---------|--------------|----------------|\n"

    # Rows
    rows = []
    for word in sorted(word_set):
        word_str = str(word)
        result = results.get(word, {})
        mopoe_mr = result.get("mopoe", 0.0)
        uniform_mr = result.get("uniform", 0.0)
        peak_mr = result.get("peak_onehot", 0.0)
        random_mr = result.get("random_onehot", 0.0)

        row = f"| {word_str} | {mopoe_mr:.4f} | {uniform_mr:.4f} | {peak_mr:.4f} | {random_mr:.4f} |\n"
        rows.append(row)

    return header + separator + "".join(rows)


def save_results(
    results: dict[int, dict[str, float]],
    output_path: Path,
    word_set: list[int],
) -> None:
    """Save results to file.

    Args:
        results: Dictionary mapping word to result dictionary with MR values.
        word_set: Set of all possible words.
        output_path: Path to save results.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as Markdown
    markdown_table = format_results_table(results, word_set)
    markdown_path = output_path.with_suffix(".md")
    with open(markdown_path, "w") as f:
        f.write("# Word-to-word Transition Evaluation Results\n\n")
        f.write(markdown_table)

    # Save as JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {markdown_path} and {json_path}")


def load_model(checkpoint_path: str, config_path: str | None = None) -> MoPoE_MRSSM:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to config file (optional, recommended).

    Returns:
        Loaded model in eval mode.
    """
    if config_path is not None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Load config and instantiate model directly using hydra
        # This avoids conflicts with command-line arguments
        config = OmegaConf.load(config_path)
        model_config = config.model

        # Convert LightningCLI format (class_path + init_args) to hydra format (_target_ + direct params)
        def convert_lightning_to_hydra(cfg):
            """Convert LightningCLI config format to hydra format."""
            # Use OmegaConf.is_config() which works across all versions
            is_omega_conf = OmegaConf.is_config(cfg)

            if isinstance(cfg, dict) or is_omega_conf:
                # Convert OmegaConf to dict if needed
                if is_omega_conf:
                    cfg = OmegaConf.to_container(cfg, resolve=True)

                if "class_path" in cfg:
                    # Convert class_path + init_args to _target_ + direct params
                    result = {"_target_": cfg["class_path"]}
                    if "init_args" in cfg:
                        # Recursively convert nested configs
                        for key, value in cfg["init_args"].items():
                            # For "config" parameter (plain dict without class_path), convert to appropriate config class
                            # This is needed because cnn.Encoder/Decoder expects config as EncoderConfig/DecoderConfig dataclass
                            if key == "config" and isinstance(value, dict) and "class_path" not in value:
                                # Determine which config class to use based on parent class_path
                                # We need to check the parent's class_path to know if it's Encoder or Decoder
                                parent_class_path = cfg.get("class_path", "")

                                # Convert dict to appropriate config dataclass
                                if "Encoder" in parent_class_path:
                                    from cnn.config import EncoderConfig

                                    # Convert lists to tuples for dataclass
                                    config_dict = {
                                        k: tuple(v)
                                        if isinstance(v, list)
                                        and k in ("linear_sizes", "channels", "kernel_sizes", "strides", "paddings")
                                        else v
                                        for k, v in value.items()
                                    }
                                    result[key] = EncoderConfig(**config_dict)
                                elif "Decoder" in parent_class_path:
                                    from cnn.config import DecoderConfig

                                    # Convert lists to tuples for dataclass
                                    config_dict = {
                                        k: tuple(v)
                                        if isinstance(v, list)
                                        and k
                                        in (
                                            "linear_sizes",
                                            "channels",
                                            "kernel_sizes",
                                            "strides",
                                            "paddings",
                                            "output_paddings",
                                        )
                                        else v
                                        for k, v in value.items()
                                    }
                                    # Handle conv_in_shape specially (it's a list that should stay as list or tuple)
                                    if "conv_in_shape" in config_dict and isinstance(
                                        config_dict["conv_in_shape"], list
                                    ):
                                        config_dict["conv_in_shape"] = tuple(config_dict["conv_in_shape"])
                                    result[key] = DecoderConfig(**config_dict)
                                else:
                                    # Fallback: use OmegaConf if we can't determine the type
                                    result[key] = OmegaConf.create(value)
                            else:
                                result[key] = convert_lightning_to_hydra(value)
                    return result
                else:
                    # Recursively convert nested dicts
                    return {k: convert_lightning_to_hydra(v) for k, v in cfg.items()}
            elif isinstance(cfg, list) or is_omega_conf:
                if is_omega_conf:
                    cfg = OmegaConf.to_container(cfg, resolve=True)
                return [convert_lightning_to_hydra(item) for item in cfg]
            else:
                return cfg

        hydra_config = convert_lightning_to_hydra(model_config)

        # Ensure it's a regular dict (not OmegaConf)
        if OmegaConf.is_config(hydra_config):
            hydra_config = OmegaConf.to_container(hydra_config, resolve=True)

        if not isinstance(hydra_config, dict) or "_target_" not in hydra_config:
            raise ValueError(
                f"Invalid config structure. Expected dict with '_target_', got: {type(hydra_config)}. "
                f"Keys: {list(hydra_config.keys()) if isinstance(hydra_config, dict) else 'N/A'}"
            )

        # Instantiate model from config using hydra
        # First, try to import the class to verify it exists
        target_class = hydra_config["_target_"]
        try:
            # Import the class
            import importlib

            module_path, class_name = target_class.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import class {target_class}: {e}") from e

        # Use instantiate with _convert_="none" to preserve all objects including dataclasses
        # This prevents instantiate from trying to convert EncoderConfig/DecoderConfig to OmegaConf
        # If instantiate fails, fall back to direct instantiation
        try:
            model = instantiate(hydra_config, _convert_="none")
        except Exception as e:
            # If instantiate fails, try direct instantiation
            import warnings

            warnings.warn(f"instantiate failed: {e}, trying direct instantiation")
            # Extract init_args and instantiate directly
            init_args = {k: v for k, v in hydra_config.items() if k != "_target_"}

            # Recursively instantiate nested configs
            # Use direct instantiation to avoid OmegaConf conversion issues with dataclasses
            def instantiate_nested(value):
                import dataclasses
                from omegaconf import OmegaConf

                if isinstance(value, dict) and "_target_" in value:
                    # Direct instantiation for nested classes
                    target = value["_target_"]
                    try:
                        import importlib

                        module_path, class_name = target.rsplit(".", 1)
                        module = importlib.import_module(module_path)
                        nested_cls = getattr(module, class_name)
                        nested_init_args = {k: instantiate_nested(v) for k, v in value.items() if k != "_target_"}

                        # Convert string class paths to actual class objects for certain parameters
                        # This is needed for torchrl.modules.MLP.activation_class and similar parameters
                        if "activation_class" in nested_init_args and isinstance(
                            nested_init_args["activation_class"], str
                        ):
                            try:
                                import importlib

                                act_module_path, act_class_name = nested_init_args["activation_class"].rsplit(".", 1)
                                act_module = importlib.import_module(act_module_path)
                                nested_init_args["activation_class"] = getattr(act_module, act_class_name)
                            except (ImportError, AttributeError, ValueError) as e:
                                # If conversion fails, keep the original value
                                pass

                        result = nested_cls(**nested_init_args)
                        return result
                    except (ImportError, AttributeError) as e:
                        raise ValueError(f"Failed to import class {target}: {e}") from e
                elif isinstance(value, dict):
                    return {k: instantiate_nested(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [instantiate_nested(item) for item in value]
                else:
                    return value

            init_args = {k: instantiate_nested(v) for k, v in init_args.items()}
            model = cls(**init_args)

        # Verify model is actually instantiated (not a config object)
        if not isinstance(model, MoPoE_MRSSM):
            if isinstance(model, dict):
                raise ValueError(
                    f"Failed to instantiate model. Got dict. "
                    f"Config _target_: {hydra_config.get('_target_', 'N/A')}, "
                    f"Result keys: {list(model.keys()) if isinstance(model, dict) else 'N/A'}"
                )
            raise ValueError(f"Failed to instantiate model. Got type: {type(model)}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Try to load without config (may fail if model structure changed)
        try:
            model = MoPoE_MRSSM.load_from_checkpoint(checkpoint_path, map_location="cpu")
        except Exception:
            raise ValueError("Failed to load model without config. Please provide --config argument.") from None

    model.eval()
    return model


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate word-to-word transitions using MoPoE-MRSSM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/multimodal_rssm/models/mrssm/mopoe_mrssm/configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default="../../audio-mnist-generator/data/test",
        help="Directory containing test .npz files or .pt files",
    )
    parser.add_argument(
        "--use-pt-files",
        action="store_true",
        help="Use .pt files instead of .npz files",
    )
    parser.add_argument(
        "--npz-dir-for-labels",
        type=str,
        default=None,
        help="Directory containing .npz files for labels (if using .pt files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/word_transitions_mrssm",
        help="Output path for results (without extension)",
    )
    parser.add_argument(
        "--n-intervals",
        type=int,
        default=6,
        help="Number of observation intervals per word",
    )
    parser.add_argument(
        "--n-predictions",
        type=int,
        default=10,
        help="Number of predictions per interval",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=10,
        help="Number of future frames to predict",
    )
    parser.add_argument(
        "--query-length",
        type=int,
        default=30,
        help="Length of observation interval",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default=None,
        help="Path to MNIST classifier checkpoint (if None, train new one)",
    )

    args = parser.parse_args()

    # Word set: 0-9
    word_set = list(range(10))

    print("Loading model...")
    model = load_model(args.checkpoint, args.config)
    model = model.to(args.device)

    print("Loading MNIST classifier...")
    classifier = load_pretrained_classifier(args.classifier_checkpoint, device=args.device)

    print("Loading test data...")
    test_data_dir = Path(args.test_data_dir)
    # Resolve relative paths relative to current working directory
    if not test_data_dir.is_absolute():
        test_data_dir = Path.cwd() / test_data_dir
    test_data_dir = test_data_dir.resolve()

    print(f"Test data directory: {test_data_dir}")
    print(f"Directory exists: {test_data_dir.exists()}")

    # Determine npz directory for labels if using .pt files
    npz_dir_for_labels = None
    if args.use_pt_files:
        if args.npz_dir_for_labels:
            npz_dir_for_labels = Path(args.npz_dir_for_labels)
            if not npz_dir_for_labels.is_absolute():
                npz_dir_for_labels = Path.cwd() / npz_dir_for_labels
            npz_dir_for_labels = npz_dir_for_labels.resolve()
        else:
            # Default: try to find npz files in audio-mnist-generator
            npz_dir_for_labels = Path(__file__).parent.parent.parent / "audio-mnist-generator" / "data" / "test"
            npz_dir_for_labels = npz_dir_for_labels.resolve()
        print(f"NPZ directory for labels: {npz_dir_for_labels}")

    if test_data_dir.exists():
        if args.use_pt_files:
            pt_files = list(test_data_dir.glob("act_*.pt"))
            print(f"Found {len(pt_files)} .pt files")
        else:
            npz_files = list(test_data_dir.glob("*.npz"))
            print(f"Found {len(npz_files)} .npz files")

    test_data = load_test_data_with_labels(
        test_data_dir,
        use_pt_files=args.use_pt_files,
        npz_dir_for_labels=npz_dir_for_labels,
    )
    print(f"Loaded {len(test_data)} test samples")

    # Create transforms
    audio_transform = NormalizeAudioMelSpectrogram(min_value=-80.0, max_value=0.0)
    vision_transform = NormalizeVisionImage()

    # Evaluate each word
    results = {}

    for word in word_set:
        print(f"\nEvaluating word {word}...")

        # Select intervals
        intervals = select_intervals_for_word(
            word,
            test_data,
            n_intervals=args.n_intervals,
            query_length=args.query_length,
        )

        if len(intervals) == 0:
            print(f"Warning: No intervals found for word {word}")
            # Still compute true distribution and baselines for comparison
            p_dist = compute_true_distribution(word, test_data, word_set)
            baselines = compute_baselines(p_dist, word_set)
            results[word] = {
                "mopoe": 0.0,
                "uniform": baselines["uniform"],
                "peak_onehot": baselines["peak_onehot"],
                "random_onehot": baselines["random_onehot"],
            }
            continue

        print(f"  Selected {len(intervals)} intervals")

        # Generate predictions
        all_predictions = []
        for i, interval in enumerate(intervals):
            print(f"  Processing interval {i + 1}/{len(intervals)}...")
            predictions = generate_predictions_with_classifier(
                model,
                classifier,
                interval,
                n_predictions=args.n_predictions,
                n_frames=args.n_frames,
                audio_transform=audio_transform,
                vision_transform=vision_transform,
                device=args.device,
            )
            all_predictions.extend(predictions)

        print(f"  Generated {len(all_predictions)} predictions")

        # Compute prediction distribution
        q_dist = compute_prediction_distribution(all_predictions, word_set)

        # Compute true distribution (use all frames, not just query_length)
        p_dist = compute_true_distribution(word, test_data, word_set)

        # Compute MR
        mopoe_mr = compute_matching_rate(q_dist, p_dist, word_set)

        # Compute baselines
        baselines = compute_baselines(p_dist, word_set)

        results[word] = {
            "mopoe": mopoe_mr,
            "uniform": baselines["uniform"],
            "peak_onehot": baselines["peak_onehot"],
            "random_onehot": baselines["random_onehot"],
        }

        print(f"  MoPoE-MRSSM MR: {mopoe_mr:.4f}")
        print(f"  Uniform MR: {baselines['uniform']:.4f}")
        print(f"  Peak one-hot MR: {baselines['peak_onehot']:.4f}")
        print(f"  Random one-hot MR: {baselines['random_onehot']:.4f}")

    # Save results
    output_path = Path(args.output)
    print(f"\nTotal results: {len(results)} words evaluated")
    if len(results) == 0:
        print("Warning: No results to save! Check if intervals were found for any word.")
    save_results(results, output_path, word_set)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
