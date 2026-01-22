"""Common training script utilities."""

import sys
from pathlib import Path

from lightning.pytorch.cli import LightningCLI


def run_training(default_config: str | Path) -> None:
    """Run training with the specified default config.

    Args:
        default_config: Path to the default config file.
    """
    # Convert to string if Path object
    default_config_str = str(default_config)

    # Check if -c or --config is already specified in command line arguments
    has_config = any(arg in {"-c", "--config"} for arg in sys.argv[1:])

    if not has_config:
        # Insert fit subcommand and default config before any other arguments
        sys.argv.insert(1, "fit")
        sys.argv.insert(2, "-c")
        sys.argv.insert(3, default_config_str)
    elif "fit" not in sys.argv[1:]:
        # If config is specified but fit is missing, add fit subcommand
        sys.argv.insert(1, "fit")

    LightningCLI(
        run=True,
        save_config_callback=None,  # Disable config saving to avoid conflicts
    )
