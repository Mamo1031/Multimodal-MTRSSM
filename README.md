# Multimodal-MTRSSM

![python](https://img.shields.io/badge/python-3.10-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


## Models

| Model | Modality | Description |
|-------|----------|-------------|
| **MoPoE-MRSSM** | Vision + Audio + Action | PoE and MoE are combined to create a robust multimodal model. |
| **MoPoE-MMTRSSM** | Vision + Audio + Action | MoPoE-MRSSM extended with MTRSSM. |


## Setup

### Required Environment

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (package manager)

### Installation

```bash
# Create virtual environment
uv venv

# Install dependencies (including development and training dependencies)
uv sync --editable --all-extras
```

`--all-extras` flag installs the following optional dependencies:
- **dev**: Development tools (mypy, ruff, pytest, etc.)
- **train**: Training libraries (torchvision, wandb, etc.)

## Usage

### Training

#### Available Models

| Model | Command | Config Path |
|-------|---------|-------------|
| MoPoE-MRSSM | `train-mopoe-mrssm` | `src/multimodal_rssm/models/mrssm/mopoe_mrssm/configs/default.yaml` |
| MoPoE-MMTRSSM | `train-mopoe-mmtrssm` | `src/multimodal_rssm/models/mmtrssm/mopoe_mmtrssm/configs/default.yaml` |

#### Training Commands

**Use default configuration:**
```bash
uv run poe train-mopoe-mrssm
```

**Specify a custom configuration file:**
```bash
uv run poe train-mopoe-mrssm -c src/multimodal_rssm/models/mrssm/mopoe_mrssm/configs/your_config.yaml
```

**Run in background (with logging):**
```bash
nohup uv run poe train-mopoe-mrssm > train_mopoe_mrssm.log 2>&1 &
```

### Configuration Files

Each model has a dedicated configuration file (YAML). See the table above for config paths.

The configuration file can be customized with the following items:

- **Model Architecture**: Encoder, decoder, representation network, transition network configuration
- **Optimization**: Optimizer, learning rate scheduler configuration
- **Data**: Dataset, batch size, preprocessing configuration
- **Learning**: Number of epochs, precision, gradient clipping, etc.

### Dataset

The data in the `data/` directory is automatically preprocessed.

## Project Structure

```
Multimodal-MTRSSM/
├── src/multimodal_rssm/
│   └── models/
│       ├── __init__.py
│       ├── callback.py              # Common callbacks
│       ├── core.py                  # Base RSSM class
│       ├── dataset.py               # Base dataset utilities
│       ├── networks.py              # Network definitions
│       ├── objective.py             # Loss functions
│       ├── state.py                 # State utilities
│       ├── transform.py             # Data transforms
│       ├── mrssm/                   # Multimodal RSSM implementation
│       │   ├── __init__.py
│       │   ├── callback.py          # MRSSM callbacks
│       │   ├── dataset.py           # MRSSM dataset
│       │   └── mopoe_mrssm/         # MoPoE-MRSSM model
│       │       ├── __init__.py
│       │       ├── core.py          # MoPoE-MRSSM implementation
│       │       ├── callback.py      # MoPoE-MRSSM callbacks
│       │       └── configs/
│       │           └── default.yaml
│       └── mmtrssm/                 # Multimodal MTRSSM implementation
│           ├── __init__.py
│           ├── callback.py          # MMTRSSM callbacks
│           ├── state.py             # MTRSSM-specific State (MTState)
│           └── mopoe_mmtrssm/       # MoPoE-MMTRSSM model
│               ├── __init__.py
│               ├── core.py          # MoPoE-MMTRSSM implementation
│               ├── callback.py      # MoPoE-MMTRSSM callbacks
│               └── configs/
│                   └── default.yaml
├── evaluation/                      # Evaluation scripts
│   ├── __init__.py
│   ├── evaluate_word_transitions_mrssm.py    # MoPoE-MRSSM evaluation
│   ├── evaluate_word_transitions_mtmrssm.py  # MoPoE-MMTRSSM evaluation
│   └── mnist_classifier.py          # MNIST classifier for digit recognition
├── scripts/                         # Training scripts
│   ├── __init__.py
│   ├── _train_common.py             # Common training utilities
│   ├── train_mopoe_mrssm.py         # MoPoE-MRSSM training script
│   └── train_mopoe_mmtrssm.py       # MoPoE-MMTRSSM training script
├── data/                            # Data directory
│   └── .gitkeep
├── pyproject.toml                   # Project configuration
├── .python-version                  # Python version specification
└── .gitignore                       # Git ignore rules
```

## Development

### Code quality check

This project uses the following tools:

- **Ruff**: Linter/formatter
- **mypy**: Type checking
- **pydoclint**: Documentation string check

#### Run code quality checks (with auto-fix)
```bash
uv run poe lint
```

### Run tests

```bash
uv run poe test
```
