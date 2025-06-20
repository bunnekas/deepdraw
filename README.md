# Sketch-Classifier
This repository contains a PyTorch pipeline for training a linear classifier on the Google QuickDraw dataset as part of the RWTH Deep Learning Lab 2025. We use state-of-the-art vision encoders like DinoV2 and CLIP as backbone, enabling a fine-tuning logic.

# Project Structure

```
deepdraw/
├── configs/                  # Configuration files
│   ├── default.yaml          # Default configuration
│   └── ...                   # Additional experiment configs
├── data/                     # Data loading code
│   ├── dataloader.py         # WebDataset loader for tar files
│   └── categories.txt        # Category labels
├── models/                   # Model definitions
│   ├── __init__.py           # Model registry exports
│   ├── build_model.py        # Unified model construction
│   ├── backbones/            # Feature extractors
│   │   ├── __init__.py       # Backbone registry
│   │   ├── dinov2.py         # DinoV2 backbone implementation
│   │   └── clip.py           # CLIP backbone implementation
│   └── classifiers/          # Classification heads
│       ├── __init__.py       # Classifier registry
│       ├── dino_classifier.py # DinoV2 classifier
│       └── clip_classifier.py # CLIP classifier
├── trainers/                 # Training code
│   └── trainer.py            # Training loop and evaluation
├── utils/                    # Utility functions
│   ├── utils.py              # General utilities
│   └── logging_utils.py      # Logging functionality
├── jobs/                     # SLURM batch scripts
│   ├── batch_dinov2.sh       # DinoV2 training job
│   └── batch_clip.sh         # CLIP training job
├── logs/                     # Logs directory
│   ├── wandb/                # Weights & Biases logs
│   └── slurm/                # SLURM output logs
├── outputs/                  # Model outputs and checkpoints
│   └── checkpoints/          # Periodic model checkpoints
├── setup_env.sh              # Environment setup script
├── verify_env.sh             # Environment verification script
├── main.py                   # Main entry point
└── README.md                 # This file
```

## Key Components

### Data Loading

The `dataloader.py` module uses WebDataset to efficiently load sketch images from tar files. The data is stored in three directories:
- `train/`: 275 classes for training
- `test/`: 275 classes for evaluation
- `quickdraw_zeroshot/`: 70 classes for zero-shot evaluation

### Model Architecture

The project uses a modular architecture with two main components:

1. **Backbones**: Feature extractors that convert images into embeddings
   - `dinov2.py`: Implements DinoV2 ViT-L/14 with configurable freezing
   - `clip.py`: Implements CLIP ViT-L/14 with configurable freezing

2. **Classifiers**: Classification heads that convert embeddings into class predictions
   - `dino_classifier.py`: Linear classifier for DinoV2 features
   - `clip_classifier.py`: Linear classifier for CLIP features

### Model Construction

The `build_model.py` module provides a unified interface for constructing models:

```python
model = build_model(
    backbone_name="dinov2_vitl14_reg",  # or "clip_vitl14"
    num_classes=275,
    freeze_backbone=False,
    freeze_blocks="all"  # or a number/list of block indices
)
```

### Training

The `trainer.py` module implements the training loop with:
- Mixed precision training for better GPU utilization
- Learning rate scheduling with separate rates for backbone and classifier
- Periodic checkpointing and best model saving
- Comprehensive logging with Weights & Biases

## Setting Up a Run

### Environment Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd deepdraw
   ```

2. Set up the Python environment using the provided script:
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

3. Verify the environment:
   ```bash
   chmod +x verify_env.sh
   ./verify_env.sh
   ```

4. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

### Configuration

Edit the configuration file in `configs/default.yaml` to set your training parameters:

```yaml
# Random seed for reproducibility
seed: 42

# Data paths
data:
  train_dir: "/rwthfs/rz/cluster/work/lect0149/train"
  test_dir: "/rwthfs/rz/cluster/work/lect0149/test"
  zeroshot_dir: "/rwthfs/rz/cluster/work/lect0149/quickdraw_zeroshot"

# Training parameters
train:
  batch_size: 128
  epochs: 15
  backbone_lr: 1e-5
  classifier_lr: 1e-4
  weight_decay: 1e-2
  mixed_precision: true

# Model configuration
model:
  type: "dinov2_vitl14_reg"
  num_classes: 275
  freeze_backbone: false
  # freeze_blocks: 12      # Freeze first 12 blocks
  # freeze_blocks: [0,1,2] # Freeze specific blocks
  # freeze_blocks: "all"   # Freeze all blocks
  # freeze_blocks: "none"  # Don't freeze any blocks

# Logging configuration
logging:
  log_interval: 500
  save_checkpoints: true

# Wandb configuration
wandb:
  project: "deepdraw"
  group: "DeepDraw"
  tags: ["dinov2", "sketch-classification"]
```

### Running on SLURM

1. Edit the batch script in `jobs/batch_dinov2.sh` to set your SLURM parameters:

```bash
#!/bin/bash
#SBATCH --job-name=deepdraw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

# Set environment variables for wandb
export WANDB_DIR="$HOME/logs/wandb"
mkdir -p $WANDB_DIR

# Activate the virtual environment
source $HOME/deepdraw/.venv/bin/activate

# Run the training script
python main.py \
    --config configs/default.yaml \
    --log_interval 500 \
    --mixed_precision \
    --save_dir outputs \
    --checkpoint_frequency 5 \
    --num_workers 8
```

2. Submit the job:
```bash
sbatch jobs/batch_dinov2.sh
```

### Running Locally (for debugging)

For local debugging without SLURM:

```bash
# Set environment variable to disable xformers on CPU
export DINOV2_DISABLE_XFORMERS=1

# Run with minimal settings
python main.py \
    --config configs/default.yaml \
    --log_interval 10 \
    --num_workers 0
```

## Monitoring Training

Training progress can be monitored using Weights & Biases:

1. Open the Weights & Biases dashboard in your browser
2. Navigate to your project (default: "deepdraw")
3. Key metrics to monitor:
   - `train/loss` and `test/loss`: Training and test loss
   - `train/accuracy` and `test/accuracy`: Training and test accuracy
   - `charts/lr_backbone` and `charts/lr_classifier`: Learning rates
   - `train/epoch_time` and `test/epoch_time`: Time per epoch

## Extending the Project

### Adding a New Backbone

1. Create a new file in `models/backbones/` (e.g., `swin.py`)
2. Implement a loader function that returns the backbone and feature dimension
3. Register the backbone in `models/backbones/__init__.py`

### Adding a New Classifier

1. Create a new file in `models/classifiers/` (e.g., `swin_classifier.py`)
2. Implement a classifier class that inherits from `nn.Module`
3. Register the classifier in `models/classifiers/__init__.py`
