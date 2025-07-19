# DeepDraw
This repository contains a PyTorch pipeline for training a linear classifier on the Google QuickDraw dataset as part of the RWTH Deep Learning Lab 2025. We use state-of-the-art vision encoders like DinoV2 and CLIP as backbone, enabling a fine-tuning logic.

## Project Structure

```
dinov2-classifier/
├── configs/                  
│   ├── dino_classifier.yaml  # Default configuration
│   └── ...                   # Additional experiment configs
├── data/                     
│   ├── preprocessing.py      # data conversion and sharding
│   ├── dataloader.py         # WebDataset loader for tar files
│   └── categories.txt        # Category labels
├── models/                   
│   ├── build_model.py        # Unified model construction
│   ├── backbones/            
│   │   └── dinov2.py         # DinoV2 backbone implementation
│   └── classifiers/          
│       └── dino_classifier.py # DinoV2 classifier
├── trainers/                 
│   └── trainer.py            # Training loop and evaluation
├── utils/                    
│   ├── utils.py              # General utilities
│   └── logging_utils.py      # Logging functionality
├── jobs/                     
│   ├── batch_train.sh        # Job to start model training
│   └── batch_zeroshot.sh     # Job to start zeroshot evaluation
├── logs/                     
│   ├── wandb/                # Weights & Biases logs
│   └── slurm/                # SLURM output logs
├── outputs/                 
│   └── checkpoints/          # Periodic model checkpoints
├── train.py                  # Train and evaluate the model
├── zeroshot.py               # Perform zeroshot evaluation
└── README.md                 
```

## Key Components

### Data Preprocessing
We preprocess the raw version of the QickDraw dataset and convert the raw strokes stored as ndjson to shards of JPGs and archive them into tar files. In order to apply the DinoV2 or CLIP image encoder, we store each sketch as a 224x224 RGB image.

### Data Loading

The `dataloader.py` module uses WebDataset to efficiently load sketch images from tar files. The data is stored in three directories:
- `train/`: 275 classes for training
- `test/`: 275 classes for evaluation
- `quickdraw_zeroshot/`: 70 classes for zero-shot evaluation

### Model Architecture

The project uses a modular architecture:

1. **Backbones**: Feature extractors that convert images into embeddings
   - `dinov2.py`: Implements DinoV2 ViT-L/14 with configurable freezing

2. **Classifiers**: Classification heads that convert embeddings into class predictions
   - `dino_classifier.py`: Linear classifier for DinoV2 features

### Model Construction

The `build_model.py` module provides a unified interface for constructing models:

```python
model = build_model(
    backbone_name="dinov2_vitl14_reg",
    num_classes=275,
    freeze_backbone=True,
    freeze_blocks="all"
)
```

### Training

The `trainer.py` module implements the training loop with:
- Mixed precision training for better GPU utilization
- Learning rate scheduling with separate rates for backbone and classifier
- Periodic checkpointing and best model saving
- Comprehensive logging with Weights & Biases

## Setting Up a Run

### Configuration

Edit the configuration file in `configs/dino_classifier.yaml` to set your training parameters:

```yaml
# Random seed for reproducibility
seed: 42

# Data paths
data:
  train_dir: "/rwthfs/rz/cluster/hpcwork/lect0149/train"
  test_dir: "/rwthfs/rz/cluster/hpcwork/lect0149/test"
  num_workers: 8
  prefatch_factor: 4

# Training parameters
train:
  batch_size: 128
  epochs: 6
  backbone_lr: 1e-5
  classifier_lr: 1e-4
  weight_decay: 1e-2
  mixed_precision: true

# Model configuration
model:
  type: "dinov2_vitl14_reg"
  num_classes: 275
  freeze_backbone: true
  freeze_blocks: "all"

# Wandb configuration
wandb:
  project: "DinoV2-Classifier-all-data"
  group: "DeepDraw"

# Logging configuration
logging:
  log_interval: 500
  save_checkpoints: true
  checkpoint_frequency: 1
  name_appendix: 'frozen'

# Wandb configuration
wandb:
  project: "DinoV2-Classifier-all-data"
  group: "DeepDraw"
  tags: ["dinov2", "sketch-classification"]

# Paths
paths:
  save_dir: 'outputs'
```

### Running on SLURM

1. Edit the batch script in `jobs/batch_train.sh` to set your SLURM parameters:

```bash
#!/bin/bash
#SBATCH --job-name=dinov2_frozen
#SBATCH --output=../logs/slurm/%x_%j.out
#SBATCH --error=../logs/slurm/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=lect0149

### Load modules
module purge
module load GCCcore/11.3.0 Python/3.10.4 CUDA/12.6

### Activate virtual environment
cd ..
source .venv/bin/activate

### Run your script
python train.py --config configs/dinov2_classifier.yaml --name_appendix frozen
```

2. Submit the job:
```bash
sbatch jobs/batch_dinov2.sh
```

## Extending the Project

### Adding a New Backbone

1. Create a new file in `models/backbones/` (e.g., `swin.py`)
2. Implement a loader function that returns the backbone and feature dimension
3. Register the backbone in `models/backbones/__init__.py`

### Adding a New Classifier

1. Create a new file in `models/classifiers/` (e.g., `swin_classifier.py`)
2. Implement a classifier class that inherits from `nn.Module`
3. Register the classifier in `models/classifiers/__init__.py`
