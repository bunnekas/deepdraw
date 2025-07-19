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