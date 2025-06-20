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

# Activate the virtual environment
source $HOME/deepdraw/.venv/bin/activate

# Run the training script
python main.py \
    --config configs/dino_classifier.yaml \
    --log_interval 500 \
    --save_dir outputs \
    --checkpoint_frequency 5 \
    --num_workers 8