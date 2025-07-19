#!/bin/bash
#SBATCH --job-name=DinoV2_finetuned_epoch_2
#SBATCH --output=DinoV2_finetuned_epoch_2_%j.out
#SBATCH --error=DinoV2_finetuned_epoch_2_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=lect0149

### Load modules
module purge
module load GCCcore/11.3.0 Python/3.10.4 CUDA/12.6

cd ..
source deepdraw/.venv/bin/activate

# Navigate to your project directory
cd /home/ig244735/DinoV2-Zeroshot

# Execute the Python script
python3 zeroshot.py