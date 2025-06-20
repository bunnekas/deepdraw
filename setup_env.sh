#!/bin/bash
# Setup script for DeepDraw environment with Python 3.10.4, CUDA 12.6, and uv

# Exit on error
set -e

echo "Setting up DeepDraw environment with Python 3.10.4, CUDA 12.6, and uv..."

# Use specific project directory path
PROJECT_DIR="/home/lect0149"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment with Python 3.10.4
echo "Creating virtual environment with Python 3.10.4..."
uv venv --python=3.10.4 .venv

# Activate the virtual environment
source .venv/bin/activate

# Install PyTorch with CUDA 12.6 support
echo "Installing PyTorch with CUDA 12.6 support..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install xformers
echo "Installing xformers..."
uv pip install xformers

# Install other project dependencies
echo "Installing project dependencies..."
uv pip install webdataset wandb tqdm pyyaml matplotlib pillow

# Create requirements.txt file
cat > requirements.txt << EOL
torch
torchvision
torchaudio
xformers
webdataset
wandb
tqdm
pyyaml
matplotlib
pillow
EOL

echo "Environment setup complete!"
echo "To activate the environment, run: source $PROJECT_DIR/.venv/bin/activate"
echo "To verify PyTorch installation, run: python -c 'import torch; print(torch.__version__ ); print(torch.cuda.is_available())'"
