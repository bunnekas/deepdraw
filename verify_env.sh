#!/bin/bash
# Verification script for DeepDraw environment

# Exit on error
set -e

echo "Verifying DeepDraw environment setup..."

# Check if virtual environment exists
if [ ! -d "/home/lect0149/deepdraw/.venv" ]; then
    echo "ERROR: Virtual environment not found. Please run setup_env.sh first."
    exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Python version: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" != "3.10.4" ]]; then
    echo "WARNING: Python version is $PYTHON_VERSION, not 3.10.4 as requested."
fi

# Check PyTorch installation and CUDA availability
echo "Checking PyTorch and CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Check xformers installation
echo "Checking xformers..."
if python -c "import xformers" 2>/dev/null; then
    echo "xformers is installed."
    python -c "import xformers; print(f'xformers version: {xformers.__version__}')"
else
    echo "WARNING: xformers is not installed or not working properly."
fi

# Check other dependencies
echo "Checking other dependencies..."
for package in webdataset wandb tqdm pyyaml matplotlib pillow; do
    if python -c "import $package" 2>/dev/null; then
        echo "$package is installed."
    else
        echo "WARNING: $package is not installed or not working properly."
    fi
done

# Check GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU information:"
    nvidia-smi
else
    echo "nvidia-smi not available. Cannot check GPU information."
fi

# Deactivate the virtual environment
deactivate

echo "Verification complete!"
