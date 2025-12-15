#!/bin/bash
# Training script for Devstral-Small-2-24B-Instruct-2512
# Run inside screen session: screen -S devstral2_train -dm bash train_devstral2.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv-ministral3"
AXOLOTL_DIR="$SCRIPT_DIR/axolotl"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/devstral2_train.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Change to axolotl directory
cd "$AXOLOTL_DIR"

# Print environment info
echo "=============================================="
echo "  Devstral-Small-2-24B Training"
echo "=============================================="
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================================="
echo ""

# Enable HuggingFace transfer for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Disable telemetry warning (optional)
export AXOLOTL_DO_NOT_TRACK=0

# Run training
echo "Starting training..."
axolotl train examples/devstral2/devstral-small-2-24b-qlora.yaml

echo ""
echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="

