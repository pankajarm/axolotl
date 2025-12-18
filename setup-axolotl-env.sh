#!/bin/bash
# Quick setup script for Axolotl Ministral training on 4xA100
# Usage: bash setup-axolotl-env.sh /path/to/venv

set -e

VENV_PATH=${1:-".venv"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Creating virtual environment at $VENV_PATH ==="
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "=== Installing uv for fast package management ==="
pip install uv

echo "=== Installing PyTorch with CUDA ==="
uv pip install torch==2.7.0 torchvision==0.24.1

echo "=== Installing core dependencies ==="
uv pip install \
    transformers==4.57.1 \
    tokenizers==0.22.1 \
    accelerate==1.11.0 \
    datasets==4.4.1 \
    huggingface-hub==0.36.0 \
    peft==0.18.0 \
    trl==0.25.0 \
    bitsandbytes==0.48.2 \
    xformers \
    deepspeed \
    wandb \
    einops \
    sentencepiece \
    fire \
    gradio

echo "=== Installing Axolotl from source ==="
if [ -d "$SCRIPT_DIR/axolotl" ]; then
    uv pip install -e "$SCRIPT_DIR/axolotl"
else
    echo "Axolotl not found at $SCRIPT_DIR/axolotl"
    echo "Clone it with: git clone https://github.com/axolotl-ai-cloud/axolotl.git"
    echo "Then run: uv pip install -e axolotl"
fi

echo "=== Setup complete! ==="
echo "Activate with: source $VENV_PATH/bin/activate"
echo ""
echo "Key versions installed:"
pip show transformers peft trl | grep -E "^(Name|Version):"
