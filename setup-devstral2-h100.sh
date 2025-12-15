#!/bin/bash
# Setup script for Devstral-Small-2-24B-Instruct-2512 training on H100
# This model is FP8 quantized and requires GPUs with compute capability >= 8.9
# H100 has compute capability 9.0, so it can run this model natively
#
# Usage: bash setup-devstral2-h100.sh
# 
# This creates a fresh environment with:
# - Axolotl on transformers-v5 branch
# - Transformers from git main (v5 RC)
# - Cut Cross Entropy for VRAM optimization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv-devstral2"
AXOLOTL_DIR="$SCRIPT_DIR"
BASE_DIR="/home/ubuntu/us-south-2-nano-chat-exp"

echo "=============================================="
echo "  Devstral-Small-2-24B + H100 Setup"
echo "=============================================="
echo ""

# Check GPU compute capability
echo "[0/7] Verifying H100 GPU..."
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ "$GPU_CC" < "8.9" ]]; then
    echo "ERROR: This script requires compute capability >= 8.9 (H100/RTX 4090)"
    echo "Found: compute capability $GPU_CC"
    exit 1
fi
echo "  → H100 detected with compute capability $GPU_CC ✓"
echo ""

# Step 1: Create fresh virtual environment
echo "[1/7] Creating virtual environment at $VENV_PATH..."
if [ -d "$VENV_PATH" ]; then
    echo "  → Removing existing venv..."
    rm -rf "$VENV_PATH"
fi
uv venv "$VENV_PATH" --python 3.10
source "$VENV_PATH/bin/activate"

# Step 2: Switch Axolotl to transformers-v5 branch
echo ""
echo "[2/7] Setting up Axolotl branch..."
cd "$AXOLOTL_DIR"

# Fetch latest and ensure we're on transformers-v5
git fetch origin transformers-v5
if git show-ref --verify --quiet refs/heads/pankaj_devstral2_h100; then
    echo "  → Branch pankaj_devstral2_h100 exists, checking out..."
    git checkout pankaj_devstral2_h100
    git merge origin/transformers-v5 --no-edit || true
else
    echo "  → Creating new branch pankaj_devstral2_h100 from origin/transformers-v5..."
    git checkout -b pankaj_devstral2_h100 origin/transformers-v5
fi

# Step 3: Install PyTorch
echo ""
echo "[3/7] Installing PyTorch 2.7.0 with CUDA..."
uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Step 4: Install Transformers v5 from git main
echo ""
echo "[4/7] Installing Transformers v5 (from git main)..."
uv pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"

# Step 5: Install core dependencies (transformers-v5 branch requirements)
echo ""
echo "[5/7] Installing core dependencies..."
uv pip install \
    "huggingface_hub>=1.1.7" \
    "peft>=0.18.0" \
    "tokenizers>=0.22.1" \
    "accelerate==1.12.0" \
    "datasets==4.4.1" \
    "deepspeed>=0.18.2" \
    "trl==0.25.1" \
    "hf_xet==1.2.0" \
    "kernels>=0.11.2" \
    "bitsandbytes==0.48.2" \
    "triton>=3.0.0" \
    "xformers>=0.0.23.post1" \
    "liger-kernel==0.6.4" \
    "packaging==23.2" \
    "trackio" \
    "optimum==1.16.2" \
    "hf_transfer" \
    "sentencepiece" \
    "gradio==5.49.1" \
    "pydantic>=2.10.6" \
    "addict" \
    "fire" \
    "PyYAML>=6.0" \
    "requests" \
    "wandb" \
    "einops" \
    "colorama" \
    "numba>=0.61.2" \
    "numpy>=2.2.6" \
    "evaluate==0.4.1" \
    "scipy" \
    "nvidia-ml-py==12.560.30" \
    "art" \
    "tensorboard" \
    "python-dotenv==1.0.1" \
    "zstandard==0.22.0" \
    "fastcore" \
    "torchao==0.13.0" \
    "schedulefree==1.4.1" \
    "axolotl-contribs-lgpl==0.0.7" \
    "axolotl-contribs-mit==0.0.5" \
    "posthog==6.7.11" \
    "mistral-common==1.8.5"

# Step 6: Install Cut Cross Entropy
echo ""
echo "[6/7] Installing Cut Cross Entropy..."
uv pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@f643b88"

# Step 7: Install Axolotl from source
echo ""
echo "[7/7] Installing Axolotl from source..."
cd "$AXOLOTL_DIR"
uv pip install --no-build-isolation --no-deps -e .

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p "$BASE_DIR/dataset/last_run_prepared/Devstral-Small-2-24B-Instruct-2512-sft-v1"
mkdir -p "$BASE_DIR/dataset/last_run_prepared/Devstral-Small-2-24B-Instruct-2512-quicktest"
mkdir -p "$BASE_DIR/outputs/Devstral-Small-2-24B-Instruct-2512-sft-v1/lora-out"
mkdir -p "$BASE_DIR/outputs/Devstral-Small-2-24B-Instruct-2512-quicktest/lora-out"
mkdir -p "$BASE_DIR/logs"

# Summary
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Environment: $VENV_PATH"
echo "Branch: pankaj_devstral2_h100"
echo ""
echo "Key versions:"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import peft; print(f'  peft: {peft.__version__}')"
python -c "import accelerate; print(f'  accelerate: {accelerate.__version__}')"
echo ""
echo "Model: mistralai/Devstral-Small-2-24B-Instruct-2512 (FP8)"
echo "GPU: H100 (compute capability 9.0)"
echo ""
echo "To activate this environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To run quicktest:"
echo "  cd $AXOLOTL_DIR"
echo "  axolotl train examples/devstral2/devstral-small-2-24b-lora-quicktest-h100.yaml"
echo ""
echo "To start full training:"
echo "  cd $AXOLOTL_DIR"
echo "  axolotl train examples/devstral2/devstral-small-2-24b-lora-h100.yaml"
echo ""

