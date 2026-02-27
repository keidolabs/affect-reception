#!/bin/bash
# cloud_setup.sh — first-time setup for a new Vast.ai instance
# Usage: bash scripts/cloud_setup.sh
# Requires: HF_TOKEN env var set before running
#
# Run sequence:
#   1. Spin up Vast.ai instance (PyTorch template, Ubuntu 22.04)
#   2. SSH in: ssh -i ~/.ssh/vast_key root@<ip> -p <port>
#   3. export HF_TOKEN=hf_yourtoken
#   4. git clone https://github.com/<your-org>/emotion-circuits ~/emo-circuits && cd ~/emo-circuits
#   5. bash scripts/cloud_setup.sh

set -e

echo "=== Checking HF_TOKEN ==="
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN env var not set."
    echo "  export HF_TOKEN=hf_yourtoken"
    exit 1
fi

echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y -qq tmux git curl

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env 2>/dev/null || source ~/.cargo/env 2>/dev/null || true

echo "=== Syncing Python environment ==="
uv sync

echo "=== Writing .env for HF auth ==="
echo "HF_ACCESS_TOKEN=$HF_TOKEN" > .env
echo "  .env written"

echo ""
echo "=== Setup complete ==="
echo "Start experiments with:"
echo "  tmux new -s exp"
echo "  uv run python experiments/11_v2_extract_seta/extract_llama8b_base.py"
echo ""
echo "Monitor GPU:"
echo "  watch -n1 nvidia-smi"
