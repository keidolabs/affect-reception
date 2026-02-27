#!/bin/bash
# sync_down.sh — pull experiment outputs from Vast.ai → /Volumes/mechlab/
# Usage: ./scripts/sync_down.sh <instance-ip> <port> <exp_id> [model_key]
#
# Examples:
#   # Download all outputs for an experiment (CSV, plots, manifests):
#   ./scripts/sync_down.sh 123.45.67.89 12345 11_v2_extract_seta
#
#   # Download activations for a specific model only (faster):
#   ./scripts/sync_down.sh 123.45.67.89 12345 11_v2_extract_seta llama8b_base
#
#   # Download patching results:
#   ./scripts/sync_down.sh 123.45.67.89 12345 14_v2_patching
#
# After downloading, DESTROY the instance:
#   vast destroy instance <id>

set -e

IP=$1
PORT=$2
EXP=$3
MODEL_KEY=${4:-""}

if [ -z "$IP" ] || [ -z "$PORT" ] || [ -z "$EXP" ]; then
    echo "Usage: $0 <instance-ip> <port> <exp_id> [model_key]"
    exit 1
fi

# Check external SSD is mounted
if [ ! -d "/Volumes/mechlab" ]; then
    echo "ERROR: /Volumes/mechlab not mounted. Connect the Samsung T7 first."
    exit 1
fi

if [ -n "$MODEL_KEY" ]; then
    # Single-model activation download
    SRC="root@$IP:~/emo-circuits/experiments/$EXP/outputs/activations/$MODEL_KEY/"
    DEST="/Volumes/mechlab/outputs/$EXP/activations/$MODEL_KEY/"
    mkdir -p "$DEST"
    echo "Downloading $EXP activations for $MODEL_KEY → $DEST"
else
    # Full experiment output (manifests, CSVs, plots)
    SRC="root@$IP:~/emo-circuits/experiments/$EXP/outputs/"
    DEST="/Volumes/mechlab/outputs/$EXP/"
    mkdir -p "$DEST"
    echo "Downloading all $EXP outputs → $DEST"
fi

rsync -avz --progress \
    -e "ssh -i ~/.ssh/vast_key -p $PORT" \
    "$SRC" "$DEST"

echo ""
echo "Downloaded → $DEST"
du -sh "$DEST"
echo ""
echo "IMPORTANT: Destroy the instance when done downloading all experiments:"
echo "  vast destroy instance <id>"
