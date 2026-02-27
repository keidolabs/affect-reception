#!/bin/bash
# sync_up.sh — push code from MacBook to Vast.ai instance
# Usage: ./scripts/sync_up.sh <instance-ip> <port>
# Example: ./scripts/sync_up.sh 123.45.67.89 12345
#
# Excludes: outputs/, activation files, .venv, .env (secrets stay local)
# Run this before starting a cloud session, or after making local code changes.

set -e

IP=$1
PORT=$2

if [ -z "$IP" ] || [ -z "$PORT" ]; then
    echo "Usage: $0 <instance-ip> <port>"
    echo "  Get IP and port from Vast.ai console after instance spins up"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing $REPO_ROOT → root@$IP:~/emo-circuits/ (port $PORT)"

rsync -avz --progress \
    --exclude='experiments/*/outputs/' \
    --exclude='*.npz' \
    --exclude='*.npy' \
    --exclude='*.pt' \
    --exclude='*.safetensors' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    --exclude='.env' \
    -e "ssh -i ~/.ssh/vast_key -p $PORT" \
    "$REPO_ROOT/" root@$IP:~/emo-circuits/

echo ""
echo "Sync complete → root@$IP:~/emo-circuits/"
echo "Next: ssh -i ~/.ssh/vast_key root@$IP -p $PORT"
