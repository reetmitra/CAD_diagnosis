#!/usr/bin/env bash
# Fine-tuning script for SC-Net v7 (Phase 1 Quick Wins)
#
# Usage:
#   bash scripts/finetune_v7.sh <pretrained_checkpoint_path> [data_root]
#
# Example:
#   bash scripts/finetune_v7.sh ./checkpoints_v6/best_model.pth
#   bash scripts/finetune_v7.sh ./checkpoints_v6/best_model.pth ./dataset/train

set -euo pipefail

PRETRAINED="${1:?Usage: $0 <pretrained_checkpoint> [data_root]}"
DATA_ROOT="${2:-./dataset/train}"
CHECKPOINT_DIR="${3:-./checkpoints_v7}"

echo "============================================"
echo "SC-Net v7 Fine-tuning (Phase 1 Quick Wins)"
echo "============================================"
echo "  Pretrained checkpoint: ${PRETRAINED}"
echo "  Data root:             ${DATA_ROOT}"
echo "  Checkpoint dir:        ${CHECKPOINT_DIR}"
echo "  Config:                configs/finetune_v7.yaml"
echo "============================================"

python train.py \
    --config configs/finetune_v7.yaml \
    --pretrained "${PRETRAINED}" \
    --data_root "${DATA_ROOT}" \
    --checkpoint_dir "${CHECKPOINT_DIR}"
