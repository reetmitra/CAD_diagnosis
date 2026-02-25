#!/usr/bin/env bash
# Evaluation script for fine-tuned SC-Net model
#
# Usage:
#   bash scripts/eval_finetune.sh <checkpoint_path> [data_root]
#
# Examples:
#   bash scripts/eval_finetune.sh ./checkpoints_v2_finetune/best_model.pth
#   bash scripts/eval_finetune.sh ./checkpoints_v2_finetune/best_model.pth ./dataset/test

set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> [data_root]}"
DATA_ROOT="${2:-./dataset/test}"

echo "============================================"
echo "SC-Net Fine-tuning Evaluation"
echo "============================================"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Data root:  ${DATA_ROOT}"
echo "============================================"

python eval.py \
    --checkpoint "${CHECKPOINT}" \
    --pattern fine_tuning \
    --data_root "${DATA_ROOT}" \
    --eval_sc \
    --batch_size 2
