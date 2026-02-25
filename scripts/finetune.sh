#!/usr/bin/env bash
# Fine-tuning script for SC-Net
#
# Usage:
#   bash scripts/finetune.sh <pretrained_checkpoint_path>
#
# Example:
#   bash scripts/finetune.sh ./checkpoints_v2/best_model.pth
#   bash scripts/finetune.sh ./checkpoints_v2/best_model.pth ./dataset/train

set -euo pipefail

PRETRAINED="${1:?Usage: $0 <pretrained_checkpoint> [data_root]}"
DATA_ROOT="${2:-./dataset/train}"
CHECKPOINT_DIR="${3:-./checkpoints_v2_finetune}"
NUM_GPUS="${NUM_GPUS:-2}"

echo "============================================"
echo "SC-Net Fine-tuning"
echo "============================================"
echo "  Pretrained checkpoint: ${PRETRAINED}"
echo "  Data root:             ${DATA_ROOT}"
echo "  Checkpoint dir:        ${CHECKPOINT_DIR}"
echo "  GPUs:                  ${NUM_GPUS}"
echo "============================================"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    train.py \
    --distributed \
    --pattern fine_tuning \
    --pretrained "${PRETRAINED}" \
    --data_root "${DATA_ROOT}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --epochs 100 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --grad_clip 0.1 \
    --warmup_epochs 5 \
    --layerwise_lr \
    --amp \
    --ema \
    --ema_decay 0.999 \
    --augment \
    --save_every 10 \
    --print_every 1
