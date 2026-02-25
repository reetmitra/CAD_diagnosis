#!/usr/bin/env bash
# Pre-training script for SC-Net (3-class plaque composition)
#
# Usage:
#   bash scripts/pretrain.sh
#   bash scripts/pretrain.sh ./dataset/train
#   bash scripts/pretrain.sh ./dataset/train ./checkpoints_v2

set -euo pipefail

DATA_ROOT="${1:-./dataset/train}"
CHECKPOINT_DIR="${2:-./checkpoints_v2}"
NUM_GPUS="${NUM_GPUS:-2}"

echo "============================================"
echo "SC-Net Pre-training"
echo "============================================"
echo "  Data root:       ${DATA_ROOT}"
echo "  Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "  GPUs:            ${NUM_GPUS}"
echo "============================================"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    train.py \
    --distributed \
    --pattern pre_training \
    --data_root "${DATA_ROOT}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --epochs 200 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --grad_clip 0.1 \
    --warmup_epochs 10 \
    --layerwise_lr \
    --amp \
    --ema \
    --ema_decay 0.999 \
    --augment \
    --save_every 10 \
    --print_every 1
