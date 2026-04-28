#!/bin/bash
# v15 full pipeline: wait for fine-tuning → calibrate → evaluate
# Runs as a background job; logs to v15_pipeline.log

set -euo pipefail

LOG=v15_pipeline.log
CKPT=./checkpoints_v15_finetune/best_model.pth
THRESH=./calibration_thresholds_v15.json
THRESH_CON=./calibration_thresholds_v15_constrained.json
PYTHON=.venv/bin/python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ── Stage 0: wait for training PID ────────────────────────────────────────────
TRAIN_PID="${1:-}"
if [[ -n "$TRAIN_PID" ]]; then
    log "Waiting for training PID $TRAIN_PID to finish..."
    while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
    log "Training process $TRAIN_PID has exited."
else
    log "No training PID provided — assuming training already complete."
fi

# Verify checkpoint exists
if [[ ! -f "$CKPT" ]]; then
    log "ERROR: best_model.pth not found at $CKPT — aborting."
    exit 1
fi
log "Checkpoint found: $CKPT"

# ── Stage 1: Calibration (standard 2D search) ─────────────────────────────────
log "=== Stage 1: Calibration (standard) ==="
$PYTHON calibrate.py \
    --checkpoint "$CKPT" \
    --pattern fine_tuning \
    --output "$THRESH" \
    --grid_steps 50 \
    2>&1 | tee -a "$LOG"
log "Calibration done → $THRESH"

# ── Stage 2: Calibration (constrained — ensures Non-sig recall ≥ 10%) ─────────
log "=== Stage 2: Calibration (constrained Non-sig recall ≥ 0.10) ==="
$PYTHON calibrate.py \
    --checkpoint "$CKPT" \
    --pattern fine_tuning \
    --output "$THRESH_CON" \
    --grid_steps 50 \
    --constrain_nonsig_recall 0.10 \
    2>&1 | tee -a "$LOG"
log "Constrained calibration done → $THRESH_CON"

# ── Stage 3: Evaluation — raw (no thresholds) on test split ───────────────────
log "=== Stage 3: Evaluation — raw argmax (test split) ==="
$PYTHON eval.py \
    --checkpoint "$CKPT" \
    --pattern fine_tuning \
    --data_split test \
    --detailed \
    --save_results ./predictions_v15 \
    2>&1 | tee -a "$LOG"

# ── Stage 4: Evaluation — standard calibration (test split) ───────────────────
log "=== Stage 4: Evaluation — standard calibrated (test split) ==="
$PYTHON eval.py \
    --checkpoint "$CKPT" \
    --pattern fine_tuning \
    --data_split test \
    --thresholds "$THRESH" \
    --detailed \
    2>&1 | tee -a "$LOG"

# ── Stage 5: Evaluation — constrained calibration (test split) ────────────────
log "=== Stage 5: Evaluation — constrained calibrated (test split) ==="
$PYTHON eval.py \
    --checkpoint "$CKPT" \
    --pattern fine_tuning \
    --data_split test \
    --thresholds "$THRESH_CON" \
    --use_constrained \
    --detailed \
    2>&1 | tee -a "$LOG"

log "=== v15 pipeline complete. Results in $LOG ==="
