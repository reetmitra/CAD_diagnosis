#!/bin/bash
# v16 full pipeline: fine-tuning → calibrate (standard + Sig-constrained) → evaluate
# All stages run automatically in sequence. Logs to v16_pipeline.log.

set -euo pipefail

LOG=v16_pipeline.log
CKPT=./checkpoints_v16_finetune/best_model.pth
THRESH=./calibration_thresholds_v16.json
THRESH_SIG=./calibration_thresholds_v16_sig_constrained.json
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

if [[ ! -f "$CKPT" ]]; then
    log "ERROR: best_model.pth not found at $CKPT — aborting."
    exit 1
fi
log "Checkpoint found: $CKPT"

# ── Stage 1: Calibration — standard ───────────────────────────────────────────
log "=== Stage 1: Calibration (standard) ==="
$PYTHON calibrate.py \
    --checkpoint "$CKPT" \
    --pattern fine_tuning \
    --output "$THRESH" \
    --grid_steps 50 \
    2>&1 | tee -a "$LOG"
log "Calibration done → $THRESH"

# ── Stage 2: Calibration — Sig-recall constrained (Sig recall >= 0.70) ────────
log "=== Stage 2: Calibration (Sig recall >= 0.70 constrained) ==="
$PYTHON calibrate.py \
    --checkpoint "$CKPT" \
    --pattern fine_tuning \
    --output "$THRESH_SIG" \
    --grid_steps 50 \
    --constrain_sig_recall 0.70 \
    2>&1 | tee -a "$LOG"
log "Sig-constrained calibration done → $THRESH_SIG"

# ── Stage 3: Evaluation — raw argmax (testing split) ──────────────────────────
log "=== Stage 3: Evaluation — raw argmax (testing split) ==="
$PYTHON eval.py \
    --checkpoint "$CKPT" \
    --pattern fine_tuning \
    --data_split testing \
    --detailed \
    --save_results ./predictions_v16 \
    2>&1 | tee -a "$LOG"

# ── Stage 4: Evaluation — standard calibrated (testing split) ─────────────────
log "=== Stage 4: Evaluation — standard calibrated (testing split) ==="
$PYTHON eval.py \
    --checkpoint "$CKPT" \
    --pattern fine_tuning \
    --data_split testing \
    --thresholds "$THRESH" \
    --detailed \
    2>&1 | tee -a "$LOG"

# ── Stage 5: Evaluation — Sig-recall constrained calibration (testing split) ──
log "=== Stage 5: Evaluation — Sig-constrained calibrated (testing split) ==="
$PYTHON eval.py \
    --checkpoint "$CKPT" \
    --pattern fine_tuning \
    --data_split testing \
    --thresholds "$THRESH_SIG" \
    --use_constrained \
    --detailed \
    2>&1 | tee -a "$LOG"

# ── Stage 6: CPR visualizations ───────────────────────────────────────────────
log "=== Stage 6: CPR visualizations (testing split) ==="
$PYTHON visualize.py \
    --data_root ./dataset/test \
    --pattern testing \
    --checkpoint "$CKPT" \
    --thresholds "$THRESH_SIG" \
    --use_constrained \
    --output_dir ./viz_v16 \
    --filter all \
    --save_predictions ./predictions_v16_detail \
    2>&1 | tee -a "$LOG"

log "=== v16 pipeline complete. Results in $LOG ==="
