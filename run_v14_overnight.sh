#!/usr/bin/env bash
# run_v14_overnight.sh — Resume v14 fine-tuning then run full eval + viz pipeline.
#
# Stages:
#   1. Resume training from checkpoint_epoch_129.pth
#   2. Raw eval on best_model.pth
#   3. Constrained calibration
#   4. Calibrated eval
#   5. Visualisations — all, incorrect, correct, + save_predictions JSON
#
# Logs: each stage writes its own log file.
# On any failure the script exits immediately (set -e).

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

CKPT_DIR="./checkpoints_v14_finetune"
DATA_ROOT="./dataset/train"
VIZ_DIR="./viz_v14"
PRED_DIR="./predictions_v14"
CAL_FILE="./calibration_thresholds_v14_constrained.json"

echo "========================================================"
echo "Stage 1: Resume v14 fine-tuning from epoch 129"
echo "========================================================"
torchrun --nproc_per_node=2 --master_port=29508 train.py --distributed \
    --config configs/finetune_v14.yaml \
    --pretrained ./checkpoints_v14/best_model.pth \
    --resume "${CKPT_DIR}/checkpoint_epoch_129.pth" \
    >> logs_finetune_v14_resumed.log 2>&1
echo "[OK] Training complete."

echo "========================================================"
echo "Stage 2: Raw evaluation on best_model.pth"
echo "========================================================"
python eval.py \
    --checkpoint "${CKPT_DIR}/best_model.pth" \
    --pattern fine_tuning \
    --data_root "${DATA_ROOT}" \
    --eval_sc --detailed \
    --save_results ./results_v14_best.json \
    > logs_eval_v14_best.log 2>&1
echo "[OK] Raw eval complete. See logs_eval_v14_best.log"

echo "========================================================"
echo "Stage 3: Constrained calibration (Non-sig recall >= 0.10)"
echo "========================================================"
python calibrate.py \
    --checkpoint "${CKPT_DIR}/best_model.pth" \
    --pattern fine_tuning \
    --data_root "${DATA_ROOT}" \
    --constrain_nonsig_recall 0.10 \
    --output "${CAL_FILE}" \
    > logs_calibrate_v14_constrained.log 2>&1
echo "[OK] Calibration complete. Thresholds: ${CAL_FILE}"

echo "========================================================"
echo "Stage 4: Calibrated evaluation"
echo "========================================================"
python eval.py \
    --checkpoint "${CKPT_DIR}/best_model.pth" \
    --pattern fine_tuning \
    --data_root "${DATA_ROOT}" \
    --eval_sc --detailed \
    --thresholds "${CAL_FILE}" \
    --use_constrained \
    --save_results ./results_v14_best_calibrated.json \
    > logs_eval_v14_best_calibrated.log 2>&1
echo "[OK] Calibrated eval complete. See logs_eval_v14_best_calibrated.log"

echo "========================================================"
echo "Stage 5a: Visualise all samples"
echo "========================================================"
python visualize.py \
    --data_root "${DATA_ROOT}" \
    --pattern all \
    --checkpoint "${CKPT_DIR}/best_model.pth" \
    --model_pattern fine_tuning \
    --thresholds "${CAL_FILE}" \
    --use_constrained \
    --output_dir "${VIZ_DIR}/all" \
    > logs_viz_v14_all.log 2>&1
echo "[OK] All-samples visualisation complete."

echo "========================================================"
echo "Stage 5b: Visualise incorrect predictions only"
echo "========================================================"
python visualize.py \
    --data_root "${DATA_ROOT}" \
    --pattern all \
    --checkpoint "${CKPT_DIR}/best_model.pth" \
    --model_pattern fine_tuning \
    --thresholds "${CAL_FILE}" \
    --use_constrained \
    --filter incorrect \
    --output_dir "${VIZ_DIR}/incorrect" \
    > logs_viz_v14_incorrect.log 2>&1
echo "[OK] Incorrect-only visualisation complete."

echo "========================================================"
echo "Stage 5c: Visualise correct predictions only"
echo "========================================================"
python visualize.py \
    --data_root "${DATA_ROOT}" \
    --pattern all \
    --checkpoint "${CKPT_DIR}/best_model.pth" \
    --model_pattern fine_tuning \
    --thresholds "${CAL_FILE}" \
    --use_constrained \
    --filter correct \
    --output_dir "${VIZ_DIR}/correct" \
    > logs_viz_v14_correct.log 2>&1
echo "[OK] Correct-only visualisation complete."

echo "========================================================"
echo "Stage 5d: Save per-artery prediction JSONs (traceability)"
echo "========================================================"
mkdir -p "${PRED_DIR}"
python visualize.py \
    --data_root "${DATA_ROOT}" \
    --pattern all \
    --checkpoint "${CKPT_DIR}/best_model.pth" \
    --model_pattern fine_tuning \
    --thresholds "${CAL_FILE}" \
    --use_constrained \
    --output_dir "${VIZ_DIR}/predictions" \
    --save_predictions "${PRED_DIR}" \
    > logs_viz_v14_predictions.log 2>&1
echo "[OK] Per-artery prediction JSONs saved to ${PRED_DIR}"

echo ""
echo "========================================================"
echo "ALL STAGES COMPLETE"
echo "========================================================"
echo "  Raw results:        ./results_v14_best.json"
echo "  Calibrated results: ./results_v14_best_calibrated.json"
echo "  Calibration:        ${CAL_FILE}"
echo "  Visualisations:     ${VIZ_DIR}/"
echo "  Prediction JSONs:   ${PRED_DIR}/"
echo "  Logs:               logs_finetune_v14_resumed.log"
echo "                      logs_eval_v14_best.log"
echo "                      logs_calibrate_v14_constrained.log"
echo "                      logs_eval_v14_best_calibrated.log"
echo "                      logs_viz_v14_*.log"
