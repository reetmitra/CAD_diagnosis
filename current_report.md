# SC-Net v14 Fine-tuning — Results Report
**Date:** 28 April 2026 | **Checkpoint:** `checkpoints_v14_finetune/best_model.pth` (epoch 188) | **Baseline:** v12 fine-tuning

---

## 1. What Changed in This Run (Phase 26)

The paper defines C⁻¹(ŷ_od) as a pure function of model predictions — transforming OD query outputs into SC pseudo-labels. The previous implementation violated this by running Hungarian matching against ground-truth targets inside C⁻¹, leaking GT into what should be a prediction-only operation.

**Changes implemented:**

| Component | Old | New |
|-----------|-----|-----|
| `_get_sampling_point_classification_targets` | Hungarian match vs GT → take matched subset | Softmax all Q queries → filter by confidence → winner-takes-all per sampling point |
| `dual_task_contrastive_loss.forward` | GT always required | GT optional (`od_targets=None`); hard-label path receives no GT |
| `spatio_temporal_contrast_loss.forward` | Deep-copied GT passed to dc_loss | GT not passed at all |
| Confidence annealing | None | Starts at 0.7, anneals linearly to 0.4 over epochs 20–60 (mirroring DC weight ramp) |

**v14 config differences from v13:**

| | v13 | v14 |
|--|-----|-----|
| Pre-trained backbone | 110/300 epochs (aborted) | **300/300 epochs (fully converged)** |
| C⁻¹ in L_dc | GT-dependent (Hungarian) | **GT-free + confidence annealing** |
| Peak LR | 2.5e-5 | 2.0e-5 |
| dc_confidence_start | — | 0.7 → 0.4 |

---

## 2. Training Run

| | |
|--|--|
| Started from | `checkpoints_v14/best_model.pth` (v14 pre-train, 300 ep) |
| Crash | NCCL ALLREDUCE timeout at ep135 — hardware issue, not code |
| Resumed from | `checkpoint_epoch_129.pth` |
| Completed | All 300 epochs |
| Best checkpoint | **Epoch 188** (selected on validation stenosis F1) |
| SWA model | Discarded — Healthy collapsed to 0 (SWA averaged weights from poor-phase epochs) |

**Loss trajectory (resumed run, ep130–299):**

| Epoch | Train loss | Val loss | Checkpoint saved? |
|-------|-----------|---------|-------------------|
| 130 | 3.82 | 4.86 | Yes (F1=0.408) |
| 146 | 3.70 | 4.94 | Yes (F1=0.414) |
| 165 | 3.54 | 5.05 | Yes (F1=0.419) |
| 186–188 | 3.53 | 5.17 | Yes (F1=0.427) — **final best** |
| 299 | 3.13 | 5.59 | — |

Train loss fell steadily (-0.69) while val loss rose monotonically (+0.73) — a clear overfit. The model peaked at ep188 and no further generalisation improvement occurred. DC loss at ep299 averaged ~0.9–2.2 per batch, indicating the contrastive signal remained active but was not driving further generalisation.

---

## 3. Raw Evaluation (no calibration, argmax)

Evaluated on all 3182 samples.

### Stenosis

| Metric | v14 raw | v12 (calibrated) |
|--------|---------|-----------------|
| ACC | 0.585 | 0.736 |
| F1 (macro) | 0.475 | 0.739 |
| Precision | 0.449 | 0.743 |
| Recall | 0.569 | 0.736 |
| AUC (macro) | **0.804** | — |

**Per-class (raw):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Healthy | 0.000 | 0.000 | 0.000 | 979 |
| Non-significant | 0.421 | 0.928 | 0.579 | 974 |
| Significant | 0.926 | 0.779 | 0.846 | 1229 |

The uncalibrated model never predicts Healthy — all 979 Healthy arteries are routed to Non-significant. Sig is handled well (0.846 F1). AUC of 0.804 confirms the model has genuine discriminative power; the class collapse is a threshold/bias issue, not a representational failure.

**Raw prediction distribution (3182 arteries):** Healthy = 0 | Non-sig = 2147 | Sig = 1035

**Confusion matrix (raw):**
```
                  Healthy   Non-sig     Sig
Healthy  (979)        0       972        7
Non-sig  (974)        0       904       70
Sig     (1229)        0       271      958
Predicted:            0      2147     1035
```

---

## 4. Calibration

Constrained 3D threshold search on the validation split (477 samples). Constraint: Non-sig recall ≥ 0.10.

| Class | Threshold | Effect |
|-------|-----------|--------|
| Healthy (t0) | 0.300 | Lowered — makes Healthy easier to predict |
| Non-sig (t1) | **1.150** | Raised above 1.0 — suppresses raw model's Non-sig over-prediction |
| Significant (t2) | 0.100 | Very low — aggressively pulls samples into Sig |
| Calcified | 0.729 | Standard |
| Non-calcified | 1.756 | Raised — suppresses Non-calc over-prediction |
| Mixed | 0.619 | Lowered — makes Mixed easier to predict |

Val Macro-F1 improved from 0.414 (argmax) → 0.612 (constrained calibration), a +0.198 gain from calibration alone.

---

## 5. Calibrated Evaluation — v14 vs v12

| Metric | v14 calibrated | v12 baseline | Δ |
|--------|---------------|-------------|---|
| **Stenosis ACC** | 0.686 | 0.736 | **-0.050** |
| **Stenosis F1** | 0.654 | 0.739 | **-0.085** |
| Stenosis Precision | 0.663 | 0.743 | -0.080 |
| Stenosis Recall | 0.678 | 0.736 | -0.058 |
| Specificity | 0.841 | 0.867 | -0.026 |
| **Healthy Recall** | **0.939** | — | — |
| **Non-sig Recall** | 0.317 | **0.639** | **-0.322** |
| **Sig Recall** | 0.777 | 0.733 | **+0.044** |
| **Plaque F1** | **0.640** | 0.502 | **+0.138** |
| Plaque ACC | 0.771 | — | — |
| Stenosis AUC | 0.804 | — | — |

**Per-class breakdown (calibrated):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Healthy | 0.730 | **0.939** | 0.821 | 979 |
| Non-significant | 0.563 | 0.317 | 0.406 | 974 |
| Significant | 0.695 | **0.777** | 0.734 | 1229 |

**Calibrated prediction distribution (3182 arteries):** Healthy = 1,259 | Non-sig = 549 | Sig = 1,374

**Confusion matrix (calibrated):**
```
                  Healthy   Non-sig     Sig
Healthy  (979)      919        50       10
Non-sig  (974)      256       309      409
Sig     (1229)       84       190      955
Predicted:         1259       549     1374
```

**Plaque breakdown (calibrated):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Calcified | 0.827 | 0.891 | 0.858 | 1383 |
| Non-calcified | 0.756 | 0.668 | 0.710 | 585 |
| Mixed | 0.388 | 0.323 | 0.353 | 235 |

Mixed remains the hardest class (only 235 instances, high confusion with Calcified).

**Calibrated plaque prediction distribution:** Calcified = 1,490 | Non-calcified = 517 | Mixed = 196 (raw: 1,667 / 516 / 20 — calibration substantially rescues Mixed from near-zero to 196 predictions).

---

## 6. Prediction Traceability (per-artery JSONs)

Per-artery JSON files saved to `predictions_v14/` (3182 files + `predictions_summary.jsonl`). Each JSON records all OD query outputs, confidence scores, bounding boxes, and the final stenosis/plaque assignment. Stats derived from the summary:

| Stat | Value |
|------|-------|
| Total arteries | 3,182 |
| Correct (raw inference) | 1,827 (57.4%) |
| Avg surviving OD queries per artery | 5.60 |
| Arteries with 0 surviving queries | 28 (0.9%) |
| Avg max query confidence | 0.806 |
| Min / Max confidence | 0.440 / 0.995 |

*"Surviving" = foreground OD query above confidence threshold that contributes to the stenosis bar.*

**Error breakdown (raw inference, 1355 errors):**

| Error type | Count | % of dataset |
|-----------|-------|-------------|
| Healthy → Non-significant | 968 | 30.4% |
| Non-significant → Significant | 189 | 5.9% |
| Significant → Non-significant | 187 | 5.9% |
| Healthy → Significant | 11 | 0.3% |

The dominant failure (30.4% of all 3182 arteries) is Healthy vessels being flagged as Non-significant, driven by the model's uncalibrated bias toward predicting Non-sig. Over-escalation (Non-sig→Sig) and under-escalation (Sig→Non-sig) are symmetric at 5.9% each. True cross-class misses (Healthy→Sig) are rare at 0.3%.

---

## 7. Visualisation Samples

**Viz output:** `viz_v14/` — 3182 total | 1827 correct (57.4%) | 1355 incorrect
*(Note: viz uses single-sample inference without calibration threshold application — accuracy reflects raw model, not the 68.6% calibrated batched-eval figure)*

---

### Correct: Significant stenosis detected

**APNHC00016_LAD** — GT: Significant | Pred: Significant ✓

![Correct Sig](viz_v14/correct/APNHC00016_LAD__sten_Sig_pred_Sig_CORRECT.png)

Model correctly identifies significant stenosis along the LAD. The OD bar captures the dominant stenosis region. Cross-section thumbnails show bright calcified plaque deposits (correctly predicted as Calcified) across multiple axial slices.

---

### Correct: Non-significant stenosis detected

**APNHC00029_LCX** — GT: Non-significant | Pred: Non-significant ✓

![Correct NonSig](viz_v14/correct/APNHC00029_LCX__sten_NonSig_pred_NonSig_CORRECT.png)

Model correctly identifies a focal non-significant lesion (localised Non-sig/Mixed plaque segment). The model's prediction bar captures the lesion window reasonably well despite the predominantly healthy surrounding vessel.

---

### Failure: Healthy artery over-classified as Non-significant

**APNHC00002_LCX** — GT: Healthy | Pred: Non-significant ✗

![Healthy as NonSig](viz_v14/incorrect/APNHC00002_LCX__sten_Healthy_pred_NonSig_WRONG.png)

The most common error mode — accounts for 972/979 Healthy misclassifications in the raw model. The vessel appears visually clean but the model assigns Non-sig across almost the full length. Calibration reduces this by raising t0 (Healthy threshold), recovering Healthy recall to 0.939, but Non-sig recall is sacrificed as a result.

---

### Failure: Non-significant over-escalated to Significant

**APNHC00034_D1** — GT: Non-significant | Pred: Significant ✗

![NonSig as Sig](viz_v14/incorrect/APNHC00034_D1__sten_NonSig_pred_Sig_WRONG.png)

The second major error mode (409/974 Non-sig cases predicted as Sig after calibration). The vessel shows localised mixed plaque (GT confirms Non-sig with Mixed composition) but the model escalates severity. The aggressive Sig threshold (t2=0.10) amplifies this pattern — calibration is trading Non-sig precision for Sig sensitivity.

---

## 8. Root Cause Analysis

**Why v14 regressed on stenosis despite the better backbone:**

1. **Overfitting in the resumed run.** Val loss rose monotonically from 4.86 (ep130) to 5.59 (ep299) while train loss fell from 3.82 to 3.13. The 1.42-point train/val divergence indicates the model memorised training patterns without generalising.

2. **Checkpoint was the least-bad overfit point, not a convergence point.** Best epoch 188 had val F1=0.427 — the raw model at its selected checkpoint still predicts Healthy exactly 0 times out of 979 cases.

3. **Early stopping did not help.** Patience=120 on F1 would have fired at ep308 (120 after ep188 peak), past the 300-epoch limit. Val loss criterion would have been more appropriate here — it peaked at ep130.

4. **DC loss variance.** DC component fluctuated 0.27–3.91 per batch at ep299, suggesting the GT-free pseudo-labels were still noisy and contributing to val loss inflation (DC is evaluated on val samples too).

**What improved:**

- **Plaque F1 +0.138** — the fully converged v14 backbone with properly trained SE fusion gates and parallel 2D/3D streams gives richer spatial representations that plaque classification directly benefits from.
- **Sig Recall +0.044** — model is more aggressive at catching significant disease, which has higher clinical priority.

---

## 9. Pipeline Status

| Stage | Status |
|-------|--------|
| Resume training (ep130–299) | Complete |
| Raw evaluation | Complete — `results_v14_best.json` |
| Constrained calibration | Complete — `calibration_thresholds_v14_constrained.json` |
| Calibrated evaluation | Complete — `results_v14_best_calibrated.json` |
| Visualise all / correct / incorrect | Complete — `viz_v14/` |
| Per-artery prediction JSONs (Stage 5d) | **Complete** — `predictions_v14/` (3182 JSONs + summary) |

*Stage 5d originally crashed at sample 149 with an IndexError (`STENOSIS_NAMES` had 3 entries; fine-tuning mode uses 6 OD classes). Fixed at `visualize.py:410` and rerun successfully.*

---

## 10. Recommended Next Steps

| Option | Rationale |
|--------|-----------|
| **Evaluate ep130 checkpoint directly** | Val loss was lowest at ep130 (4.86). Despite lower raw F1, calibrated performance may be better than ep188 since the model was less overfit. |
| **Re-run with val-loss early stopping** | Current F1-based stopping allowed 170 epochs of overfitting post-peak. Stopping on val loss would have fired much earlier. |
| **Lower LR on resume** | Resume used the same lr=2.0e-5 as fresh training. After ep129 of prior learning, 5.0e-6 would reduce the overfit rate. |
| **Keep v14 plaque results** | Plaque F1 0.640 vs v12's 0.502 is a real gain from the better backbone. The next run should not regress this. |
