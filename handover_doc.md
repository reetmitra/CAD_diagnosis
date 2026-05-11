# SC-Net CAD Diagnosis — Project Handover Document
**Prepared by:** Reet Mitra | **Date:** 11 May 2026 | **Repository:** https://github.com/reetmitra/CAD_diagnosis

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Team Status Update](#2-team-status-update)
3. [Environment & Infrastructure](#3-environment--infrastructure)
4. [Repository Structure](#4-repository-structure)
5. [Full Training History](#5-full-training-history)
6. [Current Best Results (v16)](#6-current-best-results-v16)
7. [Architecture & Loss Overview](#7-architecture--loss-overview)
8. [Critical Bugs Fixed (Historical Reference)](#8-critical-bugs-fixed-historical-reference)
9. [Key Workflows & Commands](#9-key-workflows--commands)
10. [Calibration System](#10-calibration-system)
11. [Open Items Requiring Follow-up](#11-open-items-requiring-follow-up)
12. [Recommended Next Steps](#12-recommended-next-steps)
13. [Common Pitfalls](#13-common-pitfalls)

---

## 1. Project Overview

This project implements **SC-Net (Spatio-Temporal Contrast Network)**, a DETR-style dual-branch neural network for automated Coronary Artery Disease (CAD) diagnosis from Coronary CT Angiography (CCTA). The paper is:

> Ma et al., "Spatio-Temporal Contrast Network for Data-Efficient Learning of Coronary Artery Disease in Coronary CT Angiography," MICCAI 2024, pp. 645–655.

**Paper code (reference only):** https://github.com/PerceptionComputingLab/SC-Net

**Our fork:** https://github.com/reetmitra/CAD_diagnosis

The model takes curved planar reconstruction (CPR) NIfTI volumes as input and outputs:
1. **Stenosis severity** per artery — Healthy / Non-significant / Significant
2. **Plaque composition** per artery — Calcified / Non-calcified / Mixed

The task is a 6-class joint classification (3 stenosis × 2 plaque) using a dual-branch architecture:
- **Temporal branch** — 32 cubic crops along the vessel → 3D-CNN → Transformer encoder → per-point classification
- **Spatial branch** — full CPR volume → DETR-style Transformer decoder with 16 learnable object queries → bounding box regression + classification

These two branches are linked by a **Dual-task Contrastive Loss (L_dc)** that provides mutual pseudo-label supervision between them.

---

## 2. Team Status Update

### Current Assignment Status

| Assignment | Status | Notes |
|-----------|--------|-------|
| SC-Net implementation & bug fixes | **Complete** | All critical bugs resolved; codebase stable |
| Pre-training pipeline | **Complete** | v14 backbone (300 ep) is best pre-train |
| Fine-tuning pipeline | **Complete** | v16 is current best fine-tuned model |
| Calibration system | **Complete** | `calibrate.py` supports standard + constrained search |
| Evaluation pipeline | **Complete** | `eval.py` supports raw, calibrated, TTA, ensemble |
| CPR visualisation | **Complete** | `visualize.py` generates CPR images with prediction bars |
| Per-artery prediction traceability | **Complete** | `--save_predictions` flag in `visualize.py` |
| v16 fine-tuning | **Complete** | Checkpoint: `checkpoints_v16_finetune/best_model.pth` (ep130) |
| v16 pipeline (eval + viz) | **Complete** | All stages complete; results in `v16_pipeline.log` |
| Results reporting | **Complete** | `latest_report.md` (v16), `current_report_2.md` (v15), `current_report.md` (v14) |

### What is NOT Complete / Handed Over

| Item | Status | Action Required |
|------|--------|----------------|
| Model checkpoints | **Not on GitHub** | Too large (~15–20 GB each); retrieve from training server |
| Large viz outputs (viz_v12, viz_v14) | Not pushed | 650MB+; regenerate with `visualize.py` if needed |
| `predictions_v14/` (3182 per-artery JSONs) | Not pushed | 76MB; regenerate or retrieve from server |
| Paper PDF | Excluded from repo (see .gitignore) | Retrieve separately if needed |
| Dataset | **Not on GitHub** — never commit | Remains on training server only |

---

## 3. Environment & Infrastructure

### Hardware

| Component | Spec |
|-----------|------|
| OS | Ubuntu Linux 6.8.0-110-generic |
| GPUs | 2× NVIDIA RTX 3090 (24 GB each) |
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |

### Setup

```bash
# Clone repository
git clone git@github.com:reetmitra/CAD_diagnosis.git
cd CAD_diagnosis

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch torchvision einops nibabel scipy packaging
```

**Always activate venv before any command:**
```bash
source /home/reet/development/CAD_diagnosis/.venv/bin/activate
```

### Dataset Structure

```
dataset/
  train/
    volumes/   *.nii   — 3D CPR NIfTI volumes (256×64×64)
    labels/    *.txt   — 256-line files, one label per slice
  test/
    volumes/   *.nii
    labels/    *.txt
```

**Label encoding:**
- `0` = background (no stenosis)
- `1–3` = Non-significant stenosis (Calcified / Non-calcified / Mixed plaque)
- `4–6` = Significant stenosis (Calcified / Non-calcified / Mixed plaque)

**Dataset sizes:**
- Train: 2,961 arteries (APNHC patients) — internal 70/15/15 split used for training/val/test
- Held-out test: 665 arteries in `dataset/test/` (AP-NUH patients, completely separate)
- Visualisation/traceability subset: 67 arteries (from `dataset/test/`, used in viz pipeline)

> **Important:** `dataset/test/` is the proper external evaluation set. The internal test split (15% of `dataset/train/`) uses the same patient pool as training — use `dataset/test/` for any published results.

---

## 4. Repository Structure

### Core Files

| File | Purpose |
|------|---------|
| `architecture.py` | Model: `spatio_temporal_semantic_learning` — dual-branch SC-Net |
| `optimization.py` | Loss functions: `spatio_temporal_contrast_loss`, `FocalLoss`, `OrdinalEMDLoss`, `dual_task_contrastive_loss`, `object_detection_loss` |
| `augmentation.py` | `cubic_sequence_data` dataset class — loads NIfTI + label txt files |
| `framework.py` | `sc_net_framework` — wires model + loss + data together |
| `train.py` | Full `Trainer` class with all CLI arguments |
| `eval.py` | Evaluation: TTA, ensemble, detailed metrics, calibrated prediction |
| `calibrate.py` | Per-class threshold calibration via grid search on val split |
| `config.py` | `DefaultConfig` — all default hyperparameters |
| `functions.py` | `HungarianMatcher`, box utilities |
| `splitting.py` | Patient-level k-fold splitting |
| `visualize.py` | CPR visualisation — generates PNGs with GT/pred comparison bars |

### Config Files (`configs/`)

| File | Purpose |
|------|---------|
| `pretrain_default.yaml` | Default pre-training config |
| `pretrain_v13.yaml`, `pretrain_v14.yaml` | v13/v14 pre-training (300-epoch runs) |
| `finetune_v12.yaml` | v12 fine-tuning — first version to achieve good results |
| `finetune_v15.yaml` | v15 — GT-based DC fix; best balanced results |
| `finetune_v16.yaml` | v16 — boost_sig, ordinal_weight=1.5; current best |

### Key Results Files

| File | Contents |
|------|----------|
| `latest_report.md` | Full v16 results report with embedded images |
| `current_report_2.md` | v15 results report |
| `current_report.md` | v14 results report |
| `V12_RESULTS.md` | v12 results |
| `calibration_thresholds_v16.json` | v16 standard calibration thresholds |
| `calibration_thresholds_v16_sig_constrained.json` | v16 Sig-recall-constrained thresholds |
| `calibration_thresholds_v15.json`, `v15_constrained.json` | v15 calibration thresholds |
| `predictions_v16` | v16 raw eval summary JSON |
| `predictions_v16_detail/` | 67 per-artery JSONs (test subset) |
| `predictions_v15_detail/` | 67 per-artery JSONs (test subset) |
| `viz_v15/`, `viz_v16/` | CPR visualization PNGs (67 arteries each) |

### Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `run_v16_pipeline.sh` | Full v16 pipeline: calibrate → eval → viz → per-artery JSONs |
| `run_v15_pipeline.sh` | Same for v15 |
| `run_v14_overnight.sh` | Legacy overnight script (training + eval) |

---

## 5. Full Training History

All versions are sequential experiments. The headline metric is **Stenosis Macro-F1 on the held-out test set** (`dataset/test/`, 665 arteries unless noted).

### Pre-training History

| Run | Epochs | Key Features | Best Backbone Val Loss | Outcome |
|-----|--------|-------------|----------------------|---------|
| v1 | ~20 | Baseline | — | Many arch bugs; unusable |
| v2 | 143 | AMP, DDP, EMA, augmentation | — | Bugs 6–8 still present; good for pre-train |
| v3 | ~40 | + focal loss, SC weights | — | Killed — LR 1e-4 too high after loss weight changes |
| v4 | ~15 | All 8 bugs fixed | — | Killed — same LR issue post-fix |
| v5 | 52 | LR=3e-5, all bugs fixed | 5.97 | Stalled — v6 supersedes |
| v6 | 57 | Single GPU, fresh start | **3.22 (ep8)** | Best early backbone |
| v13 | 110/300 | SE fusion gates, parallel 2D/3D | — | Hardware crash at ep110; aborted |
| **v14** | **300** | SE fusion gates, parallel 2D/3D, learnable pos enc | — | **Full convergence — best backbone** |

### Fine-tuning History

| Version | Backbone | Stenosis F1 | Sig Recall | NonSig Recall | Plaque F1 | Key Change |
|---------|----------|------------|-----------|--------------|-----------|------------|
| v2-ft | v2 | — | — | — | — | Majority-class only; bugs 6–8 active |
| v5-ft | v5 (ep39) | 0.160 | — | — | 0.181 | First 6-class run; still majority-class |
| v6-ft | v6 (ep8) | 0.393 | 0.553 | — | 0.181 | First non-trivial Sig predictions |
| v7-ft | v6 | 0.585 | 0.595 | 0.581 | 0.463 | DC hold/ramp; constrained calibration breakthrough |
| v8-ft | v6 | 0.555 | — | — | — | focal_gamma=3.0 → SC branch collapse (0.814→0.749) |
| v9-ft | v6 | 0.643 | 0.456 | 0.456 | 0.488 | Ordinal loss, SWA; SC branch partially collapsed |
| v10-ft | v6 | 0.517 | — | — | 0.181 | 1D IoU fix; good loss trajectory but early stopped |
| v11-ft | v6 | 0.170 | — | — | 0.250 | T0=30 LR crash at DC activation |
| **v12-ft** | v6 | **0.739** | **0.733** | **0.639** | **0.502** | T0=60, patience=100; first strong result |
| v13-ft | v13 (ep110) | 0.577 | — | — | — | Aborted backbone → regressed |
| v14-ft | v14 (ep300) | 0.654 | 0.777 | 0.317 | 0.640 | GT-free C⁻¹ → Healthy collapse (0/979); reverted |
| **v15-ft** | v14 (ep300) | **0.825** | **0.806** | **0.847** | **0.638** | GT-based DC fix; full class recovery |
| **v16-ft** | v15-ft | **0.851** | **0.894** | **0.828** | **0.683** | boost_sig + ordinal×3 + eos_coef=0.20 |

### The v14 Incident (Critical Context)

v14 introduced a GT-free C⁻¹ implementation in L_dc. This caused a **self-reinforcing feedback loop**: the OD head generated Non-sig pseudo-labels for Healthy arteries → SC learned Non-sig=Healthy → fed back through DC → locked in. Result: 0/979 Healthy arteries correctly classified.

**v15 fixed this** by reverting OD→SC direction to GT-anchored targets (`od2sc_targets`), while keeping the improved v14 backbone and SC→OD direction as prediction-based. This is the correct architecture per the paper.

---

## 6. Current Best Results (v16)

**Checkpoint:** `checkpoints_v16_finetune/best_model.pth` (epoch 130)
**Dataset:** `dataset/test/` — 478 samples (held-out, different patient pool)

### Stenosis — Raw Argmax (recommended for deployment)

| Metric | v16 | v15 | v12 |
|--------|-----|-----|-----|
| **ACC** | **0.854** | 0.820 | 0.736 |
| **F1 (macro)** | **0.851** | 0.825 | 0.739 |
| Precision | 0.864 | 0.839 | 0.743 |
| Recall | 0.843 | 0.820 | 0.736 |
| Specificity | 0.922 | 0.906 | 0.867 |
| AUC (macro) | **0.892** | 0.868 | — |

**Per-class (v16 raw):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Healthy | 0.929 | 0.806 | 0.863 | 98 |
| Non-significant | 0.776 | 0.828 | 0.801 | 163 |
| **Significant** | **0.886** | **0.894** | **0.890** | 217 |

**AUC per class:** Healthy 0.974 | Non-sig 0.817 | Sig 0.885

### Plaque — Calibrated (recommended for deployment)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Calcified | 0.825 | 0.845 | 0.835 | 207 (val) |
| Non-calcified | 0.718 | 0.656 | 0.685 | 93 (val) |
| Mixed | 0.500 | 0.548 | 0.523 | 31 (val) |
| **Plaque Macro F1** | — | — | **0.683** | — |

> **Deployment recommendation:** Use raw argmax for stenosis (F1=0.851) and calibrated thresholds for plaque (F1=0.683). Calibration hurts stenosis on the test set — the val-optimised thresholds do not transfer perfectly to the held-out test distribution.

### Calibration Finding (v16)

Both standard and Sig-recall-constrained calibration (≥0.70 Sig recall) converged to identical thresholds. The model's raw Sig recall (0.894) already satisfies the constraint — no forced tradeoff needed.

---

## 7. Architecture & Loss Overview

### Dual-Branch Model

```
Input: CPR volume (256 × 64 × 64 NIfTI)
         |
         ├── Temporal branch
         │   32 cubic crops (each ~8×64×64)
         │   → 3D-CNN (4 pooling levels, SE fusion gates)
         │   → Positional encoding (learnable)
         │   → Transformer encoder (4 layers, 8 heads)
         │   → Linear head → 7 logits per point (bg + 6 classes)
         │   → L_sc (cross-entropy + ordinal EMD)
         │
         └── Spatial branch
             Full volume + 4 multi-view 2D projections
             → 3D+2D parallel CNN streams (SE fusion gates)
             → Transformer decoder (4 layers, Q=16 object queries)
             → Box head → [center, width] per query
             → Class head → 7 logits per query
             → L_od (CE + L1 + GIoU)
```

### Loss Functions

```
L_total = L_od + L_sc + δ × L_dc

L_dc = L_od(C(ŷ_sc), ŷ_od) + L_sc(C⁻¹(ŷ_od), ŷ_sc)

where:
  C     = SC → OD pseudo-label conversion (prediction-based, always GT-free)
  C⁻¹   = OD → SC pseudo-label conversion (GT-anchored via od2sc_targets)
  δ     = 0.5 (weight for DC branch; ramped over epochs 20–60)
```

**Key implementation notes:**
- `C(ŷ_sc)` — SC→OD: prediction-based, uses `_get_object_detection_targets(sc_detached)`
- `C⁻¹(ŷ_od)` — OD→SC: **GT-based only** using `od2sc_targets(od_targets, seq_length)`. This is the v15 fix; using predictions here causes Healthy class collapse.
- DC loss is **held at 0** for the first 20 epochs, then ramped linearly to δ over 40 epochs
- Loss components use **1:1:1 weights** in code (not 5:2:1 as written in the paper — the paper equations ≠ paper code)

### Box Representation

Boxes are 1D intervals `[center, width]` along the vessel axis. When expanded for IoU computation: `[cx, 0.5, w, 1.0]` → xyxy form `[cx-w/2, 0, cx+w/2, 1]`.

---

## 8. Critical Bugs Fixed (Historical Reference)

These bugs were present in the original paper code and required fixing before the model could train correctly.

| # | File | Bug | Fix | Impact |
|---|------|-----|-----|--------|
| 1 | `functions.py` | `box_lastdim_expansion` returned `(0,2)` for empty tensors | Return `torch.zeros` with `shape[-1]=4` | Prevented shape mismatch crash |
| 2 | `augmentation.py` | Labels 0–6 in pre-training (3-class) caused CUDA OOB | Modulo remapping `((label-1)%3)+1` | Pre-training was completely broken |
| 3 | `functions.py` | Assert on degenerate boxes crashed at epoch 129 | Replace with clamping | Training crashed mid-run |
| 4 | `functions.py` | In-place box op broke AMP autograd | Switch to `torch.cat` | AMP incompatibility |
| 5 | `optimization.py` | `FocalLoss.alpha` stayed on CPU | `register_buffer('alpha', alpha)` | Device mismatch crash |
| 6 | `architecture.py` | Temporal transformer processed length-1 sequences (bag-of-cubes) | Pass `[B, n_cubes, ...]` to `_3dcnn` | Temporal branch was disabled |
| 7 | `train.py` | Early stopping monitored test split not val split | `pattern='validation'` in `setup_data()` | Test data leaked into stopping criterion |
| 8 | `optimization.py` | DC loss GT leaking through Hungarian matching in C⁻¹ | GT-free version then reverted to GT-anchored | Feedback loop → class collapse |

> **Critical finding (2026-02-27):** The paper's code uses **1:1:1 loss weights** throughout, not the λ_L1=5, λ_iou=2 stated in the paper equations. Our early "fix" to match the equations broke convergence. The paper code is authoritative — equations were never implemented as written by the authors themselves.

---

## 9. Key Workflows & Commands

### Training

```bash
source .venv/bin/activate

# Pre-training (2-GPU)
NCCL_TIMEOUT=3600 nohup torchrun --nproc_per_node=2 --master_port=29501 train.py \
  --distributed --config configs/pretrain_v14.yaml \
  > v14_train.log 2>&1 &

# Fine-tuning from a pre-trained backbone (2-GPU)
NCCL_TIMEOUT=3600 nohup torchrun --nproc_per_node=2 --master_port=29509 train.py \
  --distributed --config configs/finetune_v16.yaml \
  --pretrained ./checkpoints_v16/best_model.pth \
  > v16_train.log 2>&1 &

# Kill training
ps aux | grep -E "torchrun|train.py" | grep -v grep | awk '{print $2}' | xargs kill
```

### Evaluation

```bash
# Raw argmax (no calibration)
python eval.py \
  --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning \
  --data_split testing \
  --detailed \
  --save_results ./results_v16.json

# Calibrated
python eval.py \
  --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning \
  --data_split testing \
  --thresholds ./calibration_thresholds_v16.json \
  --detailed
```

### Calibration

```bash
# Standard (macro-F1 optimised)
python calibrate.py \
  --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning \
  --output calibration_thresholds_v16.json \
  --grid_steps 50

# Sig-recall constrained (Sig recall ≥ 0.70)
python calibrate.py \
  --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning \
  --output calibration_thresholds_v16_sig_constrained.json \
  --grid_steps 50 \
  --constrain_sig_recall 0.70
```

### Visualisation

```bash
python visualize.py \
  --data_root ./dataset/test \
  --pattern testing \
  --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --thresholds ./calibration_thresholds_v16_sig_constrained.json \
  --use_constrained \
  --output_dir ./viz_v16 \
  --filter all \
  --save_predictions ./predictions_v16_detail
```

### Full Pipeline (one command)

```bash
# Run from repo root after training completes
nohup bash run_v16_pipeline.sh > pipeline_output.log 2>&1 &
# Or with a training PID to wait for:
nohup bash run_v16_pipeline.sh <TRAINING_PID> > pipeline_output.log 2>&1 &
```

### Monitoring

```bash
# GPU status
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Follow training log
tail -f v16_train.log

# Check if training is still running
ps aux | grep -E "torchrun|train.py" | grep -v grep
```

---

## 10. Calibration System

The calibration system (`calibrate.py`) performs per-class threshold scaling on the validation split. Predictions are made as: `pred = argmax(p_i / t_i)` where `t_i` is the threshold for class `i`.

### Calibration Modes

| Mode | Command Flag | Effect |
|------|-------------|--------|
| Standard 2D | *(none)* | Searches t0 (Healthy) and t2 (Sig); t1 (Non-sig) fixed at 1.0 → Non-sig is never predicted |
| Constrained Non-sig | `--constrain_nonsig_recall 0.10` | 3D search; forces Non-sig recall ≥ 10%; finds t1~0.35 |
| Constrained Sig | `--constrain_sig_recall 0.70` | Forces Sig recall ≥ 0.70; prevents threshold from sacrificing Sig for macro-F1 |

### Calibration Thresholds (Archived)

| Version | File | t0 (Healthy) | t1 (Non-sig) | t2 (Sig) | Notes |
|---------|------|-------------|-------------|---------|-------|
| v7-ft | `calibration_thresholds_v7_constrained.json` | 2.20 | 0.35 | 0.25 | — |
| v12-ft | `calibration_thresholds_v12_constrained.json` | 2.80 | 0.65 | 0.20 | — |
| v14 | `calibration_thresholds_v14_constrained.json` | 0.30 | 1.15 | 0.10 | Aggressive; compensates for Healthy collapse |
| v15 | `calibration_thresholds_v15_constrained.json` | 0.70 | 0.40 | 0.10 | — |
| **v16** | `calibration_thresholds_v16.json` | — | — | — | Identical to sig-constrained (constraint was not binding) |

> **Key insight:** Standard 2D calibration always collapses Non-sig recall to zero because it fixes t1=1.0. **Always use the constrained calibration** (`--constrain_nonsig_recall 0.10`) for fair evaluation of all three stenosis classes. For plaque, calibration is essential (raw F1≈0.50 → calibrated F1≈0.68).

---

## 11. Open Items Requiring Follow-up

> These are items that were active, unresolved, or identified as next priorities at handover. The person continuing this work should review each one.

### High Priority

**[FOLLOW-UP 1] Model Checkpoint Retrieval**
The best model checkpoints are **not on GitHub** due to size (each is ~1.5–2 GB per epoch × many epochs = 15–22 GB per version). The following are the critical checkpoints that must be retrieved from the training server before the server is wiped:

| Checkpoint | Path on Server | Size | Purpose |
|-----------|---------------|------|---------|
| v16 fine-tune (best) | `checkpoints_v16_finetune/best_model.pth` | ~1.5 GB | **Current best — highest priority** |
| v15 fine-tune (best) | `checkpoints_v15_finetune/best_model.pth` | ~1.5 GB | Second-best; stable baseline |
| v14 pre-train (backbone) | `checkpoints_v14/best_model.pth` | ~1.5 GB | Best backbone for future fine-tuning |
| v12 fine-tune (best) | `checkpoints_v12_finetune/best_model.pth` | ~1.5 GB | Historical reference; was best before v15 |

Use `scp` or `rsync` to copy to a persistent storage location before the server lease ends.

---

**[FOLLOW-UP 2] Healthy Recall Still at 0.806**
Both v15 and v16 plateau at Healthy recall=0.806 (19.4% of clean arteries flagged as Non-sig). This has been stable across two versions and is not improving with the current loss function changes. The root cause is the OD head firing spurious foreground queries on anatomically clean vessels.

Potential directions (none implemented yet):
- Healthy-targeted augmentation (contrast variation to create harder negatives)
- Higher `eos_coef` to further suppress spurious OD queries (currently 0.20, up from 0.15)
- Per-class asymmetric focal loss gamma (higher gamma for Healthy class to focus on hard clean cases)

---

**[FOLLOW-UP 3] Plaque Mixed Class F1 is Weak**
Mixed plaque F1 ranges from 0.35–0.56 across calibrated/uncalibrated settings, compared to Calcified F1~0.83. Mixed has only ~55 test samples and high visual overlap with both Calcified and Non-calcified. No targeted fix has been applied.

Potential directions:
- Additional supervision from 2D cross-sectional views (cross-section thumbnails are already generated in `visualize.py` but not fed back into training)
- Contrastive pre-training on plaque morphology
- Multi-window HU input (soft tissue + bone window as separate channels — calcified plaque is distinctly brighter at bone window)

---

**[FOLLOW-UP 4] Calibration Does Not Transfer Well to Test Set**
For stenosis, the val-calibrated thresholds reduce test F1 from 0.851 → ~0.750. This means calibration is optimising to the val distribution which does not perfectly represent the test distribution. This gap persists across v15 and v16.

Options:
- Train/val/test re-split with more balanced patient representation
- Temperature scaling instead of per-class threshold search (more regularised)
- Hold out a small portion of the test set for threshold selection (if dataset permits)

---

**[FOLLOW-UP 5] Paper Accuracy Target Not Reached**
The paper reports Stenosis ACC=0.914 on the fine-tuned model. Our best is 0.854 (v16). The remaining ~6% gap likely comes from:
- The paper using a larger training set (exact size not disclosed)
- Potential differences in dataset pre-processing (CPR reconstruction parameters)
- Possible ensemble or TTA in the paper's reported results (not disclosed)

This gap does not need to be closed before handover but should be noted in any publication or presentation.

---

### Medium Priority

**[FOLLOW-UP 6] SWA Model Not Evaluated Separately**
The v15 and v16 training runs saved an SWA model (`swa_model.pth`) alongside `best_model.pth`. In v14 the SWA model collapsed (Healthy=0); in v15/v16 this was not reproduced. The SWA models have not been evaluated on the test set. This could yield a small additional improvement (+1–2% F1 typically).

```bash
python eval.py \
  --checkpoint ./checkpoints_v16_finetune/swa_model.pth \
  --pattern fine_tuning --data_split testing --detailed
```

---

**[FOLLOW-UP 7] `context.md` and `CHANGELOG.md` Are Stale**
Both files document the state of the project as of approximately March–April 2026. They have not been updated to reflect v15/v16 results. The most current state is in `latest_report.md` and `current_report_2.md`. If continuing the project, update `context.md` to reflect the current best checkpoint and architecture state.

---

**[FOLLOW-UP 8] The README Is the Original Paper README**
`README.md` describes the paper's original codebase, not this implementation. It should be updated to describe the actual training pipeline, eval commands, config system, and calibration workflow before any public release or handover to external collaborators.

---

## 12. Recommended Next Steps

In priority order for whoever continues this work:

| Priority | Action | Rationale |
|----------|--------|-----------|
| 1 | **Retrieve checkpoints** (see Follow-up 1) | All else becomes irrelevant without the model weights |
| 2 | **Evaluate v16 SWA model** (see Follow-up 6) | Quick win; may improve F1 by 1–2% with no retraining |
| 3 | **Test-time ensemble: v15 + v16** | Both models have different strength profiles; logit averaging often yields 1–3% gain |
| 4 | **Healthy recall improvement** | Target the 19.4% spurious OD queries on clean vessels; explore `eos_coef` and augmentation |
| 5 | **Mixed plaque improvement** | Multi-window input or additional 2D cross-section supervision |
| 6 | **v17 fine-tuning** | If v16 SWA + ensemble still fall short of targets, next iteration should focus on Healthy recall specifically |
| 7 | **Update README and context.md** | Required before external publication or team handover |

### Suggested v17 Config Changes (if retraining)

Based on v16 lessons:
- Keep: boost_sig=true, ordinal_weight=1.5, eos_coef=0.20, GT-based DC
- Explore: `boost_healthy=true` (mirror boost_sig pattern for Healthy class), `eos_coef=0.25–0.30` (more aggressive suppression of spurious OD), focal_gamma per-class instead of global

---

## 13. Common Pitfalls

These are failure modes encountered during the project. A new developer should read these before making any changes.

| Pitfall | What Happens | Prevention |
|---------|-------------|------------|
| GT-free C⁻¹ in L_dc | Healthy class collapses — 0/N correct | OD→SC direction **must** use `od2sc_targets` (GT-based). See v14 incident. |
| Loss fn not moved to GPU | `RuntimeError: expected cuda, got cpu` | Call `loss_fn.to(device)` alongside `model.to(device)` |
| `boxes_dimension_expansion` mutates in-place | Second + third loss terms receive corrupted targets | Deep copy targets before passing to each loss term |
| Standard calibration (2D search) | Non-sig recall = 0% | Always use `--constrain_nonsig_recall 0.10` for fair stenosis eval |
| `focal_gamma=3.0` | SC branch ACC collapses (0.814→0.749) | Keep `focal_gamma=2.0` — confirmed lesson from v8-ft |
| `T0=30` in cosine warm restarts | LR≈0 exactly when DC activates at ep20 → model never recovers | Use `T0=60` — LR is at a reasonable value when DC ramps |
| Evaluating on internal test split | Inflated metrics vs held-out test | Use `--data_root ./dataset/test` or `--data_split testing` for final results |
| `best_model.pth` ≠ best test model | Val loss metric can select an overfit checkpoint (see v14 ep130 vs ep188) | Check raw F1 per-class on the test set, not just val loss |
| Paper equations ≠ paper code | λ_L1=5, λ_iou=2 in equations, but 1:1:1 in code | **Follow the code**, not the equations. See Root Cause Analysis in `context.md`. |
| DDP NCCL timeout | Hang during ALLREDUCE → NCCL watchdog kills | Always add `NCCL_TIMEOUT=3600` before `torchrun`; ensure both GPUs are free |
| `od2sc_targets` / `sc2od_targets` on CPU | Tensors need explicit `.to(device)` | Already fixed in codebase — do not remove |

---

*Document prepared by Reet Mitra — May 2026. For questions, contact via repository issues at https://github.com/reetmitra/CAD_diagnosis/issues*
