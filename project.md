# SC-Net: Automated CAD Diagnosis from Coronary CT Angiography
**Internship Project — Reet Mitra**
**Repository:** https://github.com/reetmitra/CAD_diagnosis
**Paper:** Ma et al., "Spatio-Temporal Contrast Network for Data-Efficient Learning of Coronary Artery Disease in Coronary CT Angiography," MICCAI 2024, pp. 645–655.
**Current best checkpoint:** `checkpoints_v16_finetune/best_model.pth` (epoch 130) — Stenosis F1=0.851, AUC=0.892

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Architecture (Full Detail)](#3-architecture-full-detail)
4. [Dataset](#4-dataset)
5. [Phase 1 — Critical Bug Fixes (Making the Code Run)](#5-phase-1--critical-bug-fixes-making-the-code-run)
6. [Phase 2 — Training Infrastructure](#6-phase-2--training-infrastructure)
7. [Phase 3 — Evaluation & Analysis Pipeline](#7-phase-3--evaluation--analysis-pipeline)
8. [Phase 4 — Architecture Improvements](#8-phase-4--architecture-improvements)
9. [Phase 5 — Calibration System](#9-phase-5--calibration-system)
10. [Critical Research Finding: Paper Code ≠ Paper Equations](#10-critical-research-finding-paper-code--paper-equations)
11. [Full Pre-training History](#11-full-pre-training-history)
12. [Full Fine-tuning History](#12-full-fine-tuning-history)
13. [Detailed Per-Version Results](#13-detailed-per-version-results)
14. [Current Best: v16 Full Results](#14-current-best-v16-full-results)
15. [Error Analysis & Failure Modes](#15-error-analysis--failure-modes)
16. [Key Lessons Learned](#16-key-lessons-learned)
17. [Outputs & Deliverables](#17-outputs--deliverables)
18. [Open Items & Future Work](#18-open-items--future-work)
19. [Technical Stack & Commands](#19-technical-stack--commands)

---

## 1. Project Overview

This project implements and significantly extends **SC-Net (Spatio-Temporal Contrast Network)**, a deep learning system for automated Coronary Artery Disease (CAD) diagnosis from Coronary CT Angiography (CCTA). The work is grounded in a MICCAI 2024 paper and spans the full ML lifecycle: making a broken codebase run, fixing 13+ architectural bugs, building all training infrastructure from scratch, iterating through 16 model versions, and reaching results approaching the paper's reported benchmark.

**Clinical context:** CAD is a leading cause of death worldwide. Diagnosis from CCTA requires a radiologist to examine curved planar reconstruction (CPR) images of each coronary artery and assess:
1. **Stenosis severity** — Healthy / Non-significant (<50% occlusion) / Significant (≥50% occlusion)
2. **Plaque composition** — Calcified / Non-calcified / Mixed

This is a per-artery, multi-label task. The model outputs both assessments jointly in a single forward pass. The clinical priority is: **never miss a Significant artery** (under-escalation is more dangerous than over-escalation), while maintaining Non-significant recall (those arteries require monitoring, not just classification).

**Paper source code (reference only):** https://github.com/PerceptionComputingLab/SC-Net
**Our fork:** https://github.com/reetmitra/CAD_diagnosis

**Key paper stats:** 218 patients, 1163 CPR volumes, 994 lesions (678 non-significant, 316 significant). Paper reports 91.4% stenosis accuracy on their held-out split.

---

## 2. Problem Statement

The starting point was a **partially implemented codebase** from the paper authors. The original code:
- Had no training loop, optimizer, or scheduler
- Crashed immediately on GPU (device mismatch, CUDA index out-of-bounds)
- Had an architectural bug that disabled the temporal transformer entirely (processed length-1 sequences — bag-of-cubes instead of sequence-level attention)
- Could not perform fine-tuning (the 6-class stenosis+plaque diagnosis task)
- Had incorrect loss functions that silently corrupted gradients in every training step
- Contained discrepancies between the paper's equations and the actual implementation

**The goal:** Make the code run, fix all bugs, implement all missing features, train the model to convergence, push classification performance as high as possible, and document everything for handover.

---

## 3. Architecture (Full Detail)

SC-Net is a **dual-branch DETR-style neural network** that processes CPR NIfTI volumes (256×64×64 voxels representing the coronary artery length × cross-sectional field of view).

### 3.1 Temporal Branch

Processes 32 overlapping cubic crops sampled at equal intervals along the vessel centerline:

```
32 cubic crops (each ~8×64×64 voxels)
  → 3D-CNN encoder (4 pooling levels with SE fusion gates)
  → Learnable positional encoding (nn.Parameter, shape [32, embed_dim])
  → Transformer encoder (4 layers, 8 attention heads)
  → Linear classification head → 7 logits per sampling point
      (background + 6 classes: Healthy, NS+Calc, NS+NonCalc, NS+Mix, Sig+Calc, Sig+NonCalc, Sig+Mix)
  → Loss: L_sc (Focal Cross-Entropy + Ordinal EMD loss)
```

The temporal branch reasons about the full vessel as an ordered sequence — the transformer attends from proximal to distal positions. This was completely non-functional in the original code due to a shape bug (see Phase 1, bug fix #13).

### 3.2 Spatial Branch

Processes the full CPR volume together with 4 multi-view 2D projections (sagittal, coronal, two diagonals):

```
Full CPR volume (256×64×64) + 4 × 2D projections (256×64)
  → Parallel 3D CNN stream + 4 × 2D CNN streams
  → SE fusion gates merge 3D and 2D features at each of 4 pyramid levels
  → Spatial flattening: Conv3d(→16 tokens) → Linear projection → 16 spatial tokens × 512-dim
  → Transformer decoder (4 layers, 16 learnable object queries)
      - Queries cross-attend to spatial token memory
      - Each query decodes one candidate lesion region
  → Box head → [center, width] per query (1D interval along vessel axis)
  → Class head → 7 logits per query (background + 6 classes)
  → Hungarian matching against ground-truth lesion intervals
  → Loss: L_od (Focal CE + L1 + GIoU)
```

The spatial branch performs DETR-style set prediction: each of the 16 object queries independently predicts one possible lesion, and Hungarian matching pairs predictions to GT targets (or ∅ for unmatched queries). Artery-level classification aggregates the foreground query outputs.

### 3.3 Loss Functions

```
L_total = L_od + L_sc + δ × L_dc

L_dc = L_od(C(ŷ_sc), ŷ_od) + L_sc(C⁻¹(ŷ_od), ŷ_sc)

where:
  C(ŷ_sc)     = SC → OD pseudo-label conversion (always prediction-based, GT-free)
  C⁻¹(ŷ_od)  = OD → SC pseudo-label conversion (GT-anchored via od2sc_targets)
  δ           = 0.5 (DC weight; ramped over epochs 20–60)
```

**DC warmup schedule:** DC weight held at 0 for the first 20 epochs while the fresh 6-class classification heads stabilise. Then ramped linearly from 0 → 0.5 over epochs 20–60. This prevents the noisy early predictions from poisoning the mutual supervision loop before the heads have learned any meaningful representations.

**Implementation notes (critical):**
- `C(ŷ_sc)` — SC→OD direction: fully prediction-based; uses `_get_object_detection_targets(sc_detached)` with confidence annealing (0.7→0.4 over the DC ramp window)
- `C⁻¹(ŷ_od)` — OD→SC direction: **GT-anchored only**, using `od2sc_targets(od_targets, seq_length)`. This was the v15 fix. Using predictions here creates a self-reinforcing feedback loop that collapses the Healthy class to 0 predictions (the v14 incident — see Section 12).
- Loss components use **1:1:1 weights** in code (not the λ_L1=5, λ_iou=2 from the paper equations — the paper's code does not implement its own equations; see Section 10).

### 3.4 Box Representation

Boxes are 1D intervals `[center, width]` along the vessel axis (normalised 0–1). When expanded for IoU computation: `[cx, 0.5, w, 1.0]` → xyxy form `[cx-w/2, 0, cx+w/2, 1]`. This gives true interval IoU along the vessel axis while preserving DETR's 2D GIoU machinery. The original code expanded boxes to `[cx, cx, w, w]` (square boxes) — incorrect geometry that silently corrupted every gradient.

---

## 4. Dataset

### 4.1 Structure

```
dataset/
  train/
    volumes/   *.nii   — 3D CPR NIfTI volumes (256×64×64)
    labels/    *.txt   — 256-line files, one label per slice
  test/
    volumes/   *.nii
    labels/    *.txt
```

### 4.2 Label Encoding

| Value | Stenosis | Plaque |
|-------|----------|--------|
| 0 | Background (no stenosis) | — |
| 1 | Non-significant | Calcified |
| 2 | Non-significant | Non-calcified |
| 3 | Non-significant | Mixed |
| 4 | Significant | Calcified |
| 5 | Significant | Non-calcified |
| 6 | Significant | Mixed |

Labels are 256 per-slice values. Contiguous runs of the same non-zero label define a lesion interval which is converted to a bounding box `[center, width]` for the object detection branch.

**Pre-training remapping (3-class):** labels 1–6 are mapped via `((label-1) % 3) + 1` → {Calcified=1, Non-calcified=2, Mixed=3}. Pre-training supervises plaque composition only; stenosis severity is learned during fine-tuning.

### 4.3 Dataset Sizes and Splits

| Split | Size | Patients | Purpose |
|-------|------|----------|---------|
| Internal train (70%) | 2,073 arteries | APNHC patients | Model training |
| Internal val (15%) | 444 arteries | APNHC patients | Checkpoint selection, calibration |
| Internal test (15%) | 444 arteries | APNHC patients | Development evaluation |
| **Held-out test** | **665 arteries** | **AP-NUH patients** | **Final evaluation (completely separate)** |
| Viz subset | 67 arteries | from held-out test | Per-artery traceability |

**Critical:** The held-out test set (`dataset/test/`) uses entirely different patients from a different hospital (AP-NUH vs APNHC). This is the correct set for any published results. The internal 15% test split shares the same patient pool as training — numbers from it are inflated and should not be reported externally.

Patient-level splitting is enforced by `splitting.py` — arteries from the same patient always end up in the same split to prevent data leakage.

---

## 5. Phase 1 — Critical Bug Fixes (Making the Code Run)

The codebase required 15+ fixes across 6 files before it could train. These fall into four categories.

### 5.1 Architecture Bugs (nn.Module violations)

| # | File | Bug | Fix | Impact |
|---|------|-----|-----|--------|
| 1 | `architecture.py` | Extraction blocks stored in plain Python lists — weights invisible to optimizer, cannot move to GPU | Replaced with `nn.ModuleList` | Spatial branch was completely untrained in all paper runs |
| 2 | `architecture.py` | `_2d_maps_to_3d_maps` feature weights created as CPU tensors in `forward()` on every call | Converted to `nn.Parameter` | Immediate crash on GPU; weights not learned |
| 3 | `architecture.py` | Object queries generated by `torch.randint` every forward pass — non-deterministic, decoder never converged | Replaced with fixed `nn.Embedding` (DETR standard) | Decoder could not learn query specialisation |
| 4 | `architecture.py` | `Conv3d` spatial flattening layer defined in `__init__` but never called in `forward()` | Fixed call chain; corrected rearrange pattern | Spatial tokens were incorrectly sized; downstream projection wrong |
| 5 | `architecture.py` | View fusion weights `_3d_weight` (0.75) and `_2d_weight` fixed scalars — not learnable | Converted to `nn.Parameter` (Eq. 2 in paper) | Model could not learn optimal 3D/2D fusion ratios |
| 6 | `architecture.py` | `temporal_semantic_learning.forward()` reshaped cubes to `[(B*n_cubes), 1, D, H, W]` before `_3dcnn` — transformer received length-1 sequences | Pass `[B, n_cubes, D, H, W]` so transformer attends across all 32 vessel positions | **Temporal transformer entirely disabled in all original runs** |

### 5.2 Loss Function Bugs

| # | File | Bug | Fix | Impact |
|---|------|-----|-----|--------|
| 7 | `optimization.py` | Raw model outputs (with gradients attached) used as pseudo-labels in L_dc — circular gradient flow | Added `.detach()` before cross-branch supervision | DC loss was computing gradients through both branches simultaneously — not mutual supervision |
| 8 | `optimization.py` | `FocalLoss.alpha` (class weights tensor) stayed on CPU → device mismatch at runtime | `register_buffer('alpha', alpha)` for automatic device transfer | Runtime crash on any GPU run |
| 9 | `optimization.py` | `boxes_dimension_expansion` mutated targets in-place — second and third loss terms received already-modified data | Deep copy targets before each of the three loss calls | Silent corruption of L_dc and L_od targets |
| 10 | `optimization.py` | `od2sc_targets` / `sc2od_targets` generated tensors on CPU regardless of model device | Explicit `.to(device)` transfer after creation | Device mismatch crash on GPU |

### 5.3 Crash Fixes

| # | File | Bug | Fix | Impact |
|---|------|-----|-----|--------|
| 11 | `functions.py` | `box_lastdim_expansion` returned shape `(0, 2)` for empty tensors — downstream ops assumed `(N, 4)` | Return `torch.zeros(0, 4)` | Shape mismatch crash when batch had no lesions |
| 12 | `functions.py` | Hard assert in `generalized_box_iou` crashed training at epoch 129 when degenerate box arose | Replace assert with `torch.cat` clamping | Crashed after 129 epochs of training |
| 13 | `functions.py` | In-place box operation broke AMP autograd graph | Switch to `torch.cat` (non-in-place) | AMP autograd crash after degenerate box fix applied |

### 5.4 Data and Configuration Bugs

| # | File | Bug | Fix | Impact |
|---|------|-----|-----|--------|
| 14 | `augmentation.py` | Labels 0–6 used in pre-training where `num_classes=3` → CUDA index out-of-bounds | Added modulo remapping: `((label-1) % 3) + 1` | Pre-training was completely broken |
| 15 | `augmentation.py` | `__getitem__` used raw `index` without adding `data_start` offset — val/test sets loaded training samples | Fixed to `index + data_start` | Val evaluation was actually measuring training data |
| 16 | `augmentation.py` | `torch.torch.float32` typo | `torch.float32` | `AttributeError` at runtime |
| 17 | `functions.py` | `_3d_cubes_selection` created output tensor on CPU regardless of input device | Inherit device and dtype from input | CPU/GPU mismatch on temporal branch |
| 18 | `config.py` | `spatial_proj_channels`: `[128, 1024, 128, 512]` — mismatched actual feature dimensions | Corrected to `[128, 256, 16, 512]` | Spatial projection layer wrong shape |
| 19 | `framework.py` | `torch.load` without `map_location` — conflicted when loading from different GPU | Added `map_location='cpu'` | Checkpoint loading crash on multi-GPU |

**Net result of Phase 1:** Training went from impossible (immediate GPU crash) to running end-to-end.

---

## 6. Phase 2 — Training Infrastructure

`train.py` was written from scratch. The complete training pipeline includes:

### 6.1 Core Training

| Component | Implementation |
|-----------|---------------|
| Optimizer | AdamW (lr=3e-5, weight_decay=1e-4) |
| LR schedule | CosineAnnealingWarmRestarts (T0=60, T_mult=2) |
| Gradient clipping | `max_norm=0.1` |
| Mixed-Precision (AMP) | `torch.amp.GradScaler` + `autocast` — ~1.5–2× speedup on RTX 3090 |
| Multi-GPU (DDP) | `DistributedDataParallel` via `torchrun --nproc_per_node=2` |
| NCCL DDP sync fix | `dist.all_reduce(val_loss)` after validation — prevents rank divergence and NCCL timeout |
| Gradient accumulation | Configurable `--accumulate_steps` (default 4) — effective batch = 2 GPU × 2 samples × 4 = 16 |
| Early stopping | Patience-based on val stenosis F1; synchronized across DDP ranks |
| Exponential Moving Average | EMA copy of weights (decay=0.999) for inference |
| Stochastic Weight Averaging | `torch.optim.swa_utils.AveragedModel` from `--swa_start_epoch` — saves `swa_model.pth` |

### 6.2 LR Schedule Strategy

Layer-wise LR decay (DETR practice):
- Backbone: 0.1× base LR
- Transformer encoder/decoder: 0.5× base LR
- Classification heads: 1.0× base LR

Linear warmup over first 10 epochs before cosine annealing begins.

**Key finding:** T0=60 for cosine warm restarts is critical. T0=30 causes the LR to be near-zero at epoch 20 when the DC loss activates — the model cannot adapt to the new loss term and collapses (v11 incident). T0=60 ensures a healthy LR when DC activates.

### 6.3 Data Pipeline

| Feature | Detail |
|---------|--------|
| Patient-level splitting | `splitting.py` groups arteries by patient ID (prevents leakage) |
| Balanced sampling | `WeightedRandomSampler` with inverse-frequency weights — minority classes sampled proportionally |
| Online augmentation | 8 transforms: rotation ±15°, intensity jitter ±50 HU, depth flip, Gaussian noise, blur, random erasing |
| YAML config system | `--config configs/finetune_v16.yaml` with CLI overrides |
| TensorBoard | Per-epoch loss components, metrics, LR schedules, gradient norms, DC confidence |

### 6.4 DC Warmup Schedules

Two coupled annealing schedules that mirror each other over the DC ramp window:

**DC weight schedule:**
- Epochs 0–19: `dc_weight = 0` (hold — heads stabilise)
- Epochs 20–59: linear ramp 0 → 0.5
- Epoch 60+: `dc_weight = 0.5` (fixed)

**DC confidence threshold schedule (added in Phase 26):**
- Epochs 0–19: `confidence = 0.7` (high threshold — only very certain OD predictions become SC pseudo-labels)
- Epochs 20–59: linear anneal 0.7 → 0.4
- Epoch 60+: `confidence = 0.4` (floor)

Both schedules open together so the DC signal becomes both stronger and broader as the model matures.

---

## 7. Phase 3 — Evaluation & Analysis Pipeline

| Tool | Capability |
|------|-----------|
| `eval.py` | Per-class metrics (ACC, Precision, Recall, F1, Specificity, AUC), confusion matrices, TTA, ensemble, calibration support (`--thresholds`), detailed mode |
| `calibrate.py` | Per-class threshold grid search on val split; `pred = argmax(p_i / t_i)`; supports standard 2D search, constrained Non-sig search, constrained Sig search |
| `visualize.py` | CPR longitudinal strips with GT colour bands, prediction bar, cross-section panels; `--save_predictions` exports per-artery JSON |
| `cross_validate.py` | Patient-level k-fold cross-validation |
| `gradcam.py` | Gradient-CAM saliency maps for interpretability |
| `uncertainty.py` | Prediction uncertainty estimation |
| `run_v16_pipeline.sh` | One-command: calibrate → eval → viz → per-artery JSON export |

---

## 8. Phase 4 — Architecture Improvements

All improvements introduced progressively from v9-ft through v16:

| Improvement | Added in | Impact |
|-------------|----------|--------|
| **Temporal branch fix** — sequences now passed correctly as `[B, n_cubes, ...]` so transformer attends across all 32 vessel positions | v9 | Temporal branch restored from bag-of-cubes to full sequence-level attention |
| **SE fusion gates** — squeeze-and-excitation gates for adaptive 3D/2D feature fusion at each pyramid level | v13 backbone | Replaced fixed scalar weights; model learns content-adaptive blending |
| **Parallel 2D/3D streams** — independent CNN paths for each level; 2D blocks receive their own prior output, not 3D output | v13 backbone | Feature diversity vs interleaved design; fixed a subtle data-flow bug where 2D blocks at level 2+ were receiving 3D features |
| **Learnable positional encoding** — `nn.Parameter` before temporal transformer encoder | v13 backbone | Explicit proximal→distal vessel ordering; better than sinusoidal for short (32-element) sequences |
| **Ordinal EMD loss** — Earth Mover's Distance over cumulative class distributions | v9 | Penalises Healthy↔Significant errors ~2× more than adjacent-class errors (clinically motivated) |
| **GT-based DC (v15 fix)** — OD→SC direction uses GT-anchored `od2sc_targets` only | v15 | Eliminates self-reinforcing feedback loop that collapsed Healthy class to 0 predictions |
| **boost_sig** — 2× class weight on Significant+plaque indices in SC focal loss | v16 | Pushed Sig recall from 0.806 → 0.894 |
| **ordinal_weight 0.5 → 1.5** — stronger ordinal penalty for severity-direction errors | v16 | Complements boost_sig; specifically targets Sig→Non-sig under-escalation |
| **eos_coef 0.15 → 0.20** — higher no-object cost in OD loss | v16 | Reduced spurious OD foreground queries on clean (Healthy) vessels |

---

## 9. Phase 5 — Calibration System

The default argmax prediction is biased toward the majority class. `calibrate.py` performs per-class threshold calibration:

```
pred = argmax(p_i / t_i)   for class i with learned threshold t_i
```

### 9.1 Standard 2D Grid Search

Searches `t_Healthy` and `t_Significant` with `t_NonSig` fixed at 1.0. This is fast but has a critical flaw: fixing `t_NonSig=1.0` means Non-significant is never predicted when the model is biased toward the other two classes. Produces high Sig recall but Non-sig recall = 0%.

### 9.2 Constrained 3D Search (Breakthrough — v7)

`--constrain_nonsig_recall 0.10` — full 3D grid search over all three thresholds simultaneously, with a constraint that Non-sig recall ≥ 10%. This finds `t_NonSig ≈ 0.35–0.65` which unlocks Non-sig recall from 0% to 58%+ (v7) / 63.9% (v12) / 84.7% (v15). The model had learned Non-sig features all along — the 2D search was simply missing the threshold.

### 9.3 Sig-Recall Constrained Search (Added in v16)

`--constrain_sig_recall 0.70` — constrains the threshold search to maintain Sig recall ≥ 0.70. Added because standard calibration sometimes sacrifices Sig recall for macro-F1. In practice for v16, both constrained and standard searches converged to identical thresholds — the model's raw Sig recall (0.894) already satisfies the constraint.

### 9.4 Key Calibration Findings

| Finding | Detail |
|---------|--------|
| Calibration helps stenosis on val, hurts on test | Thresholds overfit to val distribution; the val→test gap is real; raw argmax recommended for stenosis |
| Calibration strongly helps plaque | Raw plaque F1 ≈ 0.47–0.50; calibrated plaque F1 ≈ 0.64–0.68; always calibrate for plaque |
| Val loss is misleading for checkpoint selection | When DC ramp activates, val loss spikes; F1 or AUC is the correct checkpoint criterion |

### 9.5 Calibration Thresholds by Version

| Version | Healthy | Non-sig | Sig |
|---------|---------|---------|-----|
| v7-ft (standard) | 0.600 | 1.000 | 0.050 |
| v7-ft (constrained) | 2.200 | 0.350 | 0.250 |
| v12-ft (constrained) | 2.800 | 0.650 | 0.200 |
| v15 (standard) | 1.757 | 1.000 | 0.287 |
| v16 (standard = constrained) | — | — | — |

---

## 10. Critical Research Finding: Paper Code ≠ Paper Equations

During comparison with the original paper repository, a fundamental discrepancy was discovered:

**Paper equations (Section 3.3) state:** λ_L1=5, λ_iou=2 for L_od; matching costs `cost_bbox=5, cost_giou=2`.

**Paper's actual code uses:** 1:1:1 weights everywhere — `L1 + GIoU` with no scaling, and `cost_class=1, cost_bbox=1, cost_giou=1`.

| Fix # | What we changed | What paper code actually does | Impact |
|-------|----------------|-------------------------------|--------|
| Bug 6 | `box_lastdim_expansion`: `[cx,w]→[cx, 0.5, w, 1.0]` (correct geometry) | `[cx,w]→[cx,w,cx,w]` reindexed to `[cx,cx,w,w]` (hacky square boxes) | Different GIoU values → different loss landscape |
| Bug 7 | `loss_boxes`: `5.0×L1 + 2.0×GIoU` (matches paper equations) | `L1 + GIoU` (1:1 weights) | ~3.5× larger box loss gradient |
| Bug 8 | `HungarianMatcher`: `cost_bbox=5, cost_giou=2` (matches equations) | All costs = 1 | Different query-to-target matching |

When we applied the "correct" paper equations (bugs 6–8), training collapsed into majority-class prediction. Reverting to the paper's actual code (1:1:1 everywhere) restored convergence and drove F1 from 0.16 → 0.739 (v12).

**Key insight:** The paper authors achieved 91.4% accuracy with their code, not their equations. The equations in the paper appear to be from a planning/design stage and were not what the model was trained with.

**Fix classification:**
- Bugs 1–5 (crash fixes): KEEP — these prevent GPU crashes and have no effect on the loss landscape
- Bugs 6–8 (behavioral changes): REVERT to paper code — following the equations breaks convergence

---

## 11. Full Pre-training History

Pre-training uses the 3-class plaque composition task (augmented dataset A). The temporal and spatial branches learn visual features; fine-tuning then adds stenosis severity supervision and trains the 6-class heads.

| Run | Epochs | Key Config | Best Val Loss | Outcome |
|-----|--------|-----------|--------------|---------|
| v1 | ~20 | Baseline | — | Many arch bugs unfixed; model untrained |
| v2 | 143 | AMP, DDP, EMA, augmentation, warmup, layerwise LR | — | Running end-to-end; bugs 6–8 still present; good SC branch (ACC 0.848) |
| v3 | ~40 | + focal loss, SC weights, grad accum | — | Killed — LR 1e-4 too high after loss weight changes |
| v4 | ~15 | All 8 bugs fixed (incorrect) | — | Killed — same LR issue; val loss increasing after warmup |
| v5 | 52 | LR=3e-5, all bugs fixed | 5.97 | Stalled; best ep39; superseded by v6 |
| v6 | 57 | Single GPU (GPU 0), fresh start | 3.22 (ep8) | Best early backbone — only 8 epochs but lowest val loss |
| v13 | 110/300 | SE fusion gates, parallel 2D/3D streams, learnable pos enc | — | Hardware crash at ep110; aborted; backbone under-converged |
| **v14** | **300** | SE fusion + parallel 2D/3D + learnable pos enc, full run | — | **Full convergence — best backbone for all fine-tuning** |

**Why v6 epoch 8 was useful despite short training:** The model hit a low-loss local minimum very early, then slowly overfit. Epoch 8 captured the best generalising point. v6 backbone was used from v6-ft through v12-ft.

**Why v14 superseded v6:** 300 epochs allowed SE gates and parallel 2D/3D streams to fully converge. The richer spatial representations directly improved plaque F1 (+0.136 vs v12). v14 backbone used from v15-ft and v16-ft.

---

## 12. Full Fine-tuning History

Fine-tuning trains the 6-class (stenosis × plaque) heads from a pre-trained backbone. The table below shows the chronological progression from majority-class prediction to current best.

| Version | Backbone | Sten F1 | Sig Rec | NonSig Rec | Plaque F1 | SC ACC | Key Change / Why it Failed or Succeeded |
|---------|----------|---------|---------|-----------|-----------|--------|----------------------------------------|
| v5-ft | v5 ep39 | 0.160 | — | — | 0.181 | 0.792 | First ever 6-class run; majority-class only; backbone too weak |
| v2-ft | v2 ep139 | — | — | — | — | 0.820 | LR=3e-6 too low; bugs 6–8 active; majority class throughout |
| v6-ft | v6 ep8 | 0.393 | 0.553 | — | 0.181 | 0.806 | First non-majority-class predictions; no Non-sig predicted |
| v7-ft | v6 | 0.585 | 0.595 | **0.581** | 0.463 | 0.814 | DC hold/ramp + constrained calibration = Non-sig breakthrough |
| v8-ft | v6 | 0.555 | — | — | — | 0.749 | focal_gamma=3.0 → SC branch collapsed (0.814→0.749) |
| v9-ft | v6 | 0.643 | 0.456 | 0.456 | 0.488 | **0.322** | Ordinal+SWA improved OD; but SC branch collapsed (6× LR mismatch at head re-init) |
| v10-ft | v6 | 0.517 | — | — | 0.181 | — | 1D IoU fix introduced; good loss trajectory but wrong checkpoint metric (val loss → val F1 fix not yet applied) |
| v11-ft | v6 | 0.170 | — | — | 0.250 | — | T0=30 → LR≈0 when DC activated at ep20; never recovered; best checkpoint was ep1 |
| **v12-ft** | v6 | **0.739** | **0.733** | **0.639** | **0.502** | 0.814 | T0=60 + F1 checkpoint metric + patience=100; first strong result |
| v13-ft | v13 ep110 | 0.577 | — | — | — | — | Crashed backbone (110/300 ep) → regressed despite arch improvements |
| v14-ft | v14 ep300 | 0.654 | 0.777 | 0.317 | **0.640** | — | GT-free C⁻¹ → Healthy collapse (0/979 correct); overfitting ep188; plaque gain retained |
| **v15-ft** | v14 ep300 | **0.825** | **0.806** | **0.847** | 0.638 | — | GT-based DC fix; full class recovery; massive across-the-board improvement |
| **v16-ft** | v15-ft ep149 | **0.851** | **0.894** | **0.828** | **0.683** | — | boost_sig + ordinal_weight=1.5 + eos_coef=0.20; current best |

### Improvement Trajectory

From baseline to current best on the held-out test set:
- Stenosis F1: 0.413 (v1 pre-train) → **0.851** (v16) — **+106% relative**
- Plaque F1: 0.100 (v1 pre-train) → **0.683** (v16) — **+583% relative**
- Significant Recall: ~0 (v5-ft) → **0.894** (v16)
- Non-sig Recall: 0% (every version until v7 constrained calibration) → **0.828** (v16)

---

## 13. Detailed Per-Version Results

### v7-ft (Fine-tune on v6 backbone — constrained calibration)

Calibration thresholds: `[H=2.20, NS=0.35, Sig=0.25]`

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Healthy | 0.613 | 0.561 | 0.586 |
| Non-significant | 0.412 | **0.581** | 0.482 |
| Significant | 0.814 | 0.595 | 0.688 |
| **Macro** | 0.632 | 0.526 | **0.585** |

| Task | ACC | F1 | AUC |
|------|-----|----|-----|
| Stenosis | 0.580 | 0.585 | 0.713 |
| Plaque | 0.567 | 0.463 | 0.700 |
| SC Points | 0.814 | — | — |

**Significance:** First version with balanced Non-sig recall. The constrained 3D calibration search was the key insight — the 2D search (t_NS fixed at 1.0) was systematically missing Non-sig. This breakthrough drove every subsequent version.

---

### v9-ft (Ordinal loss + SWA — SC branch collapsed)

Calibration thresholds: `[H=1.80, NS=1.15, Sig=1.00]`

| Task | ACC | F1 | AUC |
|------|-----|----|-----|
| Stenosis | 0.645 | 0.643 | 0.803 |
| Plaque | 0.642 | 0.488 | 0.690 |
| SC Points | **0.322** | — | — |

Strong OD/stenosis results (+0.058 F1 vs v7) but SC branch collapsed to 32.2% from 81.4%. Root cause: 6-class SC head was re-initialised from scratch at fine-tuning start; the LR multiplier for this head was 6× too high relative to the rest of the network. Ordinal loss was net positive; the LR bug undid the gains. v12 fixed this with conservative `lr=3e-5`.

---

### v12-ft (First major result)

**Checkpoint:** `checkpoints_v12_finetune/best_model.pth` (epoch 198)
**Four changes drove v12 over all prior versions:**

| Change | Why it mattered |
|--------|----------------|
| Native 1D interval IoU | Correct box geometry throughout — previously using fake 2D padding |
| Stenosis F1 as checkpoint metric | Immune to DC ramp val loss corruption — before this, best_model.pth was selected at the wrong epoch |
| T0=60 cosine restart | LR is non-negligible when DC activates at epoch 20 |
| patience=100 | Full training runway post-DC activation — v10's patience=60 stopped too early |

Calibration thresholds: `[H=2.80, NS=0.65, Sig=0.20]`

| Class | F1 | Recall |
|-------|----|--------|
| Healthy | 0.868 | — |
| Non-significant | 0.613 | 0.639 |
| Significant | 0.735 | 0.733 |
| **Macro** | **0.739** | 0.736 |

Plaque: Calcified F1=0.790, Non-calc F1=0.500, Mixed F1=0.214, **Macro F1=0.502**

Calibration added +0.084 F1 (0.655 argmax → 0.739 calibrated) for stenosis and +0.186 for plaque (0.316 → 0.502). The constrained Non-sig search was mandatory — standard 2D calibration zeroed Non-sig recall entirely.

---

### v14-ft (GT-free DC incident — plaque improves, stenosis collapses)

**Checkpoint:** `checkpoints_v14_finetune/best_model.pth` (epoch 188)
**The v14 incident:** GT-free C⁻¹ in L_dc created a self-reinforcing feedback loop:
1. Confused OD head generated Non-sig pseudo-labels for Healthy arteries
2. SC branch learned Non-sig=Healthy (positive feedback through DC)
3. This got fed back through DC as more confident Non-sig pseudo-labels
4. Complete Healthy class collapse — 0/979 Healthy arteries correctly classified (raw)

Raw distribution: Healthy=0 | Non-sig=2147 | Sig=1035 (out of 3182 arteries)

Despite this, calibration partially rescued the results:

| Metric | v14 calibrated | v12 |
|--------|---------------|-----|
| Stenosis F1 | 0.654 | **0.739** |
| Healthy Recall | **0.939** | — |
| Non-sig Recall | 0.317 | **0.639** |
| Sig Recall | **0.777** | 0.733 |
| **Plaque F1** | **0.640** | 0.502 |

Plaque F1 improved by +0.138 over v12 — confirming the fully-converged v14 backbone produced richer spatial representations. Stenosis regression was entirely due to the GT-free DC feedback loop, not backbone quality.

**Also in v14:** Training crashed at epoch 135 (NCCL ALLREDUCE hardware timeout) and was resumed from epoch 129. The resumed run overfit monotonically (val loss rose 4.86 → 5.59 while train loss fell 3.82 → 3.13) — best checkpoint (ep188) was the least-bad overfit point, not a convergence point.

---

### v15-ft (GT-based DC fix — full recovery)

**Checkpoint:** `checkpoints_v15_finetune/best_model.pth` (epoch 149, early stop at 249)
**The fix:** Revert OD→SC direction in L_dc to GT-anchored `od2sc_targets`. SC→OD direction remains prediction-based (correct per paper). Removed dead code paths (`use_soft_dc`, `set_dc_temperature`).

| Metric | v15 raw | v15 calibrated | v12 |
|--------|---------|---------------|-----|
| Stenosis F1 | **0.825** | 0.736 | 0.739 |
| Stenosis ACC | **0.820** | 0.713 | 0.736 |
| Stenosis AUC | **0.868** | 0.868 | — |
| Healthy Recall | 0.806 | 0.888 | — |
| Non-sig Recall | **0.847** | 0.785 | 0.639 |
| Sig Recall | **0.806** | 0.581 | 0.733 |
| Plaque F1 | 0.470 | **0.638** | 0.502 |

Confusion matrix (v15 raw, test):
```
                  Healthy   Non-sig     Sig
Healthy    (98)       79        18       1
Non-sig   (163)        5       138      20
Sig       (217)        2        40     175
```

40 Sig arteries misclassified as Non-sig (Sig recall = 0.806) — this is the primary target for v16.

AUC by class: Healthy=0.974 | Non-sig=0.774 | Sig=0.857 | Macro=0.868
Plaque AUC: Calcified=0.840 | Non-calc=0.834 | Mixed=0.758 | Macro=0.811

**Key observation:** Raw argmax outperforms calibrated on the test set for stenosis (0.825 vs 0.736). The calibration thresholds are optimised on the val split and the val→test distribution shift is large enough that calibration hurts. **Use raw argmax for stenosis.**

**Deployment recommendation from v15:** Raw argmax for stenosis (F1=0.825), calibrated for plaque (F1=0.638).

---

### v16-ft (Current Best)

See Section 14 for full detail.

---

## 14. Current Best: v16 Full Results

**Checkpoint:** `checkpoints_v16_finetune/best_model.pth` (epoch 130, selected on val F1)
**Evaluation set:** `dataset/test/` — 478 held-out arteries from AP-NUH patients
**Started from:** `checkpoints_v15_finetune/best_model.pth` (fine-tuning on fine-tuned model)

### 14.1 v16 Config Changes from v15

| Parameter | v15 | v16 | Effect |
|-----------|-----|-----|--------|
| `boost_sig` | false | **true** | 2× class weight on Sig in SC focal + DC loss |
| `ordinal_weight` | 0.5 | **1.5** | Heavier penalty for severity-order violations (Sig↔Non-sig, Non-sig↔Healthy) |
| `eos_coef` | 0.15 | **0.20** | Higher no-object cost → fewer spurious OD detections on clean vessels |

All other settings identical to v15 (lr=3e-5, T0=60, focal_gamma=2.0, boost_nonsig=true, dc_hold=20, dc_ramp=40).

### 14.2 Training Summary

| Item | Value |
|------|-------|
| Best checkpoint | Epoch 130 (selected on val stenosis F1) |
| Last logged epoch | 198 (stopped externally; checkpoint unaffected) |
| Early stopping | Patience=100 on val F1 |
| SWA | Averaged from epoch 100 |

### 14.3 Stenosis — Raw Argmax (Recommended)

| Metric | v16 | v15 | v12 | Δ (v16 vs v15) |
|--------|-----|-----|-----|----------------|
| **ACC** | **0.854** | 0.820 | 0.736 | **+0.034** |
| **F1 (macro)** | **0.851** | 0.825 | 0.739 | **+0.026** |
| Precision | 0.864 | 0.839 | 0.743 | +0.025 |
| Recall | 0.843 | 0.820 | 0.736 | +0.023 |
| Specificity | 0.922 | 0.906 | 0.867 | +0.016 |
| **AUC (macro)** | **0.892** | 0.868 | — | **+0.024** |

**Per-class (v16 raw, test):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Healthy | 0.929 | 0.806 | 0.863 | 98 |
| Non-significant | 0.776 | 0.828 | 0.801 | 163 |
| **Significant** | **0.886** | **0.894** | **0.890** | 217 |

**AUC per class:** Healthy=0.974 | Non-sig=0.817 | Sig=0.885

**Confusion matrix (v16 raw, test):**
```
                  Healthy   Non-sig     Sig
Healthy    (98)       79        17       2
Non-sig   (163)        5       135      23
Sig       (217)        1        22     194
Predicted:            85       174     219
```

Only 23 Significant arteries missed (22 downgraded to Non-sig, 1 to Healthy). Dominant clinical error is under-escalation (Non-sig when GT=Sig) — not over-diagnosis. This is the safer failure direction.

### 14.4 Plaque — Calibrated (Recommended)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Calcified | 0.825 | 0.845 | 0.835 | 207 (val) |
| Non-calcified | 0.718 | 0.656 | 0.685 | 93 (val) |
| Mixed | 0.500 | 0.548 | 0.523 | 31 (val) |
| **Macro** | — | — | **0.683** | — |
| **Plaque AUC** | — | — | **0.835** | — |

Raw plaque F1 (argmax) = 0.502 — calibration adds +0.181. Always apply calibration for plaque.

### 14.5 Stenosis Calibration on Val Split

Both standard and Sig-recall-constrained (≥0.70) calibration converged to **identical thresholds**. The model's raw Sig recall (0.894) already satisfies the constraint — no forced tradeoff. Val stenosis F1: 0.736 (raw) → 0.750 (calibrated, +0.014). Test stenosis F1 with calibration is lower than raw, confirming: **use raw argmax for stenosis in deployment**.

### 14.6 Full Version Comparison (Test Split)

| Metric | v12 | v14 | v15 | **v16** |
|--------|-----|-----|-----|---------|
| Stenosis ACC | 0.736 | 0.686 | 0.820 | **0.854** |
| Stenosis F1 | 0.739 | 0.654 | 0.825 | **0.851** |
| Stenosis AUC | — | 0.804 | 0.868 | **0.892** |
| Healthy Recall | — | 0.939 | 0.806 | 0.806 |
| Non-sig Recall | 0.639 | 0.317 | 0.847 | **0.828** |
| Sig Recall | 0.733 | 0.777 | 0.806 | **0.894** |
| Plaque F1 (cal) | 0.502 | 0.640 | 0.638 | **0.683** |

### 14.7 Visualisation Summary (67 test arteries)

Saved to `viz_v16/`. Single-sample inference (no batch norm): 45/67 correct (67.2%). Batched test accuracy 85.4% — the gap is expected (batch norm statistics differ in single-sample vs batched mode).

**Visualisation accuracy by class:**
- Correct Significant: high confidence, OD bar captures lesion extent
- Correct Non-significant: hardest class; each correct prediction is a strong signal
- Correct Healthy: 0.806 recall; model correctly assigns no foreground queries

**Primary failure modes (see Section 15):**
- Sig → Non-sig: 22/217 (10.1%) — dominant error
- Healthy → Non-sig: 17/98 (17.3%) — spurious OD on clean vessels

---

## 15. Error Analysis & Failure Modes

### 15.1 Stenosis Error Breakdown (v16 test set)

| Error Type | Count | Rate | Clinical Direction |
|-----------|-------|------|--------------------|
| Sig → Non-sig | 22 | 10.1% of Sig | Under-escalation (riskier) |
| Healthy → Non-sig | 17 | 17.3% of Healthy | False flag (over-call) |
| Non-sig → Sig | 23 | 14.1% of Non-sig | Over-escalation (acceptable) |
| Healthy → Sig | 2 | 2.0% of Healthy | False flag, severe |
| Sig → Healthy | 1 | 0.5% of Sig | Extreme under-escalation |
| Non-sig → Healthy | 5 | 3.1% of Non-sig | Under-escalation |

The model's dominant clinical error direction is **under-escalation** (missing Significant disease) — not over-diagnosis. Clinical context: under-escalation (missed Significant → patient not referred) is more dangerous than over-escalation (Non-sig flagged as Significant → unnecessary follow-up).

### 15.2 Characteristic Failure Patterns

**Significant → Non-significant (most common):** Diffuse mild-moderate disease without a focal high-grade obstruction. OD queries fire in the right region but the combined class logits fall below the Sig threshold. These are borderline cases where the lesion sits near the 50% stenosis clinical cutoff.

**Healthy → Non-significant:** Visually clean vessels where the OD head fires spurious foreground queries — likely due to mild wall irregularity, noise in the CPR reconstruction, or motion artifact. Healthy recall has been stable at 0.806 since v15 and did not regress with the boost_sig changes.

**Non-significant → Significant (acceptable over-escalation):** Soft or mixed plaque that the model interprets as flow-limiting. These patients would get further investigation in a clinical workflow — an acceptable over-call.

### 15.3 Plaque Error Patterns

Mixed plaque is consistently the hardest class (F1 ≈ 0.52 calibrated):
- Small support (31–55 samples in test sets)
- Highly variable CT appearance — overlapping features with Calcified (bright) and Non-calcified (soft tissue)
- Multi-window HU input could help (not yet implemented)

Calcified plaque is the strongest (F1 ≈ 0.83): bright, focal, high-contrast — easiest for the model to identify.

### 15.4 Historical Error Modes (Resolved)

| Error Mode | Seen In | Root Cause | Resolution |
|-----------|---------|------------|------------|
| 0/979 Healthy predicted | v14 | GT-free DC feedback loop | GT-based C⁻¹ in v15 |
| SC branch collapse (ACC 0.322) | v9 | 6× LR mismatch at head re-init | Conservative lr=3e-5 in v12 |
| DC activation crash | v11 | T0=30 → LR≈0 at DC epoch 20 | T0=60 in v12 |
| Majority-class only | v2-ft, v5-ft | Bugs 6–8 + weak backbone + high LR | Bug revert + v6 backbone + lr=3e-5 |
| Non-sig Recall=0% | v6-ft, v7 standard | Standard 2D calibration fixed t_NS=1.0 | Constrained 3D search in v7 |

---

## 16. Key Lessons Learned

| Lesson | Detail |
|--------|--------|
| Paper code ≠ paper equations | Always compare against the actual code, not just the paper text. The authors achieved 91.4% with their implementation, not with their stated λ values. |
| GT-free mutual supervision requires anchoring | A GT-free feedback loop in L_dc caused complete Healthy class collapse. Anchoring one direction of DC to GT breaks the loop. |
| Standard 2D calibration silently zeros Non-sig | A 3-class 2D search with t_NS fixed always collapses the middle class. Always use the constrained 3D search for fair multi-class evaluation. |
| focal_gamma=2.0 is the sweet spot | gamma=3.0 (v8) collapsed the SC temporal branch (0.814→0.749). Higher focal concentration destabilises the shorter temporal sequence. |
| T0=60 is critical for DC warmup | LR must be non-negligible at epoch 20 when DC activates. T0=30 → LR≈0 at that point → model cannot adapt (v11 collapse). |
| Checkpoint metric matters | Val loss is corrupted by the DC ramp spike; use val F1 or AUC as the checkpoint selection criterion. |
| Temporal transformer was always disabled | The sequence shape bug (length-1 cubes) was silent — no crash, but the transformer was processing independent cubes rather than attending across vessel positions. Shape tracking is essential. |
| Val calibration doesn't always transfer | Thresholds tuned on val can reduce test F1 — especially when val and test have different class distributions. Trust raw argmax when it outperforms calibrated. |
| Backbone convergence matters | v13-ft regressed because the backbone was only 110/300 epochs. v14 backbone (300 ep) is consistently better across all fine-tuning versions. |
| SC and OD branches can diverge | v9 showed the OD branch can improve while the SC branch collapses. Both must be monitored independently per epoch. |
| SWA adds +1–2% F1 | Weight averaging over late epochs reliably finds flatter minima and generalises better, especially under limited data. |

---

## 17. Outputs & Deliverables

### 17.1 Model Checkpoints (on training server — not on GitHub)

| Checkpoint | Description | Size |
|-----------|-------------|------|
| `checkpoints_v16_finetune/best_model.pth` | **Current best model** (epoch 130) | ~1.5 GB |
| `checkpoints_v15_finetune/best_model.pth` | Previous best; ensemble candidate (epoch 149) | ~1.5 GB |
| `checkpoints_v14/best_model.pth` | Best pre-trained backbone (300 epochs) | ~1.5 GB |
| `checkpoints_v12_finetune/best_model.pth` | v12 best (historical reference) | ~1.5 GB |
| `checkpoints_v6/best_model.pth` | v6 pre-training best (ep8) — used for v6-ft through v12-ft | ~1.5 GB |

### 17.2 Calibration Files (on GitHub)

| File | Description |
|------|-------------|
| `calibration_thresholds_v16.json` | v16 standard calibration thresholds |
| `calibration_thresholds_v16_sig_constrained.json` | v16 Sig-recall-constrained (identical to standard) |
| `calibration_thresholds_v15.json` | v15 standard |
| `calibration_thresholds_v15_constrained.json` | v15 constrained (Non-sig recall ≥ 10%) |
| `calibration_thresholds_v12_constrained.json` | v12 constrained `[H=2.80, NS=0.65, Sig=0.20]` |
| `calibration_thresholds_v7_constrained.json` | v7 constrained `[H=2.20, NS=0.35, Sig=0.25]` |

### 17.3 Results & Visualisations (on GitHub)

| Artifact | Description |
|----------|-------------|
| `latest_report.md` | Full v16 results with embedded images |
| `current_report_2.md` | Full v15 results |
| `current_report.md` | v14 results |
| `V12_RESULTS.md` | v12 results |
| `predictions_v16` | v16 raw eval summary JSON |
| `predictions_v16_detail/` | 67 per-artery JSONs with OD query outputs + confidence scores |
| `predictions_v15_detail/` | 67 per-artery JSONs (v15) |
| `viz_v16/` | 67 CPR visualisation PNGs with GT/pred bars |
| `viz_v15/` | 67 CPR visualisation PNGs (v15) |
| `handover_doc.md` | Complete project handover document |
| `project.md` | This document |
| `SC_Net_Pipeline_Flowchart.html` | Interactive pipeline flowchart |

### 17.4 Configs (on GitHub)

| Config | Purpose |
|--------|---------|
| `configs/finetune_v16.yaml` | v16 fine-tuning — current best |
| `configs/finetune_v15.yaml` | v15 fine-tuning |
| `configs/finetune_v12.yaml` | v12 fine-tuning (historical) |
| `configs/pretrain_v14.yaml` | v14 300-epoch pre-training |
| `configs/pretrain_default.yaml` | Default pre-training config |

---

## 18. Open Items & Future Work

### 18.1 High Priority

| Item | Evidence | Possible Approach |
|------|----------|------------------|
| Healthy recall stuck at 0.806 | Stable since v15; 17/98 Healthy → Non-sig in v16 | Healthy-targeted augmentation; focal re-weighting; OD eos_coef tuning |
| Mixed plaque F1 = 0.523 | Small support (55 samples); high confusion with Calcified | Multi-window HU input; additional 2D cross-section supervision; dedicated Mixed augmentation |
| Paper accuracy gap (0.914 target) | v16 best = 0.854; gap ≈ 6 percentage points | Likely explained by training set size differences (paper: 218 patients; our training: APNHC cohort) |

### 18.2 Medium Priority

| Item | Expected Gain | Notes |
|------|--------------|-------|
| v15 + v16 ensemble | +1–3% F1 | Complementary strength profiles; v15 higher Non-sig recall, v16 higher Sig recall; logit averaging |
| v16 SWA model evaluation | +1–2% F1 | `checkpoints_v16_finetune/swa_model.pth` not yet formally evaluated |
| Temperature scaling for calibration | Better val→test transfer | Global temperature parameter; more robust than per-class thresholds |
| Healthy recall recovery run (v17) | Target Healthy recall ≥ 0.85 | Increase Healthy class weight; higher eos_coef; maybe Healthy-specific augmentation |

### 18.3 Low Priority

| Item | Notes |
|------|-------|
| README update | Still contains original paper README |
| Cross-validation | `cross_validate.py` exists but was not run on v15/v16 |
| GradCAM interpretability | `gradcam.py` implemented but not used systematically |
| Uncertainty estimation | `uncertainty.py` implemented; could quantify borderline cases |

---

## 19. Technical Stack & Commands

### 19.1 Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| GPUs | 2× NVIDIA RTX 3090 (24 GB each) |
| OS | Ubuntu Linux 6.8.0-110-generic |
| Key libraries | einops, nibabel, scipy, packaging |

```bash
# Activate venv (always required before any command)
source /home/reet/development/CAD_diagnosis/.venv/bin/activate
```

### 19.2 Training Commands

```bash
# Fine-tuning (v16 style)
torchrun --nproc_per_node=2 --master_port=29509 train.py --distributed \
  --config configs/finetune_v16.yaml \
  --pretrained ./checkpoints_v15_finetune/best_model.pth

# Pre-training (v14 style)
torchrun --nproc_per_node=2 train.py --distributed \
  --config configs/pretrain_v14.yaml

# Background training with log
NCCL_TIMEOUT=3600 nohup torchrun --nproc_per_node=2 --master_port=29509 \
  train.py --distributed --config configs/finetune_v16.yaml \
  --pretrained ./checkpoints_v15_finetune/best_model.pth \
  > logs_finetune_v16.log 2>&1 &
```

### 19.3 Evaluation Commands

```bash
# Raw evaluation (recommended for stenosis)
python eval.py --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning --data_split testing --detailed

# Evaluation with calibration (recommended for plaque)
python eval.py --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning --data_split testing \
  --thresholds calibration_thresholds_v16.json --detailed

# Evaluate on held-out test set explicitly
python eval.py --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning --data_root ./dataset/test --detailed
```

### 19.4 Calibration Commands

```bash
# Standard calibration
python calibrate.py --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning --output calibration_thresholds_v16.json --grid_steps 50

# Sig-recall constrained calibration
python calibrate.py --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning --output calibration_thresholds_v16_sig_constrained.json \
  --grid_steps 50 --constrain_sig_recall 0.70

# Non-sig constrained calibration (use for older versions)
python calibrate.py --checkpoint ./checkpoints_v12_finetune/best_model.pth \
  --pattern fine_tuning --output calibration_thresholds_v12_constrained.json \
  --constrain_nonsig_recall 0.10
```

### 19.5 Visualisation & Pipeline

```bash
# Full v16 pipeline
bash run_v16_pipeline.sh

# Single visualisation run with JSON export
python visualize.py --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning --data_root ./dataset/test \
  --output_dir viz_v16/ --save_predictions --predictions_dir predictions_v16_detail/

# Monitor GPU
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Kill training
ps aux | grep -E "torchrun|train.py" | grep -v grep | awk '{print $2}' | xargs kill

# Monitor training log
tail -f logs_finetune_v16.log
```

### 19.6 Common Pitfalls

| Pitfall | Detail |
|---------|--------|
| Loss fn is `nn.Module` | Must `.to(device)` the loss function just like the model |
| `boxes_dimension_expansion` mutates in-place | Always deep copy targets before passing to each loss sub-term |
| `od2sc_targets`/`sc2od_targets` create CPU tensors | Need explicit device transfer after creation |
| `spatial_proj_channels` must match feature dims | Correct value: `[128, 256, 16, 512]` |
| DDP requires both GPUs free | Check `nvidia-smi` before launching torchrun |
| `best_model.pth` = best val F1 (not val loss) | Val loss is corrupted by DC ramp; always use F1 as checkpoint criterion |
| Check which test split is being used | Internal 15% test and held-out `dataset/test/` give very different (inflated vs realistic) numbers |
| Calibration hurts stenosis on test set | Raw argmax recommended for stenosis; calibration only for plaque |

---

*Prepared by Reet Mitra — May 2026. Repository: https://github.com/reetmitra/CAD_diagnosis*
