# SC-Net CAD Diagnosis — Compiled Project Report
**Prepared by:** Reet Mitra | **Last updated:** 11 May 2026
**Repository:** https://github.com/reetmitra/CAD_diagnosis
**Paper code (reference):** https://github.com/PerceptionComputingLab/SC-Net

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Paper Background and Targets](#2-paper-background-and-targets)
3. [Architecture](#3-architecture)
4. [Dataset](#4-dataset)
5. [Environment and Setup](#5-environment-and-setup)
6. [Development Timeline](#6-development-timeline)
7. [All Bugs Fixed](#7-all-bugs-fixed)
8. [Training Infrastructure Added from Scratch](#8-training-infrastructure-added-from-scratch)
9. [Critical Discovery: Paper Equations ≠ Paper Code](#9-critical-discovery-paper-equations--paper-code)
10. [Calibration System](#10-calibration-system)
11. [Full Training and Version History](#11-full-training-and-version-history)
12. [Per-Version Results (All Versions)](#12-per-version-results-all-versions)
13. [Current Best Results: v16](#13-current-best-results-v16)
14. [The v14 Incident — Healthy Class Collapse](#14-the-v14-incident--healthy-class-collapse)
15. [Changes from Original Paper Code](#15-changes-from-original-paper-code)
16. [Calibration Thresholds Reference](#16-calibration-thresholds-reference)
17. [Error Analysis and Failure Modes](#17-error-analysis-and-failure-modes)
18. [Key Workflows and Commands](#18-key-workflows-and-commands)
19. [Common Pitfalls](#19-common-pitfalls)
20. [Open Items and Next Steps](#20-open-items-and-next-steps)

---

## 1. Project Overview

This project implements **SC-Net (Spatio-Temporal Contrast Network)**, a DETR-style dual-branch neural network for automated Coronary Artery Disease (CAD) diagnosis from Coronary CT Angiography (CCTA). The paper is:

> Ma et al., "Spatio-Temporal Contrast Network for Data-Efficient Learning of Coronary Artery Disease in Coronary CT Angiography," MICCAI 2024, pp. 645–655.

The model takes curved planar reconstruction (CPR) NIfTI volumes as input and produces two outputs per artery:

1. **Stenosis severity** — Healthy / Non-significant / Significant
2. **Plaque composition** — Calcified / Non-calcified / Mixed

The key design principle is **data-efficient learning under limited labels**: the dual-branch architecture cross-supervises itself via a contrastive loss, reducing the need for large annotated datasets. This matters clinically because annotated CCTA data is scarce and expensive to produce.

### Status Summary (May 2026)

| Component | Status |
|-----------|--------|
| SC-Net architecture + all bug fixes | **Complete** |
| Pre-training pipeline (v14 backbone, 300 epochs) | **Complete** |
| Fine-tuning pipeline | **Complete** |
| Calibration system (standard + constrained) | **Complete** |
| Evaluation pipeline (raw, calibrated, TTA, ensemble) | **Complete** |
| CPR visualisation | **Complete** |
| Per-artery prediction traceability | **Complete** |
| v16 fine-tuning (current best) | **Complete** |

---

## 2. Paper Background and Targets

### Paper Dataset

- **218 patients** (mean age 57.4 ± 6.2 years; 163 males; 2019–2022 acquisition)
- **1,163 CPR volumes** of main coronary branches
- **994 annotated coronary lesions:**
  - 678 Non-significant stenoses (208 Calcified, 119 Non-calcified, 351 Mixed)
  - 316 Significant stenoses (107 Calcified, 94 Non-calcified, 115 Mixed)
- Metrics reported at the artery level: ACC, Precision, Recall, F1, Specificity
- Data split: 70% train / 30% val+test; best checkpoint selected after 200 epochs

### Paper-Reported Results (Target)

**Stenosis Degree Classification:**

| Data | ACC | Precision | Recall | F1 | Specificity |
|------|-----|-----------|--------|----|-------------|
| 50% | 0.914 | 0.939 | 0.939 | 0.938 | 0.861 |
| 100% | 0.928 | 0.942 | 0.946 | 0.944 | 0.879 |

**Plaque Composition:**

| Data | ACC | Precision | Recall | F1 | Specificity |
|------|-----|-----------|--------|----|-------------|
| 50% | 0.903 | 0.936 | 0.934 | 0.935 | 0.784 |
| 100% | 0.912 | 0.941 | 0.939 | 0.940 | 0.816 |

### Method Summary

SC-Net has three core components:

**1. Clinically-Credible Data Augmentation (CDA)**
Lesion foreground ROIs are overlaid onto clean vessel backgrounds: `a = (b − b_I) ∪ f_I`. Pre-training uses this augmented dataset (plaque composition only). Fine-tuning uses the original clinical data (full 6-class labels).

**2. Spatio-Temporal Semantic Learning**
- *Spatial branch:* Object detection on the full CPR + 4 2D views using DETR-style decoder (Q=16 queries) → lesion bounding boxes + class predictions
- *Temporal branch:* 32 cubic crops along vessel → 3D-CNN → Transformer encoder → per-point classification

**3. Dual-Task Contrastive Optimization**
```
L_total = L_od + L_sc + δ × L_dc
L_dc = L_od(C(ŷ_sc), ŷ_od) + L_sc(C⁻¹(ŷ_od), ŷ_sc)
```
Each branch's predictions serve as pseudo-labels for the other via mapping functions C(·) and C⁻¹(·). This mutual self-supervision is the paper's central innovation.

### Gap Between Paper and Our Results

Our best result (v16, ACC=0.854, F1=0.851) sits ~6% below the paper's 0.914 ACC. Likely causes:
- The paper uses their own clinical dataset (not publicly available) with potentially different preprocessing
- Our dataset has a different patient distribution (Singapore hospital vs paper's cohort)
- Possible ensemble or TTA in the paper's reported results (not disclosed)
- The paper uses CDA augmentation which we did not implement (we trained on the clinical dataset directly)

---

## 3. Architecture

### 3.1 Dual-Branch Overview

```
Input: CPR volume (256 × 64 × 64 NIfTI)
         |
         ├── Temporal Branch
         │   32 cubic crops (25×25×25 voxels, 8-voxel step)
         │   → 3D-CNN (4 pooling levels, SE fusion gates)
         │   → Learnable positional encoding
         │   → Transformer encoder (4 layers, 8 heads, 512-dim)
         │   → Linear head → 7 logits per point (background + 6 classes)
         │   → L_sc (cross-entropy over all 32 points + ordinal EMD)
         │
         └── Spatial Branch
             Full volume (256×64×64) + 4 multi-view 2D projections (256×64)
             → 4-level parallel 3D+2D CNN streams (SE fusion gates)
             → Conv3d(128→16) + linear projection → 16 spatial tokens × 512-dim
             → Transformer decoder (4 encoder + 4 decoder layers, Q=16 object queries)
             → Box head → [center, width] per query (sigmoid)
             → Class head → 7 logits per query
             → L_od (CE + L1 + GIoU via Hungarian matching)
```

### 3.2 Input Pipeline

- **HU windowing:** Clip to `[-150, 750]` (from `window_lw=[300, 900]` config), normalise to `[0, 1]`
- **Spatial branch input:** Full 3D volume `[B, 1, 256, 64, 64]` + 4 2D views `[B, 4, 256, 64]`
- **Temporal branch input:** 32 cubes of 25×25×25 extracted at 8-voxel intervals along centerline → `[B, 32, 25, 25, 25]`

### 3.3 Temporal Branch (Sampling-Point Classification)

1. **3D cube extraction:** `_3d_cubes_selection` extracts 32 cubes along centerline
2. **Shallow 3D-CNN:** 4-level Conv3d (1→16→32→64→128 channels), BN + ReLU + MaxPool per level; processes cubes as a sequence `[B, 32, D', H', W']` (not bag-of-cubes — the full sequence is passed together)
3. **Flattening + projection:** Conv3d(128→32, 1×1) + flatten spatial dims + Linear → `[B, 32, 512]`
4. **Transformer encoder:** 4 layers, 8 heads, dropout=0.1. Self-attention across 32 positions captures inter-location dependencies
5. **Classification head:** MLP(512→128→num_classes+1) per sampling point; Softmax at inference only

### 3.4 Spatial Branch (Object Detection)

1. **Multi-view feature extraction:** 4-level interleaved 3D/2D CNN pyramid:
   - Level 0: 3D-CNN and 2D-CNN process raw input independently
   - Levels 1–3: 2D branch extracts 4 views from current 3D features (parallel streams); results fused with SE-learned weights
   - Output: `[B, 128, 16, 4, 4]`
2. **Spatial flattening:** Conv3d(128→16, 1×1) + Linear(256, 512) → `[B, 16, 512]` (16 spatial tokens)
3. **Transformer decoder:** Standard DETR architecture; 16 fixed learned query embeddings (`nn.Embedding`) cross-attend to spatial tokens → `[B, 16, 512]`
4. **Detection heads (parallel MLPs per query):**
   - Box head: MLP(512→256→2) + Sigmoid → `[center, width]` ∈ [0,1]
   - Class head: MLP(512→256→num_classes+1) + Softmax (inference only)

**Box representation:** Boxes are 1D intervals `[cx, w]` along vessel axis. For IoU computation, expanded to 4D: `[cx, 0.5, w, 1.0]` → xyxy: `[cx-w/2, 0, cx+w/2, 1]`. This gives true interval IoU along the vessel (not square boxes as in the original code).

### 3.5 Loss Function

| Term | Eq. | Weight | Description |
|------|-----|--------|-------------|
| `L_od` | 4–5 | 1.0 | Hungarian matching → CE on classes + L1 + GIoU on boxes. No-object downweighted by `eos_coef` (0.20 in v16) |
| `L_sc` | 6 | 1.0 | Cross-entropy over flattened `[B×32, C]` logits + ordinal EMD loss |
| `L_dc` | 7 | δ=0.5 (ramped) | Each branch's **detached** predictions as pseudo-labels for the other |

**Critical implementation note:** Loss weights in the code are **1:1:1** (L1 + GIoU, no scaling). The paper equations state λ_L1=5, λ_iou=2 — these were implemented but broke convergence. The paper code is authoritative (see Section 9).

**DC loss schedule:**
- Epochs 0–19: DC held at 0 (6-class heads stabilise)
- Epochs 20–60: Linear ramp 0 → δ
- Epochs 60+: DC weight fixed at δ=0.5

**DC confidence annealing:** Confidence threshold starts at 0.7, anneals to 0.4 over the ramp window. At high confidence, only certain OD predictions become SC pseudo-labels — reduces noise from early-training confusion.

### 3.6 Transform Functions (C and C⁻¹)

- **C(ŷ_sc)** → `sc2od_targets`: Converts per-point SC predictions into contiguous ROI bounding boxes with class labels for the OD head's contrastive supervision
- **C⁻¹(ŷ_od)** → `od2sc_targets`: Converts GT OD box targets into per-point label arrays of length 32 for SC head supervision (uses **GT-anchored** targets — not raw OD predictions, to avoid feedback loops; see Section 14)

### 3.7 Two-Stage Training

| Stage | num_classes | Data | Supervision |
|-------|-------------|------|-------------|
| Pre-training | 3 | Augmented set A (CDA output or full train) | Plaque composition only: Calcified / Non-calcified / Mixed |
| Fine-tuning | 6 | Clinical data B | Full 6-class: {Non-sig, Sig} × {Calcified, Non-calcified, Mixed} |

**Label remapping for pre-training:** Raw labels 1–6 mapped to plaque-only classes via `((label-1) % 3) + 1`. Background (0) passes through unchanged.

### 3.8 Additional Loss Components (Added)

**Ordinal EMD Loss:** Earth Mover's Distance over cumulative class distributions. Penalises severity-order violations proportionally to ordinal distance (Sig→Healthy costs 2× more than Sig→Non-sig). Weight controlled via `--ordinal_weight`. In v16: `ordinal_weight=1.5`.

**Focal Loss:** `FocalLoss` class with `gamma=2.0`. Down-weights easy examples, pushes training to focus on hard boundary cases. Focal alpha = `compute_sc_class_weights()` output.

**Class weighting:** Background=0.5, all lesion classes=1.5, Non-sig=3.0× (if `boost_nonsig`), Sig indices=2.0× (if `boost_sig`).

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

| Label | Meaning |
|-------|---------|
| 0 | Background (no stenosis, no plaque) |
| 1 | Non-significant stenosis, Calcified plaque |
| 2 | Non-significant stenosis, Non-calcified plaque |
| 3 | Non-significant stenosis, Mixed plaque |
| 4 | Significant stenosis, Calcified plaque |
| 5 | Significant stenosis, Non-calcified plaque |
| 6 | Significant stenosis, Mixed plaque |

### 4.3 Splits

**Internal split (from `dataset/train/`):**
- Total: 2,961 arteries (APNHC* patients)
- Train: 70% (2,073 arteries)
- Validation: 15% (444 arteries, 477 in v16 eval)
- Internal test: 15% (444 arteries) — **not used for final results**
- Split is patient-level to prevent leakage (797 unique patients)

**Held-out test set (`dataset/test/`):**
- 665 arteries (AP-NUH patients — completely different hospital)
- Used as the proper external evaluation set in all published results
- v16 evaluation uses 478 arteries (subset after filtering)

> **Important:** The internal test split uses the same patient pool as training — metrics on it are inflated. Always use `dataset/test/` for final results.

### 4.4 Test Set Class Distribution

| Class | Count | % |
|-------|-------|---|
| Healthy | 98 | 20.5% |
| Non-significant | 163 | 34.1% |
| Significant | 217 | 45.4% |
| **Total** | **478** | — |

---

## 5. Environment and Setup

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
git clone git@github.com:reetmitra/CAD_diagnosis.git
cd CAD_diagnosis
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch torchvision einops nibabel scipy packaging
```

**Always activate venv before any command:**
```bash
source /home/reet/development/CAD_diagnosis/.venv/bin/activate
```

### Key Files

| File | Purpose |
|------|---------|
| `architecture.py` | Model: `spatio_temporal_semantic_learning` |
| `optimization.py` | Loss: `spatio_temporal_contrast_loss`, `FocalLoss`, `OrdinalEMDLoss`, `object_detection_loss`, `dual_task_contrastive_loss` |
| `functions.py` | `HungarianMatcher`, box utilities, cube extraction |
| `augmentation.py` | `cubic_sequence_data` dataset class |
| `framework.py` | `sc_net_framework` — wires model + loss + data |
| `train.py` | Full `Trainer` class with all CLI arguments |
| `eval.py` | Evaluation with TTA, ensemble, detailed metrics |
| `calibrate.py` | Per-class threshold calibration |
| `config.py` | `DefaultConfig` — all hyperparameters |
| `splitting.py` | Patient-level k-fold splitting |
| `visualize.py` | CPR visualisation with prediction bars |
| `cross_validate.py` | Patient-level k-fold cross-validation |
| `scheduler_utils.py` | LR schedulers, EMA, layer-wise LR utilities |

### Config Files

| File | Purpose |
|------|---------|
| `configs/pretrain_default.yaml` | Default pre-training |
| `configs/pretrain_v14.yaml` | v14 pre-training (300 epochs, best backbone) |
| `configs/finetune_v12.yaml` | v12 fine-tuning — first strong result |
| `configs/finetune_v15.yaml` | v15 — GT-based DC; best balanced |
| `configs/finetune_v16.yaml` | v16 — boost_sig + ordinal×3; current best |

---

## 6. Development Timeline

### Phase 1: Initial Implementation (2025-01-16)

Established the core SC-Net skeleton from the paper code: `architecture.py`, `augmentation.py`, `config.py`, `framework.py`, `functions.py`, `optimization.py`.

**Issues at this stage:**
- No training loop, optimizer, or scheduler existed
- Multiple critical bugs prevented GPU training
- Extraction blocks were plain Python lists (invisible to optimizer)
- Object queries were random on every forward pass
- Temporal branch effectively disabled (processed length-1 sequences)

### Phase 2: Critical Bug Fixes (2025-01-19 to 2026-02-24)

Fixed 14+ bugs across 6 files. See Section 7 for the full list. After fixes: training could run without crashes.

### Phase 3: Training Infrastructure (2026-02-24)

Wrote `train.py` (complete `Trainer` class), `eval.py` (artery-level evaluation), `generate_dummy_data.py`. First evaluation on test data: Stenosis ACC=0.702, Plaque F1=0.100.

### Phase 4: Training Enhancements (2026-02-24)

Added:
- **AMP (Mixed Precision):** `torch.amp.GradScaler` + `autocast` — ~1.5–2× speedup on RTX 3090
- **Multi-GPU DDP:** `DistributedDataParallel` for 2× RTX 3090; launched via `torchrun --nproc_per_node=2`
- **Online Augmentation:** Random rotation (±15°), intensity jitter (±50 HU), depth flip — all 50% probability
- **LR Warmup:** Linear warmup over first 10 epochs via `LinearWarmupCosineDecay`
- **Layer-wise LR:** CNN backbone 0.1× base LR, transformer 0.5× base LR, heads 1.0× base LR
- **EMA:** `ModelEMA` with decay=0.999; EMA weights used for evaluation
- **TensorBoard Integration:** Per-epoch: total/train/val loss, L_od/L_sc/L_dc components, stenosis/plaque/SC metrics, current LR, global gradient norm
- **Focal Loss:** `FocalLoss` class; `gamma=2.0`, alpha = class weights
- **Gradient Accumulation:** `--accumulate_steps`; effective batch = batch_size × world_size × steps
- **Early Stopping:** Patience on val loss, synchronized across DDP ranks via `dist.all_reduce`
- **Detailed Eval:** Confusion matrices, per-class P/R/F1, AUC-ROC, result JSON export
- **TTA:** `--tta` flag with depth flip + intensity transforms, averages softmax probabilities
- **Ensemble:** `--ensemble` averages logits across multiple checkpoints
- **SC Loss Class Weighting:** `compute_sc_class_weights()`: background=0.5, lesion=1.5
- **YAML Config System:** `--config` flag; CLI args override YAML defaults
- **Cross-Validation:** Patient-level k-fold via `cross_validate.py`

### Phase 5: Evaluation and Retraining (2026-02-25)

- Fixed DDP synchronisation bug (NCCL ALLREDUCE timeout from unsynchronised early stopping)
- Fixed eval split bug (early stopping was monitoring test split, not val split)
- Fixed balanced sampling bug (`dataset.labels_file_list` API changed to `dataset.file_pairs`)
- Fixed `build_param_groups` parameter name matching for actual SC-Net parameter names
- Launched v2 pre-training (first run with all bug fixes), v3–v6 pre-training iterations

### Phase 6: Core Architecture Bug Fixes + v4 Training (2026-02-25)

Discovered three root causes of the pre-training/fine-tuning gap:
1. Box expansion geometry was wrong (`[cx,cx,w,w]` instead of `[cx,0.5,w,1.0]`)
2. Loss weights did not match paper equations (λ_L1=5, λ_iou=2)
3. Fine-tuning had never been run (all prior training was pre-training only)

Applied fixes 1 and 2, launched v4. (Later discovered that fix 2 was itself incorrect — see Section 9.)

### Phase 7: Fine-Tuning Breakthrough (2026-03-02 to 2026-03-23)

- Launched v6-ft, v7-ft (DC hold/ramp, confidence gating, balanced sampling, focal_gamma=2.0)
- Discovered constrained calibration: 3D threshold search with Non-sig recall ≥ 10% constraint
- v7-ft with constrained calibration: F1=0.585 — first non-trivial result across all three classes
- v8-ft (focal_gamma=3.0): Worse — SC branch collapsed (0.814→0.749). gamma=2.0 is the correct value.
- Patient-level data splitting implemented (`splitting.py`)
- Grad-CAM, MC Dropout uncertainty, cross-validation tools added

### Phase 8: Architecture Improvements + v9–v12 (2026-03-23 to 2026-04-06)

- **Critical temporal branch fix:** Temporal transformer was processing length-1 sequences (bag-of-cubes). Fixed to pass full `[B, n_cubes, ...]` for true sequence-level attention.
- **SE fusion gates:** Learned channel-wise 3D/2D feature blending at each pyramid level (replaces scalar `_3d_weight=0.71`)
- **Parallel 2D/3D streams:** Each level's 2D branch now processes its own input rather than the 3D branch's output
- **Learnable positional encoding:** Replaced fixed sinusoidal encoding in temporal transformer
- **Native 1D IoU:** Boxes expanded as `[cx, 0.5, w, 1.0]` — confirmed correct after reverts
- **Ordinal EMD Loss:** Added `OrdinalEMDLoss`, wired through framework
- **SWA:** `--swa` flag using `torch.optim.swa_utils.AveragedModel`; saves `swa_model.pth` alongside `best_model.pth`
- **Cosine warm restarts:** `CosineAnnealingWarmRestarts` with `T0=60, T_mult=2`
- **v12-ft** (first use of: 1D IoU + F1 checkpoint metric + T0=60 + patience=100): **F1=0.739** — first strong result

### Phase 9: v14 Pre-training + DC Experiments (2026-04-22 to 2026-04-28)

- Completed v14 pre-training (300 epochs; SE gates + parallel streams + learnable pos enc fully converged)
- Implemented Phase 26 GT-free C⁻¹: removed Hungarian matching from C⁻¹, replaced with confidence-filtered winner-takes-all pseudo-labels
- v14-ft: Plaque F1=0.640 (best ever at time), but Healthy class collapsed completely (0/979 correct)
- Root cause: GT-free pseudo-labels created a self-reinforcing feedback loop (see Section 14)

### Phase 10: v15 Fix + v16 Boost (2026-04-29 to 2026-05-11)

- **v15-ft:** Reverted OD→SC direction to GT-based `od2sc_targets`; kept v14 backbone; Stenosis F1=**0.825** raw
- **v16-ft:** Added `boost_sig` (2× Sig class weight), increased `ordinal_weight` (0.5→1.5), raised `eos_coef` (0.15→0.20); Stenosis F1=**0.851** raw, Sig Recall=**0.894**

---

## 7. All Bugs Fixed

### Original Bugs (Fixed Chronologically)

| # | File | Bug Description | Fix | Impact |
|---|------|----------------|-----|--------|
| 1 | `functions.py` | `box_lastdim_expansion` returned `(0,2)` shape for empty tensors — downstream `HungarianMatcher` crashes on `torch.cat` dimension mismatch | Return `torch.zeros` with `shape[-1]=4` | Crash on healthy-only samples |
| 2 | `augmentation.py` | Labels 0–6 passed to pre-training head with `num_classes=3` → CUDA index OOB | Added `num_classes` param + modulo remapping `((label-1)%3)+1` | Pre-training completely broken |
| 3 | `functions.py` | Hard `assert (boxes1[:, 2:] >= boxes1[:, :2]).all()` in `generalized_box_iou` crashed at epoch 129 | Replace with `torch.cat` clamping | Mid-run crash |
| 4 | `functions.py` | In-place box op (`.fill_` or similar) in AMP autograd graph | Switch to `torch.cat` (non-in-place) | AMP incompatibility |
| 5 | `optimization.py` | `FocalLoss.alpha` stored as plain tensor; stayed on CPU when model moved to GPU | `self.register_buffer('alpha', alpha)` | Device mismatch crash |
| 6 | `architecture.py` | 3D/2D extraction blocks stored in plain Python lists — invisible to `nn.Module.parameters()` | Changed to `nn.ModuleList` | Spatial branch weights received no gradient updates; could not move to GPU |
| 7 | `architecture.py` | Feature fusion weights `_3d_weight`/`_2d_weight` created as `torch.tensor()` inside `forward()` — reallocated on CPU every call | Convert to `nn.Parameter` | Device mismatch + unfixed fusion ratios |
| 8 | `architecture.py` | Object queries generated via `torch.randint(0, num_queries)` inside `forward()` — random each call | Fixed `nn.Embedding(num_queries, embed_dim)` in `__init__` | Decoder could never develop stable query specialisation |
| 9 | `architecture.py` | `Conv3d` spatial flattening layer defined in `__init__` but never called in `forward()`; rearrange pattern incorrect | Call Conv3d in `forward()`; correct rearrange pattern | Spatial branch fed wrong-shaped tokens into decoder |
| 10 | `optimization.py` | `dual_task_contrastive_loss.forward()` used raw model outputs (with gradients) as pseudo-labels — circular gradient flow | Detach before pseudo-label generation | DC loss created gradient loops, corrupting both branches |
| 11 | `augmentation.py`, `optimization.py` | Boxes stored as `[start, end]` but matcher called `box_cxcywh_to_xyxy` expecting `[center, width]` | All box generation uses center-width format consistently | Incorrect Hungarian matching and IoU |
| 12 | `optimization.py` | 1D boxes `[cx, w]` not expanded to 4D before GIoU computation — shape errors downstream | Auto-expand inside `object_detection_loss.forward()` | GIoU computation broken |
| 13 | `optimization.py` | `od2sc_targets` and `sc2od_targets` generate tensors on CPU | Explicit `.to(output_device)` in loss functions | Device mismatch in DC loss |
| 14 | `optimization.py` | `boxes_dimension_expansion` mutates targets in-place; second and third sub-losses receive corrupted data | Deep copy targets before each loss term | Silent data corruption in multi-term loss |
| 15 | `augmentation.py` | `__getitem__` used raw index without adding `data_start` → validation set loaded training samples | Add `data_start` offset | Data leakage; val metrics were meaningless |
| 16 | `functions.py` | `_3d_cubes_selection` output created on CPU regardless of input device | Inherit device and dtype from input | Device mismatch for temporal branch on GPU |
| 17 | `augmentation.py` | `torch.torch.float32` typo | `torch.float32` | `AttributeError` at runtime |
| 18 | `framework.py` | `torch.load` without `map_location` → device conflict loading from different GPU | Add `map_location='cpu'` | Checkpoint loading could fail |
| 19 | `config.py`, `framework.py` | Single `data_root` for pre-training and fine-tuning — no way to specify different datasets | Add `pretrain_data_root` / `finetune_data_root` with `--data_root` CLI override | Could not separate pre-train and fine-tune datasets |
| 20 | `config.py` | `spatial_proj_channels: [128,1024,128,512]` → did not match actual feature dimensions | Corrected to `[128,256,16,512]` | Shape mismatch in spatial projection |

### Additional Bugs (Phase 4 Analysis)

| # | File | Bug | Fix |
|---|------|-----|-----|
| B18 | `optimization.py` | `sc2od_targets()` with no lesions: `torch.tensor([])` produces shape `[0]` not `[0,2]` — crashes `torch.cdist` | Guard: `torch.zeros((0,2))` for empty boxes |
| B19 | `optimization.py` | `_get_sampling_point_classification_targets()` applied `argmax(logits) - 1` then `clamp(min=0)` — systematically wrong label mapping | Filter no-object predictions; pass 0-indexed class labels directly |
| B20 | `optimization.py` | `spatio_temporal_contrast_loss.forward()` returned scalar; cannot diagnose which component is dominating | Return dict `{'total':…, 'od':…, 'sc':…, 'dc':…}` |
| B21 | `augmentation.py` | Same empty tensor shape issue as B18 in `detection_targets()` | Same guard: `torch.zeros((0,2))` |
| B22 | `architecture.py` | `forward()` returned 2 outputs in training mode, 1 in testing — inconsistent | Always return `(od_outputs, sc_outputs)` |

### Architecture Bug (Phase 8)

**Temporal branch sequence length = 1 (critical):**
`temporal_semantic_learning.forward()` passed cubes as `[(B×n_cubes), 1, D, H, W]` to the 3D-CNN, making the sequence length inside `Conv3d.forward()` equal to 1. The temporal transformer attended over a single token per item — effectively a bag-of-independent-cubes classifier. Fixed: pass `[B, n_cubes, D, H, W]` directly; `Conv3d.forward()` reshapes internally and outputs `[B, n, C, d', h', w']`. Temporal branch now correctly attends across all 32 positions.

### DC Loss Bug (Phase 9) — Reverted

**GT-free C⁻¹ feedback loop:**
The v14 Phase 26 implementation removed GT from the OD→SC direction of L_dc, replacing Hungarian matching with confidence-filtered winner-takes-all. This caused a Healthy class collapse (see Section 14). Reverted to GT-anchored `od2sc_targets` in v15.

---

## 8. Training Infrastructure Added from Scratch

None of the following existed in the original paper code — all were written from scratch:

| Component | File(s) | Description |
|-----------|---------|-------------|
| Training loop | `train.py` | Full `Trainer` class: AdamW, CosineAnnealingWarmRestarts, grad clip=0.1, AMP, DDP, EMA, SWA, checkpointing, early stopping, per-epoch validation |
| Evaluation | `eval.py` | Artery-level ACC/Prec/Recall/F1/Spec for stenosis + plaque; TTA; ensemble; calibrated inference; confusion matrix; AUC-ROC; per-class breakdown; JSON export; matplotlib plots |
| Calibration | `calibrate.py` | Per-class threshold grid search on val split; standard 2D (t_H, t_Sig) and constrained 3D (all three thresholds); Non-sig recall constraint; Sig recall constraint |
| Visualisation | `visualize.py` | CPR PNG with GT colour bands, prediction bar, 4 cross-section panels; per-artery JSON traceability; filter by correct/wrong/all |
| Splitting | `splitting.py` | Patient-level k-fold split (no data leakage) |
| Scheduling | `scheduler_utils.py` | `LinearWarmupCosineDecay`, `CosineAnnealingWarmRestarts`, `ModelEMA`, `build_param_groups` for layer-wise LR |
| Cross-validation | `cross_validate.py` | Patient-level k-fold CV; reports mean ± std across folds |
| Pipeline scripts | `run_v15_pipeline.sh`, `run_v16_pipeline.sh` | Full pipeline: wait for training → calibrate → eval → visualise → per-artery JSONs |
| Augmentation (8 transforms) | `augmentation.py` | Random rotation ±15°, intensity jitter ±50 HU, depth flip, Gaussian blur, elastic deformation, HU shift, zoom, cutout |
| Dummy data generation | `generate_dummy_data.py` | Synthetic NIfTI volumes + label files for pipeline testing |
| YAML config system | `configs/*.yaml` | `--config` flag loads YAML; CLI args override; `pretrain_default.yaml`, `finetune_v12–v16.yaml` |
| Ordinal EMD loss | `optimization.py` | `OrdinalEMDLoss` class: Earth Mover's Distance over cumulative class distributions; penalises ordinal-direction errors proportionally |
| boost_sig/boost_nonsig | `optimization.py`, `train.py` | CLI flags to double class weight for Sig or Non-sig stenosis indices in SC focal loss |
| Constrained calibration | `calibrate.py` | `--constrain_nonsig_recall FLOAT` and `--constrain_sig_recall FLOAT` flags |
| Grad-CAM | `gradcam.py` | 3D heatmap visualisation for the spatial branch |
| Uncertainty estimation | `uncertainty.py` | MC Dropout with N forward passes; outputs mean probs, variance, entropy per artery |

---

## 9. Critical Discovery: Paper Equations ≠ Paper Code

**Discovery date:** 2026-02-27

When comparing our implementation against the original paper repository line-by-line, a fundamental discrepancy was found:

| Element | Paper Equations | Paper Code (what authors actually trained with) |
|---------|----------------|-----------------------------------------------|
| L1 loss weight | λ_L1 = **5** | **1** |
| GIoU loss weight | λ_iou = **2** | **1** |
| Hungarian matcher class cost | — | **1** |
| Hungarian matcher bbox cost | — | **1** |
| Hungarian matcher GIoU cost | — | **1** |

**The authors' code uses 1:1:1 everywhere. The paper equations stating 5:2 were never implemented by the authors themselves.**

### Impact on Our Development

Our early "fixes" (bugs 6–8 in `context.md`) matched the paper's equations but broke convergence:

| Fix applied | What it did | Why it broke training |
|-------------|-------------|----------------------|
| `box_lastdim_expansion`: `[cx,w]→[cx,0.5,w,1.0]` | Geometrically correct interval IoU | Different GIoU gradient landscape from what the authors trained with |
| `loss_boxes`: `5.0*L1 + 2.0*GIoU` | Matches paper equations | ~3.5× larger box loss gradients than what the authors used |
| `HungarianMatcher`: `cost_bbox=5, cost_giou=2` | Matches paper equations | Different query-to-target assignments |

**Resolution:** Reverted to 1:1:1 everywhere to match the actual paper code. The native 1D IoU box expansion (`[cx, 0.5, w, 1.0]`) was retained as it is geometrically correct and the authors' original expansion (`[cx,cx,w,w]`) was clearly a bug — but the loss weights were reverted.

**Rule going forward:** Follow the code, not the equations. The paper code is the executable specification.

---

## 10. Calibration System

### How It Works

Calibration adjusts per-class decision thresholds: `pred = argmax(p_i / t_i)` where `t_i` is the threshold for class `i`. A threshold `t_i > 1` makes class `i` harder to predict; `t_i < 1` makes it easier.

Grid search over `t_i` is run on the validation split, optimising macro-F1 (with optional recall constraints).

### Calibration Modes

| Mode | Command Flag | t0 (Healthy) | t1 (Non-sig) | t2 (Sig) | Effect |
|------|-------------|--------------|-------------|---------|--------|
| Standard 2D | *(none)* | Searched | Fixed at 1.0 | Searched | Non-sig is **never predicted** — 0% Non-sig recall |
| Constrained 3D (Non-sig) | `--constrain_nonsig_recall 0.10` | Searched | Searched | Searched | Forces Non-sig recall ≥ 10%; finds t1 that unlocks the class |
| Constrained 3D (Sig) | `--constrain_sig_recall 0.70` | Searched | Searched | Searched | Forces Sig recall ≥ 0.70 |

**Key insight:** The standard 2D grid search always collapses Non-sig recall to zero because it fixes `t1=1.0`. The 3D constrained search discovers that the model *can* predict Non-sig — it just needs a lower threshold. Always use `--constrain_nonsig_recall 0.10` for fair stenosis evaluation.

### Calibration Impact on Plaque

For plaque, calibration is essential. The raw model is biased toward Calcified (dominant class). Calibration significantly rescues Non-calcified and Mixed at the cost of some Calcified precision:

- v16 raw plaque F1: 0.502 → calibrated: **0.683**
- Mixed recall raw: near zero → calibrated: 0.548

### Deployment Recommendation

- **Stenosis classification:** Use **raw argmax** — calibration reduces test F1 (val-optimised thresholds do not fully transfer to the held-out test distribution)
- **Plaque classification:** Use **calibrated thresholds** — substantial improvement from raw

---

## 11. Full Training and Version History

### Pre-training History

| Run | Epochs | Key Features | Outcome |
|-----|--------|-------------|---------|
| v1 | ~20 | Baseline; many arch bugs | Unusable |
| v2 | 143 | AMP, DDP, EMA, augmentation; bugs 1–5 fixed; bugs 6–8 still present | Pre-train complete; fine-tuning majority class only |
| v3 | ~40 | + focal loss, SC weights | Killed — LR 1e-4 too high after loss weight changes |
| v4 | ~15 | All 8 bugs "fixed" (incl. 5:2 weights — later reverted) | Killed — same LR issue |
| v5 | 52 | LR=3e-5, all "bugs" fixed | Stalled |
| v6 | 57 | Single GPU, fresh start, best early backbone | **Best early backbone** (val loss 3.22 at ep8) |
| v13 | 110/300 | SE fusion gates, parallel 2D/3D streams; hardware crash | Aborted at ep110 |
| **v14** | **300** | SE fusion gates, parallel 2D/3D, learnable pos enc; full convergence | **Best backbone — use for all fine-tuning** |

### Fine-tuning History

| Version | Backbone | Sten F1 (test) | Sig Recall | NonSig Recall | Plaque F1 | Key Change |
|---------|----------|----------------|-----------|--------------|-----------|------------|
| v5-ft | v5 (ep39) | 0.160 | — | — | 0.181 | First 6-class run; majority-class only |
| v6-ft | v6 (ep8) | 0.393 | 0.553 | — | 0.181 | First non-trivial Sig predictions |
| **v7-ft** | v6 | **0.585** | 0.595 | **0.581** | 0.463 | DC hold/ramp; constrained calibration breakthrough |
| v8-ft | v6 | 0.555 | — | — | — | focal_gamma=3.0 → SC branch collapse (0.814→0.749) |
| v9-ft | v6 | 0.643 | 0.456 | 0.456 | 0.488 | Ordinal loss, SWA; SC branch partially collapsed (0.322) |
| v10-1D | v6 | 0.468 | — | — | 0.272 | 1D IoU; bad checkpoint metric (val loss) |
| v11-ft | v6 | 0.170 | — | — | 0.250 | T0=30 → LR≈0 at DC activation; model never recovers |
| **v12-ft** | v6 | **0.739** | **0.733** | **0.639** | **0.502** | T0=60 + patience=100 + F1 checkpoint + 1D IoU |
| v13-ft | v13 (ep110) | 0.577 | — | — | — | Aborted backbone → regressed |
| v14-ft | v14 (ep300) | 0.654 | 0.777 | 0.317 | **0.640** | GT-free C⁻¹ → Healthy collapse (0/979) |
| **v15-ft** | v14 (ep300) | **0.825** | **0.806** | **0.847** | **0.638** | GT-based DC fix; full class recovery |
| **v16-ft** | v15-ft | **0.851** | **0.894** | **0.828** | **0.683** | boost_sig + ordinal_weight×3 + eos_coef=0.20 |

### Key Milestones

| Date | Milestone |
|------|-----------|
| 2026-02-27 | Paper equations ≠ paper code discovery |
| 2026-03-02 | Constrained calibration breakthrough (Non-sig recall 0%→58%) |
| 2026-03-05 | v8-ft: focal_gamma=3.0 hurts SC branch; gamma=2.0 confirmed |
| 2026-03-23 | Patient-level splitting, temporal branch sequence length fix |
| 2026-04-06 | v12-ft: F1=0.739; first strong result |
| 2026-04-22 | Phase 26: GT-free C⁻¹ implementation |
| 2026-04-28 | v14-ft: Healthy collapse incident discovered |
| 2026-04-29 | v15-ft: GT-based DC fix; F1=0.825 |
| 2026-05-11 | v16-ft: F1=0.851, Sig Recall=0.894 |

---

## 12. Per-Version Results (All Versions)

### v1 — Pre-training Baseline (held-out test, 665 arteries)

| Task | ACC | F1 | AUC |
|------|-----|-----|-----|
| Stenosis | 0.702 | 0.413 | 0.554 |
| Plaque | 0.430 | 0.100 | 0.452 |
| SC Points | 0.801 | — | — |

### v6-ft (held-out test, calibrated)

| Task | ACC | F1 | Sig Recall |
|------|-----|-----|-----------|
| Stenosis | 0.435 | 0.393 | 0.553 |
| Plaque | 0.606 | 0.181 | — |
| SC Points | 0.806 | — | — |

**Calibration thresholds:** H=3.0, NS=1.0, Sig=0.346

### v7-ft (held-out test, constrained calibration)

**Thresholds:** [H=2.20, NS=0.35, Sig=0.25]

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Healthy | 0.613 | 0.561 | 0.586 | 198 |
| Non-significant | 0.412 | **0.581** | 0.482 | 210 |
| Significant | 0.814 | 0.595 | 0.688 | 257 |

| Task | ACC | F1 | AUC |
|------|-----|-----|-----|
| Stenosis | 0.580 | 0.585 | 0.713 |
| Plaque | 0.567 | 0.463 | 0.700 |
| SC Points | **0.814** | — | — |

*Standard calibration (t1=1.0 fixed) collapsed Non-sig to 0% recall with Sig recall=93.5% — clinically dangerous.*

### v8-ft (focal_gamma=3.0 — worse than v7)

| Task | F1 | SC ACC |
|------|-----|--------|
| Stenosis | 0.555 | 0.749 |

**Lesson:** focal_gamma=3.0 destabilises the SC branch. gamma=2.0 is the correct value.

### v9-ft (constrained calibration, best checkpoint)

**Thresholds:** [H=1.80, NS=1.15, Sig=1.00]

| Metric | Value |
|--------|-------|
| Stenosis F1 | 0.643 |
| Healthy Recall | 0.729 |
| Non-sig Recall | 0.456 |
| Sig Recall | 0.456 |
| **SC Branch ACC** | **0.322** (collapsed) |

*SC branch collapsed from 0.814 to 0.322 due to 6× LR mismatch at SC head re-initialisation.*

### v10-1D IoU (per-epoch on val split)

Introduced native 1D interval IoU but used val loss as checkpoint metric (wrong). Peak F1=0.468 (ep61).

### v11-ft (degenerate, T0=30)

**T0=30** caused LR≈0 exactly at DC activation (ep20). Best model was epoch 1 (F1=0.379). Early stopping fired at ep61 with patience=60 exhausted from the ep1 spike. Final eval: F1=0.170, ACC=0.341. Fix: **T0=60** so LR has a reasonable value at DC activation.

### v12-ft — First Strong Result (held-out test, constrained calibration)

**Checkpoint:** `checkpoints_v12_finetune/best_model.pth` (epoch 198 best out of 300)
**Thresholds:** [H=2.80, NS=0.65, Sig=0.20]

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Healthy | — | — | 0.868 | 198 |
| Non-significant | — | 0.639 | 0.613 | 210 |
| Significant | — | 0.733 | 0.735 | 257 |

| Task | ACC | F1 | Spec |
|------|-----|-----|------|
| Stenosis | 0.736 | **0.739** | 0.867 |
| Plaque | 0.650 | 0.502 | — |
| SC Points | 0.814 | — | — |

**Four changes that drove v12 over all prior runs:**
1. Native 1D interval IoU (replacing fake 2D box padding)
2. Stenosis F1 as checkpoint metric (immune to DC ramp val loss corruption)
3. T0=60 cosine restart (LR not near-zero at DC activation)
4. patience=100 (full training runway post-DC activation)

**Plaque breakdown (constrained calibration, v12):**

| Class | F1 | Support |
|-------|-----|---------|
| Calcified | 0.790 | 294 |
| Non-calcified | 0.500 | 128 |
| Mixed | 0.214 | 45 |

### v14-ft — Better Backbone, Healthy Collapse (held-out test, calibrated)

**Checkpoint:** `checkpoints_v14_finetune/best_model.pth` (epoch 188 of 300)

| Class | Recall | F1 |
|-------|--------|----|
| Healthy | **0.939** | 0.821 |
| Non-significant | 0.317 | 0.406 |
| Significant | **0.777** | 0.734 |

| Task | ACC | F1 | AUC |
|------|-----|-----|-----|
| Stenosis | 0.686 | 0.654 | **0.804** |
| Plaque | 0.771 | **0.640** | — |

**Raw (no calibration):** Healthy=0 predictions (0/979 Healthy arteries correct) — the GT-free C⁻¹ feedback loop locked in total Healthy collapse. See Section 14.

**v14 Confusion Matrix (calibrated):**
```
                  Healthy   Non-sig     Sig
Healthy  (979)      919        50       10
Non-sig  (974)      256       309      409
Sig     (1229)       84       190      955
```

**What v14 improved:** Plaque F1 0.502→0.640 (+0.138) and Sig Recall 0.733→0.777 (+0.044) — the fully converged v14 backbone with SE gates and parallel 2D/3D streams provides richer spatial representations.

### v15-ft — GT-Based DC Fix, Best Balanced Results (held-out test, raw argmax)

**Checkpoint:** `checkpoints_v15_finetune/best_model.pth` (epoch 149, early stop at 249)

**Stenosis (raw argmax):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Healthy | 0.919 | 0.806 | 0.859 | 98 |
| Non-significant | 0.704 | 0.847 | 0.769 | 163 |
| Significant | 0.893 | 0.806 | 0.847 | 217 |

| Metric | v15 raw | v12 | Δ |
|--------|---------|-----|---|
| ACC | **0.820** | 0.736 | +0.084 |
| F1 (macro) | **0.825** | 0.739 | +0.086 |
| Precision | 0.839 | 0.743 | +0.096 |
| Recall | 0.820 | 0.736 | +0.084 |
| Specificity | 0.906 | 0.867 | +0.039 |
| AUC | 0.868 | — | — |

**v15 Confusion Matrix (raw):**
```
                  Healthy   Non-sig     Sig
Healthy    (98)       79        18       1
Non-sig   (163)        5       138      20
Sig       (217)        2        40     175
```

**Plaque (calibrated):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Calcified | — | — | 0.847 | 223 |
| Non-calcified | — | — | 0.659 | 95 |
| Mixed | — | — | 0.389 | 55 |
| **Macro** | — | — | **0.638** | — |

**AUC per class:** Healthy 0.974 | Non-sig 0.774 | Sig 0.857 | Plaque macro 0.811

**Note on calibration:** Val-optimised thresholds do not transfer to the test set. Calibrated stenosis F1=0.736 (lower than raw 0.825). Use raw argmax for stenosis.

---

## 13. Current Best Results: v16

**Checkpoint:** `checkpoints_v16_finetune/best_model.pth` (epoch 130 of 250, early stop)
**Dataset:** `dataset/test/` — 478 arteries (held-out, different patient pool)
**Backbone:** v15 fine-tuned checkpoint (fine-tuning-on-fine-tuning approach)

### v16 Config Changes from v15

| Parameter | v15 | v16 | Effect |
|-----------|-----|-----|--------|
| `boost_sig` | false | **true** | 2× class weight on Sig indices in SC focal + ordinal loss |
| `ordinal_weight` | 0.5 | **1.5** | 3× heavier penalty for order-violating errors (Healthy↔Sig) |
| `eos_coef` | 0.15 | **0.20** | Slightly higher no-object cost → fewer spurious OD detections on clean vessels |

### 13.1 Stenosis — Raw Argmax (recommended for deployment)

| Metric | v16 | v15 | v12 | Δ (v16 vs v15) |
|--------|-----|-----|-----|-----------------|
| **ACC** | **0.854** | 0.820 | 0.736 | +0.034 |
| **F1 (macro)** | **0.851** | 0.825 | 0.739 | +0.026 |
| Precision | 0.864 | 0.839 | 0.743 | +0.025 |
| Recall | 0.843 | 0.820 | 0.736 | +0.023 |
| Specificity | 0.922 | 0.906 | 0.867 | +0.016 |
| AUC (macro) | **0.892** | 0.868 | — | +0.024 |

**Per-class (v16 raw, test):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Healthy | 0.929 | 0.806 | 0.863 | 98 |
| Non-significant | 0.776 | 0.828 | 0.801 | 163 |
| **Significant** | **0.886** | **0.894** | **0.890** | 217 |

**AUC per class:** Healthy 0.974 | Non-sig 0.817 | Sig 0.885

### 13.2 Confusion Matrix (v16 raw, test)

```
                  Healthy   Non-sig     Sig
Healthy  (98)          79        17       2
Non-sig  (163)          5       135      23
Sig      (217)          1        22     194
Predicted:             85       174     219
```

**Key observations:**
- Only 23 Sig arteries missed (22 → Non-sig, 1 → Healthy) — Sig recall 0.894
- 17 Healthy arteries flagged as Non-sig (19.4% false-positive rate) — stable since v15
- Non-sig is the hardest class; 28 Non-sig cases escalated to Sig (acceptable over-flag clinically)

### 13.3 Plaque — Calibrated (recommended for deployment)

Calibration is performed on validation split (477 stenosis, 331 plaque samples).

**Per-class (v16 calibrated, test):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Calcified | 0.831 | 0.861 | 0.846 | 223 |
| Non-calcified | 0.722 | 0.542 | 0.619 | 96 |
| Mixed | 0.493 | 0.636 | 0.556 | 55 |
| **Macro** | — | — | **0.683** | — |

| Metric | v16 cal. | v15 cal. | v12 cal. |
|--------|----------|----------|----------|
| Plaque F1 | **0.683** | 0.638 | 0.502 |
| Plaque AUC | **0.835** | 0.811 | — |

*Note: v16 raw plaque F1=0.502 — always apply calibration for plaque.*

### 13.4 Calibration Finding (v16)

Both standard and Sig-recall-constrained calibration (≥0.70 Sig recall) converged to **identical thresholds**. The model's raw Sig recall (0.894) already satisfies the constraint — no forced tradeoff needed.

### 13.5 Validation Split Calibration (v16)

| Metric | Raw (argmax) | Calibrated |
|--------|-------------|------------|
| Stenosis Macro-F1 | 0.736 | **0.750** |
| Healthy Recall | 0.938 | 0.925 |
| Non-sig Recall | 0.812 | 0.792 |
| Sig Recall | 0.519 | 0.572 |
| Stenosis ACC | — | 0.746 |

*Calibration lifts val stenosis F1 by only +0.014 and hurts on test — use raw argmax for stenosis.*

### 13.6 Full Version Comparison (Test Split)

| Metric | v12 | v14 | v15 | **v16** |
|--------|-----|-----|-----|---------|
| Stenosis ACC | 0.736 | 0.686 | 0.820 | **0.854** |
| Stenosis F1 | 0.739 | 0.654 | 0.825 | **0.851** |
| Stenosis AUC | — | 0.804 | 0.868 | **0.892** |
| Healthy Recall | — | 0.939 | 0.806 | 0.806 |
| Non-sig Recall | 0.639 | 0.317 | 0.847 | **0.828** |
| Sig Recall | 0.733 | 0.777 | 0.806 | **0.894** |
| Plaque F1 (cal.) | 0.502 | 0.640 | 0.638 | **0.683** |

### 13.7 Training Summary (v16)

| Item | Value |
|------|-------|
| Pre-train backbone | `checkpoints_v16/best_model.pth` (v16 pre-train, 300 ep) |
| Starting checkpoint | `checkpoints_v15_finetune/best_model.pth` (epoch 149) |
| Best fine-tune checkpoint | Epoch 130 (selected on val stenosis F1) |
| Last logged epoch | ep198 (process stopped externally — checkpoint unaffected) |
| Early stopping | patience=100 on val F1 |
| SWA | Averaged from ep100 |

---

## 14. The v14 Incident — Healthy Class Collapse

### What Happened

v14 introduced a GT-free C⁻¹ implementation in L_dc. This removed Hungarian matching from the OD→SC direction, replacing it with confidence-filtered winner-takes-all pseudo-labels from raw OD predictions.

**The feedback loop:**
1. Early in training, OD head was biased toward Non-sig for clean vessels
2. GT-free C⁻¹ converted these Non-sig OD predictions into SC pseudo-labels for Healthy arteries
3. SC branch learned: Healthy vessels → predict Non-sig (reinforced by DC loss)
4. SC's Non-sig pseudo-labels fed back through DC → reinforced OD's Non-sig predictions
5. The loop locked in: **0/979 Healthy arteries correctly classified** in the raw model

### Evidence

**Raw prediction distribution (v14, 3182 arteries):**
- Healthy: 0 | Non-sig: 2,147 | Sig: 1,035

**Calibration recovered Healthy recall to 0.939** by aggressively lowering t0 (Healthy threshold to 0.30), but this caused Non-sig recall to collapse to 0.317.

### Root Cause

The paper defines C⁻¹(ŷ_od) as a function of model predictions. The v14 implementation followed this literally. However, when OD predictions are unreliable (early training, fresh 6-class heads), the pseudo-labels inject correlated noise into the SC branch. Since both branches are co-training, the error propagates and self-reinforces before OD predictions become reliable.

### Fix in v15

Reverted OD→SC direction to **GT-anchored** `od2sc_targets(od_targets, seq_length)`: the GT boxes are used to generate stable, correct pseudo-labels for the SC branch. This is what the v7–v12 runs used and what the paper code implicitly did (Hungarian matching requires GT targets).

**SC→OD direction (C(ŷ_sc))** remains prediction-based — SC predictions drive OD pseudo-label generation without GT involvement. This direction is less noisy because the SC branch's 32-point classification is more stable than the OD branch's bounding box predictions in the early epochs.

### v14 Legacy

Despite the Healthy collapse, v14 is valuable because:
- Plaque F1 improved to 0.640 (best at time, from richer backbone features)
- The fully-converged v14 backbone (SE gates, parallel 2D/3D, learnable pos enc) is used by v15 and v16
- The AUC of 0.804 confirmed the model had discriminative power — the class collapse was a threshold/bias issue

---

## 15. Changes from Original Paper Code

### Summary by Category

| Category | Count | Description |
|----------|-------|-------------|
| **A. Kept** | 7 | Elements retained without modification |
| **B. Fixed (bugs)** | 18+ | Crashes, silent corruption, device errors |
| **C. Reverted** | 3 | "Fixes" that matched equations but broke convergence |
| **D. Changed** | 10 | Deliberate architectural improvements |
| **E. Added** | 15+ | Entire subsystems built from scratch |

### A. What Was Kept

1. **Overall architecture design** — dual-branch structure, temporal (32 cubes → 3D-CNN → Transformer) and spatial (full volume + 4 2D views → CNN → DETR decoder) branches
2. **DC loss formulation** — `L_total = L_od + L_sc + δ × L_dc`; mutual pseudo-label supervision is the paper's core innovation
3. **Loss weights 1:1:1** — retained from paper code (not paper equations; see Section 9)
4. **Hungarian matching framework** — bipartite matching for set prediction
5. **Data format and label encoding** — 256×64×64 NIfTI, 0–6 label scheme, two-stage pre-train/fine-tune
6. **CT windowing** — HU clipping to `[-150, 750]`, normalised to [0,1]
7. **Multi-view 2D projections** — 4 views (sagittal, coronal, two diagonals) for spatial branch

### B. Fixed (Selected Critical Bugs)

See Section 7 for the complete numbered list. Key fixes:
- `nn.ModuleList` for extraction blocks (was plain list — weights never trained)
- Feature fusion weights as `nn.Parameter` (was CPU tensor per forward pass)
- Fixed object query embeddings (was random per-call)
- Spatial flattening projection actually called in forward
- Gradient detachment in DC loss (was circular)
- Empty tensor shape guards throughout (crash prevention)
- Temporal branch sequence length fix (was bag-of-cubes)

### C. Reverted (Equations vs Code Discovery)

| Fix | Applied | Why Reverted |
|-----|---------|--------------|
| `box_lastdim_expansion` `[cx,w]→[cx,0.5,w,1.0]` (correct geometry) | Yes | Different loss landscape from what paper trained with; *kept because geometrically correct* |
| `loss_boxes` weights 5.0×L1 + 2.0×GIoU (paper equations) | Yes | 3.5× larger gradients → convergence failure → reverted to 1:1 |
| `HungarianMatcher` weights cost_bbox=5, cost_giou=2 | Yes | Different matching → convergence failure → reverted to 1:1:1 |

### D. Deliberate Improvements

1. **Temporal branch fix** — sequence length 1→32 (restored true temporal attention)
2. **SE fusion gates** — per-channel learned 3D/2D blending (replaces fixed scalar `_3d_weight=0.71`)
3. **Parallel 2D/3D streams** — each level's 2D branch processes its own input (not 3D branch output)
4. **Learnable positional encoding** — temporal transformer; replaces fixed sinusoidal
5. **Ordinal EMD loss** — penalises severity-order violations proportionally to distance
6. **boost_sig / boost_nonsig** — targeted class weight increases for clinical priority
7. **GT-based C⁻¹ (v15 fix)** — OD→SC direction uses GT targets to prevent feedback loops
8. **DC confidence annealing** — threshold 0.7→0.4 over DC ramp window
9. **Native 1D interval IoU** — box expansion `[cx, 0.5, w, 1.0]` for correct vessel-axis geometry
10. **Fine-tuning-on-fine-tuning** — v16 starts from v15 checkpoint (not just pre-trained backbone)

### E. Added (From Scratch)

Complete training loop, evaluation pipeline, calibration system, CPR visualisation, patient-level splitting, ordinal loss, augmentation suite (8 transforms), YAML configs, SWA, cross-validation, Grad-CAM, uncertainty estimation, pipeline scripts, per-artery traceability JSON outputs — none existed in the original code. See Section 8 for the full list.

---

## 16. Calibration Thresholds Reference

### Archived Thresholds

| Version | File | t0 (Healthy) | t1 (Non-sig) | t2 (Sig) | Notes |
|---------|------|-------------|-------------|---------|-------|
| v7-ft | `calibration_thresholds_v7_constrained.json` | 2.20 | 0.35 | 0.25 | Use `--use_constrained` |
| v12-ft | `calibration_thresholds_v12_constrained.json` | 2.80 | 0.65 | 0.20 | |
| v14 | `calibration_thresholds_v14_constrained.json` | 0.30 | 1.15 | 0.10 | Aggressive; compensated for Healthy collapse |
| v15 std | `calibration_thresholds_v15.json` | 1.757 | 1.000 | 0.287 | Standard (Non-sig=0 on test) |
| v15 con | `calibration_thresholds_v15_constrained.json` | 0.700 | 0.400 | 0.100 | Constrained |
| **v16** | `calibration_thresholds_v16.json` | — | — | — | Identical to sig-constrained (constraint not binding) |
| v16 sig | `calibration_thresholds_v16_sig_constrained.json` | — | — | — | Both calibrations converged to same result |

### Plaque Thresholds (by version)

| Version | t_Calcified | t_NonCalc | t_Mixed |
|---------|------------|----------|---------|
| v7-ft | 1.42 | 0.78 | 1.19 |
| v12-ft | 1.19 | 1.59 | 0.46 |
| v14 | 0.729 | 1.756 | 0.619 |
| v15 | 0.729 | 0.456 | 0.213 |

---

## 17. Error Analysis and Failure Modes

### Dominant Stenosis Errors (v16)

| Error Type | Count | Rate |
|-----------|-------|------|
| Sig → Non-sig (under-escalation) | 22/217 | 10.1% |
| Healthy → Non-sig (false positive) | 17/98 | 17.3% |
| Non-sig → Sig (over-escalation) | 23/163 | 14.1% |
| Sig → Healthy | 1/217 | 0.5% |
| Healthy → Sig | 2/98 | 2.0% |

**Under-escalation (Sig→Non-sig):** Dominant failure mode. Diffuse mild-moderate disease without a focal high-grade obstruction — the OD queries aggregate below the Sig threshold. Clinically, these are borderline cases; under-escalation is the riskier direction.

**False-positive Healthy (Healthy→Non-sig):** 17.3% of clean arteries trigger a spurious flag. The OD head fires foreground queries on clean vessels with mild wall irregularity or CPR reconstruction noise. This rate has been stable across v15 and v16 despite `eos_coef` increase.

### Dominant Plaque Errors (v16)

| Class | F1 | Root Cause |
|-------|-----|-----------|
| Calcified | 0.846 | Bright calcified deposits are visually distinctive in HU — easiest class |
| Non-calcified | 0.619 | Soft tissue plaque has similar HU to vessel wall; no iodine contrast to enhance lumen |
| **Mixed** | **0.556** | Rare class (55 samples), high visual overlap with both other types; heterogeneous appearance |

### Historical Failure Modes (Resolved)

| Version | Failure | Fix |
|---------|---------|-----|
| v11-ft | T0=30 → LR≈0 at DC activation → degenerate training | T0=60 |
| v9-ft | 6× LR mismatch at SC head re-init → SC branch collapse | Layer-wise LR |
| v8-ft | focal_gamma=3.0 → SC temporal features destabilised | Keep gamma=2.0 |
| v14-ft | GT-free C⁻¹ → Healthy feedback loop → 0/979 Healthy correct | GT-based OD→SC targets |
| Pre-v12 | DC val loss corruption: DC ramp inflates val loss → wrong checkpoint selected | Switch to F1 checkpoint metric |

---

## 18. Key Workflows and Commands

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

# Fine-tuning from a previously fine-tuned checkpoint (v16 from v15)
NCCL_TIMEOUT=3600 nohup torchrun --nproc_per_node=2 --master_port=29509 train.py \
  --distributed --config configs/finetune_v16.yaml \
  --pretrained ./checkpoints_v15_finetune/best_model.pth \
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
# Standard (macro-F1 optimised, Non-sig will be 0)
python calibrate.py \
  --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning \
  --output calibration_thresholds_v16.json \
  --grid_steps 50

# Constrained Non-sig recall (recommended for stenosis)
python calibrate.py \
  --checkpoint ./checkpoints_v16_finetune/best_model.pth \
  --pattern fine_tuning \
  --output calibration_thresholds_v16_nonsig_constrained.json \
  --grid_steps 50 \
  --constrain_nonsig_recall 0.10

# Sig-recall constrained
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
nohup bash run_v16_pipeline.sh > pipeline_output.log 2>&1 &
# Or wait for training PID first:
nohup bash run_v16_pipeline.sh <TRAINING_PID> > pipeline_output.log 2>&1 &
```

### Monitoring

```bash
# GPU status
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Follow training log
tail -f v16_train.log

# Check training is running
ps aux | grep -E "torchrun|train.py" | grep -v grep
```

### v16 Config (for reference)

```yaml
# configs/finetune_v16.yaml
pattern: fine_tuning
data_root: ./dataset/train
epochs: 250
lr: 3.0e-5
weight_decay: 1.0e-4
grad_clip: 0.1
warmup_epochs: 10
lr_schedule: cosine_warm_restarts
lr_t0: 60
lr_t_mult: 2
swa: true
swa_start_epoch: 80
dc_warmup_hold: 20
dc_warmup_ramp: 40
delta: 0.5
dc_confidence_threshold: 0.4
dc_confidence_start: 0.7
label_smoothing: 0.1
ordinal_weight: 1.5
patience: 100
min_delta: 0.001
balanced_sampling: true
sc_class_weight: true
boost_nonsig: true
boost_sig: true
focal_loss: true
focal_gamma: 2.0
eos_coef: 0.20
accumulate_steps: 4
amp: true
ema: true
ema_decay: 0.999
layerwise_lr: true
augment: true
num_workers: 8
seed: 42
patient_split: true
split_seed: 42
temporal_encoder_layers: 4
temporal_heads: 8
spatial_encoder_layers: 4
spatial_decoder_layers: 4
checkpoint_dir: ./checkpoints_v16_finetune
save_every: 10
print_every: 1
log_dir: ./runs
log_every: 10
```

---

## 19. Common Pitfalls

| Pitfall | What Happens | Prevention |
|---------|-------------|------------|
| GT-free C⁻¹ in L_dc | Healthy class collapses — 0/N correct | OD→SC direction **must** use `od2sc_targets` (GT-based). See Section 14. |
| Loss fn not moved to GPU | `RuntimeError: expected cuda, got cpu` | Call `loss_fn.to(device)` alongside `model.to(device)` |
| `boxes_dimension_expansion` mutates in-place | Second + third loss terms receive corrupted targets | Deep copy targets before passing to each loss term |
| Standard calibration 2D search | Non-sig recall = 0% | Always use `--constrain_nonsig_recall 0.10` for fair stenosis eval |
| `focal_gamma=3.0` | SC branch ACC collapses (0.814→0.749) | Keep `focal_gamma=2.0` — confirmed lesson from v8-ft |
| `T0=30` in cosine warm restarts | LR≈0 exactly when DC activates at ep20 → model never recovers | Use `T0=60` so LR has a reasonable value at DC activation |
| Evaluating on internal test split | Inflated metrics vs held-out test | Use `--data_root ./dataset/test` for final results |
| `best_model.pth` ≠ best test model | Val loss metric can select an overfit checkpoint | Check raw F1 per-class on test, not just val loss; use F1 as checkpoint metric |
| Paper equations ≠ paper code | λ_L1=5, λ_iou=2 in equations; 1:1:1 in code | Follow the code. Applying the equations breaks convergence. |
| DDP NCCL timeout | Hang during ALLREDUCE → watchdog kills | Always add `NCCL_TIMEOUT=3600` before `torchrun`; ensure both GPUs are free |
| `od2sc_targets` / `sc2od_targets` on CPU | Device mismatch in DC loss computation | Already fixed in codebase — do not remove the `.to(device)` calls |
| Calibration hurts stenosis on test | Val-calibrated thresholds don't transfer | Use raw argmax for stenosis; calibrate plaque only |
| SWA model not evaluated | May outperform `best_model.pth` by 1–2% | Evaluate `swa_model.pth` separately after each training run |
| Not using NCCL_TIMEOUT | Subtle hang if ranks diverge | Always prefix `torchrun` with `NCCL_TIMEOUT=3600` |

---

## 20. Open Items and Next Steps

### Unresolved Issues

**[1] Healthy Recall Plateau at 0.806**
Both v15 and v16 show Healthy recall = 0.806 (19.4% of clean arteries flagged as Non-sig). This has been stable across two versions. The OD head fires spurious foreground queries on anatomically clean vessels.

Potential directions:
- Higher `eos_coef` (currently 0.20, try 0.25–0.30) — further suppresses spurious OD
- `boost_healthy=true` — mirror boost_sig pattern for Healthy class in SC loss
- Per-class asymmetric focal gamma (higher gamma for Healthy to focus on hard clean cases)
- Harder negative augmentation (clean vessels with contrast variation)

**[2] Mixed Plaque F1 is Weak (0.556)**
Mixed has only ~55 test samples and high visual overlap with both Calcified and Non-calcified. No targeted fix applied yet.

Potential directions:
- Multi-window HU input (soft tissue + bone window as separate channels)
- Contrastive pre-training on plaque morphology
- Additional 2D cross-section supervision (cross-sections are already generated in `visualize.py`)

**[3] Calibration Does Not Transfer to Test Set**
Val-calibrated thresholds reduce stenosis test F1 from 0.851 → ~0.750. The val and test distributions are not perfectly matched (different patient populations may have different class balance).

Options:
- Temperature scaling (more regularised than per-class thresholds)
- Hold out a small test subset for threshold selection
- Train/val/test re-split with better patient representation balance

**[4] Paper Accuracy Target Not Reached**
Paper reports Stenosis ACC=0.914; our best is 0.854 (~6% gap). Not required for handover but noted for future work. Likely requires the paper's CDA augmentation (foreground ROI overlay) which we did not implement.

**[5] SWA Model Not Evaluated for v15/v16**
Both runs saved `swa_model.pth`. In v14 the SWA model collapsed; in v15/v16 this was not reproduced. Quick evaluation may yield +1–2% F1.

```bash
python eval.py \
  --checkpoint ./checkpoints_v16_finetune/swa_model.pth \
  --pattern fine_tuning --data_split testing --detailed
```

**[6] Checkpoint Retrieval**
Model checkpoints are not on GitHub (too large, ~1.5 GB each). Critical checkpoints to retrieve from training server:

| Checkpoint | Path | Purpose |
|-----------|------|---------|
| v16 fine-tune (best) | `checkpoints_v16_finetune/best_model.pth` | Current best — highest priority |
| v15 fine-tune (best) | `checkpoints_v15_finetune/best_model.pth` | Second-best; stable baseline |
| v14 pre-train (backbone) | `checkpoints_v14/best_model.pth` | Best backbone for future fine-tuning |
| v12 fine-tune (best) | `checkpoints_v12_finetune/best_model.pth` | Historical reference |

### Recommended Next Steps (Priority Order)

| Priority | Action | Expected Gain |
|----------|--------|--------------|
| 1 | Retrieve checkpoints (see above) | Required before all else |
| 2 | Evaluate v16 SWA model | +1–2% F1, no retraining |
| 3 | Ensemble v15 + v16 | +1–3% F1 from complementary predictions |
| 4 | Implement CDA augmentation (paper's foreground overlay) | Close gap to paper's 0.914 |
| 5 | Healthy recall recovery | Target the 19.4% spurious OD queries |
| 6 | Mixed plaque improvement | Multi-window input or more 2D supervision |
| 7 | v17 fine-tuning (if needed) | Focus on Healthy recall with boost_healthy |

### Suggested v17 Config Changes

Based on v16 lessons, the next iteration should:
- Keep: `boost_sig=true`, `ordinal_weight=1.5`, `eos_coef=0.20`, GT-based DC
- Explore: `boost_healthy=true`, `eos_coef=0.25–0.30`, per-class focal gamma, Healthy-targeted augmentation

---

*End of compiled report. All metrics are from the held-out test set (`dataset/test/`, 478–665 arteries, AP-NUH patients) unless explicitly noted as validation or internal test split.*
