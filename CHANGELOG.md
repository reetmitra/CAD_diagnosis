# SC-Net Code Improvements & Roadmap

## Recent Changes (2026-02-25)

### Baseline Evaluation

- Evaluated `checkpoints/best_model.pth` (v1, epoch 20, pre_training) on all 665 test files in `dataset/test/`
- Results: Stenosis ACC 0.702, F1 0.413 | Plaque ACC 0.430, F1 0.100 | SC Points ACC 0.801 (17,035/21,280)
- All v1 checkpoints are pre_training mode (num_classes=3); test data labels 0-6 require fine_tuning (num_classes=6)

### Bug Fix: Empty Box Dimension Mismatch (`1c43ae7`)

- **File:** `functions.py`, `box_lastdim_expansion()`
- `box_lastdim_expansion()` returned empty tensors with shape `(0, 2)` instead of `(0, 4)` for samples with no lesions
- Caused `RuntimeError` in `HungarianMatcher` when `torch.cat` mixed `(N, 4)` and `(0, 2)` tensors across batch elements
- Fix: return `torch.zeros` with `shape[-1]=4` for empty tensors

### Bug Fix: Label Remapping for pre_training (`1c43ae7`)

- **Files:** `augmentation.py`, `framework.py`, `train.py`
- Training data has labels 0-6 but pre_training expects labels 0-3
- Added `num_classes` parameter to `cubic_sequence_data`
- When `num_classes=3`, labels remapped via `((label - 1) % 3) + 1` (6 fine-tuning classes -> 3 plaque composition classes)
- Parameter threaded through `framework.py` and `train.py`

### Retraining v2 (In Progress)

- Full 200-epoch training with all bug fixes and enhancements applied
- Config: AMP, DDP (2x RTX 3090), EMA (0.999), augmentation (rotation/jitter/flip), warmup (10 ep), layer-wise LR, grad clip 0.1
- Dataset: `dataset/train/` (2,961 samples), pre_training mode (num_classes=3), 70/15/15 split
- Checkpoints saving to `checkpoints_v2/`
- Smoke test (1 epoch): val Stenosis ACC 0.734, Plaque ACC 0.497 (already exceeds v1 final model)
- Estimated ~3 min/epoch, ~10 hours total

### Fine-Tuning Pipeline Preparation

- Fixed `pre_training_load()` in `framework.py` to handle checkpoint format (`model_state_dict` key extraction)
- Fixed `eval.py` stenosis class boundary for 6-class mode
- Fixed `eval.py` plaque composition mapping for 6-class evaluation
- Created launch scripts: `scripts/pretrain.sh`, `scripts/finetune.sh`, `scripts/eval_finetune.sh`
- Updated `config.py` data paths from placeholders to `dataset/train`

### TensorBoard Integration

- Added `SummaryWriter` to `Trainer` class in `train.py`
- Logging: per-epoch losses (total, training, validation), component losses (L_od, L_sc, L_dc), validation metrics (Stenosis ACC/F1, Plaque ACC/F1), LR schedules, gradient norms
- New CLI args: `--log_dir` (default: `runs/`), `--log_every` (logging frequency)

### Test-Time Augmentation (`a313e27`)

- Added `--tta` flag and `--tta_k` (default 5) to `eval.py`
- Augmentations: depth flip, intensity scale (±5%), intensity shift (±0.02 normalized)
- Depth-flipped SC logits flipped back before averaging; box predictions from original pass only

### SC Loss Class Weighting & Delta Tuning (`a313e27`)

- Added `compute_sc_class_weights()` to `optimization.py` (background=0.5, lesion=1.5)
- Class weights registered as buffer for automatic device transfer
- `--sc_class_weight` flag (default on), `--no_sc_class_weight` to disable
- Exposed `--delta` CLI arg (default 1.0) for contrastive loss weighting
- Delta stored as `self.delta` in `spatio_temporal_contrast_loss` constructor

### YAML Config System (`a313e27`)

- Added `--config` CLI arg to `train.py` for loading YAML config files
- YAML values serve as defaults; CLI args override
- Created `configs/pretrain_default.yaml`, `configs/finetune_default.yaml`, `configs/sweep_example.yaml`

### Cross-Validation (`a313e27`)

- Created `cross_validate.py` with patient-level k-fold splitting
- `PatientKFoldSplitter` extracts patient IDs, groups arteries, implements manual k-fold (no sklearn)
- Added `file_indices` parameter to `cubic_sequence_data` for flexible fold-based splits
- CLI args: `--n_folds` (default 5), `--cv_seed` (default 42)

### Transformer Hyperparameter Tuning (`a313e27`)

- Exposed `--temporal_encoder_layers`, `--temporal_heads`, `--spatial_encoder_layers`, `--spatial_decoder_layers` as CLI args
- Overrides applied at runtime in `framework.py`, `DefaultConfig` unchanged

---

## Implemented Changes (Prior)

### Critical Bug Fixes

| Fix | File | Impact |
|-----|------|--------|
| `nn.ModuleList` for extraction blocks | `architecture.py` | 3D/2D extraction block weights were invisible to the optimizer and couldn't be moved to GPU. **Without this fix, the spatial branch was completely untrained.** |
| Feature weight device handling | `architecture.py` | `_2d_maps_to_3d_maps` created a new CPU tensor every forward pass, causing an immediate crash on GPU. Now stored as `nn.Parameter` on the correct device. |
| Fixed query embeddings | `architecture.py` | `torch.randint` generated random query indices every forward pass, making the decoder non-deterministic and preventing learned object queries from converging. Replaced with fixed learned `nn.Embedding` weights (DETR standard). |
| Spatial flattening projection | `architecture.py` | The `Conv3d` flattening layer defined in `__init__` was never called in `forward()`. The rearrange pattern was also incorrect. Now properly applies Conv3d(128→16) before flattening and linear projection, producing the correct 16 spatial tokens of 512 dimensions. |
| Training infrastructure | `train.py` | No training loop, optimizer, or scheduler existed. Added complete training script with AdamW (lr=1e-4, weight_decay=1e-4), CosineAnnealingLR, gradient clipping (max_norm=0.1), validation, and checkpointing. |

### Architectural Corrections

| Fix | File | Impact |
|-----|------|--------|
| Gradient detachment in contrastive loss | `optimization.py` | The dual-task contrastive loss used raw model outputs (with gradients) as pseudo ground truth, creating circular gradient flow between the two branches. Detaching prevents this and ensures each branch receives clean supervision from the other — this is the core novelty of the paper. |
| Learnable view fusion weights | `architecture.py` | `_3d_weight` (0.75) and `_2d_weight` ([0.25, 0.25, 0.25, 0.25]) were fixed scalars. Now `nn.Parameter` so the model can learn optimal fusion ratios between the 3D-CNN and multi-view 2D-CNN features (Eq. 2 in paper). |
| Box format: center-width | `augmentation.py`, `optimization.py` | Boxes were stored as `[start, end]` but the matcher called `box_cxcywh_to_xyxy` which assumes `[center, width]`. This mismatch caused incorrect Hungarian matching and IoU computation. All box generation now uses center-width format consistently. |
| Auto box dimension expansion | `optimization.py` | 1D boxes `[center, width]` are automatically expanded to 4D `[cx, cy, w, h]` inside `object_detection_loss.forward()`, ensuring the DETR-style matcher and GIoU loss work correctly regardless of input format. |

### Robustness Fixes

| Fix | File | Impact |
|-----|------|--------|
| Device-aware target tensors | `optimization.py` | `od2sc_targets` and `sc2od_targets` generate tensors on CPU. Loss functions now explicitly move targets to the model's device before computation. |
| Deep copy targets before each loss term | `optimization.py` | `boxes_dimension_expansion` mutates tensors in-place. The composite loss calls three sub-losses on the same targets — without cloning, the second and third losses received already-mutated data. |
| Dataset index offset | `augmentation.py` | `__getitem__` used raw index without adding `data_start`, so the validation set loaded training samples. |
| `_3d_cubes_selection` device | `functions.py` | Output tensor was created on CPU regardless of input device. Now inherits device and dtype from input. |
| `torch.torch.float32` typo | `augmentation.py` | Would cause `AttributeError` at runtime. |
| `torch.load` with `map_location` | `framework.py` | Pre-trained weights now loaded to CPU first, preventing device conflicts when loading checkpoints saved on different GPUs. |

### Configuration Updates

| Change | File | Reason |
|--------|------|--------|
| `spatial_proj_channels`: [128,1024,128,512] → [128,256,16,512] | `config.py` | Matches actual feature dimensions after 4 pooling levels (16×4×4 = 256 spatial elements, 16 output tokens). |
| Two-stage data roots | `config.py`, `framework.py` | Added `pretrain_data_root` and `finetune_data_root` with CLI override via `--data_root`. |
| Default HU window: [-200,800] → [-150,750] | `functions.py` | Matches the actual values computed from config `window_lw=[300,900]`. |

### Training Infrastructure Enhancements (v2)

| Feature | File(s) | Impact |
|---------|---------|--------|
| Mixed-Precision Training (AMP) | `train.py` | `torch.amp.GradScaler` and `autocast` for ~1.5-2x speedup on RTX 3090. Enabled by default with `--amp` flag. |
| Multi-GPU Training (DDP) | `train.py` | `DistributedDataParallel` support for both RTX 3090s. Launch via `torchrun --nproc_per_node=2`. Uses existing distributed utilities in `functions.py`. |
| Online Data Augmentation | `augmentation.py` | Random rotation (±15°), intensity jitter (±50 HU), and random depth flip integrated into training pipeline. Each applied with 50% probability. Enabled via `--augment` flag. |
| Evaluation Metrics Integration | `train.py`, `eval.py` | Per-epoch validation now computes accuracy, precision, recall, F1, and specificity for both stenosis degree and plaque composition classification. |
| Learning Rate Warmup | `scheduler_utils.py`, `train.py` | Linear warmup over first 10 epochs before cosine decay. Stabilizes early training for transformer components. |
| Layer-wise Learning Rate Decay | `scheduler_utils.py`, `train.py` | CNN backbone at 0.1x, transformer at 0.5x, detection heads at 1.0x base LR. Standard DETR practice for fine-tuning. |
| Exponential Moving Average (EMA) | `scheduler_utils.py`, `train.py` | Maintains EMA copy of weights (decay=0.999) for evaluation. Typically improves accuracy by 0.5-1% for DETR-style models. |
| `eval.py` data root fix | `eval.py` | When an explicit `--data_root` is provided, evaluation now uses all files in that directory instead of being limited by the default split. |

---

## Performance Impact

### Before fixes
- Training was **impossible** — no training loop existed, extraction blocks were unregistered, GPU crashes on forward pass.

### After fixes
- Full training pipeline runs end-to-end on GPU
- Loss decreases consistently across epochs (10.1 → 3.9 on dummy data over 200 epochs)
- Both branches (temporal + spatial) now have trainable parameters and receive gradients
- Contrastive loss provides correct cross-task supervision without gradient leakage
- Hungarian matching produces correct assignments with consistent box format

### v2 improvements
- **AMP** reduces memory usage and provides ~1.5-2x training speedup on RTX 3090, enabled by default via `--amp`
- **DDP** enables multi-GPU training across both RTX 3090s, doubling effective batch size or halving wall-clock time (`torchrun --nproc_per_node=2`)
- **Online augmentation** (rotation, intensity jitter, depth flip) improves generalization on small medical imaging datasets, enabled via `--augment`
- **LR warmup** (10 epochs linear) stabilizes early transformer training, avoiding loss spikes
- **Layer-wise LR decay** (backbone 0.1x, transformer 0.5x, heads 1.0x) preserves pre-trained spatial features during fine-tuning
- **EMA** (decay=0.999) maintains a smoothed weight copy for evaluation, typically adding 0.5-1% accuracy
- **Evaluation metrics** (accuracy, precision, recall, F1, specificity) are now computed per epoch for both classification tasks

---

## Future Improvements

### Medium Priority

**1. Parallel 2D/3D Feature Streams**
- Currently the 2D branch takes 3D features as input after level 0 (interleaved)
- Paper describes independent parallel paths that fuse after extraction
- Implementing true parallel streams may improve feature diversity

**2. Model Compression**
- Knowledge distillation from the full model to a smaller variant
- Pruning unused attention heads in the transformer
- Important for potential clinical deployment

**3. Vessel-Aware Preprocessing**
- Centerline extraction and straightening before feeding to the model
- Adaptive cube sampling based on vessel curvature rather than fixed step size
- Could significantly improve the temporal branch's ability to capture lesion context

### Previously Listed — Now Implemented

- ~~Transformer Configuration Tuning~~ — CLI args for layer/head counts (commit `a313e27`)
- ~~Test-Time Augmentation~~ — `--tta` flag in eval.py (commit `a313e27`)
- ~~Cross-Validation~~ — `cross_validate.py` with patient-level k-fold (commit `a313e27`)
