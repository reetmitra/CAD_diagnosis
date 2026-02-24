# SC-Net Training Improvements Design

**Date:** 2026-02-24
**Scope:** Implement CHANGELOG.md future improvements #1-7

## Architecture

Refactor `train.py` into a `Trainer` class. Extract LR scheduling and EMA into `scheduler_utils.py`. Add online data augmentation to `augmentation.py`.

## Features

### 1. Mixed-Precision Training (AMP)
- `torch.amp.GradScaler` + `autocast('cuda')` in training loop
- Scaler handles NaN gradients, auto loss scaling
- ~1.5-2x speedup on RTX 3090

### 2. Multi-GPU Training (DDP)
- `DistributedDataParallel` wrapping in Trainer
- `DistributedSampler` for data loading
- Launch via `torchrun --nproc_per_node=2 train.py`
- Uses existing `functions.py` distributed utilities

### 3. Data Augmentation Pipeline
- `online_augment()` function in `augmentation.py`
- Random rotation along vessel axis (±15 degrees)
- Intensity jitter (±50 HU)
- Elastic deformation (small sigma)
- Controlled by `--augment` flag in training

### 4. Evaluation Metrics
- Integrate per-class accuracy, precision, recall, F1, specificity into validation step
- Log metrics each epoch alongside loss
- Reuse metric computation from `eval.py`

### 5. Learning Rate Warmup
- Linear warmup for first 10 epochs
- Then cosine annealing decay to 0
- `LinearWarmupCosineDecay` scheduler class in `scheduler_utils.py`

### 6. Layer-wise Learning Rate Decay
- Param groups: CNN backbone (0.1x base LR), transformer (0.5x), detection heads (1.0x)
- Standard practice for DETR-style fine-tuning

### 7. Exponential Moving Average (EMA)
- `ModelEMA` class in `scheduler_utils.py`
- Decay rate 0.999, updated after each optimizer step
- EMA weights used for evaluation, original for training

## File Changes
- `train.py` — Trainer class (~300 lines)
- `scheduler_utils.py` — new file (~80 lines)
- `augmentation.py` — add online_augment()
- `eval.py` — fix data loading for dedicated test dirs
- `CHANGELOG.md` — updated
- `report.md` — new, full project history
