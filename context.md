# SC-Net Project Context
> Living document — updated every session. New agents: read this first, then check report.md for full detail.

---

## What This Project Is

Implementation of **SC-Net** (Spatio-Temporal Contrast Network) for automated CAD diagnosis from Coronary CT Angiography (CCTA). Based on MICCAI 2024 paper:
> Ma et al., "Spatio-Temporal Contrast Network for Data-Efficient Learning of Coronary Artery Disease in Coronary CT Angiography," MICCAI 2024, pp. 645–655.

Paper source code: https://github.com/PerceptionComputingLab/SC-Net (reference only)
Our fork: https://github.com/reetmitra/CAD_diagnosis

---

## Environment

```
OS:       Ubuntu, Linux 6.8.0-90-generic
GPUs:     2× NVIDIA RTX 3090 (24 GB each)
Python:   3.10, venv at .venv/
PyTorch:  2.5.1+cu121
Activate: source .venv/bin/activate
```

**Always activate venv before running anything:**
```bash
source /home/reet/development/CAD_diagnosis/.venv/bin/activate
```

---

## Key Files

| File | Purpose |
|------|---------|
| `architecture.py` | Model: `spatio_temporal_semantic_learning` |
| `optimization.py` | Loss: `spatio_temporal_contrast_loss`, `FocalLoss`, `sampling_point_classification_loss`, `object_detection_loss`, `dual_task_contrastive_loss` |
| `functions.py` | `HungarianMatcher`, `box_lastdim_expansion`, `generalized_box_iou`, `box_cxcywh_to_xyxy` |
| `augmentation.py` | `cubic_sequence_data` — loads NIfTI volumes + label txt files |
| `framework.py` | `sc_net_framework` — wires model + loss + data together |
| `train.py` | Full `Trainer` class with all CLI args |
| `eval.py` | Evaluation with TTA, ensemble, detailed metrics, plots |
| `config.py` | `DefaultConfig` with all hyperparameters |
| `cross_validate.py` | Patient-level k-fold cross-validation |
| `scripts/pretrain.sh` | Launch pre-training |
| `scripts/finetune.sh` | Launch fine-tuning |
| `configs/*.yaml` | YAML config files |

---

## Dataset Structure

```
dataset/
  train/
    volumes/   *.nii   — 3D CPR NIfTI volumes (256×64×64)
    labels/    *.txt   — 256-line files, one label per slice
  test/
    volumes/   *.nii
    labels/    *.txt
```

**Label values (0–6):**
- `0` = background
- `1–3` = non-significant stenosis (calcified / non-calcified / mixed plaque)
- `4–6` = significant stenosis (calcified / non-calcified / mixed plaque)

**Pre-training (3-class):** labels remapped via `((label-1) % 3) + 1` → plaque composition only (calcified/non-cal/mixed)
**Fine-tuning (6-class):** all labels 1–6 used as-is

Train set: 2,961 samples | Test set: 665 samples

---

## Architecture Summary

**Dual-branch:**
1. **Temporal branch** — 32 cubic crops along vessel → 3D-CNN → Transformer encoder → per-point classification head (stenosis/plaque)
2. **Spatial branch** — full CPR volume + 4 2D-views → multi-view 3D+2D CNN → Transformer decoder (DETR-style, Q=16 queries) → box regression + classification heads

**Loss (paper Eq. 5–7):**
- `L_od` = CE + λ_L1×L1 + λ_iou×GIoU, where **λ_L1=5, λ_iou=2**
- `L_sc` = cross-entropy over 32 sampling points
- `L_dc` = L_od(C(ŷ_sc), ŷ_od) + L_sc(C⁻¹(ŷ_od), ŷ_sc) — mutual pseudo-label supervision
- `L_total` = L_od + L_sc + δ × L_dc

**1D box representation:** boxes are `[center, width]` along vessel axis. When expanded to 4D for IoU: `[cx, 0.5, w, 1.0]` → xyxy = `[cx-w/2, 0, cx+w/2, 1]`.

---

## All Bugs Fixed (in order)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `functions.py` | `box_lastdim_expansion` returned `(0,2)` shape for empty tensors | Return `torch.zeros` with `shape[-1]=4` |
| 2 | `augmentation.py` | Labels 0–6 exceeded pre_training num_classes=3 (CUDA index OOB) | Added `num_classes` param + modulo remapping |
| 3 | `functions.py` | Hard assert on degenerate boxes crashed at epoch 129 | Replace assert with `torch.cat` clamping |
| 4 | `functions.py` | In-place box op broke AMP autograd | Switched to `torch.cat` (non-in-place) |
| 5 | `optimization.py` | `FocalLoss.alpha` stayed on CPU (device mismatch) | `register_buffer('alpha', alpha)` |
| 6 | `functions.py` | **`box_lastdim_expansion` expanded `[cx,w]→[cx,cx,w,w]`** (square boxes, wrong GIoU) | **Correct: `[cx, 0.5, w, 1.0]`** — full-height 1D intervals |
| 7 | `optimization.py` | Loss weights 1:1 instead of paper λ_L1=5, λ_iou=2 | `5.0*L1 + 2.0*GIoU` |
| 8 | `functions.py` | `HungarianMatcher` weights 1:1:1 | `cost_class=1, cost_bbox=5, cost_giou=2` |

**Bugs 6–8 are the most critical** — they corrupted L_od and L_dc in every prior training run.

---

## Training History

| Run | Epochs | LR | Key config | Status | Notes |
|-----|--------|----|-----------|--------|-------|
| v1 | ~20 | 1e-4 | baseline | Done | Many arch bugs unfixed |
| v2 | 143 | 1e-4 | AMP, DDP, EMA, warmup, layer-wise LR | Done | Bugs 1–5 fixed; bugs 6–8 still present |
| v3 | ~40 | 1e-4 | + focal loss, SC weights, grad accum | Killed | Killed: LR too high for new loss weights |
| v4 | ~15 | 1e-4 | All bugs 1–8 fixed | Killed | Killed: same LR issue, val loss increasing after warmup |
| v5 | 40 | **3e-5** | All bugs 1–8 fixed | **RESUMING** | NCCL DDP timeout fixed (val_loss all_reduce). Last checkpoint: epoch 39. patience_counter resets to 0 on resume. |

### Current best checkpoints
- `checkpoints_v2/checkpoint_epoch_139.pth` — best pre-training before bug fixes
- `checkpoints/checkpoint_epoch_39.pth` — best pre-training with all bugs fixed ← **USE THIS**
  (Note: v5 run saved to `./checkpoints/`, not a separate `checkpoints_v5/` directory)

---

## Evaluation Results

### v1 Baseline (epoch 20, pre_training mode, 665 test files)
| Task | ACC | F1 |
|------|-----|----|
| Stenosis | 0.702 | 0.413 |
| Plaque | 0.430 | 0.100 |
| SC Points | 0.801 | — |

### v2 Epoch 139 (pre_training mode)
| Task | ACC | F1 | Change vs v1 |
|------|-----|----|-------------|
| Stenosis | 0.702 | 0.413 | — |
| Plaque | **0.486** | **0.218** | +5.6% / +118% |
| SC Points | **0.848** | — | +4.7% |

### Paper Target (fine-tuned model)
| Task | ACC |
|------|-----|
| Stenosis | **0.914** |

**We have never run fine-tuning.** The paper's 0.914 comes from the fine-tuned (6-class) model. This is the next critical step.

---

## Errors Encountered & Fixes

### NCCL DDP Timeout (v5, epoch 40) — **FIXED**
```
[rank0]: Exception (either an error or timeout) detected by watchdog at work: 233625
terminate called after throwing an instance of 'c10::DistBackendError'
```
**Root cause:** `validate()` returned a *local* `val_loss` computed only on each rank's shard of the eval set (via `DistributedSampler`). The early-stopping `patience_counter` was updated independently on each rank using different data, so the counters drifted. By epoch 40, rank 1's counter hit 30 (the patience limit) while rank 0's was at 29. Rank 1 broke from the training loop and called `dist.destroy_process_group()`. Rank 0 entered epoch 41 and hung at ALLREDUCE for 10 minutes before the NCCL watchdog killed it.
**Fix applied (`train.py`, `validate()`):** After computing local `val_loss`, do `dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)` so all ranks see the same globally-averaged validation loss before updating `patience_counter`. Both ranks now make identical stop/continue decisions.

### FocalLoss CPU/GPU Device Mismatch (v3 start)
```
RuntimeError: Expected all tensors to be on the same device, but found cuda and cpu
```
**Fix:** `register_buffer('alpha', alpha)` in `FocalLoss.__init__`

### Degenerate Box AssertionError (v2, epoch 129)
```
assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
```
**Fix:** Replace assert with `torch.cat` clamping in `generalized_box_iou`

### In-place AMP Autograd Error (v2, after epoch 129 fix)
```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```
**Fix:** Changed in-place box op to `torch.cat` (fully non-in-place)

### Val Loss Blowing Up After Warmup (v3, v4)
**Symptom:** Val loss decreases during warmup, then sharply increases after LR hits full value
**Cause:** LR 1e-4 too high after loss weights changed to 5:2 (effectively 3.5× larger gradients)
**Fix:** Reduced LR to `3e-5` for v5

---

## Standard Run Commands

### Check training progress
```bash
tail -f ~/development/CAD_diagnosis/train_v5.log
# or snapshot:
tail -20 ~/development/CAD_diagnosis/train_v5.log
```

### Resume v5 from epoch 39
```bash
source .venv/bin/activate
NCCL_TIMEOUT=3600 nohup torchrun --nproc_per_node=2 train.py \
  --distributed \
  --pattern pre_training \
  --data_root ./dataset/train \
  --checkpoint_dir ./checkpoints \
  --resume ./checkpoints/checkpoint_epoch_39.pth \
  --epochs 200 \
  --lr 3e-5 --weight_decay 1e-4 --grad_clip 0.1 \
  --warmup_epochs 10 --layerwise_lr \
  --amp --ema --ema_decay 0.999 \
  --augment --sc_class_weight \
  --focal_loss --focal_gamma 2.0 \
  --accumulate_steps 2 \
  --patience 30 --min_delta 0.001 \
  --delta 1.0 \
  --log_dir ./runs_v5 \
  --save_every 10 --print_every 1 \
  > train_v5_resumed.log 2>&1 &
```

### Run fine-tuning (after v5 has good checkpoint)
```bash
source .venv/bin/activate
NCCL_TIMEOUT=3600 nohup torchrun --nproc_per_node=2 train.py \
  --distributed \
  --pattern fine_tuning \
  --pretrained ./checkpoints/best_model.pth \
  --data_root ./dataset/train \
  --checkpoint_dir ./checkpoints_v5_finetune \
  --epochs 100 \
  --lr 1e-5 --weight_decay 1e-4 --grad_clip 0.1 \
  --warmup_epochs 5 --layerwise_lr \
  --amp --ema --ema_decay 0.999 \
  --augment --sc_class_weight \
  --focal_loss --focal_gamma 2.0 \
  --accumulate_steps 2 \
  --patience 20 --min_delta 0.001 \
  --delta 1.0 \
  --log_dir ./runs_finetune \
  --save_every 10 \
  > train_finetune.log 2>&1 &
```

### Evaluate a checkpoint
```bash
source .venv/bin/activate
python eval.py \
  --checkpoint ./checkpoints_v5/best_model.pth \
  --pattern pre_training \
  --data_root ./dataset/test \
  --batch_size 2 --eval_sc --detailed \
  --plot --plot_dir ./plots_v5 \
  --save_results results_v5.json
```

### Kill any running training
```bash
ps aux | grep -E "torchrun|train.py" | grep -v grep | awk '{print $2}' | xargs kill
```

### GPU status
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

### Push to GitHub (if credentials needed)
```bash
git push  # may need manual auth in terminal
```

---

## Pending Next Steps

1. **Commit DDP fix** — `train.py` val_loss all_reduce (uncommitted, fixes NCCL crash)
2. **Resume v5** from `checkpoints/checkpoint_epoch_39.pth` — command above; note patience_counter resets to 0
3. **Evaluate current best** (`checkpoints/best_model.pth`) on test set to baseline v5
4. **Run fine-tuning** from best v5 checkpoint — **highest-impact step** (paper: 0.914 ACC vs our 0.702)
5. **Evaluate fine-tuned model** in `fine_tuning` mode (6 classes)
6. **Push all commits** to GitHub (`a7ee5e4` and DDP fix were noted as unpushed)

---

## Common Pitfalls for New Agents

- Loss fn is `nn.Module` with registered buffers — must `.to(device)` like model
- `boxes_dimension_expansion` mutates targets in-place — always deep copy targets before passing to each loss term
- `od2sc_targets` / `sc2od_targets` create CPU tensors — need device transfer
- `spatial_proj_channels` must match actual feature dims `[128, 256, 16, 512]`
- DDP training: both GPUs must be free — check with `nvidia-smi` before launching
- Fine-tuning LR should be ~3–5× lower than pre-training LR (use `1e-5` not `3e-5`)
- `best_model.pth` is saved by lowest val loss — may not be best test performance (see v2: best_model was epoch 17 but epoch 139 was better on test)
- Always check `tail -f <log>` after launching — don't assume it started cleanly
