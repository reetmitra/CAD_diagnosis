# SC-Net Retraining Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retrain SC-Net with all training improvements (AMP, DDP, EMA, augmentation, warmup, layer-wise LR) and evaluate on held-out test set.

**Architecture:** Single 200-epoch training run on `dataset/train/` (2,961 samples, 70/15/15 split → 2,073 train / 444 val / 444 test) with pre_training mode (num_classes=3). DDP across 2x RTX 3090. Then evaluate new checkpoint on all 665 files in `dataset/test/`.

**Tech Stack:** PyTorch 2.5.1+cu121, torchrun for DDP, existing train.py Trainer class

---

### Task 1: Smoke Test — Verify Training Runs Without Errors

**Files:**
- Read: `train.py`, `framework.py`, `augmentation.py`

**Step 1: Back up existing checkpoints**

```bash
cp -r ./checkpoints ./checkpoints_backup_v1
```

**Step 2: Run 1-epoch smoke test with all flags**

```bash
source .venv/bin/activate
torchrun --nproc_per_node=2 train.py \
  --data_root ./dataset/train \
  --pattern pre_training \
  --epochs 1 \
  --lr 1e-4 \
  --augment \
  --amp \
  --ema \
  --layerwise_lr \
  --warmup_epochs 1 \
  --grad_clip 0.1 \
  --distributed \
  --checkpoint_dir ./checkpoints_smoke \
  --save_every 1
```

Expected: Completes without errors, prints training summary, loss values, and validation metrics for 1 epoch.

**Step 3: Verify checkpoint was saved**

```bash
ls ./checkpoints_smoke/
```

Expected: `checkpoint_epoch_0.pth`, `best_model.pth`, `final_model.pth`

**Step 4: Clean up smoke test**

```bash
rm -rf ./checkpoints_smoke
```

---

### Task 2: Full Training Run (200 Epochs)

**Step 1: Launch training**

```bash
source .venv/bin/activate
torchrun --nproc_per_node=2 train.py \
  --data_root ./dataset/train \
  --pattern pre_training \
  --epochs 200 \
  --lr 1e-4 \
  --augment \
  --amp \
  --ema \
  --layerwise_lr \
  --warmup_epochs 10 \
  --grad_clip 0.1 \
  --distributed \
  --checkpoint_dir ./checkpoints_v2 \
  --save_every 10
```

Key details:
- Effective batch size: 4 (2 per GPU × 2 GPUs)
- Training split: 2,073 samples → ~518 batches/epoch
- Validation split: 444 samples → ~222 batches/epoch
- Checkpoints saved to `./checkpoints_v2/` (separate from v1)
- Best model tracked by validation loss

**Step 2: Monitor progress**

Check that loss decreases over first 10 epochs. Expected pattern:
- Epoch 0-10 (warmup): loss ~5-10, gradually decreasing
- Epoch 10-50: loss should drop to ~2-4
- Epoch 50-200: loss should stabilize around 1-3

---

### Task 3: Evaluate on Test Set

**Step 1: Run evaluation with new best checkpoint**

```bash
source .venv/bin/activate
python eval.py \
  --checkpoint ./checkpoints_v2/best_model.pth \
  --pattern pre_training \
  --data_root ./dataset/test \
  --batch_size 2 \
  --eval_sc
```

Expected: Processes all 665 test files, prints Stenosis/Plaque/SC metrics.

**Step 2: Compare against baseline**

| Metric | Baseline (v1) | New (v2) |
|--------|:------------:|:--------:|
| Stenosis ACC | 0.702 | ? |
| Stenosis F1 | 0.413 | ? |
| Plaque ACC | 0.430 | ? |
| Plaque F1 | 0.100 | ? |
| SC Points ACC | 0.801 | ? |

**Step 3: If v2 is worse, also evaluate final_model.pth**

```bash
python eval.py \
  --checkpoint ./checkpoints_v2/final_model.pth \
  --pattern pre_training \
  --data_root ./dataset/test \
  --batch_size 2 \
  --eval_sc
```

---

### Task 4: Update Documentation

**Files:**
- Modify: `report.md`
- Modify: `CHANGELOG.md`

**Step 1: Update report.md with new training run results**

Add a new section documenting:
- Training configuration used
- Training loss curve summary
- Test evaluation results (v1 vs v2 comparison table)

**Step 2: Commit all changes**

```bash
git add report.md CHANGELOG.md checkpoints_v2/best_model.pth checkpoints_v2/final_model.pth
git commit -m "Add v2 training results with all improvements"
git push origin master
```
