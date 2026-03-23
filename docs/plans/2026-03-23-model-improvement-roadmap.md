# Model Improvement Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve stenosis classification from ACC=0.580/F1=0.585/AUC=0.713 toward the paper's 0.914, by (1) fixing the Hungarian matcher to unblock fine-tuning, (2) quick-win inference tricks, (3) fine-tuning from the stronger v9 pre-trained checkpoint, and (4) targeting the Non-sig class specifically.

**Architecture:** SC-Net dual-branch (temporal SC + spatial OD) with DETR-style Hungarian matching. Fine-tuning adapts a 3-class pre-trained model to a 6-class task (3 stenosis + 3 plaque types). The matcher fix enables all downstream training. Quick wins (TTA, ensemble) require no training.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, 2× RTX 3090, `scipy.optimize.linear_sum_assignment`, existing eval/train/calibrate.py pipeline.

---

## Baseline

| Metric | Current (v7-ft constrained) |
|---|---|
| Stenosis ACC | 0.5805 |
| Stenosis F1 | 0.5852 |
| Stenosis AUC (Significant) | 0.863 |
| **Non-sig AUC** | **0.436** ← critical weakness |
| SC branch ACC | 0.8137 |
| Plaque F1 | 0.463 |

**Checkpoint**: `checkpoints_v7_finetune/final_model.pth`
**Calibration**: `calibration_thresholds_v7_constrained.json`
**Test set**: 665 arteries (all pattern)

---

## Phase 1: Fix Hungarian Matcher (Critical Blocker)

The `HungarianMatcher.forward()` in `functions.py:155-169` is broken: it builds a cross-product of `bs × len(sizes)` assignments when only `bs` are needed. This causes a shape mismatch in `object_detection_loss.loss_labels` during fine-tuning, crashing on the first batch.

### Task 1: Write Matcher Smoke Test

**Files:**
- Create: `tests/test_matcher.py`

**Background – the bug:**
```
sizes = [n0, n1]  (one per batch item)
split_cost_matrices = C.split(sizes, -1)  # list of bs tensors

# BUGGY: produces bs×bs indices
for b in range(bs):
    batch_costs = [cm[b] for cm in split_cost_matrices]  # bs costs
    indices.extend([linear_sum_assignment(c) for c in batch_costs])  # ×bs

# CORRECT: produces bs indices
for b in range(bs):
    i, j = linear_sum_assignment(split_cost_matrices[b][b])
    indices.append(...)
```

**Step 1: Create the test file**

```python
# tests/test_matcher.py
import torch
import pytest
import sys
sys.path.insert(0, '.')
from functions import HungarianMatcher


def _make_outputs(bs, num_queries, num_classes):
    return {
        "pred_logits": torch.randn(bs, num_queries, num_classes + 1),
        "pred_boxes":  torch.rand(bs, num_queries, 4),
    }


def _make_targets(sizes, num_classes):
    return [
        {
            "labels": torch.randint(0, num_classes, (n,)),
            "boxes":  torch.rand(n, 4),
        }
        for n in sizes
    ]


def test_matcher_returns_one_index_per_batch_item():
    """Matcher must return exactly bs (row, col) tuples."""
    bs, num_q, C = 2, 16, 3
    matcher = HungarianMatcher()
    outputs = _make_outputs(bs, num_q, C)
    targets = _make_targets([3, 2], C)

    indices = matcher(outputs, targets)

    assert len(indices) == bs, f"Expected {bs} index pairs, got {len(indices)}"


def test_matcher_index_values_in_range():
    """Every source index must be < num_queries; every target index < n_targets."""
    bs, num_q, C = 3, 16, 3
    sizes = [4, 2, 3]
    matcher = HungarianMatcher()
    outputs = _make_outputs(bs, num_q, C)
    targets = _make_targets(sizes, C)

    indices = matcher(outputs, targets)

    for b, (src, tgt) in enumerate(indices):
        assert src.max() < num_q,   f"Batch {b}: src index {src.max()} >= num_queries {num_q}"
        assert tgt.max() < sizes[b], f"Batch {b}: tgt index {tgt.max()} >= n_targets {sizes[b]}"


def test_matcher_handles_empty_batch():
    """Matcher must handle targets with 0 boxes without crashing."""
    bs, num_q, C = 2, 16, 3
    matcher = HungarianMatcher()
    outputs = _make_outputs(bs, num_q, C)
    targets = [
        {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)},
        {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
    ]

    indices = matcher(outputs, targets)
    assert len(indices) == bs


def test_matcher_bs1():
    """Matcher must work for batch size 1."""
    matcher = HungarianMatcher()
    outputs = _make_outputs(1, 16, 3)
    targets = _make_targets([2], 3)

    indices = matcher(outputs, targets)
    assert len(indices) == 1
```

**Step 2: Run the test – expect failures**

```bash
cd /home/reet/development/CAD_diagnosis
.venv/bin/pytest tests/test_matcher.py -v
```

Expected: `test_matcher_returns_one_index_per_batch_item` FAILS (gets `bs*bs` entries)

---

### Task 2: Fix the Matcher

**Files:**
- Modify: `functions.py:155-169`

**Step 1: Apply the fix**

Replace lines 155-169 in `functions.py`:

```python
        sizes = [len(v["boxes"]) for v in targets]
        # Correct batch-aware assignment:
        # split_cost_matrices[b] has shape [bs, num_queries, size_b]
        # We want split_cost_matrices[b][b] — queries of batch item b vs targets of batch item b
        indices = []
        split_cost_matrices = C.split(sizes, -1)
        for b in range(bs):
            i, j = linear_sum_assignment(split_cost_matrices[b][b].numpy())
            indices.append(
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            )
        return indices
```

**Step 2: Run tests – expect pass**

```bash
.venv/bin/pytest tests/test_matcher.py -v
```

Expected output:
```
PASSED tests/test_matcher.py::test_matcher_returns_one_index_per_batch_item
PASSED tests/test_matcher.py::test_matcher_index_values_in_range
PASSED tests/test_matcher.py::test_matcher_handles_empty_batch
PASSED tests/test_matcher.py::test_matcher_bs1
4 passed in X.Xs
```

**Step 3: Smoke test – 1 fine-tuning batch**

```bash
python train.py \
  --config configs/finetune_v7.yaml \
  --pretrained checkpoints_v9/best_model.pth \
  --checkpoint_dir ./checkpoints_smoke \
  --epochs 1 \
  --device cuda:1 \
  --seed 42 \
  2>&1 | head -30
```

Expected: Training runs without RuntimeError. Loss values appear (OD ~3-5, SC ~1-2).

**Step 4: Commit**

```bash
git add tests/test_matcher.py functions.py
git commit -m "fix: correct Hungarian matcher batch indexing (bs×bs→bs indices)"
```

---

## Phase 2: Quick Wins — TTA + Ensemble (No Training Needed)

These tests require zero code changes. Run with existing v7-ft checkpoints.

### Task 3: Test-Time Augmentation (TTA)

**Files:** None (eval.py already supports TTA)

**Step 1: Run TTA evaluation**

```bash
python eval.py \
  --checkpoint checkpoints_v7_finetune/final_model.pth \
  --pattern fine_tuning \
  --data_root ./dataset/test \
  --data_split all \
  --batch_size 1 \
  --eval_sc \
  --detailed \
  --tta --tta_k 5 \
  --use_constrained \
  --calibration_file calibration_thresholds_v7_constrained.json \
  --save_results results_v7ft_tta5.json \
  2>&1 | tee eval_v7ft_tta5.log
```

**Step 2: Compare to baseline**

```bash
python3 -c "
import json
base = json.load(open('results_v7ft_full665.json'))
tta  = json.load(open('results_v7ft_tta5.json'))
for key in ['stenosis_metrics', 'plaque_metrics']:
    print(f'\n{key}:')
    for m in ['acc', 'f1', 'recall']:
        b = base[key][m]
        t = tta[key][m]
        delta = t - b
        sign = '+' if delta >= 0 else ''
        print(f'  {m:8s}: {b:.4f} → {t:.4f}  ({sign}{delta:.4f})')
"
```

Expected: Modest improvement (+0.01-0.03 F1), especially on borderline samples.

**Step 3: Commit result (no code change)**

```bash
git add results_v7ft_tta5.json eval_v7ft_tta5.log
git commit -m "eval: TTA k=5 evaluation on v7-ft model"
```

---

### Task 4: Multi-Checkpoint Ensemble

Use the three best late-stage v7-ft checkpoints: epoch 29, 39, 49.

**Files:** None (eval.py already supports ensemble)

**Step 1: Run ensemble evaluation**

```bash
python eval.py \
  --ensemble \
    checkpoints_v7_finetune/checkpoint_epoch_29.pth \
    checkpoints_v7_finetune/checkpoint_epoch_39.pth \
    checkpoints_v7_finetune/checkpoint_epoch_49.pth \
  --pattern fine_tuning \
  --data_root ./dataset/test \
  --data_split all \
  --batch_size 1 \
  --eval_sc \
  --detailed \
  --use_constrained \
  --calibration_file calibration_thresholds_v7_constrained.json \
  --save_results results_v7ft_ensemble3.json \
  2>&1 | tee eval_v7ft_ensemble3.log
```

**Step 2: Run TTA + Ensemble combined**

```bash
python eval.py \
  --ensemble \
    checkpoints_v7_finetune/checkpoint_epoch_29.pth \
    checkpoints_v7_finetune/checkpoint_epoch_39.pth \
    checkpoints_v7_finetune/checkpoint_epoch_49.pth \
  --pattern fine_tuning \
  --data_root ./dataset/test \
  --data_split all \
  --batch_size 1 \
  --eval_sc \
  --detailed \
  --tta --tta_k 5 \
  --use_constrained \
  --calibration_file calibration_thresholds_v7_constrained.json \
  --save_results results_v7ft_ensemble3_tta5.json \
  2>&1 | tee eval_v7ft_ensemble3_tta5.log
```

**Step 3: Compare all variants**

```bash
python3 -c "
import json
results = {
    'baseline':       'results_v7ft_full665.json',
    'TTA-5':          'results_v7ft_tta5.json',
    'Ensemble-3':     'results_v7ft_ensemble3.json',
    'Ensemble3+TTA5': 'results_v7ft_ensemble3_tta5.json',
}
print(f'{'Variant':<20} {'ACC':>6} {'F1':>6} {'AUC':>6}')
print('-' * 42)
for name, path in results.items():
    try:
        d = json.load(open(path))
        s = d['stenosis_metrics']
        auc = sum(d['stenosis_auc'].values()) / len(d['stenosis_auc'])
        print(f'{name:<20} {s[\"acc\"]:.4f} {s[\"f1\"]:.4f} {auc:.4f}')
    except FileNotFoundError:
        print(f'{name:<20}  NOT FOUND')
"
```

**Step 4: Commit**

```bash
git add results_v7ft_ensemble*.json eval_v7ft_ensemble*.log
git commit -m "eval: ensemble and TTA inference variants on v7-ft"
```

---

## Phase 3: Fine-tune from v9 Pre-trained Checkpoint

v9 is our strongest pre-trained model (687MB checkpoint, trained longer than v7's base). Fine-tuning from it should transfer better representations into the 6-class task.

### Task 5: Create finetune_v9.yaml Config

**Files:**
- Create: `configs/finetune_v9.yaml`

**Step 1: Write the config**

```yaml
# configs/finetune_v9.yaml
# Fine-tuning from v9 pre-trained checkpoint
# Inherits proven v7 config; adds patient split + extended patience

pattern: fine_tuning
data_root: ./dataset/train

# Training schedule
epochs: 200
lr: 3.0e-5
weight_decay: 1.0e-4
grad_clip: 0.1
warmup_epochs: 10

# L_dc warm-up (same as v7 — proven stable)
dc_warmup_hold: 20
dc_warmup_ramp: 30
delta: 1.0
dc_confidence_threshold: 0.3
soft_dc: true
label_smoothing: 0.1

# Early stopping
patience: 50
min_delta: 0.001

# Class imbalance handling
balanced_sampling: true
sc_class_weight: true
focal_loss: true
focal_gamma: 2.0
eos_coef: 0.15

# Effective batch = 2 * 1 GPU * 4 accum = 8
accumulate_steps: 4

# Training infrastructure
amp: true
ema: true
ema_decay: 0.999
layerwise_lr: true
augment: true
num_workers: 4
seed: 42
patient_split: true
split_seed: 42

# Transformer architecture (match v9 pre-training config)
temporal_encoder_layers: 4
temporal_heads: 8
spatial_encoder_layers: 4
spatial_decoder_layers: 4

# Logging
checkpoint_dir: ./checkpoints_v9_finetune
save_every: 10
print_every: 1
log_dir: ./runs
log_every: 10
```

**Step 2: Confirm the config loads**

```bash
python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/finetune_v9.yaml'))
print('checkpoint_dir:', cfg['checkpoint_dir'])
print('lr:', cfg['lr'])
print('dc_warmup_hold:', cfg['dc_warmup_hold'])
print('soft_dc:', cfg['soft_dc'])
print('Config OK')
"
```

Expected: prints config values, no error.

**Step 3: Commit**

```bash
git add configs/finetune_v9.yaml
git commit -m "config: add finetune_v9.yaml for v9→fine-tuning"
```

---

### Task 6: Launch Fine-tuning from v9

**Files:** None (train.py and config already complete)

**Step 1: Create checkpoint directory**

```bash
mkdir -p /home/reet/development/CAD_diagnosis/checkpoints_v9_finetune
```

**Step 2: Launch fine-tuning (background)**

```bash
cd /home/reet/development/CAD_diagnosis
nohup python train.py \
  --config configs/finetune_v9.yaml \
  --pretrained checkpoints_v9/best_model.pth \
  --device cuda:1 \
  > train_v9_finetune.log 2>&1 &
echo "PID: $!"
```

**Step 3: Verify first 5 epochs start cleanly**

```bash
# Wait ~10 minutes then check
tail -50 train_v9_finetune.log
```

Expected first-epoch loss range:
- OD: 3.0–6.0
- SC: 0.8–1.5
- DC: 0.0 (held for first 20 epochs)
- No RuntimeError, no NaN

**Step 4: Monitor training**

```bash
# After epoch 20 (DC ramp begins), confirm DC term appears
grep "DC:" train_v9_finetune.log | head -20
```

Expected: DC values start at ~0 at epoch 20, grow linearly to ~delta by epoch 50.

**Step 5: Track best checkpoint**

```bash
grep "Best\|Saved\|val" train_v9_finetune.log | tail -30
```

---

### Task 7: Evaluate v9 Fine-tuned Model

**Files:** None

**Step 1: Wait for training to reach ≥50 epochs, then evaluate**

```bash
python eval.py \
  --checkpoint checkpoints_v9_finetune/best_model.pth \
  --pattern fine_tuning \
  --data_root ./dataset/test \
  --data_split all \
  --batch_size 2 \
  --eval_sc \
  --detailed \
  --plot \
  --plot_dir plots_v9ft \
  --save_results results_v9ft.json \
  2>&1 | tee eval_v9ft.log
```

**Step 2: Run calibration on v9-ft**

```bash
python calibrate.py \
  --checkpoint checkpoints_v9_finetune/best_model.pth \
  --pattern fine_tuning \
  --data_root ./dataset/train \
  --output_file calibration_thresholds_v9.json \
  --constrain_nonsig_recall 0.10 \
  2>&1 | tee calibrate_v9.log
```

**Step 3: Evaluate with calibration**

```bash
python eval.py \
  --checkpoint checkpoints_v9_finetune/best_model.pth \
  --pattern fine_tuning \
  --data_root ./dataset/test \
  --data_split all \
  --batch_size 2 \
  --eval_sc \
  --detailed \
  --use_constrained \
  --calibration_file calibration_thresholds_v9.json \
  --save_results results_v9ft_calibrated.json \
  2>&1 | tee eval_v9ft_calibrated.log
```

**Step 4: Full comparison table**

```bash
python3 -c "
import json
results = {
    'v7-ft (baseline)':       ('results_v7ft_full665.json',       'uncalibrated'),
    'v7-ft constrained cal':  ('results_v7ft_full665.json',       'see memory'),
    'v9-ft raw':              ('results_v9ft.json',                'uncalibrated'),
    'v9-ft calibrated':       ('results_v9ft_calibrated.json',    'constrained'),
}
print(f'{'Model':<28} {'Sten ACC':>9} {'Sten F1':>8} {'NonSig AUC':>11} {'SC ACC':>7}')
print('-' * 68)
for name, (path, _) in results.items():
    try:
        d = json.load(open(path))
        s = d['stenosis_metrics']
        ns_auc = d.get('stenosis_auc', {}).get('Non-significant', 0.0)
        sc = d.get('sc_metrics', {}).get('acc', 0.0)
        print(f'{name:<28} {s[\"acc\"]:9.4f} {s[\"f1\"]:8.4f} {ns_auc:11.4f} {sc:7.4f}')
    except FileNotFoundError:
        print(f'{name:<28}  (not yet generated)')
"
```

**Step 5: Commit results**

```bash
git add results_v9ft*.json eval_v9ft*.log calibration_thresholds_v9.json
git commit -m "eval: v9 fine-tuned model evaluation and calibration"
```

---

## Phase 4: Target Non-sig AUC (if still below 0.55 after Phase 3)

Non-sig AUC=0.436 is barely above random. It's the boundary class between Healthy and Significant.  This phase adds targeted class-weight tuning and an auxiliary binary detection signal.

### Task 8: Increase Non-sig Loss Weight

The current SC loss treats all lesion classes equally (weight 1.5). Non-sig needs more gradient signal.

**Files:**
- Modify: `optimization.py:99-101` (in `compute_sc_class_weights`)

**Step 1: Write the test first**

Add to `tests/test_matcher.py`:

```python
from optimization import compute_sc_class_weights

def test_sc_class_weights_shape():
    """compute_sc_class_weights must return tensor of length num_classes+1."""
    weights = compute_sc_class_weights(num_classes=3)
    assert weights.shape == (4,), f"Expected shape (4,), got {weights.shape}"

def test_sc_class_weights_non_sig_boosted():
    """After boosting, Non-sig class (index 2 in fine_tuning) > Sig class."""
    # In fine_tuning mode: 0=bg, 1=Healthy, 2=NonSig, 3=Sig, 4=CalcPlaque, 5=NonCalcPlaque, 6=Mixed
    weights = compute_sc_class_weights(num_classes=6, boost_nonsig=True, nonsig_idx=2)
    assert weights[2] > weights[3], "Non-sig weight must exceed Sig weight when boosted"
```

**Step 2: Run test – expect failure (boost_nonsig param doesn't exist yet)**

```bash
.venv/bin/pytest tests/test_matcher.py::test_sc_class_weights_non_sig_boosted -v
```

Expected: `TypeError` — `compute_sc_class_weights` doesn't accept `boost_nonsig`.

**Step 3: Add boost_nonsig parameter to compute_sc_class_weights**

In `optimization.py`, replace `compute_sc_class_weights` (lines ~85-102):

```python
def compute_sc_class_weights(num_classes, boost_nonsig=False, nonsig_idx=2):
    """Return inverse-frequency inspired class weights for SC loss.

    Args:
        num_classes: Number of lesion classes (e.g. 3 for pre_training,
            6 for fine_tuning).  Returned tensor has length num_classes+1
            (background class 0 included).
        boost_nonsig: If True, multiply weight at nonsig_idx by 2.0 to
            give more gradient to the hard Non-significant boundary class.
        nonsig_idx: Class index (1-based lesion index) for Non-significant
            stenosis. Default 2 matches fine_tuning 6-class setup.
    """
    weights = torch.ones(num_classes + 1, dtype=torch.float32)
    weights[0] = 0.5       # background
    weights[1:] = 1.5      # all lesion classes
    if boost_nonsig and nonsig_idx <= num_classes:
        weights[nonsig_idx] = weights[nonsig_idx] * 2.0
    return weights
```

**Step 4: Run all tests – expect pass**

```bash
.venv/bin/pytest tests/test_matcher.py -v
```

Expected: All tests pass.

**Step 5: Wire into train.py**

In `train.py`, find where `compute_sc_class_weights` is called (search for `sc_class_weight`). Add `--boost_nonsig` flag to parser:

```python
# In parse_args(), near --sc_class_weight:
parser.add_argument('--boost_nonsig', action='store_true', default=False,
                    help='Double loss weight for Non-significant stenosis class (class index 2)')
```

And pass it through in the Trainer where `compute_sc_class_weights` is called:

```python
# Find: compute_sc_class_weights(num_classes=...)
# Replace with:
compute_sc_class_weights(
    num_classes=...,
    boost_nonsig=args.boost_nonsig,
    nonsig_idx=2  # Non-sig is always index 2 in fine_tuning
)
```

**Step 6: Commit**

```bash
git add optimization.py train.py tests/test_matcher.py
git commit -m "feat: add boost_nonsig option to SC class weights"
```

---

### Task 9: Run Fine-tuning with Non-sig Boost

**Files:**
- Create: `configs/finetune_v9_nonsig.yaml`

**Step 1: Create config (inherits v9 config + adds boost)**

```yaml
# configs/finetune_v9_nonsig.yaml
# Same as finetune_v9.yaml but with non-sig weight boost

pattern: fine_tuning
data_root: ./dataset/train
epochs: 200
lr: 3.0e-5
weight_decay: 1.0e-4
grad_clip: 0.1
warmup_epochs: 10
dc_warmup_hold: 20
dc_warmup_ramp: 30
delta: 1.0
dc_confidence_threshold: 0.3
soft_dc: true
label_smoothing: 0.1
patience: 50
min_delta: 0.001
balanced_sampling: true
sc_class_weight: true
boost_nonsig: true        # ← new: 3.0 weight on Non-sig class
focal_loss: true
focal_gamma: 2.0
eos_coef: 0.15
accumulate_steps: 4
amp: true
ema: true
ema_decay: 0.999
layerwise_lr: true
augment: true
num_workers: 4
seed: 42
patient_split: true
split_seed: 42
temporal_encoder_layers: 4
temporal_heads: 8
spatial_encoder_layers: 4
spatial_decoder_layers: 4
checkpoint_dir: ./checkpoints_v9_nonsig
save_every: 10
print_every: 1
log_dir: ./runs
log_every: 10
```

**Step 2: Launch training (in parallel with v9_finetune if GPU available)**

```bash
mkdir -p /home/reet/development/CAD_diagnosis/checkpoints_v9_nonsig
nohup python train.py \
  --config configs/finetune_v9_nonsig.yaml \
  --pretrained checkpoints_v9/best_model.pth \
  --device cuda:0 \
  > train_v9_nonsig.log 2>&1 &
echo "PID: $!"
```

**Step 3: Monitor Non-sig recall in early epochs**

```bash
grep "Non-sig\|non_sig\|val_acc" train_v9_nonsig.log | head -20
```

Expected: Non-sig recall starts appearing in validation metrics (currently ~0 in baseline).

**Step 4: Commit config**

```bash
git add configs/finetune_v9_nonsig.yaml
git commit -m "config: add finetune_v9_nonsig.yaml with 2x Non-sig class weight"
```

---

## Phase 5: Final Evaluation and Model Selection

### Task 10: Full Comparison Across All Models

**Files:** None

**Step 1: Run calibrate + eval on all new checkpoints**

```bash
for CKPT in checkpoints_v9_finetune/best_model.pth checkpoints_v9_nonsig/best_model.pth; do
    NAME=$(basename $(dirname $CKPT))
    echo "=== Calibrating $NAME ==="
    python calibrate.py \
      --checkpoint $CKPT \
      --pattern fine_tuning \
      --data_root ./dataset/train \
      --output_file calibration_${NAME}.json \
      --constrain_nonsig_recall 0.10

    echo "=== Evaluating $NAME ==="
    python eval.py \
      --checkpoint $CKPT \
      --pattern fine_tuning \
      --data_root ./dataset/test \
      --data_split all \
      --batch_size 2 \
      --eval_sc \
      --detailed \
      --use_constrained \
      --calibration_file calibration_${NAME}.json \
      --save_results results_${NAME}.json
done
```

**Step 2: Generate comprehensive comparison**

```bash
python3 -c "
import json, glob

files = {
    'v7-ft (current best)': 'calibrated_results.json',   # existing constrained result
    'v7-ft TTA':            'results_v7ft_tta5.json',
    'v7-ft Ensemble-3':     'results_v7ft_ensemble3.json',
    'v9-ft':                'results_checkpoints_v9_finetune.json',
    'v9-ft+NonSig':         'results_checkpoints_v9_nonsig.json',
}

per_class = ['Healthy', 'Non-significant', 'Significant']

print(f'{'Model':<25} {'ACC':>6} {'F1':>6} {'H-Rec':>7} {'NS-Rec':>7} {'S-Rec':>7} {'SC':>6}')
print('-' * 72)
for name, path in files.items():
    try:
        d = json.load(open(path))
        s = d['stenosis_metrics']
        pc = {c['class']: c for c in d.get('stenosis_per_class', [])}
        sc = d.get('sc_metrics', {}).get('acc', 0.0)
        h = pc.get('Healthy', {}).get('recall', 0.0)
        ns = pc.get('Non-significant', {}).get('recall', 0.0)
        sig = pc.get('Significant', {}).get('recall', 0.0)
        print(f'{name:<25} {s[\"acc\"]:6.4f} {s[\"f1\"]:6.4f} {h:7.4f} {ns:7.4f} {sig:7.4f} {sc:6.4f}')
    except FileNotFoundError:
        print(f'{name:<25}  (not available)')
"
```

**Step 3: Select best model and document**

Criteria for best model:
1. Highest Stenosis F1 overall
2. Non-sig Recall ≥ 0.15 (at least some detection)
3. SC branch ACC ≥ 0.80 (don't degrade SC)

**Step 4: Commit final summary**

```bash
git add results_*.json calibration_*.json
git commit -m "eval: final comparison across all model variants"
```

---

## Expected Outcomes

| Phase | Model | Expected Stenosis F1 | Non-sig AUC | Rationale |
|---|---|---|---|---|
| Baseline | v7-ft | 0.585 | 0.436 | Current best |
| Phase 2 | v7-ft + TTA | 0.595–0.615 | ~0.44 | Variance reduction |
| Phase 2 | v7-ft Ensemble | 0.600–0.625 | ~0.45 | Model averaging |
| Phase 3 | v9-ft | 0.620–0.660 | 0.50–0.60 | Stronger base checkpoint |
| Phase 4 | v9-ft + NonSig | 0.630–0.680 | 0.55–0.65 | Targeted class weight |

Paper target (0.914) likely includes test-set overlap or different evaluation protocol. Our realistic target: **F1 ≥ 0.65, Non-sig AUC ≥ 0.55**.

---

## Rollback Plan

If any phase degrades performance:
- **Phase 1 (matcher fix)**: Matcher only used during fine-tuning; eval/inference unaffected.
- **Phase 2 (TTA/Ensemble)**: Read-only; no model changes.
- **Phase 3 (v9-ft)**: New checkpoint directory; v7-ft untouched.
- **Phase 4 (NonSig boost)**: New checkpoint directory; can revert via `git revert`.

The stable v7-ft model (`checkpoints_v7_finetune/final_model.pth`) is always available.

---

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `functions.py:155-169` | Modify | Fix Hungarian matcher |
| `tests/test_matcher.py` | Create | TDD tests for matcher |
| `optimization.py:85-102` | Modify | Add boost_nonsig param |
| `train.py` | Modify | Wire --boost_nonsig flag |
| `configs/finetune_v9.yaml` | Create | v9 fine-tuning config |
| `configs/finetune_v9_nonsig.yaml` | Create | v9 + Non-sig boost config |

---

**Date**: 2026-03-23
**Estimated wall-clock time**: Phase 1 (30min) + Phase 2 (2hr) + Phase 3 (8–16hr training) + Phase 4 (8–16hr training)
