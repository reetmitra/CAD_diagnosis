# CPR Visualization Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `visualize.py` — a batch script that renders one PNG per artery showing a longitudinal CPR strip with ground truth label bands, optional model prediction boxes, and cross-sectional slice panels.

**Architecture:** Standalone script at project root following the same CLI pattern as `eval.py` and `calibrate.py`. Imports reusable utilities (`od_predictions_to_artery_level`, `_load_model_from_checkpoint`, `normalize_ct_data`) from existing modules. No new dependencies.

**Tech Stack:** Python, matplotlib (Agg backend), nibabel, numpy, torch, argparse — all already installed in `.venv/`.

---

## Label Encoding Reference

For `fine_tuning` (num_classes=6), raw label file values mean:

| Raw label | `detection_targets` class | Stenosis       | Plaque         | Colour  |
|-----------|--------------------------|----------------|----------------|---------|
| 0         | — (background)           | Healthy        | None           | none    |
| 1, 2      | 0, 1                     | Non-significant| Calcified      | yellow  |
| 3, 4      | 2, 3                     | Significant    | Non-calcified  | orange  |
| 5, 6      | 4, 5                     | Significant    | Mixed          | red     |

For `pre_training` (num_classes=3), raw labels 1–3 → class 0–2 (Calcified / Non-calc / Mixed).

---

## Task 1: Scaffold — parse_args, main loop, file enumeration

**Files:**
- Create: `visualize.py`

**Step 1: Create the file with imports, constants, parse_args, and a main() stub**

```python
#!/usr/bin/env python3
"""
CPR Visualization Tool

Renders one PNG per artery: longitudinal CPR strip with ground truth label
bands and (optionally) model prediction boxes.

Usage:
  # Ground truth only
  python visualize.py --data_root ./dataset/test --pattern testing --output_dir ./viz_gt

  # With model predictions
  python visualize.py --data_root ./dataset/test --pattern testing \\
      --checkpoint checkpoints_v7_finetune/final_model.pth \\
      --thresholds calibration_thresholds_v7_constrained.json --use_constrained \\
      --output_dir ./viz_v7ft

  # Error analysis only
  python visualize.py ... --filter incorrect
"""

import argparse
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from functions import normalize_ct_data
from eval import (
    od_predictions_to_artery_level,
    targets_to_artery_level,
    _load_model_from_checkpoint,
    get_device,
)
import augmentation as aug

# ─── Constants ────────────────────────────────────────────────────────────────

STENOSIS_NAMES = ['Healthy', 'Non-significant', 'Significant']
PLAQUE_NAMES   = ['Calcified', 'Non-calcified', 'Mixed']

# Colour per raw label value (0 = no colour)
RAW_LABEL_COLOURS = {
    0: None,
    1: '#FFD700',   # gold   — Non-sig + Calcified
    2: '#FFA500',   # orange — Non-sig + Calcified (variant)
    3: '#FF6600',   # dark-orange — Sig + Non-calc
    4: '#FF4400',   # red-orange  — Sig + Non-calc (variant)
    5: '#FF0000',   # red         — Sig + Mixed
    6: '#CC0000',   # dark-red    — Sig + Mixed (variant)
}

# Prediction-box colours by artery-level stenosis prediction
PRED_COLOURS = {
    0: '#00CC00',   # green  — predicted Healthy
    1: '#FFD700',   # gold   — predicted Non-sig
    2: '#FF0000',   # red    — predicted Significant
}

# HU windowing (must match training: window=[300, 900])
HU_MIN = 300 - 900 / 2   # = -150
HU_MAX = 300 + 900 / 2   # =  750


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize CPR images with ground truth and model predictions')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset root containing volumes/ and labels/')
    parser.add_argument('--pattern', type=str, default='testing',
                        choices=['training', 'validation', 'testing'],
                        help='Dataset split to visualize (default: testing)')
    parser.add_argument('--output_dir', type=str, default='./viz',
                        help='Directory to save PNG files (default: ./viz)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional: SC-Net checkpoint for prediction overlay')
    parser.add_argument('--model_pattern', type=str, default='fine_tuning',
                        choices=['pre_training', 'fine_tuning'],
                        help='Model pattern (determines num_classes) (default: fine_tuning)')
    parser.add_argument('--thresholds', type=str, default=None,
                        help='Path to calibration JSON from calibrate.py')
    parser.add_argument('--use_constrained', action='store_true',
                        help='Use constrained_stenosis_thresholds from --thresholds JSON')
    parser.add_argument('--filter', type=str, default='all',
                        choices=['all', 'correct', 'incorrect',
                                 'healthy', 'nonsig', 'sig'],
                        help='Only save PNGs matching this filter (default: all)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Stop after N samples (0 = unlimited)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto / cuda / cpu')
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # TODO: tasks 2-7 go here

    print("visualize.py scaffold OK")


if __name__ == '__main__':
    main()
```

**Step 2: Verify scaffold runs without errors**

```bash
cd /home/reet/development/CAD_diagnosis && \
  .venv/bin/python visualize.py --data_root ./dataset/test --output_dir ./viz_test
```

Expected: prints `visualize.py scaffold OK`, no import errors.

**Step 3: Commit**

```bash
git add visualize.py
git commit -m "feat(viz): scaffold visualize.py with CLI and imports"
```

---

## Task 2: Data loading — file list + load_volume_and_labels

**Files:**
- Modify: `visualize.py`

**Step 1: Add `get_file_pairs()` that returns sorted (volume_path, label_path, artery_id) for a split**

Insert after the `main()` stub, before `if __name__ == '__main__':`:

```python
def get_file_pairs(data_root, pattern):
    """Return sorted list of (vol_path, lbl_path, artery_id) for the given split.

    Uses the same deterministic sort + train_ratio split as cubic_sequence_data
    (train_ratio=0.8) so indices are consistent with training/eval.
    """
    vol_dir = os.path.join(data_root, 'volumes')
    lbl_dir = os.path.join(data_root, 'labels')
    vol_files = sorted(os.listdir(vol_dir))
    lbl_files = sorted(os.listdir(lbl_dir))
    n = len(vol_files)
    train_ratio = 0.8

    if pattern == 'training':
        start = 0
        end = int(n * train_ratio)
    elif pattern == 'validation':
        start = int(n * train_ratio)
        end = int(n * (train_ratio + (1 - train_ratio) / 2))
    else:  # testing
        start = int(n * (train_ratio + (1 - train_ratio) / 2))
        end = n

    pairs = []
    for i in range(start, end):
        vf = vol_files[i]
        lf = lbl_files[i]
        artery_id = os.path.splitext(vf)[0]   # strip .nii
        pairs.append((
            os.path.join(vol_dir, vf),
            os.path.join(lbl_dir, lf),
            artery_id,
        ))
    return pairs
```

**Step 2: Add `load_volume_and_labels()`**

```python
def load_volume_and_labels(vol_path, lbl_path):
    """Load a CPR NIfTI volume and its per-slice label file.

    Returns:
        volume: np.ndarray shape (256, 64, 64), raw HU values
        labels: np.ndarray shape (256,), int32, raw label values 0-6
    """
    img = nib.load(vol_path)
    vol = img.get_fdata()                 # may be (64, 64, 256) or (256, 64, 64)
    if vol.shape[0] == vol.shape[1]:      # (64, 64, 256) → transpose to (256, 64, 64)
        vol = vol.transpose(2, 0, 1)
    labels = np.loadtxt(lbl_path).astype(np.int32)
    return vol, labels
```

**Step 3: Wire into main() — iterate and print artery IDs**

Replace `# TODO: tasks 2-7 go here` with:

```python
    pairs = get_file_pairs(args.data_root, args.pattern)
    print(f"Found {len(pairs)} arteries in split '{args.pattern}'")

    saved = 0
    for vol_path, lbl_path, artery_id in pairs:
        if args.max_samples > 0 and saved >= args.max_samples:
            break
        vol, labels = load_volume_and_labels(vol_path, lbl_path)
        print(f"  {artery_id}: vol={vol.shape}, labels={np.unique(labels)}")
        saved += 1

    print(f"Done: {saved} arteries listed.")
```

**Step 4: Smoke test data loading**

```bash
.venv/bin/python visualize.py \
  --data_root ./dataset/test --pattern testing --max_samples 5 \
  --output_dir ./viz_test
```

Expected: prints 5 artery IDs with correct shapes `(256, 64, 64)` and label values.

**Step 5: Commit**

```bash
git add visualize.py
git commit -m "feat(viz): add get_file_pairs and load_volume_and_labels"
```

---

## Task 3: decode_label_segments + GT-only rendering

**Files:**
- Modify: `visualize.py`

**Step 1: Write `decode_label_segments()` and verify by hand**

```python
def decode_label_segments(labels):
    """Convert a 256-length label array into contiguous segments.

    Returns:
        list of (start_idx, end_idx_exclusive, raw_label) for each non-zero run.
        Background (0) runs are omitted.
    """
    segments = []
    i = 0
    n = len(labels)
    while i < n:
        lbl = labels[i]
        if lbl == 0:
            i += 1
            continue
        j = i + 1
        while j < n and labels[j] == lbl:
            j += 1
        segments.append((i, j, int(lbl)))
        i = j
    return segments
```

Verify manually:
```python
# Quick inline check (not a test file — just paste in a REPL or add a __main__ guard)
segs = decode_label_segments(np.array([0,0,1,1,1,0,5,5,0]))
assert segs == [(2, 5, 1), (6, 8, 5)], segs
print("decode_label_segments OK")
```

Run this check by adding it temporarily to main() and running the script.

**Step 2: Add `render_artery_gt_only()`**

The function renders a matplotlib figure: one longitudinal strip (row 0) and up to
4 cross-section panels (row 1). No model predictions yet.

```python
def render_artery_gt_only(artery_id, volume, labels, save_path):
    """Render longitudinal CPR strip with GT label bands + cross-section panels.

    Args:
        artery_id:  str, used in title
        volume:     np.ndarray (256, 64, 64), raw HU
        labels:     np.ndarray (256,), int32 raw labels
        save_path:  str, output PNG path
    """
    segments = decode_label_segments(labels)

    # ── Longitudinal strip: centre cross-section row across all z ──────────
    # volume[:, 32, :] → (256, 64), transpose → (64, 256) so x=vessel axis
    strip = volume[:, 32, :].T   # shape (64, 256)
    strip_norm = normalize_ct_data(strip, hu_min=HU_MIN, hu_max=HU_MAX)

    # ── Cross-section positions: centre of each labelled segment ───────────
    cs_positions = [((s + e) // 2, lbl) for s, e, lbl in segments]
    cs_positions = cs_positions[:4]   # cap at 4 panels
    n_cs = max(len(cs_positions), 1)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(14, n_cs * 4), 7))
    gs = fig.add_gridspec(2, n_cs, height_ratios=[3, 2], hspace=0.4, wspace=0.3)

    # Row 0: longitudinal strip (spans all columns)
    ax_long = fig.add_subplot(gs[0, :])
    ax_long.imshow(strip_norm, cmap='gray', aspect='auto', origin='upper',
                   vmin=0, vmax=1)

    # GT label bands
    for seg_start, seg_end, raw_lbl in segments:
        colour = RAW_LABEL_COLOURS.get(raw_lbl)
        if colour is not None:
            ax_long.axvspan(seg_start, seg_end, alpha=0.35, color=colour,
                            label=f'Raw {raw_lbl}')

    ax_long.set_xlabel('Vessel axis position (slice index)')
    ax_long.set_ylabel('Cross-section (px)')
    ax_long.set_title(f'{artery_id}   |   GT labels: {np.unique(labels[labels>0]).tolist()}')

    # Row 1: cross-section panels
    for col, (z_idx, raw_lbl) in enumerate(cs_positions):
        ax_cs = fig.add_subplot(gs[1, col])
        cs_img = normalize_ct_data(volume[z_idx], hu_min=HU_MIN, hu_max=HU_MAX)
        ax_cs.imshow(cs_img, cmap='gray', vmin=0, vmax=1, origin='upper')
        colour = RAW_LABEL_COLOURS.get(raw_lbl, 'white')
        ax_cs.set_title(f'z={z_idx}  raw={raw_lbl}', color=colour or 'white',
                        fontsize=9)
        ax_cs.axis('off')

    # Fill unused cross-section columns with blank axes
    for col in range(len(cs_positions), n_cs):
        ax_blank = fig.add_subplot(gs[1, col])
        ax_blank.axis('off')

    # Legend
    legend_patches = []
    for raw_lbl, colour in RAW_LABEL_COLOURS.items():
        if raw_lbl == 0 or colour is None:
            continue
        name = f'Raw {raw_lbl}'
        legend_patches.append(mpatches.Patch(color=colour, alpha=0.6, label=name))
    if legend_patches:
        ax_long.legend(handles=legend_patches, loc='upper right', fontsize=7,
                       ncol=len(legend_patches))

    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
```

**Step 3: Wire into main() — replace the print loop with rendering**

Replace the `print(f"  {artery_id}...")` block with:

```python
        # Derive GT stenosis class for filename
        gt_labels = labels[labels > 0]
        if len(gt_labels) == 0:
            sten_tag = 'Healthy'
        elif np.any(gt_labels >= 3):
            sten_tag = 'Sig'
        else:
            sten_tag = 'NonSig'

        filename = f'{artery_id}__sten_{sten_tag}.png'
        save_path = os.path.join(args.output_dir, filename)
        render_artery_gt_only(artery_id, vol, labels, save_path)
        print(f"  Saved: {filename}")
        saved += 1
```

**Step 4: Smoke test GT rendering**

```bash
.venv/bin/python visualize.py \
  --data_root ./dataset/test --pattern testing --max_samples 5 \
  --output_dir ./viz_gt
```

Expected: 5 PNG files appear in `./viz_gt/`. Open one and verify:
- Grayscale longitudinal strip (64px tall × 256px wide)
- Coloured bands where labels are non-zero
- Up to 4 cross-section panels below

**Step 5: Commit**

```bash
git add visualize.py
git commit -m "feat(viz): GT-only rendering with longitudinal strip and cross-sections"
```

---

## Task 4: Model loading + single-sample inference

**Files:**
- Modify: `visualize.py`

This task adds the ability to run a loaded model on each volume and get raw OD outputs.

**Step 1: Add `load_thresholds()`**

```python
def load_thresholds(thresholds_path, use_constrained):
    """Load per-class thresholds from a calibrate.py JSON file.

    Returns:
        stenosis_t: list of 3 floats, or None
        plaque_t:   list of 3 floats, or None
    """
    if thresholds_path is None:
        return None, None
    with open(thresholds_path) as f:
        data = json.load(f)
    if use_constrained and 'constrained_stenosis_thresholds' in data:
        stenosis_t = data['constrained_stenosis_thresholds']
    else:
        stenosis_t = data.get('stenosis_thresholds')
    plaque_t = data.get('plaque_thresholds')
    return stenosis_t, plaque_t
```

**Step 2: Add `predict_artery()`**

```python
@torch.no_grad()
def predict_artery(model, volume, device, num_classes,
                   stenosis_t=None, plaque_t=None):
    """Run single-sample inference and return artery-level predictions + raw OD outputs.

    Args:
        model:       SC-Net model in eval mode
        volume:      np.ndarray (256, 64, 64), raw HU
        device:      torch device
        num_classes: 3 or 6
        stenosis_t:  list of 3 thresholds or None (argmax)
        plaque_t:    list of 3 thresholds or None (argmax)

    Returns:
        stenosis_pred: int 0-2
        plaque_pred:   int 0-2 or -1
        od_outputs:    raw dict with 'pred_logits' [Q, C+1] and 'pred_boxes' [Q, 2]
    """
    # Normalise and convert to tensor [1, D, H, W]
    vol_norm = normalize_ct_data(volume, hu_min=HU_MIN, hu_max=HU_MAX)
    tensor = torch.tensor(vol_norm, dtype=torch.float32).unsqueeze(0).to(device)

    outputs = model(tensor)
    od_out_raw = outputs[0]   # (od_outputs, sc_outputs) — take OD

    # Squeeze batch dimension from logits/boxes
    od_outputs = {
        'pred_logits': od_out_raw['pred_logits'][0],   # [Q, C+1]
        'pred_boxes':  od_out_raw['pred_boxes'][0],    # [Q, 2]
    }

    if stenosis_t is not None:
        # Apply per-class threshold scaling: pred = argmax(p_i / t_i)
        logits = od_outputs['pred_logits']               # [Q, C+1]
        probs  = F.softmax(logits, dim=-1)               # [Q, C+1]
        # Build threshold vector matching logits dimension
        t_vec = torch.ones(logits.shape[-1], dtype=torch.float32, device=device)
        t_vec[:3] = torch.tensor(stenosis_t, dtype=torch.float32)
        scaled_logits = torch.log(probs / t_vec.unsqueeze(0) + 1e-9)
        od_outputs['pred_logits'] = scaled_logits

    stenosis_pred, plaque_pred = od_predictions_to_artery_level(od_outputs, num_classes)
    return stenosis_pred, plaque_pred, od_outputs
```

**Note on model output shape:** `sc_net_framework.model(x)` returns `(od_outputs_dict, sc_outputs_dict)` where `od_outputs_dict['pred_logits']` has shape `[B, Q, C+1]`. Verify this in `architecture.py` before proceeding if there's any doubt.

**Step 3: Verify model output shape**

```bash
.venv/bin/python -c "
import torch
from framework import sc_net_framework
from config import opt
fw = sc_net_framework(pattern='fine_tuning', state_dict_root=None)
fw.model.eval()
x = torch.randn(1, 256, 64, 64)
with torch.no_grad():
    out = fw.model(x)
print('type:', type(out))
print('keys od:', out[0].keys())
print('pred_logits shape:', out[0]['pred_logits'].shape)
print('pred_boxes shape:', out[0]['pred_boxes'].shape)
"
```

Expected: `pred_logits` shape `[1, Q, 7]` (6 classes + 1 no-object), `pred_boxes` shape `[1, Q, 2]`.
Adjust `predict_artery()` batch-dimension squeezing if the output shape differs.

**Step 4: Wire model loading into main()**

At the top of `main()`, after `os.makedirs`:

```python
    device = get_device(args.device)
    model = None
    num_classes = 6 if args.model_pattern == 'fine_tuning' else 3
    if args.checkpoint:
        print(f"Loading model from {args.checkpoint}...")
        model = _load_model_from_checkpoint(
            args.checkpoint, args.model_pattern, device, args.data_root)
        model.eval()

    stenosis_t, plaque_t = load_thresholds(args.thresholds, args.use_constrained)
    if stenosis_t:
        print(f"  Stenosis thresholds: {stenosis_t}")
    if plaque_t:
        print(f"  Plaque thresholds:   {plaque_t}")
```

And in the loop, after `vol, labels = load_volume_and_labels(...)`, add:

```python
        stenosis_pred, plaque_pred = None, None
        od_outputs = None
        if model is not None:
            stenosis_pred, plaque_pred, od_outputs = predict_artery(
                model, vol, device, num_classes, stenosis_t, plaque_t)
```

**Step 5: Smoke test model loading (no prediction overlay yet)**

```bash
.venv/bin/python visualize.py \
  --data_root ./dataset/test --pattern testing --max_samples 3 \
  --checkpoint checkpoints_v7_finetune/final_model.pth \
  --thresholds calibration_thresholds_v7_constrained.json --use_constrained \
  --output_dir ./viz_model_test
```

Expected: runs without error, prints loaded checkpoint epoch, saves 3 GT-only PNGs (predictions not overlaid yet).

**Step 6: Commit**

```bash
git add visualize.py
git commit -m "feat(viz): add model loading, load_thresholds, predict_artery"
```

---

## Task 5: Prediction overlay + filter + filenames

**Files:**
- Modify: `visualize.py`

**Step 1: Refactor `render_artery_gt_only` → `render_artery` that accepts optional prediction data**

Rename the function and add prediction parameters:

```python
def render_artery(artery_id, volume, labels, save_path,
                  stenosis_pred=None, plaque_pred=None, od_outputs=None,
                  num_classes=6):
```

Inside, after the GT bands block, add the prediction box overlay:

```python
    # ── Predicted bounding boxes (dashed) ──────────────────────────────────
    if od_outputs is not None:
        pred_logits = od_outputs['pred_logits']   # [Q, C+1]
        pred_boxes  = od_outputs['pred_boxes']    # [Q, 2]  (center, width in [0,1])
        pred_probs  = F.softmax(pred_logits, dim=-1)
        pred_classes = pred_probs.argmax(dim=-1)  # [Q]
        D = volume.shape[0]  # 256

        for q in range(pred_classes.shape[0]):
            cls = pred_classes[q].item()
            if cls >= num_classes:   # no-object query
                continue
            prob = pred_probs[q, cls].item()
            if prob < 0.15:          # low-confidence: skip
                continue
            cx_norm, w_norm = pred_boxes[q, 0].item(), pred_boxes[q, 1].item()
            x_left  = (cx_norm - w_norm / 2) * D
            x_right = (cx_norm + w_norm / 2) * D

            # Stenosis group from class index
            if num_classes == 6:
                sten_group = 0 if cls < 2 else 2  # 0=non-sig, 2=sig
            else:
                sten_group = 1
            colour = PRED_COLOURS.get(sten_group, '#FFFFFF')

            ax_long.axvline(x_left,  linestyle='--', color=colour, alpha=0.8, linewidth=1.5)
            ax_long.axvline(x_right, linestyle='--', color=colour, alpha=0.8, linewidth=1.5)
            ax_long.text((x_left + x_right) / 2, 3,
                         f'P{cls}', color=colour, fontsize=7, ha='center', va='top')
```

Also update the title to include prediction info:

```python
    if stenosis_pred is not None:
        sten_gt_name  = STENOSIS_NAMES[_sten_gt_from_labels(labels)]
        sten_pred_name = STENOSIS_NAMES[stenosis_pred]
        correct_str = '✓' if stenosis_pred == _sten_gt_from_labels(labels) else '✗'
        ax_long.set_title(
            f'{artery_id}   |   GT: {sten_gt_name}   Pred: {sten_pred_name}  {correct_str}')
    else:
        ax_long.set_title(f'{artery_id}   |   GT labels: {np.unique(labels[labels>0]).tolist()}')
```

**Step 2: Add `_sten_gt_from_labels()` helper**

```python
def _sten_gt_from_labels(labels):
    """Derive artery-level stenosis GT (0/1/2) from raw label array."""
    gt = labels[labels > 0]
    if len(gt) == 0:
        return 0   # Healthy
    if np.any(gt >= 3):
        return 2   # Significant
    return 1       # Non-significant
```

**Step 3: Add `--filter` logic**

In the main loop, after computing `stenosis_pred`:

```python
        sten_gt = _sten_gt_from_labels(labels)

        # Apply filter
        if args.filter == 'correct':
            if stenosis_pred is None or stenosis_pred != sten_gt:
                continue
        elif args.filter == 'incorrect':
            if stenosis_pred is None or stenosis_pred == sten_gt:
                continue
        elif args.filter == 'healthy':
            if sten_gt != 0:
                continue
        elif args.filter == 'nonsig':
            if sten_gt != 1:
                continue
        elif args.filter == 'sig':
            if sten_gt != 2:
                continue
```

**Step 4: Update filenames in main()**

```python
        sten_tag = ['Healthy', 'NonSig', 'Sig'][sten_gt]
        if stenosis_pred is not None:
            pred_tag = ['Healthy', 'NonSig', 'Sig'][stenosis_pred]
            correct_tag = 'CORRECT' if stenosis_pred == sten_gt else 'WRONG'
            filename = f'{artery_id}__sten_{sten_tag}_pred_{pred_tag}_{correct_tag}.png'
        else:
            filename = f'{artery_id}__sten_{sten_tag}.png'

        save_path = os.path.join(args.output_dir, filename)
        render_artery(artery_id, vol, labels, save_path,
                      stenosis_pred=stenosis_pred,
                      plaque_pred=plaque_pred,
                      od_outputs=od_outputs,
                      num_classes=num_classes)
        print(f"  Saved: {filename}")
        saved += 1
```

**Step 5: Smoke test full pipeline**

```bash
# GT only — all testing arteries
.venv/bin/python visualize.py \
  --data_root ./dataset/test --pattern testing \
  --output_dir ./viz_gt_all

# With model, save only wrong predictions
.venv/bin/python visualize.py \
  --data_root ./dataset/test --pattern testing \
  --checkpoint checkpoints_v7_finetune/final_model.pth \
  --thresholds calibration_thresholds_v7_constrained.json --use_constrained \
  --filter incorrect \
  --output_dir ./viz_v7ft_errors
```

Expected:
- First run: one PNG per artery, GT bands visible, no dashed boxes.
- Second run: only wrong predictions saved; each PNG shows GT bands AND dashed prediction boxes in a different colour; title ends with `✗`.

**Step 6: Commit**

```bash
git add visualize.py
git commit -m "feat(viz): prediction overlay, filter, and informative filenames"
```

---

## Task 6: Polish + progress reporting

**Files:**
- Modify: `visualize.py`

**Step 1: Add a summary printout at the end of main()**

```python
    print(f"\nDone. Saved {saved} PNGs to '{args.output_dir}'.")
    if model is not None and args.filter in ('correct', 'incorrect'):
        print(f"  Filter '{args.filter}' was active — totals reflect filtered count only.")
```

**Step 2: Add a progress counter**

In the loop header, replace `print(f"  Saved: {filename}")` with:

```python
        print(f"  [{saved}/{len(pairs)}] {filename}")
```

**Step 3: Final smoke test — full test split**

```bash
.venv/bin/python visualize.py \
  --data_root ./dataset/test --pattern testing \
  --checkpoint checkpoints_v7_finetune/final_model.pth \
  --thresholds calibration_thresholds_v7_constrained.json --use_constrained \
  --output_dir ./viz_v7ft_all
```

Expected: ~133 PNGs saved (10% of 665 = test split), progress counter updates each item,
summary line at end. Visually inspect 5-10 WRONG cases to confirm dashed boxes differ
from GT bands.

**Step 4: Final commit**

```bash
git add visualize.py
git commit -m "feat(viz): progress counter and summary; visualize.py complete"
```

---

## Verification Checklist

- [ ] `python visualize.py --data_root ./dataset/test --output_dir ./viz_gt` runs without errors
- [ ] GT-only PNGs show coloured bands where labels are non-zero
- [ ] Cross-section panels appear at lesion centres
- [ ] Model mode: dashed boxes appear over the longitudinal strip
- [ ] `--filter incorrect` only saves arteries where `stenosis_pred != sten_gt`
- [ ] Filenames include `_CORRECT` / `_WRONG` tags when checkpoint is provided
- [ ] No matplotlib windows pop up (Agg backend)
- [ ] No CUDA OOM — inference runs one sample at a time
