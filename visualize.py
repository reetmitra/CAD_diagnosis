#!/usr/bin/env python3
"""
CPR Visualization Tool

Renders one PNG per artery: longitudinal CPR strip with ground truth label
bands and (optionally) model prediction boxes.

Usage:
  # Ground truth only
  python visualize.py --data_root ./dataset/test --pattern testing --output_dir ./viz_gt

  # Single-model with GT/Pred bars (recommended)
  python visualize.py --data_root ./dataset/test --pattern testing \\
      --checkpoint checkpoints_v7_finetune/final_model.pth \\
      --thresholds calibration_thresholds_v7_constrained.json --use_constrained \\
      --output_dir ./viz_v7ft
  #
  #  Output layout per PNG:
  #    Row 0: longitudinal CT strip with GT bands + TP/FN/FP boxes
  #    Row 1 (GT bar):   black=normal slice, red=abnormal (per-slice GT)
  #    Row 2 (Pred bar): black=no prediction, red=prediction fires here
  #    Row 3: cross-section panels (up to 4 labelled segments)

  # Before/After comparison (pre-trained vs fine-tuned)
  python visualize.py --data_root ./dataset/test --pattern testing \\
      --checkpoint  checkpoints_v6/best_model.pth --model_pattern pre_training \\
      --label "Pre-trained (v6)" \\
      --checkpoint2 checkpoints_v7_finetune/final_model.pth \\
      --thresholds2 calibration_thresholds_v7_constrained.json --use_constrained2 \\
      --label2 "Fine-tuned v7 (constrained)" \\
      --output_dir ./viz_comparison

  # Error analysis only
  python visualize.py ... --filter incorrect
"""

import argparse
import math
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

from functions import normalize_ct_data, normalize_ct_data_multiwindow, DEFAULT_MULTI_WINDOWS
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

# Paper Figure 3 colour legend (6 semantic classes):
#   raw label 0              → No-lesion       (green)
#   raw labels 1,2,3  (non-sig + any plaque)   → stenosis bars: Non-significant (yellow)
#   raw labels 4,5,6  (sig + any plaque)       → stenosis bars: Significant (orange)
#   plaque component decoded separately:
#     calcified (labels 1,4) → blue
#     non-calcified (2,5)    → pink
#     mixed (3,6)            → purple

# Dual-bar colour maps — matching Figure 3 legend exactly
# Top bar: stenosis severity
STEN_BAR_COLOURS = {
    0: '#4CAF50',   # green  — No-lesion
    1: '#FFC107',   # yellow — Non-significant stenosis
    2: '#FF6600',   # orange — Significant stenosis
}
# Bottom bar: plaque composition
PLAQUE_BAR_COLOURS = {
    0: '#4CAF50',   # green  — No-lesion
    1: '#2196F3',   # blue   — Calcified plaque   (raw labels 3, 6)
    2: '#FF80AB',   # pink   — Non-calcified plaque (raw labels 1, 4)
    3: '#9C27B0',   # purple — Mixed plaque         (raw labels 2, 5)
}

# Keep for cross-section border colouring (raw label → dominant colour)
RAW_BAR_COLOURS = {
    0: '#4CAF50',
    1: '#FF80AB', 2: '#9C27B0', 3: '#2196F3',
    4: '#FF80AB', 5: '#9C27B0', 6: '#2196F3',
}


def _raw_to_sten_class(lbl):
    """Raw label (0-6) → stenosis class index for STEN_BAR_COLOURS."""
    if lbl == 0:
        return 0
    return 1 if lbl <= 3 else 2


def _raw_to_plaque_class(lbl):
    """Raw label (0-6) → plaque class index for PLAQUE_BAR_COLOURS."""
    if lbl == 0:
        return 0
    if lbl in (1, 4):
        return 2   # Non-calcified → pink
    if lbl in (2, 5):
        return 3   # Mixed → purple
    return 1       # Calcified (3, 6) → blue

# Prediction-box overlay colours (used in comparison/annotation mode)
PRED_COLOURS = {
    0: '#4CAF50',   # green  — predicted Healthy
    1: '#FFD700',   # yellow — predicted Non-sig
    2: '#FF6600',   # orange — predicted Significant
}

# HU windowing (must match training: window=[300, 900])
HU_MIN = 300 - 900 / 2   # = -150
HU_MAX = 300 + 900 / 2   # =  750

# Paper legend entries — matches Figure 3 legend exactly
PAPER_LEGEND = [
    ('#4CAF50', 'No-lesion'),
    ('#FFC107', 'Non-significant stenosis'),
    ('#FF6600', 'Significant stenosis'),
    ('#2196F3', 'Calcified plaque'),
    ('#FF80AB', 'Non-calcified plaque'),
    ('#9C27B0', 'Mixed plaque'),
]


# ─── Label helpers ────────────────────────────────────────────────────────────

def _sten_gt_from_labels(labels):
    """Derive artery-level stenosis GT (0/1/2) from raw label array."""
    gt = labels[labels > 0]
    if len(gt) == 0:
        return 0   # Healthy
    if np.any(gt >= 3):
        return 2   # Significant
    return 1       # Non-significant


def iou_1d(a0, a1, b0, b1):
    """1D intersection-over-union of intervals [a0,a1] and [b0,b1].

    All arguments are floats in the same coordinate space.
    Returns a float in [0, 1].
    """
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(od_outputs, gt_segments, num_classes, D,
                             conf_thresh=0.15, iou_thresh=0.3):
    """Classify each predicted box and GT segment as TP/FN/FP via 1D IoU.

    Args:
        od_outputs: dict with 'pred_logits' shape [Q, C+1] and 'pred_boxes'
                    shape [Q, 2] (center_norm, width_norm).
        gt_segments: list of (start_idx, end_idx_exclusive, raw_label) in
                     pixel coords.
        num_classes: int C; queries whose predicted class == num_classes are
                     no-object and are skipped.
        D: int, vessel axis length (e.g. 256).
        conf_thresh: float; skip queries whose max-class prob < this.
        iou_thresh: float; IoU cutoff for a match.

    Returns:
        tp_pred_idx:    set of indices into surviving_queries that matched a GT.
        fn_seg_idx:     set of GT segment indices that were never matched.
        fp_pred_idx:    set of indices into surviving_queries that were not matched.
        pred_intervals: list of (x0_norm, x1_norm) for each surviving query.
        surviving_queries: list of original query indices [0..Q-1] that passed
                           the conf/class filter.
    """
    pred_logits = od_outputs['pred_logits']   # [Q, C+1]
    pred_boxes  = od_outputs['pred_boxes']    # [Q, 2]

    probs        = F.softmax(pred_logits, dim=-1)          # [Q, C+1]
    pred_classes = torch.argmax(probs, dim=-1)             # [Q]

    # Step 1: filter surviving queries
    # A query is kept only when its best foreground-class probability strictly
    # exceeds the no-object-class probability AND meets conf_thresh.
    surviving_queries = []
    pred_intervals    = []
    Q = pred_logits.shape[0]
    for q in range(Q):
        cls = pred_classes[q].item()
        # Skip if argmax already landed on the no-object class
        if cls >= num_classes:
            continue
        fg_prob = probs[q, cls].item()
        no_obj_prob = probs[q, num_classes].item()
        # Skip if no-object is at least as likely (handles ties with uniform logits)
        if fg_prob <= no_obj_prob:
            continue
        if fg_prob < conf_thresh:
            continue
        cx = pred_boxes[q, 0].item()
        w  = pred_boxes[q, 1].item()
        x0 = cx - w / 2.0
        x1 = cx + w / 2.0
        surviving_queries.append(q)
        pred_intervals.append((x0, x1))

    # Step 2: build GT intervals in normalised coords
    gt_intervals_norm = [(s / D, e / D) for s, e, _ in gt_segments]

    # Step 3: greedy matching
    matched_gt   = set()
    tp_pred_idx  = set()

    for i, (x0, x1) in enumerate(pred_intervals):
        best_iou  = -1.0
        best_gt_j = -1
        for j, (g0, g1) in enumerate(gt_intervals_norm):
            if j in matched_gt:
                continue
            iou = iou_1d(x0, x1, g0, g1)
            if iou >= iou_thresh and iou > best_iou:
                best_iou  = iou
                best_gt_j = j
        if best_gt_j >= 0:
            matched_gt.add(best_gt_j)
            tp_pred_idx.add(i)

    fp_pred_idx = set(range(len(surviving_queries))) - tp_pred_idx
    fn_seg_idx  = set(range(len(gt_segments))) - matched_gt

    return tp_pred_idx, fn_seg_idx, fp_pred_idx, pred_intervals, surviving_queries


# ─── Model helpers ────────────────────────────────────────────────────────────

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


@torch.no_grad()
def predict_artery(model, volume, device, num_classes,
                   stenosis_t=None, plaque_t=None):
    """Run single-sample inference and return artery-level predictions + raw OD outputs.

    Model output verified:
        pred_logits shape: [B, Q, C+1] = [1, 16, 7]  (6 classes + 1 no-object)
        pred_boxes  shape: [B, Q, 2]   = [1, 16, 2]

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
    # Normalise and convert to tensor
    if hasattr(model, 'in_channels') and model.in_channels > 1:
        vol_norm = normalize_ct_data_multiwindow(volume, DEFAULT_MULTI_WINDOWS)
    else:
        vol_norm = normalize_ct_data(volume, hu_min=HU_MIN, hu_max=HU_MAX)
    
    tensor = torch.tensor(vol_norm, dtype=torch.float32).unsqueeze(0).to(device)

    outputs = model(tensor)
    od_out_raw = outputs[0]   # (od_outputs, sc_outputs) — take OD

    # Squeeze batch dimension from logits/boxes
    # Verified shapes: pred_logits [1, 16, 7], pred_boxes [1, 16, 2]
    od_outputs = {
        'pred_logits': od_out_raw['pred_logits'][0],   # [Q, C+1] = [16, 7]
        'pred_boxes':  od_out_raw['pred_boxes'][0],    # [Q, 2]   = [16, 2]
    }

    if stenosis_t is not None:
        # Apply per-class threshold scaling: pred = argmax(p_i / t_i)
        logits = od_outputs['pred_logits']               # [Q, C+1]
        probs  = F.softmax(logits, dim=-1)               # [Q, C+1]
        # Build threshold vector matching logits dimension
        t_vec = torch.ones(logits.shape[-1], dtype=torch.float32, device=device)
        t_vec[:3] = torch.tensor(stenosis_t, dtype=torch.float32, device=device)
        scaled_logits = torch.log(probs / t_vec.unsqueeze(0) + 1e-9)
        od_outputs['pred_logits'] = scaled_logits

    stenosis_pred, plaque_pred = od_predictions_to_artery_level(od_outputs, num_classes)
    return stenosis_pred, plaque_pred, od_outputs


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize CPR images with ground truth and model predictions')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset root containing volumes/ and labels/')
    parser.add_argument('--pattern', type=str, default='all',
                        choices=['all', 'training', 'validation', 'testing'],
                        help='Dataset split to visualize; "all" uses the full folder (default: all)')
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
    # Comparison mode (--checkpoint2)
    parser.add_argument('--checkpoint2', type=str, default=None,
                        help='Optional: second SC-Net checkpoint for comparison strip')
    parser.add_argument('--model_pattern2', type=str, default='fine_tuning',
                        choices=['pre_training', 'fine_tuning'],
                        help='Model pattern for checkpoint2 (default: fine_tuning)')
    parser.add_argument('--thresholds2', type=str, default=None,
                        help='Calibration JSON for checkpoint2')
    parser.add_argument('--use_constrained2', action='store_true',
                        help='Use constrained_stenosis_thresholds from --thresholds2 JSON')
    parser.add_argument('--multi_window', action='store_true', default=False,
                        help='Use multi-window inference for checkpoint(s) if applicable')
    parser.add_argument('--label', type=str, default='Model A',
                        help='Display label for model 1 strip (default: "Model A")')
    parser.add_argument('--label2', type=str, default='Model B',
                        help='Display label for model 2 strip (default: "Model B")')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='1D IoU threshold for TP/FN/FP classification (default: 0.3)')
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device(args.device)

    # ── Model 1 ────────────────────────────────────────────────────────────
    model = None
    num_classes = 6 if args.model_pattern == 'fine_tuning' else 3
    if args.checkpoint:
        print(f"Loading model 1 from {args.checkpoint}...")
        model = _load_model_from_checkpoint(
            args.checkpoint, args.model_pattern, device, args.data_root, args.multi_window)
        model.eval()

    stenosis_t, plaque_t = load_thresholds(args.thresholds, args.use_constrained)
    if stenosis_t:
        print(f"  Stenosis thresholds: {stenosis_t}")
    if plaque_t:
        print(f"  Plaque thresholds:   {plaque_t}")

    # ── Model 2 (comparison mode) ───────────────────────────────────────────
    model2 = None
    num_classes2 = 6 if args.model_pattern2 == 'fine_tuning' else 3
    stenosis_t2 = plaque_t2 = None
    if args.checkpoint2:
        print(f"Loading model 2 from {args.checkpoint2}...")
        model2 = _load_model_from_checkpoint(
            args.checkpoint2, args.model_pattern2, device, args.data_root, args.multi_window)
        model2.eval()
        stenosis_t2, plaque_t2 = load_thresholds(args.thresholds2, args.use_constrained2)
        if stenosis_t2:
            print(f"  Stenosis thresholds (m2): {stenosis_t2}")
        if plaque_t2:
            print(f"  Plaque thresholds   (m2): {plaque_t2}")

    pairs = get_file_pairs(args.data_root, args.pattern)
    print(f"Found {len(pairs)} arteries in split '{args.pattern}'")

    comparison_mode = model2 is not None

    saved = 0
    for vol_path, sten_path, plaq_path, comb_path, artery_id in pairs:
        if args.max_samples > 0 and saved >= args.max_samples:
            break
        vol, labels = load_volume_and_labels(vol_path, sten_path, plaq_path, comb_path)

        stenosis_pred, plaque_pred = None, None
        od_outputs = None
        if model is not None:
            stenosis_pred, plaque_pred, od_outputs = predict_artery(
                model, vol, device, num_classes, stenosis_t, plaque_t)

        stenosis_pred2, plaque_pred2 = None, None
        od_outputs2 = None
        if model2 is not None:
            stenosis_pred2, plaque_pred2, od_outputs2 = predict_artery(
                model2, vol, device, num_classes2, stenosis_t2, plaque_t2)

        sten_gt = _sten_gt_from_labels(labels)

        # Apply filter (considers both models in comparison mode)
        if args.filter == 'correct':
            m1_correct = stenosis_pred is not None and stenosis_pred == sten_gt
            m2_correct = stenosis_pred2 is None or stenosis_pred2 == sten_gt
            if not (m1_correct and m2_correct):
                continue
        elif args.filter == 'incorrect':
            m1_wrong = stenosis_pred is not None and stenosis_pred != sten_gt
            m2_wrong = stenosis_pred2 is not None and stenosis_pred2 != sten_gt
            if not (m1_wrong or m2_wrong):
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

        sten_tag = ['Healthy', 'NonSig', 'Sig'][sten_gt]

        # Build filename — include both model tags in comparison mode
        if comparison_mode:
            def _tag(pred): return ['Healthy', 'NonSig', 'Sig'][pred] if pred is not None else 'None'
            def _result(pred): return ('CORRECT' if pred == sten_gt else 'WRONG') if pred is not None else 'NOPRED'
            filename = (f'{artery_id}__sten_{sten_tag}'
                        f'_m1_{_tag(stenosis_pred)}_{_result(stenosis_pred)}'
                        f'_m2_{_tag(stenosis_pred2)}_{_result(stenosis_pred2)}.png')
        elif stenosis_pred is not None:
            pred_tag   = ['Healthy', 'NonSig', 'Sig'][stenosis_pred]
            correct_tag = 'CORRECT' if stenosis_pred == sten_gt else 'WRONG'
            filename = f'{artery_id}__sten_{sten_tag}_pred_{pred_tag}_{correct_tag}.png'
        else:
            filename = f'{artery_id}__sten_{sten_tag}.png'

        save_path = os.path.join(args.output_dir, filename)
        render_artery(artery_id, vol, labels, save_path,
                      stenosis_pred=stenosis_pred,
                      plaque_pred=plaque_pred,
                      od_outputs=od_outputs,
                      num_classes=num_classes,
                      stenosis_pred2=stenosis_pred2,
                      plaque_pred2=plaque_pred2,
                      od_outputs2=od_outputs2,
                      num_classes2=num_classes2,
                      label1=args.label,
                      label2=args.label2,
                      iou_threshold=args.iou_threshold)
        print(f"  [{saved + 1}/{len(pairs)}] {filename}")
        saved += 1

    print(f"\nDone. Saved {saved} PNGs to '{args.output_dir}'.")
    if model is not None and args.filter in ('correct', 'incorrect'):
        print(f"  Filter '{args.filter}' was active — totals reflect filtered count only.")


def _build_viz_file_pairs(vol_dir, lbl_dir):
    """Build (vol_path, sten_path_or_None, plaq_path_or_None, comb_path_or_None, artery_id) tuples.

    Matches volumes to labels by stem name. Prefers new .nii.gz + separate label format.
    """
    lbl_set = set(os.listdir(lbl_dir))
    pairs = []
    seen_stems = set()
    for vf in sorted(os.listdir(vol_dir)):
        if vf.endswith('.nii.gz'):
            stem = vf[:-7]
        elif vf.endswith('.nii'):
            stem = vf[:-4]
        else:
            continue
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        sten_f = stem + '_stenosis.txt'
        plaq_f = stem + '_plaque.txt'
        comb_f = stem + '.txt'
        # Prefer new format: .nii.gz with separate stenosis/plaque labels
        if vf.endswith('.nii.gz') and sten_f in lbl_set and plaq_f in lbl_set:
            pairs.append((
                os.path.join(vol_dir, vf),
                os.path.join(lbl_dir, sten_f),
                os.path.join(lbl_dir, plaq_f),
                None,
                stem,
            ))
        elif comb_f in lbl_set:
            # Fall back to old format: .nii with combined label
            old_vf = stem + '.nii'
            vol_path_candidate = os.path.join(vol_dir, old_vf)
            if os.path.exists(vol_path_candidate):
                vol_path = vol_path_candidate
            else:
                vol_path = os.path.join(vol_dir, vf)
            pairs.append((vol_path, None, None, os.path.join(lbl_dir, comb_f), stem))
    return pairs


def get_file_pairs(data_root, pattern):
    """Return sorted list of (vol_path, sten_path, plaq_path, comb_path, artery_id) for the given split.

    Uses the same deterministic sort + train_ratio split as cubic_sequence_data
    (train_ratio=0.8) so indices are consistent with training/eval.
    """
    vol_dir = os.path.join(data_root, 'volumes')
    lbl_dir = os.path.join(data_root, 'labels')
    all_pairs = _build_viz_file_pairs(vol_dir, lbl_dir)
    n = len(all_pairs)
    train_ratio = 0.8

    if pattern == 'all':
        start = 0
        end = n
    elif pattern == 'training':
        start = 0
        end = int(n * train_ratio)
    elif pattern == 'validation':
        start = int(n * train_ratio)
        end = int(n * (train_ratio + (1 - train_ratio) / 2))
    else:  # testing
        start = int(n * (train_ratio + (1 - train_ratio) / 2))
        end = n

    return all_pairs[start:end]


def load_volume_and_labels(vol_path, sten_path=None, plaq_path=None, comb_path=None):
    """Load a CPR NIfTI volume and labels (new or old format).

    Handles both:
      - Old format: combined label in one file
      - New format: separate stenosis and plaque labels

    Returns:
        volume: np.ndarray shape (256, 64, 64), raw HU values
        labels: np.ndarray shape (256,), int32, raw label values 0-6
    """
    from scipy.ndimage import zoom

    # Load volume
    img = nib.load(vol_path)
    vol = img.get_fdata()                 # may be (H, W, D) or (D, H, W)
    # NIfTI stores (H, W, D); transpose to (D, H, W) when depth is the last (largest) dim.
    # Use vessel-axis check (same as augmentation.py) — avoids false trigger on 95×95 cross-sections.
    if vol.shape[2] > vol.shape[0]:
        vol = vol.transpose(2, 0, 1)

    # Resize to model input shape — must match opt.net_params["input_shape"] = [256, 64, 64]
    target_shape = (256, 64, 64)
    if list(vol.shape) != list(target_shape):
        zf = [target_shape[i] / vol.shape[i] for i in range(3)]
        vol = zoom(vol, zf, order=3)

    # Load labels
    if comb_path is not None:
        labels = np.loadtxt(comb_path).astype(np.int32)
    else:
        from augmentation import merge_new_labels
        sten = np.loadtxt(sten_path).astype(np.int32)
        plaq = np.loadtxt(plaq_path).astype(np.int32)
        labels = merge_new_labels(sten, plaq)

    # Resize labels to match vessel axis length
    if len(labels) != target_shape[0]:
        labels = zoom(labels.astype(float), target_shape[0] / len(labels), order=0).astype(np.int32)

    return vol, labels




def render_artery(artery_id, volume, labels, save_path,
                  stenosis_pred=None, plaque_pred=None, od_outputs=None,
                  num_classes=6,
                  stenosis_pred2=None, plaque_pred2=None, od_outputs2=None,
                  num_classes2=6,
                  label1="Model A", label2="Model B",
                  iou_threshold=0.3):
    """Render a CPR visualisation matching the layout of Figure 3 in the paper.

    Layout per PNG (top to bottom):
        Row 0: Longitudinal CPR grayscale strip (thin MIP over central rows)
              with red X sampling-point markers matching the paper figure
        Row 1: Ground Truth combined bar
        Row 2: Model 1 prediction bar
        Row 3: Model 2 prediction bar  (comparison mode only)
        Row 4: Cross-section panels at the centre of each GT lesion segment

    Each bar is the same pixel-width as the CPR strip, using the paper legend:
        green=No-lesion, yellow=Non-sig stenosis, orange=Significant stenosis,
        blue=Calcified, pink=Non-calcified, purple=Mixed
    """
    import matplotlib.colors as mcolors

    segments = decode_label_segments(labels)
    sten_gt  = _sten_gt_from_labels(labels)
    D        = volume.shape[0]

    # ── CPR strip: thin MIP over 5 central rows — less noise than a single row
    H        = volume.shape[1]
    cy       = H // 2
    r0, r1   = max(0, cy - 2), min(H, cy + 3)
    strip      = volume[:, r0:r1, :].max(axis=1).T    # (W, D)
    strip_norm = normalize_ct_data(strip, hu_min=HU_MIN, hu_max=HU_MAX)

    # ── Cross-section positions: centre of each GT segment (up to 4)
    cs_positions = [((s + e) // 2, lbl) for s, e, lbl in segments][:4]
    n_cs  = max(len(cs_positions), 1)

    comparison_mode = od_outputs2 is not None
    n_models        = 3 if comparison_mode else 2   # GT + model1 [+ model2]

    # ── Figure layout — 2 thin bars per model (stenosis + plaque), matching Fig 3
    bar_h = 0.22   # height of each individual thin bar
    # Build height_ratios: CPR strip | pairs of bars | cross-sections
    hr = [3] + [bar_h, bar_h] * n_models + [1.8]
    fig = plt.figure(figsize=(max(14, n_cs * 3.5), sum(hr) * 1.5))
    gs  = fig.add_gridspec(1 + n_models * 2 + 1, n_cs,
                           height_ratios=hr, hspace=0.04, wspace=0.25)
    ax_strip   = fig.add_subplot(gs[0, :])
    # model_axes[i] = (sten_ax, plaque_ax) for model i (0=GT, 1=model1, 2=model2)
    model_axes = []
    for i in range(n_models):
        sten_ax   = fig.add_subplot(gs[1 + i * 2, :])
        plaque_ax = fig.add_subplot(gs[2 + i * 2, :])
        model_axes.append((sten_ax, plaque_ax))
    cs_row_idx = 1 + n_models * 2

    # ── Helper: draw dual bars (stenosis on top, plaque below) for one model row
    def _draw_dual_bars(sten_ax, plaque_ax, labels_1d, row_label):
        n          = len(labels_1d)
        sten_rgb   = np.zeros((1, n, 3))
        plaque_rgb = np.zeros((1, n, 3))
        for i, lbl in enumerate(labels_1d):
            lbl = int(lbl)
            sten_rgb[0, i, :]   = mcolors.hex2color(
                STEN_BAR_COLOURS[_raw_to_sten_class(lbl)])
            plaque_rgb[0, i, :] = mcolors.hex2color(
                PLAQUE_BAR_COLOURS[_raw_to_plaque_class(lbl)])
        sten_ax.imshow(sten_rgb,   aspect='auto', origin='upper',
                       interpolation='nearest')
        plaque_ax.imshow(plaque_rgb, aspect='auto', origin='upper',
                         interpolation='nearest')
        # Label on the left of the stenosis bar (vertically centred over both)
        sten_ax.set_yticks([0])
        sten_ax.set_yticklabels([row_label], fontsize=8, fontweight='bold')
        plaque_ax.set_yticks([])
        for ax in (sten_ax, plaque_ax):
            ax.set_xticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # ── Helper: OD outputs → per-slice combined-label array (0-6)
    def _od_to_combined_labels(od_out, n_cls, conf_thresh=0.15):
        out = np.zeros(D, dtype=np.int32)
        if od_out is None:
            return out
        pred_logits = od_out['pred_logits']
        pred_boxes  = od_out['pred_boxes']
        probs       = F.softmax(pred_logits, dim=-1)
        pred_cls    = probs.argmax(dim=-1)
        for q in range(pred_logits.shape[0]):
            cls = pred_cls[q].item()
            if cls >= n_cls:
                continue
            fg_prob  = probs[q, cls].item()
            no_obj_p = probs[q, n_cls].item()
            if fg_prob <= no_obj_p or fg_prob < conf_thresh:
                continue
            cx  = pred_boxes[q, 0].item()
            w   = pred_boxes[q, 1].item()
            x0  = int(max(0, (cx - w / 2) * D))
            x1  = int(min(D, (cx + w / 2) * D))
            if x1 <= x0:
                continue
            raw_lbl = cls + 1
            out[x0:x1] = np.where(out[x0:x1] < raw_lbl, raw_lbl, out[x0:x1])
        return out

    # ── CT strip
    ax_strip.imshow(strip_norm, cmap='gray', aspect='auto', origin='upper',
                    vmin=0, vmax=1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    ax_strip.set_ylabel('CPR', fontsize=8)

    # Red X sampling-point markers (32 cubes, step=8, matching paper figure)
    cube_step = 8
    num_cubes = 32
    strip_cy  = strip_norm.shape[0] // 2
    for i in range(num_cubes):
        sx = cube_step // 2 + cube_step * i - 1
        if 0 <= sx < D:
            ax_strip.plot(sx, strip_cy, 'rx', markersize=4,
                          markeredgewidth=1.0, alpha=0.85)

    # Title
    if od_outputs is not None and stenosis_pred is not None:
        tick  = '✓' if stenosis_pred == sten_gt else '✗'
        title = (f'{artery_id}  |  GT: {STENOSIS_NAMES[sten_gt]}'
                 f'  |  {label1}: {STENOSIS_NAMES[stenosis_pred]} {tick}')
    else:
        title = f'{artery_id}  |  GT: {STENOSIS_NAMES[sten_gt]}'
    ax_strip.set_title(title, fontsize=10, fontweight='bold', loc='left')

    # ── Bars — dual (stenosis + plaque) per model row, matching Figure 3
    _draw_dual_bars(*model_axes[0], labels, 'Ground Truth')
    _draw_dual_bars(*model_axes[1], _od_to_combined_labels(od_outputs, num_classes), label1)
    if comparison_mode:
        _draw_dual_bars(*model_axes[2],
                        _od_to_combined_labels(od_outputs2, num_classes2), label2)

    # ── Legend (paper colours, 2 rows matching Figure 3 legend layout)
    legend_patches = [mpatches.Patch(facecolor=c, label=n) for c, n in PAPER_LEGEND]
    model_axes[0][0].legend(handles=legend_patches, loc='upper right',
                            fontsize=7, ncol=3, framealpha=0.85, borderpad=0.4)

    # ── TP/FN detection for cross-section border styling
    if od_outputs is not None:
        _, fn_idx1, _, _, _ = match_predictions_to_gt(
            od_outputs, segments, num_classes, D, iou_thresh=iou_threshold)
    else:
        fn_idx1 = set()
    if comparison_mode and od_outputs2 is not None:
        _, fn_idx2, _, _, _ = match_predictions_to_gt(
            od_outputs2, segments, num_classes2, D, iou_thresh=iou_threshold)
    else:
        fn_idx2 = set()

    # ── Cross-section panels
    for col, (z_idx, raw_lbl) in enumerate(cs_positions):
        ax_cs  = fig.add_subplot(gs[cs_row_idx, col])
        cs_img = normalize_ct_data(volume[z_idx], hu_min=HU_MIN, hu_max=HU_MAX)
        ax_cs.imshow(cs_img, cmap='gray', vmin=0, vmax=1, origin='upper')
        ax_cs.set_xticks([])
        ax_cs.set_yticks([])

        seg_idx = None
        for si, (s, e, _) in enumerate(segments):
            if s <= z_idx < e:
                seg_idx = si
                break

        border_c  = RAW_BAR_COLOURS.get(raw_lbl, 'white')
        lw        = 3
        linestyle = '-'
        if seg_idx is not None and raw_lbl != 0:
            if comparison_mode:
                if seg_idx in fn_idx1 and seg_idx in fn_idx2:
                    linestyle = '--'   # both models missed
                elif seg_idx in fn_idx1:
                    lw = 5             # model 2 caught what model 1 missed
            elif seg_idx in fn_idx1:
                linestyle = '--'       # single-model miss

        for spine in ax_cs.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_c)
            spine.set_linewidth(lw)
            spine.set_linestyle(linestyle)
        ax_cs.set_title(f'z={z_idx}', color=border_c, fontsize=8)

    # Fill unused columns
    for col in range(len(cs_positions), n_cs):
        fig.add_subplot(gs[cs_row_idx, col]).axis('off')

    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close(fig)



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


if __name__ == '__main__':
    main()
