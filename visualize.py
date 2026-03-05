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
    # Normalise and convert to tensor [1, D, H, W]
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

    pairs = get_file_pairs(args.data_root, args.pattern)
    print(f"Found {len(pairs)} arteries in split '{args.pattern}'")

    saved = 0
    for vol_path, lbl_path, artery_id in pairs:
        if args.max_samples > 0 and saved >= args.max_samples:
            break
        vol, labels = load_volume_and_labels(vol_path, lbl_path)

        stenosis_pred, plaque_pred = None, None
        od_outputs = None
        if model is not None:
            stenosis_pred, plaque_pred, od_outputs = predict_artery(
                model, vol, device, num_classes, stenosis_t, plaque_t)

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

    print(f"Done: {saved} PNGs saved to '{args.output_dir}'.")


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
