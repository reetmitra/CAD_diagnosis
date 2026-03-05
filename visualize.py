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


if __name__ == '__main__':
    main()
