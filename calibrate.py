"""
calibrate.py -- Per-class threshold calibration for SC-Net stenosis predictions.

Runs inference on the VALIDATION split, collects 3-class stenosis probabilities,
then grid-searches per-class thresholds that maximise macro-F1. Saves thresholds
to JSON for use with eval.py --thresholds.

The calibration decision rule replaces argmax with: pred = argmax(p_i / t_i)
where t = [t_healthy, t_nonsig, t_sig]. Lowering t_sig makes the model more
willing to predict Significant.

Usage:
  python calibrate.py --checkpoint ./checkpoints_v6_finetune/best_model.pth \
      --pattern fine_tuning --output calibration_thresholds.json

  python calibrate.py --checkpoint ./checkpoints_v6_finetune/best_model.pth \
      --pattern fine_tuning --output calibration_thresholds.json --grid_steps 100
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from framework import sc_net_framework
from config import opt
import augmentation as aug
from eval import (
    _collect_artery_probs,
    targets_to_artery_level,
    od_predictions_to_artery_level,
    compute_per_class_metrics,
    compute_metrics,
    STENOSIS_CLASSES,
    PLAQUE_CLASSES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='SC-Net per-class threshold calibration')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--pattern', type=str, default='fine_tuning',
                        choices=['pre_training', 'fine_tuning'])
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override dataset root path')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--output', type=str,
                        default='calibration_thresholds.json',
                        help='Output JSON path (default: calibration_thresholds.json)')
    parser.add_argument('--grid_steps', type=int, default=50,
                        help='Grid points per threshold dimension (default: 50)')
    return parser.parse_args()


def threshold_predict(probs, thresholds):
    """Apply per-class thresholds: pred = argmax(p_i / t_i)."""
    t = np.array(thresholds, dtype=np.float64)
    scaled = probs / t[np.newaxis, :]
    return scaled.argmax(axis=1)


def macro_f1(gts, preds):
    from sklearn.metrics import f1_score
    return f1_score(gts, preds, average='macro', zero_division=0)


def search_thresholds(probs, gts, grid_steps=50):
    """Grid search over t0 (Healthy) and t2 (Significant), with t1=1.0 fixed.

    Lower t2 means more willing to predict Significant.
    Higher t0 means less willing to predict Healthy.
    """
    best_f1 = -1.0
    best_t = [1.0, 1.0, 1.0]

    t0_grid = np.linspace(0.1, 3.0, grid_steps)
    t2_grid = np.linspace(0.05, 1.5, grid_steps)

    for t0 in t0_grid:
        for t2 in t2_grid:
            t = [t0, 1.0, t2]
            preds = threshold_predict(probs, t)
            f1 = macro_f1(gts, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_t = [float(t0), 1.0, float(t2)]

    return best_t, best_f1


@torch.no_grad()
def collect_val_probs(model, val_loader, device, num_classes):
    """Run inference on the validation set, collect artery-level probs + GTs."""
    model.eval()
    all_stenosis_probs = []
    all_stenosis_gts = []
    all_stenosis_preds = []  # baseline argmax predictions
    all_plaque_probs = []
    all_plaque_gts = []

    print(f"\nCollecting probabilities on validation set "
          f"({len(val_loader)} batches)...")

    for batch_idx, (images, targets) in enumerate(val_loader):
        images = images.to(device)
        od_outputs, sc_outputs = model(images)
        batch_size = images.shape[0]

        for i in range(batch_size):
            od_out_i = {
                'pred_logits': od_outputs['pred_logits'][i],
                'pred_boxes': od_outputs['pred_boxes'][i],
            }
            target_i = targets[i]

            stenosis_gt, plaque_gt = targets_to_artery_level(
                target_i, num_classes)
            stenosis_pred, _ = od_predictions_to_artery_level(
                od_out_i, num_classes)

            all_stenosis_gts.append(stenosis_gt)
            all_stenosis_preds.append(stenosis_pred)

            _collect_artery_probs(
                od_out_i, num_classes, stenosis_gt, plaque_gt,
                all_stenosis_probs, all_plaque_probs)

            if plaque_gt != -1:
                all_plaque_gts.append(plaque_gt)

        if (batch_idx + 1) % 10 == 0:
            print(f"  {batch_idx + 1}/{len(val_loader)} batches processed")

    print(f"Collected {len(all_stenosis_gts)} stenosis samples, "
          f"{len(all_plaque_gts)} plaque samples")

    return (np.array(all_stenosis_probs),
            np.array(all_stenosis_gts),
            all_stenosis_preds,
            np.array(all_plaque_probs) if all_plaque_probs else None,
            np.array(all_plaque_gts) if all_plaque_gts else None)


def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    num_classes = 3 if args.pattern == 'pre_training' else 6

    # Load model
    print(f"\nLoading model in {args.pattern} mode...")
    fw = sc_net_framework(
        pattern=args.pattern,
        state_dict_root=None,
        data_root=args.data_root,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    if 'model_state_dict' in checkpoint:
        fw.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        fw.model.load_state_dict(checkpoint)
        print("Loaded model weights")

    model = fw.model.to(device)
    model.eval()

    # Use VALIDATION split (not test)
    val_loader = fw.dataLoader_eval
    print(f"Validation set: {len(val_loader)} batches "
          f"(batch_size={args.batch_size})")

    # Collect probabilities
    sten_probs, sten_gts, sten_preds_baseline, plaque_probs, plaque_gts = \
        collect_val_probs(model, val_loader, device, num_classes)

    # Baseline metrics (argmax, t=[1,1,1])
    baseline_preds = sten_probs.argmax(axis=1)
    baseline_f1 = macro_f1(sten_gts, baseline_preds)
    unique, counts = np.unique(baseline_preds, return_counts=True)
    pred_dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"\n{'='*60}")
    print("BASELINE (argmax)")
    print(f"{'='*60}")
    print(f"  Macro-F1: {baseline_f1:.4f}")
    print(f"  Prediction distribution: {pred_dist}")

    baseline_pc = compute_per_class_metrics(
        sten_gts.tolist(), baseline_preds.tolist(), STENOSIS_CLASSES)
    for m in baseline_pc:
        print(f"    {m['class']:<20} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  N={m['support']}")

    # Grid search
    print(f"\n{'='*60}")
    print(f"THRESHOLD SEARCH (grid_steps={args.grid_steps})")
    print(f"{'='*60}")
    best_t, best_f1 = search_thresholds(
        sten_probs, sten_gts, args.grid_steps)

    cal_preds = threshold_predict(sten_probs, best_t)
    unique, counts = np.unique(cal_preds, return_counts=True)
    cal_dist = dict(zip(unique.tolist(), counts.tolist()))

    print(f"  Best thresholds: Healthy={best_t[0]:.3f}  "
          f"Non-sig={best_t[1]:.3f}  Significant={best_t[2]:.3f}")
    print(f"  Calibrated Macro-F1: {best_f1:.4f}  "
          f"(baseline: {baseline_f1:.4f}, delta: {best_f1 - baseline_f1:+.4f})")
    print(f"  Prediction distribution: {cal_dist}")

    cal_pc = compute_per_class_metrics(
        sten_gts.tolist(), cal_preds.tolist(), STENOSIS_CLASSES)
    for m in cal_pc:
        print(f"    {m['class']:<20} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  N={m['support']}")

    # Also compute standard metrics
    cal_metrics = compute_metrics(sten_gts.tolist(), cal_preds.tolist(),
                                  num_classes=3)
    print(f"\n  Calibrated overall metrics:")
    print(f"    ACC={cal_metrics['acc']:.3f}  Prec={cal_metrics['prec']:.3f}  "
          f"Rec={cal_metrics['recall']:.3f}  F1={cal_metrics['f1']:.3f}  "
          f"Spec={cal_metrics['spec']:.3f}")

    # Save
    output = {
        'stenosis_thresholds': best_t,
        'stenosis_class_names': STENOSIS_CLASSES,
        'val_macro_f1_baseline': float(baseline_f1),
        'val_macro_f1_calibrated': float(best_f1),
        'val_prediction_distribution': {
            STENOSIS_CLASSES[k]: int(v)
            for k, v in zip(unique.tolist(), counts.tolist())
        },
        'val_calibrated_metrics': {
            k: float(v) for k, v in cal_metrics.items()
        },
        'checkpoint': args.checkpoint,
        'pattern': args.pattern,
        'grid_steps': args.grid_steps,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nThresholds saved to: {args.output}")


if __name__ == '__main__':
    main()
