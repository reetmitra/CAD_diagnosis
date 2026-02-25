"""
eval.py — Evaluation script for SC-Net

Computes artery-level metrics from the MICCAI 2024 paper:
- Stenosis Degree: ACC, Precision, Recall, F1, Specificity (3 classes)
- Plaque Composition: ACC, Precision, Recall, F1, Specificity (3 classes)
- (Optional) Sampling Point Classification accuracy

Note: compute_metrics, od_predictions_to_artery_level, and targets_to_artery_level
are imported by train.py and other evaluation utilities. Avoid circular imports.

Usage:
  python eval.py --checkpoint ./checkpoints/best_model.pth --pattern fine_tuning
  python eval.py --checkpoint ./checkpoints/pretrain.pth --pattern pre_training --data_root ./test_data
  python eval.py --checkpoint ./checkpoints/best_model.pth --pattern fine_tuning --eval_sc --batch_size 4
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from framework import sc_net_framework
import augmentation as aug
from config import opt


def parse_args():
    parser = argparse.ArgumentParser(description='SC-Net Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--pattern', type=str, default='fine_tuning',
                        choices=['pre_training', 'fine_tuning'],
                        help='Evaluation mode (determines num_classes)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override test dataset root path')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: "auto", "cpu", "cuda:0", etc.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for evaluation (default: 2)')
    parser.add_argument('--eval_sc', action='store_true',
                        help='Also evaluate the sampling point classification branch')
    return parser.parse_args()


def get_device(device_str):
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def compute_metrics(y_true, y_pred, num_classes):
    """Compute classification metrics including specificity."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Specificity: TN / (TN + FP) per class, then macro-average
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    spec = np.mean(specificities)

    return {
        'acc': acc,
        'prec': prec,
        'recall': recall,
        'f1': f1,
        'spec': spec
    }


def od_predictions_to_artery_level(od_outputs, num_classes):
    """
    Convert object detection predictions to artery-level classifications.

    Uses a confidence-based approach: instead of only looking at the argmax class,
    we also consider the maximum object-class probability across all queries.
    This is more robust when the model predicts most queries as "no object".

    Returns:
        stenosis_pred: int (0=healthy, 1=non-significant, 2=significant)
        plaque_pred: int (0/1/2 = plaque composition group, -1=none)
            For num_classes=3: 0=calcified, 1=non-calcified, 2=mixed
            For num_classes=6: class//2 gives group (0, 1, 2)
    """
    pred_logits = od_outputs['pred_logits']  # [num_queries, num_classes+1]

    # Apply softmax to get class probabilities
    pred_probs = F.softmax(pred_logits, dim=-1)  # [num_queries, num_classes+1]

    # Get predicted classes via argmax
    pred_classes = pred_probs.argmax(dim=-1)  # [num_queries]

    # Filter out no-object predictions (class index == num_classes)
    is_object = pred_classes < num_classes
    object_classes = pred_classes[is_object]
    object_scores = pred_probs[is_object, :num_classes].max(dim=-1).values if is_object.any() else None

    # Fallback: if argmax gives no detections, check if any query has a
    # non-trivial object probability (> 0.1). This catches cases where the
    # model softly predicts plaque but "no object" still wins the argmax.
    if len(object_classes) == 0:
        # Look at the max object-class probability across all queries
        object_probs = pred_probs[:, :num_classes]  # [num_queries, num_classes]
        max_obj_prob_per_query = object_probs.max(dim=-1)  # values, indices
        best_query_idx = max_obj_prob_per_query.values.argmax()
        best_obj_prob = max_obj_prob_per_query.values[best_query_idx]

        if best_obj_prob.item() > 0.1:
            # Use this soft prediction
            best_class = max_obj_prob_per_query.indices[best_query_idx]
            object_classes = best_class.unsqueeze(0)
            object_scores = best_obj_prob.unsqueeze(0)
        else:
            # Genuinely no detection — artery is healthy
            return 0, -1

    # For 6-class fine-tuning:
    #   Raw labels 1-2 -> classes 0-1: non-significant stenosis
    #   Raw labels 3-4 -> classes 2-3: significant stenosis (plaque group B)
    #   Raw labels 5-6 -> classes 4-5: significant stenosis (plaque group C)
    # Stenosis: classes 0-1 = non-significant, classes 2-5 = significant
    # Plaque:   class // 2 gives plaque group (0, 1, 2) — three compositions
    if num_classes == 6:
        # Stenosis degree: classes >= 2 means significant
        if (object_classes >= 2).any():
            stenosis_pred = 2  # significant
        else:
            stenosis_pred = 1  # non-significant

        # Plaque composition: extract plaque group from class index
        # Classes 0-1 -> group 0, classes 2-3 -> group 1, classes 4-5 -> group 2
        plaque_types = object_classes // 2
        # Use the most confident detection for plaque type
        if object_scores is not None and len(object_scores) > 1:
            best_idx = object_scores.argmax()
            plaque_pred = plaque_types[best_idx].item()
        else:
            plaque_pred = plaque_types[0].item()

    # For 3-class pre-training (plaque composition only):
    # Class 0 = calcified, Class 1 = non-calcified, Class 2 = mixed
    elif num_classes == 3:
        stenosis_pred = 1  # all plaques are considered non-significant in pre-training
        plaque_types = object_classes
        if object_scores is not None and len(object_scores) > 1:
            best_idx = object_scores.argmax()
            plaque_pred = plaque_types[best_idx].item()
        else:
            plaque_pred = object_classes[0].item()

    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")

    return stenosis_pred, plaque_pred


def targets_to_artery_level(targets, num_classes):
    """
    Convert ground truth targets to artery-level classifications.

    Returns:
        stenosis_gt: int (0=healthy, 1=non-significant, 2=significant)
        plaque_gt: int (0=calcified, 1=non-calcified, 2=mixed, -1=none)
    """
    labels = targets['labels']  # [N] where N is number of ground truth boxes

    # If no ground truth boxes, artery is healthy
    if len(labels) == 0:
        return 0, -1

    # Same logic as predictions
    if num_classes == 6:
        # Classes >= 2 (raw labels 3-6) are significant stenosis
        if (labels >= 2).any():
            stenosis_gt = 2  # significant
        else:
            stenosis_gt = 1  # non-significant
        # Plaque group: class // 2 -> 0, 1, or 2
        plaque_types = labels // 2
        plaque_gt = plaque_types[0].item()
    elif num_classes == 3:
        stenosis_gt = 1
        plaque_gt = labels[0].item()
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")

    return stenosis_gt, plaque_gt


def _build_sc_point_labels(targets_i, seq_length):
    """
    Build per-point ground truth labels for sampling point classification.

    This mirrors the od2sc_targets logic from optimization.py:
    - Background points get label 0
    - Points within a box get label (class + 1)

    Args:
        targets_i: dict with 'labels' [N] and 'boxes' [N, 2 or 4] (center-width format)
        seq_length: number of sampling points

    Returns:
        point_labels: LongTensor of shape [seq_length], values in [0, num_classes]
    """
    point_labels = torch.zeros(seq_length, dtype=torch.long)
    labels = targets_i.get('labels', None)
    boxes = targets_i.get('boxes', None)

    if labels is None or boxes is None or labels.numel() == 0 or boxes.numel() == 0:
        return point_labels

    interval = 1.0 / (seq_length + 1)
    # Use first dimension as center and second as width (center-width format)
    centers = boxes[:, 0]
    widths = boxes[:, 1]
    x1 = centers - widths / 2.0
    x2 = centers + widths / 2.0
    starts = torch.round(x1 / interval).int()
    ends = torch.round(x2 / interval).int()
    starts = torch.clamp(starts, min=1, max=seq_length) - 1
    ends = torch.clamp(ends, min=1, max=seq_length) - 1

    for k in range(starts.shape[0]):
        point_labels[starts[k]:ends[k] + 1] = labels[k] + 1

    return point_labels


@torch.no_grad()
def evaluate(model, test_loader, device, num_classes, eval_sc=False):
    """Run evaluation on test set.

    Args:
        model: SC-Net model
        test_loader: DataLoader for test data
        device: torch device
        num_classes: number of object classes (3 or 6)
        eval_sc: if True, also evaluate sampling point classification branch

    Returns:
        stenosis_metrics: dict of stenosis classification metrics
        plaque_metrics: dict of plaque classification metrics
        sc_metrics: dict of sampling point classification metrics (or None if eval_sc=False)
    """
    model.eval()

    all_stenosis_preds = []
    all_stenosis_gts = []
    all_plaque_preds = []
    all_plaque_gts = []

    # Sampling point classification tracking
    sc_correct = 0
    sc_total = 0

    print("\nRunning inference on test set...")
    for batch_idx, (images, targets) in enumerate(test_loader):
        images = images.to(device)

        # Forward pass
        od_outputs, sc_outputs = model(images)

        # Process each sample in the batch
        batch_size = images.shape[0]
        for i in range(batch_size):
            # --- Object detection branch ---
            od_out_i = {
                'pred_logits': od_outputs['pred_logits'][i],
                'pred_boxes': od_outputs['pred_boxes'][i]
            }
            target_i = targets[i]

            # Convert to artery-level predictions
            stenosis_pred, plaque_pred = od_predictions_to_artery_level(od_out_i, num_classes)
            stenosis_gt, plaque_gt = targets_to_artery_level(target_i, num_classes)

            all_stenosis_preds.append(stenosis_pred)
            all_stenosis_gts.append(stenosis_gt)

            # Only include plaque predictions for samples with plaques in GT
            if plaque_gt != -1:
                # Even if pred is -1 (no detection), count it against the GT
                # by mapping -1 to a dummy class (num_classes) so it counts as wrong
                effective_plaque_pred = plaque_pred if plaque_pred != -1 else num_classes
                all_plaque_preds.append(effective_plaque_pred)
                all_plaque_gts.append(plaque_gt)
            elif plaque_pred != -1 and plaque_gt == -1:
                # False positive: predicted plaque on a healthy artery
                # We still track this for stenosis (already done above)
                pass

            # --- Sampling point classification branch ---
            if eval_sc and sc_outputs is not None and 'pred_logits' in sc_outputs:
                sc_logits_i = sc_outputs['pred_logits'][i]  # [seq_length, num_classes+1]
                sc_preds_i = sc_logits_i.argmax(dim=-1)     # [seq_length]

                seq_length = sc_logits_i.shape[0]
                sc_gt_i = _build_sc_point_labels(target_i, seq_length).to(device)

                sc_correct += (sc_preds_i == sc_gt_i).sum().item()
                sc_total += seq_length

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

    print(f"Evaluation complete: {len(all_stenosis_gts)} samples\n")

    # Compute stenosis metrics
    stenosis_metrics = compute_metrics(all_stenosis_gts, all_stenosis_preds, num_classes=3)

    # Compute plaque metrics
    if len(all_plaque_preds) > 0:
        # Determine the effective number of classes for plaque metric computation.
        # If any prediction mapped to the dummy "no detection" class, we need
        # num_classes+1 labels; otherwise use 3 (the standard plaque classes).
        plaque_num_classes = 3
        if any(p >= 3 for p in all_plaque_preds):
            # There are "no detection" predictions for samples that have GT plaques.
            # Include a 4th dummy label so confusion matrix is correct.
            plaque_num_classes = max(max(all_plaque_preds), max(all_plaque_gts)) + 1
        plaque_metrics = compute_metrics(all_plaque_gts, all_plaque_preds,
                                         num_classes=plaque_num_classes)
    else:
        print("Warning: No plaque samples found in ground truth of the test set.")
        plaque_metrics = {'acc': 0, 'prec': 0, 'recall': 0, 'f1': 0, 'spec': 0}

    # Compute sampling point classification metrics
    sc_metrics = None
    if eval_sc:
        if sc_total > 0:
            sc_acc = sc_correct / sc_total
            sc_metrics = {'acc': sc_acc, 'total_points': sc_total, 'correct_points': sc_correct}
        else:
            print("Warning: No sampling points evaluated (sc_total=0).")
            sc_metrics = {'acc': 0.0, 'total_points': 0, 'correct_points': 0}

    return stenosis_metrics, plaque_metrics, sc_metrics


def print_results(stenosis_metrics, plaque_metrics, sc_metrics=None):
    """Print evaluation results in paper format."""
    print("=" * 70)
    print("SC-Net Evaluation Results (Artery-Level)")
    print("=" * 70)
    print()
    print("Stenosis Degree Classification (Healthy / Non-significant / Significant):")
    print(f"  ACC:         {stenosis_metrics['acc']:.3f}")
    print(f"  Precision:   {stenosis_metrics['prec']:.3f}")
    print(f"  Recall:      {stenosis_metrics['recall']:.3f}")
    print(f"  F1:          {stenosis_metrics['f1']:.3f}")
    print(f"  Specificity: {stenosis_metrics['spec']:.3f}")
    print()
    print("Plaque Composition Classification (Calcified / Non-calcified / Mixed):")
    print(f"  ACC:         {plaque_metrics['acc']:.3f}")
    print(f"  Precision:   {plaque_metrics['prec']:.3f}")
    print(f"  Recall:      {plaque_metrics['recall']:.3f}")
    print(f"  F1:          {plaque_metrics['f1']:.3f}")
    print(f"  Specificity: {plaque_metrics['spec']:.3f}")

    if sc_metrics is not None:
        print()
        print("Sampling Point Classification:")
        print(f"  ACC:         {sc_metrics['acc']:.3f}")
        print(f"  Points:      {sc_metrics['correct_points']}/{sc_metrics['total_points']}")

    print()
    print("=" * 70)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Initialize framework
    print(f"\nLoading model in {args.pattern} mode...")
    fw = sc_net_framework(
        pattern=args.pattern,
        state_dict_root=None,  # Will load checkpoint manually
        data_root=args.data_root,
    )

    # Determine num_classes based on pattern
    num_classes = 3 if args.pattern == 'pre_training' else 6
    print(f"Number of classes: {num_classes}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        fw.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch {epoch}")
    else:
        fw.model.load_state_dict(checkpoint)
        print("Loaded model weights")

    model = fw.model.to(device)

    # When an explicit data_root is given, use ALL files in that directory
    # (bypassing the 70/15/15 train/val/test split used during training)
    if args.data_root is not None:
        dataset_test = aug.cubic_sequence_data(
            dataset_root=args.data_root,
            pattern='testing',
            train_ratio=opt.data_params["train_ratio"],
            input_shape=opt.net_params["input_shape"],
            window=opt.data_params["window_lw"],
            num_classes=num_classes)
        dataset_test.data_start = 0
        dataset_test.data_end = dataset_test.file_total
        dataset_test.length = dataset_test.file_total
        test_loader = DataLoader(dataset_test, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=aug.collate_fn)
        print(f"Using all {dataset_test.file_total} files from {args.data_root}")
    else:
        test_loader = fw.dataLoader_test

    print(f"Test set: {len(test_loader)} batches (batch_size={args.batch_size})")

    # Run evaluation
    stenosis_metrics, plaque_metrics, sc_metrics = evaluate(
        model, test_loader, device, num_classes, eval_sc=args.eval_sc
    )

    # Print results
    print_results(stenosis_metrics, plaque_metrics, sc_metrics)


if __name__ == '__main__':
    main()
