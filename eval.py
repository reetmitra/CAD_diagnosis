"""
eval.py — Evaluation script for SC-Net

Computes artery-level metrics from the MICCAI 2024 paper:
- Stenosis Degree: ACC, Precision, Recall, F1, Specificity (3 classes)
- Plaque Composition: ACC, Precision, Recall, F1, Specificity (3 classes)

Usage:
  python eval.py --checkpoint ./checkpoints/best_model.pth --pattern fine_tuning
  python eval.py --checkpoint ./checkpoints/pretrain.pth --pattern pre_training --data_root ./test_data
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from framework import sc_net_framework


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

    Returns:
        stenosis_pred: int (0=healthy, 1=non-significant, 2=significant)
        plaque_pred: int (0=calcified, 1=non-calcified, 2=mixed, -1=none)
    """
    pred_logits = od_outputs['pred_logits']  # [16, num_classes+1]
    pred_boxes = od_outputs['pred_boxes']    # [16, 4 or 2]

    # Apply softmax to get class probabilities
    pred_probs = F.softmax(pred_logits, dim=-1)  # [16, num_classes+1]

    # Get predicted classes and confidence scores
    pred_classes = pred_probs.argmax(dim=-1)  # [16]
    pred_scores = pred_probs.max(dim=-1).values  # [16]

    # Filter out no-object predictions (class index == num_classes)
    is_object = pred_classes < num_classes
    object_classes = pred_classes[is_object]

    # If no detections, artery is healthy
    if len(object_classes) == 0:
        return 0, -1  # healthy, no plaque

    # For 6-class fine-tuning:
    # Classes 0-2: non-significant stenosis (calcified/non-calcified/mixed)
    # Classes 3-5: significant stenosis (calcified/non-calcified/mixed)
    if num_classes == 6:
        # Stenosis degree
        if (object_classes >= 3).any():
            stenosis_pred = 2  # significant
        else:
            stenosis_pred = 1  # non-significant

        # Plaque composition: extract plaque type from class index
        # Class 0,3 = calcified; Class 1,4 = non-calcified; Class 2,5 = mixed
        plaque_types = object_classes % 3
        # Take majority vote or most confident prediction
        plaque_pred = plaque_types[0].item()  # Use first detection for simplicity

    # For 3-class pre-training (plaque composition only):
    # Class 0 = calcified, Class 1 = non-calcified, Class 2 = mixed
    elif num_classes == 3:
        stenosis_pred = 1  # all plaques are considered non-significant in pre-training
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
        if (labels >= 3).any():
            stenosis_gt = 2  # significant
        else:
            stenosis_gt = 1  # non-significant
        plaque_types = labels % 3
        plaque_gt = plaque_types[0].item()
    elif num_classes == 3:
        stenosis_gt = 1
        plaque_gt = labels[0].item()
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")

    return stenosis_gt, plaque_gt


@torch.no_grad()
def evaluate(model, test_loader, device, num_classes):
    """Run evaluation on test set."""
    model.eval()

    all_stenosis_preds = []
    all_stenosis_gts = []
    all_plaque_preds = []
    all_plaque_gts = []

    print("\nRunning inference on test set...")
    for batch_idx, (images, targets) in enumerate(test_loader):
        images = images.to(device)

        # Forward pass
        od_outputs, sc_outputs = model(images)

        # Process each sample in the batch
        batch_size = images.shape[0]
        for i in range(batch_size):
            # Extract single-sample outputs
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

            # Only include plaque predictions for samples with plaques
            if plaque_pred != -1 and plaque_gt != -1:
                all_plaque_preds.append(plaque_pred)
                all_plaque_gts.append(plaque_gt)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

    print(f"Evaluation complete: {len(all_stenosis_gts)} samples\n")

    # Compute metrics
    stenosis_metrics = compute_metrics(all_stenosis_gts, all_stenosis_preds, num_classes=3)

    if len(all_plaque_preds) > 0:
        plaque_metrics = compute_metrics(all_plaque_gts, all_plaque_preds, num_classes=3)
    else:
        print("Warning: No plaque samples found in test set")
        plaque_metrics = {'acc': 0, 'prec': 0, 'recall': 0, 'f1': 0, 'spec': 0}

    return stenosis_metrics, plaque_metrics


def print_results(stenosis_metrics, plaque_metrics):
    """Print evaluation results in paper format."""
    print("=" * 70)
    print("SC-Net Evaluation Results (Artery-Level)")
    print("=" * 70)
    print()
    print("Stenosis Degree Classification (Healthy / Non-significant / Significant):")
    print(f"  ACC:        {stenosis_metrics['acc']:.3f}")
    print(f"  Precision:  {stenosis_metrics['prec']:.3f}")
    print(f"  Recall:     {stenosis_metrics['recall']:.3f}")
    print(f"  F1:         {stenosis_metrics['f1']:.3f}")
    print(f"  Specificity:{stenosis_metrics['spec']:.3f}")
    print()
    print("Plaque Composition Classification (Calcified / Non-calcified / Mixed):")
    print(f"  ACC:        {plaque_metrics['acc']:.3f}")
    print(f"  Precision:  {plaque_metrics['prec']:.3f}")
    print(f"  Recall:     {plaque_metrics['recall']:.3f}")
    print(f"  F1:         {plaque_metrics['f1']:.3f}")
    print(f"  Specificity:{plaque_metrics['spec']:.3f}")
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
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        fw.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch {epoch}")
    else:
        fw.model.load_state_dict(checkpoint)
        print("Loaded model weights")

    model = fw.model.to(device)
    test_loader = fw.dataLoader_test

    print(f"Test set: {len(test_loader)} batches")

    # Run evaluation
    stenosis_metrics, plaque_metrics = evaluate(model, test_loader, device, num_classes)

    # Print results
    print_results(stenosis_metrics, plaque_metrics)


if __name__ == '__main__':
    main()
