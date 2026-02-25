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
  python eval.py --checkpoint ./checkpoints/best_model.pth --pattern fine_tuning --tta --tta_k 5
  python eval.py --checkpoint ./checkpoints/best_model.pth --pattern fine_tuning --detailed
  python eval.py --checkpoint ./checkpoints/best_model.pth --pattern fine_tuning --detailed --save_results results.json
  python eval.py --checkpoint ./checkpoints/best_model.pth --pattern fine_tuning --ensemble ckpt1.pth ckpt2.pth ckpt3.pth
  python eval.py --checkpoint ./checkpoints/best_model.pth --pattern fine_tuning --ensemble ./checkpoints/checkpoint_epoch_*.pth --tta --detailed
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from framework import sc_net_framework
import augmentation as aug
from config import opt

# Class name constants
STENOSIS_CLASSES = ["Healthy", "Non-significant", "Significant"]
PLAQUE_CLASSES = ["Calcified", "Non-calcified", "Mixed"]


def parse_args():
    parser = argparse.ArgumentParser(description='SC-Net Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (required unless --ensemble is used)')
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
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation')
    parser.add_argument('--tta_k', type=int, default=5,
                        help='Number of augmented versions for TTA (default: 5)')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed output: confusion matrices, per-class metrics, and AUC-ROC')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Save metrics to a JSON file (e.g., --save_results results.json)')
    parser.add_argument('--ensemble', type=str, nargs='+', default=None,
                        help='Ensemble inference: one or more checkpoint paths '
                             '(e.g., --ensemble ckpt1.pth ckpt2.pth ckpt3.pth)')
    return parser.parse_args()


def get_device(device_str):
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def print_confusion_matrix(gt_list, pred_list, class_names):
    """Print a formatted confusion matrix with row/column labels.

    Rows are ground truth, columns are predictions.
    """
    n = len(class_names)
    matrix = [[0] * n for _ in range(n)]
    for gt, pred in zip(gt_list, pred_list):
        if 0 <= gt < n and 0 <= pred < n:
            matrix[gt][pred] += 1

    # Determine column width for alignment
    max_name_len = max(len(name) for name in class_names)
    col_width = max(max_name_len, 6) + 2

    # Header row
    header = " " * (max_name_len + 4)
    for name in class_names:
        header += f"{name:>{col_width}}"
    print(header)
    print(" " * (max_name_len + 4) + "-" * (col_width * n))

    # Data rows
    for i, row_name in enumerate(class_names):
        row_str = f"  {row_name:<{max_name_len + 2}}"
        for j in range(n):
            row_str += f"{matrix[i][j]:>{col_width}}"
        row_str += f"  | {sum(matrix[i])}"
        print(row_str)

    # Column totals
    totals_str = " " * (max_name_len + 4)
    for j in range(n):
        col_total = sum(matrix[i][j] for i in range(n))
        totals_str += f"{col_total:>{col_width}}"
    print(totals_str)
    print()


def compute_per_class_metrics(gt_list, pred_list, class_names):
    """Compute per-class precision, recall, F1, and support.

    Returns a list of dicts, one per class.
    """
    n = len(class_names)
    # Build confusion matrix
    matrix = [[0] * n for _ in range(n)]
    for gt, pred in zip(gt_list, pred_list):
        if 0 <= gt < n and 0 <= pred < n:
            matrix[gt][pred] += 1

    per_class = []
    for i in range(n):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n)) - tp
        fn = sum(matrix[i][j] for j in range(n)) - tp
        support = sum(matrix[i][j] for j in range(n))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class.append({
            'class': class_names[i],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        })

    return per_class


def print_per_class_metrics(per_class_metrics):
    """Print a formatted per-class metrics table."""
    header = f"  {'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    print(header)
    print("  " + "-" * 58)
    for m in per_class_metrics:
        print(f"  {m['class']:<18} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>10.3f} {m['support']:>10}")
    print()


def compute_auc_ovr(gt_list, prob_array, class_names):
    """Compute one-vs-rest AUC-ROC for each class.

    Uses sklearn if available, otherwise falls back to a simple trapezoidal
    implementation.

    Args:
        gt_list: list of int ground truth labels
        prob_array: np.ndarray of shape [N, num_classes] with softmax probabilities
        class_names: list of class name strings

    Returns:
        dict mapping class name to AUC value (or None if not computable)
    """
    n_classes = len(class_names)
    gt_array = np.array(gt_list)
    auc_results = {}

    try:
        from sklearn.metrics import roc_auc_score
        for i, name in enumerate(class_names):
            binary_gt = (gt_array == i).astype(int)
            if binary_gt.sum() == 0 or binary_gt.sum() == len(binary_gt):
                auc_results[name] = None  # Only one class present
            else:
                auc_results[name] = roc_auc_score(binary_gt, prob_array[:, i])
    except ImportError:
        # Fallback: simple trapezoidal AUC
        for i, name in enumerate(class_names):
            binary_gt = (gt_array == i).astype(int)
            scores = prob_array[:, i]
            if binary_gt.sum() == 0 or binary_gt.sum() == len(binary_gt):
                auc_results[name] = None
            else:
                auc_results[name] = _trapezoidal_auc(binary_gt, scores)

    return auc_results


def _trapezoidal_auc(labels, scores):
    """Compute AUC via sorting and trapezoidal integration.

    Args:
        labels: binary np.ndarray (1 = positive, 0 = negative)
        scores: np.ndarray of predicted scores for the positive class

    Returns:
        float AUC value
    """
    # Sort by score descending
    desc_idx = np.argsort(-scores)
    labels_sorted = labels[desc_idx]

    num_pos = labels.sum()
    num_neg = len(labels) - num_pos
    if num_pos == 0 or num_neg == 0:
        return None

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / num_pos
        fpr = fp / num_neg
        # Trapezoidal rule
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr

    return auc


def tta_augment(volume, tta_k):
    """Generate TTA-augmented versions of a volume tensor.

    All transforms are normalization-invariant (work on already-normalized data).

    Args:
        volume: tensor of shape [D, H, W] (single sample, already normalized)
        tta_k: maximum number of augmented versions (not counting original)

    Returns:
        augmented: list of (augmented_volume, needs_depth_flip_reversal) tuples.
            The original is always first with needs_depth_flip_reversal=False.
            Length is min(tta_k, num_available_transforms) + 1.
    """
    augmented = [(volume, False)]  # original, no flip reversal needed

    # Define the pool of transforms: (transform_fn, is_depth_flip)
    transforms = [
        # 1. Depth flip — reverse along dim 0 (the 256-length axis)
        (lambda v: torch.flip(v, dims=[0]), True),
        # 2. Intensity scale +5% (multiplicative, normalization-invariant)
        (lambda v: v * 1.05, False),
        # 3. Intensity scale -5% (multiplicative, normalization-invariant)
        (lambda v: v * 0.95, False),
        # 4. Intensity shift +0.02 (small additive shift in normalized space)
        (lambda v: v + 0.02, False),
        # 5. Intensity shift -0.02
        (lambda v: v - 0.02, False),
    ]

    # Extended transforms if tta_k > 5
    if tta_k > 5:
        transforms.extend([
            # 6. Intensity scale +2%
            (lambda v: v * 1.02, False),
            # 7. Intensity scale -2%
            (lambda v: v * 0.98, False),
            # 8. Combined depth flip + intensity scale +5%
            (lambda v: torch.flip(v * 1.05, dims=[0]), True),
            # 9. Intensity shift +0.05
            (lambda v: v + 0.05, False),
            # 10. Intensity shift -0.05
            (lambda v: v - 0.05, False),
        ])

    # Take up to tta_k transforms
    for i, (transform_fn, is_flip) in enumerate(transforms):
        if i >= tta_k:
            break
        augmented.append((transform_fn(volume), is_flip))

    return augmented


def tta_forward(model, image, tta_k, device):
    """Run TTA inference on a single sample.

    Averages softmax probabilities for classification logits across
    the original and K augmented versions. Box predictions come from
    the original (unaugmented) forward pass only.

    Args:
        model: SC-Net model (in eval mode)
        image: tensor of shape [D, H, W] (single sample)
        tta_k: number of augmented versions
        device: torch device

    Returns:
        od_outputs_i: dict with averaged 'pred_logits' [num_queries, C] and
                       original 'pred_boxes' [num_queries, 2]
        sc_outputs_i: dict with averaged 'pred_logits' [seq_len, C] or None
    """
    augmented_pairs = tta_augment(image, tta_k)

    od_probs_list = []
    sc_probs_list = []
    od_boxes_original = None
    sc_outputs_available = False

    for aug_idx, (aug_volume, is_depth_flip) in enumerate(augmented_pairs):
        # Run single-sample batch through the model
        inp = aug_volume.unsqueeze(0).to(device)  # [1, D, H, W]
        od_out, sc_out = model(inp)

        # OD branch: softmax over class logits
        od_logits = od_out['pred_logits'][0]  # [num_queries, num_classes+1]
        od_probs = F.softmax(od_logits, dim=-1)
        od_probs_list.append(od_probs)

        # Keep boxes from the original (unaugmented) pass only
        if aug_idx == 0:
            od_boxes_original = od_out['pred_boxes'][0]  # [num_queries, 2]

        # SC branch
        if sc_out is not None and 'pred_logits' in sc_out:
            sc_outputs_available = True
            sc_logits = sc_out['pred_logits'][0]  # [seq_len, num_classes+1]
            sc_probs = F.softmax(sc_logits, dim=-1)
            # For depth-flipped augmentations, flip SC logits back along
            # the sequence dimension before averaging
            if is_depth_flip:
                sc_probs = torch.flip(sc_probs, dims=[0])
            sc_probs_list.append(sc_probs)

    # Average OD probabilities — convert back to log-space so downstream
    # code that calls softmax again still produces the same ranking
    # (softmax of log(p) = p). We store as log-probs.
    avg_od_probs = torch.stack(od_probs_list, dim=0).mean(dim=0)
    # Use log to convert averaged probs back to logit-like values.
    # Clamp to avoid log(0).
    avg_od_logits = torch.log(avg_od_probs.clamp(min=1e-8))

    od_outputs_i = {
        'pred_logits': avg_od_logits,
        'pred_boxes': od_boxes_original,
    }

    sc_outputs_i = None
    if sc_outputs_available and len(sc_probs_list) > 0:
        avg_sc_probs = torch.stack(sc_probs_list, dim=0).mean(dim=0)
        avg_sc_logits = torch.log(avg_sc_probs.clamp(min=1e-8))
        sc_outputs_i = {'pred_logits': avg_sc_logits}

    return od_outputs_i, sc_outputs_i


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


def _collect_artery_probs(od_out_i, num_classes, stenosis_gt, plaque_gt,
                          stenosis_probs_list, plaque_probs_list):
    """Collect artery-level softmax probabilities for AUC-ROC computation.

    Derives 3-class stenosis probabilities and 3-class plaque probabilities
    from the raw per-query OD logits. Uses the max object-class probability
    across queries as a proxy for artery-level confidence.

    Args:
        od_out_i: dict with 'pred_logits' [num_queries, num_classes+1]
        num_classes: number of object classes (3 or 6)
        stenosis_gt: int ground truth stenosis class
        plaque_gt: int ground truth plaque class (-1 if none)
        stenosis_probs_list: list to append [3] stenosis probabilities to
        plaque_probs_list: list to append [3] plaque probabilities to
    """
    pred_logits = od_out_i['pred_logits']
    pred_probs = F.softmax(pred_logits, dim=-1)  # [num_queries, num_classes+1]

    # Max probability that any query is an object (vs no-object)
    obj_probs = pred_probs[:, :num_classes]  # [num_queries, num_classes]
    no_obj_probs = pred_probs[:, num_classes]  # [num_queries]

    # Best query: highest total object probability
    total_obj_per_query = obj_probs.sum(dim=-1)  # [num_queries]
    best_q = total_obj_per_query.argmax()

    # Stenosis: derive 3-class probs [healthy, non-significant, significant]
    # healthy ~ no-object probability of best query
    p_healthy = no_obj_probs[best_q].item()

    if num_classes == 6:
        # classes 0-1 -> non-significant, classes 2-5 -> significant
        p_nonsig = obj_probs[best_q, :2].sum().item()
        p_sig = obj_probs[best_q, 2:].sum().item()
    elif num_classes == 3:
        # In pre-training mode all detections are non-significant
        p_nonsig = obj_probs[best_q].sum().item()
        p_sig = 0.0
    else:
        p_nonsig = 0.0
        p_sig = 0.0

    total = p_healthy + p_nonsig + p_sig
    if total > 0:
        stenosis_probs_list.append([p_healthy / total, p_nonsig / total, p_sig / total])
    else:
        stenosis_probs_list.append([1.0 / 3, 1.0 / 3, 1.0 / 3])

    # Plaque: derive 3-class probs [calcified, non-calcified, mixed]
    # Only collect for samples that have a plaque GT
    if plaque_gt != -1:
        if num_classes == 6:
            # classes 0-1 -> group 0 (calcified), 2-3 -> group 1, 4-5 -> group 2
            p_calc = obj_probs[best_q, 0:2].sum().item()
            p_noncalc = obj_probs[best_q, 2:4].sum().item()
            p_mixed = obj_probs[best_q, 4:6].sum().item()
        elif num_classes == 3:
            p_calc = obj_probs[best_q, 0].item()
            p_noncalc = obj_probs[best_q, 1].item()
            p_mixed = obj_probs[best_q, 2].item()
        else:
            p_calc = p_noncalc = p_mixed = 1.0 / 3

        plaque_total = p_calc + p_noncalc + p_mixed
        if plaque_total > 0:
            plaque_probs_list.append([p_calc / plaque_total, p_noncalc / plaque_total,
                                      p_mixed / plaque_total])
        else:
            plaque_probs_list.append([1.0 / 3, 1.0 / 3, 1.0 / 3])


@torch.no_grad()
def evaluate(model, test_loader, device, num_classes, eval_sc=False,
             tta=False, tta_k=5, detailed=False):
    """Run evaluation on test set.

    Args:
        model: SC-Net model
        test_loader: DataLoader for test data
        device: torch device
        num_classes: number of object classes (3 or 6)
        eval_sc: if True, also evaluate sampling point classification branch
        tta: if True, enable test-time augmentation
        tta_k: number of augmented versions for TTA (default 5)
        detailed: if True, also collect softmax probabilities for AUC-ROC

    Returns:
        stenosis_metrics: dict of stenosis classification metrics
        plaque_metrics: dict of plaque classification metrics
        sc_metrics: dict of sampling point classification metrics (or None if eval_sc=False)
        detailed_data: dict with raw lists/probs (or None if detailed=False)
    """
    model.eval()

    all_stenosis_preds = []
    all_stenosis_gts = []
    all_plaque_preds = []
    all_plaque_gts = []

    # Softmax probability collection for AUC-ROC (only when detailed=True)
    all_stenosis_probs = [] if detailed else None
    all_plaque_probs = [] if detailed else None

    # Sampling point classification tracking
    sc_correct = 0
    sc_total = 0

    if tta:
        print(f"\nRunning inference on test set with TTA (K={tta_k})...")
    else:
        print("\nRunning inference on test set...")

    for batch_idx, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        batch_size = images.shape[0]

        if tta:
            # TTA path: process each sample individually
            for i in range(batch_size):
                image_i = images[i]   # [D, H, W]
                target_i = targets[i]

                od_out_i, sc_out_i = tta_forward(model, image_i, tta_k, device)

                # Convert to artery-level predictions
                stenosis_pred, plaque_pred = od_predictions_to_artery_level(
                    od_out_i, num_classes)
                stenosis_gt, plaque_gt = targets_to_artery_level(
                    target_i, num_classes)

                all_stenosis_preds.append(stenosis_pred)
                all_stenosis_gts.append(stenosis_gt)

                # Collect softmax probabilities for AUC-ROC
                if detailed:
                    _collect_artery_probs(
                        od_out_i, num_classes, stenosis_gt, plaque_gt,
                        all_stenosis_probs, all_plaque_probs)

                if plaque_gt != -1:
                    effective_plaque_pred = plaque_pred if plaque_pred != -1 else num_classes
                    all_plaque_preds.append(effective_plaque_pred)
                    all_plaque_gts.append(plaque_gt)

                # SC branch
                if eval_sc and sc_out_i is not None and 'pred_logits' in sc_out_i:
                    sc_logits_i = sc_out_i['pred_logits']  # [seq_length, num_classes+1]
                    sc_preds_i = sc_logits_i.argmax(dim=-1)
                    seq_length = sc_logits_i.shape[0]
                    sc_gt_i = _build_sc_point_labels(target_i, seq_length).to(device)
                    sc_correct += (sc_preds_i == sc_gt_i).sum().item()
                    sc_total += seq_length
        else:
            # Standard (non-TTA) path
            od_outputs, sc_outputs = model(images)

            for i in range(batch_size):
                od_out_i = {
                    'pred_logits': od_outputs['pred_logits'][i],
                    'pred_boxes': od_outputs['pred_boxes'][i]
                }
                target_i = targets[i]

                stenosis_pred, plaque_pred = od_predictions_to_artery_level(
                    od_out_i, num_classes)
                stenosis_gt, plaque_gt = targets_to_artery_level(
                    target_i, num_classes)

                all_stenosis_preds.append(stenosis_pred)
                all_stenosis_gts.append(stenosis_gt)

                # Collect softmax probabilities for AUC-ROC
                if detailed:
                    _collect_artery_probs(
                        od_out_i, num_classes, stenosis_gt, plaque_gt,
                        all_stenosis_probs, all_plaque_probs)

                if plaque_gt != -1:
                    effective_plaque_pred = plaque_pred if plaque_pred != -1 else num_classes
                    all_plaque_preds.append(effective_plaque_pred)
                    all_plaque_gts.append(plaque_gt)
                elif plaque_pred != -1 and plaque_gt == -1:
                    pass

                if eval_sc and sc_outputs is not None and 'pred_logits' in sc_outputs:
                    sc_logits_i = sc_outputs['pred_logits'][i]
                    sc_preds_i = sc_logits_i.argmax(dim=-1)
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

    # Build detailed data dict if requested
    detailed_data = None
    if detailed:
        detailed_data = {
            'stenosis_gts': all_stenosis_gts,
            'stenosis_preds': all_stenosis_preds,
            'stenosis_probs': all_stenosis_probs,
            'plaque_gts': all_plaque_gts,
            'plaque_preds': all_plaque_preds,
            'plaque_probs': all_plaque_probs,
        }

    return stenosis_metrics, plaque_metrics, sc_metrics, detailed_data


def print_results(stenosis_metrics, plaque_metrics, sc_metrics=None,
                   detailed_data=None):
    """Print evaluation results in paper format.

    Args:
        stenosis_metrics: dict of macro-averaged stenosis metrics
        plaque_metrics: dict of macro-averaged plaque metrics
        sc_metrics: dict of sampling point classification metrics (or None)
        detailed_data: dict with raw lists/probs from evaluate() (or None)
    """
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

    if detailed_data is not None:
        # Per-class metrics for stenosis
        print()
        print("  Per-class metrics:")
        stenosis_pc = compute_per_class_metrics(
            detailed_data['stenosis_gts'], detailed_data['stenosis_preds'],
            STENOSIS_CLASSES)
        print_per_class_metrics(stenosis_pc)

        # Confusion matrix for stenosis
        print("  Confusion Matrix (rows=GT, cols=Pred):")
        print_confusion_matrix(
            detailed_data['stenosis_gts'], detailed_data['stenosis_preds'],
            STENOSIS_CLASSES)

    print()
    print("Plaque Composition Classification (Calcified / Non-calcified / Mixed):")
    print(f"  ACC:         {plaque_metrics['acc']:.3f}")
    print(f"  Precision:   {plaque_metrics['prec']:.3f}")
    print(f"  Recall:      {plaque_metrics['recall']:.3f}")
    print(f"  F1:          {plaque_metrics['f1']:.3f}")
    print(f"  Specificity: {plaque_metrics['spec']:.3f}")

    if detailed_data is not None and len(detailed_data['plaque_gts']) > 0:
        # Per-class metrics for plaque
        print()
        print("  Per-class metrics:")
        plaque_pc = compute_per_class_metrics(
            detailed_data['plaque_gts'], detailed_data['plaque_preds'],
            PLAQUE_CLASSES)
        print_per_class_metrics(plaque_pc)

        # Confusion matrix for plaque
        print("  Confusion Matrix (rows=GT, cols=Pred):")
        print_confusion_matrix(
            detailed_data['plaque_gts'], detailed_data['plaque_preds'],
            PLAQUE_CLASSES)

    if sc_metrics is not None:
        print()
        print("Sampling Point Classification:")
        print(f"  ACC:         {sc_metrics['acc']:.3f}")
        print(f"  Points:      {sc_metrics['correct_points']}/{sc_metrics['total_points']}")

    # AUC-ROC (only when detailed and probabilities are available)
    if detailed_data is not None:
        print()
        print("-" * 70)
        print("AUC-ROC (One-vs-Rest)")
        print("-" * 70)

        if detailed_data['stenosis_probs'] and len(detailed_data['stenosis_probs']) > 0:
            stenosis_prob_array = np.array(detailed_data['stenosis_probs'])
            stenosis_auc = compute_auc_ovr(
                detailed_data['stenosis_gts'], stenosis_prob_array, STENOSIS_CLASSES)
            print()
            print("  Stenosis:")
            valid_aucs = []
            for name in STENOSIS_CLASSES:
                auc_val = stenosis_auc[name]
                if auc_val is not None:
                    print(f"    {name:<20} AUC = {auc_val:.3f}")
                    valid_aucs.append(auc_val)
                else:
                    print(f"    {name:<20} AUC = N/A (single-class)")
            if valid_aucs:
                print(f"    {'Macro-average':<20} AUC = {np.mean(valid_aucs):.3f}")
        else:
            print("\n  Stenosis: No probability data collected.")

        if detailed_data['plaque_probs'] and len(detailed_data['plaque_probs']) > 0:
            plaque_prob_array = np.array(detailed_data['plaque_probs'])
            plaque_auc = compute_auc_ovr(
                detailed_data['plaque_gts'], plaque_prob_array, PLAQUE_CLASSES)
            print()
            print("  Plaque:")
            valid_aucs = []
            for name in PLAQUE_CLASSES:
                auc_val = plaque_auc[name]
                if auc_val is not None:
                    print(f"    {name:<20} AUC = {auc_val:.3f}")
                    valid_aucs.append(auc_val)
                else:
                    print(f"    {name:<20} AUC = N/A (single-class)")
            if valid_aucs:
                print(f"    {'Macro-average':<20} AUC = {np.mean(valid_aucs):.3f}")
        else:
            print("\n  Plaque: No probability data collected.")

    print()
    print("=" * 70)


def _resolve_ensemble_paths(paths):
    """Resolve ensemble checkpoint paths, expanding glob patterns.

    Args:
        paths: list of file paths or glob patterns

    Returns:
        list of resolved file paths (sorted for reproducibility)
    """
    import glob as glob_module
    resolved = []
    for p in paths:
        expanded = glob_module.glob(p)
        if expanded:
            resolved.extend(expanded)
        else:
            # Not a glob, treat as literal path
            resolved.append(p)
    resolved = sorted(set(resolved))
    return resolved


def _load_model_from_checkpoint(checkpoint_path, pattern, device, data_root=None):
    """Load a single model from a checkpoint file.

    Args:
        checkpoint_path: path to checkpoint .pth file
        pattern: 'pre_training' or 'fine_tuning'
        device: torch device
        data_root: optional data root override

    Returns:
        model: loaded model on device in eval mode
    """
    fw = sc_net_framework(
        pattern=pattern,
        state_dict_root=None,
        data_root=data_root,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        fw.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"  Loaded {checkpoint_path} (epoch {epoch})")
    else:
        fw.model.load_state_dict(checkpoint)
        print(f"  Loaded {checkpoint_path}")
    model = fw.model.to(device)
    model.eval()
    return model


def ensemble_forward(models, image, device, tta=False, tta_k=5):
    """Run ensemble inference on a single sample.

    Averages softmax probabilities across all models (and optionally TTA
    augmentations within each model). Box predictions come from the first
    model's original (unaugmented) pass.

    Args:
        models: list of SC-Net models (all in eval mode)
        image: tensor of shape [D, H, W] (single sample)
        device: torch device
        tta: whether to apply TTA within each model
        tta_k: number of TTA augmentations

    Returns:
        od_outputs_i: dict with averaged 'pred_logits' and first model's 'pred_boxes'
        sc_outputs_i: dict with averaged 'pred_logits' or None
    """
    od_probs_list = []
    sc_probs_list = []
    od_boxes_first = None
    sc_available = False

    for m_idx, model in enumerate(models):
        if tta:
            od_out_m, sc_out_m = tta_forward(model, image, tta_k, device)
        else:
            inp = image.unsqueeze(0).to(device)
            od_out_raw, sc_out_raw = model(inp)
            od_out_m = {
                'pred_logits': od_out_raw['pred_logits'][0],
                'pred_boxes': od_out_raw['pred_boxes'][0],
            }
            sc_out_m = None
            if sc_out_raw is not None and 'pred_logits' in sc_out_raw:
                sc_out_m = {'pred_logits': sc_out_raw['pred_logits'][0]}

        # Collect OD probabilities
        od_logits = od_out_m['pred_logits']
        od_probs = F.softmax(od_logits, dim=-1)
        od_probs_list.append(od_probs)

        # Keep boxes from first model only
        if m_idx == 0:
            od_boxes_first = od_out_m['pred_boxes']

        # Collect SC probabilities
        if sc_out_m is not None and 'pred_logits' in sc_out_m:
            sc_available = True
            sc_logits = sc_out_m['pred_logits']
            sc_probs = F.softmax(sc_logits, dim=-1)
            sc_probs_list.append(sc_probs)

    # Average OD probabilities across models, convert back to log-space
    avg_od_probs = torch.stack(od_probs_list, dim=0).mean(dim=0)
    avg_od_logits = torch.log(avg_od_probs.clamp(min=1e-8))

    od_outputs_i = {
        'pred_logits': avg_od_logits,
        'pred_boxes': od_boxes_first,
    }

    sc_outputs_i = None
    if sc_available and len(sc_probs_list) > 0:
        avg_sc_probs = torch.stack(sc_probs_list, dim=0).mean(dim=0)
        avg_sc_logits = torch.log(avg_sc_probs.clamp(min=1e-8))
        sc_outputs_i = {'pred_logits': avg_sc_logits}

    return od_outputs_i, sc_outputs_i


@torch.no_grad()
def evaluate_ensemble(models, test_loader, device, num_classes, eval_sc=False,
                      tta=False, tta_k=5, detailed=False):
    """Run ensemble evaluation on test set.

    Same interface as evaluate() but uses multiple models and averages their
    softmax probabilities per sample.

    Args:
        models: list of SC-Net models (all in eval mode)
        test_loader: DataLoader for test data
        device: torch device
        num_classes: number of object classes (3 or 6)
        eval_sc: if True, also evaluate sampling point classification branch
        tta: if True, enable TTA within each model
        tta_k: number of TTA augmentations per model
        detailed: if True, also collect softmax probabilities for AUC-ROC

    Returns:
        Same as evaluate(): (stenosis_metrics, plaque_metrics, sc_metrics, detailed_data)
    """
    for m in models:
        m.eval()

    all_stenosis_preds = []
    all_stenosis_gts = []
    all_plaque_preds = []
    all_plaque_gts = []

    all_stenosis_probs = [] if detailed else None
    all_plaque_probs = [] if detailed else None

    sc_correct = 0
    sc_total = 0

    mode_str = "ensemble"
    if tta:
        mode_str += f" + TTA (K={tta_k})"
    print(f"\nRunning inference on test set with {mode_str} ({len(models)} models)...")

    for batch_idx, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        batch_size = images.shape[0]

        for i in range(batch_size):
            image_i = images[i]  # [D, H, W]
            target_i = targets[i]

            od_out_i, sc_out_i = ensemble_forward(
                models, image_i, device, tta=tta, tta_k=tta_k)

            stenosis_pred, plaque_pred = od_predictions_to_artery_level(
                od_out_i, num_classes)
            stenosis_gt, plaque_gt = targets_to_artery_level(
                target_i, num_classes)

            all_stenosis_preds.append(stenosis_pred)
            all_stenosis_gts.append(stenosis_gt)

            if detailed:
                _collect_artery_probs(
                    od_out_i, num_classes, stenosis_gt, plaque_gt,
                    all_stenosis_probs, all_plaque_probs)

            if plaque_gt != -1:
                effective_plaque_pred = plaque_pred if plaque_pred != -1 else num_classes
                all_plaque_preds.append(effective_plaque_pred)
                all_plaque_gts.append(plaque_gt)

            if eval_sc and sc_out_i is not None and 'pred_logits' in sc_out_i:
                sc_logits_i = sc_out_i['pred_logits']
                sc_preds_i = sc_logits_i.argmax(dim=-1)
                seq_length = sc_logits_i.shape[0]
                sc_gt_i = _build_sc_point_labels(target_i, seq_length).to(device)
                sc_correct += (sc_preds_i == sc_gt_i).sum().item()
                sc_total += seq_length

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

    print(f"Evaluation complete: {len(all_stenosis_gts)} samples\n")

    # Compute metrics (same logic as evaluate())
    stenosis_metrics = compute_metrics(all_stenosis_gts, all_stenosis_preds, num_classes=3)

    if len(all_plaque_preds) > 0:
        plaque_num_classes = 3
        if any(p >= 3 for p in all_plaque_preds):
            plaque_num_classes = max(max(all_plaque_preds), max(all_plaque_gts)) + 1
        plaque_metrics = compute_metrics(all_plaque_gts, all_plaque_preds,
                                         num_classes=plaque_num_classes)
    else:
        print("Warning: No plaque samples found in ground truth of the test set.")
        plaque_metrics = {'acc': 0, 'prec': 0, 'recall': 0, 'f1': 0, 'spec': 0}

    sc_metrics = None
    if eval_sc:
        if sc_total > 0:
            sc_acc = sc_correct / sc_total
            sc_metrics = {'acc': sc_acc, 'total_points': sc_total, 'correct_points': sc_correct}
        else:
            print("Warning: No sampling points evaluated (sc_total=0).")
            sc_metrics = {'acc': 0.0, 'total_points': 0, 'correct_points': 0}

    detailed_data = None
    if detailed:
        detailed_data = {
            'stenosis_gts': all_stenosis_gts,
            'stenosis_preds': all_stenosis_preds,
            'stenosis_probs': all_stenosis_probs,
            'plaque_gts': all_plaque_gts,
            'plaque_preds': all_plaque_preds,
            'plaque_probs': all_plaque_probs,
        }

    return stenosis_metrics, plaque_metrics, sc_metrics, detailed_data


def _build_results_dict(stenosis_metrics, plaque_metrics, sc_metrics,
                        detailed_data, args):
    """Build a JSON-serializable results dictionary.

    Args:
        stenosis_metrics: dict of stenosis metrics
        plaque_metrics: dict of plaque metrics
        sc_metrics: dict of SC metrics (or None)
        detailed_data: dict with raw lists/probs (or None)
        args: parsed CLI arguments

    Returns:
        dict suitable for json.dump()
    """
    from datetime import datetime

    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint if not args.ensemble else args.ensemble,
        'pattern': args.pattern,
        'tta': args.tta,
        'tta_k': args.tta_k if args.tta else None,
        'ensemble': args.ensemble is not None,
        'num_ensemble_models': len(args.ensemble) if args.ensemble else 1,
        'stenosis_metrics': {k: float(v) for k, v in stenosis_metrics.items()},
        'plaque_metrics': {k: float(v) for k, v in plaque_metrics.items()},
    }

    if sc_metrics is not None:
        results['sc_metrics'] = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                                 for k, v in sc_metrics.items()}

    if detailed_data is not None:
        # Per-class metrics
        stenosis_pc = compute_per_class_metrics(
            detailed_data['stenosis_gts'], detailed_data['stenosis_preds'],
            STENOSIS_CLASSES)
        results['stenosis_per_class'] = stenosis_pc

        if len(detailed_data['plaque_gts']) > 0:
            plaque_pc = compute_per_class_metrics(
                detailed_data['plaque_gts'], detailed_data['plaque_preds'],
                PLAQUE_CLASSES)
            results['plaque_per_class'] = plaque_pc

        # AUC-ROC
        if detailed_data['stenosis_probs'] and len(detailed_data['stenosis_probs']) > 0:
            stenosis_prob_array = np.array(detailed_data['stenosis_probs'])
            stenosis_auc = compute_auc_ovr(
                detailed_data['stenosis_gts'], stenosis_prob_array, STENOSIS_CLASSES)
            results['stenosis_auc'] = {k: float(v) if v is not None else None
                                        for k, v in stenosis_auc.items()}

        if detailed_data['plaque_probs'] and len(detailed_data['plaque_probs']) > 0:
            plaque_prob_array = np.array(detailed_data['plaque_probs'])
            plaque_auc = compute_auc_ovr(
                detailed_data['plaque_gts'], plaque_prob_array, PLAQUE_CLASSES)
            results['plaque_auc'] = {k: float(v) if v is not None else None
                                      for k, v in plaque_auc.items()}

        # Sample counts
        results['num_stenosis_samples'] = len(detailed_data['stenosis_gts'])
        results['num_plaque_samples'] = len(detailed_data['plaque_gts'])

    return results


def main():
    args = parse_args()

    # Validate: need at least --checkpoint or --ensemble
    if not args.checkpoint and not args.ensemble:
        print("Error: Must provide --checkpoint or --ensemble.")
        return

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Determine num_classes based on pattern
    num_classes = 3 if args.pattern == 'pre_training' else 6
    print(f"Number of classes: {num_classes}")

    # --- Ensemble mode ---
    if args.ensemble:
        checkpoint_paths = _resolve_ensemble_paths(args.ensemble)
        if len(checkpoint_paths) == 0:
            print("Error: No checkpoint files found for --ensemble paths.")
            return
        print(f"\nEnsemble mode: loading {len(checkpoint_paths)} models...")
        models = []
        for ckpt_path in checkpoint_paths:
            model = _load_model_from_checkpoint(
                ckpt_path, args.pattern, device, data_root=args.data_root)
            models.append(model)

        # Build test loader from first model's framework (for dataset access)
        fw = sc_net_framework(
            pattern=args.pattern,
            state_dict_root=None,
            data_root=args.data_root,
        )
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

        stenosis_metrics, plaque_metrics, sc_metrics, detailed_data = evaluate_ensemble(
            models, test_loader, device, num_classes, eval_sc=args.eval_sc,
            tta=args.tta, tta_k=args.tta_k, detailed=args.detailed
        )

    # --- Single model mode ---
    else:
        print(f"\nLoading model in {args.pattern} mode...")
        fw = sc_net_framework(
            pattern=args.pattern,
            state_dict_root=None,
            data_root=args.data_root,
        )

        # Load checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            fw.model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded model from epoch {epoch}")
        else:
            fw.model.load_state_dict(checkpoint)
            print("Loaded model weights")

        model = fw.model.to(device)

        # Build test loader
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

        stenosis_metrics, plaque_metrics, sc_metrics, detailed_data = evaluate(
            model, test_loader, device, num_classes, eval_sc=args.eval_sc,
            tta=args.tta, tta_k=args.tta_k, detailed=args.detailed
        )

    # Print results
    print_results(stenosis_metrics, plaque_metrics, sc_metrics, detailed_data)

    # Save results to JSON if requested
    if args.save_results:
        results = _build_results_dict(
            stenosis_metrics, plaque_metrics, sc_metrics, detailed_data, args)
        save_path = args.save_results
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {save_path}")


if __name__ == '__main__':
    main()
