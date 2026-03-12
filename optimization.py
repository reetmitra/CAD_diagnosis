import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

import functions as funcs


class object_detection_loss(nn.Module):
    def __init__(self, num_classes=2, eos_coef=0.2, matcher=funcs.HungarianMatcher()):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        empty_batch = False

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if target_classes_o.numel() != 0:
            target_classes_o = target_classes_o.to(device=src_logits.device, dtype=torch.long)
            target_classes[idx] = target_classes_o
        else:
            empty_batch = True

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return loss_ce, empty_batch

    def loss_boxes(self, outputs, targets, indices, num_boxes):

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_boxes.device)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(funcs.generalized_box_iou(funcs.box_cxcywh_to_xyxy(src_boxes),
                                                            funcs.box_cxcywh_to_xyxy(target_boxes)))
        # Paper Eq. 5: λ_L1=5, λ_iou=2
        return 5.0 * loss_bbox.sum() / num_boxes + 2.0 * loss_giou.sum() / num_boxes

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        batch_idx = batch_idx.to(dtype=torch.long)
        src_idx = src_idx.to(dtype=torch.long)
        return batch_idx, src_idx

    def forward(self, outputs, targets):

        # Expand 2D boxes [center, width] to 4D [cx, cy, w, h] for matcher/loss
        device = next(iter(outputs.values())).device
        outputs = {k: v.clone() for k, v in outputs.items()}
        targets = [{k: v.clone().to(device) for k, v in t.items()} for t in targets]
        if outputs['pred_boxes'].shape[-1] == 2:
            outputs = funcs.boxes_dimension_expansion(outputs, dtype='outputs')
            targets = funcs.boxes_dimension_expansion(targets, dtype='targets')

        indices = self.matcher(outputs, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / funcs.get_world_size(), min=1).item()

        loss_labels, empty_batch = self.loss_labels(outputs, targets, indices)

        if empty_batch == True:
            return loss_labels

        loss_boxes = self.loss_boxes(outputs, targets, indices, num_boxes)
        return loss_labels + loss_boxes


def compute_sc_class_weights(num_classes):
    """Return inverse-frequency inspired class weights for SC loss.

    Background (class 0) accounts for ~64% of sampling points, so it gets
    a lower weight (0.5) while all lesion classes share weight 1.5.

    Args:
        num_classes: Number of lesion classes (e.g. 3 for pre_training,
            6 for fine_tuning).  The returned tensor has length
            ``num_classes + 1`` (background class 0 included).

    Returns:
        Tensor of shape [num_classes + 1].
    """
    weights = torch.ones(num_classes + 1, dtype=torch.float32)
    weights[0] = 0.5       # background
    weights[1:] = 1.5      # all lesion classes
    return weights


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in classification tasks.

    Applies a modulating factor (1 - p_t)^gamma to the standard cross-entropy
    loss so that well-classified examples contribute less and hard, misclassified
    examples dominate the gradient.

    Args:
        alpha: Per-class weight tensor (same role as ``weight`` in CE).
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        reduction: ``'mean'`` or ``'sum'``.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class sampling_point_classification_loss(nn.Module):
    def __init__(self, num_classes=3, seq_length=32, class_weights=None,
                 use_focal=False, focal_gamma=2.0, label_smoothing=0.0):
        super().__init__()

        self.num_classes = num_classes
        self.seq_length = seq_length
        self.use_focal = use_focal
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        if self.use_focal:
            self.focal_loss_fn = FocalLoss(
                alpha=self.class_weights, gamma=focal_gamma, reduction='mean')

    def loss_labels(self, outputs, targets):
        if self.use_focal:
            return self.focal_loss_fn(outputs, targets)
        return F.cross_entropy(outputs, targets, weight=self.class_weights,
                               label_smoothing=self.label_smoothing)

    def loss_soft(self, outputs, soft_targets):
        """Compute KL-divergence loss against soft probability targets."""
        log_probs = F.log_softmax(outputs, dim=1)
        return F.kl_div(log_probs, soft_targets, reduction='batchmean')

    def forward(self, outputs, targets):

        logits = rearrange(outputs["pred_logits"], 'b l c -> (b l) c').to(torch.float32)
        labels = torch.cat([t["labels"] for t in targets], dim=0).to(device=logits.device, dtype=torch.long)

        return self.loss_labels(logits, labels)


def od2sc_targets(od_box_data, seq_length):

    sc_point_data, interval = [], 1 / (seq_length + 1)
    for box_data in od_box_data:
        point_data = torch.zeros(seq_length, dtype=torch.long)
        if box_data['boxes'].numel() == 0:
            sc_point_data += [{"labels": point_data}]
            continue
        centers = box_data['boxes'][:, 0]
        widths = box_data['boxes'][:, 1]
        x1 = centers - widths / 2.0
        x2 = centers + widths / 2.0
        starts = torch.round(x1 / interval).int()
        ends = torch.round(x2 / interval).int()
        starts = torch.clamp(starts, min=1, max=seq_length) - 1
        ends = torch.clamp(ends, min=1, max=seq_length) - 1
        for k in range(starts.shape[0]):
            point_data[starts[k]:ends[k] + 1] = box_data['labels'][k] + 1
        sc_point_data += [{"labels": point_data}]
    return sc_point_data


def sc2od_targets(sc_point_data, seq_length):

    od_box_data = []
    for point_data in sc_point_data:
        tmp_data = point_data['labels']

        boxes, labels = [], []
        start, label, length, last = None, 0, seq_length, -1

        for i in range(tmp_data.shape[0]):
            if start is not None:
                if tmp_data[i] != last:
                    x1 = (start + 1) / length
                    x2 = min((i + 1) / length, 1.0)
                    center = (x1 + x2) / 2.0
                    width = x2 - x1
                    boxes.append([center, width])
                    labels.append(label - 1)
                    if tmp_data[i] != 0:
                        start, label, last = i, tmp_data[i], tmp_data[i]
                    else:
                        start, label, last = None, 0, -1
                else:
                    continue
            else:
                if tmp_data[i] == 0:
                    start, label, last = None, 0, -1
                else:
                    start, label, last = i, tmp_data[i], tmp_data[i]

        if start is not None:
            x1 = (start + 1) / length
            x2 = 1.0
            center = (x1 + x2) / 2.0
            width = x2 - x1
            boxes.append([center, width])
            labels.append(label - 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 2), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        od_box_data.append({"labels": labels, "boxes": boxes})
    return od_box_data


class dual_task_contrastive_loss(nn.Module):
    def __init__(self, od_contrastive_loss, sc_contrastive_loss, seq_length,
                 confidence_threshold=0.0, use_soft_labels=False):
        super().__init__()

        self.od_contrastive_loss = od_contrastive_loss
        self.sc_contrastive_loss = sc_contrastive_loss
        self.matcher = self.od_contrastive_loss.matcher
        self.seq_length = seq_length
        self.confidence_threshold = confidence_threshold
        self.use_soft_labels = use_soft_labels

    def _get_object_detection_targets(self, sc_outputs):

        ret_sc_targets = []
        for batch in sc_outputs["pred_logits"]:
            probs = torch.softmax(batch, dim=1)
            max_probs, labels = torch.max(probs, dim=1)

            # Confidence gating: zero out low-confidence predictions
            if self.confidence_threshold > 0:
                labels = labels.clone()
                labels[max_probs < self.confidence_threshold] = 0  # treat as background

            ret_sc_targets.append({"labels": labels})
        return sc2od_targets(ret_sc_targets, self.seq_length)

    def _get_sampling_point_classification_targets(self, od_outputs, od_targets):

        od_outputs = funcs.boxes_dimension_expansion(
            {k: v.clone() for k, v in od_outputs.items()}, dtype='outputs')
        od_targets = funcs.boxes_dimension_expansion(
            [{k: v.clone() for k, v in t.items()} for t in od_targets], dtype='targets')
        indices = self.matcher(od_outputs, od_targets)
        selected_indices = [item[0] for item in indices]

        ret_od_targets = []
        for batch_idx, indices in enumerate(selected_indices):

            logits = od_outputs["pred_logits"][batch_idx]
            boxes = od_outputs["pred_boxes"][batch_idx]
            selected_logits = logits[indices]
            selected_boxes = boxes[indices][:, [0, 2]]

            # Confidence gating: only use high-confidence predictions
            probs = torch.softmax(selected_logits, dim=1)
            max_probs, pred_classes = torch.max(probs, dim=1)

            # Filter out no-object predictions (class index == num_classes)
            num_classes = selected_logits.shape[-1] - 1  # last class is no-object
            is_object = pred_classes < num_classes

            # Also filter by confidence threshold
            if self.confidence_threshold > 0:
                is_confident = max_probs >= self.confidence_threshold
                mask = is_object & is_confident
            else:
                mask = is_object

            filtered_labels = pred_classes[mask]
            filtered_boxes = selected_boxes[mask]

            ret_od_targets.append({"labels": filtered_labels, "boxes": filtered_boxes})

        return od2sc_targets(ret_od_targets, self.seq_length)

    def _compute_soft_sc_loss(self, sc_outputs, od_outputs, od_targets):
        """Compute SC contrastive loss using soft probability targets from OD.

        Instead of converting OD predictions to hard point labels, we create
        soft probability distributions for each sampling point based on OD
        prediction confidences. Uses KL divergence as the loss.
        """
        od_out = funcs.boxes_dimension_expansion(
            {k: v.clone() for k, v in od_outputs.items()}, dtype='outputs')
        od_tgt = funcs.boxes_dimension_expansion(
            [{k: v.clone() for k, v in t.items()} for t in od_targets], dtype='targets')
        indices = self.matcher(od_out, od_tgt)
        selected_indices = [item[0] for item in indices]

        batch_size = sc_outputs["pred_logits"].shape[0]
        num_sc_classes = sc_outputs["pred_logits"].shape[2]  # num_classes + 1 (includes bg)

        # Build soft targets for each point in the sequence
        soft_targets_list = []
        for batch_idx, sel_idx in enumerate(selected_indices):
            logits = od_out["pred_logits"][batch_idx]
            boxes = od_out["pred_boxes"][batch_idx]
            selected_logits = logits[sel_idx]
            selected_boxes = boxes[sel_idx][:, [0, 2]]  # [cx, w]

            # Start with uniform background distribution
            soft_target = torch.zeros(self.seq_length, num_sc_classes,
                                      device=logits.device, dtype=torch.float32)
            soft_target[:, 0] = 1.0  # default to background

            probs = torch.softmax(selected_logits, dim=1)
            num_od_classes = selected_logits.shape[-1] - 1

            interval = 1.0 / (self.seq_length + 1)
            for k in range(selected_boxes.shape[0]):
                cx, w = selected_boxes[k, 0], selected_boxes[k, 1]
                x1 = cx - w / 2.0
                x2 = cx + w / 2.0
                start = int(torch.clamp(torch.round(x1 / interval), min=1, max=self.seq_length).item()) - 1
                end = int(torch.clamp(torch.round(x2 / interval), min=1, max=self.seq_length).item()) - 1

                # Get class probabilities (exclude no-object class)
                class_probs = probs[k, :num_od_classes]  # (num_classes,)
                no_obj_prob = probs[k, num_od_classes]

                # For each point in the interval, set soft targets
                for p in range(start, end + 1):
                    if p < self.seq_length:
                        # Blend: background gets no_obj_prob, classes get their probs
                        soft_target[p, 0] = no_obj_prob
                        soft_target[p, 1:num_od_classes + 1] = class_probs

            # Normalize to valid probability distribution
            soft_target = soft_target / (soft_target.sum(dim=1, keepdim=True) + 1e-8)
            soft_targets_list.append(soft_target)

        # Stack and compute KL-div loss
        soft_targets = torch.stack(soft_targets_list, dim=0)  # (B, L, C)
        sc_logits = sc_outputs["pred_logits"].to(torch.float32)  # (B, L, C)

        # Reshape for KL-div
        sc_flat = rearrange(sc_logits, 'b l c -> (b l) c')
        soft_flat = rearrange(soft_targets, 'b l c -> (b l) c')

        return self.sc_contrastive_loss.loss_soft(sc_flat, soft_flat)

    def forward(self, od_outputs, sc_outputs, od_targets):

        od_detached = {k: v.detach() for k, v in od_outputs.items()}
        sc_detached = {k: v.detach() for k, v in sc_outputs.items()}

        # OD contrastive: always uses hard labels from SC predictions
        od_con_targets = self._get_object_detection_targets(sc_detached)
        od_loss_values = self.od_contrastive_loss(od_outputs, od_con_targets)

        # SC contrastive: soft or hard labels from OD predictions
        if self.use_soft_labels:
            sc_loss_values = self._compute_soft_sc_loss(
                sc_outputs, od_detached, od_targets)
        else:
            sc_con_targets = self._get_sampling_point_classification_targets(
                od_detached, od_targets)
            sc_loss_values = self.sc_contrastive_loss(sc_outputs, sc_con_targets)

        return sc_loss_values + od_loss_values


class spatio_temporal_contrast_loss(nn.Module):
    def __init__(self, num_classes=2, seq_length=32, eos_coef=0.2,
                 delta=1.0, sc_class_weights=None,
                 use_focal=False, focal_gamma=2.0,
                 dc_confidence_threshold=0.0,
                 label_smoothing=0.0, use_soft_dc=False):
        super().__init__()

        self.num_classes = num_classes
        self.seq_length = seq_length
        self.eos_coef = eos_coef
        self.delta = delta
        # dc_weight can be overridden per-epoch for delayed ramp
        self.dc_weight = delta

        self.od_loss = object_detection_loss(num_classes=self.num_classes, eos_coef=self.eos_coef,
                                             matcher=funcs.HungarianMatcher())
        self.sc_loss = sampling_point_classification_loss(
            num_classes=self.num_classes + 1, seq_length=self.seq_length,
            class_weights=sc_class_weights,
            use_focal=use_focal, focal_gamma=focal_gamma,
            label_smoothing=label_smoothing)
        self.dc_loss = dual_task_contrastive_loss(
            self.od_loss, self.sc_loss, seq_length=self.seq_length,
            confidence_threshold=dc_confidence_threshold,
            use_soft_labels=use_soft_dc)

    def set_dc_weight(self, weight):
        """Set the current dc loss weight (for delayed ramp scheduling)."""
        self.dc_weight = weight

    def forward(self, od_outputs, sc_outputs, od_targets):

        # Deep copy targets to prevent in-place mutation across loss terms
        od_targets_dc = [{k: v.clone() for k, v in t.items()} for t in od_targets]
        od_targets_od = [{k: v.clone() for k, v in t.items()} for t in od_targets]
        od_targets_sc = [{k: v.clone() for k, v in t.items()} for t in od_targets]

        dc_loss_val = self.dc_loss(od_outputs, sc_outputs, od_targets_dc) * self.dc_weight
        od_loss_val = self.od_loss(od_outputs, od_targets_od)
        sc_loss_val = self.sc_loss(sc_outputs, od2sc_targets(od_targets_sc, self.seq_length))
        total_loss = dc_loss_val + od_loss_val + sc_loss_val

        return {
            'total': total_loss,
            'od': od_loss_val,
            'sc': sc_loss_val,
            'dc': dc_loss_val
        }