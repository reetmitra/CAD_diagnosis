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
        # Convert [cx, w] → [start, end] for native 1D interval IoU
        src_se = funcs.box_cxw_to_se(src_boxes)
        tgt_se = funcs.box_cxw_to_se(target_boxes)
        loss_bbox = F.l1_loss(src_se, tgt_se, reduction='none')
        loss_giou = 1 - torch.diag(funcs.generalized_box_1d_iou(src_se, tgt_se))
        # Paper Eq. 5: λ_L1=5, λ_iou=2
        return 5.0 * loss_bbox.sum() / num_boxes + 2.0 * loss_giou.sum() / num_boxes

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        batch_idx = batch_idx.to(dtype=torch.long)
        src_idx = src_idx.to(dtype=torch.long)
        return batch_idx, src_idx

    def forward(self, outputs, targets):

        device = next(iter(outputs.values())).device
        outputs = {k: v.clone() for k, v in outputs.items()}
        targets = [{k: v.clone().to(device) for k, v in t.items()} for t in targets]

        indices = self.matcher(outputs, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / funcs.get_world_size(), min=1).item()

        loss_labels, empty_batch = self.loss_labels(outputs, targets, indices)

        if empty_batch == True:
            return loss_labels

        loss_boxes = self.loss_boxes(outputs, targets, indices, num_boxes)
        return loss_labels + loss_boxes


def compute_sc_class_weights(num_classes, boost_nonsig=False, nonsig_idx=2,
                              boost_sig=False, sig_multiplier=2.0):
    """Return inverse-frequency inspired class weights for SC loss.

    Args:
        num_classes: Number of lesion classes (e.g. 3 for pre_training,
            6 for fine_tuning). Returned tensor has length num_classes+1
            (background class 0 included).
        boost_nonsig: If True, double the weight for the Non-significant
            stenosis class to push the model to differentiate it.
        nonsig_idx: Index into the weights tensor for Non-significant stenosis
            (0=background, 1=Healthy, 2=NonSig, 3=Sig, ...). Default 2.
        boost_sig: If True, multiply weight for all Significant+plaque classes
            by sig_multiplier. Only applied in fine_tuning (num_classes==6)
            where Sig+plaque occupy indices 4,5,6.
        sig_multiplier: Multiplier applied to Sig class weights. Default 2.0.
    """
    weights = torch.ones(num_classes + 1, dtype=torch.float32)
    weights[0] = 0.5       # background
    weights[1:] = 1.5      # all lesion classes
    if boost_nonsig and nonsig_idx <= num_classes:
        weights[nonsig_idx] = weights[nonsig_idx] * 2.0
    if boost_sig and num_classes == 6:
        # fine_tuning 6-class: Sig+Calc=4, Sig+NonCalc=5, Sig+Mix=6
        for i in range(4, 7):
            weights[i] = weights[i] * sig_multiplier
    return weights


class OrdinalEMDLoss(nn.Module):
    """Earth Mover's Distance loss for ordinal classification.

    Penalises predictions proportional to the *ordinal distance* between the
    predicted class distribution and the ground-truth class.  This is more
    appropriate than cross-entropy for ordered labels such as stenosis
    severity (Healthy < Non-significant < Significant) because it assigns a
    larger gradient to gross ordinal errors (Healthy ↔ Significant) than to
    adjacent-class errors (Healthy ↔ Non-significant).

    Implementation: squared EMD over the cumulative distribution functions:
        loss = mean( ||CDF(pred) − CDF(target)||^2 )

    References:
        Hou et al., "Squared Earth Mover's Distance-based Loss for Training
        Deep Neural Networks on Ordered-classes", arXiv:1611.05916.

    Args:
        num_classes: Number of ordered classes.
        weight: Optional per-class weight tensor (same shape as for CE).
    """

    def __init__(self, num_classes: int, weight: torch.Tensor | None = None):
        super().__init__()
        self.num_classes = num_classes
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  Float tensor of shape (N, C) — raw logits.
            targets: Long tensor of shape (N,)   — ground-truth class indices.
        """
        probs = torch.softmax(inputs.float(), dim=1)  # (N, C)
        cum_probs = torch.cumsum(probs, dim=1)         # (N, C)

        target_onehot = torch.zeros_like(probs)
        target_onehot.scatter_(1, targets.view(-1, 1), 1.0)
        cum_targets = torch.cumsum(target_onehot, dim=1)  # (N, C)

        emd = ((cum_probs - cum_targets) ** 2).sum(dim=1)  # (N,)

        if self.weight is not None:
            sample_weights = self.weight[targets]
            emd = emd * sample_weights

        return emd.mean()


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
                 use_focal=False, focal_gamma=2.0, label_smoothing=0.0,
                 ordinal_weight=0.0):
        super().__init__()

        self.num_classes = num_classes
        self.seq_length = seq_length
        self.use_focal = use_focal
        self.label_smoothing = label_smoothing
        self.ordinal_weight = ordinal_weight

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        if self.use_focal:
            self.focal_loss_fn = FocalLoss(
                alpha=self.class_weights, gamma=focal_gamma, reduction='mean')

        if ordinal_weight > 0.0:
            self.ordinal_loss_fn = OrdinalEMDLoss(
                num_classes=num_classes, weight=class_weights)

    def loss_labels(self, outputs, targets):
        if self.use_focal:
            base = self.focal_loss_fn(outputs, targets)
        else:
            base = F.cross_entropy(outputs, targets, weight=self.class_weights,
                                   label_smoothing=self.label_smoothing)

        if self.ordinal_weight > 0.0:
            base = base + self.ordinal_weight * self.ordinal_loss_fn(outputs, targets)

        return base

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
                 confidence_threshold=0.0):
        super().__init__()

        self.od_contrastive_loss = od_contrastive_loss
        self.sc_contrastive_loss = sc_contrastive_loss
        self.matcher = self.od_contrastive_loss.matcher
        self.seq_length = seq_length
        self.confidence_threshold = confidence_threshold

    def set_dc_confidence(self, threshold: float) -> None:
        """Update the confidence gate applied in GT-free C⁻¹. Called each epoch by Trainer."""
        self.confidence_threshold = threshold

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

    def _get_sampling_point_classification_targets(self, od_outputs):
        """GT-free C⁻¹(ŷ_od): convert OD predictions → SC sampling-point labels.

        No ground truth consulted. Softmax all Q queries, keep only foreground
        queries above self.confidence_threshold, then for each sampling point
        the highest-confidence overlapping query wins (winner-takes-all).

        Output label space: 0 = background; OD class k → SC label k+1.
        Matches rounding convention of od2sc_targets (torch.round / clamp).
        """
        batch_size = od_outputs["pred_logits"].shape[0]
        interval = 1.0 / (self.seq_length + 1)
        sc_targets = []

        for b in range(batch_size):
            logits = od_outputs["pred_logits"][b]       # (Q, C+1)
            boxes  = od_outputs["pred_boxes"][b]        # (Q, 2) [cx, w]

            probs = torch.softmax(logits.float(), dim=-1)
            max_probs, pred_classes = torch.max(probs, dim=-1)  # (Q,)

            num_od_classes = logits.shape[-1] - 1       # last dim = no-object
            is_foreground = pred_classes < num_od_classes
            is_confident  = max_probs >= self.confidence_threshold
            mask = is_foreground & is_confident

            fg_labels = pred_classes[mask]              # (K,) OD-space 0-indexed
            fg_boxes  = boxes[mask]                     # (K, 2)
            fg_confs  = max_probs[mask]                 # (K,)

            sc_labels   = torch.zeros(self.seq_length, dtype=torch.long,
                                      device=logits.device)
            point_confs = torch.zeros(self.seq_length, dtype=torch.float32,
                                      device=logits.device)

            for k in range(fg_labels.shape[0]):
                cx = fg_boxes[k, 0].item()
                w  = fg_boxes[k, 1].item()
                x1, x2 = cx - w / 2.0, cx + w / 2.0
                # Same rounding as od2sc_targets: round(x/interval) then clamp(1..L)-1
                start = max(0, min(self.seq_length - 1, round(x1 / interval) - 1))
                end   = max(0, min(self.seq_length - 1, round(x2 / interval) - 1))
                conf  = fg_confs[k].item()
                lbl   = fg_labels[k].item() + 1         # shift OD→SC space

                for p in range(start, end + 1):
                    if conf > point_confs[p].item():
                        sc_labels[p]   = lbl
                        point_confs[p] = conf

            sc_targets.append({"labels": sc_labels})

        return sc_targets

    def forward(self, od_outputs, sc_outputs, od_targets=None):

        od_detached = {k: v.detach() for k, v in od_outputs.items()}
        sc_detached = {k: v.detach() for k, v in sc_outputs.items()}

        # SC→OD: C(ŷ_sc) — always prediction-based, as per paper Eq. 7
        od_con_targets = self._get_object_detection_targets(sc_detached)
        od_loss_values = self.od_contrastive_loss(od_outputs, od_con_targets)

        # OD→SC: C⁻¹ direction
        # When GT OD targets are provided (training): use them directly via od2sc_targets.
        # This prevents the GT-free feedback loop that causes Healthy class collapse when
        # OD predictions are unreliable early in training. Matches paper code behaviour.
        # GT-free fallback (od_targets=None): used for inference or ablation only.
        if od_targets is not None:
            sc_con_targets = od2sc_targets(od_targets, self.seq_length)
        else:
            sc_con_targets = self._get_sampling_point_classification_targets(od_detached)
        sc_loss_values = self.sc_contrastive_loss(sc_outputs, sc_con_targets)

        return sc_loss_values + od_loss_values


class spatio_temporal_contrast_loss(nn.Module):
    def __init__(self, num_classes=2, seq_length=32, eos_coef=0.2,
                 delta=1.0, sc_class_weights=None,
                 use_focal=False, focal_gamma=2.0,
                 dc_confidence_threshold=0.0,
                 label_smoothing=0.0,
                 ordinal_weight=0.0):
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
            label_smoothing=label_smoothing,
            ordinal_weight=ordinal_weight)
        self.dc_loss = dual_task_contrastive_loss(
            self.od_loss, self.sc_loss, seq_length=self.seq_length,
            confidence_threshold=dc_confidence_threshold)

    def set_dc_weight(self, weight):
        """Set the current dc loss weight (for delayed ramp scheduling)."""
        self.dc_weight = weight

    def set_dc_confidence(self, threshold: float) -> None:
        """Delegate to dc_loss — called each epoch by Trainer to anneal the confidence gate."""
        self.dc_loss.set_dc_confidence(threshold)

    def forward(self, od_outputs, sc_outputs, od_targets):

        # Clone targets to prevent in-place mutation across loss terms
        od_targets_od = [{k: v.clone() for k, v in t.items()} for t in od_targets]
        od_targets_sc = [{k: v.clone() for k, v in t.items()} for t in od_targets]
        od_targets_dc = [{k: v.clone() for k, v in t.items()} for t in od_targets]

        # dc_loss receives GT targets: OD→SC uses od2sc_targets(GT), SC→OD stays prediction-based
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