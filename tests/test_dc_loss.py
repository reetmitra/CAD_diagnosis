# tests/test_dc_loss.py
import torch
import pytest
import sys
sys.path.insert(0, '.')

from optimization import (
    dual_task_contrastive_loss,
    object_detection_loss,
    sampling_point_classification_loss,
    spatio_temporal_contrast_loss,
)

SEQ_LENGTH = 16
NUM_OD_CLASSES = 2  # → num_sc_classes = 3, no-object index = 2


def _make_dc_loss(confidence_threshold=0.0):
    od_loss = object_detection_loss(num_classes=NUM_OD_CLASSES)
    sc_loss = sampling_point_classification_loss(
        num_classes=NUM_OD_CLASSES + 1, seq_length=SEQ_LENGTH)
    return dual_task_contrastive_loss(
        od_contrastive_loss=od_loss,
        sc_contrastive_loss=sc_loss,
        seq_length=SEQ_LENGTH,
        confidence_threshold=confidence_threshold,
    )


def _make_od_outputs(batch_size=1, num_queries=4, logits=None, boxes=None):
    if logits is None:
        logits = torch.zeros(batch_size, num_queries, NUM_OD_CLASSES + 1)
    if boxes is None:
        boxes = torch.rand(batch_size, num_queries, 2)
    return {"pred_logits": logits, "pred_boxes": boxes}


# ── shape ──────────────────────────────────────────────────────────────────────

def test_gt_free_output_shape():
    """Returns a list of B dicts, each with 'labels' of shape (seq_length,) dtype long."""
    dc = _make_dc_loss()
    od_out = _make_od_outputs(batch_size=2, num_queries=5)
    result = dc._get_sampling_point_classification_targets(od_out)
    assert len(result) == 2
    for item in result:
        assert "labels" in item
        assert item["labels"].shape == (SEQ_LENGTH,), item["labels"].shape
        assert item["labels"].dtype == torch.long


# ── no-object filtering ────────────────────────────────────────────────────────

def test_gt_free_all_background_when_noobj():
    """All queries predicting no-object → all sampling points stay background (0)."""
    dc = _make_dc_loss(confidence_threshold=0.0)
    logits = torch.full((1, 4, NUM_OD_CLASSES + 1), -10.0)
    logits[0, :, NUM_OD_CLASSES] = 100.0   # strong no-object signal on every query
    od_out = _make_od_outputs(logits=logits, boxes=torch.rand(1, 4, 2))
    result = dc._get_sampling_point_classification_targets(od_out)
    assert (result[0]["labels"] == 0).all(), \
        f"Expected all zeros, got {result[0]['labels']}"


# ── confidence threshold ───────────────────────────────────────────────────────

def test_gt_free_confidence_threshold_excludes_low_conf_query():
    """A foreground query whose max_prob < threshold must not assign any points."""
    dc = _make_dc_loss(confidence_threshold=0.9)
    # logits [1.0, 0.0, 0.0] → softmax ≈ [0.576, 0.212, 0.212] → class 0, conf ≈ 0.576
    logits = torch.tensor([[[1.0, 0.0, 0.0]]])   # (1, 1, 3)
    boxes  = torch.tensor([[[0.5, 0.9]]])          # wide box covering most points
    od_out = _make_od_outputs(logits=logits, boxes=boxes)
    result = dc._get_sampling_point_classification_targets(od_out)
    assert (result[0]["labels"] == 0).all(), \
        "Below-threshold query should not assign any point labels"


def test_gt_free_high_conf_query_assigns_points():
    """A foreground query with conf > threshold must assign points within its box."""
    dc = _make_dc_loss(confidence_threshold=0.5)
    # logits [5.0, 0.0, 0.0] → softmax ≈ [0.993, 0.003, 0.003] → class 0, conf ≈ 0.993
    # seq_length=16, interval=1/17; cx=8/17≈0.471, w=0.05 → covers point 7
    cx = 8.0 / 17.0
    logits = torch.tensor([[[5.0, 0.0, 0.0]]])
    boxes  = torch.tensor([[[cx, 0.05]]])
    od_out = _make_od_outputs(logits=logits, boxes=boxes)
    result = dc._get_sampling_point_classification_targets(od_out)
    assert result[0]["labels"][7].item() != 0, \
        "High-confidence query should assign point 7"


# ── label shift (OD class k → SC label k+1) ───────────────────────────────────

def test_gt_free_label_shift_class0():
    """OD class 0 (first foreground) → SC label 1."""
    dc = _make_dc_loss(confidence_threshold=0.0)
    cx = 8.0 / 17.0
    logits = torch.tensor([[[5.0, 0.0, 0.0]]])    # strong class 0
    boxes  = torch.tensor([[[cx, 0.05]]])
    od_out = _make_od_outputs(logits=logits, boxes=boxes)
    result = dc._get_sampling_point_classification_targets(od_out)
    assert result[0]["labels"][7].item() == 1, \
        f"OD class 0 should map to SC label 1, got {result[0]['labels'][7].item()}"


def test_gt_free_label_shift_class1():
    """OD class 1 (second foreground) → SC label 2."""
    dc = _make_dc_loss(confidence_threshold=0.0)
    cx = 8.0 / 17.0
    logits = torch.tensor([[[0.0, 5.0, 0.0]]])    # strong class 1
    boxes  = torch.tensor([[[cx, 0.05]]])
    od_out = _make_od_outputs(logits=logits, boxes=boxes)
    result = dc._get_sampling_point_classification_targets(od_out)
    assert result[0]["labels"][7].item() == 2, \
        f"OD class 1 should map to SC label 2, got {result[0]['labels'][7].item()}"


# ── winner-takes-all deduplication ────────────────────────────────────────────

def test_gt_free_winner_takes_all_highest_conf_wins():
    """When two queries overlap the same point, the higher-confidence one sets the label."""
    dc = _make_dc_loss(confidence_threshold=0.0)
    # seq_length=16, interval=1/17; cx=8/17 → covers point 7
    cx = 8.0 / 17.0
    # Query A: class 0, conf ≈ 0.993 (logit 5.0)
    # Query B: class 1, conf ≈ 0.789 (logit 2.0)
    # Both centered on the same point → A should win
    logits = torch.tensor([[[5.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0]]])   # (1, 2, 3)
    boxes  = torch.tensor([[[cx, 0.05],
                             [cx, 0.05]]])          # (1, 2, 2) both at same position
    od_out = _make_od_outputs(logits=logits, boxes=boxes)
    result = dc._get_sampling_point_classification_targets(od_out)
    # A (class 0 → SC label 1) has higher confidence than B (class 1 → SC label 2)
    assert result[0]["labels"][7].item() == 1, \
        f"Higher-confidence Query A (SC label 1) should win, got {result[0]['labels'][7].item()}"


def test_gt_free_non_overlapping_queries_both_assigned():
    """Queries covering disjoint regions both assign their labels independently."""
    dc = _make_dc_loss(confidence_threshold=0.0)
    # seq_length=16, interval=1/17
    # Query A at point 3 (cx = 4/17), Query B at point 12 (cx = 13/17)
    cx_a = 4.0 / 17.0
    cx_b = 13.0 / 17.0
    logits = torch.tensor([[[5.0, 0.0, 0.0],    # class 0 → SC label 1
                             [0.0, 5.0, 0.0]]])  # class 1 → SC label 2
    boxes  = torch.tensor([[[cx_a, 0.05],
                             [cx_b, 0.05]]])
    od_out = _make_od_outputs(logits=logits, boxes=boxes)
    result = dc._get_sampling_point_classification_targets(od_out)
    labels = result[0]["labels"]
    assert labels[3].item() == 1, f"Point 3 should be label 1 (from Query A), got {labels[3].item()}"
    assert labels[12].item() == 2, f"Point 12 should be label 2 (from Query B), got {labels[12].item()}"


# ── set_dc_confidence ──────────────────────────────────────────────────────────

def test_set_dc_confidence_updates_threshold():
    """set_dc_confidence must update self.confidence_threshold on the dc loss."""
    dc = _make_dc_loss(confidence_threshold=0.0)
    dc.set_dc_confidence(0.85)
    assert dc.confidence_threshold == pytest.approx(0.85), \
        f"Expected 0.85, got {dc.confidence_threshold}"


def test_set_dc_confidence_affects_filtering():
    """After raising threshold, a previously-assigned point becomes background."""
    dc = _make_dc_loss(confidence_threshold=0.0)
    cx = 8.0 / 17.0
    # logits [1.0, 0.0, 0.0] → conf ≈ 0.576
    logits = torch.tensor([[[1.0, 0.0, 0.0]]])
    boxes  = torch.tensor([[[cx, 0.05]]])
    od_out = _make_od_outputs(logits=logits, boxes=boxes)

    # With threshold=0.0, point 7 gets assigned
    result_before = dc._get_sampling_point_classification_targets(od_out)
    assert result_before[0]["labels"][7].item() != 0, "Should assign point 7 with threshold=0.0"

    # Raise threshold above 0.576 → point 7 becomes background
    dc.set_dc_confidence(0.9)
    result_after = dc._get_sampling_point_classification_targets(od_out)
    assert result_after[0]["labels"][7].item() == 0, \
        "Should not assign point 7 with threshold=0.9 (conf ≈ 0.576 < 0.9)"


def test_spatio_temporal_set_dc_confidence_delegates():
    """spatio_temporal_contrast_loss.set_dc_confidence must delegate to dc_loss."""
    loss_fn = spatio_temporal_contrast_loss(num_classes=NUM_OD_CLASSES, seq_length=SEQ_LENGTH)
    loss_fn.set_dc_confidence(0.75)
    assert loss_fn.dc_loss.confidence_threshold == pytest.approx(0.75), \
        f"Expected 0.75, got {loss_fn.dc_loss.confidence_threshold}"


# ── forward no longer requires od_targets ─────────────────────────────────────

def test_dc_forward_works_without_od_targets():
    """dual_task_contrastive_loss.forward must work when called with only 2 args."""
    dc = _make_dc_loss()
    od_out = _make_od_outputs(batch_size=2, num_queries=5)
    sc_out = {"pred_logits": torch.randn(2, SEQ_LENGTH, NUM_OD_CLASSES + 1)}
    # Must not raise TypeError
    loss = dc(od_out, sc_out)
    assert loss.item() >= 0.0


def test_spatio_temporal_forward_does_not_pass_gt_to_dc():
    """spatio_temporal_contrast_loss.forward must call dc_loss without GT targets.

    Verified by monkey-patching dc_loss.forward to assert it receives no od_targets.
    """
    loss_fn = spatio_temporal_contrast_loss(num_classes=NUM_OD_CLASSES, seq_length=SEQ_LENGTH)
    captured = {}

    original_forward = loss_fn.dc_loss.forward

    def patched_forward(od_outputs, sc_outputs, od_targets=None):
        captured['od_targets'] = od_targets
        return original_forward(od_outputs, sc_outputs, od_targets)

    loss_fn.dc_loss.forward = patched_forward

    bs, Q = 2, 5
    od_out = {
        "pred_logits": torch.randn(bs, Q, NUM_OD_CLASSES + 1),
        "pred_boxes":  torch.rand(bs, Q, 2),
    }
    sc_out = {"pred_logits": torch.randn(bs, SEQ_LENGTH, NUM_OD_CLASSES + 1)}
    targets = [
        {"labels": torch.tensor([0, 1]), "boxes": torch.rand(2, 2)},
        {"labels": torch.tensor([0]),    "boxes": torch.rand(1, 2)},
    ]

    loss_fn(od_out, sc_out, targets)
    assert captured.get('od_targets') is None, \
        "spatio_temporal_contrast_loss must not pass od_targets to dc_loss.forward"


# ── _compute_dc_confidence annealing ──────────────────────────────────────────
# Tests the annealing schedule as a pure function matching Trainer._compute_dc_confidence.
# Tested standalone (no Trainer instantiation required).

def _compute_dc_confidence(epoch, hold, ramp, start, floor):
    """Mirror of Trainer._compute_dc_confidence — tested here as a pure function."""
    if hold == 0 and ramp == 0:
        return floor
    if epoch < hold:
        return start
    elif ramp > 0 and epoch < hold + ramp:
        progress = (epoch - hold) / ramp
        return start - progress * (start - floor)
    else:
        return floor


def test_dc_confidence_during_hold():
    """Returns start value for any epoch within the hold period."""
    result = _compute_dc_confidence(5, hold=10, ramp=20, start=0.7, floor=0.0)
    assert result == pytest.approx(0.7)


def test_dc_confidence_at_hold_ramp_boundary():
    """At epoch=hold (start of ramp), progress=0 so result equals start."""
    result = _compute_dc_confidence(10, hold=10, ramp=20, start=0.7, floor=0.0)
    assert result == pytest.approx(0.7)


def test_dc_confidence_mid_ramp():
    """Mid-ramp: epoch=hold+ramp/2, progress=0.5 → linearly interpolated."""
    # hold=10, ramp=20 → mid-point at epoch 20
    # progress = (20-10)/20 = 0.5 → 0.7 - 0.5 * (0.7 - 0.0) = 0.35
    result = _compute_dc_confidence(20, hold=10, ramp=20, start=0.7, floor=0.0)
    assert result == pytest.approx(0.35)


def test_dc_confidence_end_of_ramp():
    """At epoch=hold+ramp, progress=1 so result equals floor."""
    result = _compute_dc_confidence(30, hold=10, ramp=20, start=0.7, floor=0.0)
    assert result == pytest.approx(0.0)


def test_dc_confidence_after_ramp():
    """Any epoch past hold+ramp returns floor."""
    result = _compute_dc_confidence(99, hold=10, ramp=20, start=0.7, floor=0.0)
    assert result == pytest.approx(0.0)


def test_dc_confidence_no_schedule_returns_floor_immediately():
    """With hold=0, ramp=0, always returns floor regardless of epoch."""
    assert _compute_dc_confidence(0,   hold=0, ramp=0, start=0.7, floor=0.3) == pytest.approx(0.3)
    assert _compute_dc_confidence(100, hold=0, ramp=0, start=0.7, floor=0.3) == pytest.approx(0.3)


def test_dc_confidence_non_zero_floor():
    """Annealing stops at floor, not zero, when floor > 0."""
    # hold=0, ramp=10, start=0.8, floor=0.2; epoch=10 → end of ramp
    result = _compute_dc_confidence(10, hold=0, ramp=10, start=0.8, floor=0.2)
    assert result == pytest.approx(0.2)
