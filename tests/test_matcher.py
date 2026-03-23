# tests/test_matcher.py
import torch
import pytest
import sys
sys.path.insert(0, '.')
from functions import HungarianMatcher


def _make_outputs(bs, num_queries, num_classes):
    return {
        "pred_logits": torch.randn(bs, num_queries, num_classes + 1),
        "pred_boxes":  torch.rand(bs, num_queries, 4),
    }


def _make_targets(sizes, num_classes):
    return [
        {
            "labels": torch.randint(0, num_classes, (n,)),
            "boxes":  torch.rand(n, 4),
        }
        for n in sizes
    ]


def test_matcher_returns_one_index_per_batch_item():
    """Matcher must return exactly bs (row, col) tuples."""
    bs, num_q, C = 2, 16, 3
    matcher = HungarianMatcher()
    outputs = _make_outputs(bs, num_q, C)
    targets = _make_targets([3, 2], C)
    indices = matcher(outputs, targets)
    assert len(indices) == bs, f"Expected {bs} index pairs, got {len(indices)}"


def test_matcher_index_values_in_range():
    """Every source index must be < num_queries; every target index < n_targets."""
    bs, num_q, C = 3, 16, 3
    sizes = [4, 2, 3]
    matcher = HungarianMatcher()
    outputs = _make_outputs(bs, num_q, C)
    targets = _make_targets(sizes, C)
    indices = matcher(outputs, targets)
    for b, (src, tgt) in enumerate(indices):
        if src.numel() > 0:
            assert src.max() < num_q,    f"Batch {b}: src {src.max()} >= {num_q}"
        if tgt.numel() > 0:
            assert tgt.max() < sizes[b], f"Batch {b}: tgt {tgt.max()} >= {sizes[b]}"


def test_matcher_handles_empty_batch():
    """Matcher must handle targets with 0 boxes without crashing."""
    bs, num_q, C = 2, 16, 3
    matcher = HungarianMatcher()
    outputs = _make_outputs(bs, num_q, C)
    targets = [
        {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)},
        {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
    ]
    indices = matcher(outputs, targets)
    assert len(indices) == bs
    empty_src, empty_tgt = indices[0]
    assert empty_src.numel() == 0
    assert empty_tgt.numel() == 0


def test_matcher_all_empty_targets():
    """When all targets are empty, matcher returns bs pairs of empty tensors."""
    bs, num_q, C = 2, 16, 3
    matcher = HungarianMatcher()
    outputs = _make_outputs(bs, num_q, C)
    targets = [
        {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)},
        {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)},
    ]
    indices = matcher(outputs, targets)
    assert len(indices) == bs
    for src, tgt in indices:
        assert src.numel() == 0
        assert tgt.numel() == 0


def test_matcher_bs1():
    """Matcher must work for batch size 1."""
    matcher = HungarianMatcher()
    outputs = _make_outputs(1, 16, 3)
    targets = _make_targets([2], 3)
    indices = matcher(outputs, targets)
    assert len(indices) == 1
