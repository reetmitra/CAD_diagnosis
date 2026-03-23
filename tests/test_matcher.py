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
        assert src.max() < num_q,    f"Batch {b}: src index {src.max()} >= num_queries {num_q}"
        assert tgt.max() < sizes[b], f"Batch {b}: tgt index {tgt.max()} >= n_targets {sizes[b]}"


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


def test_matcher_bs1():
    """Matcher must work for batch size 1."""
    matcher = HungarianMatcher()
    outputs = _make_outputs(1, 16, 3)
    targets = _make_targets([2], 3)
    indices = matcher(outputs, targets)
    assert len(indices) == 1
