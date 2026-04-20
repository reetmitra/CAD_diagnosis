# GT-Free C⁻¹ with Curriculum Confidence Annealing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the ground-truth dependency from C⁻¹(ŷ_od) in L_dc so the dual-task contrastive loss matches paper Eq. 7 faithfully, while keeping stability via confidence annealing tied to the existing DC warmup schedule.

**Architecture:** Replace the Hungarian-matching step in `_get_sampling_point_classification_targets` with a pure OD-prediction function: argmax all Q queries, filter foreground by confidence threshold, resolve per-point conflicts with winner-takes-all (highest confidence wins). Add `set_dc_confidence(threshold)` to the loss chain so the trainer can anneal the threshold from a high start value (default 0.7) down to `dc_confidence_threshold` over `dc_warmup_ramp` epochs — reusing the existing DC warmup infrastructure.

**Tech Stack:** PyTorch, existing `optimization.py` / `train.py` infrastructure, pytest.

---

## File Map

| File | Change |
|------|--------|
| `optimization.py` | Replace `_get_sampling_point_classification_targets`; add `set_dc_confidence` to both loss classes; update `dual_task_contrastive_loss.forward` + `spatio_temporal_contrast_loss.forward` |
| `train.py` | Add `--dc_confidence_start` arg; add `_compute_dc_confidence` to `Trainer`; wire into `train_one_epoch` and `run` |
| `tests/test_dc_loss.py` | New file — all tests for this feature |

---

## Background: What Is Changing and Why

The paper defines:
```
L_dc = L_od(C(ŷ_sc), ŷ_od) + L_sc(C⁻¹(ŷ_od), ŷ_sc)
```

C⁻¹(ŷ_od) is a pure function of OD predictions. The current implementation runs Hungarian matching against ground-truth targets to pick candidate queries first, which introduces a GT dependency the paper does not have. The new implementation takes argmax over all Q queries directly.

The stability risk: early in training, OD predicts mostly "no-object" → near-empty pseudo-targets → no SC contrastive signal. The mitigation is a high confidence threshold at the start of training (only very certain predictions become pseudo-labels), annealed down as OD quality improves during the DC warmup ramp.

**Label space:** SC class space is `{0=background, 1..N=OD class k+1}`. OD class space is `{0..N-1=foreground, N=no-object}`. The new method produces SC-space labels directly (no `od2sc_targets` call needed).

---

## Task 1: Write failing tests for GT-free `_get_sampling_point_classification_targets`

**Files:**
- Create: `tests/test_dc_loss.py`

- [ ] **Step 1: Create the test file**

```python
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
```

- [ ] **Step 2: Run tests to verify they all fail (method not yet changed)**

```bash
cd /home/reet/development/CAD_diagnosis && source .venv/bin/activate && pytest tests/test_dc_loss.py -v 2>&1 | head -60
```

Expected: several tests FAIL or ERROR — `_get_sampling_point_classification_targets` still uses GT matching, so shape/output may differ from expected.

---

## Task 2: Implement GT-free `_get_sampling_point_classification_targets` + update `dual_task_contrastive_loss.forward`

**Files:**
- Modify: `optimization.py:339-374` (replace `_get_sampling_point_classification_targets`)
- Modify: `optimization.py:440-458` (update `forward`)

- [ ] **Step 1: Replace `_get_sampling_point_classification_targets` (optimization.py:339-374)**

Find the old method:
```python
    def _get_sampling_point_classification_targets(self, od_outputs, od_targets):
```

Replace the entire method (from that `def` line through the `return od2sc_targets(ret_od_targets, self.seq_length)` line) with:

```python
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
```

- [ ] **Step 2: Update `dual_task_contrastive_loss.forward` to not require `od_targets` (optimization.py:440)**

Find:
```python
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
```

Replace with:
```python
    def forward(self, od_outputs, sc_outputs, od_targets=None):

        od_detached = {k: v.detach() for k, v in od_outputs.items()}
        sc_detached = {k: v.detach() for k, v in sc_outputs.items()}

        # OD contrastive: C(ŷ_sc) → pseudo-targets for L_od
        od_con_targets = self._get_object_detection_targets(sc_detached)
        od_loss_values = self.od_contrastive_loss(od_outputs, od_con_targets)

        # SC contrastive: C⁻¹(ŷ_od) → pseudo-targets for L_sc (GT-free hard path)
        # Soft path retains od_targets for backwards compat but is rarely used.
        if self.use_soft_labels and od_targets is not None:
            sc_loss_values = self._compute_soft_sc_loss(
                sc_outputs, od_detached, od_targets)
        else:
            sc_con_targets = self._get_sampling_point_classification_targets(
                od_detached)
            sc_loss_values = self.sc_contrastive_loss(sc_outputs, sc_con_targets)

        return sc_loss_values + od_loss_values
```

- [ ] **Step 3: Run Task 1 tests to verify they now pass**

```bash
cd /home/reet/development/CAD_diagnosis && source .venv/bin/activate && pytest tests/test_dc_loss.py -v 2>&1
```

Expected: all tests in `test_dc_loss.py` **PASS**.

- [ ] **Step 4: Commit**

```bash
cd /home/reet/development/CAD_diagnosis && git add optimization.py tests/test_dc_loss.py && git commit -m "feat: GT-free C⁻¹ in dual_task_contrastive_loss — removes ground-truth dependency from Ldc SC direction"
```

---

## Task 3: Write failing tests for `set_dc_confidence` + `spatio_temporal_contrast_loss.forward`

**Files:**
- Modify: `tests/test_dc_loss.py` (append new tests)

- [ ] **Step 1: Append the following tests to `tests/test_dc_loss.py`**

```python
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
```

- [ ] **Step 2: Run only the new tests to verify they fail**

```bash
cd /home/reet/development/CAD_diagnosis && source .venv/bin/activate && pytest tests/test_dc_loss.py -v -k "set_dc_confidence or without_od_targets or does_not_pass_gt" 2>&1
```

Expected: all 5 new tests **FAIL** (`AttributeError: 'dual_task_contrastive_loss' object has no attribute 'set_dc_confidence'`).

---

## Task 4: Implement `set_dc_confidence` methods + update `spatio_temporal_contrast_loss.forward`

**Files:**
- Modify: `optimization.py` (3 edits)

- [ ] **Step 1: Add `set_dc_confidence` to `dual_task_contrastive_loss` (after `set_dc_temperature`, optimization.py:321)**

Find:
```python
    def set_dc_temperature(self, temperature: float) -> None:
        self.dc_temperature = max(temperature, 1.0)
```

Replace with:
```python
    def set_dc_confidence(self, threshold: float) -> None:
        """Update the confidence gate applied in GT-free C⁻¹. Called each epoch by Trainer."""
        self.confidence_threshold = threshold

    def set_dc_temperature(self, temperature: float) -> None:
        self.dc_temperature = max(temperature, 1.0)
```

- [ ] **Step 2: Add `set_dc_confidence` to `spatio_temporal_contrast_loss` (after `set_dc_temperature`, optimization.py:495)**

Find:
```python
    def set_dc_temperature(self, temperature: float) -> None:
        self.dc_loss.set_dc_temperature(temperature)
```

Replace with:
```python
    def set_dc_confidence(self, threshold: float) -> None:
        """Delegate to dc_loss — called each epoch by Trainer to anneal the confidence gate."""
        self.dc_loss.set_dc_confidence(threshold)

    def set_dc_temperature(self, temperature: float) -> None:
        self.dc_loss.set_dc_temperature(temperature)
```

- [ ] **Step 3: Update `spatio_temporal_contrast_loss.forward` to not pass GT to `self.dc_loss` (optimization.py:498)**

Find:
```python
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
```

Replace with:
```python
    def forward(self, od_outputs, sc_outputs, od_targets):

        # Deep copy targets to prevent in-place mutation across loss terms
        od_targets_od = [{k: v.clone() for k, v in t.items()} for t in od_targets]
        od_targets_sc = [{k: v.clone() for k, v in t.items()} for t in od_targets]

        # dc_loss uses GT-free C⁻¹ — no od_targets passed
        dc_loss_val = self.dc_loss(od_outputs, sc_outputs) * self.dc_weight
        od_loss_val = self.od_loss(od_outputs, od_targets_od)
        sc_loss_val = self.sc_loss(sc_outputs, od2sc_targets(od_targets_sc, self.seq_length))
        total_loss = dc_loss_val + od_loss_val + sc_loss_val

        return {
            'total': total_loss,
            'od': od_loss_val,
            'sc': sc_loss_val,
            'dc': dc_loss_val
        }
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
cd /home/reet/development/CAD_diagnosis && source .venv/bin/activate && pytest tests/test_dc_loss.py -v 2>&1
```

Expected: **all tests PASS**.

- [ ] **Step 5: Run existing test suite to check for regressions**

```bash
cd /home/reet/development/CAD_diagnosis && source .venv/bin/activate && pytest tests/ -v 2>&1
```

Expected: all tests in `tests/test_matcher.py` still **PASS**.

- [ ] **Step 6: Commit**

```bash
cd /home/reet/development/CAD_diagnosis && git add optimization.py tests/test_dc_loss.py && git commit -m "feat: add set_dc_confidence interface; forward no longer passes GT to dc_loss"
```

---

## Task 5: Write failing tests for `_compute_dc_confidence` annealing logic

**Files:**
- Modify: `tests/test_dc_loss.py` (append)

- [ ] **Step 1: Append the following tests to `tests/test_dc_loss.py`**

```python
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
```

- [ ] **Step 2: Run the new tests to verify they fail (method not yet in train.py)**

```bash
cd /home/reet/development/CAD_diagnosis && source .venv/bin/activate && pytest tests/test_dc_loss.py -v -k "dc_confidence" 2>&1
```

Expected: The `_compute_dc_confidence` helper function is defined locally in the test file so these tests will actually **PASS** immediately. Confirm they do. This validates the schedule math before wiring it into `train.py`.

---

## Task 6: Implement `--dc_confidence_start` + `_compute_dc_confidence` + wiring in `train.py`

**Files:**
- Modify: `train.py` (4 edits)

- [ ] **Step 1: Add `--dc_confidence_start` arg to `parse_args` (after `--dc_temperature_start`, train.py:163)**

Find:
```python
    parser.add_argument('--dc_temperature_start', type=float, default=3.0,
                        help='Initial softmax temperature for DC soft pseudo-labels; anneals to 1.0 over dc_warmup_ramp (default: 3.0)')
```

Replace with:
```python
    parser.add_argument('--dc_temperature_start', type=float, default=3.0,
                        help='Initial softmax temperature for DC soft pseudo-labels; anneals to 1.0 over dc_warmup_ramp (default: 3.0)')
    parser.add_argument('--dc_confidence_start', type=float, default=0.7,
                        help='Initial confidence threshold for GT-free C⁻¹(ŷ_od) pseudo-labels; '
                             'linearly anneals to --dc_confidence_threshold over dc_warmup_ramp epochs '
                             '(default: 0.7). Only active when dc_warmup_ramp > 0.')
```

- [ ] **Step 2: Add `_compute_dc_confidence` method to `Trainer` (after `_compute_dc_weight`, train.py:606)**

Find the end of `_compute_dc_weight`:
```python
        else:
            return self.args.delta
```

The full `_compute_dc_weight` method ends with that line. Add the new method immediately after (before `def train_one_epoch`):

```python
    def _compute_dc_confidence(self, epoch: int) -> float:
        """Anneal the GT-free C⁻¹ confidence threshold from dc_confidence_start to floor.

        Mirrors the dc_weight schedule:
          - During hold: return dc_confidence_start (aggressive filtering)
          - During ramp: linearly interpolate from start → dc_confidence_threshold
          - After ramp: return dc_confidence_threshold (floor)
          - No schedule (hold=ramp=0): return dc_confidence_threshold immediately
        """
        hold  = self.args.dc_warmup_hold
        ramp  = self.args.dc_warmup_ramp
        start = getattr(self.args, 'dc_confidence_start', 0.7)
        floor = self.args.dc_confidence_threshold

        if hold == 0 and ramp == 0:
            return floor
        if epoch < hold:
            return start
        elif ramp > 0 and epoch < hold + ramp:
            progress = (epoch - hold) / ramp
            return start - progress * (start - floor)
        else:
            return floor

```

- [ ] **Step 3: Call `set_dc_confidence` in `train_one_epoch` (after the dc_weight block, train.py:615)**

Find:
```python
        # Update dc weight for delayed ramp
        dc_weight = self._compute_dc_weight(epoch)
        self.loss_fn.set_dc_weight(dc_weight)
```

Replace with:
```python
        # Update dc weight and confidence threshold for delayed ramp
        dc_weight = self._compute_dc_weight(epoch)
        self.loss_fn.set_dc_weight(dc_weight)
        dc_confidence = self._compute_dc_confidence(epoch)
        self.loss_fn.set_dc_confidence(dc_confidence)
```

- [ ] **Step 4: Log `dc_confidence` in the epoch summary in `run()` (train.py:860)**

Find:
```python
                dc_w = self._compute_dc_weight(epoch)
                print(f"Epoch [{epoch}/{self.args.epochs}] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"LR: {current_lr:.6f} | DC_w: {dc_w:.3f} | Time: {elapsed:.1f}s")
```

Replace with:
```python
                dc_w    = self._compute_dc_weight(epoch)
                dc_conf = self._compute_dc_confidence(epoch)
                print(f"Epoch [{epoch}/{self.args.epochs}] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"LR: {current_lr:.6f} | DC_w: {dc_w:.3f} | "
                      f"DC_conf: {dc_conf:.3f} | Time: {elapsed:.1f}s")
```

- [ ] **Step 5: Add TensorBoard scalar for `dc_confidence` (train.py:893, in the Schedule section)**

Find:
```python
                    self.writer.add_scalar('Schedule/dc_weight',
                                           dc_w, epoch)
```

Replace with:
```python
                    self.writer.add_scalar('Schedule/dc_weight',
                                           dc_w, epoch)
                    self.writer.add_scalar('Schedule/dc_confidence',
                                           dc_conf, epoch)
```

Note: `dc_conf` is now defined in the same block (Step 4 above renamed it from `dc_w`). Confirm the variable name is `dc_conf` in both places.

- [ ] **Step 6: Update `_print_summary` to show `dc_confidence_start` (train.py:1003)**

Find:
```python
        if self.args.dc_warmup_hold > 0 or self.args.dc_warmup_ramp > 0:
            print(f"  DC warmup:        hold={self.args.dc_warmup_hold}, "
                  f"ramp={self.args.dc_warmup_ramp}")
```

Replace with:
```python
        if self.args.dc_warmup_hold > 0 or self.args.dc_warmup_ramp > 0:
            dc_conf_start = getattr(self.args, 'dc_confidence_start', 0.7)
            print(f"  DC warmup:        hold={self.args.dc_warmup_hold}, "
                  f"ramp={self.args.dc_warmup_ramp}, "
                  f"conf_start={dc_conf_start:.2f}→{self.args.dc_confidence_threshold:.2f}")
```

- [ ] **Step 7: Run full test suite**

```bash
cd /home/reet/development/CAD_diagnosis && source .venv/bin/activate && pytest tests/ -v 2>&1
```

Expected: **all tests PASS**.

- [ ] **Step 8: Smoke-test the training path with dummy data**

```bash
cd /home/reet/development/CAD_diagnosis && source .venv/bin/activate && python train.py \
    --pattern pre_training \
    --epochs 3 \
    --dc_warmup_hold 1 \
    --dc_warmup_ramp 2 \
    --dc_confidence_start 0.7 \
    --dc_confidence_threshold 0.0 \
    --delta 1.0 \
    --data_root ./data \
    --checkpoint_dir /tmp/smoke_test_gt_free \
    2>&1 | head -60
```

Expected: training runs 3 epochs without error; DC_conf line in epoch printout shows 0.70 → 0.35 → 0.00.

- [ ] **Step 9: Commit**

```bash
cd /home/reet/development/CAD_diagnosis && git add train.py tests/test_dc_loss.py && git commit -m "feat: add --dc_confidence_start; Trainer anneals GT-free C⁻¹ confidence over dc_warmup_ramp"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] GT-free C⁻¹ — no Hungarian matching against GT: Task 2
- [x] Winner-takes-all per-point deduplication: Task 2 (inside new method)
- [x] `set_dc_confidence` interface on both loss classes: Task 4
- [x] `spatio_temporal_contrast_loss.forward` stops passing GT to dc_loss: Task 4
- [x] `--dc_confidence_start` CLI arg: Task 6
- [x] `_compute_dc_confidence` annealing schedule: Task 6
- [x] Wiring in `train_one_epoch`: Task 6
- [x] TensorBoard logging: Task 6
- [x] No-schedule fallback (hold=ramp=0 → floor immediately): Task 5 tests + Task 6 impl
- [x] Existing test suite regression check: Task 4 Step 5

**Placeholder scan:** No TBDs, no "similar to Task N", no missing implementations.

**Type consistency:**
- `_get_sampling_point_classification_targets` returns `List[Dict[str, torch.Tensor]]` with key `"labels"` — consistent with what `sampling_point_classification_loss.forward` expects (it accesses `t["labels"]` in `torch.cat([t["labels"] for t in targets], dim=0)`). ✅
- `set_dc_confidence(threshold: float)` called on `loss_fn` (a `spatio_temporal_contrast_loss`) from `Trainer` — method added in Task 4. ✅
- `dc_conf` variable name used consistently in `run()` printout and TensorBoard block (both set in the same `if self.is_main:` block after Step 4's edit). ✅

---

## Usage Notes for Future Runs

**With DC warmup schedule (recommended):**
```bash
python train.py \
    --dc_warmup_hold 20 \
    --dc_warmup_ramp 30 \
    --dc_confidence_start 0.7 \
    --dc_confidence_threshold 0.0 \
    --delta 1.0
```
During epochs 0-19: only OD predictions with ≥70% confidence contribute SC pseudo-labels.
During epochs 20-49: threshold linearly falls 0.7→0.0.
After epoch 50: all foreground predictions contribute.

**Without DC warmup (same as before):**
```bash
python train.py --delta 1.0
# dc_confidence_start has no effect; threshold stays at dc_confidence_threshold (0.0)
```

**With a non-zero floor (conservative late-training):**
```bash
python train.py \
    --dc_warmup_hold 10 \
    --dc_warmup_ramp 20 \
    --dc_confidence_start 0.7 \
    --dc_confidence_threshold 0.3
# Threshold anneals 0.7→0.3 and stays at 0.3 for the rest of training
```
