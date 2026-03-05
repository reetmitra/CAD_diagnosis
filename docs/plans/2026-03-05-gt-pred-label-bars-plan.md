# GT/Pred Label Bars Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two thin colour-coded label bars below the longitudinal CPR strip — one showing ground-truth normality/abnormality per slice, one showing predicted normality/abnormality — so discrepancies are immediately visible.

**Architecture:** All changes are inside `render_artery()` in `visualize.py`. A new `_draw_label_bar` inner helper renders a 1×D imshow with a two-colour (black/red) palette. The single-model gridspec gains two extra thin rows between the main strip and the cross-section panels. `_draw_strip` is extended to return `pred_intervals` so the pred bar can be built from the same matching pass without a second `match_predictions_to_gt` call.

**Tech Stack:** matplotlib (imshow, ListedColormap), numpy

---

## Context

Current single-model layout in `render_artery` (`visualize.py:530-543`):
```
row 0 (ratio 3): longitudinal CT strip      → gs[0, :]
row 1 (ratio 2): cross-section panels       → gs[1, col]
figsize height: 7
```

Target layout:
```
row 0 (ratio 3.0): longitudinal CT strip   → gs[0, :]
row 1 (ratio 0.3): GT label bar            → gs[1, :]
row 2 (ratio 0.3): Pred label bar          → gs[2, :]
row 3 (ratio 2.0): cross-section panels    → gs[3, col]
figsize height: 8
```

Bars are 1-pixel-tall (resized to fill the thin row by `aspect='auto'`).
- **Black** (0) = normal (labels[z] == 0 for GT; z not covered by any predicted box)
- **Red**   (1) = abnormal (labels[z] > 0 for GT; z covered by ≥1 predicted box)

The comparison mode (two strips, `od_outputs2 is not None`) is **unchanged** — GT/Pred bars are added to single-model mode only.

---

### Task 1: Add `_draw_label_bar` helper and GT bar

**Files:**
- Modify: `visualize.py` — inside `render_artery()`, single-model layout block

**Background:**
`render_artery` has a single-model branch at line 537–543. The gridspec is:
```python
fig = plt.figure(figsize=(max(14, n_cs * 4), 7))
gs  = fig.add_gridspec(2, n_cs, height_ratios=[3, 2], hspace=0.4, wspace=0.3)
ax_long  = fig.add_subplot(gs[0, :])
ax_long2 = None
cs_row   = 1
```

**Step 1: Locate the single-model else-branch**

It's at `visualize.py:537`. The block to replace is:
```python
    else:
        fig = plt.figure(figsize=(max(14, n_cs * 4), 7))
        gs  = fig.add_gridspec(2, n_cs, height_ratios=[3, 2],
                               hspace=0.4, wspace=0.3)
        ax_long  = fig.add_subplot(gs[0, :])
        ax_long2 = None
        cs_row   = 1
```

Replace with:
```python
    else:
        fig = plt.figure(figsize=(max(14, n_cs * 4), 8))
        gs  = fig.add_gridspec(4, n_cs,
                               height_ratios=[3, 0.3, 0.3, 2],
                               hspace=0.4, wspace=0.3)
        ax_long   = fig.add_subplot(gs[0, :])
        ax_gt_bar  = fig.add_subplot(gs[1, :])
        ax_pred_bar = fig.add_subplot(gs[2, :])
        ax_long2  = None
        cs_row    = 3
```

**Step 2: Add `_draw_label_bar` inner function**

Insert this directly after the `_draw_strip` inner function closes (after `return tp_count, fn_count, fp_count` at ~line 643), still inside `render_artery`:

```python
    def _draw_label_bar(ax_bar, coverage_1d, label_text):
        """Render a thin 1×D label bar.

        coverage_1d: np.ndarray shape (D,), int; 0=normal(black), 1=abnormal(red)
        label_text:  short string shown on the y-axis tick ('GT' or 'Pred')
        """
        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(['black', 'red'])
        bar  = coverage_1d.reshape(1, -1)   # (1, D)
        ax_bar.imshow(bar, cmap=cmap, vmin=0, vmax=1,
                      aspect='auto', origin='upper')
        ax_bar.set_xticks([])
        ax_bar.set_yticks([0])
        ax_bar.set_yticklabels([label_text], fontsize=7)
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
```

**Step 3: Draw GT bar in single-model mode**

After the `_draw_strip(ax_long, ...)` call and before the comparison_mode block, add:

```python
    # ── Label bars (single-model mode only) ─────────────────────────────────
    if not comparison_mode:
        gt_coverage = (labels > 0).astype(int)   # shape (D,)
        _draw_label_bar(ax_gt_bar, gt_coverage, 'GT')
```

**Step 4: Smoke-test that GT bar renders (no crash)**

```bash
python visualize.py \
  --data_root ./dataset/test --pattern testing \
  --max_samples 1 \
  --output_dir ./viz_bar_test
```

Open `./viz_bar_test/*.png` — expect: main strip + thin black/red GT bar row + thin grey (all-black, no pred yet) row + CS panels.

> Note: `ax_pred_bar` will display nothing yet (it's undefined). The script will crash with `NameError: ax_pred_bar`. That's expected at this step.

Actually — to avoid crash, add a temporary placeholder in this step:
```python
    if not comparison_mode:
        gt_coverage = (labels > 0).astype(int)
        _draw_label_bar(ax_gt_bar, gt_coverage, 'GT')
        # placeholder until Task 2
        _draw_label_bar(ax_pred_bar, np.zeros(D, dtype=int), 'Pred')
```

**Step 5: Run smoke test**

```bash
python visualize.py \
  --data_root ./dataset/test --pattern testing \
  --max_samples 2 \
  --output_dir ./viz_bar_test
ls viz_bar_test/
```

Expected: 2 PNGs created without error. Open one — verify a thin GT bar appears below the main strip (red where disease labels exist, black elsewhere).

**Step 6: Commit**

```bash
git add visualize.py
git commit -m "feat: add GT/Pred label bars — layout + GT bar (Task 1)"
```

---

### Task 2: Add real Pred bar from predicted intervals

**Files:**
- Modify: `visualize.py` — `_draw_strip` return value + pred bar logic

**Background:**
`_draw_strip` currently returns `(tp_count, fn_count, fp_count)` at line 643. The `pred_intervals` list (one entry per surviving query, each `(x0_norm, x1_norm)`) is computed inside `_draw_strip` but not returned. We need it to build the pred bar.

**Step 1: Extend `_draw_strip` return value**

Locate the current return at `visualize.py:643`:
```python
        return tp_count, fn_count, fp_count
```

Replace with:
```python
        return tp_count, fn_count, fp_count, pred_intervals_out
```

And earlier in `_draw_strip`, just before the `if od_out is not None:` block, initialise:
```python
        pred_intervals_out = []
```

After the `match_predictions_to_gt` call, save it:
```python
            tp_idx, fn_idx, fp_idx, pred_intervals, surv_q = match_predictions_to_gt(
                od_out, segments, n_cls, D, iou_thresh=iou_threshold)
            pred_intervals_out = pred_intervals
```

The full `_draw_strip` signature and structure stays the same; only the init + return changes.

**Step 2: Update `_draw_strip` call sites**

There are two call sites:

1. Single-model call at ~line 646:
```python
    _draw_strip(ax_long, od_outputs, num_classes, label1, stenosis_pred)
```
Replace with:
```python
    _, _, _, pred_ivs = _draw_strip(ax_long, od_outputs, num_classes, label1, stenosis_pred)
```

2. Comparison-mode call at ~line 648:
```python
        _draw_strip(ax_long2, od_outputs2, num_classes2, label2, stenosis_pred2)
```
Replace with:
```python
        _draw_strip(ax_long2, od_outputs2, num_classes2, label2, stenosis_pred2)
```
(comparison mode discards return value — unchanged)

**Step 3: Build pred coverage and draw Pred bar**

Replace the Task 1 placeholder `_draw_label_bar(ax_pred_bar, np.zeros(...), 'Pred')` with:

```python
    if not comparison_mode:
        gt_coverage = (labels > 0).astype(int)
        _draw_label_bar(ax_gt_bar, gt_coverage, 'GT')

        pred_coverage = np.zeros(D, dtype=int)
        for x0_norm, x1_norm in pred_ivs:
            start = max(0, int(x0_norm * D))
            end   = min(D, int(x1_norm * D) + 1)
            pred_coverage[start:end] = 1
        _draw_label_bar(ax_pred_bar, pred_coverage, 'Pred')
```

**Step 4: Smoke test with a model checkpoint**

```bash
python visualize.py \
  --data_root ./dataset/test --pattern testing \
  --checkpoint checkpoints_v7_finetune/final_model.pth \
  --thresholds calibration_thresholds_v7_constrained.json --use_constrained \
  --max_samples 3 \
  --output_dir ./viz_bar_test2
```

Expected output per PNG:
- Row 0: CT longitudinal strip with GT bands + TP/FN/FP markers
- Row 1 (GT bar): thin strip — black where healthy, red where GT lesion exists
- Row 2 (Pred bar): thin strip — black where no prediction fires, red where ≥1 prediction box covers the slice
- Row 3: cross-section panels

Verify by opening a PNG and checking that:
- GT bar's red segments align with the coloured GT bands in the strip above
- Pred bar's red segments appear near (but possibly offset from) GT bands — discrepancies visible as red-in-GT-only or red-in-Pred-only

**Step 5: Verify GT-only mode still works**

```bash
python visualize.py \
  --data_root ./dataset/test --pattern testing \
  --max_samples 2 \
  --output_dir ./viz_bar_nomodel
```

Expected: no crash; GT bar appears (red where disease); Pred bar is all-black (no predictions). No `NameError` for `pred_ivs`.

**Step 6: Verify comparison mode is unbroken**

```bash
python visualize.py \
  --data_root ./dataset/test --pattern testing \
  --checkpoint  checkpoints_v6/best_model.pth --model_pattern pre_training \
  --checkpoint2 checkpoints_v7_finetune/final_model.pth \
  --thresholds2 calibration_thresholds_v7_constrained.json --use_constrained2 \
  --max_samples 2 \
  --output_dir ./viz_bar_compare
```

Expected: comparison mode renders 2-strip layout as before, no GT/Pred bars (bars only added for single-model mode), no crashes.

**Step 7: Commit**

```bash
git add visualize.py
git commit -m "feat: add Pred label bar — shows predicted vs GT abnormality per slice"
```

---

### Task 3: Update docstring and run full test set

**Files:**
- Modify: `visualize.py` — module docstring (lines 1–29)

**Step 1: Update the module docstring**

The module docstring at the top of `visualize.py` (lines 6–28) currently shows three usage examples. Add a new example block describing the GT/Pred bar feature and the recommended single-model invocation:

Replace the existing "# With model predictions" block with text that notes the GT/Pred bars:

```
  # Single-model with GT/Pred bars (recommended for analysis)
  python visualize.py --data_root ./dataset/test --pattern testing \
      --checkpoint checkpoints_v7_finetune/final_model.pth \
      --thresholds calibration_thresholds_v7_constrained.json --use_constrained \
      --output_dir ./viz_v7ft
  #
  #  Output layout per PNG:
  #    Row 0: longitudinal CT strip with GT bands + TP/FN/FP boxes
  #    Row 1: GT label bar  — black=normal, red=abnormal (per-slice GT)
  #    Row 2: Pred label bar — black=no prediction, red=prediction fires here
  #    Row 3: cross-section panels (up to 4 labelled segments)
```

**Step 2: Run full test set with latest model**

```bash
python visualize.py \
  --data_root ./dataset/test --pattern testing \
  --checkpoint checkpoints_v7_finetune/final_model.pth \
  --thresholds calibration_thresholds_v7_constrained.json --use_constrained \
  --output_dir ./viz_v7ft_bars
```

Expected: 67 PNGs, no crashes. Spot-check a few — verify GT/Pred bars look sensible.

**Step 3: Commit**

```bash
git add visualize.py
git commit -m "docs: update visualize.py docstring for GT/Pred label bars"
```

---

## Summary of Changes

| File | Lines changed | What |
|------|--------------|------|
| `visualize.py` | ~537–543 | Single-model gridspec: 2→4 rows, add `ax_gt_bar`, `ax_pred_bar`, `cs_row=3` |
| `visualize.py` | ~643 (inside `_draw_strip`) | Add `pred_intervals_out = []` init + save after match; extend return to 4-tuple |
| `visualize.py` | ~646 | Unpack 4-tuple from `_draw_strip` call in single-model mode |
| `visualize.py` | ~648+ | Draw GT bar + Pred bar in single-model mode (not comparison mode) |
| `visualize.py` | ~545+ | Add `_draw_label_bar(ax_bar, coverage_1d, label_text)` inner function |
| `visualize.py` | 6–28 | Update module docstring |

**Comparison mode**: completely unchanged.
**GT-only mode** (no `--checkpoint`): GT bar appears (red = disease), Pred bar all-black.
