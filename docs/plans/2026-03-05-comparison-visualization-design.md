# Before/After Comparison Visualization Design
Date: 2026-03-05

## Purpose

Extend `visualize.py` with a `--checkpoint2` comparison mode that places two models'
predictions side-by-side in the same PNG. Designed to showcase the full pipeline
improvement: **pre-trained only** (before) vs **fine-tuned with constrained calibration**
(after).

## Data & Model Context

- "Before" checkpoint: `checkpoints_v6/best_model.pth` (pre-training only, 3-class)
- "After"  checkpoint: `checkpoints_v7_finetune/final_model.pth` + `calibration_thresholds_v7_constrained.json`
- CPR volumes: `(256, 64, 64)` after transpose; vessel axis = dim 0 (256 slices)
- Labels: 256-int array, raw values 0–6; non-zero segments → GT lesion intervals

## Figure Layout (comparison mode)

```
┌────────────────────────────────────────────────────────────────────────┐
│ Title: APNHC00002_LAD  |  GT: Non-significant                          │
├────────────────────────────────────────────────────────────────────────┤
│ Strip 1: "Pre-trained (no fine-tuning)"                                │
│   GT bands (filled, alpha=0.35) — always shown                         │
│   FN: GT band + diagonal hatch (/'/) — GT segment with no matched pred │
│   TP: solid-border box, class colour — matched prediction              │
│   FP: dashed orange box — prediction with no GT match                  │
│   Strip title: "Pre-trained | Pred: Healthy ✗ | TP:0 FN:1 FP:0"       │
├────────────────────────────────────────────────────────────────────────┤
│ Strip 2: "Fine-tuned v7 (constrained cal.)"                            │
│   Same GT bands                                                         │
│   TP: solid-border box — correctly fires on GT segment                 │
│   Strip title: "Fine-tuned | Pred: Non-sig ✓ | TP:1 FN:0 FP:0"        │
├────────┬────────┬────────┬───────────────────────────────────────────  │
│ Cross-section panels (shared, up to 4)                                 │
│   Border colour encodes TP/FN outcome across both models:              │
│     green  = TP in fine-tuned model (improvement achieved)            │
│     red    = FN in both models      (both missed this segment)         │
│     orange = FN in model 1 only     (improvement from fine-tuning)    │
│     white  = no GT at this position (healthy / background)            │
└────────────────────────────────────────────────────────────────────────┘
```

When `--checkpoint2` is absent the tool renders identically to the existing
single-model mode (no regression).

## TP/FN/FP Matching Logic (1D IoU on vessel axis)

For each model independently:

1. Filter queries: skip no-object class (`cls >= num_classes`) and low-confidence (`prob < 0.15`)
2. Convert each surviving prediction `(cx_norm, w_norm)` → 1D interval `[cx-w/2, cx+w/2]`
3. Convert each GT segment `(start, end)` → normalised interval `[start/D, end/D]`
4. Compute 1D IoU between every pred–GT pair
5. **TP**: prediction whose best-matching GT interval has IoU ≥ `iou_threshold` (default 0.3)
6. **FN**: GT segment whose best-matching prediction has IoU < `iou_threshold`
7. **FP**: prediction whose best-matching GT interval has IoU < `iou_threshold`

Returns: `(tp_preds, fn_segments, fp_preds)` — indices into the pred/segment lists.

## New CLI Arguments

| Argument           | Default    | Description                                        |
|--------------------|------------|----------------------------------------------------|
| `--checkpoint2`    | None       | "After" model checkpoint path                      |
| `--thresholds2`    | None       | Calibration JSON for model 2                       |
| `--use_constrained2` | False    | Use `constrained_stenosis_thresholds` from JSON 2  |
| `--label`          | `"Model A"`| Display name for model 1 (strip 1 title)           |
| `--label2`         | `"Model B"`| Display name for model 2 (strip 2 title)           |
| `--iou_threshold`  | `0.3`      | 1D IoU cutoff for TP/FN/FP classification          |

## `render_artery` Signature Extension

```python
render_artery(
    artery_id, volume, labels, save_path,
    # existing params (unchanged)
    stenosis_pred=None, plaque_pred=None, od_outputs=None, num_classes=6,
    # new comparison params
    stenosis_pred2=None, plaque_pred2=None, od_outputs2=None, num_classes2=6,
    label1="Model A", label2="Model B",
    iou_threshold=0.3,
)
```

When `od_outputs2 is None`: renders single-strip layout (existing behaviour, no change).

## Helper Functions to Add

- `match_predictions_to_gt(od_outputs, gt_segments, num_classes, D, conf_thresh, iou_thresh)`
  → Returns `(tp_pred_indices, fn_seg_indices, fp_pred_indices, pred_intervals_norm)`
- `iou_1d(a_start, a_end, b_start, b_end)` → float

## Filter Behaviour in Comparison Mode

- `--filter incorrect` → keep arteries where **either** model is wrong
- `--filter correct` → keep arteries where **both** models are correct
- Class filters (`healthy`, `nonsig`, `sig`) apply to GT class as before

## Output Filename in Comparison Mode

`{artery_id}__sten_{sten_tag}_m1_{pred1_tag}_{m1_result}_m2_{pred2_tag}_{m2_result}.png`

e.g. `APNHC00002_LAD__sten_NonSig_m1_Healthy_WRONG_m2_NonSig_CORRECT.png`

## Colour Reference

| Marker            | Visual                                      |
|-------------------|---------------------------------------------|
| GT band           | Filled, alpha=0.35, per raw-label colour    |
| FN hatch          | `hatch='///'`, same colour as GT band       |
| TP box            | Solid border, width=2, pred class colour    |
| FP box            | Dashed orange border, width=1.5             |
| CS border: green  | TP in model 2 (fine-tuned correctly detects)|
| CS border: orange | FN in model 1 but TP in model 2             |
| CS border: red    | FN in both models                           |
| CS border: white  | No GT at this z-position                    |
