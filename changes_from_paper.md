# SC-Net: Changes from the Original Paper Code
**Prepared by:** Reet Mitra | **Date:** 11 May 2026
**Scope:** A complete account of every deviation from the paper authors' released code — what was kept, what was fixed, what was reverted after the paper-equations discovery, what was deliberately changed, and what was built from scratch.

---

## Overview

The original paper codebase (https://github.com/PerceptionComputingLab/SC-Net) was used as the starting point. It contained the conceptual skeleton of SC-Net: the dual-branch architecture, the loss function definitions, and the data loading structure. However, the code could not run at all — it crashed immediately on GPU, had no training loop, and contained numerous silent bugs that would have corrupted training even if it could run. Our work falls into five distinct categories:

| Category | Description |
|----------|-------------|
| **A. Kept** | Elements of the original code retained without modification |
| **B. Fixed** | Bugs that caused crashes or silent corruption — all strictly correct fixes |
| **C. Reverted** | "Fixes" we initially applied to match the paper's equations, then had to undo because the paper's equations differ from the paper's actual code |
| **D. Changed** | Deliberate architectural and algorithmic improvements beyond crash-fixing |
| **E. Added** | Entire subsystems written from scratch that did not exist in the original code |

---

## A. What We Kept

These elements were taken from the original code and remain in their original form (or very close to it). They represent the authors' design decisions that were conceptually correct and worked as intended.

### A.1 Overall Architecture Design

The fundamental two-branch structure of SC-Net was kept as designed. The temporal branch (32 cubic crops → 3D-CNN → Transformer encoder → per-point classification) and the spatial branch (full CPR volume + 4 multi-view 2D projections → CNN feature extraction → DETR-style Transformer decoder with 16 learnable object queries → bounding box regression + classification) are both inherited from the paper. The high-level design — processing the same vessel at two scales simultaneously and having them supervise each other — is the paper's core contribution and was not altered in concept.

### A.2 The Dual-Task Contrastive Loss Formulation

The mathematical formulation of L_dc was kept exactly:

```
L_total = L_od + L_sc + δ × L_dc
L_dc = L_od(C(ŷ_sc), ŷ_od) + L_sc(C⁻¹(ŷ_od), ŷ_sc)
```

Where each branch's predictions are converted into pseudo-labels for the other via the mapping functions C(·) and C⁻¹(·). This mutual self-supervision under limited labeled data is the paper's central innovation and remains structurally unchanged. The DC warmup schedule concept (holding DC at zero for initial epochs then ramping it in) also originated in the paper code.

### A.3 Loss Weights — 1:1:1 (Paper Code, Not Paper Equations)

The 1:1:1 weighting of all loss components — `L1 + GIoU` with no scaling in `loss_boxes`, and `cost_class=1, cost_bbox=1, cost_giou=1` in the Hungarian matcher — was retained as the paper code actually implements it. This is a critical distinction: the paper's equations state λ_L1=5 and λ_iou=2, but the code the authors trained with uses uniform 1:1:1 weights throughout. We discovered this empirically when applying the equations broke convergence entirely (see Section C).

### A.4 Hungarian Matching Framework

The bipartite matching approach in `HungarianMatcher` — using the scipy linear sum assignment solver to optimally pair predicted queries to ground-truth lesion intervals — was kept from the original code. The matching cost matrix formulation (class cost + box L1 cost + GIoU cost) is the standard DETR approach and was correct in concept.

### A.5 Data Format and Label Encoding

The NIfTI volume format (256×64×64 CPR volumes), the 256-line label text files (one label per slice), the 0–6 label encoding (0=background, 1–3=Non-significant plaque types, 4–6=Significant plaque types), and the two-stage training concept (pre-training on 3-class plaque composition, fine-tuning on 6-class stenosis+plaque) were all inherited from the original code and kept unchanged. The label→interval box conversion logic (identifying contiguous runs of the same non-zero label and converting to normalized `[center, width]` format) was also retained.

### A.6 CT Windowing

The HU windowing approach — clipping intensities to a defined window and normalising to [0, 1] — was kept from the original code. The window values were corrected to match what the config actually computes (`[-150, 750]` rather than the erroneous `[-200, 800]` default), but the method itself was not changed.

### A.7 Multi-View 2D Projections

The spatial branch's use of four 2D projection views (sagittal, coronal, and two diagonal projections of the CPR volume) alongside the 3D input was kept as designed. The paper argues these projections give the model complementary cross-sectional information about the coronary lumen, and we retained this without modification.

---

## B. What We Fixed

These are bugs that caused crashes, silent gradient corruption, or silent data loading errors. Every fix in this section is unambiguously correct — there is no debate about whether these should have been fixed.

### B.1 nn.ModuleList for Extraction Blocks (`architecture.py`)

**Original code:** The 3D-CNN and 2D-CNN extraction blocks across the four pyramid levels were stored in plain Python lists inside the model class.

**Problem:** PyTorch's optimizer and device-management machinery only discover parameters through `nn.Module.parameters()`, which walks the module tree. Parameters stored in plain Python lists are completely invisible to this traversal. Consequently, the extraction block weights received no gradient updates and could not be moved to GPU with `.to(device)`. The spatial branch — the entire CNN feature extraction backbone — was completely untrained in the original code. Every forward pass used randomly initialised, CPU-resident weights.

**Fix:** All extraction block lists replaced with `nn.ModuleList`. This is a standard PyTorch requirement for any collection of submodules inside a module class.

### B.2 Feature Fusion Weights as nn.Parameter (`architecture.py`)

**Original code:** The weights for fusing 3D and 2D features (`_3d_weight=0.75`, `_2d_weight=[0.25, 0.25, 0.25, 0.25]`) were created as plain tensors inside `_2d_maps_to_3d_maps`. Specifically, they were instantiated as `torch.tensor(...)` inside the `forward()` method on every call.

**Problem:** Creating tensors in `forward()` means they are freshly allocated on the CPU every time the method runs, regardless of where the model lives. This caused an immediate device mismatch crash on GPU. Additionally, even if the crash were avoided, these tensors would not have been tracked by the optimizer, so the fusion ratios would have remained fixed at 0.75/0.25 forever rather than being learned.

**Fix:** Converted to `nn.Parameter` so they are registered as learnable parameters, moved to the correct device with the model, and updated during training.

### B.3 Learned Object Query Embeddings (`architecture.py`)

**Original code:** The 16 object queries fed into the DETR-style Transformer decoder were generated using `torch.randint(0, num_queries, size=[num_queries])` inside `forward()`, producing random integer indices on every call, which were then passed through an embedding table.

**Problem:** Because the indices changed every forward pass, the decoder queries were different vectors at every step. The decoder could never learn stable, specialised query representations — each query had no persistent identity across training steps. This is a fundamental violation of the DETR design principle, where each query learns to detect a specific type of object. The original code effectively disabled the decoder's ability to develop query specialisation.

**Fix:** The queries were replaced with a fixed `nn.Embedding(num_queries, embed_dim)` initialised once in `__init__` and used identically on every forward pass. This is the standard DETR implementation.

### B.4 Spatial Flattening Projection (`architecture.py`)

**Original code:** A `Conv3d` layer for compressing the spatial feature map before flattening was defined in `__init__` but never actually called in `forward()`. The `rearrange` pattern used for flattening also produced incorrect token dimensions.

**Problem:** The spatial branch's feature map was being fed into the Transformer decoder at the wrong size. The tokens were not the 512-dimensional vectors the decoder expected, causing shape mismatches or silent misalignment between what the paper describes (16 spatial tokens of 512 dimensions) and what the code produced.

**Fix:** The `Conv3d(128→16)` compression layer is now called in `forward()` before flattening, correctly producing 16 spatial tokens of the expected 512-dimensional embedding. The rearrange pattern was corrected to match the actual tensor layout after the convolution.

### B.5 Gradient Detachment in Dual-Task Contrastive Loss (`optimization.py`)

**Original code:** In `dual_task_contrastive_loss.forward()`, the raw model outputs from both branches (with computation graphs attached) were passed directly into the pseudo-label generation functions.

**Problem:** The dual-task contrastive loss is supposed to use each branch's predictions as supervision for the other — detached predictions that do not carry gradients back to the source branch. Without detachment, L_dc was computing gradients that flowed simultaneously through both branches in a circular pattern: the OD branch's loss depended on SC outputs, and the SC branch's loss depended on OD outputs, creating an entangled gradient graph that could not represent the intended independent supervision signal. This corrupted the paper's core novelty — the mutual pseudo-label mechanism — in every training step.

**Fix:** Added `.detach()` to branch outputs before they are used to generate pseudo-labels for the other branch. Each branch receives clean, gradient-free supervision from its peer.

### B.6 FocalLoss Alpha as Registered Buffer (`optimization.py`)

**Original code:** The `alpha` class-weights tensor in `FocalLoss` was stored as a plain attribute: `self.alpha = alpha`.

**Problem:** When the model was moved to GPU via `.to(device)`, the `FocalLoss` module moved with it — but plain attributes are not tracked by PyTorch's device-transfer machinery. Only tensors registered as buffers or parameters follow the module to its new device. At inference time, `self.alpha` remained on CPU while the input activations were on GPU, causing an immediate `RuntimeError: Expected all tensors to be on the same device`.

**Fix:** `self.register_buffer('alpha', alpha)` — the alpha tensor is now automatically transferred when the module is moved to a different device.

### B.7 Deep Copy Targets Before Each Loss Term (`optimization.py`)

**Original code:** In `spatio_temporal_contrast_loss.forward()`, the same `od_targets` list was passed sequentially to `object_detection_loss`, then to `dual_task_contrastive_loss`, then used again for DC weight computation.

**Problem:** `boxes_dimension_expansion` — called inside the loss computation — mutates the target tensors in-place, expanding 1D `[center, width]` boxes to 4D. Once mutated, the targets passed to the second and third loss calls were already expanded, causing incorrect geometry in subsequent computations. The corruption was silent — no crash, just wrong gradients in L_dc.

**Fix:** A deep copy of the targets is made before each of the three loss calls: `[{k: v.clone() for k, v in t.items()} for t in od_targets]`. Each loss term receives an independent copy of the original targets.

### B.8 Device-Aware Target Tensors (`optimization.py`)

**Original code:** `od2sc_targets()` and `sc2od_targets()` — the functions that convert detection targets to classification targets and vice versa — created output tensors using `torch.zeros(...)` and `torch.tensor(...)` without specifying a device.

**Problem:** These tensors were always created on CPU. When the loss function computed operations involving these targets alongside GPU-resident model outputs, a device mismatch error was triggered. In configurations where the error didn't immediately crash (e.g., if some operations happen to tolerate mixed devices), the CPU tensors would silently bottleneck training by forcing data transfer.

**Fix:** Explicit `.to(device)` calls after creation, inheriting the device from the model outputs already in scope.

### B.9 Empty Tensor Shape in box_lastdim_expansion (`functions.py`)

**Original code:** When `box_lastdim_expansion` received an empty tensor (zero lesions in a healthy artery), it returned a tensor of shape `(0, 2)` rather than `(0, 4)`.

**Problem:** Downstream operations — `HungarianMatcher` calling `torch.cdist`, and `torch.cat` combining predictions with targets across batch elements — all assumed 4-column tensors. An empty `(0, 2)` tensor caused a shape mismatch crash. Since healthy arteries with no lesions are common in real clinical data, this crash would occur regularly.

**Fix:** Guard the empty case: return `torch.zeros((0, 4), dtype=torch.float32)` when the input has zero rows.

### B.10 Degenerate Box Assert Replaced with Clamping (`functions.py`)

**Original code:** `generalized_box_iou` contained a hard `assert (boxes[:, 2:] >= boxes[:, :2]).all()` to verify box validity before IoU computation.

**Problem:** During long training runs, stochastic mini-batches occasionally produce degenerate boxes (e.g., predicted width near-zero after softmax) where the assertion fails. This crashed training at epoch 129 of v2 — after many hours of compute. An assert that kills a long training run for an edge case is unacceptable in practice.

**Fix:** The assert was replaced with `torch.cat` clamping that enforces `x2 >= x1` without crashing. Degenerate boxes produce zero IoU, which is correct behaviour.

### B.11 In-Place Box Operation Breaks AMP (`functions.py`)

**Original code:** After the degenerate box fix, a box coordinate clamping operation was applied in-place.

**Problem:** PyTorch's Automatic Mixed Precision (AMP) maintains a computation graph for fp16/fp32 gradient scaling. In-place operations on tensors that are part of this graph invalidate the saved intermediate activations, causing `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`. This only manifested after AMP was added.

**Fix:** Replaced the in-place operation with `torch.cat(...)` to construct a new tensor, leaving the original untouched.

### B.12 Dataset Index Offset in __getitem__ (`augmentation.py`)

**Original code:** `cubic_sequence_data.__getitem__(index)` used `index` directly to retrieve the file at position `index` in the full file list.

**Problem:** The dataset class stores a `data_start` offset that determines which subset of the files each split (train/val/test) should access. Using the raw `index` without adding `data_start` meant the validation set was loading samples from the beginning of the training set — it was evaluating training data and calling it validation. Early stopping was therefore monitoring training performance, creating a false impression of the model's generalisation and preventing correct checkpoint selection.

**Fix:** Changed to `index + data_start` so each data split correctly accesses its designated portion of the file list.

### B.13 _3d_cubes_selection Device Inheritance (`functions.py`)

**Original code:** `_3d_cubes_selection` created its output tensor using `torch.zeros(...)` without specifying device or dtype, defaulting to CPU float32.

**Problem:** The input to this function was a GPU tensor, but the output was always a CPU tensor. Any subsequent operation involving both would immediately cause a device mismatch.

**Fix:** Output tensor created with `torch.zeros(..., device=input.device, dtype=input.dtype)` to inherit device and dtype from the input.

### B.14 torch.torch.float32 Typo (`augmentation.py`)

**Original code:** A dtype specification was written as `torch.torch.float32`.

**Problem:** `torch.torch` does not exist. This would raise `AttributeError` at any code path that reached this line.

**Fix:** Corrected to `torch.float32`.

### B.15 torch.load Without map_location (`framework.py`)

**Original code:** Checkpoint loading in `pre_training_load()` called `torch.load(path)` without specifying a `map_location`.

**Problem:** When a checkpoint saved from GPU 0 is loaded without `map_location`, PyTorch attempts to restore tensors to their original device. If that device is not available, or if loading is happening on a machine with a different GPU layout, this causes a device mismatch error. In multi-GPU DDP setups this was an especially common crash point.

**Fix:** `torch.load(path, map_location='cpu')` loads the checkpoint onto CPU first; the caller then moves it to the correct device. This is the robust standard practice.

### B.16 spatial_proj_channels Dimension Mismatch (`config.py`)

**Original code:** `DefaultConfig.spatial_proj_channels` was set to `[128, 1024, 128, 512]`.

**Problem:** After 4 pooling levels, the actual feature map dimensions are 16×4×4 = 256 spatial elements (not 1024), and the projection should produce 16 spatial tokens (not 128) of 512 dimensions. The mismatched config caused shape errors in the spatial flattening projection.

**Fix:** Corrected to `[128, 256, 16, 512]` to match the actual post-pooling tensor dimensions.

### B.17 Label Offset Corruption in DC Loss (`optimization.py`)

**Original code:** `dual_task_contrastive_loss._get_sampling_point_classification_targets()` converted OD outputs to SC pseudo-labels via `labels = torch.argmax(selected_logits, dim=1) - 1` followed by `clamp(min=0)`.

**Problem:** This systematically corrupted every pseudo-label. The -1 offset was intended to convert from OD class indices to SC class indices, but `od2sc_targets()` already handles this conversion internally by adding +1. The result was that OD class 0 coincidentally mapped correctly (0-1=-1, clamped to 0, +1=1), but OD class 1 mapped to the wrong SC class (1-1=0, +1=1 instead of 2), and OD class 2 similarly mapped incorrectly. The no-object class (index = num_classes) was also not filtered, resulting in background pseudo-labels contaminating foreground supervision. This silently corrupted L_dc — the paper's core contribution — in every training step.

**Fix:** The -1 offset removed; foreground/background filtering added (no-object predictions are excluded from pseudo-label generation); 0-indexed class labels passed directly to `od2sc_targets` which applies the correct +1 shift internally.

### B.18 Loss Function Returns Scalar Only (`optimization.py`)

**Original code:** `spatio_temporal_contrast_loss.forward()` returned a single scalar — the total combined loss.

**Problem:** With no visibility into individual loss components, it was impossible to diagnose which term was dominating or exploding during training. This made debugging training instabilities (particularly during the DC warmup ramp) significantly harder.

**Fix:** The forward pass now returns a dictionary `{'total': ..., 'od': ..., 'sc': ..., 'dc': ...}`. The training loop unpacks `loss_dict['total']` for backpropagation and logs each component to TensorBoard separately.

---

## C. What We Reverted

This section documents the most important discovery of the project: a set of "fixes" we initially applied to match the paper's equations that we subsequently had to undo because the paper's equations do not match the paper's actual code.

### C.1 The Discovery

In February 2026, after convergence failures across v2–v4 fine-tuning, we conducted a direct comparison of our code against the original paper repository. The comparison revealed a fundamental discrepancy:

**Paper equations (Section 3.3, Eq. 5) state:** λ_L1=5, λ_iou=2 for the object detection box loss, and implicitly the same ratios for the Hungarian matching costs.

**Paper's actual code uses:** 1:1:1 weights throughout — `L1 + GIoU` with no scaling coefficients, and `cost_class=1, cost_bbox=1, cost_giou=1` in the Hungarian matcher.

The paper's authors achieved their reported 91.4% stenosis accuracy with the code, not with the equations. The equations in the paper text appear to reflect an earlier design stage that was not what the final model was trained with.

### C.2 Box Loss Weights Reverted (5:2 → 1:1)

**What we had changed:** We applied the paper's equations directly: `loss_boxes = 5.0 * L1_loss + 2.0 * GIoU_loss` in `optimization.py`.

**Why we reverted:** This made the box regression loss approximately 3.5× larger than what the paper actually trained with, overwhelming the classification loss and driving all gradients toward box geometry optimisation. The model collapsed into majority-class prediction — predicting the same class for every artery — because the classification signal was drowned out. Reverting to `L1 + GIoU` (1:1) restored correct loss balance.

**Current state:** `L1_loss + GIoU_loss` — matching the paper code.

### C.3 Hungarian Matching Costs Reverted (5:2 → 1:1:1)

**What we had changed:** We set `cost_bbox=5, cost_giou=2` in `HungarianMatcher` to match the paper equations.

**Why we reverted:** The matching cost matrix determines which predicted query gets paired with which ground-truth lesion during each training step. With 5:2 costs, the matcher strongly preferred geometric closeness over class accuracy when pairing queries to targets. This produced bipartite matchings that were geometrically reasonable but class-confused, leading to noisy classification gradients. Reverting to `cost_class=1, cost_bbox=1, cost_giou=1` — the paper code's actual values — restored stable training.

**Current state:** All matching costs equal to 1.0 — matching the paper code.

### C.4 Box Expansion Geometry — Partial Revert

**What we had initially changed:** We corrected `box_lastdim_expansion` to expand `[cx, w]` → `[cx, 0.5, w, 1.0]` (a full-height 1D interval centred at y=0.5). The paper code expanded to `[cx, w, cx, w]` which, after reindexing as `[x1, y1, x2, y2]`, produced `[cx, w, cx, w]` — square boxes where the y-coordinates happened to equal the vessel-axis coordinates.

**The revert decision:** After the paper-equations discovery, we classified this correction as a "behavioral change" and initially reverted to the paper code's square-box expansion to match the loss landscape the paper was trained on.

**What happened after:** In v10, we reintroduced the geometrically correct `[cx, 0.5, w, 1.0]` expansion as a deliberate architectural choice — now explicitly called "native 1D IoU" — paired with the 1:1 loss weights. This combination worked correctly, whereas the earlier combination of correct geometry + 5:2 weights had not. The key was that the loss weights, not the box geometry, were the root cause of the original convergence failure.

**Current state:** `[cx, 0.5, w, 1.0]` (correct 1D interval geometry) with 1:1 loss weights. This is a deviation from the paper code, but a correct one that is paired with the weight regime the paper code used.

---

## D. What We Deliberately Changed

These are intentional modifications and improvements to the existing code — not crash fixes, but targeted enhancements to architecture quality, training stability, and clinical performance. Each was introduced at a specific version and motivated by a concrete observed failure or limitation.

### D.1 Temporal Branch Sequence Shape Fix (`architecture.py`)

**Original code:** `temporal_semantic_learning.forward()` reshaped all 32 cubic crops into a flattened batch before passing them to `_3dcnn`. Concretely: `[B, 32, 1, D, H, W]` was reshaped to `[B×32, 1, D, H, W]`, making each cube an independent sample. After the CNN, the sequence length passed to the Transformer encoder was effectively 1 (all cubes were treated as independent items in the batch, not as a sequence).

**Problem:** The Transformer encoder in the temporal branch is supposed to attend across all 32 vessel positions simultaneously — that is the entire point of the temporal branch. With a sequence length of 1, the self-attention mechanism had nothing to attend to. The temporal branch functioned purely as a bag of independent cubes with no cross-position reasoning. The Transformer was computationally present but functionally disabled.

**Fix:** The input is passed as `[B, 32, D, H, W]` directly to `_3dcnn`. The CNN internally handles the `(B×n_cubes)` reshaping for convolution and then restores the sequence dimension, producing `[B, 32, C_out, d', h', w']`. The Transformer encoder now receives a proper 32-element sequence and can attend across all vessel positions from proximal to distal.

**Impact:** The temporal branch was restored to its intended function. This fix was silent — no crash, no obvious symptom — which is what made it particularly dangerous. The transformer was running but doing nothing meaningful.

### D.2 SE Fusion Gates for 3D/2D Feature Blending (`architecture.py`)

**Original code:** The spatial branch fused 3D-CNN and 2D-CNN features at each pyramid level using fixed scalar weights: `_3d_weight=0.75` and `_2d_weight=0.25` (uniform across all four 2D views). The scalars were the same at all four pyramid levels.

**Problem:** The relative importance of 3D volumetric information versus 2D cross-sectional projections varies by pyramid level and by artery region. High-level semantic features may benefit from different 3D/2D ratios than low-level texture features. Fixed uniform scalars prevent the model from learning these optimal ratios.

**Fix:** Squeeze-and-excitation (SE) fusion gates were introduced at each pyramid level. Each gate applies channel-wise learned weights to the concatenated 3D and 2D features before summing, allowing the model to learn content-adaptive blending. This replaces the fixed scalars with a small learned network (global average pooling → two linear layers → sigmoid) at each level.

**Impact:** The SE gates allow the model to upweight, for example, the 2D cross-sectional views when calcified plaque produces high-contrast features in the axial plane, and upweight 3D features when the spatial extent of a diffuse lesion is more informative.

### D.3 Parallel 2D/3D CNN Streams (`architecture.py`)

**Original code:** The 2D CNN blocks at pyramid levels 2, 3, and 4 received the output of the 3D CNN block at the same level as their input, rather than their own prior 2D output. This created an interleaved rather than truly parallel architecture.

**Problem:** When 2D blocks process 3D-derived features, the two streams are not independent — the 2D path becomes a correction branch on top of the 3D path rather than an alternative viewpoint. The paper describes parallel streams; the code implemented a serial dependency.

**Fix:** The 2D blocks are restructured so that each level's 2D block receives the output of the previous level's 2D block (establishing a true 2D feature hierarchy), while the 3D blocks maintain their own chain. The two streams merge only at the SE fusion gate, which is the intended design.

**Impact:** Greater feature diversity across the two paths. The 3D and 2D streams learn genuinely complementary representations rather than the 2D stream simply refining the 3D stream's output.

### D.4 Learnable Positional Encoding (`architecture.py`)

**Original code:** The temporal branch used fixed sinusoidal positional encoding (if any), which encodes position as a predetermined function of frequency without any learned component.

**Fix:** Replaced with `nn.Parameter` of shape `[seq_length, embed_dim]` — a fully learnable positional encoding. For the temporal branch's short sequence (32 cubic crops along the vessel), learnable encodings are better suited than sinusoidal: the model can directly learn that proximal and distal vessel positions have different diagnostic relevance, rather than relying on a generic frequency basis.

**Impact:** The model can develop vessel-specific position representations that reflect domain knowledge (e.g., the LAD proximal segment is the most clinically critical region).

### D.5 Ordinal EMD Loss (`optimization.py`)

**Original code:** The temporal branch classification loss was standard cross-entropy (or focal cross-entropy after our additions), which treats all misclassification errors equally regardless of severity direction.

**Problem:** In the clinical stenosis context, confusing Healthy with Non-significant is a much smaller error than confusing Healthy with Significant. Cross-entropy penalises both equally. Worse, confusing Significant with Non-significant (under-escalation — potentially missing critical disease) is penalised the same as confusing Significant with Healthy (an implausible jump over one class).

**Fix:** An Ordinal Earth Mover's Distance (EMD) loss was added as a secondary term for the temporal branch. EMD computes the "work" required to transform one probability distribution into another over the ordered class sequence. For three ordered classes {Healthy, Non-sig, Significant}, it penalises Healthy↔Significant errors approximately twice as heavily as adjacent-class errors (Healthy↔Non-sig, Non-sig↔Significant). The total temporal branch loss becomes:

```
L_sc = L_focal_CE + ordinal_weight × L_EMD
```

Controlled via `--ordinal_weight` (0 = disabled; 0.5 in v12–v15; 1.5 in v16).

**Impact:** The ordinal loss specifically targets the severity-order violation errors that are clinically most dangerous. In v16, increasing ordinal_weight from 0.5 to 1.5, combined with boost_sig, reduced Sig→Non-sig misses from 40 to 22 (a 45% reduction in the most dangerous error type).

### D.6 Significant Class Weight Boosting (`optimization.py`, `train.py`)

**Original code:** The SC branch class weights treated all non-background lesion classes uniformly.

**Rationale:** In the 6-class fine-tuning setting, Significant stenosis maps to indices 4, 5, 6 (Sig+Calcified, Sig+Non-calcified, Sig+Mixed). Clinically, missing a Significant artery is a high-consequence error. The model was under-predicting Significant relative to its true prevalence because the ordinal weight alone was insufficient.

**Fix:** `boost_sig=True` doubles the class weight for indices 4–6 in `compute_sc_class_weights`. This mirrors the existing `boost_nonsig` pattern that boosted Non-significant class weights. The 2× boost acts directly on the focal loss, penalising Sig→Non-sig misclassifications more heavily at every training step.

**Impact (v15 → v16):** Sig recall improved from 0.806 to 0.894 — the primary performance gain in v16.

### D.7 GT-Based C⁻¹ in L_dc (v15 Fix) (`optimization.py`)

**Original code and our initial v14 attempt:** The C⁻¹(ŷ_od) function — which converts OD predictions into SC branch pseudo-labels — was implemented as a purely prediction-based operation. OD query outputs were filtered by confidence and converted to sampling-point labels without any ground-truth involvement.

**Problem (the v14 incident):** A GT-free feedback loop is unstable when the OD head is confused early in fine-tuning. In v14, the freshly re-initialised OD head misclassified many Healthy arteries as Non-significant. These incorrect Non-sig pseudo-labels were fed to the SC branch via L_dc. The SC branch learned that Non-sig and Healthy were synonymous. This reinforced the OD head's Non-sig predictions for Healthy vessels via the SC→OD direction of L_dc. The loop locked in: 0 out of 979 Healthy arteries were correctly classified in the raw model.

**Fix:** The OD→SC direction of L_dc is anchored to the ground-truth OD targets: `od2sc_targets(od_targets, seq_length)`. This function converts the GT lesion intervals into per-point SC labels without involving model predictions, breaking the self-reinforcing feedback. The SC→OD direction remains prediction-based (correct per the paper design).

**Current state:** A hybrid of paper design and a GT-based correction. The paper describes both directions as prediction-based; our v15 fix makes one direction GT-anchored. This is a deliberate deviation from the paper that was necessary for stable training on our dataset.

### D.8 DC Confidence Annealing (`train.py`, `optimization.py`)

**Original code:** The confidence threshold for filtering OD predictions before they become SC pseudo-labels was a fixed hyperparameter throughout training.

**Problem:** Early in fine-tuning (epochs 0–20), OD predictions are unreliable — the freshly initialised 6-class heads produce mostly noise. Using a fixed low confidence threshold allows low-quality predictions to generate noisy pseudo-labels during the period when the model is most vulnerable to learning wrong associations.

**Fix:** A curriculum annealing schedule for the DC confidence threshold was introduced. It starts high (0.7 — only very confident OD predictions generate pseudo-labels) and anneals linearly down to the floor (0.4) over the same window as the DC weight ramp (epochs 20–60). The confidence gate and the DC loss weight open together: as the DC signal grows stronger, the pseudo-label quality also improves proportionally.

### D.9 Native 1D Interval IoU (`functions.py`, `optimization.py`)

**Original code box expansion:** Expanded `[cx, w]` to `[cx, cx, w, w]` — a square box where the y-coordinates equalled the x-coordinates. GIoU then computed the area overlap of these squares rather than the interval overlap along the vessel axis.

**Current approach (v10 onward):** Expand to `[cx, 0.5, w, 1.0]` so the converted xyxy box spans `[cx-w/2, 0, cx+w/2, 1]`. This represents a full-height rectangle in the normalised 2D plane, but whose x-extent encodes the 1D vessel-axis interval exactly. GIoU operating on these boxes computes true 1D interval IoU — the width overlap along the vessel axis divided by the union of the two intervals.

**Why this matters:** Coronary artery lesions are 1D objects — they have a location and extent along the vessel axis, not a 2D area. The correct geometric primitive is a 1D interval, and the IoU between two intervals is simply the length of their overlap divided by the length of their union. The paper's square-box approach computed a geometrically incorrect IoU that varied with absolute vessel position rather than purely measuring interval overlap.

---

## E. What We Added from Scratch

None of the following existed in the original paper code. Everything in this section was implemented entirely by us.

### E.1 Training Loop — train.py

The original code had no training loop. It had a `sc_net_framework` class that wired together the model, loss, and data, but no optimizer, no gradient updates, no validation loop, and no checkpoint saving. The entire `Trainer` class in `train.py` was written from scratch.

The trainer implements:

- **AdamW optimizer** with configurable learning rate and weight decay
- **Cosine Annealing with Warm Restarts** (`CosineAnnealingWarmRestarts`, T0=60, T_mult=2) — periodic LR restarts to escape local minima
- **Linear LR warmup** over the first 10 epochs — stabilises transformer parameters before full LR is applied
- **Layer-wise LR decay** — backbone at 0.1×, transformer at 0.5×, heads at 1.0× base LR, matching standard DETR fine-tuning practice
- **Gradient clipping** at `max_norm=0.1` — prevents gradient explosions in transformer attention
- **Gradient accumulation** (`--accumulate_steps=4`) — effective batch size of 16 (2 GPUs × 2 samples × 4 steps) without requiring 16-sample GPU memory
- **Mixed-Precision Training (AMP)** — `torch.amp.GradScaler` + `autocast`, approximately 1.5–2× training speedup on RTX 3090
- **Multi-GPU DDP** — `DistributedDataParallel` via `torchrun --nproc_per_node=2`, using both available RTX 3090s; NCCL timeout set to 3600s
- **NCCL sync fix** — `dist.all_reduce(val_loss)` after validation so all ranks see the same globally-averaged loss before making early-stopping decisions (the original pattern of per-rank decisions caused NCCL ALLREDUCE timeout at epoch 40 of v5)
- **Exponential Moving Average** (`ModelEMA`, decay=0.999) — shadow copy of model weights updated every step; used for evaluation while training weights continue updating
- **Stochastic Weight Averaging** (`torch.optim.swa_utils.AveragedModel`) — averages weights across late epochs for flatter minima and better generalisation; saves `swa_model.pth`
- **Early stopping** — configurable patience on val stenosis F1; synchronised across DDP ranks
- **Checkpoint saving** — every N epochs + best model by val F1 (not val loss — val loss is corrupted by DC ramp spikes)
- **DC warmup schedules** — both weight and confidence thresholds ramped together
- **TensorBoard logging** — per-component losses, metrics, LR schedules, gradient norms, DC schedules

### E.2 Evaluation Pipeline — eval.py

The original code had no systematic evaluation. We wrote a complete evaluation script with:

- Artery-level metric computation for stenosis (ACC, Precision, Recall, F1, Specificity, AUC-ROC) and plaque (same metrics)
- Per-class breakdown for all three stenosis classes and all three plaque classes
- 3×3 confusion matrices for both tasks
- Full support for calibrated evaluation via `--thresholds calibration_thresholds.json`
- Test-Time Augmentation (TTA) — depth flip, intensity scale/shift, logit averaging
- Ensemble evaluation across multiple checkpoints — logit averaging
- `--save_results` — exports all metrics to JSON
- `--detailed` mode — confusion matrices, ROC curves, per-class bar charts
- Separate evaluation of the SWA model (`swa_model.pth`)
- Both `pre_training` (3-class) and `fine_tuning` (6-class) mode support

### E.3 Calibration System — calibrate.py

No calibration existed in the original code. We built a complete per-class threshold calibration system.

The prediction rule is `pred = argmax(p_i / t_i)` where `t_i` is a learned per-class threshold. Thresholds are found by grid search on the validation split to maximise macro-F1.

Three search modes:
- **Standard 2D search** — searches `t_Healthy` and `t_Significant`, fixes `t_NonSig=1.0`. Fast but systematically collapses Non-sig recall to 0%.
- **Constrained 3D search** (`--constrain_nonsig_recall 0.10`) — searches all three thresholds simultaneously with a minimum Non-sig recall constraint. This was the breakthrough in v7 that took Non-sig recall from 0% to 58.1% — the model had learned Non-sig features but the 2D search was missing the threshold sweet spot.
- **Sig-recall constrained search** (`--constrain_sig_recall 0.70`) — prevents the search from sacrificing Significant recall for macro-F1 gain. Added in v16.

The calibration insight is that the paper's "significant recall" clinical priority and Non-sig's diagnostic importance (requires monitoring, not dismissal) cannot both be satisfied by an unconstrained argmax or a 2D threshold search. The constrained 3D search is the only approach that produces clinically valid calibrated predictions.

### E.4 CPR Visualisation — visualize.py

No visualisation existed in the original code. We wrote a complete CPR image rendering pipeline:

- Reads NIfTI volumes and renders the 256-slice longitudinal vessel strip as a horizontal image
- Overlays GT label colour bands (green=Healthy, yellow=Non-sig, orange=Significant) along the vessel length
- Renders a prediction bar below the GT bar using model output
- Extracts and renders up to 4 cross-sectional panels at clinically relevant positions along the vessel
- Outputs one PNG per artery with the artery name, GT class, predicted class, and CORRECT/WRONG label in the filename for easy sorting
- `--save_predictions` flag exports per-artery JSON files containing all OD query outputs, per-query confidence scores, bounding boxes, and final stenosis/plaque assignment for full prediction traceability
- Batch pipeline `run_v16_pipeline.sh` — runs calibrate → eval → visualize → JSON export in sequence

### E.5 Patient-Level Data Splitting — splitting.py

The original code used random file-level splits that could leak data — arteries from the same patient could appear in both training and validation sets. We wrote `splitting.py` with:

- Patient ID extraction from filenames (APNHC/AP-NUH patient ID prefix)
- Patient-level grouping — all arteries from a patient are assigned to a single split
- Configurable 70/15/15 split with deterministic seeding
- Used by `framework.py` to compute `train_idx`, `val_idx`, `test_idx` at setup time

### E.6 Balanced Sampling

The original code used uniform random sampling, which undersampled minority classes (Non-significant, Significant) relative to the majority class (Healthy). We added `WeightedRandomSampler` with inverse-frequency weights computed over the training set, ensuring all classes are seen at approximately equal frequency per epoch. Controlled via `--balanced_sampling`.

### E.7 Online Data Augmentation

The original code applied no online augmentation during training. We added 8 randomised transforms applied per-sample at training time:

- Random rotation (±15°, probability 0.5) applied to the 3D CPR volume
- Intensity jitter (±50 HU additive, probability 0.5)
- Random depth flip along the vessel axis (probability 0.5) — note: SC logit outputs must be reversed before aggregation if flip is applied
- Gaussian noise injection
- Gaussian blur
- Random erasing (random rectangular region set to zero)
- Intensity scaling (multiplicative, mild)
- Intensity shifting (additive, mild)

These are all label-preserving transforms for the artery-level classification task. For the OD branch, box coordinates are adjusted geometrically to remain consistent with spatial augmentations.

### E.8 YAML Config System

The original code used a `DefaultConfig` class with hardcoded values and no CLI override mechanism. We added:

- YAML config loading via `--config configs/finetune_v16.yaml`
- YAML values serve as defaults; any CLI argument overrides the YAML value
- Per-experiment config files committed to the repository for full reproducibility
- Version-named configs: `finetune_v12.yaml`, `finetune_v15.yaml`, `finetune_v16.yaml`, `pretrain_v14.yaml`

### E.9 Cross-Validation — cross_validate.py

Patient-level k-fold cross-validation written from scratch (no sklearn dependency):

- `PatientKFoldSplitter` extracts patient IDs, groups arteries, implements manual k-fold
- `--n_folds` (default 5), `--cv_seed` for reproducibility
- `file_indices` parameter added to `cubic_sequence_data` for flexible fold-based splits

### E.10 Interpretability Tools

- **gradcam.py** — Gradient-CAM saliency maps for both the temporal and spatial branches. Highlights which vessel segments and cross-sectional regions most influenced the model's classification decision.
- **uncertainty.py** — Prediction uncertainty estimation via Monte Carlo dropout. Runs multiple stochastic forward passes and reports prediction entropy, which flags cases where the model is uncertain (clinically useful for borderline cases near the 50% stenosis threshold).

### E.11 Prediction Traceability System

Added `--save_predictions` flag to `visualize.py`, which exports per-artery JSON files to `predictions_vXX_detail/`. Each JSON contains:

- All 16 OD query outputs (class logits, softmax probabilities, confidence scores, predicted box `[cx, w]`)
- Number of foreground queries (those not assigned to no-object) and their indices
- Per-query predicted class and confidence
- Final aggregated stenosis and plaque predictions and probabilities
- Ground-truth labels for comparison

This traceability layer was essential for understanding model failures and complemented the CPR visualisations with machine-readable data. 3182 per-artery JSONs were generated for v14, 67 for v15 and v16 (test subset).

### E.12 Pipeline Automation Scripts

End-to-end bash scripts that run the full evaluation pipeline in sequence:

- `run_v16_pipeline.sh` — calibrate (standard + constrained) → eval (raw + calibrated) → visualise (67 arteries) → per-artery JSON export
- `run_v15_pipeline.sh` — same for v15
- Scripts log all output, handle checkpoint paths, and can run overnight without intervention

---

## Summary Table

| Element | Status | Notes |
|---------|--------|-------|
| Dual-branch architecture | **Kept** | Temporal + spatial design unchanged |
| L_dc mutual supervision formulation | **Kept** | Conceptually unchanged |
| 1:1:1 loss weights | **Kept** | Paper code, not paper equations |
| Hungarian matching costs 1:1:1 | **Kept** | Paper code, not paper equations |
| Data format + label encoding | **Kept** | NIfTI, 0–6 labels, 2-stage training |
| CT windowing approach | **Kept** | Values corrected |
| Multi-view 2D projections | **Kept** | 4 views as designed |
| nn.ModuleList for extraction blocks | **Fixed** | Was Python lists — weights untrained |
| Feature fusion weights as nn.Parameter | **Fixed** | Was CPU tensor in forward() |
| Fixed object query embeddings | **Fixed** | Was torch.randint per call |
| Spatial flattening projection | **Fixed** | Defined but never called |
| Gradient detachment in L_dc | **Fixed** | Circular gradients without detach |
| FocalLoss alpha as buffer | **Fixed** | Device mismatch crash |
| Deep copy targets per loss term | **Fixed** | In-place mutation corrupted L_dc |
| Device-aware target tensors | **Fixed** | Always created on CPU |
| Empty tensor shape guard | **Fixed** | (0,2) → (0,4) crash |
| Degenerate box clamping | **Fixed** | Assert crash at epoch 129 |
| In-place AMP fix | **Fixed** | Autograd graph corruption |
| Dataset index offset | **Fixed** | Val loaded training data |
| _3d_cubes_selection device | **Fixed** | Always CPU output |
| torch.torch.float32 typo | **Fixed** | AttributeError |
| torch.load map_location | **Fixed** | Device conflict on load |
| spatial_proj_channels | **Fixed** | Wrong dimensions |
| Label offset in DC pseudo-labels | **Fixed** | Corrupted every L_dc target |
| Loss returns scalar only | **Fixed** | Added component dict |
| 5:2 loss weights (equations) | **Reverted** | Broke convergence; 1:1 is correct |
| 5:2 Hungarian costs (equations) | **Reverted** | Broke convergence; 1:1:1 is correct |
| Temporal branch sequence shape | **Changed** | Was length-1; now length-32 |
| SE fusion gates | **Changed** | Replaced fixed scalar weights |
| Parallel 2D/3D CNN streams | **Changed** | Fixed serial dependency |
| Learnable positional encoding | **Changed** | Replaced fixed sinusoidal |
| Ordinal EMD loss | **Changed** | New secondary loss term |
| Significant class weight boost | **Changed** | New class weighting |
| GT-based C⁻¹ in L_dc | **Changed** | Deviation from paper; prevents collapse |
| DC confidence annealing | **Changed** | New curriculum schedule |
| Native 1D interval IoU | **Changed** | Correct box geometry |
| Training loop (train.py) | **Added** | Written from scratch |
| Evaluation pipeline (eval.py) | **Added** | Written from scratch |
| Calibration system (calibrate.py) | **Added** | Written from scratch |
| CPR visualisation (visualize.py) | **Added** | Written from scratch |
| Patient-level splitting (splitting.py) | **Added** | Written from scratch |
| Balanced sampling | **Added** | WeightedRandomSampler |
| Online data augmentation | **Added** | 8 transforms |
| YAML config system | **Added** | Replaces hardcoded DefaultConfig |
| Cross-validation (cross_validate.py) | **Added** | Written from scratch |
| GradCAM interpretability | **Added** | Written from scratch |
| Uncertainty estimation | **Added** | Written from scratch |
| Prediction traceability (JSON export) | **Added** | Written from scratch |
| Pipeline automation scripts | **Added** | End-to-end bash pipelines |

---

*Prepared by Reet Mitra — May 2026.*
