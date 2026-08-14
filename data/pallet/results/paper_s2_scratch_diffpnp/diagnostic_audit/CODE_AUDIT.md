# PAPER_S2 ep57 code audit

## Scope and verdict

This audit describes the canonical PAPER_S2 Stage-B checkpoint and the code paths
that can affect it:

- checkpoint: `weights/paper_s2_stageB/net_epoch_0057.pth`
- SHA-256:
  `c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896`
- selected checkpoint evidence:
  `data/pallet/results/paper_s2_scratch_diffpnp/stageB_val_select.md:7-18`
- recorded training arguments: `weights/paper_s2_stageB/header.txt:1-2`
- launch command: `scripts/stage0/_run/diffpnp3d_full_run.sh:72-104`

The checkpoint is the direct DiffPnP model. Later LOO/flip/self-training
checkpoints are derivatives and are not evidence for the direct DiffPnP
objective.

The method implemented by the canonical path is:

> A GT-pose-initialized, four-step, differentiable unrolled
> Gauss--Newton pose regularizer whose output is supervised with a normalized
> 3-D corner loss.

It is **not** the legacy `BPnP` autograd function, is not an OpenCV PnP forward
pass, and is not the same solver used at evaluation time. Calling it a
“DiffPnP3D regularizer” is accurate. Calling it an end-to-end deployment PnP
solver or the repository's BPnP implementation is not.

This document audits the current working tree. Relevant source files are
modified or untracked relative to Git, including
`Deep_Object_Pose/train/diffpnp3d_loss.py`. In particular, the current
pre-solve NaN guard and its 2026-07-28 unit audit postdate the saved ep57
artifact. They make the current training implementation safer but do not
retroactively change ep57 weights.

## Artifact and architecture check

A read-only checkpoint inspection found:

| property | ep57 value |
|---|---:|
| state-dict tensors | 208 |
| `module.` prefix | absent |
| tensor dtype | `torch.float32` |
| segmentation head | present (`m_seg1.*`, `m_seg2.*`) |
| corner-quality head | absent |
| mask-belief-fusion head | absent |

`net_epoch_0057_noseg.pth` only removes segmentation parameters. Its 184
parameters shared with ep57 are bitwise identical; it is not another learned
heatmap model.

The model source constructs VGG with `pretrained=False` regardless of the
constructor argument (`Deep_Object_Pose/common/models.py:14-18,43-57`).
The six belief and affinity stages are declared at
`Deep_Object_Pose/common/models.py:59-95` and connected recurrently at
`Deep_Object_Pose/common/models.py:167-217`. Stage 1 consumes the 128-channel
VGG feature. Stages 2--6 consume `128 + 9 + 16 = 153` channels.

For a canonical `(B,3,400,400)` input, the effective graph is:

```text
image float32 (B,3,400,400)
  -> VGG/custom feature float32 (B,128,50,50)
  -> six belief stages    [float32 (B,9,50,50)] x 6
  -> six affinity stages  [float32 (B,16,50,50)] x 6
  -> two segmentation logits [float32 (B,1,50,50)] x 2
```

The outputs are raw real-valued maps. There is no final sigmoid or spatial
softmax in `DopeNetwork`.

## Canonical ep57 configuration

The launch script fixes Stage B to the six data directories, a 60:40
base-to-mask sampling balance, batch 12, sigma 2, input size 400, learning rate
`5e-5`, seed 42, Stage-A epoch 42 initialization, mask auxiliary weight `0.01`,
and DiffPnP weight `0.005` with a 1000-step ramp
(`scripts/stage0/_run/diffpnp3d_full_run.sh:78-91`). The persisted namespace agrees
(`weights/paper_s2_stageB/header.txt:1-2`).

The following flags were off in the saved namespace:

- `geo_loss`
- `vis_coord_loss`
- `rel_loss`
- `struct_loss`
- `symmetric_loss`

The ep57 checkpoint has no later refinement heads. Therefore teacher
distillation, mask-belief fusion, corner quality, projected-span,
signed-footprint, mask-extent, DiffPnP undercoverage, and DiffPnP fit-coverage
are not part of the canonical result.

## Source-of-truth table

The table distinguishes the historical ep57 path from optional or diagnostic
code that merely exists in the current repository.

| quantity or decision | source | shape / dtype | coordinate space | detach, mask, and gradient behavior | ep57 status |
|---|---|---|---|---|---|
| raw keypoint order | `Deep_Object_Pose/common/utils_dataset.py:432-449` | 8 corners, then centroid to make 9 | original JSON pixels | no remapping; invisible object gets sentinels | active |
| pseudo-label validity | `Deep_Object_Pose/common/utils_dataset.py:451-472`; `Deep_Object_Pose/common/heatmap_refinement.py:45-64` | belief `(B,9)`, affinity `(B,16)`, float32 | channel metadata | affinity pair is valid only if corner and centroid are valid | active |
| image spatial transform | `Deep_Object_Pose/common/utils_dataset.py:542-580`; `Deep_Object_Pose/train/train.py:994-1028` | image becomes `400x400` | original `640x480` -> anisotropically squashed input | DiffPnP-eligible frames skip rotation; all still receive photometric augmentation | active |
| target-grid transform | `Deep_Object_Pose/common/utils_dataset.py:583-603` | target image/keypoints become `50x50` | input 400 grid -> belief 50 grid | used for belief, affinity, mask targets | active |
| network input | `Deep_Object_Pose/common/utils_dataset.py:638-645,664-672` | `(B,3,400,400)`, float32 | squashed input pixels | ImageNet normalization; gradient begins at network input | active |
| belief GT | `Deep_Object_Pose/common/utils_belief.py:127-175`; `Deep_Object_Pose/common/utils_dataset.py:623-630` | `(B,9,50,50)`, **float64** | belief grid | NumPy default float64; no gradient | active |
| affinity GT | `Deep_Object_Pose/common/utils_belief.py:97-124`; `Deep_Object_Pose/common/utils_dataset.py:631-636` | `(B,16,50,50)`, float32 | belief grid | corner-to-centroid unit vectors; no gradient | active |
| belief/affinity MSE | `Deep_Object_Pose/common/heatmap_refinement.py:67-99`; `Deep_Object_Pose/train/train.py:172-196` | scalar per stage | map values | global valid-element average, not equal per-frame average; six stages summed | active |
| shared feature and six stages | `Deep_Object_Pose/common/models.py:43-95,167-217` | feature `(B,128,50,50)`; belief `(B,9,50,50)`; affinity `(B,16,50,50)`, float32 | belief grid | recurrent stage dependency | active |
| segmentation output | `Deep_Object_Pose/common/models.py:107-115,234-238` | two logits `(B,1,50,50)`, float32 | belief grid | shares VGG, but is not a belief hard gate | active |
| mask target and validity | `Deep_Object_Pose/common/utils_dataset.py:720-734` | mask `(B,1,50,50)`, validity `(B,)`, float32 | belief grid | only decoded real `mask_rle` frames are valid | active |
| mask BCE | `Deep_Object_Pose/train/train.py:278-302` | scalar | mask grid | pixel mean per frame, then valid-frame mean; gradients to both seg heads and shared VGG | active |
| DiffPnP frame eligibility | `Deep_Object_Pose/common/utils_dataset.py:474-485,736-775` | validity `(B,)`; fixed-shape fallbacks | audit metadata plus transformed belief grid | requires audit `pnp_valid_3d`, `V8`, original `640x480`, and all eight GT corners inside the belief support margin | active |
| DiffPnP targets | `Deep_Object_Pose/common/utils_dataset.py:761-775`; `Deep_Object_Pose/train/diffpnp3d_loss.py:501-524` | `X (B,8,3)`, `K/R (B,3,3)`, `t (B,3)`, `diag (B,)`, float32 after loader cast | object/camera 3-D and original image pixels | invalid frames receive a well-conditioned placeholder and mask 0 | active |
| local heatmap decode | `Deep_Object_Pose/train/diffpnp3d_loss.py:40-101` | `(B,8,2)`, float32 | belief grid -> original pixels using `x*12.8`, `y*9.6` | hard argmax selects window; softmax-weighted coordinate is differentiable within the selected window | active |
| local confidence/moments | `Deep_Object_Pose/train/diffpnp3d_loss.py:65,86-100`; `Deep_Object_Pose/train/train.py:461` | peak, ratio, sigma, `var_x`, `var_y`, `cov_xy` | belief grid | peak value retains a max-value gradient, but second peak, ratio, and moments are under `no_grad`; the whole `_conf` result is discarded | diagnostic only |
| GT pose initialization | `Deep_Object_Pose/train/diffpnp3d_loss.py:287-314` | `rvec/tvec (B,3)` | object-to-camera pose | `R_gt` is detached to CPU/SciPy; `t_gt` is detached | active |
| unrolled GN | `Deep_Object_Pose/train/diffpnp3d_loss.py:317-333` | `J (B,16,6)`, `A (B,6,6)`, `delta (B,6)` | original-pixel reprojection residual | four differentiable updates; damping `1e-3`, update norm clipped at `0.5` | active |
| 3-D geometry loss | `Deep_Object_Pose/train/diffpnp3d_loss.py:335-341` | scalar after frame reduction | camera-frame 3-D, normalized by object diagonal | GT transformed corners detached; gradient flows through predicted pose to heatmaps | active |
| undercoverage and fit coverage | `Deep_Object_Pose/train/diffpnp3d_loss.py:343-417` | per-frame diagnostics/losses | projected 2-D span and camera depth | PCA axes/targets detached; weights default to zero in ep57 path | off |
| final DiffPnP guards | `Deep_Object_Pose/train/diffpnp3d_loss.py:419-438` | boolean `(B,)` | mixed | audit mask AND positive depth AND finite values AND condition threshold | active |
| total finite guard | `Deep_Object_Pose/train/train.py:527-535` | scalar | loss | checks loss before backward; does not inspect gradients or parameters | active |
| learned corner quality | `Deep_Object_Pose/common/models.py:131-145,246-253`; `Deep_Object_Pose/common/heatmap_refinement.py:833-873` | log-sigma map `(B,9,50,50)` | belief-grid cells | all quality-head inputs detached; only quality head learns from this loss | absent |
| evaluation sigma | `scripts/stage0/paper_s2_rgb1_eval.py:411-431` | one scalar sigma per keypoint | converted to scalar original-pixel RMS | available only if a quality head exists | absent |

### Mixed belief-target dtype

`CreateBeliefMap` starts from `np.zeros` without a dtype
(`Deep_Object_Pose/common/utils_belief.py:145-170`), so the loader creates a
float64 belief target at
`Deep_Object_Pose/common/utils_dataset.py:623-630`. Consequently,
`channel_masked_mse` performs the belief error arithmetic in float64. However,
`loss_belief` is initialized as a float32 scalar and updated with in-place
`+=` (`Deep_Object_Pose/train/train.py:172-194`), so every stage result is cast
back into the float32 accumulator. The final canonical loss and parameter
gradients are float32, but the belief MSE incurs unnecessary mixed-dtype /
float64 work.

This is a reproducibility and efficiency defect, not evidence that ep57
parameters are float64.

### Border target and DiffPnP gate

With the canonical `sigma=2` and `clip_at_border=False`, a belief channel is
drawn only if the full support extending `2*sigma` cells on each side fits in
the grid (`Deep_Object_Pose/common/utils_belief.py:138-169`). The DiffPnP target
gate mirrors this requirement for all eight transformed corners
(`Deep_Object_Pose/common/utils_dataset.py:745-760`).

The gate is based on GT corners. A prediction may still peak at the map border.
`LocalSoftArgmax2D` clamps every window index
(`Deep_Object_Pose/train/diffpnp3d_loss.py:69-78`), which duplicates an edge
cell multiple times in the softmax window. Thus early or failed boundary
predictions can be multiplicity-biased even though active GT targets are
interior.

## Canonical ep57 loss graph

Let `B_s` and `A_s` be belief and affinity outputs for stage
`s in {1,...,6}`, `Y_B` and `Y_A` their targets, and `M_B`, `M_A` the
channel masks. `masked_mse` averages over all valid batch/channel/spatial
elements (`Deep_Object_Pose/common/heatmap_refinement.py:67-99`).

```text
L_base =
    sum(s=1..6) [
        masked_mse(B_s, Y_B, M_B)
      + masked_mse(A_s, Y_A, M_A)
    ]

L_seg =
    sum(j=1..2)
      sum_i mask_valid_i * mean_hw BCEWithLogits(S_j[i], mask_gt_i)
      ----------------------------------------------------------------
                         sum_i mask_valid_i

ramp(global_step) = min(1, global_step / 1000)

u = LocalSoftArgmax2D(B_6[:, 0:8])       # (B,8,2), original pixels
T_pred = GN4(init=T_gt, observations=u)
L_3d = valid-frame mean of
       mean_corner Huber(||T_pred X - T_gt X||_2 / object_diagonal)

L_ep57 = L_base + 0.01 * L_seg + 0.005 * ramp(global_step) * L_3d
```

The outer implementation is at
`Deep_Object_Pose/train/train.py:175-196,278-302,442-473`.

Important reduction and gradient details:

- The six stages are summed, not divided by six.
- Both segmentation stages are summed.
- A batch with no real mask contributes exactly zero mask loss.
- DiffPnP is skipped at the trainer level when the batch has no active
  `diffpnp_valid` frame (`Deep_Object_Pose/train/train.py:457-460`).
- Valid DiffPnP frames are normalized by their valid count
  (`Deep_Object_Pose/train/diffpnp3d_loss.py:434-438`).
- The final belief stage receives the direct DiffPnP gradient.
- Because stage 6 consumes stage-5 belief and affinity outputs, this gradient
  also reaches the backbone and belief/affinity stages 1--5 through recurrence.
  It does **not** reach the final affinity head `m6_1`; that head receives only
  its affinity-map MSE.
- The segmentation loss reaches the segmentation heads and the shared VGG
  feature. There is no ep57 mask-to-belief fusion or hard mask gate.
- `global_step` is reset to zero after loading a checkpoint
  (`Deep_Object_Pose/train/train.py:1198-1206,1402-1406`), so the Stage-B
  1000-step ramp starts again at the Stage-B run rather than continuing a
  Stage-A optimizer/global-step state.

## Current DiffPnP3D numerical path

### Decoder

`LocalSoftArgmax2D`:

1. finds a hard raw-map argmax per channel;
2. gathers a clamped `7x7` window;
3. applies a temperature-`0.1` softmax to raw map values;
4. computes a weighted coordinate in the 50-grid;
5. maps it to original pixels by `(12.8, 9.6)`.

The hard window selection is non-differentiable. Given a fixed window, the
weighted coordinate is differentiable with respect to the gathered heatmap
values. The returned peak value retains the gradient of `max`, while the
second peak, ratio, and second moments are explicitly outside autograd; the
trainer discards the entire confidence dictionary
(`Deep_Object_Pose/train/diffpnp3d_loss.py:65-100`).

### Pose update

For each of four iterations, the current code computes

```text
r = project(rvec, tvec, X, K) - observed_xy
J = approximate projection Jacobian wrt [rvec, tvec]
A = J^T J + 1e-3 I
delta = solve(A, J^T r)
delta = delta * min(1, 0.5 / ||delta||)
[rvec, tvec] = [rvec, tvec] - delta
```

See `Deep_Object_Pose/train/diffpnp3d_loss.py:109-146,317-333`.
There is no camera distortion term.

The starting pose is the GT pose, not a pose inferred from scratch. This
substantially narrows the optimization problem and is the principal reason the
module should be described as a local training regularizer.

### Current pre-solve NaN fix

The original style of checking finiteness only after the solver is too late:
one non-finite observation can make `solve` or `eigvalsh` throw, aborting an
otherwise healthy batch; moreover, masking a non-finite branch with
`torch.where` can still leave non-finite derivatives in backward.

The current working tree guards observations before the batched solve:

- record per-frame `pred_finite`;
- replace NaN and both infinities in `pred_xy` with zero using
  `torch.nan_to_num`;
- run the fixed-shape batched solve on the safe tensor;
- include the original `pred_finite` flag in the final frame-valid mask.

Source: `Deep_Object_Pose/train/diffpnp3d_loss.py:301-309,419-438`.

The regression audit deliberately places a NaN frame next to a healthy frame
and checks backward
(`scripts/stage0/paper_s2_geometry_unit_audit.py:831-859`;
`challenge/tests/test_paper_s2_geometry_unit_audit.py:27-32`). The recorded
result is:

| diagnostic | result |
|---|---:|
| valid frames | 1 of 2 |
| skipped as NaN | 1 |
| invalid-frame gradient norm | `0.0` |
| healthy-frame gradient norm | `1.245283e-6` |
| finite gradients by frame | `[true, true]` |

This fix is narrowly scoped to non-finite predicted 2-D observations. It does
not pre-sanitize corrupt `X`, `K`, `R_gt`, `t_gt`, or `diag`, nor does it catch
all possible exceptions from a non-finite iterate before `solve/eigvalsh`.
Those targets are expected to be finite by the audit index and fallback
construction. Gradient and parameter finiteness are still not checked after
`loss.backward()`.

## Legacy BPnP is a separate method

The legacy path is enabled by `--geo_loss`, which was false for ep57. Its main
entry is `Deep_Object_Pose/train/geo_loss.py:46-143`.

It consists of:

1. a global spatial softmax over every heatmap cell
   (`Deep_Object_Pose/train/geo_loss_bpnp.py:75-105`);
2. detached CPU OpenCV `SOLVEPNP_EPNP` in a custom
   `torch.autograd.Function` forward
   (`Deep_Object_Pose/train/geo_loss_bpnp.py:108-147`);
3. a hand-written damped pseudo-inverse/Jacobian backward
   (`Deep_Object_Pose/train/geo_loss_bpnp.py:149-177`).

It is a repository-local approximation inspired by BPnP, not the official
reference implementation. The recorded geometry audit intentionally reports
it separately:

| gate | legacy BPnP | current DiffPnP3D |
|---|---:|---:|
| oracle pose/reprojection | pass | pass |
| finite-difference relative error | `5.429e-1` at epsilon `0.01` (fail) | `1.014e-9` (pass) |
| canonical ep57 path | no | yes |

Evidence:
`data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit/unit_audit.md:9-22`
and
`data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit/unit_audit.json`.
The legacy result is not a single finite-difference step-size artifact: an
epsilon sweep over `0.1, 0.05, 0.01, 0.005, 0.001` remains in the
`0.537--0.545` relative-L2 range, with best error `0.536847`, still far above
the `1e-2` gate.

Additional legacy-path defects:

- `input_size` is accepted but unused
  (`Deep_Object_Pose/train/geo_loss.py:59-78`).
- The path uses one fixed camera matrix scaled to the 50-grid and one fixed
  `1.1 x 1.1 x 0.15` model
  (`Deep_Object_Pose/train/train.py:1208-1222`). If used with the loader's
  random crop/rotation, K is not transformed with the augmentation.
- A broad `except Exception: pass` silently removes all PnP-derived terms
  (`Deep_Object_Pose/train/geo_loss.py:105-138`).
- Its `volume_loss` compares edge lengths before and after rigid transforms.
  Rigid transforms preserve those lengths, so the predicted and GT volumes are
  identical and the term is mathematically zero-gradient
  (`Deep_Object_Pose/train/geo_loss_pose.py:51-73`).
- The face-diagonal midpoint constraint
  (`Deep_Object_Pose/train/geo_loss_pose.py:30-38`) assumes a projected
  parallelogram, which is not generally invariant under perspective.
- Reliability uses a detached scalar inverse-sigma reweighting, while its
  `lambda_log` parameter is never used
  (`Deep_Object_Pose/train/geo_loss_coord.py:46-75,100-126`).
- Structural flip assumes that the network returns exactly two values
  (`Deep_Object_Pose/train/geo_loss_struct.py:34-40`). It is incompatible with
  a segmentation/refinement network returning four or five values.

## Covariance and uncertainty status

The correct status is:

> Local heatmap second moments are implemented and numerically audited, but
> covariance-aware training and covariance-aware PnP are not implemented in
> the canonical ep57 path.

Specifically:

- `LocalSoftArgmax2D` computes `var_x`, `var_y`, and `cov_xy`, but does so under
  `torch.no_grad` and the trainer discards `_conf`
  (`Deep_Object_Pose/train/diffpnp3d_loss.py:86-100`;
  `Deep_Object_Pose/train/train.py:461`).
- The “local soft-argmax/covariance PASS” in `unit_audit.md` verifies moment
  arithmetic, PSD behavior, coordinate scaling, and finite coordinate
  gradients. Every recorded covariance has
  `covariance_requires_grad=false`; it is a diagnostic gate, not a learned or
  deployed contribution.
- Legacy `SpatialSoftArgmax2D` returns only the scalar radial
  `sqrt(var_x + var_y)`, not a full `2x2` covariance
  (`Deep_Object_Pose/train/geo_loss_bpnp.py:98-105`).
- The optional corner-quality head predicts a separate isotropic localization
  sigma and receives detached feature/belief inputs
  (`Deep_Object_Pose/common/models.py:131-145,246-253`). ep57 does not contain
  this head.
- If a later checkpoint has the quality head, the evaluator converts the
  scalar grid sigma to a scalar pixel RMS
  (`scripts/stage0/paper_s2_rgb1_eval.py:411-431`) and the safe PnP path converts
  it to clipped, median-normalized inverse-variance weights
  (`challenge/scripts/annotate_pnp.py:380-432`).
- “Covariance” in the projected-span code is the covariance of corner
  coordinates used to obtain PCA axes, not prediction uncertainty
  (`Deep_Object_Pose/train/diffpnp3d_loss.py:174-187,351-360`).

No ep57 loss uses a Mahalanobis residual, no full covariance is propagated
through PnP, and no ep57 evaluation result is uncertainty-weighted.

## Training versus evaluation mismatch

### Coordinate decoder

Training DiffPnP decodes:

- all eight corner channels;
- raw argmax;
- `7x7` raw-value softmax at temperature `0.1`;
- no smoothing, NMS, confidence threshold, or missing correspondence;
- fixed original-pixel scale `(12.8, 9.6)`.

Canonical validation/evaluation decodes:

- Gaussian smoothing with sigma 2;
- four-neighbor local NMS and threshold `0.3`;
- strongest raw-valued surviving peak;
- raw `11x11` weighted centroid with `+0.4395` offset;
- missing correspondences when no valid peak;
- an optional centroid channel.

Sources:
`Deep_Object_Pose/train/diffpnp3d_loss.py:59-101` versus
`scripts/data_prep/eval/filter_pr_camfacing.py:124-156`.
The canonical single-object evaluator also discards the predicted affinity
maps (`scripts/stage0/diffpnp3d_q1_eval.py:125-128`), although all six affinity
stages remain supervised by the base training loss.

The current torch refinement decoder also clamps negative patch values to zero
(`Deep_Object_Pose/common/heatmap_refinement.py:166-195`), whereas the
historical SciPy/NumPy evaluator uses raw patch weights. That helper is not the
canonical ep57 coordinate source.

### Pose solver

Training:

- exactly eight ordered corners;
- per-frame audited `X`, `K`, and dimensions;
- GT pose initialization;
- four local GN steps;
- no centroid, missing points, RANSAC, W/D hypothesis selection, or discrete
  seed search;
- supervision by camera-frame 3-D corner distance.

Canonical selection evaluation:

- the same 400-square input squash and inverse coordinate scales
  (`scripts/stage0/diffpnp3d_q1_eval.py:4-8,51-69`);
- thresholded/missing keypoints and optional centroid
  (`scripts/stage0/diffpnp3d_q1_eval.py:125-153`);
- at least six detected correspondences;
- `(W,D,H)` and `(D,W,H)` hypotheses;
- OpenCV-based candidates and reprojection/invariant ranking
  (`scripts/stage0/diffpnp3d_q1_eval.py:89-108`);
- the selection wrapper reuses that evaluator verbatim
  (`scripts/stage0/diffpnp3d_val_select.py:4-10,91-101`).

The current paper evaluator additionally exposes a legacy `>=6` path and a
fail-closed safe `>=7` path
(`scripts/stage0/paper_s2_rgb1_eval.py:491-536`). Its production solver searches
IPPE-face, EPNP, SQPNP, all-point IPPE, and manual seeds, applies symmetry and
LM refinement, and evaluates W/D candidates
(`challenge/scripts/annotate_pnp.py:435-590,803-972,1034-1123`).

This mismatch does not invalidate DiffPnP as an auxiliary regularizer, but it
does invalidate a claim that training differentiates through the deployment
solver.

The stock `Deep_Object_Pose/common/detector.py` is a third, incompatible path:
it defines its own older `DopeNetwork`, performs no canonical 400-square
resize, uses a fixed scale factor 8 and classic affinity association, and
strict-loads a checkpoint without segmentation keys
(`Deep_Object_Pose/common/detector.py:32-65,247-270,460-581`). It must not be
used to reproduce PAPER_S2 ep57 results.

## Configuration and dead-path findings

| finding | evidence | consequence |
|---|---|---|
| `-c/--config` is parsed but never loaded/applied | `Deep_Object_Pose/train/train.py:1470-1484` | passing a YAML file directly to `train.py` has no effect |
| `--save` is parsed but unused | `Deep_Object_Pose/train/train.py:1608` | documented “save a batch and quit” behavior does not exist |
| VGG pretrained flag is ignored | `Deep_Object_Pose/common/models.py:14-18,43` | model always starts from non-ImageNet VGG unless weights are loaded from a checkpoint |
| `config/default.yaml` says input 448 and pretrained true | `config/default.yaml:7-15` | it does not describe canonical ep57 |
| loader input is hard-coded to 400 | `Deep_Object_Pose/common/utils_dataset.py:542-580`; target grid is fixed at `Deep_Object_Pose/train/train.py:987` | `--imagesize` does not resize the actual loader input |
| main creates an unused torchvision resize | `Deep_Object_Pose/train/train.py:840-844` | apparent `--imagesize` wiring is dead |
| YAML loss weights are not consumed | `config/default.yaml:43-48`; `Deep_Object_Pose/train/train.py:172-196` | base belief/affinity terms are unweighted sums |
| shell wrapper supports only a subset of flags | `scripts/train_dope.sh:61-87,177-239` | unknown options are warned and discarded; PAPER_S2 correctly uses a separate runner |
| resume does not restore global step/optimizer | `Deep_Object_Pose/train/train.py:1198-1206,1402` | ramps restart on every resumed process |
| `DiffPnP3DLoss.temperature` member is unused | `Deep_Object_Pose/train/diffpnp3d_loss.py:232-285` | temperature is controlled only by the separate local-soft-argmax module |
| optional loss detail dictionaries are not logged | `Deep_Object_Pose/train/train.py:223-276,537-600` | only selected aggregate scalars are retained |

## Bugs and limitations ranked

### High

1. **Historical non-finite observation handling occurred after the solve.**
   The current pre-solve sanitization at
   `Deep_Object_Pose/train/diffpnp3d_loss.py:301-309` fixes the reproduced
   batch-abort/backward-poisoning case. The saved ep57 artifact is unchanged.
2. **Legacy BPnP backward fails its finite-difference audit.**
   Relative error is `0.5429` at epsilon `0.01`; the five-epsilon sweep has
   best error `0.536847`, so the mismatch is not a finite-difference step-size
   artifact. It must remain separate from the validated current unrolled-GN
   path.
3. **Configuration provenance is fragmented.** `train.py -c` is dead and
   `config/default.yaml` contradicts the actual runner. The persisted header,
   runner, checkpoint hash, and selection report are the canonical source set.

### Medium

1. Belief targets are float64 while the network and accumulator are float32.
2. `global_step` and optimizer state restart during Stage-B resume, changing
   ramp semantics from continuous training.
3. Local soft-argmax duplicates clamped boundary cells; the GT-interior gate
   does not guarantee the prediction is interior.
4. The symmetric-loss option takes one scalar minimum for the entire batch
   (`Deep_Object_Pose/train/train.py:180-190`), rather than choosing the
   symmetry hypothesis per sample.
5. The trainer checks only scalar loss finiteness before backward, not gradient
   or parameter finiteness.
6. Training and deployment use different coordinate decoders and radically
   different PnP solvers.

### Dormant because ep57 flags are off

1. Legacy geo K is inconsistent with crop/rotation augmentation.
2. Legacy PnP exceptions are silently swallowed.
3. Legacy volume loss is zero-gradient.
4. Legacy reliability's `lambda_log` is dead.
5. Structural flip is incompatible with segmentation/refinement return arity.
6. Visibility targets are computed from original JSON coordinates and image
   shape, not from the actual crop/rotate/resize coordinates
   (`Deep_Object_Pose/common/utils_dataset.py:662,779-829`).

## Reproducibility statement

For a paper or result table, the minimally accurate description is:

> PAPER_S2 Stage-B epoch 57 was initialized from Stage-A epoch 42 and trained
> on six datasets with 60:40 base/mask sampling. The network used 400x400
> anisotropically resized input, 50x50 nine-channel beliefs, six-stage belief
> and affinity MSE, a two-stage segmentation auxiliary BCE weighted by 0.01,
> and a GT-initialized four-step differentiable Gauss--Newton 3-D corner
> regularizer weighted by 0.005 after a 1000-step ramp. Covariance estimates
> were diagnostic only and were not used by the canonical loss or evaluator.

Any claim beyond this should explicitly distinguish later refinement heads,
the legacy BPnP implementation, and the deployment PnP solver.
