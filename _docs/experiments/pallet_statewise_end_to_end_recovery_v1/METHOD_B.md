# B: fixed RGB normal-line boundary distributions

Status: automatic observation implementation and CAL input inspection complete. Pose accuracy and final arm adoption are separate campaign results. Model source was frozen before any new PRIMARY accuracy evaluation.

The module is `scripts/research/pallet_statewise_end_to_end_recovery_v1/boundary_model.py`, SHA-256 `935adf0622315365aba61bd050ea78ab05043d59ac8c4e6108712b8b29d1c6d7`. It uses original RGB, the original clean target mask, original B states/observations, registered cuboid/camera geometry and the prediction-only coarse pose. No manual six-case boundaries, reference pose, historical success category, frame identity, or candidate pose enters observation construction.

## Literature correspondence

[SRT3D, IJCV 2022](https://link.springer.com/article/10.1007/s11263-022-01579-8), equations 1, 5, 12–14, supplies fixed normal-line sampling, foreground/background color posteriors and a normalized offset distribution. Its one-transition assumption is consequential when several material transitions occur. This module retains all offset bins instead of replacing a multimodal distribution by a mean/covariance. The original tracker uses detailed model contours, multiple scales and temporal updates; this experiment uses one image and a cuboid guide. Known-object occlusion rendering is not an unknown-cone detector.

Official implementation inspected: [DLR-RM/3DObjectTracking at f02106186a1e756450226b3b793e309506817c58](https://github.com/DLR-RM/3DObjectTracking/tree/f02106186a1e756450226b3b793e309506817c58), `SRT3D/include/srt3d/region_modality.h` and `SRT3D/src/region_modality.cpp`; [MIT license](https://github.com/DLR-RM/3DObjectTracking/blob/f02106186a1e756450226b3b793e309506817c58/LICENSE), copyright 2023 Manuel Stoiber / DLR. The NumPy implementation is separately written; it does not execute the C++ tracker. Pinned downloaded-file hashes are preserved in private `literature_srt3d/BINDINGS.json`.

## Fixed inputs and coordinates

The loader verifies `packet.image_sha256` and `packet.provenance.original_mask_binding.sha256`, then loads RGB and the NPZ array `pallet_mask_clean`. This is the original mask consumed by `evaluate_fixed_wd_archived_tangent_v4_15.py:66` and `evaluate_face_partial_BI_outline_v4_13d.py:77`. Missing inputs are not replaced and no new segmentation runs inside B.

> Postseal contract clarification; the original frozen description below is retained. The implemented candidate score intersects an infinite supporting line and does **not** enforce the finite candidate segment bound `t∈[0,1]`. Its endpoint inputs therefore do not certify finite physical-edge support. The three extreme outputs and this contract validity limitation are documented in [CONTRACT_VALIDITY_NOTE_KO.md](CONTRACT_VALIDITY_NOTE_KO.md) and [BOUNDARY_EXTREME_FORENSICS_KO.md](BOUNDARY_EXTREME_FORENSICS_KO.md). Numerical verification PASS is not physical-support certification. Original predictions and model are preserved; no posthoc B2 was fitted or evaluated.

All coordinates remain on the raw image canvas. Canonical distances are multiplied by `hypot(height,width)/800`; thus 640×480 has scale 1. A fixed normal profile intersects each candidate's projected native edge, including both projected endpoints. The projected corner alone is not the offset when the profile lies along the edge.

The initial projected cuboid hull proposes incident native edges. Each B anchor proposes four along-edge positions per incident outline edge. S mates are excluded before RGB queries. An O/T/U mate can provide a geometric direction, but its neighborhood is protected and no precise mate observation is created. Off-canvas normal profiles are explicitly unsupported; there is no border padding. The projected cuboid edge may be a virtual bounding feature rather than a visible material edge; no independently certified physical owner is claimed.

## Frozen parameters and distributions

| Quantity | Fixed value | Basis |
|---|---:|---|
| Step amplitude α | 0.36 | Official SRT3D default |
| Step slope | 0 | Official sharp-step default |
| RGB bins/channel | 32 | Official default |
| Appearance samples | negative/positive 1…18 canonical px | Official appearance distances; mask class must agree |
| Normal samples | −43…43, step 1 canonical px | Existing local reach = 36 + 7 |
| Offset grid | −36…36, step 1 canonical px | Same reach with 7 px endpoint context |
| Along-edge locations | 4, 14⅔, 25⅓, 36 canonical px | Four fixed locations within the existing 4…36 interval |
| Protected O/T/U mate radius | 7 canonical px | Existing boundary context |
| Appearance updates per candidate | 0 | Shared fixed evidence |
| Parameter search / fit | 0 | No CAL or PRIMARY accuracy tuning |

Histogram pixels are deduplicated and pooled once over the frame. Foreground/background use equal class priors. A color absent from both histograms has probability 0.5. The fixed profile distribution is

`p(d) ∝ product_r [(.5 − α sign(r−d)) p_f(r) + (.5 + α sign(r−d)) (1−p_f(r))]`.

The complete probability/log-probability vectors, sample coordinates, actual RGB values and hashes are retained. Exact-flat, missing-class, one-sided-color and wrong-direction profiles remain explicitly unsupported. Multiple peaks remain in supported distributions; the nearest-to-initial peak is never selected as truth. Unsupported groups preserve their original B factors and stay in the full-frame population.

Candidate region energy is `−2(log p(d) − max log p)` on the fixed grid, with linear interpolation. Outside the grid, a fixed Laplace continuation prevents an observation from disappearing merely because a candidate moves beyond support. Peak subtraction is a fixed, observation-only constant shared by every hypothesis. These are conditional engineering energies; RGB samples, original point fits and mask statistics are not independent sensors, and probability calibration is not claimed.

## Same-support simple control and replacement

The edge control uses the identical profiles, RGB colors, masks, shared histograms and support count. It fixes one edge location at the largest signed foreground-to-background color-posterior drop. Ties use the first location in normal-coordinate order, independent of candidates. Its Laplace distance slope is `2 log(.86/.14)`, matching the ideal sharp region response. This is a simple point-location compression of the same evidence, not an extra-pixel comparator.

Within a B corner group, profile energies are averaged and the original B reliability is applied once. Duplicate support IDs do not add information. `consumed_native_ids` and exact `full:k` / `partial:k` IDs specify replacement: remove those B full/partial factors and disable the shared global hull factor when a B replacement is active. I and unsupported B factors, O/T constraints and U regions remain. B is never appended on top of full point + partial normal + hull evidence from the same support. The campaign adapter owns and audits this replacement.

For AB, all owner hypotheses use the same bundle and consumed-group set. Pass `q[owner_permutation]` to B, add its energy within the owner hypothesis, and then perform the global owner mixture. Scatter `dE/dq` back through the permutation. No owner hypothesis can hide a profile or refit appearance.

## API and checks

```python
image, mask = load_bound_frame(source_root, packet)
bundle = build_boundary_bundle(image, mask, packet)
value = candidate_energy(bundle, projected_native_xy, model="region")
value, gradient_8x2 = energy_and_gradient(bundle, projected_native_xy, model="edge")
```

Ten synthetic tests passed: foreground direction; physical-edge intersection instead of corner displacement; exact uniform-resolution scaling; retained multiple peaks and flat distributions; S/O/T/U/I gates; unsupported missing appearance; real-border exclusion and wrong-direction handling; immutable support/appearance and duplicate invariance; both analytic projected-coordinate gradients versus central finite differences; and poisoned reference/manual/ID fields having no effect.

All original CAL46 were processed without reference or pose-accuracy scoring: 45 frames had supported B replacements; 218/245 B corners and 1,354/1,544 queried profiles were supported. All 46 remain in the inventory. Reasons include 13 missing-appearance profiles, 173 wrong-direction profiles and 4 flat/one-sided profiles; off-canvas and other pre-query exclusions are separately recorded. This is observation coverage, not a pose success rate.

RGB file authentication/loading took 0.521 s and observation construction 1.004 s in aggregate. The complete probe, including reading the original compressed packet inventory, writing the compressed bundles and plotting, took 8.503 s wall / 8.500 s process CPU. Process high-water RSS was 1,055.13 MiB, not an incremental B allocation measurement. BLAS/OpenMP threads were 1; new NN/PnP calls were 0.

The first three CAL frames in immutable input order are shown in private `b_boundary_v1/CAL_BOUNDARY_OVERVIEW.png`; its actual RGB, normal directions, raw-coordinate axes and distribution labels were visually inspected. The unsupported first frame is retained. Private `CAL_BOUNDARY_BUNDLES.jsonl.gz` and `CAL_PROBE_RECEIPT.json` preserve complete support, input hashes, reasons and resources. No PRIMARY case selection or accuracy was performed by this probe.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
$PRIVATE_HOME/anaconda3/envs/pallet-yolo26/bin/python -B -m pytest -q \
  scripts/research/pallet_statewise_end_to_end_recovery_v1/test_boundary_model.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
$PRIVATE_HOME/anaconda3/envs/pallet-yolo26/bin/python -B -m \
  scripts.research.pallet_statewise_end_to_end_recovery_v1.boundary_model \
  --source-root $PRIVATE_HOME/Documents/github/pallet-pose \
  --packets $OLD_STATE_PRIVATE/e_trace_v2/TYPED_PACKETS.jsonl.gz \
  --output-dir $RUN_PRIVATE/b_boundary_v1
```
