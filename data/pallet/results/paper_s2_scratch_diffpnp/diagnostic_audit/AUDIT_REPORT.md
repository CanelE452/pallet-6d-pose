# PAPER_S2 ep57 DiffPnP diagnostic audit

Audit date: 2026-07-28  
Repository: `/home/minjae/Documents/github/pallet-pose`  
Branch / HEAD: `main` / `0baa6dfc2ba850dd498f59b74e42663828d166c7`

## 1. Executive decision

The paper-safe direct DiffPnP checkpoint in the current repository is:

```text
weights/paper_s2_stageB/net_epoch_0057.pth
SHA-256 c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
```

`stageB_best.pth` resolves to this file. `net_epoch_0057_noseg.pth` is only an
export with 24 segmentation tensors removed; all 184 common tensors are
bitwise equal. The newest dated derivative is the h8 LOO+flip self-training
checkpoint, but it is excluded from paper model selection because its pool
shares filter-validation sessions and its historical outside evaluator
included sealed final-test sessions.

The full frozen audit completed 623/623 frames with zero errors:

| population | role | frames | status |
|---|---|---:|---|
| Q1 synthetic validation | order-free/channel-agnostic only | 500 | complete |
| Outdoor–Day + Night | strict primary filter-val | 87 | complete |
| `capturepallet11` manual | exploratory PL-pool only | 36 | complete |

The sealed final-test sessions `capturepallet07`, `capturepallet09`,
`capturenight08`, and `capturenight09` had zero input opens. `handannot17` also
had zero input opens. No final-test score was computed.

Primary causal verdict:

1. The dominant error begins in the predicted heatmap/keypoint geometry,
   especially kp5/kp6 and the far/depth footprint under truncation. It is not
   principally caused by K, units, 3-D dimensions, or the choice among tested
   direct PnP solvers.
2. The local covariance arithmetic is finite and PSD, but its raw values are
   severely undercalibrated. It ranks difficult predictions, but cannot yet be
   treated as calibrated uncertainty or plugged into PnP as a paper method.
3. The current DiffPnP3D regularizer passes its gradient audit. The separate
   legacy BPnP implementation fails finite differences and remains blocked.
4. The centroid is not an isolated yaw cause. It follows the biased predicted
   corner footprint; GT replacement, corner mean, geometry projection, and
   removal do not provide a consistent yaw cure.
5. Replacing kp5 with GT helps, but kp6 helps at least as much. This supports a
   structured far/depth-face localization problem, not a unique kp5 code bug
   or a justified fixed kp5 down-weight.

## 2. Model and implementation identity

The canonical ep57 model is a six-stage VGG DOPE network:

```text
RGB Bx3x400x400
  -> VGG/custom feature Bx128x50x50
  -> six raw belief stages Bx9x50x50
  -> six raw affinity stages Bx16x50x50
  -> two segmentation logits Bx1x50x50
```

Its saved training configuration is:

| item | value |
|---|---|
| convention | camera-facing 0123 |
| input | 640x480 anisotropic squash to 400x400 |
| belief grid | 50x50, sigma 2 |
| batch / seed | 12 / 42 |
| Stage B data | Arm A:Arm B sampled 60:40 |
| optimizer / LR | Adam / `5e-5` |
| base loss | six belief MSE + six affinity MSE |
| mask auxiliary | two segmentation BCE stages, weight `0.01` |
| DiffPnP3D | final belief corners 0–7, local 7x7 softargmax T=0.1 |
| pose update | GT-pose initialized, four unrolled GN steps |
| geometry loss | normalized camera-frame 3-D corner Huber |
| DiffPnP weight | `0.005`, 1,000-step linear ramp |

The exact active objective is:

```text
L = sum_s=1..6 (belief_MSE_s + affinity_MSE_s)
    + 0.01 * sum_j=1..2 segmentation_BCE_j
    + 0.005 * min(1, global_step/1000) * DiffPnP3D_corner_loss
```

Affinity has no direct DiffPnP gradient. Only the final belief map is decoded,
although recurrence propagates that gradient into earlier stages and the
backbone. Covariance moments are computed under `no_grad`, discarded by the
trainer, and are not consumed by the canonical PnP evaluator.

The method name should be “GT-initialized unrolled-GN DiffPnP3D regularizer.”
It is not the legacy BPnP function and is not the deployment PnP solver.
Detailed graph and executable pseudocode are in `CODE_AUDIT.md` and
`DIFFPNP_PSEUDOCODE.md`.

## 3. Gate results

| gate | result | evidence / consequence |
|---|---|---|
| Gate 1: GT geometry oracle | PASS | strict GT+centroid PnP 87/87; reprojection median 1.913 px, p95 6.015 px |
| Gate 2: covariance unit test | PASS for arithmetic | known Gaussian mean ≤0.1 cell, covariance error ≤5%, PSD and finite gradients |
| Real covariance calibration | FAIL for method use | detected 95% ellipse coverage 67.1%, not 95% |
| Gate 3: current DiffPnP3D | PASS | oracle pass; finite-difference relative error `1.014e-9`; NaN mixed-batch backward pass |
| Gate 3: legacy BPnP | FAIL / BLOCKED | relative error `0.537–0.545` over epsilon `0.001–0.1`, versus limit `1e-2` |
| Gate 4: frozen cause decomposition | PASS | GT points recover pose; argmax/softargmax and solver changes do not |
| Gate 5: smoke | PASS_ORDER_FREE_HISTORICAL_ONLY | existing λ=0.005 screen: corner 15.7→13.2 px and good rate 32.0→38.1% |
| Gate 6: paper main claim | NOT REACHED | no prospective clean selector, matched three seeds, or clean multi-domain ablation |

The historical λ screen is single-seed and undertrained. Its front/rear and
kp-channel labels are invalid because the synthetic validation convention is
mixed; only order-free corner/honest8 and channel-agnostic success quantities
remain usable. It supports keeping the already trained λ=0.005 checkpoint as
the frozen audit target, not a new main-effect claim.

## 4. Data, GT, split, and convention audit

### 4.1 Split validity

| issue | result |
|---|---|
| Stage-B train vs val exact JSON/PNG duplicates | 0 / 0 |
| strict real primary | 87 frames from eight non-final sessions |
| manual36 | `capturepallet11` PL-pool; exploratory only |
| later descendant exact strict-filter RGB hits | 23 unique frames across saved descendants |
| historical broad outside evaluator | included sealed cp07/cp09; those results are invalid |
| final-test use in this audit | none |

### 4.2 Synthetic convention

The six Stage-B training roots are nearly fully camera-facing, but
`training_data/val` is not: only 690/1,500 frames pass all current LR-pair
checks and 810 violate at least one. Therefore:

- order-free corner and honest8 metrics remain valid;
- synthetic front/rear and per-channel kp5 claims are invalid;
- ep57 is the historical selector winner, not a prospectively clean
  current-convention winner.

Post-hoc valid terms nevertheless favor ep57: among ep45/48/51/54/57 it has
the lowest honest8 (8.4), tied-lowest corner (8.0), highest good rate (67.0%),
lowest gross rate (7.5%), and highest detection rate (88.9%). This makes ep57
the defensible frozen target, but it does not erase the original selector
caveat.

### 4.3 Camera, geometry, and centroid definition

- Real JSON K exactly matches the corresponding allowed-session `cam_K.txt`.
- Dimensions are named width/depth/height in metres; correct versus W/D swap
  is unambiguous on 87/87 strict frames.
- GT id8 is the projection of the 3-D origin. Against stored-pose origin its
  strict p95 difference is 0.065 px.
- GT id8 is not the arithmetic mean of projected corners; that difference has
  strict median 3.731 px and p95 12.198 px.
- Replacing id8 by the 2-D mean would introduce a definition error.

### 4.4 Sentinel

Only exact `(-1,-1)` is missing. Strict N87 contains 13 such corner entries in
9 frames and 29 other legitimate off-image corners. Blanket negative-coordinate
masking is wrong. Keeping sentinels in PnP inflates strict reprojection p95 from
6.843 to 82.04 px and rotation-error p95 from 0.312° to 20.17°.

The diagnostic uses exact-pair masking. Existing common evaluator/annotator
paths that accept raw `[-1,-1]` remain documented as unsafe; they were not
silently changed in this dirty user worktree.

## 5. Full frozen ep57 result

All continuous pose numbers below are conditioned on successful finite poses;
success/failure is shown separately. Yaw is 180°-symmetry-aware and referenced
to Y0. ADD-S counts vary because candidates with incompatible selected W/D
hypotheses are not compared.

### 5.1 Yaw cause ladder

| stage | input | success | yaw median ° | fixed-GT reproj median px | ADD-S median m (n) |
|---|---|---:|---:|---:|---:|
| Y0 | GT 2-D → current APNP | 87/87 | 0.000 | 1.913 | 0.000 (87) |
| Y1 | predicted argmax | 70/87 | 6.796 | 22.228 | 0.403 (34) |
| Y2 | predicted local softargmax | 70/87 | 6.025 | 23.162 | 0.423 (34) |
| Y3 | Y2 + GT centroid | 70/87 | 5.697 | 20.046 | 0.387 (36) |
| Y4 | Y2 + corners-only-PnP origin | 70/87 | 5.779 | 24.887 | 0.446 (36) |
| Y5 | Y2 + GT kp5 | 71/87 | 5.455 | 19.655 | 0.376 (40) |
| Y6 | Y2 without kp5 | 70/87 | 5.839 | 21.991 | 0.440 (35) |
| Y7 | Y2 without centroid | 70/87 | 5.708 | 24.914 | 0.446 (36) |
| Y8 | Y2 masked by GT in-frame only | 70/87 | 6.025 | 23.162 | 0.412 (33) |
| Y9 | Y2 → SQPnP+RefineLM | 70/87 | 6.719 | 24.566 | 0.365 (70) |

Key paired 10,000-replicate session-cluster CIs:

| comparison, candidate−Y2 | mean yaw Δ ° [95% CI] | mean reproj Δ px [95% CI] | interpretation |
|---|---:|---:|---|
| Y0 | −11.473 [−16.342, −7.258] | −33.578 [−42.468, −27.845] | predicted 2-D geometry is the dominant loss |
| Y1 | +0.164 [−0.175, +0.520] | −0.129 [−0.347, +0.143] | argmax vs local softargmax is not causal |
| Y3 | +0.408 [+0.016, +1.030] | −1.665 [−2.758, −0.659] | GT centroid helps reproj but slightly worsens yaw |
| Y4 | +0.723 [−0.104, +1.678] | +3.790 [+1.043, +6.722] | geometry-derived centroid is not a cure |
| Y5 | −2.242 [−5.107, −0.575] | −5.005 [−7.359, −3.371] | accurate kp5 matters |
| Y6 | −0.568 [−1.304, +0.0004] | −0.269 [−1.177, +0.283] | simply dropping kp5 is not reliable |
| Y7 | +0.658 [−0.116, +1.553] | +3.746 [+0.981, +6.724] | removing centroid hurts reprojection |
| Y8 | −0.244 [−0.562, −0.005] | −0.387 [−1.265, −0.001] | very small in-frame effect, not visibility evidence |
| Y9 | −0.350 [−1.760, +0.699] | +0.088 [−0.646, +0.988] | tested solver swap does not recover yaw |

Y0 also improves pose success over Y2 by 19.54 percentage points,
CI [7.55, 38.36]. Y5 rescues one additional frame. Y8 only changes four
frames/two points each and uses coordinate-in-frame, not an occlusion mask.

### 5.2 Keypoint localization

This table conditions on detected points with finite GT. “Good” is <10 px and
“gross” is >20 px.

| kp | detected/87 | soft median px | p90 px | good % | gross % |
|---:|---:|---:|---:|---:|---:|
| 0 | 65 | 4.59 | 15.93 | 83.1 | 9.2 |
| 1 | 59 | 12.11 | 125.5 | 44.1 | 39.0 |
| 2 | 60 | 12.55 | 126.2 | 40.0 | 40.0 |
| 3 | 65 | 6.21 | 17.95 | 73.8 | 9.2 |
| 4 | 68 | 18.50 | 66.63 | 20.9 | 47.8 |
| 5 | 68 | 25.99 | 113.1 | 29.4 | 57.4 |
| 6 | 68 | 28.25 | 113.1 | 14.7 | 61.8 |
| 7 | 68 | 18.96 | 61.97 | 23.9 | 46.3 |
| 8 | 74 | 16.94 | 54.74 | 27.0 | 40.5 |

Pooled detection is 595/783 channels (76.0%); 54/87 frames detect all nine.
Local softargmax improves paired point error over argmax by only 0.688 px in
median and wins 56.5% of points. The upstream belief geometry, rather than
argmax versus local softargmax, is therefore the larger real-data problem.

### 5.3 kp5, kp6, and footprint distortion

Replacing one prediction with GT produces these mean deltas on the 70
Y2-success frames:

| kp | yaw-error Δ ° | fixed-GT reproj Δ px |
|---:|---:|---:|
| 0 | −1.565 | −2.488 |
| 1 | −0.134 | −6.431 |
| 2 | −0.357 | −6.790 |
| 3 | −1.333 | −2.571 |
| 4 | −1.256 | −3.154 |
| 5 | −2.242 | −5.005 |
| 6 | **−2.343** | **−5.326** |
| 7 | −1.079 | −2.889 |
| 8 | +0.408 | −1.665 |

At a ±1 belief-cell x perturbation, fixed-dimension SQPnP+LM mean yaw pose
displacement is 1.576° for kp5 and 1.621° for kp6; no other point exceeds
1.236°. Their y sensitivity is small (0.059° and 0.072°). This supports a
shared horizontal/far-depth sensitivity, not a kp5-only defect.

For detected kp5, error median is 25.986 px, dx median −8.159 px, and dy median
+1.023 px. Median predicted/GT ratios are:

| geometry | median ratio | interpretation |
|---|---:|---|
| far top edge 4–5 | 0.960 | slightly short |
| depth edge 1–5 | 0.813 | compressed |
| vertical edge 5–6 | 1.372 | expanded |
| far-face perimeter | 0.975 | near-neutral aggregate |

Thus “kp5 is uniformly over-compressed” is too simple. The pattern is
anisotropic and shared with kp6/far-face geometry. Fixed kp5 down-weighting is
not yet justified.

### 5.4 Centroid

The detected learned centroid has median error 16.94 px. On common frames,
corner-mean and corners-only-PnP-origin alternatives are not better: the
alternative-minus-learned median errors are +0.666 and +0.801 px. Learned
centroid error correlates strongly with the two alternatives
(Spearman 0.936 and 0.900), showing a shared footprint displacement.

Bias changes materially with domain and conditioning. For detected points,
night/outdoor dx means are −0.16/−20.72 px; over all finite local maps the
means change to +12.46/−6.04 px. The canonical evaluator and exploratory
manual set also give opposite x-bias signs. There is no stable universal
`x` or `y` pixel offset.

The Y3/Y4/Y7 interventions independently refute centroid as the primary yaw
cause. A fixed upward shift, arithmetic corner mean, or automatic centroid
replacement should not be adopted.

### 5.5 PnP solver-only comparison

Direct solvers below use the same predicted softargmax points, fixed
dimensions, cheirality checks, and locked-Y0 reference.

| solver | success | yaw med ° | fixed-GT reproj med px | ADD-S med m | runtime med ms |
|---|---:|---:|---:|---:|---:|
| EPnP | 70/87 | 7.484 | 28.459 | 0.328 | 0.086 |
| EPnP+RANSAC | 70/87 | 5.891 | 20.886 | 0.301 | 0.303 |
| SQPnP | 70/87 | 7.007 | 27.093 | 0.315 | 0.076 |
| SQPnP+RefineLM | 70/87 | 6.719 | 24.566 | 0.365 | 0.135 |
| ITERATIVE | 60/87 | 6.438 | 20.945 | 0.260 | 0.125 |

Against EPnP, every predicted-softargmax yaw-difference CI includes zero.
ITERATIVE loses 10 successes; its conditional lower medians are therefore not
a fair global win. Y9 also fails to recover Y2. Solver tuning may affect
secondary reprojection/runtime, but it does not explain the yaw collapse.

With GT inputs, all main solvers solve 87/87 except ITERATIVE at 86/87; median
yaw errors are 0–0.103°. This further refutes a geometry/solver root cause.

### 5.6 Covariance calibration

All 783 strict local covariance matrices are finite and PSD. No eigenvalue is
below `−1e-9`; detected-point median 1σ axes are 7.74 and 10.71 px.

| nominal coverage | all finite N770 | detected N593 |
|---:|---:|---:|
| 50% | 35.1% | 43.3% |
| 80% | 47.1% | 57.8% |
| 90% | 52.5% | 63.6% |
| 95% | 56.0% | 67.1% |

Detected kp6 has only 44.1% coverage at nominal 95%. The detected mean squared
Mahalanobis distance is 17.96 versus an ideal 2, with a heavy tail.

The signal is still useful for ranking:

| diagnostic vs error | all-finite Spearman ρ | detected ρ |
|---|---:|---:|
| peak | −0.646 | −0.412 |
| peak/second | −0.587 | −0.334 |
| entropy | +0.668 | +0.458 |
| covariance area | +0.667 | +0.456 |

Entropy and covariance-area rankings are nearly identical (ρ=0.9997), so they
are not independent evidence. The finding is “useful but uncalibrated local
difficulty score,” not “valid predicted likelihood.” C0–C4 weighted PnP was
not run and is not part of ep57.

### 5.7 Flip

- Both original and flipped predictions are detected for 555/783 keypoints;
  553 have finite GT.
- Matched consistency median is 13.34 px and p90 54.67 px.
- Consistency-versus-error ρ is 0.224, but within `(kp, domain)` ranking drops
  to 0.081.
- Low-consistency accuracy AUROC is about 0.62. Even consistency ≤5 px has
  only 34% <10 px accuracy: consistently wrong predictions are common.
- On matched points, flip-unwarped error is worse by median +6.54 px and flip
  wins only 33.6%.
- Original/flip pose successes are 70/73; both succeed on 68 frames, with
  median mutual yaw difference 5.96°.

Flip consistency can be an exploratory high-inconsistency reject signal. It is
not calibrated enough to be a standalone pseudo-label acceptance filter or a
paper representation claim.

### 5.8 Truncation, view, domain, and session

True per-keypoint occlusion labels are absent. `gt_visibility` is constant 1.0
and `Vgeom` means coordinate-in-frame only. The following is a
truncation/in-frame slice, not an occlusion experiment:

| slice | frames | kp detection | full9 | detected error med px | Y2 success |
|---|---:|---:|---:|---:|---:|
| not truncated / Vgeom=8 | 70 | 86.4% | 71.4% | 13.73 | 91.4% |
| truncated | 17 | 33.3% | 23.5% | 38.91 | 35.3% |

Outdoor–Day and Night pooled detected medians are similar (14.57 and 13.94
px), while session behavior varies strongly. `capturepallet04` has only 11.1%
keypoint detection and 0/6 Y2 success; `capturepallet03` and
`capturepallet05` have large tails. The evidence favors session/view/truncation
structure over a simple day-versus-night explanation.

The manual36 PL-pool is much easier (97.5% detection, 7.71 px detected median,
100% Y2 success). It is a selected exploratory population and must not be
pooled with strict N87.

### 5.9 Synthetic order-free result

Synthetic channels are not assigned current kp identities. Across N500:

- mean matched corners: 7.43/8;
- full eight matches: 418/500 (83.6%);
- zero matches: 8/500 (1.6%);
- among 492 frames with a match, local-softargmax order-free frame-median
  error median/p90: 5.98/16.02 px;
- full-eight subset median: 5.51 px.

Partial-match metrics are detection-conditioned and can look better when hard
corners are absent. No synthetic kp5, front/rear, fixed-correspondence yaw, or
PnP conclusion is made.

## 6. Self-training and existing later models

Only acceptance metadata is valid:

| run | R1 accepted/rate | R2 accepted/rate |
|---|---:|---:|
| night RANSAC+LOO | 64 / 12.8% | 169 / 33.8% |
| combined RANSAC+LOO | 151 / 10.1% | 360 / 24.0% |
| combined RANSAC+LOO+flip | 110 / 7.3% | 241 / 16.1% |

Historical R0/R1/R2 performance is invalid for paper use because the
real-unlabeled pool shares validation sessions and broad outside evaluation
included sealed sessions. There is no nested quantity sweep or equal-count
quality sweep. No self-training model is selected.

## 7. Required experiment tables

| requested table | status | usable result |
|---|---|---|
| A frozen baseline | COMPLETE | strict N87, manual36 separate, synthetic order-free |
| B PnP solver | COMPLETE frozen-only | five direct solvers with success, pose, runtime, paired CI |
| C loss ablation | PARTIAL | historical single-seed order-free λ screen only |
| D covariance weighting | BLOCKED | calibration-only; no W0–W5 matched run |
| E centroid/kp5 | COMPLETE frozen counterfactual / BLOCKED training | Y3–Y7, LOO/GT replacement, perturbation |
| F occlusion weighting | BLOCKED | no occlusion metadata or O0–O4 runs |
| G flip representation | PARTIAL | frozen consistency/accuracy only; no clean F0–F3 |
| H self-training ratio/quality | BLOCKED | acceptance metadata only; required sweeps absent |
| I domain robustness | DESCRIPTIVE ONLY | strict domain/session slices; no matched 3-seed method comparison |

## 8. Figures and machine-readable artifacts

| requested figure | artifact / status |
|---|---|
| 1 yaw cause ladder | `full_ep57_frozen_20260728/yaw_cause_ladder.png` |
| 2 keypoint yaw influence | `full_ep57_frozen_20260728/keypoint_influence_delta_yaw.png` |
| 3 kp5 perturbation | `full_ep57_frozen_20260728/kp5_perturbation_sensitivity.png` |
| 4 centroid vs elevation | `full_ep57_frozen_20260728/centroid_residual_vs_elevation.png` |
| 5 covariance ellipses | `full_ep57_frozen_20260728/covariance_ellipse_examples.png` |
| 6 covariance coverage | `full_ep57_frozen_20260728/covariance_coverage_calibration.png` |
| 7 confidence vs error | `full_ep57_frozen_20260728/confidence_vs_error.png` |
| 8 solver comparison | `full_ep57_frozen_20260728/solver_yaw_reproj_add.png` |
| 9 flip reliability | `full_ep57_frozen_20260728/flip_reliability.png` |
| 10 self-training ratio | BLOCKED; acceptance-only substitute is `self_training_acceptance_curve.png` |
| 11 R0/R1/R2 domain | INVALID/BLOCKED by leakage |
| 12 metric + CI | `full_ep57_frozen_20260728/metric_ci_barplot.png` |

The CI plot uses candidate-minus-baseline. Negative improves error; positive
improves success rate. Its right success-rate panel must be read by the panel
title rather than the generic error-axis phrase.

Machine-readable primary outputs:

- `full_ep57_frozen_20260728/{frames,keypoints,yaw_ladder,keypoint_influence,keypoint_perturbation,kp5_perturbation,kp5_geometry,solver_comparison,flip_consistency,flip_keypoints}.csv`
- `full_ep57_frozen_20260728/{summary,manifest,analysis_summary}.json`
- `full_ep57_frozen_20260728/frozen_tables.csv`
- `full_ep57_frozen_20260728/frozen_tables.md`
- `full_ep57_frozen_20260728/RUN_PROVENANCE.md`

## 9. Code findings and changes

One shared-code safety defect was reproduced and minimally fixed in the
untracked current `Deep_Object_Pose/train/diffpnp3d_loss.py`: non-finite
predicted 2-D observations were previously checked only after batched
solve/eigendecomposition, so one NaN frame could abort or poison a healthy
frame. The fix records the original finite mask, substitutes finite internal
values before algebra, and gates the invalid frame from loss and gradients.
Finite inputs are value-identical.

Regression coverage:

- new frozen-core convention tests: yaw axis, exact sentinel, clamped boundary
  local moments;
- geometry audit: Gaussian mean/covariance, operational local decoder, legacy
  BPnP versus current DiffPnP gradients, mixed NaN/healthy batch;
- relevant final pytest set: 41 passed;
- full frozen inference: 623/623, zero errors.

The ep57 checkpoint was not modified. No GT, data, or checkpoint was
overwritten. Development smokes a–d are explicitly marked `INVALIDATED`; smoke
e is the corrected schema smoke.

## 10. Blocked work and next admissible steps

```text
BLOCKED:
필요한 항목: calibrated covariance mapping and matched W0–W5 checkpoints/runs
현재 확인한 위치: full_ep57_frozen_20260728/keypoints.csv and covariance figures
시도한 명령: full frozen local-moment calibration on strict N87
실패 원인: raw local covariance is severely undercalibrated and ep57 PnP is unweighted
대체로 수행한 진단: PSD, ellipse coverage, Mahalanobis tail, keypoint/domain correlations
이 blocker가 전체 결론에 미치는 영향: covariance can be reported as a diagnostic score, not a pose-weighting contribution
```

```text
BLOCKED:
필요한 항목: real per-keypoint visibility/occlusion metadata or a valid existing proxy
현재 확인한 위치: frame/keypoint CSV metadata and source JSON schemas
시도한 명령: Vgeom, truncation, bbox, domain, and session slicing
실패 원인: visibility is constant, gt_occluded is missing, and Vgeom is only coordinate-in-frame
대체로 수행한 진단: truncation/in-frame robustness and keypoint detection slices
이 blocker가 전체 결론에 미치는 영향: no occlusion-weighting or occluded-keypoint causal claim is valid
```

```text
BLOCKED:
필요한 항목: session-independent PL pool plus nested quantity and equal-count quality manifests
현재 확인한 위치: data/pallet/results/ralph_selftrain/* and SELFTRAIN_AUDIT.md
시도한 명령: metadata-only R1/R2 acceptance audit
실패 원인: saved pools/evaluations are contaminated and required sweeps do not exist
대체로 수행한 진단: accepted counts/rates and fresh-extraction history
이 blocker가 전체 결론에 미치는 영향: existing self-training performance cannot support selection or a paper claim
```

```text
BLOCKED:
필요한 항목: immutable current-semantic/order-free selector, matched 3-seed candidates, paired CI, domain guards
현재 확인한 위치: stageB_val_select.*, quick_screen_results.md, strict frozen N87
시도한 명령: valid-term checkpoint review, unit gates, full frozen cause decomposition
실패 원인: historical selection used invalid rear semantics; later real models leak sessions; no matched three-seed ablation
대체로 수행한 진단: ep57 order-free dominance, historical λ smoke, strict N87 frozen diagnostics
이 blocker가 전체 결론에 미치는 영향: no new core training claim and no final-test evaluation are admissible yet
```

Recommended next order:

1. Make every metric/PnP consumer exact-sentinel-safe and regression-test the
   shared evaluator, without discarding legitimate off-image points.
2. Freeze an immutable current-semantic or explicitly order-free validation
   selector and re-score the existing Stage-B checkpoints.
3. Align or explicitly separate the training local decoder and deployment
   decoder; test decoder parity before another geometry-loss sweep.
4. Address kp5/kp6/far-depth structure together. First smoke one intervention
   at a time: robust residual, calibrated uncertainty scaling, or conditional
   keypoint weighting. Do not hard-code a kp5 or centroid pixel shift.
5. Treat truncation/in-frame robustness as the primary data/model gap. Do not
   call it occlusion until metadata supports that label.
6. Only after a clean Gate-5 candidate exists, run λ=0 versus λ=0.005 or the
   selected single intervention with three matched seeds and strict paired CI.
7. Freeze every choice, then evaluate the sealed final-test once.

관찰:
GT 2-D를 쓰면 strict N87이 87/87 해결되지만 ep57 local-softargmax 좌표는 70/87만 해결되고 yaw 중앙값 6.025°, fixed-GT reprojection 중앙값 23.162 px이다. 오차는 kp5/kp6, far/depth footprint, truncation과 특정 session에 집중된다. Covariance는 PSD지만 95% coverage가 detected 기준 67.1%뿐이다.

원인 후보:
주원인은 구조화된 heatmap/keypoint localization과 truncation/session sim-to-real gap이다. 보조 후보는 보정되지 않은 local-moment uncertainty와 학습/평가 decoder mismatch다. Legacy BPnP gradient 오류는 별도 경로의 확정 버그다. Geometric loss가 kp5/kp6 bias를 직접 만들었다는 가설은 아직 미해결이다.

지지 증거:
Y0−Y2 yaw 평균 차이는 −11.473°이고 95% CI는 [−16.342,−7.258]이다. GT kp5 치환은 yaw −2.242°와 reprojection −5.005 px를 개선하며, kp6 치환도 yaw −2.343°로 동등 이상이다. Truncation에서 검출률은 86.4%에서 33.3%, Y2 성공률은 91.4%에서 35.3%로 하락한다. Covariance area는 오차와 ρ=0.456(detected)이라 난이도 순위 신호는 있다.

반증 증거:
Y1−Y2와 Y9−Y2 yaw CI는 0을 포함해 좌표 추출 방식이나 solver 교체가 주원인이라는 가설을 지지하지 않는다. GT centroid 치환은 yaw를 평균 +0.408° 악화시키고, geometry centroid와 centroid 제거는 reprojection을 악화시킨다. 모든 covariance가 PSD이므로 covariance 산술/축 교환 자체가 붕괴했다는 가설은 반증된다. Flip consistency의 조건부 상관과 AUROC도 약하다.

현재 판정:
후보 1은 “softargmax 한 줄의 버그”가 아니라 upstream belief와 구조화된 keypoint geometry 문제로 지지된다. 후보 2는 계산 버그가 아니라 calibration 실패로 판정한다. 후보 3은 current DiffPnP3D PASS, legacy BPnP FAIL이다. 후보 4는 GT oracle과 solver 비교로 주원인에서 제외한다. 후보 5인 geometric-loss-induced bias는 관찰상 가능하지만 인과적으로 미확정이며 논문 핵심 기여로 승격할 수 없다.

남은 불확실성:
깨끗한 λ=0 full baseline/3 seeds가 없어 DiffPnP가 kp5/kp6 편향을 만들었는지 분리할 수 없다. 실제 occlusion metadata가 없고 manual36은 PL-pool 선택군이다. Synthetic validation channel convention이 혼재하고 strict 표본은 8개 session뿐이다. ADD paired 수는 W/D hypothesis 일치 조건 때문에 작다.

권장 수정:
Legacy `--geo_loss`는 사용하지 말고 방법명을 정확히 분리한다. Exact sentinel mask를 공용 평가 경로에 우선 적용한다. Current-semantic/order-free selector를 동결한 뒤 decoder parity를 확인하고, kp5/kp6/far-depth 전체를 대상으로 robust 또는 calibration-aware한 단일 smoke를 수행한다. Fixed centroid/kp5 pixel shift와 원시 covariance weighting은 금지한다. Gate 5를 넘은 한 후보만 3 seeds로 확장하고 그 전까지 final-test는 봉인한다.

수정 후 재검증 결과:
DiffPnP3D의 pre-solve non-finite guard 수정 후 current finite-difference 오차 `1.014e-9`, mixed NaN/healthy backward, 관련 pytest 41개가 모두 통과했다. 수정은 ep57 weights와 frozen inference를 바꾸지 않았으며, 전체 frozen run은 623/623 오류 0과 final-test 접근 0으로 완료됐다. 성능 향상 수정은 아직 적용하지 않았으므로 before/after 학습 성능 주장은 없고 final-test도 실행하지 않았다.
