# Pallet line pose geometry and head contract

This module adds a trainable spatial line branch and a differentiable corner
layer to predictions and neck features from the same frozen YOLO26n forward.
The YOLO model, highest-confidence detection selection, image loading,
letterboxing, source hashes, train/selection/evaluation splits, and conversion
back to the original image are owned by the experiment driver. No GT enters
the head's `forward`. This is a new trained branch, not a replay of stored DHT
postprocessing; the original YOLO parameters remain frozen in this experiment.

## Interface and units

```python
head = PalletLinePoseHead(c3=64, c4=128, hidden=16, along_samples=32)
out = head(p3, p4, points, boxes, point_valid, input_shape,
           lam=1.0, geometry_only=False)
losses = compute_loss(out, gt_points, gt_valid,
                      corner_weight=1.0, gt_support=None)
losses['loss'].backward()
```

The installed checkpoint has neck P3 channels64/stride8 and P4
channels128/stride16, as independently inspected by the parent agent.

| Input | Shape | Meaning |
|---|---|---|
| `p3` | B,C3,80,80 | Actual rectangular neck feature, zero-padded bottom/right |
| `p4` | B,C4,40,40 | Same; FP16-rounded cache may be supplied |
| `points` | B,9,2 | Actual predicted points in letterboxed input pixels |
| `boxes` | B,4 | Predicted xyxy boxes in the same pixels |
| `point_valid` | B,9 | Predicted availability, never a GT mask |
| `input_shape` | B,2 | True input H,W before feature-cache padding |

Production points/boxes and the head are float32; cached features are converted
to the head dtype. AMP is disabled by the driver. CPU float64 analytic tests use
float64 geometry. Pixel conversion after the head must preserve the original
baseline decoder convention, e.g. original baseline plus `(q-p)/gain`; lambda0
should bypass conversion and copy the stored original-image baseline exactly.

For `grid_sample(..., align_corners=False)`, the explicit feature-cell mapping is

```
grid_x = 2*x/(stride*padded_feature_width) - 1
grid_y = 2*y/(stride*padded_feature_height) - 1
```

Thus cell index j denotes input coordinate `(j+0.5)*stride`. This is the
declared sampling convention, **not a claim about convolution receptive-field
centers**. The cache dimensions are used for normalization; each sample's true
H,W masks adapter bias and samples outside the actual input rectangle. This
prevents artificial feature-cache padding from supplying image evidence.

## Semantic lines and candidates

Camera-facing 0123 is retained: front0–3, rear4–7, center8, top0/1/4/5,
bottom2/3/6/7. The ordered eight edges are:

```
height: (1,2), (3,0), (5,6), (7,4)
depth:  (0,4), (1,5), (2,6), (3,7)
incident roles by corner:
0:(1,4), 1:(0,5), 2:(0,6), 3:(1,7),
4:(3,4), 5:(2,5), 6:(2,6), 7:(3,7)
```

These are amodal cuboid supporting lines. Four width-axis edges are excluded.
We do not label physical visibility, pallet slats, or image gradients as GT.

For predicted endpoints a,b define midpoint m, length L, normal
`n0=(-(b-a).y,(b-a).x)/L`, and predicted-box diagonal D. A local candidate is

```
n_j = R(delta_theta_j) n0
rho_j = n_j dot m + delta_r_j D
h_j = [n_j.x, n_j.y, -rho_j]
```

The grid is theta-major: 13 angles from -12 to12 degrees, step2; 17 offsets
from -0.08 to0.08 times D, step0.01; 221 lines plus null at index221. Center
candidate index110 reproduces the original predicted line. Angle changes pivot
around m, not the image origin. All normals lie within12 degrees of n0, so
undirected normal-sign seams cannot cancel the local mean. Each candidate is
sampled at32 equally spaced positions over the predicted length L, centered on
the projection of m onto that candidate line. The +/- normal/tangent direction
does not change line geometry, but the implementation keeps a fixed ordering.

P3/P4 each have a 1x1 conv+SiLU adapter to16 channels. Concatenated samples pass
through two32-channel width3 Conv1d+SiLU layers. Mean and max preserve64 image
features per candidate after the learned longitudinal processing. A small MLP
also receives predicted initial normal, normalized midpoint, normalized length,
predicted box log-aspect, an8-dimensional semantic-role embedding, normalized
candidate angle/offset, and sample coverage. A separate MLP produces null.
No Fourier features, GT-coordinate proposals, PnP, fixed aspect ratio, parallel
constraint, visibility predictor, or learned classification reranking is added.

## Distribution and differentiable corner layer

`logits[B,8,222]` jointly include null. Non-null conditional probabilities are
computed by softmax of the first221 logits, independently of null's probability
`p_null` in the full222-class distribution. Define

```
raw_mean = sum_j p(j | non-null) h_j
h_mean = raw_mean / norm(raw_mean[:2])
M = sum_j p(j | non-null) (h_j-h_mean)(h_j-h_mean)^T
```

M is a second moment **about the returned unit-normal mean line**, not exactly
the covariance about unnormalized E[h]. At predicted corner q, z=[qx,qy,1],
`z^T M z` gives the conditional expected squared residual difference from that
returned line, in input-pixel squared units. It summarizes the entire local
candidate distribution; it is neither calibrated coverage nor global angular
uncertainty outside the candidate window.

The corner anchor standard deviation is `max(1,0.025*D)` pixels. Each incident
line has variance `max(1,z^T M z)` and precision
`(1-p_null)*predicted_line_valid/variance`. For each available corner p, solve

```
A = I/anchor_std^2 + lambda sum_r precision_r n_r n_r^T
b = -lambda sum_r precision_r n_r (h_r dot [px,py,1])
q = p + solve(A,b)
```

The positive anchor makes A nonsingular even for parallel incident lines. A
single line constrains its normal direction and cannot determine tangent
position. The ninth center is copied. An unavailable point is copied, including
NaN or -1 sentinels. A line requires finite available predicted endpoints,
length>=2 input pixels, a valid positive box, and at least one in-frame sample
somewhere on its candidate grid. Otherwise it exerts no force. Fully null lines
have exactly zero precision. There is no displacement clipping in this head.

Lambda is an external nonnegative scalar, not a GT-dependent per-frame choice.
Lambda0 is an explicit copy before solving. A zero-precision corner is also
copied. Original box and classification/keypoint confidences are not changed
by this module; no score input is consumed. The driver must retain them.

## Training-only targets and objective

`make_targets(out,gt_points,gt_valid,gt_support=None)` has no gradient and does
not mutate predictions. Targets use the actual predicted line chart from the
same forward. GT normals are aligned to n0 before computing relative angle and
offset. In-range targets bilinearly interpolate four neighboring grid bins.
Out-of-range targets assign full mass to null, rather than forcing an incorrect
boundary candidate. Null means unsupported by this local candidate family,
not invisible. Support requires two available finite GT endpoints and
length>=2 input pixels, a valid predicted chart, and optional additional
`gt_support[B,8]`. Unsupported lines have zero target mass and are masked from
CE. Dataset-derived original-image support can be supplied here only; it never
enters forward inference.

```
L = mean_image(mean_supported_line(cross_entropy))
    + corner_weight * mean_image(mean_available_corner(
        mean_xy(SmoothL1((q-GT)/anchor_std, beta=1))))
```

Only corners0–7 contribute. The point loss uses the intersection of predicted
availability and GT availability. No missing point is filled using GT. Empty
images do not dilute nonempty images; a completely empty batch has a finite,
differentiable zero loss. All target construction must use already augmented
canonical corners, without an independent unsynchronized line augmentation.

Controls share this code: image-line-only uses `corner_weight=0`; geometry-only
uses `geometry_only=True`, zeroing image adapters' output before sampling while
retaining all predicted geometry and the same scorer capacity. Its learned
convolution biases are constants and cannot carry image information. Baseline
is lambda0. A geometry-only gain cannot be attributed to learned line evidence.

## Validation and remaining scope

The following CPU command completed successfully with15 tests:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
  /home/minjae/Documents/github/pallet-pose/scripts/research/pallet_line_pose_v1/test_model.py
```

Tests cover fixed incidence, grid/pivot/center reconstruction, cell sampling and
roundtrip, sign equivalence, null/out-of-range/missing targets, PSD moments and
full-distribution quadratic equality, analytic orthogonal and parallel solves,
uncertain-line suppression, isotropic affine geometry, lambda0, center and
missing preservation, invalid boxes, absence of GT in forward, image-feature
sensitivity, geometry-only image invariance, padded-feature invariance, empty
losses, and final-corner gradients into both adapters and both scorers. A CPU
optimizer step uses real64/128 channels, FP16-rounded80/40 feature caches and a
batch with384x640 and640x448 true inputs. No GPU, source annotation, original
YOLO checkpoint, or prior frozen experiment was changed for these tests.

These checks establish the software and geometry contracts, not an accuracy
gain. Main synthetic training, raw-YOLO parity, held-out selection, final
same-protocol paper evaluation, negative-image behavior, timing/memory, and
6D-pose consequences remain driver responsibilities. This local branch cannot
recover an undetected object or reliably recover a line outside its proposal
window; it may learn overconfident incorrect lines under domain shift.

The spatial evidence aggregation follows the ideas of [Deep Hough Transform,
ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/779_ECCV_2020_paper.php)
and [L-CNN's candidate line sampling](https://github.com/zhou13/lcnn/blob/master/lcnn/models/line_vectorizer.py).
This is a task-specific implementation, not an exact reproduction of either.
Earlier repository strip refiners already sampled features, but used jittered
GT lines for their capacity experiment and single-line correction rather than
this predicted-YOLO-instance distribution with train-through corner fusion.
