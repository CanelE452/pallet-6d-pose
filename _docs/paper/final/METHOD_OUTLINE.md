# Method section outline

Section 3 describes the **frozen study protocol**, not the history of the
experiments. V2 through V5 and the teacher probes are development follow-ups and
belong to Section 6 and the Appendix.

## 3.1 Problem formulation

A single RGB frame is processed by a one-stage keypoint detector (YOLO26n-Pose)
that predicts, per instance, a bounding box and nine 2D keypoints: the eight corners
of the pallet cuboid plus its projected centroid.

Keypoint indices follow a camera-facing convention:

```text
0 1 2 3      the face nearer the camera
4 5 6 7      the far face
{0,1,4,5}    upper corners
{2,3,6,7}    lower corners
8            projected centroid
```

The keypoints are converted to a 6D pose by PnP downstream. **The quantitative
claims in this paper are at the detection and 2D keypoint level.** Pose metrics are
reported nowhere, because the pose selector is unresolved for this evaluation set
(see Limitations); the PnP stage is described because it is the consumer of the
keypoints, not because it is measured here.

## 3.2 Synthetic source supervision

The source model is trained only on rendered images with exact geometric labels.
No real image and no real pose annotation enters source training. This model is
referred to as **R0** and is the baseline against which every adapted student is
compared.

State: renderer, number of frames, pallet models, domain randomisation axes,
backbone, input resolution, and the fact that horizontal-flip augmentation is
disabled. Flip augmentation must stay off: if enabled, horizontal-flip keypoint
consistency becomes an identity the training objective already enforces, and it
would stop being an independent reliability signal.

## 3.3 Target-domain self-training

A **static** teacher — the frozen R0 checkpoint — is run once over unlabeled target
RGB. Its predictions become pseudo bounding boxes and pseudo 2D keypoints. There is
no teacher refresh and no iterative round in the main protocol: every student starts
from the same checkpoint and consumes the same frozen prediction cache, so teacher
drift never enters the comparison between selection rules.

State explicitly that the adaptation pool contains no frame from the evaluation
sessions.

## 3.4 Confidence pre-filter

Two thresholds gate every downstream rule: a box-confidence floor and a per-keypoint
validity floor.

Note plainly that the detector's confidence output is **not a calibrated
probability**. It is used as an ordering signal, not as a likelihood.

## 3.5 Projective consistency

Given the predicted 2D keypoints and the known object geometry, a pose is fitted and
the keypoints are reprojected; large reprojection residuals indicate a prediction
that is not geometrically realisable.

**Single-keypoint-removal reprojection consistency** strengthens this: the fit is
repeated with one keypoint withheld at a time. A single badly placed corner
contaminates the fit that would otherwise expose it, so the removal variant
identifies which corner is responsible rather than only that the frame is bad.

Terminology: this is never abbreviated to "LOO" in any paper-facing text.

## 3.6 Equivariant consistency

The image is mirrored horizontally, the model is run again, and the predicted
keypoints are un-mirrored and re-indexed through the flip permutation
`[1,0,3,2,5,4,7,6,8]`. Disagreement between the two views marks predictions that are
not equivariant under a transformation the object class is symmetric to.

Terminology: **horizontal-flip keypoint consistency**. Never "flip reprojection" —
no reprojection is involved.

The umbrella term for 3.5 and 3.6 together is **projective and equivariant
consistency**.

## 3.7 Exposure-matched student training

Every arm holds constant:

```text
initialisation checkpoint     identical, hash-verified
optimiser updates             identical
pseudo-label exposures        identical
synthetic-replay exposures    identical
synthetic-replay membership   identical, hash-verified
learning rate, batch, seed    identical
augmentation                  identical, horizontal flip disabled
checkpoint selection          fixed final checkpoint, never selected on a metric
```

Only the pseudo-label selection rule varies. When a stricter rule yields fewer unique
labels, the fixed exposure slots are filled by sampling with replacement, so that
selection quality is never confounded with training quantity.

A synthetic-replay-only control arm replaces the pseudo slots with additional
synthetic replay under the same budget. It answers how much of any observed change
comes from further optimisation alone.

## 3.8 Evaluation separation

Three families of measurement, reported separately and never merged:

```text
detection coverage      per acquisition condition
confidence ranking      positives against a real negative set
keypoint localisation   original-image pixel error on supervised keypoints,
                        compared on the keypoints both models detected
```

The localisation comparison is **paired**: a rule that discards hard keypoints would
otherwise improve its own score by shrinking its population. Coverage is reported
alongside so that any such shrinkage is visible.
