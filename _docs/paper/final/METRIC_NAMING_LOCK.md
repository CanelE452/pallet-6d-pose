# Metric naming lock

The reader-facing naming authority for this paper. The frozen protocol it describes
is `metric_split_lock.md`, which is preserved unchanged as the historical contract.

**`px` is a 2D localisation metric, not a 6D pose metric.**

## The frozen metric contract has four layers

`metric_split_lock.md` §2 defines them. Only the first two are currently measurable.

```text
layer            status        what it contains
────────────────────────────────────────────────────────────────────────────────
2.1 Detection    READY         detection rate, box AP, ranking
2.2 Keypoint     READY         2D keypoint error, Proj@N, PCK, gross, catastrophic
2.3 Pose         BLOCKED       translation, rotation, yaw, ADD, ADD-S, AUC, 3D IoU
2.4 Operational  NOT EVALUATED fork-pocket alignment success
```

The pixel metric was **not introduced for this paper**. It has been in the keypoint
layer of the frozen contract from the start. What changed is that the pose layer
became blocked, which leaves the keypoint layer as the finest localisation endpoint
currently computable. That is a statement about availability, not a promotion.

## Detection layer — reader-facing names

```text
Detection coverage         IoU@0.5 match rate
Box AP50                   evaluator key box_ap50
Box AP50-95                evaluator key box_ap50_95
AUROC                      frame-level, positives against the real negative set
FPR95                      false-positive rate at 95 percent true-positive rate
```

There is **no** metric called "ranking AP" in the artifacts. Do not invent one.

## 2D keypoint localisation layer — reader-facing names

```text
Pooled supervised keypoint median [px]   headline; lower is better
Pooled supervised keypoint p90 [px]
Proj@5px / Proj@10px / Proj@20px         secondary
gross20                                  2D error above 20 px
catastrophic40                           2D error above 40 px
```

### The exact definition

> 2D keypoint localisation error, measured as the Euclidean distance between the
> predicted and the annotated keypoint in the original-image pixel coordinate
> system, pooled over supervised keypoints of correctly matched detections.

Implementation, verified by reading the evaluator:

```text
challenge/evaluation_v2/paper_real_eval.py:2409
    distances = np.linalg.norm(prediction.keypoints_xy - target.keypoints_xy, axis=1)

coordinates are un-padded first (INFERENCE_PAD = 100 subtracted at :1756 and :1759),
so the comparison happens in original-image coordinates.
```

### Two things this metric is not

```text
not "reprojection error"   nothing is reprojected from a fitted pose. It is a
                           prediction-to-annotation 2D distance. Do not generalise
                           the name.
not order-free             the frozen contract text in 2.2 says "8 corner order-free
                           Hungarian", but the evaluator uses index-wise
                           correspondence and calls no assignment solver
                           (linear_sum_assignment: 0 occurrences).
```

The second point is a **documented divergence between the contract text and the
implementation**, and it matters: index-wise correspondence is why a 90-degree
keypoint index permutation registers as a large error at all. Under order-free
matching those frames would score as correct and the axis-permutation analysis in
the diagnostics could not exist. The reported numbers are index-wise.

### Scale sensitivity

Raw pixel error depends on projected object size. A pallet that projects larger
yields a larger absolute error at the same relative accuracy.

```text
interpret       model-to-model difference within one subgroup
do not          compare absolute pixel values across subgroups and call one
                condition harder than another
```

## Paired diagnostic localisation

```text
paired frame delta       R0 versus an arm on the same frames and same keypoints
NME                      keypoint error normalised by the projected cuboid diagonal
```

**NME is a post-hoc scale-normalised diagnostic metric.** It was introduced during
the V2-V5 diagnosis, after results on the development population had been seen. It
is never promoted to a frozen primary metric and never appears in Table 1, Table 2,
or the abstract headline.

It is used for one purpose: testing whether an observed condition difference could
be explained by object scale. It is not a replacement for the frozen pixel metric,
and no sentence in this paper says the metric was changed because the pixel metric
was unfair.

Permitted locations: Discussion, diagnostic appendix, failure analysis — always
labelled *post-hoc scale-normalised diagnostic*.

## 6D pose layer

```text
POSE_METRICS_STATUS = REPORTABLE          (amended 2026-09-04)
```

```text
rotation error      REPORTABLE   name it "rotation error [deg]"
yaw error           REPORTABLE   folded to 0-90 under the 180-degree class
translation error   REPORTABLE   report in cm
3D IoU              REPORTABLE   name it "oriented IoU3D" — never "IoU"
ADD / ADD-S AUC     REPORTABLE   name it "symmetry-aware ADD AUC"
5cm5deg             WITHDRAWN    by the frozen contract's 2026-08-26 revision,
                                 independently of the pose layer
```

The 6D reference is named **geometry-reconstructed 6D reference pose** (or
*geometry-resolved*).  Never "metrology-grade GT", "ground-truth sensor pose", or
"motion-capture GT" — it is reconstructed from manual 2D cuboid keypoints,
calibrated intrinsics and registered physical dimensions.

Pose columns now appear in a **separate pose table**, not merged into the
2D/detection table.

Pipeline description is permitted and accurate:

```text
allowed     "the predicted 2D keypoints are consumed by a PnP solver"
forbidden   "our method improves 6D pose"
forbidden   "our method reduces yaw error"
```

## Operational layer

```text
NOT EVALUATED
```

Fork-pocket alignment success requires a closed-loop experiment that was not run.
No sentence in this paper claims an operational or insertion-success benefit.

## Forbidden phrasings

```text
primary pose metric                 there is none; the pose layer is blocked
pose accuracy (as our result)       blocked layer
6D localisation                     the measured quantity is 2D
corner error, used alone            define as 2D keypoint error on first use
reprojection error                  nothing is reprojected
leave-one-keypoint-out / LOO        use single-keypoint-removal reprojection consistency
flip reprojection                   use horizontal-flip keypoint consistency
```

## Permitted phrasings

```text
2D keypoint localisation error [px]
pooled supervised keypoint median error [px]
the reported 2D localisation endpoint
fine 2D keypoint localisation
gross 2D localisation error          (for gross20)
catastrophic 2D localisation error   (for catastrophic40)
post-hoc scale-normalised diagnostic (for NME)
```
