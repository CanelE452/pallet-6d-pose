# GT axis resolution — frozen before any pose result

```text
STATUS   FROZEN
DATE     2026-09-03
SCOPE    how the physical long axis of the ground-truth pose is determined
```

Written before a single pose metric was computed. Nothing here changes once a model
number has been seen.

## What this rule does

Each pallet has a rectangular footprint, so its pose is ambiguous under a 90-degree
turn **only if you do not know which physical dimension lies along which
camera-facing axis**. The annotations record keypoints but never confirmed that
assignment: `axis_assignment_confirmed` is `false` on all 319 frames.

This rule fixes the assignment using ground-truth information only.

## Reference inputs — the complete list

```text
REFERENCE_INPUTS
    manually annotated GT 2D keypoints
    calibrated camera intrinsics K
    registered physical pallet dimensions
    object type
```

```text
FORBIDDEN_INPUTS
    R0 prediction
    R5 prediction
    any model prediction
    selector score
    model confidence
    any downstream ADD / yaw / translation / IoU result
```

The builder imports no model, loads no checkpoint, and reads no prediction cache. It
is a function of the annotation, the calibration and the registry.

## Object dimensions

From `POSE_EVAL_OBJECT_CONTRACT.json`, which agrees with the frozen registry and does
not modify it.

```text
plastic_standard_110x130x11    long 1.30 m   short 1.10 m   height 0.11 m
wood_small_80x59x14            long 0.80 m   short 0.59 m   height 0.14 m
```

## The rule

For every frame:

```text
1  build hypothesis A — the camera-facing WIDTH axis is the physical long axis
       edges (0,1) (2,3) (4,5) (6,7)
2  build hypothesis B — the camera-facing DEPTH axis is the physical long axis
       edges (0,4) (1,5) (2,6) (3,7)
3  solve each independently from the GT keypoints:  SQPnP, then RefineLM
4  compute the same reprojection residual for each — mean Euclidean distance
   between the reprojected model corners and the annotated corners, over the
   supervised corners
5  select the hypothesis with the lower residual
6  keep the 180-degree sign as an equivalence class; it is never resolved
```

Both hypotheses are solved by identical code with identical settings. The only
difference is which physical dimension is placed on which axis.

## What is deliberately not decided

```text
the 180-degree sign     orientation is a 180-degree equivalence class throughout
                        this evaluation, so front/back is never pinned down
90 degrees              is NOT in the equivalence class, for either object
```

## Quality condition — reusing an existing threshold, not inventing one

The repository already carries an annotation-quality bar:

```text
scripts/annotate/_audit_annotate.py:200
    ok("reproj 가 합리적", p["reproj_error_px"] < 5.0)
```

Measured on the stored annotations for these 319 frames:

```text
reproj_error_px   median 1.13   p90 2.47   p95 2.78   max 4.48
```

All 319 already satisfy the existing 5.0 px bar. That bar is therefore adopted
unchanged as the condition under which a GT reference is considered trustworthy. No
new numeric threshold is created for this track.

## Exclusion policy

```text
default                      NO AUTOMATIC EXCLUSION
unresolved only when         the solver fails outright (fewer than six usable
                             corners, or SQPnP returns no solution)
always preserved             both hypothesis residuals and their margin, per frame
```

A frame is not dropped for having a small margin. Margins are recorded and reported
so that a reader can see how separable each frame was, but no margin threshold
selects frames in or out. Inventing a cut-off such as "ratio > 2" or "difference >
5 px" after seeing the distribution is exactly the post-hoc choice this lock exists
to prevent.

## The human review is not the source of truth

```text
HUMAN_AXIS_REVIEW_ROLE = ANNOTATION_RELIABILITY_DIAGNOSTIC
```

The first pass and the blind second pass are retained in full. They measure how
reliable a human visual judgement of physical long-axis identity is, and under which
viewing conditions it degrades. They do not define the ground truth.

This is not a claim that the reviewer made mistakes. The reviewer judged **apparent
image length**, which under strong perspective foreshortening is not the same
quantity as physical long-axis length. That distinction is the finding, and it is
reported as such.

## A prior diagnosis that is withdrawn

An earlier note in this track said:

> at low elevation the axis identification is impossible in principle

**That is not supported by the measurements and is withdrawn.** With manually
annotated keypoints and known geometry, the two hypotheses separate by residual in
almost every frame, including at 0-2 degrees of elevation. The two things must be
kept apart:

```text
GEOMETRIC_UNOBSERVABILITY        the information is not in the image
PREDICTION_OR_SELECTOR_FAILURE   the information is there and is not recovered
```

The evidence points at the second, not the first. The exact numbers belong in the
audit and the diagnostics, not in this lock.

## Frozen

```text
reference inputs          fixed
forbidden inputs          fixed
hypothesis construction   fixed
solver                    SQPnP then RefineLM, identical for both hypotheses
selection criterion       lower reprojection residual on GT keypoints
sign                      180-degree equivalence class, never resolved
quality bar               the pre-existing 5.0 px annotation bar
exclusion                 solver failure only
```

None of these changes after a pose number exists. If the resulting pose results are
unfavourable, they are reported unfavourably.
