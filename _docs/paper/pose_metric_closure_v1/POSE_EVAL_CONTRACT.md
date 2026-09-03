# Pose evaluation contract

Fixed before any model result is computed. Every definition here is backed by a
passing test in `scripts/paper/pose_metric_closure_v1/`.

## The symmetry group

Both pallets are rectangular, so a half turn about the vertical axis maps the object
onto itself and a quarter turn does not.

```text
plastic   1.10 m x 1.30 m footprint     aspect 1.182
wood      0.80 m x 0.59 m footprint     aspect 1.356

S = { I , Ry(180 deg) } = { I , diag(-1, +1, -1) }
```

**90 and 270 degrees are not in the group.** A rotation that changes which face the
forks enter is a different pose, however symmetric the outline looks. Admitting 90
degrees would erase the axis-selection failure from every metric at once — the test
`test_group_aware_add_does_not_behave_like_unrestricted_nearest_neighbour` shows the
unrestricted nearest-neighbour ADD-S returning exactly 0.0 for a 90-degree swap on a
square footprint.

## Yaw error

```text
delta      = | wrap180( yaw_pred - yaw_gt ) |
yaw_error  = min( delta , 180 - delta )

range      0 .. 90 degrees
```

A wrong long/short axis choice lands at 90, which is the maximum. That is deliberate:
the metric must not be able to hide the failure this study is about.

## Rotation error

```text
R_error = min over S of  geodesic( R_pred , R_gt @ S )

geodesic(A, B) = arccos( ( trace(B^T A) - 1 ) / 2 )
```

Not the bare geodesic. Under the declared 180-degree equivalence a half-turn must
score zero, and it does.

## Translation error

Measured at the pallet centre.

```text
t_error_m  = || t_pred - t_gt ||_2
t_med [cm] = 100 x median( t_error_m )
```

Appendix split, since a monocular estimator fails differently along depth:

```text
lateral_cm = 100 x || (dx, dy) ||
depth_cm   = 100 x | dz |
```

## Oriented 3D IoU

Exact, as `metric_split_lock.md` 2.3 requires — twelve half-spaces, vertex
enumeration over plane triples, convex-hull volume. No axis-aligned approximation and
no sampling.

```text
IoU3D = vol(A ∩ B) / ( vol(A) + vol(B) - vol(A ∩ B) )
```

Implementation `challenge/evaluation_v2/oriented_iou3d.py`, 16 analytical tests.
Closed-form checks that pin it down:

```text
same box                        1.0
disjoint                        0.0
faces touching                  0.0
shift by fraction f along x     (1-f) / (1+f)
180-degree turn                 1.0    both pallets
90-degree turn, plastic         1.10^2 / (2 x 1.10 x 1.30 - 1.10^2)   about 0.79
90-degree turn, wood            0.59^2 / (2 x 0.80 x 0.59 - 0.59^2)   about 0.53
90-degree turn, square          1.0    the control that shows why this matters
45-degree turn                  < 0.95, which an axis-aligned approximation
                                would report as about 1.0
```

Wood penalises the swap harder than plastic because its aspect ratio is further from
square. That is the same quantity that makes the plastic selector hard.

## Symmetry-aware ADD

```text
ADD_sym = min over S of  mean_i || T_pred X_i  -  T_gt S X_i ||
```

Corresponding points, minimised over the **declared** group only. This is not the
unrestricted nearest-neighbour ADD-S, and it is not named ADD-S in the paper unless
the two are first shown to coincide for these objects.

Strict signed ADD is **not** a primary metric here. The human review deliberately
does not resolve the 180-degree sign, so a strict front/back-sensitive ADD would be
reporting a quantity the ground truth does not pin down. If it appears at all it goes
in the appendix, labelled.

## Pose AUC

```text
AUC = area under accuracy(tau) for tau in [ 0 , 0.1 x diameter ]
      normalised to [0, 1]

diameter               the model's maximum pairwise point distance
integration points     1001        FROZEN
```

Threshold-free by construction, which is why the frozen metric contract moved to it
from `5cm5deg`: there is no threshold left to pick after seeing results.

**One implementation is canonical.** The repository had two with different
discretisation (`pose_metrics.py` at 1001 points on normalised errors, and
`re_metrics.py` at 100 steps on raw errors plus diameter). This track uses
`symmetry_aware_pose_metrics.pose_auc`, at 1001 points, and the resolution does not
change after results are seen.

## Unresolved poses

The selector may decline to choose, returning `POSE_UNRESOLVED`. Dropping those
frames from the table would inflate every number, so three quantities are always
reported together:

```text
pose coverage            fraction of frames with a resolved pose
conditional pose error   computed on resolved frames only
failure-aware score      computed over all frames
```

Axis-GT coverage is reported the same way: frames the human marked unclear are
counted, not quietly discarded.

## Where the ground truth enters

```text
predict_pose_without_gt(keypoints, intrinsics, dimensions, config)
    no ground-truth parameter, by design and by test

score_pose_against_gt(..., target_rotation, target_translation, ...)
    called only after the prediction path has finished
```

The test `test_prediction_path_signature_takes_no_ground_truth` asserts the selector's
parameter set is exactly `{predicted_keypoints, camera_intrinsics,
physical_dimensions, config}`, so a reviewer can confirm the separation by reading a
signature rather than trusting a claim.

## Oracle path

A second, clearly separated path uses the human-reviewed long axis to pick the
hypothesis. Every record it emits carries `is_oracle = True` and `mode = "oracle"`,
and no oracle value appears in a main table.

It exists to split one question in two:

```text
main poor / oracle good    the axis selector is the bottleneck
main poor / oracle poor    keypoint localisation or PnP geometry is also a bottleneck
```

## What the human was asked

```text
asked       which axis is the physical LONG side, A or B
not asked   the 180-degree sign
not asked   keypoint positions, yaw in degrees, rotation matrices, translations,
            which PnP candidate to prefer, or anything about a model
```

The review GUI never displays a model prediction, a selector score, or an error
value. An annotator who can see the model's answer stops being an independent
measurement.

## Frozen before results

```text
symmetry group            { I, Ry(180) }
yaw folding               min(delta, 180 - delta)
rotation minimisation     over S
AUC range                 [0, 0.1 x diameter]
AUC resolution            1001 points
IoU3D                     exact oriented, not axis-aligned
ADD variant               group-aware over S
coverage reporting        mandatory
```

None of these changes once a model number has been seen. If the result is
unfavourable it is reported unfavourably.
