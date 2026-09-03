# Table — 6D pose, main comparison

Population: the in-house real-image evaluation set, 319 positive frames
(plastic 194, wood 125), `population_contract.role = DEV` — a repeatedly used
development population, never held-out.

The reference is a **geometry-reconstructed 6D reference pose**
(`GEOMETRY_RESOLVED_POSE_GT.json`): the physical long axis is resolved from manually
annotated 2D cuboid keypoints, calibrated intrinsics and the registered pallet
dimensions, under a rule frozen before any 6D result was seen. No model prediction
enters it. It is not metrology-grade sensor ground truth and it inherits the
annotation noise of the manual keypoints.

Orientation is a 180-degree equivalence class, so yaw folds to 0-90 and a wrong
long/short assignment is never absorbed. `ADDsym AUC` is the group-aware ADD over
{I, Ry(180)}, integrated over [0, 0.1 x diameter] at 1001 points.

Every arm was inferred once under one frozen recipe. The R0 replay reproduced its
existing cache exactly, which is what makes a cached arm and a freshly inferred arm
comparable.

```text
Method                              PoseCov  AxisAcc   R med  Yaw med  t med cm   IoU3D  ADDsym AUC
───────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                   1.000    0.749    2.26     1.23      7.90   0.603       0.428
Source-only continuation              0.997    0.726    2.50     1.26      8.54   0.594       0.409
Naive self-training                   1.000    0.768    2.37     1.25      7.80   0.590       0.420
Confidence self-training              1.000    0.724    2.48     1.33      7.78   0.599       0.416
Reprojection self-training            1.000    0.749    2.35     1.21      7.74   0.600       0.415
Removal self-training                 1.000    0.727    2.52     1.28      8.05   0.600       0.412
Full consistency self-training        1.000    0.734    2.53     1.29      8.83   0.587       0.400
```

`AxisAcc` is reported beside the pose metrics on purpose. On this data a
change in axis accuracy does not translate proportionally into a change in
pose accuracy, so the two must be read together rather than one standing
in for the other.

`PoseCov` is not decoration: the source-only continuation arm solves 318 of
the 319 frames and every other arm solves all 319. The paired bootstrap
against that arm therefore runs on 318 shared frames, which is why its
reference values differ in the fourth decimal from this table's R0 row.
The difference is a population difference, not a discrepancy.

No arm improves on the synthetic-only baseline here. Of the 24 metric blocks
in `POSE_PAIRED_BOOTSTRAP.json`, zero have a session-cluster interval that
excludes zero in the improvement direction. With 13 recording groups that is
a statement about resolving power, not a statement that the arms are equal.

Pose columns other than these are not reported. Strict signed ADD is absent
because the 180-degree sign is deliberately unresolved in the reference.
