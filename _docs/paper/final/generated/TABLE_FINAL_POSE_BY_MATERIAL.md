# Table — 6D pose by material

Population: the in-house real-image evaluation set, 319 positive frames.
Ground truth comes from `GEOMETRY_RESOLVED_POSE_GT.json` — the physical long axis is
resolved from manually annotated keypoints, calibrated intrinsics and the registered
pallet dimensions. No model prediction enters the ground truth.

Orientation is a 180-degree equivalence class, so yaw folds to 0-90 and a wrong
long/short assignment is never absorbed. `ADDsym AUC` is the group-aware ADD over
{I, Ry(180)}, integrated over [0, 0.1 x diameter] at 1001 points.

Every arm was inferred once under one frozen recipe. The R0 replay reproduced its
existing cache exactly, which is what makes a cached arm and a freshly inferred arm
comparable.

Wood is included. Its footprint aspect ratio is 1.356 against plastic's
1.182, which makes its two axis hypotheses further apart geometrically.

## Plastic  (N = 194)

```text
Method                              PoseCov  AxisAcc   R med  Yaw med  t med cm   IoU3D  ADDsym AUC
───────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                   1.000    0.706    2.13     1.19     10.47   0.586       0.345
Source-only continuation              0.995    0.674    2.41     1.32     10.78   0.572       0.325
Naive self-training                   1.000    0.727    2.20     1.17     10.68   0.582       0.331
Confidence self-training              1.000    0.665    2.28     1.31     10.28   0.587       0.336
Reprojection self-training            1.000    0.696    2.23     1.20     10.68   0.595       0.343
Removal self-training                 1.000    0.670    2.42     1.21     10.75   0.587       0.337
Full consistency self-training        1.000    0.655    2.33     1.29     11.16   0.573       0.320
```

## Wood  (N = 125)

```text
Method                              PoseCov  AxisAcc   R med  Yaw med  t med cm   IoU3D  ADDsym AUC
───────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                   1.000    0.816    2.98     1.38      4.20   0.626       0.423
Source-only continuation              1.000    0.808    3.02     1.15      3.95   0.624       0.419
Naive self-training                   1.000    0.832    2.60     1.34      3.76   0.635       0.432
Confidence self-training              1.000    0.816    2.70     1.34      4.42   0.626       0.418
Reprojection self-training            1.000    0.832    2.66     1.23      4.30   0.622       0.406
Removal self-training                 1.000    0.816    2.68     1.32      3.97   0.616       0.406
Full consistency self-training        1.000    0.856    2.65     1.29      4.09   0.634       0.399
```

