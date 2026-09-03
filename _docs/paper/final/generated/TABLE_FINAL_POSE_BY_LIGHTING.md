# Table — 6D pose by lighting

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

The frozen acquisition-condition subgroups, unchanged: Daytime N = 70 and
Nighttime N = 50, both plastic only. No new subgroup was created.

## Daytime  (N = 70)

```text
Method                              PoseCov  AxisAcc  Yaw med  t med cm   IoU3D  ADDsym AUC
───────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                   1.000    0.486     2.00     11.03   0.564       0.290
Source-only continuation              1.000    0.457     2.45     10.91   0.577       0.278
Naive self-training                   1.000    0.471     2.22     10.87   0.575       0.269
Confidence self-training              1.000    0.386     2.77     11.80   0.510       0.243
Reprojection self-training            1.000    0.429     2.25     11.44   0.535       0.279
Removal self-training                 1.000    0.400     2.12     11.92   0.500       0.257
Full consistency self-training        1.000    0.400     2.36     11.19   0.556       0.272
```

## Nighttime  (N = 50)

```text
Method                              PoseCov  AxisAcc  Yaw med  t med cm   IoU3D  ADDsym AUC
───────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                   1.000    0.760     2.35     12.59   0.532       0.288
Source-only continuation              0.980    0.673     2.45     17.00   0.429       0.220
Naive self-training                   1.000    0.820     2.02     14.35   0.574       0.275
Confidence self-training              1.000    0.740     2.01     16.38   0.550       0.258
Reprojection self-training            1.000    0.800     1.87     16.27   0.566       0.270
Removal self-training                 1.000    0.760     1.75     17.14   0.557       0.260
Full consistency self-training        1.000    0.780     1.47     14.40   0.556       0.230
```

