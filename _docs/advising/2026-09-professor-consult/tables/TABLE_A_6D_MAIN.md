# Table A — downstream 6D pose (main comparison)

population   PAPER_EVAL positive 319 (plastic 194 + wood 125), role DEV
reference    geometry-reconstructed 6D reference pose (manual 2D keypoints + intrinsics + registered dimensions)
source       data/pallet/results/paper_pose_metric_closure_v1/POSE_EVALUATION_<ARM>.json, key paths.MAIN.ALL

```text
arm                               PoseCov ↑  AxisAcc ↑   R med ↓  Yaw med ↓   t cm ↓   IoU3D ↑   ADDsym ↑
---------------------------------------------------------------------------------------------------------
R0  합성만 학습                       1.000     0.7492     2.262      1.231    7.897   0.60318    0.42847
R0-CONT  추가학습만 (ST 없음)         0.997     0.7264     2.499      1.261    8.544   0.59432    0.40915
R1  ST 필터없음                       1.000     0.7680     2.372      1.245    7.796   0.58966    0.42045
R2  ST 신뢰도                         1.000     0.7241     2.477      1.326    7.785   0.59947    0.41580
R3  ST +재투영                        1.000     0.7492     2.349      1.207    7.737   0.59979    0.41489
R4  ST +코너제거                      1.000     0.7273     2.524      1.276    8.055   0.59966    0.41205
R5  ST 전체일관성 (제안)              1.000     0.7335     2.535      1.294    8.827   0.58677    0.40010
```

How to read it in the meeting

- 24 metric blocks over 6 paired comparisons against R0.
- session-cluster intervals excluding zero, improvement direction: **0**
- session-cluster intervals excluding zero, any direction: 0
- With 13 recording groups an interval containing zero means the data cannot
  separate the arms — not that the arms are equal.
- R0_CONT solves 318 of 319 frames (PoseCov 0.997); every other arm solves 319.

Paper role: **main table**. 6D is reportable; a 6D improvement is not claimed.
