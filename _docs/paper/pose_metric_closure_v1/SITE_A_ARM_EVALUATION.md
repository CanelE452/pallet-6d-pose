# SITE_A — existing arms on the site's held-out recordings

Population: **88 frames** from SITE_A recordings that supply no
adaptation image. Built from `SITE_A_EVAL_ELIGIBLE.csv`; the twelve
`FT_OVERLAP` frames in the earlier 100-row file were removed before any
model result was read, because they overlap fine-tuning data.

No model was trained. Only A8 was inferred, under the frozen replay recipe.

```text
Method                          Det    KpPx  PoseCov      R    Yaw     tcm   IoU3D   ADDsym
───────────────────────────────────────────────────────────────────────────────────────────
R0 synthetic-only             1.000   11.71    1.000   2.44   1.31    8.84   0.609    0.395
R0-CONT replay only           1.000   11.88    1.000   2.66   1.70    8.99   0.614    0.388
R1 naive joint                1.000   12.18    1.000   2.66   1.42    8.28   0.622    0.380
R5 proposed joint             1.000   12.11    1.000   2.79   1.71    9.07   0.592    0.384
A8 day-only site-matched      1.000   12.33    1.000   2.75   1.84    9.64   0.572    0.370
```

## Pre-registered contrasts

```text
contrast                    metric               diff    recording-cluster 95% CI
─────────────────────────────────────────────────────────────────────────────────
A8_DAY_ONLY - R5_PROPOSED   iou3d              -0.020            [-0.132, +0.021]
A8_DAY_ONLY - R5_PROPOSED   add_sym_m          -0.014            [-0.058, +0.010]
A8_DAY_ONLY - R5_PROPOSED   yaw_error_deg      +0.124            [-0.400, +0.572]
A8_DAY_ONLY - R5_PROPOSED   translation_error_cm   +0.565           [-1.217, +10.924]

A8_DAY_ONLY - R0            iou3d              -0.037            [-0.123, +0.031]
A8_DAY_ONLY - R0            add_sym_m          -0.026            [-0.046, +0.033]
A8_DAY_ONLY - R0            yaw_error_deg      +0.530            [-0.225, +1.177]
A8_DAY_ONLY - R0            translation_error_cm   +0.791           [-0.799, +10.265]

A8_DAY_ONLY - R0_CONT       iou3d              -0.043            [-0.147, +0.017]
A8_DAY_ONLY - R0_CONT       add_sym_m          -0.018            [-0.071, +0.052]
A8_DAY_ONLY - R0_CONT       yaw_error_deg      +0.139            [-0.872, +0.680]
A8_DAY_ONLY - R0_CONT       translation_error_cm   +0.645            [-1.321, +8.602]

```

Clusters: 7 recordings. An interval containing zero
means this data does not resolve the comparison — not that the arms perform
the same.
