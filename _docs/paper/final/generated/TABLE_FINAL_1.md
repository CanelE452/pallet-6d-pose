# Table 1 — Main comparison

Population `PAPER_EVAL`: 319 positive frames and 2,689 real negative frames,
scored by one evaluator with one metric definition. `population_contract.role`
is **DEV** and `held_out_final` is **false** — these are development numbers,
not held-out results.

`Keypoint` is the median error of supervised keypoints in **original-image
pixels**, pooled across frames. Lower is better. `Detection` is the IoU@0.5
match rate. `AP50-95` is the evaluator's `box_ap50_95`. `AUROC` and `FPR95`
are frame-level ranking metrics computed against the 2,689 negatives.

```text
Method                                   Keypoint[px]     Det  AP50-95    AP50   AUROC   FPR95
──────────────────────────────────────────────────────────────────────────────────────────────
DOPE (same-data control, Tier C)               10.916   0.737   0.3412  0.6395  0.9903  0.0409
Synthetic-only (R0)                             6.616   0.975   0.7688  0.9363  0.9921  0.0417
Synthetic-replay control                        6.911   0.966   0.7609  0.9367  0.9872  0.0573
Naive self-training                             7.120   0.981   0.7622  0.9292  0.9913  0.0558
Confidence                                      7.037   0.987   0.7635  0.9467  0.9923  0.0469
+ reprojection consistency                      7.044   0.987   0.7643  0.9417  0.9920  0.0487
+ keypoint-removal consistency                  6.999   0.987   0.7578  0.9366  0.9911  0.0502
+ horizontal-flip consistency (full)            7.210   0.984   0.7585  0.9580  0.9953  0.0283
```

## Uncertainty on the R0 versus full-filter difference

Paired bootstrap. `p_better` is the probability that the full filter is
better than R0 on that axis.

```text
axis                             frame   session-clustered
──────────────────────────────────────────────────────────
detection                       0.1210              0.2437
corner                          0.0282              0.0653
pooled_corner_median            0.0056              0.0952
```

Read this together with the table: the detection gain is **not** separated
from noise, and the localisation difference favours R0.

## Reference rows are not comparison rows

The DOPE row is a **reference**, not a controlled comparison. DOPE has no box
head, so its boxes are derived from detected cuboid corners and its score is a
belief peak rather than a box confidence. `AP50-95`, `AUROC`, and `FPR95` are
therefore not the same quantity across that row. The columns that compare
directly are `Keypoint` and `Det`.

Real-supervision fine-tuning is deliberately absent from this table. It is a
reference upper bound trained with real labels and does not belong in a block
of unlabeled-adaptation arms.

`R med`, `yaw`, and every other 6D quantity are omitted because
`POSE_METRICS_STATUS = BLOCKED`.
