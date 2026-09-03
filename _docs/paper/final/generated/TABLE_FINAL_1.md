# Table 1 — Main comparison

Population `PAPER_EVAL`: 319 real positive frames and 2,689 real negative
frames, scored by one evaluator with one metric definition.

The backing artifacts declare `population_contract.role = DEV` and
`held_out_final = false`, and their own reports warn that these are development
values. They are reported here as development results and are never described as
held-out or independently confirmed.

`Pooled kp median [px]` is the **2D keypoint layer** metric of the frozen metric
contract (`metric_split_lock.md` 2.2), not a pose metric. It is the Euclidean
distance between the predicted and the annotated keypoint, index by index, in the
original-image coordinate system after the inference padding is removed, pooled
over supervised keypoints of correctly matched detections. Lower is better.

Raw pixel error is **scale-sensitive**: a pallet that projects larger yields a
larger absolute error at the same relative accuracy. Compare models within a row,
not absolute values across rows.

Pose columns are absent because `POSE_METRICS_STATUS = BLOCKED`. They are removed
rather than left blank, so that no reader mistakes an empty cell for a measured
zero.

`Det` is the IoU@0.5 match rate. `AP50` and `AP50-95` are the evaluator's
`box_ap50` and `box_ap50_95`. `AUROC` and `FPR95` are frame-level ranking
metrics against the 2,689 negatives.

## Panel A — Controlled YOLO arms

Every arm shares one initialisation, optimiser budget, pseudo-label exposure,
synthetic replay membership, augmentation and seed. Only the selection rule
differs.

```text
Method                                    Pooled kp med[px]     Det     AP50   AP50-95    AUROC    FPR95
────────────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                                   6.616   0.975   0.9363    0.7688   0.9921   0.0417
Synthetic-replay control                              6.911   0.966   0.9367    0.7609   0.9872   0.0573
Naive self-training                                   7.120   0.981   0.9292    0.7622   0.9913   0.0558
Confidence                                            7.037   0.987   0.9467    0.7635   0.9923   0.0469
Full consistency filter                               7.210   0.984   0.9580    0.7585   0.9953   0.0283
```

### Secondary 2D keypoint metrics (same frozen layer)

`Proj@Npx` is the fraction of supervised keypoints within N pixels of the
annotation. These are part of the same frozen keypoint layer as the median and
are reported as secondary; the headline column stays the pooled median.

```text
Method                                    Proj@5px  Proj@10px  Proj@20px  gross20
─────────────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                          0.380      0.652      0.828    0.172
Synthetic-replay control                     0.370      0.630      0.818    0.182
Naive self-training                          0.360      0.622      0.820    0.180
Confidence                                   0.374      0.616      0.806    0.194
Full consistency filter                      0.375      0.608      0.803    0.197
```

`gross20` is the fraction of supervised keypoints more than 20 px from the
annotation — a **gross 2D localisation error**, not a pose failure.

Standard reprojection and keypoint-removal selection are not shown here; they
belong to the selection ablation and appear in Table 3.

Real-supervision fine-tuning is deliberately absent: it is a reference upper
bound trained with real labels and does not belong in a block of
unlabeled-adaptation arms.

## Panel B — Architecture reference

```text
Model                                     Pooled kp med[px]     Det
───────────────────────────────────────────────────────────────────
DOPE (same-data backbone)                            10.916   0.737
YOLO26n-Pose (synthetic-only)                         6.616   0.975
```

**Ranking and AP columns are omitted from Panel B on purpose.** DOPE has no box
head, so the box needed for IoU matching is derived from its detected cuboid
corners, and its score is a belief-map peak rather than a box confidence.
`AP50-95`, `AUROC` and `FPR95` would not be the same quantity across the two
rows, so the columns are removed rather than filled with values that invite an
invalid comparison. The keypoint and detection columns are directly comparable:
both are
measured against the same 2D ground-truth keypoints by the same evaluator.

This panel is a **reference**, not a controlled comparison.

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

The detection difference is **not** separated from noise, and the
localisation difference favours R0. `AUROC` and `FPR95` have no matching
interval in the artifacts, which is why the ranking result is stated as
*best observed* rather than as an established improvement.
