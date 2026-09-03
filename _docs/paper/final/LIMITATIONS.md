# Limitations

Stated plainly. No limitation here ends with a sentence that takes it back.

## 1. The evaluation population is a development population

```text
PAPER_EVAL   319 positive frames, 2,689 real negative frames
             population_contract.role = DEV
             held_out_final = false
```

It was consulted repeatedly during development. **No number in this paper is a
held-out result**, and none is described as one. There is no sealed test set for
this study.

## 2. The diagnostic tracks are post-hoc

V2, V3, V4, V5, the separability analysis, and the three teacher-consensus probes
were all designed after results on this population had been seen. They support
mechanism analysis; they are not independent confirmation.

For three of them the version history cannot even order contract against result:

```text
V4                     method lock and result in the same commit
strong-teacher audit   purpose, lock and result in the same commit
fast-teacher probes    membership freeze and results in the same commit
```

File modification times suggest the intended order, but a file timestamp is not
provenance. These are reported as diagnostics and nothing more.

Two arms in the V1 namespace are also post-hoc: `R6_CONF_FLIP` and the `B_CONF_*`
family do not appear in the exposure lock's `arms[]` and were created after the
main results were seen. They are not in any main table.

## 3. The pose reference is reconstructed, not measured by a sensor

```text
POSE_METRICS_STATUS = REPORTABLE   (amended 2026-09-04)
```

The 6D layer is reported, but the reference it is scored against is a
**geometry-reconstructed 6D reference pose**: manual 2D cuboid keypoints,
calibrated intrinsics and registered physical dimensions, resolved under a rule
frozen before any 6D result was seen. It is not metrology-grade sensor ground
truth, not a motion-capture pose, and it inherits the annotation noise of the
manual keypoints. No model prediction was used to choose it.

An earlier version of this document stated that pose metrics were blocked. That
was the first-pass diagnosis, which attributed the block to the axis selector. The
actual blocker was the absence of a ground-truth physical axis; resolving the
reference opened the metrics without modifying the selector. The first-pass record
is preserved in `PAPER_CLAIM_LOCK.json` under `pose_metrics.historical_first_pass`.

What did **not** change: no improvement in 6D pose is claimed. Of 24 metric blocks
in the paired bootstrap, zero resolve in the improvement direction under session
clustering. The axis selector also remains weak — 0.59 to 0.65 measured against a
0.95 gate — and the square-ish footprint of one pallet type makes width and depth
visually interchangeable, which is the same ambiguity behind the 90-degree keypoint
permutations discussed in the analysis. That is now reported as a diagnostic
finding rather than used to withhold the metrics.

## 4. A 2D keypoint metric is not insertion success

The paper reports original-image pixel error on supervised keypoints. That is not
the same quantity as final 6D pose accuracy, and neither is the same as whether a
forklift's forks enter the pockets. A model could improve on this metric and fail
in deployment, or the reverse. No closed-loop result is claimed.

## 5. A constrained pallet family

A small number of pallet types under one deployment setting. Nothing in this study
supports a claim about pallets outside the studied category, and none is made. The
words *unseen*, *arbitrary*, and *generalisation to new pallets* do not appear as
claims anywhere in the paper.

## 6. No independent confirmation population

Related to (1) but worth stating separately: the study never opened a population
that had not been used for development. This is why the paper's positive claims are
restricted to the Tier A frozen arms and why every diagnostic is labelled.

## 7. The nighttime subgroup is small

```text
subgroups.Nighttime   N = 50, plastic only
```

The headline nighttime detection comparison rests on 50 frames of one material.
Every use of this subgroup prints its N. A separate, broader lighting split exists
(`Lighting_night`, N = 106, plastic and wood) and the two are never interchanged.

## 8. Ranking uncertainty is only partly available

Paired bootstrap intervals exist for detection and for keypoint error. As of
2026-09-04 a **frame-level** paired interval also exists for the ranking metrics,
computed from the frozen per-frame scores with no new inference:

```text
R5 - R0   AUROC  +0.00318  95% frame CI [+0.000092, +0.006898]   excludes zero
R5 - R0   FPR95  -0.01339  95% frame CI [-0.02566,  +0.005578]   contains zero
session_cluster_bootstrap_95ci = UNAVAILABLE_FOR_CURRENT_DEV_NEGATIVE_CAPTURE
```

The session-clustered interval remains uncomputable for a concrete reason: every
negative row in the per-frame artifacts carries an empty session identifier, so the
negative pool cannot be resampled by cluster. A partial interval that resamples
only the positive sessions was computed and is recorded, but it does not cover
negative-side variability and is never reported as a session-clustered interval.

The frame-level interval was computed **after** the ranking was observed, so it is
a Tier-B follow-up to Tier-A point estimates. The ranking result is therefore
phrased as *the highest observed* AUROC with a positive paired frame-level
difference — never as a statistically confirmed or session-level significant
improvement.

The detection result carries the same caution from the other direction: the overall
R0-versus-full-filter detection difference is not resolved
(`p_better` 0.121 frame-level, 0.244 session-clustered).

## 8b. No untouched confirmation population remains

Every selection track, teacher probe and no-train screen consumed PAPER_EVAL as a
development population. There is no population left that the current method search
has never consulted, so nothing in this study can be upgraded to confirmatory by
re-running it. Opening confirmation would require a new capture and a protocol
frozen before any result on it is observed.

## 8c. Post-hoc diagnostics are kept separate from the frozen comparisons

The exploratory tracks — the no-train pose screens, the temporal pilot, the depth
gates — were designed after PAPER_EVAL results had been seen. They appear only as
diagnostics and never in the same table block as the frozen adaptation arms.

## 9. No closed-loop forklift evaluation

Every number is offline, on recorded frames. There is no experiment in which a
vehicle acted on the model's output. Claims about deployment benefit would need
that experiment and are not made.

## 10. Raw pixel error depends on object scale

The keypoint metric is measured in original-image pixels, so a pallet that projects
larger yields larger absolute errors for the same relative accuracy. This makes
absolute pixel values **not comparable across condition rows** in the robustness
table — a far pallet is not "easier" because its pixel error is smaller.

Only the R0-versus-adapted comparison *within* a row is interpreted. Where a
scale-normalised quantity is needed, the diagnostics use error normalised by the
projected cuboid diagonal, and the two are never mixed in one column.

The evaluator also compares keypoints **index by index**, not with an order-free
assignment, although the frozen contract text in 2.2 mentions Hungarian matching.
The reported numbers are index-wise. This is why a 90-degree index permutation
registers as a large error rather than as a correct detection.

## 11. The reported localisation metric is 2D, and NME is diagnostic only

The headline localisation number is the **2D keypoint layer** of the frozen metric
contract: Euclidean prediction-to-annotation distance in original-image pixels. It
is not a 6D pose error, and it is not an operational forklift-insertion metric.

NME — error normalised by the projected cuboid diagonal — appears only in the
diagnostics. It was introduced during the post-hoc V2-V5 analysis, after results on
this population had been seen. It is never substituted for the frozen pixel metric,
and no result is re-reported under it.

## 12. Mostly single-seed

Replicate runs exist for a subset of arms only, and those replicates were introduced
after the main results had been seen. Effects smaller than the seed-to-seed spread
are not claimed anywhere.
