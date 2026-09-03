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

## 3. Pose metrics are blocked

```text
POSE_METRICS_STATUS = BLOCKED
blocked_reason      = POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR;
                      FINAL_MANIFEST_NOT_FROZEN;
                      wood: CANONICAL_MIGRATION_NOT_PASS;
                      SYMMETRY_NOT_FROZEN
```

No claim about rotation, translation, yaw, ADD, ADD-S, 3D IoU, 5cm5deg, or 6D pose
AUC appears anywhere in this paper. These columns are **removed** from the
paper-facing tables rather than left as blank cells, so that a reader never sees an
empty pose column and infers the measurement merely failed to run.

The blocker is algorithmic, not a matter of unfinished labelling: the best axis
selector measured reaches 0.65 against a gate of 0.95. More annotation would not
open it. The square footprint of one pallet type makes width and depth visually
interchangeable, which is the same ambiguity that produces the 90-degree keypoint
permutations discussed in the analysis.

Pipeline description remains accurate and permitted: the predicted 2D keypoints are
consumed by a Perspective-n-Point solver.

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

## 8. Ranking differences carry no separate confidence interval

Paired bootstrap intervals exist for detection and for keypoint error. The AUROC and
FPR95 differences have **no** matching interval in the artifacts, and
session-clustered bootstrap is unavailable for the current negative capture:

```text
session_cluster_bootstrap_95ci = UNAVAILABLE_FOR_CURRENT_DEV_NEGATIVE_CAPTURE
```

The ranking result is therefore phrased as *the best observed* AUROC and FPR95 among
the frozen arms, not as a statistically established improvement.

The detection result carries the same caution from the other direction: the overall
R0-versus-full-filter detection difference is not resolved
(`p_better` 0.121 frame-level, 0.244 session-clustered).

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
