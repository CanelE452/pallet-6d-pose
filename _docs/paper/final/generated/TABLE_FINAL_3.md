# Table 3 — Pseudo-label selection

## Table 3A — Frozen filter-quality evaluation

Population `PAPER_EVAL_PLASTIC_POS`, N = 194. This measures the labels each
rule would pass, not the student it would produce.

`Accepted` is the number of teacher predictions kept; `Retention` is that
as a fraction. `Pass median`, `Pass p90` and `Pass gross20` describe the
keypoint error of what passed, in original-image pixels, with gross =
more than 20 px.

```text
Selection rule                                Accepted  Retention  Pass med[px]  Pass p90[px]  gross20
──────────────────────────────────────────────────────────────────────────────────────────────────────
No filter                                          194      1.000         7.689        55.648    0.209
Confidence                                         150      0.773         6.615        25.347    0.141
Confidence + standard reprojection                 143      0.737         6.388        23.117    0.128
Confidence + keypoint-removal consistency          149      0.768         6.633        25.528    0.141
Full removal + horizontal-flip consistency         142      0.732         6.551        23.881    0.135
```

Standard reprojection consistency gives the best pass median, pass p90 and
pass gross20 of the five rules on this proxy. The full variant is not the
best label filter here, and the table reports that rather than hiding it.

Naming caution: the last row is the frozen `F4_PROPOSED` arm, defined as
confidence **and** keypoint-removal **and** horizontal-flip consistency —
reprojection is not part of it. A separate `F5_CONF_FLIP` arm exists in the
same artifact with different values; the two must not be confused.

## Table 3B — Downstream student

The same selection rules, measured by the student each one produced.

```text
Selection rule                            Pooled kp med[px]  gross20     Det    AUROC    FPR95
──────────────────────────────────────────────────────────────────────────────────────────────
Naive                                                 7.120    0.180   0.981   0.9913   0.0558
Confidence                                            7.037    0.194   0.987   0.9923   0.0469
+ standard reprojection                               7.044    0.194   0.987   0.9920   0.0487
+ keypoint-removal consistency                        6.999    0.194   0.987   0.9911   0.0502
+ horizontal-flip consistency (full)                  7.210    0.197   0.984   0.9953   0.0283
```

Synthetic-only reference: keypoint 6.616 px, gross20 0.172.

## Why the two panels are separate

```text
3A asks   does the rule pass better labels?      answer: reprojection does best
3B asks   does the student get better?           answer: no rule beats R0
```

The rule with the best pass quality in 3A is not the rule with the best student
in 3B, and no rule in 3B reaches the synthetic-only baseline on keypoint error.
Reporting only 3A would misrepresent the result; that is why both panels exist.

Post-hoc separability AUCs are not in this table. They are development
diagnostics and live in `TABLE_FINAL_DIAGNOSTIC.md`.
