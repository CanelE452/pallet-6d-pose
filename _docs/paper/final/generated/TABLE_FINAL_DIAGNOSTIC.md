# Appendix table — diagnostic interventions

**Every row is development evidence (Tier B).** Each was designed after
PAPER_EVAL diagnostics had been seen, so none is an independent confirmation
and none may be described as a held-out result.

The purpose of the table is not to count failures. It is to record which
candidate mechanisms were isolated and ruled out.

## Student arms

`base` and `arm` are paired NME medians on that arm's own common-frame subset,
so the base column differs slightly per row and the columns are not a single
shared baseline.

```text
arm                             n_fr  base NME  arm NME     delta                      CI95
───────────────────────────────────────────────────────────────────────────────────────────
V2A_CONF25                       309   0.02075  0.02188  +0.00091      [+0.00023, +0.00160]
V2B_KP_MASK                      308   0.02072  0.02174  +0.00030      [-0.00017, +0.00089]
V2C_AMBIG                        310   0.02080  0.02164  +0.00093      [+0.00052, +0.00148]
V2D_FULL                         309   0.02087  0.02121  +0.00030      [-0.00024, +0.00090]
V3A_TRUE_IGNORE                  307   0.02082  0.02176  +0.00072      [+0.00010, +0.00147]
V3B_TRUE_IGNORE_AMBIG            310   0.02080  0.02194  +0.00067      [-0.00008, +0.00146]
V5_RELIABILITY_WEIGHTED          308   0.02072  0.02201  +0.00086      [+0.00018, +0.00140]
```

No arm's interval lies entirely below zero. Improvement in the student's
keypoint localisation was never observed.

## Geometry repair — why no student was trained

```text
repair status               count
─────────────────────────────────
AMBIGUOUS_VIEW                 15
HYPOTHESIS_DISAGREE             5
OUT_OF_IMAGE                    2
REPAIRED                        2
NO_VALID_HYPOTHESIS             1
```

Repair candidates were about one percent of supervised corners, and the
competing geometric hypotheses disagree precisely on the corners that need
repairing. The intervention had no population to act on, so no student was
trained — the track stopped at the mechanism stage rather than producing a
null training result.

## Reliability weighting — label quality improved, student did not

```text
metric                     uniform    weighted      change
──────────────────────────────────────────────────────────
frame_gross                 0.5190      0.4615     -0.0574
corner_gross                0.2078      0.1823     -0.0255
median_error_px            20.0479     18.2491     -1.7988
p90_error_px               31.4206     28.7036     -2.7170
```

The reliability score ranks frames with AUC 0.7625, above every
individual signal it is built from, and the labels the student sees do get
cleaner. The student's localisation did not move. This is the most direct
evidence that pseudo-label purity is not the binding constraint.

## Multi-view teacher consensus — median better, tail worse

Paired comparison on the identical keypoint set: a candidate that discards
hard keypoints would otherwise flatter itself. Coverage is listed
separately for exactly that reason.

```text
probe       coverage   n_kp   R0 NME  cand NME   R0 p90  cand p90  R0 gross  cand gross
───────────────────────────────────────────────────────────────────────────────────────
FAST-A           230   1807  0.01928   0.01888  0.07840   0.09933    0.1422      0.1599
FAST-B           221   1671  0.01908   0.01899  0.08197   0.09964    0.1376      0.1460
FAST-C           235   1801  0.01896   0.01831  0.07946   0.10538    0.1438      0.1588
```

All three probes move the same way: the median improves slightly and the
tail gets clearly worse. Averaging or taking a median across views pulls
good predictions toward bad ones exactly where the views disagree, which is
where the hard keypoints are. Pooling more observations from the same
teacher does not raise the teacher's ceiling. No student was trained on any
of them.
