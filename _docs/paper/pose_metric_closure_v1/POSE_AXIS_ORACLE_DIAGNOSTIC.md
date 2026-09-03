# MAIN versus ORACLE-AXIS diagnostic

**Post-hoc diagnostic. The oracle column is not a deployable result** — it
is produced by handing the model the ground-truth physical axis, which a
deployed system does not have. It appears in no main table.

Its purpose is to split one question in two:

```text
main poor / oracle good    axis selection is the bottleneck
main poor / oracle poor    keypoint geometry is also a bottleneck
```

```text
Method                              MAIN Axis  MAIN IoU  ORA IoU    dIoU  MAIN AUC  ORA AUC    dAUC
───────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                     0.749     0.603    0.609  +0.006     0.428    0.456  +0.027
Source-only continuation                0.726     0.594    0.614  +0.020     0.409    0.441  +0.032
Naive self-training                     0.768     0.590    0.602  +0.012     0.420    0.429  +0.009
Confidence self-training                0.724     0.599    0.618  +0.018     0.416    0.446  +0.030
Reprojection self-training              0.749     0.600    0.619  +0.020     0.415    0.435  +0.020
Removal self-training                   0.727     0.600    0.613  +0.013     0.412    0.441  +0.029
Full consistency self-training          0.734     0.587    0.623  +0.036     0.400    0.433  +0.033
```

## Same-population selector comparison

An earlier note compared a simple minimum-reprojection selector at 75.2%
against the existing selector at 65.0%. Those figures came from different
populations and that comparison is withdrawn.

```text
Method                              existing frozen  simple reprojection
────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                           0.749                0.752
Source-only continuation                      0.726                0.745
Naive self-training                           0.768                0.768
Confidence self-training                      0.724                0.727
Reprojection self-training                    0.749                0.749
Removal self-training                         0.727                0.727
Full consistency self-training                0.734                0.743
```

On the common 319-frame population the two selectors show nearly identical
axis accuracy. The sentence "the simple residual selector is ten percentage
points better" is not supported and is not used.

`POST_HOC_DIAGNOSTIC_ONLY`

generated 2026-09-03T06:18:47.469046+00:00
