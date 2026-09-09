# Paired bootstrap — 6D pose, each arm against R0

Every interval is **paired**: the two arms are compared on the same frame
and it is the per-frame difference that gets resampled.

Two resampling schemes are reported. `frame` treats frames as independent.
`cluster` resamples whole sessions, which respects the fact that frames from
one recording resemble each other — that is the interval to quote.

10000 resamples, seed 20260903, 95% interval. No model ran again.

**13 sessions is a small number of clusters.** An interval containing zero
means this data cannot separate the arms — not that they perform equally.

## IoU3D median  (higher is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
geometry_joint_seed1   +0.001      [-0.006, +0.018]      [-0.009, +0.021]   contains 0
```

## ADDsym AUC  (higher is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
geometry_joint_seed1   +0.007      [+0.003, +0.012]      [+0.000, +0.018]   excludes 0
```

## yaw median [deg]  (lower is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
geometry_joint_seed1   -0.009      [-0.080, +0.050]      [-0.081, +0.084]   contains 0
```

## translation median [cm]  (lower is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
geometry_joint_seed1   -0.094      [-0.507, +0.238]      [-0.675, +0.256]   contains 0
```

