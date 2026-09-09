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
geometry_joint_seed3   +0.001      [-0.007, +0.022]      [-0.009, +0.025]   contains 0
```

## ADDsym AUC  (higher is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
geometry_joint_seed3   +0.008      [+0.004, +0.013]      [+0.000, +0.021]   excludes 0
```

## yaw median [deg]  (lower is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
geometry_joint_seed3   -0.019      [-0.093, +0.049]      [-0.094, +0.090]   contains 0
```

## translation median [cm]  (lower is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
geometry_joint_seed3   -0.121      [-0.586, +0.206]      [-0.749, +0.232]   contains 0
```

