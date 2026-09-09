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
image_joint_seed2      +0.030      [+0.012, +0.054]      [+0.004, +0.053]   excludes 0
```

## ADDsym AUC  (higher is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
image_joint_seed2      +0.023      [+0.014, +0.033]      [+0.013, +0.036]   excludes 0
```

## yaw median [deg]  (lower is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
image_joint_seed2      -0.091      [-0.250, -0.011]      [-0.339, -0.002]   excludes 0
```

## translation median [cm]  (lower is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
image_joint_seed2      -0.422      [-1.183, -0.028]      [-1.410, -0.024]   excludes 0
```

