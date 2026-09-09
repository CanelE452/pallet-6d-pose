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
image_line_only_seed1   +0.034      [+0.017, +0.057]      [+0.009, +0.056]   excludes 0
```

## ADDsym AUC  (higher is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
image_line_only_seed1   +0.022      [+0.012, +0.031]      [+0.012, +0.032]   excludes 0
```

## yaw median [deg]  (lower is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
image_line_only_seed1   -0.116      [-0.210, +0.024]      [-0.254, +0.035]   contains 0
```

## translation median [cm]  (lower is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
image_line_only_seed1   -0.288      [-1.083, +0.061]      [-1.412, -0.045]   excludes 0
```

