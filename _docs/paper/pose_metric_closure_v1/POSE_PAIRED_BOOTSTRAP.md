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
R5_PROPOSED            -0.016      [-0.043, +0.028]      [-0.044, +0.022]   contains 0
R1_NAIVE               -0.014      [-0.029, +0.025]      [-0.037, +0.025]   contains 0
R2_CONF                -0.004      [-0.032, +0.036]      [-0.041, +0.039]   contains 0
R3_CONF_REPROJ         -0.003      [-0.027, +0.034]      [-0.025, +0.030]   contains 0
R4_CONF_REMOVE         -0.004      [-0.031, +0.033]      [-0.038, +0.037]   contains 0
R0_CONT                -0.009      [-0.037, +0.021]      [-0.051, +0.021]   contains 0
```

## ADDsym AUC  (higher is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
R5_PROPOSED            -0.028      [-0.052, -0.005]      [-0.062, +0.004]   contains 0
R1_NAIVE               -0.008      [-0.030, +0.014]      [-0.027, +0.014]   contains 0
R2_CONF                -0.013      [-0.037, +0.012]      [-0.046, +0.024]   contains 0
R3_CONF_REPROJ         -0.014      [-0.034, +0.007]      [-0.027, +0.011]   contains 0
R4_CONF_REMOVE         -0.016      [-0.039, +0.006]      [-0.044, +0.014]   contains 0
R0_CONT                -0.020      [-0.040, -0.001]      [-0.048, +0.004]   contains 0
```

## yaw median [deg]  (lower is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
R5_PROPOSED            +0.063      [-0.139, +0.271]      [-0.202, +0.400]   contains 0
R1_NAIVE               +0.015      [-0.205, +0.224]      [-0.195, +0.213]   contains 0
R2_CONF                +0.096      [-0.143, +0.319]      [-0.202, +0.335]   contains 0
R3_CONF_REPROJ         -0.024      [-0.224, +0.207]      [-0.189, +0.374]   contains 0
R4_CONF_REMOVE         +0.046      [-0.223, +0.250]      [-0.247, +0.206]   contains 0
R0_CONT                +0.035      [-0.144, +0.232]      [-0.143, +0.318]   contains 0
```

## translation median [cm]  (lower is better)

```text
arm vs R0                diff          frame 95% CI        cluster 95% CI   cluster
───────────────────────────────────────────────────────────────────────────────────────
R5_PROPOSED            +0.930      [-0.456, +1.926]      [-0.492, +2.277]   contains 0
R1_NAIVE               -0.101      [-1.157, +0.728]      [-1.000, +0.926]   contains 0
R2_CONF                -0.112      [-1.339, +0.596]      [-1.286, +0.816]   contains 0
R3_CONF_REPROJ         -0.160      [-1.130, +0.804]      [-0.939, +0.834]   contains 0
R4_CONF_REMOVE         +0.158      [-0.871, +1.056]      [-0.957, +1.266]   contains 0
R0_CONT                +0.569      [-0.423, +1.306]      [-0.705, +1.537]   contains 0
```

