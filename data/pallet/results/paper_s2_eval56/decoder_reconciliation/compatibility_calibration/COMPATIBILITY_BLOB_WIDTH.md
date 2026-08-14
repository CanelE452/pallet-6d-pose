# How wide is ep57's response

Per channel, on H6, measured around each channel's own peak.  `sigma from
half-max` converts the half-maximum area to the sigma an isotropic Gaussian
would need: a Gaussian's half-maximum diameter is 2*sqrt(2 ln 2)*sigma.

```
            model      role    n  raw peak  half-max area  eq radius  sigma (half-max)  eff sigma  11x11 mass frac
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
          M0_ep57      near  348     0.798             19      2.459             2.089      1.918            0.936
          M0_ep57       far  348     0.804             21      2.585             2.196      1.979            0.960
          M0_ep57  centroid   87     0.886             19      2.459             2.089      1.893            0.955
 M1_challenge0123      near   48     0.900             72      4.787             4.066      2.791            0.691
 M1_challenge0123       far   48     0.848             74      4.837             4.108      2.798            0.686
 M1_challenge0123  centroid   12     0.954             71      4.754             4.038      2.797            0.715
M2_challengenight      near   48     0.919             70      4.720             4.009      2.777            0.715
M2_challengenight       far   48     0.926             70      4.720             4.009      2.789            0.689
M2_challengenight  centroid   12     0.821             72      4.804             4.080      2.784            0.716
```

ep57's blobs are **about half the width** of the two challenge checkpoints:
sigma 2.09-2.20 against 4.01-4.11, half-maximum area 19-21 cells against 70-73.
That is the training target showing through -- ep57 was trained at belief
sigma 2, the challenge models at 4 -- and it is the whole of the incompatibility.

The 11x11 window tells the same story from the other side.  On ep57, 94-96% of
the positive mass already sits inside 11x11; on the challenge models only
69-72% does.  The deployment decoder's 11x11 weighted average was sized for the
wider blob.

## Centroid against corner, within each model

```
            model  centroid sigma  corner sigma  ratio
──────────────────────────────────────────────────────
          M0_ep57           2.089         2.143  0.975
 M1_challenge0123           4.038         4.094  0.986
M2_challengenight           4.080         4.009  1.018
```

**No model separates the two roles.**  The ratio is 0.98-1.02 everywhere: the
centroid channel is trained to exactly the same width as a corner channel, in
ep57 and in both challenge checkpoints.  Phase H shows the deployment decoder
does not ask for the same width from both.

## Peak retention against smoothing sigma (centroid, median)

```
            model    0.0    0.5    1.0    1.5    2.0    2.5    3.0
──────────────────────────────────────────────────────────────────
          M0_ep57  1.000  0.954  0.813  0.658  0.517  0.404  0.319
 M1_challenge0123  1.000  0.986  0.938  0.873  0.798  0.720  0.641
M2_challengenight  1.000  0.986  0.940  0.877  0.802  0.723  0.644
```

At the deployment sigma of 3, ep57 keeps 0.319 of its centroid peak and the
challenge models keep 0.641 and 0.644.  With a raw median of 0.886 that leaves
ep57 at 0.282 against a 0.30 gate -- clearing it on 11 of 87 frames.
