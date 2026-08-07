# What an edge predictor must reach for direct CIGM -> PnP

Sensitivity diagnostic, not selection.  Controlled noise is injected into
ground-truth edge lines; no model predicts anything here.  Run on the
deterministic first 128 frames of the frozen validation512, which was already
opened for post-validation diagnostics.  untouched, eval56, wood45 and
final-test stay closed.

Lines are perturbed in the form CIGM actually consumes -- unit normal and rho,
sign canonicalised -- not as a centre and a half-length.

Usable, fixed before the sweep ran:

```
solve rate      >= 95%
GT-corner reprojection median  <= 5 px
GT-corner reprojection p90     <= 20 px
catastrophic (> 50 px)         <= 5%
```

Error is measured against `projected_cuboid`, never against the solver's own
input.

## Angle only

```
  0.0 deg   solve 0.969   median  0.033 px   p90   0.064   usable
  0.5       0.987          2.147             4.449          usable
  1.0       0.977          4.278             8.223          usable
  2.0       0.974          8.629            16.032          --
  4.0       0.971         17.000            30.965          --
  8.0       0.979         33.096            58.567          --
 12.0       0.977         50.213            90.146          --
 16.0       0.990         67.988           124.037          --
```

## Offset only

```
  0.000 cell  solve 0.969   median  0.033 px   p90   0.064   usable
  0.125       0.982          1.181             2.212          usable
  0.250       0.982          2.370             4.422          usable
  0.500       0.979          4.846             9.024          usable
  1.000       0.977          9.877            17.144          --
  2.000       0.971         19.673            33.262          --
  4.000       0.966         38.238            61.384          --
  6.000       0.987         57.318            92.746          --
  8.000       0.992         75.241           120.794          --
```

```
ANGLE_BUDGET    1.0 degree
OFFSET_BUDGET   0.5 cell        (0.5 cell = 4 image px at the 400/50 factor)
```

## The gap this exposes

```
                       required        Round-1 PEQ delivered      factor
orientation median     <= 1.0 deg              16.0 deg            16x
perpendicular offset   <= 0.5 cell              6.6 cell           13x
```

The response is close to linear in both axes -- roughly 4.3 px of corner error
per degree and 9.9 px per cell -- and the solve rate barely moves, staying near
97% even at 16 degrees.  So the solver does not fail loudly under bad edges; it
returns a confident and wrong pose.  A predictor cannot be screened by solve rate.

This is the target for the next architecture, and it is a demanding one: an order
of magnitude better edge localisation than PEQ reached, on a path that has no
belief map to absorb the error.  It does not change any Round-1 gate.

## Reading it fairly

The budget is derived from ground-truth geometry perturbed synthetically.  Real
predictor error will not be zero-mean Gaussian and independent across the twelve
roles, which is why a correlated-bias condition is measured separately in
`edge_noise_budget.csv`.  Treat 1.0 degree and 0.5 cell as an optimistic bound
rather than a sufficient condition.
