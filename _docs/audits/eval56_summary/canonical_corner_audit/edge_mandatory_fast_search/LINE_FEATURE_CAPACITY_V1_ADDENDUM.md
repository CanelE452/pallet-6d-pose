# V1 was a refiner failure, not a feature failure

`6ad0e39` and its files stand unedited.  The numbers are real; the conclusion
drawn from them was too broad.

```
observed          F50   3.834 deg / 1.561 cell
                  F100  3.947 / 1.919
                  MULTI 3.916 / 1.800
                  RGB   3.918 / 2.020
```

```
primary interpretation    LOCAL_LINE_REFINER_V1_FAIL

not established by V1     FROZEN_FEATURE_LINE_CAPACITY_FAIL
                          RGB_EDGE_CAPACITY_FAIL
```

## Four confounds, any one of which could produce this result

**The loss scale, which alone explains the headline observation.**  V1 used
`L_angle = (1 - cos).mean()` against `L_offset = SmoothL1`, combined as
`L_angle + 0.1 * L_offset`.  Near alignment `1 - cos` behaves like `theta^2 / 2`:
at 4 degrees it is 2.4e-3 while the offset term is order 1.  So despite the
nominal 10:1 weighting in favour of angle, offset dominated the gradient by
roughly two orders of magnitude.  V1 reported that offset improved and angle did
not.  That is what this loss does, and I read it as a statement about features.

**The sampler did not follow the line.**  Longitudinal samples ran
`t in [-grid/2, +grid/2]` from `centre = n * rho`, the foot of the perpendicular
from the origin.  That is a fixed window anchored to the origin, not the line's
visible extent, so for oblique or distant lines much of the strip fell outside
the image entirely.

**The strip was narrower than the jitter.**  `TRANSVERSE = 7` gives offsets
`-3..+3` cells while offset jitter was uniform +/-4.  Whenever the draw exceeded
3 cells the ground-truth line lay outside the sampled strip, so the evidence
needed to correct it was not in the refiner's input at all.

**The runner was not preserved.**  V1 trained from a scratchpad heredoc, so the
exact run is not reproducible from the repository.

## Consequence

V1 does not support a statement about what F50, F100, multi-scale or RGB can
carry.  It supports one about the refiner as built.  A V2 gate fixes the sampler
to the line-image intersection, widens the strip past the jitter, normalises both
losses by their own budget so an error of 1 means the gate boundary in each, and
runs from a committed script -- and adds a perfect-raster oracle first, so the
refiner's own identifiability is settled before any feature is judged.

The V1 files are kept as the record of the attempt.
