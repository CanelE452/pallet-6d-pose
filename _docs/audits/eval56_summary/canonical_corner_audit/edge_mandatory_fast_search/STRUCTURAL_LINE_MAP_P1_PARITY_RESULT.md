# LOCKED_SOFTPLUS_TLS_FAIL

The oracle now simulates the decoder the screen actually locked, and it still
fails -- by more than the mismatched one did, and not for the reason the first
run gave.

```
                 P0 target-as-weight   P1 forward parity   gate
angle median            0.0015               0.0485       <=0.05   pass
angle p90               0.3780               0.4378       <=0.10   FAIL
offset median           0.0006               0.0172       <=0.05   pass
offset p90              0.1380               0.1889       <=0.10   FAIL
eigen ratio median         8.5                 12.8

LINE_DEV, 2,393 frames, 27,684 supported roles
OMAP_PASS = False  ->  LOCKED_SOFTPLUS_TLS_FAIL
                   ->  MAP_TO_LINE_DECODER_FAIL_CONFIRMED
```

Units: angle in degrees, offset in canonical50 cells, sigma 1.5 MAP100 pixel =
0.75 canonical50 cell, border threshold 1.5 canonical50 cell = 3 MAP100 pixel =
2 sigma, visible threshold 2.0 canonical50 cell = 4 MAP100 pixel.

## The parity weight makes it worse, not better

`-log1p(-target)` maps the ridge value 1.0 to 15.94 and 0.1 to 0.105, so it
concentrates mass on the line and thins the skirt the frame can cut.  That was
the reason to think the verdict might move.  It moved the wrong way: the angle
median rose thirty-fold, from 0.0015 to 0.0485, landing on the gate boundary.

Sharpening the weight shrinks the effective footprint of the moment estimate.
A tube two sigma wide contributes almost all of its mass from a two-or-three
pixel core once expanded, so the covariance is estimated from fewer effective
samples and the direction becomes more sensitive to the discrete grid.  The
eigen ratio rising from 8.5 to 12.8 is that same concentration seen from the
other side.

## The border was not the cause

`548585b` said the tail was border truncation.  Under the correct oracle that
does not hold.

```
quadrant                              n      angle med  angle p90  offset p90  eig p10
A  border >= 1.5, visible >= 2   16,404       0.0199     0.1513      0.0624      4.84
B  border >= 1.5, visible <  2    5,854       0.1546     0.5047      0.2236      1.28
C  border <  1.5, visible >= 2    4,706       0.1584     1.3240      0.5857      6.49
D  border <  1.5, visible <  2      720       4.9088    65.6833     22.5728      1.22
```

Quadrant A is 59% of the population, away from the boundary and long enough to
have a direction, and it *still* misses the p90 gate at 0.1513.  Border and
short stub each cost about the same on their own (B 0.5047, C 1.3240) and
compound catastrophically together (D 65.68), but neither is what keeps the
clean majority out.

So the claim in `548585b` is withdrawn.  What P0 showed was that the *target
read as a weight* is border-sensitive; what P1 shows is that the locked decoder
misses the budget across the whole population, boundary or not.

## Verdict and consequence

```
O_MAP_P0_TARGET_AS_WEIGHT     TARGET_AS_WEIGHT_TLS_FAIL      historical, kept
O_MAP_P1_FORWARD_PARITY       LOCKED_SOFTPLUS_TLS_FAIL       current
STRUCTURAL_LINE_MAP_CAPACITY  NOT MEASURED
M0_F50_MAP / M1_F50_RGB_MAP   BLOCKED, 0 training runs
LINE_MAP_GO                   NOT REACHED
SLQ / CIGM / PnP              NOT BUILT
```

No decoder repair is implemented here.  Recorded as the preferred next family,
not built:

**normalised Radon / Hough line readout** -- integrate ridge evidence along
candidate lines instead of taking the centroid and covariance of a mass.  Its
failure mechanism is different in kind: it never forms a spatial mean, so a
truncated or sharply peaked mass does not shift an estimator that has no mean to
shift, and its angular resolution is set by the accumulator rather than by how
many effective samples survive in the covariance.

That belongs in its own locked screen with its own oracle stage, which is the
part of this protocol that has now caught two decoder faults before any arm
trained.

No PnP, no CIGM, no dimensions, no intrinsics, no `validation512`.  `untouched`,
`eval56`, `wood45` and final-test remain unopened.
