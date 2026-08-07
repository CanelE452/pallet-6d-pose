# SUPPORTING_LINE_HOUGH_DECODER_VALID

The mismatch was the whole failure.  Changing nothing but the target -- same
decoder, same lattice, same support rule, same coarse and fine search, same
sigma, same gates, same 27,684 roles -- takes the decoder from missing every
gate to clearing all four by an order of magnitude.

```
O_NUM, the identical 10,000 synthetic lines

                    S0 finite segment      S1 supporting line     gate
angle median              0.0114                 0.0063          <=0.02
angle p99                51.9619                 0.0170          <=0.08
offset median             0.0093                 0.0063          <=0.02
offset p99               19.3990                 0.0171          <=0.08
                                             ONUM_PASS = True
```

```
LINE_DEV, 2,393 frames, 27,684 supported roles, population 00c605b9116e214b

                    S0 finite segment      S1 supporting line     gate
angle median              1.4661                 0.0066          <=0.05
angle p90                14.5698                 0.0128          <=0.10
offset median             0.5309                 0.0061          <=0.05
offset p90                5.7738                 0.0128          <=0.10
                                                 PASS = True
```

Paired on the same roles: the supporting-line target is better on **98.8%** of
them in angle and 97.4% in offset, median delta 1.457 degrees.

## The stratum that settles it

```
stratum          S0 median   S0 p99      S1 median   S1 p99
short_chord         8.1565    80.3762       0.0063    0.0176
interior_long       0.0490     0.3138       0.0061    0.0187
border              0.0146     0.1903       0.0065    0.0187
theta_0             0.0063     0.0169       0.0063    0.0124
theta_90            0.0066     0.0175       0.0066    0.0124
theta_180           0.0062     0.0173       0.0062    0.0125

Q1  interior_long p99 <= 0.08   YES
Q2  short_chord   p99 <= 0.08   YES
->  FINITE_SEGMENT_EXTENT_MISMATCH_CONFIRMED
```

`short_chord` goes from 80.38 to 0.0176 at p99, a factor of 4,600.  I had written
that those cases were "unidentifiable, by any decoder".  That was wrong, and it
was wrong in a specific way worth naming: the claim was about the *line*, while
every measurement had been about a *finite-segment target*.  A 1-3 cell tube is
a blob with no orientation; the supporting line it lies on is a different object,
and the decoder recovers it to the lattice ceiling.

The same reversal runs through the real cross-tab.  The quadrant that looked
catastrophic under every previous decoder is now the same as the rest.

```
quadrant                     n      S0 med   S0 p90     S1 med   S1 p90
A border>=1.5 visible>=2  16,404    0.9809   5.3648     0.0064   0.0123
B border>=1.5 visible< 2   5,854    5.1390  29.6020     0.0064   0.0120
C border< 1.5 visible>=2   4,706    1.0410  14.1425     0.0072   0.0176
D border< 1.5 visible< 2     720   41.8257  70.9336     0.0080   0.0368

IN_FRAME_FULL             23,943    1.4444  12.1805     0.0064   0.0124
IN_FRAME_PARTIAL           3,741    1.6033  39.1858     0.0073   0.0215
```

Every earlier screen read those gradients as evidence about boundaries, chord
length or representation capacity.  They were the extent mismatch, seen four
different ways.

## Anisotropy: nothing left to see

With the mismatch gone, the theta bins are flat.  Angle medians run 0.0055 to
0.0072 across all eighteen 10-degree bins and p99 runs 0.0126 to 0.0219, against
a fine-lattice ceiling of 0.0125 degree.  Square-grid rasterisation leaves no
residual worth naming; the decoder is quantisation-limited everywhere.

## What this does and does not establish

```
SUPPORTING_LINE_TARGET_DECODER_VALID          yes
the decoder problem is solved                 not the claim
STRUCTURAL_LINE_MAP_CAPACITY                  STILL UNMEASURED
M0 / M1                                       ZERO TRAINING RUNS
```

This is a decoder oracle.  It reads a *perfect* supporting-line probability map.
Nothing here says a network can predict one.

The finite-segment structural map family closes as `TARGET_SEMANTICS_MISMATCH`,
which retroactively narrows what the earlier failures were about:

```
P0  TARGET_AS_WEIGHT_TLS_FAIL                  stands
P1  LOCKED_SOFTPLUS_TLS_FAIL                   stands
    both measured against a finite-segment target
O_NUM(S0) HOUGH_DISCRETIZATION_OR_IMPLEMENTATION_FAIL
    stands as a protocol label; the cause was the target, not discretization
```

The `20eb590` M0/M1 training protocol **may not be reused**: it supervises
`SLM.raster_targets`, the finite-segment tube.  A new training protocol has to be
locked against the supporting-line target -- new target construction, new map
loss, and a fresh decision about what "support" means when the target no longer
depends on segment extent.

`STRUCTURAL_LINE_MAP_CAPACITY` is unblocked for the new target only.

## Left open, on purpose

Roles whose physical segment misses the image while its supporting line crosses
it are still outside the population.  Under a supporting-line target that
question becomes live -- the target is now well defined for them -- but mixing it
in here would have moved the population and the target together.  It belongs in
its own screen.

No PnP, no CIGM, no dimensions, no intrinsics, no `validation512`.  `untouched`,
`eval56`, `wood45` and final-test remain unopened.
