# The target and the decoder were describing different objects

`1b9685c` stands unedited.  Its numbers are real; what they compared was not one
geometric object against itself.

```
historical target    FINITE_VISIBLE_SEGMENT_TUBE
historical decoder   INFINITE_SUPPORTING_LINE_HOUGH
->                   FINITE_SEGMENT_TO_INFINITE_LINE_MISMATCH = PRESENT
```

Every structural-line screen has rasterised a tube around the *clipped visible
segment* and then scored it against a template for the *infinite supporting
line*.  The template extends past the segment, so mass outside the segment
enters `sum K` and `sum K^2` but never `sum pK`, and the optimum tilts.

The O_NUM strata already pointed at it and I read them as two competing
mechanisms.  `border`, whose segment spans the whole chord and therefore matches
the template's extent, scored three times better than `interior_long`, whose
segment is shorter than its chord, at the same lengths.  That is the
extent-match prediction exactly.

## Labels

```
historical, kept          HOUGH_DISCRETIZATION_OR_IMPLEMENTATION_FAIL
narrowed                  SYNTHETIC_BOUNDARY_BIAS_REDUCED
                          FINITE_SEGMENT_TARGET_HOUGH_FAIL
```

## One claim withdrawn

`1b9685c` wrote that short chords are "unidentifiable, by any decoder".  That was
stated about the line when the evidence only covers the finite-segment *target*:
a 1-3 cell tube is a blob, but the supporting line it lies on is a different
object and was never given to the decoder.  Replaced by

```
SHORT_SEGMENT_TARGET_ORIENTATION_POORLY_CONDITIONED
```

and the question is now tested rather than asserted -- `short_chord` stays in the
synthetic set, and under the S1 target its segment length simply stops entering
the map.

## What this screen changes, and what it does not

```
changed     the target only
            S0_FINITE_SEGMENT    SLM.raster_targets, unmodified
            S1_SUPPORTING_LINE   exp(-d^2 / 2 sigma^2), d to the infinite line,
                                 rasterised over the whole MAP100 grid
unchanged   decoder H2_ZERO_MEAN_NCC, lattice, support rule, coarse and fine
            search, sigma 1.5 MAP100 pixel, both gates, and the population
```

The population is deliberately *not* touched: the same supported roles, the same
27,684 on LINE_DEV, `short_chord` kept, nothing added for off-frame.  Changing
the population in the same screen that changes the target would move two factors
at once, and the off-frame question -- roles whose physical segment misses the
image while its supporting line crosses it -- is a global-extrapolation question
that belongs in its own screen.

No segment-aware template, no extent predictor, no half-length.  Those would
reintroduce the extent nuisance that the final CIGM does not need; the point of
S1 is to remove it.

## Pre-declared questions

```
Q1   interior_long   S1 angle p99 <= 0.08 degree ?
Q2   short_chord     S1 angle p99 <= 0.08 degree ?
both YES  ->  FINITE_SEGMENT_EXTENT_MISMATCH_CONFIRMED
```

If both hold, the short-chord collapse was a property of the finite-segment
representation rather than of line identifiability.

A theta anisotropy diagnostic follows on S1 only, in 10-degree bins, to see
whether square-grid rasterisation leaves a residual once the extent mismatch is
gone.  It is reported, never used to select.

## Gates, unchanged

```
O_NUM (S1 primary)   angle median <= 0.02   p99 <= 0.08
                     offset median <= 0.02  p99 <= 0.08   canonical50 cell
LINE_DEV (S1)        angle median <= 0.05   p90 <= 0.10
                     offset median <= 0.05  p90 <= 0.10
```

`SUPPORTING_LINE_TARGET_NUMERICAL_VALID` gates LINE_DEV.  A LINE_DEV pass reads
as `SUPPORTING_LINE_TARGET_DECODER_VALID` -- not "the decoder problem is solved"
-- and closes the finite-segment structural map family as
`TARGET_SEMANTICS_MISMATCH`.

In that case the `20eb590` M0/M1 training protocol may **not** be reused: it
supervises the finite-segment target, so a new training protocol has to be
locked against the supporting-line target first.  M0/M1 stay at zero training
runs either way, and `STRUCTURAL_LINE_MAP_CAPACITY` stays unmeasured.

If S1 fails, no further decoder repair happens in the same session.  The recorded
next candidate, not built: **DIRECT_HOUGH_SPACE_LINE_MAP** -- predict twelve
role-specific `(theta, rho)` heatmaps directly, without an image-space map, which
is still a spatial prediction and not a coordinate regression.

No PnP, no CIGM, no dimensions, no `validation512`.  `untouched`, `eval56`,
`wood45` and final-test remain unopened.
