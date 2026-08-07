# HOUGH_DISCRETIZATION_OR_IMPLEMENTATION_FAIL

O_NUM runs before LINE_DEV, and it fails, so LINE_DEV was not opened and no
`NORMALIZED_HOUGH_DECODER_VALID/FAIL` verdict exists.

```
10,000 synthetic lines, H2_ZERO_MEAN_NCC (primary)

                measured    gate
angle median      0.0114   <=0.02   pass
angle p99        51.9619   <=0.08   FAIL
offset median     0.0093   <=0.02   pass
offset p99       19.3990   <=0.08   FAIL
```

## Per stratum, which is why the strata were separated

```
stratum          n      angle median   angle p99
theta_0        1,667       0.0063        0.0169
theta_180      1,666       0.0062        0.0173
theta_90       1,666       0.0066        0.0175
border         1,667       0.0146        0.1903
interior_long  1,667       0.0490        0.3138
short_chord    1,667       8.1565       80.3762
```

The aggregate p99 is one stratum.  Three separate readings follow, and the label
`HOUGH_DISCRETIZATION_OR_IMPLEMENTATION_FAIL` is accurate for only one of them.

**Boundary clipping is not this decoder's problem.**  `border` -- a long chord
whose endpoints lie on the rectangle, so its Gaussian tube is cut -- sits at
median 0.0146 and p99 0.1903.  The same condition destroyed the weighted-moment
decoder, where the `border < 1.5` group ran at angle p90 5.84 and IN_FRAME_PARTIAL
at 8.27.  Removing the spatial mean removed that failure mode, which is the one
thing this screen was designed to test and the one thing it establishes.

**Axis-aligned lines sit at the lattice ceiling.**  0.0063 to 0.0066 median and
0.017 p99 against a fine ceiling of 0.0125 degree.  Discretization is not the
binding constraint, and a pre-run check already showed the coarse peak inside
the locked fine window for every long stratum (max 0.46 against a 0.75 degree
half-width).

**Oblique long lines carry a real bias.**  `interior_long` is 0.0490 median and
0.3138 p99 -- eight times the axis-aligned strata, and above the 0.08 numerical
gate on its own.  Two mechanisms are consistent with it and this run does not
separate them: the template is an infinite line while the target is a finite
segment, so mass outside the segment enters `sum K` and `sum K^2` but not
`sum pK`; and a Gaussian tube rasterised on a square grid is not isotropic in
theta.  The first is supported by `border` (segment spans the full chord, so the
template matches its extent) scoring three times better than `interior_long`
(segment shorter than the chord) at the same lengths.

**Short chords are unidentifiable, by any decoder.**  A 1-3 canonical50 cell
segment is a blob roughly 2-6 MAP100 pixels across with a 1.5 pixel tube; it has
no orientation to recover.  Median 8.16 degrees, and the coarse peak itself is
7-64 degrees off for 94% of them, so no refinement window reaches the truth.
The weighted-moment decoder failed the same cases.  This is a property of the
population, not of the readout.

## What the label should be read as

```
HOUGH_DISCRETIZATION_OR_IMPLEMENTATION_FAIL      as the protocol defines it
```

but the evidence says discretization is *not* what failed.  The gate is an
aggregate p99 over a set that deliberately contains an unidentifiable stratum,
so it was always going to be decided by that stratum.  Written plainly:

```
boundary clipping        SOLVED by removing the spatial mean
lattice / fine window    not binding (0.017 p99 on axis-aligned)
finite-segment template  OPEN, 0.31 p99 on oblique long lines, above 0.08
short chords             unidentifiable, decoder-independent
```

## Consequence

LINE_DEV is not opened.  `NORMALIZED_HOUGH_DECODER_VALID` and
`NORMALIZED_HOUGH_DECODER_FAIL` are both unreached; the background-floor
diagnostic did not run.  M0/M1 remain untrained with zero runs, and
`STRUCTURAL_LINE_MAP_CAPACITY` is still unmeasured.

Nothing is repaired here.  Two things a next locked screen would have to settle
before this decoder can be judged, recorded and not built:

1. A segment-aware template -- correlate against a finite segment with an
   estimated extent rather than an infinite line -- which is what the
   `border` versus `interior_long` gap points at.
2. Whether the short-chord stratum belongs in a decoder gate at all, or whether
   the population for a *decoder* oracle should be the roles a decoder could in
   principle resolve, with the unidentifiable ones counted and excluded the way
   OFF_FRAME_FULL already is.

Deciding the second after seeing this result would be redefining a population to
pass a gate, which is exactly the move the earlier screens were built to prevent.
It belongs in a locked protocol, not here.

No PnP, no CIGM, no dimensions, no intrinsics, no `validation512`.  `untouched`,
`eval56`, `wood45` and final-test remain unopened.
