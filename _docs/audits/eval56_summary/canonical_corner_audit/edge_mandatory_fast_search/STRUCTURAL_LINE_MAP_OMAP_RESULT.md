# MAP_TO_LINE_DECODER_FAIL

The screen stopped where it was designed to stop.  The decoder oracle runs
before any arm trains, and it fails, so no capacity number about the
representation was produced and none should be quoted.

```
O_MAP   ground-truth target maps -> weighted TLS -> line
        LINE_DEV, 2,393 frames, 27,684 supported roles

                 measured    gate
angle median      0.0015    <=0.05    pass
offset median     0.0006    <=0.05    pass
angle p90         0.3780    <=0.10    FAIL
offset p90        0.1380    <=0.10    FAIL

OMAP_PASS = False  ->  MAP_TO_LINE_DECODER_FAIL  ->  training blocked
```

## The readout is not broken; its footprint is

On a clean interior segment the readout is exact to `4e-5` degrees, and the
median over the whole split is 0.0015 degrees.  The failure is entirely in a
tail, and the tail has one cause.

```
                    n        angle med   angle p90   offset p90
border >= 1.5    22,258        0.0007      0.0132      0.0053
border <  1.5     5,426        0.3920      5.8421      2.6406

IN_FRAME_FULL    23,943        0.0009      0.0249      0.0098
IN_FRAME_PARTIAL  3,741        0.5654      8.2677      3.8127

visible < 2 cell  6,574        0.0084      0.3692      0.0998
```

Away from the image border the oracle clears the gate by an order of magnitude.
Within 1.5 cells of it, the Gaussian tube is clipped asymmetrically by the grid,
which drags the weighted centroid off the line and tilts the covariance.  Since
`IN_FRAME_PARTIAL` is by definition the set of roles whose segment runs into the
border, that population is the same failure seen from the other side.

Short visible stubs are *not* the main cause: roles under 2 cells long have p90
0.369 degrees, well below the border group's 5.84.

## Why this is a real finding rather than a bug to patch

Weighted moments over a bounded grid are not translation-equivariant near the
boundary.  Any readout of this shape -- centroid plus covariance of a mass that
the frame can cut -- inherits the bias.  That is a property of the proposed
decoder, which is exactly what the oracle stage exists to expose, and it would
have been read as a representation failure had the arms trained first: a model
predicting a *perfect* map still could not have produced an in-budget line for
19.6% of the population.

Three admissible repairs, none applied here, because changing the decoder after
seeing its result is the move this project keeps paying for:

1. Normalise the moments by the visible support -- accumulate over the same
   window for prediction and for a reference mask so the truncation cancels.
2. Restrict the moment window to a band around a current line estimate, making
   the footprint symmetric by construction and the readout iterative.
3. Pad the map beyond the image rectangle so a tube near the border is not cut,
   at the cost of predicting mass where there is no evidence.

Choosing among them is a design decision and belongs in the next locked screen,
not in this one.

## Standing

```
STRUCTURAL_LINE_MAP_CAPACITY     NOT MEASURED
M0_F50_MAP / M1_F50_RGB_MAP      NOT TRAINED
LINE_MAP_GO                      NOT REACHED
SLQ / CIGM / PnP                 NOT BUILT
```

`GENERIC_LOCAL_STRIP_REPRESENTATION_FAIL` from `062363a` is unchanged, and
`EXPLICIT_STRUCTURAL_LINE_REPRESENTATION` remains untested.  No PnP, no
dimensions, no intrinsics, no `validation512`.  `untouched`, `eval56`, `wood45`
and final-test remain unopened.
