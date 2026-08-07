# How far the extent-mismatch finding reaches

`f5ac650` stands unedited.  `FINITE_SEGMENT_EXTENT_MISMATCH_CONFIRMED` is real
and its evidence is strong, but I wrote that "every earlier screen read those
gradients as evidence about boundaries, chord length or representation capacity;
they were the extent mismatch seen four ways."  That sentence is too broad and is
withdrawn.

The finding is about a *map target read by a line decoder*.  It applies to
exactly the results that pair one with the other:

```
applies to
  P0    target-as-weight TLS
  P1    softplus-TLS (the locked forward)
  O_NUM S0 finite-segment Hough
  the finite-segment structural-map decoder oracles generally

does NOT apply to
  V2 local strip refiner        no map, no line template; a strip around a
                                coarse line reduced by a coordinate head
  PEQ                           query-per-role regression
  the dense 12-edge predictor   field activation, judged by top-k selection
  HCRM                          spatial residual on the belief map
```

None of those four ever rasterised a segment tube and correlated it against an
infinite-line template, so the mismatch cannot be what they measured.

```
GENERIC_LOCAL_STRIP_REPRESENTATION_FAIL     remains CONFIRMED
6.8X_OPTIMIZATION_BUDGET_INSUFFICIENT       remains as recorded
```

The four earlier gradients that *did* dissolve -- border, chord length,
IN_FRAME_PARTIAL, quadrant D -- all came from the same decoder-oracle family, and
that is what the sentence should have said.
