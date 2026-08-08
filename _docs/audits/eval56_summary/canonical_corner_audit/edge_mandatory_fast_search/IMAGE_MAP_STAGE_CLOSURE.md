# The image-space map family is closed

`0f41954` stands unedited.  This records what the family measured and corrects
one sentence in it.

```
same output representation, three conditioning shapes
  GAP-FiLM global content      -3.4% angle    weak
  absolute XY                 -20.1% angle    useful, insufficient
  role-query nonlocal global   -6.5% angle    weak incremental, on top of XY

best              4.1793 degree / 1.8788 canonical50 cell
against budget    4.2x median, 18x the p90 safety line
verdict           ROLE_CONDITIONED_GLOBAL_MAP_FAIL
```

## One sentence corrected

I wrote: *"Hough margin and peak entropy are unchanged, so the decoder is no
more certain about Q1's maps; only the ridge position moved."*

Margin and peak entropy are two proxies for one thing.  Their being flat does
not establish that the decoder's certainty is unchanged, and "only the ridge
moved" is a mechanism claim the measurement does not carry.

```
recorded instead
  decoded geometry improved without improvement in the measured
  Hough confidence proxies
```

## What the next screen is, and is not

Replacing the image raster with a line-native Hough output changes both the
representation and the readout family at once.

```
not claimed   a clean causal effect of representation alone
recorded as   REPRESENTATION_AND_READOUT_FAMILY_SWITCH
```

The encoder is held fixed -- frozen A1 F50, normalised XY, the same twelve role
queries and the same cross-attention block as Q1 -- so what changes is
everything downstream of the role descriptor.  That is a deliberate family
switch, not a factorial arm, and it will be reported as one.
