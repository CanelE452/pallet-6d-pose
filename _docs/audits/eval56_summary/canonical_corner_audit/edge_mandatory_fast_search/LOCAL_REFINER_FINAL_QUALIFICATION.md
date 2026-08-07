# What 062363a closed, and what it did not

`062363a` stands unedited.  Its taxonomy label is kept; this names the family the
evidence actually covers.

```
SEARCH2K_LINE_REFINEMENT_FAIL               CONFIRMED
DATA_SCALE_UNRESOLVED                       CLOSED
LOCAL_EDGE_REPRESENTATION_PRECISION_FAIL    original taxonomy label

narrowed reading
GENERIC_LOCAL_STRIP_REPRESENTATION_FAIL     CONFIRMED

still untested
EXPLICIT_STRUCTURAL_LINE_REPRESENTATION
LINE_SUPERVISED_ENCODER
```

Everything measured so far shares one architecture: a generic frozen feature,
sampled into a strip around a coarse line, reduced to two scalars by a
coordinate head.  A failure of that pipeline is not a failure of every way of
representing a line, and O1C in particular is a statement about raw Scharr, not
about whether structural line information exists in the image.

## The optimizer wording, corrected

`062363a` should not be read as "the optimizer saturated".  What was measured is

```
6.8X_OPTIMIZATION_BUDGET_INSUFFICIENT
```

835 to 5,675 steps bought -0.385 to -0.559 degrees and the curve was still
moving.  Nothing in the run shows where it stops; only that 6.8x of it does not
reach 1.0 degree.

## Terminology

The twelve targets are **structural cuboid lines** -- supporting lines defined by
3D cuboid incidence.  They are not required to be strong photometric image edges
along their whole length, and calling them "image edges" quietly assumes the
thing the screens keep failing to find.  The task is **image-conditioned
structural line estimation**, not edge detection.

## What the next screen asks

One question: can an explicit spatial structural-line map reach 1.0 degree and
0.5 cell?  Twelve role-fixed maps at 100x100, decoded to a line by weighted
total least squares over the map itself -- no coordinate head, no Hungarian
matching, no pose, no dimensions, no intrinsics.

`STRUCTURAL_LINE_MAP_CAPACITY` is screened alone.  CIGM, PnP and the SLQ query
are attached only after `LINE_MAP_GO`.
