# Known-dimension pallet pose: the problem this work solves

Addendum.  `bf9d8ab` and the note appended to
`DIRECT_PNP_DIMENSION_POLICY.md` both stand; neither is edited or reversed.  The
block was right for the setting it assumed, and this names the setting the work
actually targets.

## Formulation

```
inputs    monocular RGB
          camera intrinsics K
          known target pallet dimensions (W, D, H)

output    metric rotation R
          metric translation t
```

"Monocular" describes the sensing modality.  It does not exclude known object
geometry.  This is **known-model / known-dimension pallet pose**, not
category-level unknown-size monocular metric pose, and the two should not be
compared as if they were the same task.

## Why dimensions cannot simply be dropped

Under a perspective camera, scaling the object and the translation together
leaves the image unchanged:

```
X' = sX,  t' = st
    =>  K (R X' + t') = K (s (R X + t)) = s · K (R X + t)
```

Homogeneous coordinates absorb the factor `s`, so the projection is identical for
every `s > 0`.  Absolute metric translation is therefore **not identifiable**
from a single RGB image when the object's metric size is unknown.

Three consequences follow, and all three are choices rather than omissions:

- No dimension regressor is added to SLQ.  Predicting size would be inventing the
  unidentifiable quantity and hiding it inside the model.
- No unknown-size metric pose claim is made anywhere.
- The task is not switched to scale-free pose, which would be a different
  problem, not this one solved differently.

## How paper20k is read

Its randomised dimensions are **known parametric instance geometry**, not an
unknown target label.  Each synthetic frame is a different known cuboid instance,
which is what forces a predictor to generalise across shape rather than memorise
one ratio.

Where dimensions may and may not appear:

```
allowed     predicted lines -> CIGM corners -> PnP
forbidden   line predictor input
forbidden   feature extraction
forbidden   training target construction
```

Any of the forbidden three would leak size into the part of the system that is
supposed to read geometry out of pixels.

## Deployment reading

`(W, D, H)` arrives as system metadata -- a pallet specification or a prior
measurement -- and is never estimated from the image.

`eval56` and `wood45` are the user's own pallets with measured dimensions, so
they satisfy this setting.  They stay sealed; using their locked metadata is a
question for a later final evaluation, not now.

## Earlier results, qualified

```
DIRECT_EDGE_TO_PNP_INTERFACE_VALID     condition = KNOWN_DIMENSIONS
ANGLE_BUDGET   1.0 degree              condition = KNOWN_DIMENSIONS
OFFSET_BUDGET  0.5 cell                condition = KNOWN_DIMENSIONS
```

Not re-run.  They were already recorded with the dimension precondition named, so
they carry over unchanged.

```
HARD_BLOCKED_DIMENSION_ORACLE
    -> HARD_BLOCKED_FOR_UNKNOWN_DIMENSION_SETTING
DIRECT_PNP_ALLOWED_FOR_DECLARED_KNOWN_DIMENSION_SETTING
```

## What is still oracular

The dimension precondition is settled.  The correspondence one is not: R1C and
the budget both consume ground-truth-derived 2D correspondences, which no problem
setting supplies.  They remain capacity oracles, and 1.0 degree / 0.5 cell is a
target for a predictor rather than a measured deployable accuracy.

That is precisely why the next step measures whether any available feature can
read a line to that precision, before a predictor is written.
