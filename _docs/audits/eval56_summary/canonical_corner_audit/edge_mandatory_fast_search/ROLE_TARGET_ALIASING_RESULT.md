# STRUCTURAL_ROLE_TARGET_ALIASING_PRESENT

Two of the twelve targets frequently describe the same line, and it does not
explain the underfit.

## The audit

Geometry only, whole `LINE_TRAIN`, 13,618 frames x 66 role pairs.  A pair is
aliased when the two supporting lines agree to within both reused thresholds --
1.0 degree, the task angle budget, and 0.75 canonical50 cell, the label's own
tube sigma.  Neither number is new.

```
                     frames   with alias   share    alias pairs   min-angle p1   min-rho p1
LINE_TRAIN           13,618       10,211   74.98%
G0_nondegenerate     10,660        7,274   68.24%       16,283          0.002        0.001
G1_FACE_COLLAPSE      2,500        2,483   99.32%       10,721          0.000        0.001
G2_LINE_LIKE            458          454   99.13%        1,631          0.000        0.001
```

Three quarters of the training set contains at least one pair of roles the
supervision cannot separate.

## Which pairs, and why it is structural

```
rank  roles     corner edges          meaning
1     (4, 6)    (1,5) vs (2,6)        top-face right edge vs bottom-face right edge
2     (2, 7)    (0,4) vs (3,7)        top-face left  vs bottom-face left
3     (0, 5)    (0,1) vs (2,3)        top-face front vs bottom-face front
4     (8, 11)   (4,5) vs (6,7)        top-face rear  vs bottom-face rear
5     (0, 11)   (0,1) vs (6,7)        top-front vs bottom-rear (true edge-on only)
```

Under `camera_dynamic_0123_v4`, {0,1,4,5} is the top and {2,3,6,7} the bottom.
The top four pairs are therefore each a top-face edge against the bottom-face
edge directly beneath it, separated in 3D only by the pallet's height -- 0.098 to
0.243 m against a width of 1.4 m.  A pallet is a thin slab, so those two outlines
project onto nearly the same image line from most viewpoints.

This is a property of the object and the target definition, not a labelling
error.  It is also exactly the quantity `d799101` called `thickness`, which is
why `G1` is 99% aliased by construction.

## It does not predict the error

Existing M0 epoch-5 checkpoint, one forward pass on `D0_SEEN512`, frames split by
whether they contain any aliased pair.

```
                roles    angle med   angle p90   offset med   offset p90
with_alias      4,661       6.3044      54.877       2.5332       26.734
no_alias        1,267       7.9082      52.888       3.4134       29.092
```

Aliased frames are **better**, not worse, by 1.6 degree in the median.  Whatever
aliasing costs, it is outweighed by something else those frames have -- they are
the flatter, more overhead views, which also tend to be larger in the image.  The
comparison is confounded and the direction is recorded rather than explained.

```
STRUCTURAL_ROLE_TARGET_ALIASING_PRESENT   TRUE
CURRENT_UNDERFIT_CAUSE                    not this
TRAINING_FILTER                           KEEP
```

`G0` alone sits at 5.83 degree and is itself 68% aliased, so aliasing does not
separate the frames the model handles from the ones it does not.  Declaring it
the cause of a 6.60 degree underfit would not survive either number.

## Next

```
M0 data x optimizer-step 2x2    UNCHANGED
pool                            unchanged
step counts                     unchanged
```

Nothing was trained, filtered or deleted.  No PnP, no CIGM, no dimensions.
`untouched`, `eval56`, `wood45` and final-test remain unopened.
