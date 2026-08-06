# Paper dataset projection convention

Closed on the full 20,000 frames.  `ANNOTATION_PROJECTION_STATUS = OK`.

## The path

```python
from scripts.data_prep.blender.blender_math import build_view_matrix

X_world       = objects[0]["cuboid"]                    # world coordinates, 8 corners
R_w2c, t_w2c  = build_view_matrix(camera_data["location_worldframe"],
                                  camera_data["look_worldframe"])
X_camera      = R_w2c @ X_world.T + t_w2c
uv            = K @ X_camera ;  uv = uv[:2] / uv[2]
```

`K` is built per frame from `camera_data["intrinsics"]`.  Corners are compared at
raw index against `objects[0]["projected_cuboid"]` -- no permutation of any kind.

Implementation is not restated anywhere: the generator's own function is
imported.

```
scripts/data_prep/blender/blender_math.py :: build_view_matrix
file sha256  bcad33632a645273...
```

## Measured, 20,000 frames

```
corner points     160,000
  median  5.68e-14 px    p95 3.06e-13    p99 5.68e-13    max 3.07e-12
centroid points    20,000
  median  1.69e-06 px                    p99 1.34e-04    max 2.54e-04

non-finite 0 · negative-depth frames 0 · dynamic permutation 0
max by run          run1 3.07e-12   run2 1.67e-12
max by resolution   720x480 1.93e-12 · 640x480 1.59e-12 · 560x560 2.16e-12 · 960x540 3.07e-12
max by pallet       Pallet_0 1.83e-12 · Pallet_1 1.59e-12 · Pallet_2 1.59e-12 · Pallet_3 3.07e-12
```

Corners are at machine precision.  The centroid is consistent to sub-millipixel
rather than machine precision, which is why edge and CIGM targets use the eight
corners only; the centroid stays an A1 passthrough and is never a fusion target.

## Facts that fix the path

1. `cuboid` holds **world** coordinates, not object-local and not camera-frame.
2. `build_view_matrix` already returns an **OpenCV** world-to-camera pair, so no
   further `diag(1,-1,-1)` and no image Y flip belong anywhere near it.
3. `pose_transform` is not the matrix for these points.  Feeding world cuboid
   corners through it is a frame-composition error, and it is what produced the
   272 px and 412 px residuals in the first attempt.
4. Raw corner order already reflects `camera_dynamic_0123_v4`; `perm_v4` is
   baked into `projected_cuboid` and must not be applied again.
5. Four resolutions exist -- 640x480, 720x480, 960x540 and 560x560 -- so every
   step reads the frame's own width, height and K.  A fixed 640x480 assumption
   is wrong on three quarters of the shards.

The schema census supports this independently: `quaternion_xyzw_worldframe`,
`up_worldframe`, `view_matrix` and `projection_matrix` are absent from every
frame, so a quaternion-based camera path was never available.  Look-at is the
only camera path this dataset carries.

## Paths that must not be used again

```
world cuboid + pose_transform            272 - 412 px
world cuboid + inverse pose_transform    477 px
canonical cuboid regenerated from dims   272 - 313 px
world cuboid with R=I, t=0               4.9e9 px
any re-application of perm_v4 or its inverse
Hungarian assignment between predicted and stored corners
```

## Files

```
annotation_reprojection_summary.json   gate results and per-slice maxima
annotation_reprojection_audit.csv      one row per frame (1.86 MB)
annotation_schema_census.json          field presence over 5,000 sampled frames
```
