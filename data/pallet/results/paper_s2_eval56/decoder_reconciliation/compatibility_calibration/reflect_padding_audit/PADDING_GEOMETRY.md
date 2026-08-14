# Geometry and intrinsics

```
original          W x H            = 640 x 480 for every D13 and C13 frame
pad               P                = 100 px on all four sides
padded canvas     Wp x Hp          = 840 x 680
K on the canvas   cx' = cx + P,  cy' = cy + P,  fx and fy unchanged
model input       preprocess_squash -> 400 x 400
belief            50 x 50 spanning the padded canvas
decode            x_original = bx * (Wp / 50) - P
```

`pad_frame` resizes the padded canvas back to (W, H) before the model's own
squash to 400x400.  The two resizes compose to a single squash of the 840x680
canvas onto 400x400, which is why the intrinsics above are exact.

## Unit tests (Phase C)

```
padded-K projection equals original projection + (P, P)      max err 1.14e-13 px
unpad round trip                                             max err 5.68e-14 px
PnP from (original points, K) vs (padded points, K_padded)   |dr| 2.08e-16  |dt| 4.44e-16
```

All far inside the 1e-6 requirement, so the two coordinate conventions are
interchangeable and the audit reports everything in original-image coordinates.
