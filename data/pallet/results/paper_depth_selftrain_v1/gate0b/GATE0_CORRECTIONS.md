# Corrections to the Gate 0 report

Gate 0 is not edited. Its verdict of PARTIAL stands as written. Four statements in
it went further than the evidence, and this records the corrected version.

## 1. Differing intrinsics do not forbid pooling

Gate 0 wrote that day and night "do not share the same intrinsics, so they must not
be pooled". That is wrong as stated. What must not happen is applying one shared
matrix to both. If each recording is projected with its own calibrated matrix, day
and night frames can sit in the same pool without any problem. Gate 0B carries each
recording's own matrix throughout.

## 2. The presence of cam_K.txt does not identify the stream

Gate 0's inventory set `color_intrinsics_available` from the file merely existing.
A three-by-three matrix with no stream label proves only that some intrinsic was
saved. Gate 0B replaces that field with a graded `K_role`, and `COLOR_K_CONFIRMED`
is reserved for acquisition provenance that nobody has. The day recordings reach
`LIKELY_COLOR_K` because their matrix matches the DOPE baseline RGB header exactly;
the night recordings stay `UNPROVEN` because no such counterpart exists.

## 3. A same-matrix round trip proves the algebra, nothing else

Gate 0 reported a back-projection residual of about 1e-13 pixels. Projecting with a
matrix and unprojecting with the same matrix must return the original pixel; the
number confirms the implementation and says nothing about whether that matrix
belongs to the colour stream, whether the depth is aligned to it, or whether the
metric scale is right. Gate 0B states that limitation in the artifact itself.

## 4. Whole-region contamination does not mean the local signal is absent

This is the substantive one. Gate 0 measured depth across the whole box and the
whole corner hull, found a spread far larger than a pallet is deep, and concluded
the object was not separable. That inference does not follow: a box drawn around a
pallet contains ground and background by construction, so its spread is dominated
by them whatever the pallet surface looks like.

Gate 0B measures inside the projected cuboid faces instead, shrunk toward their own
centroids. The whole-box spread and the face spread are not the same quantity and
they do not agree.
