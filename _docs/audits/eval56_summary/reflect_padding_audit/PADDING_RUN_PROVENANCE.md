# Provenance

```
HEAD at start          82ec98b2dad63b0cf1c0507e17a67905fac7fc20
HEAD at write          82ec98b2dad63b0cf1c0507e17a67905fac7fc20
ep57 SHA256            c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896
python 3.10.20  torch 2.1.1+cu118  opencv 4.9.0  numpy 1.26.4
training steps         0
optimizers constructed 0
checkpoints written    0
tensor dtype           float32
final-test sessions    not read
```

## Membership

```
D13  13 frames   C13 13 frames   control sha 9230daa96f515e11
E44  44 frames   sha dff15140ff5d7a4c   NOT SPENT
W45  45 frames                                  NOT SPENT
D13 inter E44 0   C13 inter E44 0
D13 inter W45 0   C13 inter W45 0
```

## Varied

```
input padding only: pad = 100 px per side, border mode in
{reflect (BORDER_REFLECT_101), replicate (BORDER_REPLICATE), constant (127,127,127)}
plus the unpadded A0 baseline.  Four arms, fixed before running, none added.
```

## Held fixed

```
ep57 checkpoint, preprocess_squash, thresh_map 0.30, thresh_points 0.30,
threshold 0.30, thresh_angle 0.50, deployment sigma 3, NMS, 11x11 window,
+0.4395, affinity grouping, EPNP solver, live gates, dimensions.
K is shifted by the padding offset only; fx and fy are untouched.
```

No GT enters preprocessing, the padding is the same for every frame, and no
frame-, domain- or direction-specific padding was used.

## Outputs

```
PADDING_PATH_TRACE.md  PADDING_GEOMETRY.md  PADDING_RESPONSE_RECOVERY.md
PADDING_CORNER_PRECISION.md  PADDING_D0_POSE.md  PADDING_P2_DEPLOYMENT.md
PADDING_MATCHED_CONTROL.md  PADDING_CONFIRMATORY.md
PADDING_ARCHITECTURE_DECISION.md  PADDING_FINAL_DECISION.md
PADDING_RUN_PROVENANCE.md
padding_membership.json  padding_frames.csv  padding_corner_rows.csv
padding_response_metrics.csv  padding_gate.json  selected_padding.json
figures/ (9 png, local only)
```
