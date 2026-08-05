# Does the response come back?

D13, 13 frames that produced nothing at all without padding.

```
        arm  centroid>0.30  R4  R6  corners (med)  D0 PnP  rescued reproj px  yaw deg
─────────────────────────────────────────────────────────────────────────────────────
   original              0   0   0              0       0                nan      nan
    reflect              9   6   6              3       7               66.6     12.0
  replicate             11  10   7              6      10               51.9     13.6
constant127             11  10   8              8      10               46.8      6.7
```

**Yes, and substantially.**  From 0 of 13 on every count, padding restores the
centroid on 9-11 frames, the full R4 condition on 6-10, and a D0 pose on 7-10.
The global collapse is therefore **not** an architectural inability to see a
truncated pallet -- it is an input-boundary effect.

The ranking is the surprise: **constant grey is the best arm and reflect the
worst.**  A3 recovers 11 centroids, 10 R4, 8 R6 and a median of 8 of 8 corners;
A1 recovers 9, 6, 6 and a median of 3.

## What that says about the mechanism

Constant grey adds **no context at all** -- every padded pixel is uninformative.
If the recovery came from continuing the scene past the frame edge, A3 would be
the weakest arm.  It is the strongest.  What all three arms share is that the
pallet occupies a smaller fraction of the network input and no longer touches the
border, so the effect is **scale and canvas margin**, not context continuation.

Reflect being worst is consistent with the same reading: the mirrored texture
puts a second, false pallet edge just outside the true one, and that is
misleading rather than helpful.
