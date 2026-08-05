# Every PnP rescue, decomposed

```
   set  arm             frame_id  domain  base_n  arm_n  new_ch  new_err  reproj   yaw               cause
──────────────────────────────────────────────────────────────────────────────────────────────────────────
eval56   T3  1778653017736058368     cad       3      4       0    22.75   235.2  72.6  four_point_minimum
eval56   T4  1778653017736058368     cad       3      4       0    22.75   235.2  72.6  four_point_minimum
eval56   R3  1778653017736058368     cad       3      4       0    22.75   235.2  72.6  four_point_minimum
```

There is exactly one, and it is the same frame in T3, T4 and R3.

`1778653017736058368` (cad) had three accepted corners, one short of the
`valid < 4` guard in `current_solve` (`paper_s2_frozen_diagnostic.py:1152`).
Dropping the gate to 0.225 accepts channel 0 at 22.75px from GT, the frame
reaches four correspondences, and PnP returns a pose.

That pose has a **fixed-GT reprojection of 235.2px and a yaw error of 72.6
degrees**.  The gate allows 17.34px.  `threshold_rescue_examples.png` shows why:
the four surviving points sit in a near-degenerate cluster along one edge, so
the solve is unconstrained in depth.

The rescue cause is `four_point_minimum` -- purely crossing the correspondence
count, not a better-conditioned correspondence set.

## Comparison with PFDR N3

N3 rescued **two different cad frames**, `1778653020759897344` and
`1778653033056483584`.  Neither appears here at any threshold down to 0.20.
The threshold arms and N3 do not rescue the same frames, and the reason is in
`THRESHOLD_GO_STOP_GATE.md`: the corners N3 recovered were not near the gate.
