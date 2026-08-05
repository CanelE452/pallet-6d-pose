# Parity

## D1 -- P0 reproduces the existing results exactly

```
   set  PnP   reproj  corner    near      far  >50  >100  NaN  parity
─────────────────────────────────────────────────────────────────────
eval56   50  11.5578  7.2411  4.6755  11.4063   45    17  119      OK
  wood   44   9.2839  9.2255  6.7325  14.1798   40    36   51      OK
```

Every arm's P0 column also reproduces the number it was originally judged on:
E2 eval56 far 9.6422 and reproj 11.7433, S1 eval56 reproj 8.5191, C1 eval56 PnP
55, N2 eval56 reproj 11.6680, N3 eval56 PnP 52, E2 wood far 11.8776, N2 wood
reproj 8.8733.  Pinned in `test_arm_p0_parity`.

## D2 -- P1 is the repository's own extractor

`decoder_paths.decode_p1` calls
`filter_pr_camfacing.extract_keypoints_from_belief` with its default threshold
and converts the returned grid coordinates to image space with the same
`image_size / 50` scale P0 uses.  No part of the Gaussian, the NMS, the 11x11
average or the offset is restated in the wrapper; the test asserts those tokens
are absent from `decoder_paths.py`.

## D3 -- P2 direct forward vs cache wrapper

10 deterministic eval56 frames, decoded once straight from the forward and once
after a float32 round trip through an npz:

```
max 2D coordinate difference   0.000e+00 px
max pose difference (R and t)  0.000e+00
objects direct vs cached       0 vs 0
selected hypothesis            identical on every frame
```

Both are exactly zero, well inside the 1e-6 tolerance, so the cache wrapper is
not the reason P2 behaves as it does below.
