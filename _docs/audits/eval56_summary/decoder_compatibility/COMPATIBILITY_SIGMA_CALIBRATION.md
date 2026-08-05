# The sigma sweep, on N87 only

Seven pre-registered sigmas.  `thresh_map`, `thresh_points`, `thresh_angle`,
`threshold`, the NMS, the 11x11 window, the +0.4395 offset, the affinity
grouping, the PnP solver and the live gates are all untouched --
`decoder_paths.config_with_sigma` copies every other field and changes only the
one `find_objects` hands to `gaussian_filter` (detector.py:684).

Reference on the same tensors: D0 gives PnP **70/87** and a fixed-GT
reprojection median of **23.161629px**, reproducing the canonical value.

```
arm  sigma  centroid>0.30  objects  PnP  live gate  pos depth  reproj    vs D0  cat>=10  obj med  obj p95  gate
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
S00    0.0             74       74   70         45      1.000  27.807   +20.1%        2        1        1  FAIL
S05    0.5             74       74   70         46      1.000  28.049   +21.1%        2        1        1  FAIL
S10    1.0             73       73   70         46      1.000  28.057   +21.1%        2        1        1  FAIL
S15    1.5             71       71   68         47      1.000  27.306   +22.8%        2        1        1  FAIL
S20    2.0             70       70   67         45      1.000  25.617   +18.0%        2        1        1  FAIL
S25    2.5             65       65   63         39      1.000  25.618   +21.7%        4        1        1  FAIL
S30    3.0             11       11    2          0      1.000  27.773  +159.8%        1        0        1  FAIL
```

## The ceiling is not the smoothing

At **sigma = 0**, with no smoothing at all, centroid survival is **74 of 87**.
Thirteen frames have a raw centroid peak at or below 0.30:

```
0.0015  0.0032  0.0034  0.0126  0.0131  0.0146  0.0181
0.0220  0.0268  0.0952  0.1584  0.2521  0.2816      (8 night, 5 outside)
```

Nine of those are essentially zero.  No choice of smoothing can lift them, and
the gate asks for 83.  So the sweep fails on its first condition before the
smoothing question is even reached, and lowering the threshold -- the only other
lever -- was rejected in the previous audit, where wood corners at 0.25 landed a
median 402px from GT.

## Pose quality does not simply get worse; it flattens

The gate's reprojection condition compares medians over the common-success
frames and fires at every sigma (+18% to +22%).  The paired view is more
informative and, at sigma = 0 over the 70 common frames:

```
D0 median 23.162px    P2 median 27.807px    paired delta median -0.003px
improved 35   worsened 35   >=10px worse 2
frames D0 handled well  (D0 median 12.60px)  ->  P2 14.47px, delta +1.16px
frames D0 handled badly (D0 median 51.97px)  ->  P2 51.11px, delta -2.03px
```

The deployment decoder degrades the frames the mechanism decoder was good at
and marginally helps the ones it was bad at.  The two statistics do not
disagree -- a median of differences and a difference of medians are different
quantities -- but the second is the honest description.
