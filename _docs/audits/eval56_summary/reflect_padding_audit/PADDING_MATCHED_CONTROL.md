# What padding costs the healthy frames

C13, the 13 matched frames whose centroid was alive without padding.

```
        arm  centroid lost  D0 PnP  reproj relative  improved  worsened  catastrophic >=10px
────────────────────────────────────────────────────────────────────────────────────────────
   original              -      11                -         -         -                    -
    reflect              1      11            +7.3%         4         6                    4
  replicate              0      12           +10.1%         4         7                    4
constant127              0      13           +16.8%         5         6                    3
```

Frame-clustered paired bootstrap, 10,000 resamples, seed 1:

```
        arm   n  median delta px     p90  95% CI of mean  P(improvement)
────────────────────────────────────────────────────────────────────────
    reflect  10            +1.10  +70.21  [-1.08,+58.20]           0.037
  replicate  11            +2.17  +59.62  [+0.88,+56.32]           0.017
constant127  11            +1.90  +65.83  [-5.28,+51.50]           0.099
```

Padding **helps detection and hurts pose** on these frames.  PnP successes go
11 -> 12 -> 13, so more frames solve, but the frames that already solved get
worse: the reprojection median rises 7.3% to 16.8% against a 5% allowance,
worsened outnumbers improved on every arm, and 3 to 4 frames regress by 10px or
more.  Reflect additionally loses one centroid outright.

This is the behaviour already recorded for this padding path -- useful on
truncated frames, neutral to harmful on frames where the object is fully
inside -- and it is why an always-on padded pass is not admissible.  Any use of
padding would have to be conditional, and the condition cannot be GT.
