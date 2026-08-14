# How far the deployment path gets

```
        arm  1 no raw response  2 lost in smoothing  3 no object  4 no association  6 PnP failed  7 gate failed  8 reached pose
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   original                 13                    0            0                 0             0              0               0
    reflect                  4                    9            0                 0             0              0               0
  replicate                  2                   11            0                 0             0              0               0
constant127                  2                   11            0                 0             0              0               0
```

Without padding all 13 frames fail at stage 1: there is no raw response to
smooth.  With padding 11 of 13 move to **stage 2** -- the raw response exists and
the sigma = 3 blur takes the centroid back under 0.30 before an object can be
built.  Not one frame reaches stage 3.

**This separates the two problems cleanly.**  The truncation collapse and the
belief-bandwidth mismatch are independent: fixing the input boundary moves every
frame from "no response" to "response destroyed by the deployment blur", which is
exactly the failure the compatibility audit measured on the other 74 frames.  A
model that is to work in deployment has to satisfy both conditions at once.
