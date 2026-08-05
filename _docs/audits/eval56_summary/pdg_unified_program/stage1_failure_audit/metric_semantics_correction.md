# Metric semantics: t100

`t100` in `development_metrics.csv` counts **corner errors above 100px**, not
pose reprojection.

```
A2 | D13 | D0    PnP solved 0/13     t100 sum 3
```

Those two are not in conflict.  The three are corner-level errors spread over
two frames (one frame contributes 1, another 2), on frames where PnP never
solved.  Pose catastrophic count is 0 by definition here, because there is no
pose.

**Correction.**  An earlier summary said "A2 produced 3 poses beyond 100px on
D13".  That is wrong and is withdrawn.  No pose was produced at all on D13 by
any arm.
