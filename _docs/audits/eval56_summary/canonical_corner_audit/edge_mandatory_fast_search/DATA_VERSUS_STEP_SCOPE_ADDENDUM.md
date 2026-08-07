# What the 2x2 is allowed to be called

`72f080b` stands unedited.  Its labels are narrowed here so they cannot be read
as claims about data or optimization in general.

```
DATA_POOL_SCALE_NOT_PRIMARY_BOTTLENECK_UNDER_M0
K2_STEP_SCALE_WEAK
CURRENT_RECIPE_PLATEAU_AT_8515
```

Each is conditioned on the architecture that produced it.  Neither of these
follows and neither is claimed:

```
not claimed   more data is unnecessary
not claimed   optimization is intrinsically saturated
```

A 6.8x step increase bought under 40% and a 6.8x pool increase bought about 6%
on the holdout **for M0**.  A different architecture could have a different
appetite for either, and the plateau is a property of this recipe at this budget.

## The reference the next screen measures against

```
M0, FULL pool, 8,515 steps, D2_LINE_DEV512
                     5.5966 degree / 2.2597 canonical50 cell

absolute budget      <= 1.0 degree / <= 0.5 cell
safety               p90 <= 2.0 degree / <= 1.0 cell
40% reduction        <= 3.35796 degree / <= 1.35582 cell
```

Both thresholds are derived from that baseline before the architecture screen
runs and are not recomputed afterwards.
