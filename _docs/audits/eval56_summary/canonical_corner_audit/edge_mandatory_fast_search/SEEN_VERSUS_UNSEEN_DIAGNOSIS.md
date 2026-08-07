# What 5dd6036 measured, and the ratio I should not have taken

`5dd6036` stands unedited.  Its protocol verdict is kept.

```
historical protocol decision
  SUPPORTING_LINE_MAP_GENERALIZATION_FAIL

narrowed to what the evidence supports
  SEARCH2K_DEV_BUDGET_FAIL      CONFIRMED
  GENERALIZATION_COLLAPSE       UNRESOLVED
  SEARCH2K_UNDERFIT             UNRESOLVED
```

The screen evaluated the epoch-5 checkpoint on held-out frames only.  It never
measured that checkpoint on frames it had trained on, so it cannot say whether
6.845 degree is a gap between train and test or simply where the model sits
everywhere.

## The ratio was invalid

I wrote that M0 goes "from 0.1268 degree on memorised frames to 6.8450 on unseen
ones, 54 times worse".  Those two numbers come from **different training runs**:
0.1268 is a dedicated 1,500-step fit on 32 frames, 6.8450 is a 1,250-step run on
2,000 frames.  Dividing one by the other measures the difference between two
experiments, not a generalization gap.

`54x` is withdrawn.  A train-versus-test ratio may only be quoted once the same
checkpoint has been measured on frames it saw, which is what D0 is for.

## Role semantics, stated precisely

```
kept        shuffle 6.845 -> 50.90 degree, margin +44.05 against a +5 gate
            ROLE_CHANNEL_IDENTITY_LEARNED = TRUE
withdrawn   "the model knows which structural line each output owns"
accurate    predictions are strongly channel-role specific
```

A permutation test shows the channels are not interchangeable.  It does not show
that the network has any notion of what a role *means*.

## The three populations

Fixed before any forward pass.

```
D0_SEEN512          512 frames drawn from line_search2k -- the model trained on
                    every one of them
D1_TRAIN_UNSEEN512  512 frames from LINE_TRAIN \ line_search2k, allocated with
                    the identical per-group quota as D0
D2_LINE_DEV512      the existing line_dev512, unchanged, no new manifest
```

The split's own structure makes the matching exact rather than approximate:
`line_search2k` and the remaining 11,618 LINE_TRAIN frames cover the **same 260
appearance groups**, and every group has enough frames outside search2k to fill
its D0 quota, so the shortfall is zero.  `line_dev512`'s 46 groups are disjoint
from search2k's, which is what makes D2 the appearance-combination holdout.

Allocation is largest-remainder over sorted group ids and frames are taken in
sorted index order -- no RNG, so the manifests are a function of the split alone.

```
assert   D0 n D1 = 0      D0 n DEV = 0      D1 n DEV = 0
```

## Reading

```
D0 FAIL                    SEARCH2K_MODEL_UNDERFIT_CONFIRMED
                           "generalization collapse" is then forbidden, and D1/D2
                           are recorded but not read as architecture evidence
D0 PASS, D1 FAIL           WITHIN_LINE_TRAIN_GENERALIZATION_GAP
D0/D1 PASS, D2 FAIL        APPEARANCE_COMBINATION_GENERALIZATION_GAP
all three PASS             HARD_BLOCKED_DIAGNOSTIC_INCONSISTENCY
```

M0 is the diagnostic arm; M1 is recorded identically but the cause does not
depend on it.  The task gate is unchanged and shown only as a reference line.

## Why neither MAP200 nor a deeper head is the next move

The perfect-map oracle at MAP100 already decodes to about 0.006 degree and 0.006
canonical50 cell, so a 100x100 grid is far more than enough to *represent* 1
degree / 0.5 cell.  Raising the output grid is not where the current evidence
points.

And the 32-frame fit reached 0.1268 degree, so there is no evidence the present
head cannot represent a supporting-line map at all.  What is unknown is whether
the 2k checkpoint fits its own 2k training data -- which is exactly this
diagnostic, and why it comes before any architecture change.

Nothing is trained here.  No new architecture, no MAP200, no deeper head, no
off-frame expansion, no CIGM, no PnP, no `validation512`.  `untouched`,
`eval56`, `wood45` and final-test remain unopened.
