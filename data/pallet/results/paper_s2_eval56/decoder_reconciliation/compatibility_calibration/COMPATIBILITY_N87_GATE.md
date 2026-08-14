# N87 gate

Nine conditions, all fixed before the sweep ran.  An arm passes only if every
one holds.

```
arm  sigma  centroid survival >= 83  objects built >= 83  PnP candidates >= 63  positive depth >= 95%  reproj not >10% worse  catastrophic <= 1  objects median <= 2  objects p95 <= 5  association not collapsed  verdict
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
S00    0.0                        -                    -                     Y                      Y                      -                  -                    Y                 Y                          Y     FAIL
S05    0.5                        -                    -                     Y                      Y                      -                  -                    Y                 Y                          Y     FAIL
S10    1.0                        -                    -                     Y                      Y                      -                  -                    Y                 Y                          Y     FAIL
S15    1.5                        -                    -                     Y                      Y                      -                  -                    Y                 Y                          Y     FAIL
S20    2.0                        -                    -                     Y                      Y                      -                  -                    Y                 Y                          Y     FAIL
S25    2.5                        -                    -                     Y                      Y                      -                  -                    Y                 Y                          Y     FAIL
S30    3.0                        -                    -                     -                      Y                      -                  Y                    Y                 Y                          -     FAIL
```

**No sigma passes.  CONFIG_ONLY_RESCUE = FAIL.**

The binding condition is the first one: centroid survival must reach 83 of 87
and the best any sigma achieves is 74, at sigma = 0.  The reprojection and
catastrophic-regression conditions also fail everywhere, so even a relaxed
reading of the survival requirement would not produce a candidate.

Per the audit's own rule, eval56 and wood were **not** run: they are one-shot
holdouts and are only spent once a sigma has been selected on N87.

## Selection

```
{
 "selected": null,
 "reason": "no sigma cleared the N87 gate",
 "verdict": "CONFIG_ONLY_RESCUE = FAIL"
}
```

## A leakage disclosure the plan assumed away

The plan treats N87 as a clean development set and eval56/wood as holdouts.
N87 and wood are disjoint, but **N87 and eval56 share 12 frames** (all in the
`outside` domain; 12 of eval56's 56).  Had a sigma been selected here and then
validated on eval56, 21% of that holdout would have been contaminated.  Since
no sigma was selected the point is moot for this run, but any future run of
this calibration must either report the leak-free 44-frame subset alongside the
full 56, or use a development set disjoint from eval56.
