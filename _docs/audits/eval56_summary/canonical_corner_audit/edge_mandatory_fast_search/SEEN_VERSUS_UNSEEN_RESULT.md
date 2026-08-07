# SEARCH2K_MODEL_UNDERFIT_CONFIRMED

The model never fit its own training data.  Nothing about generalization was
observed, and the word does not apply to any of these numbers.

```
M0_F50_SLINE          D0 seen              D1 train-unseen      D2 line-dev
e1              9.2391 / 3.9424      9.3091 / 3.9028      9.4189 / 3.9292
e3              7.2092 / 3.0027      7.5678 / 3.0729      7.2438 / 3.0222
e5              6.6040 / 2.7023      6.9439 / 2.8335      6.8450 / 2.7717
e5 p90         54.35 / 27.24        49.89 / 25.31        52.29 / 24.74
n                      5,928                5,936                5,921

M1_F50_RGB_SLINE
e1             11.5815 / 5.0670     11.6603 / 5.0579     11.4823 / 4.8387
e3              8.3449 / 3.6652      8.6208 / 3.5924      8.2026 / 3.5537
e5              7.3727 / 3.1009      7.5186 / 3.0775      7.3931 / 3.0916

budget          <= 1.0 degree / <= 0.5 canonical50 cell
```

`D0_SEEN512` is 512 frames the checkpoint trained on.  At epoch 5 M0 sits at
6.6040 degree there -- **6.6 times outside the budget on data it saw**.  The
protocol's first branch decides on that alone.

## The ratios, now that they are meaningful

```
arm    D1/D0 angle   D1/D0 offset   D2/D0 angle   D2/D0 offset
M0          1.051          1.049         1.036          1.026
M1          1.020          0.992         1.003          0.997
```

Seen and unseen differ by 5% and less.  M1's offset ratio is below 1 -- it is
marginally *better* on frames it never saw.

This is the number that belongs where I wrote "54 times worse" in `5dd6036`.
That figure divided a dedicated 32-frame fit by a 2,000-frame run -- two
different experiments -- and it is withdrawn.  Measured properly, on one
checkpoint across three populations, the train-to-holdout ratio is **1.04**.

```
withdrawn    GENERALIZATION_COLLAPSE, and "54x", and the sentence in 5dd6036
             calling the result a generalization collapse
confirmed    SEARCH2K_DEV_BUDGET_FAIL
             SEARCH2K_MODEL_UNDERFIT_CONFIRMED
```

`5dd6036`'s protocol label `SUPPORTING_LINE_MAP_GENERALIZATION_FAIL` is kept as
the historical verdict of that screen, and it should now be read as
"the epoch-5 checkpoint fails the dev budget", not as a statement about
generalization.

## The three populations really were separable

```
D0 and D1   identical group histogram over 246 groups, zero frame overlap
D2          46 appearance groups, disjoint from search2k's
D2 measured here reproduces 5dd6036 to four decimals on all six points
            (M0 e1/e3/e5 and M1 e1/e3/e5)
```

So the harness did not change between screens, and the only thing that differs
between D0 and D1 is whether the model saw the frame.  It made no difference.

## The maps are identical across populations too

```
M0 e5              positive MSE   negative MSE    NCC     mass    peak
D0_SEEN512              0.13400        0.02324   0.2920   1469.0  0.8094
D1_TRAIN_UNSEEN512      0.13521        0.02321   0.2856   1467.3  0.8153
D2_LINE_DEV512          0.13445        0.02327   0.2912   1468.5  0.8201
```

Every map diagnostic agrees to the third decimal across seen and unseen frames.
The network produces the same quality of map everywhere, and that quality is not
good enough.  (NCC is a correlation diagnostic only; it is used here to compare
populations against each other, not to attribute a share of error.)

## Next branch, per the locked protocol

`SEARCH2K_MODEL_UNDERFIT_CONFIRMED` forbids architecture changes.  The next
screen is M0 alone in a data x optimizer-step 2x2:

```
2k / short      2k / long
full / short    full / long
```

which separates optimization from data.  Neither head depth nor MAP200 is
touched, and the reasons stand: the perfect-map oracle at MAP100 decodes to
about 0.006 degree, so the grid represents the budget with room to spare, and
the 32-frame fit reached 0.1268 degree, so the head can represent the map.  What
this screen adds is that the 2k run does not get near either.

`GLOBAL_CONTEXT x POSITIONAL_COORDINATES` was pre-registered as the first
architecture factorial, but only for the branch where D0 and D1 pass and D2
fails.  That branch did not happen, so it is not opened.

```
STRUCTURAL_LINE_MAP_CAPACITY   still unmeasured -- an underfit model cannot
                               bound what the architecture can represent
off-frame                      NOT TESTED
CIGM / PnP                     NOT BUILT
```

Nothing was trained here.  No PnP, no CIGM, no dimensions, no `validation512`.
`untouched`, `eval56`, `wood45` and final-test remain unopened.
