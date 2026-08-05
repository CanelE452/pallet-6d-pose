# 6 arms x 3 paths x 2 sets

Same cached belief and affinity per set x arm; only the decoder changes.
Corner statistics are pooled over corners, matching `summarise`, not a median of
per-frame medians.

## eval56 (56 frames)

```
arm  path  PnP   reproj    yaw   corner    near      far  >50  >100  NaN  P2obj
───────────────────────────────────────────────────────────────────────────────
 B0    P0   50  11.5578  1.672   7.2411  4.6755  11.4063   45    17  119      0
 B0    P1   46  13.0587  1.929   9.9539  6.6494  13.5668   32     9  145      0
 B0    P2    0      nan    nan      nan     nan      nan    0     0  448      0
 E2    P0   50  11.7433  1.361   6.3975  4.6755   9.6422   45    16  120      0
 E2    P1   47  13.3819  1.662   8.9541  6.6494  11.0374   32     9  147      0
 E2    P2    0      nan    nan      nan     nan      nan    0     0  448      0
 S1    P0   50   8.5191  1.339   7.4162  6.0759  10.9807   45    15   73      2
 S1    P1   44  11.8207  1.403   9.7794  6.4561  12.3490   32     9  147      2
 S1    P2    0      nan    nan      nan     nan      nan    0     0  448      2
 C1    P0   55  10.4157  1.958   8.4082  6.2775  11.3068   66    28   53      1
 C1    P1   43  12.9277  1.483  10.9151  7.4678  14.1851   40    12  143      1
 C1    P2    0      nan    nan      nan     nan      nan    0     0  448      1
 N2    P0   50  11.6680  1.411   6.4023  4.6755   9.5603   45    16  120      0
 N2    P1   47  13.4436  1.683   8.9706  6.6494  11.0888   31     8  147      0
 N2    P2    0      nan    nan      nan     nan      nan    0     0  448      0
 N3    P0   52  11.8004  1.798   7.8796  5.3089  11.4063   45    17  105      0
 N3    P1   46  13.0561  1.929   9.9539  6.6687  13.5668   32     9  145      0
 N3    P2    0      nan    nan      nan     nan      nan    0     0  448      0
```

## wood (45 frames)

```
arm  path  PnP   reproj    yaw    corner      near       far  >50  >100  NaN  P2obj
───────────────────────────────────────────────────────────────────────────────────
 B0    P0   44   9.2839  1.471    9.2255    6.7325   14.1798   40    36   51      0
 B0    P1   42  13.8395  1.304   13.7612   13.3859   14.6975   23    22   77      0
 B0    P2    0      nan    nan       nan       nan       nan    0     0  360      0
 E2    P0   44   9.0329  1.408    8.7754    6.7325   11.8776   38    34   53      0
 E2    P1   41  13.7111  1.331   13.5593   13.3859   13.8759   23    21   76      0
 E2    P2    0      nan    nan       nan       nan       nan    0     0  360      0
 S1    P0   44   9.5830  1.365    9.4887    7.3793   13.4677   40    35   43      3
 S1    P1   40  13.1199  0.985   13.4495   13.2028   14.4816   23    20   79      3
 S1    P2    0      nan    nan       nan       nan       nan    0     0  360      3
 C1    P0   45   8.2292  1.245    9.8202    7.7372   12.8440   51    44   28      7
 C1    P1   43  14.7494  1.472   14.0886   13.2443   14.9793   30    28   69      7
 C1    P2    1  12.1462  2.426  317.8503  513.9043  291.0618    6     6  354      7
 N2    P0   44   8.8733  1.381    8.7790    6.7325   11.5472   38    34   53      0
 N2    P1   41  13.7039  1.551   13.4985   13.3859   13.7599   23    21   78      0
 N2    P2    0      nan    nan       nan       nan       nan    0     0  360      0
 N3    P0   44   9.2921  1.556    9.3580    7.1376   14.1798   40    36   50      0
 N3    P1   42  14.1107  1.325   13.7655   13.3003   14.6975   23    22   77      0
 N3    P2    0      nan    nan       nan       nan       nan    0     0  360      0
```

Two things are visible immediately.

**P2 produces no object at all.**  Across 101 frames and six arms the
deployment decoder built 13 object
hypotheses in total, and only one of them (wood, C1) reached a pose -- a pose
with a 317px corner error.  The reason is in
`DECODER_ASSOCIATION_COMPARISON.md`.

**P1 is worse than P0 on this model, not better.**  eval56 PnP 50 -> 46,
reprojection 11.56 -> 13.06px, corner 7.24 -> 9.95px; wood PnP 44 -> 42,
reprojection 9.28 -> 13.84px.  The one thing it improves is the tail: >50px
corners fall 45 -> 32 (eval56) and 40 -> 23 (wood), because the D2 extractor
rejects far more channels (NaN 119 -> 145, 51 -> 77) and the ones it rejects are
the bad ones.
