# GT pose reference audit

Run **before** any model pose result exists. The question here is only whether
the ground-truth reference is trustworthy on its own terms.

```text
total        319
resolved     319
unresolved   0
```

## Per population

```text
population                          n  chosen med     p90     p95     max   alt med  margin med
───────────────────────────────────────────────────────────────────────────────────────────────
ALL                               319        0.96    2.76    3.39    6.23     10.19        8.55
domain:daytime                     70        1.32    3.56    3.95    6.23     10.55        8.07
domain:nighttime                   50        1.23    2.65    2.96    3.56      7.79        6.38
domain:none                       199        0.70    2.57    3.08    4.76     11.33       10.93
plastic_standard_110x130x11       194        1.08    2.76    3.39    6.23      7.37        6.32
wood_small_80x59x14               125        0.69    2.64    3.37    4.76     13.54       13.03
```

`chosen` is the reprojection residual of the selected hypothesis; `alt` is the
rejected one. The gap between them is what makes the axis identifiable.

## Quality bar

```text
bar            5.0 px
source         scripts/annotate/_audit_annotate.py:200 — the repository's pre-existing annotation-quality bar
invented here  no
```

Frames whose refit residual reaches the bar: **1**.
They are listed for review and are **not** excluded — the resolution lock
permits exclusion only on solver failure.

```text
frame                                chosen      alt   stored    elev
─────────────────────────────────────────────────────────────────────
eval_pallet07__1778652172717607680     6.23    13.77     4.48    11.5
```

## Least separated frames

Smallest margin between the two hypotheses. Reported so a reader can see how
close the closest calls were; no margin threshold selects frames in or out.

```text
frame                                margin   ratio   chosen    elev
────────────────────────────────────────────────────────────────────
plastic_day_01__014632                 0.84     1.5     1.55    18.4
wood_night_01__031153                  0.92     2.4     0.67     4.0
plastic_night_01__040717               0.92     2.2     0.80     3.1
plastic_night_01__040923               1.11     9.9     0.13     6.0
wood_night_01__031953                  1.21     5.8     0.25    10.1
plastic_day_01__020954                 1.31     4.5     0.38     9.0
plastic_day_01__020955                 1.43     7.9     0.21     9.0
plastic_day_01__014704                 1.47     3.0     0.73    15.0
wood_night_01__031298                  1.48     2.1     1.37     6.0
plastic_day_01__021348                 1.49     8.9     0.19     8.5
```

## Verdict

```text
all frames resolved        True
counts match 319           True
ready for pose evaluation  True
```

No model prediction was read while producing this reference or this audit.
