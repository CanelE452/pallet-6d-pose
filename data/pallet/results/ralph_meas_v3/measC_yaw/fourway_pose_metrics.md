# fourway — 논문 Method 표 (self-domain, median)

```
domain   method                 N  succ%   det%    yaw°  cent cm   ADD m
----------------------------------------------------------------------
outside  Synthetic only    73/113   45.1   64.6    6.54    27.56   0.358
outside  Naive ST          95/113   48.7   84.1    6.81    35.52   0.417
outside  Reproj+flip ST    97/113   44.2   85.8    7.94    41.56   0.452
outside  Ours (loo+flip)   99/113   48.7   87.6    7.19    25.10   0.296
----------------------------------------------------------------------
night    Synthetic only    28/40    47.5   70.0    6.04    23.61   0.267
night    Naive ST          33/40    60.0   82.5    7.23    32.07   0.332
night    Reproj+flip ST    34/40    40.0   85.0    6.77    31.49   0.451
night    Ours (loo+flip)   32/40    65.0   80.0    6.48    17.54   0.189
----------------------------------------------------------------------
noapril  Synthetic only    15/18    83.3   83.3    0.89     5.46   0.056
noapril  Naive ST          15/18    83.3   83.3    0.91     5.76   0.061
noapril  Reproj+flip ST    15/18    83.3   83.3    1.11     7.43   0.076
noapril  Ours (loo+flip)   15/18    83.3   83.3    0.73     4.97   0.053
----------------------------------------------------------------------
```

n_gt (GT reproj<=5): {'outside': 113, 'night': 40, 'noapril': 18}
ADD 는 표준 ADD(대칭 fold 없음) — ADD-S 아님.
pseudo-GT floor: ADD m outside 0.027 / night 0.028 / noapril 0.078.
