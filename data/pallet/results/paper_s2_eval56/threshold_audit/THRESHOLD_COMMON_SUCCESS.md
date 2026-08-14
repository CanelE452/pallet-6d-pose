# The frames that already solved

Paired, on frames where both the baseline and the arm produced a pose.  A new
rescue never enters this table, so nothing here can be inflated by detection.
Bootstrap is frame-clustered, 10,000 resamples, seed 1.

## eval56

```
arm   n    base     arm   chg%  d_med  imp  wor    p90  cat>=10  P(imp)
───────────────────────────────────────────────────────────────────────
 T1  50  11.558  11.558  +0.00  +0.00    0    1  +0.00        0   0.000
 T2  50  11.558  11.266  -2.53  +0.00    1    2  +0.00        0   0.637
 T3  50  11.558  11.266  -2.53  +0.00    1    4  +0.00        1   0.246
 T4  50  11.558  11.266  -2.53  +0.00    1    6  +0.22        2   0.067
 R1  50  11.558  11.558  +0.00  +0.00    0    1  +0.00        0   0.000
 R2  50  11.558  11.266  -2.53  +0.00    1    1  +0.00        0   0.637
 R3  50  11.558  11.266  -2.53  +0.00    1    3  +0.00        1   0.246
 C1  50  11.558  11.558  +0.00  +0.00    0    1  +0.00        0   0.000
```

## wood

```
arm   n   base    arm   chg%  d_med  imp  wor    p90  cat>=10  P(imp)
─────────────────────────────────────────────────────────────────────
 T1  44  9.284  8.984  -3.23  +0.00    1    0  +0.00        0   0.630
 T2  44  9.284  8.984  -3.23  +0.00    2    1  +0.00        1   0.316
 T3  44  9.284  8.984  -3.23  +0.00    2    2  +0.00        2   0.192
 T4  44  9.284  8.984  -3.23  +0.00    2    4  +0.00        4   0.033
 R1  44  9.284  8.984  -3.23  +0.00    1    0  +0.00        0   0.630
 R2  44  9.284  8.984  -3.23  +0.00    2    1  +0.00        1   0.316
 R3  44  9.284  8.984  -3.23  +0.00    2    1  +0.00        1   0.316
 C1  44  9.284  9.284  +0.00  +0.00    0    0  +0.00        0   0.000
```

The median falls a little on both sets -- eval56 -2.53%, wood -3.23% -- and
that looks like a win until you count frames.  On eval56 **every arm has more
frames getting worse than better**: T1 0 improved / 1 worsened, T2 1/2, T3 1/4,
T4 1/6.  The median moves because one frame improves by more than several
frames degrade, not because the pose is generally better.  `P(improvement)` is
0.000 to 0.637 on eval56 and never above 0.630 on wood.

From T3 down, catastrophic regressions appear: frames that already solved now
solve **10px or more worse** (eval56 1 then 2, wood 2 then 4).  Accepting a
corner that is 25px off does not merely add a bad point, it moves the pose of a
frame that was fine.

wood T1/R1 is the only place the count is not adverse (1 improved, 0 worsened,
0 catastrophic) -- one frame, on one set.
