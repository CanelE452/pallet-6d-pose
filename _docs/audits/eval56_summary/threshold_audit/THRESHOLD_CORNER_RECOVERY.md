# What a lower gate accepts

Newly accepted corners only (channels 0-7, GT present).  `lost` is corners the
base accepted and the arm did not -- zero everywhere, as it must be when the
threshold only moves down.

## eval56

```
arm  new_corners  median_error_px  le10_frac  le20_frac  gt50_frac  lost_corners
────────────────────────────────────────────────────────────────────────────────
 T1            1             7.36       100%       100%         0%             0
 T2            3             7.36        67%       100%         0%             0
 T3            6            17.41        33%        50%        17%             0
 T4            8            31.52        25%        38%        38%             0
 R1            1             7.36       100%       100%         0%             0
 R2            2             6.89       100%       100%         0%             0
 R3            5            22.75        40%        40%        20%             0
 C1            1            12.06         0%       100%         0%             0
```

## wood

```
arm  new_corners  median_error_px  le10_frac  le20_frac  gt50_frac  lost_corners
────────────────────────────────────────────────────────────────────────────────
 T1            1            12.37         0%       100%         0%             0
 T2            3           402.51         0%        33%        67%             0
 T3            4           332.40         0%        25%        75%             0
 T4            7           375.97        14%        29%        71%             0
 R1            1            12.37         0%       100%         0%             0
 R2            3           402.51         0%        33%        67%             0
 R3            3           402.51         0%        33%        67%             0
 C1            0              n/a        n/a        n/a        n/a             0
```

The gate asks for at least 70% of new corners within 20px and at most 10%
beyond 50px.

On eval56 that holds only while the threshold barely moves (T1/T2/R1/R2 recover
1-3 corners).  From 0.225 down, half or more of what arrives is wrong.

On wood it collapses immediately.  At 0.25 the three new corners have a median
error of **402px** -- they are not the corner at all, they are whatever the
map's largest remaining bump happens to be on a pallet the model has not seen.
`threshold_false_corner_examples.png` shows them.

The count of detected corners rises monotonically as the gate falls.  That is
arithmetic, not improvement.
