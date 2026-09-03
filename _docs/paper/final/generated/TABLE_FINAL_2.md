# Table 2 — Daytime and nighttime adaptation

The backing artifacts declare `population_contract.role = DEV` and
`held_out_final = false`, and their own reports warn that these are development
values. They are reported here as development results and are never described as
held-out or independently confirmed.

```text
Method                                    Day det  Night det  Day pooled kp med[px]  Night pooled kp med[px]
────────────────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                         1.000      0.840                 10.556                    7.686
Naive self-training                         0.971      0.960                 11.852                    8.309
Confidence                                  0.971      0.980                 12.380                    9.465
Full consistency filter                     0.986      0.960                 11.576                   10.072
```

```text
Daytime N = 70      Nighttime N = 50, plastic only
broad lighting split, not used here:  Lighting_day N = 168   Lighting_night N = 106   (plastic + wood)
```

## Reading this table

Nighttime N = 50 and the subgroup is plastic only. Every claim drawn
from it carries that sample size.

**The nighttime detection increase is not attributed to the geometry filter.**
Naive self-training already reaches 0.960 and confidence-only selection reaches
0.980 — higher than the full consistency filter. The movement belongs to
self-training as a whole.

In the same rows, 2D keypoint error does not fall in either lighting
condition. Detection up, 2D localisation not: that contrast is the paper's
main result.

**Raw pixel error is scale-sensitive.** The absolute Daytime and Nighttime
values must therefore not be read as a direct measure of relative condition
difficulty — a daytime value of 10.556 px against a nighttime 7.686 px does not
mean daytime is the harder condition. What is interpretable is the change from
R0 to an adapted arm **within** one lighting condition.
