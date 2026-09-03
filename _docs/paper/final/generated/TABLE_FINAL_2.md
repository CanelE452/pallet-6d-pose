# Table 2 — Daytime and nighttime adaptation

Two different nighttime subgroups exist in the artifacts and are **not**
interchangeable. This table uses the narrow acquisition-condition split; the
broad lighting split is reported underneath with its own sample size.

```text
Method                                    Day det  Night det  Day kp[px]  Night kp[px]
──────────────────────────────────────────────────────────────────────────────────────
Synthetic-only (R0)                         1.000      0.840      10.556         7.686
Synthetic-replay control                    1.000      0.800      10.555         7.964
Naive self-training                         0.971      0.960      11.852         8.309
Confidence                                  0.971      0.980      12.380         9.465
+ reprojection consistency                  0.971      0.960      11.461         8.642
+ keypoint-removal consistency              0.986      0.960      11.118         8.737
+ horizontal-flip consistency (full)        0.986      0.960      11.576        10.072
```

```text
subgroup sizes    Daytime N = 70   Nighttime N = 50   (plastic only)
broad split       Lighting_day N = 168   Lighting_night N = 106   (plastic + wood)
```

## What this table says

Nighttime detection rises from 0.840 to 0.960 or higher for every adapted arm.
**Naive self-training already reaches 0.960 and confidence-only selection
reaches 0.980**, so the nighttime detection gain cannot be attributed to the
geometric consistency filters.

In the same rows, keypoint error does not fall in either lighting condition.
That contrast — detection up, localisation not — is the paper's main result.

Nighttime N is 50; every claim drawn from this subgroup must carry that sample size.
