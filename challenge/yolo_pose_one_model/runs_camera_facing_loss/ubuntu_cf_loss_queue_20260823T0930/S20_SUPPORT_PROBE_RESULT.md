# S20 SUPPORT PROBE — Render 최종 판정 (30ep, effective 9,704)

```
                        B10        S20        T10
--------------------------------------------------
ALL cbox              0.812      0.766      0.859
ALL median            18.79      14.84      15.66
ALL p90               93.74     109.15      56.83
ALL gross20           0.478      0.411      0.419

DAY cbox              0.840      0.770      0.910
DAY median            16.14      13.71      15.98
DAY p90               92.49      88.30      57.78
DAY gross20           0.451      0.370      0.427

NIGHT cbox            0.714      0.750      0.679
NIGHT median          30.01      25.45      14.32
NIGHT p90             94.38     126.84      46.44
NIGHT gross20         0.594      0.560      0.382

```

## NIGHT candidate
```
                        B10        S20        T10
--------------------------------------------------
any-cbox              0.929      0.857      0.750
top1-cbox             0.714      0.750      0.679
cand/frame           12.250      8.464      4.964
wrong%                0.964      1.000      0.786
margin                0.155      0.104      0.753
```

## correct candidate rank (night)
```
B10    {'rank3+': 5, 'rank1': 20, 'absent': 2, 'rank2': 1}
S20    {'rank1': 21, 'absent': 4, 'rank3+': 2, 'rank2': 1}
T10    {'rank1': 19, 'absent': 7, 'rank3+': 1, 'rank2': 1}
```

hits 2/6 → {'top1_cbox_+3f': False, 'median_-15%': True, 'p90_-15%': False, 'cand_-20%': True, 'wrong_-15pp': False, 'margin_+0.15': False}
guards {'all_cbox': False, 'all_median': True, 'night_any_cbox': False}

**SUPPORT_PROBE_SIGNAL = HARM**
**GENERIC_SUPPORT_CAUSAL_SIGNAL = False**
**RENDER_ACTION = CLOSE_RENDER_MOVE_TO_ADAPTATION**

support 1933 + broad 7771 = 9704, target 0
r3d_rendered median 0.134689, strict-thin(<= 0.15885) 87.1%

★ T10 은 target-specific diagnostic reference — paper winner 아님. NIGHT n=28, seed 1.