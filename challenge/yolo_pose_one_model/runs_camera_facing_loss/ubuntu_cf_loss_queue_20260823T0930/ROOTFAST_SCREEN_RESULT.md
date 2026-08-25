# ROOTCAUSE FAST SCREEN (30ep, common N=9,704)

```
                        B10        M10        T10
--------------------------------------------------
ALL cbox              0.812      0.656      0.859
ALL median            18.79      18.82      15.66
ALL p90               93.74     153.86      56.83
ALL gross20           0.478      0.473      0.419

DAY cbox              0.840      0.750      0.910
DAY median            16.14      16.50      15.98
DAY p90               92.49     129.52      57.78
DAY gross20           0.451      0.420      0.427

NIGHT cbox            0.714      0.321      0.679
NIGHT median          30.01     140.68      14.32
NIGHT p90             94.38     209.11      46.44
NIGHT gross20         0.594      0.917      0.382

```

## NIGHT candidate
```
                        B10        M10        T10
--------------------------------------------------
any-cbox              0.929      0.750      0.750
top1-cbox             0.714      0.321      0.679
cand/frame           12.250     13.107      4.964
wrong%                0.964      1.000      0.786
margin                0.155     -0.064      0.753
```

MATCH_GOOD_ENOUGH = False   (SMD r3d -2.24, elev +1.15)

**CASE = TARGET_RESIDUAL**
GENERIC_SUPPORT_EFFECT = False
TARGET_SPECIFIC_RESIDUAL = True

**RENDER_ACTION = NO_RENDER_MOVE_TO_ADAPTATION**

times(min) {'B10': 26.0, 'M10': 32.6, 'T10': 32.8}  total 91.8

★ M10 은 B10 과 7,990/10,000 겹친다 — V2 풀에 OT support 가 없어 실질 대조가 약하다.
  T10−M10 을 target identity 효과로 단정하지 않는다. NIGHT n=28, seed 1, 30ep.