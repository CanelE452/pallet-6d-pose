# depth Kabsch yaw — 앞면 코너 depth 로 자세 풀기

```
stage   dom      arm              N  yaw med  <3deg%  <6deg%
------------------------------------------------------------
ORACLE  outside  inset0_med      99    13.27      16      25
ORACLE  outside  inset0_p20      99    12.08      14      28
ORACLE  outside  inset6_p20     101     8.61      24      42
ORACLE  outside  inset12_p20    102     4.10      37      59
ORACLE  outside  inset12_p35    102     5.46      33      54
------------------------------------------------------------
ORACLE  night    inset0_med      32    12.33       3      19
ORACLE  night    inset0_p20      32    11.16       3      25
ORACLE  night    inset6_p20      33     7.51      18      42
ORACLE  night    inset12_p20     33     5.82      27      55
ORACLE  night    inset12_p35     33     5.81      27      52
------------------------------------------------------------
REAL    outside  inset0_med      58    13.46      14      26
REAL    outside  inset0_p20      58    15.05      16      29
REAL    outside  inset6_p20      58     7.81      31      47
REAL    outside  inset12_p20     59     5.77      31      54
REAL    outside  inset12_p35     59     6.18      31      47
------------------------------------------------------------
REAL    night    inset0_med      20    12.98      10      30
REAL    night    inset0_p20      20    12.29      20      30
REAL    night    inset6_p20      20     9.94      30      35
REAL    night    inset12_p20     20     5.25      35      55
REAL    night    inset12_p35     20     6.05      30      50
------------------------------------------------------------
```

비교 기준: 현재 모델 yaw {'outside': 6.54, 'night': 6.04}, GT 불확실성 바닥 {'outside': 0.41, 'night': 0.69} (deg).
ORACLE 이 모델보다 나쁘면 이 경로는 기각(상한 논증).
