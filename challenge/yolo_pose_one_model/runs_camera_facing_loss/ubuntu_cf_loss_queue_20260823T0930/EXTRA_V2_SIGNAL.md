**EXTRA_V2_SIGNAL = NULL_OR_WORSE**
```
                   C V2-10K  E V2-12.5K
----------------------------------------
V1val mAP            0.0272      0.0231
V1val median          27.05       27.38
V1val p90             53.29       56.38
V2val mAP            0.9266      0.9264
V2val median           2.36        2.31
V2val p90              8.62        8.58
real det              0.993       0.993
real cbox             0.679       0.579
real median           12.57       12.03
real p90              60.14       69.08
real gross20         0.3064      0.3381
real bottom           60.83       67.78
real DAY p90          60.14       69.08
real NIGHT p90          nan         nan
```

GAP TO yolo26n-ft
```
corner_median  FT    7.06  C_gap    +5.50  E_gap    +4.96  추가closure +9.8%
corner_p90     FT   19.25  C_gap   +40.88  E_gap   +49.83  추가closure -21.9%
```

hits {'cbox_+3pp': False, 'median_10pct': False, 'p90_10pct': False, 'gross20_10pct': False}
harm {'median': False, 'p90': True, 'cbox': True, 'det': False}
INTERIM_BEST_DATASET = C_V2_EARLY10K
★ C→E 는 순수 N 효과가 아니다 (unique sample + optimization update 동시 증가).
★ scanner: C train eff 9704 / E train eff 12173, val 은 둘 다 123 (동일).
engineering screen. real 은 EXPLORATORY membership. FINAL dataset 아님(interim).