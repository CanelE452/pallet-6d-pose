**COVERAGE_EFFECT_10K = POSITIVE**
```
                 A V1-10K   C V2-10K
--------------------------------------
V1val mAP          0.7806     0.0272
V1val median         4.63      27.05
V1val p90           34.09      53.29
V2val mAP          0.1013     0.9266
V2val median        24.00       2.36
V2val p90           57.59       8.62
real det            1.000      0.993
real cbox           0.414      0.679
real median         52.76      12.57
real p90            87.48      60.14
real gross20       0.8922     0.3064
real bottom         87.58      60.83
real day p90        87.48      60.14
real night p90        nan        nan
```

GAP TO yolo26n-ft (closure = (A_gap − C_gap)/A_gap)
```
corner_median  FT    7.06  A_gap   +45.69  C_gap    +5.50  closure +88.0%
corner_p90     FT   19.25  A_gap   +68.22  C_gap   +40.88  closure +40.1%
```

hits {'cbox_recall_+5pp': True, 'median_10pct': True, 'p90_10pct': True, 'gross20_10pct': True}
harm {'median': False, 'p90': False, 'cbox': False, 'det': False}
★ effective N: V1 9867 vs V2 9704 (corrupt 171/2) — 완전한 matched-N 은 아니다.
engineering screen. real 은 EXPLORATORY membership.