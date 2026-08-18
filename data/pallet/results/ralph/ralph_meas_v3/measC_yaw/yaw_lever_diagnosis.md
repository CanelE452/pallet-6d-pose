# yaw lever diagnosis — 6deg 는 어디서 오는가

```
dom      model    N     ALL  GTrear  GTfront  GTtoprear   e2d_f   e2d_r
------------------------------------------------------------------------
outside  R0      73    6.54    1.08     5.56       3.12    8.03   24.38
outside  Ours    99    7.19    1.21     5.81       3.76    8.83   30.67
------------------------------------------------------------------------
night    R0      28    6.04    1.77     5.91       4.27    8.75   31.18
night    Ours    32    6.48    1.94     5.13       3.35    8.38   25.84
------------------------------------------------------------------------
```

elevation 층화 (ALL_PRED / GT_REAR, deg):
```
outside R0
   elev    0-5  n= 44  yaw= 6.63  GTrear= 1.15  e2d_rear= 24.76
   elev   5-10  n= 20  yaw= 7.59  GTrear= 4.60  e2d_rear= 38.82
   elev  10-20  n=  9  yaw= 3.90  GTrear= 0.50  e2d_rear= 24.29
   elev  20-90  n=  0  yaw=  n/a  GTrear=  n/a  e2d_rear=  n/a
outside Ours
   elev    0-5  n= 55  yaw= 7.44  GTrear= 0.96  e2d_rear= 38.77
   elev   5-10  n= 34  yaw=11.83  GTrear= 3.41  e2d_rear= 27.55
   elev  10-20  n=  9  yaw= 3.47  GTrear= 0.46  e2d_rear= 21.71
   elev  20-90  n=  0  yaw=  n/a  GTrear=  n/a  e2d_rear=  n/a
night R0
   elev    0-5  n=  5  yaw= 8.29  GTrear= 1.18  e2d_rear= 37.50
   elev   5-10  n= 23  yaw= 3.80  GTrear= 1.83  e2d_rear= 29.22
   elev  10-20  n=  0  yaw=  n/a  GTrear=  n/a  e2d_rear=  n/a
   elev  20-90  n=  0  yaw=  n/a  GTrear=  n/a  e2d_rear=  n/a
night Ours
   elev    0-5  n=  6  yaw= 6.96  GTrear= 0.67  e2d_rear= 30.96
   elev   5-10  n= 26  yaw= 6.26  GTrear= 2.49  e2d_rear= 25.02
   elev  10-20  n=  0  yaw=  n/a  GTrear=  n/a  e2d_rear=  n/a
   elev  20-90  n=  0  yaw=  n/a  GTrear=  n/a  e2d_rear=  n/a
```

yaw floor (measB, GT 불확실성): outside 0.41 / night 0.69 deg
순환 주의: GT rear 코너 6,7 은 92~100% 가 GT pose 의 PnP 외삽 -> GT_REAR 는 상한.
