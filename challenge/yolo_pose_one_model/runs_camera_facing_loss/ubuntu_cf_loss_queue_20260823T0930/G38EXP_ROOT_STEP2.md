# G38_EXP73916 — ROOT CAUSE STEP 2 (exposure-matched generic-only)

```
             unique  exposure  batch/ep  total steps  target
------------------------------------------------------------
G38           38002     38002      1188        71280       0
G38EXP        38002     73916      2310       138600       0
OLD           55959     73916      2310       138600   17957
```

## SAME REAL n=128
```
model    scope    cbox     med     p90  gross
----------------------------------------------
A42      ALL     0.422   53.46   87.91  0.917
A42      DAY     0.540   53.46   87.91  0.917
A42      NIGHT   0.000     n/a     n/a    n/a

G38      ALL     0.852   12.03   66.66  0.314
G38      DAY     0.940   11.61   63.90  0.298
G38      NIGHT   0.536   16.38   78.16  0.417

G38EXP   ALL     0.867   12.92   89.71  0.331
G38EXP   DAY     0.950   11.87   72.41  0.307
G38EXP   NIGHT   0.571   18.91  143.99  0.477

OLD      ALL     0.969    9.68   40.99  0.222
OLD      DAY     0.980    9.61   54.56  0.247
OLD      NIGHT   0.929    9.84   22.76  0.125

C43      ALL     0.797   14.28   91.98  0.401
C43      DAY     0.840   13.61   71.76  0.387
C43      NIGHT   0.643   17.30  145.66  0.465

FT       ALL     0.984    6.47   25.40  0.135
FT       DAY     0.990    6.55   27.53  0.141
FT       NIGHT   0.964    6.31   21.37  0.111

```

## NIGHT candidate
```
model     any-cbox    top1  cand/fr  wrong%   margin
--------------------------------------------------
G38          0.821   0.536     7.93     86%  +0.0362
G38EXP       0.750   0.571     4.68     89%  +0.0103
OLD          0.964   0.929     2.25     29%  +0.8793
```

## RECOVERY (G38 → OLD 구간에서 G38EXP 위치)
```
R_cbox            +9.1%
R_margin          -3.1%
R_candidate      +57.2%
```

**TARGET_CONTENT_REQUIRED**

guards: {'any_cbox_drop': -0.0714285714285714, 'DAY_cbox_delta': 0.010000000000000009, 'DAY_med_delta': 0.2670184294789042}
bootstrap(night paired): {'n_paired_night': 12, 'delta_median_G38_minus_G38EXP': -21.403722174826626, 'ci95': [-51.2968299146287, 1.5101230297266959]}

★ alias 는 동일 RGB 반복 — unique 다양성은 그대로다. exposure/update 통제이지
  'generic 을 늘린 것' 이 아니다. NIGHT n=28.