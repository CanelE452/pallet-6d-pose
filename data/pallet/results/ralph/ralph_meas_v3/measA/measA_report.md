# measA — self-training keypoint motion / sharpening / error (MEASUREMENT ONLY)

```
R0    = weights/paper_s2_stageB/net_epoch_0057_noseg.pth
self  outside = data/pallet/results/ralph/ralph_selftrain/h6_s2_outside/round_02.pth
self  night   = data/pallet/results/ralph/ralph_selftrain/h3_s2_night/round_02.pth
self  noapril = data/pallet/results/ralph/ralph_selftrain/h7_s2_noapril/round_02.pth
self  cad     = data/pallet/results/ralph/ralph_selftrain/h4_s2_combined/round_02.pth  [combined (no cad-specific self)]
decode= raw argmax per belief(9,50,50)->orig(W/50,H/50); peak=max; squash400
THRESH_main(S1)=0.3  sweep=[0.1, 0.2, 0.3, 0.4, 0.5]  reproj_gate>5.0px excluded
leakage: infer(model,img)=PNG only; GT projected_cuboid used for err only
```

## 6. err_R0 median + 1. Δ distributions (S1 @THRESH=0.3)
```
dom        nFr   S1n  S1gt  errR0med  Δpos_med        Δpos_IQR  Δpeak_med  Δerr_med        Δerr_IQR
----------------------------------------------------------------------------------------------------
outside    117   689   689    15.308      12.8   [0.0, 27.341]     0.0443       0.0[-11.309, 1.278]
night       43   246   244    12.313       9.6     [0.0, 12.8]     0.0781       0.0   [-6.08, 0.02]
noapril     18   121   121     6.291       0.0      [0.0, 0.0]     0.0116       0.0      [0.0, 0.0]
cad         44   126   126     9.623       0.0      [0.0, 9.6]    -0.0924       0.0    [0.0, 0.715]
```

## 4. paired Wilcoxon err_R0 vs err_self (S1 @THRESH=0.3)
median Δerr≈0 for all (front corners stable=zeros); direction is in the SKEW.
HL=Hodges-Lehmann pseudomedian (the location Wilcoxon tests); n_impr/n_wors among GT corners.
```
dom          n  medΔerr  meanΔerr      HL      dir          p   eff_r  n_impr  n_wors    n_0
----------------------------------------------------------------------------------------------
outside    689      0.0    -5.784  -6.133  improve   1.07e-06   0.186     256     188    245
night      244      0.0    -2.535  -3.673  improve   6.58e-03   0.174      84      62     98
noapril    121      0.0    -1.405  -1.627  improve   3.25e-01   0.089      15      14     92
cad        126      0.0     1.798   2.849   worsen   9.95e-03    0.23      23      33     70
```

### Wilcoxon sweep (THRESH x reproj-gate ON/OFF) — flag if verdict flips
```
dom        thr  gate     n  med_Δerr          p
outside    0.1  True   792       0.0     0.0000
outside    0.1 False   818       0.0     0.0000
outside    0.2  True   732       0.0     0.0000
outside    0.2 False   758       0.0     0.0000
outside    0.3  True   689       0.0     0.0000
outside    0.3 False   715       0.0     0.0000
outside    0.4  True   661       0.0     0.0000
outside    0.4 False   687       0.0     0.0000
outside    0.5  True   629       0.0     0.0008
outside    0.5 False   651       0.0     0.0006
night      0.1  True   256       0.0     0.0116
night      0.1 False   267       0.0     0.0151
night      0.2  True   244       0.0     0.0066
night      0.2 False   253       0.0     0.0162
night      0.3  True   244       0.0     0.0066
night      0.3 False   252       0.0     0.0162
night      0.4  True   235       0.0     0.0220
night      0.4 False   243       0.0     0.0484
night      0.5  True   227       0.0     0.1054
night      0.5 False   235       0.0     0.1970
noapril    0.1  True   123       0.0     0.2134
noapril    0.1 False   123       0.0     0.2134
noapril    0.2  True   122       0.0     0.2134
noapril    0.2 False   122       0.0     0.2134
noapril    0.3  True   121       0.0     0.3252
noapril    0.3 False   121       0.0     0.3252
noapril    0.4  True   119       0.0     0.3869
noapril    0.4 False   119       0.0     0.3869
noapril    0.5  True   118       0.0     0.3869
noapril    0.5 False   118       0.0     0.3869
cad        0.1  True   161       0.0     0.9708
cad        0.1 False   163       0.0     0.9708
cad        0.2  True   137       0.0     0.1224
cad        0.2 False   139       0.0     0.1224
cad        0.3  True   126       0.0     0.0100
cad        0.3 False   128       0.0     0.0100
cad        0.4  True   119       0.0     0.0484
cad        0.4 False   121       0.0     0.0484
cad        0.5  True   110       0.0     0.0373
cad        0.5 False   112       0.0     0.0373
```

## 5. S1/S2/S3 counts per THRESH (S2=R0 miss→self det; S3=R0 det→self miss)
```
[outside]  (self: outside)
  THRESH        0.1     0.2     0.3     0.4     0.5
  S1 both       792     732     689     661     629
  S2 R0->s       70     116     150     164     180
  S3 s->R0        7       8       9      11      13
  S2 err_self_med@0.3=25.809 (n=150)

[night]  (self: night)
  THRESH        0.1     0.2     0.3     0.4     0.5
  S1 both       258     246     246     237     229
  S2 R0->s       35      36      34      37      32
  S3 s->R0        2       2       2       4       4
  S2 err_self_med@0.3=79.93 (n=34)

[noapril]  (self: noapril)
  THRESH        0.1     0.2     0.3     0.4     0.5
  S1 both       123     122     121     119     118
  S2 R0->s        5       0       1       3       3
  S3 s->R0        2       2       1       0       0
  S2 err_self_med@0.3=4.405 (n=1)

[cad]  (self: combined)
  THRESH        0.1     0.2     0.3     0.4     0.5
  S1 both       161     137     126     119     110
  S2 R0->s       39      34      31      17      11
  S3 s->R0       29      23      20      15      13
  S2 err_self_med@0.3=28.678 (n=28)

```

## 7. VERDICT per domain (measurement classification, NOT prescription)
```
criteria (fixed a priori; direction via HL pseudomedian since medΔerr≈0):
  Δpos_med<2px & Δpeak_med>0                 -> POSITION-STABLE, SHARPEN-ONLY
  Δpos_med large & Δerr not sig / HL~0        -> MOVED-BUT-RANDOM
  Δpos_med large & Wilcoxon sig & HL<0        -> REAL-POSITION-IMPROVE
  Δpos_med large & Wilcoxon sig & HL>0        -> MOVED + ERROR-WORSE
------------------------------------------------------------
outside   Δpos=12.8   Δpeak=0.0443   HL=-6.133  p=1.1e-06   n_impr/wors=256/188 -> MOVED + REAL-POSITION-IMPROVE (net, HL<0)
night     Δpos=9.6    Δpeak=0.0781   HL=-3.673  p=6.6e-03   n_impr/wors=84/62 -> MOVED + REAL-POSITION-IMPROVE (net, HL<0)
noapril   Δpos=0.0    Δpeak=0.0116   HL=-1.627  p=3.3e-01   n_impr/wors=15/14 -> POSITION-STABLE, SHARPEN-ONLY
cad       Δpos=0.0    Δpeak=-0.0924  HL=2.849   p=9.9e-03   n_impr/wors=23/33 -> POSITION-STABLE + small sig err-shift (HL=2.849, worsen); Δpeak_med=-0.0924
```
