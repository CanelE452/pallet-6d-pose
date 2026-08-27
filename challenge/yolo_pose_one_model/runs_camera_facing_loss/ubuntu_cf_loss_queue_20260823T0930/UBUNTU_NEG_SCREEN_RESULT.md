# UBUNTU NEGATIVE SCREEN — N0 vs N1 (15ep, exposure 13,554)

## SAME REAL n=128
```
                         N0         N1          Δ
--------------------------------------------------
ALL cbox              0.836      0.789     -0.047
ALL med              12.339     11.938     -0.401
ALL p90             103.419     61.463    -41.956
ALL gross             0.325      0.287     -0.038

DAY cbox              0.920      0.920     +0.000
DAY med              11.092     11.439     +0.347
DAY p90              88.117     60.920    -27.197
DAY gross             0.285      0.274     -0.011

NIGHT cbox            0.536      0.321     -0.214
NIGHT med            24.894     16.100     -8.794
NIGHT p90           132.708    114.828    -17.880
NIGHT gross           0.567      0.417     -0.150

```
## NIGHT candidate
```
                         N0         N1
----------------------------------------
any_cbox              0.679      0.357
top1_cbox             0.536      0.321
cand_per_frame        5.036      2.571
wrong_present_frac      0.857      0.714
margin_median         0.426      0.546
```
## NEG HELD-OUT n=2,689
```
                               N0         N1          Δ
--------------------------------------------------------
AP                         0.6934     0.7306    +0.0372
AUROC                      0.9499     0.9446    -0.0053
FPR@TPR95                  0.2674     0.2012    -0.0662
neg_median                 0.0059     0.0000    -0.0059
neg_p90                    0.1735     0.0219    -0.1515
neg_p95                    0.3969     0.0788    -0.3181
neg_p99                    0.8392     0.6575    -0.1817
neg_cand_mean              3.3563     0.7542    -2.6021
separation_margin         -0.1001    -0.0082    +0.0919
neg_detect_rate@0.05       0.2064     0.0640    -0.1424
FP_per_image@0.05          0.2726     0.0725    -0.2001
recall_TPR@0.05            0.9219     0.7891    -0.1328
neg_detect_rate@0.25       0.0774     0.0245    -0.0528
FP_per_image@0.25          0.0811     0.0253    -0.0558
recall_TPR@0.25            0.7969     0.6875    -0.1094
neg_detect_rate@0.4        0.0498     0.0171    -0.0327
FP_per_image@0.4           0.0506     0.0175    -0.0331
recall_TPR@0.4             0.7656     0.6406    -0.1250
```

hits 4/6 → {'d_AP': True, 'd_FPR_at_TPR95': True, 'd_neg_detect_rate_040': False, 'd_FP_per_image_040': False, 'd_neg_cand_mean_rel': True, 'd_neg_p90': True}
guards {'d_all_cbox_pp': False, 'd_night_any_cbox_frames': False, 'd_all_median_rel_worse': True}

**NEG_SIGNAL = HARM**

batches/ep N0 424 = N1 424 · 15ep · init sha 1a806ca497fde517

★ host-internal DELTA 만 유효 — Windows P0/P1 과 absolute 비교 금지.
★ NEG heldout = 2,689 전체. 'FT 학습분 259 제외 2,430' 은 실측상 틀렸다(내용 교집합 0).
★ NIGHT n=28, seed 1, 15ep FAST SCREEN.