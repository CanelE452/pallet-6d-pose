# P26 INFERENCE PATH ABLATION (training-0)

## real positive n=128
```
mode            ALLcbox     med      p90   n_kpt  NIGHTany  NIGHTtop1
--------------------------------------------------------------
M0 E2E            0.828  12.523   69.710     848     0.714      0.500
M1 O2M_RAW        0.828  12.940   77.928     848     0.786      0.429
M2 O2M_NMS        0.828  12.940   77.928     848     0.786      0.429
```

## real positive DAY / NIGHT (n_correct_box 동반)
```
mode           scope     cbox  n_cb     med      p90  gross20
----------------------------------------------------------
M0             ALL      0.828   106  12.523   69.710    0.334
M0             DAY      0.920    92  11.861   59.701    0.300
M0             NIGHT    0.500    14  23.674  151.213    0.554
M1             ALL      0.828   106  12.940   77.928    0.325
M1             DAY      0.940    94  12.495   63.139    0.305
M1             NIGHT    0.429    12  19.804  152.615    0.490
M2             ALL      0.828   106  12.940   77.928    0.325
M2             DAY      0.940    94  12.495   63.139    0.305
M2             NIGHT    0.429    12  19.804  152.615    0.490
```

## real negative n=2,689
```
mode                 AP    AUROC   FPR@95   det@.4  FP/img@.4     p90     p95     p99
----------------------------------------------------------------------------------
M0 E2E           0.6460   0.9374   0.3020   0.0915     0.0941   0.363   0.554   0.818
M1 O2M_RAW       0.6790   0.9377   0.3737   0.1123     0.4429   0.429   0.592   0.748
M2 O2M_NMS       0.6790   0.9377   0.3737   0.1123     0.1131   0.429   0.592   0.748
```

## NIGHT 전이 (M0 -> M2, n=28)
```
A_M0wrong_M2correct      1
B_M0correct_M2wrong      3
C_both_correct           11
D_both_wrong             13
top1 correct  M0 14  M1 12  M2 12  / 28
any  correct  M0 20  M1 22  M2 22  / 28
```

## synthetic G38 val 1,998 (secondary)
```
mode            boxmAP50  poseMAP  9kp med  9kp p90    cbox
------------------------------------------------------------
M0 E2E            0.9938   0.9059     2.28     8.88   0.994
M1 O2M_RAW        0.2371   0.2238     2.26     8.93   0.995
M2 O2M_NMS        0.9947   0.9072     2.26     8.93   0.995
```

## delta M2 - M0
```
ALL_cbox_pp                +0.0000
NIGHT_any_pp               +0.0714
ALL_median_degrade_rel     +0.0333
NIGHT_top1_pp              -0.0714
neg_AP_gain                +0.0330
FPR95_rel_drop             -0.2377
detect040_rel_drop         -0.2276
```

safety {'ALL_cbox': True, 'NIGHT_any': True, 'ALL_median': True}
benefits {'A_NIGHT_top1': False, 'B_neg_AP': False, 'C_FPR95': False, 'D_detect040': False}  (0/4, 필요 2)

**VERDICT = INFERENCE_PATH_NOT_FACTOR**

M0 PARITY  raw max diff {'conf': 0.0, 'box': 0.0, 'kps': 0.0, 'n': 0}  evaluator {'conf': 0.0, 'iou': 0.0, 'err': 0.0, 'correct_box_mismatch': 0}
checkpoint sha 불변 True · mtime 불변 True · 학습 0 회

★ 후보 수 기반 지표는 mode 간 정의가 달라 primary evidence 아님.
★ FP/image@0.40 단독 해석 금지 — AP/AUROC/FPR95/detect-rate 와 함께 본다.
★ NIGHT n=28, seed 1, training-0.