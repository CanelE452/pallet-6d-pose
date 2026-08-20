# REAL_DEV F0 vs F3 — 최종 추론 경로 실측

이전 real 수치는 corner decoder + Point PnP 만 썼다. 그건 F0 이고,
SplitLate 의 line branch 를 버린 것이라 **최종 추론 성능이 아니었다.**
이번에는 정본 `mh_fusion.solve_arms` 의 F3 를 그대로 붙였다 — 새 solver 는
쓰지 않았다.

```
population   REAL_DEV_POS_V1  (비봉인 정본 eval, positive 56장)
SEALED       eval_pallet07, eval_pallet09, eval_night08, eval_night09  — 열지 않음
lambda_theta 3.0
```

## 두 가지 결정 (조용히 다른 걸 재지 않도록 명시)

**corner 입력** — `solve_arms` 는 8코너 전부를 요구하므로 solver 는
`_decode_peaks`(argmax, threshold 없음)를 읽는다. synthetic F3 검증과
같은 입력이다. threshold 0.3 검출기는 **따로** 보고한다. 둘을 섞으면
검출 실패가 pose 오차로 둔갑한다.

**line support** — synthetic 에서는 GT 코너에서 뽑았다. real 에서 그러면
oracle 이므로, 주 결과는 **예측 코너**에서 같은 `visible_segments` 로
뽑는다. GT 판은 parity 대조로만 남긴다.

## seed1

```
A 전체 positive      56
B corner 검출 성공   35  (62.5%)
C PnP 성공           56  ← argmax 입력이라 항상 8점, 정보량 없음
```

MAIN = A 기준 unconditional

```
arm      R med    R p90    t med    t p90    ADD-S    IoU3D   5cm5deg
F0       7.155  25.6806   0.0959   1.8966   0.1381   0.4969      0.25
F3      5.2875  21.6398   0.0585   2.0169   0.1109   0.5094    0.3036
```

paired F3 − F0 (같은 프레임, bootstrap 95% CI)

```
R     F0     7.155 F3    5.2875  delta   -0.6409  CI [-1.2781, 0.2504]  0 포함 — 미확립
t     F0    0.0959 F3    0.0585  delta    0.0023  CI [-0.0051, 0.0079]  0 포함 — 미확립
adds  F0    0.1381 F3    0.1109  delta    0.0041  CI [-0.0076, 0.0226]  0 포함 — 미확립
iou   F0    0.4969 F3    0.5094  delta   -0.0125  CI [-0.0456, 0.0]  0 포함 — 미확립
```

세트별 (F3)

```
set                n    R med    t med     IoU    5cm5
eval_outside      22   6.5486   0.3037  0.3676  0.1364
eval_noapril      12   4.7972   0.0436  0.5575  0.3333
eval_cad          22   4.7783   0.0319  0.5518  0.4545
```

support parity: 예측 support R med 5.2875 vs GT support 5.2875 — 차이가 사실상 없다.
즉 F3 는 GT 없이도 같은 성능을 낸다 (배포 가능).

score_4kp (positive 만)

```
min 0.211  median 0.7396  max 0.87
recall @ 0.1:1.0  0.2:1.0  0.3:0.9643  0.4:0.8929  0.5:0.8214  0.6:0.75  0.7:0.6429  0.8:0.1964  0.9:0.0
```

**threshold 를 고르지 않는다.** real negative 가 없어 precision /
AP / FPR 을 계산할 수 없다.

## seed2

```
A 전체 positive      56
B corner 검출 성공   32  (57.1%)
C PnP 성공           56  ← argmax 입력이라 항상 8점, 정보량 없음
```

MAIN = A 기준 unconditional

```
arm      R med    R p90    t med    t p90    ADD-S    IoU3D   5cm5deg
F0     10.7839  44.0786   0.1488   2.0112   0.2139   0.3427    0.1607
F3      6.6032  35.7266   0.1003   1.7233   0.1174    0.404     0.125
```

paired F3 − F0 (같은 프레임, bootstrap 95% CI)

```
R     F0   10.7839 F3    6.6032  delta   -0.9394  CI [-2.3682, 0.2754]  0 포함 — 미확립
t     F0    0.1488 F3    0.1003  delta   -0.0085  CI [-0.0362, 0.0033]  0 포함 — 미확립
adds  F0    0.2139 F3    0.1174  delta     -0.01  CI [-0.0441, 0.0023]  0 포함 — 미확립
iou   F0    0.3427 F3     0.404  delta       0.0  CI [-0.0155, 0.0165]  0 포함 — 미확립
```

세트별 (F3)

```
set                n    R med    t med     IoU    5cm5
eval_outside      22  13.3513   0.4763   0.265  0.0455
eval_noapril      12   3.8544   0.0681  0.4712    0.25
eval_cad          22   6.2555   0.0813  0.3861  0.1364
```

support parity: 예측 support R med 6.6032 vs GT support 6.7325 — 차이가 사실상 없다.
즉 F3 는 GT 없이도 같은 성능을 낸다 (배포 가능).

score_4kp (positive 만)

```
min 0.0125  median 0.7847  max 0.938
recall @ 0.1:0.9643  0.2:0.9464  0.3:0.9464  0.4:0.8929  0.5:0.8929  0.6:0.8571  0.7:0.7321  0.8:0.4821  0.9:0.0893
```

**threshold 를 고르지 않는다.** real negative 가 없어 precision /
AP / FPR 을 계산할 수 없다.

