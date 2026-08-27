# HYBRID POINT x LINE — 판정

**VERDICT: HOUGH_INCREMENTAL_VALUE_NOT_ESTABLISHED**

질문: robust 한 point estimator 위에서도 Direct-Hough + F3 가 독립적
가치를 가지는가. 학습 0, 기존 추론 결과만 사용.

## 공정성

```
코너 순서   index-wise / order-free 비율 1.02~1.17 -> 순열 일치 [확인]
3D 점       annotate_pnp, top={0,1,4,5} = mh_fusion.TOP 과 일치
K / dims    프레임 라벨 동일
solver      base = SQPnP->refineLM,  hybrid = mh_fusion F3 정본
신뢰도      필터 없음 (visibility 와 belief peak 은 다른 양)
support     예측 점에서 생성. GT 는 O1/O2 에만
```

## REAL_DEV_OPEN_56 (n=56)

```
arm     avail   R med ↓   t med ↓   ADD-S ↓   IoU ↑   5cm5 ↑
B0      1.000      7.16    0.0959    0.1381   0.497    0.250
P0      1.000      5.29    0.0585    0.1109   0.509    0.304
B1      0.768      2.92    0.0430    0.0501   0.698    0.429
P1      0.768      5.25    0.0589    0.0838   0.584    0.196
B2      0.964      2.03    0.0495    0.0548   0.750    0.482
P2      0.964      4.22    0.0369    0.0709   0.664    0.375
B3      0.982      1.49    0.0232    0.0334   0.740    0.750
P3      0.982      4.21    0.0333    0.0653   0.653    0.518
O1      1.000      3.39    0.0178    0.0434   0.737    0.696
O2      1.000      0.05    0.0003    0.0008   0.996    0.946
```

P1 vs B1 paired (bootstrap 95% CI)

```
R     base    2.9214 -> hybrid    5.2452  delta    1.3298  CI [0.4803, 2.4101]  win 0.2558
t     base     0.043 -> hybrid    0.0589  delta    0.0044  CI [-0.0009, 0.0161]  win 0.3721
adds  base    0.0501 -> hybrid    0.0838  delta     0.012  CI [0.0014, 0.026]  win 0.3256
iou   base    0.6982 -> hybrid    0.5837  delta   -0.0748  CI [-0.1381, -0.0162]  win 0.6744
```

## REAL_CHALLENGE_DEV_105 (n=105)

```
arm     avail   R med ↓   t med ↓   ADD-S ↓   IoU ↑   5cm5 ↑
B0      1.000     34.54    1.6110    1.0541   0.000    0.009
P0      1.000     23.92    1.7214    1.1788   0.000    0.000
B1      0.476      7.84    0.1139    0.1654   0.398    0.057
P1      0.476      7.55    0.1320    0.1708   0.303    0.048
B2      0.838      3.47    0.0810    0.1030   0.632    0.248
P2      0.838      5.04    0.0768    0.1073   0.537    0.191
B3      0.971      2.96    0.0635    0.0760   0.681    0.400
P3      0.971      4.52    0.0742    0.0974   0.573    0.324
O1      1.000      2.86    0.0123    0.0337   0.768    0.743
O2      1.000      0.11    0.0005    0.0017   0.992    0.886
```

P1 vs B1 paired (bootstrap 95% CI)

```
R     base     7.841 -> hybrid    7.5457  delta    0.1733  CI [-0.2575, 0.7685]  win 0.44
t     base    0.1139 -> hybrid     0.132  delta   -0.0017  CI [-0.0071, 0.0047]  win 0.54
adds  base    0.1654 -> hybrid    0.1708  delta    0.0018  CI [-0.0076, 0.0085]  win 0.48
iou   base    0.3978 -> hybrid    0.3034  delta   -0.0104  CI [-0.0406, 0.0]  win 0.62
```

## 사전등록 판정 입력

```
open_R_rel_improve           -0.7955
challenge_R_rel_improve      0.0377
open_t_degrade               0.3698
challenge_t_degrade          0.1589
open_R_win                   0.2558
challenge_R_win              0.44
challenge_CI                 [-0.2575, 0.7685]
```

