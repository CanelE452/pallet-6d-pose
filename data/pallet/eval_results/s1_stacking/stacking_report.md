# Paper-S1 stacking of 9 PL-accept signals over dev real GT sets

- weights: `weights/paper_s1/paper_s1_maskaux/net_epoch_0065.pth`
- inference: reflect-pad100 (near-field PL 후보 확보용; **NOT official eval**)
- domains(dev): outside/night/cad/noapril | SEAL 제외(pallet07/09,night08/09)
- total frames=120 | detected(>=6 corner)=82
- good = order-free Hungarian corner_med < 10px (12px 병기: GT 임계 취약)
- ★ N 도메인별 작음 → 예비. 필터 천장 = base 코너 정확도 (memory).

## (Domain) good PL distribution — good PL 이 어디에 있나
```
domain      N  det  good10  good12  best_cm  med_cm
---------------------------------------------------
outside    44   38       6      10      6.1    17.3
night      48   35       4       7      9.0    19.8
cad        22    4       0       2     11.1    13.7
noapril     6    5       3       3      6.8     9.0
```

## (a) Consensus curve — k신호 이상 통과 시 PL precision
```
 k>=  N_pass  good10  prec10  good12  prec12  recall10
------------------------------------------------------
   1      82      13   0.159      22   0.268       1.0
   2      82      13   0.159      22   0.268       1.0
   3      82      13   0.159      22   0.268       1.0
   4      81      13    0.16      22   0.272       1.0
   5      75      13   0.173      21    0.28       1.0
   6      69      13   0.188      21   0.304       1.0
   7      58      13   0.224      20   0.345       1.0
   8      46      12   0.261      19   0.413     0.923
   9      11       5   0.455       7   0.636     0.385
```

## (b) Per-signal discriminability (oriented so higher=better)
```
signal          AUC(good10) AUC(good12)  Spearman    n
------------------------------------------------------
f1_peak               0.848       0.806     0.441   82
f2_peak_ratio         0.595       0.545     0.121   82
f3_flip               0.672       0.666     0.218   82
f4_tta_stab           0.851       0.807     0.444   82
f5_rear_conf          0.858       0.797      0.46   79
f6_frsep              0.582       0.586     0.104   82
f7_posdepth             0.5         0.5      None   82
f8_size_env            0.53       0.542     0.055   82
f9_bbox_iou           0.838       0.758     0.428   82
```
- AUC>0.5 = good 을 양의 방향으로 가름 / ~0.5 = 무관 / <0.5 = 역방향.

## (c) Weighted combo (logistic, LOO-CV) vs best single signal
```
combo LR LOO-CV AUC = 0.877
best single signal = f5_rear_conf (AUC 0.857)
```

## 판정 (stacking 이 clean PL 을 만드나)
- 전체 good10=13, good12=22 (detected=82).
- 최고 precision consensus: k>=9 → prec10=0.455 (N=11, good=5).
- combo LR LOO-CV AUC=0.877 vs best single f5_rear_conf=0.857 → stacking 이 단일 대비 이득 있으면 combo AUC 가 유의미하게 높아야 함.
- ★ good PL 존재 도메인에서만 stacking 유효 (도메인 분포 표 참조). cad/noapril=unseen·near-field 라 good≈0 예상, outside 저앙각이 주력.
- ★ 과결론 금지: N 작고 pad(비공식)·GT 10px 취약. 데이터로만 판단.
