# 동일 TRAIN 입력에서 CLEAR/OCC 학습 모델 교차 진단

동일 seed42 사전 K8의62 occurrence/45 unique,476지원코너(계획 covered21)를 모든 모델에 입력했다. 새 fit0. 값은 augmented640px의 **각 모델 RAW/REF 의사 타깃** 추종이며 물리 GT·자연 가림6D·attention의 증명이 아니다. seed43 모델도 동일 seed42 진단 입력을 사용했다.

| seed/target | 같은 입력 | covered L2 mean CLEAR-trained→OCC-trained | unmasked L2 mean CLEAR-trained→OCC-trained | covered 개선/악화/동률 |
|---|---|---:|---:|---:|
| 42/RAW | CLEAR | 1.60706→1.63793 | 3.16711→3.19153 | 7/14/0 |
| 42/RAW | OCC | 10.97300→10.89535 | 3.41273→3.43242 | 15/6/0 |
| 42/REF | CLEAR | 2.06290→2.06850 | 3.70440→3.73042 | 13/8/0 |
| 42/REF | OCC | 11.32618→11.27199 | 3.92005→3.94075 | 16/5/0 |
| 43/RAW | CLEAR | 1.70023→1.76053 | 3.17915→3.20174 | 9/12/0 |
| 43/RAW | OCC | 10.88442→10.81864 | 3.42754→3.44611 | 10/11/0 |
| 43/REF | CLEAR | 2.12791→2.13329 | 3.70127→3.71840 | 8/13/0 |
| 43/REF | OCC | 11.25207→11.22740 | 3.91343→3.92474 | 12/9/0 |

CLEAR 입력의 covered는 실제 가림이 아닌 동일 계획 위치다. missing과640대각선 패널티, paired 평균/중앙값, 양쪽 RAW/REF 타깃 전체 결과는 JSON에 있다. 가림학습 모델의 이득/손해는 동일 입력에 대해 비교하지만, 이 소표본만으로 노출 부족과 학습능력·타깃 오차·instance 선택을 완전히 분해할 수 없다.

[결과 JSON](TRAIN_CROSS_INPUT_RESULTS.json) · [사전 입력 잠금](TRAIN_CROSS_INPUT_PROTOCOL.json) · [예측 잠금](TRAIN_CROSS_INPUT_PREDICTIONS_LOCK.json)
