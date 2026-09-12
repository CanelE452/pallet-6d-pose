# v2 WLS 구현 오류 정정 재현

기존 187589a의 실행과 저장 성능은 보존한다. 기존 `DHT_LOCAL_TRACK_CLOSED`는 잘못 구현된 실행의 역사적 판정이며 정상 v2 방법의 결론으로 사용하지 않는다. 기존 oracle은 `INVALID_FOR_CAUSAL_DECOMPOSITION_DUE_TO_WLS_BUG`로 해석 보류한다.

실제 6개 START의 경로·SHA·1792개 순서를 승인 train manifest와 대조했다. INIT_PARITY의 512장 기록은 사전 감사 호출 오류였으며, 확인된 START는 train 1792장에 일치한다. 사후 정정 parity는 학습 전 감사가 올바르게 수행됐다는 소급 증명이 아니다.

수정은 single_mode_wls에서 signed와 force를 각 corner별로 계산하도록 옮긴 두 줄뿐이다. loss/utility/decoder/lattice/threshold/seed/batch/학습량과 나머지 학습 코드는 그대로다. 새 6개 run을 각각 2000 step 실행했으며 마지막 checkpoint만 평가했다.

정정 재학습의 고정 gate 판정: `DHT_LOCAL_TRACK_CLOSED`. Real DEV 실행: `False`. FINAL은 열지 않았다.

| 모델 | seed | primary | Point 대비 개선 % | median px | P90 px | good damage % |
|---|---:|---:|---:|---:|---:|---:|
| Point | — | 0.009474144 | 0 | 2.0264 | 7.3339 | 0 |
| direct | 1 | 0.010264925 | -8.347 | 2.6554 | 9.1782 | 1.017 |
| direct | 2 | 0.010493882 | -10.763 | 2.8554 | 9.4462 | 1.346 |
| direct | 3 | 0.010688872 | -12.822 | 2.9389 | 9.9207 | 1.764 |
| hough | 1 | 0.009653579 | -1.894 | 2.0897 | 7.9877 | 0.538 |
| hough | 2 | 0.009667522 | -2.041 | 2.0995 | 7.9087 | 0.508 |
| hough | 3 | 0.009658698 | -1.948 | 2.1185 | 7.9517 | 0.359 |

정상 WLS로 재학습한 Hough도 세 seed 모두 Point의 primary를 넘지 못했다. 따라서 WLS 오류만으로 이전 열세가 모두 설명되지는 않는다. 이 고정 예산·구조의 DHT local fusion 트랙은 종료한다. GT 선 oracle의 개선은 현재 보정식에 진단상 여지가 남음을 보여주며, 예측선과 utility 선택의 결합이 여전히 병목이라는 해석을 지지한다.

양의 변화율이 개선이다. 세 Hough seed가 모두 고정 gate를 통과해야 실사 DEV 평가가 허용된다.

| Hough seed | buggy primary | fixed-WLS posthoc | 정상 WLS 재학습 |
|---|---:|---:|---:|
| 1 | 0.009748398 | 0.009662919 | 0.009653579 |
| 2 | 0.009726279 | 0.009655999 | 0.009667522 |
| 3 | 0.009739518 | 0.009660481 | 0.009658698 |

posthoc는 기존 raw_line/utility를 그대로 사용하는 GT-free 재결합이다. 기존 buggy gradient로 학습한 가중치의 실행 영향이며 정상 학습을 대체하지 않는다.

Oracle은 공통 eligible-edge 교집합과 동일한 gain>0.25px GT 선택 규칙으로 재계산했다. 선이 다르면 실제 선택된 edge도 달라진다. 따라서 아래 차이는 순수 격자/예측 오차의 인과 기여율이 아니다.

| Oracle 가지 | primary | baseline 대비 개선 % |
|---|---:|---:|
| exact_GT | 0.007662988 | 19.117 |
| point | 0.009474144 | 0.000 |
| soft_GT | 0.007812419 | 17.540 |
| v1_hough_seed1 | 0.008749577 | 7.648 |

예측선 oracle의 출처는 기존 v1 Hough seed1이다. GT-assisted 선택 결과를 배포 정확도나 달성 가능한 성능 상한으로 해석하지 않는다.

| Hough seed | utility AUROC | AUPRC | signed gain Spearman |
|---|---:|---:|---:|
| 1 | 0.6091 | 0.2708 | -0.0711 |
| 2 | 0.5994 | 0.2773 | -0.0632 |
| 3 | 0.6067 | 0.2834 | -0.0636 |

Utility 통계는 선택된 단일 edge의 unit-weight counterfactual gain>0.25px 사건에 대한 것이다. 여러 edge를 예측 utility로 동시에 결합한 최종 보정의 calibration이나 기대 순이익과 같지 않다. 정상 WLS에서는 단일 edge의 감독·실행 동작 일치를 회귀검사로 확인했다.

고정 프로토콜 내 정정 재현의 결과만 논문 결론에 사용한다. 정확한 GT 선 oracle의 개선은 선 기하에 진단상 여지가 있음을 보이나, GT 선택을 포함하므로 예측선/utility 각각의 병목 기여도는 분리하지 못한다. DHT 전체의 불가능성이나 미래 성공을 확정하지 않는다.

원본 v1/v2 결과 JSON·checkpoint·예측·GT 보존 해시 검사는 PRESERVATION_VERIFIED.json에, 수식과 전체 저장 예측 독립 검사는 REGRESSION_TESTS.json/OFFLINE_REFERENCE_AUDIT.json에 기록한다. 정정 재학습의 전체 7×512 frame 좌표 지표와 WLS 독립 검사는 FINAL_INDEPENDENT_AUDIT.json에 기록한다.
