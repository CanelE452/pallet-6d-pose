# Native385 추가의 독립 TRAIN 검산

**독립 수치 검산 PASS는 고정 목적식의 수렴·기록 확인이며 T/R 일반화 성공 판정이 아니다.** 이전 Q의271 prefix·9해시·target·TRAIN2,598행(유효2,597·실패1)을 유지하고 고정 native385를 추가했다.

Huber=Huber_symmetric+Huber_underprediction이며 J=Huber+Sign_logistic+L2이다. Q와 같은 목적식·전체2,598행 분모·원래 valid·λ1e−4·Newton/Armijo 상한이다. 모든1,312개 계수에 ridge를 적용한다.

독립 Torch64 autograd와1,312×1,312 가중 design Hessian을 비교했다. 초기/최종 값과 모든 Armijo trial 기록을 확인했다. 중간 weight는 저장되지 않아 중간 Newton 방향 전수 재계산을 주장하지 않는다.

| 모델 | 대칭 Huber | 추가 Huber | sign logistic | J | gap 상한 | 호출 |
|---|---:|---:|---:|---:|---:|---:|
| R0_ONLY | 0.210733849 | 0.013788587 | 0.128270159 | 0.358772422 | 1.03e-18 | 54 |
| UNION_s1 | 0.381706431 | 0.108626377 | 0.271392698 | 0.774957642 | 8.68e-18 | 62 |
| UNION_s2 | 0.382499367 | 0.108675079 | 0.271219547 | 0.776579818 | 7.29e-25 | 142 |
| UNION_s3 | 0.382636339 | 0.108819658 | 0.272064217 | 0.777541943 | 3.77e-20 | 54 |

## 같은 목적식에서 이전 Q와 비교

이전271×2 가중치 뒤에0의385행을 붙여 같은656입력·같은 목적식으로 평가했다. 이전 native271 인증 gradient는271공간에서만 재확인하며 추가385 gradient는 별도다. padding weight는 비교용이고 새 학습 초기값은 모두0이다.

| 모델 | 이전 Q의 동일 J | 현재656 J | 감소 | 실제 선택 변경 |
|---|---:|---:|---:|---:|
| R0_ONLY | 0.371675472 | 0.358772422 | 0.012903050 | 3 |
| UNION_s1 | 0.801800523 | 0.774957642 | 0.026842881 | 55 |
| UNION_s2 | 0.804028872 | 0.776579818 | 0.027449054 | 60 |
| UNION_s3 | 0.804845394 | 0.777541943 | 0.027303451 | 62 |

JSON에 고정 TRAIN 선택 T/R·anchor·safe/unsafe·실패와 회귀 MAE/RMSE/sign confusion을 기록했다. 출력은 학습된 signed-axis 점수이며 cm/degree 또는 보정된 확률이 아니다. target 일치율은 과거 discrete target의 부가 진단이며 현재 supervision은 signed 두 축이다.

새 fit·VAL/실사 참조·정책 탐색0. T/R 성공은 별도 봉인된 source45와 원래+matched 실사5기준을 모두 확인해야 한다.

[전체 검산 JSON](TRAIN_CONVERGENCE.json) · [사전 검산](PREFIT_REVIEW_KO.md)
