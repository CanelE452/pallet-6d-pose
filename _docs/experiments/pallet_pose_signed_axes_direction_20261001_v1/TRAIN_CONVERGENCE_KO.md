# 고정 잔차 방향18 추가: 독립 TRAIN 검산

**독립 수치 검산 PASS는 고정 목적식의 수렴·기록 확인이며 source/실사 T/R 개선 판정이 아니다.** TRAIN2,598행(유효2,597·실패1)을 유지했다. 기존253 입력에 O에서 고정한 signed residual18을 추가해271×2=542개 계수를 학습했다.

같은 Huber+nonzero-target sign logistic+ridge 목적식·타깃·scale·유효 후보를 유지한다. 각 frame의 원래 유효후보×2축 평균 후 전체2,598행 평균이며 실패1행은 손실0으로 분모에 남는다. 94/18 정규화는 고정 R0 TRAIN FP32 값이고, FP64 변환 뒤 candidate−anchor 차이를 계산한다.

Torch64 autograd gradient와 별도542×542 가중 design Hessian을 trainer·인증과 대조했다. 초기/최종 값을 독립 계산하고 모든 호출·accepted iteration·Armijo 반감·최종 accepted-state 기록을 검사했다. 중간 weight는 저장되지 않아 중간 Newton 방향을 모두 재계산했다고 주장하지 않는다. 새 fit·VAL·실사 참조는0이다.

| 모델 | Huber | sign logistic | J | gradient Linf | gap 상한 | 호출 |
|---|---:|---:|---:|---:|---:|---:|
| R0_ONLY | 0.218209317 | 0.126375760 | 0.350751831 | 7.59e-09 | 3.53e-12 | 32 |
| UNION_s1 | 0.385407879 | 0.269140479 | 0.664632551 | 6.07e-10 | 2.36e-14 | 35 |
| UNION_s2 | 0.386669353 | 0.269508051 | 0.666946002 | 2.41e-10 | 8.26e-15 | 48 |
| UNION_s3 | 0.386660367 | 0.270399955 | 0.667821303 | 4.69e-10 | 8.52e-15 | 72 |

## 동일 목적식의 이전 모델 비교

직전 sign253×2 가중치에18개 영행을 붙여 새271입력에서 계산했다. 원래 예측·목적식과253축 gradient를 재현했다. 추가18축 gradient는 일반적으로0이 아니므로 이전 인증과 비교하지 않는다. 두 J는 같은 Huber+sign+ridge이며 이전 weight를 초기값으로 사용하지 않았다.

| 모델 | 이전253+zero18 J | 새271 J | 감소 | 실제 선택 변경 |
|---|---:|---:|---:|---:|
| R0_ONLY | 0.352123902 | 0.350751831 | 0.001372071 | 1 |
| UNION_s1 | 0.667962978 | 0.664632551 | 0.003330427 | 94 |
| UNION_s2 | 0.670587373 | 0.666946002 | 0.003641372 | 83 |
| UNION_s3 | 0.670432604 | 0.667821303 | 0.002611301 | 71 |

JSON에는 이전·현재의 고정 TRAIN 선택 T/R, anchor 유지·safe/unsafe 수, 두 축 회귀 MAE/RMSE/sign confusion을 보존했다. 회귀 출력은 혼합 손실로 학습한 signed-axis 점수이며 cm/degree나 보정된 확률이 아니다. Anchor 입력·예측이0이므로 nonanchor 지표를 별도로 제공한다. 이전 discrete anchored-target 일치율은 부가 진단이며 현재 회귀 목적함수가 아니다.

[전체 검산 JSON](TRAIN_CONVERGENCE.json) · [사전 입력·수학 검산](PREFIT_REVIEW_KO.md)
