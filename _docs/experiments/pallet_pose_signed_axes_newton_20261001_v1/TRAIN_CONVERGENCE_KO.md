# Newton solver를 적용한 Signed T/R 회귀의 독립 TRAIN 검산

**독립 수치 검산 PASS.** 이 판정은 실제 T/R 개선이나 실사 일반화 판정이 아니다. 고정253 특징·anchor 차분·signedlog1p target을 독립 복원하고 TRAIN2,598행(유효2,597·실패1)을 모두 유지했다. runtime은 참조나 안전 mask 없이 max(predicted T,predicted R)를 사용한다.

zero 초기화에서 고정 generalized-Newton/Armijo로 얻은 네 accepted 상태를 검산했다. 초기·모든 trial 호출과 반감 alpha, Armijo 부등식, 최종 accepted-call 연결을 확인하며 재학습하지 않았다. 각 frame의 유효 후보×두 축 Huber(delta1) 평균을 전체2,598행으로 평균하고 모든506 가중치에λ=1e−4 ridge를 적용했다. Torch64 autograd와 독립 가중 design-matrix의506×506 block curvature를 대조했다. residual 절댓값1에서는 고전적 Hessian이 없으므로 사전 선언한0 곡률을 사용하며, gradient-gap 인증은 C1 강볼록성에 근거한다. START/CK/FIT/COMPLETE/PREFIT의 basis와6개SHA 및 전체trace를 검산했다.

| 모델 | Huber | objective | gradient L2 | gap 상한 | 호출 수 |
|---|---:|---:|---:|---:|---:|
| R0_ONLY | 0.216026019 | 0.220022577 | 5.7e-16 | 1.63e-27 | 103 |
| UNION_s1 | 0.386049295 | 0.394591254 | 7.37e-16 | 2.72e-27 | 157 |
| UNION_s2 | 0.387978628 | 0.396978862 | 6.09e-16 | 1.85e-27 | 156 |
| UNION_s3 | 0.386856372 | 0.396047982 | 1.04e-15 | 5.44e-27 | 112 |

## 동일 TRAIN의 실제 선택 비교

이전 RBF 모델의 고정 score/선택/물리오차 통계를 정확히 재현한 뒤 현재 고정 모델과 비교했다. 이전 CE와 새 Huber는 다른 목적식이므로 손실값을 직접 비교하지 않는다. 다음 수는 전체2,598행 중 선택 수이며 실패1행은 별도 유지한다. 조건부2,597행의 비율과 각축 위반·T/R 중앙값/P90은 JSON에 보존했다.

| 모델 | anchor 이전→현재 | safe 개선 이전→현재 | unsafe 이전→현재 | 바뀐 선택 |
|---|---:|---:|---:|---:|
| R0_ONLY | 2575 → 2593 | 21 → 4 | 1 → 0 | 18 |
| UNION_s1 | 2451 → 2291 | 92 → 135 | 54 → 171 | 247 |
| UNION_s2 | 2463 → 2294 | 85 → 120 | 49 → 183 | 288 |
| UNION_s3 | 2506 → 2320 | 56 → 106 | 35 → 171 | 254 |

## 회귀 target 오차

다음은 anchor를 제외한 유효 후보에 대한 signedlog1p 단위 MAE/RMSE와 정확 부호 일치율이다. cm/degree 오차가 아니며, anchor(정답·예측 모두0)를 포함해 회귀 성공률을 부풀리지 않는다. exact0도 독립 class로 남기고 별도 threshold를 도입하지 않았다.

| 모델 | 축 | MAE | RMSE | 부호 일치율 |
|---|---|---:|---:|---:|
| R0_ONLY | T | 0.555766 | 1.041547 | 90.720% |
| R0_ONLY | R | 0.727744 | 2.180201 | 92.799% |
| UNION_s1 | T | 0.603477 | 0.999442 | 79.682% |
| UNION_s1 | R | 1.068557 | 1.764764 | 81.658% |
| UNION_s2 | T | 0.605806 | 1.009189 | 79.977% |
| UNION_s2 | R | 1.070961 | 1.765318 | 81.337% |
| UNION_s3 | T | 0.602772 | 1.002716 | 79.977% |
| UNION_s3 | R | 1.070295 | 1.767804 | 81.158% |

anchor 예측0과 선택된 max예측≤0은 모델 내부 성질이며 실제 두 물리오차 비악화를 보장하지 않는다. 기존 anchored discrete target 일치도는 JSON의 부가 진단이고 새 회귀의 학습 target이 아니다. 새 fit·optimizer step·VAL/실사 품질 열람·추가 선택 정책은 실행하지 않았다. 사전 source45와 원래/개입 실사5개 AND 판정은 별도로 유지한다.

[검산 JSON](TRAIN_CONVERGENCE.json) · [학습 계약](TRAIN_PROTOCOL.json) · [학습 전 검산](PREFIT_REVIEW_KO.md)
