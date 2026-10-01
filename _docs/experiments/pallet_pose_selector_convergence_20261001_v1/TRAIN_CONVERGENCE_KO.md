# 凸 scorer의 TRAIN 수렴 독립 검증

**네 fit 모두 명시적 CE+ridge objective의 수렴 인증을 독립 재계산으로 통과했다.** 이는 T/R 개선 판정이 아니다.

[검증 JSON](TRAIN_CONVERGENCE.json)은 원래 입력·각 checkpoint·START·TRACE·protocol hash를 연결한다. 저장 weight를 수정하지 않고 별도 PyTorch float64 cross-entropy/autograd로 objective와 gradient를 계산했다. 새 fit, optimizer step, VAL quality 또는 실사 GT 읽기는0회다.

| 모델 | CE | L2 penalty | objective | gradient L2 | objective gap 상한 | iterations / calls |
|---|---:|---:|---:|---:|---:|---:|
| R0_ONLY | 0.13250970 | 0.00469996 | 0.13720966 | 3.65e-08 | 6.66e-12 | 181 / 206 |
| UNION_s1 | 0.79701407 | 0.00696837 | 0.80398244 | 3.41e-08 | 5.82e-12 | 346 / 377 |
| UNION_s2 | 0.78539713 | 0.00741250 | 0.79280962 | 1.82e-08 | 1.66e-12 | 340 / 370 |
| UNION_s3 | 0.80758663 | 0.00740917 | 0.81499581 | 2.99e-08 | 4.48e-12 | 341 / 379 |

objective는2598행 전체의 CE 평균에 `0.5×1e−4×||w94||²`를 더한다. 모든 fit에서 후보 없는1행도 원래 분모에 남는다. bias는 순위에서 상쇄되어0으로 고정했다. 명시적 강볼록 계수λ=1e−4에 대해 `J(w)−J* ≤ ||∇J(w)||²/(2λ)`이며, 네 결과 모두 사전 기준1e−6보다 작다. optimizer success와1000 iterations/2000 objective closure 상한도 함께 확인했다.

| 원래 AdamW weight → 대응 수렴 모델 | 동일 입력의 old CE | new CE | CE 변화 | 명시적 ridge objective 변화 |
|---|---:|---:|---:|---:|
| R0_ONLY_s1 → R0_ONLY | 0.20770543 | 0.13250970 | -0.07519573 | -0.07054789 |
| R0_ONLY_s2 → R0_ONLY | 0.20884346 | 0.13250970 | -0.07633377 | -0.07170041 |
| R0_ONLY_s3 → R0_ONLY | 0.21381721 | 0.13250970 | -0.08130751 | -0.07665957 |
| UNION_s1 → UNION_s1 | 0.85433663 | 0.79701407 | -0.05732256 | -0.05040241 |
| UNION_s2 → UNION_s2 | 0.84832597 | 0.78539713 | -0.06292884 | -0.05557605 |
| UNION_s3 → UNION_s3 | 0.87282542 | 0.80758663 | -0.06523878 | -0.05788113 |

저장 certificate와 독립 계산의 최대 objective 차이는2.22e-16, gradient norm 차이는1.44e-16다. 모든 trace call/iteration 번호가 연속이고 최종 weight hash가 checkpoint와 일치한다.

이 비교에서 old weight도 정확히 같은 float32 정규화 입력을 float64로 올린 함수에 적용했다. 원래 실사 예측을 수정하거나 다른 seed를 선택하지 않았다. R0_ONLY는 하나의 결정론적 공통 control이고, UNION의 세 경우는 기존 동결 refiner3개를 각각 유지한다.

이전6개 모델의 CE가 더 낮아질 수 있었음은 확인된다. 따라서 원래 실패를 Linear94의 표현력 한계만으로 설명해서는 안 된다. 다만 이번 통제는 optimizer·precision·초기값과 명시적 ridge를 바꾸므로 순수하게 optimizer 하나만의 인과 효과를 식별한 실험도 아니다. 기존 AdamW의 decoupled weight decay와 새로운 ridge objective를 같은 목적함수라고 부르면 안 된다.

**실질적인 T/R 판단은 별도 source VAL과, 그 통과 이후에만 허용되는 기존 실사 gate에서 해야 한다.** 수렴 인증이나 TRAIN CE 감소가 중앙값·tail·recording 안정성 개선을 대신하지 않는다.
