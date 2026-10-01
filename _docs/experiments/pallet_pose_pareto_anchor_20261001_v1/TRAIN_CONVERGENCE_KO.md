# Anchor target 학습의 독립 수렴·TRAIN 진단

**독립 검산 PASS**. 수렴 인증은 실제 T/R 개선 판정을 대신하지 않는다.

정답은 별도 scalar-loop 구현으로 복원하고 target/safe/anchor/원래 valid hash를 START 및 source feasibility와 대조했다. PyTorch float64 CE/autograd로 목적함수와 gradient를 재계산했다. 후보 없는 1행도 전체 2,598행 분모에 남으며, CE 및 runtime에는 원래 valid 후보 전체가 참여한다.

| 모델 | 새 CE | 새 objective | gradient L2 | gap 상한 | 호출 수 |
|---|---:|---:|---:|---:|---:|
| R0_ONLY | 0.109204504 | 0.113997170 | 3.37e-08 | 5.67e-12 | 171 |
| UNION_s1 | 0.703295569 | 0.710768652 | 1.71e-08 | 1.46e-12 | 320 |
| UNION_s2 | 0.696586075 | 0.705869718 | 3.16e-08 | 5e-12 | 320 |
| UNION_s3 | 0.718397407 | 0.727765143 | 2.23e-08 | 2.49e-12 | 332 |

이전 수렴 CE weight도 동일한 새 target에 적용해 비교했다. 아래 비율의 분모는 anchor가 유효한 TRAIN 2,597행이며, 실패 1행은 별도 집계하고 full-frame T/R에는 +inf로 유지했다.

| 모델 | target 정확도 이전→현재 | T 위반 이전→현재 | R 위반 이전→현재 | 한 축 이상 위반 이전→현재 |
|---|---:|---:|---:|---:|
| R0_ONLY | 95.110% → 95.264% | 1.271% → 0.924% | 1.001% → 0.770% | 1.463% → 1.117% |
| UNION_s1 | 64.613% → 67.077% | 13.477% → 9.781% | 11.090% → 7.432% | 18.752% → 13.015% |
| UNION_s2 | 64.035% → 67.693% | 14.286% → 9.357% | 12.707% → 7.894% | 20.216% → 12.745% |
| UNION_s3 | 64.382% → 66.731% | 14.363% → 9.935% | 11.706% → 8.394% | 19.754% → 13.631% |

위반은 실제 unrestricted scorer가 선택한 후보의 T/R가 같은 프레임 R0 anchor를 초과하는지로 집계했다. 학습 target의 safe mask를 inference에 적용하지 않았다. 따라서 oracle의 pointwise 보존은 학습된 scorer에 자동으로 전달되는 보장이 아니다.

이 비교는 TRAIN 설명용이며 추가 fit·optimizer step·VAL 품질·실사 GT 읽기·선택 정책 변경은 없었다. 실질 성능 판정은 사전 고정 source VAL과 그 통과 후에만 허용되는 실사 평가로 분리한다.

[검산 JSON](TRAIN_CONVERGENCE.json), [source target 진단](SOURCE_FEASIBILITY.json), [학습 계약](TRAIN_PROTOCOL.json)
