# Anchor context189 학습의 독립 수렴·TRAIN 진단

**독립 검산 PASS**. 수렴 인증은 실제 T/R 개선 판정을 대신하지 않는다.

정답은 기존 독립 scalar-loop 구현으로 복원하고 target/safe/anchor/원래 valid hash를 START 및 이전 source feasibility와 대조했다. 94차원 float32 정규화→float64 승격→절댓값 anchor 차이94·identity1 연결은 새 독립 행별 loop로 계산했다. PyTorch float64 CE/autograd로 목적함수·gradient 벡터·수렴 상한을 검산했다. 후보 없는 1행도 전체 2,598행 분모에 남으며, CE 및 runtime에는 원래 valid 후보 전체가 참여한다.

| 모델 | 새 CE | 새 objective | gradient L2 | gap 상한 | 호출 수 |
|---|---:|---:|---:|---:|---:|
| R0_ONLY | 0.094383614 | 0.100079963 | 3.06e-08 | 4.68e-12 | 253 |
| UNION_s1 | 0.627688976 | 0.635706119 | 3.17e-08 | 5.01e-12 | 408 |
| UNION_s2 | 0.620493696 | 0.627836859 | 3.83e-08 | 7.35e-12 | 401 |
| UNION_s3 | 0.629390916 | 0.637801101 | 3.91e-08 | 7.65e-12 | 433 |

바로 이전 pareto_anchor의94차원 weight 뒤에95개의0을 붙여 같은 target·context189에서 비교했다. 원래94 score/objective와의 동등성도 검증했다. 추가95축에서의 gradient는 기존94 수렴 인증과 다른 공간이므로 동일시하지 않는다. 아래 정확도와 위반 비율의 분모는 전체 TRAIN 2,598행이다. 실패1행은 target 정답으로 세지 않으며, 위반 여부를 평가할 수 없는 실패로 별도 보고한다. 유효 anchor 2,597행 기준의 조건부 비율도 JSON에 함께 제공한다.

| 모델 | target 정확도 이전→현재 | T 위반 이전→현재 | R 위반 이전→현재 | 한 축 이상 위반 이전→현재 |
|---|---:|---:|---:|---:|
| R0_ONLY | 95.227% → 95.574% | 0.924% → 0.462% | 0.770% → 0.462% | 1.116% → 0.577% |
| UNION_s1 | 67.052% → 72.363% | 9.777% → 1.963% | 7.429% → 1.039% | 13.010% → 2.540% |
| UNION_s2 | 67.667% → 72.941% | 9.353% → 1.655% | 7.891% → 1.347% | 12.741% → 2.309% |
| UNION_s3 | 66.705% → 71.824% | 9.931% → 1.771% | 8.391% → 1.270% | 13.626% → 2.348% |

위반은 실제 unrestricted scorer가 선택한 후보의 T/R가 같은 프레임 R0 anchor를 초과하는지로 집계했다. 학습 target의 safe mask를 inference에 적용하지 않았다. 따라서 oracle의 pointwise 보존은 학습된 scorer에 자동으로 전달되는 보장이 아니다.

이 비교는 TRAIN 설명용이며 추가 fit·optimizer step·VAL 품질·실사 GT 읽기·선택 정책 변경은 없었다. 실질 성능 판정은 사전 고정 source VAL과 그 통과 후에만 허용되는 실사 평가로 분리한다.

[검산 JSON](TRAIN_CONVERGENCE.json), [재사용 source target 진단](../pallet_pose_pareto_anchor_20261001_v1/SOURCE_FEASIBILITY.json), [학습 계약](TRAIN_PROTOCOL.json)
