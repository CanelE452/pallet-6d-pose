# O 위치 원 hard-bound 유보 대조군 v4.19

O 상태와 원래 raw-N3 원을 감사 기록에 보존하고, 해당 원의 강제 위치 제한만 기존 closed-canvas box로 바꿨습니다. 초기 S·U 범위·T·B/I 좌표와 Σ/신뢰도·실제 NMS 선·CAL 잡음·알려진 크기/Q·후보집합을 그대로 사용했습니다. 원래 추가 halfplane도 보존합니다. O 정확점 비용이나 soft loss는 만들지 않았습니다.

원래 O 원은 실제 occluder mask의 관측 지지가 아닌 CAL7 오차 기반 위치 가설입니다. 이 실험은 그 위치 제약의 인과 대조군입니다. O/T/S/U 방법 전체의 검증이나 새 기본 방법 채택으로 해석하지 않습니다.

전체238장(CAL46+TEST49+추가143, severe0)을 같은 규칙으로 예측·봉인한 뒤 같은 참조로 채점했습니다. 연속 탐색은 각 W/D에서 동일한 최대6개 native-diverse 후보와 자체 coarse 초기화 정책을 사용합니다. feasible set이 넓어져 실제 선택된6개 초기화 부분집합은 달라질 수 있습니다.

## 기존 TEST49

| 방법 | 위치 평균 cm | 위치 최대 cm | 위치≤5cm | 위치·회전 5cm/5° |
|---|---:|---:|---:|---:|
| 보존된 기본 e | 1.2994 | 4.3389 | 49 | 49 |
| native 기존18c | 3.3331 | 40.3581 | 42 | 41 |
| native O 위치 유보 | 5.3922 | 52.8434 | 40 | 39 |
| native+고정 mask 기존18c | 3.0658 | 40.3979 | 43 | 43 |
| native+고정 mask O 위치 유보 | 3.0658 | 40.3979 | 43 | 43 |

## 추가143

| 방법 | 위치 평균 cm | 위치 최대 cm | 위치≤5cm | 위치·회전 5cm/5° |
|---|---:|---:|---:|---:|
| 보존된 기본 e | 18.5025 | 260.4117 | 66 | 65 |
| native 기존18c | 16.6091 | 182.4996 | 68 | 68 |
| native O 위치 유보 | 20.9332 | 576.8840 | 68 | 67 |
| native+고정 mask 기존18c | 14.4633 | 189.2627 | 72 | 72 |
| native+고정 mask O 위치 유보 | 15.3491 | 189.2627 | 72 | 72 |

O가 없는159장에서는 두 선택 대조군 모두 이전18c의 R/t/Q·W/D 가족·accepted 상태가 정확히 같았습니다. 기존에 feasible이던 저장 후보60,542개 token이 모두 유지됐고 새로 feasible인 token을 포함해62,135개가 됐습니다. 이는 후보 표현 token 수이며 독립적인 정보량이나 전역 해의 개수가 아닙니다.

위치 원을 제거해도 추가143장 전부5cm 목표는 해결되지 않았습니다. native-only의 최악은032645에서576.884cm로 커졌습니다. 기본 e와 이전 결과는 그대로 보존하며 이 대조군은 채택하지 않습니다. 미충족 목표와 참조의 한계도 유지합니다.

원본49장과 추가143장은 이미 반복적으로 살펴본 exploratory 자료입니다. 추가143 참조는 annotation-derived pose이고 물리 축/등록의 독립 측정 보장은 없습니다. CAL46은 calibration 집단이며 holdout 성능으로 부르지 않습니다. 각 방법을 모든 집단에 동일 적용했고 정답·ID·집단에 따라 방법을 골라 배포하지 않았습니다.

- [프로토콜](PROTOCOL.json) · [입력 봉인](INPUT_SEAL.json) · [전체 예측 봉인](PREDICTION_SEAL.json) · [점수 봉인](SCORE_SEAL.json)
- [기존49 점수](TEST_SCORE.json) · [추가143 점수](REMAINDER_SCORE.json)
- [소유자 계약·paired receipt](../o_localization_abstention_v4_19_checks/COMPLETION_AND_PAIRED_RECEIPT_V1.json)
- [실제 CAL smoke](../o_localization_abstention_v4_19_checks/SMOKE_RECEIPT_V1.json) · [독립 source/실제133-domain 검토](../independent_O_localization_abstention_review_v4_19/PEER_REVIEW_V1.json)
