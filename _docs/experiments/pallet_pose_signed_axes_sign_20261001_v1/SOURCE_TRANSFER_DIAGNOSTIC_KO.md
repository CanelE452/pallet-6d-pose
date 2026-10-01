# 부호 항 추가 후의 고정 source 선택 진단

**네 모델의 수렴 검산은 PASS이나 source 조건은43/45이며 전체 FAIL이다.** 직전 Newton을 비교 control로 고정했다. 이미 잠긴 선택·두 축 예측·기존 물리 오류만 요약했다. 새 fit/forward/argmin/threshold/GT 재채점/실사 routing은0이다. 진단 PASS는 방법의 성공이 아니다.

UNION_s3 T 중앙값은 1.683872412413cm이며 R0_GEO 및 R0_ONLY의 1.674594148930cm보다 크다. 실패는 두 T strict-median 조건이고 이전 Newton과 같은 범주다. 작은 차이에 tolerance를 새로 넣거나 seed1·2만 채택하지 않는다. 이번 모델의 실사 성능은 평가하지 않았다.

## 고정 Newton 대비 실제 선택

각 모델1,024행 전체이며 실패0행이다. anchor는 운영 R0 GEO와 후보 정체성이 같음, safe 개선은 다른 후보에서 두 실제 오차가 비증가하고 적어도 하나가 엄격 감소함이다. 정확한 부등호를 쓰며 근사 tolerance를 적용하지 않는다.

| 모델 | anchor 이전→현재 | safe 개선 이전→현재 | unsafe 이전→현재 | 바뀐 후보 |
|---|---:|---:|---:|---:|
| R0_ONLY | 1023 → 1021 | 1 → 3 | 0 → 0 | 2 |
| UNION_s1 | 922 → 872 | 30 → 41 | 72 → 111 | 50 |
| UNION_s2 | 919 → 868 | 33 → 47 | 72 → 109 | 53 |
| UNION_s3 | 925 → 886 | 26 → 33 | 73 → 105 | 43 |

## 실제 선택의 위험과 부호

선택 score는 두 예측의 max이고 anchor 예측은0이다. 선택한 두 예측이0 이하라는 사실은 실제 두 축 비악화를 보장하지 않는다. 아래는 실제 선택에서만 집계한 위반이다.

| 모델 | 선택 nonanchor | T 악화 | R 악화 | 양축 악화 | unsafe 위험 P90 |
|---|---:|---:|---:|---:|---:|
| R0_ONLY | 3 | 0 | 0 | 0 | 해당 없음 |
| UNION_s1 | 152 | 87 | 65 | 41 | 1.329715 |
| UNION_s2 | 156 | 85 | 63 | 39 | 1.041560 |
| UNION_s3 | 138 | 79 | 62 | 36 | 1.339080 |

위험은 `max((T−T_anchor)/sT,(R−R_anchor)/sR,0)`이다. JSON의 MAE/RMSE는 signed-log1p 단위이며 cm/degree가 아니다. anchor0을 포함한 정확도를 학습된 분류 능력으로 해석하지 않도록 selected/nonanchor/known-pool 통계를 나눴다. 이전과 현재의 selected 집단은 서로 달라 직접 인과 비교 대상이 아니며, known nonanchor는 같은 부분후보 집합으로 비교한다.

| 모델 | 같은 known nonanchor 쌍 | T sign 이전→현재 | R sign 이전→현재 |
|---|---:|---:|---:|
| R0_ONLY | 10 | 80.00% → 90.00% | 10.00% → 30.00% |
| UNION_s1 | 564 | 32.62% → 34.75% | 44.33% → 44.68% |
| UNION_s2 | 582 | 32.82% → 33.51% | 46.22% → 44.67% |
| UNION_s3 | 580 | 35.52% → 35.17% | 42.93% → 43.45% |

## 이미 확인된 개선 기회와 미포착 하한

현재 sign·직전 Newton·더 오래된 RBF의 실제 선택, 기존 oracle CSV에 남은 learned/oracle 후보의 이미 채점된 오류만 합쳤다. RBF는 부분 캐시 보충 자료이며 이번 비교 control은 Newton이다. source poses/metadata/features/prediction lock 및 기존 oracle protocol의 feature-lock SHA를 대조했다. 전체 후보의 참조 오류를 새로 계산하거나 새로운 oracle/argmin을 만들지 않았다.

아래 기회와 miss는 이 부분 캐시에서 확인된 하한이다. 전체 후보 recall/false-negative가 아니다. 이전·현재 모두 **같은 합쳐진 known pool**로 계산하므로, 앞선 진단의 더 작은 캐시에서 발표한 하한과 직접 비교하지 않는다. 다른 safe 후보를 선택해도 capture다.

| 모델 | 확인된 기회 | safe capture 이전→현재 | miss 하한 이전→현재 | 현재 miss 중 anchor | 알려진 nonanchor/가능 수 |
|---|---:|---:|---:|---:|---:|
| R0_ONLY | 10 | 1 → 3 | 9 → 7 | 7 | 10/1024 |
| UNION_s1 | 226 | 30 → 41 | 196 → 185 | 182 | 564/3072 |
| UNION_s2 | 238 | 33 → 47 | 205 → 191 | 189 | 582/3072 |
| UNION_s3 | 219 | 26 → 33 | 193 → 186 | 185 | 580/3072 |

## TRAIN과 source를 구분한 관측

TRAIN은 기존 독립 검산의2,598행(유효2,597·실패1) 값을 인용한다. 새 목적식은 이전 Newton의 고정 weight에서도 채점된 동일 목적식과 비교해 감소했으나, Huber-only와 Huber+sign의 서로 다른 원시 loss값을 직접 비교하지 않는다. 수렴과 목적식 감소가 source45 통과를 보장하지 않는다.

| 모델 | TRAIN safe 이전→현재 | TRAIN unsafe 이전→현재 | source safe 이전→현재 | source unsafe 이전→현재 |
|---|---:|---:|---:|---:|
| R0_ONLY | 4 → 8 | 0 → 0 | 1 → 3 | 0 → 0 |
| UNION_s1 | 135 → 211 | 171 → 282 | 30 → 41 | 72 → 111 |
| UNION_s2 | 120 → 191 | 183 → 277 | 33 → 47 | 72 → 109 |
| UNION_s3 | 106 → 144 | 171 → 234 | 26 → 33 | 73 → 105 |

이번 고정 계수1의 부호 항 추가는 최종 source gate 실패를 해소하지 못했다. 세부 sign/선택 변화는 관측 결과이며, 표현력·후보 정보·감독 목표·합성 지원범위 중 무엇이 유일 원인인지는 확정하지 않는다. 이 자료는 반복 사용된 source VAL 개발 집합이며 새로운 독립 검증이나 실사 개선의 증거가 아니다.

이 진단은 다음 loss·weight·threshold·seed를 제안하거나 시험하지 않는다. source45의 어느 조건이라도 실패했으므로 이번 모델의 실사 routing은 금지된 상태다.

[진단 JSON](SOURCE_TRANSFER_DIAGNOSTIC.json) · [source 판정](SOURCE_VAL_GATE.json) · [독립 source 검산](SOURCE_VAL_VERIFICATION_KO.md) · [TRAIN 검산](TRAIN_CONVERGENCE_KO.md) · [직전 Newton 진단](../pallet_pose_signed_axes_newton_20261001_v1/SOURCE_TRANSFER_DIAGNOSTIC_KO.md)
