# 고정 TRAIN 선택의 anchor 위험 진단

현재189 모델의 기존 TRAIN 선택만 다시 재현했다. 모든 선택 index SHA는 독립 TRAIN 검산의 기존 값과 일치한다. 새로운 후보 선택 규칙, 임계값, loss 또는 fit을 실행하지 않았다. VAL 배열·실사 GT는 읽지 않았다.

전체2,598행 중 유효 anchor는2,597행이다. 실패1행은 별도 유지하며 위험을0이나 안전한 선택으로 간주하지 않는다. JSON에는 전체·유효행 두 분모의 비율을 모두 기록했다. 아래 표는 행 수다. `safe improvement`는 두 축 모두 anchor 이하이면서 적어도 한 축이 엄격히 작은 non-anchor pose다. 두 오차가 정확히 같은 non-anchor는 별도 `safe equal`이다.

| 모델 | target anchor | target non-anchor safe | 선택 anchor | 선택 safe improvement | 선택 safe equal | 선택 unsafe | 실패 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0_ONLY | 2454 | 143 | 2538 | 44 | 0 | 15 | 1 |
| UNION_s1 | 1812 | 785 | 2400 | 131 | 0 | 66 | 1 |
| UNION_s2 | 1817 | 780 | 2401 | 136 | 0 | 60 | 1 |
| UNION_s3 | 1835 | 762 | 2450 | 86 | 0 | 61 | 1 |

위험은 같은 frame의 R0 operational GEO에 대한 정규화 초과량 `r=max((T-Ta)/sT,(R-Ra)/sR,0)`이다. 기존 TRAIN 고정 scale을 그대로 사용한다. 다음 분위수의 분모는 실제 해당 위반을 선택한 행만이며 실패1행은 정의되지 않는다. T·R 위반 집합은 두 축 모두 위반한 경우 겹친다.

| 모델 | 위반 | 행 수 | r 중앙값 | r P90 | r 최대 |
|---|---|---:|---:|---:|---:|
| R0_ONLY | T | 12 | 73.165702 | 78.516871 | 79.568211 |
| R0_ONLY | R | 12 | 73.165702 | 78.516871 | 79.568211 |
| R0_ONLY | either | 15 | 69.333657 | 78.484400 | 79.568211 |
| UNION_s1 | T | 51 | 1.648411 | 7.028250 | 64.958666 |
| UNION_s1 | R | 27 | 1.491709 | 5.197864 | 64.958666 |
| UNION_s1 | either | 66 | 1.218936 | 5.941837 | 64.958666 |
| UNION_s2 | T | 43 | 2.373536 | 9.817681 | 79.507309 |
| UNION_s2 | R | 35 | 0.882688 | 76.468881 | 79.507309 |
| UNION_s2 | either | 60 | 1.097772 | 9.786487 | 79.507309 |
| UNION_s3 | T | 46 | 2.183294 | 9.201146 | 138.757578 |
| UNION_s3 | R | 33 | 1.646126 | 48.572971 | 138.757578 |
| UNION_s3 | either | 61 | 1.570571 | 9.186269 | 138.757578 |

현재 CE는 target y가 고정되면 `logsumexp(-score)+score_y`다. 같은 target·valid·score라면 오선택의 물리 초과량이 달라도 loss와 gradient가 같다. score 확률이나 입력 특징을 통한 간접 구분은 가능하므로, 이 사실만으로 현재 표현력이 부족하거나 CE가 위험을 전혀 학습할 수 없다고 결론내리지 않는다.

## 다음 개입 후보 하나: TRAIN 위험으로 경쟁 logit을 보정하는 CE

실행하지 않은 후보로, 고정 target·189특징·후보·scale은 유지하고 TRAIN에서 각 후보의 anchor 위험 r_c를 구해 `m_c=log1p(r_c)`를 정한다. 손실만 `logsumexp(-score_c+m_c)+score_y+(lambda/2)||w||²`로 바꾼다. anchored target은 r_y=0이므로 target 항에는 추가 보정이 없다. 이 식은 unsafe 경쟁 후보의 softmax 질량에1+r_c를 곱하며, 안전한 경쟁 후보에는 기존 CE와 같은0 margin을 적용한다.

이는 TRAIN label에서 정한 상수 margin을 쓰는 log-sum-exp이므로 고정 선형189 weight에 대해 convex이고, 기존 모든 weight의 L2를 유지하면 lambda-strongly-convex 수렴 인증을 그대로 적용할 수 있다. 원시 r를 직접 margin으로 쓰면 지수적으로 큰 경쟁 가중이 생기므로, 제안은 추가 scale이나 sweep 없이 `log1p(r)` 한 형태만 지정한다. 현재 수치로 가중치·절단값을 최적화한 것은 아니다. 런타임은 기존 input-only score와 전체 valid 후보의 argmin을 유지하며 GT 위험·margin을 입력하지 않는다.

이 개입은 단순 row별 class weighting이나 local4-edge pairwise가 아니라, 현재 whole-pose target 하나와 모든 경쟁 후보의 physical anchor 초과량을 같은 listwise loss에 넣는다. 따라서 모든 오선택을 같은 비용으로 다루는 현재 감독과 차이가 명확하다. 그러나 위험 가중의 감소가 실제 두 중앙값·P90을 함께 개선한다는 보장은 없고, anchor 쪽으로 과도하게 보수적으로 이동해 strict gain을 잃을 수도 있다.

## 선행과 중단 조건

비용·보존 감독 자체는 새 개념이 아니다. [기존 목적함수 감사](../pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md)에 DHT의2D soft-cost CE/회귀, whole-layout gain 회귀, RGB utility 회귀, 좌표 헤드의 baseline-relative squared regret 실패가 정리돼 있다. [고정4-edge pairwise 실험](../pallet_pose_selector_pairwise_20261001_v1/REPORT_KO.md)도 수렴 후 전체 pose 선택과 위험이 악화했다. 이들은 현재 physical T/R anchor 위험을 TRAIN-only logit margin으로 쓰는 global189 CE와 같지 않지만, 위험/비용 정보를 추가하면 전이가 해결된다는 주장에 대한 반례다. 저장소 제한 검색에서 이 정확한 조합의 실행은 찾지 못했으며 학술적 신규성 주장은 하지 않는다.

다음 실제 실험을 택하더라도 단일 고정 loss, 같은4모델·zero init·L2·solver 예산·모든2,598행을 유지한다. 실패한 수렴을 연장하거나 좋은 seed만 선택하지 않는다. 기존 source45개 조건을 전부 통과하기 전 실사 routing은 금지하며, 통과 후에도 원래 실사5개 안정성 조건과 모든 seed 조건을 그대로 적용한다. 이 문서는 제안과 TRAIN 진단이며 새 실험 protocol이나 T/R 개선 결과가 아니다.

[전체 수치와 해시](TRAIN_RISK_DIAGNOSTIC.json) · [기존 독립 TRAIN 검산](TRAIN_CONVERGENCE_KO.md)
