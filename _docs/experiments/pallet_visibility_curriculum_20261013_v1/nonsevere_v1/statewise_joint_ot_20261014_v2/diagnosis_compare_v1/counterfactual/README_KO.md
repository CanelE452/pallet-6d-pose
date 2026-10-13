# 봉인 입력의 통제된 사후 원인 진단

이 결과는 이미 본 CAL46/TEST49에서의 posthoc intervention이며 새 production 방법이나 일반화 성능 비교가 아니다. 기존 실제3조건은 그대로 보존했다. 새11조건과 cached2selector에 대해 source/input/output을 바인딩·봉인하고, 그 뒤 동일 physical pose metric을 집계했다. Oracle에는 human state 또는 역사적 derived reference가 입력된다는 점을 명시한다. 원RGB·마스크·head를 다시 추론하지 않았다.

| 진단 조건, TEST49 | 위치 cm | 회전 ° | 성공 /49 | 공동 채택 | 모호성 |
|---|---:|---:|---:|---:|---:|
| ORIGINAL_PNP | 3.3630 | 6.5271 | 40 | 0 | 0 |
| CURRENT_CORRECTED_BI_NO_OT | 4.6924 | 8.2853 | 38 | 30 | 3 |
| CURRENT_CORRECTED_BI_OT | 3.5139 | 6.5675 | 39 | 29 | 0 |
| RAW_BI_OT | 3.7974 | 6.7026 | 38 | 28 | 0 |
| RAW_BI_NO_OT | 4.6234 | 6.4277 | 38 | 29 | 2 |
| CURRENT_RAW_STANDARD_COV | 3.7897 | 6.7067 | 39 | 29 | 2 |
| ORACLE_BINARY_RAW_STANDARD_COV | 3.8539 | 6.5934 | 38 | 26 | 1 |
| FULL_HUMAN_RAW_STANDARD_COV | 3.8380 | 6.5910 | 38 | 27 | 0 |
| CURRENT_DROP_KNOWN_FALSE_OT | 3.5371 | 6.7096 | 38 | 30 | 0 |
| CURRENT_BI_BUDGET6 | 3.5302 | 6.7616 | 37 | 44 | 0 |
| CURRENT_REFERENCE_MODEL_CENTER_O | 3.6048 | 6.7044 | 38 | 30 | 0 |
| CURRENT_REFERENCE_INIT | 3.5139 | 6.5675 | 39 | 29 | 0 |
| MODEL_IDEAL_PROJECTIONS | 0.0000 | 0.0000 | 49 | 49 | 0 |
| FULL_HUMAN_RAW_STANDARD_COV_BUDGET6 | 3.7069 | 6.5832 | 38 | 39 | 0 |
| SELECT_CLOSEST_COARSE_FIXED | 3.3256 | 6.6467 | 39 | 29 | 0 |
| ORACLE_BEST_FOUND | 3.0920 | 6.9870 | 39 | 29 | 0 |

RAW_BI_OT/NO_OT는 q만 바꾸고 기존 corrected covariance를 그대로 둔 인위적 xy 통제다. CURRENT_RAW_STANDARD_COV는 별도 B2px/I3px control이며 binary/full-human oracle은 같은 covariance를 쓴다. FULL_HUMAN_BUDGET6는 compatible86에서만 budget을6으로 바꾸고 비호환9는 정확히 standard3 control을 유지한다. 가중 objective/Sigma와 hard budget은 다른 조작이다.

Native reference 대응이 호환되는 CAL44/TEST42=86을 pixel oracle 해석 범위로 제한했다. 비호환9에서 state/center/drop oracle은 control을 유지한다. MODEL_IDEAL9는 physical phase0에서 현재 native 모델을 자체 투영한 virtual positive control이며 human E를 적용하지 않는다. 이 모델 투영을 실제 영상 관측이나 정확한 physical visibility로 표현하지 않는다.

Reference-init은 objective/feasible set을 바꾸지 않지만 탐색과 BI seed fallback을 바꿀 수 있다. 각 phase의 BI Maha/geometry/canvas/depth와 개별O/T 제약, fixed native reference의 가능성과 any allowed physical-symmetry reference의 가능성을 분리했다. MODEL_IDEAL의 construction pose가 가능했는지·found samples에 남았는지·최종 선택이 모호한지는 별도 항목이다. 모호한 경우 모든 pose error가0이어야 한다고 요구하지 않는다.

SELECT_CLOSEST_COARSE_FIXED는 기존 feasible sample의 all8 모델 투영과 초기 모델 투영 사이 RMS/초기bboxdiag를 최소화한다. S raw나GT를 쓰지 않는다. ORACLE_BEST_FOUND는 기존 pool에서 translation error 최소 후보 하나를 선택하고 그 pose의 모든 metric을 보고한다. 각metric의 attainable floor는 별도로 제공하며 서로 다른 후보의 최소값을 가짜 한 pose로 합치지 않는다. 새 전역 search의 완전성이나 production selector 우수성은 증명하지 않는다.

POSTSEAL_SCORING.json에는 전체95 및 compatible 부분집합·짝비교·성공flip·session-cluster bootstrap가 있다. POSTSEAL_DIAGNOSTIC_ANALYSIS.json에는 세션별 기여, 큰 손실/개선, 상태/fallback/모호성, reference exclusion과 q보정×OT 상호작용이 있다. PREDICTIONS에는 모든 선언 입력, pose, 원인별 제약 감사와 provenance를 보존했고 source/real predictions/보호 사용자파일은 유지했다.
