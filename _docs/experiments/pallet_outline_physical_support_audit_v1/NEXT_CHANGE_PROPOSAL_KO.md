# 다음 변경안 한 가지: 전역 외곽선 부모 구간의 지지·출처 보존

**whole-outline 관측 패킷에 각 원 부모 선분의 생성 출처와 물리 지지 근거를 보존하는 계약을 추가하는 안을 우선 제안한다.** 독립적인 다른 물체 지지가 실제로 제공되는지가 선행 조건이다. 이번에는 이 변경을 실행하거나 관측·가중치·선택을 바꾸지 않았다. 새 성능도 산출하지 않았다.

이번6장 검토에서는 cone이 포함된 원 마스크와 그 convex 연결이 실제 비용 표본으로 전달된 사례3장이 확인됐다. 같은 부모 구간이 `OTHER_OR_CONSTRUCTED_BOUNDARY`로 분류된 관측→모델 비용차는 다음과 같다. 양수는 실패4장에서는 실제 실패 선택을 지지한다. 모델 육안 해석이며 사람 검수·픽셀 정답·학습 레이블이 아니다.

| 사례 | 해당 부모 / 전체 부모 | 해당 표본 / 전체 표본 | 해당 구간의 기존 signed Δ |
|---|---:|---:|---:|
| plastic_night_01:038630 | 6/12 | 328/403 | +0.425418393 |
| wood_day_01:005286 | 7/12 | 256/281 | +2.735624622 |
| plastic_day_01:020955 | 10/18 | 336/363 | +4.563473466 |

원 mask→hull→표본 변환은 재현됐다. 따라서 위 세 사례는 변환 구현의 불일치와 구분되는 **원 마스크의 비팔레트 포함 및 그로부터 파생된 연결 구간**이다. convex 연결이라는 이유만으로 그 안의 모든 픽셀이나 모든 가상 외곽을 틀렸다고 판정하지 않는다. 어두운 목재2장은 별도 미확정으로 남긴다.

제안하는 한 계약은 원 mask/RGB SHA, 부모 hull edge와 유한 선분, sample index, 직접 관측 지지인지 hull 연결인지, 독립 non-target 지지의 출처·적용 범위, 미확정 상태를 같은 부모 구간에 연결하는 것이다. 다른 물체 지지가 없으면 `UNRESOLVED`를 유지한다. 감사용 판정을 정답으로 사용하거나 cone 뒤의 보이지 않는 팔레트 선을 새 측정으로 만들지 않는다. 어떤 근거를 후속 추정기에서 어떻게 사용할지는 별도 변경·검증 대상이며 여기서 새 규칙을 채택하지 않는다.

## 기존 처리와의 차이 및 필요한 입력

가장 가까운 기존 구현은 [edge_visibility_v4_28.py:89](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/edge_visibility_v4_28.py#L89)의 `EdgeVisibilityContext`다. 별도로 출처·같은 RGB·canvas·pixel SHA가 결합된 `NON_TARGET_OCCLUDER` mask를 요구하며, [`_occluder_blocked`:161](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/edge_visibility_v4_28.py#L161)와 [`query_allowed`:172](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/edge_visibility_v4_28.py#L172)는 국소 RGB/target-mask 샘플링 전에 지지 여부를 처리한다. [`classify_expected_edge`:196](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/edge_visibility_v4_28.py#L196)은 실제 chain 지지·독립 occluder가 지지한 gap·미확인을 구분한다. 관측 유효성과 출처를 구분하는 이 원리는 이미 저장소에 있다.

반면 원 전역 관측 생성 함수는 [G10 `observe_global_outline`:75](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_global_outline_solver_v4_10.py#L75), [G13의 같은 함수:85](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_partial_BI_outline_solver_v4_13.py#L85)에서 global target mask만 받는다. native corner·mate·B/I anchor·실제 chain을 전제로 하는 국소 함수를 whole-outline에 그대로 적용할 수는 없다. 전체 선분의 생성·지지 출처를 보존하는 입력 계약이 먼저 필요하다. 이를 새로운 발명으로 부르지 않는다.

**핵심 입력 공백은 이번 여섯 장의 전역 부모 구간에 연결된 독립 non-target 지지 패킷이 확인되지 않았다는 점이다.** 확인한 기존 실험 호출도 [v4.28:189–190](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/run_joint_geometry_guided_line_cases_v4_28.py#L189)과 [v4.32:103](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/run_initial_only_observed_ray_CAL_v4_32.py#L103)에서 `independent_occluder_mask=None`이다. 별도 코드나 class-agnostic mask가 있다는 사실이 해당 구간의 다른 물체 인증을 보장하지 않는다. target mask의0·보완값·예측 O 상태를 그 입력 대신 사용할 수 없다. [기존 `_occluder_evidence`:105](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/boundary_evidence.py#L105)도 독립 출처가 없는 target 음성만으로 가림을 확인하지 않는다.

따라서 우선 확인할 것은 **같은 RGB·원좌표에 결합된 독립 foreign-instance 지지가 실제로 가용한가**이다. 없다면 필요한 증거와 획득 범위를 정한 뒤 별도로 승인받아야 한다. 새 학습이 반드시 필요하다는 근거는 없고, 이번에는 학습·분할기 실행·새 관측 생성 모두0회다.

## 지켜야 할 반례와 미확정

성공 반례 `plastic_day_01:014704`는22부모·417표본 모두 보이는 팔레트 envelope 지지가 있으며 관측→모델 Δ+6.872412890가 올바른 선택을 보호했다. `wood_night_01:030898`도 outline 전체 Δ+0.774584020가 성공을 보호했지만, 그중 어두운7부모·201표본의 물리 지지는 미확정이다. **미확정은 오류 판정이 아니다.** 이 두 반례에 실패 사례와 같은 지지·출처 기준을 적용해야 한다.

전체25부모·686표본의 미확정과 여섯 장 모두의 K/축/기하 참조 공백은 [REVIEW_REQUIRED](REVIEW_REQUIRED.md)에 원좌표·부모 ID로 보존했다. 이번6장은 이전 결과로 고른 설명 사례여서 전체32/114의 물리 판정이나 새로운 독립 검증이 아니다. 기존192장의 고정후보 진단에서도 G13 outline 제외는9회복·11손상이었다. 이 제안에서 항 삭제·점수 재선택·가중치 탐색·개선 성능을 주장하지 않는다.

후보부재46장과 `wood_night_01:033421`의 G10→G13 전달 공백은 그대로 남는다. N3·SubPix·기존 e·상태별 처리·운영 기본값·논문·기존 결과를 바꾸지 않았다. 이 단일 변경안의 효과는 **미검증**이다.

[원 구간 판정](SUPPORT_REVIEW.jsonl) · [기존 비용 연결](COST_SUPPORT_SUMMARY.csv) · [사례별 A/B/C](CASE_ASSESSMENTS.json) · [등록·참조 근거](REGISTRATION_PROVENANCE_KO.md) · [실제 검토 화면](gallery/index.html)
