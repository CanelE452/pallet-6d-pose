# 무엇을 고쳤고 무엇이 아직 해결되지 않았는가

“가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?”

**이번 알려진 평가245장(Clean153·Moderate92)의 고정 GEOMETRIC_PROXY에서는 개선을 확인하지 못했다. 기능 오류 두 개는 수정했지만 정확도 목표는 미해결이다.** Severe74는 제외했다. 과거319장 결과와 이번245장 결과를 구분한다.

| 단계 | 실제 문제·상태 | 증거 |
|---|---|---|
| 경계 대응 감독 | 이전 ray 수치 miss를 존재하지 않는 대응 NONE으로 처리한2,164 query는 잘못됐다. 실제 mesh·wire·depth 근거와 POSITIVE/NONE/IGNORE로 수정했다. 최초 감사 당시 재학습은 없었고 이후 수정 감독3조건×3,000 update는 완료된 별도 역사 실험이다. 이번 추가 학습0. | [최신 상태](LATEST_KO.md), [V18 전체 결과의 역사 감독 구분](../pallet_consensus_decision_repair_20261011_v18/RESULT_KO.md) |
| PnP 결정 | 같은 support의 다른 비용까지 거부한 추가 veto161 frame을 제거하고 원 count→전체 관측 truncated SSE 결정을 복원했다. 진짜 numerical tie1과 관측·기하 부족4는 유지했다. | [결정 검산](../pallet_consensus_decision_repair_20261011_v18/VALIDATION_CHECKS.json), [코드](../../../scripts/research/pallet_consensus_decision_repair_20261011_v18/solver.py) |
| 관측 선택·LOO | H∪{시험corner}를 뺀 관측으로 LOO를 수행한다. 작은 conditional LOO 잔차가 실제 대응의 정확성을 보증하지는 않는다. proxy에서 틀린 채택 q13개가 두 backend final inlier로 남았다. | [13 query literal 인덱스](REMAINING_THIRTEEN_QUERIES.json), [LOO 코드](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py) |
| 최종 fit·재투영 | NEW fit/score에서 초기 self-occ H XY를 제외하고, 확정된 R,t로 H를 한 번 재투영하여 초기 좌표를 대체한다. 재투영을 새 독립 관측처럼 다시 fit하지 않는다. H 불일치 자체가 frame 실패가 아니다. | [point 검사626,482 PASS](../pallet_consensus_decision_repair_20261011_v18/VALIDATION_CHECKS.json), [pipeline](../../../scripts/research/pallet_consensus_decision_repair_20261011_v18/pipeline.py) |
| 평가 정답·물리 ownership | 기존 point source는 unknown6/manual_click6/pnp_projected1이고13개 axis review는 None이다. 현재 proxy는 독립 물리 R,t나 actual incident edge 소속의 인증이 아니다. | [원주석 태그13개와 SHA](REMAINING_THIRTEEN_SOURCE_TAGS.json), [그 확인 코드](SOURCE_TAG_PROVENANCE_INDEX.py) |

13개 모두 기존 사람 상태는 DIRECT_VISIBLE이다. 첫 사례 `eval_cad:1778653029965134592`, native corner4는 저장 proxy 오차14.306876px, LOO2.957661px, 최종 fit1.409801px다. 이는 작은 잔차가 proxy 대응 정확성을 보증하지 않음을 보여준다. 그 점의 실제 physical ownership이나 다른 점이 물리적으로 틀렸다는 것을 이 수치만으로 확정하지 않는다. mask가 틀렸다는 이유로 frame을 버리는 것이 문제 해결책은 아니다.

[fallback5개 원행 인덱스](FALLBACK_FIVE_CASES.json)의 최종 저장 상태는 INSUFFICIENT_POINT_OBSERVATIONS1개, INSUFFICIENT_ACTUAL_INLIER_GEOMETRY3개, AMBIGUOUS_FINITE_CONSENSUS1개다. 이 frame 분류는 내부 root/optimizer의 수치 거절0을 주장하는 것이 아니다. 산출은79 NEW/166 fallback에서240 NEW/5 fallback으로 복원됐지만, 전체 운용245장의 위치 평균은 N3의9.754548cm→10.194734cm, 회전 평균은10.914842°→13.639333°로 악화됐다. 같은 hybrid q ordinary PnP의9.207154cm/10.613464°, native N3 ordinary PnP의8.951574cm/10.617639°보다도 나쁘다. hidden corner 재투영 오차가 작아 보이거나 산출률이 높아져도 자세 개선으로 판정하지 않는다. 모든 원행·평균·분산·표준편차·중앙값·P90·paired CI·fresh 전체 경로 시간은 [고정 V18 보고서](../pallet_consensus_decision_repair_20261011_v18/RESULT_KO.md)에 있다.

기존 source-test의 mesh 소속 증거는 합성 source 장면의 증거다. 실사13 query의 물리 corner·두 incident edge 인증으로 옮길 수 없다. `manual_click`은 입력 방법이며 physical material corner/가상 cuboid 구분의 승인이 아니다. `unknown`이나 원주석 출처 누락을 `NONE` 감독으로 바꿔서는 안 된다. 수치/출처 불확실성과 실제 대응 부재를 구분해야 한다.

현재 승인된 기존자료 내 최소 확인은 수행했다: 공개ID→source cache ID→고정 AXIS manifest의 원주석 경로→저장 phase permutation→point source/visibility/reason을 연결했고 원주석 바이트를 바꾸지 않았다. 이 provenance 집계는972 pose moment/135 CI 독립검산과 별도이며, 원주석 source 태그를 새 자세 정답으로 승격하지 않았다. [초기 인덱스 해시 전사 실패](FIRST_REMAINING_CASE_INDEX_FAILURE.json)도 zero-row 실패로 보존했고 원 원행은 producer SHA와 일치한다.

남은 가장 작은 확인은 이13 query에 대해 **같은 이미지 SHA·native/annotation ID·실제 corner 정의·두 물리 incident edge의 소속을 확인한 기존 독립 승인 기록**을 연결하는 것이다. 현재 연결한13 원주석과 AXIS review에는 그러한 승인이 없다. 다른 미검토 기록에도 절대 없다고 주장하지 않는다. 독립 근거가 확보되면 고정 설정·같은 관측의 pose 결과를 그 근거로 판정하여 observation 선택 문제와 proxy 문제를 구분할 수 있다. 이 확인 없이 q를 oracle로 거절하거나 새 설정을 반복해 성공을 만들지 않는다. 추가 성능 실험·학습·새 RGB·수동 주석은 이번 provenance 확인에서0이다.

검증된 payload 게시 commit은 `8a863c0044a74378fe5688ad3896654dcab495f6`이며 당시 원격 SHA·main 불변·사용자 workspace 보존은 [실제 영수증](PUSH_V16_V18_RECEIPT.json)에 기록한다. 이 후기 metadata는 고정 실험 code/원행/수치와 이미지8개를 수정하지 않는다. 전용 브랜치에만 정상 push했고 main 변경·병합·force push는 없다. 목표를 complete로 표시하지 않았다.
