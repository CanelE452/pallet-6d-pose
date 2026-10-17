# 새 관측과 옛 결측 구간·기존 점/선의 관계

이 문서는 새 입력과24개 예측을 동결한 뒤 작성한 **사후 공간 진단**이다. 관측·점수·선택을 바꾸지 않았다. 새 polyline 적분점 각각에서 옛 G13/ORIGINAL_WD 관측의 모든 유한 선분까지 거리를 계산해, 거리 최저의 원 segment를 원 segment 순서의 `argmin`으로 연결했다. 그 원 부모가 옛 censor 대상인지 집계했다. margin이나 새 거리 threshold를 만들지 않았고 삭제 부모만 대상으로 nearest를 다시 찾지 않았다. 이6장의 옛 관측 기하는 두 family/두 stage에서 같지만 계산 문맥은 명시적으로 고정했다.

| ID | 옛 원부모 / 결측부모 | 새 길이질량 중 nearest가 옛 결측부모인 비율 | 새점→모든 옛선분 거리 질량평균 px | 거리 최대 px |
|---|---:|---:|---:|---:|
| plastic_night_01:038630 | 12 / 5 | 0.568399 | 5.033 | 16.268 |
| wood_day_01:005286 | 12 / 7 | 0.875705 | 3.382 | 11.925 |
| wood_night_01:030698 | 35 / 0 | 0.000000 | 2.320 | 4.676 |
| plastic_day_01:020955 | 18 / 9 | 0.910660 | 7.521 | 26.419 |
| plastic_day_01:014704 | 22 / 0 | 0.000000 | 1.528 | 3.868 |
| wood_night_01:030898 | 28 / 0 | 0.000000 | 2.248 | 7.352 |

새 관측은 사례1·2·4에서 옛 결측 부모가 가장 가까운 공간에도 존재한다. 긴 부모 전체 제외에서 잃은 구간의 주변에 새 RGB 외곽을 공급했다는 공간적 연결은 있지만, 같은 물리 변을 되살렸다는 인증은 아니다. 특히 nearest 원 hull 선이 새 관측에서 멀 수 있으며 원 hull 자체에 다른 물체/비물리 연결이 포함됐기 때문에 이 비율을 실제 팔레트 지지 복원율·정확도·독립 정보량으로 부르지 않는다. 원영상 전체와 옛 target hull/삭제부모/새 열린 span을 [갤러리](gallery/index.html)에서 함께 확인한다.

새 관측과 원 full B/I·partial normal은 같은 RGB에서 나온다. [원 비용 지도](../pallet_selection_score_diagnosis_v1/OBJECTIVE_MAP.md)에 따라 partial의 native ID는 full 2D likelihood에서 철회되고 원 corner별 normal은 grouped robust로 유지된다. 이번 outline 교체가 철회된2D를 복원하지 않는다. 원 source provenance와 새 L01 등 임의 polyline ID 사이에는 공통의 인증된 물리 edge ID가 없다. 가까운 픽셀이나 별도 입력 파일만으로 중복/독립을 확정하지 않는다.

`wood_night_01:030898`의 사후 원 packet 확인에서 full B/I 활성 native corner는0–4이고 partial은 없었다. 저장된 원 visible measurement audit은 `CALLER_CAL_TOTAL_ERROR_COVARIANCE_OVERRIDE`, reliability override, `covariance_is_calibrated_coverage_probability=False`를 기록한다. 예를 들어 한 저장 seed의 native0 fit covariance는 약[[12.332675,−0.251591],[−0.251591,12.686131]]px², fit reliability는0.981502였다. 이는 최종 모든 후보의 잔차나 관측 정확도를 새로 검증한 값이 아니라 기존 likelihood 입력 경로가 이미 존재한다는 출처 확인이다. 다음 과제는 불확실성 항의 부재를 메우는 것이 아니라 이 기존 계약의 타당성과 충돌을 확인하는 것이다.

직육면체 projection hull은 둥근 플라스틱 모서리와 개별 나무 판재의 계단/빈틈을 정확히 나타내지 못할 수 있다. 수작업에 준하는 모델 보조로 실제 외곽을 표시해도 cuboid hull에 완전히 대응한다는 보장은 없다. 입력 작성 의미오차·model approximation·원 K/참조의 물리 오차를 이 공간 표로 분해하지 않았다.

산술 재현: [observation_context.py](../../../scripts/research/pallet_visible_boundary_pilot_v1/observation_context.py)와 [OBSERVATION_CONTEXT.csv](OBSERVATION_CONTEXT.csv), [OBSERVATION_CONTEXT_RECEIPT.json](OBSERVATION_CONTEXT_RECEIPT.json)에 있다. 실행은 새 후보나 평가를 만들지 않고 봉인된 입력과 옛 공개 부모 기록의 공간 산술만 수행한다.
