# 실패 원인·학습·일반화

V20~V44. 손실·문맥·PCGrad, 실제 신규 합성 관측, 독립 점/선 전문가, 적응형 보정·실제 가시 경계 감독·점별 선/끝점 대응.

[전체 노트 목차](../../README.md) · [현재 점·선 연구 종합](../STATUS_AND_NEXT_STEPS.md)

> 각 원문은 실험 당시의 기록이다. 당시의 중단/일시정지/다음 제안은 최신 상태를 뜻하지 않는다. 현재 결론은 종합 문서를 우선한다.

| 문서 | 내용 |
|---|---|
| [pallet_fresh_decision_v44.md](pallet_fresh_decision_v44.md) | 새24그룹·부모미학습감독·점/선 대조24회귀, 작은효과재현과안정성실패·현재계열종료 |
| [pallet_useful_constraint_v43.md](pallet_useful_constraint_v43.md) | 개선 가능한 유효 후보 감독·4배 비중·순위 15분류기, AP 개선과 후보 보존/피해·기권 분리 |
| [pallet_constraint_validity_v42.md](pallet_constraint_validity_v42.md) | 목표점 제약 유효성 18분류기, 높은 정밀도와 추가 점 이득 실패·유효 후보 누락/순위 손실 분리 |
| [pallet_constraint_residual_v41.md](pallet_constraint_residual_v41.md) | 기하 계산+잔차와 직접회귀27모델, 물리법선/기존후보 분리·GT조건 이득과 실제 잘못된 제약 피해 |
| [pallet_segment_correspondence_v40.md](pallet_segment_correspondence_v40.md) | 실제 선분/GT대응/물리직선/가시끝점/역할교란30모델, 기하 제약과 끝점 의미 분리 |
| [pallet_hard_direction_v39.md](pallet_hard_direction_v39.md) | 큰 오류 비중·전용 보정·그룹 OOF·실제/정답 선택48모델, 방향 일반화 실패 분리 |
| [pallet_line_uncertainty_v38.md](pallet_line_uncertainty_v38.md) | OOF 잔차·불확실성·범위 보정36모델, 거절/선택 안정성·잘못된 보정 방향·oracle 분해 |
| [pallet_line_utility_v37.md](pallet_line_utility_v37.md) | 개선 선 후보의 점 잔차·관측가능상한·144개 회귀식, 연장 오차와 보정 목표 대조 |
| [pallet_continuous_lines_v36.md](pallet_continuous_lines_v36.md) | 연속 θ/ρ·후보 품질·후속 인코더 적응21모델42k, 선 정확도와 점 기하 효용/안정성 분리 |
| [pallet_role_offset_v35.md](pallet_role_offset_v35.md) | 실제 경계↔가상 점 offset·관측 방향·예측 후보와1152격자 정밀도 분리, 고정3시드 합성 진단 |
| [pallet_line_validation_v34.md](pallet_line_validation_v34.md) | 24모델 반복 재현성 수정·검증, 선 위치/영상 특징 분리 개입·실제 사례와 최종 판단 |
| [pallet_correspondence_v33.md](pallet_correspondence_v33.md) | 점별 Hough support/끝점 query, 선만·끝점만 등6대조18모델, 실제 attention·재현 변동·기하 효용 분석 |
| [pallet_visible_geometry_v32.md](pallet_visible_geometry_v32.md) | 실제 CAD 가시 경계/끝점 감독·점/Hough/가상 감독·보조 감독만 대조15모델, GT와 선 기여 분석 |
| [pallet_separated_gain_v31.md](pallet_separated_gain_v31.md) | 새 원본·점 선택/선 추가 이득 분리·일반 방향 대조3시드 학습과 실패 분석 |
| [pallet_cause_v20_design.md](pallet_cause_v20_design.md) | V20 설계와 단계별 계획 |
| [pallet_cause_v20_results.md](pallet_cause_v20_results.md) | V20 원인 분석과 수정 검증 |
| [pallet_residual_v21_design.md](pallet_residual_v21_design.md) | V21 사전 설계 |
| [pallet_residual_v21_results.md](pallet_residual_v21_results.md) | V21 새 잔차 데이터와 gradient surgery 검증 |
| [pallet_objective_v22_design.md](pallet_objective_v22_design.md) | V22 사전 대조 설계 |
| [pallet_objective_v22_results.md](pallet_objective_v22_results.md) | V22 점 거리 손실과 부모 문맥 검증 |
| [pallet_quality_v23_design.md](pallet_quality_v23_design.md) | V23 설계와 학습 데이터 진단 후 추가 대조 |
| [pallet_quality_v23_results.md](pallet_quality_v23_results.md) | V23 실제 합성 관측과 정확도 균형 검증 |
| [pallet_identifiability_v24_design.md](pallet_identifiability_v24_design.md) | V24 사전 설계 |
| [pallet_identifiability_v24_results.md](pallet_identifiability_v24_results.md) | V24 보정 방향 정보와 작은 집합 학습 검증 |
| [pallet_observation_v25.md](pallet_observation_v25.md) | V25 조건 분해→반대쪽 실제 관측→방향 보정7대조·21모델의 결과와 시각화 |
| [pallet_camera_trust_v26.md](pallet_camera_trust_v26.md) | V26 실제 CAD180렌더·360관측, 점/선 정확도와 보정 방향·oracle·카메라별 검출 진단 |
| [pallet_direction_utility_v27.md](pallet_direction_utility_v27.md) | V27 방향별 효용·이동량·역할 구조12모델/24,000업데이트 학습과 안정성 검증 |
| [pallet_direction_utility_v27_cases.md](pallet_direction_utility_v27_cases.md) | V27 oracle8.78px 실제 크기·개선/실패 사례의 원본/점/Hough 선/확대 시각화 |
| [pallet_line_only_v28.md](pallet_line_only_v28.md) | V28 사례3의 선 후보만 보정·전체6조건 확인·점 힌트 gate를 우회한 DHT 선 추론 |
| [pallet_adaptive_experts_v29.md](pallet_adaptive_experts_v29.md) | V29 점/점가이드DHT/점독립DHT와 점별 선택기24모델 학습, 3조건 가설·독립 선 사례·선택 실패 원인 검증 |
| [pallet_directional_adaptive_v30.md](pallet_directional_adaptive_v30.md) | V30 기존 강한 점 부모 보존·상태 없는 선 법선 부분 보정·방향/이득12모델 학습, 추가 효과와 적응 선택 실패 분해 |
