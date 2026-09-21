# E1-C pool audit

[확인] 기존 후보/시각검토 2276행 → SHA 중복 제거 2240장. 기존 메타데이터로 clean이며 평가·teacher 학습 세션과 분리된 후보 0장.

[확인] unknown/미검토를 clean으로 간주하지 않았다. CAD8은 이번 지시에서 평가 세션 제외 조건에 걸리므로 자동 재사용하지 않는다. GT 정확도로 프레임을 선택하지 않았다.

[확인] 실제 이유별 수: `{'no_verified_clean_condition': 2240}`. 8장 미만이면 E2 STOP.
