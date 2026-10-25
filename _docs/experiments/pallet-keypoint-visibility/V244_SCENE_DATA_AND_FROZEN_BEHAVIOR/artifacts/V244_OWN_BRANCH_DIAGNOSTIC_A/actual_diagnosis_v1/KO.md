# Frozen TREAT 내부 분기 SOURCE 진단

같은 최종 checkpoint의 `model.base(active_features)`와 `model.relational(동일 입력)`을 각각 계산한 뒤 FP32 합으로 TOTAL을 만들었습니다. 이는 별도 CONTROL과의 비교가 아닙니다. 두 family는 각자의 단일 logit에 0 문턱을 적용하며 두 출력을 argmax하지 않습니다.

저장 TOTAL의 7617행/15234 logits을 기존 CPU 재생의 atol=rtol=1e-5로 검산하고, 7617개 예측 부호를 정확히 확인해야 완료됩니다. 원 q/L/native/role/target/available는 유지했습니다.

행별 표의 target_aligned_residual은 정답 방향의 잔차 부호·크기이고, threshold crossing은 ownbase와 total의 문턱 교차입니다. 잔차의 단독 sign 분류는 추가항의 행동 진단이며 별도 훈련·보정된 모델이 아닙니다. 수치분해는 시각적 원인·정답 native geometry·인과관계·해결불가능성의 증명이 아닙니다. SOURCE DEV는 이미 노출·재사용됐습니다.

[SUMMARY.json](SUMMARY.json) · [전체 행 CSV](ALL7617_OWN_BRANCH_ROWS.csv) · [family/role/target 통계](FAMILY_ROLE_TARGET_BRANCH_STATS.csv) · [TOTAL 재생 QA](TOTAL_REPLAY_QA.json)
