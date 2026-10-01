# Pareto anchor 실사 진단의 구현 검토

**현재 동결 입력에 관한 구현 검토 PASS**다. R0 operational GEO의 후보 이름과 기존 T/R 오류를 anchor로 명시적으로 복원하고, T와 R가 모두 anchor 이하인 완전한 pose 중에서 기존 TRAIN scale의 minmax 규칙으로 선택한다. 동일 후보 identity의 오류 쌍은 정확히 같아야 하며, 동률에서는 기존 Pareto 제거·R0·가설·expert 순서를 유지한다. 실제 선택/5개 gate의 독립 수치 검산은 별도 산출물로 구분한다.

압축된 후보에서 빠진 pose가 anchor 조건을 만족하면 그 pose를 약하게 지배하는 보존 후보도 같은 조건을 만족한다. 따라서 이 압축은 정해진 단조 cost/Pareto 선택에 필요한 오류 쌍을 보존한다. 다만 `eligible_candidates`, `anchor_only_eligible` 등의 수는 **압축 후보 + 복원 anchor**에 대한 집계다. 원래 네 후보 identity 전체의 가용/허용 개수를 모두 복원한 집계로 해석하지 않는다.

프로토콜과 코드에는 **실행되지 않은 결측 경계 조건 차이**가 있다. 프로토콜은 unavailable anchor를 실패로 유지한다고 쓰지만, 구현은 기존 9모델×173개의 운영 오류가 모두 유한함을 확인하고 예상 밖 결측이면 중단한다. 동결 입력 감사가 현재의 유한 입력을 확인했으므로 이번 계산에서 프레임 제외·대체나 결과 차이를 만들지 않는다. 다른 결측 모집단에서 failure-retention 분기까지 구현·검증했다고 주장할 수는 없다. 동결 코드·프로토콜은 수정하지 않았다.

R0 대비 pointwise 비증가는 R0 자연 tail/clean 보존을 보장하지만, 엄격한 중앙값 개선·recording 불확실성·paired SINGLE 비교 등 나머지 원래 조건까지 자동으로 보장하지 않는다. 이 검사는 GT 기반 진단이며 배포 가능한 선택기나 실제 목표 달성이 아니다. learned R0_ONLY의 real 비교 결과도 없어 후속 UNION 확장 계약을 충족했다고 주장하지 않는다.

[기계 판독 검토](REAL_METHOD_REVIEW.json), [동결 프로토콜](PROTOCOL.json), [동결 구현](../../../scripts/research/pallet_pose_pareto_anchor_20261001_v1/real_feasibility.py)
