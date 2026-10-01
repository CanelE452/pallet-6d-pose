# 고정 real union 가능성 진단의 독립 검산

**검산 PASS**. 이것은 후보 가능성 진단의 재현 확인이며 실제 모델 개선이나 전체 목표 달성이 아니다.

- 원래 5개 gate와 두 진단의 5개 gate를 독립 계산했다. 1,038개 프레임/seed/진단 행, 8,437개 수치·상태 leaf를 확인했다.
- repository 평가 helper를 import하지 않았다. 독립 dominance matrix와 정렬 순위 보간으로 후보 선택·median/P90을 계산하고, recording×seed bootstrap 2,000회(seed 20261001)와 6회 recording 제외 검사를 재현했다.
- 원래 baseline 오차와 gate, 각 모집단·seed·recording 요약이 저장 결과와 일치했다. Boolean 판정은 정확히 같고 부동소수점 값의 검산 허용치는 1e-12이다.

| 진단 | 원래 5개 gate 전체 | 확인 가능한 결론 |
|---|---|---|
| AXIS_LOWER_BOUND | PASS | 필요조건 통과이며 공동 달성 증명은 아님 |
| FIXED_TRAIN_COST_ORACLE | FAIL | 고정 TRAIN cost 규칙 실패이며 모든 다른 선택 조합을 배제하지 않음 |

고정 TRAIN-cost target을 매 프레임 완벽하게 선택해도 자연 T P90의 seed 평균은 129.601418690011 cm로 R0 기준 상한 126.494938608040 cm를 넘는다. 따라서 현재 고정 target을 완벽히 모방하는 것 자체로 원래 전체 목표가 달성되지 않는다. 이 실패는 다른 whole-pose 선택 조합의 불가능성을 증명하지 않으며, 축별 낙관적 하한이 통과했다는 결과와 모순되지 않는다.

모든 기존 운영 모델 및 union 진단의 오류가 유한하고, 원래 실패 수 증가 금지 조건을 유지했다. 따라서 같은 분모·bootstrap draw에서 축별 하한의 단조 필요조건 해석이 유효하다. 축별 최솟값은 서로 다른 pose에서 올 수 있으므로 물리적 pose로 해석하지 않는다.

고정 cost oracle은 T와 R를 같은 완전한 후보 pose에서 가져온다. source TRAIN의 기존 scale, 정확한 cost 동률·Pareto 제거·R0/hypothesis/expert 우선순위를 그대로 사용했다. 이미지·checkpoint·raw GT를 읽거나 새 학습·PnP·learned real routing을 수행하지 않았다.

**learned R0_ONLY real 비교 결과는 없다.** 이 진단의 원래 SINGLE251/R0/PRIOR1/FULL125 비교는 후속 UNION 실험의 확장 비교 계약을 대체하지 않는다. Oracle 통과는 배포 가능성·학습 가능성·새 recording 일반화·전체 목표 달성의 증거가 아니다.

[독립 검산 JSON](VERIFICATION.json), [진단 결과](RESULTS.json), [동결 프로토콜](PROTOCOL.json), [검산 코드](../../../scripts/research/pallet_pose_real_union_feasibility_20261001_v1/verify_results.py)
