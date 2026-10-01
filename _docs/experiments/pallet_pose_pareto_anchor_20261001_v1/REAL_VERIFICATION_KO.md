# Pareto anchor 실사 진단의 독립 검산

**검산 PASS**. 원래 5개 안정성 gate의 진단 판정은 **PASS**이며, 실제 배포 가능한 모델이나 전체 목표 성공을 뜻하지 않는다.

519개 frame×seed 선택을 독립 복원하고 4,472개 통계·상태 leaf를 확인했다. 원래 기준 gate, 필요한 2,000회 recording×seed bootstrap(seed 20261001), 6회 recording 제외, 모집단·seed·recording 요약이 저장 결과와 일치한다. Boolean은 정확히 같으며 float 허용치는 1e-12다.

자연99의 seed별 quantile 평균은 T median 7.414874407053 cm, R median 3.136052428130°, T P90 120.471370102895 cm, R P90 71.426585778022°다.

R0 operational GEO anchor를 명시적으로 복원하고, 같은 identity가 캐시에 존재한 444건의 오류 쌍이 정확히 같음을 확인했다. 519/519 선택은 T와 R 모두 R0 anchor 이하이며 완전한 단일 pose다. Anchor 복원 75건, anchor identity 반환 291건, 보존한 후보 집합에서 anchor만 허용된 경우 291건이다. 이 후보 수는 압축 캐시 + anchor 기준이며 원래 네 후보 identity 전체의 census가 아니다.

원래 고정 cost oracle의 실패 기록을 그대로 대조했다. 여기서 바뀐 것은 같은 frame의 R0 T/R를 모두 보존하는 허용 집합이다. 이것은 기존 target의 한계를 보완하는 GT 기반 후보 가능성 증거이며, 새 학습 결과나 실제 runtime 개선으로 바꾸어 말하지 않는다.

현재 모든173 입력이 유효하므로 결측 anchor 분기는 실행되지 않았다. 프로토콜의 실패 유지 서술과 구현의 예상 밖 결측 STOP 차이는 [구현 검토](REAL_METHOD_REVIEW_KO.md)에 명시했다. learned R0_ONLY real 비교는 없고 원래 SINGLE251/R0/PRIOR1/FULL125 기준만 검사하므로 후속 UNION 확장 계약의 전체 성공도 아니다.

[검산 JSON](REAL_VERIFICATION.json), [진단 결과](REAL_FEASIBILITY.json), [독립 검산 코드](../../../scripts/research/pallet_pose_pareto_anchor_20261001_v1/verify_real.py)
