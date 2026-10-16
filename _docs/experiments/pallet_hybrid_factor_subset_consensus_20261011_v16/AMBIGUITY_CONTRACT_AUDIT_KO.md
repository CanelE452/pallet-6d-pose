# 원 지시문과 저장 후보의 모호성 정책 대조

이 자료는 2026-10-11 코드·문서만 읽은 감사다. 자세 재계산, GT/실사 점수 읽기, 설정 변경, 새 실험은 하지 않았다. 원 사용자의 요청은 실제 위치·회전 개선과 다중해·관측 부족의 구분이며, 첨부 문서는 그 요청을 구체화하는 참고 계약이다. 이후 사용자가 지정한 쉬움·중간 245장 범위를 유지한다.

원 첨부 파일은 `/home/minjae/Downloads/pallet_cli_observation_refiner_robust_pnp_20261009.md`이다. SHA-256은 `e517bf684f8fc7484b756d5c21451ce77f765120377d10cf9fcc92659bbc060b`이다. 158–166행은 모든 4점·치수 후보와 유효 반환 해를 보존하도록 한다. 168–178행은 동일한 관측 풀에서 inlier 수, 다음으로 truncated SSE를 비교한다. 175행은 “수치적으로 동점이면 고정된 tie-break”라고 하며, 177행은 4점 조건에서 대안 해와 점수 차이 기록을 요구한다. 같은 inlier ID를 가진 모든 다른 비용의 해를 실패시키라는 요구는 없다. C2의 402–414행도 부분 선 관측의 기하 조건을 검사하도록 하지만 이 추가 veto를 요구하지 않는다.

| 구현 | 실제 모호성 처리 | 코드 위치 |
|---|---|---|
| V4 일반 PnP | 전체 U의 SSE가 수치적으로 동점이고 물리적으로 동등하지 않은 대안이 있으면 `AMBIGUOUS_PNP` | [pose.py](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py#L295) |
| V4 강건 PnP | inlier 수가 같고 truncated SSE 차이가 고정 `1e-8` 이내인 비동등 대안이면 위와 동일 | [pose.py](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py#L295) |
| S15 / V16 | 위 수치적 동점뿐 아니라, 최종 점·선 inlier ID가 같은 비동등 accepted 대안이면 비용 차이에 관계없이 `OBSERVATION_CONSENSUS_AMBIGUITY` | [solver.py](../../../scripts/research/pallet_factor_subset_consensus_20261011_v15/solver.py#L309) |

V4의 정책은 26–48행, 분기는 295–314행이다. 다른 비용의 비동등 후보는 `multiple_solutions`와 `alternatives`에 기록되며 그 존재 자체로 신규 자세를 거부하지 않는다. 동등성은 모든 8개 모델 투영 및 물리 R,t로 판단한다. ADD 대칭성이나 실제 물리 정답을 증명하는 검사가 아니다.

S15의 309–327행은 먼저 inlier 수와 truncated SSE로 best를 고른다. 이후 `same_support`는 `final_inliers`와 `final_line_inliers`의 일치만 검사하며 `fit_input_ids`·`fit_line_edges` 일치를 요구하지 않는다. 비동등 후보가 `same_support`이면 서로 다른 적합 풀과 다른 비용에서도 veto한다. `numerical_objective_tie`는 별도로 저장된다. 따라서 같은 8px 이진 합의가 같은 수치 목적값이나 동일한 다중근을 뜻하지 않는다. 모든 후보는 같은 전체 관측 풀에서 채점하므로 최종 truncated SSE는 비교할 수 있지만, 서로 다른 적합 풀의 `soft_l1_cost`는 같은 목적함수 값으로 취급하면 안 된다.

이 추가 veto는 S15에서 새로 생긴 코드 오타가 아니다. [V13 solver.py](../../../scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/solver.py#L317)의 317–330행과 정책 40–42행에서 이미 선언된 보수적 정책을 계승한다. [V16 pipeline_ready.py](../../../scripts/research/pallet_hybrid_factor_subset_consensus_20261011_v16/pipeline_ready.py#L157)는 같은 S15 솔버를 그대로 호출한다.

감사에 사용한 소스 SHA-256:

- V4 `pose.py`: `ef827022bef6807c5f7e4016d7ebefa1838c5f013f6d48be6f4932e0c9ad4a40`
- V13 `solver.py`: `3b8cdeb5bc189cb196be4066d095ead081f8a5c3de1b6ee62c28e9402621e4d7`
- S15 `solver.py`: `4f447f8fa21ae11d90be19f7423b6c42b8dbde4cd8bacb32450b54dfe4c495a9`
- V16 `pipeline_ready.py`: `b940cde6eee5023ddd1b1a7c18ff3e5cb77a48567aa58a44f0067fe66481a7d3`

결론은 계약 범위의 차이에 한정한다. 모든 비용의 same-support veto는 원 요청의 필수 조건이 아니라 추가 보수 정책이다. 저장 accepted 대안의 실제 관계를 확인하기 전에는 이 정책이 어떤 프레임을 실패시켰는지, 실제 정확도가 좋아지거나 나빠졌는지 주장하지 않는다. 수치적 비용 차이로 물리 정답이나 전역 해의 완전성을 증명할 수 없고, 비용 마진·새 임계값·새 변형은 제안하거나 실행하지 않았다.
