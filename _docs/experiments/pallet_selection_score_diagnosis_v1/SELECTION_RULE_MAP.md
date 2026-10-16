# 고정 원후보의 선택 재현 규칙

이번 코드는 저장된 후보 레코드만 읽는다. 원 G10/G13 solver, PnP, initializer,
수치 최적화 함수를 호출하지 않는다. 참조상 성공 표시는 원 봉인 오라클의
동일 후보 ID를 읽으며 선택 재현 자체에는 사용하지 않는다.

| 결정 | 실제 원 소스 | 이번 재현 |
|---|---|---|
| outline 후보 진입 | `statewise_global_outline_solver_v4_10.py:223`, `statewise_partial_BI_outline_solver_v4_13.py:247` | 원 순서의 `joint_feasible`, `global_outline_audit.valid_projection`가 참인 저장 endpoint |
| 최초 중복 대표 유지 | 위 소스의 `consider`, `joint_ot_pnp.py:272` | rotation magnitude `<1e-5` rad AND translation/원모델 대각선 `<1e-5`; 원 순서의 첫 대표를 보존 |
| G10/G13 outline 내부 winner | `statewise_global_outline_solver_v4_10.py:261`, `statewise_partial_BI_outline_solver_v4_13.py:297` | `(objective, initialization name)`; 정확 동률은 Python stable 순서 |
| typed partial-only 내부 winner | `statewise_partial_BI_u_solver_v4_13.py:498` | 동일 `(objective, initialization name)` |
| legacy partial 내부 winner | `statewise_partial_boundary_u_solver_v4_6.py:408` | 동일 `(objective, initialization name)` |
| 저장 U parent 위임 | `statewise_u_region_solver_v4_5.py:136`, `joint_ot_pnp.py:718` | U를 통과한 원 parent sample을 유지; parent의 objective-only stable 순서. 위임한 결과를 새 생성으로 세지 않음 |
| W/D family 선택 | `statewise_two_wd_family_v4_9.py:131` | `accepted` family 중 `(objective, family index)` 최솟값. 정확 동률 ORIGINAL_WD; 둘 다 거부면 index0의 자체 fallback |
| 최종 e | `select_frozen_wd_partial_refinement_v4_13e.py:27` | G10의 family를 고정하고 G13 그 family의 이미 저장된 결과 또는 그 자체 fallback을 읽음 |

G10과 G13은 서로 다른 관측·목적함수 패키지다. 서로 다른 단계의 objective를
빼거나 비교하지 않는다. typed revocation이 없어 G13이 G10을 정확 재사용한
영상에서는 원 `exact_D_G10_cache_reuse`를 따라 같은 G10 endpoint 로그를
조회한다. 그렇더라도 G10/G13 retained record의 단계 역할은 합치지 않는다.

각 원 retained sample의 `ORIGINAL_<stage>/<family>/feasible/<index>`를
그대로 유지한다. 실제 초기화 이름·branch·원 endpoint 순서·원 보존 순위·Q·XYZ를
연결한다. 최종 fallback은 출력 역할이며 가짜 generated/feasible 후보로 넣지 않는다.

검산 허용치는 이전 `parity.py`와 동일한 `atol=1e-9, rtol=0`이다. 후보 순서,
초기화 이름, branch, family, accepted/fallback 및 출처 문자열은 정확 비교한다.
숫자 R/t, physical Q/회전, 중심, objective는 고정 절대 허용치로 비교한다.

`selection.py.rank_retained`는 §8의 사전 고정된 점수 항 영향 진단용이다.
이미 eligible인 단일 frame/stage/family 풀에 같은 후보 수의 고정 scalar를
받아 원 comparator로만 순위를 매긴다. hard 제약·dedup·registration은 바꾸지
않는다. G10-only 진단의 family 선택과 G13-internal-only 진단은 서로 분리한다.

그룹은 원 `earliest_observed_failure_stage`의 `family_selection`을
`FAMILY_SELECTION_LOSS`, `within_family_final_selection`을
`WITHIN_FAMILY_SELECTION_LOSS`,
`confirmed_original_consumer_observation_or_proposal_limit`를
`NO_SUCCESS_CANDIDATE`로 매핑한다. 원 최종 공동성공은
`FINAL_SUCCESS_CONTROL`이다. 이는 이번 분석 별칭이며 원 오라클을 수정하지 않는다.
