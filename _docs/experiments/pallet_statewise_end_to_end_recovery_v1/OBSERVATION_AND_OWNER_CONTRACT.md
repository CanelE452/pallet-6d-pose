# 관측 가능성·물리 owner·조건부 정밀도의 입력 계약

원 RGB·고정 upstream·등록 cuboid·기존 상태의 자동 관측을 재사용한다. PRIMARY 참조/성공 후보/원 실패분류는 아래 입력에 없다. 물리 owner의 독립 확인이 없다는 이유로 A 전체를 멈추지 않고, 허용한 합성 검증과 자동 global 대안으로 개발 실험을 수행한다. 확인되지 않은 owner를 새 정답으로 학습하지 않는다.

| 의미 | 실제 원 필드/소스 | 새 경로에서의 의미 |
|---|---|---|
| canvas/기하 | typed `solver_input.K/image_hw`, family `native_xyz/physical_registration_Q` | 원 pixel/미터/원점/Q를 그대로 보존 |
| 원 상태 | `solver_input.states` | S/B/I/O/T/U는 관측 역할. owner 정답 확률이 아님 |
| 원 complete point | typed `activation.complete_ids`; authoritative prepared measurements | 실제 RGB 교점 또는 upstream fallback. 같은 채널의 pose projection으로 교체하지 않음 |
| partial | `solver_input.boundary_observations`; `activation.all_partial_ids` | full 철회를 보존한 normal + finite tangent. 원점 archive는 정밀2D factor가 아님 |
| S | states S 및 `S_image_terms=0` | 영상 좌표 회귀/새 RGB query 없음. pose를 구한 후 model projection으로만 보고 |
| O/T | prepared `domains`, 원 outside-canvas/T union | 실제/초기 기하에 조건부인 region의 원 출처를 유지. point likelihood로 승격하지 않음 |
| U | `u_regions` | 가설 구간/보류. 작은 covariance의 점으로 대체하지 않음 |
| 선 지원 | trace `rays`, typed partial `association_provenance/line.provenance` | 실제 observed pixels·TLS와 model 안내를 구분. 같은 line의 여러 표본은 독립 센서 아님 |
| 조건부 precision | prepared `covariance_px2`, `normal_variance_px2`, `fit_reliability_weight` | 원 CAL total-error override를 사용. 상태 confidence를 owner probability/none 비율로 쓰지 않음 |
| 가상 geometry | projected original8 cuboid vertices | 측정이 아닌 계산값. 실제 재료 경계가 아니라 외접/등록 box 꼭짓점일 수 있음 |

## owner 보조 상태

원 레이블을 수정하지 않고 다음 보조 상태를 구분한다.

- `confirmed_physical_owner`: 독립 physical corner/축 확인이 있는 경우만. 이번 자동 실사 packet에는 새로 인증하지 않는다.
- `set_of_possible_owners`: A가 제공한 네 전역 proper permutation에 따른 owner 집합. 영상에서 실제 인증됐다는 뜻이 아니다.
- `unknown`: 관측이 그 어떤 model corner/edge에도 대응하지 않을 수 있음. A의 none branch가 이를 표현한다.
- `virtual_cuboid_feature`: cuboid 좌표로 정의되지만 물리 재료의 실제 교점인지 확인되지 않은 model feature. registered vertex라는 사실만으로 `confirmed_physical_owner`로 바꾸지 않는다.

모델의8개 꼭짓점은 항상 등록 cuboid vertex지만, 팔레트의 둥근 모서리·상판 cutout·포크 구멍·재료 경계가 그 위치에 실제 존재하는지는 별도다. 따라서 실제 외곽/포크 입구/상판 edge를 nearest projected corner에 붙였다는 이유로 정답 owner라고 인증하지 않는다. B도 이 한계를 유지한다. 새 정밀 메시나 다른 시점은 현재 입력에 몰래 추가하지 않는다.

## 재현하는 실제 활성 집합

`fixed_cost.prepare(packet,'G13',family)`가 원 typed/legacy backend와 full/partial 철회 및 outline delegation을 결정한다. prepared measurements에 원 complete archive가 남을 수 있으므로 A는 boundary owner에 속한 full factor를 공통 집합에서 한 번 철회한다. 이는 h마다 관측 수를 줄이는 분기가 아니다. `full:k/partial:k`가 새 group ID이며 한 native channel의 full과 partial은 동시에 평가하지 않는다.

prepared 단계에서 누락된 source/state는 `ctx.packet.solver_input.visible`와 `ctx.states`에서 복원한다. 신뢰도·covariance·좌표 값은 바꾸지 않는다. 같은 parent RGB 선을 여러 모듈에서 소비할 수 있는 점은 dependence ledger로 남기고 독립 정보량/calibrated posterior를 주장하지 않는다. AB에서 B가 소비하는 group을 제외하려면 동일한 fixed ID 집합을 모든 h에 적용해야 한다.

## global owner 운반

하나의 source observation k를 `p_h[k]`로 옮길 때, 그 점·incident line 양끝·partial tangent guard·O/T/U owner·state vector를 함께 옮긴다. 실제 RGB 좌표/normal/offset/covariance/support interval은 변하지 않는다. `object_xyz`와 model incident normals는 새 owner에서 재계산해야 하며 예전 vertex의 normal을 단순 복사하지 않는다. `map_owner_records`가 이 재계산을 수행한다.

초기 S/B/I와 새 h의 기하 가시성이 충돌할 수 있다. 이때 한 h만 더 많은 관측을 S/O/none으로 보내 raw cost를 줄이면 안 된다. 동일 원 관측 집합을 설명하고 원 hard constraint를 해당 h로 운반하여 위반을 명시하거나 +∞로 남긴다. 통과한 h의 prior를 다시 정규화해 삭제 보상을 주지 않는다. 실제로 확인된 외부 가림·화면 밖 제약도 임의 해제하지 않는다.

원 full/partial/region과 A owner는 물리 GT가 아닌 고정 자동 입력이다. 참조와의416px 같은 잔차를 새 검출 오류 레이블로 쓰지 않는다. 기존 reference/치수/대칭/성공 기준은 평가 프로세스에서 그대로 유지한다.
