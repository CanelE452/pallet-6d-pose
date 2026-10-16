# 고정 후보의 원 목적함수와 관측 장부

이번 코드는 저장된 자세에서 기존 함수를 읽기 전용으로 평가한다. 새 pose fit, 초기 후보 생성, 모델 forward, mask/선 추출은 하지 않는다. 원점·등록·K·상태·공분산·reliability·관측과 순서는 원 packet 그대로다. 실제 후보 선택의 모집단과 comparator는 별도 `selection.py`의 장부에 있으며, 이 문서의 비용 평가 자체가 새 선택기는 아니다.

## 실제 활성 backend 연결

`fixed_cost.prepare(packet, stage, family)`는 해당 family의 `native_xyz`, `physical_registration_Q`, `coarse_pose`와 packet의 `solver_input`을 사용한다. 같은 영상 채널 ID라도 family에 따라 가정하는 physical owner가 달라지므로 서로 다른 Q를 섞지 않는다.

| 단계·분기 | 실제 원 함수와 관측 |
|---|---|
| G10, 원 outline 활성·진입 가능 | `statewise_global_outline_solver_v4_10._prepare`; legacy partial만 사용. `_joint_objective_pair`는 `statewise_partial_boundary_u_solver_v4_6`이고 outline은 v4.10의 원 scalar/수치 gradient |
| G13, 명시적 typed revocation 있음 | `statewise_partial_BI_outline_solver_v4_13._prepare`; typed와 legacy가 함께 든 원 `boundary_observations`; `_joint_objective_pair`는 `statewise_partial_BI_u_solver_v4_13` |
| G13, 명시적 revocation 없음 | v4.13의 실제 위임 대상 `statewise_global_outline_solver_v4_11`; 원 G13 boundary를 전달하고 legacy partial 목적함수 사용 |
| G13 typed, outline 불지원·진입 부족 | 원 v4.13의 `partial_only()` 패키지. 저장된 partial solver 자세를 normal+full BI+range로 평가하며 outline 비용을 만들지 않음 |
| G10/비typed 위임, outline 불지원·진입 부족 | 원 outline solver는 첫 저장 WD solution을 그대로 반환. 그 delegated full BI/U 패키지를 평가하며 적용되지 않았던 legacy partial을 새로 넣지 않음 |
| partial-only 내부 BI seed fit | 실제 seed objective는 full BI와 prior이며 partial/outline/U/domain은 없음. 역할을 `BI_SEED_ONLY`로 분리하고 최종 eligible 후보와 혼합하지 않음 |

원 호출 연결은 `typed_replay.original_stage_inputs`, `_family_run`의 **소스만** 참조한다. 이 진단에서 해당 solve/replay 함수를 실행하지 않는다. V15의 후속 d/G10 winner starts를 원래 consumer 후보로 추가하지 않는다. 기존 원 소비자 full-parity와 현 분석의 저장 후보 선택 재현은 별도 seal로 연결한다.

## 항별 수학과 원 함수

| 항 | 원 구현·가중 후 scalar | 관측 장부와 집계 |
|---|---|---|
| 완전 B/I | `statewise_uncertain_solver_v4_bias._objective_pair`: `sum_i w_i * 2*delta²*(sqrt(1 + ||L_i(q_i-y_i)||²/delta²)-1)` | `L_i=inv(cholesky(effective covariance))`. caller `fit_covariance_px2`가 우선이고 없으면 `covariance_scale*observed covariance + state floor`; reliability도 caller override가 우선. 관측별 원 px 벡터·관측/fit 공분산·whitening·강건 손실·가중 기여를 보존. **합산, 평균 아님** |
| 부분 B/I normal | `statewise_partial_BI_u_solver_v4_13._joint_objective_pair` 또는 legacy v4.6 → `statewise_boundary_normal_solver_v4_4._normal_objective_pair` | 해당 corner의 1~2 normal 잔차 `r_j=(n_j·q+b_j)/sqrt(variance_j)`를 하나의 radial robust loss로 묶고 frozen reliability를 곱함. corner 간 합산. 한 corner 안 두 선의 robust loss를 임의로 가산 분해하지 않고 잔차와 grouped 기여를 보존 |
| 전역 outline | stage 원 `outline_value_and_gradient` / `_outline_value`: `outline_factor_weight * .5 * (model_to_observed_mean + observed_to_model_mean)` | model 방향은 clipped projected convex-hull segment마다 원 15개 표본, `length/15` arc 가중 평균. observed 방향은 원 저장 boundary samples의 평균. 각 방향 pseudo-Huber는 `delta²*(sqrt(1+(distance/sigma/delta)²)-1)`이며 full BI의 factor 2와 다름. 원 관측 configuration·표본수·normalizer·양방향 기여를 보존 |
| coarse prior | `statewise_uncertain_solver_v4_bias._prior_residual_and_jacobian`, `_objective_pair` | 실제 scalar는 `effective_prior_weight * ||whitened residual||²`. outline `_prepare`와 typed partial-only가 weight를 0으로 설정. 이번 원 패키지에서는 **비활성**이며 residual 숫자가 커도 cost가 아니다. 부분-only/WD 위임에서는 원 covariance6 whitening을 복원하고, outline의 원 prior whitening은 identity 유지 |

`global_outline_robust_mean_loss`는 **아직 가중 전 값**이다. `global_outline_factor_weight`를 정확히 한 번 곱한다. `boundary_factor_weight`는 원 normal scalar에 이미 포함된 metadata이므로 다시 곱하지 않는다. rank·ambiguity·count·source·prior residual은 additive loss가 아니다.

## full 2D 철회와 원 관측 출처

partial의 **모든 native ID**를 full BI 목록에서 뺀다. typed의 `removed_complete_BI_2D_likelihood_ids`와 legacy의 `removed_raw_B_2D_likelihood_ids`를 동일 활성 규칙으로 처리하되 원 출처는 분리한다. 이전 e_trace가 full로 표시했던 legacy partial72를 다시 full point로 되살리지 않는다.

full 장부에는 원 archive 보존을 위해 철회된 점도 `active=False, weighted_contribution=0`으로 남긴다. 이 점의 좌표는 normal display나 새로운 정밀 영상점이 아니다. complete likelihood는 철회해도 기존 visible geometry/canvas guard는 원 consumer처럼 유지된다.

`VISIBLE_BACKBONE_FALLBACK`은 원 예측 상속이며 실제 RGB 교점으로 표시하지 않는다. CAL/TEST correction은 수정된 lineage가 인증한 V47이고 과거 evaluation_v3로 돌아가지 않는다. partial의 `association_provenance`, 원 RGB line ID, image/mask binding을 보존한다. 같은 source의 반복 사용 가능성을 확인할 수 있지만 source hash 차이가 독립성을 증명하지는 않는다. `SEMANTIC_OWNER_UNVERIFIED`를 유지한다.

## hard constraint: 비용에 합산하지 않음

원 `P._joint_constraint_pair`를 사용하고 각 행의 slack과 원 단위를 보존한다. 통과는 원 `feasibility_tolerance=1e-7`를 적용하며 임계값을 변경하지 않는다.

| 행 | 원 구현·실제 역할 |
|---|---|
| 모든 8 vertex positive depth | `joint_ot_jacobian_v2.constraint_values_and_jacobian`; depth/model-diagonal 정규화. S가 포함된 모델의 물리 depth guard이며 S 영상 관측이 아님 |
| visible B/I geometry·canvas | `statewise_uncertain_solver_v4_bias._constraints`; 옛 hard residual budget 행은 제거, geometry/canvas만 남김. full2D 철회 partial도 원 guard 유지 |
| O/T domain | 원 ellipse/box/halfplane slack, T outside-side branch를 별도 보존. 원 실제238장 T domain은0이며 새로운 T 효과를 주장하지 않음 |
| U ellipse | `statewise_u_region_solver_v4_5._u_constraint_pair`; `1-delta^T precision delta`. 중심은 정밀 점이 아니고 U point/visibility/canvas cost도 없음 |
| partial finite tangent | `statewise_boundary_normal_solver_v4_4._boundary_constraint_pair`; 각 선에서 lower/upper 픽셀 slack. tangent point loss로 바꾸지 않음 |

제약 행마다 `HARD_CONSTRAINT_NOT_ADDITIVE_COST`를 기록한다. soft likelihood와 hard guards의 단위가 다르므로 slack을 loss와 더하거나 서로 다른 row 단위의 절대 크기를 우열로 쓰지 않는다.

## 비용 검산과 scalar 전용 평가

원 저장 `objective_components`가 있으면 우선 보존한다. 부족한 관측별 내역은 원 objective 함수를 같은 R/t에서 평가해 계측한다. 원 저장 total·가중 components·recomputed total·관측 기여 합을 **각각 절대 1e-9, 상대0**으로 확인한다. 실패한 항목은 통과로 바꾸거나 tolerance를 늘리지 않는다.

후보에 optimizer의 `p`가 저장돼 있으면 원 R/t와 `1e-9` 이내 일치를 검사한 뒤 그대로 쓴다. 없으면 기존 `_pose_vector`로 저장 R/t를 변환할 뿐 새 자세를 풀지 않는다. native/physical Q도 검사한다.

원 outline gradient는 G10에서 수치 차분을 포함한다. `check_outline_scalar`로 원 scalar-and-gradient 반환값과 원 `_outline_value` 값의 동등성을 먼저 검사한 경우에만 unused gradient 계산을 생략할 수 있다. 다른 BI/partial/prior 및 constraint 함수는 원 gradient/Jacobian도 계산하며 이 비용을 실제 시간에 포함한다. 원 표본 수와 정규화는 바꾸지 않는다. dense outline 배열은 `dense_outline=True`로 원 sample별 residual/loss/가중 기여까지 재현한다.

G10과 G13의 총비용은 다른 관측 패키지의 값이다. 각 stage 안 같은 목적함수의 후보 간 차이만 해석하고 `G10(s)-G13(g)`는 계산하지 않는다. 외부 pose나 다른 stage pose를 고정 재평가하면 `COUNTERFACTUAL_FIXED_POSE`/`EXTERNAL_POSE_NOT_IN_E_BANK`로 구분하며 원래 실행 후보나 원 comparator 성능으로 취급하지 않는다.

## 실제 전량 검산 결과

원 CAL 순서 처음6장의 retained300후보에서 scalar-only 선행 동등성을 통과했다. 고정 `fixed_cost.py` SHA는 `d716f0a06dfecf886ce96060472ef12b10b477862784f28937d7672abed2fe6c`다. 이후 **238영상·11,603 retained후보 전량**을 같은 원 패키지로 평가했으며 비용 복원 불가0, 재평가 infeasible0이었다.

| 검산 | 최대 절대 차이 |
|---|---:|
| 원 저장 total 대 고정 자세 원함수 | 7.96e-13 |
| 원 저장 가중 항 대 재평가 항 | 6.83e-13 이하 |
| 재평가 total 대 가중 항 합 | 0 |
| 독립 `math.fsum`의 원 total 대 가중 항 합 | 4.55e-13 |
| 독립 관측 기여 합 대 해당 항 | 6.54e-13 |
| 독립 outline sample 기여 합 대 방향 기여 | 2.85e-14 |
| 독립 sample 가중식 대 저장 sample 기여 | 0 |

전체 private dense journal에는 outline23,174방향 관측, 원 sample11,966,811개의 6종 배열이 보존됐다. compact journal은 같은 `(id, candidate_id)`와 방향 총합·표본수·normalizer·원 dense 파일/행 포인터를 가진 파생본이다. candidate ID는 영상 안에서만 유일하므로 영상 ID를 빼고 join하지 않는다.

전량 평가는 두 worker로 wall282.219초, worker CPU합541.425초였다. 각 준비·비용 평가에 적용한 호출 감사197,802,058회에서 금지 함수 진입 시도0이었다. 원 cost 코드를 import하지 않는 독립 dense 검산은 wall32.951초였으며 모든 원 candidate identity, compact/dense 집합과 dense SHA도 일치했다. 이 시간은 영상 추론 지연이나 FPS가 아니다.

해당 수치의 원 증거는 private `cost_v1/COST_SUMMARY.json`, `COST_SEAL.json`, `DENSE_ARRAY_VALIDATION.json`이다. COST seal SHA는 `e799f8083f34bb9d3948a4d57fd593f5bb43ab19524d3bd797929c27d46b898e`다. 이 검산은 비용 산술과 원 입력 연결의 검산이며, 관측의 semantic owner나 참조의 독립 물리 정답을 검증한 것은 아니다.
