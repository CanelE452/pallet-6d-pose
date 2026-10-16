# 원래 e와 e 후단의 관측 계보·형식·owner 대응

이 문서는 새 감사 출력 필드와 기존 필드의 대응표다. 기존 e, e_trace, 가중치, 관측 좌표·공분산·상태·손실을 수정하지 않는다. 기본 모집단은 guarded 입력의 명시적 전체238 ID(CAL46/TEST49/추가143)이며 성능·참조·사례 오차로 행을 선택하지 않는다.

## 다섯 단계와 실제 원자료

| 단계 | 실제 기존 필드·원자료 | 새 출력 |
|---|---|---|
| A: 원 N3/SubPix | V15 `INPUTS.jsonl.gz`의 `finish_input._original_source.original_raw_points`, `original_coarse_pose`, `initial_geometry` | `A_original_N3` |
| B: 원 e 이전 상태 | 같은 원본의 `states`, `state_observation`; V15 `mask_binding` | `B_original_pre_e` |
| B: 최종 RGB 보정 | CAL/TEST: **`shared_edge_partial_b_eval_v4_7/{CAL,TEST}_PREDICTIONS.jsonl.gz`의 `correction`**; 추가143: `fresh_observations_v4_14/OBSERVATIONS.jsonl.gz`의 `correction` | `B_original_pre_e.correction` 및 저널 SHA·행 번호·행 SHA |
| C: 실제 e consumer | V15 `spec`, `finish_input.frozen_BI_fit`, `finish_input.partial_boundary_observation.boundary_observations`, `u_regions`, `outline`, `family`; `baseline.solution` | `C_original_e`, typed replay packet |
| C: e 최종 선택 | V17 `SAME_FROZEN_E`; V15 `baseline`과 전체 payload SHA 일치, guarded `E_FROZEN.coarse_pose`와 수치 차이≤기존1e−12 | `E_TRACE.final_pose`와 numeric bridge |
| D: e 이후 입력·재관측 | guarded `e_inputs/INPUTS.json`의 `native_vertex_mapping`/재번호 원좌표; `e_run/observations/*_E_FROZEN.json`의 자기 관측·mask | `D_post_e` |
| E: 새 후단 최적화·선택 | 같은 `e_run/poses`의 자기 B/I solve; guarded 봉인 예측의 `E0/E1/EA/EB` | `E_post_e_solve` |

V15는 `fixed_wd_archived_tangent_eval_v4_15`, V17은 `cached_native_edge_candidates_v4_17`을 뜻한다. 두 폴더는 기존 visibility 실험의 `boundary_recovery_v1` 아래에 있다. 원 e 이전의 mask·선과 e 이후 새 mask·선을 섞지 않는다. A에 남은 원 네트워크 배열은 감사용 원예측이며 S에 대한 영상 측정으로 사용하거나 표시하지 않는다. S readout은 모델 투영으로 분리한다.

## 기존 e_trace에서 확인한 두 가지 추적 공백

1. `e_trace.py:179–194`는 명시적인 typed revocation만 complete 목록에서 뺀다. 실제 consumer `statewise_partial_BI_u_solver_v4_13._joint_objective_pair`는 **모든 boundary ID**를 complete2D loss에서 뺀다. legacy fallback-B partial72개도 full point가 아니다. 신규 packet의 `activation.complete_ids`는 실제 consumer 기준이며 `legacy_e_trace_complete_ids`/`legacy_e_trace_false_complete_ids`로 옛 표시 분류를 따로 남긴다. 기존 e_trace 소스는 변경하지 않는다.
2. 기존 e_trace의 CAL/TEST correction 매핑은 `evaluation_v3.results.FULL`에서 끝난다. 실제 e는 이후 V47 corner-reach 추가복구를 사용해12개 B의 xy/source가 달랐다. 새 추출은 원 V15 protocol에 바인딩된 **V47 correction**을 읽어 모든 frozen fit의 xy/source를 검산한다. 최신 후단 관측으로 원 e를 대체한 것이 아니다. 처음 인증에 실패한 추출 시도는 private의 별도 실패 폴더에 보존한다.

`visible_observations`에 좌표가 남거나 `is_actual_image_observation=True`라고 쓰인 사실은 활성 full2D loss의 증거가 아니다. `VISIBLE_BACKBONE_FALLBACK`은 원예측 상속으로 분류하며 실제 교점 관측으로 표시하지 않는다.

## 원래 optimizer에 들어가는 packet

API는 `lineage.prepare(source_root, guarded_dir, output_dir)`와 `lineage.iter_typed_packets(output_dir)`다. 파일 생성기는 새 출력 경로만 허용하며 기존 출력은 덮어쓰지 않는다. streaming JSONL과 frame별 읽기로 메모리를 제한한다.

| packet 필드 | 원 필드와 의미 |
|---|---|
| `solver_input.xyz/K/states/visible/domains/image_hw/coarse_pose/config` | 원 `spec` 바이트 의미 그대로. `visible`은 revocation 검증·seed·감사용 보관까지 포함한 전체 archive |
| `solver_input.boundary_observations` | `finish_input.partial_boundary_observation.boundary_observations` = **`boundary_typed`**. 이름과 달리 기존 legacy-B partial도 포함 |
| `solver_input.u_regions` | 기존 range-only U 제약. 영역 중심을 관측점으로 바꾸지 않음 |
| `solver_input.outline_observation` | 원 mask에서 이미 계산해 저장된 전체 윤곽·표본·configuration |
| `activation.complete_ids` | `visible`의 ID에서 **typed+legacy partial 전체**를 제외 |
| `activation.typed_partial_ids/legacy_partial_ids` | 명시적 complete-BI 철회와 기존 raw-B 철회를 분리 |
| `activation.selected_objective_component_parity` | 저장된 실제 objective의 제거 ID와 비교. fallback 등 component가 없으면 `null`; 검산 성공으로 꾸미지 않음 |
| `registered_family` | e가 실제 선택한 family의 native XYZ·치수·Q·coarse |
| `e0_pose` | 기존 e 예측. SINGLE의 optimizer 시작점으로만 사용하며 위 원 `coarse_pose`를 교체하지 않음 |
| `original_e_stage1_family_seed_bank` | V15에 저장된 두 family bank 원형. 원 starts를 재구성하기 위한 자료이며 그대로 전부 solver starts로 넣지 않음 |
| `original_e_legacy_boundary_observations` | 원 G10의 legacy boundary. 이후 G13 typed 관측과 분리 |
| `original_e_selected_solution_schema/accepted/hierarchical_correspondence_audit` | 원 consumer 분기·선택·fallback 해석 자료 |
| `provenance` | 원 저널 SHA/행/행 SHA, mask SHA, correction 단계·저널·행 SHA, e 최종 저널 SHA |

V15의 `boundary_all`에는 후속 control을 위한 archived tangent 정보가 추가된다. **원 e의 손실로 가져오지 않는다.** 원 e의 coarse prior/제약 매개변수는 보존한다. E0는 참조가 아니며 SINGLE용 시작점이다.

원 G10/G13의 시작점 복원은 별도 replay 코드에서 수행한다. V15 bank의 `[D winner, WD, G10 winner, (ORIGINAL만 P1)]`을 원 e가 처음부터 전부 사용했다고 간주하면 안 된다. 원 consumer는 WD와 ORIGINAL의 P1 solution 또는 SWAPPED의 기존 P1 변환 start를 사용한다. 이름·순서·fallback·family 선택은 replay parity에서 별도 검산한다. 원 e의 원본 replay와 E0 numeric bridge는 다른 검사다.

## 실제 활성 손실·제약

| 관측 | 원 consumer 처리 | 보존할 정보 |
|---|---|---|
| complete B/I | 기존 effective covariance·reliability와 robust point loss | 관측/fit 공분산 둘 다, state weight, 원 source |
| typed 또는 legacy partial B/I | complete2D likelihood 제거, 선 normal robust loss + 유한 tangent interval 제약 | line normal/offset/variance, reliability, tangent 범위·extrapolation, 원 association hypothesis |
| O/T | 기존 spatial domain의 부등식. T의 union은 원 outside-side 선택을 유지 | shape·행렬/halfplane·margin·outside-side 등 원 레코드 전체 |
| U | 기존 ellipse/range 제약만; U point/line loss 없음 | 기존 영역 파라미터·출처 |
| S | local 영상 질의·정밀 점/선 항 없음 | 조건화한 초기 pose와 최종 모델 투영을 관측과 분리 |
| 전체 윤곽 | 기존 전역 mask의 robust shape loss | 원 표본·clip 규칙·noise/weight/configuration |

공분산이 작거나 whole-outline rank가 높다고 영상선의 physical owner가 검증되는 것은 아니다. 원 soft loss와 visible geometry/canvas·OT/U·partial tangent hard guards를 구분한다. partial의 원 xy는 seed/검산 archive로 남아도 정밀 관측 loss로 되살리지 않는다.

## owner와 source group

원 e의 W/D 비교는 두 family에서 **같은 native image channel ID**를 유지한다. family의 Q/XYZ가 바뀌면 그 채널에 가정하는 physical owner도 달라질 수 있다. 반면 e_inputs는 모델의 고정 physical vertex로 원 N3 좌표를 재번호한다. 다음 두 링크를 따로 센다.

- **C→E owner 링크:** 선택된 e family의 native k에 붙은 원 partial/full과 같은 E-native k의 새 full/state를 비교한다.
- **A→D 원좌표 링크:** 원 N3 channel k가 inverse permutation으로 어느 새 E channel에 주입되는지 기록한다.

따라서 source channel을 따라 센 partial→full과 원 e가 실제 사용한 owner를 따라 센 partial→full은 별도 수치다. 후단은 새 RGB 관측 패키지이므로 전이를 확인했다고 원래 선 표본을 직접 재사용했다고 단정하지 않는다.

모델 vertex bijection의 유일성은 기하 등록 검사다. 실제 영상선이 어느 물리 코너인지에 대한 독립 정답이 없으므로 `SEMANTIC_OWNER_UNVERIFIED`를 남긴다. point covariance·model bijection·semantic correspondence를 하나의 신뢰도에 합치지 않는다.

ray의 동일 RGB SHA와 동일한 관측 표본/confidence를 해시한 `exact_sample_source_group`으로 중복 의존을 기록한다. endpoint 또는 표본 순서를 뒤집어도 같은 group이다. 동일 group, 선 교점·partial·윤곽의 공유 가능성을 보고할 뿐 새 covariance·가중치·손실은 만들지 않는다. 다른 표본 해시가 통계적 독립성을 증명하지도 않는다.

## 출력과 검산 경계

비공개 추출에는 `E_TRACE.jsonl.gz`, `TYPED_PACKETS.jsonl.gz`, `OBSERVATION_LINEAGE.jsonl.gz`, `OBSERVATION_TYPE_AND_OWNER_AUDIT.csv`, `EXTRACTION_SEAL.json`을 쓴다. 실제 공개 범위는 별도 작은 집계·코드·허용 RGB만 포함한다. 원 mask와 대형 저널은 경로·SHA로 바인딩한다.

순수 검사는 legacy/typed 활성 분리, 잘못된 S/O/T/U 점 인자 거부, fallback 출처, 직렬화, endpoint reversal의 group 불변성, S readout/철회된 display 좌표 오염의 활성 factor-view 불변성을 확인한다. 이 정적 view 검사는 solver 수치 재현을 대신하지 않으며 실제 residual/constraint·pose 불변성은 replay에서 따로 확인한다.
