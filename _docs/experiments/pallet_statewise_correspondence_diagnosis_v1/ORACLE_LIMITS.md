# 후보 oracle의 범위와 기록 한계

이 문서는 참조 점수를 읽기 전에 고정한 분석 범위다. 대상은 CAL46과 PRIMARY192(TEST49 + 추가143)의 기존 238장이다. 성공은 **동일한 후보 하나의 중심 오차 ≤5cm AND physical/C2 회전 오차 ≤5°**다. 위치 최소값과 회전 최소값이 서로 다른 후보에서 나왔다면 공동 성공으로 합치지 않는다. 직사각형 팔렛의 기존 C2만 사용하며 90° C4 동등성을 추가하지 않는다.

분석의 참조는 기존 기하 재구성 pose다. 따라서 oracle은 그 고정 참조에 대한 **기록된 후보 집합의 상한**이며, 독립적으로 확인한 물리적 참자세에 대한 절대 상한이 아니다. 모델의 native/physical 등록 일치는 영상에서 의미상 올바른 코너를 찾았다는 인증이 아니다.

## 1. 원기록에 실제로 남아 있는 것

[candidates.py](../../../scripts/research/pallet_statewise_correspondence_diagnosis_v1/candidates.py)는 원 G10, d/G13, frozen-e 저널을 스트리밍하고 원 seal과 순서·ID·RGB·상태·선택 family를 확인한다. 원95장과 추가143장을 합치며, 원95장의 G10 cache와 e의 최종 같은-family 숫자 일치도 검사한다. 원자료는 수정하지 않는다.

| 원 단계 | family solution 수 | optimizer outcome 기록 | outcome에 R/t 누락 | 저장된 전체 feasible sample |
|---|---:|---:|---:|---:|
| G10 | 476 | 8,452 | **8,353** | **5,861** |
| d/G13 | 476 | 8,456 | **8,357** | **5,742** |

각 단계의 outcome 중 R/t가 있는 것은 99개다. R/t가 빠진 outcome이 있는 영상은 각 단계 233/238장이다. 누락은 비실현 후보만의 문제가 아니다. 원 outline consumer는 feasible 중복 제거 전 outcome에도 pose 없이 진단값을 기록한다. outcome 행 수, 고유 pose 수, 시작점 수를 같은 것으로 취급하지 않는다.

`feasible_pose_samples`는 원 consumer가 검사를 통과시키고 pose 중복을 제거한 뒤 남긴 **모든 저장 표본**이다. 추출기는 이 목록을 자르지 않는다. 이 목록으로 해당 원실행의 retained-feasible oracle을 분석할 수 있지만, 탈락한 생성 후보나 과거 탐색 전체의 oracle은 복원할 수 없다. 원 consumer가 fallback으로 이전 solver 결과를 그대로 반환한 경우도 그 위임된 원기록으로 명시한다.

후보의 R/t가 없으면 비용, 초기화 이름, 가장 가까운 다른 후보에서 pose를 추정해 채워 넣지 않는다. 과거 V15의 추가 시작점 은행과 V17의 후속 proposal 합집합을 원래 e 생성 은행으로 바꾸어 부르지 않는다.

## 2. 이번 로깅 재현이 보충하는 범위

[typed_replay.py](../../../scripts/research/pallet_statewise_correspondence_diagnosis_v1/typed_replay.py)는 원 G10 family 탐색 → 필요한 d/G13 refinement → 원 G10이 선택한 family의 d 결과라는 계층을 재현한다. 원 WD와 P1 시작점만 원순서대로 복원하며, 후속 V15가 추가했던 d/G10 winner를 새로운 원실행 시작점으로 넣지 않는다. 원 coarse prior, scalar/analytic backend, 목적함수, 제약, 반복 한도, 후보 순서, tie-break를 유지한다.

계측은 다음 위치에서 이미 계산된 R/t와 기록을 별도 ledger로 복사한다.

- 원 outline `consider()`의 unrefined 시작 pose와 최적화 끝점. feasibility 검사 및 원 고유-pose 중복 제거를 통과하지 못한 끝점도 포함한다.
- typed partial-only 경로의 `fit()`이 반환하는 BI seed 및 joint 최적화 끝점. 반환식을 한 번 평가하며 기존 목적함수나 제약을 추가 호출하지 않는다.

이는 **이번 원 e refinement consumer가 실제 반환·검사한 숫자 끝점의 로깅 범위**다. 예외 때문에 숫자 끝점을 반환하지 못한 경로는 원 오류 기록으로 남는다. 다음의 더 넓은 역사적 범위는 완전하다고 주장하지 않는다.

- 과거 WD/P1 solver가 처음 만들고 버린 모든 PnP·seed 후보.
- 초기값 생성 함수 안에서 usable 시작점 목록에 포함되지 않은 내부 중간값.
- SLSQP의 전체 내부 iterate 또는 탐색하지 않은 자세.

따라서 이번 replay endpoint ledger가 완전해도 **전체 역사적 generated bank가 완전해지는 것은 아니다**. 그러나 이 사실을 원 e consumer 은행의 무조건적인 누락으로 취급해서도 안 된다. 다음 source 경계가 중요하다.

### 2.1 과거 탈락 후보와 실제 e 입력의 경계

원 [G10 caller](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/evaluate_global_outline_pose_v4_10.py#L63)는 WD solution과 P1 solution을 전달한다. [G10 시작점 구성](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_global_outline_solver_v4_10.py#L197) 및 [G13 시작점 구성](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_partial_BI_outline_solver_v4_13.py#L221)은 **accepted winner, 모든 feasible sample, 명시적 supplied pose**만 읽는다. 과거 WD/P1의 `candidates`에 있는 rejected proposal은 읽지 않는다. 이 과거 탈락 pose들은 e가 비교하던 은행 바깥이며, 없다는 이유만으로 e의 bank를 불완전하다고 하지 않는다.

outline 경로에서는 원 dedup 전 incoming winner/sample/supplied 좌표를 봉인된 packet에서 순수 추출하고 원 `_same_pose`의 대표·탈락 정보를 보존한다. dedup을 통과한 모든 시작점은 원 `consider(..._UNREFINED)`를 거쳐 ledger에 기록되고, 이어지는 숫자 최적화 끝점도 기록된다. 이 consumer 범위의 전체 후보 검사가 가능하다. dedup에서 제외된 같은 pose 근방의 incoming 기록도 지우지 않는다.

238장 중 232장은 이 outline 경로다. 5장은 관측점 cardinality에 따른 원 admission/fallback 경로로 새 optimizer가 없다. 이때 생성 은행이 비어 있어도 최종 fallback pose는 따로 존재할 수 있으므로 generated → selected 성공 수의 단조성을 가정하지 않는다. fallback을 가짜 생성 후보로 넣지 않는다. 원 함수가 accepted saved solution을 그대로 반환한 별도 위임 경로에서는 그 정확한 cached winner를 들어온 후보로 보존하되, 새 optimizer 끝점으로 세지 않는다.

나머지 CAL 1장 `eval_cad:1778653003088339968`은 outline이 지원되지 않아 partial-only solver가 원 `_initial_poses`를 호출했다. 최초 계측은 그 fit 끝점은 모두 보존했으나 함수가 반환한 raw 시작점 목록을 별도로 남기지 않았다. 이 구현상 로깅 공백은 **같은 원 `_initial_poses`, 같은 xyz/K/visible/supplied/config**를 호출하는 [restore_starts.py](../../../scripts/research/pallet_statewise_correspondence_diagnosis_v1/restore_starts.py)로 참조를 읽기 전에 보충했다. ORIGINAL 51개·SWAPPED 50개의 수, 전체 이름 순서, 원 search metadata가 기존 cached 실행과 정확히 일치한다.

이 보충은 초기값 생성 함수 2호출(P3P 20회, SQPNP 2회)의 반복이다. 새 seed·시작점 정책·후보 예산·학습·최적화는 없으며, `minimize`/`least_squares`/새 top-level pose slot/전체 replay는 각각 0회다. 실제 호출·입력·source SHA와 시간은 `RAW_START_SUPPLEMENT.json` 및 `SUPPLEMENT_SEAL.json`에 기록하고 원 optimizer 결과는 그대로 재사용한다.

최종적으로 full strict parity, 각 consumer의 outcome/ledger 수, incoming 및 이 CAL 보충 seal이 모두 통과한 영상은 `original_consumer_generated_bank_complete=true`로 보고할 수 있다. 이것과 `entire_historical_generated_bank_complete=false`는 서로 모순되지 않는다. PoseFix, N0 또는 새 SINGLE의 좋은 pose를 원래 e 후보에 섞지 않는다.

## 3. G10과 G13을 분리하는 이유

G10은 원 complete B/I 및 legacy partial과 outline으로 family를 정한다. d/G13은 추가 complete likelihood를 철회하고 typed partial normal/finite tangent 항으로 refinement한다. 두 단계의 관측·목적함수가 다르므로 G10과 G13의 비용을 그대로 비교하거나, G10에 있던 좋은 후보가 G13에 없다는 이유만으로 순수 `PRUNING_LOSS`라고 부르지 않는다.

다음 두 사다리를 별도로 보존한다.

| 단계별 사다리 | 같은 단계 안에서 비교하는 집합 |
|---|---|
| G10 | logged generated → 원 support predicate → 원 retained feasible → 원 G10 선택 |
| G13 | logged generated → 원 support predicate → 원 retained feasible → 원 d 선택 |

typed 변경이 없었던 48장은 원 d가 G10 결과 전체를 그대로 재사용한 control이다. 이때 G13 표의 generated/support는 **동일 G10 cache의 명시적 재사용**이며 새 G13 최적화로 세지 않는다.

주 oracle의 `generated`, `after_support`, `feasible`은 최종 G13 패키지 안에서 비교한다. 그 뒤 `family`는 G10이 고정한 family 안의 G13 feasible 후보, `refinement`는 그 family의 d 선택 또는 원 fallback, `selected`는 원 frozen-e 최종 출력이다. G10의 family 선택과 G13의 내부 목적함수 차이는 `NOT_ISOLATED`로 함께 기록한다.

`after_support`는 원 `_visible_audit`의 geometry/canvas predicate와 원 `_boundary_audit`의 finite partial support predicate를 같은 저장 끝점에 사후 적용한 분해다. 새 threshold나 guard가 아니다. 원 프로그램이 이 순서의 별도 필터를 실행했다는 주장도 아니다. 계산에 필요한 기록이 부족하면 UNKNOWN으로 남긴다.

## 4. 분류에 허용되는 주장

- `SELECTION_LOSS`: 원 retained-feasible 집합에 동일 후보 공동 성공이 실제로 남아 있는데 최종 출력은 실패한 경우. 다른 family와 같은 family 내부 선택을 구분한다.
- `PRUNING_LOSS`: 최종 출력이 실패했고, 같은 최종 G13 패키지의 로깅된 공동 성공 끝점이 원 guard에서 실제 거부됐으며, 그 단계의 retained-feasible 은행에 공동 성공 후보가 하나도 남지 않은 경우. 다른 좋은 feasible 후보가 남아 있거나 최종 출력이 성공했다면 거부된 좋은 끝점의 수만 별도 진단으로 남긴다. G10 → G13 관측 변경을 이 이름에 포함하지 않는다.
- `REFINEMENT_LOSS`: 같은 stage/family/branch/start의 원 unrefined 후보가 공동 성공인데 그 뒤 최적화 끝점은 실패한 경우의 **후보 수준** flag다. 시작점 이름만 비슷한 다른 pose를 연결하지 않는다. 좋은 후보가 따로 남아 최종 출력은 성공한 영상에도 이 flag가 있을 수 있다.
- `CANDIDATE_RECORD_INCOMPLETE`: **실제 e consumer 범위**의 생성·incoming 기록 부족 또는 replay 불일치로 그 은행의 판정이 불가능한 경우. e가 읽지 않았던 과거 rejected WD/P1 proposal의 누락은 이 flag의 독립 근거가 아니다. 이 flag는 retained-feasible에서 관찰한 선택 손실 등 다른 flag와 함께 존재할 수 있다.
- `OBSERVATION_OR_PROPOSAL_LIMIT`: 위 입력·끝점·보충·parity 검사로 전체 e consumer 은행이 확인됐고, 그 은행에 동일 후보 공동 성공이 없으며, 최종 출력도 실패한 경우. 이 고정 실행의 실제 consumer 범위에 관한 판정이며 다른 관측·모델·탐색으로도 불가능하다는 뜻이 아니다.
- `REFERENCE_OR_REGISTRATION_UNRESOLVED`: 고정 참조의 회전 해석 등이 실제로 미확정인 경우 별도 flag로 보존한다. 큰 회전 오차 자체를 참조 오류의 증거로 바꾸지 않는다.

전체 역사적 generated bank는 여전히 불완전하다. 일부 기록에 성공이 없다는 이유만으로 `OBSERVATION_OR_PROPOSAL_LIMIT`을 확정하지 않으며, **실제 consumer 범위의 전량 확인**이 필요하다. `OBSERVED_REFINEMENT_OR_PRUNING_LOSS` 역시 최종-stage generated와 retained 집합 사이의 관찰이며, 탐색하지 않은 모든 자세의 원인 확정이 아니다. earliest observed stage와 추가 flag를 함께 보고하고 주원인 하나로 억지 분류하지 않는다.

집계의 `failure_flags`는 전체 분모의 coverage 및 후보 수준 flag 발생 수다. 이를 최종 실패 영상 수로 읽지 않는다. 최종 실패 영상 안의 수는 `failure_flags_final_failure_only`, 최종 성공 영상 안의 수는 `failure_flags_final_success_only`로 따로 기록하고, 각각의 성공·실패·참조 미확정 분모를 함께 제시한다. `rejected_good_endpoint_diagnostic`은 좋은 끝점의 거부가 일부 있었던 영상·끝점 수이며 전체 pruning 손실과 구별한다.

## 5. 원자료·새 봉인·재현 검사 연결

검사 순서는 다음과 같다.

1. `CANDIDATE_MANIFEST.json`: 원 저널/protocol/seal, 원 consumer source SHA, 추출 저널 SHA와 정확한 238 ID 순서를 연결한다. `CANDIDATE_COVERAGE.json`은 위 원기록 공백과 무절단 feasible 범위를 보존한다.
2. `e_trace_v2/EXTRACTION_SEAL.json`: 실제 e 이전 관측, 원 typed packet, 원 prior 및 E0 numeric bridge를 연결한다. e 이후 재관측을 원 e 입력으로 대체하지 않는다.
3. 새 `PREDICTION_SEAL.json`과 `CANDIDATE_SEAL.json`: 네 경로의 238장과 로깅된 candidate bytes를 **새 참조 점수 읽기 전** 봉인한다. candidate seal은 원 candidate manifest, strict full parity receipt, CAL raw-start 보충 seal SHA도 연결한다. 보충 start는 원 endpoint ledger와 별도 파일로 보존한다.
4. [parity.py](../../../scripts/research/pallet_statewise_correspondence_diagnosis_v1/parity.py): 원 최종 pose만이 아니라 G10/G13 양 family의 outcome, 전체 feasible sample, 목적함수·구성 항·search·rank·ambiguity 및 source/accept/fallback을 검사한다. 고정 절대 허용치는 `1e-9`, 상대 허용치는 0이며 결과를 본 뒤 완화하지 않는다. 부가 U-fallback metadata 필드의 포장 차이는 숫자·선택 parity와 분리해 명시한다.
5. [oracle.py](../../../scripts/research/pallet_statewise_correspondence_diagnosis_v1/oracle.py): 명시적 postseal GO와 prediction/candidate/lineage/reference 및 start 보충 hash 검사를 통과한 뒤에만 고정 참조를 읽는다. `ORACLE_SEAL.json`이 사용한 parity receipt, 원 manifest, 추출·보충 seal, 참조·annotation 및 분석 source SHA를 다시 연결한다.

full parity가 실패한 영상의 replay endpoint는 원 e 생성 후보로 승격하지 않고 oracle에서 보류한다. 원저널에서 확인되는 feasible·선택 진단은 계속할 수 있지만, 그 영상은 `CANDIDATE_RECORD_INCOMPLETE` 및 `replay_endpoints_withheld_on_parity_failure` 상태를 유지한다. 전체 PRIMARY192/CAL46 분모에서 실패 영상을 제거하지 않는다.

## 6. 최종 결과를 기록할 위치

참조 분석이 승인·완료된 뒤의 실제 도달 가능 성공 수는 `ORACLE_SUMMARY.json`의 `populations`에 CAL46, TEST49, 추가143, PRIMARY192로 별도 기록한다. `original_stage_oracles`가 G10/G13 사다리를 분리한다. cohort별 consumer complete/unknown 장수와 row별 `original_consumer_generated_bank_complete`를 보고하고, 역사적 범위의 별도 false flag를 이 수와 섞지 않는다. 전수 행은 `ORACLE_ROWS.jsonl.gz`와 `ORACLE_FRAME_TABLE.csv`, 기존 여섯 진단 사례는 `DIAGNOSTIC_CASE_STAGES.json`에 남긴다.

`SINGLE_FALLBACK_AUDIT.json`은 참조 없이 SINGLE의 실제 fallback과 자기 E0의 숫자 일치를 따로 검사한다. 원 backend의 위임이나 fallback을 새 E0 국소 fit으로 오해하지 않으며, 자기 E0와 다른 fallback이 있으면 strict-local 해석 제한을 표시한다. 이 검사를 위해 새로운 fallback 정책이나 자세 제한을 도입하지 않는다.

이 정적 문서 작성 시점에는 새 oracle 참조 평가를 실행하지 않았다. 최종 수치와 hash는 봉인 후 산출물 및 최종 보고서에 덧붙이며, 위 원기록 공백과 정의를 결과에 맞추어 바꾸지 않는다.

## 7. 봉인 후 실제 결과

전체 네 경로 952 slot에서 기술 실패는 0건이었고, TYPED_MULTI 238장 모두 원 G10/G13 양 family·전체 저장 outcome/feasible sample·최종 pose에 대한 고정 `1e-9` strict parity를 통과했다. 위 원 consumer 범위와 CAL 보충을 적용한 coverage는 **CAL46/46, TEST49/49, 추가143/143, PRIMARY192/192 complete**, UNKNOWN 0장이다. 더 넓은 역사적 WD/P1 내부 생성 기록은 여전히 불완전하며 그 rejected proposal은 e consumer 밖이다.

아래 숫자는 **같은 후보의 위치≤5cm AND physical/C2 회전≤5°**를 만족하는 영상 수다.

| 집합 | generated | after support | retained feasible | 고정 family 안 feasible | 같은 family d 선택/fallback | e 최종 |
|---|---:|---:|---:|---:|---:|---:|
| CAL46 | 40 | 40 | 40 | 37 | 37 | 37 |
| TEST49 | 49 | 49 | 49 | 49 | 49 | 49 |
| 추가143 | 97 | 97 | 97 | 87 | 65 | 65 |
| PRIMARY192 | **146** | **146** | **146** | **136** | **114** | **114** |

PRIMARY의 최종 실패 78장은 이 고정 은행 기준에서 다음처럼 나뉜다.

- **46장**: 확인된 원 e consumer generated 은행에 공동 성공 후보가 없다. 이 안에는 admission으로 후보를 생성하지 않은 4장이 포함된다. 다른 모델·관측·탐색으로도 불가능하다는 주장이 아니다.
- **32장**: retained-feasible 은행에는 공동 성공 후보가 있지만 e 최종 선택은 실패한다. 최초 관찰 손실은 고정 family 선택 10장과 같은 family 안 최종 선택 22장이다.
- **전체 좋은 후보 소실로서의 `PRUNING_LOSS`는 0장**이다. 일부 좋은 끝점은 거부됐어도 좋은 feasible 후보가 남았다. 좋은 끝점 거부가 있었던 133장/343끝점을 전체 pruning 손실로 세지 않는다.

146/192는 고정 참조를 본 분석용 oracle이다. 새 image-only selector의 성능도, 32장을 실제로 회복한 결과도 아니다. G10 자체 사다리는 PRIMARY generated/support/feasible 143 → G10 선택 116, G13 자체 사다리는 146 → d의 자체 family 재선택 결과 112다. 원 e는 G10 family를 고정한 뒤 같은-family d를 사용하므로 최종 114이며, 이 세 선택 정책의 차이를 섞지 않는다.

`REFINEMENT_LOSS` 후보 수준 flag는 PRIMARY 124장에 있었지만, 최종 실패 안에서는 31장이고 **최종 성공 안에서도 93장**이다. 이는 일부 시작점이 최적화 후 손상됐다는 뜻이며 124장의 최종 자세가 손상됐다는 뜻이 아니다. 이 수를 46+32에 더하지 않는다.

고정 여섯 사례 중 `plastic_day_01:010995`는 원 e와 은행 모두 공동 성공이었다. `plastic_day_01:011640`, `wood_day_01:005249`, `wood_night_01:029428`, `wood_night_01:031344`, `wood_night_01:031426`은 이번 원 e consumer generated 은행에 공동 성공 후보가 없었다. 이 다섯 사례를 기존 후보 선택만 바꾸면 해결되는 사례로 설명하지 않는다. source/physical 축 및 영상 의미 대응의 별도 한계는 그대로 유지한다.

전체238의 SINGLE fallback은 TYPED 5건, REOBS 90건이며 **모두 자기 E0와 `1e-9` 내 일치**했다. E0와 다른 fallback은 0건이다. optimizer 호출이 없었던 영상은 각각 5장/16장으로 별도 기록한다.

실제 oracle 후처리는 110.472초(과정 CPU 110.388초)였고 새 pose fit은 0회다. 결과의 hash 연결은 다음과 같다.

- prediction seal: `b7028b2ce160298e274a7035f3ea6367bcf0e14bcf3c8a5b372823c4d3b4b148`
- candidate seal: `21673626100d4fb589a9d922844fe30a78bd4c68c73d95b5f6dd991437fce8cb`
- 고정 reference manifest: `929c22a9f5298cd8bfe793fe9536b70c1b208e6f49a173f32cf7edfed532c06c`

전수 행·후보 단계·분모·fallback 및 세부 input/source/artifact hash는 `ORACLE_SUMMARY.json`, `ORACLE_ROWS.jsonl.gz`, `ORACLE_FRAME_TABLE.csv`, `DIAGNOSTIC_CASE_STAGES.json`, `SINGLE_FALLBACK_AUDIT.json`, `ORACLE_SEAL.json`에 보존한다. TEST49와 추가143은 반복 개발 자료이며 이 결과를 새로운 독립 검증으로 부르지 않는다.
