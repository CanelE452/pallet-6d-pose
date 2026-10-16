# v5 프레임워크를 v6에 재사용할 때의 정적 감사

검토한 원본은 게시된 v5 `pallet_partial_line_independent_20261010_v5`이며, 후속 검토에서는 새 v6 framework의 실제 소스도 읽었다. source code와 완료 보고서 문구만 읽었고 기존 v5에 새 오류가 있다고 소급 주장하지 않는다. 이번 감사의 모델/PnP/optimizer/ray/GT scoring/원행 통계 재실행·알고리즘 코드 수정은0이다. 이 파일과 EVALUATION_CONTRACT_KO.md만 새 namespace에 작성했다. 아래 구현 판정은 정적 읽기의 범위이며 새 CPU·실사·runtime PASS receipt를 대신하지 않는다.

## 의미가 달라지는 재사용 지점

| v5 위치 | 현재 가정 | v6에서 필요한 계약 |
|---|---|---|
| [evaluate.py:22](../../../scripts/research/pallet_partial_line_independent_20261010_v5/evaluate.py#L22) | 과거v4 point-only geometry 경로·method hardcode | 과거v5 C2 GT-free geometry와 해당 control alias를 protocol에 바인딩; `V5_CONTROL_PARITY` 의미로 pre-reference 비교 |
| [evaluate.py:58](../../../scripts/research/pallet_partial_line_independent_20261010_v5/evaluate.py#L58) | control parity의 point IDs·state·reason·prior·Rt 중심 | unchanged C2의 linepool, consumed/final lineinlier, point+line score와 endpointRMS도 scope에 포함; 허용오차와 bitexact 구분 |
| [statistics.py:223](../../../scripts/research/pallet_partial_line_independent_20261010_v5/statistics.py#L223), [verify.py:295](../../../scripts/research/pallet_partial_line_independent_20261010_v5/verify.py#L295) | primary−oldv4point 대비 contrast hardcode | primary−unchangedv5C2로 대체; nativeH/nomask/BASE/N3 contrast와3scope/3strata/기존draw 유지 |
| [statistics.py:270](../../../scripts/research/pallet_partial_line_independent_20261010_v5/statistics.py#L270), [verify.py:255](../../../scripts/research/pallet_partial_line_independent_20261010_v5/verify.py#L255) | V4_CONTROL_PARITY 파일과 receipt key | 새 v5 control parity 경로·key·SHA로 정확히 연결; 기존v4 PASS를 v6 control PASS로 대용하지 않음 |
| [verify.py:144](../../../scripts/research/pallet_partial_line_independent_20261010_v5/verify.py#L144), [public_review.py:67](../../../scripts/research/pallet_partial_line_independent_20261010_v5/public_review.py#L67) | boundary arm tuple에 PRIMARY+oldv4point | PRIMARY+oldv5C2 모두 boundary LOO/채택ID 검산·CSV 대상; native-only arm과 구분 |
| [render_repair.py:49](../../../scripts/research/pallet_partial_line_independent_20261010_v5/render_repair.py#L49) | 사례 N3/v4point/v5C2 | 고정6case ID 그대로 N3/v5C2/v6main 표시; 새 score로 사례를 선택하지 않음 |
| [render_repair.py:56](../../../scripts/research/pallet_partial_line_independent_20261010_v5/render_repair.py#L56) | PRIMARY만 line overlay | v5C2 control과 v6의 실제 used/final line을 각각 표시; gate REJECT/UNVERIFIED와 final inlier를 혼동하지 않음 |
| [line_checks.py:166](../../../scripts/research/pallet_partial_line_independent_20261010_v5/line_checks.py#L166) | self-consistent source-supported−consumed = 모든 final pool | v6 gate전 pool, consumed, heldout supported/reject/unverified-retained, finalpool을 별도 유도; 기존 원식 그대로 사용하면 정당한 제외를 실패로 오판 |
| [line_checks.py:293](../../../scripts/research/pallet_partial_line_independent_20261010_v5/line_checks.py#L293) | empty-line delegate 결과 = oldv4point control | v6 control은 v5C2라 선이 남을 수 있음. v4 point-only delegate를 별도 확인; v5control과 동일하다고 요구하지 않음 |
| [line_quality.py:25](../../../scripts/research/pallet_partial_line_independent_20261010_v5/line_quality.py#L25), [line_quality.py:418](../../../scripts/research/pallet_partial_line_independent_20261010_v5/line_quality.py#L418) | POINT_ONLY alias·vs_point_only_same_observations | control 이제 C2임을 명명/바인딩; 양 C2의 source/consumed/gate/finalpool/inlier proxy품질을 각각 기록 |
| [runtime.py:123](../../../scripts/research/pallet_partial_line_independent_20261010_v5/runtime.py#L123) | legacy v4 schema, v5 경로 boundary | 새 v6 schema/경로/새line-heldout 호출 포함. inherited label은 exact namespace 의미와 구분하며 공식600 cost를 옛 cache 시간으로 대체하지 않음 |

## 그대로 보존할 점수화 계약

[v5 evaluate.py:85](../../../scripts/research/pallet_partial_line_independent_20261010_v5/evaluate.py#L85)의 완전 seal·cleanup receipt·GT-free fields·245 ID·각 method245행 guard는 보존한다. methods가4개이면 geometry980+fixed490=1470의 counting 계약도 그대로다. 새 v5control parity를 [148행](../../../scripts/research/pallet_partial_line_independent_20261010_v5/evaluate.py#L148)에 해당하는 **reference 읽기 전** 위치에서 실행한다. 과거 geometry는 parity diagnostic이며 신규 pose/gate의 input이 아니다.

[187–220행](../../../scripts/research/pallet_partial_line_independent_20261010_v5/evaluate.py#L187)의 fresh fixedN3 branch/permutation/native GTproxy와 human state mapping은 유지한다. NOMASK arm만 mask_applied=False로 놓고 fixedcontrols도 mask성공/실패로 집계하지 않는다. H와 직접 가시·SELF·actualreprojected를 분리하고 fallback raw candidate inlier를 최종 승인inlier로 쓰지 않는다. scoring canary와 새 fits0을 유지한다.

[v5 statistics.py:96](../../../scripts/research/pallet_partial_line_independent_20261010_v5/statistics.py#L96)와 [verify.py:94](../../../scripts/research/pallet_partial_line_independent_20261010_v5/verify.py#L94)는 ref.matched를 known 판정에 직접 반영하지 않는다. 기존 동일245의 matched1470/1470 receipt가 있어 이번 기존 결과의 오류는 아니지만, v6에서도 allmatched를 명시 guard하거나 unmatched는 UNKNOWN으로 처리해야 한다. 새 label/GTmatching을 만들자는 뜻은 아니다.

mean·samplevar(ddof1)·SD·median·P90·max, 전체operational/new/fallback, common/candidateNEW/bothNEW, 기존10000×13draw·비영/empty denominator 처리는 재사용한다. fixedcontrol의 bothNEW n=0은 pose부재가 아니라 status 정의다. raw method와 rendered label, artifact path·metadata schema·actualcount를 정확히 일치시킨다.

## 수정 head와 role 의미의 provenance

현재 head는 [v2 pipeline.py:177](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py#L177)의 corrected9000 완료조건과 [184행](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py#L184)의 last ROLE SHA로 강제된다. [280행](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py#L280)은 unchanged Base anchor에서 model.inputs와 해당 head를 호출한다. v6 loader는 동일 completion/checkpoint/CAL binding을 유지해야 한다. 과거0/319 NEW snapshot을 수정학습 이후의 현재 상태로 쓰지 않는다.

historical [training.targets():59](../../../scripts/research/pallet_observation_refiner_20261009_v1/training.py#L59)의 finitehitclose 실패→[94행 no_match](../../../scripts/research/pallet_observation_refiner_20261009_v1/training.py#L94)는 보존된 연구 source supervisor 코드다. 수정 경로는 [retrain.load_arrays():100](../../../scripts/research/pallet_kp_supervision_repair_20261010_v1/retrain.py#L100)/103의 READY np.load, [152행 SHA](../../../scripts/research/pallet_kp_supervision_repair_20261010_v1/retrain.py#L152), [376행 repaired_target SHA](../../../scripts/research/pallet_kp_supervision_repair_20261010_v1/retrain.py#L376)를 사용한다. 이는 고정 main N3→SubPix correction 함수가 잘못됐다는 증거가 아니다.

role채널은 [model.py:47](../../../scripts/research/pallet_observation_refiner_20261009_v1/model.py#L47)의 initial virtual projected hull/internal/unavailable이며 실제 physical boundary owner의 정답이 아니다. source-supported wire와 source CAL uncovered edge를 실사 physical absence로 일반화하지 않는다. Baseproposal/initialN3 H 의존성이 남아 있음을 보존하며 numeric endpoint제외를 full statistical independence로 확대하지 않는다.

## pre/post GT freeze와 artifact 위치

v6 core PROTOCOL은 새 accuracy/reference 접근 **이전**에 구현·정책·CPUtest·이문서·입력 SHA를 freeze한다. v5 LINE_QUALITY/LINE_HELDOUT은 score/statistics **이후**, 각 저장행 산술 실행 전에 code/input을 freeze한 supplemental audit다. 이 순서를 뒤집어 과거 진단을 preregisteredaccuracy라고 보고하지 않는다. 새 v6의 후행 산술도 같은 구분을 지킨다.

새 common DOC/PRIVATE, archive restore/reprojection/publicreview defaults, source-bound controlgeometry, schema, receipt keys와 README/RESULT/figure link를 v6으로 바꿔야 한다. v5 코드/문서/실패 receipt/원 gzip는 변경하지 않는다. protocol에 묶인 v5 큰geometry를 public archive에서 먼저 복원해야 새 public verifier가 privateasset 대용 없이 같은 binding을 확인할 수 있다. 새 protocol/row binding helper가 in-repository 경로를 요구하면 /tmp protocol 예제를 쓰지 않는다.

## 실제 v6 framework의 후속 정적 검토

[common.py:16](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/common.py#L16)의 PRIMARY는 `N3_INDEPENDENT_CORNERWISE_ROLE_ENDPOINT_VALIDATED_LINES`, control은 변경 없는 v5 `N3_INDEPENDENT_CORNERWISE_ROLE_PARTIAL_LINES`다. [46–71행](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/common.py#L46)은 새 framework, borrowed v4/v5 solver, frozen ROLE/CAL·cohort·bootstrap draw·이 평가 계약·v5 geometry/protocol·CPU receipt를 input binding으로 묶는다. [114행 canary](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/common.py#L114)는 inference 중 score·annotation·target·기존 observation/geometry 경로를 금지한다. checksum 검사는 canary 밖에서 수행하므로 읽기 전용 SHA 검사를 pose 입력 사용으로 혼동하지 않는다.

[pipeline.py:38](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/pipeline.py#L38)은 두 boundary arm의 기존 corner selection을 공유하고, [45–50행](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/pipeline.py#L45)은 PRIMARY에만 새 wrapper, control에는 변경 없는 v5 PointLineBank를 사용한다. wrapper의 [solver.py:202–207](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/solver.py#L202)은 native N3 bank의 `excluded=caller_temporary∪{a,b}`, `hidden=H`, `robust=True`로 정확히 한 검증 solve를 요청한다. [89–132행](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/solver.py#L89)은 prior-free NEW 상태, 실제4점, used·selected generator/fit/inlier와 저장 candidate branch의 제외 증거가 불완전하면 UNVERIFIED로 처리한다. [217–247행](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/solver.py#L217)은 고정8px SUPPORTED_RETAIN / CONTRADICTED_REJECT / UNVERIFIED_RETAIN을 구분하고 원 관측의 deep copy에서 contradicted unused edge만 제거한다. consumed 선은 gate 대상이 아니며 원 관측 byte binding과 consumed support를 보존하는 assert가 [259행](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/solver.py#L259)에 있다.

이 제외 증거의 범위는 **활성 검증 후보·scoring·refit**다. 전역 numeric cache에는 다른 mask에서 제외 ID를 쓴 primitive 생성 결과가 있을 수 있고, [v4 pose.py:186](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py#L186)은 generator ID가 현재 allowed pool의 부분집합인 후보만 사용한다. [124–132행 equivalence](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py#L124)의 all8 비교는 두 후보의 model projection·physical R·정규화 t를 비교한다. heldout endpoint의 실제 관측 xy를 equivalence 잔차에 넣는 경로는 이 함수에 없다. 따라서 numeric validation의 활성 입력 제외를 확인했지만, 제외 좌표가 포함된 primitive를 전역 cache에서 전혀 생성하지 않았거나 시스템 전체가 통계적으로 독립이라는 주장은 하지 않는다. 잘못된 남은 N3 합의에 검증 자세가 맞을 가능성도 남는다.

초기 재사용 버전의 C2 parity는 존재하지 않는 옛 field name을 `.get()`으로 비교해 양쪽 None가 같아질 수 있었다. 현재 [evaluate.py:58–92](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/evaluate.py#L58)는 실제 `fit_line_edges`, `point_inlier_count`, `unique_line_inlier_count`, `total_inlier_factor_count`, `factor_pool`, empty delegate와 actual RMS/SSE·candidate/optimizer packet의 **존재 여부와 값**을 비교하며 `C2_fields_present`를 남긴다. available+nonempty line pool의 필수 C2 field도 별도 요구한다. [168행](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/evaluate.py#L168)의 v5 control parity가 [176행 reference](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/evaluate.py#L176)보다 앞이고 geometry는 pose 입력으로 쓰지 않는다. 실패는 별도 receipt 저장 후 중단하도록 작성되어 있다. 이것은 소스 판정이며 실제245 parity PASS는 아직 확인하지 않았다.

[evaluate.py:111](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/evaluate.py#L111)은 inference complete·cleanup_error=None·245/980/490을 확인하고, [115–151행](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/evaluate.py#L115)은 seal binding, method/ID population, GT-free fields와 prior-free flags를 확인한다. [209행](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/evaluate.py#L209), [statistics.py:87](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/statistics.py#L87), [verify.py:90](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/verify.py#L90)의 matched guard는 unavailable proxy를 known negative correspondence로 바꾸지 않는다. NOMASK만 mask_applied=False인 mapping과 fixedN3 phase의 DIRECT/SELF/H 분리는 유지된다. [statistics.py:224](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/statistics.py#L224)와 [verify.py:296](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/verify.py#L296)의7contrast는 PRIMARY−v5C2를 포함하며 기존3scope/3strata/10000×13 draw 계약을 유지한다.

public CSV의 초기 `RMS_px` 참조는 현재 [public_review.py:67](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/public_review.py#L67)의 실제 `endpoint_RMS_px`로 정정되었다. [render.py:49](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/render.py#L49)은 기존 고정 사례 ID를 N3/v5C2/PRIMARY에 사용하고, [59행](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/render.py#L59)은 heldout rejected 선을 최종 inlier와 다른 표현으로 표시하도록 작성되어 있다. fixed control의 solver=None도 metadata 접근에서 처리한다. [runtime.py:123–128](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/runtime.py#L123)은 새 v6 schema와 실제 dualendpoint validation·final C2 경로를 명시하고, pipeline [55–56행](../../../scripts/research/pallet_partial_line_heldout_20261010_v6/pipeline.py#L55)은 line-heldout 호출 수를 별도 기록한다. runtime 결과를 이미 산출했다는 뜻은 아니다.

## 검토 범위와 아직 실행으로 증명하지 않은 부분

- [x] 정적 소스에서 alias/input binding, native N3 dualendpoint+H 호출, active candidate/fit/scoring 제외, 상태 구분과 C2 filtered pool 경로를 확인했다.
- [x] pre-reference v5 C2 parity, complete cleanup/seal guard, matched proxy guard, fixedN3 phase,7contrast와 새 runtime/CSV 의미를 확인했다.
- [x] all8 model equivalence와 전역 cache의 제외 ID 포함 가능성, 남은 H/Base 의존성·물리 소유권 미인증을 구분했다.
- [ ] 새 CPU synthetic 테스트·endpoint validation checker의 실제 PASS receipt와 code SHA 일치는 이 정적 감사에서 실행·확인하지 않았다.
- [ ] 새245 geometry·control parity·점수·원행 moments/CI·H재투영·fresh600 runtime·공개 archive restoration은 아직 이 감사의 실행 증거가 아니다.
- [ ] v5 line_checks의 pool 식을 새 gate에 맞춘 독립 산술 checker가 실제 확인했는지는 후속 validation_checks receipt를 따른다. 이전 PASS를 새 gate의 PASS로 대용하지 않는다.

현재 읽은 framework에서 추가 실질 blocker는 찾지 못했다. 이는 구조와 guard의 정적 검토 결과이며 정확도 개선, 전역 자세 유일성, 실사 물리 경계 소유권, 새 실행 완료를 증명하지 않는다. 새 v6 accuracy/성공을 보고하지 않는다.
