# V9 같은 sparse 관측의 원래 C2 지역 점·선 진단 재현 계약

이 파일은 실사 기하·성능 평가 전에 고정하는 재현 계약이다. 아래 명령은 자동 실행되지 않는다. 실제 실행 상태, 원래 CPU 실패와 보충 검사, 호출량, 자세 성능 및 게시 SHA는 각 receipt와 별도 결과 보고서를 확인한다. 이미 완료된 공식 실행의 파일을 덮어쓰거나 성능을 보고 설정을 바꾸어 재실행하지 않는다.

## 필요한 입력과 보호

공개 V7의 완전한 geometry1,960행, observations735행, fixed490행, protocol/seal/inference receipt와 Clean153+Moderate92의 고정245 cohort가 필요하다. Severe74는 이 신규 실행에 포함하지 않는다. 동일 V7 sparse q의 공개 V8 `ROLE_BOUNDARY_H_ROBUST`245행을 comparator로 사용하며, V8 전체735 geometry와245 ledger, 봉인·cleanup·독립 검산·채점 receipt·원행도 byte bind한다. V7와 V8의 압축 자료를 복원하면 기존 SHA256와 bytes가 정확히 일치해야 한다.

V9 `PROTECTION_BEFORE.json`은 V8 게시 commit `07d5f5c616956c70717f3df9ef0dd1f4c439169e`의 tracked 파일, 추가 완료 자료와 원래 사용자 작업본을 보호한다. 더 오래된 보호 snapshot은 사용할 수 없다. 고정 입력 전체 목록은 새 `PROTOCOL.json`의 `inputs`를 따른다. GT targets, 기존 scored prediction과 visibility audit은 실사 fit 전에 SHA만 고정하고 관측·start·치수·해 선택에 decode하지 않는다.

source와 baseline은 서로 다른 checkout이다. 환경 변수와 CLI에 각각 같은 실제 경로를 전달한다. baseline은 기존 `a22fb14beb5e8df08076385000e0d53503c1ae29` dependency다. 최소 환경 context는 기존 assemble의 hidden-mask import를 지원하며 모델·checkpoint를 초기화하지 않는다.

```bash
export PALLET_SOURCE_ROOT="/actual/original/source/checkout"
export PALLET_BASELINE_ROOT="/actual/immutable/a22fb14/baseline/checkout"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
v7_input="/path/to/complete/V7/accuracy"
v8_input="/path/to/complete/V8/controls"
v9_output="/tmp/pallet-sparse-local-line-private-20261010-v9/geometry"
v9_protocol="_docs/experiments/pallet_sparse_local_line_20261010_v9/PROTOCOL.json"
mkdir -p "$v9_output"
```

공개 원본 자료가 DOC에 있으면 V7/V8 input을 각 실험 DOC의 절대 경로로 지정한다. 새로운 V9 output은 전용 DOC 또는 위 PRIVATE 하위 디렉터리만 허용하며 parent/source/baseline/fits와 겹치지 않는다.

## CPU 증거와 core freeze

CPU 검사의 원래 solver/pipeline/14 fixture 코드와 protocol, STARTED, 실제 receipt는 먼저 보존한다. 기대값을 정정하는 보충 fixture가 필요하면 원래 파일을 변경하지 않고 별도 코드·protocol·STARTED·receipt를 사용한다. 성공한 joined 검산은 원래 통과 항목과 보충 항목을 연결할 뿐이며 원래 실패·오류·호출량을 지우지 않는다. `JOINED_LOCAL_CONTRACT_CHECKS.json`이 명시적인 CPU 성공 권한이며, freeze는 그 joined 자료와 원래·보충 증거의 byte binding을 모두 확인해야 한다. CPU 명령과 자체 protocol은 `test_solver.py` 및 보충 검사 CLI를 따른다. CPU 완료 전에 core freeze 또는 실사 fit을 시작하지 않는다.

core는 one arm `ROLE_BOUNDARY_LOCAL_POINT_LINE`만 고정한다. model/N3/head/decoder/CAL forward와 새 initial PnP, 새 training, 새 RGB는0이다. 실제 SciPy 지역 최적화는0으로 기록하지 않는다. 초기 N3와 available point-only 자세는 LOCAL 시작점이며 residual prior가 아니다. 두 등록 치수를 동일하게 시도하고 whole fixed factor pool의 soft_l1 cost로 선택한다. initial rank는 진단만이며 최종 observed/model-normal J rank6, positive depth, finite optimizer와 distinct tied local solution 처리가 필요하다. `NEW`는 수치 LOCAL 산출이며 정확한 자세나 global unique 성공의 증명이 아니다.

```bash
python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.runner freeze \
  --parent "$v7_input" --point-parent "$v8_input" --output "$v9_output" \
  --protocol "$v9_protocol" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"

python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.runner preflight \
  --parent "$v7_input" --point-parent "$v8_input" --output "$v9_output" \
  --protocol "$v9_protocol" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"

python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.runner run \
  --parent "$v7_input" --point-parent "$v8_input" --output "$v9_output" \
  --protocol "$v9_protocol" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"
```

실제 authority 파일의 위치가 default와 다르면 CPU CLI 인자로 명시하고 모든 단계에 같은 경로를 전달한다. 자동으로 유리한 receipt를 고르지 않는다. freeze는 full V7 population과 V8 comparator join을 검사하고 원래 V8 proof schema를 그대로 nested evidence로 보존한다. preflight는 독립적인 CPU/GPU workload·온도 snapshot을 남긴다. 경쟁 작업이 발견되면 실패 snapshot과 실제 prefix를 보존하며 자동 retry하지 않는다.

정상 완료의 예정 수량은 새 local geometry245행, ledger245행, actual bank245개와 logical local path245개다. 한 프레임의 두 source×두 registry dimension의 logical start는 최대4개이며, 정확히 동일한 start와 factor pool의 cache hit는 실제 optimizer call과 구분한다. native N3는 V7 OBS의 원본에서 얻고 display/fallback용으로 보존하며 sparse 결측을 채워 fit하지 않는다. actual point에 소비된 source edge는 line factor로 재사용하지 않는다. 선0+점3의 LOCAL 산출은 선의 효과와 분리한다. H의 초기2D를 fit에서 제외하고 NEW 이후 한 번 재투영하며 재fit하지 않는다.

`CONTROL_RECEIPT.json`은 SciPy 진입·완료·예외·residual/J callback, 전체 OpenCV primitive, 모델/GT/asset canary, 모든 start/candidate/rank/inlier witness, streaming close/fsync, 환경·monkeypatch cleanup, 보호 및 resource snapshot을 연결한다. 실패시 `INTERRUPTED_*`와 active packet의 실제 prefix를 남긴다. geometry full witness와 receipt/cleanup/population proof가 완전한 후에 `GEOMETRY_SEAL.json`을 작성한다.

## 독립 기하 검산과 GT 평가 순서

독립 checker 코드의 SHA는 core freeze에 들어간다. checker 자체 protocol은 **완료된 GT-free geometry 봉인 후, checker의 자체 산술과 GT decode 전에** 별도 freeze한다. checker는 stdlib scalar projection/J를 재구성하고 production optimizer/OpenCV/SVD/모델/GT를 실행하지 않는다. 저장 singular value의 rank 규칙 재계산과 독립 SVD 실행은 구분해서 보고한다.

```bash
python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.validation_checks freeze \
  --input "$v9_output" --output "$v9_output" --protocol "$v9_protocol"

python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.validation_checks run \
  --input "$v9_output" --output "$v9_output" --protocol "$v9_protocol"
```

`VALIDATION_PROTOCOL.json` 및 `VALIDATION_CHECKS.json`의 code/input bindings와 complete/PASS를 확인한 뒤에만 GT 평가를 진행한다. 실제240여 장 일부 prefix나 CPU 통과를 완전한 실사 결과로 취급하지 않는다.

```bash
python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.evaluator preflight \
  --parent "$v7_input" --point-parent "$v8_input" --output "$v9_output" \
  --protocol "$v9_protocol" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"

python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.evaluator score \
  --parent "$v7_input" --point-parent "$v8_input" --output "$v9_output" \
  --protocol "$v9_protocol" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"

python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.statistics \
  --input "$v9_output" --parent "$v7_input" --point-parent "$v8_input" \
  --output "$v9_output" --protocol "$v9_protocol" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"
```

unchanged scorer는 local245, same sparse point comparator245, BASE/N3 fixed490의 **980행을 실제 채점**한다. 새 모델·initial PnP·point/line fit·local optimizer는 이 단계에서 금지한다. saved V8 point245와 saved V7 fixed490의 기존 score parity735행도 확인한다. 출력은 `PREDICTIONS.jsonl.gz`, `COMPARATOR_PREDICTIONS.jsonl.gz`, `FIXED_PREDICTIONS.jsonl.gz`이며 complete245 frame 모두 유지한다. 사람 가시성은 score 뒤 진단에만 쓰며 fit에 들어가지 않는다.

평균·표본분산·표준편차·중앙값·P90/max는 full 원행에서 계산한다. 네 방법×세 strata×세 operational/NEW/fallback scope×세 metric×여섯 field의648 moment scalar, 세 고정 대비×세 strata×세 paired scope의27 group/81 CI를 기존13 session×10,000 draws로 계산한다. 새 draw/seed는 없다. local−BASE, local−N3_SUBPIX, local−ROLE_BOUNDARY_H_ROBUST의 음수 delta가 개선이다. 전체 운용 집합과 common NEW를 구분하고 관측 부족·수치 실패·다중해·기본 반환을 삭제하지 않는다.

## 독립 통계 검산과 공개 검토

통계 완료 뒤 표준 라이브러리 검산의 code/public inputs를 별도 freeze하고 한 번 산술을 수행한다. core 정확도 정책을 다시 정하는 freeze가 아니다.

```bash
python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.verify freeze \
  --input "$v9_output" --output "$v9_output" --protocol "$v9_protocol"

python -B -m scripts.research.pallet_sparse_local_line_20261010_v9.verify run \
  --input "$v9_output" --output "$v9_output" --protocol "$v9_protocol"
```

검산은 `STATISTICS_VERIFICATION_PROTOCOL.json`, `STATISTICS_VERIFICATION_STARTED.json`, `VERIFICATION.json`을 남긴다. 최종 보고서는 line>0, line0, <4point+line rescue, weak8px inlier support와 실제 위치·회전·ADDsym을 ID별로 연결하고 수치 산출과 목표 성공을 구분한다. 직접 가시 코너 손상과 실제 H 재투영 오차도 별도로 보존한다. 같은 point pool을 사용한 line-removal ablation은 이 고정 one arm에 추가하지 않았으므로 line 효과가 완전히 인과 분리됐다고 주장하지 않는다.

render는 core에 byte-bound된 코드와 기존 visual-case protocol을 사용하고 postseal/score 이후에만 원본 RGB를 읽는다. 원본 영상·가중치·사용자 변경·기존 결과는 변경하지 않는다. 공개 inspector/압축 복원 등 보충 후단 도구는 별도의 code/input protocol과 실행 receipt를 남긴다. 전용 브랜치에 코드·원행·검산·실행량·이미지·보고서를 정상 commit/push하고 remote SHA를 확인한다. main 변경·자동 merge·force push는 없다.

이 명령은 sealed observation replay의 실제 local fit 진단이며 배포 전체 경로 latency benchmark가 아니다. `wall_seconds`는 실행량·비용 기록이다. fresh 전체 경로 시간을 평가하려면 정확도 계약을 유지한 별도 runtime code/protocol을 먼저 검토·고정하고 경쟁 workload가 없는 실제 경로로 측정해야 한다. cache replay나 과거 시간 합산을 latency로 보고하지 않는다. 현재 DEV는 이미 관찰한 기하 reference이고 unseen 또는 독립 측정 physical truth가 아니다.
