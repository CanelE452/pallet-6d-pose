# V8 동일 관측 C3 재현 절차

이 파일은 실행 전 고정할 재현 계약이다. 실제 실행 상태·호출량·성능·게시 SHA는 별도의 보고서와 receipt를 확인한다. 아래 명령은 자동 실행되지 않는다.

필요한 자료는 완전한 V7 protocol/seal/inference receipt와 geometry1960/OBS735/fixed490, 고정 Clean+Moderate245 cohort, CPU protocol과 실패 A·성공 B, 새 게시 V7 HEAD를 보호하는 V8 `PROTECTION_BEFORE.json`이다. 공개 압축 자료를 복원할 때에도 원래 SHA256·bytes가 일치해야 한다. 정확한 입력 binding은 V8 `PROTOCOL.json`에 기록한다.

원본 source와 원래 baseline은 서로 다른 checkout이다. CLI와 `PALLET_SOURCE_ROOT`, `PALLET_BASELINE_ROOT`에 같은 실제 경로를 전달한다. baseline은 기존 a22fb14beb5e8df08076385000e0d53503c1ae29 dependency를 사용한다. 최소 환경 context는 assemble의 original hidden-mask import만 지원하며 detector나 checkpoint를 초기화하지 않는다. 환경 변수가 누락된 기존 CPU attempt A는 보존되어 있고 성공 B와 구분한다.

```bash
export PALLET_SOURCE_ROOT="/actual/original/source/checkout"
export PALLET_BASELINE_ROOT="/actual/immutable/a22fb14/baseline/checkout"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
v7_input="/path/to/complete/V7/accuracy"
v8_output="/tmp/pallet-same-observation-controls-private-20261010-v8/geometry"
mkdir -p "$v8_output"
```

기존 V7 자료가 공개 DOC에 복원되어 있으면 `v7_input`을 `_docs/experiments/pallet_three_head_observation_20261010_v7`의 절대 경로로 설정한다. root 경로는 실제 실행 전에 확인한다. source/GT/model 자료는 새 관측을 만드는 용도로 읽지 않는다. 보호 snapshot의 바이트 검증은 수치 실행 외부에서 수행한다.

코드·평가 계약·parent 전체 population·성공 CPU 증거가 모두 준비된 뒤 **한 번** freeze한다. 실패·성공 receipt, protocol, pending/interrupted 파일은 덮어쓰지 않는다. 이미 완료된 공식 실행에 아래 freeze/run을 다시 호출하면 보호 guard가 거부해야 한다.

```bash
python -B -m scripts.research.pallet_same_observation_controls_20261010_v8.runner freeze \
  --parent "$v7_input" --output "$v8_output" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"

python -B -m scripts.research.pallet_same_observation_controls_20261010_v8.runner preflight \
  --parent "$v7_input" --output "$v8_output" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"

python -B -m scripts.research.pallet_same_observation_controls_20261010_v8.runner run \
  --parent "$v7_input" --output "$v8_output" \
  --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"
```

예정된 정상 결과는 geometry735행, ledger245행, 실제 bank245개와 solve735개다. Hrobust245는 저장 결과를 복사하는 것이 아니라 기존 parent 결과와 parity를 확인하는 실제 replay다. 나머지 두 진단이490개다. native N3는 반드시 OBS 원본에서 사용한다. 한 프레임의 같은 bank를 세 경로가 공유한다. sparse 결측을 native로 메우지 않는다. 원래 H는 변경하지 않고 masked NEW 이후에만 재투영으로 교체하며 재투영을 재fit하지 않는다.

`CONTROL_RECEIPT.json`이 완전하고 cleanup/error가 없으며 `GEOMETRY_SEAL.json`이 해당 receipt·원행·ledger·protocol·population proof의 SHA와 일치하기 전에는 GT 평가를 시작하지 않는다. 별도 `validation_checks.py`와 평가·통계·검산 CLI는 그 파일의 `--help` 및 고정 protocol에 따라 실행한다. BASE/N3는 부모의 sealed fixed490을 사용하며 evaluator는 다시 pose fit이나 모델 forward를 수행해서는 안 된다. 평가 뒤에는 독립 검산의 원행 수·상태·일반/강건 rank 및 ambiguity 규칙·재투영·모멘트·고정 대비를 확인한다.

후단 코드는 `evaluator.py`, `statistics.py`, `verify.py`이며 모두 실사 솔버 실행 전의 core protocol에 byte-bound된다. Scoring은 성공한 독립 `validation_checks.py` 결과까지 확인한 뒤 실행한다. 독립 통계 검산은 통계가 완전한 이후에 `STATISTICS_VERIFICATION_PROTOCOL.json`을 별도로 freeze하고 `STATISTICS_VERIFICATION_STARTED.json`, `VERIFICATION.json`을 남긴다. 이 사후 검산 freeze는 정확도 정책을 다시 설정하는 단계가 아니다. 고정 대상은 5방법×3strata×3scope×3metric×6field의 810 moment scalar와 여덟 대비의 72 group/216 CI다. GT targets, 기존 BASE/N3 reference 및 visibility audit은 사전 SHA만 고정하며 성공한 seal 이전에 decode하지 않는다.

각 geometry row는 full solver candidate/per-dimension/alternative/Jacobian witness, 원래 native N3와 parent OBS digest, original predicted H 및 실제 적용 H를 보존한다. ledger는 실제 cache 생성·hit/refit/LM counts, replay parity, 같은 U 여부와 실제 H 제외 ID를 보존한다. 완전 실패도 원행을 유지한다. resource snapshot 및 실제 entry counter를 실행량 보고와 연결한다.

이 run은 관측 재생 PnP 진단이며 latency benchmark가 아니다. `wall_seconds`는 실제 계산의 실행량·비용 기록일 뿐이다. 처리시간은 이미 따로 실행한 V7 fresh 전체 경로의 공식 기록을 확인한다. 이번 실사 기준은 이미 관찰한 DEV245의 기존 기하 reference이며 미지 평가나 독립 physical GT라고 주장하지 않는다.
