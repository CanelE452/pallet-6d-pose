# final 6D pose target의 matched GEO 실험

목적은 같은 후보 bank와 scorer에서 supervision만 바꾸어 기존 2D target과 비교하는 것이다. 새 방법 `POSE_TARGET_GEO_J`는 기존 최종 F의 ADDsym이 최소인 실제 bank index에 hard classification CE를 적용한다. RGB 특징·치수·초기화·order·모델 20,259개 파라미터·AdamW·6,000 update·hard J·최종 F는 그대로다. OLD는 soft 2D target, NEW는 hard 6D target이므로 비용 정렬과 label sharpness 각각의 효과를 분리하는 비교는 아니다. 추가 비교나 hyperparameter 탐색은 하지 않는다.

[확인] 실제 결과는 `POSE_TARGET_NOT_SUPPORTED`다. seed1의 SYNTH NEW−OLD paired 평균 T/R/ADDsym이 모두 악화하고 RAW good5→bad10 손상이 증가해 고정 gate `STOP`으로 seed2/3을 실행하지 않았다. 전체 비용 11,238,915 F, 정식 fit 1개/6,000 update, SYNTH 1,985/REAL_DEV 319 전수 평가를 완료했다. 계약 테스트 30 PASS와 I/O fixture 12 PASS, 전체 원입력 종료 SHA와 독립 원행 검산 PASS다. 수치와 분모는 [RESULT_KO.md](RESULT_KO.md)에 있다.

최신 직접 지시에 따라 최종 commit/push 대상은 `main`이다. 기준 `a22fb14beb5e8df08076385000e0d53503c1ae29`에서 분리한 detached 작업 공간을 사용하며 새 branch를 사용하지 않는다. 기존 사용자 변경과 이전 작업 공간을 보존한다. 이번 결과에는 논문·원고·TeX·PDF·참고문헌 변경을 포함하지 않는다.

## 계약과 데이터

`PROTOCOL.json`은 첫 새 F/forward/fit 전에 최종 판정까지 고정했다. `INPUT_BINDINGS.json`/`PREFLIGHT.json`은 원 SHA 기대값에 연결한 710개 외부 입력 76,031,296,408bytes의 전체 내용 검사와 a22 보호 파일 351개의 검사를 담는다. 이전 protocol 초안과 최종 실행 전 코드 고정 영수증을 `audit/`에 보존했다. 실제 source/cache/bank는 복사하지 않는다.

TRAIN은 기존 usable 55,915장이다. 전체가 기존 201 action이어서 최종 F 비용은 11,238,915개다. source row와 candidate index를 유지하고, F unavailable은 +inf, 전체 실패 target은 −1로 기록한다. 같은 비용의 첫 index를 고르며 NoOp는 0이다. 원 float64 native candidate를 평가하고 center를 바꾸지 않는다. W/D 선택과 SQPnP·기존 refinement는 prediction-only이며 GT pose는 TRAIN 비용 채점에만 들어간다.

SYNTH_HELDOUT은 반복 사용한 개발 1,985장, REAL_DEV는 같은 2D 레이블·치수로 참조를 만든 반복 사용한 319장/13 sessions다. REAL은 exploratory이고 screening에 사용하지 않는다. 실제 모델이 inference에서 GT/oracle index를 읽지 않는다. 독립 물리 계측 정확도를 증명하는 비교가 아니다.

## 실행 경로

실제로 사용한 Python은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`이다. 새 코드는 unchanged a22의 scorer/F/Data와 기존 원행을 읽기 전용으로 호출한다. main에 이전 논문이나 실험 전체를 복사하지 않으므로 별도의 a22 baseline이 명시적 의존성이다. 기본 baseline은 `/home/minjae/Documents/github/pallet-pose-handoff-20261006`; 다른 위치라면 `PALLET_BASELINE_ROOT`를 지정한다. `baseline.py`가 해당 저장소의 a22 Git object와 실제 파일 SHA를 대조한다. baseline이 없거나 다르면 `BLOCKED_DATA`/`BLOCKED_INTEGRITY`이며 새 bank로 대체하지 않는다.

```bash
PALLET_PY=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
PALLET_BASELINE_ROOT=/home/minjae/Documents/github/pallet-pose-handoff-20261006
PALLET_SOURCE_ROOT=/home/minjae/Documents/github/pallet-pose
PALLET_BANK_CACHE=/tmp/pallet-joint-action-cache
PALLET_COST_CACHE=/tmp/pallet-pose-target-6d-cache
export PALLET_BASELINE_ROOT

"$PALLET_PY" -B -m scripts.research.pallet_pose_target_6d_20261006_v1.contract_tests
"$PALLET_PY" -B -m scripts.research.pallet_pose_target_6d_20261006_v1.training --stage parity --source-root "$PALLET_SOURCE_ROOT" --bank-cache "$PALLET_BANK_CACHE" --cost-cache "$PALLET_COST_CACHE"
```

최초 원입력 감사는 전용 `preflight.audit(source_root, bank_cache, cost_cache)`로 실행했다. 원입력 SHA는 기존 a22 영수증의 기대값을 바꾸지 않는다. 잠긴 INPUT_BINDINGS가 있으면 시작 계약을 덮어쓰지 않고 종료 감사로 전환한다. 아래 비용/검산 명령은 PREFLIGHT PASS와 현재 protocol/code/binding SHA를 요구한다.

```bash
"$PALLET_PY" -B -m scripts.research.pallet_pose_target_6d_20261006_v1.geometry_parity --source-root "$PALLET_SOURCE_ROOT" --bank-cache "$PALLET_BANK_CACHE" --cache-dir "$PALLET_COST_CACHE" --bindings _docs/experiments/pallet_pose_target_6d_20261006_v1/INPUT_BINDINGS.json --preflight _docs/experiments/pallet_pose_target_6d_20261006_v1/PREFLIGHT.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PALLET_PY" -B -u -m scripts.research.pallet_pose_target_6d_20261006_v1.cost_cache --source-root "$PALLET_SOURCE_ROOT" --bank-cache "$PALLET_BANK_CACHE" --cache-dir "$PALLET_COST_CACHE" --bindings _docs/experiments/pallet_pose_target_6d_20261006_v1/INPUT_BINDINGS.json --preflight _docs/experiments/pallet_pose_target_6d_20261006_v1/PREFLIGHT.json --workers 16

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PALLET_PY" -B -u -m scripts.research.pallet_pose_target_6d_20261006_v1.run --stage finish --source-root "$PALLET_SOURCE_ROOT" --bank-cache "$PALLET_BANK_CACHE" --cost-cache "$PALLET_COST_CACHE"
```

`finish`는 이미 실행 중인 비용 생성에 추가 F를 실행하지 않고 완료 manifest를 기다린다. 이후 aux → seed1 fit → final6000 eval → SYNTH-only screen → 조건부 seed2/3 fit/eval → 실행량 집계 → 실제 산출물 계약 테스트 30개 → 최종 분석/보고 순서다. 최초 대기의 producer는 3927620이었으며, 아래 I/O 전환 후 실제 대기 실행에는 `--cost-producer-pid 4052685`를 지정했다. PID와 Linux start ticks가 바뀌거나 생성 과정이 중단되면 manifest를 최종 확인한 뒤 실패 의존상태를 기록한다. `finish.lock`은 동시 실행을 막고, 기존 미해결 stage ATTEMPT는 자동으로 반복하지 않는다. 미완료 시도를 없애거나 같은 F를 조용히 재시도하지 않는다.

각 module의 실제 `--help`와 `PIPELINE_EXECUTIONS.json`에 기록한 command를 확인한다. 비용 cache는 binary attempt/complete journal을 재생하여 완료 후보를 읽으며, 동일 binding의 완료 F를 반복하지 않는다. 동일 bank라도 비용/F/target/code 계약을 바꾼 결과를 기존 run처럼 재사용하지 않는다. 실행 전·후 검사와 publication 이후 Git 상태의 정상 변경을 구분한다.

`audit/MONITOR_PROGRESS.json`과 `audit/GUARDED_ORCHESTRATOR_LAUNCH.json`은 I/O 전환 전의 시각·PID·진행률을 보존한 과거 snapshot이다. 그 ETA를 현재 예상 시간으로 사용하지 않는다. `STATUS.stage`는 대기 시작과 전체 종료 때 갱신되므로 중간 aux/fit/eval 단계는 `PIPELINE_EXECUTIONS.json`의 마지막 stage와 각 `TRAIN_PROGRESS_seed*.json`을 확인한다.

사용자의 속도 개선 요청 후 실제 병목을 확인했다. 원 16 worker는 CPU 약 15%와 ext4 commit 대기였고, 후보당 두 번의 O_DSYNC 작은 쓰기가 지연을 만들었다. 원 producer parent만 멈춰 bounded queue를 끝까지 처리한 뒤, 161개 chunk/10,304 TRAIN/2,071,104 완료 F와 미완료 시도 0을 검산하고 idle worker를 종료했다. 원 F·cost·bank·protocol 코드는 바꾸지 않았다. `audit/IO_TRANSITION.json`과 `audit/SLOW_COST_COMPLETED_SNAPSHOT.json`에 중단 전 완료 증거와 CPU/벽시간을 보존했다.

이후 `fast_cost_cache`가 남은 동일 후보만 계산한다. 행별 예약을 fsync하고 원 37byte WAL 형식을 그대로 기록하며, 행 끝에 fdatasync한다. 예약 수와 실제 호출 수는 구분한다. 중단된 예약에 완료 증거가 없으면 재계산하지 않고 BLOCKED로 남긴다. 전체 native WAL/배열/first-argmin/실패/실제 PnP와 원 사용자/입력 보존을 `native_cost_audit`로 검산한 다음에만 학습용 PASS manifest를 공개한다. `FAST_IO_TEST_RESULTS.json`의 12개 검사는 실제 solver/model을 실행하지 않은 임시 fixture다. 최초 느린 단계와 빠른 단계의 wall 합을 기록하며 두 단계 사이의 준비 대기는 `audit/FAST_COST_LAUNCH.json`에 별도로 기록한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PALLET_PY" -B -u -m scripts.research.pallet_pose_target_6d_20261006_v1.fast_cost_cache --source-root "$PALLET_SOURCE_ROOT" --bank-cache "$PALLET_BANK_CACHE" --cache-dir "$PALLET_COST_CACHE" --bindings _docs/experiments/pallet_pose_target_6d_20261006_v1/INPUT_BINDINGS.json --preflight _docs/experiments/pallet_pose_target_6d_20261006_v1/PREFLIGHT.json --workers 16 --transition "$PALLET_COST_CACHE/IO_TRANSITION.json"
```

이 명령은 old producer의 실제 종료와 PID start ticks를 검사하며 단독 producer lock을 요구한다. 진행 중인 다른 writer와 동시에 시작하지 않는다. `PRETRAIN_COST_MEMORY_MANIFEST.json`은 감사에 사용한 실제 메모리 manifest의 보존본이다. 감사 후 시간/검산 SHA를 덧붙인 최종 manifest와 구분하며, 감사의 `memory_manifest_sha256`은 보존본의 canonical JSON SHA다.

## 결과 파일

필수 숫자는 실제 실행이 끝난 뒤 `POSE_COST_CACHE_MANIFEST`, `POSE_TARGET_PARITY`, `TRAIN_RECEIPTS`, `SYNTH_RESULTS`, `REAL_DEV_RESULTS`, `MATCHED_COMPARISON`, `ORACLE_RECOVERY`, `WD_HYPOTHESIS_ANALYSIS`, `TWO_D_SIX_D_TRADEOFF`, `EXECUTION_COUNTS`에 기록한다. `RESULT_KO.md`와 `NEXT_DECISION_KO.md`는 실제 원행을 읽어 작성한다. 전체 cost matrix·native bank·feature cache·checkpoint·임시 로그는 저장소 밖에 유지하며 SHA와 byte count를 manifest에 남긴다.

seed1 gate와 최종 verdict는 사전 고정한 protocol을 따른다. 마지막 6,000 update checkpoint만 평가한다. seed2/3 실행 여부와 이유를 보존하며 결과를 본 뒤 LR/epoch/seed/cap/action 수를 바꾸지 않는다. recovery ratio의 음수와 1 초과를 clamp하지 않고, headroom≤1e−7m은 분모에서 따로 기록한다. OLD 개선 여부와 N3보다 좋은지의 질문은 별도로 보고한다.

학습 checkpoint와 stage 영수증은 atomic rename으로 교체하며, 프로세스 중단 후 미해결 ATTEMPT를 자동 반복하지 않는다. 학습 저장 경로는 fsync를 수행하지 않으므로 전원·커널 장애까지의 영속 저장 보장을 뜻하지 않는다. 실제 실행 종료의 checkpoint SHA·6,000 update 영수증과 optimizer 노출 수를 독립 검산한다.

시간 범위는 구분한다. `EXECUTION_COUNTS`의 fit/evaluation/aux 시간은 각 영수증의 내부 실행 구간 wall 합이며 초기화·입력 검사를 모두 포함한 subprocess 전체 시간은 `PIPELINE_EXECUTIONS.json`의 `seconds`다. 내부 구간에도 GPU 연산·CPU·I/O가 섞여 있으므로 isolated GPU kernel 시간으로 해석하지 않는다. 비용 wall에는 전체 native 검산 시간이 포함되고 I/O 전환 준비 대기는 별도 영수증에 남긴다.

최종 main push의 실제 원격 SHA 비교 영수증은 새 외부 cache의 `PUSH_VERIFICATION.json`에 둔다. 결과 commit 자체에 자신의 SHA를 쓰기 위한 재귀 commit은 만들지 않는다.

저장소의 `STATUS.json`은 commit 생성 전 계산·검산 snapshot이다. 실제 commit SHA·정상 push 종료 코드·push 후 remote main SHA의 일치는 위 외부 영수증에서 확인한다. 이 구분을 통해 push 성공을 미리 기록하거나 commit에 자신의 SHA를 포함하지 않는다.

main의 최초 사용자 checkout은 원격보다 오래됐으며 이미 내려받은 원격 파일들과 사용자 변경이 섞여 있다. `publication --stage plan`은 최신 원격 SHA, 기존 파일 바이트, 사용자 symlink와 index 상태를 읽기 전용으로 검증한다. 실제 종료 검산 전에는 `--stage integrate`가 차단된다. 종료 검산 뒤에는 원 index와 사용자 파일을 외부에 백업한 후 `read-tree`로 index만 통합하고 `update-ref`의 예상 이전 SHA 검사로 main을 전진시킨다. 원격 working 파일을 checkout하지 않으며, 원래 없던 92개 파일과 symlink 아래 1개 tracked child는 명시적 skip-worktree 93개로 유지한다. 이는 사용자 파일 내용 변경과 논문 파일 생성 없이 main의 최신 parent를 사용하기 위한 로컬 Git 메타데이터 처리다. 새 namespace만 명시적으로 stage하며 기존 a22 논문 변경은 main에 병합하지 않는다.

main의 기존 사용자 checkout에서 옮긴 코드를 재검사한 기록도 보존했다. 과학 산출물을 포함한 원 계산 workspace의 30개 계약 검사는 PASS이며 이 유효한 영수증을 재사용한다. main 재검사는 29 PASS/1 FAIL이었다: frozen 원고 검사가 기존 untracked 원고46개까지 신규 생성으로 세었다. 이46개는 시작 상태에 부모 경로가 이미 있고 전부 시작 전 mtime이며 내용46/46이 a22의 기존 blob과 동일하다. 실패를 `audit/MAIN_CONTEXT_CONTRACT_TEST_ATTEMPT.json`에 보존하고, 원30 PASS 영수증을 덮어쓴 실패 기록과 구분했다. 실제 main 게시 검증은 이번 staged 경로의 strict allowlist와 원고 staged diff0 및 기존 파일 보존으로 수행한다. ignore/index/test 코드를 조작해 재검사 실패를 숨기지 않았다.
