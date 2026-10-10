[확인] 고정 입력의 Stage0/1, S1–S3, 공식 depth 정확도 gate와 조건부 S4를 재현하는 명령입니다. 새 학습·데이터 생성은 없습니다.

# 환경과 입력

게시 저장소 checkout 루트에서 기존 pose 환경을 사용합니다. 기존 private 입력·RGB·checkpoint는 공개 저장소에 포함되지 않으며 [INPUT_AUDIT.json](INPUT_AUDIT.json)의 경로와 SHA가 일치해야 합니다. 코드·공식 depth weight·reference·수치 정의를 바꾸면 기존 실행의 재현으로 간주하지 않습니다.

```bash
export PALLET_PYTHON=/path/to/existing/pallet-pose/bin/python
export PALLET_SOURCE_ROOT=/path/to/original/private/source
export PALLET_BASELINE_ROOT=/path/to/existing/pallet-pose-handoff
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

`PALLET_WD_OUTPUT`은 checkout 내부의 존재하지 않는 새 폴더여야 합니다. 예를 들어 `$PWD/_docs/experiments/pallet_wd_reproduce_run1`을 사용합니다. 기존 산출물·private cache가 있으면 보존 검사로 중단하며 덮어쓰지 않습니다. 아래 두 경로는 목적이 다르므로 같은 출력 폴더로 이어 실행하지 않습니다.

# 기존 게시 Stage1에서 Stage2·Stage3 재현

이 경로는 최초에 실제 게시한 Stage1의 입력 감사·봉인·예측·reference score·사전 방법 lock을 SHA 검증하여 새 폴더에 복사한 후, 수정하지 않은 S1/S2/S3 driver를 실행합니다. 새로 계산한 Stage1을 게시했다고 주장하지 않습니다. 복사한 파일과 SHA는 `REPRODUCTION_INPUT_LOCK.json`에 남습니다.

```bash
export PALLET_WD_OUTPUT="$PWD/_docs/experiments/pallet_wd_reproduce_run1"
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.fresh_stage2
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.stage2_verify \
  --doc "$PALLET_WD_OUTPUT" --source-root "$PALLET_SOURCE_ROOT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.stage2_report \
  --doc "$PALLET_WD_OUTPUT" --stage1-commit 2502db29e77c26b88c6c681b054902ea997f441d
```

depth 준비는 새 private parent의 `metric3d` 하위 폴더에 고정 공식 source와 Small checkpoint만 받습니다. 기존 환경은 바꾸지 않고 private target에 부족한 세 패키지만 설치합니다. 공식 source와 checkpoint SHA는 [DEPTH_PREPARATION.json](DEPTH_PREPARATION.json)에 있습니다. fetch·prepare는 모델 생성과 forward를 실행하지 않습니다.

```bash
export PALLET_RUN_PRIVATE=/path/to/new/private/reproduce_run1
export PALLET_DEPTH_PRIVATE="$PALLET_RUN_PRIVATE/metric3d"
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.depth fetch --private-root "$PALLET_DEPTH_PRIVATE"
"$PALLET_PYTHON" -B -m pip install --no-deps --no-compile --no-cache-dir --no-build-isolation \
  --target "$PALLET_DEPTH_PRIVATE/deps" mmcv==1.7.2 addict==2.4.0 yapf==0.40.1
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.depth prepare --private-root "$PALLET_DEPTH_PRIVATE"
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.stage3_fresh --private-dir "$PALLET_RUN_PRIVATE"
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.stage3_verify \
  --doc "$PALLET_WD_OUTPUT" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RUN_PRIVATE"
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.stage3_report --doc "$PALLET_WD_OUTPUT"
```

Stage3는 Stage2 보고 receipt가 COMPLETE인 뒤에만 실행됩니다. 원래 실행의 strict-key guard 실패와 읽기 전용 진단을 재현할 필요는 없습니다. `stage3_fresh`는 동일 공식 strict=False 경로에서 추론에 사용하지 않는 zero mask_token 하나만 허용하는 계약을 모델 생성 전에 봉인합니다. 공식 Small·float32·eval·no_grad·batch 1·전처리·K 역정규화는 동일합니다. 추론 호출의 network 입력은 RGB tensor이고 K는 외부 공식 전처리에서 사용합니다.

정확도 gate는 REAL231 주 경로의 abs→세 seed 평균→영상 median≤5%이며 어느 주 seed라도 누락하면 FAIL입니다. FAIL이면 m 선택·final-test 88장 추가 추론·S4를 실행하지 않고 depth 결과를 보존합니다. PASS이면 SYNTH에서 m={1,2,3} 중 혼동률 최소값을 고르고 동률은 작은 m을 택한 뒤 REAL319에 한 번 적용합니다. GT는 봉인 뒤의 accuracy·pose 채점에만 쓰며 final-test 네 session으로 모델이나 m을 고르지 않습니다.

# Stage0·Stage1 자체의 별도 재계산

새로운 solver 실행으로 Stage0 parity와 Stage1만 다시 계산하려면 다른 새 출력 폴더를 사용합니다. bounded-memory wrapper는 SYNTH 참조 배열을 선택 봉인 뒤 한 번 로드하며 원래 여섯 과학 source를 수정하지 않습니다. 이후 Stage2를 자동으로 이어 실행하는 경로가 아니며, 실제 게시·사전 방법 lock 요건을 생략하지 않습니다.

```bash
export PALLET_WD_OUTPUT="$PWD/_docs/experiments/pallet_wd_stage1_reproduce_run1"
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.fresh_stage1
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.summarize
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.summarize_failures
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.report --doc "$PALLET_WD_OUTPUT"
```

# 공개 수치의 검산·표·PNG 재생성

private RGB와 모델이 없는 환경에서도 게시한 표·PNG는 저장 숫자로 재생성할 수 있습니다. 최종 cumulative 보고서는 `stage3_report`를 사용합니다. `--documents-only`는 기존 PNG와 figure index를 보존합니다. Stage1/Stage2 생성기는 당시 문서를 만들므로 최종 보고서 재생성 명령과 구분합니다.

```bash
unset PALLET_WD_OUTPUT
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.stage3_report \
  --doc _docs/experiments/pallet_wd_hypothesis_diag_20261010
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.stage3_report \
  --doc _docs/experiments/pallet_wd_hypothesis_diag_20261010 --documents-only
```

공식 코드·checkpoint·K 처리 근거와 checkpoint별 별도 license 미표시 제한은 [METHOD_KO.md](METHOD_KO.md), [DEPTH_PREPARATION.json](DEPTH_PREPARATION.json)에 남겼습니다. RGB·검수 overlay·전체 depth map은 private이고 공개는 숫자·SHA·자체 그래프입니다. 기존 방법·source lock을 수정하지 않습니다. 독립 검산은 추가 모델이나 PnP를 실행하지 않습니다.
