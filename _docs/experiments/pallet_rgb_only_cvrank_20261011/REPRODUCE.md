RGB_ONLY는 고정 두 seed를 3,000스텝씩 실제 학습했으며, 아래 실행 기록과 빈 출력 폴더에서의 재현 절차를 구분한다.

[확인] 원 입력·기존 checkpoint·기존 결과·원고는 수정하지 않는다. 새 공개 출력은 `scripts/research/pallet_rgb_only_cvrank_20261011/`과 이 문서의 실험 폴더만 사용한다. 원본 RGB, 정답 원좌표, 학습 checkpoint, 절대 비공개 경로는 공개하지 않는다. 따라서 공개 저장소만으로 학습·영상 평가를 완전히 재실행할 수는 없다. 아래 절차는 등록된 비공개 입력에 접근할 수 있는 환경을 대상으로 한다.

[확인] 실제 환경은 [원 실행 ENVIRONMENT.json](https://github.com/CanelE452/pallet-6d-pose/blob/main/_docs/experiments/pallet_rgb_only_cvrank_20261011/ENVIRONMENT.json)에 기록했다. 모델 연산 FP32, AMP/TF32 off이며 기존 특징 경로의 neck FP16 capture 반올림은 유지된다. RGB_ONLY 후단은 p3/p4를 읽지 않는다. 검출기 자체는 그대로 사용하며 현재 Base 좌표·예측 bbox·support를 공통 입력으로 제공한다.

## 실제 실행 기록

[확인] 아래 상태는 해당 실행 영수증으로 확인한 것만 적었다. 명령의 경로만 `PALLET_*` 변수로 정규화했으며 수치 인수와 단계 이름은 그대로다. 준비·환경 수집·PLAN 생성은 root의 inline 작업이었고, 존재하지 않는 별도 CLI를 실행했다고 적지 않는다. 최초 준비 중 실패와 수정은 [원 실행 IMPLEMENTATION_REPAIRS.json](https://github.com/CanelE452/pallet-6d-pose/blob/main/_docs/experiments/pallet_rgb_only_cvrank_20261011/IMPLEMENTATION_REPAIRS.json)에 보존했다.

| 단계 | 실제 수행 명령의 핵심 | 확인된 상태 |
|---|---|---|
| 입력 준비 | `contract_eval.py prepare` 직접 실행 | 완료, 입력·이전 좌표·해시 대조, 새 F/모델 forward 0 |
| CVRANK | 기존 `stage_runner.py --stage cvrank --device cpu -- … contract_eval.py cvrank` | 완료, [STAGE_cvrank.json](STAGE_cvrank.json), 110장 CPU 재계산·330 F |
| RGB preflight | 기존 `stage_runner.py --stage rgb_preflight --device cuda -- … train_rgb.py preflight … --device cuda` | 완료, [STAGE_rgb_preflight.json](STAGE_rgb_preflight.json), 업데이트 0 |
| RGB 학습 | 기존 `stage_runner.py --stage rgb_train --device cuda -- … train_rgb.py train … --device cuda` | 완료, [STAGE_rgb_train.json](STAGE_rgb_train.json), 2 fits·6,000 updates |
| 특징 제거 독립 검산 | `feature_audit.py final` 직접 실행, 아래 4개 경로 인수 | 완료, [FEATURE_EXCLUSION_AUDIT.json](FEATURE_EXCLUSION_AUDIT.json), CPU 검산·forward/F/fit 0 |
| 합성 calibration·선택 | 기존 `stage_runner.py --stage rgb_calibration_selection --device cuda -- … calibration_selection.py …` | 완료, 65.401초, T=1/1, 고정 VAL에서 RGB_ONLY 919/1564로 선택 |
| 기존 입력 보존 검산 | 기존 `stage_runner.py --stage preservation --device cpu -- … preserve_new.py verify …` | 완료, PASS, 4.493초 |
| RGB 실사 추론 최초 시도 | 기존 `stage_runner.py --stage rgb_accuracy --device cuda -- … inference_new.py …` | 실패 기록 보존, 1.957초, bbox JSON list 복원 TypeError, 새 F·학습 0 |
| RGB 실사 추론 직렬화 수선 | 기존 `stage_runner.py --stage rgb_accuracy_serialization_retry --device cuda -- … inference_new.py …` | 완료, 5.165초, 220개 좌표 봉인; 학습·설정 변경 없음 |
| RGB 실사 평가 | 기존 `stage_runner.py --stage evaluation_rgb --device cpu -- … contract_eval.py rgb --predictions …` | 완료, 4.566초, 440행·440 F |
| 기존 행 재사용·합치기 | 기존 `stage_runner.py --stage evaluation_finalize --device cpu -- … contract_eval.py finalize` | 완료, 8.921초, 기존 2,200행 metric digest exact, 총3,300행, 새 F 0 |
| 통계 | 기존 `stage_runner.py --stage statistics --device cpu -- … summarize_new.py …` | 완료, 1.476초 |
| CVRANK 비용 최초 시도 | 기존 `stage_runner.py --stage latency_CVRANK_NATIVE --device cuda -- … latency_new.py --method CVRANK_NATIVE …` | 실패 보존, 1.096초, COMMON 입력의 sessions 필드 KeyError, detector/F 0 |
| CVRANK 비용 패널 수선 | 기존 `stage_runner.py --stage latency_CVRANK_NATIVE_panel_retry --device cuda -- … latency_new.py --method CVRANK_NATIVE …` | 완료, 5.576초, 이전 18개 panel ID 그대로 사용 |
| 나머지 새 비용 4경로 | 기존 `stage_runner.py --stage latency_방법명 --device cuda -- … latency_new.py --method 방법명 …` | 모두 완료: CAP1 5.537초, CVRANK+SP 5.668초, RGB_ONLY 4.587초, RGB_ONLY+SP 4.492초 |
| 비용 합치기 | 기존 `stage_runner.py --stage latency_aggregate --device cpu -- … latency_new.py … --aggregate` | 완료, 1.017초, 새 5경로·기존 10경로 |
| 그림 | 기존 `stage_runner.py --stage figures --device cpu -- … figures_new.py --public-dir …` | 완료, 2.260초, 공개 PNG 5개, 새 RGB/GPU/F 0 |
| 전체 예산 | `compute_new.py --public-dir … --analysis-start-epoch 1791725400` 직접 실행 | 완료, GPU 예약1,206.172초·CPU 실측40.467초·보수 CPU 상한2,570.535초 |
| 한국어 결과·논문 구성 판단 | `report_new.py --public-dir …` 직접 실행 | 완료, M1 구성 추천, 결합 증분 근거 미확인 |

[확인] 최종 독립 검산은 아래 `final_verify` 명령으로 한 번 실행하며 판정은 [INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json) 및 [STAGE_final_verify.json](STAGE_final_verify.json)에 별도로 기록한다. 이 문서는 감사 결과를 예측하지 않는다. 최종 판정 뒤 이 실행 기록을 다시 고쳐 감사 대상 바이트를 바꾸지 않는다.

[확인] 특징 제거 검산은 두 seed의 학습 전후 실제 forward 캡처와 최종 checkpoint를 CPU로 독립 확인했다. 정상 64칸·null 61칸·residual 81칸의 첫 48칸은 정확히 0이며 original/zero/other/missing p3/p4의 base logits·residual logits·확률·실제 기대좌표가 같았다. adapter/patch CNN 호출은 0이었다. RGB 변경 시 학습 후 network 좌표 최대 변화는 seed1 1.71631px, seed2 1.53644px였다. 이는 TRAIN 검산이며 실사 6D 수치가 아니다.

[확인] seed1/2의 마지막 3,000스텝 checkpoint만 사용한다. 실제 두 fit 시간은 1,095.338초, imports·load·사후 검산을 포함한 학습 process wall은 1,100.756초였다. [RGB_ONLY_TRAIN_RECEIPTS.json](RGB_ONLY_TRAIN_RECEIPTS.json)에 업데이트·순서·초기값·소스 SHA·개별 checkpoint SHA가 있다. 초기 공통 가중치 SHA는 이전 M3와 같지만 특징을 제거한 step0 출력은 full-feature 출력과 같다고 주장하지 않는다.

실제 preflight·학습 명령은 다음과 같았다. `--prior-public-dir`를 생략하여 현재 코드 ROOT의 고정 이전 실험 DOC를 사용했다.

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PALLET_GPU_PYTHON" -B "$PALLET_RUNNER" --stage rgb_preflight --device cuda --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_GPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/train_rgb.py" preflight --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --device cuda
"$PALLET_GPU_PYTHON" -B "$PALLET_RUNNER" --stage rgb_train --device cuda --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_GPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/train_rgb.py" train --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --device cuda
```

실제 입력 준비·CVRANK·특징 독립 검산은 CPU 환경에서 `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`로 실행했다.

```bash
"$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/contract_eval.py" prepare
"$PALLET_CPU_PYTHON" -B "$PALLET_RUNNER" --stage cvrank --device cpu --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/contract_eval.py" cvrank
"$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/feature_audit.py" final --public-dir "$PALLET_RGB_ONLY_DOC" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --source-root "$PALLET_SOURCE_ROOT" --prior-public-dir "$PALLET_PRIOR_DOC"
```

실제 calibration·accuracy 실행은 아래 5개 경로 인수를 썼다. 최초 accuracy 실패와 직렬화 수선 재실행은 서로 다른 stage 영수증으로 남겼다.

```bash
"$PALLET_GPU_PYTHON" -B "$PALLET_RUNNER" --stage rgb_calibration_selection --device cuda --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_GPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/calibration_selection.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --prior-private-dir "$PALLET_NONSEVERE_PRIVATE" --prior-public-dir "$PALLET_PRIOR_DOC"
"$PALLET_GPU_PYTHON" -B "$PALLET_RUNNER" --stage rgb_accuracy --device cuda --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_GPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/inference_new.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --prior-private-dir "$PALLET_NONSEVERE_PRIVATE" --prior-public-dir "$PALLET_PRIOR_DOC"
"$PALLET_GPU_PYTHON" -B "$PALLET_RUNNER" --stage rgb_accuracy_serialization_retry --device cuda --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_GPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/inference_new.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --prior-private-dir "$PALLET_NONSEVERE_PRIVATE" --prior-public-dir "$PALLET_PRIOR_DOC"
```

실제 비용 argv의 `--accuracy-predictions`는 CVRANK 3경로에서 `${PALLET_RGB_ONLY_PRIVATE}/CVRANK_PREDICTIONS_SEALED.jsonl.gz`, RGB_ONLY 2경로에서 `${PALLET_RGB_ONLY_PRIVATE}/RGB_ONLY_ALL_PREDICTIONS_SEALED.jsonl.gz`였다. CVRANK_NATIVE 최초 실패 후 성공 stage 이름은 `latency_CVRANK_NATIVE_panel_retry`이며, 나머지 네 stage는 `latency_방법명`이었다. aggregate는 위 4개 경로 인수와 `--aggregate`를 CPU wrapper로 실행했다. 새 5경로의 90회 timed 출력 모두 accuracy 좌표 최대 차이0을 기록했으며 기존 PoseFix+SubPix의 제한을 지우지 않았다.

[확인] 실패 수선은 bbox list→ndarray 직렬화 복원과 기존 18개 latency panel ID→COMMON row 매핑이다. 학습·후처리·패널·수치 설정을 변경하거나 좋은 결과를 고르는 재실행이 아니며 실패 process wall도 전체 예산에 포함한다. 아래 전체 recipe는 실행 가능한 재현 명령이며, 별도 clean replay를 이미 끝냈다는 주장이 아니다. 수정된 코드로 깨끗하게 실행할 때 역사 실패 영수증을 새로 만들지 않는다.

## 고정 입력과 빈 작업 폴더

실행 전에 다음 환경 변수를 실제 접근 가능한 경로로 지정한다. `PALLET_SOURCE_ROOT`, `PALLET_PUBLIC_ROOT`, `PALLET_WD_PUBLIC_ROOT`, `PALLET_NONSEVERE_PRIVATE`는 기존 입력이다. `PALLET_REPLAY_ROOT`와 `PALLET_RGB_ONLY_PRIVATE`는 **존재하지 않는 새 경로**여야 한다. 기존 공개 실험 폴더에서 실행하지 않는다.

```bash
set -euo pipefail
: "${PALLET_RELEASE_CHECKOUT:?published checkout required}"
: "${PALLET_RELEASE_REF:?pin the publication commit, not a moving branch}"
: "${PALLET_REPLAY_ROOT:?new absent replay root required}"
: "${PALLET_SOURCE_ROOT:?frozen original source and private dataset required}"
: "${PALLET_PUBLIC_ROOT:?frozen previous publication inputs required}"
: "${PALLET_WD_PUBLIC_ROOT:?frozen W/D publication inputs required}"
: "${PALLET_NONSEVERE_PRIVATE:?authenticated prior private predictions and reference required}"
: "${PALLET_RGB_ONLY_PRIVATE:?new absent private output directory required}"
: "${PALLET_GPU_PYTHON:?compatible CUDA Python interpreter required}"
: "${PALLET_CPU_PYTHON:?compatible CPU analysis Python interpreter required}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PALLET_SOURCE_ROOT PALLET_PUBLIC_ROOT PALLET_WD_PUBLIC_ROOT
export PALLET_NONSEVERE_PRIVATE PALLET_RGB_ONLY_PRIVATE
export PALLET_ANALYSIS_START_EPOCH="$(date +%s)"
test ! -e "$PALLET_REPLAY_ROOT"
test ! -e "$PALLET_RGB_ONLY_PRIVATE"
mkdir "$PALLET_REPLAY_ROOT"
mkdir "$PALLET_RGB_ONLY_PRIVATE"
export PALLET_RGB_ONLY_NS=scripts/research/pallet_rgb_only_cvrank_20261011
export PALLET_PRIOR_NS=scripts/research/pallet_nonsevere_precision_20261011
git -C "$PALLET_RELEASE_CHECKOUT" archive --format=tar "$PALLET_RELEASE_REF" \
  "$PALLET_RGB_ONLY_NS" "$PALLET_PRIOR_NS" \
  _docs/experiments/pallet_nonsevere_precision_20261011 \
  scripts/research/pallet_training_free_compare_20261007_v1 \
  _docs/experiments/pallet_training_free_compare_20261007_v1 \
  | tar -xf - -C "$PALLET_REPLAY_ROOT"
cd "$PALLET_REPLAY_ROOT"
export PALLET_RGB_ONLY_DOC="$PALLET_REPLAY_ROOT/_docs/experiments/pallet_rgb_only_cvrank_20261011"
export PALLET_PRIOR_DOC="$PALLET_REPLAY_ROOT/_docs/experiments/pallet_nonsevere_precision_20261011"
export PALLET_RUNNER="$PALLET_PRIOR_NS/stage_runner.py"
mkdir -p "$PALLET_RGB_ONLY_DOC"
```

[확인] `contract_eval.py`와 `preserve_new.py`의 DOC는 코드 ROOT 아래의 위 고정 namespace다. `--public-dir`만 임의 경로로 바꾸면 이 두 script의 출력 경로가 바뀌지 않는다. 그래서 code-only replay ROOT를 만들고 같은 상대 namespace를 사용한다. 이전 namespace는 고정 입력으로 archive하며 수정하지 않는다.

비공개 입력은 원본 TRAIN memmap·등록된 synthetic RGB·OLD_P seed1/2 last checkpoint와 order, 기존 calibration·selection metadata, 실사 110장의 원본 RGB 및 SHA, 이전 현재 FP32 `NEW_PREDICTIONS_SEALED.jsonl.gz`, `NEW_POSES_SEALED.jsonl.gz`, `RAW_NEW_PREDICTIONS.npz`, 현재 detector captures/matches, `REFERENCE.json`, 기존 `SYNTHETIC_SELECTION_ROWS.json`을 포함한다. 실제 요구 경로·SHA는 [INPUT_REUSE_CONTRACT.json](INPUT_REUSE_CONTRACT.json), 이전 [REPRO_DEPENDENCIES.json](../pallet_nonsevere_precision_20261011/REPRO_DEPENDENCIES.json), 이전 [SOURCE_BINDINGS.json](../pallet_nonsevere_precision_20261011/SOURCE_BINDINGS.json)로 확인한다. RGB_ONLY fitting은 p3/p4 배열을 읽지 않지만 기존 data loader 초기화에 필요한 파일 존재 조건은 유지한다.

새 실행의 기존 변경 보존 기준을 먼저 기록한다. 아래 조회는 기존 source를 수정하지 않는다.

```bash
git --no-optional-locks -C "$PALLET_SOURCE_ROOT" status --porcelain=v1 -z --untracked-files=normal > "$PALLET_RGB_ONLY_PRIVATE/original-status.bin"
git --no-optional-locks -C "$PALLET_SOURCE_ROOT" diff --binary HEAD > "$PALLET_RGB_ONLY_PRIVATE/original-diff.bin"
"$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/preserve_new.py" snapshot --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --prior-private-dir "$PALLET_NONSEVERE_PRIVATE" --published-root "$PALLET_PUBLIC_ROOT" --wd-published-root "$PALLET_WD_PUBLIC_ROOT"
"$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/contract_eval.py" prepare
```

공개 PLAN은 이전 실행의 사전 고정 계약이다. replay에서는 그 바이트를 **학습 전에** 복사하고 두 training source SHA를 확인한다. 원래 작성 시각을 새 사전등록 시각이라고 표현하지 않는다. 입력 prepare의 source binding 검산이 실패하면 중단하고 입력 차이를 확인한다.

```bash
"$PALLET_CPU_PYTHON" -B - <<'PY'
import hashlib, json, os
from pathlib import Path
published = Path(os.environ['PALLET_RELEASE_CHECKOUT']) / '_docs/experiments/pallet_rgb_only_cvrank_20261011/PLAN_AND_CONTRACT.json'
raw = published.read_bytes()
plan = json.loads(raw)
assert plan['status'] == 'LOCKED_BEFORE_FITS'
for name, expected in plan['training_code_sha256'].items():
    source = Path(os.environ['PALLET_RGB_ONLY_NS']) / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
output = Path(os.environ['PALLET_RGB_ONLY_DOC']) / 'PLAN_AND_CONTRACT.json'
with output.open('xb') as stream:
    stream.write(raw)
PY
```

## 순차 재현 명령

GPU 작업은 아래 순서대로 한 프로세스씩 실행한다. preflight나 코드·입력·초기화·기울기·시간 게이트가 실패하면 뒤 단계로 진행하지 않는다. 완료 파일과 stage log는 exclusive 생성하며, 존재하면 중단한다. 실패를 삭제하고 같은 파일명으로 덮어쓰는 방식은 사용하지 않는다.

```bash
"$PALLET_CPU_PYTHON" -B "$PALLET_RUNNER" --stage cvrank --device cpu --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/contract_eval.py" cvrank
"$PALLET_GPU_PYTHON" -B "$PALLET_RUNNER" --stage rgb_preflight --device cuda --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_GPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/train_rgb.py" preflight --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --device cuda
"$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/feature_audit.py" prefit --public-dir "$PALLET_RGB_ONLY_DOC" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --source-root "$PALLET_SOURCE_ROOT" --prior-public-dir "$PALLET_PRIOR_DOC"
"$PALLET_GPU_PYTHON" -B "$PALLET_RUNNER" --stage rgb_train --device cuda --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_GPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/train_rgb.py" train --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --device cuda
"$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/feature_audit.py" final --public-dir "$PALLET_RGB_ONLY_DOC" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --source-root "$PALLET_SOURCE_ROOT" --prior-public-dir "$PALLET_PRIOR_DOC"
"$PALLET_GPU_PYTHON" -B "$PALLET_RUNNER" --stage rgb_calibration_selection --device cuda --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_GPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/calibration_selection.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --prior-private-dir "$PALLET_NONSEVERE_PRIVATE" --prior-public-dir "$PALLET_PRIOR_DOC"
"$PALLET_GPU_PYTHON" -B "$PALLET_RUNNER" --stage rgb_accuracy --device cuda --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_GPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/inference_new.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --prior-private-dir "$PALLET_NONSEVERE_PRIVATE" --prior-public-dir "$PALLET_PRIOR_DOC"
"$PALLET_CPU_PYTHON" -B "$PALLET_RUNNER" --stage evaluation_rgb --device cpu --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/contract_eval.py" rgb --predictions "$PALLET_RGB_ONLY_PRIVATE/RAW_RGB_ONLY_PREDICTIONS.npz"
"$PALLET_CPU_PYTHON" -B "$PALLET_RUNNER" --stage evaluation_finalize --device cpu --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/contract_eval.py" finalize
"$PALLET_CPU_PYTHON" -B "$PALLET_RUNNER" --stage statistics --device cpu --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/summarize_new.py" --public-dir "$PALLET_RGB_ONLY_DOC" --prior-public-dir "$PALLET_PRIOR_DOC"
```

[확인] 학습은 기존 OLD_P last6000을 공통 warm-start로 사용하고, 이전 M3와 같은 NativeEvidence 초기값과 seed별 첫 3,000×16 replay 순서를 확인한다. AdamW lr0.001, weight decay0.0001, betas(0.9,0.999), warmup100, cosine horizon3000·최종 fraction0.1, clip5를 유지한다. 손실은 CE + 0.15214983019356332×COORD, GEO0이다. 단독 RGB 모델을 처음부터 학습했을 때의 성능이나 모든 RGB-only의 가능성을 검증한 실험으로 해석하지 않는다.

[확인] calibration은 고정 합성 1,004장, T∈{0.5,1,2,4}, λ=1만 사용한다. 좁은 선택은 signed-front C1 249장을 제외한 기존 C2 selection 782장×두 seed이며 M1/M3의 기존 selection 원행을 재사용한다. 이전 M4 선택 기록은 보존한다. 새 REAL 통계가 나온 뒤 합성 선택을 바꾸지 않는다. 이전 분석에서 frozen synthetic 전체에 대한 수학·decoder 감사를 수행했던 이력도 보존되므로 untouched independent TEST라고 주장하지 않는다.

[확인] 현재 CVRANK cache는 Base q0 exact parity를 통과하지 못해 110장을 새 CPU 재계산했다. seed 복제는 독립 학습이 아니다. RGB_ONLY와 CVRANK_THEN_SUBPIX 모두 uncapped 좌표 → cornerSubPix → 원 Base 기준 영상 대각선 1% cap 한 번이며 center8·원 support를 유지한다. 자세는 공통 production F를 사용한다.

새 비용은 다섯 경로를 별도 프로세스로 순차 측정한다. 각 프로세스는 같은 18장 패널·warmup20·영상당5회·batch1이며 모델 load/영상 decode는 latency에서 제외한다. 기존 10경로 비용은 이전 영수증 SHA를 확인해 재사용하고 다시 실행하지 않는다.

```bash
for PALLET_COST_METHOD in RGB_ONLY RGB_ONLY_SUBPIX CVRANK_NATIVE CVRANK_CAP1 CVRANK_THEN_SUBPIX
do
  case "$PALLET_COST_METHOD" in
    CVRANK*) PALLET_COST_PREDICTIONS="$PALLET_RGB_ONLY_PRIVATE/CVRANK_PREDICTIONS_SEALED.jsonl.gz" ;;
    RGB_ONLY*) PALLET_COST_PREDICTIONS="$PALLET_RGB_ONLY_PRIVATE/RGB_ONLY_ALL_PREDICTIONS_SEALED.jsonl.gz" ;;
  esac
  "$PALLET_GPU_PYTHON" -B "$PALLET_RUNNER" --stage "latency_${PALLET_COST_METHOD}" --device cuda --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_GPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/latency_new.py" --method "$PALLET_COST_METHOD" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --prior-public-dir "$PALLET_PRIOR_DOC" --accuracy-predictions "$PALLET_COST_PREDICTIONS"
done
"$PALLET_CPU_PYTHON" -B "$PALLET_RUNNER" --stage latency_aggregate --device cpu --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/latency_new.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" --prior-public-dir "$PALLET_PRIOR_DOC" --aggregate
"$PALLET_CPU_PYTHON" -B "$PALLET_RUNNER" --stage figures --device cpu --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/figures_new.py" --public-dir "$PALLET_RGB_ONLY_DOC"
"$PALLET_CPU_PYTHON" -B "$PALLET_RUNNER" --stage preservation --device cpu --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/preserve_new.py" verify --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --prior-private-dir "$PALLET_NONSEVERE_PRIVATE" --published-root "$PALLET_PUBLIC_ROOT" --wd-published-root "$PALLET_WD_PUBLIC_ROOT"
"$PALLET_CPU_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
metadata = dict(status='CLEAN_REPLAY', repairs=[],
    preparation_failure_wall_seconds_not_instrumented=False,
    scientific_failed_runs=0)
with (Path(os.environ['PALLET_RGB_ONLY_DOC']) / 'IMPLEMENTATION_REPAIRS.json').open('x') as stream:
    json.dump(metadata, stream, indent=2)
    stream.write('\n')
PY
"$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/compute_new.py" --public-dir "$PALLET_RGB_ONLY_DOC" --analysis-start-epoch "$PALLET_ANALYSIS_START_EPOCH"
"$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/report_new.py" --public-dir "$PALLET_RGB_ONLY_DOC"
"$PALLET_CPU_PYTHON" -B - <<'PY'
import os
from pathlib import Path
original = Path(os.environ['PALLET_RELEASE_CHECKOUT']) / '_docs/experiments/pallet_rgb_only_cvrank_20261011/REPRODUCE.md'
with (Path(os.environ['PALLET_RGB_ONLY_DOC']) / 'REPRODUCE.md').open('xb') as stream:
    stream.write(original.read_bytes())
PY
"$PALLET_CPU_PYTHON" -B "$PALLET_RUNNER" --stage final_verify --device cpu --private-dir "$PALLET_RGB_ONLY_PRIVATE" --public-dir "$PALLET_RGB_ONLY_DOC" -- "$PALLET_CPU_PYTHON" -B "$PALLET_RGB_ONLY_NS/independent_verify.py" --public-dir "$PALLET_RGB_ONLY_DOC" --prior-public-dir "$PALLET_PRIOR_DOC" --private-dir "$PALLET_RGB_ONLY_PRIVATE" --source-root "$PALLET_SOURCE_ROOT" --prior-private-dir "$PALLET_NONSEVERE_PRIVATE" --allow-pending STAGE_final_verify.json --output-name INDEPENDENT_VERIFICATION.json
```

`STAGE_final_verify.json`는 wrapper의 child가 끝난 뒤 생성되므로 독립 검산 실행 중 그 한 영수증만 pending으로 허용한다. `COMPUTE_RECEIPT.json`는 직전까지의 단계만 합산하고 최종 감사의 wall을 별도로 검사한다. 재현 실행은 첫 준비 전에 새 CPU 분석 시작 epoch를 기록해 넘긴다. 원 실행의 시작 epoch와 실패 시도는 기존 기록으로 남기며 미래 replay에 원 시각을 새 시작으로 재사용하지 않는다. clean recipe의 `IMPLEMENTATION_REPAIRS.json`는 실패 없는 새 실행의 metadata이며 과거 실패를 복사하거나 재연하지 않는다. 실제 새 실패가 있으면 이 성공 recipe를 계속 실행하지 말고 별도 실패 기록을 보존한다.

통계는 방법×영상×seed 정렬을 확인하고 110장 안 두 seed를 묶는다. 9세션 cluster bootstrap 10,000회(seed20260917), 고정 family3의 중심화 양측 bootstrap p와 Holm을 재계산한다. 모든 산출 실패·회복·손상과 코너 개선/자세 악화 사례를 남긴다. 비유의 결과를 동등성·비열등성으로 해석하지 않는다. 110장은 반복 노출된 개발 집단이고 독립 최종 실사 시험이 아니다.

## 게시 및 재현 범위

학습·평가·검산이 끝나고 공개 필드·문서 링크·PNG·source 보존 검산이 통과한 결과만 게시한다. 실제 게시 작업은 root가 새 두 namespace의 파일을 명시적으로 stage하고 `main`에 일반 commit/push한다. 다른 변경을 자동 병합하지 않고 `git add .`, reset/stash, force push를 사용하지 않는다. 원격 main SHA·보고서·그림 링크 확인은 최종 게시 기록과 사용자 보고로 확인한다. 이 문서의 recipe는 Git push를 자동 실행하지 않는다.

공개 그래프는 실제 평가 원행에서 만든 통계·오류 사례이며 원본 영상·정답 원좌표를 포함하지 않는다. 추가 model/seed/loss/치수/geometry cache/합성 생성/하이퍼파라미터 탐색은 없다. 이전 N3/M0/M2/M4·PoseFix 및 원고/LaTeX/PDF는 보존한다. 공개 코드만으로 개인 데이터 권한을 대체할 수 없고, GPU 독립 실행의 미세 수치 차이와 기존 PoseFix+SubPix 최대 약0.94px parity 제한은 기존 기록과 함께 보고한다.
