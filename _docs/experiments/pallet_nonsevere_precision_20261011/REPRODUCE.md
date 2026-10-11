Severe를 제외한 동일 110장에서 실제 30,000 updates·18경로 평가·독립 latency 측정을 완료했으며 재현 recipe와 실제 실행 기록을 구분한다.

[확인] 모델 학습·synthetic calibration/선택·새 실사 평가·통계·그림·비용 측정은 실제 완료했다. 단계별 receipt에 원래 실패와 수선 기록도 보존한다. 공개 코드만 clone한 환경에서는 비공개 RGB, 기존 feature cache 및 checkpoint가 없어 같은 실행을 완료할 수 없다. 필요한 기존 입력과 해시는 [REPRO_DEPENDENCIES.json](REPRO_DEPENDENCIES.json), [SOURCE_BINDINGS.json](SOURCE_BINDINGS.json), [DATA_CONTRACT.json](DATA_CONTRACT.json)에 있다.

[확인] 본 실험은 Clean 68장·Moderate 42장, 9개 세션의 동일 110장만 사용한다. Severe와 보호된 물리 FINAL 구간은 사용하지 않는다. 수동 코너로 만든 참조 자세와 비교하므로 독립 센서로 계측한 물리 정확도 검증이라고 하지 않는다. 기존 결과 폴더·원고·checkpoint는 수정하지 않으며 새 결과만 이 namespace에 쓴다.

실제 환경

[확인] CPU introspection을 실제 실행했다. Python 3.10.20, PyTorch 2.1.1+cu118, torchvision 0.16.1+cu118, OpenCV 4.9.0, NumPy 1.26.4, SciPy 1.12.0, Matplotlib 3.10.9, Ultralytics 8.4.60을 사용한다. GPU 장치는 RTX 3080 10,240 MiB, driver 580.178.04이며 PyTorch의 CUDA build는 11.8이다. 이 introspection은 CUDA context를 만들지 않았고 forward·optimizer update·PnP를 수행하지 않았다([ENVIRONMENT.json](ENVIRONMENT.json)).

[확인] model arithmetic은 FP32이며 AMP와 matmul/cuDNN TF32를 끈다. 기존 frozen neck feature capture의 p3/p4는 FP16으로 반올림된 다음 GenericPointRefiner에서 features.float()로 계산된다. 이 중간 반올림은 모든 feature 기반 경로에 공통이다. PoseFix native RGB는 FP32다. Native RGB tile은 원본 픽셀 간격을 유지하고 `/255 − 0.5`로 정규화한다.

입력 경로의 의미

| 변수 | 입력 또는 출력 |
|---|---|
| PALLET_SOURCE_ROOT | 원래 데이터·checkpoint·기존 source가 있는 저장소 |
| PALLET_PUBLIC_ROOT | 이전 paper-table 게시 worktree |
| PALLET_WD_PUBLIC_ROOT | 이전 W/D 게시 worktree |
| PALLET_NONSEVERE_PRIVATE | 이번 비공개 cache·checkpoint·sealed 좌표 폴더 |
| public-dir CLI 인자 | 이번 `_docs/experiments/pallet_nonsevere_precision_20261011` 출력 폴더 |

[확인] 공개 문서에 로컬 절대 경로나 원본 RGB는 적지 않는다. 기존 input을 동일 해시로 공급하고 환경 변수를 설정한 후, 각 단계는 code guard가 확인하는 사전 lock 순서대로 실행해야 한다. 존재하는 결과를 덮어쓰는 rerun은 허용하지 않는다. `contract.py`의 DOC는 해당 코드 ROOT 아래의 canonical experiment 경로로 고정되어 있다. 다른 `--public-dir`만 주면 contract 출력 위치까지 바뀌는 구조가 아니다.

[확인] 아래는 **새 재현을 위한 recipe**이며 이번 실험에서 이 전체 recipe를 다시 실행했다는 기록이 아니다. 새 namespace가 포함된 게시 commit을 checkout하고 code만 빈 ROOT로 archive한다. 감사의 문서 링크가 해석되도록 static 재현·dependency 설명 문서 두 개도 복사하되 과학 결과·receipt·checkpoint는 복사하지 않는다. SOURCE/PUBLIC/WD 입력은 별도로 공급하는 기존 읽기 전용 경로다. `PY`는 [ENVIRONMENT.json](ENVIRONMENT.json)의 패키지를 가진 interpreter이며 `PALLET_REPLAY_PARENT`와 `PALLET_PRIVATE_REPLAY_PARENT`는 사용자가 지정한 출력 parent 폴더다. 기존 게시 DOC는 삭제하지 않는다.

```bash
set -euo pipefail
: "${PY:?same recorded Python environment required}"
: "${PALLET_SOURCE_ROOT:?authenticated original inputs required}"
: "${PALLET_PUBLIC_ROOT:?authenticated prior paper inputs required}"
: "${PALLET_WD_PUBLIC_ROOT:?authenticated prior WD inputs required}"
: "${PALLET_REPLAY_PARENT:?new output parent required}"
: "${PALLET_PRIVATE_REPLAY_PARENT:?new private output parent required}"
NS=scripts/research/pallet_nonsevere_precision_20261011
REPLAY_PUBLICATION_ROOT="$PWD"
REPLAY_MAIN_SHA="$(git rev-parse HEAD)"
REPLAY_ROOT="$(mktemp -d "$PALLET_REPLAY_PARENT/pallet-nonsevere-replay-XXXXXX")"
export PALLET_NONSEVERE_PRIVATE="$(mktemp -d "$PALLET_PRIVATE_REPLAY_PARENT/pallet-nonsevere-private-XXXXXX")"
git archive HEAD "$NS" | tar -x -C "$REPLAY_ROOT"
cd "$REPLAY_ROOT"
DOC="$REPLAY_ROOT/_docs/experiments/pallet_nonsevere_precision_20261011"
mkdir -p "$DOC"
for STATIC_DOC in REPRODUCE.md REPRO_DEPENDENCIES.json; do
  git -C "$REPLAY_PUBLICATION_ROOT" show "$REPLAY_MAIN_SHA:_docs/experiments/pallet_nonsevere_precision_20261011/$STATIC_DOC" > "$DOC/$STATIC_DOC"
done
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" -B "$NS/contract.py"
"$PY" -B "$NS/evaluate.py" baseline
"$PY" -B "$NS/environment.py" --public-dir "$DOC"
"$PY" -B "$NS/preservation.py" snapshot --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" --published-root "$PALLET_PUBLIC_ROOT" --wd-published-root "$PALLET_WD_PUBLIC_ROOT"
"$PY" -B "$NS/geometry.py" --build --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/geometry_contract_audit.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/dry_run.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/train.py" calibrate --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/decoder_replay.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/decoder_completion.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/decoder_artifacts.py" --public-dir "$DOC"
"$PY" -B "$NS/selection.py" preflight --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/lock_plan.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" --published-root "$PALLET_PUBLIC_ROOT" --wd-published-root "$PALLET_WD_PUBLIC_ROOT" --main-sha "$REPLAY_MAIN_SHA"
"$PY" -B "$NS/train.py" train --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/stage_runner.py" --stage selection --device cuda --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/selection.py" run --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/detector_match_gate.py" prepare
"$PY" -B "$NS/stage_runner.py" --stage accuracy --device cuda --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/inference.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/detector_match_gate.py" validate
"$PY" -B "$NS/evaluate.py" new --predictions "$PALLET_NONSEVERE_PRIVATE/RAW_NEW_PREDICTIONS.npz"
for METHOD in BASE SUBPIX OLD_P OLD_P_SUBPIX N3_DIM_SYM N3_THEN_SUBPIX POSEFIX POSEFIX_SUBPIX M0 M0_SUBPIX M1 M1_SUBPIX M2 M2_SUBPIX M3 M3_SUBPIX M4 M4_SUBPIX; do
  "$PY" -B "$NS/latency.py" --method "$METHOD" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" --accuracy-predictions "$PALLET_NONSEVERE_PRIVATE/NEW_PREDICTIONS_SEALED.jsonl.gz"
done
"$PY" -B "$NS/latency.py" --aggregate --public-dir "$DOC"
"$PY" -B "$NS/stage_runner.py" --stage statistics --device cpu --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/stats_report.py" --public-dir "$DOC" --rows "$DOC/ROW_METRICS.jsonl.gz" --contract "$DOC/DATA_CONTRACT.json" --status "$DOC/TRAIN_RECEIPTS.json" --selection "$DOC/SYNTHETIC_SELECTION.json" --benchmark "$DOC/LATENCY.json" --report
"$PY" -B "$NS/stage_runner.py" --stage figures --device cpu --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/figures.py" --public-dir "$DOC" --rows "$DOC/ROW_METRICS.jsonl.gz" --selection "$DOC/SYNTHETIC_SELECTION.json" --private-predictions "$PALLET_NONSEVERE_PRIVATE/NEW_PREDICTIONS_SEALED.jsonl.gz" --private-reference "$PALLET_NONSEVERE_PRIVATE/REFERENCE.json" --private-contact-dir "$PALLET_NONSEVERE_PRIVATE/CASE_REVIEW" --source-root "$PALLET_SOURCE_ROOT" --runtime-inputs "$PALLET_NONSEVERE_PRIVATE/BASELINE_INPUTS.npz"
"$PY" -B "$NS/preservation.py" verify --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" --published-root "$PALLET_PUBLIC_ROOT" --wd-published-root "$PALLET_WD_PUBLIC_ROOT"
"$PY" -B "$NS/compute_receipt.py" --public-dir "$DOC"
"$PY" -B "$NS/stage_runner.py" --stage independent --device cpu --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/independent_verify.py" --public-dir "$DOC" --rows "$DOC/ROW_METRICS.jsonl.gz" --contract "$DOC/DATA_CONTRACT.json" --private-dir "$PALLET_NONSEVERE_PRIVATE" --owner-root "historical_source=$PALLET_SOURCE_ROOT" --owner-root "published_source=$PALLET_PUBLIC_ROOT" --owner-root "wd_published_source=$PALLET_WD_PUBLIC_ROOT"
```

[확인] 마지막 독립 감사는 완료된 source preservation·compute·보고서·그림 receipt를 읽는다. 원래 환경 기록과 no-update dry run은 inline으로 실제 실행했다. 위 `environment.py` 및 `dry_run.py`는 그 뒤 추가한 clean-replay entrypoint이고 원래 receipt를 다시 생성하지 않았다. 이 전체 recipe의 재학습·새 실사 평가를 두 번째로 완료했다고 주장하지 않는다. 위 recipe는 original interrupted draft를 의도적으로 재실행하지 않고 동일한 수식의 repaired decoder wrapper를 사용한다. 이 차이를 과거 실패가 없었던 것처럼 기록하지 않는다.

계산 정의

[확인] 후보 CE는 기존 222개 후보(221개 위치+null)를 유지한다. M1은 calibration 이전 temperature 1의 확률 기대좌표에 native-pixel SmoothL1을 직접 적용한다. Weighted expectation을 통해 logits까지 미분되며, 학습 후 synthetic calibration temperature를 별도로 정한다. 이 개념의 선행 연구는 [Integral Human Pose Regression](https://arxiv.org/abs/1711.08229)이다.

[확인] M2/M4는 reference 주변의 local pose sensitivity A를 미리 계산하고 detach한다. residual r에 대해 `C = A diag(r²) Aᵀ`에 해당하는 transformed cuboid corner 24차원 분산을 구한 뒤, 각 3D 코너 trace의 제곱근을 8개 평균한다. A와 고정 가중치를 학습하지 않으며 residual만 미분한다. 이는 [Linear-Covariance Loss §3.3](https://arxiv.org/html/2303.11516v2)의 E_cov를 사용한 local surrogate다. E_prior/E_linear까지 포함하는 LC 전체 Eq17이나 실제 T/R loss 재현은 아니다.

[확인] M3/M4의 작은 RGB CNN은 기존 위치 후보의 native tile에서 증거를 읽고 shared residual scorer에 공급한다. [Deep ChArUco](https://arxiv.org/abs/1812.03247)는 이미지 기반 subpixel refinement 후 PnP를 적용한 관련 선행 연구다. 마커 전용 원래 네트워크를 재현한 것은 아니며 그 논문의 정확도를 본 팔레트 실험에 이전하지 않는다.

[확인] candidate hull은 13방향×17반경의 outer radius `box diagonal × 0.08`이다. `stencil_fraction = 0.131037399...`는 feature context patch sampling 위치이며 후보 hull 반경과 다른 설정이다. 좌표 auxiliary eligibility는 원래 support, candidate hull, 영상 대각선 1% cap 안에 있는 참조 코너의 교집합으로 고정한다. 최종 SubPix 경로는 uncapped native 보정 좌표에서 SubPix를 적용한 뒤 원래 Base를 기준으로 cap을 마지막에 한 번 적용한다.

[확인] 모든 arm은 동일 24,387 parameters(기존 18,962, 추가 5,425)를 가진다. M0/M1/M2에서는 RGB CNN 2,768 parameters의 실행을 건너뛴다. RGB scorer 입력 열 512 parameters는 같은 81차원 matmul 안에 유지되지만 RGB 입력은 0이다. 이 두 수치를 합쳐 active CNN parameter 수라고 하지 않는다.

[확인] synthetic 자료 노출 범위를 구별한다. 추가 학습·optimizer·aux 가중치 산출은 TRAIN 55,980장만 사용한다. Decoder와 geometry의 수학적/frozen-source 검사는 60,000장 전체, 즉 heldout 1,985장을 포함한다. 기존 frozen OLD_P의 probability 진단은 seed별 calibration+selection+heldout 4,020장(지원 4,006장)을 재사용한다. 따라서 synthetic heldout을 미열람 독립 TEST라고 하지 않는다. 최종 새 모델 선택에는 calibration 1,004장의 동일 temperature 규칙과 selection 1,031장 중 사전 C2 조건의 782장만 사용하며 새 M0–M4의 heldout 성능 추론을 하지 않는다. 실사의 보호 physical FINAL 및 Severe의 현재 scoring은 0이다.

[확인] 첫 synthetic calibration/selection 프로세스는 미검출 row의 nonfinite box 때문에 native RGB tile 반경을 정수로 만들지 못해 60.986초 후 실패했다. 첫 stage receipt를 보존했다. 수선은 source Generic의 `[0,0,1,1]` fallback을 미사용 tile allocation에만 적용한 것으로 원래 model boxes/support/points/masks와 모든 782장 분모를 유지한다. 완료된 학습 30,000 updates 및 optimizer·가중치·선택 기준은 바뀌지 않았다. 실패와 수선 프로세스 시간을 모두 예산에 합산한다([게시된 원래 amendment](https://github.com/CanelE452/pallet-6d-pose/blob/main/_docs/experiments/pallet_nonsevere_precision_20261011/SELECTION_INVALID_BOX_ADAPTER_AMENDMENT.json), [게시된 원래 실패 receipt](https://github.com/CanelE452/pallet-6d-pose/blob/main/_docs/experiments/pallet_nonsevere_precision_20261011/STAGE_selection.json)). Clean replay는 수선된 현재 source를 사용하므로 과거 실패를 의도적으로 다시 생성하지 않는다.

실행 기록

[확인] 아래는 실제 실행한 명령에서 interpreter와 입력/출력 경로만 위 환경 변수로 바꾼 기록이다. 당시 source·원행·단계 로그는 receipt의 SHA로 연결된다. 시작 당시의 inline environment 및 no-update dry run은 실제 실행했지만 이후 작성한 `environment.py`/`dry_run.py`를 원래 실행한 것처럼 적지 않는다.

| 실제 단계 | 실행 결과 | 근거 |
|---|---|---|
| M0–M4 seed 1·2 마지막 step | 3,000 updates씩, 총 30,000, 10 fits | [TRAIN_RECEIPTS.json](TRAIN_RECEIPTS.json) |
| 최초 synthetic selection | FAILED, 60.986초, 추가 fit 없음 | [게시된 원래 STAGE_selection.json](https://github.com/CanelE452/pallet-6d-pose/blob/main/_docs/experiments/pallet_nonsevere_precision_20261011/STAGE_selection.json) |
| 수선 synthetic selection | DONE, 159.958초, M4 고정 | [게시된 원래 STAGE_selection_repaired.json](https://github.com/CanelE452/pallet-6d-pose/blob/main/_docs/experiments/pallet_nonsevere_precision_20261011/STAGE_selection_repaired.json), [SYNTHETIC_SELECTION.json](SYNTHETIC_SELECTION.json) |
| fresh precision prediction | DONE, 2,420 prefix rows | [STAGE_accuracy.json](STAGE_accuracy.json), [PRECISION_PREDICTION_SEAL.json](PRECISION_PREDICTION_SEAL.json) |
| detector-match·공통 평가 | DONE, 4,840 rows·18경로 | [CURRENT_DETECTOR_MATCH_PARITY.json](CURRENT_DETECTOR_MATCH_PARITY.json), [NEW_EVALUATION_RECEIPT.json](NEW_EVALUATION_RECEIPT.json) |
| 독립 프로세스 비용 | 18경로×20 warmup+90 timed calls, GPU 예약 wall 57.183초 | [LATENCY.json](LATENCY.json) |
| 통계·보고서 | DONE, 추가 F/forward/update 0 | [STAGE_statistics.json](STAGE_statistics.json), [STATISTICS_RECEIPT.json](STATISTICS_RECEIPT.json) |
| 사례·방법·CDF 그림 | DONE, 공개 PNG4 및 비공개 RGB 검수 시트2 | [STAGE_figures.json](STAGE_figures.json), [FIGURE_RECEIPT.json](FIGURE_RECEIPT.json) |

```bash
"$PY" -B "$NS/train.py" train --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/stage_runner.py" --stage selection --device cuda --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/selection.py" run --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/stage_runner.py" --stage selection_repaired --device cuda --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/selection.py" run --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/stage_runner.py" --stage accuracy --device cuda --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/inference.py" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC"
"$PY" -B "$NS/stage_runner.py" --stage detector_match_gate --device cpu --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/detector_match_gate.py" validate
"$PY" -B "$NS/stage_runner.py" --stage evaluation --device cpu --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/evaluate.py" new --predictions "$PALLET_NONSEVERE_PRIVATE/RAW_NEW_PREDICTIONS.npz"
for METHOD in BASE SUBPIX OLD_P OLD_P_SUBPIX N3_DIM_SYM N3_THEN_SUBPIX POSEFIX POSEFIX_SUBPIX M0 M0_SUBPIX M1 M1_SUBPIX M2 M2_SUBPIX M3 M3_SUBPIX M4 M4_SUBPIX; do
  "$PY" -B "$NS/latency.py" --public-dir "$DOC" --private-dir "$PALLET_NONSEVERE_PRIVATE" --source-root "$PALLET_SOURCE_ROOT" --method "$METHOD" --accuracy-predictions "$PALLET_NONSEVERE_PRIVATE/NEW_PREDICTIONS_SEALED.jsonl.gz"
done
"$PY" -B "$NS/latency.py" --aggregate --public-dir "$DOC"
"$PY" -B "$NS/stage_runner.py" --stage statistics --device cpu --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/stats_report.py" --public-dir "$DOC" --rows "$DOC/ROW_METRICS.jsonl.gz" --contract "$DOC/DATA_CONTRACT.json" --status "$DOC/TRAIN_RECEIPTS.json" --selection "$DOC/SYNTHETIC_SELECTION.json" --benchmark "$DOC/LATENCY.json" --report
"$PY" -B "$NS/stage_runner.py" --stage figures --device cpu --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/figures.py" --public-dir "$DOC" --rows "$DOC/ROW_METRICS.jsonl.gz" --selection "$DOC/SYNTHETIC_SELECTION.json" --private-predictions "$PALLET_NONSEVERE_PRIVATE/NEW_PREDICTIONS_SEALED.jsonl.gz" --private-reference "$PALLET_NONSEVERE_PRIVATE/REFERENCE.json" --private-contact-dir "$PALLET_NONSEVERE_PRIVATE/CASE_REVIEW" --source-root "$PALLET_SOURCE_ROOT" --runtime-inputs "$PALLET_NONSEVERE_PRIVATE/BASELINE_INPUTS.npz"
```

[확인] 위 latency 18개 실제 실행은 경로별 독립 프로세스였다. 같은 설정의 좌표 검산은 16경로 최대 0px, PoseFix 0.0001348px, PoseFix+SubPix 0.940033px 차이를 기록했다. 정확도와 비용 실행의 PoseFix+SubPix 좌표가 완전히 일치한다고 하지 않으며 이 수치 차이를 비용 비교의 제한으로 남긴다. 좋은 결과를 얻기 위한 재학습·재선택·측정 재시도는 하지 않았다.

[확인] 별도 CPU source probe는 같은 입력의 runtime/evaluator SubPix+cap 차이 0px와 작은 PoseFix 입력 차이가 0.940033px로 확대되는 현상을 재현했다. 정확한 PoseFix+SubPix 실행의 pre-SubPix tensor가 없어 GPU 차이의 발생 원인은 미확인이다. 이 진단은 CNN/F/update 0이며 조건을 변경하지 않았다([게시된 runtime SubPix probe](https://github.com/CanelE452/pallet-6d-pose/blob/main/_docs/experiments/pallet_nonsevere_precision_20261011/RUNTIME_SUBPIX_PARITY_PROBE.json)).

[확인] 첫 방법 그림의 화살표와 긴 라벨은 시각 검수 후 보완했다. 원PNG·원source·원FIGURE_RECEIPT는 보존하고 K/치수를 detector로 넣는 것처럼 보이던 그림만 고쳤다. 같은 CPU wrapper로 `figures.py --repair-pipeline --public-dir "$DOC" --private-dir "$PALLET_NONSEVERE_PRIVATE"`를 실제 실행했으며 다른 PNG·통계·CNN/F/update는 바뀌지 않았다([게시된 시각 수선 receipt](https://github.com/CanelE452/pallet-6d-pose/blob/main/_docs/experiments/pallet_nonsevere_precision_20261011/PIPELINE_VISUAL_REPAIR.json)). Clean replay는 보완된 그림을 처음부터 생성하므로 이 역사 수선 CLI를 다시 실행하지 않는다.

[확인] 마지막 독립 감사는 아래 CPU 명령으로 실제 실행했다. 산술·bootstrap·복구/손상 전이·원 source·학습량·비용 예산·그림 및 공개 계약의 판정은 [INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json)에 기록한다. 검산에는 새 학습·CNN forward·PnP를 넣지 않는다.

```bash
"$PY" -B "$NS/stage_runner.py" --stage independent --device cpu --private-dir "$PALLET_NONSEVERE_PRIVATE" --public-dir "$DOC" -- "$PY" -B "$NS/independent_verify.py" --public-dir "$DOC" --rows "$DOC/ROW_METRICS.jsonl.gz" --contract "$DOC/DATA_CONTRACT.json" --private-dir "$PALLET_NONSEVERE_PRIVATE" --owner-root "historical_source=$PALLET_SOURCE_ROOT" --owner-root "published_source=$PALLET_PUBLIC_ROOT" --owner-root "wd_published_source=$PALLET_WD_PUBLIC_ROOT"
```

검산 기준

[확인] headline 통계는 fresh `ROW_METRICS.jsonl.gz`만 사용한다. 역사적 `B0_ROW_METRICS.jsonl.gz`는 재사용 cache parity의 보존 기록이며 headline에 합치지 않는다. Primary는 M0–M4+SubPix 각각과 PoseFix-style+SubPix의 strict T<5cm AND R<5° 성공률 5개 대비다. Mechanism은 M1−M0/M2−M1/M3−M1/M4−M3의 4개 대비이며 별도 Holm family다.

[확인] 실제 9개 세션을 10,000회 재표집(seed 20260917)하고 영상 내 seed 1·2와 paired 방법을 함께 묶는다. 영상 수에 가중된 estimand를 사용한다. 중심화 양측 p는 정수 교차곱으로 동률을 계산하고 plus-one을 적용한다. 연속 오차는 산출된 seed 값을 먼저 영상 안에서 평균하고, 영상별 평균·표본 분산(ddof 1)·SD·중앙값·P90·P95 및 세션 bootstrap 평균 CI를 계산한다. Raw PCK@1/2/3/5/10px는 동일 valid reference 코너 분모이며 미예측은 miss다.

[확인] 독립 검산은 producer 통계 함수를 import하지 않고 math.fsum, 별도 표본 분산·sorted linear percentile, 별도 session accumulation으로 재계산한다. 통계 산술 PASS와 방법의 정확도 개선 여부는 서로 다른 결론이다. 성공 복구·손상 전이 및 모든 자세 실패를 보존한다. 공개 사례는 evaluation-aligned 실제 코너 도식이고 원본 RGB 검수 시트는 비공개 폴더에만 남긴다.
