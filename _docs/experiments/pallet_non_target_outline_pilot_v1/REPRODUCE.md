# 재현 안내

기본 재현은 **공개된 고정 non-target 마스크와 기존 원후보/원비용 cache를 이용하는 무추론 재선택**이다. 새 MobileSAM·YOLO forward, PnP, 자세 최적화, 후보 생성, 학습이나 threshold 탐색이 필요하지 않다. 원 실행의 supplier 추론은 encoder3·decoder3으로 완료됐으며 누락된 timing을 채우기 위해 반복하지 않는다.

## 필요한 원자료와 공개 자료

고정 원결과는 `407de865445e4987614f50f893d2ccea78d507b6`, 선택/비용 복원은 `c56a4e92027a8e56f6704773f24d7c3da78082f1`, 시작 main은 `4d509f66e0e544a970cd039afb5d33c94680d394`다. [PLAN_AND_CONTRACT.md](PLAN_AND_CONTRACT.md), [PUBLIC_DERIVATIONS.json](PUBLIC_DERIVATIONS.json), 공개 evidence와 최종 검산을 먼저 읽는다.

기존 승인 cache/Release에서 필요한 항목만 복원한다. 저장소나 대형 archive 전체를 새로 복제하지 않는다. 원자료 위치와 봉인 규약은 [이전 선택 점수 진단 재현 안내](../pallet_selection_score_diagnosis_v1/REPRODUCE.md)를 따른다.

| 별칭 | 필요한 내용 |
|---|---|
| `SOURCE_ROOT` | 이 버전의 저장소 코드,6개 원 RGB와 원 target mask의 원 상대경로/바이트 |
| `SCORE_CACHE` | 기존 `selection_v1/FRAME_SELECTIONS.jsonl.gz`, `RETAINED_CANDIDATES.jsonl.gz`, `cost_v1/COST_SEAL.json`, 봉인된 해당6개의 `cost_v1/frames/*_COSTS.jsonl.gz` |
| `STATE_CACHE` | 기존 `e_trace_v2/TYPED_PACKETS.jsonl.gz`, `score_v2/FIXED_REFERENCES.json`, `score_v2/SCORE_SEAL.json` |
| `PILOT_PUBLIC` | 이 문서의 디렉터리. `INPUTS_FROZEN.jsonl`, `inputs/masks/*.npy`, 원 실행 예측/비용/부모 장부와 evidence 포함 |
| `TASK_RUN` | 존재하는 이전 실험 출력과 겹치지 않는 새 재현 디렉터리 |

전351개 retained 후보의 dense outline 배열은 이미 저장된 값을 재사용한다. 원 bank에 reference flag 등이 함께 있어도 `prepare.py`가 allowed-field whitelist로 내보낸 neutral package만 selector에 전달한다. `prepare.py`는 privileged exporter이고 supplier/selector와 별도 프로세스다. supplier에는 RGB/지시만, selector에는 neutral geometry/cost/pose와 frozen mask만 전달한다. 참조는 prediction seal이 만들어진 뒤 evaluator에만 전달한다.

공개 `INPUTS_FROZEN.jsonl`의 mask 경로는 이 manifest 기준 `inputs/masks/{session__frame}.npy`다. ndarray는 원640×480 canvas의 uint8 0/1이며 SHA와 픽셀 바이트를 검사한다. 입력 없음3행은 검증된 빈 배경이 아니라 `NO_ADMISSIBLE_NON_TARGET_EVIDENCE`다. 마스크를 다시 그리거나 원 target의 차집합으로 대체하지 않는다. 원 지시는 `PROMPT_MANIFEST.jsonl`, 모든 대안/선택과 검토 경계는 `NON_TARGET_INPUT_MANIFEST.jsonl`, `evidence/MASK_REVIEW.json`, `evidence/INPUT_SEAL.json`에 있다. 상대경로/별칭 파생 manifest의 SHA는 원 private 실행 manifest의 SHA와 다를 수 있으며 그 연결을 `PUBLIC_DERIVATIONS.json`에 보존한다.

## 원후보 재선택과 평가

아래 경로는 예시 변수다. 실제 기존 cache 경로로 지정한다. 기존 환경에서 Python3.10, NumPy/SciPy/Pillow/pytest를 사용했으며 새 패키지 설치는 하지 않았다. 사용한 interpreter는 실행 evidence와 [INPUT_SUPPLIER_AUDIT.md](INPUT_SUPPLIER_AUDIT.md)에 기록했다. `TASK_RUN` 하위 출력은 모두 새 이름이어야 한다. 원 예측/참조/마스크를 덮어쓰지 않는다.

```bash
SOURCE_ROOT=/path/to/pallet-6d-pose
SCORE_CACHE=/path/to/existing/pallet-selection-score-diagnosis-20261016-v1
STATE_CACHE=/path/to/existing/pallet-statewise-correspondence-diagnosis-20261016-v1
TASK_RUN=/path/to/new/pallet-non-target-reproduction
PYTHON_BIN=/path/to/existing/python
PILOT_PUBLIC="$SOURCE_ROOT/_docs/experiments/pallet_non_target_outline_pilot_v1"
cd "$SOURCE_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TASK_RUN"

"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider \
  scripts/research/pallet_non_target_outline_pilot_v1/test_kernel.py

"$PYTHON_BIN" -B -m scripts.research.pallet_non_target_outline_pilot_v1.prepare \
  --source-root "$SOURCE_ROOT" --score-private "$SCORE_CACHE" \
  --state-private "$STATE_CACHE" --output "$TASK_RUN/neutral"

"$PYTHON_BIN" -B -m scripts.research.pallet_non_target_outline_pilot_v1.replay \
  --neutral-package "$TASK_RUN/neutral/NEUTRAL_PACKAGE.json" \
  --output "$TASK_RUN/empty"

"$PYTHON_BIN" -B -m scripts.research.pallet_non_target_outline_pilot_v1.replay \
  --neutral-package "$TASK_RUN/neutral/NEUTRAL_PACKAGE.json" \
  --input-manifest "$PILOT_PUBLIC/INPUTS_FROZEN.jsonl" \
  --output "$TASK_RUN/rerank"
```

각 replay는 전6장·351후보를 처리하고18개 prediction을 봉인한다. BASELINE_E 재현, 무입력 scalar·순위·pose 항등성과 원표본/부모 연결을 통과해야 한다. 원 G10 family 선택 후 G13 같은 family의 winner를 읽는 계층을 유지하며, 서로 다른 단계의 목적함수를 합쳐 순위화하지 않는다. reference/GT/audit-label 파일 읽기와 network/subprocess는 selector의 cooperative Python audit boundary에서 금지된다. 이는 OS/native 보안 sandbox를 주장하는 장치가 아니다.

`PREDICTION_SEAL.json`과 그 artifact SHA가 만들어진 뒤에만 다음 별도 프로세스를 실행한다. `--postseal-go`는 봉인 후 참조 결합의 명시적 gate다. 새 후보를 만들거나 다시 풀지 않는다.

```bash
"$PYTHON_BIN" -B -m scripts.research.pallet_non_target_outline_pilot_v1.evaluate \
  --prediction-folder "$TASK_RUN/rerank" \
  --neutral-package "$TASK_RUN/neutral/NEUTRAL_PACKAGE.json" \
  --state-private "$STATE_CACHE" --output "$TASK_RUN/evaluation" --postseal-go

"$PYTHON_BIN" -B -m scripts.research.pallet_non_target_outline_pilot_v1.diagnose \
  --run-folder "$TASK_RUN/rerank" \
  --neutral-package "$TASK_RUN/neutral/NEUTRAL_PACKAGE.json" \
  --state-private "$STATE_CACHE" --output "$TASK_RUN/diagnostic"

"$PYTHON_BIN" -B -m scripts.research.pallet_non_target_outline_pilot_v1.independent_check \
  --neutral-package "$TASK_RUN/neutral/NEUTRAL_PACKAGE.json" \
  --input-manifest "$PILOT_PUBLIC/INPUTS_FROZEN.jsonl" \
  --run "$TASK_RUN/rerank" --empty-run "$TASK_RUN/empty" \
  --evaluation "$TASK_RUN/evaluation" \
  --fixed-references "$STATE_CACHE/score_v2/FIXED_REFERENCES.json" \
  --reference-seal "$STATE_CACHE/score_v2/SCORE_SEAL.json" \
  --output "$TASK_RUN/INDEPENDENT_RECHECK.json"
```

위 독립 검산의 범위는 재선택·무입력·평가 산술이다. 원 실행의 사용자 파일 보호·공급기 raw NPZ·vendor/weight·공개 허가 검산은 공개 [INDEPENDENT_VALIDATION.json](INDEPENDENT_VALIDATION.json), [INDEPENDENT_PUBLICATION_CHECK.json](INDEPENDENT_PUBLICATION_CHECK.json)과 해당 evidence를 함께 확인한다. 과거 시작 시 snapshot을 새로 만든 것으로 바꿔 과거의 자료 보호를 증명하지 않는다.

## 비교할 항목

18행의 candidate/family/R/t/Q/accepted/fallback,351후보의 새 점수,508부모 레코드,18평가,12전이를 비교한다. 원 실행 prediction seal SHA256은 `004215e30bd9722e76649de1f62be9f238a83f05a26430b595a8d6c4436c789b`, prediction gzip SHA256은 `6c1c43d0fa25228035eee12d19f06c5cf81219ec7a612733539f21a55cc1cd1e`다. 공개 원 seal은 원 실행 증거로 변경하지 않는다.

재현 입력의 절대경로, 공개 manifest의 상대경로화, 실행 시간·소스 버전에 따라 재생성 seal SHA는 달라질 수 있다. 기존 seal에 맞추기 위해 시간이나 출처를 고치지 않는다. 원 수치는 `atol=1e-9, rtol=0`, ID/family/tie/순위/metadata는 정확 일치, 입력 없음은 scalar/순위/pose 정확 일치를 요구한다. 평가는 기존 참조에 대한 중심 오차와 physical/C2이며 동일 자세에서≤5cm AND≤5°다.

6행의 입력 상태와2/6 성공의 분모를 남기고 입력 없음·실패 지속·악화도 제외하지 않는다. 새 방법의 성공 후보만 고른 비교로 바꾸지 않는다. 이전 감사 범주나 참조 자세를 mask/선택 입력에 다시 넣지 않는다.

## 새 분할 추론을 실행하지 않는 이유

기존 MobileSAM vendor/checkpoint/API/고정 prompt와 최대 quality 규칙,9개 대안 보존,raw 출력 복구 이력은 [INPUT_SUPPLIER_AUDIT.md](INPUT_SUPPLIER_AUDIT.md)에 있다. 이 작업에서는 입력3개에 허용된 encoder/decode 호출을 이미 수행했다. 공개 frozen mask의 재선택에는 새 NN이 필요하지 않으며 마지막 packaging 실패나 저장되지 않은 프레임별 시간을 채우기 위해 재추론하지 않는다.

AUTO 의미 공급기는 확보되지 않았고 AUTO238,PRIMARY192/CAL46의 새 결과는 없다. 다른 모델 확보·새 prompt·새 mask·threshold/가중치 탐색은 위 cached 재현에 포함하지 않는다. 향후 별도 승인된 새 supplier 실행도 원 출력을 보존하고 별도 출력·실행 기록으로 남긴다.
