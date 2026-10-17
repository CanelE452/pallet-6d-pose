# 재현 안내

이 실험은 **고정된 RGB 모델 보조 경계 + 기존 neutral 후보은행**의 점수/선택 재현이다. 입력을 다시 그리거나 새 detector/SAM/NN/PnP/최적화를 실행하지 않는다. 공개 작은 결과만으로 후보은행의 원 자료를 복원할 수는 없으며, 승인된 기존 cache의 정확한 bank와 fixed reference가 필요하다. 미보유 cache를 새 후보 생성으로 대신하지 않는다.

## 입력·환경

시작 main은 `f87a3d4fcfbf5c2bc638b3fbe5613a83f707bb72`, 원 결과는 `407de865445e4987614f50f893d2ccea78d507b6`, 기존 선택/비용 진단은 `c56a4e92027a8e56f6704773f24d7c3da78082f1`이다. [PLAN_AND_CONTRACT.md](PLAN_AND_CONTRACT.md), [INPUT_BINDINGS.json](INPUT_BINDINGS.json), [PROTOCOL_LOCK.json](PROTOCOL_LOCK.json)으로 실제 파일을 연결한다. 전체 저장소·release·11,603개 원 비용을 재계산하지 않는다.

기존 환경 Python3.10의 NumPy/OpenCV/Pillow/SciPy/pytest를 사용한다. 실제 interpreter는 `pallet-yolo26` 환경이었고 새 설치가 없다. 아래 변수는 예시 위치이며 기존 파일을 가리켜야 한다. `TASK_RUN`은 기존 결과와 겹치지 않는 새 디렉터리다. 원 neutral package SHA는 `ec62902cf6e42ccabaa6a66a65339957d08c17abc445a2900a7735b9a806c7dc`이며 같은 디렉터리에 `NEUTRAL_FRAMES.jsonl.gz`, `NEUTRAL_CANDIDATES.jsonl.gz`, `NEUTRAL_RGB.jsonl`, `EXPORT_RECEIPT.json`이 함께 있다. 이전 입력보조 시험의 whitelist exporter 산출물을 바이트 그대로 재사용했다. 이전 bank exporter 안내는 [이전 재현 문서](../pallet_non_target_outline_pilot_v1/REPRODUCE.md)에 있다.

새 `VISIBLE_INPUTS.jsonl`은 원640×480 좌표의 열린 polylines이며6장 전부 AVAILABLE이다. 원 RGB/입력 비교만으로 동결한 관측을 재사용한다. 공개 경로 별칭 파생본은 원 manifest SHA와 다를 수 있으므로 [PUBLIC_DERIVATIONS.json](PUBLIC_DERIVATIONS.json)을 대조한다. 입력 경계는 [VALIDATION_METHOD.md](VALIDATION_METHOD.md)에 설명한다.

## 점수·선택: 참조 없는 별도 프로세스

```bash
SOURCE_ROOT=/path/to/pallet-6d-pose
NEUTRAL_DIR=/path/to/existing/neutral_v1
STATE_CACHE=/path/to/existing/pallet-statewise-correspondence-diagnosis-20261016-v1
TASK_RUN=/path/to/new/visible-boundary-reproduction
PYTHON_BIN=/path/to/existing/python
PUBLIC="$SOURCE_ROOT/_docs/experiments/pallet_visible_boundary_pilot_v1"
cd "$SOURCE_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TASK_RUN"

"$PYTHON_BIN" -B -m pytest -q -p no:cacheprovider \
  scripts/research/pallet_visible_boundary_pilot_v1/test_kernel.py

"$PYTHON_BIN" -B -m scripts.research.pallet_visible_boundary_pilot_v1.replay \
  --neutral-package "$NEUTRAL_DIR/NEUTRAL_PACKAGE.json" \
  --output "$TASK_RUN/empty"

"$PYTHON_BIN" -B -m scripts.research.pallet_visible_boundary_pilot_v1.replay \
  --neutral-package "$NEUTRAL_DIR/NEUTRAL_PACKAGE.json" \
  --inputs "$PUBLIC/VISIBLE_INPUTS.jsonl" \
  --output "$TASK_RUN/actual"
```

각 replay는6장·351후보·네 점수와24개 출력을 처리한다. `empty`는 세 신규 경로의 scalar/전체 rank/pose가 원 e와 같은 기술 검사이며 가시경계 처치의 안전성 성능으로 집계하지 않는다. `actual/PREDICTION_SEAL.json`은 reference를 열기 전에 원 e candidate/family/pose, 전체 후보 수·선택 출력·artifact SHA·실제 read/runtime guard를 고정한다. G10 family 선택 후 같은 family의 G13을 선택하며 stage별 목적함수를 서로 비교하지 않는다. cooperative Python scientific-input guard는 참조/감사 범주와 NN/solver 호출을 차단하고 OS/native 보안 sandbox라고 주장하지 않는다.

## 봉인 후 평가·원인 진단·독립 검산

```bash
"$PYTHON_BIN" -B -m scripts.research.pallet_visible_boundary_pilot_v1.evaluate \
  --prediction-folder "$TASK_RUN/actual" \
  --neutral-package "$NEUTRAL_DIR/NEUTRAL_PACKAGE.json" \
  --state-private "$STATE_CACHE" --output "$TASK_RUN/evaluation" --postseal-go

"$PYTHON_BIN" -B -m scripts.research.pallet_visible_boundary_pilot_v1.diagnose \
  --run-folder "$TASK_RUN/actual" \
  --neutral-package "$NEUTRAL_DIR/NEUTRAL_PACKAGE.json" \
  --state-private "$STATE_CACHE" --output "$TASK_RUN/diagnostic"

"$PYTHON_BIN" -B -m scripts.research.pallet_visible_boundary_pilot_v1.independent_check \
  --neutral-package "$NEUTRAL_DIR/NEUTRAL_PACKAGE.json" \
  --input-manifest "$PUBLIC/VISIBLE_INPUTS.jsonl" \
  --run "$TASK_RUN/actual" --empty-run "$TASK_RUN/empty" \
  --evaluation "$TASK_RUN/evaluation" \
  --fixed-references "$STATE_CACHE/score_v2/FIXED_REFERENCES.json" \
  --reference-seal "$STATE_CACHE/score_v2/SCORE_SEAL.json" \
  --output "$TASK_RUN/independent.json"
```

fixed reference SHA는 `d488564817a8cac8d0103350f03ac61d78bdeb56163c6a5a4daffe414f0f2af1`이다. 평가자는 prediction seal을 확인한 뒤 참조를 열고 동일 후보 중심 T≤5cm AND physical/C2 R≤5°를 적용한다. 사후 진단에서만 모든351후보의 성공 표시를 결합해 순수 A_E/결합점수/단계·family를 비교한다. 진단의 성공 후보 ID를 selector 입력으로 다시 쓰지 않는다.

독립 검산기는 새 kernel/replay/selection을 import하지 않고 자기 hull·finite distance·길이 질량·robust loss·원 stable comparator·계층과 C2 오차를 계산한다. 의미별 숫자 허용치는 기존1e-9,rtol0이고 후보/family/order/status는 exact다. 새 입력이 없을 때 전체24출력 항등성도 따로 검증한다. 후보의 native R/t/Q/XYZ를 함께 사용하며 다른 family 회전만 섞지 않는다.

[METRICS.csv](METRICS.csv)24행, [CANDIDATE_SCORES.jsonl.gz](CANDIDATE_SCORES.jsonl.gz)351행, [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz)24행, [TRANSITIONS.csv](TRANSITIONS.csv)18행, [VISIBLE_VS_MASS_TRANSITIONS.csv](VISIBLE_VS_MASS_TRANSITIONS.csv)6행을 각각 집합/행수/전량값으로 대조한다. 재현의 gzip 포장 시각·경로 metadata가 달라지면 gzip 바이트 SHA 자체보다 봉인된 논리 레코드/출처 관계를 함께 확인한다. 원 실행 공개 파생본의 SHA 연결은 별도 manifest에 있다.

공간 맥락 진단도 새 threshold 없이 재현할 수 있다. 이 명령은 evaluation receipt가 해당 prediction seal에 연결되고 frozen input SHA가 맞는지 먼저 검사한다.

```bash
"$PYTHON_BIN" -B -m scripts.research.pallet_visible_boundary_pilot_v1.observation_context \
  --run-folder "$TASK_RUN/actual" \
  --input-manifest "$PUBLIC/VISIBLE_INPUTS.jsonl" \
  --prior-parents "$SOURCE_ROOT/_docs/experiments/pallet_non_target_outline_pilot_v1/PARENT_SUPPORT.jsonl" \
  --evaluation-receipt "$TASK_RUN/evaluation/EVALUATION_RECEIPT.json" \
  --output "$TASK_RUN/observation_context"
```

공간 비율은 nearest 옛 부모의 삭제 여부이며 물리 edge 정답·복원율이 아니다. [OBSERVATION_CONTEXT_KO.md](OBSERVATION_CONTEXT_KO.md)에 해석 범위를 남겼다.

이전 N0/PoseFix/부모 제외는 [CONTROL_COMPARISON.csv](CONTROL_COMPARISON.csv)의 저장 결과를 읽으며 새 forward나 pose solve를 하지 않는다. 갤러리는 먼저 [gallery/index.html](gallery/index.html)의 원 RGB/새 관측만 보고, 그 뒤 저장 target mask/hull·옛 삭제 부모·네 최종 pose를 본다. 원영상 crop은 확대 보조용이고 전체 장면을 함께 보존한다.

이번 실제 실행 비용·기술 실패 및 수정·원 파일 보존은 [COST.json](COST.json), [VALIDATION.json](VALIDATION.json)에 있다. 새 contour 의미 버전·가중치/threshold 탐색·192장 확장·운영 기본값 변경·논문 표 변경은 재현 범위에 없다.
