# 재현 및 공개 산출물 검산

[확인] 기존 학습 환경을 사용한다. 설치, 재학습, 새 RGB·합성 데이터·수동 주석 생성은 없다. 정확도 재현에는 원본 비공개 입력이 필요하며 공개 JSON의 통계·그림 검산은 원본 RGB나 checkpoint 없이 가능하다.

```bash
export PALLET_PYTHON="/path/to/existing/pallet-pose/bin/python"
export PALLET_SOURCE_ROOT="/path/to/original/pallet-pose"
export PALLET_BASELINE_ROOT="/path/to/existing/baseline-inputs"
export PALLET_PRIVATE_OUTPUT="/path/to/private/replay-records"
export PALLET_LEGACY_QA_SOURCE="/path/to/existing/qa_risk.py"
export PALLET_VIS_OUTPUT="/path/to/nonexistent/fresh-vis-results"
export PYTHONDONTWRITEBYTECODE=1
# 게시한 저장소 checkout의 루트에서 실행한다.
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.replay --output "$PALLET_VIS_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.square_audit --doc "$PALLET_VIS_OUTPUT" --legacy-qa "$PALLET_LEGACY_QA_SOURCE"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.figures --doc "$PALLET_VIS_OUTPUT" --private-cases "$PALLET_PRIVATE_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.verify --doc "$PALLET_VIS_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.report --doc "$PALLET_VIS_OUTPUT"
```

[확인] `replay --output`은 존재하지 않는 새 출력 디렉터리에서 `preflight`(방법·마스크 봉인, A0 ALL parity)→소스 SHA lock→`synth`(A1 ALL/VIS·집계·gate)→gate 통과 시에만 `real`과 `summarize_real`을 호출한다. A1 WORSENED/NOT_ESTIMABLE이면 REAL F를 호출하지 않는다. 기존 게시 또는 중단 산출물을 덮어쓰지 않는다. 위 명령은 학습 모델을 새로 실행하지 않으며 고정된 예측과 기존 영상·기하 입력을 사용한다.

[확인] `square_audit`와 `verify`의 새 결과는 별도로 생성된다. `verify`에서 비공개 preservation 인자를 생략하면 공개 수치 검산은 수행하고 원 checkout 보존 검사는 NOT_RUN_PUBLIC_ONLY로 표시한다. 이번 실행의 보존 기록은 원 세션의 START/snapshot을 필요로 하므로 다른 사용자 재현에서 그 기록을 만들어 낸 것처럼 보고하지 않는다.

```bash
# 원본 RGB·checkpoint 없이 게시한 공개 산출물을 다시 검산한다.
export PALLET_PUBLIC_DOC="$PWD/_docs/experiments/pallet_vispnp_square6d_20261011"
export PALLET_RECHECK_OUTPUT="/path/to/nonexistent/public-recheck.json"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.verify --doc "$PALLET_PUBLIC_DOC" --output "$PALLET_RECHECK_OUTPUT"
# 아래 두 명령은 저장된 수치의 파생 PNG와 Markdown만 재생성한다.
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.figures --doc "$PALLET_PUBLIC_DOC"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.report --doc "$PALLET_PUBLIC_DOC"
```

[확인] square_audit는 기존 `qa_risk.py`의 텍스트와 AST를 읽고 원래 threshold를 기록하며 그 QA나 PnP를 실행하지 않는다. 기존 output이 있으면 보존하고 멈춘다. source·manual JSON·이미지 SHA를 읽기 전후 비교한다. B2/B3 계약 충돌을 bypass하거나 B4 승인 없이 6D 평가하는 명령은 없다.

[확인] 공개 PNG는 수치 그림이다. 실제 RGB 사례는 `--private-cases`를 명시할 때에만 저장하며 Git 저장소 내부 출력은 거부한다. A1은 기존 SOURCE_MANIFEST의 heldout RGB에서 reflection padding 100 px를 제거한 뒤 raw qFinal을 그대로 겹친다. A2는 고정 INPUT_AUDIT의 실사 원영상 binding을 사용한다. `cases_A1_SYNTH.png` 또는 `cases_A2_REAL.png`와 receipt는 비공개 local directory에 남긴다. 정사각형의 새 참조 overlay는 생성하지 않았다.
