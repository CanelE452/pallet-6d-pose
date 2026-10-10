# 재현

[확인] 기존 환경과 비공개 원본 입력을 읽는다. 새 학습·모델 forward 없이 fresh output에만 쓴다. 원본 RGB 참조 sheet는 저장소 밖 private-dir만 허용한다.

```bash
export PALLET_PYTHON="/path/to/existing/pallet-pose/bin/python"
export PALLET_SOURCE_ROOT="/path/to/original/input-checkout"
export PALLET_BASELINE_ROOT="/path/to/existing/baseline-checkout"
export PALLET_SQUARE_OUTPUT="/path/to/nonexistent/fresh-square-output"
export PALLET_PRIVATE_SQUARE="/path/to/private/review-output"
export PYTHONDONTWRITEBYTECODE=1
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.preflight
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.references --private-dir "$PALLET_PRIVATE_SQUARE"
# 2D parity 및 REFERENCE_GATE.proceed_to_evaluation=true일 때만 계속한다.
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.evaluate
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.summarize
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.report
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.verify --source-root "$PALLET_SOURCE_ROOT" --baseline-root "$PALLET_BASELINE_ROOT"
```

[확인] 각 스크립트는 기존 출력이 있으면 멈춘다. source/annotation/checkpoint는 수정하지 않는다. 공개 수치 검산은 원 RGB나 checkpoint 없이도 가능하다:

```bash
unset PALLET_SQUARE_OUTPUT
"$PALLET_PYTHON" -B -m scripts.research.pallet_square6d_manualpnp_20261010.verify --output /path/to/nonexistent/public-recheck.json
```

[확인] 2D parity는 기존 두 manual 분모의 결과를 다시 계산하고 비교한다. reference는 예측을 열지 않는 별도 함수이며 잔차 gate를 통과해야 평가가 실행된다. 독립 verifier는 PnP·모델·F를 호출하지 않는다. 다른 사용자 재현에서는 원세션의 private 보존 snapshot을 만들어 낸 것처럼 표시하지 않는다.
