# 재현과 공개 수치 검산

[확인] 기존 Python/GPU 환경과 비공개 기존 입력을 사용한다. 새 학습·합성 생성·annotation·checkpoint 변경은 없으며 원고·기존 실험 결과를 덮어쓰지 않는다. 입력 SHA와 METHOD_LOCK, 245장 ID를 먼저 확인한다. 출력이 있으면 각 실행은 중단하며 새 출력 경로를 만들어 재현한다.

```bash
export PALLET_SOURCE_ROOT="/path/to/original/input-checkout"
export PALLET_BASELINE_ROOT="/path/to/existing/baseline-checkout"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PALLET_PYTHON="/path/to/existing/pallet-pose/bin/python"
export PALLET_RUNTIME_PYTHON="/path/to/existing/pallet-yolo26/bin/python"
export PALLET_PRIVATE_PAPER="/path/outside/repository/private-paper-output"
export PALLET_PAPER_OUTPUT="/path/to/nonexistent/fresh-paper-output"
mkdir -p "$PALLET_PAPER_OUTPUT"
for PALLET_PAPER_LOCK in METHOD_LOCK.json STATS_SOURCE_LOCK.json NUMERICAL_AMENDMENT_SOURCE_LOCK.json ROOT_EVALUATION_REVIEW_LOCK.json; do
  cp "_docs/experiments/pallet_paper_tables_245_20261011/$PALLET_PAPER_LOCK" "$PALLET_PAPER_OUTPUT/$PALLET_PAPER_LOCK"
done
```

[확인] 실행 순서는 입력/schema·표시 정밀도 parity → 기존 F seed1/4경로 및 PoseFix319 parity → 추론용 whitelist만 받는 전체319 새 좌표 생성·SHA봉인 → 기존 F 출력봉인·metric → 같은환경6경로 처리시간 → 표/짝 비교/독립 수치검산/그림/보고서다. 기존좌표 parity 채점의 참조는 새 생성기/F에 전달하지 않는다. 아래 코드의 진입점을 사용한다. 공개 수치 검산은 model·PnP를 새로 실행하지 않고 saved R/t/코너/개별 오차와 bootstrap draw를 대조한다.

- [입력 확인](../../../scripts/research/pallet_paper_tables_245_20261011/preflight.py)
- [좌표·F 평가](../../../scripts/research/pallet_paper_tables_245_20261011/evaluate.py)
- [통계·p/Holm](../../../scripts/research/pallet_paper_tables_245_20261011/stats.py)
- [표 생성](../../../scripts/research/pallet_paper_tables_245_20261011/tables.py)
- [그림·private RGB](../../../scripts/research/pallet_paper_tables_245_20261011/figures.py)
- [한국어 보고서](../../../scripts/research/pallet_paper_tables_245_20261011/report.py)

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.preflight
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.evaluate
"$PALLET_RUNTIME_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.latency
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.stats
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.correct_pvalues
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.tables
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.verify_corrected_stats
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.figures --private-dir "$PALLET_PRIVATE_PAPER"
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.report --private-dir "$PALLET_PRIVATE_PAPER"
```

[확인] 집계/평가는 기존 pallet-pose 환경을, 실제 latency는 기존 checkpoint와 호환되는 원래 pallet-yolo26 환경을 사용한다. 새 환경·package·checkpoint를 설치하거나 변경하지 않는다. 호환되지 않는 첫 로딩의0forward/F는 LATENCY_PREFLIGHT_REPAIR에 보존했다.

[확인] 수치 집계만 재현하려면 별도 fresh output에 고정 METHOD_LOCK과 공개 RECT_ROWS/SQUARE_ROWS를 복사한 뒤 stats부터 실행한다. 표2의 처리 시간 열을 동일하게 재현하려면 공개 LATENCY.json도 복사한다. 실행 시간을 새로 측정하는 경우 LATENCY를 복사하지 않고 위6경로 benchmark를 실행한다. 그림은 기존 audit RGB의 SHA가 일치할 때 private FIGURE_INPUTS.json을 자동 생성한다. 이 파일과 RGB그림8+3장은 저장소 밖에만 저장한다. 세션 master draw는 기존13세션 hash를 그대로 쓰며245장은 같은 draw의 고정 subset이다. Square는 기존118 frame draw다. 세부 실행횟수와 검산값은 실행 receipt와 METRICS/PAIRED에 남긴다.
