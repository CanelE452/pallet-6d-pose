실제 실행한 D 명령은 아래 순서이며, fitting과 solver 실행 전에 PLAN_LOCK 및 별도 D source lock을 저장했다.

[확인] `PY`는 기존 `pallet-yolo26` 환경의 Python, `SOURCE`는 원래 read-only 저장소, `PUBLIC`은 과거 게시 스냅샷, `PRIVATE`는 이번 비공개 출력 namespace, `OUT`은 이번 공개 산출 namespace다. 아래 변수는 실행에 쓴 실제 경로의 역할을 나타내며 공개 파일에 로컬 private 절대경로를 넣지 않는다. Python과 라이브러리를 업그레이드하지 않았다.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PY" -B scripts/research/pallet_wd_info_fast_20261011/d_local.py contract --source-root "$SOURCE" --private-dir "$PRIVATE" --public-dir "$OUT"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PY" -B scripts/research/pallet_wd_info_fast_20261011/d_local.py throughput --source-root "$SOURCE" --private-dir "$PRIVATE" --public-dir "$OUT"
```

[확인] throughput은 optimizer update 0이며, 동일한 zero-initialized scorer 두 arm에서 loss backward를 각각 6회 했다. 실제 fitting은 다음 명령의 각 arm1,000 updates뿐이다. root의 PLAN_LOCK 파일이 준비되고 GPU 단독 실행 시간이 조율된 뒤 실행했다.

```bash
export PALLET_SOURCE_ROOT="$SOURCE"
export PALLET_PUBLIC_ROOT="$PUBLIC"
export PALLET_FAST_PRIVATE="$PRIVATE"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PY" -B scripts/research/pallet_wd_info_fast_20261011/d_local.py train --source-root "$SOURCE" --private-dir "$PRIVATE" --public-dir "$OUT" --plan-lock "$OUT/PLAN_LOCK.json" --device-seconds-limit 300
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PY" -B scripts/research/pallet_wd_info_fast_20261011/d_evaluate.py calibrate --source-root "$SOURCE" --published-root "$PUBLIC" --private-dir "$PRIVATE" --public-dir "$OUT"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PY" -B scripts/research/pallet_wd_info_fast_20261011/d_evaluate.py predict --source-root "$SOURCE" --published-root "$PUBLIC" --private-dir "$PRIVATE" --public-dir "$OUT"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PY" -B scripts/research/pallet_wd_info_fast_20261011/d_evaluate.py evaluate --source-root "$SOURCE" --published-root "$PUBLIC" --private-dir "$PRIVATE" --public-dir "$OUT"
```

[확인] 최초 `evaluate`의 final solver에는 K가 list로 전달돼 무효 산출물이 생겼다. 이를 덮어쓰거나 방법 실패로 해석하지 않는다. 실제 실행은 봉인된 예측과 selector 후보를 재사용하여 다음 recovery로 original final fit을 완료했다. 반복 fitting은 0이며 실패 및 검산을 포함한 D 누적 OpenCV solver 호출은 8,495다. 기존 파일이 있는 곳에 그대로 재실행하면 overwrite 방지 assertion으로 종료한다.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$PY" -B scripts/research/pallet_wd_info_fast_20261011/d_recover.py --source-root "$SOURCE" --published-root "$PUBLIC" --private-dir "$PRIVATE" --public-dir "$OUT"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 "$PY" -B scripts/research/pallet_wd_info_fast_20261011/d_verify.py --source-root "$SOURCE" --private-dir "$PRIVATE" --public-dir "$OUT"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 "$PY" -B scripts/research/pallet_wd_info_fast_20261011/d_diagram.py --output _docs/experiments/pallet_wd_info_fast_20261011/FIGURES/D_LOCAL_METHOD.png
```

[확인] 기존 PRIVATE/source data가 있어야 재현할 수 있다. RAW RGB, 참조 원본, 모델 checkpoint와 대형 feature cache는 새 공개 commit에 들어가지 않는다. 최종 분석 파일은 [D_ARTIFACT_INDEX.json](D_ARTIFACT_INDEX.json)이 지정하는 REPAIRED 산출물이다.
