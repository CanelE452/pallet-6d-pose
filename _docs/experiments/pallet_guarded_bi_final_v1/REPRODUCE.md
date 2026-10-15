# 재현 안내

이 실험은 기존 고정 가중치·입력·캐시를 사용한다. 새 환경 설치/학습/seed/데이터/정답 변경 없이 원래 검증 환경에서 실행한다. SOURCE_ROOT는 원 자산 저장소, PRIOR는 앞선 minimal-bi-ablation 보존본, FOLDER는 새 후속 실험 폴더, REPORT는 새 보고서 폴더다. 기존 봉인 폴더를 덮어쓰지 않는다. 공개 작은 결과만으로 대형 입력·모델·마스크 원시 캐시가 포함됐다고 주장하지 않는다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m scripts.research.pallet_guarded_bi_final_v1.e_inputs --prior-input-dir "$PRIOR/inputs_v2" --source-root "$SOURCE_ROOT" --output-dir "$FOLDER/e_inputs"
python -m scripts.research.pallet_guarded_bi_final_v1.freeze --folder "$FOLDER" --prior "$PRIOR"
python -m scripts.research.pallet_guarded_bi_final_v1.run observe --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.run solve --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.run guard --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.score cal --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.score primary --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.analyze --folder "$FOLDER"
python -m pytest -q scripts/research/pallet_guarded_bi_final_v1
```

독립 검산은 현재 PREDICTION_SEAL.json과 score/SCORE_SEAL.json의 전체 SHA256을 명시해야 한다. 해당 값을 계산해 PREDSHA/SCORESHA 변수에 두고 새 검산 폴더로 실행한다. source/input이 달라졌거나 과거 실패가 있으면 그 시도를 보존한다.

```bash
python -m scripts.research.pallet_guarded_bi_final_v1.independent --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT" --output "$FOLDER/independent_v1" --prediction-seal-sha256 "$PREDSHA" --score-seal-sha256 "$SCORESHA"
python -m scripts.research.pallet_guarded_bi_final_v1.timing --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.report --folder "$FOLDER" --prior "$PRIOR" --output "$REPORT"
```

보고서 생성기는 score/RESULTS.json·SCORED_ROWS.json·DIAGNOSTICS.json·SELECTION_SEAL.json·독립 검산 영수증·실제 시간 결과와 ledger만 읽는다. 정답·원영상·관측·모델을 열거나 새 pose를 풀지 않는다. 보고서 입력과 소스 SHA는 REPORT_INPUTS.json에 있다. 공개 metrics·diagnostics·gallery 파일은 root의 별도 패키징 단계에서 원본 결과와 SHA를 확인해 복사한다.

성공 기준·physical/C2 alignment·전체238장·세부192/49/143/46 분모·가용 n·모든 no-pose/fallback을 유지한다. CAL 선택을 바꾸거나 PRIMARY 결과를 보고 새 후보/임계값을 추가하면 이 봉인 실험의 재현이 아니다.
