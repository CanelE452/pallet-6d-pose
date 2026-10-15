# 재현

기존환경과확정가중치·정확한원캐시를사용한다. PRIOR는앞선minimal보존본, SOURCE_ROOT는원자산저장소, FOLDER는새출력폴더, REPORT는새보고서폴더다. 기존봉인을덮어쓰지않는다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m scripts.research.pallet_single_component_bi_v1.inputs --prior-input-dir "$PRIOR/inputs_v2" --source-root "$SOURCE_ROOT" --output "$FOLDER/inputs_v1"
python -m scripts.research.pallet_single_component_bi_v1.run freeze --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_single_component_bi_v1.run observe --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_single_component_bi_v1.run solve --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_single_component_bi_v1.score --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_single_component_bi_v1.diagnostics --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT" --output "$FOLDER/diagnostics"
python -m pytest -q scripts/research/pallet_single_component_bi_v1
```

독립검산은실제완료PREDICTION_SEAL과SCORE_SEAL의전체SHA를PREDSHA/SCORESHA로명시한다. 새검산폴더를사용하고실패시도를보존한다.

```bash
python -m scripts.research.pallet_single_component_bi_v1.independent --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT" --output "$FOLDER/independent_v1" --prediction-seal-sha256 "$PREDSHA" --score-seal-sha256 "$SCORESHA"
python -m scripts.research.pallet_single_component_bi_v1.report --folder "$FOLDER" --prior "$PRIOR" --output "$REPORT"
```

감사의기존v1JSON·245/110출처감사·원격main관찰은FOLDER/compliance_v1에보존된메타데이터다. 보고서생성기는완료점수·독립검산·감사만읽으며정답·영상·모델을열거나pose를풀지않는다. 초기화별mask관측476개와solver후보등대형원배열은비공개보존본에있고공개작은파일이전체자산을포함한다고주장하지않는다.

재개는동일source/input/model/config와자기관측binding을통과한완료행만재사용한다. 기준선R0/R1/N0/N1예측과지표의정확한불변,새N3/SubPix초기캐시재현,모든실패/분모/물리C2정의를검산한다. source해시·정확명령·완료seal은evidence/metrics원결과와REPORT_INPUTS.json에있다. 사용자의최신지시에따라최종main게시는root가별도수행·검증한다. 운영기본값과논문변경은없다.
