# 재현과 보존

연구 브랜치 `research/minimal-bi-ablation-20261016-v1`, 기반 커밋 `f1a30071382f6304fe13d6c546b8f8033c6b9f49`를 사용한다. 기존 공개 데이터/캐시/가중치의 정확한 SHA는 contract/INPUT_CONTRACT_V1.json과 PROTOCOL.json에 있다. 이미지를 새로 받거나 학습하지 않는다. 자산이 없는 환경에서는 이 정확한 바인딩을 먼저 복원해야 하며 다른 seed/가중치로 대체하지 않는다.

로컬 변수 `PROJECT`는 이 브랜치의 체크아웃, `SOURCE`는 기존 자산이 복원된 저장소, `ARTIFACT_ROOT`는 새 출력 폴더로 지정한다. 간단한 재현에서는 SOURCE=PROJECT로 둔다. 별도 worktree를 쓰면 자산을 PROJECT의 같은 상대경로에 읽기 전용 링크로 연결한다. 공개 문서에 개인 절대경로를 넣지 않는다. 실행 기록의 원본 경로는 비공개 로컬 manifest에 보존하고 공개본은 역할 토큰으로 표시한다.

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest scripts/research/pallet_minimal_bi_ablation_v1 -q -p no:cacheprovider
python -m scripts.research.pallet_minimal_bi_ablation_v1.inputs prepare --source-root "$SOURCE" --output "$ARTIFACT_ROOT/inputs" --docs "$ARTIFACT_ROOT/input_contract"
python -m scripts.research.pallet_minimal_bi_ablation_v1.batch parity --root "$PROJECT" --output "$ARTIFACT_ROOT/parity" --device cuda
```

입력 준비는 predictor whitelist와 기존 F 함수만 사용한다. 기존102 RAW F를 재계산 대조하고 누락136개를 같은 F로 만든다. parity는 기존10장×2의 실제 관측과 자세를 재실행한다. 회귀 성공 검사는 봉인된 parity 결과와 기존 선택10장 정답만 사용하는 별도 `regression.py`다. 전체238장 예측의 정답 scorer는 따로 둔다.

다음 명령으로 고정 참조 루트의 경로/SHA만 이월하고 좌표/자세를 해독하지 않는다. 회귀 검사는 선택10장으로 제한되며0/7/10/10 및 모든 사례 판정 일치를 검사한다. 기존 출력 폴더는 덮어쓰지 않는다.

```bash
python - "$SOURCE" "$ARTIFACT_ROOT" <<'PY'
import sys
from pathlib import Path
from scripts.research.pallet_minimal_bi_ablation_v1 import scoring
source, artifact = map(Path, sys.argv[1:])
seal = artifact / 'inputs/INPUT_SEAL.json'
scoring.prepare_reference_manifest(source_root=source, input_seal_path=seal,
    expected_input_seal_sha256=scoring.sha(seal),
    output=artifact / 'SCORING_REFERENCE_MANIFEST.json', root_metadata_GO=True)
PY
python -m scripts.research.pallet_minimal_bi_ablation_v1.regression --source-root "$SOURCE" --parity-dir "$ARTIFACT_ROOT/parity" --manifest "$ARTIFACT_ROOT/SCORING_REFERENCE_MANIFEST.json" --output "$ARTIFACT_ROOT/REGRESSION_SUCCESS_CHECK.json" --root-regression-GO
```

```bash
python -m scripts.research.pallet_minimal_bi_ablation_v1.batch freeze --root "$PROJECT" --input-dir "$ARTIFACT_ROOT/inputs" --output "$ARTIFACT_ROOT/run" --parity-dir "$ARTIFACT_ROOT/parity"
python -m scripts.research.pallet_minimal_bi_ablation_v1.batch observe --root "$PROJECT" --input-dir "$ARTIFACT_ROOT/inputs" --output "$ARTIFACT_ROOT/run" --pilot --device cuda
python -m scripts.research.pallet_minimal_bi_ablation_v1.batch solve --root "$PROJECT" --input-dir "$ARTIFACT_ROOT/inputs" --output "$ARTIFACT_ROOT/run" --pilot --workers 3
python -m scripts.research.pallet_minimal_bi_ablation_v1.pilot_memory_probe --root "$PROJECT" --input-dir "$ARTIFACT_ROOT/inputs" --output "$ARTIFACT_ROOT/run" --device cuda
python -m scripts.research.pallet_minimal_bi_ablation_v1.batch observe --root "$PROJECT" --input-dir "$ARTIFACT_ROOT/inputs" --output "$ARTIFACT_ROOT/run" --device cuda
python -m scripts.research.pallet_minimal_bi_ablation_v1.batch solve --root "$PROJECT" --input-dir "$ARTIFACT_ROOT/inputs" --output "$ARTIFACT_ROOT/run" --workers 3
python -m scripts.research.pallet_minimal_bi_ablation_v1.batch coordinate --root "$PROJECT" --input-dir "$ARTIFACT_ROOT/inputs" --output "$ARTIFACT_ROOT/run" --workers 3
python -m scripts.research.pallet_minimal_bi_ablation_v1.timing --root "$PROJECT" --input-dir "$ARTIFACT_ROOT/inputs" --output "$ARTIFACT_ROOT/run" --device cuda
python -m scripts.research.pallet_minimal_bi_ablation_v1.real_cli_check --root "$PROJECT" --input-dir "$ARTIFACT_ROOT/inputs" --folder "$ARTIFACT_ROOT/run"
```

GPU 프로세스는 동시1개다. CPU pose worker≤3, BLAS/OpenMP1을 사용한다. CAL 파일럿은 고정6개 ID를 해시/세션 순환으로 선택한다. 파일럿 메모리 probe는 같은 이미지로 수치/마스크 동등성까지 확인하는 별도 계측이다. 실제 호출계수에는 상태 ResNet/Tiny/SAM의 forward pre-hook와 함수 wrapper를 썼다. 계측이 추가 forward를 만들지 않는다.

전체 `PREDICTION_SEAL.json`이238행/952경로·순서·입력/소스/모델 SHA를 통과한 후에만 `scoring` CLI에 `--root-scoring-GO`와 실제 전체 seal SHA를 준다. 결과 폴더에952행 CSV, 모든 전환, bootstrap/통계, SCORE_SEAL을 남긴다. `independent_check`는 새 scorer/statistics를 import하지 않고 이 저장 점수를 별도 수식으로 재집계한다. 참조 자체를 새로 측정하는 독립 검증은 아니다.

```bash
python -m scripts.research.pallet_minimal_bi_ablation_v1.scoring --folder "$ARTIFACT_ROOT/run" --input-dir "$ARTIFACT_ROOT/inputs" --source-root "$SOURCE" --reference-manifest "$ARTIFACT_ROOT/SCORING_REFERENCE_MANIFEST.json" --reference-manifest-sha256 "$REFERENCE_MANIFEST_SHA256" --prediction-seal-sha256 "$PREDICTION_SEAL_SHA256" --output "$ARTIFACT_ROOT/score" --root-scoring-GO
python -m scripts.research.pallet_minimal_bi_ablation_v1.independent_check --score-folder "$ARTIFACT_ROOT/score" --input-seal "$ARTIFACT_ROOT/inputs/INPUT_SEAL.json" --score-seal-sha256 "$SCORE_SEAL_SHA256" --output "$ARTIFACT_ROOT/independent" --root-recount-GO
python -m scripts.research.pallet_minimal_bi_ablation_v1.coordinate_analysis --folder "$ARTIFACT_ROOT/run" --input-dir "$ARTIFACT_ROOT/inputs" --score-dir "$ARTIFACT_ROOT/score" --source-root "$SOURCE" --reference-manifest "$ARTIFACT_ROOT/SCORING_REFERENCE_MANIFEST.json" --output "$ARTIFACT_ROOT/coordinate_analysis" --prediction-seal-sha256 "$PREDICTION_SEAL_SHA256" --score-seal-sha256 "$SCORE_SEAL_SHA256"
python -m scripts.research.pallet_minimal_bi_ablation_v1.postseal_diagnostics --folder "$ARTIFACT_ROOT/run" --input-dir "$ARTIFACT_ROOT/inputs" --score-dir "$ARTIFACT_ROOT/score" --source-root "$SOURCE" --output "$ARTIFACT_ROOT/diagnostics" --prediction-seal-sha256 "$PREDICTION_SEAL_SHA256" --score-seal-sha256 "$SCORE_SEAL_SHA256" --root-postscore-diagnostics-GO
python -m scripts.research.pallet_minimal_bi_ablation_v1.gallery --folder "$ARTIFACT_ROOT/run" --input-dir "$ARTIFACT_ROOT/inputs" --score-dir "$ARTIFACT_ROOT/score" --source-root "$SOURCE" --output "$ARTIFACT_ROOT/gallery" --prediction-seal-sha256 "$PREDICTION_SEAL_SHA256" --score-seal-sha256 "$SCORE_SEAL_SHA256" --root-gallery-GO
```

재개는 같은 source/model/input/config 바인딩과 자기 observation binding을 통과한 완료행만 재사용한다. 원본 시도/기술적 오류를 보존하고 원자료·사용자6개 dirty 파일·논문·기존 실험은 변경하지 않는다. 소스가 바뀌면 새 시도로 나누며 출력 구성을 같다고 추정해서 이어 붙이지 않는다. 전체 seal이 이미 있으면 새 출력 폴더로 완전 재현하거나 저장된 seal을 검산한다.

단일 영상은 `cli run`에 predictor-only JSON(`K`, native8/9 전체영상 좌표, 자기 F의 R_cf/R_physical/centroid/cf_extents, 영상)을 전달한다. 모델 경로는 명시 인자다. `cli observe`와 `cli solve`로 관측/자세를 분리할 수도 있다. N3 캐시/가중치·정답 경로는 이 인터페이스에 없다. 반환된 B/I 측정 좌표와 `model_projection_xy`를 혼동하지 않는다.

실제 환경, 시도/테스트 실패와 수리, 실행량·총 경과, 독립 검사 범위/공백은 COST_AND_VALIDATION_KO.md와 EXECUTION_RECEIPT.json에 최종 기록한다. 전체 시간/FPS, 새 학습, 새 데이터, 독립 최종 시험, 운영 승격, 논문 변경은 수행하지 않았다.
