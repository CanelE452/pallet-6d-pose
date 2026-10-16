# 재현 절차와 입력 범위

**실행 기준은 아래 [전체 명령 보완](#전체-명령-보완)이다.** 앞부분의 최초 안내는 보존했다. 특히 assemble 전 supplement, 실제 자산 경로·전체238 ID 생성, analysis/diagnose/oracle/검산/report 순서는 보완 절을 따른다.

Python은 기존 `pallet-yolo26` 환경을 사용한다. 새 모델 설치·fit·가중치 다운로드는 필요하지 않다. 이번 실행은 이미 저장된 관측을 사용하므로 새 YOLO/N3/SubPix/분할기 forward는 0회다. 아래 경로는 실행자가 가진 원자료 위치에 맞춰 지정한다. 기존 폴더를 출력 대상으로 재사용하면 실패하도록 만들었다. 실패 시도는 보존하고 새 출력 버전을 사용한다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_SOURCE=/path/to/this/research-worktree
TASK_PYTHON=/path/to/envs/pallet-yolo26/bin/python
TASK_GUARDED=/path/to/pallet-guarded-bi-final-20261016-v1
TASK_SINGLE=/path/to/pallet-single-component-bi-20261016-v1
TASK_OUTPUT=/path/to/new-empty-statewise-output
```

기존 raw visibility 저널과 data를 같은 상대경로로 읽을 수 있어야 한다. 이번 작업은 별도 sparse worktree의 두 자산 경로를 원본에 읽기 전용으로 연결해 사용했다. 링크를 통해 새 결과를 쓰지 않는다. 공개 코드만 checkout한 경우 기존 release/manifest의 정확한 필요한 파일만 복원한다. 원 archive 전체를 다시 받거나 모델 전체를 복제할 필요는 없다. 모델·관측·참조의 SHA와 전체238 ID는 공개 manifest 및 로컬 EXTRACTION_SEAL에 기록된다.

먼저 순수 검사를 수행한다. real frozen-cache tests는 해당 로컬 자료가 없으면 명시적으로 skip된다.

```bash
"$TASK_PYTHON" -B -m pytest -q scripts/research/pallet_statewise_correspondence_diagnosis_v1 -p no:cacheprovider
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.protocol --source-root "$TASK_SOURCE" --guarded-dir "$TASK_GUARDED" --output "$TASK_OUTPUT"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.lineage --source-root "$TASK_SOURCE" --guarded-dir "$TASK_GUARDED" --output-dir "$TASK_OUTPUT/e_trace_v2"
```

원 candidate 추출에는 `REPLAY_PROTOCOL.json.ordered_ids`를 그대로 JSON list로 저장한 explicit238 ID 파일을 넘긴다. 지난 outlier 부분집합을 사용하지 않는다.

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.candidates --source-root "$TASK_SOURCE" --ids-file "$TASK_OUTPUT/ORDERED_IDS.json" --output "$TASK_OUTPUT/original_candidate_archive_v1"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.runner --source-root "$TASK_SOURCE" --guarded-dir "$TASK_GUARDED" --output "$TASK_OUTPUT" --phase pilot --workers 3
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.parity --output "$TASK_OUTPUT" --candidate-folder "$TASK_OUTPUT/original_candidate_archive_v1" --phase pilot --receipt "$TASK_OUTPUT/TYPED_PILOT_PARITY.json"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.continuity --folder "$TASK_OUTPUT"
```

현재 E 후단의 원 cache와 REOBS_MULTI의 pose/Q/state/measurement/accepted/fallback/source를 기존 1e-9, rtol0으로 비교한다. TYPED 비교기는 원 G10/G13 양 family의 모든 저장 후보·sample·점수·rank·ambiguity와 원 e 최종 pose를 검산한다. numeric bridge와 solver replay를 구분한다. parity 실패 시 구현을 먼저 진단하고 결과에 맞춰 tolerance를 넓히지 않는다.

6장 parity와 예산 추정 검토 후 `PILOT_AUTHORIZATION.json`을 작성한다. 이번 실제 receipt를 복사해 다른 입력/소스를 승인하지 않는다. 네 경로/관측/손실/후보예산/전체 ID를 유지하고 CPU stage7200초·GPU process3600초 상한을 확인한다. 승인된 동일 계산의 pilot slots는 재사용한다.

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.runner --source-root "$TASK_SOURCE" --guarded-dir "$TASK_GUARDED" --output "$TASK_OUTPUT" --phase full --workers 3
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.parity --output "$TASK_OUTPUT" --candidate-folder "$TASK_OUTPUT/original_candidate_archive_v1" --phase full --receipt "$TASK_OUTPUT/REPLAY_PARITY.json"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.runner --source-root "$TASK_SOURCE" --guarded-dir "$TASK_GUARDED" --output "$TASK_OUTPUT" --phase assemble
```

prediction 및 candidate seal이 완성된 뒤 **별도 분석 프로세스**에서만 참조를 연다. analysis CLI는 `--root-scoring-go`, 두 seal의 실제 SHA, 고정 reference manifest의 SHA를 요구한다. `--candidate-file`은 봉인된 GENERATED_CANDIDATES.jsonl.gz다. reference manifest는 기존 SCORING_REFERENCE_MANIFEST_V2의 확정 바이트를 사용한다. 임의 registration/annotation을 새로 맞추지 않는다. 정확한 실제 명령과 SHA는 각 실행 receipt 및 최종 보고서에 연결한다.

독립 검산은 새 `independent_check`가 saved physical R/t와 같은 hash-bound reference에서 중심 거리와 SciPy rotation-vector C2 각도를 직접 다시 계산한다. 원 trace/acos metric helper를 import하지 않는다. 성공 분모, 평균, 표본분산과 모든 행의 수치를 비교하며, 과거 실패와 새 unavailable을 제거하지 않는다. 후보 oracle, CAL 잔차 진단과 실제 여섯 사례 갤러리도 이 봉인 이후 생성한다.

로컬의 대형 원저널·mask·typed packet·전체 candidate R/t는 manifest로 바인딩한다. 공개 패키지는 새 코드, 작은 집계·행 지표·감사표·실제 사례 이미지와 seal을 포함한다. 기존 논문·운영 기본값·과거 기록은 변경하지 않는다.

## 전체 명령 보완

앞의 초기 안내를 덮어쓰지 않고 실제 API·실행 순서를 보완한다. 각 명령은 새 출력에서만 실행한다. 이 문서·기존 영수증은 새 입력을 자동 승인하지 않는다. 새 학습·모델 forward·seed·관측·손실·후보 예산 선택은 없다.

다음 변수를 추가하고 코드 checkout에서 실행한다. `TASK_DATA`와 `TASK_LEGACY`는 읽기 전용 원자료이며 출력 대상으로 쓰지 않는다. 공개 clone의 누락 파일은 [입력/source manifest](evidence/SOURCE_AND_INPUT_MANIFEST.json), [추출 봉인](evidence/EXTRACTION_SEAL.json), [원 후보 manifest](evidence/CANDIDATE_MANIFEST.json), [점수 봉인](metrics/SCORE_SEAL.json)에 명시된 필수 경로·바이트·SHA만 선택 복원한다. archive 전체·모델 전체를 다시 받지 않는다. sparse checkout에서 빠진 추적 소스도 같은 커밋의 필요한 폴더만 materialize한다.

```bash
export TASK_SOURCE TASK_PYTHON TASK_GUARDED TASK_SINGLE TASK_OUTPUT
export TASK_DATA=/path/to/original-pallet-pose/data
export TASK_LEGACY=/path/to/original-pallet-pose/_docs/experiments/pallet_visibility_curriculum_20261013_v1
export TASK_REFERENCE=/path/to/existing/SCORING_REFERENCE_MANIFEST_V2.json
export TASK_SCORE="$TASK_OUTPUT/score_v2"
export TASK_PUBLIC=/path/to/new-empty-public-results
cd "$TASK_SOURCE"
"$TASK_PYTHON" -B - <<'PY'
import os
from pathlib import Path
source = Path(os.environ['TASK_SOURCE'])
for name, target in [('data', Path(os.environ['TASK_DATA'])),
                     ('_docs/experiments/pallet_visibility_curriculum_20261013_v1',
                      Path(os.environ['TASK_LEGACY']))]:
    if not target.is_dir():
        raise FileNotFoundError('필수 원자료가 없음: '+str(target))
    link = source/name
    if link.exists() or link.is_symlink():
        if not link.exists() or link.resolve() != target.resolve():
            raise FileExistsError('기존 경로 보존; 계약 확인 필요: '+str(link))
    else:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target.resolve(), target_is_directory=True)
for key in ('TASK_PYTHON', 'TASK_REFERENCE'):
    if not Path(os.environ[key]).is_file():
        raise FileNotFoundError(os.environ[key])
PY
```

링크 자체는 OS 쓰기 차단이 아니다. 원자료에 쓰는 명령을 실행하지 않는다. guarded 입력에 저장된 과거 절대 RGB 경로도 그대로 접근 가능해야 하며, 위 링크는 JSON 내부 경로를 재작성하지 않는다. 필요한 파일·경로·봉인이 없거나 다르면 중단하고 보고한다. 임의 cache·참조·registration으로 대체하지 않는다. 이번 원 worktree의 `data`와 legacy 문서 symlink는 게시 대상이 아니다.

새 실행 시작 시 `TASK_OUTPUT`을 새로 만들고 시각을 기록한다. 과거 목표의 시작시각을 복사하지 않는다. 환경·소스 HEAD/dirty dependency와 읽기·구현·검산 시간도 별도로 기록한다.

```bash
"$TASK_PYTHON" -B - <<'PY'
import json, os, time
from pathlib import Path
from datetime import datetime, timezone
out = Path(os.environ['TASK_OUTPUT'])
out.mkdir(parents=True, exist_ok=False)
(out/'START.json').write_text(json.dumps(dict(goal_created_epoch=time.time(),
    captured_UTC=datetime.now(timezone.utc).isoformat(),
    timing_scope='this fresh reproduction, not historical timing',
    cpu_solver_analysis_max_elapsed_seconds=7200,
    gpu_max_process_elapsed_seconds=3600, maximum_top_level_pose_slots=952,
    max_core_workers=3, max_gpu_processes=1, new_training_updates=0, new_seeds=0), indent=2)+'\n')
PY
"$TASK_PYTHON" -B -m pytest -q scripts/research/pallet_statewise_correspondence_diagnosis_v1 -p no:cacheprovider
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.protocol --source-root "$TASK_SOURCE" --guarded-dir "$TASK_GUARDED" --output "$TASK_OUTPUT"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.lineage --source-root "$TASK_SOURCE" --guarded-dir "$TASK_GUARDED" --output-dir "$TASK_OUTPUT/e_trace_v2"
"$TASK_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ['TASK_OUTPUT'])
ids = json.loads((out/'REPLAY_PROTOCOL.json').read_text())['ordered_ids']
assert len(ids) == len(set(ids)) == 238
with (out/'ORDERED_IDS.json').open('x') as stream:
    json.dump(ids, stream, indent=2)
    stream.write('\n')
PY
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.candidates --source-root "$TASK_SOURCE" --ids-file "$TASK_OUTPUT/ORDERED_IDS.json" --output "$TASK_OUTPUT/original_candidate_archive_v1"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.runner --source-root "$TASK_SOURCE" --guarded-dir "$TASK_GUARDED" --output "$TASK_OUTPUT" --phase pilot --workers 3
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.parity --output "$TASK_OUTPUT" --candidate-folder "$TASK_OUTPUT/original_candidate_archive_v1" --phase pilot --receipt "$TASK_OUTPUT/TYPED_PILOT_PARITY_V1.json"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.continuity --folder "$TASK_OUTPUT"
```

`parity`는 TYPED의 원 G10/G13 replay를 검사한다. REOBS_MULTI는 다음처럼 저장된 자기 후단과 별도 비교한다. 시간·설명 문자열을 제외한 실제 pose·상태·측정·수락·fallback·solver 설정을 기존1e-9/rtol0으로 비교한다.

```bash
"$TASK_PYTHON" -B - <<'PY'
import os
from pathlib import Path
from scripts.research.pallet_statewise_correspondence_diagnosis_v1.common import read, write
from scripts.research.pallet_statewise_correspondence_diagnosis_v1.parity import compare
out, guarded = Path(os.environ['TASK_OUTPUT']), Path(os.environ['TASK_GUARDED'])
p = read(out/'REPLAY_PROTOCOL.json')
index = {ident: i for i, ident in enumerate(p['ordered_ids'])}
checks = []
for ident in p['pilot_ids']:
    i = index[ident]
    old = read(guarded/f'e_run/poses/{i:03d}_E_FROZEN.json')['result']
    new = read(out/f'slots/{i:03d}_REOBS_MULTI.json')['result']
    keys = ('actual_pose', 'states', 'frozen_visible_measurements', 'input_native_ids',
            'accepted', 'fallback', 'solver_configuration')
    issues = compare({k: old[k] for k in keys}, {k: new[k] for k in keys})
    checks.append(dict(id=ident, passed=not issues, issues=issues))
write(out/'REOBS_PILOT_PARITY.json', dict(rows=checks, numeric_atol=1e-9,
      numeric_rtol=0., all_pass=all(v['passed'] for v in checks)))
assert all(v['passed'] for v in checks)
PY
```

실행 담당자가 실제 CAL6×4 경로 동등성·시간 및 남은 CPU7200초/GPU3600초 예산을 검토하고 **새 실행 자체의** `PILOT_AUTHORIZATION.json`에 해당 protocol/source/parity 해시·예측 비용·검토 사유를 기록한 뒤 full을 진행한다. 원 [파일럿 영수증](evidence/PILOT_AUTHORIZATION.json)을 복사해 새 실행을 승인하지 않는다. 불일치·예산 초과는 full 진행 사유가 아니다. 동일 input/source의 pilot24 slots만 재사용한다.

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.runner --source-root "$TASK_SOURCE" --guarded-dir "$TASK_GUARDED" --output "$TASK_OUTPUT" --phase full --workers 3
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.parity --output "$TASK_OUTPUT" --candidate-folder "$TASK_OUTPUT/original_candidate_archive_v1" --phase full --receipt "$TASK_OUTPUT/REPLAY_PARITY.json"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.restore_starts --folder "$TASK_OUTPUT" --output "$TASK_OUTPUT/supplemental_cal_start_logging_v1"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.runner --source-root "$TASK_SOURCE" --guarded-dir "$TASK_GUARDED" --output "$TASK_OUTPUT" --phase assemble
```

`restore_starts`는 고정 CAL1장의 빠진 raw-start 로그에 한해 동일 원 `_initial_poses`와 인자를 사용한다. generator2호출·P3P20회·SQPNP2회는 따로 계수한다. 새 seed/시작점 정책, `minimize`, `least_squares`, 새 pose fit, 전체 replay는0이며 원 optimizer 결과는 그대로다. supplement는 assemble 전·참조 개봉 전에 봉인한다.

완성된 prediction/candidate 봉인 및 참조 계약을 확인한 뒤 **별도 분석 프로세스에서만** 다음을 실행한다. `--root-scoring-go`는 그 새 실행의 실제 gate이며 과거 receipt 복사 승인이 아니다. SHA는 새 바이트에서 계산한다.

```bash
TASK_PREDICTION_SHA=$(sha256sum "$TASK_OUTPUT/PREDICTION_SEAL.json" | cut -d ' ' -f1)
TASK_CANDIDATE_SHA=$(sha256sum "$TASK_OUTPUT/CANDIDATE_SEAL.json" | cut -d ' ' -f1)
TASK_REFERENCE_SHA=$(sha256sum "$TASK_REFERENCE" | cut -d ' ' -f1)
TASK_SEAL_ARGS=(--prediction-seal-sha256 "$TASK_PREDICTION_SHA" --candidate-seal-sha256 "$TASK_CANDIDATE_SHA" --reference-manifest-sha256 "$TASK_REFERENCE_SHA" --root-scoring-go)
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.analysis --folder "$TASK_OUTPUT" --source-root "$TASK_SOURCE" --single-folder "$TASK_SINGLE" --guarded-folder "$TASK_GUARDED" --reference-manifest "$TASK_REFERENCE" --candidate-file "$TASK_OUTPUT/GENERATED_CANDIDATES.jsonl.gz" --output "$TASK_SCORE" "${TASK_SEAL_ARGS[@]}"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.diagnose --folder "$TASK_OUTPUT" --source-root "$TASK_SOURCE" --single-folder "$TASK_SINGLE" --lineage-folder "$TASK_OUTPUT/e_trace_v2" --score-folder "$TASK_SCORE" --reference-manifest "$TASK_REFERENCE" --candidate-file "$TASK_OUTPUT/GENERATED_CANDIDATES.jsonl.gz" --output "$TASK_OUTPUT/diagnostics_v1" "${TASK_SEAL_ARGS[@]}"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.oracle --folder "$TASK_OUTPUT" --source-root "$TASK_SOURCE" --lineage-folder "$TASK_OUTPUT/e_trace_v2" --original-candidate-folder "$TASK_OUTPUT/original_candidate_archive_v1" --reference-manifest "$TASK_REFERENCE" --candidate-file "$TASK_OUTPUT/GENERATED_CANDIDATES.jsonl.gz" --parity-receipt "$TASK_OUTPUT/REPLAY_PARITY.json" --output "$TASK_OUTPUT/oracle_v1" "${TASK_SEAL_ARGS[@]}"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.independent_check --folder "$TASK_OUTPUT" --score-folder "$TASK_SCORE"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.independent_check --folder "$TASK_OUTPUT" --score-folder "$TASK_SCORE" --summary-only
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.selection_cases --folder "$TASK_OUTPUT" --source-root "$TASK_SOURCE" --reference-manifest "$TASK_REFERENCE" --output "$TASK_OUTPUT/SELECTION_CASES.json"
```

analysis는238×19경로와 실패/NA를 보존한다. 독립 검산은 원 metric helper를 재사용하지 않고 저장 physical R/t·별도 중심 norm·SciPy 회전벡터/C2 식과76집계 평균·표본분산을 확인한다. diagnose의 px 잔차는 기존 기하 재구성 참조의 투영이며 독립2D 클릭·의미 owner 정답이 아니다. `oracle`은 **원래 e consumer의 고정 입력·시작점·끝점** 범위다. 과거 WD/P1 rejected proposal·optimizer 내부 iterate·모든 가능한 pose의 완전성을 주장하지 않는다. G10/G13 관측 패키지를 섞지 않는다. `selection_cases`는 봉인 뒤 기존 성공 후보를 설명용으로 추출하는 privileged oracle이며 운영 selector나 새 추론이 아니다.

이번 게시의 `score_v2`는 원 `score_v1`의 E0 초기화·SINGLE ambiguity/격리 범위 메타데이터 보완이다. [bridge](evidence/SCORE_V1_V2_METADATA_BRIDGE.json)은4,522개 metric dict·76개 수치집계·모든 짝대비/CI의 exact 일치와 동일 prediction/candidate SHA를 확인한다. v1·diagnostics·gallery는 보존했다. 새 재현은 현재 analysis로 score_v2를 한 번 생성하면 되며 과거 잘못된 메타데이터나 bridge를 꾸며 만들지 않는다. diagnostic/score 봉인이 다르면 report는 실제 검증된 bridge를 요구한다.

최종 score seal 확인 뒤 실제 RGB 갤러리를 렌더한다. `gallery`는 CLI 대신 아래 API다. 공개 허용 inventory/RGB SHA를 확인하며 S/O/T/U 정밀점, 생성형 이미지, 새 관측/fit은 사용하지 않는다.

```bash
"$TASK_PYTHON" -B - <<'PY'
import os, time
from pathlib import Path
from scripts.research.pallet_statewise_correspondence_diagnosis_v1.gallery import render
from scripts.research.pallet_statewise_correspondence_diagnosis_v1.common import record_stage
out = Path(os.environ['TASK_OUTPUT'])
started, cpu = time.monotonic(), time.process_time()
try:
    render(out, os.environ['TASK_SOURCE'], out/'gallery',
           predictions_path=out/'REPLAY_PREDICTIONS.jsonl.gz',
           metrics_path=Path(os.environ['TASK_SCORE'])/'ROW_METRICS.csv')
finally:
    record_stage(out, 'actual_eight_case_gallery_render', started,
                 worker_cpu_seconds=time.process_time()-cpu, new_pose_calls=0, new_NN_calls=0)
PY
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.cost --folder "$TASK_OUTPUT"
"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.report --folder "$TASK_OUTPUT" --score-folder "$TASK_SCORE" --diagnostic-folder "$TASK_OUTPUT/diagnostics_v1" --oracle-folder "$TASK_OUTPUT/oracle_v1" --independent-receipt "$TASK_OUTPUT/INDEPENDENT_REAGGREGATION.json" --cost-path "$TASK_OUTPUT/COST.json" --output "$TASK_OUTPUT/report_v1" --root-report-go
```

`--root-report-go`는 같은 실행의 분석·검산·공개 범위를 확인한 후 보고서를 생성한다는 gate다. 기존 영수증 복사 승인·push 완료 선언이 아니다. report 출력은 별도 새 폴더이며 기존 공개 REPRODUCE가 있는 폴더를 쓰면 보존 규칙으로 실패한다. 총 wall·stage elapsed·worker CPU를 구분하고 미계측 역사 CPU는 unknown으로 남긴다.

렌더 뒤 실제 crop/hash·basis/Q·전체 HTML 링크·표시 수치를 검산하고 이미지를 직접 연다. [실제 갤러리](gallery/index.html), [저장 출처](gallery/PROCESS.json), [QA](gallery/QA_RECEIPT.json)를 확인한다. 작은 결과 copier의 실제 CLI는 `"$TASK_PYTHON" -B -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.package_results --folder "$TASK_OUTPUT" --public "$TASK_PUBLIC"`다. 이 copier는 **이번 게시의 고정 receipt 목록**을 요구하므로 신규 재현에서 없는 과거 amendment/bridge를 만들어 통과시키지 않는다. 필요한 receipt/복사 목록을 확인하고 누락은 별도 보고한다. full raw·weights·masks·참조는 공개하지 않는다.

최종 gallery22개 JPEG는 `*.jpg` ignore 규칙에 해당하므로 검산 후 각 파일만 `git add -f`로 추가한다. 원자료 symlink·대형 private archive·보존 실패 갤러리는 게시하지 않는다. 새 소스·문서·수치·허용 그림의 최종 링크·manifest·커밋·push 성공은 별도 게시 영수증으로 확인한다. 운영 기본값·원 논문·과거 결과·사용자 수정은 바꾸지 않는다.

## 단일 이미지 CLI: 저장 자세·표·갤러리 일치 확인

아래는 **새 추론이나 pose fit이 아닌** 봉인된 한 행의 조회·대조다. `--id`는 필수 여섯 사례 중 하나를 지정한다. 위 API가 만든 `TASK_OUTPUT/gallery`를 사용하며 실제 게시 폴더를 검사하려면 `--gallery-folder`만 해당 경로로 지정한다. 저장 `R_cf`, physical `R_physical`, 중심 좌표와 기존 cm/°를 출력하고 PROCESS/CSV의 같은 값임을 확인한다. GT/annotation 파일을 열지 않는다.

```bash
"$TASK_PYTHON" -B - --folder "$TASK_OUTPUT" --score-folder "$TASK_SCORE" --gallery-folder "$TASK_OUTPUT/gallery" --id wood_night_01:029428 <<'PY'
import argparse, csv, json
from pathlib import Path
from scripts.research.pallet_statewise_correspondence_diagnosis_v1.common import read, rows, sha
p = argparse.ArgumentParser()
for key in ('folder', 'score-folder', 'gallery-folder', 'id'):
    p.add_argument('--'+key, required=True)
a = p.parse_args()
folder, scored, gallery = Path(a.folder), Path(a.score_folder), Path(a.gallery_folder)
ps, ss, gs = read(folder/'PREDICTION_SEAL.json'), read(scored/'SCORE_SEAL.json'), read(gallery/'GALLERY_SEAL.json')
assert sha(folder/'REPLAY_PREDICTIONS.jsonl.gz') == ps['predictions_sha256']
assert sha(folder/'PREDICTION_SEAL.json') == ss['prediction_seal_sha256']
assert sha(gallery/'PROCESS.json') == gs['files']['PROCESS.json']['sha256']
assert sha(scored/'ROW_METRICS.csv') == ss['artifacts']['ROW_METRICS.csv']
found = [r for r in rows(folder/'REPLAY_PREDICTIONS.jsonl.gz') if r['id'] == a.id]
assert len(found) == 1
case = next(c for c in read(gallery/'PROCESS.json')['cases'] if c['id'] == a.id)
with (scored/'ROW_METRICS.csv').open(newline='') as stream:
    table = {r['arm']: r for r in csv.DictReader(stream) if r['id'] == a.id}
answer = {}
for arm, shown in case['diagnostics'].items():
    pose = found[0]['predictions'][arm]['actual_pose']
    assert pose == shown['actual_pose']
    for field in ('translation_cm', 'rotation_deg'):
        value = float(table[arm][field]) if table[arm][field] else None
        assert value == case['metrics'][arm][field]
    assert case['metrics'][arm]['success_le5'] == {'True': True, 'False': False, '': None}[table[arm]['success_le5']]
    answer[arm] = dict(actual_pose=pose, translation_cm=case['metrics'][arm]['translation_cm'],
                       rotation_deg=case['metrics'][arm]['rotation_deg'],
                       success_le5=case['metrics'][arm]['success_le5'])
print(json.dumps(dict(id=a.id, same_stored_pose_table_and_gallery=True,
    source_RGB_sha256=case['source_RGB_sha256'], results=answer,
    new_inference_or_pose_fit=0, GT_annotation_reads=0), ensure_ascii=False, indent=2))
PY
```
