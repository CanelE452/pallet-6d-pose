# 고정 후보 선택·비용 진단 재현

이번 명령은 저장된 관측과 자세를 읽어 원 선택·비용·잔차·제약·참조 지표를 재집계한다. **새 학습·모델 forward·PnP·수치 최적화·후보/관측 생성은 실행하지 않는다.** N3·SubPix·기존 e·운영 기본값·논문·과거 결과를 수정하지 않는다. 원결과는 `407de865445e4987614f50f893d2ccea78d507b6`, 소스 기준 main은 `2fe1969b7a999d9a8d460703be4bfd11317e7f07`다. 실제 연결은 [SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)과 [실행 계약](PLAN_AND_CONTRACT.md)을 따른다.

PRIMARY192=TEST49+REMAINDER143, CAL46 별도다. 네 PRIMARY 집단114/10/22/46은 원 오라클 필드에서 매핑한다. 성공 후보는 **동일 R/t 하나**의 중심 위치≤5cm AND physical/C2 회전≤5°를 동시에 만족한다. 기존 참조는 기하 재구성값이며 독립 physical/semantic owner 정답이 아니다. 기존49·추가143은 반복 개발에 노출된 집합이다.

## 입력 복원과 환경

코드·작은 공개 집계만으로 대형 원자료가 자동 생성되지는 않는다. 원407 실행의 typed packet·관측 계보·G10/G13 원 은행·봉인된 replay 끝점·오라클·score_v2 참조를 그대로 재사용한다. [원 실행 재현·자산 안내](../pallet_statewise_correspondence_diagnosis_v1/REPRODUCE.md), [원 후보 범위](../pallet_statewise_correspondence_diagnosis_v1/ORACLE_LIMITS.md), [Release 복원 안내](../pallet_visibility_curriculum_20261013_v1/publication_followup_20261015_v11/repro_inventory_DOWNLOADS_KO_v1.md)에서 필요한 멤버·경로·SHA만 확인해 복원한다. **이번에는 그 이전 안내의 runner·typed_replay·restore_starts·새 PnP 명령을 실행하지 않는다.** 기존 보충 raw-start 파일도 봉인된 바이트만 읽는다. archive 전체를 재다운로드하거나 다른 cache로 대체하지 않는다.

대형 파일은 `SOURCE_BINDINGS.json.private_parent_artifacts`가 지정한 SHA·크기와 대조한다. 부모 evidence와 그 형제 `minimal-bi-ablation-20261016-v1/SCORING_REFERENCE_MANIFEST_V2.json` 배치는 bindings 코드의 실제 경로 계약이다. 새 ROOT와 PUBLIC은 기존 결과와 다른 빈 경로를 사용한다. 아래 `/path/to/...`는 실행자의 별칭이며 게시한 개인 절대경로가 아니다.

검증 환경은 Python3.10, NumPy1.26.4, SciPy1.12.0, OpenCV4.9.0, Pillow·Matplotlib·pytest가 있는 기존 환경이다. 새 모델이나 패키지를 설치하지 않았다. CPU 분석 단계 elapsed 상한7200초, 최대 CPU worker3개, GPU사용0이며 batch 명령은 worker2개를 사용한다. thread 환경은1로 고정한다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TASK_PYTHON=/path/to/existing/python
export TASK_SOURCE=/path/to/isolated-research-worktree
export TASK_PARENT=/path/to/evidence/pallet-statewise-correspondence-diagnosis-20261016-v1
export TASK_ROOT=/path/to/evidence/new-empty-selection-score-diagnosis
export TASK_PUBLIC="$TASK_SOURCE/_docs/experiments/pallet_selection_score_diagnosis_reproduced_v1"
export TASK_DATA=/path/to/restored-snapshot/data
export TASK_LEGACY=/path/to/restored-snapshot/_docs/experiments/pallet_visibility_curriculum_20261013_v1
cd "$TASK_SOURCE"
```

원 worktree의 사용자 dirty 파일을 보존하고 별도 checkout에서 실행한다. PUBLIC은 반드시 SOURCE 루트 아래 `_docs/experiments/새폴더이름`의3단계 상대경로이며 빈 새 디렉터리여야 한다. 기존 문서의 상대 source 링크와 최종 게시 seal의 `relative_to(source_root)` 계약 때문에 저장소 바깥 임의 public 경로는 사용할 수 없다. 저장소의 상대 `data`와 legacy 경로는 기존 readonly 자산에 연결할 수 있다. 아래 설정은 기존 경로가 다른 곳을 가리키면 중단하며 덮어쓰지 않는다. symlink 자체가 OS 쓰기 차단은 아니므로 원 경로에는 쓰지 않는다. JSON 내부 경로나 이미지·참조를 재작성하지 않는다.

```bash
"$TASK_PYTHON" -B - <<'PY'
import hashlib, json, os, time
from pathlib import Path
source=Path(os.environ['TASK_SOURCE'])
for name,key in [('data','TASK_DATA'),
                 ('_docs/experiments/pallet_visibility_curriculum_20261013_v1','TASK_LEGACY')]:
    target=Path(os.environ[key])
    if not target.is_dir(): raise FileNotFoundError(str(target))
    link=source/name
    if link.exists() or link.is_symlink():
        if not link.exists() or link.resolve()!=target.resolve():
            raise FileExistsError('기존 경로 보존: '+str(link))
    else:
        link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(target.resolve(),target_is_directory=True)
public=Path(os.environ['TASK_PUBLIC']).resolve()
relative=public.relative_to(source.resolve())
if len(relative.parts)!=3 or relative.parts[:2]!=('_docs','experiments'):
    raise ValueError('PUBLIC은 SOURCE/_docs/experiments/새폴더의 정확한 3단계 경로여야 한다')
out=Path(os.environ['TASK_ROOT']); out.mkdir(parents=True,exist_ok=False)
public.mkdir(parents=True,exist_ok=False)
protected_paths=['.gitignore','readme.md',
    '_docs/experiments/pallet_resnet18_refiner_20261001_v1/README_KO.md',
    '_docs/experiments/pallet_three_estimator_20261001_v1/README_KO.md',
    '_docs/experiments/pallet_three_estimator_20261001_v1/STATUS_KO.md',
    'data/evaluation/pallet_eval_v1/reports/ANNOTATION_PROGRESS.md']
protected={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
           for name in protected_paths if (source/name).is_file()}
missing=[name for name in protected_paths if not (source/name).is_file()]
with (out/'START.json').open('x') as f:
    json.dump(dict(start_epoch=time.time(),cpu_analysis_elapsed_cap_seconds=7200,
                   max_workers=3,GPU_use=0,new_pose_solves=0,new_model_forwards=0,
                   protected_files_SHA256=protected,
                   protected_paths_absent_from_this_sparse_source=missing),f,indent=2)
PY
"$TASK_PYTHON" -B -m pytest -q scripts/research/pallet_selection_score_diagnosis_v1 -p no:cacheprovider
```

`--help`를 bindings/selection/fixed_cost/cost_batch/check_dense/metrics/pair_analysis/baseline/no_candidate/influence/gallery/gallery_verify와 independent_influence_check/witness/report/validate에서 실제 확인했다. 아래 출력 디렉터리 이름은 후속 코드가 읽는 실제 이름에 맞췄다. 완료 폴더에 재실행하지 않는다. 구현·포장 실패 시 해당 시도와 예외를 보존하고 새 전체 ROOT 또는 새 출력 버전을 사용하며, 후속 연결도 명시적으로 갱신한다. 보호 파일6개는 이 SOURCE에 실제 존재하는 경우만 START에 hash를 기록하고, sparse SOURCE에 없는 경로는 별도로 남긴다. 최종 검사도 같은 SOURCE를 대상으로 하며 다른 사용자 작업공간의 dirty 바이트를 새 기준으로 가장하지 않는다.

## 원407 연결과 원 선택 재현

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.bindings \
  --source-root "$TASK_SOURCE" --parent "$TASK_PARENT" --output "$TASK_ROOT"
"$TASK_PYTHON" -B - <<'PY'
import json, os, time
from pathlib import Path
source=Path(os.environ['TASK_SOURCE'])
value=json.loads((source/'_docs/experiments/pallet_selection_score_diagnosis_v1/INFLUENCE_PROTOCOL.json').read_text())
value['recorded_epoch']=time.time()
value['reference_performance_table_not_yet_computed']=True
with (Path(os.environ['TASK_ROOT'])/'INFLUENCE_PROTOCOL.json').open('x') as f:
    json.dump(value,f,indent=2)
PY
"$TASK_PYTHON" -B - <<'PY'
import os
from pathlib import Path
from scripts.research.pallet_selection_score_diagnosis_v1.selection import run
from scripts.research.pallet_selection_score_diagnosis_v1.runtime_guard import FixedPoseOnly
from scripts.research.pallet_selection_score_diagnosis_v1.common import write, timed_stage
root=Path(os.environ['TASK_ROOT'])
with timed_stage(root,'records_only_selection_replay',new_pose_calls=0,new_model_forwards=0):
    with FixedPoseOnly() as guard:
        result=run(Path(os.environ['TASK_PARENT']),root/'selection_v1')
write(root/'SELECTION_RUNTIME_GUARD.json',guard.receipt())
assert result['all_passed'] and result['passed_rows']==238
PY
```

선택 단계는 저장된 입력 순서·feasible/dedup·원 backend comparator·tie-break·G10 family·G13 같은-family 출력/fallback을 복원한다. optimizer를 부르지 않는다. `SELECTION_SEAL.json`의238장 PASS와11603 retained 레코드, 114/10/22/46 및 CAL37/3/0/6을 확인한다. G10과 G13 목적함수는 구분한다. 원 tolerance는 `atol=1e-9, rtol=0`이며 source·pose·수락·index 불일치를 숨기거나 허용치를 완화하지 않는다.

## 원 CAL 처음6장의 cost 파일럿과 전체 고정 비용

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.fixed_cost \
  --pilot-old-root "$TASK_PARENT" --selection-directory "$TASK_ROOT/selection_v1" \
  --output "$TASK_ROOT/fixed_cost_pilot_v2"
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.cost_batch \
  --old-root "$TASK_PARENT" --selection-directory "$TASK_ROOT/selection_v1" \
  --pilot-receipt "$TASK_ROOT/fixed_cost_pilot_v2/PILOT_COST_RECEIPT.json" \
  --output "$TASK_ROOT/cost_v1" --workers 2
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.check_dense \
  --folder "$TASK_ROOT/cost_v1" --selection-directory "$TASK_ROOT/selection_v1"
```

파일럿은 원 CAL 순서의 처음6장·원300개 retained 후보이며 참조 오차로 선정하지 않는다. 원 analytic-gradient cost와 scalar-only cost를 **같은 고정 자세**에서 대조한다. 실제 실행의 원 저장 total 최대차이는 `5.684e-13`, scalar-only 최대차이는 `2.274e-13`이었다. 새 실행은 그 실행 자체의 source SHA·패킷·후보·parity·처리량·메모리로 다시 gate를 확인한다. 파일럿 영수증을 다른 소스에 복사하지 않는다. 예상 elapsed가 상한을 넘으면 동일 계산의 streaming/정확 캐시를 먼저 사용하며 표본을 유리하게 줄이지 않는다.

batch는 원11603개의 R/t·family·관측·가중치를 보존한 원 cost 함수 평가만 한다. full B/I·partial normal·outline·실제 prior의 가산 합과 관측별 합을 검산한다. partial의 full2D 철회, S/O/T/U의 정밀 점 항0, tangent·영역의 hard constraints 역할을 보존한다. `FixedPoseOnly` profiler는 이 분석 프로세스/worker에서 금지 함수 진입을 차단하며 다른 실행의 solver를 monkeypatch하지 않는다.

`COST_SEAL.json`의238/11603 COMPLETE와 원 objective/component parity 및 `DENSE_ARRAY_VALIDATION.json`을 확인한다. dense 검산은 cost 코드 import 없이 저장된 양방향 sample 배열·normalizer·가중 공식·합·SHA를 확인한다. 극단적인 feasible 후보와 모든 실패도 삭제하지 않는다.

## 고정 후보 지표·후보쌍·다른 방법 교차표·후보부재

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.metrics \
  --parent "$TASK_PARENT" --selection "$TASK_ROOT/selection_v1" --output "$TASK_ROOT/metrics_v2"
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.pair_analysis \
  --folder "$TASK_ROOT" --old-root "$TASK_PARENT" --output "$TASK_ROOT/pairs_v2"
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.baseline \
  --old-private "$TASK_PARENT" --output "$TASK_ROOT/baseline_v1"
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.no_candidate \
  --old-private "$TASK_PARENT" --source-root "$TASK_SOURCE" --output "$TASK_ROOT/no_candidate_v2"
```

metrics는 기존 hash-bound `score_v2/FIXED_REFERENCES.json`에서 같은 저장 pose를 평가한다. 원11603개 success ID와의 일치를 검사하고, 독립 Decimal 중심거리 및 SciPy rotation-vector/C2 계산을 함께 확인한다. **원 selector/cost의1e-9 허용치는 그대로다.** 극단적으로 먼48개 retained 후보에서는 독립 norm 연산 순서의1ULP 차이 때문에 고정1e-9cm 절대차 검사에 실패했다(최대 `1.52587890625e-5cm`). 실패한 초기 `metrics_v1`은 보존했다. `metrics_v2`는 metric 수식·값·pose·success를 바꾸지 않고 IEEE 산술의 도출된 오차 bound로 독립 거리 계산을 검산하며, 해당48행과 초기 검사 실패를 `METRIC_CHECK.json`에 모두 남긴다. 이를 새 자세 성능이나 임의 tolerance tuning으로 보고하지 않는다.

pair 분석은 같은-family22장의 s와 원 comparator상 첫 참조 성공 g, 성공114장의 s와 첫 참조 실패 b, family10장의 **G10 실제 winner 결정 표와 G13 도달 가능성 표**를 구분한다. 성공 후보 전체의 순위·범위를 보존하고 전체192/49/143/CAL46에서 같은 분석을 한다. `Delta = other − selected`이며 항별·관측별 Delta 합을 검사한다. G10(s)에서 G13(g)를 빼지 않는다.

baseline은 기존 sealed N0/PoseFix RAW/E0 row metrics만 읽는다. PoseFix+SubPix의 전체192 규칙은 만들지 않는다. no_candidate는 원 오라클·start→end·진입 로그만 읽어46장의 admission4/기존 후보42를 분리한다. T 최소와 R 최소를 합친 가상 성공을 만들지 않는다. semantic owner와 observation-vs-proposal 공백은 미검증으로 보존한다.

## gate 통과 후 사전 고정6개 순위 영향 진단

위 원407 연결 단계에서 성과표를 열기 전에 이 게시본의 [INFLUENCE_PROTOCOL.json](INFLUENCE_PROTOCOL.json)을 새 ROOT에 기록했다. source/선택/가산 cost/관측별 합/eligible pool gate를 모두 통과한 경우에만 아래를 실행한다. 실제 활성 항 full_BI·partial_normal·outline을 각각 한 번만 제외하며, G10_ONLY와 G13_INTERNAL_ONLY를 분리한 정확히6개다. prior는 실제 weight0이어서 진단 항목을 만들지 않았다. 점별·출처별 selector나 두 항 조합·hard guard 삭제·가중치 탐색은 없다.

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.influence \
  --parent "$TASK_PARENT" --root "$TASK_ROOT" --output "$TASK_ROOT/influence_attempt01"
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.independent_influence_check \
  --root "$TASK_ROOT" --parent "$TASK_PARENT" --influence "$TASK_ROOT/influence_attempt01" \
  --output "$TASK_ROOT/INDEPENDENT_INFLUENCE_CHECK.json"
```

gate가 실패하면 영향 진단만 미실행/이유를 남기고 다른 독립 분석을 계속한다. 독립 영향 검산은 원 selector·influence·cost·metric scorer를 import하지 않고 원 tuple comparator와 저장 scalar에서 이산 선택 및 Fraction 기반 연속 합을 재집계한다. 원 winner·기본 e·6개 진단의 선택·전체192/49/143/CAL46 손익을 확인한다. 기존 게시 결과를 읽은 자료는 새 독립 test가 아니다.

> 원 목적함수로 생성한 고정 후보 은행 위의 사후 순위 영향 진단이다. 항을 제거한 목적함수로 후보를 새로 생성·최적화한 방법의 성능이 아니며, 새 선택기로 채택하지 않았다.

## 실제 RGB 갤러리·독립 QA·보고 패키지

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.gallery \
  --private "$TASK_ROOT" --old-private "$TASK_PARENT" --source-root "$TASK_SOURCE" \
  --output "$TASK_PUBLIC/gallery" \
  --costs-path "$TASK_ROOT/cost_v1/CANDIDATE_COSTS.jsonl.gz" \
  --candidate-metrics-path "$TASK_ROOT/metrics_v2/CANDIDATE_METRICS.jsonl.gz" \
  --selection-folder "$TASK_ROOT/selection_v1"
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.gallery_verify \
  --gallery "$TASK_PUBLIC/gallery" --private "$TASK_ROOT" --old-private "$TASK_PARENT" \
  --source-root "$TASK_SOURCE"
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.witness \
  --root "$TASK_ROOT" --public "$TASK_PUBLIC"
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.report \
  --root "$TASK_ROOT" --parent "$TASK_PARENT" --public "$TASK_PUBLIC" --source-root "$TASK_SOURCE"
"$TASK_PYTHON" -B -m scripts.research.pallet_selection_score_diagnosis_v1.validate \
  --root "$TASK_ROOT" --public "$TASK_PUBLIC" --source-root "$TASK_SOURCE" \
  --original-workspace "$TASK_SOURCE"
```

gallery는 선택32장 전량과 성공 반례2장을 같은 RGB·초기예측 기반 crop·같은 축척으로 보여준다. 본문 대표6개는 사전 규칙의 기존 E0 위치오차 내림차순·녹화 중복 회피로 선택한다. full 점·상속 예측 fallback·partial 선/유한 지지·O/T/U 영역·저장 mask convex outline·모델 투영을 구별하며 S를 관측점으로 만들지 않는다. 후보부재46+CAL6장은 새 pose를 만들어 그리지 않고 기존 RGB crop만 연결한다.

원사진은 기존 공개 Release inventory의 **동일 SHA**가 있는 경우만 사용한다. inventory member가 공개됐다는 사실과 그 data 경로가 Git blob이라는 주장은 다르다. 원RGB 전체를 Git에 추가하지 않고 허용된 작은 crop과 원 member/SHA·복원 안내를 연결한다. 공개 범위 밖 이미지가 필요하면 해당 부분만 미게시/이유를 남긴다.

QA는88패널의 원 R/t·XYZ·cost·metric 값, 독립 homogeneous 투영,86개 RGB crop의 정확한 JPEG 재생성,120JPEG decode·HTML 링크를 확인한다. 실제 화면을 열어 위치·범례·G10/G13 표 분리를 추가 확인한다. witness는 이미 저장된 dense outline sample의 양·음 Delta를 그림으로 보여주며 새로운 비용·선·mask·후보를 평가하거나 crop을 바꾸지 않는다. 보고서 작성 후 루트 문서 링크도 확인한다.

`report`는 실제 구현된 보고·작은 결과 패키징 진입점이며 `--help`를 확인했다. `validate`는 보고 후 평면 장부의 후보·쌍·관측 합, 전체 분모·원 hash·보호 파일·이미지·링크·개인경로 부재를 검사하고, CLI 마지막에 공개 파일과 새 분석 소스의 SHA/크기를 `PUBLICATION_CHECK.json`에 봉인한다. 이는 원격 push 확인을 대신하지 않으며 원격 SHA는 게시 후 별도로 읽어 확인한다. 원 실행에만 있었던 실패/중간 시도 영수증은 새 실행에 가짜로 복사하지 않는다. 선택적 역사 영수증은 없음을 보존하고, COST는 새 실행의 실제 단계 영수증·ledger에서 집계한다. fresh ROOT의 SOURCE_BINDINGS를 사용하며 원 실행의 `final_bindings` 보충 폴더를 임의로 만들 필요는 없다.

## 보존·게시·재검산

전체49MB compact cost와 약783MB dense frame journal, 원 후보·관측·원RGB는 private evidence에 보존하며 Git에 복제하지 않는다. 공개하는 것은 새 분석 코드·작은 표·필요한 실제 관측 비용 근거·봉인/출처·허용 이미지와 재현 안내다. 참조 숫자 파일 전체와 기존 raw archive도 무차별 게시하지 않는다.

이전 실행의 실패·수정 시도(`fixed_cost_pilot_v1`, `metrics_v1`, `selection_attempt01`, no_candidate 링크/포장 수정 등)는 원본대로 남겼다. 재현 시에도 실패를 삭제하거나 다른 source/cache/ID로 바꾸어 성공 수를 맞추지 않는다. 실제 새 elapsed·worker CPU·갤러리·검산을 각 영수증과 COST에 기록하고 과거 시간을 새 실행 시간으로 복사하지 않는다. CPU 상한을 실제 넘길 전망이면 필요한 남은 계산과 자원을 보고한다.

마지막으로 full 분모·모든 결측/실패·회복−손상·연속 오차 합, 입력/관측/후보/참조·기존 소스/운영 파일 hash 불변, runtime guard의 금지 함수 body 실행0, 원격 게시 SHA를 확인한다. 다음 수정 제안은 이 사후 진단의 근거이며 효과가 검증된 새 방법으로 보고하지 않는다.
