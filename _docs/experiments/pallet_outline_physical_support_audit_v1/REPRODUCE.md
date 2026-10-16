# 외곽선 물리 지지 감사 재현

이번 실행은 기존 여섯 장의 RGB·마스크·저장 경계·두 고정 후보·참조와 **이미 저장된 비용**을 연결한다. 새 학습·forward·SAM 실행·선 검출·PnP·자세 최적화·후보 생성·비용 함수 재평가·순위 변경·성능 측정은 없다. 기존952개 replay나11,603개 비용 평가도 실행하지 않는다. 새 결과는 진단 장부와 표시 그림이며, 운영 기본값·원 예측·참조·논문은 유지한다.

원결과는 `c56a4e92027a8e56f6704773f24d7c3da78082f1`, 상위 관측·후보 결과는 `407de865445e4987614f50f893d2ccea78d507b6`, 원결과를 main에 병합한 커밋은 `560784a7f5ed7621a7ccc03eee67c19eadd8b759`다. [입력 연결](INPUT_BINDINGS.json), [게시 연결](PACKAGING_BINDINGS.json), [검산](VALIDATION.json)이 실제 파일/SHA의 권위다. Git history에서 원 c56 커밋을 읽을 수 있어야 하며, `bindings`가 원 문서14개의 바이트를 `git show` 결과와 대조한다. 다른 실행의 같은 이름 파일로 대체하지 않는다.

## 입력과 공개 범위

필요한 기존 자료는 [원 score 재현 안내](../pallet_selection_score_diagnosis_v1/REPRODUCE.md), [원 state 재현 안내](../pallet_statewise_correspondence_diagnosis_v1/REPRODUCE.md), [Release 멤버 복원 안내](../pallet_visibility_curriculum_20261013_v1/publication_followup_20261015_v11/repro_inventory_DOWNLOADS_KO_v1.md)에서 경로·SHA를 따라 복원한다. **그 안내에 있는 모델·solver·candidate·cost runner 명령은 이번에 실행하지 않는다.** archive 전체를 다시 받거나 대형 은행을 재생성하지 않는다.

- SCORE private: `cost_v1/COST_SEAL.json`과 manifest가 가리키는 여섯 frame journal의 원 두 후보, 등록 감사가 읽는 `selection_v1/RETAINED_CANDIDATES.jsonl.gz`. 필요한 행만 읽는다.
- STATE private: `e_trace_v2/TYPED_PACKETS.jsonl.gz`, `score_v2/FIXED_REFERENCES.json`, `score_v2/SCORE_SEAL.json`의 원본. 여섯 typed packet과 봉인된 기하 재구성 참조를 재사용한다.
- SOURCE 아래 기존 `data` 및 legacy 문서/자산 경로: 원 RGB6개·원 mask NPZ6개·원 관측 provenance frame·원 annotation·기존 witness CSV·대표 목록과 gallery proof. ASSET_ROOT는 보존할 원 Git 작업공간 루트이며 `challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json`을 포함해야 한다. 최종 불변 검사에는 원 SCORE `SOURCE_BINDINGS.json.private_parent_artifacts` 및 STATE score seal이 열거하는 기존 artifact/annotation/참조 파일의 SHA도 필요하다. 이는 기존 바이트 읽기이며 전량 재평가가 아니다. 파일이 없으면 원 manifest에서 해당 멤버만 복원하고, 새 추론으로 만들지 않는다.

여섯 ID와 순서는 `plastic_night_01:038630`, `wood_day_01:005286`, `wood_night_01:030698`, `plastic_day_01:020955`, `plastic_day_01:014704`, `wood_night_01:030898`로 고정한다. 앞의4장은 기존 실패 사례, 뒤의2장은 성공 반례다. 이들은 과거 결과로 고른 설명 사례이며, 독립 검증 표본·전체32/114의 물리 판정·새192장 성능이 아니다. CAL46도 합치지 않는다.

원 RGB와 NPZ의 기존 공개 허가는 legacy `publication_followup_20261015_v11/repro_inventory_current_files_v2.jsonl.gz`에서 확인한다. inventory SHA256은 `75c5aaa33d534faefc028c3358d9be6b5cbe688599488a29d34478502fac47b5`이며,12멤버의 원 경로·SHA·`prior_release_exact`를 검사한다. 새 Git 산출물에는 원 full RGB 파일·NPZ·대형 dense·가중치를 복제하지 않는다. 기존 허가와 연결된 표시용 전체 장면 JPEG, 정확한 crop PNG, mask 파생 그림만 게시한다. JPEG 재압축과 원 RGB의 바이트 동일성을 혼동하지 않는다. 권한/SHA를 확인할 수 없으면 해당 공개물을 만들지 않고 공백을 보고한다.

## 환경과 새 출력 경로

기존 Python3.10 환경의 NumPy·SciPy·OpenCV·Pillow·Matplotlib를 사용한다. 새 모델이나 패키지를 설치할 필요는 없다. CPU 분석 상한7200초, 무거운 CPU 작업 동시 최대2개, 각 라이브러리 thread1이며 아래 명령은 순차 실행한다. GPU 모델 작업은0이다. `/path/to/...`는 실행자의 별칭이다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TASK_PYTHON=/path/to/existing/python
export TASK_SOURCE=/path/to/isolated-worktree
export TASK_SCORE_PRIVATE=/path/to/existing-selection-score-evidence
export TASK_STATE_PRIVATE=/path/to/existing-statewise-evidence
export TASK_ASSET_ROOT=/path/to/preserved-original-git-workspace
export TASK_PRIVATE=/path/to/new-empty-physical-support-evidence
export TASK_DATA=/path/to/restored-snapshot/data
export TASK_LEGACY=/path/to/restored-snapshot/_docs/experiments/pallet_visibility_curriculum_20261013_v1
export TASK_PUBLISHED="$TASK_SOURCE/_docs/experiments/pallet_outline_physical_support_audit_v1"
export TASK_PUBLIC="$TASK_SOURCE/_docs/experiments/pallet_outline_physical_support_reproduced_v1"
cd "$TASK_SOURCE"
```

원 사용자 작업공간의 dirty 파일을 보존하고 별도 checkout에서 실행한다. PRIVATE와 PUBLIC은 새 경로여야 한다. PUBLIC은 상대 문서 링크를 위해 SOURCE 아래 `_docs/experiments/새이름`의3단계 경로를 쓴다. 기존 `data`/legacy 경로가 다른 대상을 가리키면 아래 설정은 중단한다. symlink는 OS 쓰기 차단을 만들지 않으므로 대상에 쓰지 않는다. 원 JSON 내부 경로나 좌표를 새 경로에 맞춰 재작성하지 않는다.

```bash
"$TASK_PYTHON" -B - <<'PY'
import hashlib, json, os, subprocess, time
from pathlib import Path
source=Path(os.environ['TASK_SOURCE']).resolve()
for relative,key in [('data','TASK_DATA'),
    ('_docs/experiments/pallet_visibility_curriculum_20261013_v1','TASK_LEGACY')]:
    target=Path(os.environ[key]).resolve()
    if not target.is_dir(): raise FileNotFoundError(target)
    link=source/relative
    if link.exists() or link.is_symlink():
        if not link.exists() or link.resolve()!=target:
            raise FileExistsError('기존 경로 보존: '+str(link))
    else:
        link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(target,target_is_directory=True)
registry=Path(os.environ['TASK_ASSET_ROOT'])/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
if not registry.is_file(): raise FileNotFoundError(registry)
private=Path(os.environ['TASK_PRIVATE']); private.mkdir(parents=True,exist_ok=False)
public=Path(os.environ['TASK_PUBLIC']).resolve()
relative=public.relative_to(source)
if len(relative.parts)!=3 or relative.parts[:2]!=('_docs','experiments'):
    raise ValueError('PUBLIC은 SOURCE/_docs/experiments/새이름이어야 한다')
public.mkdir(parents=True,exist_ok=False)
asset=Path(os.environ['TASK_ASSET_ROOT'])
def git(root,*args):
    return subprocess.run(['git',*args],cwd=root,check=True,capture_output=True).stdout
tracked=git(asset,'status','--porcelain','-z','--untracked-files=no')
all_status=git(asset,'status','--porcelain','-z')
with (private/'ORIGINAL_STATUS_NUL.bin').open('xb') as stream: stream.write(all_status)
protected_paths=['.gitignore','readme.md',
    '_docs/experiments/pallet_resnet18_refiner_20261001_v1/README_KO.md',
    '_docs/experiments/pallet_three_estimator_20261001_v1/README_KO.md',
    '_docs/experiments/pallet_three_estimator_20261001_v1/STATUS_KO.md',
    'data/evaluation/pallet_eval_v1/reports/ANNOTATION_PROGRESS.md']
protected={name:hashlib.sha256((asset/name).read_bytes()).hexdigest()
           for name in protected_paths if (asset/name).is_file()}
value=dict(start_epoch=time.time(),execution_HEAD=git(source,'rev-parse','HEAD').decode().strip(),
    origin_main_cached=git(source,'rev-parse','origin/main').decode().strip(),
    worktree_porcelain=git(source,'status','--porcelain=v1').decode(),
    original_HEAD=git(asset,'rev-parse','HEAD').decode().strip(),
    original_branch=git(asset,'branch','--show-current').decode().strip(),
    original_tracked_status_SHA256=hashlib.sha256(tracked).hexdigest(),
    original_all_status_SHA256=hashlib.sha256(all_status).hexdigest(),
    protected_files_SHA256=protected,
    protected_paths_absent=[name for name in protected_paths if not (asset/name).is_file()],
    cpu_analysis_cap_seconds=7200,max_concurrent_heavy_CPU_jobs=2,new_model_jobs=0,
    frozen_prior_review_reused=True)
with (private/'START_BEFORE_FETCH.json').open('x') as stream:
    json.dump(value,stream,ensure_ascii=False,indent=2)
PY
```

원 HEAD·작업공간 기록은 이 실행의 기록이다. 원격 최신 상태는 별도 `git fetch` 후 확인하며, 다운로드/대기/모델 육안검토 시간을 CPU 단계 시간으로 만들지 않는다. 다음 CLI의 `--help`를 실제 확인했다: bindings, mask_lineage, registration_audit, gallery, review, gallery_qa, independent_review_check, package, validate. 출력이 이미 존재하면 지우거나 덮어써 성공을 맞추지 말고 실패 시도와 예외를 보존한 새 출력 경로를 사용한다.

## 바이트 연결·원 mask 변환·등록 계약

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_outline_physical_support_audit_v1.bindings \
  --source-root "$TASK_SOURCE" --score-private "$TASK_SCORE_PRIVATE" \
  --state-private "$TASK_STATE_PRIVATE" --output "$TASK_PRIVATE/bindings_v3"
"$TASK_PYTHON" -B -m scripts.research.pallet_outline_physical_support_audit_v1.mask_lineage \
  --source-root "$TASK_SOURCE" --score-private "$TASK_SCORE_PRIVATE" \
  --state-private "$TASK_STATE_PRIVATE" --output "$TASK_PRIVATE/mask_lineage_v2"
"$TASK_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
from scripts.research.pallet_outline_physical_support_audit_v1.audit_guard import SupportAuditOnly
from scripts.research.pallet_outline_physical_support_audit_v1.registration_audit import run
root=Path(os.environ['TASK_PRIVATE'])
with SupportAuditOnly() as guard:
    run(source_root=Path(os.environ['TASK_SOURCE']),asset_root=Path(os.environ['TASK_ASSET_ROOT']),
        score_private=Path(os.environ['TASK_SCORE_PRIVATE']),state_private=Path(os.environ['TASK_STATE_PRIVATE']),
        output=root/'REGISTRATION_AUDIT_V2.json')
with (root/'REGISTRATION_RUNTIME_GUARD_V2.json').open('x') as stream:
    json.dump(guard.receipt(),stream,ensure_ascii=False,indent=2)
PY
```

등록 감사의 동일 CLI는 `registration_audit --source-root ... --asset-root ... --score-private ... --state-private ... --output FILE`이다. 위 wrapper는 같은 함수를 실행하면서 이 프로세스의 금지 함수 진입을 별도 기록한다. `bindings`·`mask_lineage`·gallery·review는 자체 `SupportAuditOnly`를 사용한다. 경계 생성 검산은 실제 원 backend의 저장 clean mask→hull→clip→표본 변환을 같은 설정으로 재현하는 것이다. 새로운 RGB 관측이나 대체 mask를 만들지 않는다. 실제 소비 mask는 `pallet_mask_clean`이며 `pallet_mask`는 같은 배열의 alias다. `pallet_mask_raw`는 저장 SAM binary, `state_pallet_mask`는 Tiny soft state, `pallet_mask_tight`는 저장돼 있지만 이번 outline에 쓰이지 않은 binary다. SAM decoder soft/logits는 NPZ에 없으며 Tiny soft를 SAM soft로 부르지 않는다.

검산값은6장·12고정후보·127유한 부모 선분·3030 observed 표본이다. polygon·유한 선분·표본의 원 순서와 좌표는 exact parity여야 한다. 등록 감사는 원 K/canvas·native/physical Q·치수·동일 R/t·기존 참조 전환을 직접 행렬 연산으로 확인한다. 원 detector letterbox 전체 계보나 실제 물리 축·카메라 독립 보정·참조의 물리 정답성을 새로 인증하지 않는다.

## 후보/비용을 숨긴 초기 표시와 동결 검토 입력

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_outline_physical_support_audit_v1.gallery \
  --source-root "$TASK_SOURCE" --score-private "$TASK_SCORE_PRIVATE" \
  --state-private "$TASK_STATE_PRIVATE" --phase initial \
  --output "$TASK_PRIVATE/initial_gallery"
```

`initial_gallery/index.html`은 표시선 없는 전체 RGB와 원좌표 확대·mask 단계·원 contour/hull·부모 ID를 담고 후보명/성공/오차/비용을 숨긴다. 과거에 이미 알고 있던 사례이므로 이를 독립 blind 검증이라고 부르지 않는다. 전체 저장 mask와 hull까지 포함한 표시 crop을 별도로 기록하며, case04의 cone 꼭대기가 잘리던 옛 crop도 원 기록으로 남긴다. 어두운 화면의 gamma0.5 사본은 원본과 별도 표시하며 수치 입력에는 사용하지 않는다.

다음은 **게시된 모델 판독과 그 산술을 정확히 재현**하는 경로다. 기존 실제 이미지 검토의 모델 종류·시각·근거·미확정·비독립 한계를 그대로 입력으로 재사용한다. 새 실행이 과거 시각에 새 육안검토를 수행했다고 기록하지 않는다. 판독 범주는 진단 전용이며 GT·학습 타깃·자동 물리 판정기가 아니다. 별도 인간/모델 재검토는 새로운 판독 기록을 남겨야 하며 이 동결 장부의 동일성 검산과 구별한다.

```bash
"$TASK_PYTHON" -B - <<'PY'
import hashlib, os
from pathlib import Path
published=Path(os.environ['TASK_PUBLISHED']); root=Path(os.environ['TASK_PRIVATE'])
names=['REVIEW_PROTOCOL.json','RAW_RGB_INITIAL_REVIEW.json',
       'MASK_FIRST_SUPPORT_DECISIONS.json','FINAL_VISUAL_REVIEW.json']
for name in names:
    original=published/'evidence/review'/name
    value=original.read_bytes()
    with (root/name).open('xb') as stream: stream.write(value)
    assert hashlib.sha256(value).digest()==hashlib.sha256((root/name).read_bytes()).digest()
assert (root/'MASK_FIRST_SUPPORT_DECISIONS.json').read_bytes()==(published/'SUPPORT_DECISIONS.json').read_bytes()
# 기존 선행/출처 감사는 새 실험이 아니며 원 기록 그대로 보존한다.
name='OCCLUDER_PRECEDENT_SOURCE_AUDIT.json'
with (root/name).open('xb') as stream:
    stream.write((published/'evidence'/name).read_bytes())
PY
```

이전에 실제로 원 RGB→초기 mask 화면→후보/비용 패널 순서로 본 기록은 [검토 프로토콜](evidence/review/REVIEW_PROTOCOL.json), [초기 RGB 검토](evidence/review/RAW_RGB_INITIAL_REVIEW.json), [동결 부모 판독](evidence/review/MASK_FIRST_SUPPORT_DECISIONS.json), [최종 그림 검토](evidence/review/FINAL_VISUAL_REVIEW.json)이다. `SUPPORT_DECISIONS.json`은 동결 판독과 같은 바이트여야 한다. 판독시각을 현재로 바꾸거나 모델 검토를 사람 검수로 바꾸지 않는다. 공개 영수증의 경로 별칭 치환은 의미 범주·시각·출처를 바꾸지 않는다.

## 다섯 층 표시·저장 비용 집계·독립 검산

보고서와 해석 문서는 원 실행의 모델 판독에 관한 정적 기록이다. 최종 QA가 보고서 링크도 확인할 수 있게 먼저 복사하되, 정적 Markdown의 `(VALIDATION.json)`/`(VALIDATION.md)` 링크만 보존된 원 게시 폴더로 바꾼다. 해석·판독시각·수치는 그대로이며 원 JSON 영수증은 재작성하지 않는다. 따라서 해당 링크의 **과거 원 완료 영수증**을 fresh 실행의 PASS로 보지 않는다. 새 검산 결과는 아래에서 생성하는 `INDEPENDENT_REVIEW_CHECK.json`과 `GALLERY_FINAL_QA.json`이다. 새 성능 주장이나 새 검토 결과로 바꾸지 않는다.

```bash
"$TASK_PYTHON" -B - <<'PY'
import os
from pathlib import Path
source=Path(os.environ['TASK_PUBLISHED']); out=Path(os.environ['TASK_PUBLIC'])
for name in ['REPORT_KO.md','NEXT_CHANGE_PROPOSAL_KO.md',
             'REGISTRATION_PROVENANCE_KO.md','REVIEW_REQUIRED.md','REPRODUCE.md']:
    text=(source/name).read_text()
    for suffix in ['json','md']:
        text=text.replace('(VALIDATION.'+suffix+')',
            '(../pallet_outline_physical_support_audit_v1/VALIDATION.'+suffix+')')
    with (out/name).open('x') as stream: stream.write(text)
PY
"$TASK_PYTHON" -B -m scripts.research.pallet_outline_physical_support_audit_v1.gallery \
  --source-root "$TASK_SOURCE" --score-private "$TASK_SCORE_PRIVATE" \
  --state-private "$TASK_STATE_PRIVATE" --phase complete \
  --analysis-private "$TASK_PRIVATE" --initial-folder "$TASK_PRIVATE/initial_gallery" \
  --initial-review-receipt "$TASK_PRIVATE/MASK_FIRST_SUPPORT_DECISIONS.json" \
  --output "$TASK_PUBLIC/gallery"
"$TASK_PYTHON" -B -m scripts.research.pallet_outline_physical_support_audit_v1.review \
  --source-root "$TASK_SOURCE" --score-private "$TASK_SCORE_PRIVATE" \
  --state-private "$TASK_STATE_PRIVATE" --root "$TASK_PRIVATE" --output "$TASK_PUBLIC" \
  --decisions "$TASK_PRIVATE/MASK_FIRST_SUPPORT_DECISIONS.json"
"$TASK_PYTHON" -B -m scripts.research.pallet_outline_physical_support_audit_v1.gallery_qa \
  --source-root "$TASK_SOURCE" --score-private "$TASK_SCORE_PRIVATE" \
  --state-private "$TASK_STATE_PRIVATE" --analysis-private "$TASK_PRIVATE" \
  --gallery "$TASK_PUBLIC/gallery" --output "$TASK_PRIVATE/GALLERY_FINAL_QA.json"
"$TASK_PYTHON" -B -m scripts.research.pallet_outline_physical_support_audit_v1.independent_review_check \
  --source-root "$TASK_SOURCE" --private "$TASK_PRIVATE" --score-private "$TASK_SCORE_PRIVATE" \
  --public "$TASK_PUBLIC" --output "$TASK_PRIVATE/INDEPENDENT_REVIEW_CHECK.json"
```

완성 gallery는 RGB / 실제 mask / 원 비용 경계·표본 / 두 고정 후보 / 별도 기하 참조를 분리한다. 후보 `(image_id,candidate_id)`·stage·family·R/t·Q·등록 크기는 원 목록 그대로다. 기존 참조는 독립 물리 정답이 아니다. 원 가장 큰 양/음 표본60개의 전체 부모와 이웃을18개 확대 그림으로 보존하며, 작은 값 목록에 실제 음수가 없어도 부호를 바꾸지 않는다. 같은 부모의 표본을 독립 증거로 세지 않는다.

`Delta = comparator_contribution − selected_contribution`이다. observed→model의3030개 공유 표본은 원 index·좌표·순서를 유지하고 `math.fsum`으로 기존 방향 차와1e-9 안에서 대조한다. model→observed는 후보마다 다른 자체 표본·호 길이·normalizer·비용을 각각 보존하며 index끼리 빼지 않는다. 원 outline weight와1/2가 포함된 기여를 다시 곱하거나 px 오차라고 부르지 않는다. 점·부분 선 비용은 기존 값의 문맥이며 재fit·삭제·새 선택을 하지 않는다.

QA는 원 crop PNG의 정확한 픽셀,12후보+6참조의 독립 homogeneous projection,3030표본·127부모·60극값·18확대와 이미지 decode·링크를 확인한다. 독립 review checker는 review/cost 함수 없이 원 witness와 저장 dense의 합·부모/범주 분모·각 방향 normalizer·JSONL/CSV·동결 판독 시각을 검산한다. `UNRESOLVED`25부모/686표본을 포함한 전체 분모를 유지한다. 숫자 검산 PASS는 의미 범주의 물리적 참을 증명하지 않는다.

## 작은 결과 포장과 보존

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_outline_physical_support_audit_v1.package \
  --source-root "$TASK_SOURCE" --asset-root "$TASK_ASSET_ROOT" \
  --score-private "$TASK_SCORE_PRIVATE" --state-private "$TASK_STATE_PRIVATE" \
  --analysis-private "$TASK_PRIVATE" --output "$TASK_PUBLIC"
```

package는 경로 별칭 치환·작은 영수증/입력 SHA·실제 시간 집계만 수행한다. 새 계산을 수행한 것처럼 이전 실행 시간을 복사하지 않는다. 원 `bindings_v1/v2`, `mask_lineage_v1`, 옛 등록 검사, crop/폰트 표시 수정 시도와 `GALLERY_COST_AND_ATTEMPTS.json` 등 **역사 영수증은 새 실행의 필수 입력이 아니다.** 이미 존재하는 역사 자료는 그대로 보존하며 원 실행의 실패를 숨기지 않는다. 새 실행에서는 자기 단계의 elapsed/process CPU 영수증을 사용하고, 동결 육안 판독의 과거 시각은 입력 provenance로만 남긴다.

## 원 실행 완료 QA의 범위

위 fresh 재현의 수치 검산은 `independent_review_check`와 `gallery_qa`가 수행한다. `validate`는 **원 실행 완료 당시의 고정 workspace snapshot**을 검사하는 별도 도구다. 원 작업공간의 HEAD/branch/NUL status·보호 파일·고정 namespace 밖 변경까지 검사하므로 다른 시점의 fresh workspace에서 같은 완료 PASS를 만들기 위한 명령이 아니다. 다른 이름의 fresh PUBLIC을 원 허용 namespace로 가장하거나 기존 `VALIDATION.json`을 새 실행 결과로 표기하지 않는다. 공개 [VALIDATION.json](VALIDATION.json)은 원 실행의 완료 영수증으로 읽는다.

원 완료 명령의 실제 CLI 형식은 다음과 같다. 여기의 `ORIGINAL_*`는 원 완료 snapshot의 별칭이며 위 fresh `TASK_*` 출력과 구별한다. 원 snapshot과 예상 불변 상태가 없는 환경에서는 실행하지 않는다. 기존 private 완료 영수증을 덮어쓰지 않도록 출력 이름도 새로 지정해야 한다.

```bash
"$TASK_PYTHON" -B -m scripts.research.pallet_outline_physical_support_audit_v1.validate \
  --source-root "$ORIGINAL_SOURCE" --asset-root "$ORIGINAL_ASSET_ROOT" \
  --score-private "$ORIGINAL_SCORE_PRIVATE" --state-private "$ORIGINAL_STATE_PRIVATE" \
  --analysis-private "$ORIGINAL_AUDIT_PRIVATE" --public "$ORIGINAL_PUBLIC" \
  --output-private "$ORIGINAL_AUDIT_PRIVATE/FINAL_VALIDATION_NEW_RECEIPT.json"
```

원파일·예측·참조·기존 소스 불변, ID/좌표/합·분모·미확정 보존, 개인 경로/권한·게시 링크를 최종 검사한다. Git에 올리는 것은 새 namespace의 코드·작은 표/영수증·허용 파생 이미지다. 대형 원자료와 전체 cost 은행은 기존 manifest/Release·private evidence에 남긴다. 게시 시 타인의 main 변경을 보존하고 원격 main에서 실제 SHA와 산출물 경로를 다시 읽는다. 이 진단에서 제안한 다음 변경의 개선 효과는 이번에 실험하지 않았다.
