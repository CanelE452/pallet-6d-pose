# V222 원바이트 게시 — 준비 상태

이 도구는 CODEONLY로 준비했습니다. 현재 실제 V222 열거·원문 SHA·복사·Git·HTTP 실행은 0입니다. Root가 실제 학습 결과·독립 QA·최종 phase/freeze를 닫은 뒤 exact actual entry로 실행합니다. 결과 숫자·학습 예산·개선 여부를 이 준비 도구가 추정하지 않습니다. 최신 실사 평가 V220과 SOURCE 학습 V222는 구분하고, 모델 선택·채택·목표 해결을 주장하지 않습니다.

핵심 파일은 `publish_completed_v222.py`와 이 runbook입니다. 바이트/NOFOLLOW/링크 11개 및 Git/index/blob 4개 함수는 검토된 V220 kernel과 AST가 같습니다. `GENERATED_INVENTED_CHECKS.json`과 `FINAL_READY.json`은 준비 봉인 metadata입니다. 별도 false-entry 파일 다수나 추가 closing commit은 만들지 않습니다.

## 고정 범위

- Repository: `/home/minjae/Documents/pallet-visibility-publication-20261024`
- Parent main: `8e8cda5a8d6e0ce8cc02426fa633ceec36d02a9f`
- Source: `/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_expected_part_counterfactual_source_bridge_20261025_v222`
- Stage: `/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V222_PUBLICATION_STAGE_C`
- Delta: `_docs/experiments/pallet-keypoint-visibility/publication/deltas/V222_20261025_v1/`
- Prior index: `_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_THROUGH_V221_KO.md`（変更なし）
- New index: `_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_THROUGH_V222_KO.md`
- 최신 prefix만 추가하는 기존 파일: `readme.md`, `_docs/experiments/pallet-keypoint-visibility/README.md`, `_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md`. 기존 전체 bytes를 suffix와 delta snapshot으로 보존합니다.

원본 positive text/code/table/SVG/PNG와 정확히 다음 새 PT 두 개만 원바이트 SHA/size로 게시합니다. 모델·이미지·배열 해석은 없습니다.

```
paired_source_fit_v1/C66CONTINUATION_CONTROL/FINAL_CHECKPOINT.pt
paired_source_fit_v1/C67PLUS_EXPECTED_PART_CF_WEAK/FINAL_CHECKPOINT.pt
```

`owned_binary_whitelist`의 path/SHA/size는 root가 두 final 상태 동결 후 제공한 binding만 사용합니다. 두 checkpoint binding은 `paired_source_fit_v1/COMPLETE.json` 또는 `ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json` 저장 metadata에도 있어야 합니다. 원 C65 PT·입력/feature/cache/예측 NPZ·NPY·다른 weights는 게시하지 않습니다. 실제 phase 안에 RAW가 존재한다면 root의 정확한 phase-relative `excluded_RAW_paths` 목록에 있어야 하며, 아니면 **stat/hash/header/body 전에 중지**합니다. 제외 목록에 든 RAW도 stat/hash/header/body를 읽지 않습니다. 새 scalar prediction은 JSON이므로 기본 NPZ 허용은 0입니다. source binding pointer는 따라가지 않습니다.

원 phase만 열거하며 cache 디렉터리는 stat 전에 제외합니다. 모든 경로에서 V112 문자열은 파일시스템 전에 거부하고, parent/leaf NOFOLLOW를 사용합니다. 기존 ignored `tools/__pycache__/check_publication_qa.cpython-313.pyc`, `tools/__pycache__/verify_published_results.cpython-313.pyc` 두 경로는 Git 목록의 이름 비교 예외뿐이며 직접 stat/read/hash/stage하지 않습니다.

기본 상한은 source 1,200개/180MiB, file 각각 strictly <100MiB, 한 batch strictly <480MiB입니다. Root는 실제 entry의 `source_caps`로 기본 상한 이내에서 줄일 수 있습니다. 실제 자료가 상한을 넘으면 완화·재시도하지 않고 기록 후 중지합니다.

## Root actual entry의 공통 필드

네 단계는 같은 code와 runbook binding을 사용하되 별도 entry를 기록합니다. 모든 binding은 `{path,sha256}` 또는 명시된 경우 `{path,sha256,size_bytes}`입니다. 경로는 절대경로입니다.

```
approved_actual_operation: "stage" | "verify" | "push" | "post"
automatic_retry: false
publisher: 정확한 publish_completed_v222.py binding
runbook: 정확한 RUNBOOK_KO.md binding
root_actual_scope: R 아래 실제 root 승인 JSON binding
initial_remote_main: "8e8cda5a8d6e0ce8cc02426fa633ceec36d02a9f"
repository: "/home/minjae/Documents/pallet-visibility-publication-20261024"
```

실제 scope에는 해당 `approved_actual_stage`, `approved_actual_verify`, `approved_actual_push`, `approved_actual_post`가 True인 단계만 허용됩니다. `stage` entry에는 **`approved_actual_enumeration_or_copy:true`**가 추가로 필요합니다. 누락/false는 원문 접근 전 거부합니다. 준비 단계에는 모두 false이며 실제 entry가 아직 없습니다.

## stage entry 추가 필드

```
source_root: 위 고정 source 절대경로
phase_result: {binding: ROOT_PHASE_RESULT.json, expected_fields: 실제 completed/phase 등}
final_freeze: {binding: ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json, expected_fields: 실제 동결 필드들}
paired_fit_complete: {binding: paired_source_fit_v1/COMPLETE.json, expected_fields: 실제 완료 필드들}
all2_freeze: {binding: paired_source_fit_v1/ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json,
              expected_fields: completed:true + 실제 final barrier 필드}
owned_binary_whitelist: [새 PT 두 개의 정확한 {path,sha256,size_bytes}]
excluded_RAW_paths: [] 또는 실제 root 확인 phase-relative RAW closed list (최대 64개)
source_caps: {files:1200,bytes:188743680}
history_before: {
  "EXPERIMENT_INDEX_THROUGH_V221_KO.md": clone prior versioned index binding,
  "readme.md": clone root README binding,
  "_docs/experiments/pallet-keypoint-visibility/README.md": docs README binding,
  "_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md": canonical index binding
}
publication_prefixes: {index_utf8: root 최종문구, root_readme_utf8: root 최종문구,
                       docs_readme_utf8: root 최종문구}
portable_pages: ["EXPERIMENT_INDEX_THROUGH_V222_KO.md",
  "archive/task-2/pallet-keypoint-visibility/<V222 phase basename>/PORTABLE_REPORT_KO.md", ...]
commit_message: root가 정한 한 줄
```

Freeze의 `phase_result` binding은 entry의 result binding과 같아야 합니다. `expected_fields`는 실제 봉인 원문 필드를 사용합니다. 지원 동결 schema는 `frozen && all_scientific_phase_files_final_no_more_writes`, `scientific_phase_frozen && no_further_phase_writes`, 또는 `scientific_phase_frozen_no_further_phase_writes`가 정확히 True인 경우입니다. `completed`만인 가짜 freeze는 거부합니다. Root scope가 fit/독립 QA/보고서의 과학적 완료를 책임지고 이 도구는 해당 완료 metadata와 원바이트를 보존합니다. 새 모델 성능·physical gold·훈련 손실 재현 gate를 추가하지 않습니다.

`portable_pages`는 PUB 기준 phase 소유 MD/HTML과 새 index만 지정합니다(JSON은 일반 record로 자동 게시). 현재 phase 페이지의 모든 상대 링크와 새 global/index prefix 링크를 검사합니다. 예전 index/README suffix는 opaque 보존하므로 과거 archive를 재순회하지 않습니다. 새 글에서 V112 경로 링크는 허용하지 않습니다.

stage 성공 시 `V222_PUBLICATION_STAGE_C/{COMPLETE.json,QA.json,ROOT_V222_PUSH_PLAN.json}`과 clone delta manifest를 반환합니다. 현 시점 source PNG 수·PT SHA·전체 file/bytes는 확정하지 않습니다. 실패 시 과거 stage나 source를 되돌리거나 지우지 않고 그대로 보존합니다.

## verify / push / post

세 단계 공통 추가 binding은 `stage_complete`, `push_plan`, `manifest`입니다. `verify`는 closed manifest만 읽어 모든 원본/대상 SHA·size·최종 원문 및 전체 plan bytes, PT2 allowlist 일치, 네 historical suffix를 재확인합니다. 실제 delta가 plan과 같고 index가 비었으며 HEAD=remote=예상 parent인지 확인합니다. 출력은 `R/ROOT_V222_SOURCE_AND_PLAN_BYTE_QA.json`입니다. 같은 작성자의 저장 byte 검산이며 별도 사람/모델의 독립 과학 QA라고 주장하지 않습니다.

`push`에는 `approved_single_nonforce_main_push:true`, `byte_QA` binding이 추가됩니다. 모든 plan SHA를 재확인한 뒤 normal pathspec과 **정확한 declared PNG/new2PT pathspec만** 나눠 add합니다. 후자만 `--force`를 사용하며 디렉터리나 ignored cache는 추가하지 않습니다. CSV를 수정하지 않고 `core.autocrlf=false`로 add하며 다음을 실행합니다.

```
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check
```

실제 index와 commit tree의 정확한 name set, 각 explicit blob의 SHA256/size를 확인한 뒤 한 번만 `git push origin HEAD:refs/heads/main`(nonforce)을 실행합니다. HEAD/remote/clean 결과를 `R/V222_MAIN_PUSH_ROOT_TRANSACTION/COMPLETE.json`에 보존합니다. Root만 actual push entry를 실행합니다. 첫 실패 이후 자동 retry/reset/checkout/clean은 없습니다.

`post`에는 `transaction`, `expected_commit`, `HTTP_HEAD_targets:[{role,path}×6]`가 추가됩니다. INDEX와 ROOT_REPORT 역할은 새 versioned index와 새 PORTABLE_REPORT로 고정하며 나머지도 exact plan 내 non-RAW 파일입니다. 전체 source/physical 재hash, HEAD=remote, normal clean/index empty, immutable commit URL 6개 HEAD200을 확인합니다(HTTP body 0). `R/ROOT_FINAL_V222_MAIN_PUBLICATION_LOCAL_RECEIPT.json`만 기록하며 과학 phase·ACTIVE_HANDOFF를 수정하지 않습니다. Canonical handoff의 새 SOURCE 상태는 root가 별도로 관리합니다.

## 명령 순서

```
PY=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
H=/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V222_PUBLICATION_PREPARATION_C
$PY -B "$H/publish_completed_v222.py" --self-test
$PY -B "$H/publish_completed_v222.py" --mode stage  --entry <root 실제 stage entry 절대경로>
$PY -B "$H/publish_completed_v222.py" --mode verify --entry <root 실제 verify entry 절대경로>
$PY -B "$H/publish_completed_v222.py" --mode push   --entry <root 실제 push entry 절대경로>
$PY -B "$H/publish_completed_v222.py" --mode post   --entry <root 실제 post entry 절대경로>
```

매 단계 exit0와 completed/PASS 후 다음 단계로 진행합니다. 실행이 yield되면 같은 session을 poll합니다. 현재 단계는 CODEONLY 완료 뒤 root 지시 대기입니다.
