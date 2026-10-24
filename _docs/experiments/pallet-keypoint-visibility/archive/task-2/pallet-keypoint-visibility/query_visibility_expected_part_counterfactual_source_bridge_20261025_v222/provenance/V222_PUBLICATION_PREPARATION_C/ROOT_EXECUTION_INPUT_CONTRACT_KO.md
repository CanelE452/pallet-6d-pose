# V222 root 실행 입력 점검 메모

기존 publisher/core/runbook/checks/READY 네 파일은 바꾸지 않았습니다. 이 메모는 frozen code의 실제 읽는 필드와 조건을 대조한 추가 문서입니다. 전체 science freeze는 아직 전이며, source 열거·hash·모델 size 조회·복사·Git·network 실행을 하지 않았습니다.

Root가 실제 SOURCE 학습·eval/QA 완료 및 generalization unsuccessful라고 전달했습니다. 이것은 새 current223 추론 결과가 아니며 최신 실사 평가는 V220으로 유지합니다. 실제 결과의 새 수치나 개선을 이 메모가 추정·검산하지 않습니다.

## 가장 먼저 맞출 binding shape

`publisher`와 `runbook`은 **정확히 `{path,sha256}` 두 키**여야 합니다. `entry_gate()`가 `bind()`의 2-key dict와 통째로 비교하므로 FINAL_READY의 `size_bytes`까지 그대로 복사하면 실패합니다. SHA는 다음과 같습니다.

- publisher: `c026896d5271525eaa5860cb4118ab46fea5d5fe9e67649199297f12d230c726`
- runbook: `739efa559a3ccd150303e27f316ea34e6c540f9c19a4dbbab14e922f5f523915`

나머지 일반 `checked_bytes()` binding은 `{path,sha256}` 또는 `{path,sha256,size_bytes}`를 허용합니다. 단 `stage_complete`에서 가져온 `entry`, `manifest`, `push_plan` graph는 **반환된 객체 모양 그대로** 재사용해야 합니다. Freeze 원문의 `phase_result`와 stage entry의 `phase_result.binding`도 객체가 정확히 같아야 합니다. PT whitelist 각 항목만은 정확히 `{path,sha256,size_bytes}` 세 키가 필수이며 `size_bytes`는 실제 정수여야 합니다.

## 네 actual entry 공통 필드

```
approved_actual_operation: "stage" / "verify" / "push" / "post"
automatic_retry: false
publisher: {path: H/publish_completed_v222.py, sha256: 위 값}
runbook: {path: H/RUNBOOK_KO.md, sha256: 위 값}
root_actual_scope: R 아래 root 실제 승인 JSON binding
initial_remote_main: "8e8cda5a8d6e0ce8cc02426fa633ceec36d02a9f"
repository: "/home/minjae/Documents/pallet-visibility-publication-20261024"
```

Entry 자체도 R 아래여야 합니다. root scope의 해당 `approved_actual_stage` / `approved_actual_verify` / `approved_actual_push` / `approved_actual_post`가 정확히 True여야 해당 mode만 열립니다. 현재 이 메모는 실제 entry나 승인 scope를 만들지 않습니다.

## stage 입력

- `approved_actual_enumeration_or_copy:true` 누락/false는 원문 접근 전에 거부합니다.
- `source_root`: 고정 V222 source 절대경로.
- `phase_result`: `{binding:S/ROOT_PHASE_RESULT.json, expected_fields:{...실제 필드...}}`.
- `final_freeze`: `{binding:S/ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json, expected_fields:{...실제 동결 필드...}}`.
- `paired_fit_complete`: `{binding:S/paired_source_fit_v1/COMPLETE.json, expected_fields:{...실제 완료 필드...}}`.
- `all2_freeze`: `{binding:S/paired_source_fit_v1/ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json, expected_fields:{completed:true, ...실제 final barrier 필드...}}`.
- `owned_binary_whitelist`: 아래 고정 PT 두 개의 절대경로/SHA/실제 size 정수. 다른 필드를 붙이지 않습니다.
- `excluded_RAW_paths`: 기본 `[]`. 존재하는 입력/feature/cache/이전 PT 등 RAW를 제외하려면 root가 확인한 **phase-relative** 정확한 closed list를 넣습니다(최대64, 중복0). 목록 외 RAW는 stat/hash/header/body 전에 STOP하며, 목록에 넣은 RAW도 읽지 않습니다. 목록과 실제 발견한 제외 leaf가 정확히 같아야 합니다. 외부 binding이 가리키는 원입력 경로를 넣거나 따라가지 않습니다.
- `source_caps`: 기본 `{files:1200,bytes:188743680}`. entry는 이 이하만 허용합니다.
- `history_before`: 정확히 네 키 `EXPERIMENT_INDEX_THROUGH_V221_KO.md`, `readme.md`, `_docs/experiments/pallet-keypoint-visibility/README.md`, `_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md`. 첫 키의 binding.path는 clone/PREFIX/THROUGH221이며 다른 셋은 clone-relative 경로의 실제 원문 binding입니다.
- `publication_prefixes`: 정확히 `index_utf8`, `root_readme_utf8`, `docs_readme_utf8` 세 개의 비어 있지 않은 root 최종 문자열. 각각 64KiB 미만입니다.
- `portable_pages`: PUB-relative MD/HTML 리스트(2..40, 중복0). 새 THROUGH222와 `archive/task-2/pallet-keypoint-visibility/<V222 basename>/PORTABLE_REPORT_KO.md`는 필수입니다. 나머지는 V222 archive 아래 MD/HTML만 허용합니다. JSON/CSV/PNG는 일반 record로 자동 게시되므로 이 목록에 넣지 않습니다.
- `commit_message`: 줄바꿈 없는 root의 최종 제목.

완료 receipt의 expected_fields는 nonempty이며, 원문 `completed`가 정확히 True여야 합니다. `all2_freeze.expected_fields`는 completed와 실제 barrier 필드 등 최소2개여야 합니다. Final freeze 기대값에는 다음 중 실제 저장 schema를 넣습니다: `frozen:true`와 `all_scientific_phase_files_final_no_more_writes:true`; 또는 `scientific_phase_frozen:true`와 `no_further_phase_writes:true`; 또는 `scientific_phase_frozen_no_further_phase_writes:true`. Root phase freeze의 `phase_result` binding과 entry가 같아야 합니다. 새 근거 없는 필드명을 추측하지 않습니다.

### Root 선언 PT 두 개 — 아직 size/opaque hash 재조회 0

```
paired_source_fit_v1/C66CONTINUATION_CONTROL/FINAL_CHECKPOINT.pt
9c34e9edda3938ff5daed45cf9684c42e2058940e4ea94f2e2cbc3f7285e14fd
paired_source_fit_v1/C67PLUS_EXPECTED_PART_CF_WEAK/FINAL_CHECKPOINT.pt
5b9efd6a9c7b6768ca5a0204130385e880e33adb527906e8e43a94ec13fcecdb
```

각 `size_bytes`는 실제 fit 완료 metadata의 선언값으로 채우면 되고 여기서는 null로 보존했습니다. 두 path/SHA tuple은 fit COMPLETE 또는 all2 barrier 저장 metadata 안에도 있어야 합니다. 모델값을 decode하지 않습니다. Root가 준 modelQA SHA `be9e42cf6d02a8306497eca7176beacf8200b88e98a6b9447a4ba654f6c9584f`는 선언 참고값이며, 이 게시기가 추가 모델 재검산을 요구하는 gate는 아닙니다. 그 실제 QA 코드/영수증은 phase archive에 들어 있으면 원바이트로 게시합니다.

## 실제 실행 순서와 반환 binding

1. Wholephase ROOT_PHASE_RESULT/freeze 완료와 root actual stage entry 이후 `--mode stage --entry <stage_entry>`를 한 번 실행합니다. 처음 HEAD/remote base·empty index·normal clean을 read-only 확인한 다음 원본을 bounded 열거/복사합니다. 원문과 destination SHA/size, 최종 원문 재hash, PT2·zeroNPZ, 상대 링크와 과거 suffix를 검증합니다. Phase 추가 쓰기0입니다.
2. Stage exit0 및 `V222_PUBLICATION_STAGE_C/COMPLETE.json` completed/PASS를 확인합니다. 그 객체의 `push_plan`과 `manifest`를 **그대로** 받아 verify entry를 만듭니다. `--mode verify --entry <verify_entry>` 한 번 후 `R/ROOT_V222_SOURCE_AND_PLAN_BYTE_QA.json` completed/PASS 확인. 이 mode는 closed manifest 원본 전수와 destination/plan을 다시 opaque 검산하고 index empty·HEAD/remote base·whole working delta exact plan을 확인합니다. 별도 주체의 독립 과학 QA라고 부르지 않습니다.
3. Root만 push entry에 공통 graph 세 binding과 `approved_single_nonforce_main_push:true`, `byte_QA`를 넣고 `--mode push --entry <push_entry>`를 실행합니다. PNG와 새 PT2만 명시 leaf pathspec force-add; 나머지는 normal add. CSV는 core.autocrlf=false 및 요청된 cr-at-eol whitespace 설정을 사용합니다. 모든 index blob와 commit tree blob SHA256/size 및 exact name set·single parent를 검증하고 nonforce `HEAD:refs/heads/main` 한 번. 반환 `R/V222_MAIN_PUSH_ROOT_TRANSACTION/COMPLETE.json` completed/PASS 및 commit을 확인합니다.
4. Post entry에 동일 graph 세 binding, `transaction`, `expected_commit`, `HTTP_HEAD_targets` 6개를 넣습니다. INDEX는 새 THROUGH222, ROOT_REPORT는 새 PORTABLE_REPORT로 고정하고 나머지 네 개도 exact plan의 non-RAW 파일이어야 합니다. `--mode post --entry <post_entry>` 후 `R/ROOT_FINAL_V222_MAIN_PUBLICATION_LOCAL_RECEIPT.json` 확인. 원본·destination 재hash, HEAD=remote·normal clean/index empty, 6 HEAD200/body0을 기록합니다. ACTIVE_HANDOFF나 science 파일은 수정하지 않습니다.

모든 actual CLI는 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B H/publish_completed_v222.py --mode <mode> --entry <R 아래 실제 entry 절대경로>`입니다. 매 단계 exit0/PASS 후 다음으로 갑니다. 첫 실패를 보존하고 자동 retry/rollback/reset/clean하지 않습니다. Yield되면 같은 session만 poll합니다.

## 변하지 않는 검증 경계

- Source wholephase만 열거, source binding follow0, 다른 phase나 GT/input/C65/private/protected 원문0.
- V112 lexical reject before filesystem, 부모와 leaf NOFOLLOW.
- PNG는 encoded opaque bytes, 새 PT2도 opaque bytes. INPUT/feature NPZ·NPY·old weights 게시0.
- source≤1200/180MiB, 각 file<100MiB, 한 batch<480MiB.
- 새 index는 THROUGH221 bytes를 suffix로, root/docs README 및 canonical index는 각각 현재 원문 bytes를 suffix로 보존하고 원문 snapshot을 delta에 둡니다.
- 과거 ignored cache 두 경로는 이름 예외뿐, 직접 stat/read/hash/stage0.
- Source 실험 결과가 좋거나 나쁜지와 무관하게 완료 기록을 그대로 공개합니다. No model/threshold selection/adoption; latest real V220 unchanged, goal active.
