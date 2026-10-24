# V187 immutable delta 실행 순서

1. Root가 source 보고서·표·32개 gallery·실패기록과 최종 report QA를 봉인한다. `ROOT_V187_PUBLICATION_SCOPE_BEFORE_SOURCE_ENUMERATION_OR_COPY.json`을 bind하고 `ROOT_DELTA_INVENTORY_ENTRY_TEMPLATE_APPROVED_FALSE.json`의 별도 실제 entry를 만든다.
2. `prepare_completed_phase_delta.py --entry <절대 root inventory entry>`로 V187만 열거·해시한다. 실제 목록은 root admin에 새로 기록되며 source·repo를 수정하지 않는다.
3. Root가 새 candidate/INVENTORY_COMPLETE, 기존 PUBLICATION_MANIFEST 및 index, 현재 global readme의 exact SHA와 최종 phase receipts를 `ROOT_DELTA_BUILD_ENTRY_TEMPLATE_APPROVED_FALSE.json`에 bind한다. `build_completed_phase_delta.py --build-entry <절대 실제 entry>`는 기존 copier로 새 V187 archive와 gzip/descriptor를 보관하고, 새 delta manifest/index/chain을 만든다. 기존 manifest·기존 index·기존 covered paths는 불변이다. Global readme만 이전 바이트를 별도 보관하고 새 내용의 suffix로 유지한다. Git 작업은 없다.
4. A가 새 delta 원문 roundtrip·gzip·stored SHA·구본 경로 불변·credential·상대 링크를 독립 검산한다. 원본 provenance MD의 절대 경로는 수정하지 않고, 공개용 portable phase report/새 index/global readme의 세 페이지를 검사한다.
5. Root가 `ROOT_DELTA_PUSH_PLAN_ENTRY_TEMPLATE_APPROVED_FALSE.json`에 BUILD_COMPLETE/delta manifest/독립 completed+PASS receipts를 bind한다. `build_completed_phase_delta.py --push-plan-entry <절대 실제 entry>`는 모든 새 물리 바이트를 다시 검증하고 QA/BUILD receipts까지 포함하는 480MiB 이하 새 plan을 만든다.
6. Root만 기존 `publication_root_preparation/push_publication_batches.py --plan <새 delta plan> --state <새 V187 progress 경로>`를 실행한다. 이전 main/local HEAD와 일치해야 하며 강제 push·관계없는 파일은 금지한다. 이후 새 V187 closing receipt 파일명으로 실제 remote payload commit을 기록한다. 기존 초기 closing receipt helper의 고정 이름을 재사용하지 않는다.

새 파일: `publication/deltas/V187_20261024_v1/DELTA_MANIFEST.json`, `EXPERIMENT_INDEX_THROUGH_V187_KO.md`, `MANIFEST_CHAIN_AFTER_V187.json`. Original `PUBLICATION_MANIFEST.json`과 `EXPERIMENT_INDEX_KO.md`는 historical 검산을 계속 통과해야 한다. Restore helper는 base 또는 delta manifest를 명시적으로 받는다.

`INVENTED_MANIFEST_AND_PLAN_SCHEMA_EXAMPLE.json`은 필드 예시일 뿐 실제 숫자/SHA가 아니다. 발명 end-to-end 검사는 실제 source 데이터 대신 임시 두 텍스트와 제외 raw fixture만 사용해 copier→gzip→delta/index/chain→검산→pushplan 연결을 검증했다. 실제 phase 파일 열거·해시·복사 및 Git 작업은 root entry 전 0이다.
