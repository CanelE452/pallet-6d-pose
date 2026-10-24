# V189 완료 phase의 새 immutable delta 절차

현재는 코드·가상 검사 준비만 완료했습니다. 실제 V189 inventory, parent manifest, 저장소, Git/remote는 읽지 않았습니다.

1. Root가 V189 완료 범위와 최종 ROOT_PHASE_RESULT/report QA를 동결합니다. fit 준비와 실제 fit 완료를 publisher가 추론하지 않습니다.
2. Root가 false inventory entry의 scope/완료영수증/코드·protocol SHA를 채워 승인하고 inventory를 한 번 실행합니다. 원래 모든 코드·실패·UNKNOWN·표·PNG·portable MD를 선택 규칙대로 보존하며 arrays/weights/banks는 제외 provenance만 남깁니다.
3. Root build entry는 BASE와 순서대로 V187/V188의 delta/chain/index 7개를 exact binding 합니다. 최신 alias는 V188과 같아야 합니다. BASE→V187→V188 closure 검증 뒤 새 V189에만 복사합니다.
4. 기존 copier가 lossless gzip/parts/descriptors를 쓰고 새 V189 delta와 THROUGH_V189 index, AFTER_V189 chain을 만듭니다. 이전 index 전체 bytes는 suffix 그대로 보존합니다. global readme의 갱신은 별도 root binding한 기존 bytes를 보존한 prepend뿐입니다.
5. 독립 byte roundtrip/credential/relative-link/size QA 후 push-plan entry를 승인하여 finalizer를 실행합니다. strict pushrunner가 검증된 최신 main X에서 새 physical paths만 push하며 force push는 하지 않습니다. 이 도구 안에는 Git 명령이 없습니다.
6. 원격 증명과 closing receipt는 후속 root 절차로 별도 기록합니다. 자체 manifest/chain/후속 QA는 자기 hash 순환을 피하고 strict physical plan에 바인딩합니다.

실행 예: pinned Python + prepare_completed_phase_delta.py --entry ROOT_INVENTORY_ENTRY / build_completed_phase_delta.py --build-entry ROOT_BUILD_ENTRY / 같은 builder --push-plan-entry ROOT_PLAN_ENTRY. 현재 false entries는 실행 권한이 아닙니다. 실패는 보존하고 자동 재시도하지 않습니다.
