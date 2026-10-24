# V192 누적 immutable delta 게시 절차

이 도구는 공개 source metadata 진단과 bbox-only 가상 SHAM 구성 가능성 기록을 게시합니다. 새 영상·alpha·라벨·학습·실사 평가·모델 개선이나 채택을 뜻하지 않습니다. 원 V191 도구와 이전 게시 원문은 변경하지 않았습니다.

1. Root가 V192 보고서·독립 QA·최종 동결을 끝낸 뒤 inventory false-entry를 실제 승인합니다. `phase_completion_receipts`는 phase_result, independent_report_QA, final_phase_freeze 세 역할과 실제 metadata의 expected_fields를 바인딩합니다. 그 전에는 원 phase 파일 열거/해시/복사를 하지 않습니다.
2. 정확 V192 한 폴더만 inventory합니다. V112 이름은 stat 전에 거절하고 symlink·pycache·환경은 따라가지 않습니다. 코드·문서·표·검토 그림은 포함하고 모든 배열·가중치·cache는 제외 사유를 보존합니다. 두 whitelist는 반드시 빈 배열이며 prediction/fit seal은 받지 않습니다.
3. Root가 실제 main을 확인하여 expected_remote_main을 넣고 BASE+V187~V191의 delta/chain/index 및 BASE+V187~V191 closing pushplan 6개를 정확히 바인딩합니다. 이전 manifest-covered logical/raw/gzip/descriptor/closing 경로는 변경하지 않습니다. BASE 역사 README 1개 예외는 이전 규칙 그대로입니다.
4. 별도 root build-entry가 기존 copier로 새 archive와 lossless gzip descriptor를 만듭니다. THROUGH_V192 인덱스와 AFTER_V192 chain, 새 tools 경로만 추가합니다. 기존 root readme.md는 원문 바이트를 별도 보존하고 새 README의 suffix로 유지합니다.
5. 별도 독립 byte/압축/자격증명/relative-link/parent 불변 검산 후 root push-plan-entry가 delta-only ≤480MiB batch를 봉인합니다. 실제 Git/main push는 기존 strict pushrunner를 root가 별도 실행합니다. 이 도구는 Git/network를 실행하지 않습니다.

원문 byte 보존·실패/U/UNKNOWN 분모 보존은 게시 일관성 검사이며 과학적 성능 검증을 대신하지 않습니다. 가상 roundtrip은 임시 가상 source/repo만 사용했습니다.
