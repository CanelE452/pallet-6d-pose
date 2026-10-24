# V190 main 게시 확인

원 q/L을 유지한 C54/C55 source-only 학습 각 8,000회와 TRAIN apparent·전체 UNKNOWN·코드·실패·검산을 보존한 게시 기록입니다. 자체 head 2개와 선언된 source 출력 8개를 포함하며 원본 RGB/SAM/input cache는 제외 provenance로 남깁니다. TRAIN 적합은 실사 개선·물리 native 정답 인증이나 전체 목표 해결의 증거가 아닙니다. V191 실사 벤치마크는 별도 단계이며 그 결과는 이 영수증에 포함하지 않습니다.

- [V190까지의 전체 실험 인덱스](../../../EXPERIMENT_INDEX_THROUGH_V190_KO.md)
- [V190 source 보고서](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_direct_end_paired_source_training_20261024_v190/PORTABLE_REPORT_KO.md)
- [독립 source 학습 감사](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_direct_end_paired_source_training_20261024_v190/independent_source_fit_review_b/actual_review_v1/REVIEW.json)
- [게시 영수증 JSON](PUBLICATION_RECEIPT_KO.json)
- [독립 원문 바이트 검산](closing/independent_QA/BYTE_ROUNDTRIP.json)
- [독립 링크·credential 검산](closing/independent_QA/PORTABLE_LINKS_CREDENTIALS.json)

- [독립 원격 public/main·5개 blob 검산](closing/remote/REVIEW.json)
- [복사된 게시 원문 검산 코드](closing/tools/verify_v190_delta.py)

검증된 payload는 101개 물리 파일, 21398205바이트입니다. 신규 원문 source 기록 66개와 지정 5개 원격 blob metadata의 독립 검산을 해당 영수증에 바인딩했습니다. 원격 콘텐츠·픽셀 재다운로드는 하지 않았습니다.

검증된 payload commit X: [9732d5b](https://github.com/CanelE452/pallet-6d-pose/commit/9732d5bb2683d71d29b730ae853895bcc5c1b2e9). 이 closing 기록의 후속 commit Y는 push 뒤 로컬 영수증에만 추가하여 자기 commit 순환 참조를 피합니다.
