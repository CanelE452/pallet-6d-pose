# V189 main 게시 확인

원 q/L의 sampled-domain source 제어·직접 가림 구현과 전체 UNKNOWN·실패·그림·검산을 보존한 게시 기록입니다. 실제 cache·학습은 별도 V190 단계입니다. 새 모델 학습·실사 개선·물리 native 정답 인증이나 전체 목표 해결을 뜻하지 않습니다.

- [V189까지의 전체 실험 인덱스](../../../EXPERIMENT_INDEX_THROUGH_V189_KO.md)
- [V189 source 보고서](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_sampled_domain_direct_end_20261024_v189/PORTABLE_REPORT_KO.md)
- [합성 전체 그림](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_sampled_domain_direct_end_20261024_v189/synthetic_end_review_visuals_b/PORTABLE_GALLERY_KO.md)
- [게시 영수증 JSON](PUBLICATION_RECEIPT_KO.json)
- [독립 원문 바이트 검산](closing/independent_QA/BYTE_ROUNDTRIP.json)
- [독립 링크·credential 검산](closing/independent_QA/PORTABLE_LINKS_CREDENTIALS.json)

- [독립 원격 public/main·5개 blob 검산](closing/remote/REVIEW.json)
- [복사된 게시 원문 검산 코드](closing/tools/verify_v189_delta.py)

검증된 payload는 726개 물리 파일, 72198555바이트입니다. 신규 원문 source 기록 695개와 지정 5개 원격 blob metadata의 독립 검산을 해당 영수증에 바인딩했습니다. 원격 콘텐츠·픽셀 재다운로드는 하지 않았습니다.

검증된 payload commit X: [d1dfeed](https://github.com/CanelE452/pallet-6d-pose/commit/d1dfeed3bc4cc9454cd2ecdbf992e73c3ab88945). 이 closing 기록의 후속 commit Y는 push 뒤 로컬 영수증에만 추가하여 자기 commit 순환 참조를 피합니다.
