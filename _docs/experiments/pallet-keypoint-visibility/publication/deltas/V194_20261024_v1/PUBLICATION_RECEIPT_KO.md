# V194 main 게시 확인

공개 source의 그룹 분할과 두 조건의 고정8000회 학습, source TRAIN/HOLDOUT 결과·표·코드·UNKNOWN·실패·독립 검산을 보존한 게시 기록입니다. source는 합성과 공개 실사 약한 라벨을 포함하며 TRAIN327은 apparent 응답, direct HOLDOUT45는 합성뿐이고 실사 끝점 holdout은0입니다. 지정한 자체 final head2개와 source 저장 예측24개만 포함하고 raw RGB/SAM/feature/cache 배열 및 외부 가중치는 제외 기록으로 보존합니다. 자연 실사 성능 개선·새 blind 일반화·모델 채택·전체 목표 해결을 뜻하지 않으며 V195 실사 benchmark는 별도입니다.

- [V194까지의 전체 실험 인덱스](../../../EXPERIMENT_INDEX_THROUGH_V194_KO.md)
- [V194 source 보고서](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_source_group_split_direct_end_training_20261024_v194/PORTABLE_REPORT_KO.md)
- [독립 source 학습·저장 결과 감사](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_source_group_split_direct_end_training_20261024_v194/independent_grouped_source_fit_review_a/actual_review_v1/REVIEW.json)
- [게시 영수증 JSON](PUBLICATION_RECEIPT_KO.json)
- [독립 원문 바이트 검산](closing/independent_QA/BYTE_ROUNDTRIP.json)
- [독립 링크·credential 검산](closing/independent_QA/PORTABLE_LINKS_CREDENTIALS.json)

- [독립 원격 public/main·5개 blob 검산](closing/remote/REVIEW.json)
- [복사된 게시 원문 검산 코드](closing/tools/verify_v194_delta.py)

검증된 payload는 207개 물리 파일, 37228081바이트입니다. 신규 원문 source 기록 157개와 지정 5개 원격 blob metadata의 독립 검산을 해당 영수증에 바인딩했습니다. 원격 콘텐츠·픽셀 재다운로드는 하지 않았습니다.

검증된 payload commit X: [1438480](https://github.com/CanelE452/pallet-6d-pose/commit/143848012f7403f320896d283e7d629074c3eeb7). 이 closing 기록의 후속 commit Y는 push 뒤 로컬 영수증에만 추가하여 자기 commit 순환 참조를 피합니다.
