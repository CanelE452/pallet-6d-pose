# V195 main 게시 확인

동결된8개 모델의 동일 current1784와 primary1237(D1147/E90), 원입력U45 및 별도 public360 assistantweak 비교·고정68case 그림·표·코드·실패·독립 검산을 보존한 평가 전용 게시 기록입니다. C56/C57 source 학습과 자체 가중치는 V194에 별도로 보존되며 이전6개 모델은 저장 출력만 사용합니다. 이 단계는 자체 가중치0개, 새 저장 예측4개만 수치 출력으로 포함하고 raw 입력·RGB/SAM/cache·외부 가중치는 제외합니다. 반복 노출된 고정 평가이며 새 blind 일반화·인과 결론·모델 채택·전체 목표 해결을 뜻하지 않습니다.

- [V195까지의 전체 실험 인덱스](../../../EXPERIMENT_INDEX_THROUGH_V195_KO.md)
- [V195 source 보고서](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_grouped_direct_end_heads_frozen_real_benchmark_20261024_v195/PORTABLE_REPORT_KO.md)
- [독립 동결 평가 저장 결과 감사](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_grouped_direct_end_heads_frozen_real_benchmark_20261024_v195/independent_saved_result_review_preparation_a/actual_review_v1/REVIEW.json)
- [게시 영수증 JSON](PUBLICATION_RECEIPT_KO.json)
- [독립 원문 바이트 검산](closing/independent_QA/BYTE_ROUNDTRIP.json)
- [독립 링크·credential 검산](closing/independent_QA/PORTABLE_LINKS_CREDENTIALS.json)

- [독립 원격 public/main·5개 blob 검산](closing/remote/REVIEW.json)
- [복사된 게시 원문 검산 코드](closing/tools/verify_v195_delta.py)

검증된 payload는 447개 물리 파일, 30196550바이트입니다. 신규 원문 source 기록 420개와 지정 5개 원격 blob metadata의 독립 검산을 해당 영수증에 바인딩했습니다. 원격 콘텐츠·픽셀 재다운로드는 하지 않았습니다.

검증된 payload commit X: [33a341a](https://github.com/CanelE452/pallet-6d-pose/commit/33a341a02ea68481be3d920b4eaa367630eac90a). 이 closing 기록의 후속 commit Y는 push 뒤 로컬 영수증에만 추가하여 자기 commit 순환 참조를 피합니다.
