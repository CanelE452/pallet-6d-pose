# V200 동결 실사 비교 main 게시 확인

V200은 source V199의 RAW nativezero8·onehot8 두 동결 상태를 원 query/crop 그대로 실사에서 비교한 단계입니다. Current humansemantic 10모델과 public assistant-weak 9모델의 저장 수치 및 같은 고정68카드를 구분해 기록합니다. 새 C58/C59가 기존 C57의 human D/E 개수보다 나은 개선을 입증하지 못했습니다. 기존 개발 노출이 반복된 비블라인드 평가이며 새로운 학습·source생성·SAM·PnP·좌표·문턱·모델/데이터/architecture 선택은 없습니다. Source 학습에 publicreal이 허용되지만 current/publicheld/COMMON/VAL을 fit하지 않습니다. SELF136/paired21, reference461/manual376 및 source direct-known native6/7 부재를 유지합니다. Humansemantic과 publicweak의 권한을 섞지 않으며 68카드 pixelparity와 실제12장 판독 범위도 구분합니다. 새 native physicalgold·실사 일반화·인과적 개선·채택·전체 목표 해결을 주장하지 않습니다.

- [V200까지의 전체 실험 인덱스](../../../EXPERIMENT_INDEX_THROUGH_V200_KO.md)
- [V200 원 query 실사 비교·방법·한계](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/PORTABLE_REPORT_KO.md)
- [독립 저장 실사 수치 검토](closing/independent_QA/SAVED_REAL_NUMERIC_QA.json)
- [고정 카드 실제 시각 검토](closing/independent_QA/ACTUAL_FIXED_CARD_VISUAL_QA.json)
- [독립 최종 실사 보고서 검토](closing/independent_QA/FINAL_REAL_REPORT_QA.json)
- [게시 영수증 JSON](PUBLICATION_RECEIPT_KO.json)
- [독립 원문 바이트 검산](closing/independent_QA/BYTE_ROUNDTRIP.json)
- [독립 링크·credential 검산](closing/independent_QA/PORTABLE_LINKS_CREDENTIALS.json)

- [독립 원격 public/main·5개 blob 검산](closing/remote/REVIEW.json)
- [복사된 게시 원문 검산 코드](closing/tools/verify_v200_delta.py)

검증된 payload는 415개 물리 파일, 32634392바이트입니다. 신규 원문 기록 383개와 지정5개 원격 blob metadata 검산을 영수증에 바인딩했습니다. 원격 콘텐츠·픽셀·모델·배열은 다운로드하거나 해석하지 않았습니다.

검증된 payload commit X: [fa718c1](https://github.com/CanelE452/pallet-6d-pose/commit/fa718c118ecf7c5e9a978bea9570d25de38aff8d). 후속 closing commit Y는 push 뒤 로컬 영수증에만 기록하여 자기 commit 순환 참조를 피합니다.
