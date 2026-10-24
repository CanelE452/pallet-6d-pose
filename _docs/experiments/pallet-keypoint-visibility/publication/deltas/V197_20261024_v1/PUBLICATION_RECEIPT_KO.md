# V197 main 게시 확인

SOURCE TRAIN의 고정12 triplet(합성8·공개실사4), 모든 native의288행을 보존한 입력·목표 대응 판독 기록입니다. weak D24/E12와 UNKNOWN252, 입력U24를 구분합니다. 전체109 TRAIN triplet 개별 전수뷰나 물리 native gold·인식 충분성·실사 전이·새 blind 검증을 뜻하지 않습니다. 실제 고정뷰 범위는 독립 QA 원문에만 따릅니다. 새 모델·NPZ·fit·forward·SAM·PnP·라벨·모델/데이터/문턱 선택은0이며 모델 채택·전체 목표 해결을 주장하지 않습니다.

- [V197까지의 전체 실험 인덱스](../../../EXPERIMENT_INDEX_THROUGH_V197_KO.md)
- [V197 SOURCE TRAIN 입력·목표 대응 판독](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_train_only_target_contract_readout_20261024_v197/PORTABLE_REPORT_KO.md)
- [독립 저장 source viewer 검산](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_train_only_target_contract_readout_20261024_v197/independent_saved_train_viewer_review_b/REVIEW.json)
- [독립 최종 보고서 검토](closing/independent_QA/FINAL_REPORT_QA.json)
- [게시 영수증 JSON](PUBLICATION_RECEIPT_KO.json)
- [독립 원문 바이트 검산](closing/independent_QA/BYTE_ROUNDTRIP.json)
- [독립 링크·credential 검산](closing/independent_QA/PORTABLE_LINKS_CREDENTIALS.json)

- [독립 원격 public/main·5개 blob 검산](closing/remote/REVIEW.json)
- [복사된 게시 원문 검산 코드](closing/tools/verify_v197_delta.py)

검증된 payload는 758개 물리 파일, 35767308바이트입니다. 신규 원문 source 기록 729개와 지정5개 원격 blob metadata 검산을 영수증에 바인딩했습니다. 원격 콘텐츠·픽셀은 다운로드하지 않았습니다.

검증된 payload commit X: [3cf4d60](https://github.com/CanelE452/pallet-6d-pose/commit/3cf4d6099bf3761268af0b8f96076eacfc39e947). 후속 closing commit Y는 push 뒤 로컬 영수증에만 기록하여 자기 commit 순환 참조를 피합니다.
