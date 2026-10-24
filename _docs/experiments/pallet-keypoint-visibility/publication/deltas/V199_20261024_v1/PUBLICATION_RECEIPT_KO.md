# V199 source 학습 기록 main 게시 확인

V199 완료 source-only matched RAW nativezero8/onehot8 대조 학습 및 저장 source integrity 검토입니다. 각116407 parameters/8000 fixed steps이며 source TRAIN apparent와 설계가 이미 알려진 약한 합성 source HOLDOUT 결과를 분리합니다. native-ID 개선이나 실사 일반화·native physicalgold·freshblind·선택·채택·전체 목표 해결을 입증한 단계가 아닙니다. 원본 publicreal SOURCE_DEVELOPMENT도 기존 학습풀에 포함되므로 synthetic-only training이라고 부르지 않습니다. Source freeze의 신규 실사 결과 미열람은 새 V200 metric에 한정되며 과거 평가 노출을 부정하지 않습니다. 새 owned states2와 source outputs24는 게시 원문바이트로만 취급하며 이 도구는 과학 수치를 재평가하지 않습니다.

- [V199까지의 전체 실험 인덱스](../../../EXPERIMENT_INDEX_THROUGH_V199_KO.md)
- [V199 source-only 대조 학습·방법·한계](../../../archive/task-2/pallet-keypoint-visibility/query_visibility_native_conditioned_grouped_source_training_20261024_v199/PORTABLE_REPORT_KO.md)
- [독립 source-fit 무결성 검토](closing/independent_QA/SOURCE_FIT_INTEGRITY_QA.json)
- [독립 최종 source 보고서 검토](closing/independent_QA/FINAL_SOURCE_REPORT_QA.json)
- [source 도표 실제 시각 검토](closing/independent_QA/ACTUAL_SOURCE_CHART_VISUAL_QA.json)
- [게시 영수증 JSON](PUBLICATION_RECEIPT_KO.json)
- [독립 원문 바이트 검산](closing/independent_QA/BYTE_ROUNDTRIP.json)
- [독립 링크·credential 검산](closing/independent_QA/PORTABLE_LINKS_CREDENTIALS.json)

- [독립 원격 public/main·5개 blob 검산](closing/remote/REVIEW.json)
- [복사된 게시 원문 검산 코드](closing/tools/verify_v199_delta.py)

검증된 payload는 142개 물리 파일, 32655131바이트입니다. 신규 원문 source 기록 108개와 지정5개 원격 blob metadata 검산을 영수증에 바인딩했습니다. 원격 콘텐츠·픽셀·모델·배열은 다운로드하거나 해석하지 않았습니다.

검증된 payload commit X: [84efbed](https://github.com/CanelE452/pallet-6d-pose/commit/84efbed493bdb8a2f4abdf7c7691809647521aa7). 후속 closing commit Y는 push 뒤 로컬 영수증에만 기록하여 자기 commit 순환 참조를 피합니다.
