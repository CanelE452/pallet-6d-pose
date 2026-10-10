# V13 finite-edge consensus 실험

**전체245장에서 N3보다 위치·회전 평균이 나쁘며 목표는 미해결이다.** [결과·모든810moments/81CI/56timing values](RESULT_KO.md), [실행 전 방법](METHOD_CONTRACT_KO.md), [동결 PROTOCOL](PROTOCOL.json), [실행량/보존 실패](BUILD_LEDGER.json)를 먼저 읽는다.

[METRICS.json](METRICS.json)/[CSV](METRICS.csv), [245 기하](GEOMETRY_SEALED.jsonl.gz)/[245 ledger](CONTROL_LEDGERS.jsonl.gz)/[245 scores](PREDICTIONS.jsonl.gz)/[posthoc](POSTHOC_ROWS.jsonl.gz), [독립 통계 검산](REVIEW_VERIFICATION.json), [원 FAIL74](VALIDATION_CHECKS.json)/[별도 검산 PASS](REVIEW_VALIDATION_CHECKS.json), [fresh150 runtime](RUNTIME_RECEIPT.json)/[원행](RUNTIME_ROWS.jsonl.gz)/[독립 검산](RUNTIME_CHECKS.json)을 함께 확인한다.

이 보고서는 known DEV GEOMETRIC_PROXY 결과이며 독립 physical pose/unseen test를 증명하지 않는다. [8이미지·24패널](RESULT_KO.md#이미지공개-재현게시-상태)은 원 renderer 끝검사 FAIL을 보존한 [별도 artifact PASS](SUPPLEMENTAL_FIGURE_CHECKS.json)와 [실제8이미지 열람](ROOT_VISUAL_REVIEW.json)을 갖춘다. [245 metadata 해석](POSTHOC_METADATA_CHECKS.json)은 generic marker와 실제 hinge/point-only/fullpool 차이를 분리했다. 공개 fresh restore·최종 게시 SHA는 [PUBLIC_REVIEW_KO](PUBLIC_REVIEW_KO.md)에 별도로 기록할 예정이며 아직 대기다. [V9](../pallet_sparse_local_line_20261010_v9/RESULT_KO.md)의 구실험319/현재245와 수정 전 감독 실패를 구분한다. main 자동병합/force push 없이 전용 research 브랜치에 정상 게시할 예정이다.
