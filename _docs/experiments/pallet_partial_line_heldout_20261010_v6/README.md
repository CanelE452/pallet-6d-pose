# v6: 양끝점 heldout 선 검증의 실제 결과

**최상위 목표 미달.** 쉬움153·중간92, 전체245장의 새 실제 평가에서 주 방법은10.52131cm/13.00650°, 고정 N3→SubPix는9.75455cm/10.91484°다.241NEW·4N3기본반환·0완전실패이며 모든 출력을 운용 통계에 남겼다. 양끝점과 H를 제외한 검증은 구현·검산했지만, v5 C2 대비 회전도 악화됐다.

- [상세 결과·역할/감독/선 gate 구분·전체 분산·8개 실제 그림](RESULT_KO.md)
- [공개 원행 복원과 재현](REPRODUCE.md)
- [실제 실행량·실패 CLI·primitive·보호 ledger](BUILD_LEDGER.json)
- [사전 고정 평가 계약](EVALUATION_CONTRACT_KO.md), [정적 재사용 감사](STATIC_REUSE_AUDIT_KO.md), [formal PROTOCOL](PROTOCOL.json)
- [245장 수치](METRICS.json), [행별 CSV](PER_FRAME.csv), [독립189CI 검산](VERIFICATION.json)
- [3500726개 저장 기하 계약](VALIDATION_CHECKS.json), [H 재투영 검산](REPROJECTION_CHECKS.json)
- [후행 선 gate 품질](GATE_QUALITY_CHECKS.json), [실제600 전체 경로 시간](RUNTIME.json)
- [공개 byte archive](ARCHIVE_MANIFEST.json), [새 bundle 복원 후 검토](PUBLIC_ARCHIVE_REVIEW.json)
- [전체 분포 그림](figures/01_all_operational.png), [동일 frame 비교 그림](figures/02_same_frame_pairs.png), [그림 바인딩](FIGURE_BINDINGS.json)

core protocol은 새 실사 GT 전에 고정했다. gate 품질 감사의 별도 protocol은 점수화·통계 뒤 own 산술 전에 고정한 후행 감사이며 정확도 사전등록이 아니다. 새 학습·RGB·annotation·threshold tuning은0이다. 공개 수치 검산·byte 복원은 물리적 소유권이나 성능 성공을 인증하지 않는다. 이전 세 head의 수정 학습과245장66-way 평가가 완료된 사실과 현재 ROLE-only decoder의 필요성 검증 공백을 구분한다. 별도 v7 개발/실행은 이 v6 결과에 포함하지 않는다.
