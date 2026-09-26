# 문장별 주장–근거

| 원고 문장/주장 | 근거/범위 |
| --- | --- |
| 보정 pseudo 가시점 PCK10 개선 | PSEUDO_LABEL_QUALITY.json: FINAL_V2 fixed-ID44/66→50/66; 217 unlabeled 학습영상 자체의 GT 품질 측정은 아님 |
| 동일 조건 corrected 학생 > raw 학생 | CORE_COMPARABILITY_AUDIT.json + CORE_RESULTS.json: LR5 PCK10/공통D9 AUC; 모든LR/order도 보존 |
| R0보다 추가 가치 | CORE_RESULTS.groups.ALL: PCK10/AUC 개선; 전체 지표 우월성 아님 |
| 분산·실패·악화 | PAIRED_ANALYSIS:92/28/8,51/12; TABLE3 severe/tail, TABLE6 recording |
| 실사 감독 공개 | 기존 teacher INPUT_LOCK/TRAIN_SUPPORT:9이미지38점; 66평가점과 분리 |
| 순수 teacher-free raw 정책 비교 아님 | RAW/REF 동일 보정후 승인집합. 좌표 intervention만 격리; 선택정책효과는 측정하지 않음 |
| 독립 일반화/물리6D 검증 아님 | INDEPENDENT_CONFIRMATION_AUDIT + geometry reference 계약 |
| 신규기법 주장 아님 | PoseFix/STAC/Self6D/BOP 원문 확인; application-specific controlled evidence |
| Hard8/selector/occlusion main으로 섞지 않음 | EXPERIMENT_ROLE_MAP.md |

각 수치의 source path/hash는 generated_tables/NUMBER_PROVENANCE.json 및 MANUSCRIPT_NUMBER_PROVENANCE.json에 있다.
