# V240 저장 SOURCE 오류 조인

기존 YCB known 1,521행을 새 두 CSV와 row_id **및** sample_index로 전수 연결했습니다. 역할·associated-r D/E와 기존 object×aux 31개 구성표가 모두 동일합니다. FIT/DEV known 분모는 1,124/397이며 DEV E는 57행입니다.

CONTROL E 정답 0/57 → TREAT 1/57입니다. 회복 1, 손상 0, 계속 미검출 56, 둘 다 정답 0입니다.

|scene|object|logical aux|q–r 범주|DEV E|회복|손상|계속 미검출|FIT 같은 object×aux D/E|
|---|---|---|---|---:|---:|---:|---:|---|
|6|7|3|GT3|13|0|0|13|79/0|
|6|7|3|LE3|16|0|0|16|79/0|
|6|7|7|GT3|13|1|0|12|20/0|
|6|7|7|LE3|11|0|0|11|20/0|
|7|2|2|GT3|1|0|0|1|46/0|
|7|2|2|LE3|3|0|0|3|46/0|

기존과 같이 53/57은 scene6/object7이며, 같은 object×aux의 FIT E 지원은 57행 모두 0입니다. saved frame 구분은 57개이나 독립된 자연가림 사건 수로 인증하지 않습니다. 이 구성이 오류의 원인이라는 결론도 내리지 않습니다.

q–r 분류는 기존 진단표의 저장 거리입니다. nominal/associated reference와 query의 거리이지 새 물리점 gold나 현재 실사 원인 분석이 아닙니다. YCB native는 UNKNOWN·aux는 논리 슬롯이고, SOURCE 내부 DEV는 이미 노출·재사용됐습니다. 모델/data/architecture 선택, 추가 학습, label 수정은 하지 않았습니다.

[57행 조인표](YCB_DEV_E57_ERROR_JOIN.csv) · [그룹표](YCB_DEV_E_GROUP_COUNTS.csv) · [요약·한계](SUMMARY.json)
