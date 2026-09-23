# 기존 자료 EASY→HARD 전이 결정 실험

## 결론과 데이터 계약

**COMPLETED_3_FITS** — T0/T1/T2 각320 update, 총960. 신규 촬영0, 신규 수동 어노테이션0, 신규 교사 학습0. 입력 대조: `MIXED_OR_NO_CONSISTENT_HARD_GAIN`. 좌표 대조: `MIXED_OR_NO_CONSISTENT_HARD_GAIN`.

가림 자세에서는 T0가 가장 좋았다. 중간 ADD AUC T0/T1/T2=0.45717/0.44136/0.44888, 심함0.21826/0.18517/0.17894. T2는 TRAIN manual33/36점은 맞췄지만 다른 기록의 전이 이득은 확인되지 않았다. 다음은 새 학습이 아닌 남은 TRAIN 잔차 감사 설계 하나만 제안한다.

TRAIN 기록: ['REC_001', 'REC_002']. HELDOUT 기록: ['REC_007', 'REC_021', 'REC_022', 'REC_025', 'REC_027', 'REC_041', 'REC_044']. 66장의 TRAIN 기록 pool 중 실제 학생 고유 영상은 조건별20장(C0 10 + E 또는 H 10). 과거 EVAL에서56장이 TRAIN 기록 pool로 역할 변경됐지만, 실제 추가 학습 원본은 조건별10장이다. 세 조건 합집합으로는 E/H20장이다.

새 HELDOUT128 = Clean29 / Moderate21 / Severe78. 과거 평가300과 분모·recording 계약이 다르므로 이전 표에 끼워 넣거나 직접 향상률을 주장하지 않는다. 이미 열람된 DEV이며 새로운 TEST가 아니다. 목재 개선을 주장하지 않는다.


[전체 이미지 보고서](REPORT_KO.md)
