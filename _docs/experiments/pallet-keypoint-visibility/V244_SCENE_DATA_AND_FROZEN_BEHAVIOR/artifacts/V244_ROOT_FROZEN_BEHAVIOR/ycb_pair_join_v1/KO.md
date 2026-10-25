# 기존 YCB SOURCE FIT CF 행동·draw join

새 이미지·배열·모델 실행 없이 기존 row_id/sample_index와 object/aux scalar metadata만 결합했습니다. 전체 YCB 성공830쌍: COVER E 789/830、SHAM D 785/830、両方正解 751/830。

고정 주목 그룹 obj7 aux3/7 또는 obj2 aux2은 성공143쌍입니다. 실제 학습 draw는 COVER 650회, SHAM 694회입니다. pool과 반복 draw는 별도 분모입니다.

기존 SOURCE FIT 합성 pair 기능의 진단입니다. 자연 SOURCE DEV나 current 성능, 실패 원인·인과관계의 증명이 아닙니다. YCB native는 UNKNOWN을 유지합니다.

[SUMMARY](SUMMARY.json) · [group metrics](GROUP_METRICS.csv) · [YCB rows](ALL841_YCB_PAIR_ROWS.csv) · [actual draw exposure](ALL5406_ACTUAL_DRAW_EXPOSURE.csv)
