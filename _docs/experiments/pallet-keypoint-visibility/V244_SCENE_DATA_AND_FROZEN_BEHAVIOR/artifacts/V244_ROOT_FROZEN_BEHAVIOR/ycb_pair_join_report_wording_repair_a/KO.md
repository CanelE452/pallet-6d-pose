# 기존 YCB SOURCE FIT CF 행동·학습 draw 진단

기존 저장 scalar 자료를 row_id와 원 sample_index로 결합했습니다. 새 이미지·배열·모델·GT 읽기, forward, 학습은 없습니다.

| 그룹 | 성공쌍 | COVER E | SHAM D | 두 역할 모두 정답 | COVER logit > SHAM |
|---|---:|---:|---:|---:|---:|
| 전체 YCB | 830 | 789 | 785 | 751 | 817 |
| 고정 주목 조합: obj7 aux3/7 또는 obj2 aux2 | 143 | 136 | 137 | 130 | 143 |
| 나머지 YCB | 687 | 653 | 648 | 621 | 674 |

주목 조합은 전체145시도 중 성공143, 실패 U 2입니다. 성공143쌍은 모두 두 역할에서 실제 학습 draw가 있었습니다. COVER 650회 중 정답 625회, SHAM 694회 중 정답 662회입니다. 이는 반복 draw 분모이며 고유 pair 수와 다릅니다.

전체 YCB 성공 pool830쌍 중 COVER는823개 고유쌍, SHAM은815개를 실제 사용했습니다. 둘 중 적어도 한 역할 사용은830쌍, 양 역할 사용은808쌍입니다. SYN 성공 pool4537쌍에서는 COVER 2654, SHAM 2630개만 실제 draw됐습니다. 따라서 pool의 모든 역할 이미지가 실제 gradient로 사용됐다고 해석하지 않습니다.

이 결과는 기존의 합성 SOURCE FIT pair에 대한 고정 head 기능과 draw 노출 진단입니다. 자연 SOURCE DEV나 current 성능, 실패 원인·인과관계의 증명이 아닙니다. 기존 metadata의 과거 모델 점수·GT reference·owner 값은 이번 join에 사용하지 않았습니다. YCB native는 UNKNOWN이며 associated-r 약한 교사와 literal q 중심 상태를 구분합니다.

[저장 SUMMARY](../ycb_pair_join_v1/SUMMARY.json) · [group metrics](../ycb_pair_join_v1/GROUP_METRICS.csv) · [YCB 전체841행](../ycb_pair_join_v1/ALL841_YCB_PAIR_ROWS.csv) · [5406 실제 draw 노출](../ycb_pair_join_v1/ALL5406_ACTUAL_DRAW_EXPOSURE.csv)

원 실행의 혼합 언어 보고서는 보존했습니다. 이 문서는 같은 저장 summary를 한국어로 표시한 교정본이며 숫자·원 CSV·원 COMPLETE·원 join code를 변경하지 않았습니다.
