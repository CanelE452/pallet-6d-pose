# C_EXPOSURE 결과

상태: **FINAL**. EXECUTION=PARTIAL_BUDGET; POSE_OUTCOME=MIXED; REPEAT=NOT_RUN.

사전 SPEC과 실제 RESULTS는 그대로 보존했다. 아래 표는 완료 결과의 JSON을 직접 집계한다. 음의 Δ가 개선이다. 기존 REF·R0·matched RAW 대비를 모두 공개한다.

| 조건 / 모델 | valid / N | T cm median / P90 | R deg median / P90 | yaw deg median / P90 | axis 오류 | pose 실패 |
| --- | --- | --- | --- | --- | --- | --- |
| C_EXPOSURE/PLASTIC/S42/R0 | 99/99 | 12.147922 / 120.471370 | 6.037944 / 89.218262 | 5.563436 / 88.888879 | 45 | 0 |
| C_EXPOSURE/PLASTIC/S42/OLD_REF | 99/99 | 12.423010 / 127.143068 | 8.694696 / 89.197272 | 7.694101 / 88.913988 | 46 | 0 |
| C_EXPOSURE/PLASTIC/S42/NEW_RAW | 99/99 | 12.322978 / 118.880869 | 8.283290 / 89.297929 | 7.235244 / 89.188717 | 46 | 0 |
| C_EXPOSURE/PLASTIC/S42/NEW_REF | 99/99 | 12.387281 / 126.911220 | 8.682595 / 89.266298 | 7.688518 / 88.999830 | 46 | 0 |

| 조건 / after − before | common / N | Δ T 중앙값 cm | Δ R 중앙값 deg | median frame ΔT cm | median frame ΔR deg | valid→fail | fail→valid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C_EXPOSURE/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-OLD_REF | 99/99 | -0.035730 | -0.012101 | -0.055007 | +0.019063 | 0 | 0 |
| C_EXPOSURE/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-R0 | 99/99 | +0.239359 | +2.644651 | -0.195561 | -0.238403 | 0 | 0 |
| C_EXPOSURE/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-NEW_RAW | 99/99 | +0.064303 | +0.399305 | -0.052887 | -0.065493 | 0 | 0 |
| C_EXPOSURE/PLASTIC/S42/ALL/NEW_REF-minus-OLD_REF | 128/128 | +0.303825 | -0.021899 | -0.007747 | +0.012548 | 0 | 0 |
| C_EXPOSURE/PLASTIC/S42/ALL/NEW_REF-minus-R0 | 128/128 | +0.136649 | +0.214210 | -0.290628 | -0.116590 | 0 | 0 |
| C_EXPOSURE/PLASTIC/S42/ALL/NEW_REF-minus-NEW_RAW | 128/128 | +0.358897 | -0.199355 | -0.163439 | -0.038420 | 0 | 0 |

| 조건 / 대비 | T_↓__R_↓ | T_↓__R_= | T_↓__R_↑ | T_=__R_↓ | T_=__R_= | T_=__R_↑ | T_↑__R_↓ | T_↑__R_= | T_↑__R_↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_EXPOSURE/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-OLD_REF | 18 | 0 | 39 | 0 | 0 | 0 | 13 | 0 | 29 |
| C_EXPOSURE/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-R0 | 43 | 0 | 13 | 0 | 0 | 0 | 23 | 0 | 20 |
| C_EXPOSURE/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-NEW_RAW | 34 | 0 | 17 | 0 | 0 | 0 | 27 | 0 | 21 |
| C_EXPOSURE/PLASTIC/S42/ALL/NEW_REF-minus-OLD_REF | 23 | 0 | 42 | 0 | 0 | 0 | 25 | 0 | 38 |
| C_EXPOSURE/PLASTIC/S42/ALL/NEW_REF-minus-R0 | 52 | 0 | 28 | 0 | 0 | 0 | 26 | 0 | 22 |
| C_EXPOSURE/PLASTIC/S42/ALL/NEW_REF-minus-NEW_RAW | 46 | 0 | 30 | 0 | 0 | 0 | 31 | 0 | 21 |

| 조건 / after − before | common / N | Δ T 중앙값 cm | Δ R 중앙값 deg | median frame ΔT cm | median frame ΔR deg | valid→fail | fail→valid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C-minus-A/NEW_RAW/PRIMARY_MODERATE_PLUS_SEVERE | 99/99 | -0.395690 | -0.723098 | +0.004938 | +0.004192 | 0 | 0 |
| C-minus-A/NEW_REF/PRIMARY_MODERATE_PLUS_SEVERE | 99/99 | +0.131561 | -0.047662 | -0.007634 | +0.007467 | 0 | 0 |

기존 REF 대비 작은 joint sign만 보였으나 matched RAW와 R0보다 양 축에서 나쁘다. C−A 역시 tradeoff이다. 따라서 반복 대상으로만 잠갔고, 같은 추가 seed의 원 recipe 대조와 Wood 적용성 결과는 상위 최종 보고서에서 확인한다.

[전체 보고서](../../REPORT_KO.md), [전체 난도·recording 표](../../FINAL_TABLES.md), [진단 한계](../../LOSS_SIGNAL_AUDIT.md).
