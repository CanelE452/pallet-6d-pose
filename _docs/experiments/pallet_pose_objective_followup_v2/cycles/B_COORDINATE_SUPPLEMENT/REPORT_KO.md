# B_COORDINATE_SUPPLEMENT 결과

상태: **FINAL**. EXECUTION=PARTIAL_BUDGET; POSE_OUTCOME=MIXED; REPEAT=NOT_RUN.

사전 SPEC과 실제 RESULTS는 그대로 보존했다. 아래 표는 완료 결과의 JSON을 직접 집계한다. 음의 Δ가 개선이다. 기존 REF·R0·matched RAW 대비를 모두 공개한다.

| 조건 / 모델 | valid / N | T cm median / P90 | R deg median / P90 | yaw deg median / P90 | axis 오류 | pose 실패 |
| --- | --- | --- | --- | --- | --- | --- |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/R0 | 99/99 | 12.147922 / 120.471370 | 6.037944 / 89.218262 | 5.563436 / 88.888879 | 45 | 0 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/OLD_REF | 99/99 | 12.423010 / 127.143068 | 8.694696 / 89.197272 | 7.694101 / 88.913988 | 46 | 0 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/NEW_RAW | 99/99 | 12.467667 / 119.254773 | 9.061617 / 89.312663 | 8.984703 / 89.153948 | 47 | 0 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/NEW_REF | 99/99 | 12.430253 / 127.116991 | 8.704029 / 89.203264 | 7.707125 / 88.925986 | 46 | 0 |

| 조건 / after − before | common / N | Δ T 중앙값 cm | Δ R 중앙값 deg | median frame ΔT cm | median frame ΔR deg | valid→fail | fail→valid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-OLD_REF | 99/99 | +0.007242 | +0.009333 | +0.004626 | +0.003940 | 0 | 0 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-R0 | 99/99 | +0.282331 | +2.666084 | -0.193857 | -0.231524 | 0 | 0 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-NEW_RAW | 99/99 | -0.037415 | -0.357588 | -0.057479 | -0.082482 | 0 | 0 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/ALL/NEW_REF-minus-OLD_REF | 128/128 | +0.003802 | -0.001950 | +0.004267 | +0.002390 | 0 | 0 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/ALL/NEW_REF-minus-R0 | 128/128 | -0.163374 | +0.234158 | -0.394640 | -0.127288 | 0 | 0 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/ALL/NEW_REF-minus-NEW_RAW | 128/128 | -0.045796 | -0.177947 | -0.228022 | -0.046412 | 0 | 0 |

| 조건 / 대비 | T_↓__R_↓ | T_↓__R_= | T_↓__R_↑ | T_=__R_↓ | T_=__R_= | T_=__R_↑ | T_↑__R_↓ | T_↑__R_= | T_↑__R_↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-OLD_REF | 19 | 0 | 22 | 0 | 0 | 0 | 16 | 0 | 42 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-R0 | 42 | 0 | 12 | 0 | 0 | 0 | 24 | 0 | 21 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/PRIMARY_MODERATE_PLUS_SEVERE/NEW_REF-minus-NEW_RAW | 32 | 0 | 19 | 0 | 0 | 0 | 29 | 0 | 19 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/ALL/NEW_REF-minus-OLD_REF | 22 | 0 | 32 | 0 | 0 | 0 | 26 | 0 | 48 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/ALL/NEW_REF-minus-R0 | 51 | 0 | 27 | 0 | 0 | 0 | 27 | 0 | 23 |
| B_COORDINATE_SUPPLEMENT/PLASTIC/S42/ALL/NEW_REF-minus-NEW_RAW | 40 | 0 | 35 | 0 | 0 | 0 | 32 | 0 | 21 |

기존 REF 대비 T와 R 모두 NO_GAIN이다. 유효 좌표 신호의 감쇠를 보완한 이 구현은 이 고정 조건에서 유리하지 않았다. 전체 gradient 소실이나 pseudo target의 정답성은 입증하지 못했으며, 보완 항의 gradient 크기 혼입도 남는다. 결합 후보로 승격하지 않는다.

[전체 보고서](../../REPORT_KO.md), [전체 난도·recording 표](../../FINAL_TABLES.md), [진단 한계](../../LOSS_SIGNAL_AUDIT.md).
