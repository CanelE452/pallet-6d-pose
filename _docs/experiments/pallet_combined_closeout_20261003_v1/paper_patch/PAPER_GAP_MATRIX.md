# 원고 셀 연결과 남은 x

| 원고 위치 | 결과 | 상태 | 원시 근거 | 분모·단위 |
|---|---|---|---|---|
| `tab:composition` | 직사각형 151/87/81 | INSERTED_IN_COPY | `LABEL_PROVENANCE_AUDIT.json` | 영상 319장 |
| `tab:occlusion_results` | YOLO Base/N3 전체 가림 집단 | INSERTED_IN_COPY | `STATIC_REAGGREGATION.json` | 참조 2,499, 유효 2,445 코너; px/cm/deg |
| `sup:occlusion_results` | Base/P/N2/N3 가림 집단 | INSERTED_IN_COPY | 같은 JSON | N3·N2·P 각 3 seed 통계 평균 |
| `tab:case` 영상/프레임 | 4 / 8,910 | INSERTED_IN_COPY | `LIFTER_CONTINUITY_SUMMARY.json` | 전체 저장 프레임 |
| `tab:case` 자세 산출률 | 98.451 / 98.451% | INSERTED_IN_COPY | 같은 JSON | fresh 8,772 / 8,910; held 0 |
| `tab:case` 최장 완료 결측 | 2.535 / 2.535 s | INSERTED_IN_COPY | 같은 JSON | 첫 no-pose부터 다음 fresh까지 센서 시간 |
| `tab:case` 인접 변화 | x/z/yaw 중앙값·P90 | INSERTED_IN_COPY | 같은 JSON | 양쪽 유효 같은 8,737 인접쌍; 정확도 아님 |
| `tab:case` 가시 코너 오차 | x | WAITING_HUMAN_CORNERS | 제출된 review-120 없음 | 승인 직접 가시 코너 전체 분모 필요 |
| `tab:case` 정지 구간 변동 | x | WAITING_HUMAN_CORNERS | 확인된 정지 구간 없음 | 이동 중 인접 변화를 대신 쓰지 않음 |
| `tab:case` 물리 위치·방향 오차 | x | BLOCKED_REFERENCE | 독립 동기 참조 없음 | 운용 CSV·같은 PnP는 정답 아님 |
| 정사각형 6D | x | BLOCKED_REFERENCE | 독립 6D 참조 없음 | 600/602 코너 2D와 분리 |
| 정사각형 가림 집단 | x | WAITING_STATIC_REVIEW | 119장 UI 준비 | 사람 frame severity 필요 |
| 전체 코너 가시성 분석 | 부분본만 유지 | WAITING_STATIC_REVIEW | 기존 66 direct + 5 external | 나머지 직사각형 2,428점과 정사각형 602점 대기 |

`x`는 측정되지 않았거나 계약이 성립하지 않은 값이고, `NA`는 정의되지 않는 값이다. 실제 0은 0으로 적는다.
