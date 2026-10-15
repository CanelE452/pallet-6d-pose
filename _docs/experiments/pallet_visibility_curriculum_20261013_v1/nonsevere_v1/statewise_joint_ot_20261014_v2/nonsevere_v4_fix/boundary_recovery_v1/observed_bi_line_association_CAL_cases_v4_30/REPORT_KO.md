# v30 원 B/I 점과 실제 선 연결: 고정 CAL3 재순위

새 이미지 측정/키포인트/PnP/정답 읽기 없이 원 cached 선을 재순위했다. 모든 raw chain은 원 archive에 보존한다. 기존43px normal reach를 무한 TLS 선과 named-edge의 모든 eligible B/I q 사이에 적용했다. 새 선택 집단이므로 기존 v29 noise를 검증된 모델로 쓰지 않는다. 별도 CAL46/noise/full238 평가 이전에는 새 방법 채택이나 정확도 개선을 주장하지 않는다.

| CAL 사례 | 원/새 후보 수 | gate 제외 | 원/새 선택 선 수 | rank 시간(s) |
|---|---:|---:|---:|---:|
| wood_184309:000925 | 501/177 | 531 | 6/6 | 7.925 |
| wood_184309:000804 | 942/615 | 597 | 7/9 | 61.176 |
| eval_cad:1778653017736058368 | 126/111 | 17 | 5/4 | 1.824 |

## 기존 큰 잔차 native 변의 고정 CAL 투영 대조

원래 봉인된 top3 진단 projection만 재사용했다. 재순위에는 기준이 들어가지 않는다. 다른 새 선택 변은 기준값을 만들지 않고 미제공으로 남긴다. 같은 변이 선택되지 않으면 정밀 대체 성공이 아닌 미선택이다.

- wood_184309:000925: 원 [{'line_name': '0-4/231:4', 'reference': 'same saved fixed CAL diagnostic edge, not a measurement/guide/ranking term', 'native_edge': [4, 0], 'signed_normal_residual_px': -492.6951876126087, 'angle_deg': 68.21227089006116}] → 새 [{'line_name': '0-4/62:6', 'reference': 'same saved fixed CAL diagnostic edge, not a measurement/guide/ranking term', 'native_edge': [4, 0], 'signed_normal_residual_px': -0.5115442141660175, 'angle_deg': 0.2080121433276003}]
- wood_184309:000804: 원 [{'line_name': '1-2/124:0', 'reference': 'same saved fixed CAL diagnostic edge, not a measurement/guide/ranking term', 'native_edge': [1, 2], 'signed_normal_residual_px': 401.92384937138945, 'angle_deg': 47.0669368117675}] → 새 [{'line_name': '1-2/6:3', 'reference': 'same saved fixed CAL diagnostic edge, not a measurement/guide/ranking term', 'native_edge': [1, 2], 'signed_normal_residual_px': 6.065160538761326, 'angle_deg': 4.760224213494584}]
- eval_cad:1778653017736058368: 원 [{'line_name': '3-7/158:0', 'reference': 'same saved fixed CAL diagnostic edge, not a measurement/guide/ranking term', 'native_edge': [7, 3], 'signed_normal_residual_px': -348.70230871239477, 'angle_deg': 74.36894574518946}] → 새 []

모든 새 선택선의 원 B/I q 법선 거리·source/양끝 검사는 RESULTS.json에 보존한다. 후보0/선택0 abstention 및 전체 대안 수/계산 수도 숨기지 않는다. gate 통과 자체를 실제 native 대응 정확도 증명으로 표현하지 않는다.
