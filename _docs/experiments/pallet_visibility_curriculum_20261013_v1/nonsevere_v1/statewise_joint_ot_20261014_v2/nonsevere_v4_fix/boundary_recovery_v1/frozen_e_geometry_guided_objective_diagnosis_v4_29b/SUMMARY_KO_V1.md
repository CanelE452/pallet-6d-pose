# V29 TEST49 저장 자세 비용 진단

새 관측·최적화·정답 decode 없이 저장된 e와 선택 자세를 같은 frozen 목적함수로 한 번 재평가했습니다. 원 e의 WD 대응을 본문에 보존했고, 물리 상자가 같은 다른 WD 운반은 별도 부록입니다.

| 방법 | TEST 악화 | 통과한 e보다 나쁜 선택 자세의 비용이 낮음 | e guard 제외 | fallback |
|---|---:|---:|---:|---:|
| GEOMETRY_EDGE_ALL_BI_NATIVE | 18 | 9 | 7 | 2 |
| GEOMETRY_EDGE_ALL_BI_CONTAINMENT | 18 | 7 | 9 | 2 |
| GEOMETRY_EDGE_SOURCE_NATIVE | 10 | 5 | 5 | 0 |
| GEOMETRY_EDGE_SOURCE_CONTAINMENT | 10 | 6 | 4 | 0 |

악화는 저장 e의 translation≤5cm에서 선택 자세가 >5cm인 경우입니다. 실제 선 지지 guard가 e를 제외한 모든 기록은 `OWNER_COMPLETION_V1.json`에 원 finite/directed slack과 함께 있습니다. 제외 guard는 actual-edge support audit이며 다른 guard 실패와 구분했습니다.

전체 49행에서 ALL 두 arm은 비용선호 37 / e 제외 9 / fallback 3, SOURCE 두 arm은 43 / 5 / 1입니다. 두 WD의 유효한 e 물리 운반 대안 중 더 낮은 비용은 없었고, 저장 선택 비용 검산 최대차는 7.11e−15였습니다.

두 경로 모두 관측되며, 전역최적·WD/선의 인과 증명이나 모든 실패가 최적화와 무관하다는 결론은 아닙니다. V30의 법선 거리 association은 유한 along-line 대응 보장이 아니므로 수정 완료로 간주하지 않습니다. 기본 e와 모든 원 산출물은 유지했습니다.
