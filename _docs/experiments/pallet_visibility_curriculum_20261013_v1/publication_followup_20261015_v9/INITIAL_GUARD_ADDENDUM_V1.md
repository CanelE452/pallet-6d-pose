# 초기 자세의 guard와 선택 비용 추가 확인

[V9 평가 보고서](REPORT_KO_FINAL_V1.md)를 작성한 뒤 고정3사례의 저장 초기 자세와 최종 자세를 독립식으로 정지 재계산했다. 새 PnP·후보 생성·최적화·영상 질의·GT/참조 재개방은0이다. 결과를 방법 선택이나 파라미터 조정에 넣지 않았다.

세 사례의 ORIGINAL own coarse는 ALL/SOURCE 모두 원 hardguard와 실제 지지 guard를 통과했다. noapril SOURCE는 초기 N+O+S=.806598보다 선택된 SWAPPED의.137011이 낮았고, C 포함도 초기.900330보다 선택.455271이 낮았다. 초기값의 guard 탈락이 아니라 현재 목적함수의 모드 선호임을 확인했다.

031426은 새 PROJECTED_NATIVE_EDGE 요인0개이지만, full B2 관측과 기존 B0/B4 부분 법선·T4 접선 정보는 남는다. 새 선이0개라는 말을 모든 영상 정보가 없다는 뜻으로 해석하면 안 된다.

| 031426의 SOURCE 후보 | N+O+S | 가중 전경 포함 비용(C) | 합 |
|---|---:|---:|---:|
| 보존된 native 최종, SWAPPED (13.03cm) | .932913 | 4.857064 | 5.789976 |
| C가 선택한 ORIGINAL (191.31cm) | 1.598098 | 2.500146 | 4.098244 |

포함 비용은2.356917 감소했고 native+기존 O/S 합은.665185 증가했다. 그 결과3D 오차가 더 큰 후보가 총비용에서1.691732 낮아 선택됐다. 표의 cm는 이미 봉인된 점수의 사후 설명이며 선택 비용에는 넣지 않았다.

[전체24정지값·독립 검산과 한계](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_initial_guard_cost_probe_v4_32_v1/SUMMARY_KO_V1.md)는 저장된 selected 총비용12개의 최대 차이3.747e-15를 확인했다. 전체seed/start/비선택 자세를 재생성하거나 모든 모드의 물리적 정체를 인증한 것은 아니다.
