# 유한 physical support의 계약 유효성 한계

**확인된 분류: finite physical support unguarded / contract validity limitation.** 단순히 효과가 아직 검증되지 않았다는 수준이 아니다. 실제로 선택한 pose의 scoring context에서 모든 지원 normal-line 교차가 finite projected edge 밖이었다. 원 관측·동결모델·예측·선택은 그대로 보존한다.

| 사례/arm | 실제 선택 family/h | 유한변 내부/유효 교차 | t 범위 | optimizer success / feasible | 같은family 비용: 선택 / N0 / E0 |
|---|---|---:|---|---|---|
| wood_day_01:005249 / B_FULL | ORIGINAL_WD / 0 | 0/1 | [-18.3955, -18.3955] | True / True | 2.4337e-09 / 16.2049 / 1.07386 |
| plastic_day_01:012545 / CTRL_EDGE_BOUNDARY | ORIGINAL_WD / 0 | 0/2 | [-453926, 628558] | False / True | 36.7162 / 105.438 / 105.256 |
| wood_184309:000432 / AB_FULL | ORIGINAL_WD / 2 | 0/6 | [-5.9054e+06, -1.7043e+06] | True / True | 383.545 / 3144.57 / 35683 |

B_FULL에서는 full:7을 단1개의 supported profile로 대체했다. 남은 partial 비용과 B region 비용은 각각 약1.34e-10·2.30e-9지만 교차 t=-18.4다. 선택 pose가 원 모델의 비용을 거의0으로 설명한다는 사실은 실제 finite edge가 관측 profile에 존재한다는 인증이 되지 않는다.

EDGE 선택은 iteration limit(status9,80회)로 optimizer_success=False지만 hard-constraint feasible이었다. 원 selector는 finite score와 feasible을 사용하며 optimizer_success를 별도 기록한다. 선택 bbox는 약0.00142×0.00248px로 붕괴했지만 두 지지선 연장의 offset 비용은 낮게 남았다. 이를 수렴한 최적해라고 부르지 않는다.

CAL AB는 SWAPPED_WD/h3에서 생성됐지만 최종 common-score 선택은 ORIGINAL_WD/h2다. 선택 context의 유일 feasible h2 책임은1이고 finite edge 교차는0/6이다. 생성 SWAPPED/h3를 별도로 검사해도0/6이다. 남은 full4그룹의 none 책임은 거의1이므로 full 위치항은 bounded none 비용으로 포화하며, B의 지지선 연장과 결합해 극단 깊이 pose가 더 낮은 총비용으로 남을 수 있었다. conditional 생성비용380.772와 최종 owner-mixture 비용383.545를 섞지 않는다.

**관측을 새로 만들어 낸 것과 관측-모델 연결의 유효성은 구분한다.** 원 RGB profile과 supportID는 실재하며 후보마다 새로 query하거나 삭제하지 않았다. 그러나 후보의 무한 지지선을 유한 변 밖으로 연장해 얻은 교차를 finite physical/model edge의 지지로 해석할 수는 없다. 따라서 관측하지 않은 후보 경계를 계속 이어 설명하는 연결이 허용돼 있었다는 점에서 instruction13.1.5의 유효성 점검에 걸리는 한계다.

동결 config/원 제약에는 명시적인 t∈[0,1] bound가 없었다. 존재하던 수치 bound를 실행 코드가 무시했다고 주장하는 것은 아니다. 동시에 METHOD_B와 docstring의 physical projected edge라는 표현보다 실제 구현이 보장하는 범위는 약하므로, 이를 단순한 성능 불확실성으로 축소해서도 안 된다. 기존 cuboid/material owner 미인증과 별도로, 등록 cuboid의 유한 변 자체도 이 교차를 지지하지 않는 것이 확인됐다.

세 사례는 결과를 보고 고른 극단 포렌식이다. 원 N0/E0에도 finite-segment 밖 교차가 있는 context가 있으므로 segment 검사 하나로 전체가 회복된다고 주장하지 않는다. 새 B2·가중치·임계값·후보·선택기·추론·최적화는 만들거나 실행하지 않았다. 모든 원 numeric/seal은 보존했고 이 문서는 별도 해석 기록이다.
