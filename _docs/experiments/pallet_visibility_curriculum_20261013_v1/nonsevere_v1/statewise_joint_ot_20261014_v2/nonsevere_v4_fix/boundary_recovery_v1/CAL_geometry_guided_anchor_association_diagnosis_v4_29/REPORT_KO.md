# v29 CAL 상위 선: 원 B/I q와 탐색 anchor 연결 진단

기존 top3/두정책 합집합을 그대로 사용한다. 이 단계는 정답·이미지 재조회 없이 보관된 실제 TLS 선과 모든 기존 B/I q의 법선 거리를 계산했다. S/U는 좌표 접근 전에 제외했다. nearest q를 골라 native 대응을 바꾸지 않는다.

| 사례/edge | search anchor 종류 | 원 source q→TLS 거리(px) | mate q→TLS(px) | 실제 search anchor→TLS(px) | q→search anchor 2D(px) | search seed 법선방향 q이동(px) | reach(px) |
|---|---|---:|---:|---:|---:|---:|---:|
| wood_184309:000925 [4, 0] | MODEL_PROJECTION_SEARCH_ONLY_BAND | -322.780 | -14.187 | 9.723 | 614.395 | -398.859 | 43.0 |
| wood_184309:000804 [1, 2] | MODEL_PROJECTION_SEARCH_ONLY_BAND | 667.033 | 627.632 | -2.368 | 670.358 | 670.050 | 43.0 |
| eval_cad:1778653017736058368 [7, 3] | MODEL_PROJECTION_SEARCH_ONLY_BAND | 48.037 | 미사용/비BI | -18.711 | 384.390 | 0.000 | 43.0 |

## 모든 원 B/I q에 대한 실제 TLS 법선 거리

| 사례/edge | q ID/상태 | 기존 source | signed 거리(px) | 절대거리 43px 이내 |
|---|---|---|---:|---|
| wood_184309:000925 [4, 0] | 0/B | VISIBLE_BACKBONE_FALLBACK | -14.187 | True |
| wood_184309:000925 [4, 0] | 1/B | B_TARGET_BACKGROUND_LINES | -12.425 | True |
| wood_184309:000925 [4, 0] | 2/B | B_ACTUAL_BOUNDARY_RECOVERY | 53.229 | False |
| wood_184309:000925 [4, 0] | 3/B | B_ACTUAL_BOUNDARY_RECOVERY | 52.038 | False |
| wood_184309:000925 [4, 0] | 4/B | VISIBLE_BACKBONE_FALLBACK | -322.780 | False |
| wood_184309:000925 [4, 0] | 5/B | VISIBLE_BACKBONE_FALLBACK | -316.186 | False |
| wood_184309:000804 [1, 2] | 0/I | VISIBLE_BACKBONE_FALLBACK | 10.123 | True |
| wood_184309:000804 [1, 2] | 1/B | B_TARGET_BACKGROUND_LINES | 667.033 | False |
| wood_184309:000804 [1, 2] | 2/B | VISIBLE_BACKBONE_FALLBACK | 627.632 | False |
| wood_184309:000804 [1, 2] | 3/B | B_TARGET_BACKGROUND_LINES | 3.686 | True |
| wood_184309:000804 [1, 2] | 4/B | B_TARGET_BACKGROUND_LINES | -14.440 | True |
| wood_184309:000804 [1, 2] | 5/B | B_TARGET_BACKGROUND_LINES | 470.033 | False |
| wood_184309:000804 [1, 2] | 7/B | VISIBLE_BACKBONE_FALLBACK | -20.016 | True |
| eval_cad:1778653017736058368 [7, 3] | 0/I | VISIBLE_BACKBONE_FALLBACK | 351.983 | False |
| eval_cad:1778653017736058368 [7, 3] | 5/B | B_ACTUAL_BOUNDARY_RECOVERY | -2.126 | True |
| eval_cad:1778653017736058368 [7, 3] | 7/B | VISIBLE_BACKBONE_FALLBACK | 48.037 | False |

43px는 각 실제 탐색 strip anchor 주변의 공학적 normal reach다. MODEL_GUIDE_IMAGE_ANCHOR가 원 q에서 멀리 이동한 경우 같은43px라도 원 q 주변 제한이라는 뜻이 아니다. 현재 graph의 q displacement 항은 선택된 incident 선 두 개 이상이 교점을 만들 때만 계산되며, 단독 선의 native-point 연결을 자동 증명하지 않는다. 위 수치는 이 계약을 진단할 뿐 새 q gate/threshold/키포인트나 방법을 구현하지 않는다.

[원 실제 RGB/mask 패널](../CAL_geometry_guided_noise_diagnosis_v4_29/REPORT_KO.md) · [전체 선의 coherent C2 비교](../CAL_geometry_guided_C2_identity_diagnosis_v4_29/REPORT_KO.md). 기존 runtime/관측/모델·CAL 기준·primary 평가는 변경하지 않았다.
