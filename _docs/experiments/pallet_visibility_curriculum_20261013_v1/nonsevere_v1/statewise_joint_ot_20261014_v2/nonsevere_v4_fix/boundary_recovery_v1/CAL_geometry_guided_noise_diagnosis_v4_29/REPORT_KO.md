# v29 CAL 선 잡음 진단

각 정책의 호환 CAL44에서 절대 법선 잔차 상위3개를 고정 규칙으로 선정했다. 두 정책의 같은 선은 한 그림으로 합쳤다. 정답 기반 선택은 CAL 진단에만 사용하며 runtime guide/관측·모델·파라미터는 바꾸지 않는다. 마젠타는 **CAL 진단 기준**이고 이미지 측정이나 추론 guide가 아니다.

법선 잔차는 실제 support anchor에서 지정 native 3D edge의 **무한 투영선**까지의 거리다. 이미지 상 실제 모서리를 측정한 값이 아니다. anchor는 unweighted actual-inlier mean을 저장된 weighted-IRLS TLS 선에 정사영한 통계다. 유한 지지/방향 순서는 별도 guard다.

| 사례 | 정책/순위 | edge/축 | normal(px) | 각도(°) | 실제 지지점 | RMS(px) | context |
|---|---|---|---:|---:|---:|---:|---|
| wood_184309:000925 | ALL_BI_REPLACE#1, RGB_NONFALLBACK_PRESERVE#1 | [4, 0]/Z | -492.695 | 68.212 | 148 | 0.3441 | TARGET_BACKGROUND_OUTLINE |
| wood_184309:000804 | ALL_BI_REPLACE#2, RGB_NONFALLBACK_PRESERVE#2 | [1, 2]/Y | 401.924 | 47.067 | 44 | 0.4035 | TARGET_BACKGROUND_OUTLINE |
| eval_cad:1778653017736058368 | ALL_BI_REPLACE#3, RGB_NONFALLBACK_PRESERVE#3 | [7, 3]/X | -348.702 | 74.369 | 12 | 0.0550 | TARGET_BACKGROUND_OUTLINE |

## wood_184309:000925 · edge [4, 0]

![wood_184309:000925](case0_actual.png)

## wood_184309:000804 · edge [1, 2]

![wood_184309:000804](case1_actual.png)

## eval_cad:1778653017736058368 · edge [7, 3]

![eval_cad:1778653017736058368](case2_actual.png)

실제 선 지지가 있다는 것과 해당 선이 지정된 cuboid 물리 변이라는 것은 다른 주장이다. 다른 평행 구조/슬랫·면 내부 경계·마스크 오류 가능성은 사진·mask와 함께 해석해야 하며 그림만으로 원인을 확정하지 않는다. 이 결과는 새 방법 채택이나 전체192 성능 개선 근거가 아니다.
