# 일차 출처의 실제 확인 범위

2026-10-17에 아래 공개 일차 출처를 확인했다. 논문 본문 전체·수식·성능표를 읽었다는 기록이 아니다. 이번 열린 가시 경계의 단방향·길이 비율·고정 후보 재선택 조합은 입력보조 미검증 제안으로 시작했으며, 아래 방법의 재현이나 그 성과를 물려받는다고 주장하지 않는다.

| 출처 | 실제 접근 범위 | 이번 작업과의 연결·차이 |
|---|---|---|
| [SRT3D, DLR 기관 페이지](https://www.dlr.de/en/rm/research/publications-and-downloads/papers/srt3d-a-sparse-region-based-3d-object-tracking-approach-for-the-real-world) | 기관 페이지의 제목·서지·공개 초록을 live 열람 | contour correspondence line과 불확실성 처리가 선행한다. 그 추적 모델·사전 viewpoint·Newton/Tikhonov 최적화를 이번에 도입하지 않음 |
| [Fast Directional Chamfer Matching, MERL TR2010-045](https://www.merl.com/publications/TR2010-045) | 연구기관 공개 서지·초록을 live 열람 | 경계 거리와 방향을 쓰는 선행이다. 이번은 방향을 쓰지 않는 유한 선분 최소거리이며 directional matching의 재현이 아님 |
| [Uncertainty-Aware Camera Pose Estimation From Points and Lines, CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Vakhitov_Uncertainty-Aware_Camera_Pose_Estimation_From_Points_and_Lines_CVPR_2021_paper.html) | 공식 URL의 검색 인덱스에 노출된 초록·저자·서지만 열람. 페이지와 공식 PDF의 직접 open은403 실패. 본문 미열람 | 점·선의 2D/3D uncertainty를 구분하는 선행 원리. 이번 full/partial 보존 이유와 관련되지만 보조 경계가 실제 cuboid 물리 변이라는 인증을 주지 않음. 새 PnPL/BA 실행 없음 |

가장 단순한 대조는 새 공급기를 먼저 만드는 것이 아니라 원 reverse outline 요구만 제거한 HULL_ONEWAY다. HULL_ONEWAY_MASS는 같은 r의 약화 대조다. 두 대조를 모두 실행해 실제 입력 경계의 추가 효과를 분리했다. 더 강한 비대상 분할, CAL uncertainty 재학습, 새 후보 전달/최적화는 이번 시험에서 실행하지 않았다. 실제 선택 결과를 근거로 다음 작업 한 가지를 [NEXT_DECISION_KO.md](NEXT_DECISION_KO.md)에 정한다.

저장소에는 이미 `scripts/research/pallet_visibility_curriculum_20261013_v1/image_supported_edge_chains_v4_17c.py`가 있다. `observe_image_supported_edge_chains`와 `_edge_chains`는 RGB NMS peak와 chain, 실제 지지 범위, conditional uncertainty를 남긴다. 그 탐색은 초기 projection strip·B/I anchor·native neighbor·mask context에 조건화되고 S/O/T/U query를 구분한다. 따라서 “영상선 사용”은 이번에 처음 생긴 원리가 아니다. 이번 입력 작성자는 이 추출기를 실행하거나 초기 모델에 맞춰 경계를 고르지 않았다.

원 목적함수와 단계별 backend는 [기존 OBJECTIVE_MAP](../pallet_selection_score_diagnosis_v1/OBJECTIVE_MAP.md), comparator와 위임은 [기존 SELECTION_RULE_MAP](../pallet_selection_score_diagnosis_v1/SELECTION_RULE_MAP.md), native Q·물리 치수·원점·참조의 한계는 [REGISTRATION_PROVENANCE_KO](../pallet_outline_physical_support_audit_v1/REGISTRATION_PROVENANCE_KO.md)를 유지한다. 이번 관련 source diff는 시작 main에 대해 없었고 새 namespace만 추가했다.
