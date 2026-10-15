# guide anchor·방향·W/D 소스 추적

모델 anchor 대안과 부호 검사가 빠진 단일선 경로가 확인됐습니다. 물리 R/Q 변환 자체가 틀렸다는 증거는 아닙니다. 현재 V30은 변경하지 않습니다.

- `joint_geometry_guided_lines_v4_28.py:178–195`: 원 BI q anchor의 VP 검색 뒤 모델 투영 anchor 대안을 추가합니다. `dict(proposal, …)` 때문에 `guide_kind`는 “관측 BI anchor” 이름 그대로이고, 실제 위치는 `search_anchor_kind=MODEL_PROJECTION_SEARCH_ONLY_BAND`입니다.
- 같은 파일 `235–250`: 실제 lattice는 원17c의 q·BI 방향을 무시하고 proposal anchor·direction으로 검색합니다. bothBI의 원 q 거리로 만든 lo/hi도 그 모델 anchor·방향에서 적용됩니다.
- `177/194`의 방향 부호는 각 trial pose 투영 mate−k로 정합니다. bothBI 원 q 방향과의 부호 검사는 이 생성 단계에 없습니다. 측정 BI 방향은 경쟁 대안 중 하나입니다.
- `ordered_geometry_support_v4_28b.py:41–100`에는 원 BI hemisphere 검사가 있지만, `182–206`에서 교점이 생긴 incident 두 선에만 실행합니다. 단일선은 이 검사를 거치지 않고, 이후 PnP의 새 모델 꼭짓점만 실제 support에 맞출 수 있습니다.
- `image_edge_observations_v4_29.py:83–105`: 다른 family의 physical pose/center를 ORIGINAL XYZ/Q로 재표현합니다. 같은 unordered box임은 맞지만 원 image channel k와의 동일 대응이 보장되는 것은 아닙니다. 이것은 조건부 guide이지 native 변의 정답 인증이 아닙니다.

이 사례의 guide는 `SWAPPED_WD/CACHED_329`입니다. 원 B0 q에서 약 **41.076px** 떨어진 모델 anchor를 사용했고, query 방향은 원 initial0→3와 원 B0→B3 방향 모두 반대 hemisphere입니다. 저장된 **9.9498°**는 새 generating guide와 TLS의 무방향 각도이며, 원 native ray의 부호 검사 통과가 아닙니다.

원 initial 투영은 선택 ref의 `original_reference_projection_xy`에 실제로 남아 있습니다. source는 ORIGINAL_INITIAL guide/base17c·측정 BI 대안·모든 원 chain을 보존합니다. 올바른 실제 선 후보가 측정·통과·선택됐다는 뜻은 아니며, 이번에는 추가 raw row를 읽지 않았습니다.

후속 세대에서는 **원 BI q anchor + 불변 signed native ray**를 주 경로로 하고, 모델 anchor는 별도 진단으로 분리하는 것이 맞습니다. 단일선에도 원 q 기준 finite/order association을 먼저 확인하고, 부분 지지는 남기되 다른 collinear 구간을 source로 인증하지 않아야 합니다. 무방향 TLS normal의 부호 불변성은 유지합니다. O/T/U는 model/range만, S는 사전 제외를 유지합니다. 새 정책은 별도 profile/CAL 교정과 전체 평가가 필요하며, 현재 V30의 해결 완료로 설명하지 않습니다.

정확한 line excerpts·source SHA·수정 위치는 [JSON](GUIDE_ANCHOR_DIRECTION_SOURCE_AUDIT_V1.json)에 있습니다.
