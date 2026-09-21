# E1-B geometry sanity

[확인] 検証된 physical clean/occlusion pair가 없어 CAD18×3후보 자기 가림 출력54개만 재현했다. GT-visible external recovery oracle은 미실행이다.

[확인] 실패/적용 이유: `{'fewer_than_6_reliable_visible_inframe_corners': 14, 'replaced_hidden_only': 40}`. 각 camera-facing 치수축 ±5%의6변형에 대해 출력 이동, fit residual, hidden set을 기록했다. 사용점 수·convex hull 면적도 기록했다.

[확인] 이상적 cuboid regression: `{'front_view_hidden_far4': True, 'oblique_hidden_corner': True, 'synthetic_hidden_error_recovered': True, 'visible_and_center_exact': True}`. 이는 실제 팔레트 camera/GT contract의 완전 검증이 아니다.

[추정] PnP 유래 가능 hidden GT와의 일치는 물리적 복구를 독립 입증하지 못한다.
