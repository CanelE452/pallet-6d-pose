실제 생성된 PNG 8개 중 `01_all_operational.png`와 `case_02.png`를 열어 확인했다. 새 추론·PnP·영상 생성은 하지 않았다.

전체 분포 그림에는 Clean153+Moderate92, 운용 출력 전체, 평균/분위 구간과 GEOMETRIC_PROXY 한계가 표시돼 있다. 긴 방법 이름과 위치/회전 축이 잘리지 않으며 큰 오차도 표시돼 있다. 그림의 다이아몬드는 평균이고 원행·METRICS가 수치의 기준이다.

고정 사례 02에는 같은 RGB의 N3/v5 C2/v6 양 끝점 LOO C2가 나란히 표시돼 있다. 프레임 ID, NEW/FRESH 상태, 위치·회전 수치, 코너 ID와 관측/재투영/선 색 범례가 읽힌다. 이 사례에서는 두 C2 결과가 같은 수치를 반환하므로 v6만 좋아졌다고 해석하지 않는다. 선택되지 않은 자기 가림 코너의 최종 재투영은 주황 사각형으로 표시돼 있다. 그림에서 보이는 선의 물리적 소유나 정답을 독립 인증한 것은 아니다.

6개 사례는 기존에 고정된 VISUAL_CASE_PROTOCOL을 재사용했다. 이번 결과를 보고 유리한 사례를 새로 고르지 않았다. 나머지 PNG의 경로·바이트·SHA는 FIGURE_BINDINGS.json에 있고, 기하 계약은 VALIDATION_CHECKS.json과 REPROJECTION_CHECKS.json으로 검산한다.
