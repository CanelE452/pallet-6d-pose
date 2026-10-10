# V14: 유한 선분 fit과 안정적인 점·선 관측성을 구분하는 단일 절제

이 정책의 원인과 방향은 V13 자세 채점 전에 저장한 [CUSP_OBSERVABILITY_DIAGNOSIS_KO.md](../pallet_finite_edge_consensus_repair_20261011_v13/CUSP_OBSERVABILITY_DIAGNOSIS_KO.md)에 근거한다. V13의 frame111에서 실제 점0개·선2개의 끝점 hinge가 약 −1.39e−14/−1.18e−13px인 한쪽 미분으로 rank6에 기여했다. 원 보정기·원행·713,802검사/74FAIL은 보존하며, 이를 별도 roundoff 검산 review로 확인하는 것과 운용 충분성의 수정은 구분한다. 새 GT·성능에 맞춘 threshold·학습·합성·seed 선택은 없다.

유한 지원 hinge는 실제 support가 물리 선분의 범위에 놓이는 조건이다. support의 위치는 그 edge를 따라 자유롭고, 지원점에 독립적인 3D 코너 ID를 부여하지 않는다. 따라서 hinge 활성 여부의 한쪽 미분을 별도 등식 관측처럼 세어 안정적 관측성을 승인하지 않는다. 새로운 cusp pixel epsilon을 만들지 않고 **모든 longitudinal hinge 행을 관측성 J에서 제외**한다. 이는 cusp 근방만 제외하는 것보다 보수적인 명시적 운용 절제다. finite hinge는 원 V13 optimizer 잔차, 실제8px edge-block inlier, 공통 full-pool 점수에 그대로 남는다.

단일 method는 `FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE`다. V13이 이미 선택·승인한 한 candidate의 실제 fit point IDs와 fit line IDs만 사용한다. 그 후보의 R,t/치수·관측 q·실제 normalized line·inlier/refit pool은 바꾸지 않는다. 변경 없는 V5.geometry_info의 실제 point J와 각 semantic edge의 두 endpoint-normal J/√2로 observed와 modeled-normal 관측성을 다시 계산한다. J column L2 정규화, rank 상대1e−10, weak 판정 smin<smax×1e−4는 기존 값 그대로다. modeled normal은 이 국소 기하 검사의 미분에서 고정하며 전역 유일해를 증명하지 않는다.

둘 다 rank6이고 weak가 아니면 원 V13 결과를 같은 자세·같은 출력 좌표로 유지한다. rank가 부족하면 `STABLE_INCIDENCE_INSUFFICIENT`, rank6이나 weak이면 `STABLE_INCIDENCE_WEAK`, 관측성 계산의 수치 실패이면 `STABLE_INCIDENCE_NUMERICAL_FAILURE`로 새 자세를 사용하지 않는다. 기본 자세가 있으면 원 N3 전체 좌표와 원 초기 자세로 fallback하며, 없으면 완전 실패로 남긴다. 다른 candidate 재선택·다른 초기화·추가 optimizer/refit/PnP는0이다. 원 V13이 NEW를 만들지 못한 frame도 그대로 유지하며 이 guard의 기하 연산을 생략한다.

점0개·선2개는 normal 등식이4개이므로 이 안정적 J의 rank는 최대4다. 반면 충분한 다른 실제 점·선의 J가 rank6이면 일부 관측이 없어도 NEW를 유지할 수 있다. H 초기2D 좌표는 두 J에도 들어가지 않는다. 같은 선에서 만든 점/선은 원 V13의 consumed-edge 제외를 그대로 유지한다. H에 속한 3D 끝점을 가진 실제 선은 H 초기 pixel을 쓰지 않고 모델 기하로 검사할 수 있다. mask 오판 개수나 mask 변경은 frame veto가 아니다.

NEW 유지 시 원 V13의 H 재투영 출력을 그대로 인수하며 좌표를 다시 할당하거나 fit하지 않는다. decline 시 새 R,t를 최종 자세로 승인하지 않고 full N3 fallback을 반환한다. 원 candidate 전체·finite geometry·원 rank·optimizer 실행량과 새로운 incidence J·rank·weak·원 source row SHA·guard 상태를 별도 witness로 보존한다. 위생적인 guard 이후의 numerical NEW도 실제 위치·회전 개선이나 실제 edge ownership의 증명은 아니다.

새 [pipeline.py](../../../scripts/research/pallet_stable_incidence_guard_20261011_v14/pipeline.py)의 API는 `guard_solver(solved,points,K,xyz,hidden) -> (guarded,witness)`와 `guard_row(unscored_V13_row) -> (row,witness)`다. V5 geometry_info를 완료 호출한 frame의 projection은2회이며 rank_info는5회다. 실제 OpenCV/SVD entry·return·exception은 driver가 따로 계수한다. 노출 J는 원 selected finite J에서 point 행과 각 edge의 첫 두 normal 행만 정확히 추출한 것으로, 새 incidence spectrum/column scale과 대조한다. 저장 singular spectrum을 복사해 새 SVD를 실행한 척하지 않는다.

전체245장(Clean153/Moderate92)을 유지하고 실제 운용 NEW/fallback/완전 실패 및 원 V13과의 같은 frame 결과를 모두 비교해야 한다. 숨은 corner pixel만 좋아졌다고 자세 개선을 주장하지 않는다. 새 정확도·독립 검산·fresh runtime은 별도 source/protocol/STARTED 동결 후 root가 실제 실행한다. 이 CODE ONLY 구현 문서는 학습·실사·GT 채점·CPU 검사·자원 측정을 완료했다는 보고가 아니다.
