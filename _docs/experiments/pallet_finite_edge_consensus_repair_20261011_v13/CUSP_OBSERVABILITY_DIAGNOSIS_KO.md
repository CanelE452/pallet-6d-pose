# 채점 전 확인한 끝점 미분과 관측 충분성의 별도 문제

이 기록은 V13 기하 계산과 원 독립 검산 FAIL 이후, V13 자세 채점 전에 작성했다. 성능 수치에 맞춘 threshold 변경 제안이 아니다. 원 solver·원행·검산기·실패 영수증은 보존한다.

`VALIDATION_CHECKS.json`은 전체 245장을 처리했고 713,802개 검사 중 74개가 실패했으며 exception은 없다. 모두 frame111 `plastic_night_01:035421`의 proposal0/refit0에 모인다. edge2/query20의 저장 longitudinal 값은 −1.3911779893699083e−14px, 투영 선분 길이는 220.3024853359103px다. 보정기는 선언한 u<0 분기에 따라 BEFORE_FIRST_ENDPOINT와 한쪽 미분을 기록했고 독립 scalar 계산은 INSIDE_CLOSED_SEGMENT로 반올림했다. 후보마다 12개 Jacobian 원소, 1개 분기, 그 행에서 파생한 24개 column norm 검사, 합계 37개가 달라졌다. 자세·투영·잔차·cost·선택 결과 자체의 불일치는 이 실패에서 확인되지 않았다.

별도 검산 review는 독립 u/L 계산을 먼저 확인하고, 부동소수점 연산 오차 범위 안의 끝점 분기 차이에서만 저장 scalar의 엄격한 분기를 사용해 미분을 다시 계산한다. 저장 Jacobian 또는 branch 문자열을 정답으로 복사하지 않는다. 8px inlier, weak, score tie, optimizer 정책은 변경하지 않는다.

검산기 반올림 문제와 별개로, 이 후보의 fit은 점 0개와 선 2개다. edge4/query34도 u=−1.177103065811337e−13px로 첫 끝점에 사실상 닿아 있다. 저장된 한쪽 미분이 rank6에 기여하더라도, 반대쪽에서는 hinge 미분이 0이므로 이것을 안정적인 관측 충분성의 증명으로 확대할 수 없다. 원 acceptance는 [solver.py](../../../scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/solver.py)의 `_geometry` 및 `solve`에서 저장 observed/modeled Jacobian의 rank·weak를 사용한다. 원 `NEW_POSE`는 수치적 산출을 뜻하며 이 사례의 안정성 또는 실제 정확도 성공을 뜻하지 않는다.

이 원인을 확인하는 최소 후속 수리는 같은 관측·같은 optimizer 결과에서, 실제 fit 점의 미분과 각 독립 edge의 두 법선 incidence 미분만으로 선택된 후보의 rank·weak를 확인하는 것이다. 자유롭게 edge 안에 놓일 수 있는 부분 지지점의 longitudinal hinge는 최적화·합의·score에 유지하되, 점 등식 관측처럼 관측 충분성을 인증하는 행으로 세지 않는다. 따라서 점0개+선2개는 법선 scalar4개로 rank≤4이며 끝점 반올림 행으로 rank6이 되지 않는다. 충분한 실제 점·선 조합은 계속 허용한다. 기존 rank/weak 수치를 유지하고, 불충분하면 기본 출력을 반환한다. 다른 후보를 새로 선택하거나 GT·mask 오판 개수로 선택하지 않고 새 cusp 픽셀 threshold도 도입하지 않는다. 이는 선분 지지점 끝을 물리적 코너 대응이라고 가정하지 않는 보수적 절제다. 아직 이 후속 수리의 실행·채점 결과는 없다.

LOO는 H∪검사 대상 코너/선의 끝점을 제외한다. 최종 새 R,t의 fit도 H 초기 좌표를 제외하며, 산출되면 H를 재투영으로 한 번 대체하고 다시 fit하지 않는다. 마스크 하나의 오판은 프레임 실패 조건이 아니다.
