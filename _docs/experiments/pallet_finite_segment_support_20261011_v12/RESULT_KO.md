# V12: 유한 선분 지원과 서로 다른 관측 합의 branch의 진단

“가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?”

**이번 진단은 그 개선을 입증하지 않았다.** 새 자세와 새 GT 채점은0이며 V11의 출력·선택·NEW/fallback을 그대로 유지했다. [V11 결과](../pallet_endpoint_observability_repair_20261011_v11/RESULT_KO.md)의 SAFE는 같은 쉬움153+중간92의245장에서 N3보다 위치·회전 평균이 나빴고 목표는 미해결이다. V12는 그 실패를 설명할 때 빠진 관측 조건이 실제 저장 기하에서 얼마나 나타나는지 확인한 독립 진단이다.

## 보정기의 어느 부분에 빠진 조건이 있었는가

기존 [V5 residual_jacobian:69](../../../scripts/research/pallet_partial_line_independent_20261010_v5/solver.py#L69)는 등록 3D edge의 투영 끝점에서 관측 **무한직선**까지의 법선 잔차를 fit한다. 실제 query의 support_points와 query_radii_px는 [source 검증:176](../../../scripts/research/pallet_partial_line_independent_20261010_v5/solver.py#L176)에 보존되어도, 최종 투영된 **유한 물리 선분**이 실제 관측 구간까지 닿는다는 조건을 그 잔차가 보장하지 않는다. 짧은 선분이 관측 구간에서 멀리 떨어져 있어도 같은 무한직선에 놓일 수 있다.

따라서 점·선의 rank6, 작은 법선 잔차와 기존8px inlier 합의만으로 유효한 유한 edge 관측을 설명했다고 결론내릴 수 없다. 이는 자기 가림 H를 fit에서 무시한 구현의 문제가 아니다. V11의 H fit 제외·NEW 후 재투영·재투영점 재fit 금지, 같은 선의 점/선 중복 금지 계약을 그대로 확인했다. 특정 실제 edge ownership이 틀렸다는 정답 판정도 아니다.

[audit.py:85](../../../scripts/research/pallet_finite_segment_support_20261011_v12/audit.py#L85)는 보존 candidate의 투영 끝점으로 닫힌 segment를 만들고, 실제 선택 CAL query에서 최근접 segment까지의 Euclidean 거리·끝점 전/후 초과 길이·무한선 법선 거리·원 source 잔차를 나눠 기록했다. 원 query radius는 그대로 썼다. 이 radius는 합성 CAL의 조건부 불확실성 surrogate이며 인증된 물리 오차 상한이나 완전한2D covariance가 아니다. radius gate는 실제 출력에 적용하지 않았다.

## 전체245장의 실제 진단 수치

[DIAGNOSTIC_RECEIPT](DIAGNOSTIC_RECEIPT.json)는 별도 protocol 동결 후 한 번 실행해 완료됐다. 기존 V11 기하490행·ledger245행, V7 source 기하1960행·관측735행을 모두 연결했고 후보가 없는 프레임도245행에 유지했다. 아래는 그 저장 집계이며 새 산술로 다시 계산한 값이 아니다.

|진단|실제 저장 수|분모와 해석|
|---|---:|---|
|선택 후보의 실제 지원점이 원 radius 밖인 frame|64|전체245장 중64장. SAFE NEW94장의 분모로 바꾸지 않는다.|
|비동일 대안이 선택 후보와 같은 전체 factor의8px 합의를 만족한 frame|106|전체245장 중106장. 수치 동점과 구분한다.|
|보존 candidate|380|여러 초기화·치수·branch를 포함하며380장의 의미가 아니다.|
|같은 전체 합의를 만족한 비동일 alternative|124|106frame에 속한 저장 대안 수다.|
|candidate별 선 factor|515|같은 frame/edge가 여러 candidate에서 집계될 수 있다.|
|실제 support query 검사|2958|candidate별 실제 query 지원 검사 수다.|
|지원 query가 원 radius 밖인 candidate-line|254|515 candidate-line 중254개. frame 수와 다르다.|

두 후보가 accepted이고 모든 camera depth가 양수이며 관측/모델 J가 모두rank6이고, 모든 실제 point/line factor가 기존8px inlier이며 저장된 numerically_equivalent_to_selected=False일 때 합의 다중 branch를 표시했다. [audit.py:98](../../../scripts/research/pallet_finite_segment_support_20261011_v12/audit.py#L98)의 이 정의는 비용 차이를 무시해 새 후보를 선택하거나 통계적 동점 threshold를 만든 것이 아니다. 어느 branch가 실제 GT에 가까운지, 실제 물체 대칭인지, 올바른 소유 edge인지는 결정하지 않는다.

|원 운용 view|NEW|N3 fallback|V12가 새로 산출한 자세|
|---|---:|---:|---:|
|FILTERED_LOCAL|97|148|0|
|FILTERED_LOCAL_WEAK_SAFE|94|151|0|

64와106의 교집합이나 실패의 인과 기여율은 이 집계만으로 주장하지 않는다. 일부 radius 초과 또는 다중 branch가 발견됐다는 이유로 원 frame을 실패 처리하거나 pose를 바꾸지 않았다.

## 독립 검산과 다른 사람이 확인할 증거

[SEGMENT_CHECKS](SEGMENT_CHECKS.json)는 별도 [검사 protocol](SEGMENT_CHECK_PROTOCOL.json) 동결 후 실제80,320검사/실패0으로 complete=True/passed=True다. producer 함수를 import하지 않는 stdlib [validation_checks.py](../../../scripts/research/pallet_finite_segment_support_20261011_v12/validation_checks.py)가 endpoint tangent의 signed longitudinal 좌표로 segment 거리를 독립 검산했다. 원 선택 query·semantic edge·등록 끝점·원 line record·consumed/rejected/retained 선·한 solver의 두 view·전체 candidate와490/245/1960/735 입력 수를 대조했다.

검산의 floating 허용치 relative1e−9/absolute_px1e−8은 수식 비교에만 쓰며 새로운 fit/분류 threshold가 아니다. R,t 재투영, J/rank/SVD 재계산은0이며 이 부분은 원 V11의 별도 review 바이트 권위를 유지했다. 실제 모든 saved output이 unchanged이고 모델·영상·학습·RGB·GT score·새 pose·새 fit0이다. 진단 wall21.10131147503853s는 저장 자료 처리시간이며 latency가 아니다.

[SEGMENT_ROWS](SEGMENT_ROWS.jsonl.gz)는245개 frame의 모든 candidate/query 원행이고4,664,914bytes, SHA e5ec035fbf4623f92da9cb6ae907a3e8ef5629ec46a7a3a266510dc98a8718bc다. 원 frame ID로 V11 기하·ledger와 연결해64/106의 포함 여부, 각 실제 query의 segment 초과, 비동일 후보의 저장 cost/J/depth/inlier를 확인할 수 있다. [DIAGNOSTIC_CONTRACT](DIAGNOSTIC_CONTRACT_KO.md)의 실행 전 세 사례는 GT나 이후 성능으로 선택하지 않았고, [CHECKER_CONTRACT](CHECKER_CONTRACT_KO.md)는 독립 검산 범위와 한계를 명시한다.

현재 확인한 문제는 **무한선 법선 잔차가 유한 물리 edge의 관측 구간을 설명하는 조건을 빠뜨렸고, 서로 다른 pose branch도 같은 기존 합의를 만족할 수 있다는 것**이다. 이 진단은 그 누락을 운용 정책에서 수정하거나 자세 개선을 평가한 결과가 아니다. 다음 최소 검증은 같은 관측·가중치·H 계약을 유지한 채 유한 지원과 branch 처리를 명시한 별도 동결 정책을 실제 자세 평가하는 것이다. 올바른 대응이 남으면 일부 mask 오판을 허용한다는 조건을 유지해야 하며, 현재 결과만으로 ownership·원인 기여율·방법 성공을 주장하지 않는다. 공개 검토와 정상 브랜치 게시 상태는 별도 실제 receipt로 확정한다.
