# 저장된 유한 선분 진단의 독립 검산

[validation_checks.py](../../../scripts/research/pallet_finite_segment_support_20261011_v12/validation_checks.py)는 V12 producer의 계산 함수를 import하지 않는 stdlib 검사기다. 원 producer protocol·STARTED·완료 receipt·245행과 모든 부모 입력의 실제 SHA/bytes/path, V11 원 검사 실패와 별도 전체 review PASS를 자체 `SEGMENT_CHECK_PROTOCOL.json`에 arithmetic 전에 고정한다. `SEGMENT_CHECK_STARTED.json` 뒤에만 아래 검산을 수행한다. 실패 원인과 완료 전까지의 검사·집계 수는 `SEGMENT_CHECKS.json`에 보존하며 기존 파일은 덮어쓰지 않는다.

V7 원 1,960 기하 행·735 관측 행, V11 490 기하 행·245 ledger, V12 245행을 동일 프레임 순서로 읽는다. Clean153/Moderate92 전체 집합과 마지막 extra row가 없는지 확인한다. 원 flat ROLE 관측, 선택 CAL query 좌표, 실제 semantic edge/등록 끝점, line 원 record, 코너에서 이미 소비한 incident edge, rejected/retained edge, 한 solver 제안에서 나온 두 view와 모든 candidate를 대조한다. 원 source line 법선 정규화와 지원 합의를 독립 scalar 수식으로 확인한다. source contract가 원래 typed invalid인 경우는 그 상태와 빈 후보를 유지한다.

각 저장 candidate의 투영 endpoint에서 unit tangent를 만들고 지원점의 signed longitudinal coordinate로 닫힌 선분의 최근접 위치와 최소 Euclidean 거리, 앞/뒤 초과 길이, 투영 무한선 거리 및 원 source-line 잔차를 다시 구한다. producer의 parameter 기반 계산과 다른 수식 경로를 사용한다. 0길이 선분은 점 거리와 별도 degenerate 표식으로 검산한다. 반경·지원점은 변경하지 않는다. 원 point/line의 8px 잔차와 inlier를 2D 저장 투영에서 대조하고, 양의 저장 depth/두 joint rank6/accepted/동일 전체 factor inlier를 가진 비동일 저장 대안의 predicate를 확인한다. R,t 재투영, Jacobian/rank/SVD 재계산이나 다른 branch 선택은 하지 않는다. 그 부분은 원 V11 독립 review의 바이트 권위를 유지한다.

부동소수점 수식 대조에는 `relative=1e-9`, `absolute_px=1e-8`을 쓴다. 이는 검산 오차 허용치이며 pose나 source 분류 threshold가 아니다. `distance <= original radius`, `residual <= 8px`의 경계 bool은 독립적으로 대조된 producer의 저장 scalar에 원 연산자를 적용하여 libm 반올림 차이로 새로운 분류를 만들지 않는다. 전체 candidate/line/query 수, ambiguous frame/대안 수, radius 초과 candidate-line/frame 수 및 두 view의 모든 상태를 원 receipt와 대조한다.

반경은 source-CAL의 조건부 불확실성 surrogate다. 반경 밖의 지원점이 있는 64개 프레임이라는 producer 집계는 전체 245개 중 선택 후보가 존재하고 이 조건을 만족한 프레임 수이며, 이를 safe NEW 94개의 전체 범위로 대신 해석하지 않는다. 이 진단만으로 실제 edge 소유권 오판, 자세 오차 또는 단독 실패 원인을 인증하지 않는다. 여러 branch가 전체 8px 합의를 동시에 만족한 106개 프레임도 어느 branch가 실제로 옳은지 증명하지 않는다. 원 pose·분류·출력·원행은 변경하지 않으며 fit/model/image/GT score/Jacobian/SVD 호출과 새 pose 출력은 모두 0이다.

```bash
python -m scripts.research.pallet_finite_segment_support_20261011_v12.validation_checks freeze
python -m scripts.research.pallet_finite_segment_support_20261011_v12.validation_checks run
```

코드 작성자의 자체 숫자 실행은 0이다. 루트가 전체 source를 검토한 뒤 별도로 자체 동결·실행한다.
