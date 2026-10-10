# V13 finite hinge 끝점 검산 보완

원본 `validation_checks.py`와 자체 freeze·STARTED·complete 713,802 checks/74 FAIL 영수증은 보존한다. 실패는 `plastic_night_01:035421`의 proposal0/refit0, edge2/query20에 집중됐다. 저장된 종방향 위치는 `-1.3911779893699083e-14px`로 BEFORE 분기를 택했고 독립 scalar 회전·투영의 반올림은 INSIDE를 택했다. 두 Jacobian의 한 행, branch 문자열, 그 행으로 계산한 column norms가 각각 37개씩 달랐다. 잔차·비용·자세·8px 합의·선택 결과는 이 검산에서 실패하지 않았다.

새 `validation_review.py freeze/run`은 원본 검산 AST의 정확히 일곱 위치만 바꾼다. 저장 후보의 관측/모델-normal edge block을 전달하고, 독립 `u,L` 검산을 거친 뒤 서로 다른 분기가 같은 끝점의 floating-roundoff 범위에 있을 때만 저장 수치가 택한 one-sided 분기로 독립 analytic Jacobian을 재구성한다. 원본 AST를 역으로 복구해 다른 변경이 없음을 확인한다. 출력은 `REVIEW_VALIDATION_PROTOCOL.json`, `REVIEW_VALIDATION_STARTED.json`, `REVIEW_VALIDATION_CHECKS.json`이며 원본 전체 입력과 실패 영수증, 원본 protocol, 새 코드와 이 계약을 먼저 바이트 고정한다.

roundoff 상한은 `64*ulp(max(1,L,storedL,abs(support/A/B coordinates)))*(1+distance(support,A)/L)`이다. 독립·저장 `u`와 `u-L`이 같은 끝점의 이 상한 안에 있어야 하며 저장 branch 문자열도 저장 `u,L`의 strict predicate와 일치해야 한다. 기존 near 허용값·8px·rank·weak `1e-4`·tie·consensus·관측·fit·출력은 변경하지 않는다. 실제 보완 대상은 proposal/refit의 observed/modeled 네 branch 재구성이며 모든 245행과 모든 후보의 기존 검산을 다시 실행한다.

끝점 hinge는 미분 불가능하다. 저장 수치가 택한 one-sided Jacobian의 rank6은 안정적인 two-sided 관측 가능성이나 전역 유일해의 증명이 아니다. 이 한계를 명시하며 원본 실패를 PASS로 바꾸거나 실제 pose 개선의 증거로 쓰지 않는다. 새 RGB·GT·모델·PnP·optimizer·SVD·학습·성능 임계값 탐색을 실행하지 않는다.
