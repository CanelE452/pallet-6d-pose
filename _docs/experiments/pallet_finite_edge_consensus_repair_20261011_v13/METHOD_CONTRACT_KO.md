# 유한 실제 edge와 최종 inlier pool의 고정 국소 합의 수리

V12의 독립 저장 진단은 전체 245장과 모든 후보를 유지하여 검산했다. 무한선 법선 incidence는 유효한 필요조건이지만 실제 물리 변의 관측 구간까지 설명하는 충분조건은 아니다. 또한 기존 전체 pool의 rank6는 최종 유효 대응의 수·배치·정확도와 같지 않다. V13은 이 두 관측 계약을 수리하는 한 조건 `FINITE_EDGE_CONSENSUS`이며, 알려진 DEV에 대한 구조 수리다. 성능을 보고 threshold·seed·학습량을 조절하거나 보지 않은 평가라고 부르지 않는다. 원 V7~V12 코드/원행/결과와 main은 바꾸지 않는다.

[solver.py](../../../scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/solver.py)의 `FiniteEdgeConsensusBank.solve`는 기존 V11 filtered 동일 sparse q와 실제 unused physical/CAL line만 받는다. H 초기 image 좌표를 점 score/fit에서 제외하고, 새 native 좌표/관측을 만들지 않는다. `_seed`와 `_lines`의 기존 검증을 재사용하지만 기존 V9 `.solve`는 호출하지 않는다. 점에서 이미 소비한 incident edge와 일관된 duplicate line은 중복 factor로 세지 않는다. 점이 최종 outlier가 되더라도 그 점에서 소비했던 선을 새 독립 관측으로 복원하지 않는다. 중심8은 fit·score에서 제외하고 그대로 보존한다. 초기 N3와 available point-only pose는 optimizer start에만 쓰며 residual/치수 prior로 쓰지 않는다.

## 한 물리 edge의 잔차

등록 3D 끝점의 최종 투영을 A,B, L=||B−A||, g=(B−A)/L로 둔다. 기존 실제 관측 법선 n과 offset d에서 `(n·A−d, n·B−d)/sqrt(2)`를 유지한다. 같은 실제 edge의 N개 저장 CAL query support s에 `u=(s−A)·g`를 계산하고 `h=min(u,0)+max(u−L,0)`를 `sqrt(N)`으로 나누어 같은 edge block에 붙인다. 정상적인 부분 관측이 투영 edge 안에 있으면 h=0이다. 부분 관측의 처음/끝을 실제 물체 endpoint라고 가정하거나 숨은 endpoint의 초기 2D 좌표를 관측으로 만들지 않는다. query fraction을 3D 점 ID로 바꾸지 않는다.

이 block은 query 수와 관계없이 하나의 edge 합의 표다. 같은 선의 별도 point 또는 두 번째 독립 line로 세지 않는다. query radius는 원 CAL 불확실성 surrogate로 보존하며 hard truth gate나 인증된 covariance로 쓰지 않는다. 실제 edge ownership은 여전히 UNKNOWN이다. 기존 soft_l1/f_scale8/max_nfev50의 scalar objective를 새 명시 block에 적용한다. 추가된 접선 성분 때문에 objective가 기존 V9과 같다고 주장하지 않는다. sqrt(N)은 같은 지원을 반복하여 edge의 squared weight가 늘지 않도록 고정한 정규화이며 성능으로 선택한 가중치가 아니다.

OpenCV의 정확한 forward JA,JB를 사용한다. `du=−g·JA+(s−A)^T(I−gg^T)(JB−JA)/L`, `dL=g·(JB−JA)`이고, 처음 끝점 이전에는 dh=du, 둘째 끝점 이후에는 dh=du−dL, 닫힌 구간 안에서는 dh=0이다. u=0,L에는 안쪽 derivative0을 쓴다. 실제 목적함수의 법선은 관측 n으로 고정한다. 별도 modeled-normal geometry에서만 현재 모델 법선을 고정해 구조 rank를 비교한다. 투영 segment 길이<=기존 수치 기준1e−12이면 실제 optimizer의 해당 branch는 typed numerical failure로 남기며 jitter나 새 start를 추가하지 않는다. final score에만 들어가는 비채택 edge가 collapse하면 그 edge는 별도 degenerate outlier로 기록하여 다른 유효 관측을 전체 frame 예외로 막지 않는다.

## 유효 관측으로 한 번 다시 fit

두 기존 start source를 각각 두 등록 치수에 remap하여 최대4개 proposal을 만들고, 정확히 같은 시작 vector는 cache한다. 각 proposal을 항상 같은 전체 원 input pool로 score한다. 점은 Euclidean residual, edge는 위 전체 block의 L2 norm으로 기존 8px inlier를 정한다. 초기 rank는 optimizer veto에 쓰지 않는다.

각 proposal의 실제 8px point/edge inlier만 고정 fit pool로 만들고 branch당 한 번 refit한다. 추가 random/P3P/새 seed와 반복 threshold 탐색은 없다. 최대4 proposal dispatch+4 fixed refit dispatch이며 cache 재사용과 logical entry를 따로 보존한다. 원 full `used`/`line_edges`와 실제 최종 `fit_input_ids`/`fit_line_edges`를 구분한다. 최종 비교는 각 candidate의 서로 다른 fit cost가 아니라 항상 동일 full input pool의 inlier 수 우선, factor squared residual을64에서 자른 합 다음이다.

최종 refit이 실제 fit pool의 모든 factor를 8px 안에서 설명하고, positive depth/finite/optimizer success, 실제 fit pool의 observed 및 modeled-normal joint rank6, 동일 unit-column normalization의 기존 `smin < smax*1e−4` weak 검사를 모두 통과해야 NEW가 된다. 전체 pool rank6를 inlier pool의 관측 충분성으로 대신 쓰지 않는다. refit 후 일부 fit factor가 다시 outlier면 해당 branch를 `UNSTABLE_INLIER_REFIT`로 기록하고 다른 branch는 계속 검사한다. 한 branch가 실패했다는 이유로 프레임을 즉시 실패시키지 않는다. 추가 refit을 반복하거나 이미 제외한 점/선을 fit에 있다고 주장하지 않는다. 모든 branch가 실패하면 `failure_category_counts`에 numeric/insufficient/weak/unstable/other를 보존하고 순수한 실패 유형은 `NUMERICAL_FAILURE`, `INSUFFICIENT_ACTUAL_INLIER_GEOMETRY`, `WEAK_OBSERVABILITY`, `UNSTABLE_INLIER_REFIT`로, 혼합은 `MIXED_NO_VALID_CONSENSUS`로 구분한다. caller의 기본 출력 반환 여부는 solver의 새 자세 없음과 별도로 표시한다.

서로 다른 accepted physical branch가 동일한 최종 실제 point/edge 8px inlier 집합을 설명하면 SSE 비동점이어도 보수적인 `OBSERVATION_CONSENSUS_AMBIGUITY`로 구분한다. 기존1e−8의 수치 score 동점은 별도 `NUMERICAL_CONSENSUS_TIE`다. 어느 branch가 GT에 더 가깝거나 실제로 같은 팔레트 대칭인지 이 규칙으로 인증하지 않는다. 국소 후보들의 관측 합의 판정이며 전역 유일해 증명이 아니다. 모든 proposal/refit/exception/candidate/fit pool/full score/J/singular/condition/alternative/cache와 실제 callback 계수를 보존한다.

NEW의 최종 R,t가 있으면 H 코너를 해당 투영으로 한 번 교체한다. 재투영점이나 선을 독립 관측으로 다시 fit하지 않는다. 가림 오판 개수만으로 전체 frame을 거부하지 않는다. 수치 NEW는 실제 위치·회전 개선 또는 최상위 목표 달성의 증명이 아니며 전체 245장의 산출률·기본 반환·실제 T/R/ADDsym와 공통 산출 집합을 별도로 평가한다.

## 동결 전 CPU 계약 검사

[test_solver.py](../../../scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/test_solver.py)는 code-only 초안이다. 15개 group으로 정확한 forward J와 before/inside/after hinge derivative, 무한선에는 맞지만 유한 구간을 넘는 경우, 부분 관측 끝점 비강제, one-edge 정규화, 0–3 실제 점+선의 자연 SciPy 경로, wrong line 또는 wrong point 각각 하나를 제거한 뒤 남은 대응 refit, consumed/duplicate/H/no-refit, invalid support/optimizer exception, rank6지만 weak, scoring-only degenerate edge, 두 등록 치수/정확 cache를 검증한다. 같은 합의를 가진 비동점 물리 branch는 명시 controlled optimizer 반환으로 ambiguity 계약만 검증하며 자연 수렴 결과라고 주장하지 않는다. 같은 support의 반복 검사는 query 수로 edge squared weight가 늘지 않는 대수 성질만 확인하며 새 관측 공급을 주장하지 않는다.

작성자의 모델·fit·GT·수치·unit test 실행은0이고 AST만 확인한다. 루트가 전체 코드·test와 의존 SHA를 읽고 별도 CPU protocol/STARTED를 arithmetic 전에 동결한 뒤 실제 검사한다. 테스트 실패가 생기면 원 실패와 원본을 보존하며 성능에 맞춰 수치 정책을 바꾸지 않는다. 그 뒤 전체245 GT-free 기하와 독립 scalar 검사·cleanup·seal을 완료하고 실제 자세 평가를 실행한다. cached replay wall time을 배포 latency로 보고하지 않는다.

```bash
python -m scripts.research.pallet_finite_edge_consensus_repair_20261011_v13.cpu_checks freeze
python -m scripts.research.pallet_finite_edge_consensus_repair_20261011_v13.cpu_checks run
```
