첨부 원문의 C2에서 허용한 지역적인 점·선 정제를, 현재 V7의 같은 IMAGE_ROLE 원시 관측에 연결한다. 새 조건은 `ROLE_BOUNDARY_LOCAL_POINT_LINE` 하나다. 이 문서는 결과를 보기 전에 고정할 기하·평가 계약이며, 코드 작성이나 CPU 검사 통과를 실사 성능 개선으로 보고하지 않는다.

## 범위와 기존 실행의 구분

사용자 실제 요청의 최신 대상은 Clean153+Moderate92의 245장이다. 첨부 첫 문서의 전체319장과 Severe74는 이 신규 실행에 적용하지 않는다. main의 고정 RGB 추정기, N3, cornerSubPix, 원본 영상·가중치·참조·기존 결과와 사용자 작업본은 보존한다. 학습, 새 합성 RGB, 새 실사 촬영, 수동 주석, GT를 이용한 관측 선택, 실사 오차를 보고 threshold를 바꾸는 반복은 하지 않는다.

원첨부 `/home/minjae/Downloads/pallet_cli_observation_refiner_robust_pnp_20261009.md`의 399행은 4코너 미만에서도 실제 부분선이 남으면 별도 C2 경로를 사용하도록 했고, 402–413행은 기존 초기 자세를 optimizer 시작점으로 쓰는 작은 비선형 정제, 선 끝점이 hidden이어도 실제 관측선 유지, 관측 중복 금지, 지역 rank6와 부족시 fallback을 명시했다. 151행의 주 점 경로에는 3점 P3P를 추가하지 않는다. 여기의 3점+초기 자세 지역 정제는 P3P의 독립 초기화나 모든 근을 열거하는 새 해법이 아니다.

수정 전 및 수정 감독의 원 v1 C2·C3와 66조건 실행은 이미 완료됐다. [원 v1 point_line.py](../../../scripts/research/pallet_observation_refiner_20261009_v1/point_line.py#L19)는 4점 미만도 기존 초기 자세에서 지역 정제했으며, 그 과거 결과를 이번 원행으로 복사하지 않는다. [V5 solver](../../../scripts/research/pallet_partial_line_independent_20261010_v5/solver.py#L278)는 실제 점4개와 점 inlier4개를 요구하고, V6도 그 final solver를 감싼다. [V7 pipeline](../../../scripts/research/pallet_three_head_observation_20261010_v7/pipeline.py#L333)은 부분선을 자세 fit에 사용하지 않는다. 따라서 최신 V7 sparse 관측의 부족한 코너를 부분선으로 보완하는 이 지역 C2는 그 실행들과 구분한다. V8의 같은 sparse q에 대한 masked robust / ordinary / no-mask robust 대조는 보존하며, 이 문서는 그 대조를 반복하거나 대체하지 않는다.

## 입력과 관측의 고정

입력은 봉인된 V7 `IMAGE_ROLE_BOUNDARY_ONLY`의 `input_points`, K, 물리 W/H/D, 원영상 크기, initial N3 R,t, 초기 H, 관측 계약 및 같은 프레임의 full IMAGE_ROLE observation row다. `parent.native_points`는 보정 후 display이므로 원본 N3로 쓰지 않는다. 원본 N3는 observation의 `native_N3_points`를 사용하고, fit 결측을 채우는 데 사용하지 않는다.

드라이버는 V7 전체 geometry1,960행 / observations735행 / fixed490행 / cohort245 ID, 봉인·cleanup·추론 receipt, 공개 코드와 입력 SHA를 모두 검증한 뒤 한 프레임씩 pure packet으로 전달해야 한다. V8 point-only control245행과 its seal/receipt/standalone 검산도 별도로 byte bind한다. pure 함수의 행·필드 검사는 전체 파일 provenance 증명이 아니다. GT와 사람 상태는 봉인·cleanup 및 독립 기하 검산 PASS 이전에 읽지 않는다.

[pipeline.py](../../../scripts/research/pallet_sparse_local_line_20261010_v9/pipeline.py#L1)는 V8의 parent packet 검사에 추가로 원 V7 flat observation의 frame/session/head, GT_input=False, 초기 N3 pose/H, 원본 native N3, logits hash, actual support query 좌표와 source CAL coverage를 확인한다. parent sparse q의 각 실제 코너와 supporting edge identity를 원 observation에 연결한다. 기존 head logits·decode·CAL threshold를 다시 계산하지 않는다.

점 factor는 eligible이고 H에 속하지 않는 실제 sparse 코너 ID만이다. 결측은 NaN으로 유지한다. 중심점8은 fit에서 제외하고 display에서 보존한다. 같은 실제 채택 코너를 만들었던 두 edge는 선 factor에서 소비한다. [V5 `_lines`](../../../scripts/research/pallet_partial_line_independent_20261010_v5/solver.py#L144)를 그대로 재사용해 동일 semantic edge 중복을 collapse하고 불일치 중복은 계약 실패로 처리한다. 실제 최종 support와 반경의 일관성, native endpoint identity, 기존 actual source-wire registry coverage를 확인한다. H인 3D endpoint를 포함하는 실제 보이는 선은 버리지 않으며, 그 endpoint의 초기 2D 좌표는 읽지 않는다.

선 하나는 독립 semantic edge factor 한 개다. 등록된 3D endpoint 두 개를 현재 R,t로 투영한 뒤 실제 관측선에 대한 법선 거리 두 성분을 √2로 나눈다. 선의 여러 query, 선에서 만든 가짜 점, 해당 선으로 만든 코너와 원선의 중복을 독립 측정으로 세지 않는다. 남은 코너가 공유한 source edge의 상관은 원 관측 계약에 보존한다. source CAL 지원과 직선성은 실사의 물리적 edge 소유권 인증이 아니다.

## 지역 optimizer와 해 판정

[solver.py](../../../scripts/research/pallet_sparse_local_line_20261010_v9/solver.py#L1)의 numeric bank는 V4 등록 치수와 V5 factor validator를 사용한다. V4/V5의 4점 solve는 호출하지 않는다. fixed 전체 actual point+unused line pool이 모든 시작점과 치수 후보에 동일하다.

optimizer 시작점 원천은 봉인된 initial N3 pose와 **available인** 봉인 point-only pose의 최대 두 개다. 각각 physical R,t를 기존 W/D convention으로 확인하고 두 등록 치수에 모두 변환해 시작한다. 초기 치수 선택을 final constraint나 선택 prior로 쓰지 않는다. 초기 R,t, hidden 좌표, 정답, 재투영 좌표·선을 residual prior 또는 관측으로 추가하지 않는다. 랜덤 시작점과 새 P3P는 없다. 정사각형 치수가 deduplicate되면 기존 registry대로 한 치수만 사용한다.

전체 scalar 관측이 6개 미만이면 `INSUFFICIENT_LOCAL_FACTORS`다. 이는 진짜 관측 부족이고 4점 게이트의 숫자를 바꾸는 방식이 아니다. 부분선이 없어도 코너3개가 공급한 6개 scalar 관측이면 원 v1 C2의 point-local 정제를 수행할 수 있다. 이 경우는 **선의 효과와 분리해서 보고**한다. 코너0–3개+부분선 경로와 코너3개+선0 경로 모두 독립 PnP/PnL 또는 global uniqueness로 소개하지 않는다.

고정 수학은 기존 원 v1/V5의 `least_squares(loss='soft_l1', f_scale=8, max_nfev=50)`와 point weight1, line component1/√2다. exact forward projection Jacobian을 optimizer에 제공한다. 매 source×등록치수의 모든 logical start를 기록하고, 정확히 동일한 dimension/z0와 동일 factor pool의 fit만 cache reuse한다. 최대2 source×2 등록치수의 4개 logical start이며, 모든 fit·실패·cache reuse를 실행량에 기록한다.

초기 observed/model-normal J rank는 진단뿐이며 start를 veto하지 않는다. 초기 modeled edge projection의 collapse 때문에 modeled-normal 진단이 불가능해도 typed reason을 보존하고 실제 observed-normal residual/J가 유효하면 optimizer를 수행한다. 실제 residual/J의 수치 실패는 해당 start의 실패로 기록하며 다른 고정 start는 계속 처리한다.

최종 후보는 optimizer success, 유한 R,t/residual/J, 모든8 모델점의 positive camera depth, **actual pool의 observed-normal exact J와 현재 modeled normal을 고정한 geometry J가 각각 rank6**인 경우만 유효하다. column L2 normalization과 relative rank threshold1e-10은 V5 그대로다. SciPy robust loss로 수정한 `fit.jac`만으로 geometry rank를 판정하지 않는다. 초기나 최종 모형 투영선 자체를 새 이미지 관측으로 바꾸지 않는다.

유효 후보 선택은 같은 전체 pool의 soft_l1 cost 최소값이다. 원 v1의 `fit.cost` 목적함수를 유지하며, 원 residual에서 `64×sum(sqrt(1+(r/8)^2)-1)`로 다시 계산한 cost와 대조한다. untruncated scalar SSE, point/line factor SSE, truncated SSE, point norm≤8px 및 semantic line RMS≤8px의 inlier는 진단으로 함께 저장한다. inlier4개를 새 승인 조건으로 넣지 않는다. 최종 실제 자세 성능은 이 수치적 산출과 별도 평가한다.

`NEW_POSE`는 여기서 수치적인 **LOCAL 자세 산출**을 뜻하며 올바른 대응의 충분성이나 정확한 자세, 최상위 목표 달성의 성공 판정이 아니다. 전체 fixed factor pool의 rank6와 별도로 최종8px 진단 inlier만의 scalar 관측 수, observed/model-normal J·rank를 모든 유효 후보에 저장한다. inlier scalar가6개 미만이거나 그 두 J의 rank가6이 아니면 관측 지지가 부족한 수치 산출로 식별한다. 이 진단의 실패를 잘못된 실제 T/R과 분리해서 기록하며, real 성능을 확인하고 새 inlier gate를 추가하지 않는다. 올바른 대응인지 여부는 GT가 허용된 후 원행과 실제 자세·좌표 오차로 평가한다. 잘못된 대응들이 일관된 full rank 자세를 만든 경우를 목표 성공으로 처리하지 않는다.

모든 optimizer attempt와 최종 candidate, 시작 원천, 치수, actual factor IDs, scalar residual, full model projection, physical R,t, 두 J, singular values/rank/condition, cost/SSE, 모든 대안을 보존한다. 서로 다른 후보의 physical R/t와 all8 model projection을 기존 V4 수치 equivalence 규약으로 비교한다. 서로 비등가인 두 local 해의 whole-pool cost가 기존1e-8 수치 tolerance로 동점이면 `AMBIGUOUS_LOCAL_POINT_LINE`으로 unavailable 처리한다. 점수 차이가 있는 다른 local 해도 기록한다. 저장된 starts에서 최저값을 찾았다는 사실은 발견되지 않은 다른 basin 또는 독립 global unique 해의 부재를 보장하지 않는다.

## 출력, 평가와 실행량

새 local R,t가 승인되면 H 초기 2D 좌표를 최종 fit에서 제외한 상태로, 최종 모델 projection의 H 좌표로 output을 교체한다. 그 점을 다시 fit하지 않는다. 직접 관측 코너는 선택된 실제 sparse 좌표이고 결측 코너의 native N3 display는 numeric fit에 참여하지 않는다. 실패시 unchanged initial N3 기본 자세와 원본 native display 반환을 fallback으로 기록하고, 기본 자세도 없으면 완전 실패로 구분한다. mask가 바뀌거나 한 점 틀린 사실만으로 프레임을 삭제하지 않는다.

모든245장을 유지하여 NEW local pose / 기본 반환 / 완전 실패, point count·배치·실제unused line·진단 inlier·rank를 보고한다. `U<4 && unused_line_count>0`에서 NEW인 경우와, `unused_line_count=0`에서 point-local NEW인 경우를 분리한다. point-only보다 신규 산출이 늘어도 실제 T/R/ADDsym이 좋아졌다는 결론으로 바로 바꾸지 않는다. 전체 운용 결과, 서로 새 자세를 산출한 공통집합, 직접 가시점 손상과 H 재투영 오차를 구분한다. 평균·표본분산·표준편차·중앙값·P90/max와 고정 comparator의 paired 차이는 원행에서 계산·독립 검산한다.

고정1arm의 주 comparator는 같은 sparse 관측의 V8 `ROLE_BOUNDARY_H_ROBUST`다. Base와 N3→cornerSubPix의 원래 기준선도 함께 보존해 실제 우위를 평가한다. source/test/dev 노출된 데이터임을 밝히며 새 unseen 평가라 부르지 않는다. GT는 점·선 선택, 시작점, 치수 선택, rank·cost·threshold·fallback에 사용하지 않는다.

CPU 계약 fixture는 CPU 실행 전에 별도 protocol/STARTED/code SHA에 고정한다. 독립 기하 checker의 코드도 실사 실행 전의 core protocol에 bind하지만, 해당 checker 자체의 protocol은 **완료된 실사 geometry의 봉인 뒤, checker 자체 산술과 GT 평가 전에** freeze한다. CPU 검사는 point0–3+actual/CAL lines, point3+line0의 local-only와 다중해 처리, <6scalar 부족, noisy parallel선의 observed phantom rank와 modeled rank 차이, 초기진단불가에서 최종 rank6로의 회복, line/corner 중복, H 좌표 불변·재fit금지, 두 등록치수와 seed remap, 비등가 cost 동점, optimizer failure, source/CAL/wrong-packet rejection을 확인해야 한다.

CPU 실행의 실제 통과·실패·수치 호출량은 각 원래 receipt를 보존해 확인한다. fixture의 잘못된 기대를 정정한 보충 검사와 그 결과를 원래 검사에 연결하는 별도 검산을 수행한 경우, 원래 실패를 성공 receipt로 덮어쓰지 않는다. core freeze는 원래 실행과 보충 실행의 코드·protocol·STARTED·receipt 및 명시적인 joined 검산 PASS를 모두 bind해야 한다. 본 계약의 고정은 실사 기하와 GT 성능 평가보다 앞서며, CPU 통과만으로 실사 산출 또는 성능 개선을 주장하지 않는다.

실제 드라이버는 full original witnesses·원행·ledger를 스트리밍 기록하고, quiet resource guard·모델/GT/asset canary·OpenCV primitive와 SciPy 진입 카운터·cleanup·입력 보호를 기록한다. geometry와 실행 receipt/cleanup을 봉인한 후에만 GT 채점한다. 기존 head/model forward, 최초 initial PnP, CAL/decoder, 학습, 합성 생성은 이 경로에서 모두0이다. local optimizer의 실제 진입·완료·실패·residual/J 호출과 primitive 호출은0으로 숨기지 않는다. stored-observation replay wall time은 배포 전체 경로 latency가 아니며 latency 개선을 주장하지 않는다.
