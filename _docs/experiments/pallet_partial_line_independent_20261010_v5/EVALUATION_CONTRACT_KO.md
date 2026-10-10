# 같은 IMAGE_ROLE 관측의 부분 선 절제 — 실행 전 고정 계약

이 문서는 결과가 아니다. C2에 허용된 부분 선 제약을 별도 경로로 한 번 비교한다. 새로운 경계 선택기, 학습, source RGB, seed, 임계값 탐색을 추가하지 않는다. v4의 기존 관측 선택과 point-only LOO는 그대로 유지한다.

쉬움153+중간92=245장/13세션만 새로 실행한다. Severe74가 들어 있는 과거319장 기록은 보존한다. 평가 참조는 기존 GEOMETRIC_PROXY이며 독립 물리 실측 자세가 아니다. 모든 새 기하 출력을 봉인하고 기존 대조의 parity를 확인한 뒤 참조·사람 상태를 읽는다. 배포 함수는 BGR 원영상, K, 실제 W/H/D, 영상 identity만 받으며 GT·등급·사람 가시성·점수 cache는 받지 않는다.

주경로 `N3_INDEPENDENT_CORNERWISE_ROLE_PARTIAL_LINES`를 고정 N3→cornerSubPix, 같은 IMAGE_ROLE의 변경 없는 v4 point-only 경로, 경계 교체 없는 H 강건 경로, 무마스크 강건 경로, Base와 비교한다. point-only 선택은 주경로와 point-only 대조 사이에서 공유하므로 서로 다른 후보를 만들어 선의 효과를 부풀리지 않는다. 최종 전체245 운용 위치·회전 평균이 모두 고정 N3보다 작아야 주 목표 개선으로 판정한다. 새 자세 산출, N3 기본 반환, 완전 실패를 구분하며 기본 반환·큰 오차도 전체 평균에 포함한다.

선은 같은 calibrated IMAGE_ROLE `.lines`에서 고른다. 각 선의 최종 support 잔차가 저장된 query 허용 반경을 만족해야 한다. 실제로 채택해 fit에 공급하는 경계 코너의 두 원천 edge는 선 factor에서 제외한다. 동일 edge는 한 번만 센다. raw `.partial_lines`는 downstream에서 거절된 코너의 edge까지 제거하므로 최종 관측 pool로 그대로 쓰지 않는다. Native N3 점은 해당 TLS 선에서 만들어진 점이 아니다. 가려진 초기 2D 좌표는 fit에 쓰지 않으며, 선의 등록 3D 끝점이 H라는 이유로 실제 관측 선을 삭제하지 않는다.

수치 솔버는 초기 R/t·재투영·초기 치수 분기를 받지 않는다. H를 제외한 실제 점이 4개 이상이면 v4의 같은 finite4 부분집합, 두 치수 가설, 반환된 모든 표준 해를 시작 후보로 사용한다. 초기 자세의 dimension을 정답으로 고정하지 않는다. 4점 미만에는 검증된 독립 PnL 생성기가 없으므로 관측 부족으로 기록한다. 선을 여러 가짜 점으로 바꾸어 최소 점 수를 맞추지 않는다.

점 factor 잔차는 2D 투영 차이이다. 선 factor는 등록 edge의 두 끝점 A/B를 투영한 뒤 관측 단위 법선 n/offset c까지의 거리 `[n·q(A)-c,n·q(B)-c]/sqrt(2)`이다. 선 하나의 RMS를 하나의 factor 잔차로 채점한다. 점 하나와 edge 하나를 각각 한 inlier 단위로 센다. endpoint 둘 또는 여러 query를 여러 독립 inlier로 세지 않는다. 고정 8px, 전체 factor inlier 수 우선, 동일 공통 pool의 truncated SSE 다음 순서를 사용한다. 이 정책은 4점+7선 합의를 7점+0선보다 우선할 수 있는 명시적 정보 융합 절제이며, point-only inlier 우선 정책과 같다고 주장하지 않는다. 신규 자세에는 최종 실제 point inlier가 4개 이상이어야 한다.

최대3개의 서로 다른 물리 해만 fixed soft_l1/f_scale8/max_nfev50으로 정제한다. point weight1, edge 두 scalar의 정규화 sqrt(2), rank 상대 임계1e-10을 고정한다. 결과에 point/line inlier 수, 원천 query·edge, consumed edge, 잔차, 시작 해와 대안 해, 수치 rank를 남긴다. observed-normal Jacobian과 현재 model edge 법선으로 계산한 기하 Jacobian을 구분한다. noise가 rank를 가짜로 늘리는 경우를 검사한다. rank6은 전역 유일성·물리 소유권 증명이 아니며, 수치 동점인 다른 물리 해는 AMBIGUOUS로 처리한다. 다른 초기 H와 재추정 후 H의 차이는 진단이고 전체 프레임 veto가 아니다.

새 최종 R,t가 구해지면 H의 초기 좌표를 그 자세의 재투영으로 교체한다. 재투영점을 다시 fit하지 않는다. 같은 선에서 만든 경계 코너와 선 factor를 중복 관측으로 세지 않는 검사를 수행한다. H의 raw coordinate를 큰 오답·sentinel·NaN으로 바꾸어도 고정된 다른 입력에 대한 수치 해가 바뀌지 않는 단위검사를 먼저 한다.

평균·표본분산(ddof1)·SD·중앙값·P90·최대, 공통 운용/새 자세 조건부/양쪽 새 자세 집합, 고정10000×13 session draw의 CI를 원행에서 계산한다. 직접 가시 코너 손상, H 재투영 오차, 마스크 오류와 자세 개선의 교차표를 분리한다. 결과를 본 뒤 설정·가중치·seed를 바꾸어 반복하지 않는다. 실사 edge의 물리적 소유권과 source 네 통제 변형, role3 입력의 독립 replay는 이 절제가 인증하지 않는다.

시간은 다른 수치 작업이 없는 상태에서 4경로×(warmup20+측정130)=600번의 fresh API 호출로 측정한다. detector/N3/초기 pose/ROLE/LOO/final point+line solve/H 재투영/출력 조립을 포함한다. 저장 좌표 재생·기존 시간 합산을 latency로 쓰지 않는다. 원본 사용자 checkout과 기존 게시 파일을 해시로 보호한다. 코드·원행·검산·그림·실행량은 전용 연구 브랜치에 정상 commit/push하며 main 변경·merge·force push는 하지 않는다.
