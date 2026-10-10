# 고정된 cornerwise hybrid 관측과 V15 합의 솔버의 두 경로

이 문서는 새 실사 산술 전에 고정할 코드 계약이다. 아직 실행 결과나 개선 주장이 아니다. 첨부 원문 C1의 sparse boundary-only를 native 좌표로 채우는 변경이 아니다. 기존 C2의 별도 IMAGE_ROLE hybrid 관측 절제로, V7에서 이미 선택한 native RGB 대응과 실제 경계 대응을 함께 사용한다. 사용자 최신 평가 범위인 Clean153와 Moderate92 전체245장을 유지한다.

기존 V4 H∪{k} LOO native 대 경계 선택, V5/V6 hybrid 점·선 경로, V7 IMAGE_ROLE_CORNERWISE_HYBRID 점 전용 평가가 이미 수행된 사실을 유지한다. 누락된 조합은 동일한 V7 hybrid 관측에 최신 V15의 모든 finite4 raw root와 mixed3-factor 후보, 실제 inlier refit 및 candidate별 stable normal incidence를 적용하는 것이다. V4 선택이나 모델·CAL·decoder를 재실행하지 않는다. 이미 본 개발 자료로 수행하는 구조 절제이며 미관측 시험이 아니다.

입력은 GT-free V7 IMAGE_ROLE_CORNERWISE_HYBRID 원행, 같은 프레임 original IMAGE_ROLE OBS의 native_N3_points/lines/queries, 완료·정리된 전체 부모의 byte SHA와 population 증명이다. selected q/K/physical XYZ/초기 H/초기 N3 자세/채택 boundary IDs를 그대로 유지한다. 채택한 q는 original 두 incident-line 교점이고, 나머지는 original native N3 RGB 좌표이다. LOO 재투영 좌표를 관측으로 사용하지 않는다. native 채택을 사람 가시성·정답·실제 오차로 선택하지 않는다.

두 고정 방법은 `HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS`와 `HYBRID_FACTOR_SUBSET_POINT_ONLY`이다. 전자는 실제 채택 boundary q를 만든 sourceedge만 소비하고, native 대응은 edge를 소비하지 않는다. 남은 물리/CAL line pool에 기존 native endpoint LOO(H∪두 끝점)를 새로 수행한다. 변경된 pool에 V10 boundary-only 결정을 그대로 재생하지 않는다. CONTRADICTED만 제외하고 UNKNOWN은 남긴다. 후자는 정확히 같은 selected q에서 line factors만 모두 제거하며 채택 q의 출처는 부모 계약에 그대로 남긴다.

실제로 두 V15 bank solve를 수행한다. 동일 좌표/K/XYZ/크기/H의 하나의 prior-free PoseBank를 두 경로가 공유하여 모든 4-ID raw numerical hypothesis를 재사용한다. 두 번째 점 전용 경로의 추가 Generic 호출·cache miss는0이어야 한다. refit과 candidate incidence는 각 경로에서 실제 수행하며 점 전용 원행은 선 경로의 결과 복사본이 아니다. 최종 fit에는 H 초기2D와 center가 들어가지 않는다. 동일 line-derived point와 그 sourceedge를 중복 fit하지 않는다. 경계점이 후단 outlier가 되어도 이미 소비된 sourceedge를 새 관측으로 되살리지 않는다.

V15 solver.py를 그대로 호출한다. 두 등록 치수, 모든 finite4 root, mixed3blocks(line≥1), 초기 N3 LOCAL start만, soft_l1=8/max_nfev50, finite support hinge와 one-edge block, 8px consensus/cap64, 실제 inlier refit, normal rank1e−10/weak1e−4, 동일 합의의 물리 비동일 branch ambiguity를 변경하지 않는다. point-only 실제 fit은≥4점이다. 혼합 경로의6 normal scalar는 충분조건이 아니며 실제 rank/weak/depth 조건을 모두 검사한다. root 완전성과 global uniqueness는 증명하지 않는다. 성능 기반 threshold·weight·dimension·seed 검색은0이다.

프레임별 선 경로는 기존 V15 상한 M≤1140, D≤2 및 LOCAL calls≤2DM+R≤4560+R을 사용한다. 점 전용 경로는 mixed requests0이며 direct raw roots를 공유하고 실제 refit≤R이다. 따라서 두 경로 합의 보수적 LOCAL 상한은≤4560+2R이다. 실제 M/R/생성 실패/중복 cache/후보/refit/primitive·J·SVD·endpoint 호출과 각 상태를 원행·ledger에 보존한다. 이는 재생 solver 작업량이며 배포 전체 latency 주장으로 사용하지 않는다.

새 R,t가 승인되면 원 H를 한 번 재투영하고 다시 PnP/fit하지 않는다. unavailable/수치 실패/관측 부족/weak/다중해는 각각 solver witness와 출력 상태로 기록하며, 기존 N3 자세가 있으면 full N3 fallback이다. fallback의 숨은 좌표 유지와 승인 NEW에서 H 교체를 구분한다. NEW는 수치적 산출이며 실제 정확도 개선 인증이 아니다.

대조는 두 새 경로, 동일 q의 기존 V7 hybrid 점 전용, 기존 V15 boundary-only, Base/N3이다. 같은 hybrid에서 점 전용↔점·선이 선 효과를, 같은 V15 구조의 boundary-only↔hybrid가 관측 공급 효과를 설명한다. 기존 source physical-wire 감독 검증은 실사 ownership 인증이 아니며 실사 경계·코너 ownership은 UNKNOWN으로 유지한다. 본 코드 작성 단계의 신규 실사/모델/GT/학습/합성/수치 실행은0이며 실제 수행 권위는 root의 별도 protocol/STARTED/receipt이다.
