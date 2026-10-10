# V16 실제 전체 배포 경로 시간 계약

이 보충 측정은 V16 primary `HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS`의 실제 fresh 호출을 측정한다. accuracy 실험의 paired point-only arm은 이 호출에서 실행하지 않는다. 이미 저장한 좌표·endpoint 결정·자세를 추론 입력으로 사용하지 않으며, 저장 기하는 반환 후 재현성 대조에만 사용한다. 결과는 이미 알려진 DEV의 운용 측정이지 새 독립 정확도 평가가 아니다.

## 같은 호출에서 실제로 수행할 작업

1. 원 detector RGB 경로와 고정 N3→cornerSubPix, 초기 N3 자세, 원 Base feature 자세, 고정 IMAGE_ROLE head 및 source-CAL decoder를 실행한다.
2. immutable V7 `outcomes(IMAGE_ROLE_CORNERWISE_HYBRID)`를 실행한다. 원 V4 선택은 native 관측으로 `H∪{k}`를 제외한 동일 조건의 LOO에서 실제 경계 좌표를 검증한다. unavailable·tie·모호함은 해당 native RGB 관측을 보존한다. H와 center8은 새 fit에 넣지 않는다.
3. V7이 실제로 생성한 native/selected `PoseBank`를 scoped constructor observer로 보존한다. 두 좌표가 같으면 동일 bank 하나, 다르면 각각 하나이다. selected bank를 unchanged S15 solver에 전달하며 동일 selected q/K/치수/4-ID Generic 가설을 다시 계산하지 않는다.
4. 같은 native bank를 endpoint 생성 요청에 scoped adapter로 전달하여 `H∪{a,b}`의 실제 endpoint LOO를 다시 실행한다. 이미 생성한 native4-ID 수치 문제를 재사용한다. 이전 corner LOO와 다른 5/6-ID refit은 별도 수치 문제이므로 새 호출을 허용하고 정확히 기록한다. 아직 native subset bank가 built되지 않았다면 실제 새 native4-ID 생성도 숨기지 않는다. 새 cache key는 dim·ID로 기록하며 기존 key를 보존한다.
5. 같은 선택 q와 실제 사용하지 않은 physical/CAL 선에서 CONTRADICTED만 제외하고 UNKNOWN은 유지한다. unchanged S15의 raw4 모든 해·mixed3 부분집합·전체 관측 합의·실제 inlier refit·후보별 stable normal incidence를 실행한다. 값·치수·임계값·seed·loss·반복량을 변경하지 않는다.
6. NEW면 H를 새 R,t의 재투영으로 한 번 대체한다. 재투영점은 독립 관측으로 재fit하지 않는다. unavailable이면 선언한 전체 N3 기본 출력으로 돌아가며 상태와 새 자세 산출을 구분한다. center/confidence/box/비선택 후보를 보존하고 최종 prediction을 반환한다.

## 측정 범위와 실제 실행량

기존 13 sessions × 2 frames의 고정 26장 패널과 원 RGB hash를 사용한다. warmup20 + 5×26 measured130 = 실제 fresh150 호출이다. 각 호출의 detector/N3/head/원 V4 selection/point control/endpoint/S15/최종 H 대체와 metadata 보존은 CUDA 동기화 및 `perf_counter_ns`의 전체 t0–t1 안에서 실행한다. 파일 decode, 모델 loading, resource/protection probe, 반환 후 saved parity, logging, GT scoring은 밖이다. 원 runtime harness의 첫 detector 내부 warmup2 forward와 이후149×1을 따로 기록하고 N3/head는 호출당1을 확인한다. 임의의 사전 forward를 추가하지 않는다.

다른 실험이나 시간 측정과 겹치지 않도록 독립 preflight와 측정 중 기록된 quiet/temperature 관측을 요구한다. 기록 사이의 일시적 경합은 인증하지 못한다. 측정 receipt의 환경·하드웨어 값과 실제 전체 시간만 보고하고, 캐시 재생 시간·accuracy solver replay wall time·기존 시간 단순 합산을 배포 latency로 쓰지 않는다.

6개 OpenCV primitive·실제 SciPy optimizer/callback·SVD의 entries/returns/exceptions, 실제 native/selected constructor lifecycle, 새 numeric cache key, 원 corner LOO calls, head hooks, local/direct/native 연산을 원행마다 기록한다. native=selected로 bank 하나를 공유할 때 P15의 end-of-repair native ledger에는 direct cache hit도 포함될 수 있다. 원 cumulative ledger를 보존하고 마지막 endpoint record의 `bank_after`로 endpoint phase를 분리하여 direct/native를 중복 집계하지 않는다. 반영된 P16 accuracy policy 사본의 model/selection0은 sealed replay의 설명이며 fresh 실행량은 이 측정의 별도 실제 계수가 권위이다.

## 별도 게시 및 실패 보존

complete245/490 GT-free 기하 seal·독립 scalar PASS와 postseal scoring490 완료 후, 새 `RUNTIME_PROTOCOL.json`에 코드·배포·계약·모델/교정·원 입력·패널·이미지 hash·정확도 원행·검산·보호 snapshot을 고정한다. `RUNTIME_PREFLIGHT`, `RUNTIME_STARTED`, 전체 timestamp/tick 원행 및 resources/receipt는 배타적으로 생성한다. 첫 오류·active packet·부분 gzip·restore 오류를 보존하며 자동 retry하지 않는다. scoped constructor·primitive·optimizer·SVD·head hook·threads·환경을 finally 복원하고 protected bytes를 대조한다. 전체150 및 복구 성공 전에 공식 시간 통계를 승인하지 않는다.

이 코드 작성 단계에서는 모델/이미지/PnP/SciPy/SVD/GT/시간 측정을 실행하지 않았다. 원행 scalar 시간 검산은 별도 helper를 측정 후 고정하고 수행한다. 이 계약은 정확도 개선, 실제 edge ownership, 물리 자세 truth 또는 모든 local root 탐색 완전성을 인증하지 않는다.
