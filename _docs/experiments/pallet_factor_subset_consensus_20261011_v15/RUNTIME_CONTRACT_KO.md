# V15 새 전체 경로 시간 측정 계약

이 문서는 고정 V15 방법의 추가 시간 측정 계약이다. 계약을 작성한 사실은
실행 완료를 뜻하지 않는다. 실제 실행량과 성공 여부는 별도 own protocol,
STARTED, 원행, receipt 및 후단 독립 검산에서 확인한다. 기존 solver, 정책,
학습과 평가 입력을 변경하지 않으며 known DEV의 GEOMETRIC_PROXY 검증이다.

기존 26장/13세션 패널을 고정한다. 워밍업 20호출 뒤 동일 패널을 5회 직렬
실행하여 공식 측정 130호출, 전체 fresh 호출 150회를 보존한다. 한 호출은
하나의 `FACTOR_SUBSET_FINITE_EDGE_CONSENSUS` 출력이다. 호출 사이에 이미지
캡처, head 관측, 가설은행, endpoint LOO 또는 자세를 공유하지 않는다.

RAM의 원본 RGB/K/치수를 입력받은 순간부터 최종 반환까지 CUDA 동기화와
`perf_counter_ns`로 측정한다. detector, 고정 N3→cornerSubPix, N3 초기
자세, Base 역할 특징, IMAGE_ROLE head, sparse point robust 경로, native
endpoint LOO, 모든 finite4 raw root의 직접 proposal 평가, mixed3 block
LOCAL proposal, 실제 inlier refit, 후보별 stable normal incidence 검사,
H 재투영과 전체 후보 메타데이터 보존을 포함한다. 초기 자세도 구간에 포함한다.

6개 stage는 `detector`, `N3_correction`, `initial_pose`,
`boundary_observation`, `endpoint_validation_and_filtered_local`,
`final_pose_reprojection_and_metadata`이다. 각 stage, 반환 처리와 전체
구간을 원 timestamp로 대조한다. 배포 경로의 반환 전 증거 복사 비용도 포함한다.
모델 로딩, 최초 파일 해독, 자원/보호 검사, 반환 후 parity, 로깅, GT 점수
계산은 구간 밖이다. 기존 시간 합산과 sealed 관측 replay 시간을 쓰지 않는다.

동일 fresh q/K/xyz/치수/4-ID의 Generic 가설을 다시 계산하지 않는다. 변경
없는 V7 `outcomes`가 생성하는 실제 PoseBank 생성자를 호출 구간 안에서만
관찰하고 정확히 한 bank 객체를 보존한다. 원 생성자를 `finally`에서 복구하고
그 bank를 V15의 `point_bank` 인자로 전달한다. 좌표/치수/hash/외부 prior
계약을 검증하며 bank before/after/delta와 cache hit를 저장한다. 추가 동일
4-ID Generic 호출과 cache miss는 0이어야 한다. 직렬화된 R,t로 cache를
재구성하지 않는다. point 경로의 available/best 판정으로 다른 raw roots를
삭제하지 않는다. 전체 fresh CV Generic 계수에는 point/native/초기 경로의
다른 실제 계산도 있으므로 전체 Generic 호출이 0이라는 주장은 하지 않는다.

N3 초기 자세는 mixed LOCAL 시작에만 쓰며 선택된 point 자세를 별도 LOCAL
시작으로 넣지 않는다. endpoint LOO는 H와 해당 edge endpoint를 함께 제외한
새 native 대응에서 수행한다. CONTRADICTED line만 제외하고 UNKNOWN은
남기며 실제 채택 point에 소비된 edge는 line factor로 중복 사용하지 않는다.
실험 fit/score에는 H 초기 xy와 center를 넣지 않는다. NEW일 때 H를 최종
R,t로 한 번 재투영하며 다시 fit하지 않는다. 불가 상태는 같은 fresh capture의
전체 N3 출력으로 반환하고 box/confidence/center/비선택 후보를 보존한다.

최초 실제 predict 안의 detector 내부 초기화 forward를 그대로 기록한다. 첫
호출 detector 2회, 나머지 149호출 각 1회로 실제 detector 151회/N3 150회,
IMAGE_ROLE hook 진입/반환 150회를 검산한다. 인위적 사전 forward는 0회이다.

실행 전 전체245 기하 seal/cleanup, 실제 독립 scalar PASS, postseal 평가
protocol/격리 score authority/245 scored output을 확인한다. 새 runtime
protocol은 코어·모델·보정·원본 입력과 evaluation closure 및 새 runtime,
deployment, 이 문서를 bytebind한다. 별도 quiet/온도 preflight를 요구한다.
저장된245 자세/좌표는 predict 인자가 아니며 반환 후 parity에만 사용한다.
GT/reference 파일 bytebinding은 예측 입력 또는 새 점수 계산을 의미하지 않는다.

원행은 시작/끝/6 stage timestamp, fresh 전체 prediction/원 detector 후보,
point bank ledger/실제 공유 bank witness, endpoint/전체 finite ledger와 실행
event를 보존한다. 호출별 11개 계수 그룹은 cv, optimizer, head, base, fresh,
local, native, model, svd, direct, bank_capture이다. CV6/SVD/SciPy와 residual/
Jacobian callback은 진입=반환+예외로 검산한다. 처리된 수치 예외도 삭제하지
않는다. local solver와 native endpoint bank, 공유 direct point bank 계수를
구분한다. SVD 계수는 전체 fresh 경로이며 normal guard만의 계수라고 부르지 않는다.

오류 시 첫 오류/완료 prefix/활성 packet을 보존하고 자동 재시도하지 않는다.
hook, pipeline, 계수 wrapper, point bank 생성자, OpenCV thread와 split
environment를 복구하고 원본 보호/자원 검사를 통과해야 공식 통계를 허용한다.
독립 시간 검산은 저장 timestamp·계수 일관성 범위에 한정하며 하드웨어 시계를
독립 계측했다고 주장하지 않는다. 수치적 NEW가 실제 정확도 성공을 뜻하지 않는다.
