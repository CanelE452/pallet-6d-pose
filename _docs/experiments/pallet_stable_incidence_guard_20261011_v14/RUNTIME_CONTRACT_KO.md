# V14 새 전체 경로 시간 측정 계약

이 파일은 추가 시간 측정의 실행 계약이다. 고정된 V14 기하/정책을 수정하거나
성능으로 설정을 선택하지 않는다. 실제 실행량과 성공 여부는 별도 protocol,
STARTED, 원행, receipt의 권위이며 이 문서의 존재가 실행 완료를 뜻하지 않는다.

동일한 기존 26장/13세션 패널을 사용한다. 워밍업 20호출 뒤 패널 26장을 5회
직렬 실행하여 공식 측정 130호출, 전체 새 호출 150회를 보존한다. 한 호출마다
하나의 고정 IMAGE_ROLE 모델 경로를 새로 수행하며 호출 사이에 캡처, 관측,
PnP 가설은행, endpoint LOO 결과 또는 자세를 공유하지 않는다.

RAM의 원본 RGB/K/치수를 입력받은 순간부터 최종 반환까지 CUDA 동기화와
perf_counter_ns로 측정한다. 포함되는 작업은 detector, N3→cornerSubPix,
원래 N3 초기 자세, Base 기반 역할 특징, IMAGE_ROLE head, 새 sparse point
강건 PnP, 새 native endpoint LOO, 새 필터링 관측의 유한 선분 합의 proposal와
실제 inlier refit, 자기 가림 재투영, 선택된 자세의 stable incidence 검사,
최종 좌표 및 후보 메타데이터 보존이다. 초기 자세 계산도 전체 구간에 포함된다.

6개 stage 이름은 V13과 동일하게 유지한다. 마지막
`final_pose_reprojection_and_metadata`의 timestamp를 guard 및 최종 후보
메타데이터 처리 뒤로 미루어 검사 비용을 포함한다. 모든 stage와 return의
합은 직접 측정한 전체 구간과 대조한다. 기존 시간의 단순 합산이나 sealed
관측 재생 시간은 처리시간으로 보고하지 않는다.

모델 로딩, 최초 파일 해독, 자원/보호 검사, 반환 후 저장 결과와의 parity,
로깅, GT 점수 계산은 측정 구간 밖이다. 원 detector의 첫 predict 내부 초기화
호출을 그대로 기록한다. 예상 actual detector forward는 첫 호출 2회,
나머지 149호출 각 1회로 151회이며 N3는 150회이다. 인위적인 사전 forward는
추가하지 않는다. head hook은 150회 진입/반환을 확인한다.

배포 구현은 변경 없는 V13 fresh pipeline을 한 번 호출하고 그 fresh 결과에
고정 G14.guard_row를 한 번 적용한다. 관측/초기 자세/후보는 accuracy 원행을
입력으로 사용하지 않는다. guard는 실제 선택된 fit의 점과 선 법선만 검사하며
추가 optimizer/PnP/후보 선택은 0회이다. guard가 허용한 NEW는 기존 H 재투영을
다시 배정하지 않는다. decline은 같은 fresh capture의 원래 N3 전체 출력으로
돌아가며 후보 box/confidence/center/비선택 후보를 보존한다.

실행 전에 전체245 기하 seal, cleanup, 독립42715 scalar PASS, postseal 점수,
새 evaluation protocol/source authority를 확인하고 별도 RUNTIME_PROTOCOL을
고정한다. protocol은 모든 코어/모델/보정/원본 입력 바이트와 새 runtime,
deployment 및 이 문서를 묶는다. 별도 quiet/온도 preflight를 요구한다.
첫 score dispatch의 전역 METHODS 충돌은 GT/STARTED 이전 0행 실패로 보존한다.
실제 점수 권위는 부모 전역을 바꾸지 않는 단일 local import AST review의 별도
protocol/STARTED/SCORING_AUTHORITY_SUPPLEMENT이며 이 바이트를 시간 계약에
함께 묶는다. 원 실패 authority를 실제 성공 authority로 바꾸어 설명하지 않는다.

원행은 시작/끝/stage timestamp, 실제 full prediction, 원 detector 전체 후보,
새 point bank/endpoint/finite/guard ledger, 각 호출의 CV6/SVD/SciPy/head/model
및 관측 경로 계수 차이를 보존한다. SVD 계수는 전체 fresh 경로의 실제 호출이며
guard만의 계수로 부르지 않는다. guard ledger의 source 계수는 별도로 구분한다.
SciPy와 CV/SVD는 진입=반환+예외를 검산하고 처리된 수치 예외도 삭제하지 않는다.

저장245 결과는 반환 후에만 parity 대조에 사용한다. 새 자세의 fit에서 H 초기
좌표와 center를 제외하고 NEW일 때 H 최종 좌표가 기존 새 자세의 재투영인지
확인한다. 실패하면 첫 오류와 완료 prefix/활성 packet을 보존한다. 종료 시 hook,
pipeline, 계수 wrapper, OpenCV thread 및 split environment를 복구하고 보호
바이트와 자원 검사를 완료해야 공식 통계를 허용한다. 자동 재시도는 하지 않는다.

known DEV의 GEOMETRIC_PROXY 검증이며 독립 실측 자세 정답 또는 보이지 않은
평가 집합의 성능을 증명하지 않는다. 시간 통계의 독립 stdlib 검산도 저장된
timestamp/계수의 일관성 범위에 한정하며 하드웨어 시계를 독립 계측했다고
주장하지 않는다.
