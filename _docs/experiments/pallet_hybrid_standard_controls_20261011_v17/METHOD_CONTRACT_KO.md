# V17: 같은 일반 PnP에서 원래 N3와 고정 hybrid 관측 비교

이 계약은 새 관측 선택기를 만들지 않고, 이미 고정된 V7 `IMAGE_ROLE_CORNERWISE_HYBRID`의 실제 좌표 선택이 원래 N3 좌표보다 유용했는지 같은 일반 PnP로 확인한다. 이전 단순 predicted-H 일반 PnP의 개선과 후속 복잡한 방법의 결과를 구분할 필요가 있다. 역사적 V1 일반 솔버는 현재 V4의 동점 다중해 거절과 같지 않으므로 그 결과를 V17 결과로 바꾸어 인용하지 않는다. V7 hybrid 최종 경로와 이후 같은 좌표 대조는 강건 솔버였고, V8의 일반 솔버 대조는 boundary-only 좌표였다. 여기서는 원래 N3와 hybrid를 같은 V4 일반 솔버로 비교하는 빠진 고정 대조만 수행한다.

사용자가 현재 지정한 쉬움 Clean153과 중간 Moderate92의 전체 245장을 유지한다. 첨부 문서의 원래 319장 지시 및 과거 Severe74 결과는 보존하지만 이번 평가 모집단에 Severe를 넣지 않는다. 알려진 DEV 모집단이며 미관측 실사 평가가 아니다. 파일 내 지시문과 실제 사용자 범위 변경을 구분한다.

1. `HYBRID_STANDARD_PNP`: V7 봉인 행의 `input_points`를 그대로 사용한다. 코너별 원래 `H∪{k}` LOO 선택, native 대 boundary 비교와 채택 ID를 다시 실행하거나 바꾸지 않는다. 채택되지 않은 좌표는 원래 `OBS.native_N3_points`이며 보간·재투영·빈 좌표 채움이 아니다.
2. `NATIVE_STANDARD_PNP`: 같은 원래 `OBS.native_N3_points`만 사용한다. head는 최종 fit에 관여하지 않고 채택 boundary ID는 빈 목록이다. parent의 선택 계약은 provenance로 따로 보존한다.

두 경로 모두 동일한 초기 H, K, 물리 치수, 원영상 크기와 변경 없는 [V4 `PoseBank.solve`](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py)에서 `hidden=H, robust=False`를 사용한다. 새 알려진 치수 prior는 없으며 두 등록 축 후보를 동일하게 허용한다. H와 중심 ID8의 초기 영상 좌표는 fit·점수에 넣지 않는다. 초기 N3 pose는 H의 출처와 실패 시 기본 출력에만 쓰며 자세 잔차 prior·해 선택 prior가 아니다. 같은 좌표·K·치수·실제 ID 집합의 수치 가설은 한 in-memory bank에서 재사용한다. 좌표가 다른 두 입력은 별도 bank다. bank 이전/이후와 실제 OpenCV 호출을 함께 남겨 재사용을 검산한다.

일반 PnP의 목적함수는 전체 유효 `U`의 SSE이다. 최소 4개 실제 대응, 영상/형상 배치, 양의 깊이, 전체 U의 자세 Jacobian rank6과 변경 없는 수치 동점·물리 비동일 다중해 처리를 사용한다. 4/5점과 평면/다중해 처리는 이미 봉인된 V8 CPU 10그룹의 성공 B 및 실패 A를 함께 결속한다. 기존 테스트를 이번 작업으로 다시 실행했다고 세지 않는다. 일반 솔버의 8px inlier 수는 진단이므로 강건 솔버의 최소 4-inlier 게이트를 몰래 추가하지 않는다. 수치 NEW가 실제 정확한 대응이나 목표 달성의 증명은 아니다. 부족 관측, 배치 퇴화, 수치 생성 실패, rank 부족, 동점 다중해를 원 상태로 보존하고 최종 평가에서 기본 출력 반환·완전 실패와 분리한다.

NEW R,t가 있으면 초기 H의 좌표를 최종 재투영으로 바꾸며 재투영점을 다시 PnP에 넣지 않는다. V4 solver와 변경 없는 V2 assemble이 저장된 투영을 사용하고 새 fit을 수행하지 않는다. 자기 가림 제외 좌표를 초기 값으로 유지하는 대체 경로는 없다. 새 자세가 없으면 원래 전체 N3 좌표와 초기 pose를 기본 출력으로 돌려주며 이 반환을 NEW로 표시하지 않는다. H 재추정 차이는 기록하고 프레임 실패 조건으로 삼지 않는다. 중심·confidence·box·선택 candidate metadata를 보존한다.

이번 대조는 선 관측을 fit하지 않으며 endpoint LOO·선 consensus·LOCAL·새 RGB/head/decoder/학습/seed/threshold/weight 변경을 모두 0회로 고정한다. 부분 선의 물리 ownership을 진실로 인증하지 않는다. 같은 q를 새 숫자 설정으로 반복 탐색하는 작업이 아니라 두 관측 공급을 고정한 paired ordinary 대조다.

실행 전 새 protocol에 코드 SHA, 현재 게시 HEAD의 별도 보호 snapshot, 기존 V7 1,960개 기하 행·735개 OBS·490개 fixed 행의 전체 245 frame ID/order 및 원래 hybrid 245행을 결속한다. 실제 입력인 V7 원좌표와 hybrid 선택 좌표에는 V7 독립 point scalar 검산의 실제 PASS를 요구한다. V16의 완료 245/490 geometry seal과 정리 완료는 별도 계보로 확인한다. 원 V16 scalar 환경 실패 `NameError: ast`와 원 protocol/STARTED/실패 기록·checker를 보존하며, 환경만 교정한 별도 `VALIDATION_READY_*`도 완료 198,134,007개 검사 중 564개 실패, `passed=False`인 실제 결과 그대로 결속한다. V17은 V16의 추정 자세·선 Jacobian·LOCAL 결과를 입력으로 사용하지 않으므로 V16 선 미분 검산의 PASS를 필요조건으로 두지 않는다. 이는 독립적으로 가능한 점 전용 대조의 의존성 정정이며 좌표·일반 PnP·임계값·선택 정책 변경이 아니다. 원 실패를 성공으로 다시 표시하지 않는다.

root만 독립 quiet preflight 이후 실제 490 일반 solve를 실행한다. [새 pipeline](../../../scripts/research/pallet_hybrid_standard_controls_20261011_v17/pipeline.py)과 [runner](../../../scripts/research/pallet_hybrid_standard_controls_20261011_v17/runner.py)는 245개의 두 경로 원행과 245개 전체 bank ledger, 실제 primitive/J/SVD의 entry·return·exception, 복구·자원·prefix 실패 기록을 저장한다. stream fsync/promotion과 환경·함수·OpenCV thread 복구를 완료한 뒤 geometry seal을 만든다. 독립 scalar checker의 별도 freeze/PASS가 있어야 후단 GT 접근·실제 T/R/ADDsym 평가가 가능하다. 원행·전체 운용과 공통 산출집합을 모두 보존한다. replay 경과 시간은 fresh detector/head/초기 자세를 포함하지 않으므로 배포 latency로 보고하지 않는다.

후단 비교는 hybrid−native, hybrid−고정 N3, native−고정 N3의 방향을 사전에 고정한다. 음수는 오차 개선이다. 기존 점수 커널·같은 13 session bootstrap draws를 변경하지 않고 실제 245 원행에서 검산한다. N3 단계에 정합한 geometry-resolved 실사 reference는 독립 촬영으로 확보한 물리 ground truth와 구분한다. 낮은 숨은 코너 오차나 공급 q의 8px proxy 정확도만으로 위치·회전 개선 또는 learned head의 필요성을 주장하지 않는다.
