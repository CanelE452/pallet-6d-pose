# V13 독립 기하 검산

`validation_checks.py`는 완성된 245행 기하·245행 실행 원행, 최종 seal·cleanup 영수증, 원래 IMAGE_ROLE 관측과 V10 endpoint 판정, 자체 소스를 먼저 고정한 뒤 계산한다. 명령은 `python -I -S -B scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/validation_checks.py freeze`와 같은 경로의 `run`이다. 출력은 `VALIDATION_PROTOCOL.json`, `VALIDATION_STARTED.json`, `VALIDATION_CHECKS.json`이며 기존 파일은 덮어쓰지 않는다.

프로덕션 모듈을 import하지 않는다. 고정 SHA의 기존 V9 stdlib 검산기에서 허용한 회전·투영·투영 Jacobian·원래 물리 경계 관측 검증 함수만 AST로 추출한다. 유한 선분의 단위 접선, 종방향 위치, 양 끝 밖의 hinge와 `du/dL`을 별도 scalar 식으로 재구성한다. 모델 normal은 기존 정의대로 미분 중 고정하며, 접선과 선분 길이의 미분은 포함한다. 관측 support를 새로운 3D 대응점으로 만들지 않는다.

제안 자세와 실제 합의점·합의선 재적합 자세를 각각 전부 검사한다. 전체 원래 관측의 점 오차·edge block 오차, 하나의 edge당 한 표, 8px 판정, truncated SSE와 선택 순서를 검산한다. 재적합에 실제 사용한 관측만의 두 Jacobian과 잔차·soft-L1 비용을 재구성한다. 저장된 singular spectrum은 rank·기존 `1e-4` weak 판정·column normalization·Frobenius energy를 검산하며 새 SVD를 실행하지 않는다. 이 검사는 전역 유일해나 물리 경계 소유권을 증명하지 않는다.

고정된 V10 `CONTRADICTED_REJECT`만 제거하고 `UNKNOWN`을 유지하는지, 실제 코너가 소비한 edge가 선 관측으로 다시 들어오지 않는지, H 초기 좌표가 모든 점 fit·score에서 제외되는지 검사한다. 새 자세는 H를 한 번 재투영하고 다시 fit하지 않는다. 기본 출력 반환은 원래 N3 좌표와 초기 자세 전체를 보존해야 한다. center는 변경하지 않는다.

실제 optimizer·callback·OpenCV·SVD 실행 원행의 합과 전체 영수증을 조인한다. 캐시 재사용은 동일 초기 자세·차원·fit IDs의 저장 실행 증거로 검사한다. 수치 실패, 관측 부족, weak, 재적합 후 불안정, 혼합 실패, 다중 관측 합의를 구분한다. 자원·보호 snapshot의 기록 일치와 cleanup을 확인하며 과거 하드웨어의 진정성이나 순간적인 경합 부재를 독립적으로 입증했다고 주장하지 않는다.

CPU 원본 15검사 중 14 PASS와 `6 != 7`인 한 실패는 보존한다. 추가 올바른 점 제외를 허용하도록 검사기의 정확히 한 assertion을 `>=4`로 바꾼 별도 review의 실제 15 PASS를 소스·프로토콜·영수증에 묶는다. 잘못된 점 제외와 자세 오차 assertion은 유지한다. 원본 테스트·솔버·관측·임계값을 바꾸거나 원본 실패를 PASS로 표기하지 않는다. 검산기는 모델·영상·GT·PnP·optimizer·SVD·학습을 새로 실행하지 않는다.
