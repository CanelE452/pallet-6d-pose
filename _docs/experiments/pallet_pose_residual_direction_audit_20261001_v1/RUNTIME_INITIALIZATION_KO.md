# CPU 초기화 단계의 중단과 복구

첫 표현 감사 명령은 입력 검증 뒤, 모델별 검사에 들어가기 전에 종료 코드1로 중단됐습니다. threadpoolctl이 libc를 찾으려고 `/dev/null`을 읽기/쓰기 모드로 연 것이 감사의 쓰기 제한에 걸렸습니다. 모델별 SVD·충돌 결과와 출력 파일은 아직 없었습니다.

가상 배열로 먼저 확인한 뒤, CPU 라이브러리 제어를 감사 guard 설치 전에 초기화하는 얇은 실행 진입점을 추가했습니다. 원래 감사 함수·접근 제한·봉인 코드·수학·데이터·허용오차를 그대로 유지합니다. 기존 파일 삭제나 실험 가중치 재시작은 없습니다. 후속 실제 실행 결과는 [표현 감사](REPRESENTATION_AUDIT.json)와 [독립 검산](VERIFICATION.json)에 별도로 남깁니다.

[중단·복구 기록](RUNTIME_INITIALIZATION.json) · [실행 진입점](../../../scripts/research/pallet_pose_residual_direction_audit_20261001_v1/run_representation.py)
