코너 역할 정의와 실제 경계 대응, 선 조립, N3 좌표 교체 사이에서 발생한 문제를 기존 원행으로 조사한 감사입니다. 정확도 개선을 달성한 새 방법의 결과가 아닙니다.

먼저 [결과와 그림](RESULT_KO.md)을 읽습니다. 실사 범위는 기존 쉬움 153장과 중간 92장, 총 245장입니다. 어려움 74장과 역사적 319장 결과는 보존했습니다.

| 확인할 내용 | 실행 증거 |
| --- | --- |
| 482개 교점의 선 위치 오차와 교점 증폭 분해 | [원행](ROWS.jsonl.gz), [집계](RESULTS.json), [사전 잠금](PROTOCOL.json) |
| 별도 표준 라이브러리 산술 검산 | [MECHANISM_CHECKS](MECHANISM_CHECKS.json) |
| source 35,011개 양성의 ID·bin·wire·기하 계약 | [SOURCE_CONTRACT_CHECKS](SOURCE_CONTRACT_CHECKS.json) |
| CAL 348개 가상 교점과 원래 PnP 코너 정의 대조 | [원행](CAL_CORNER_CONTRACT_ROWS.jsonl.gz), [검사](CAL_CORNER_CONTRACT_CHECKS.json) |
| source RGB 3채널의 FP16 2,096,640개 수치 일치 | [완료 검사](rgb_resume/RGB_CONTRACT_CHECKS.json), [실행 인수인계](rgb_resume/ARTIFACT_LEDGER.json) |
| RGB 감사의 첫 사전 검사 실패 | [보존된 실패](RGB_CONTRACT_CHECKS.json), [원 protocol](RGB_CONTRACT_PROTOCOL.json) |
| 실제 GPU API 생성→예측→종료 두 차례 | [검사](DEPLOYMENT_SMOKE_CHECKS.json), [원행](DEPLOYMENT_SMOKE_ROWS.jsonl.gz), [사전 잠금](DEPLOYMENT_SMOKE_PROTOCOL.json) |
| 그림·보고서와 원행의 SHA 연결 | [FIGURE_BINDINGS](FIGURE_BINDINGS.json) |
| 실제 실행량과 미달성 목표 | [BUILD_LEDGER](BUILD_LEDGER.json), [GOAL_STATUS](GOAL_STATUS.json) |

[재현 범위와 명령](REPRODUCE.md)은 공개 원행만으로 가능한 산술 검산과 외부 원본 자료가 필요한 검사를 구분합니다. 기존 결과를 덮어쓰지 않습니다. 현재 사용 API와 작은 수정 IMAGE_ROLE head는 [v2 배포 안내](../pallet_boundary_corner_refiner_20261010_v2/README.md)에 있습니다. API의 실행 가능성과 실제 자세 정확도 개선은 별개입니다.
