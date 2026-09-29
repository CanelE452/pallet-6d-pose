# Clean-to-pose 최소 구성: 추가 승인 배치

기존 `pallet_clean_to_pose_transfer_v1`의 FINAL/STOP을 변경하지 않는다. 이번 배치는 기존 DEV 결과를 본 뒤 승인된 후속 분석이다. 전체 비교를 처음부터 사전등록했다고 주장하지 않는다.

## 질문과 실행 경계

1. 같은 GEO의 R0보다 자기학습이 필요한가?
2. 같은 GEO에서 REF CLEAR/OCC의 입력 가림 효과가 있는가?
3. 같은 입력·학습량·선택기의 RAW/REF에서 보정 좌표의 추가 가치가 있는가?
4. 기존217장 REF+GEO 대비 clean78의 추가 가치가 있는가?
5. 가장 강한 단순 대조 대비 T/R 변화가 난도·recording·tail·실패·실제 반복에서 어떻게 나타나는가?

우선 기존 frozen 예측/후보에 같은 GEO를 적용한다. 평가 참조 접근 전 새 선택 결과를 잠그고 별도 scoring 단계에서 기존 후보 metric을 연결한다. 기존 OCC42/43는 재사용한다. 없는 CLEAR43 모델을 가정하지 않는다.

추가 학습은 필요 대조 또는 검증 가능한 단일 원인의 실험에만 사용한다. 이번 시작 시 누적6 student fits/1920 updates/417.7810587910935초 GPU 학습/0 selector fits다. 상한은 누적10 student fits, 1 selector fit, GPU 학습21600초이며 원장을 초기화하지 않는다. 과거 원장은 불변 입력으로 연결한다.

clean78 membership/의사 좌표/무시점, 교사·학생·선택기의 직접 및 간접 수동감독, 평가128/99와 C2 대칭/실패 처리/치수/참조는 유지한다. 새로운 학습 RGB·수동 좌표·센서·촬영·백본은 추가하지 않는다. GT로 배포 후보를 고르거나 후보별 T와 R 최솟값을 합치지 않는다.

## 완료 증거 체크리스트

| 요구 | 완료를 증명할 자료 |
|---|---|
| 로컬/원격/중복 실행 확인 | START.json, 실제 프로세스 관찰 및 remote HEAD |
| 누락된 동일 GEO 대조 | frozen 선택 잠금과 기존 후보 hash, CONTROLS 결과 |
| 동일 계약 집계 | NATURAL99/CLEAN29/MODERATE21/SEVERE78/FULL128, valid/전체, T/R median/P90 |
| paired/recording/tail | 같은 ID의 T/R 방향과 절대 차이, recording 및 leave-one-recording-out |
| 구성별 필요성 | 자기학습/보정/가림/선택기/clean78 각각의 같은 조건 대조 |
| 자율 후속 판단 | 원문·과거시도 대조, 결과를 보기 전 새 조건/판정 잠금, 제외 이유 |
| 유망 최종 비교 반복 | 실제 다른 stream과 같은 seed의 paired 입력/박스/support, protected state/update/last/예측 잠금 |
| 사례와 공개 안전성 | 사전 일관 선택 규칙, 개선/악화/큰오류 사례, 원 RGB/좌표 비공개 |
| 재현 가능한 최종 구성 | 체크포인트/데이터/프로토콜/예측 SHA, 실행 명령 |
| 실패 포함 비용/재개 | inherited+new events 원장, 완료 단계 재사용, 실제 background resume만 표시 |
| 검증 및 공개 | 테스트·독립 수치 감사·그림 검수, 관련 파일 commit/push 및 원격 SHA |

중앙값 개선만으로 안정성/독립 검증을 선언하지 않는다. 가장 강한 단순 대조를 이기지 못하면 더 단순한 구성을 인정한다. 유망 구성을 확보하거나 근거 있는 가설·예산이 닫히면 부정 결과를 포함해 종료한다.
