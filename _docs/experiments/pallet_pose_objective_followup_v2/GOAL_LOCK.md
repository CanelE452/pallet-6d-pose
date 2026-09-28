# Translation / full-rotation 중심 후속 실험

요청: `pallet_pose_objective_autonomy_goal_plan_cli.txt` (2026-09-28). 이전 C1/C2/C3와 AUC 판단을 보존한다.

## 목적과 비교

가림이 있는 실사에서 corrected-pseudo 감독이 학생의 translation과 full rotation으로 전달되는지 확인한다. 주 모집단은 고정 Plastic Moderate+Severe 99장이다. OLD_REF 대비 recipe 효과, 동일 NEW_RAW 대비 corrected 좌표의 추가 가치, R0 대비 실용 가치를 구분한다. 낮을수록 좋은 cm/degree 오차의 전후값을 보고한다. AUC/PCK는 보조이며 후보 선택에 쓰지 않는다.

## 실행 계획

1. 과거 자산·평가·단위·대칭·선택 규칙 고정, pool의 신뢰성/난도/감독 구분.
2. 현재 실행 criterion의 assignment·location/RLE/존재 손실·gradient·finite difference를 소규모 TRAIN/source에서 확인.
3. A–F를 각각 선행연구·현재 코드·최소 진단으로 판단. 기본 첫 후보는 원본 fixed target + 학생 RGB random occlusion 짝지은 대조. 변경은 하나씩 잠근다.
4. 강한 진단 근거가 있는 다음 후보 하나, 유망한 두 원리가 있으면 결합 대조. 최대 3개 주 가설 cycle.
5. 고정 예측과 공통 D9로 T/R scoring. 선택한 recipe와 대응 control을 같은 R0에서 추가 학습난수로 반복. Wood는 별도 적용성으로 구분.
6. 실제 손익·실패·자원·보류 이유·claim impact·재현·감사를 새 namespace에 기록하고 commit/push 후 종료.

## 상한과 금지

새 fit 최대 12, 실제 optimizer update 합계 최대 7,680; GPU 6시간 / active wall 10시간. 기본 학생 320 update, 마지막 4 fit은 필수 control/재현을 위해 예약한다. 실행된 실패 fit/teacher fit도 계수한다. 교사 적응 경로는 추가 학습량 control과 학생 전달 비용을 먼저 확보한다.

추가 annotation, eval→train, 새 architecture/selector, threshold sweep, 새 pretrained estimator, driver/environment 업데이트, 재부팅, 다른 프로세스 종료는 하지 않는다. sealed TEST는 열지 않는다. 기존 paper/main 결과는 바꾸지 않고 `CLAIM_IMPACT.md`에 반영안을 분리한다. 예시 RGB는 기존 공개 허용 ID만, private 좌표·원본 RGB·K·checkpoint는 공개하지 않는다.

## 종료 해석

실행 완료와 성능 성공은 별개다. T/R 중 하나만 개선하면 joint gain으로 쓰지 않는다. 같은 pretrained model의 반복은 독립 초기화/독립 TEST가 아니다. 3 cycle 또는 자원 상한 또는 남은 유효 가설 부재에서 종료하며, 한 negative 결과만으로 나머지 적격 후보를 자동 폐기하지 않는다.
