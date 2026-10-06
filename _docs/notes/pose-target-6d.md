# final 6D pose target의 matched 비교

## 1. 제안

[추정·미검증] 같은 GEO bank·JointActionScorer·frozen features·치수·seed 초기화·order·AdamW·6,000 update·hard J·최종 F에서 supervision만 final ADDsym hard firstargmin CE로 바꾼다. 전체 TRAIN 55,915장의 실제 F 비용을 저장하고 seed1 먼저 수행한다. 판정 지표는 같은 ID의 T/R/ADDsym와 코너 중앙값/P90, coverage, RAW good5→bad10 손상, oracle 회수다. seed2/3 gate는 결과 전 PROTOCOL에 고정하고 REAL_DEV를 판단에 사용하지 않는다. bank/F/index/초기화/order/GT 격리 계약이 어긋나면 BLOCKED_INTEGRITY로 학습을 중단한다. 논문 관련 파일은 수정하지 않는다.

## 2. 결과

[확인] 전체 55,915 TRAIN×201 action=11,238,915 실제 F를 완료했다. F unavailable 9개를 +inf로 기록했고 제외 행은 0개다. 전체 native 비용·index·W/D·firstargmin 저장 검산과 학습 전 원입력 보존 검사 PASS다. I/O 병목을 개선한 빠른 단계는 1,469.715초로 기존 전체 단계 대비 wall 처리량 10.22배였으며 F·모델·목표의 과학 코드는 유지했다.

[확인] seed1만 6,000 update/96,000 노출로 실행했고 최종 checkpoint를 SYNTH_HELDOUT 1,985장 및 REAL_DEV 319장에 평가했다. SYNTH의 NEW−OLD paired 평균은 T +1.984673cm [1.285732, 2.740682], R +0.658447° [0.174514, 1.146220], ADDsym +0.018989m [0.011985, 0.026177]다(같은 ID, frame bootstrap 10,000회, 95%CI). RAW good5→bad10 canonical 코너 손상은 OLD 63개→NEW 450개다. 사전 고정 gate STOP에 따라 seed2/3은 미실행이며 판정은 POSE_TARGET_NOT_SUPPORTED다. N3 대비 세 pose paired 평균도 모두 악화해 N3 개선을 지지하지 않는다.

[확인] REAL_DEV의 NEW−OLD paired 평균 ADDsym은 +0.010619m이고 frame95%CI는 [−0.002181, 0.026094]다. pose coverage는 두 방법 모두 319/319, 2D matching은 311/319다. 실제 2D 코너 중앙값은 OLD 6.4291px→NEW 8.3764px, PCK10은 64.8259%→56.3826%다. 이 자료는 같은 2D 주석·치수의 참조를 재구성한 반복 사용 DEV이며 독립 물리 계측 pair는 0/BLOCKED_DATA다.

[확인] oracle W/D 전환 집단에서 NEW가 최종 hypothesis를 회수한 수는 SYNTH 11/68(OLD 4/68), REAL 1/16(OLD 0/16)이다. exact oracle action 수는 증가했지만 전체 평균 oracle gap과 RAW 대비 회수는 악화했다. 학습 중 기존 pre-update forward의 누적 target 적중은 6,245/96,000(6.5052%), 이동 target은 1,194/90,589(1.3180%)이며 최종 checkpoint의 전체 고유 TRAIN 정확도는 측정하지 않았다. 이를 underfitting이나 특정 원인의 확증으로 해석하지 않는다.

[확인] 상세 원행·matched 계약·oracle recovery·손상·W/D 및 후속 계획은 [실험 결과](../experiments/pallet_pose_target_6d_20261006_v1/RESULT_KO.md), [다음 판단](../experiments/pallet_pose_target_6d_20261006_v1/NEXT_DECISION_KO.md)에 있다. 추가 fit·hyperparameter 탐색·후속 모델·논문 수정은 실행하지 않았다.
