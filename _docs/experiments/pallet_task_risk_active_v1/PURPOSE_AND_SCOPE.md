# 목적과 범위

기존 active-learning의 `RETROSPECTIVE_AL_NO_SIGNAL`은 보존한다. 이번에는
원본 및 고정된7개 photometric perturbation에서 전체 포즈 파이프라인의
x/z/yaw 변동과 후보·축 전환·실패율이 실제 어려운 샘플을 예측하는지 검사한다.

Stage0를 통과할 때만 기존 feature diversity와 `1+R_task`를 결합해30장을
선택하고, 기존과 같은 R0/stock loss/optimizer/300-update 조건으로3seed를
학습한다. 전체174장·seed1·300-update 참고 모델을 포함한 총예산은
4fits/1200updates다. 추론 아키텍처는 일반 RGB YOLO pose 그대로다.

Primary는 보류145장 모두에서 가장 나쁜15장 위치 오차의 평균인
translation CVaR90이다. 공통 검출 프레임만으로 primary 분모를 바꾸지 않는다.
실패한 원래 방법을 새 primary로 재판정하지 않는다.

독립 일반화·SOTA·novelty·zero-target-label 주장 및 paper/final 수정은 금지한다.
결과를 보고 score/perturbation/budget/seed를 바꾸거나 다른 연구를 자동 재개하지 않는다.

이전 GPU busy 중단은 별도 불변 RESOURCE_BLOCKED 기록으로 보존한다.
사용자의 GPU 사용 종료 확인 이후 자원 상태를 재확인하고 이번 실행을 재개했다.
