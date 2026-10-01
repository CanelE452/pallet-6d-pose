# 고정 잔차 방향18 추가의 학습 전 독립 검산

**PREFIT PASS는 입력·수학 계약 검산이며 학습이나 T/R 개선 결과가 아니다.** 기존253 특징과 O에서 고정한 bbox정규화 signed residual18을 이어 붙인271입력만 변경한다. 기존94 정규화·RBF basis·signed target·scale·유효 후보·TRAIN2,598행(실패1행 포함)을 유지했고, 원래6해시와 O의추가18/271해시를 독립 재구성해 대조했다.

새18의 mean/std는 O의 R0 TRAIN 유효5,194후보에서 고정한 FP32 값이다. FP32 정규화 뒤 FP64로 올려 candidate−R0 GEO anchor 차이를 계산한다. anchor와invalid 입력은0이다. 실제 TRAIN에서는 특징과타깃 해시만 검사했고 이전 weight·목적식·선택 정책을 적용하지 않았다.

합성 예제에서 이전253 가중치에zero18을 붙이면 같은 Huber+sign-logistic+ridge 목적식과 점수를 재현했다(수치허용1e−12). 추가18축 gradient가0이라는 주장은 하지 않는다. 독립 scalar 목적식, Torch64 logaddexp/autograd,542개 좌표 중앙차분과542×542 Hessian을 대조했다. gradient 차분 최대7.44e-11, Hessian 차분 최대1e-10이다. 모든542계수의 ridgeλ1e−4가 강볼록성을 유지한다.

기존 목적식·Hessian·signed target·Newton/Armijo·인증 함수 AST와 source45/원래+matched 실사5기준 AST를 확인했다. solver는 영초기화·최대1,000accepted iteration/2,000objective calls·gradient Linf≤1e−8 및 gap≤1e−6를 그대로 사용한다. 독립 작은 합성문제로 accepted/rejected state와 예산을 검산했다.

고정 pose와9점 관측/K/bbox의 방향 추출을 별도 성분 투영식으로 대조했다. corner0..7과 P8 중심을 유지하고, K·관측·bbox를함께100px 옮겨도 잔차가 같음을 확인했다. runtime의 target 생성함수를 강제로 막아도 점수를 계산했다. 실제 pose 생성·image forward·VAL/실사 참조 열람은0이다.

[검산 JSON](PREFIT_REVIEW.json) · [이전 동일 목적식 검산](../pallet_pose_signed_axes_sign_20261001_v1/PREFIT_REVIEW.json) · [방향 입력 독립 감사](../pallet_pose_residual_direction_audit_20261001_v1/VERIFICATION_KO.md)
