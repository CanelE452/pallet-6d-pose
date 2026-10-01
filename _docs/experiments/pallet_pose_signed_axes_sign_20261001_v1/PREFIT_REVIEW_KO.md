# 부호 logistic 항 추가 학습 전 독립 검산

**PREFIT 검산 PASS. 새 fit·VAL·실사 성능 평가는 실행하지 않았다.** 직전 Newton의 입력253, signed log1p 두 축 타깃, 원래 valid mask, TRAIN2,598행, scale, 정규화, basis, Huber δ=1과 ridge λ=1e-4를 유지한다. 독립 재구축한 모델별6개 해시가 이전 Newton PREFIT와 모두 같다.

단일 목적식 변경은 target y가0이 아닌 축에 `softplus(-sign(y)*prediction)`을 계수1로 더하는 것이다. y=0인 anchor·동률 축에는 loss ln2조차 추가하지 않고 gradient·curvature도0이다. Huber와 sign 항 모두 각 행 원래 유효 후보×2축 평균 후 전체2,598행 평균을 취하며, all-invalid1행도 전체 분모에 남는다. nonzero target 수로 새로 정규화하지 않는다.

독립 scalar softplus·sigmoid와 Torch autograd,506방향 gradient/Hessian 중앙차분을 대조했다. gradient 차분 최대 7.22e-11, Hessian 최대 1e-10이다. 큰 양·음 logit과 정확한 zero-target mask를 별도로 검사했다. Huber의 |잔차|=1에서는 기존 generalized zero curvature를 유지하며 logistic 곡률만 별도 더한다. 강볼록성은 모든506계수의 ridge에 의해 유지된다.

기존 입력·타깃·추론 함수와 source45 및 실사 두5조건의 AST를 확인했다. Newton/Armijo step·예산·성공 조건은 동일하고 Hessian에 target 인수가 추가된 부분과 새 loss component 기록만 달라진다. 합성문제에서 독립 방향/line-search 및 예산 경로를 검산했다. 이전 fitted weight를 읽거나 초기값으로 쓰지 않는다.

실제 TRAIN에서는 기존 캐시 입력·오류로 특징과 타깃만 재구축했으며 weight 적용·목적식 시험·선택/성능 probe를 수행하지 않았다. 원본 GT·VAL 품질·실사 참조는 읽지 않았다. 추론은 같은 두 예측 축의 max·원래 tie를 사용하며 학습 sign label이나 참조오차를 받지 않는다. 검산 통과는 수렴이나 실제 T/R 개선을 예고하지 않는다.

[검산 JSON](PREFIT_REVIEW.json) · [이전 동일 입력/타깃 검산](../pallet_pose_signed_axes_newton_20261001_v1/PREFIT_REVIEW.json) · [고정 basis](../pallet_pose_anchor_rbf_20261001_v1/RBF_BASIS.json)
