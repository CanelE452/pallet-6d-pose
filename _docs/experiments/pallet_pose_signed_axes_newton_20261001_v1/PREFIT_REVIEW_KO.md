# Solver만 변경한 Newton 학습 전 독립 검산

**PREFIT 검산 PASS. 새 fit·VAL·실사 성능 평가는 실행하지 않았다.** 직전 signed_axes의 고정 phi253에서 R0 GEO anchor를 뺀 입력, 두 축 signed log1p 타깃, 원래 valid mask, TRAIN2,598행, scale, 정규화, basis, Huber δ=1과 λ=1e-4를 유지한다. 독립 재구축한 모델별6개 해시가 이전 signed_axes PREFIT와 모두 같다.

변경은 L-BFGS-B에서 `BLOCK_GENERALIZED_NEWTON_ARMIJO`로의 solver 교체다. 이전 모델의 거부 weight를 초기값으로 쓰지 않고253×2 영행렬에서 시작한다. 축별 generalized Newton block과 고정 Armijo c1=1e-4, 초기 step1, 축소율0.5를 적용한다. 최대1,000 update·2,000 objective 호출은 그대로이며 초기 평가와 거부된 모든 line-search trial도 호출 예산에 포함한다. 성공에는 gradient L∞≤1e-8과 기존 gap 상한≤1e-6을 모두 요구한다.

기존 objective·Hessian·타깃·특징 차분·추론 중요 함수의 AST를 대조했다. 스키마와 solver 메타데이터 외 목적식·선택 규칙의 변경은 없다. independent scalar Huber, Torch autograd와506방향 차분 검산의 gradient 최대 차이 3.72e-11, Hessian 최대 차이 5.61e-11다. |잔차|=1의 고전적 Hessian은 정의되지 않으므로 미분 검산은 경계 밖에서 수행하며 solver는 기존 zero-curvature generalized convention을 유지한다.

합성 문제에서 독립 Newton 방향·고정 Armijo 순서·모든 평가 호출·성공 조건·예산 실패·all-invalid 분모를 검사했다. 이는 수치 구현 검산이며 실제 데이터의 수렴이나 T/R 개선을 미리 판정하지 않는다. 실제 TRAIN에서는 기존 캐시 입력·오류로 표현과 타깃만 재구축했으며 학습된 weight 적용, objective 시험, 선택률·성능 probe를 수행하지 않았다. 원본 GT·VAL 품질·실사 참조는 읽지 않았다.

추론은 이전과 같은 두 signed 예측의 max를 최소화한다. anchor 예측0은 실제 T/R 비악화 보장이 아니며 기존 source45 및 원래 SINGLE251·matched 두5조건의 AND 판정은 별도 실행 대상이다.

[검산 JSON](PREFIT_REVIEW.json) · [이전 동일 타깃 검산](../pallet_pose_signed_axes_20261001_v1/PREFIT_REVIEW.json) · [고정 basis](../pallet_pose_anchor_rbf_20261001_v1/RBF_BASIS.json)
