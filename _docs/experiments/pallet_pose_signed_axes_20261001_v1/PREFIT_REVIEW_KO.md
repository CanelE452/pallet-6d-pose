# Signed T/R 회귀 학습 전 독립 검산

**입력·목적식 검산 PASS. 새 fit·VAL·실사 성능 평가는 실행하지 않았다.** 기존 고정 RBF253을 재사용하고 후보 특징에서 같은 행 R0 GEO anchor 특징을 뺀253 입력을 두 축 선형 회귀에 사용한다. center·폭·정규화·후보·TRAIN2,598행·scale은 이전 바인딩과 같다.

정답은 각 물리 T/R 오차의 anchor 차이를 기존 TRAIN scale로 나눈 e에 `sign(e) log1p(abs(e))`를 적용한다. 유효 후보만 뺄셈하여 inf−inf를 피하고 anchor 입력·정답은 정확히0이다. invalid 입력·정답은0이며 all-invalid1행도 전체2,598 분모에 남는다. cached TRAIN 오류 외 원본 GT는 읽지 않았다.

목적식은 Huber δ=1을 행별 유효 후보와 두 축에 대해 평균한 뒤 전체 행 평균을 취하고 λ/2‖W‖²를 더한다. 독립 행·후보·축 반복 계산, Torch autograd와506방향 중앙차분을 대조했다. gradient 차분 최대 3.72e-11, Hessian 차분 최대 5.61e-11이며 Hessian 최소 고유값은 0.0001이다. |잔차|=1에서 값·gradient는 연속이지만 고전적 Hessian은 정의되지 않으므로 Hessian 검산은 경계에서 떨어진 합성 예제에 한정한다.

추론은 두 예측 축의 max를 원래 유효 후보 사이에서 최소화하며 기존 R0·hypothesis 동률 규칙을 유지한다. anchor 예측은0이므로 선택 점수는 예측상0 이하이지만 **실제 T/R 비악화 보장은 아니다.** 추론 함수에서 GT target 생성기를 금지해도 동작함을 확인했다. margin·safe GT mask·새 참조 오류를 넣지 않는다.

실제 TRAIN에서는 이전 RBF PREFIT의 phi253·raw feature·valid·unscaled 오류·정규화·anchor·scale 해시를 확인하고 signed target과 anchor 차분 입력을 독립 재구축했다. 실제 행에 임의 weight를 적용하거나 objective·선택률·T/R 개선을 시험하지 않았다. 모델별6개 해시는 학습 시작 전에 그대로 대조할 수 있도록 JSON에 기록했다.

본 결과는 학습 구현의 일관성 검증이며 개선 성과를 뜻하지 않는다. 고정 source45와 원래 SINGLE251 및 matched 두 묶음의5개 AND 조건은 별도 실행·판정 대상이다.

[검산 JSON](PREFIT_REVIEW.json) · [재사용한 고정 basis](../pallet_pose_anchor_rbf_20261001_v1/RBF_BASIS.json)
