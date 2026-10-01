# 과소예측 추가 Huber: 학습 전 독립 검산

**PREFIT PASS는 입력·목적식 계약의 검산이며 학습이나 T/R 개선 결과가 아니다.** 직전 P의271입력·타깃·원래 valid mask·TRAIN2,598행(실패1 포함)과 모든9해시가 같다. N의 원래6해시도 유지한다. R0 TRAIN 정규화94/18과 RBF basis는 다시 선택하지 않았다.

단일 변경은 e=p−y일 때 기존 Huber(e,1)에 Huber(min(e,0),1)을 고정계수1로 추가하는 것이다. 따라서 대칭 Huber와 추가 과소예측 Huber를 별도 기록하고 둘의 합을 `Huber`로 정의한다. J=Huber+Sign_logistic+L2이다. target0에도 Huber는 적용하며, 기존 sign logistic만 nonzero target에 적용한다. 원래 행별 유효후보×2축 평균 후 전체2,598행 평균과λ1e−4 ridge를 유지한다.

독립 scalar 식·Torch64 autograd·542개 좌표 중앙차분·542×542 Hessian을 대조했다. gradient 차분 최대8.51e-11, Hessian 차분 최대1.21e-10이다. 추가 Huber 곡률은−1<e<0에서1, e=0과e=−1에서는 사전 지정0이다. 원래 Huber kink·logistic 곡률·ridge는 유지한다. 추가항은 볼록이고 연속미분 가능하며 모든542개 계수의 ridge가 강볼록성을 유지한다.

같은 가중치에서 직전 runtime 점수와 byte 단위로 같고, target 생성함수를 막아도 점수를 계산한다. 실제 TRAIN에서는 입력·타깃 해시만 재구성했으며 이전 weight·목적식·선택 정책을 실행하지 않았다. 새 fit·PnP·이미지 forward·VAL/실사 참조0이다.

대칭 Huber의 방향별 비용을 바꾸는 가설 검산이다. 이전 risk-margin CE·utility/부호 회귀의 음성 결과는 반대근거이며, 보수화가 anchor 복귀만 늘려 엄격한 공동 개선을 없앨 수 있다. 비대칭 손실 자체의 학술적 새로움이나 성공을 주장하지 않는다. source45 및 원래+matched 실사5기준은 유지한다.

[검산 JSON](PREFIT_REVIEW.json) · [직전 동일 입력 검산](../pallet_pose_signed_axes_direction_20261001_v1/PREFIT_REVIEW_KO.md)
