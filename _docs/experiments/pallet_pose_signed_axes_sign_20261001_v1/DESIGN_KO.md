# 물리적 T/R 변화의 부호 감독: 사전 고정한 단일 비교

목표는 RGB 이미지 한 장과 팔레트 치수, 기존 카메라 보정 K를 이용해 자연 가림의 위치 T와 회전 R을 안정적으로 함께 개선하는 것이다. 수렴 인증, 합성 점수, 공개 완료만으로 목표 달성을 대신하지 않는다.

## 관측 근거와 경쟁 설명

[직전 Newton 실험](../pallet_pose_signed_axes_newton_20261001_v1/REPORT_KO.md)은 동일한 Huber 회귀 네 모델의 수렴 문제를 해결했으나 source45조건 중43개만 통과했다. seed3의 T 중앙값1.682086665cm가 기준1.674594149cm보다 커 실사 평가로 진행하지 않았다.

[고정 선택 진단](../pallet_pose_signed_axes_newton_20261001_v1/SOURCE_TRANSFER_DIAGNOSTIC_KO.md)에서 UNION 세 seed의 source 안전 개선은30/33/26건, 한 축 이상 악화는72/72/73건이었다. 두 예측 축이 모두0 이하인 후보를 골라도 실제 오차가 증가했다. TRAIN에서도 안전 개선135/120/106건과 악화171/183/171건이 함께 남았다. 연속 회귀의 수렴만으로 운영 선택의 부호 정확성이 확보되지 않는다는 관측이다. 손실·표현력·합성 지원범위 중 하나가 유일 원인임을 증명하지는 않는다.

이전 CE·pairwise·risk margin·signed2D utility·분류와 회귀를 섞은 structured DHT·global RGB MLP의 음성 결과를 유지한다. 이번 비교가 일반적인 혼합 손실의 새로움을 주장하거나 전이 성공을 보장하는 것은 아니다.

## 단일 변경과 수학

기존 두 축 선형 예측 `p=x@W`, signed-log1p 물리 변화 target `y`, Huber δ=1을 유지한다. 각 scalar에 다음 항 하나만 추가한다.

```text
sign_loss(p,y) = 1{y != 0} * softplus(-sign(y) * p)
J = mean_all2598[mean_original_valid_candidates_and_2axes(
       Huber(p-y,1) + sign_loss(p,y))] + (1e-4/2)||W||_F²
```

계수는1로 고정하며 탐색하지 않는다. sign 정답은 기존 TRAIN 물리 변화에서만 유도한다. target이 정확히0이면 logistic 손실·기울기·곡률은 모두0이고 log(2) 상수도 더하지 않는다. anchor와 invalid placeholder에서 새 정보가 생기지 않는다. 부호 threshold·epsilon·클래스 재가중·nonzero 항만의 재평균은 없다. 모든 후보가 실패한1행은 데이터 손실0으로 전체2,598행 분모에 남는다.

선형 예측에 대한 logistic 항은 볼록하다. 두 축 간 블록은 여전히 분리되며 λI로 강볼록성이 유지된다. Huber 경계에서는 기존 데이터 곡률0 규칙을 쓰되 logistic의 매끄러운 곡률은 유지한다. overflow를 피하는 softplus·sigmoid 계산을 사용한다. 작은 실제 변화의 부호 오류에도 분류 신호가 남는 것이 기대하는 효과이며, 실제 양축 개선·꼬리 보존을 보장하지 않는다.

## 고정한 입력·실행·판정

입력253차분, raw94 정규화·context189·고정 RBF64, TRAIN2598행, 기존 T/R scale과 target, 후보 pose·valid mask·bias0·λ=1e−4를 유지한다. 모델별 기존6개 데이터 SHA를 이전 Newton PREFIT과 대조한다. 각 모델은0에서 한 번만 학습하며 이전 가중치로 이어 학습하지 않는다.

기존 block generalized Newton와 Armijo(alpha1, 반감0.5, c1=1e−4), 최대1,000 승인 반복·초기 및 거부 trial 포함2,000호출, gradient Linf≤1e−8 및 gap 상한≤1e−6을 유지한다. 실패하면 재시작·예산확장·계수/tolerance 탐색을 하지 않는다. float64의 수치 검산을 부동소수점 오차까지 포함한 엄밀한 interval 인증이라고 부르지 않는다.

추론은 두 출력의 max를 최소화하는 기존 whole-pose 선택 및 정확 동률 규칙을 유지한다. 두 출력은 연속 변화와 부호를 함께 학습한 점수이며 보정된 물리 오차나 확률이라고 주장하지 않는다. 실제 GT·margin·safe mask·새 threshold를 추론에 넣지 않는다.

원래 source45조건을 모두 통과해야 실사173장의 네 모델 선택692개를 참조 접근 전에 잠근다. 세 seed 양축 개선·촬영/seed 불확실성·촬영 제외 민감도·natural 꼬리/실패·clean 보존의 다섯 범주와 original SINGLE251 및 matched 비교의 AND를 유지한다. 어떤 행·촬영·seed도 사후 제외하지 않는다. wood45는 별도 stress다.

source VAL과 실사 DEV는 반복 사용한 개발 자료이며 새 독립 TEST가 아니다. refiner의 source 노출 중복105/26/26과 교사 계보의 수동 코너38개/9이미지도 이전과 같다. 새 실사 정답 학습은0이며 실사 참조 pose는2D 주석·K·치수로 만든 값으로 독립 장비의 실측6D 정답이 아니다. 실제 수렴·합성·실사 판정은 구분하고 사진·치수·수치·코드를 공개한다.

[학습 계약](TRAIN_PROTOCOL.json) · [학습 전 독립 검산](PREFIT_REVIEW_KO.md)
