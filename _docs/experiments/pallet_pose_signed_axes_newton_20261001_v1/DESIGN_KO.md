# 동일 T/R 회귀의 수렴 문제: solver 한 가지 변경

목표는 단일 RGB 이미지와 팔레트 치수, 기존 카메라 보정 K로 자연 가림의 위치 T·회전 R을 안정적으로 함께 개선하는 것이다. 학습 수렴이나 합성 검증의 통과만으로 이 목표를 달성했다고 주장하지 않는다.

## 관측 근거와 범위

[직전 signed-axis 시도](../pallet_pose_signed_axes_20261001_v1/REPORT_KO.md)는 첫 R0_ONLY 학습이 고정 1,000회 반복에서 종료됐다. 1,108회 objective 호출 후 gradient-gap 상한은 0.000181007602637로, 요구한 1e−6 보증을 얻지 못했다. 나머지 학습·source VAL·실사 평가는 실행하지 않아 그 방법의 T/R 효과는 미측정이다.

[독립 검산](../pallet_pose_signed_axes_20261001_v1/REJECTED_FIT_VERIFICATION_KO.md)에서 마지막 Hessian 조건수는 약 276,286이며, 마지막 100회 반복에도 J가 약 1.26e−6 감소했다. 이 관측은 곡률을 직접 사용하는 최적화 방식을 시험할 근거다. 이것만으로 이전 실패의 유일 원인을 확정하거나 새 solver의 수렴·실사 개선을 보장하지 않는다. gap 상한 초과는 실제 최적값과의 차이가 반드시 허용치를 넘었다는 증명이 아니다.

이번 비교는 **목적함수를 바꾸지 않고 solver만 변경**한다. 기존 실패 가중치를 이어 학습하지 않으며 새 namespace와 프로토콜에서 0부터 시작한다. 동일 목적식의 인증된 해를 얻은 뒤에야 signed T/R 감독의 효과를 검증할 수 있다.

## 변경하는 계산

- 매 accepted point에서 Huber의 활성 곡률과 기존 λI로 두 축의 253×253 SPD block을 구성한다. 각 축에서 `H p = −gradient`를 풀며 추가 damping은 없다.
- 정확히 `|residual|=1`인 Huber 경계의 데이터 곡률은 0을 사용한다. λI는 유지한다. 경계에서는 고전적 Hessian이라고 주장하지 않는다.
- Armijo 조건은 `J(W+αp) ≤ J(W)+1e−4·α·<gradient,p>`로 고정한다. 매 반복 α=1부터 시작해 0.5씩 줄인다.
- 첫 objective 계산과 거부된 trial을 포함한 모든 호출을 최대 2,000회에 포함한다. 채택된 반복은 최대 1,000회다. 마지막 채택점과 이미 계산한 gradient만 인증에 사용한다.
- 성공하려면 gradient Linf≤1e−8과 기존 `||gradient||²/(2λ)≤1e−6`을 동시에 만족해야 한다. 선형계 실패·비유한 수치·하강 방향 실패·step underflow·예산 소진은 중단 조건이다. 실패 후 재시작, tolerance 탐색, 예산 증가는 없다.

SPD 방향은 gradient가 0이 아닐 때 하강 방향이다. Huber 목적식은 C1이고 λ>0으로 강볼록이므로 이 하강 방향에 Armijo backtracking을 적용할 수 있다. 실제 유한정밀도 실행의 성공 여부는 별도 인증으로 확인한다.

## 유지하는 계약

TRAIN 2,598행과 전체 분모, 기존 유효 후보, raw94 정규화·context189·고정 RBF64, 운영 R0 GEO anchor 차분253, signed-log1p 두 축 target, Huber δ=1, λ=1e−4, bias0, T/R scale, 네 모델 및 원래 후보 선택의 동률 규칙을 유지한다. 임의 threshold·새 특징·새 실사 GT 학습을 추가하지 않는다. 입력·target의 6개 SHA를 직전 사전 검산과 대조한다.

이전과 같은 source45조건을 모두 만족해야 실사로 진행한다. 실사 173장의 네 모델 692개 선택을 참조 오차 접근 전에 고정하고, 원래 SINGLE251/R0/PRIOR1/FULL125 기준과 matched 비교 기준을 항목별 AND한다. 세 seed 양축 개선, 촬영×seed 불확실성, 촬영 제외 민감도, 자연 꼬리·실패 보존, clean 보존의 다섯 범주를 모두 유지한다. 실패 행·촬영·seed를 제외하지 않으며 wood45는 별도 stress로 보고한다.

같은 실사 DEV와 source VAL을 여러 방법에서 반복 사용했다. 새로운 독립 TEST가 아니다. 기존 refiner source 노출 중복105/26/26과 교사 계보의 수동 코너38개/이미지9장을 숨기지 않는다. 실사 pose 참조는 2D 주석·K·치수로 만든 값이며 독립 장비로 측정한6D GT가 아니다. 결과는 수치 수렴, 합성 성능, 실사 T/R 안정성을 구분해 실제 RGB·치수와 함께 공개한다.

[학습 계약](TRAIN_PROTOCOL.json) · [학습 전 독립 검산](PREFIT_REVIEW_KO.md)
