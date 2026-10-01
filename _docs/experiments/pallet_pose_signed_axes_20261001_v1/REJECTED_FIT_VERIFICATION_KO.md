# 중단된 첫 signed-axis 학습의 독립 검산

**수치·기록 검산 PASS, optimizer 수렴 인증 FAIL.** 첫 R0_ONLY 학습은 사전 1,000회 반복 한도에서 중단됐다. UNION 학습은 시작하지 않았고, 승인된 checkpoint/FIT/TRAINING_COMPLETE가 없다. 실패 JSON의 506개 가중치는 중단 상태를 검산하기 위한 값이며 운영 모델이나 최종 성능 checkpoint가 아니다.

| 항목 | 독립 계산 |
|---|---:|
| 전체 목적함수 J | 0.220023167575816 |
| Huber 항 | 0.21601982514925019 |
| L2 항 | 0.0040033424265658122 |
| gradient L2 | 0.00019026697171978284 |
| gradient Linf | 3.6150967182768524e-05 |
| 강볼록 gap 상한 | 0.00018100760263708323 |
| Hessian 최소/최대 고유값 | 9.99999999978e-05 / 27.6286413484 |
| Hessian 조건수 | 276286.41349 |
| 반복 / objective 호출 | 1000 / 1108 |

고정 TRAIN 2,598행의 raw94→context189→RBF253→anchor 차분과 signed-log1p 두 축 target을 독립 재구성했다. 원래 유효 후보를 유지하여 각 frame에서 후보×두 축 Huber(delta1)를 평균한 뒤 전체 2,598행으로 평균하고 λ=1e−4 ridge를 더했다. 실패 1행은 유효 후보가 없어 데이터 손실0이지만 전체 분모에 남는다. Torch64 autograd와 독립 가중 design-matrix의 506×506 block Hessian을 실행 코드와 대조했다.

목적함수 기록 차이는 2.78e-17, gradient 벡터 최대 차이는 3.01e-17, Hessian 최대 차이는 3.33e-16이다. 정확한 Huber kink(|residual|=1)는 0개였다. kink가 있으면 사전 선언된 데이터 곡률0을 사용하되 λI는 유지한다. gradient-gap 상한은 Huber의 C1 강볼록성에 근거한다.

## 중단 판정과 감사 범위

강볼록 gap 상한 0.000181007603은 계약의 1e−6 이하 조건을 만족하지 못한다. optimizer success도 false이며, 종료 사유는 `TOTAL NO. of ITERATIONS REACHED LIMIT`이다. 2,000회 objective 호출 한도에는 도달하지 않았지만 1,000회 반복 한도에 도달했으므로 그대로 종료했다. gap **상한** 초과만으로 실제 optimality gap이 1e−6보다 크다고 증명되는 것은 아니다. 이번 판정은 정해진 수렴 인증을 얻지 못했다는 뜻이다.

START/REJECTED/FAILED/PREFIT와 전체 2108개 trace 행의 6개 데이터 SHA, basis·anchor SHA, 사건 순서, 호출 수와 반복 수를 대조했다. 각 iteration은 직전 objective 기록과 일치한다. 초기·중단 가중치의 목적함수는 독립 재계산했으며, 중간 가중치는 저장되지 않았으므로 중간 수치의 재계산을 주장하지 않는다. 900→1,000번째 반복의 기록상 J 감소는 1.26323608873e-06이다.

실패 가중치의 argmin 선택이나 T/R 성능은 계산하지 않았다. 추가 fit/optimizer step, source VAL 또는 실사 품질 열람, raw 참조 열람은 모두0이다. source45 및 실사 양쪽 5개 조건은 미평가 상태이고, 이전 CE와 현재 Huber를 직접 비교하지 않았다.

## 다음 수치 개입 제안 — 아직 실행하지 않음

별도 namespace에서 **동일 목적식의 block generalized-Newton + 고정 Armijo** solver 한 가지만 바꾸는 수치 실험을 제안한다. λI가 포함된 두 253×253 SPD block을 풀어 방향을 구하고, alpha=1에서 시작해 0.5씩 줄이며 c1=1e−4 Armijo 조건을 적용한다. 비영 gradient에서 SPD Newton 방향은 descent 방향이며, Huber의 C1 성질로 backtracking을 정의할 수 있다. 기존 zero 초기화, 특징·target·λ·분모·1,000회 반복/2,000회 objective 호출 한도를 유지하고 실패 가중치로 warm start하지 않는다. 초기 평가와 모든 line-search trial을 호출 예산에 포함한다. 채택점의 기존 gradient로 Linf≤1e−8 및 gap 상한≤1e−6을 모두 확인하며, 한도 소진·비정상 수치·선형계 실패 시 중단한다.

큰 terminal 조건수와 계속 감소한 목적함수는 곡률을 직접 사용하는 solver를 검토할 관측 근거다. 이것이 현재 실패의 유일 원인이라는 결론이나 다음 solver의 수렴·T/R 개선 보장은 아니다. 이번 namespace에서는 재시작하지 않았고, 제안의 실제 실행도 없다.

[검산 JSON](REJECTED_FIT_VERIFICATION.json) · [중단 상태](REJECTED_R0_ONLY.json) · [고정 학습 계약](TRAIN_PROTOCOL.json) · [학습 전 검산](PREFIT_REVIEW_KO.md)
