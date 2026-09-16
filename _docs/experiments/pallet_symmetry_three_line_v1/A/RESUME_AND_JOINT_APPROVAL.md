# A 재개 — 공동 대칭 선택 승인 및 학습 전 점검

사용자 승인: “일단 그렇게 해봐”. 구현·검증 후 “잠시만 멈춰봐” 요청으로 본학습 전 중지했고, “다시해줘” 요청으로 재개했다.

이 문서는 이전 `AWAITING_USER_OBJECTIVE_DECISION` 상태를 해소하는 후속 기록이다. 이전 B 결과·A0 및 중단 당시 기록은 삭제하지 않는다. `FINAL_DECISION_KO.md`의 A 대기 상태는 이전 인계 시점의 상태이며, 현재 진행률은 `A/LIVE_STATUS.json`과 `A/runs/*.json`을 따른다.

## 승인된 목적함수

물체별 허용 회전 하나를 선택하되, 두 head의 stock 가중 위치·visibility 비용과 **head별 batch-global RLE clamp**를 보존한 전체 합을 공동 최소화한다. 기존 독립 object-min과 같은 수식이라고 주장하지 않는다.

각 선택의 선형 하한이 실제 비용과 같으면 전역 최적임을 인증한다. 이 인증이 성립하지 않을 때는 one-hot 정수계획으로 gap 0 해를 구한다. 근사 좌표하강이나 head별 독립 선택으로 대체하지 않는다. 해 인증 실패는 해당 fit을 중단한다.

선택은 detach/no-grad 비용으로 수행하고, 선택한 전체 `(x,y,annotated)` tuple을 원래 `E2ELoss/PoseLoss26`에 전달해 loss와 gradient를 계산한다. 원본 라이브러리는 수정하지 않았다.

## 실제 검사

- 전체 CPU 회귀검사 36/36 PASS. 무작위 소형 60개 문제를 모든 조합 열거와 대조했고 MILP/하한 인증 경로를 모두 검사했다.
- 실제 GPU 두 이미지·두 head, C4 조합 16개에서 stock 전체 비용과 재구성 비용의 최대 차이 **3.85e-6**.
- 선택된 손실 및 gradient가 stock 선택 타깃 경로와 bit-exact.
- C4 재명명 loss/gradient 불변, C2 180도 동등·90도 비동등 확인.
- 좌표만/visibility만 잘못 순열한 fixture는 비용이 달라져 오류 검출.
- 미주석 전체 객체의 유한 loss/gradient, 여러 GT 행·이미지별 순서 섞기, 부분 swap 거부 검사 통과.
- 두 cohort × 두 군 × 3 step = **총 12 smoke optimizer updates**, 본학습 초기 모델로 재사용하지 않음.
- 실제 physical batch16, FP32, 최대 allocated memory 약 4.7GB. BN running statistics는 bit-exact 불변.

## 실행 고정

`JOINT_TRAINING_LOCK.json`에 코드·target view·sampler SHA를 고정했다. 12개 fit은 동일 R0에서 별도 시작하며 각각 2,000 update다. 동일 cohort·seed의 두 군은 같은 32,000개 표본 순서, 전체 trainable parameter set, AdamW/gradient clip/BN 설정을 쓴다.

Step 기반 레시피에는 epoch callback이 없으므로 E2E head 가중치는 stock 초기값 0.8/0.2를 양 군 모두 일정하게 사용한다. 추가 head-weight schedule을 만들지 않았다. AMP/EMA/TF32/augmentation 모두 사용하지 않는다. 마지막 raw weights만 평가하며 P/DHT를 새 A에 붙이지 않는다.

각 100 step 및 종료 시 model·optimizer·RNG·sampler offset을 저장한다. SIGINT/SIGTERM은 현재 업데이트를 마친 뒤 checkpoint를 저장하고 중지한다. 타인 프로세스를 종료하거나 시스템을 재부팅하지 않는다.
