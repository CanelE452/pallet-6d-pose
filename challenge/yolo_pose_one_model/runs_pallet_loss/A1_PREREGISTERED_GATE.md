# A1 사전등록 GATE

작성 2026-08-22, **학습 착수 전**. 결과를 보고 이 문서를 고치지 않는다.
판정 기준은 `a1_verdict.py` 에 하드코딩한다 — 문서와 코드가 갈라지지 않게.

## 0. 지표를 고르며 뒤집은 것 (기록)

처음에 "frame 단위 yaw180 flip 률 감소" 를 primary 로 잡았다가 **버렸다.**
CASE 1(전부 SYM180)에서는 뒤집힌 배정이 물리적으로 **오답이 아니다.** min() 을
쓰면 모델이 어느 쪽으로 수렴하든 정답이므로, flip 률 변화는 개선도 악화도 아닌
자유도다. 그걸 primary 로 쓰면 아무 의미 없는 수를 gate 로 삼게 된다.

⇒ primary 는 **대칭 몫 오차**(symmetry-quotient error) 다. flip 률은 서술용으로만
남긴다.

## 1. 진단셋 (학습 전 고정)

```
A1_DIAG_SET = broad40k − v1_fixed_matched10k train 9,867 stem,  목표 ≥ 4,000 프레임
```
val 133 은 검정력이 없다 (base rate 2.4% → 기대 flip 3 프레임). 근거는
`A1_LOSS_CONTRACT.md` §4.

## 2. 지표

```
M1  primary   E_sym  = median_i  min(d_id(i), d_180(i))        A1_DIAG_SET
              (SYM asset). CASE 2 의 ASYM asset 은 min 대신 d_id 를 쓴다.
M2  no-harm   pose mAP50-95 (표준 val 133, ultralytics 기본 경로)
M3  서술      frame 단위 flip 률 — gate 아님. CASE 1 에서는 오답이 아니다.
M4  paper     real DEV — ★ 지금 계산 불가. realdev161 fixed GT 미존재 +
              membership V2 미승인. gate 에 넣지 않는다.
```

## 3. 판정 (하드코딩)

frame 단위 paired bootstrap B=10,000 (A0·A1 이 같은 프레임을 보므로 paired).

```
GO         M1: ΔE_sym = E_sym(A0) − E_sym(A1) 의 95% CI 하한 > 0
       AND M2: mAP50-95(A1) ≥ mAP50-95(A0) − 0.005
NO_SIGNAL  위를 못 넘김
HARM       M2 위반 (M1 과 무관하게)
```

`NO_SIGNAL` 이면 **추가 seed·hyperparameter sweep 금지** (사용자 지시).
`GO` 여도 seed 1개이므로 논문 main claim 이 아니라 `SINGLE_SEED_SCREEN` 이다.
real 전이는 M4 가 풀린 뒤에 따로 본다 — synthetic 향상만으로 method 채택 금지.

## 4. 미리 인정하는 한계

- seed 1개. corner 계열 지표는 seed 산포가 작다고 알려져 있으나(line 계열 15–19%
  와 달리) 그 값은 다른 지표에서 얻은 것이다. 이 지표에 그대로 옮기지 않는다.
- min() 은 loss 를 **구조적으로 낮춘다**(두 값 중 작은 쪽). 따라서 loss 값 자체의
  A0/A1 비교는 무의미하다. M1 은 두 모델을 **같은 식**으로 재므로 그 함정을 피한다.
- CASE 2 라면 role 항은 §3(contract) 표대로 거의 죽어 있다. 그때 A1 의 신호는
  SYM 가지에서만 나온다 — 이건 학습 전에 예측한 것이지 사후 해석이 아니다.
