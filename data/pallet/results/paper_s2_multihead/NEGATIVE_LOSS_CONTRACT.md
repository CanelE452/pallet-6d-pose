# NEGATIVE LOSS CONTRACT

```
공통 hyperparameter = q  (target weighted gradient ratio)
raw lambda 는 파생값:   lambda_neg_s = q / r_s
q candidates            {0.05, 0.10, 0.20}
```

문서·논문 표기는 **"gradient-normalized negative weighting"** 으로 한다.
"seed-specific lambda" 라고 쓰지 않는다 — 모든 run 에 같은 개입 강도 `q` 를 적용하며,
raw lambda 가 seed 마다 다른 것은 source model 의 negative sensitivity 가 달라서다.

## 손실

```
L = L_line(B_L) + lambda_corner * L_corner(B_C) + lambda_neg * L_neg(B_N)
    backward 1회 / optimizer.step 1회
```

negative 전용 step 은 없다. N0 와 N1 은 optimizer step 수와 scheduler 궤적이 동일하고
차이는 합에 항이 하나 더 있는 것뿐이다.

## L_neg

```
stages          TRAINABLE_BELIEF_STAGES = (4,5,6)   ← positive 와 동일 계약
channels        belief 9 (8 corner + centroid)
target          0, dense (validity mask 없음)
L_neg           mean over stages of mean(B_pred^2)
affinity        미사용 — heads_from_f50 이 계산하되 corner_forward 가 버리므로 INACTIVE
```

### 왜 별도 분기가 필요한가 (실측)

```
corner_loss(valid 전부 False) = 0.000000e+00
corner_loss(valid 전부 True)  = 9.967461e-01
```

negative 를 positive 경로에 `valid=[0,…,0]` 로 넣으면 **정확히 아무것도 계산되지 않는다.**
`object_present == False` 를 명시적으로 읽어 이 분기로 보낸다.

## 검증 (전부 실측, 구조 근거 아님)

```
gradient 격리    corner_late 1.66e-05 > 0   corner_head 1.56e-03 > 0
                 line_late 0 · line_head 0 · shared_early 0   (grad 텐서 개수 0)
line parity      20-step deterministic replay, lambda_neg=1e-3 (일부러 0 아닌 값)
                 logit diff 0.0 · line loss diff 0.0 · line param diff 0.0
```

## calibration (optimizer step 0)

`autograd.grad` 만 사용했고 optimizer 를 생성하지도 호출하지도 않았다.
64 batch × 2 seed, `theta_C` = corner private late + belief head.

```
        g_pos median   g_neg median    r median    p10     p25      p75      p90
seed1     1.438e-04      1.053e-02      61.641    1.372   2.411   131.541  195.735
seed2     1.415e-04      6.000e-04       5.685    1.567   2.102    54.507   98.060

lambda = q / r
        q=0.05      q=0.10      q=0.20
seed1   0.000811    0.001622    0.003245
seed2   0.008795    0.017589    0.035178
```

### ⚠ ratio 분포가 매우 넓다 — 기록해 둘 것

`g_pos` 는 두 seed 가 거의 같은데(1.44e-04 vs 1.42e-04) `g_neg` 가 17.5배 다르다.
그리고 batch 간 분산이 더 크다 — seed1 은 p10 1.37 에서 p90 195.7 까지 **140배** 퍼져 있다.

즉 median 으로 잡은 `r` 은 대표값이지 안정적인 상수가 아니다. 같은 `q` 라도 batch 에 따라
실제 개입 강도가 크게 흔들린다. 이것은 λ 선택의 결함이 아니라 **negative gradient 자체가
프레임에 따라 극단적으로 다르다**는 성질이고, 아래 해석 제약을 만든다.

```
- q 는 "평균적으로" 목표 비율을 맞출 뿐 매 step 을 맞추지 않는다
- 따라서 q screen 결과를 "정확히 이 강도에서의 효과" 로 읽지 않는다
- 잔여 FP 가 특정 category 에 몰리는지는 PHASE 9 에서 별도로 본다
```

## 절차

```
PHASE 6   seed1 에서만 q screen (500 step), positive safety 우선 gate 로 q* 하나 선택
MAIN      seed1 lambda = q* / r_seed1 = q* / 61.641
          seed2 lambda = q* / r_seed2 = q* /  5.685
          seed2 결과를 보고 lambda 변경 금지, seed2 에서 q 재선택 금지
```
