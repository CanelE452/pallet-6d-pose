# THETA_ONLY_POSE_ALIGNED_SELECTION — GATE B

```
THETA_ONLY_POSE_ALIGNED_CONFIRMED = False
```

별도 이름의 exploratory 실험이다. 기존 판정은 그대로 유지된다 —
`THETA_ONLY_LINE_USEFUL = False`. 이 실험은 그 실패가 **line 정보** 때문인지
**selection rule** 때문인지를 가른다.

---

## PHASE 3 — 오염 감사와 D4

```
dev 총            6242
오염됨            1024   (D2 512 + D3 512)
미접촉            5218
D4_THETA_CONFIRM512   512   D0/D2/D3 겹침 각각 0
                            층화 오차 vs dev = 0.0012
                            sha256 5f3f464c512a1f9e...
```

`MH_TRAIN` 은 모든 학습 run 이 통째로 스트리밍하므로 confirmation 으로 쓸 수 없다.

⚠ `FRAME_DISJOINT_ONLY = True`, `SCENE_INDEPENDENT = False` — dev 는 17 group 이고
D4 도 같은 group 에서 뽑는다. sealed real test 는 접근하지 않았다.

---

## PHASE 4 — 새 selection rule, 그리고 그것이 아무것도 바꾸지 않았다는 사실

옛 rule 은 생존자 중 **R median 최소**를 골랐다. 새 rule 은 두 축을 상대 단위로 함께 본다.

```
J(lambda) = sqrt( (R_lam / R_point) * (t_lam / t_point) )     최소화, 동률이면 작은 lambda
안전 필터  t 열화 >3% | 5cm5deg 감소 | solve 열화 >1pp   (변경 없음)
grid       {0.03, 0.1, 0.3, 1.0, 3.0}   (기존 잠금값 재사용)
```

D0 표 위에서 재계산했다 — **어떤 프레임도 다시 풀지 않았다.**

```
seed1  P0 R 6.6752 t 0.18182            seed2  P0 R 7.4077 t 0.19497
  0.03  R 6.5623 t 0.17750  J 0.97966     0.03  R 7.0374 t 0.18808  J 0.95731
  0.1   R 6.5577 t 0.17701  J 0.97796     0.1   R 7.0174 t 0.18936  J 0.95919
  0.3   R 6.4803 t 0.17831  J 0.97574     0.3   R 6.9392 t 0.18731  J 0.94866
  1.0   R 5.9116 t 0.17726  J 0.92919     1.0   R 6.2558 t 0.18858  J 0.90378
  3.0   R 4.9057 t 0.18401  J 0.86242     3.0   R 5.2368 t 0.20333  reject: t, 5cm5deg
  → 3.0  (옛 rule 도 3.0)                  → 1.0  (옛 rule 도 1.0)
```

### ★ 가설이 여기서 무너진다

**새 objective 가 옛 objective 와 정확히 같은 λ 를 고른다.** 두 seed 모두.

이유는 표에 그대로 보인다: D0 에서 translation 은 λ 에 거의 반응하지 않는다
(seed1: 0.1775 / 0.1770 / 0.1783 / 0.1773 / 0.1840). 그래서 기하평균이 rotation 항에
지배되고, J 도 grid 끝까지 단조 감소한다.

즉 문제는 **"rule 이 translation 을 안 봤다" 가 아니었다.** D0 에서는 translation 손상이
아예 나타나지 않고, 손상은 out-of-sample 에서만 발현한다(D2 −3.9%, D3 −8.3%).
**selection objective 의 결함이 아니라 D0 → held-out 일반화 격차다.** 어떤
D0 기반 규칙도 이 비용을 미리 볼 수 없다.

---

## PHASE 5 — D4 단일 확인

λ 가 옛 값과 같으므로, D4 는 "다른 rule 의 시험" 이 아니라 **미접촉 population 에서의
재현 검증**이 되었다. 그대로 진행했다.

```
                  n     T0 (point-only)          T1 (full line)           T2 (theta-only)
s1 ALL           512  R 7.146 t 0.2200 .1016   R 6.846 t 0.1976 .0918   R 5.345 t 0.2152 .1113
s1 V<8           171  R 9.062 t 0.2415 .0643   R 8.309 t 0.2514 .0585   R 5.124 t 0.2605 .0643
s2 ALL           512  R 7.577 t 0.2277 .1035   R 6.674 t 0.2663 .0703   R 6.454 t 0.2245 .1289
s2 V<8           171  R 10.386 t 0.2693 .0643  R 7.252 t 0.3047 .0585   R 7.202 t 0.2995 .0819
```

### gate

```
s1  ALL R +25.20%  t +2.20%  5cm5 +0.97pp  | V<8 R +43.45%  t −7.87%   FAIL
s2  ALL R +14.82%  t +1.39%  5cm5 +2.54pp  | V<8 R +30.66%  t −11.20%  FAIL
```

**ALL 은 두 seed 모두 전 조건을 통과한다** — rotation 이 크게 오르고, translation 이
(악화가 아니라) **소폭 개선**되며, 5cm5deg 도 두 seed 다 오른다. 이건 D2/D3 에서
seed1 이 ALL t 와 5cm5deg 로 탈락했던 것보다 훨씬 나은 그림이다.

탈락은 **V<8 translation 절 하나**에서만 난다(−7.87%, −11.20% vs 허용 −5%).

### paired frame bootstrap (10,000, seed 분리, D4 단독)

```
                        effect      CI95                P
s1 ALL|R               +24.46  [+16.57, +31.36]     1.0000
s1 V=8|R               +11.97  [ +2.77, +21.31]     0.9945
s1 V<8|R               +43.94  [+33.02, +51.70]     1.0000
s1 near/large|R        +25.10  [ +6.92, +40.06]     0.9956
s2 ALL|R               +15.53  [ +8.98, +22.10]     1.0000
s2 V=8|R               +11.55  [ +3.96, +18.84]     0.9998
s2 V<8|R               +29.84  [+18.19, +39.79]     1.0000
s2 low-angle|R         +16.86  [ +4.80, +30.52]     0.9951

s1 ALL|t                +0.52  [−17.52, +11.49]     0.5372
s1 V<8|t               −10.61  [−48.68, +15.39]     0.2159
s1 near/large|t        −20.22  [−50.19,  +4.02]     0.0474   ← 유일하게 손상이 유의에 근접
s2 ALL|t                +3.50  [ −4.58, +10.92]     0.7841
s2 V<8|t                −3.00  [−28.23, +10.80]     0.3132
```

**rotation 은 10개 subset×seed 중 8개에서 CI 가 0 을 배제한다**(예외: seed1 low-angle).
**translation 손상은 어디에서도 확립되지 않는다** — gate 를 떨어뜨린 V<8 t 조차
CI 가 [−48.68, +15.39] / [−28.23, +10.80] 로 0 을 넉넉히 포함한다.

⚠ 즉 **gate 를 떨어뜨린 항목이 이 실험에서 불확실성이 가장 큰 항목**이다. gate 는
점추정으로 쓰여 있고 그대로 유지한다(결과를 보고 gate 를 바꾸지 않는다). 다만
"V<8 에서 translation 이 나빠진다" 를 확립된 사실로 서술해서는 안 된다.

### full-line 이 여전히 나쁘다는 것도 재현된다

```
s2 ALL   T1 t 0.2663 (T0 0.2277 대비 −17.0%)   5cm5 0.0703 (T0 0.1035)
```

`rho` 가 translation 을 망가뜨린다는 결론이 세 번째 population 에서 재현됐다.

---

## 판정

```
THETA_ONLY_POSE_ALIGNED_CONFIRMED = False     (V<8 t 절)
```

그러나 이 실험이 실제로 확립한 것은 다음이다.

```
line orientation 의 rotation 기여는 D2·D3·D4 세 population, 두 seed 에서 반복되고
CI 가 0 을 배제한다. V<8 에서 가장 크다(+30~44%).
gate 를 막는 것은 translation 이며, 그 손상은 통계적으로 확립되지 않았다.
```

브리프의 금지 문장을 지킨다 — "line 은 pose 에 쓸모없다" 로 쓰지 않는다.
