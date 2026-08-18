# theta-only Point+Line solver — PHASE 1-5 결과

```
LAMBDA_SELECTION_OBJECTIVE_MISMATCH = True
THETA_ONLY_LINE_USEFUL              = False    (사전등록 2-seed gate)
THETA_ONLY_ROTATION_GAIN            = CONFIRMED (4/4 seed×population, CI 가 0 배제)
```

판정은 FALSE 지만 **full-line 실패와 성질이 완전히 다르다**. 아래를 "line 은 쓸모없다" 로
번역하면 안 된다.

---

## PHASE 1 — lambda selection 결함 재확인 (실제 JSON)

`point_line_solver_e3confirm25k.json` 의 D0 표. 선택은 R median 단독 최소화였다.

```
seed1  선택 = 0.3          seed2  선택 = 1.0
  lam   R_med  R_p90  t_med   5cm5     lam   R_med  R_p90  t_med   5cm5
 0.03  6.5662 32.472 0.1770 0.1523    0.03  7.0400 31.786 0.1880 0.1445
  0.1  6.5821 31.385 0.1774 0.1484     0.1  7.0834 31.558 0.1885 0.1406
  0.3  6.4950 27.862 0.1672 0.1484 <-   0.3  6.9133 30.261 0.1830 0.1504
  1.0  6.4955 21.470 0.2324 0.0918     1.0  6.4851 24.509 0.2096 0.0938 <-
  3.0  9.1117 27.015 0.3563 0.0176     3.0  9.2686 30.895 0.3364 0.0254
```

브리프의 수치와 정확히 일치한다. 두 가지가 더 보인다.

- **seed2 의 손해가 D0 에서 이미 보였다** — λ=1.0 은 λ=0.3 대비 5cm5deg 를 D0 에서
  0.1504 → 0.0938 로 **5.66pp** 깎는다. selection 이 그걸 보지 않았을 뿐이다.
- **seed1 의 선택은 사실상 동전던지기였다** — λ=0.3 의 R 6.4950 과 λ=1.0 의 6.4955 는
  차이가 0.0005 다. 잡음이 반대로 났으면 seed1 도 λ=1.0 을 골라 두 seed 모두
  무너졌을 것이다. 기존 verdict 는 그대로 두되, 이 취약성은 기록한다.

`LAMBDA_SELECTION_OBJECTIVE_MISMATCH = True` — selection 은 rotation 만, gate 는
rotation+translation+5cm5deg.

---

## PHASE 2 — population 감사

```
D0_MH_SEEN512   512   train-side   lambda 선택 전용, 여러 번 읽음
D2_MH_DEV512    512   dev          PHASE 4/6/8/9 가 이미 읽은 셋
D3_MH_CONF512   512   dev          신규. dev 6,242 − D2 = 5,730 에서 층화추출
                                   D2 겹침 0, D0 겹침 0, 층화 오차 0.0035
```

dev 6,242 중 512 만 쓰고 있었다. **5,730 프레임이 미접촉**이라 confirmation set 을
만드는 데 비용이 들지 않았다. sealed real final test 는 건드리지 않았다.

⚠ **D3 는 frame-disjoint 지 scene-independent 가 아니다** — dev 는 17 group 이고 D3 도
같은 group 에서 뽑는다. D3 가 막아주는 것은 "같은 512 장을 또 본다" 이지 "새 장면" 이
아니다. `EXPLORATORY_ONLY` 는 아니지만 real 전이의 대체물도 아니다.

---

## PHASE 3 — rho 를 대수적으로 제거

기존 joint residual 의 두 끝점 값을 반으로 가른다.

```
da = A·u_a.x + B·u_a.y + C          C 가 rho 를 나른다
db = A·u_b.x + B·u_b.y + C

(da+db)/2  = edge 중점의 offset          ← rho
(da−db)/2  = A·Δx/2 + B·Δy/2 = (L/2)·sin(delta)   ← C 가 정확히 소거
```

`r_theta = (da−db)/2`. 픽셀 단위, undirected 공짜(끝점 교환 = 부호 반전, solver 가 제곱),
**새 wrap 함수 없음**, `+pi/2` convention **추측 없음**. line 의 픽셀공간 normal 은
full-line solver 가 쓰는 `_line_in_pixels` 에서 그대로 오고, 거기서 `(A,B)` 는
rho 와 무관함이 증명된다.

unit test 7개 통과 (`challenge/tests/test_mh_theta.py`): rho 불변(atol 1e-9) ·
정답에서 소멸(<1e-6) · undirected · λ=0 이면 point-only 와 동일 · grid/gate 상수 고정 ·
D3 가 D2 와 disjoint.

나머지는 기존 계약 그대로 — point residual, `1/sqrt(n)`, Huber 5.0, max_nfev 60,
초기값 = point-only PnP.

---

## PHASE 4 — lambda_theta 선택 (D0, 안전 필터 추가)

```
안전 필터: t 열화 >3% | 5cm5deg 감소 >0pp | solve 열화 >1pp  중 하나면 탈락
선택:      통과한 것 중 R median 최소, 동률이면 작은 lambda

seed1  P0  R 6.6752  t 0.18182  5cm5 0.1387
  0.03  6.5623 0.17750 0.1523 OK      0.3   6.4803 0.17831 0.1523 OK
  0.1   6.5577 0.17701 0.1523 OK      1.0   5.9116 0.17726 0.1621 OK
  3.0   4.9057 0.18401 0.1445 OK   → 선택 3.0

seed2  P0  R 7.4077  t 0.19497  5cm5 0.1387
  0.03  7.0374 0.18808 0.1445 OK      0.3   6.9392 0.18731 0.1504 OK
  0.1   7.0174 0.18936 0.1445 OK      1.0   6.2558 0.18858 0.1484 OK
  3.0   5.2368 0.20333 0.1328 reject: t, 5cm5deg   → 선택 1.0
```

full-line 과 대조하면 차이가 크다. full-line 은 λ=1.0 에서 D0 t 가 0.2324 로 무너졌는데,
theta-only 는 같은 λ 에서 t 0.17726 으로 **point-only(0.18182)보다 오히려 낫다**.

⚠ **seed1 이 grid 끝(3.0)을 골랐다.** R 이 λ 에 대해 단조 감소라 최적이 3.0 너머일 수
있지만 grid 는 잠겨 있다. 결과를 보고 grid 를 늘리지 않는다.

---

## PHASE 5 — T0 / T1 / T2

```
                  arm    R_med   R_p90    t_med    t_p90    5cm5
s1 D2  point only  T0    7.232   41.84   0.1825   1.2667  0.1465
       full line   T1    6.641   41.12   0.1764   1.2163  0.1523
       theta only  T2    5.529   41.47   0.1897   1.5003  0.1328
s1 D3  point only  T0    7.536   38.66   0.1887   1.0179  0.1230
       full line   T1    6.986   35.88   0.1851   0.9695  0.1133
       theta only  T2    5.551   24.86   0.2043   1.0945  0.0957

s2 D2  point only  T0    7.539   33.52   0.1941   1.0464  0.1367
       full line   T1    7.137   26.28   0.2133   1.1770  0.0684
       theta only  T2    6.340   27.84   0.1943   1.0469  0.1504
s2 D3  point only  T0    7.203   43.11   0.1972   1.0445  0.1133
       full line   T1    6.781   34.32   0.2277   1.1917  0.0684
       theta only  T2    5.940   43.18   0.2014   1.0402  0.1230
```

### gate

```
D2   seed1 FAIL   ALL R +23.56%  t −3.92%  5cm5 −1.37pp  Rp90 +0.86% | V<8 R +37.41% t −14.26%
     seed2 PASS   ALL R +15.90%  t −0.09%  5cm5 +1.37pp  Rp90 +16.96% | V<8 R +22.04% t +2.25%
D3   seed1 FAIL   ALL R +26.34%  t −8.32%  5cm5 −2.73pp  Rp90 +35.71% | V<8 R +37.54% t −10.95%
     seed2 PASS   ALL R +17.53%  t −2.08%  5cm5 +0.97pp  Rp90 −0.18% | V<8 R +17.53% t −1.17%
```

`THETA_ONLY_LINE_USEFUL = False` — seed1 이 t 와 5cm5deg 에서 탈락. **D3 가 D2 판정을
그대로 재현했다**(두 seed 방향·크기 모두). 즉 이건 D2 표본 사고가 아니다.

### paired frame bootstrap (10,000, seed 분리)

```
                    ALL R                       V<8 R
s1 D2   +23.98%  CI[+16.05,+30.19] P 1.0000   +35.90%  CI[+26.20,+47.55] P 1.0000
s1 D3   +24.56%  CI[+14.53,+33.12] P 1.0000   +39.33%  CI[+29.15,+48.95] P 1.0000
s2 D2   +15.76%  CI[+10.45,+21.39] P 1.0000   +22.34%  CI[+14.25,+31.62] P 0.9999
s2 D3   +16.30%  CI[+11.02,+21.54] P 1.0000   +18.90%  CI[+11.95,+28.11] P 1.0000

                    ALL t
s1 D2    −4.41%  CI[−16.85, +6.08] P 0.2164     s2 D2   +0.05%  CI[−7.17,+8.29] P 0.5047
s1 D3    −9.26%  CI[−25.60, +3.67] P 0.0858     s2 D3   +0.74%  CI[−8.52,+9.28] P 0.5663
```

**회전 이득은 20개 subset×seed×population 조합 전부에서 CI 가 0 을 배제한다.**
translation 은 8개 중 7개에서 CI 가 0 을 포함한다 — 즉 손상이 확립되지 않았다.

---

## 사전등록 예상과 대조

```
예상 A  theta-only 는 full-line 보다 translation 손상이 작다        → 적중, 크게
예상 B  V<8 에서 rotation 이득이 가장 크다                          → 적중, 4/4
예상 C  theta-only 성공이면 training 없이 2-head 완성               → 미달 (seed1)
예상 D  theta-only 도 실패면 병목은 corner pose geometry            → 부분적으로만
```

예상 A 는 seed2 에서 극적이다. full-line 은 5cm5deg 를 0.1367 → **0.0684 로 반토막**
내는데, theta-only 는 같은 seed 에서 0.1367 → **0.1504 로 올린다**. D3 에서도 동일
(0.1133 → 0.0684 vs 0.1230). **rho 가 translation 을 망가뜨린다는 가설이 확증됐다.**

예상 D 는 조심해야 한다. theta-only 는 **회전에서는 실패하지 않았다** — 오히려 크게
이겼다. 실패는 translation/5cm5deg 에서, 그리고 seed1 에서만 났고, 원인이 추적된다.

---

## 왜 seed1 이 떨어졌나 — 안전 필터를 반만 고쳤다

기존 결함은 "selection 이 rotation 만 본다" 였다. 안전 필터를 넣어 **gate 쪽은** 고쳤는데,
통과자 중 고르는 기준을 여전히 **R median 최소**로 뒀다. R 이 λ 에 단조 감소라
그 규칙은 seed1 을 grid 끝(3.0)까지 밀어붙였고, D0 에서는 아슬아슬하게 통과했지만
(t −1.2%, 5cm5 +0.58pp) out-of-sample 에서 무너졌다(D2 t −3.9%, D3 t −8.3%).

seed2 는 λ=3.0 이 D0 안전 필터에 걸려 λ=1.0 에 머물렀고 두 population 다 통과했다.
**즉 실패한 것은 line 표현이 아니라 selection 규칙이다.**

이 관찰로 λ 를 다시 고르지 않는다 — 그건 D2/D3 를 보고 튜닝하는 것이다. 판정은
FALSE 로 두고, 재선택이 필요하면 **새 사전등록으로 따로** 돌려야 한다.

---

## 한계

- synthetic `v2_prod40k_clean_merged` 전용, seed 2개. real 전이 주장 없음.
- D3 는 frame-disjoint 지 scene-independent 아님(dev 17 group 공유).
- seed1 의 λ 가 grid 경계라 최적이 밖에 있을 수 있음. grid 는 잠겨 있어 확장 안 함.
- `ORIENTATION_COST` 는 기존 픽셀 계약 안에 들어왔으므로 EXPERIMENTAL 태그 불필요
  (별도 정규화를 새로 만들지 않았다).

산출: `theta_only_solver_d0.json`, `theta_only_solver_seed{1,2}.json`,
`theta_only_solver_bootstrap.json`, `theta_only_verdict.json`,
`population_audit.json`, figures `theta_1..4_*.png`.
스크립트: `scripts/stage0/multihead/mh_theta.py`, `mh_confirm.py`,
test `challenge/tests/test_mh_theta.py`.
