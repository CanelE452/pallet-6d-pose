# CORNER_LA_OBLIQUE_V1 branch-curriculum screen — C0 vs C1

```
LOWANGLE_OBLIQUE_DATA_SIGNAL = False
LINE_ISOLATION_EXACT         = True   (seed1·seed2, 학습 3,000 step 후 param diff 0.000e+00)
```

⚠ **"효과 없음이 확립됐다" 가 아니라 "확립되지 않았다"** 다. 아래 표본 크기를 먼저 볼 것.

---

## 설정

```
ARCHITECTURE_LOCK  SPLIT_LATE_2HEAD (변경 0)
source             E3 @18k, 두 arm 동일
budget             3,000 step × 2 seed, marks 0/250/500/1000/2000/3000
two-stream         line: BROAD (C0/C1 동일)  corner: C0=BROAD / C1=BROAD+LA
LA mixture         corner batch 8 = BROAD 7 + LA 1 (12.5%), 버킷 50:50
yaw 규약           45 - facing_margin (데이터셋 정의)
평가 population    D2+D3+D4 = short-screen development population
```

**line 은 배선 불변식으로 유지됐다** — 20-step replay 에서 0.0, 그리고 3,000 step 학습
후 `line_late` 파라미터 max|diff| 가 두 seed 모두 **정확히 0.000e+00**. 따라서 두 arm 의
차이는 전부 corner stream 에서 온 것이다.

---

## 결과 (step 3000, T1/T2 = 저앙각 × |yaw|>=15)

```
                        obs_rms      R        t      5cm5
seed1 C0  T1 (n=44)      11.40     6.51   0.3168   0.0455
      C1  T1              9.91     6.17   0.3129   0.0682
seed1 C0  T2 (n=51)      26.63    24.94   0.5733   0.0000
      C1  T2             21.85    21.30   0.5606   0.0196
seed2 C0  T1              9.69     7.48   0.3712   0.0909
      C1  T1             10.26     6.27   0.3737   0.0909
seed2 C0  T2             29.31    24.23   0.6115   0.0196
      C1  T2             30.53    19.89   0.8385   0.0196
```

### gate

```
             rms      frs     scale       R        t      5cm5     판정
s1 T1     +13.03%  +16.70%  -19.03%   +5.20%   +1.25%  +2.27pp   FAIL (t)
s1 T2     +17.96%  +15.84%   +2.33%  +14.60%   +2.21%  +1.96pp   FAIL (t)
s2 T1      -5.91%   -4.52%  +30.24%  +16.16%   -0.66%  +0.00pp   FAIL (rms, t)
s2 T2      -4.15%   +2.40%   +7.25%  +17.93%  -37.13%  +0.00pp   FAIL (rms, t)
```

### 안전 절 — 여기서 실제 손상이 나온다

```
s1 T3 (저앙각 정면, n=35)   R -11.64%   t  +5.04%   SAFE=False
s2 T3                        R  -5.89%   t -65.67%   SAFE=False
s1 T4 (그 외, n=1406)        R  +2.70%   t  -1.79%   SAFE=True
s2 T4                        R  -6.75%   t  -3.45%   SAFE=False
```

**저앙각 정면(T3)이 두 seed 모두 나빠졌다.** LA 는 |yaw|>=15 만 담고 있어 corner
stream 의 12.5% 를 비정면으로 채운 만큼 정면 노출이 줄었고, 그 대가가 T3 에 나타난 것으로
보인다 [추정 — 노출 감소가 원인인지 별도 통제는 하지 않았다].

---

## paired frame bootstrap (10,000, seed 분리)

```
                              seed1                              seed2
T1 obs_rms    +10.69 [-13.30,+20.10] P.851     -5.94 [-28.93, +8.82] P.246
T1 frs        +16.84 [-21.49,+36.93] P.872     -6.04 [-46.84,+24.49] P.363
T1 R           +5.32 [-32.64,+28.82] P.642    +14.16 [-29.86,+40.29] P.806
T1 t           -0.52 [-96.40,+30.37] P.477     +5.05 [-37.84,+35.52] P.602
T2 obs_rms     +8.55 [-67.49,+44.73] P.601     -0.44 [-52.94,+46.83] P.492
T2 frs        +16.98 [-19.14,+46.62] P.883    +14.31 [-37.92,+48.20] P.688
T2 R          +17.03 [-32.88,+64.08] P.817    +20.06 [-29.30,+51.49] P.813
T2 t           -0.43 [-90.04,+52.68] P.483    -23.77 [-103.98,+31.35] P.213
T3 t           +3.25 [-59.23,+37.51] P.613    -67.41 [-171.47, +2.22] P.026  ← 손상
```

**24개 비교 중 CI 가 0 을 배제하는 것은 하나뿐이고, 그것은 개선이 아니라 손상이다.**

유일하게 방향이 일관된 것은 **rotation** 이다 — T1/T2 × 2 seed 에서 4/4 양수
(+5.3, +17.0, +14.2, +20.1), P 0.64~0.82. 그러나 어느 CI 도 0 을 배제하지 않는다.
주장된 기전인 corner pixel RMS 는 **seed 간 부호가 뒤집힌다**(+10.7/+8.6 vs −5.9/−0.4).

---

## 이 screen 은 검정력이 부족하다

```
T1 n=44   T2 n=51   T3 n=35        (T4 만 n=1406)
LA 노출 12.5%, 3,000 step
```

표본 40~50 에 CI 폭이 ±30~100% 다. 이 설계로는 20~30% 미만의 효과를 가려낼 수 없다.
따라서 `False` 는 **"이 예산에서 확립되지 않았다"** 이지 "5K 가 쓸모없다" 가 아니다.

---

## 판정과 다음

브리프의 중단 규칙대로 **여기서 멈춘다.**

```
V_vis control (C1_VCTRL)   실행 안 함 — C1 PASS 조건 미충족
EDGE_HARD (C2)             실행 안 함 — 같은 이유
theta-only fusion 재평가   실행 안 함 — corner geometry 개선이 확립되지 않음
long confirm (25k)         DO_NOT_RUN
```

데이터셋 자체에는 문제가 없다(QA 전부 재현, collision 0). 문제는 **이 screen 의
검정력**이며, 다시 시도한다면 표본이 큰 평가셋이나 더 긴 예산으로 **새 사전등록**이
필요하다. 결과를 보고 gate 를 낮추지 않는다.

산출: `branch_curriculum_{C0,C1}_seed{1,2}.json`,
`branch_curriculum_report_step3000.json`, `branch_curriculum_bootstrap.json`,
`branch_curriculum_parity_seed{1,2}.json`, `PURPOSE.md`.
스크립트: `scripts/stage0/multihead/mh_curriculum.py`, `mh_curriculum_report.py`.
