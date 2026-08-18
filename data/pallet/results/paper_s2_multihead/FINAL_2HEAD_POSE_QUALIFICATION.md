# FINAL 2-HEAD POSE QUALIFICATION — PHASE 4-9

두 gate 모두 사전등록대로 판정했고 둘 다 실패다.

```
E3_NATIVE_POINT_LINE_USEFUL   False     PHASE 4
SCALE_PREDICTABLE             False     PHASE 6
PREDICTED_SCALE_HELPS_POSE    False     PHASE 8
TWO_HEAD_POSE_QUALIFIED       False     PHASE 9
SCALE_BLOCKER_REMAINS         True
```

대상은 채택 아키텍처인 **E3_SPLIT_LATE 의 자기 예측**(`e3confirm25k`, 25k×2 seed)이다.
이전 PHASE 4 이전 판정들이 쓰던 A1(fully-shared) 예측이 아니다.

---

## PHASE 4 — native joint point+line solver

Huber-robust least squares 로 corner reprojection 과 line incidence 를 함께 푼다.
λ 는 D0 에서 calibrate, D2 에서 1회 평가. 사전등록 기준은 **두 seed 모두**.

```
run/seed      R F0    R F2     t F0     t F2   5cm5 변화
A1 seed1     7.830   7.125   0.2244   0.2389    -1.95pp
A1 seed2     8.067   6.964   0.2397   0.2951    -5.86pp
E3 seed1     7.232   6.641   0.1825   0.1733    +0.58pp
E3 seed2     7.539   7.137   0.1941   0.2133    -6.83pp
```

사전등록 예상 1("E3 에서는 translation 손상이 줄어든다")은 **맞았다** —
A1 은 −7.3%/−23.1% 손상, E3 는 +5.0%/−9.9%. 그러나 seed1 만 통과하고 seed2 가
ALL t −9.9%, 5cm5deg −6.83pp 로 떨어져 **two-of-two 미달**이다.

### 결함 기록 (고치지 않고 남긴다)

λ 선택이 **회전 median 만** 본다. 회전은 정확히 line 이 잘하는 축이라 selection 이
line 을 과대 가중하고 translation 이 대가를 치른다. seed2 의 D0 표에서 λ=0.3 이
t 0.1830 으로 λ=1.0 의 0.2096 보다 훨씬 나았는데 R 이 0.43° 나빠서 밀렸다.

```
E3 seed1  0.03→R6.566/t0.1770  0.1→6.582/0.1774  0.3→6.495/0.1672  1.0→6.496/0.2324  선택 0.3
E3 seed2  0.03→R7.040/t0.1880  0.1→7.083/0.1885  0.3→6.913/0.1830  1.0→6.485/0.2096  선택 1.0
```

결과를 본 뒤 selection objective 를 바꾸면 그 판정은 더 이상 사전등록이 아니다.
결함으로 기록만 하고 판정은 FALSE 로 둔다.

---

## PHASE 5 — scale target 계약 (재정의 없음)

`mh_diagnose.run_scaleoracle` 의 per-frame 분기를 그대로 읽어 확정했다
(`mh_diagnose.py:1119-1124`). 새 정의를 만들지 않았다.

```
corners       pred[:8] / gt[:8]                 8 코너, centroid 제외
coordinates   grid frame — grid_to_pixels 이전
spread(x)     mean_j || x_j − mean(x) ||        RMS 아니라 평균 반지름
target        s* = spread(gt) / max(spread(pred), 1e-9)
centre        pred.mean(0) — GT centroid 아니라 예측 centroid
application   pred ← centre + (pred − centre) * s*
order         grid frame 에서 적용 → grid_to_pixels → solve
```

`spread` 를 RMS 로, centre 를 GT centroid 로 "개선"하면 oracle 수치의 의미가 조용히
바뀐다. `mh_scale.apply_scale` 하나만 쓰도록 묶어 drift 를 막았다.

---

## PHASE 6 — 예측 가능한가 (Ridge, D0 fit → D2 1회)

alpha grid `[1e-3 … 1e3]` 와 gate 를 **돌리기 전에** 고정했다. GT 는 target 으로만
들어가고 feature 로는 절대 들어가지 않는다. **물체 치수도 feature 에서 뺐다** —
`CG.solve` 가 이미 model point 를 받으므로 남은 scale 오차는 치수 문제가 아니라
corner 배치 문제고, 치수를 넣으면 조용히 dims-known 방법이 된다.

```
                       seed1                    seed2
                  R2 D2   vs const %       R2 D2   vs const %
S0_constant      -0.000     +1.18         -0.000     +3.02
S1_geometry      +0.048     +5.94         +0.139    +16.46
S2P_point        +0.099    +10.94         +0.159    +18.42
S2L_line         +0.137    +14.75         +0.130    +15.64
S2PL_both        +0.107    +11.77         +0.174    +19.87
                                   gate: R2 > 0.30 AND vs const > 20%
```

신호는 **있다**(R² 가 두 seed 모두 0 보다 확실히 크다). 그러나 gate 에 한참 못 미치고,
어느 block 이 최선인지도 seed 간에 뒤집힌다(seed1 은 line, seed2 는 point/both).
`SCALE_PREDICTABLE = False`.

s\* 의 median 은 D0 1.018 / 1.023, D2 1.021 / 1.022 다. **bias 는 이미 2% 수준**으로
작다 — E3 가 A1 의 4~5% 축소를 절반으로 줄여놓은 상태다. 남은 것은 대부분
frame 마다 다른 성분이고, 그게 예측이 안 되는 부분이다.

---

## PHASE 8 — 예측된 보정이 pose 를 움직이는가

R² 가 낮아도 방향만 맞으면 pose 는 개선될 수 있다. 그래서 회귀 점수에서 멈추지 않고
실제로 풀어봤다. feature set 선택은 **D0 CV MSE 로만** 했다(D2 로 고른 PHASE 6 의
순위를 쓰면 평가셋이 선택에 관여한다) → `S2L_line`.

```
                    seed1                        seed2
                R      t     5cm5          R      t     5cm5
C0_uncorrected 7.232 0.1825 0.1465      7.539 0.1941 0.1367
C_const        7.224 0.1864 0.1328      7.527 0.1808 0.1250
C1_geometry    7.165 0.1939 0.0957      7.525 0.1827 0.1016
C2P_point      7.129 0.1985 0.1055      7.528 0.1850 0.0859
C2L_line       7.115 0.2071 0.0762      7.538 0.1870 0.0938
C2PL_both      7.135 0.1853 0.1016      7.557 0.1943 0.0977
C_oracle_GT    7.127 0.1260 0.1816      7.561 0.1296 0.1699
```

`PREDICTED_SCALE_HELPS_POSE = False`. seed1 은 t −13.4%, seed2 는 +3.7% 지만
**5cm5deg 는 두 seed 다 무너진다**(−7.03pp / −4.29pp). 예측 보정을 쓴 여섯 arm 중
5cm5deg 가 C0 를 넘는 것은 하나도 없다.

### 왜 그런지는 명확하다

PnP 의 translation 은 scale 에 거의 직접 비례한다. 그래서 곱셈 보정은 **bias 를 줄인
만큼 variance 를 넣는다**. 지금 bias 는 2% 뿐인데 R² 0.13 짜리 예측을 곱하면
줄이는 것보다 넣는 게 많다. oracle 이 되는 이유는 정확해서지 보정이라서가 아니다.
상수 보정조차 5cm5deg 를 깎는다(−1.4pp / −1.2pp)는 것이 같은 이야기다.

**oracle 은 여전히 크다** — t +30.98% / +33.26%, 5cm5deg +3.5pp / +3.3pp.
레버가 사라진 게 아니라, **이 모델의 출력만으로는 잡히지 않는다**는 것이 결론이다.

---

## PHASE 9 — 최종 2-head gate

보정된 같은 corner 를 point-only(H0) 로 풀 때와 line 을 함께 넣어(H1) 풀 때.

```
             seed1                          seed2
         R      t     5cm5             R      t     5cm5
H0    7.115  0.2071  0.0762         7.538  0.1870  0.0938
H1    6.708  0.1944  0.0938         7.211  0.2043  0.0742
      R +5.72  t +6.11  +1.76pp      R +4.33  t −9.28  −1.96pp
```

`TWO_HEAD_POSE_QUALIFIED = False`. **회전은 두 seed 모두 개선된다**(+5.7% / +4.3%,
gate 3% 통과). 그러나 seed2 가 translation −9.28%(허용 −2%)와 5cm5deg −1.96pp
(허용 −1pp)로 떨어진다. PHASE 4 와 **정확히 같은 seed2 패턴**이고, 같은 λ=1.0 이
원인이다.

---

## 그래서 논문에 뭐라고 쓰나

세 가지를 분리해서 써야 한다.

1. **2-head 아키텍처(E3)는 정당하다** — corner 가 line 의 128ch 병목 이전에 전용 late
   경로를 가지면 corner 정확도 +16~21%, PATH-C 회전 +14~15% 가 나오고 line 은
   구조적으로 무손실이다. 이건 E2/E4 대조로 확립됐다(`architecture.md`).
2. **그러나 line 은 pose contributor 로서는 자격 미달이다** — 보정 여부와 무관하게,
   corner 로 푼 pose 에 line 제약을 더하면 회전은 얻고 translation 을 잃는다.
   두 seed 를 다 통과한 적이 한 번도 없다(PHASE 4, PHASE 9). line branch 의 가치는
   pose 정확도가 아니라 다른 데서 주장해야 한다.
3. **translation 병목은 여전히 per-frame scale 이고, 열려 있다** — oracle 로 +31~33%,
   상수로 0~7%, 학습된 선형 예측으로 **0 이하**. 다음 작업은 이 양을 예측하는
   feature 를 더 찾는 게 아니라, scale 이 애초에 덜 틀어지게 만드는 표현·loss 다.

---

## 범위와 한계

- 전부 synthetic `v2_prod40k_clean_merged` 의 D2 dev split(512 frame), seed 2 개다.
  real 전이 주장 없음. sealed set 미접근.
- Ridge 는 **선형**이다. 비선형 예측기라면 R² 가 오를 수 있다. 다만 PHASE 8 의
  실패 기전(bias 2% 대비 variance 주입)은 R² 가 상당히 올라야 뒤집히고, 그 임계값을
  결과를 본 뒤에 정하는 것은 사전등록이 아니다.
- λ 선택 결함(회전 median 단독)은 두 gate 의 seed2 실패에 공통으로 얹혀 있다.
  고치면 결론이 바뀔 수 있으나, 그건 결과를 보고 고치는 것이므로 **새 사전등록으로
  따로 돌려야** 한다.

산출: `scale_ridge_e3confirm25k.json`, `corrected_point_line_e3confirm25k.json`,
`point_line_solver_e3confirm25k.json`.
스크립트: `scripts/stage0/multihead/mh_scale.py`, `mh_corrected.py`.
