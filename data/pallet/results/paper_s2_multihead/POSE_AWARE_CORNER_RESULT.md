# pose-aware corner supervision — PHASE 6-9 결과

```
POSE_AWARE_CORNER_GAIN = False
28개 비교(2 seed × 2 population × 7 지표) 전부 CI 가 0 을 포함 — 효과 미확립
→ PHASE 13 CASE D : CURRENT_CORNER_REPRESENTATION_POSE_BIAS_REMAINS
```

theta-only 가 회전에서 20/20 으로 CI 가 0 을 **배제**했던 것과 정반대다. 여기서는
어느 지표에서도 효과가 서지 않는다.

---

## 설정

```
arm      P0  E3 continuation 그대로        L = L_line + λ_corner·L_corner
         P1  pose 감독 추가                L = ... + λ_pose·L_pose
source   E3 @18k (screen_A1_CORNER_LINE_e3confirm25k_seed{1,2}/step_18000.pth)
budget   3,000 step × 2 seed, marks 0/250/500/1000/2000/3000
optim    양 arm 동일 AdamW, CONTINUATION_OPTIMIZER=FRESH, batch·순서·aug·LR·WD 동일
L_pose   DiffPnP3DLoss (GT-seed unrolled GN → 3D corner Huber / diag)
         + LocalSoftArgmax2D, 둘 다 repo 기존 구현 재사용
λ_pose   3e-05  (D0 에서 선택, 아래)
```

**배선은 학습 전에 검증했다** — `L_pose` gradient 가 `line_late` 에 정확히 0.0,
`corner_late` 에 > 0. 학습 로그에서도 P0/P1 의 line angle·offset 이 6개 마크 전부
같은 값이다(예: s2 @3000 angle 2.2994 / offset 1.1019 양쪽 동일).

**하네스 정합**: `geometry_and_pose` 의 pose 경로가 확립된 T0 값을 재현한다
(seed1 R 7.232 / t 0.1825, seed2 R 7.539 / t 0.1941, diff < 0.0003).

⚠ **`cornerC` 정의가 기존과 다르다.** 기존 `direct_cell_median` 은 전 코너를 풀링한
median 이고, 여기 `cornerC` 는 프레임별 8코너 **평균**의 median 이라 최악 코너에 끌려
올라간다(0.63 vs 0.83). **두 수를 직접 비교하지 말 것.**

---

## λ_pose 보정 — 1차 실패는 내 오류, 2차가 본 결과

### 1차 grid `{0.01, 0.1, 1.0}` = 측정 실패 (결과 아님)

loss **값**으로 스케일을 잡았다(pose ≈ 가중 corner 의 325배). 실제로 중요한 건
gradient 였다.

```
|dL_corner/d(belief)| 5.3e-05 × λ_corner 0.035 → 1.9e-06
|dL_pose  /d(belief)| 1.35e-02 (λ_pose=1.0)     → 비 253~297배
⇒ λ=0.01 에서도 가중 corner gradient 의 약 70배, λ=1.0 이면 약 7,000배
```

세 후보 전부 파괴, 게다가 **비단조**(0.01→−16%, 0.1→−2814%, 1.0→−41%) = 발산한 run
끼리의 순서라 무의미. `pose_aware_calibration_v1_misscaled.json` 에 실패한 측정으로 보존.

기전은 결함이 아니다 — `corner_loss` 는 9×50×50 에 퍼진 MSE, `L_pose` 는 soft-argmax 로
코너당 7×7 창에 민감도가 집중된다.

### 2차 grid `{1e-5, 3e-5, 1e-4, 3e-4}` (가중 corner gradient 의 0.07~2.2배)

```
λ        affine_scale   scale gap    t          cornerC      판정
0        0.9657         —            —          —
1e-5     0.9693         +10.66%      −8.06%     +0.61%       reject: t
3e-5     0.9809         +44.35%      +0.24%     +0.99%       OK ← 선택
1e-4     0.9855         +57.84%      −17.66%    −5.25%       reject: t
3e-4     0.9918         +76.02%      −7.12%     −1.49%       reject: t
```

**이번엔 깨끗한 용량-반응이다** — scale gap 이 λ 에 단조로 닫히고(10.7→44.4→57.8→76.0%)
cornerC 붕괴가 없다. pose 감독은 설계대로 작동한다.

**그런데 scale 을 잘 닫는 λ 는 전부 t 를 해친다.** 선택된 3e-5 는 t 안전 필터를 통과한
**유일한** 값이고 동시에 가장 약한 값이다. 선택은 grid 경계가 아니었다(두 번째로 작은 값).

⚠ 이 선택의 취약성은 **결과를 보기 전에 기록했다**: t 가 λ 에 비단조(−8.06/+0.24/
−17.66/−7.12)이고 3e-5 의 +0.24% 는 사실상 0 이라, 안전 필터가 잡음 위에서 통과시켰을
수 있다. 500 step·단일 seed·D0 기준의 얇은 표본이다.

---

## PHASE 9 — step 3000, D2 + D3

```
                    R      t      5cm5    scale     frs    cornerC
s1 D2  P0        7.772  0.1885  0.1445  0.9848  0.9198   0.8564
       P1        7.471  0.1782  0.1328  0.9714  0.8803   0.8377
s1 D3  P0        7.392  0.1984  0.1074  0.9860  0.8076   0.7628
       P1        7.299  0.2009  0.1191  0.9805  0.7981   0.7781
s2 D2  P0        7.621  0.2165  0.1348  0.9617  0.8789   0.8347
       P1        7.948  0.2089  0.1270  0.9718  0.8540   0.8556
s2 D3  P0        7.732  0.2127  0.1094  0.9567  0.8608   0.8102
       P1        7.787  0.1918  0.1016  0.9706  0.7855*  0.7855
```

### gate (t ≥ +10% AND R 열화 ≤3% AND 5cm5 비감소 AND max(frs, scale) ≥ +10%)

```
s1 D2:  t +5.5%  R +3.9%  5cm5 −1.17pp  frs +4.3%  scale −88.1%   FAIL
s1 D3:  t −1.3%  R +1.3%  5cm5 +1.17pp  frs +1.2%  scale −39.5%   FAIL
s2 D2:  t +3.5%  R −4.3%  5cm5 −0.78pp  frs +2.8%  scale +26.4%   FAIL
s2 D3:  t +9.9%  R −0.7%  5cm5 −0.78pp  frs +4.0%  scale +32.2%   FAIL
```

t 는 4개 중 3개에서 개선되지만 gate(+10%)에 못 미치고, s2 D3 의 +9.9% 가 가장 가깝다.
5cm5deg 는 4개 중 3개에서 하락. scale 은 seed 간 부호가 뒤집힌다(s1 악화, s2 개선).

### paired frame bootstrap (10,000, seed 분리)

```
                          effect      CI95              P(better)
s1 D2  t                  +3.79  [−10.45, +17.42]   0.695
       R                  +4.09  [ −4.98, +12.06]   0.824
       front_rear_shift   +4.09  [ −6.94, +13.75]   0.749
       affine_scale       −2.13  [−19.92, +14.05]   0.398
s1 D3  t                  +1.07  [−15.67, +17.76]   0.543
s2 D2  t                  +3.71  [−11.51, +17.85]   0.679
       R                  −5.42  [−17.74,  +5.71]   0.184
s2 D3  t                  +8.29  [ −8.80, +26.76]   0.835
       front_rear_shift   +4.03  [ −6.91, +13.60]   0.775
```

**28개 비교 전부 CI 가 0 을 포함한다.** P(better) 최대 0.895. 방향은 t 쪽으로 약하게
쏠려 있으나 어느 것도 확립되지 않는다.

---

## 이 결과가 말하는 것 — 세 번째로 같은 구조가 나왔다

`POSE_AWARE_CORNER_GAIN = False` 를 "pose 감독이 아무것도 안 한다" 로 읽으면 안 된다.
보정이 보여준 것은 정반대다: **pose 감독은 scale 통계를 단조로 개선한다.**
문제는 그게 pose 로 전이되지 않는다는 것이다.

같은 구조가 이 세션에서 세 번 독립적으로 나왔다.

```
PHASE 8   Ridge 로 per-frame scale 을 예측해 보정  → scale 잔차는 줄지만 pose 는 악화
PHASE 7B  pose loss 로 scale 통계를 개선           → scale gap 76% 닫아도 t 는 −7.1%
PHASE 9   t 를 안 해치는 유일한 λ                   → 3,000 step 에서 아무 효과 없음
```

scale oracle 의 +31~33% 는 **per-frame 정확성**에서 나온다. **집계 scale 통계를
1.0 쪽으로 옮기는 것으로는 재현되지 않는다** — 중앙값을 옮기면서 per-frame 분산을
같이 키우면 pose 는 얻는 게 없다. 이게 세 실험이 공통으로 가리키는 결론이다.

---

## 판정

```
PHASE 13 CASE D
  THETA_ONLY_LINE_USEFUL            = False  (단, 회전 이득은 20/20 확립)
  POSE_AWARE_CORNER_GAIN            = False  (28/28 미확립)
→ CURRENT_CORNER_REPRESENTATION_POSE_BIAS_REMAINS
```

브리프의 처방을 따른다: **새 loss·새 head 를 더 만들지 않고 problem definition /
data / viewpoint distribution 으로 상류 복귀.**

이 방향은 기존 진단과도 정합한다 — 학습셋 `v2_prod40k_clean_merged` 는 저앙각이 8%
인데 real capture 는 94% 이고, STAGE22/STAGE16 이 이미 "rear 레버는 데이터·appearance
지 해상도·loss 표현이 아니다" 로 수렴했다.

---

## 한계

- synthetic 전용, seed 2개, 3,000 step. 더 긴 예산에서 달라질 수 있으나 브리프가
  long run 자동 실행을 금지한다.
- **DiffPnP3D 의 V8 gating 으로 프레임의 28% (V<8) 는 pose 감독을 전혀 못 받았다.**
  truncation 이 정확히 line 이 가장 유용한 영역이라, 이 한계는 작지 않다.
- λ_pose 는 500 step·단일 seed·D0 에서 골랐고 t 기준이 그 예산에서 비단조였다.
- `L_pose` 는 GT-seeded GN 이라 추론 경로와 다른 선형화를 본다(repo 의 검증된 사용법
  이지만, predicted-seed 변형은 과거 REJECT 된 이력이 있다).

산출: `pose_aware_calibration.json`, `pose_aware_calibration_v1_misscaled.json`,
`pose_aware_loss_audit.json`, `pose_aware_P{0,1}_seed{1,2}.json`,
`pose_aware_report_step3000.json`, `pose_aware_bootstrap.json`,
frames npz 8개. 스크립트 `scripts/stage0/multihead/mh_poseaware.py`,
test `challenge/tests/test_mh_poseaware.py`.
