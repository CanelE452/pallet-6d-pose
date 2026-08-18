# PREDSEED DIFFPNP GATE — REJECT

> 모델 재학습 **0 step**, optimizer 생성 0.  ep57 heatmap 과 decoded coordinate 는 변경하지 않았다.
> canonical PnP 가 이미 성공한 **동일 70 frame** 에서만 비교했고(membership
> `10a8b40b508698df`), seed 는 canonical OpenCV PnP 가 실제 반환한 predicted pose 다.
> GN residual 은 predicted 2D correspondence(8 corner + centroid = 9)만 사용하고,
> GT 는 refinement 가 끝난 뒤 metric 계산에만 열었다.
> F1/PnP-fail 17 frame 은 대상이 아니며 "DiffPnP 가 F1 을 복구했다" 는 주장은 하지 않는다.
> final-test 미사용.

```
FAIL  1 GT reproj -5%                    -0.04502
PASS  2 3D corner -5%                     0.19200
FAIL  3 improved > 2x worsened          -14.00000
FAIL  4 yaw <= +0.25deg                   0.75754
FAIL  5 rotation <= +5%                   0.13798
PASS  6 translation <= +5%               -0.38123
PASS  7 F5-safe no 10% worse              0.00000
PASS  8 no new negative depth             0.00000
PASS  9 no NaN/Inf                        0.00000
PASS  10 fallback < 5%                    0.00000
PASS  11 no >20px catastrophic            0.00000
PASS  12 >=5 frames with -10% reproj      7.00000

-> REJECT
```

## ★ 핵심 — predicted 2D 에 더 잘 맞추자 pose 는 더 틀려졌다

```
observed reprojection (predicted 2D 적합도)   11.9591 -> 6.3452 px   -47.0%
fixed indexed GT reprojection                23.1616 -> 24.2043 px   +4.5%
```

[확인] GN 은 자기 목적함수를 **절반으로** 줄였다.  그런데 그 목적함수는
**예측된 2D 점에 대한 적합도**일 뿐이고, 실제 pose 정확도는 오히려 나빠졌다.
지시문이 미리 못박은 "observed reprojection 감소만으로 PASS 금지" 조항에 정확히 해당한다.

## 지표별로 갈린다

```
metric                       D0        D1        변화
fixed GT reproj (px)       23.1616   24.2043   +4.5%   악화  (기준 -5%)
3D corner (m)               0.4516    0.3649  -19.2%   개선  (기준 -5%)  PASS
yaw (deg)                   6.0252    6.7828   +0.76   악화  (기준 +0.25)
signed rotation             (기준 +5%)                 +13.8%  악화
translation (m)             (기준 +5%)                 -38.1%  개선  PASS
paired                     improved 42 / worsened 28 / unchanged 0
                           기준 improved > 2 x worsened (>56)  FAIL
```

[확인] **translation 은 38% 좋아지고 rotation 은 14% 나빠진다.**
3D corner 오차는 translation 이 지배하므로 19% 개선되지만,
reprojection 은 rotation 에 민감하므로 악화한다.
즉 GN 이 pose 를 "평행이동으로 끌어당기고 회전을 망가뜨린" 셈이다.

## Solver health

```
GN steps 시도   70 frame x 4 = 280
accepted        95        rejected 185 (66%)
fallback        0         negative depth 0       NaN/Inf 0
rotation update norm median      0.00695 rad
translation update norm median   0.03461 m
```

[확인] fallback 0, guard 위반 0 = 구현은 건전하다.  결과는 solver 버그가 아니다.
[확인] 66% 의 step 이 residual 증가로 reject 됐지만, 나머지 95 step 이 observed
residual 을 절반으로 낮췄다.  **canonical PnP 가 반환한 pose 는 이 9 점의
reprojection 최소점이 아니다.**

[추정] 이유는 canonical `solve_pose` 가 여러 init x 24 flip 후보를 각각 LM refine
(`cv2.solvePnP(useExtrinsicGuess=True, SOLVEPNP_ITERATIVE)`)한 뒤,
순수 reprojection 이 아니라 degeneracy guard·click violation 등을 포함한 **복합 점수**로
고르기 때문으로 보인다.  그 추가 기준이 **보호 장치**였고, 순수 2D 적합으로 밀면 깨진다.

## Slices (동일 70 frame 내부)

```
  failure class  F1_NO_RESPONSE         n=  7  median delta  -1.703 px
  failure class  F2_CONFIDENT_WRONG     n= 35  median delta  +0.000 px
  failure class  F3_GEOMETRY_AMPLIFIED  n=  4  median delta  +0.000 px
  failure class  F4_SOLVER_SPECIFIC     n=  1  median delta  -0.000 px
  failure class  F5_MIXED               n= 23  median delta  -0.000 px
  domain         night                  n= 34  median delta  -0.000 px
  domain         outside                n= 36  median delta  -0.000 px
```

## [현재 판정]

```
Predicted-seed GN refiner    REJECT
최종 구조                     A. base ep57 DOPE -> canonical decoder
                                -> centroid 포함 canonical OpenCV PnP
최근 실험에서 채택된 변경 수    0
```

추가 학습이나 loss 실험으로 자동 진행하지 않는다.

## [지지 증거]

- [확인] baseline 완전 재현 후 시작(87/87/70, yaw 6.025216, reproj 23.161629).
- [확인] 70-frame membership 을 고정하고 D0·D1 을 그 위에서만 비교.
- [확인] Phase D 검증 통과: exact pose+exact 점에서 update ~0, perturbation 에서 수렴,
  observed 좌표까지 gradient finite, 함수 signature·source 에 GT 없음.
- [확인] fallback 0, negative depth 0, NaN 0 — 실패가 구현 결함이 아니다.

## [반증 증거 / 한계]

- [확인] 3D corner 와 translation 은 실제로 개선됐다.  "전부 나쁘다" 가 아니다.
  다만 primary 조건(GT reprojection)과 rotation 이 반대 방향이다.
- [확인] 표본 70 frame.  mechanism screen 이며 일반화 수치가 아니다.
- [확인] GN 상수(4 step / damping 1e-3 / clip 0.5)는 기존 검증값 고정이고 sweep 하지 않았다.
  다른 상수에서 달라질 가능성은 이 실험이 배제하지 않는다.

## [다음 admissible experiment]

1. 이 결과는 **예측 2D 좌표에 계통 편향이 있다**는 것을 pose 쪽에서 다시 확인해 준다.
   더 잘 맞출수록 나빠지므로, 남은 레버는 solver 가 아니라 **2D 예측의 편향 자체**다.
2. canonical solver 의 후보 선택 기준이 보호 역할을 한다는 [추정]을 확인하려면
   선택 점수를 로깅해 순수 reprojection 최소 후보와 비교하면 된다(재학습 불필요).
3. 위 없이 GN 상수를 바꿔 재시도하지 않는다.
