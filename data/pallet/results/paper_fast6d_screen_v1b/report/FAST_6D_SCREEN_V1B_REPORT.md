# FAST_6D_SCREEN_V1B — bbox semantics 와 line provenance 교정

status **POST_STOP_EXPLORATORY_CORRECTION** ·
population PAPER_EVAL positive **319** (이미 개발에 반복 사용된 셋) ·
new training 0 · new checkpoint 0 · depth 0 · parameter sweep 0

> **DEVELOPMENT RESULT, NOT INDEPENDENTLY CONFIRMED.**
> held-out · independent · final · confirmed 라고 부르지 않는다.

V1 은 그대로 둔다.  S1/S3/S4 결론은 유효하다.  여기서 교정한 것은 두 개뿐이다.

## 1. 무엇이 틀렸었나

### S2 — 관측 불가능한 상자를 맞추고 있었다

`run_translation_arms.py:71-73` 은 투영된 **8 코너 전부**의 min/max 를 상자로 썼다.
YOLO 가 학습한 상자는 그것이 아니다.  `build_real_ft_dataset.to_yolo_label` 은
화면 밖 코너를 `v=0` 으로 버리고 **화면 안 코너만**의 min/max 를 쓰며,
`REAL_FT_V1_METHOD_LOCK.json` 이 이 규칙이 합성 규약과 400/400 (2e-3 이내) 일치함을
기록한다.  R0 는 그 합성 규약으로 학습됐다.

### S5 — 있는 경로를 없다고 했다

V1 은 "real line prediction cache 도 canonical adapter 도 없다" 를 이유로 S5 를
`BLOCKED_INCOMPATIBLE_PROVENANCE` 로 닫았다.  cache 가 없다는 것은 맞지만 adapter 가
없다는 것은 **틀렸다** — `ft_f0f3_eval.py` 가 real 이미지에서
`preprocess_squash → SplitLate → DH.decode → mh_fusion` 을 이미 수행한다.
cache 가 없다는 것은 만들면 되는 것이지 arm 을 닫을 사유가 아니었다.

## 2. 게이트 (측정 전 통과)

```
GEOMETRY_CONVENTION_GATE
  P1  cuboid_model_points 와 make_pallet_keypoints_3d_diagram   좌표까지 동일 (max diff 0.0)
  P2  GT pose 투영 vs 수동 어노, index-wise 중앙값
        plastic   0.94 px   (90도 재라벨 대조군 162.87 px)
        wood      0.45 px   (90도 재라벨 대조군 165.10 px)
  P3  0~3 이 near face 인 프레임 비율   plastic 1.000 · wood 1.000
  -> line_population = ALL.  WOOD_LINE_STATUS = OK (BLOCKED 아님)

LINE_FUSION_IMPLEMENTATION_GATE = PASS
  mh_fusion.run_tests 를 손대지 않고 실행(OUT 만 V1B 로 우회).  T1~T7 이 과거
  기록과 **비트 동일**.  read-only artifact 는 수정되지 않았다.
```

## 3. 결과

```
arm                                   n      R     Yaw    t cm     IoU3D    ADDsym   reproj
───────────────────────────────────────────────────────────────────────────────────────────
C0  frozen YOLO R0                  319   2.262   1.231   7.897   0.6032   0.4285    3.55
C1  관측 semantics bbox + t          319   2.262   1.231   9.816   0.5372   0.3793    4.25

L0  frozen YOLO R0                  319   2.262   1.231   7.897   0.6032   0.4285    3.55
seed1  lambda 3.0
L2    line rotation, t 고정          319   4.680   2.371   7.897   0.5434   0.3714    7.04
L3 ★  line rotation + t refit       319   4.680   2.371   8.431   0.5132   0.3623    6.47
L4    yaw-only + t refit            319   3.048   2.035   7.891   0.6034   0.4194    4.51
seed2  lambda 1.0
L2    line rotation, t 고정          319   2.718   1.358   7.897   0.5890   0.4234    3.96
L3 ★  line rotation + t refit       319   2.718   1.358   8.106   0.5879   0.4234    3.91
L4    yaw-only + t refit            319   2.401   1.344   8.200   0.5965   0.4252    3.71
```

C0 는 frozen `POSE_EVALUATION_R0` 와 **절대오차 0.00e+00** 으로 일치했다(6 지표 전부).
예외 0 건, C1_UNRESOLVED_OBSERVABLE_BOX 0 건, line 예외 0 건.
arm 이 실제로 움직였다는 증거: C1 은 translation 중앙값을 7.90 → 9.82 cm 로 옮겼고,
line rotation 은 중앙값 3.084°(seed1) / 0.836°(seed2) 만큼 회전을 돌렸다.
**no-op 이 아니라 실재하는 음성 결과다.**

## 4. paired bootstrap (10,000 재표본, seed 20260904, 13 세션)

```
contrast        metric      delta      frame CI95            session CI95
─────────────────────────────────────────────────────────────────────────────────
C1-C0           IoU3D      -0.0660   [-0.0941, -0.0274]   [-0.0903, -0.0233]  ★0 배제
C1-C0           ADDsym     -0.0492   [-0.0672, -0.0326]   [-0.0725, -0.0191]  ★0 배제
L3-L0 seed1     IoU3D      -0.0900   [-0.1176, -0.0515]   [-0.1346, -0.0460]  ★0 배제
L3-L0 seed1     ADDsym     -0.0661   [-0.0838, -0.0494]   [-0.1112, -0.0388]  ★0 배제
L3-L0 seed2     IoU3D      -0.0152   [-0.0253, +0.0085]   [-0.0258, +0.0097]
L3-L0 seed2     ADDsym     -0.0051   [-0.0093, -0.0011]   [-0.0139, -0.0001]  ★0 배제
L2-L0 seed1     IoU3D      -0.0598   [-0.0745, -0.0243]   [-0.0922, -0.0119]  ★0 배제
L2-L0 seed2     IoU3D      -0.0142   [-0.0218, +0.0040]   [-0.0179, +0.0072]
L4-L0 seed1     IoU3D      +0.0002   [-0.0098, +0.0252]   [-0.0098, +0.0253]
L4-L0 seed2     IoU3D      -0.0067   [-0.0149, +0.0112]   [-0.0116, +0.0132]
```

## 5. 사전등록 게이트 판정

```
C1   primary  ΔIoU3D -0.0660 · ΔADDsym -0.0492      둘 다 +0.020 미달   -> STOP
     translation 7.90 -> 9.82 cm (+24%)             5% 한도 초과
L3   조건 A  두 seed 모두 Δ >= 0                     불충족 (넷 다 음수) -> STOP
     조건 B  median Δ IoU -0.0526 / ADDsym -0.0356   불충족
     조건 C  coverage drop 0.000                     충족
     조건 D  rotation 비 2.07 (seed1) / 1.20 (seed2) 불충족

CORRECTED_FAST6D = NO_PROMOTABLE_SIGNAL
PROMOTED_METHOD_CANDIDATE = 없음
runtime benchmark = 미실행 (lock 은 L3 통과 또는 두 seed 양의 방향일 때만 돌린다)
```

## 6. 재료 · 조명 (선택에 쓰지 않는 기술 통계)

```
                          n      R    t cm     IoU3D    ADDsym
──────────────────────────────────────────────────────────────
plastic  C0             194   2.13   10.47    0.5857   0.3448
plastic  C1             194   2.13   13.34    0.5071   0.2837
plastic  L3 seed1       194   4.23   10.43    0.4953   0.2932
plastic  L3 seed2       194   2.52   10.58    0.5803   0.3430
wood     C0             125   2.98    4.20    0.6256   0.4230
wood     C1             125   2.98    5.21    0.6133   0.4016
wood     L3 seed1       125   5.63    5.15    0.5399   0.3223
wood     L3 seed2       125   3.28    4.01    0.5995   0.4094
```

조명은 manifest 의 기존 `paper_domain` 필드를 그대로 쓴다 — 319 중 **120 장만**
라벨돼 있다(daytime 70 · nighttime 50).  나머지 199 장은 세션 이름으로 추측하지 않고
`none` 으로 남겼다.

```
                          n      R    t cm     IoU3D    ADDsym
──────────────────────────────────────────────────────────────
daytime    C0            70   2.48   11.03    0.5636   0.2903
daytime    C1            70   2.48   12.04    0.5031   0.2540
daytime    L3 seed1      70   3.85   11.05    0.4906   0.2667
daytime    L3 seed2      70   2.81   11.08    0.5455   0.2821
nighttime  C0            50   3.03   12.59    0.5324   0.2884
nighttime  C1            50   3.03   15.30    0.4306   0.2280
nighttime  L3 seed1      50   4.81   10.45    0.4591   0.2706
nighttime  L3 seed2      50   3.25   12.16    0.5208   0.2911
```

어느 부분모집단에서도 양의 신호가 없다.  wood 는 seed1 에서 가장 크게 무너진다
(IoU 0.626 → 0.540).  line 모델의 학습 풀은 합성 BROAD 계열(FINAL_SYNTH_TRAIN_V1,
40,000)이며 **wood 종횡비 노출량은 이번 작업에서 확인하지 않았다** — wood 결과는 그
미확인을 안고 읽어야 한다.

## 7. 해석

**bbox 를 올바른 semantics 로 고치면 오히려 더 나빠진다.**  V1 의 S2 는 ΔIoU
-0.0216 이었는데 C1 은 -0.0660 이다.  잘못된 상자를 맞추던 때보다 **맞는 상자를
맞출 때 더 나쁘다** — 상자와 코너가 서로 다른 물체 위치를 말하고 있다는 뜻이고,
bbox 를 translation 단서로 쓰는 방향 자체가 닫힌다.  predicted reprojection 도
3.55 → 4.25 px 로 같이 나빠졌으므로, 이것은 "목적함수는 좋아지고 진실은 나빠지는"
과적합 함정이 아니라 두 단서의 **실제 불일치**다.

**line 은 회전을 옮기지만 옮기는 방향이 틀렸다.**  rotation 이 실제로 움직였고
(3.08° / 0.84°), 움직인 결과 회전 오차가 2.26° → 4.68° / 2.72° 로 커졌다.
lambda 가 큰 seed1 이 더 크게 나빠진다 — line 을 더 믿을수록 더 나빠진다는 뜻이라
"세기를 조절하면 될 것" 이 아니다(그리고 lambda 조정은 lock 이 금지한다).
유일하게 중립인 것은 L4(yaw-only, seed1 ΔIoU +0.0002)인데, 이는 자유도를 하나로
줄여 **line 이 pose 를 거의 못 건드리게 했을 때만 손해가 사라진다**는 뜻이다.

이 두 결과는 기존 기록과 모순되지 않는다.  `theta-only line = 회전 이득 확증` 은
**합성 모집단에서 DOPE 코너를 point cue 로 썼을 때**의 이야기다.  여기서는 point cue
가 이미 훨씬 정확한 YOLO R0 이고, 그 위에 얹은 line 은 정보가 아니라 잡음으로
들어온다.  같은 관찰의 반복이다 — 강한 base 위에서는 보조 신호가 이득을 못 낸다.

## 8. 이번 결과가 **말하지 않는** 것

- line branch 가 쓸모없다는 뜻이 아니다.  base 가 약한 곳에서의 과거 이득은 그대로다.
- YOLO bbox 가 잘못됐다는 뜻이 아니다.  검출용으로는 정상이고, **translation 단서**로
  쓰기에 코너와 정합하지 않는다는 뜻이다.
- held-out 증거가 아니다.  319 는 개발에 반복 사용된 셋이다.
