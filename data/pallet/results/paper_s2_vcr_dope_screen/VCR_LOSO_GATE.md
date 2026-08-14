# VCR Gate 0 — FAIL, architecture 구현 중단

> 교수님이 지적한 heatmap bias 의 **존재**는 기존 진단에서 확인됐다.  이번 실험은
> generic post-hoc correction 이 아니라, 그 bias 가 **viewpoint x corner-role 로 반복
> 가능한지** 를 leave-one-session-out 으로 먼저 검증한 것이다.  학습 0 step,
> ep57 read-only, centroid 는 predicted 유지, canonical PnP 는 centroid 포함,
> N87 은 mechanism development set, final-test 미사용.
> **Gate 0 가 FAIL 이므로 사전 규칙에 따라 VCR-DOPE 를 구현하지 않았다.**

## [관찰] — LOSO 결과

```
arm  median  near   far    F2far  F2signed  >20  >50  >100  PnP  reproj  yaw
B0   13.24   6.88  22.08   44.59   20.58    203  120   54    70   23.16  6.03
B1   22.58  20.17  22.89   39.09    5.51    282  115   45    70   27.28  6.71
B2   42.98  38.59  46.58   48.54   26.42    368  231   88    70   43.80  6.37
B3   51.57  51.56  52.82   68.91   20.03    376  270  155    70   47.88  9.08
```

corner rows 519, sessions 8, LOSO 8 fold.  ridge lambda 1e-3, basis 8항 사전 고정.

### B2 (role x view, scale 제외)

```
  FAIL  1 F2 signed bias -25%             -0.2834
  FAIL  2 F2 far median -15%              -0.0886
  FAIL  3 >50px tail -15%                 -0.9250
  FAIL  4 paired improved > worsened    -161.0000
  FAIL  5 near <= +5%                      4.6044
  FAIL  6 PnP >= 72 or reproj -8%         70.0000
  FAIL  7 no new >100px                   34.0000
  paired improved 179 / worsened 340
  view necessity vs B1: signed bias -10% vs B1=FAIL, F2 far -7.5% vs B1=FAIL, >50px tail -5% vs B1=FAIL, PnP rescue >= 2 vs B1=FAIL
```

### B3 (role x view, 전체 basis)

```
  FAIL  1 F2 signed bias -25%              0.0269
  FAIL  2 F2 far median -15%              -0.5454
  FAIL  3 >50px tail -15%                 -1.2500
  FAIL  4 paired improved > worsened    -223.0000
  FAIL  5 near <= +5%                      6.4886
  FAIL  6 PnP >= 72 or reproj -8%         70.0000
  FAIL  7 no new >100px                  101.0000
  paired improved 148 / worsened 371
  view necessity vs B1: signed bias -10% vs B1=FAIL, F2 far -7.5% vs B1=FAIL, >50px tail -5% vs B1=FAIL, PnP rescue >= 2 vs B1=FAIL
```

**Gate 0 FAIL** — 통과 arm 없음.

## [Bias repeatability] — 존재하지만 지배적이지 않다

B1(role-constant)이 **F2 far signed bias 를 20.58 → 5.51px 로 73% 줄인다** [확인].
즉 corner role 별 **평균 편향은 실재한다** — 교수님 지적의 이 부분은 데이터가 지지한다.

그런데 같은 B1 이 오차를 키운다: near median 6.88 → 20.17px(+193%), 전체 median
13.24 → 22.58px.  이유는 분포에 있다.

```
F2 far  mean signed (dx,dy) = (-19.28, +7.20)   |mean| = 20.58 px
        per-corner 오차 표준편차                  = 39.87 px
near    |mean| = 14.38 px                        median error = 6.88 px
```

[확인] **평균 편향(20.6px)보다 개별 산포(39.9px)가 2배 크다.**
평균을 빼면 평균은 사라지지만 그 크기만큼 다른 corner 에 더해진다.
near corner 는 median 오차가 6.88px 인데 평균 편향이 14.38px 이므로,
보정이 오히려 두 배 이상 밀어낸다.

[확인] in-fold 잔차로도 확인된다:

```
              train residual median   test residual median
B0 (무보정)          13.24 px               13.24 px
B1                   20.54 px               22.18 px
B3                   19.18 px               30.92 px
```

최소제곱은 **평균제곱**을 줄이므로, 두꺼운 꼬리 앞에서는 학습 데이터에서조차
중앙값 오차를 키운다.  이것은 fitting 버그가 아니라 손실함수와 지표의 불일치다.

## ★ [Role-only vs view-conditioned] — view 항은 도움이 아니라 해가 된다

B2/B3 는 B1 보다 **모든 지표에서 나쁘다**(median 43~52px vs 22.6px).
view necessity 4 항목은 B2·B3 둘 다 **전부 FAIL**.

원인은 데이터 구조다.

```
  capturenight05         n= 78  azimuth span    2.5 deg   elev   3.6 ~  7.8
  capturenight06         n= 94  azimuth span  280.1 deg   elev   4.6 ~  7.2
  capturenight07         n= 82  azimuth span    5.6 deg   elev   4.8 ~  9.0
  capturepallet02        n= 40  azimuth span    3.8 deg   elev   5.0 ~  7.0
  capturepallet03        n= 58  azimuth span    4.2 deg   elev   5.1 ~  6.8
  capturepallet04        n=  4  azimuth span    1.5 deg   elev   6.6 ~  7.0
  capturepallet05        n= 23  azimuth span    1.6 deg   elev   5.1 ~  6.9
  capturepallet08        n=140  azimuth span  358.7 deg   elev  -7.7 ~  7.7
```

[확인] **8 session 중 6 개가 azimuth 범위 6° 미만** = 사실상 단일 시점이다.
elevation 은 전 session 이 -7.7 ~ 9.0° 안에 있다(기존 "real 94% 가 <8° 저앙각" 과 정합).
따라서 leave-one-session-out 은 view regression 을 **학습 support 밖으로 외삽**시킨다.
B3 의 test 잔차가 train 대비 19.18 → 30.92px 로 벌어지는 것이 그 결과다.

★ 그래서 H2 의 정확한 판정은 두 갈래다.

- **Gate 판정(사전 규칙)**: FAIL → architecture 구현 금지.  이건 그대로 확정이다.
- **가설 자체**: 이 데이터로는 "viewpoint 로 bias 가 설명된다" 를 **시험할 수 없다**.
  반증됐다기보다 **설계상 검정 불가**에 가깝다.  view 다양성이 없기 때문이다.
  "viewpoint 가설이 틀렸다" 고 단정하면 과한 결론이 된다.

## [Bias atlas]

시각화 전용 bin(azimuth 4 x elevation 2, regression 에 미사용) 결과는
`figures/view_corner_bias_atlas.png` 와 `vcr_bias_atlas_cells.csv` 에 있다.
모든 cell 에서 **1σ 타원이 평균 화살표보다 크다** — 위 분산 논지와 같은 그림이다.

## [현재 판정]

```
View-conditioned bias hypothesis   REJECT (gate 기준) / 단 view 다양성 부재로 검정 불가
Role-constant mean bias            존재 확인 (signed bias -73%) 그러나 오차 감소 실패
Role-discriminative feature        NOT TESTED (Gate 0 에서 중단)
View-conditioned adapter           NOT TESTED (Gate 0 에서 중단)
Final path                         base ep57 DOPE (변경 없음)
```

Gate 1(view observability), VCR module, 32-frame sanity, 5-epoch screen 은
사전 규칙("Gate 0 FAIL 이면 architecture 구현 금지")에 따라 **실행하지 않았다**.

## [지지 증거]

- [확인] baseline 완전 재현 후 시작(87/87/70, yaw 6.025216, reproj 23.161629).
- [확인] view convention 유닛 테스트 12건 통과 — yaw+180 에서 target 동일,
  top-bottom inversion 은 동일 target 아님, elevation 은 -local Y 사용.
- [확인] LOSO session overlap 0, standardization 은 train fold 통계만 사용.
- [확인] B1 이 signed bias 를 73% 줄인다 = 평균 편향은 실재한다.

## [반증 증거 / 한계]

- [확인] view 다양성이 없다.  6/8 session 이 단일 azimuth, 전 session 이 저앙각.
  이 조건에서 view-conditioned 가설은 공정하게 시험되지 않았다.
- [확인] 표본이 얇다: corner 519, fold 당 test corner 약 65개, corner-id 당 train 약 57행에
  8개 feature.
- [확인] 최소제곱(평균제곱)과 평가지표(중앙값)의 불일치가 B1/B3 를 불리하게 만든다.
  robust 회귀였다면 다른 결과였을 수 있으나, basis·lambda·손실을 결과 보고 바꾸지 않았다.

## [다음 admissible experiment]

1. **view 다양성 확보가 선결**이다.  현 real set 으로는 Q1 을 답할 수 없다.
   azimuth 를 넓게 도는 촬영 또는 고앙각 프레임 확보 없이 이 가설을 재시험하지 않는다.
2. 재시험한다면 손실을 **robust(Huber/median)** 로 두고 지표와 정합시킨다.
   이번에는 사전 고정 원칙 때문에 바꾸지 않았다.
3. "평균 편향은 있으나 산포가 2배" 는 그 자체로 정보다 —
   보정 대상은 평균이 아니라 **산포를 만드는 요인**이며, 이는 기존 결론
   (저앙각 flat-view 에서의 depth 붕괴)과 같은 방향을 가리킨다.
