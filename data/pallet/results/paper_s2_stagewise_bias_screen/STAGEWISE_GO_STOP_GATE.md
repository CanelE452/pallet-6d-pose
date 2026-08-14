# STAGEWISE GO/STOP — STOP (12개 중 9 FAIL)

> architecture 는 기존 DOPE 그대로.  신규 prediction branch 없음.  belief stage 4~6 만 학습.
> legacy Gaussian MSE + affinity loss 유지, 신규는 full-map GT mass / wrong-peak rank /
> distance / progress 네 항.  canonical 6 roots 29,308 samples, 정확히 5 epoch,
> checkpoint selection 없음.  canonical PnP 는 **centroid 포함**.
> N87 은 epoch0/epoch5 mechanism screen, final-test 미사용.

```
FAIL  1 F2 far median -15%                 0.0094
FAIL  2 F2 signed bias -20%                0.0173
FAIL  3 >50px tail -15%                    0.1000
FAIL  4 sharpen-no-correct -30%           -0.8224
FAIL  5 F2 improved > worsened            -7.0000
FAIL  6 canonical PnP >= 72               67.0000
FAIL  7 reproj -10%                        0.0512
PASS  8 near <= +5%                       -0.0103
FAIL  9 no new PnP failure                 3.0000
PASS  10 no new >100px                    -9.0000
FAIL  11 no new NaN                        5.0000
PASS  12 no stage collapse                 0.0000

GO/STOP: STOP  ->  predicted-seed DiffPnP NOT RUN
```

## [관찰]

```
                     C0        C1        변화
median (px)         13.24     12.36     -6.7%   개선
near median          6.88      6.81     -1.0%   개선
far median          22.08     22.17     +0.4%
F2 far median       44.59     44.17     -0.9%   (기준 -15%)
F2 signed bias      20.58     20.23     -1.7%   (기준 -20%)
p90                101.16     93.04     -8.0%   개선
>20px tail            203       193     -4.9%
>50px tail            120       108    -10.0%   (기준 -15%)
>100px tail            54        45    -16.7%   개선
reproj median       23.162    21.975    -5.1%   (기준 -10%)
canonical PnP       70/87     67/87     -3      악화
NaN corner            177       182       +5    악화
sharpen-no-correct    107       195     +82.2%  ★역방향
```

## ★ 핵심 — 겨냥한 현상이 오히려 심해졌다

`sharpen-without-correction` (H6 peak > H4 peak + 0.10 이면서 H6 error 가
H4 error 대비 2px 이상 줄지 않음) 을 **30% 줄이는 것이 조건 4** 였는데,
107 → **195 로 82% 늘었다** [확인].

stage 궤적이 이유를 보여준다.

```
arm stage  far_err  far_peak  far_mass  far_wrong
C0    4     21.42    0.759     0.201     0.695
C0    5     22.35    0.838     0.199     0.749
C0    6     22.08    0.855     0.215     0.772
C1    4     21.99    0.743     0.162     0.682
C1    5     21.18    0.843     0.195     0.760
C1    6     22.17    0.889     0.185     0.775
```

[확인] C1 의 stage6 far peak 가 0.855 → **0.889 로 더 올라갔다**.
[확인] 겨냥했던 wrong peak 는 0.772 → 0.775 로 **줄지 않았다**.
[확인] GT mass 는 stage6 에서 0.215 → **0.185 로 오히려 감소**했다.
[확인] stage5 만 21.18px 로 좋아졌다가 stage6 에서 22.17px 로 되돌아간다 —
progress 제약이 real 에서 지켜지지 않는다.

## ★ synthetic 에서는 의도대로 작동했다

학습 로그의 raw mass loss 는 1.27 → 0.56 으로 내려갔다.
`-log(M) = 0.56` 이면 synthetic GT mass ≈ **0.57** 이다.
같은 모델의 real far GT mass 는 **0.185** 다.

[확인] loss 자체는 학습됐고 synthetic 에서 GT mass 를 3배 이상으로 끌어올렸다.
**real 로 전이되지 않았을 뿐이다.**  이는 PPD long-run·corner replacement 와 같은 형태다.

## 부분적으로는 좋아졌다 — 그런데 pose 로 이어지지 않는다

[확인] corner 통계는 전반적으로 개선됐다: median -6.7%, p90 -8.0%,
>20/>50/>100 tail 전부 감소.
[확인] 그런데 **canonical PnP 는 70 → 67 로 줄고 새 실패 3 건, NaN corner +5** 다.
corner 오차가 줄어도 pose 로 전환되지 않았다.  reproj -5.1% 는 기준 -10% 미달.

## [F1 no-response]

```
arm  frames  decoded  no_response  gt_mass_median  pose_success
arm  frames  decoded_corners  no_response_corners  gt_mass_median  pose_success
 C0      24               35                  157        0.003615             7
 C1      24               29                  163        0.003918             4
```

## [현재 판정]

```
Stagewise GT-mass loss        REJECT   (real 에서 mass 오히려 감소)
Wrong-peak suppression        REJECT   (wrong peak 0.772 -> 0.775, sharpening +82%)
Stage-progress constraint     REJECT   (stage5 개선이 stage6 에서 되돌아감)
Predicted-seed DiffPnP        NOT RUN  (gate FAIL)
Final path                    base ep57 DOPE
```

epoch 추가·loss/temperature/margin 재조정 없음.
