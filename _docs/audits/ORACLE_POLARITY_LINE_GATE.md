# Oracle polarity-line gate — PASS

3 class(width/depth/vertical)를 **5 class**(top_width, top_depth, base_width,
base_depth, vertical)로 확장했다.  top/base 판정은 keypoint id 하드코딩이 아니라
**object-frame vertical 좌표**에서 유도한다 (검증: top_width 2 / base_width 2 /
top_depth 2 / base_depth 2 / vertical 4).

evidence 는 GT pose 로 projection·라벨링한 **oracle upper bound** 이며, scorer 에는
GT pose 가 전달되지 않는다.

## 결과 (n=86)

```
지표                    값          기준          판정
polarity accuracy      0.965       >=0.95        PASS
inversion              3/86        <=4           PASS
signed_rot>90°         0.035       <=0.05        PASS
point-fail correct     17/17       >=15          PASS
truncated correct      17/17       —
indexed reproj median  7.39px      <=29          PASS
  (155.6px 대비 95% 감소, baseline 23.16px 의 32%)
yaw median             1.75°       악화 <=1°      PASS (line-only 1.00°)
corner_sym median      0.067 m
negative depth         0
```

## 세 scorer 비교

```
scorer              inversion    indexed reproj   point-fail correct
S0 line-only        30/86        155.6 px         —
H0 frozen heatmap   26/86         16.5 px         8/17
O0 oracle line       3/86          7.4 px        17/17
```

## 해석

- [확인] **top/base semantic line 표현은 vertical polarity 를 원리적으로 해소한다.**
  3 class 로는 부호 불변이던 에너지가 5 class 에서는 top 과 base 를 구분한다.
- [확인] 올바른 upright 후보는 **86/86 프레임에서 항상 후보 집합 안에 있었다**
  (`gt_upright_available`).  즉 이것은 순수한 **selection** 문제였고 generation 문제가 아니다.
- [판정] **Oracle polarity-line representation: ACCEPT (upper bound)**.
  learned PPD capability test 로 진행할 근거가 성립한다.

## 한계

- [확인] oracle 은 semantic top/base 라벨이 GT 에서 온다.  learned PPD 는 이를 스스로
  예측해야 하므로 이 수치는 달성 가능한 **상한**이다.
- [확인] n=86 소표본이고 inversion 3건의 원인은 분해하지 않았다.
