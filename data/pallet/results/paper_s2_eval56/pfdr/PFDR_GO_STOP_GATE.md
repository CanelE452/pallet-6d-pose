# PFDR — 전 arm FAIL, base ep57 유지

> static E2(near H6 / far H5)는 corner 기준 9 조합 중 두 셋 모두 1 위였으나 pose gate 를
> 통과하지 못했다.  PFDR 은 near·centroid 를 ep57 H6 로 **bit-exact 고정**하고 far 만
> H5 anchor + zero-init residual 로 학습한다(N1/N2), N3 는 같은 capacity 를 near 에 적용한
> negative control 이다.  base ep57 **전체 frozen**(trainable 0), canonical decoder 와
> centroid 포함 PnP 변경 없음.  eval56·wood 둘 다 통과해야 채택.  final-test 미사용.

## [관찰]

```
eval56
arm  pnp    reproj   corner     near       far  t50  t100  nan_corner
 E0   50 11.557805 7.241083 4.675503 11.406327   45    17         119
 E2   50 11.743258 6.397502 4.675503  9.642184   45    16         120
 N1   50 11.715050 6.383167 4.675503  9.585456   45    16         120
 N2   50 11.667980 6.402264 4.675503  9.560311   45    16         120
 N3   52 11.800437 7.879596 5.308887 11.406327   45    17         105

wood
arm  pnp   reproj   corner     near       far  t50  t100  nan_corner
 E0   44 9.283903 9.225494 6.732508 14.179799   40    36          51
 E2   44 9.032884 8.775418 6.732508 11.877580   38    34          53
 N1   44 9.016386 8.775418 6.732508 11.806768   38    34          53
 N2   44 8.873333 8.779047 6.732508 11.547159   38    34          53
 N3   44 9.292068 9.357992 7.137582 14.179799   40    36          50
```

## [Exact preservation] — 설계대로 지켜짐

```
N1/N2  near 0.00e+00  centroid 0.00e+00   (두 셋 모두)
N3     far  0.00e+00  centroid 0.00e+00
adapter 136,004 param, base trainable param 0
```

## [Gradient calibration] — anchor λ 가 상한에 걸림

```
arm  |g| group   anchor      l3d        lreproj    lambda
N1   4.114e-03  0.000e+00   -          -          anchor 10(clamp)
N2   3.455e-03  0.000e+00   2.788e-01  1.085e-03  anchor 10(clamp) l3d 0.002478 lreproj 0.3184
N3   3.202e-03  0.000e+00   1.055e-01  1.628e-03  anchor 10(clamp) l3d 0.006072 lreproj 0.1967
```

[확인] zero-init 에서 `Huber(0,0)` 의 gradient 가 정확히 0 이라 anchor 의 gradient-norm
측정이 성립하지 않는다.  프로토콜이 그 지점을 보지 못했고 λ 가 상한 10 으로 clamp 됐다.
**학습 전에 PURPOSE.md 에 리스크로 기록**했고, 규정대로 결과를 보고 조정하지 않았다.

## [Training] 3 arm x 3 epoch, 각 ~18-19 분

```
arm  epoch  group      anchor     l3d        lreproj
N1     1    0.000401   0.000000   -          -
N1     3    0.000406   0.000000   -          -
N2     1    0.000409   0.000000   0.140      0.00112
N2     3    0.000392   0.000000   0.113      0.00087
N3     3    0.000340   0.000000   0.110      0.00081
```

[확인] **N1 은 3 epoch 내내 group loss 가 평평**(0.000401 → 0.000406)하다.
λ_anchor=10 이 residual 을 눌러 belief objective 만으로는 움직이지 못했다.
[확인] **N2 는 움직였다** — l3d 0.397(초기) → 0.113 으로 72% 감소.
belief 를 GT heatmap 에 맞추라는 요구로는 안 움직이던 residual 이,
"9 점이 GT pose 와 맞아야 한다" 는 제약으로는 움직인다.

## [Phase M primary gate] — 6/6 FAIL

```
  eval56|N1    reproj  -1.36%  far +15.96%  FAIL  실패: ['reproj -10%', 'NaN 미증가', 'P>=0.90']
  eval56|N2    reproj  -0.95%  far +16.18%  FAIL  실패: ['reproj -10%', 'NaN 미증가', 'P>=0.90']
  eval56|N3    reproj  -2.10%  far  +0.00%  FAIL  실패: ['reproj -10%', 'near exact/+5%', 'imp>wor', 'P>=0.90']
  wood|N1      reproj  +2.88%  far +16.74%  FAIL  실패: ['reproj -5%', 'NaN 미증가', 'P>=0.80']
  wood|N2      reproj  +4.42%  far +18.57%  FAIL  실패: ['reproj -5%', 'NaN 미증가', 'P>=0.80']
  wood|N3      reproj  -0.09%  far  +0.00%  FAIL  실패: ['reproj -5%', 'near exact/+5%', 'imp>wor']
```

## ★ 핵심 — far 는 확실히 좋아지는데 pose 로 전환되지 않는다

```
        far median              reprojection
        eval56      wood        eval56    wood
E0      11.406     14.180       11.558    9.284
E2       9.642     11.878       11.743    9.033
N1       9.585     11.807       11.715    9.016
N2       9.560     11.547       11.668    8.873
```

[확인] N2 가 far 를 eval56 **-16.2%**, wood **-18.6%** 줄여 static E2 보다도 낫다.
[확인] 그런데 reprojection 은 eval56 **-0.95%**(기준 -10%), wood **+4.42%**(기준 +5%) 로
두 셋 모두 미달이다.  near·centroid 를 고정했는데도 그렇다.
[확인] paired bootstrap 도 P(improve) eval56 0.729 / wood 0.752 로 기준(0.90 / 0.80) 미달.

이는 이 프로그램에서 **일곱 번째로 반복된 같은 벽**이다 —
corner 를 고쳐도 9-point PnP 결과가 그만큼 좋아지지 않는다.

## [Phase N1] geometry objective — REJECT

```
PnP total        N1 94 -> N2 94  (감소 없음)
reproj 추가개선   eval56 +0.40%   wood +1.59%   (기준: 한 셋에서 +5%)
```

[확인] N2 가 N1 보다 far 와 reproj 모두 낫지만 **추가 개선폭이 기준에 못 미친다**.
9-point pose consistency 는 방향은 맞으나 이 조건에서 채택 근거가 되지 못한다.

단 N1 이 λ_anchor 때문에 사실상 학습되지 않았으므로, 이 비교는
"pose objective 가 belief objective 보다 나은가" 보다는
**"눌린 residual 을 움직이는 유일한 신호가 pose objective 였다"** 로 읽는 것이 정확하다.

## [Phase N2] far specificity — 조건부

```
far  eval56  N2 9.560 vs N3 11.406   (+16.2%)
far  wood    N2 11.547 vs N3 14.180  (+18.6%)
PnP total    N2 94    vs N3 96
```

[확인] far 는 두 셋 모두 10% 이상 우수해 far-specific 조건 1 을 만족한다.
[확인] 그러나 **N3 의 PnP total 이 96 으로 N2(94) 보다 높다** — 조건 2 위반.
N3 는 far 를 전혀 건드리지 않고 near 만 바꿨는데 eval56 PnP 가 50 → **52** 로 올랐다.

★ 이는 08-04 의 F4(near 만 stagewise) 가 PnP 53 으로 최고였던 것과 **같은 방향**이다.
두 번 독립적으로 "PnP 를 올리는 것은 far 가 아니라 near" 라는 신호가 나왔다.
far-specific 가설은 corner error 에서는 지지되지만 **pose 에서는 지지되지 않는다**.

## [현재 판정]

```
PFDR adapter                 REJECT (eval56·wood 6/6 FAIL)
9-point GT pose consistency  REJECT (N1 대비 추가개선 5% 미달)
Far-specific hypothesis      corner ACCEPT / pose REJECT (N3 가 PnP 에서 우수)
Final architecture           base ep57 (변경 없음)
```

## [지지 증거]

- [확인] parity 3 종(eval56·wood·E2) 재현 후 시작, exact preservation 0.00e+00.
- [확인] decoder parity median 0.022~0.039px, p90 0.151~0.285px.
- [확인] pose loss 유닛: exact GT → 0.000e+00, far 1개 +25px → l3d 1.8e-3, 음수 depth guard 작동.
- [확인] N2 가 far 를 E2 보다 더 줄였다(9.642→9.560, 11.878→11.547) — adapter 는 실제로 일했다.

## [반증 증거 / 한계]

- [확인] anchor λ=10 clamp 로 N1 은 사실상 학습되지 않았다.  N1 결과를
  "belief objective 로는 안 된다" 의 증거로 쓸 수 없다 — 그 arm 자체가 시험되지 않았다.
- [확인] NaN corner 가 eval56 119 → 120 으로 1 증가(far 를 H5 로 바꾼 데서 온다).
- [확인] N3 가 PnP 에서 앞선다.  far-decoupling 이 pose 개선의 경로라는 전제와 어긋난다.
- [확인] eval56 F2 8 frame / wood 5 frame 으로 F2 지표는 근거로 쓰지 않았다.

## [다음 admissible experiment]

1. **anchor 를 뺀 재실행이 유일하게 남은 공정한 N1 시험**이다.  단 이는 사전 고정
   프로토콜의 변경이므로 새 실험으로 선언하고 gate 를 다시 사전 고정해야 한다.
2. **near 쪽 신호를 정면으로 볼 것** — F4 와 N3 가 두 번 독립적으로 PnP 개선을 보였다.
   far 를 겨냥한 계열은 corner 는 고쳐도 pose 를 못 고쳤다.
3. corner→pose 전환이 일곱 번 연속 실패했다.  다음 제안 전에 이 계열을
   `[소비처]` 기준으로 계속할지 재평가할 것.
