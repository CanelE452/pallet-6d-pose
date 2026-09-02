# Appendix — paired uncertainty and tail analysis

## 짝지은 불확실성

R0 와 Proposed 는 같은 프레임을 본다.  그래서 프레임마다 짝지어 차이를 본다 —
프레임 난이도가 공통 요인으로 빠져 분산이 작고 정직하다.

같은 세션의 프레임은 독립이 아니므로 세션을 통째로 재표집한 구간도 함께 낸다.
**넓은 쪽(session)이 정직한 구간이다.**

```text
comparison                         metric                   n         Δ           95% CI frame           95% CI session
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Synthetic-only -> Proposed         pooled_corner_median   308    0.4355 [  0.1117,  0.7694] [   -0.1694,    1.3080]
Synthetic-only -> Proposed         corner                 308    2.3938 [  0.0191,  5.0499] [   -0.4774,    6.3682]
Synthetic-only -> Proposed         detection              319    0.0094 [ -0.0094,  0.0282] [   -0.0141,    0.0488]
Confidence -> Proposed             pooled_corner_median   313    0.1418 [ -0.1183,  0.4039] [   -0.1209,    0.3932]
Confidence -> Proposed             corner                 313   -3.2687 [ -7.7999,  0.2179] [   -7.4175,    0.0548]
Confidence -> Proposed             detection              319   -0.0031 [ -0.0157,  0.0063] [   -0.0140,    0.0071]
Reprojection -> Proposed           pooled_corner_median   313    0.1375 [ -0.1693,  0.4020] [   -0.2367,    0.6357]
Reprojection -> Proposed           corner                 313   -0.7428 [ -4.3344,  2.3229] [   -4.5360,    3.1779]
Reprojection -> Proposed           detection              319   -0.0031 [ -0.0125,  0.0063] [   -0.0136,    0.0074]
```

`pooled_corner_median` 이 primary 다 — evaluator 의 헤드라인 정의(감독
keypoint 를 전 프레임 풀링한 median)와 같은 통계량이고, 프레임을 재표집하며
그 값을 다시 계산한다.  **strict 만 쓴다** — all-annotated 로 대체하면
스케일이 다른 두 모집단을 한 median 에 섞게 된다.

`corner` 는 프레임별 median 차이의 **평균**이라 다른 통계량이고 소수의 파국
프레임에 끌린다 — 참고용으로만 둔다.

`delta = proposed - baseline`.  corner 는 음수가 개선, detection 은 양수가 개선.
구간이 0 을 포함하면 그 비교에서는 개선을 주장하지 않는다.

## 꼬리 분석

Proposed 의 프레임별 corner median 상위 10%(n=31)가
어떤 조건에 몰려 있는지 본다.  평균이 좋아져도 최악 구간이 나빠지면
배포에서는 그게 문제다.

```text
paper_domain   daytime      꼬리   9/31 = 29.03%   전체비중 21.97%   과대
paper_domain   nighttime    꼬리   3/31 = 9.68%   전체비중 15.29%
paper_domain   none         꼬리  19/31 = 61.29%   전체비중 62.74%
occlusion      medium       꼬리  15/31 = 48.39%   전체비중 42.04%
occlusion      none         꼬리  16/31 = 51.61%   전체비중 57.96%
truncation     mild         꼬리   6/31 = 19.35%   전체비중 14.97%
truncation     none         꼬리  25/31 = 80.65%   전체비중 85.03%
object_type    plastic      꼬리  17/31 = 54.84%   전체비중 60.51%
object_type    wood         꼬리  14/31 = 45.16%   전체비중 39.49%
```

프레임 목록: `data/pallet/results/paper_eval_v1/TAIL_HIGH_ERROR_FRAMES.csv`

이 산출물은 **annotation review UI 와 섞지 않는다** — review 는
prediction-blinded 이고 여기에는 모델 오차가 들어 있다.
