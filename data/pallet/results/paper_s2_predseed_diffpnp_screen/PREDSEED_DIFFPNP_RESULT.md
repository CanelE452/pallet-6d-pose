# Result

```
                              D0          D1         변화
observed reprojection      11.9591      6.3452     -47.0%   (predicted 2D 적합도)
fixed indexed GT reproj    23.1616     24.2043      +4.5%   ★ 악화
3D corner (m)               0.4516      0.3649     -19.2%
yaw (deg)                   6.0252      6.7828      +0.76
translation (m)                                    -38.1%
signed rotation                                    +13.8%
improved / worsened        42 / 28  (unchanged 0)
accepted / rejected GN step   95 / 185
fallback                       0
```

[확인] 목적함수(observed)는 절반이 됐고 진짜 pose 정확도(GT reproj, yaw, rotation)는 나빠졌다.
[확인] translation 과 3D corner 는 개선됐다 — 결과가 균일하게 나쁜 것은 아니다.
[확인] fallback·negative depth·NaN 전부 0 이므로 구현 결함이 아니다.
