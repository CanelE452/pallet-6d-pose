# Polarity metric 정정 — 기존 full-pose "PASS" 를 FAIL 로 정정

## 무엇이 잘못됐나

직전 full-pose gate 는 **subset median** 으로 판정했다.  L0 의 `rotation_sym_deg`
중앙값이 1.84° 였기 때문에 통과했지만, 실제로는 **30/86 프레임이 상하 반전**이었다.
절반이 정확하면 중앙값은 작게 나온다.

[확인] `rotation_error_sym_deg` 자체는 결함이 아니다.  검증:

```
                    기존 sym    signed     polarity   indexed reproj
yaw+180 (허용)        0.00°      0.00°      0.0°       0.00 px
width축 180 (금지)    180.00°    180.00°    180.0°     128.22 px
depth축 180 (금지)    180.00°    180.00°    180.0°     128.22 px
```

즉 metric 은 inversion 을 180° 로 표시했고, **gate 가 median 을 써서 그것을 놓쳤다**.

## 정정된 metric

- `vertical_polarity_error_deg` — object up-axis(3D 좌표에서 유도) 사이 각. >=90° = inverted.
- `signed_rotation_error_deg` — 허용 symmetry {identity, yaw+180} 로만 최소화.
- `fixed_indexed_reprojection` — Hungarian 금지, 허용 permutation 2개만 사용.

## 정정된 판정

```
arm                     inversion    signed>90°   indexed reproj   판정
S0 line-only (직전)      30/86        —            155.6 px         FAIL (정정)
```

직전 보고의 "full-pose gate PASS" 는 **FAIL** 로 정정한다.
