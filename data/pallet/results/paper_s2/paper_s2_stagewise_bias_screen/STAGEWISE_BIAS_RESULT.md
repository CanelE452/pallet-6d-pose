# Signed bias / tail / stage trajectory

```
                 C0        C1
near median      6.88      6.81
far median      22.08     22.17
F2 far median   44.59     44.17
F2 signed bias  20.58     20.23
p90            101.16     93.04
>20px            203       193
>50px            120       108
>100px            54        45
NaN corner       177       182
```

## stage 별 (far face)

```
arm stage  err     peak    GT mass  wrong peak
arm  stage  far_err_median  near_err_median  far_peak_median  far_mass_median  far_wrong_median
 C0      4       21.418953         7.570224         0.759426         0.201157          0.694511
 C0      5       22.353799         6.869148         0.837545         0.198647          0.749352
 C0      6       22.076680         6.884772         0.855087         0.215251          0.771777
 C1      4       21.991655         7.072596         0.742768         0.162333          0.681695
 C1      5       21.182839         6.779322         0.843461         0.195452          0.759521
 C1      6       22.171379         6.813829         0.889068         0.185366          0.775368
```

[확인] 목표는 stage4→6 에서 error 가 줄고 peak 상승이 억제되는 것이었다.
실제로는 C1 stage6 peak 가 0.889 로 **C0(0.855)보다 더 높다**.
[확인] wrong peak 는 0.772 → 0.775 로 사실상 불변 — suppression 이 작동하지 않았다.
[확인] stage5 에서 far error 가 21.18px 로 개선됐다가 stage6 에서 22.17px 로 회귀한다.
progress loss 가 synthetic 에서는 걸렸지만 real 에서는 stage6 회귀를 막지 못했다.
