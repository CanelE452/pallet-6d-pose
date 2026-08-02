# Mask 결정 — DROP

primary 는 **real candidate polarity** 다.  pixel F1·mask IoU 가 높다는 이유로
mask arm 을 선택하지 않는다.

```
arm  overfit polAcc  val polAcc  untouched polAcc  real polAcc  macroF1  maskIoU
L0      0.969          0.980         0.951          0.023       0.311    0.097
M0      1.000          0.980         0.950(FAIL)      —         0.344    0.706
M1      1.000          0.977         0.952          0.012       0.409    0.746
```

- [확인] mask arm 은 **pixel/mask 지표만** 올린다 (macroF1 0.311→0.409, IoU 0.097→0.746).
- [확인] **목적 지표는 개선하지 않는다**: validation 은 L0 0.980 ≥ M1 0.977,
  untouched 는 0.951 vs 0.952 로 사실상 동률, real 은 L0 0.023 > M1 0.012 로
  오히려 L0 가 낫다.
- [확인] M0 는 untouched 에서 경계 FAIL.

[판정] **Mask DROP** — 가장 단순한 L0 를 남긴다.
단 L0 자체가 real gate FAIL 이므로 "L0 채택" 이 아니라 "mask 는 불필요" 까지만 말한다.
