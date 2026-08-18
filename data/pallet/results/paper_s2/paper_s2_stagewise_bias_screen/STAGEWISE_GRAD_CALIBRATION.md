# Gradient-norm calibration

이전 screen 의 raw-magnitude calibration 이 목적함수를 왜곡한 사례가 있어,
이번에는 **gradient L2 norm** 기준으로 한 번 계산해 5 epoch 고정했다.

기준 parameter: `m6_2.12.weight` (belief stage 6 최종 출력 conv weight)
batch: 첫 8 개 train batch, optimizer update 없음, validation/N87 미사용.

```
term       raw loss median   |grad| median    lambda        weighted |grad|   target
legacy     0.006921          7.5922e-04       —             7.5922e-04        1.00
mass       0.759893          9.7098e-01       1.5638e-04    1.5184e-04        0.20
rank       0.064746          1.9055e-01       5.9764e-04    1.1388e-04        0.15
distance   0.084655          3.0122e-01       2.5205e-04    7.5922e-05        0.10
progress   0.025074          7.8044e-01       4.8640e-05    3.7961e-05        0.05
```

clamp([1e-6, 10]) 발생: 없음

[확인] 신규 loss 들의 gradient norm 이 legacy 보다 250~1300배 크다.
raw magnitude 로 맞췄다면 실제 gradient 기여가 의도와 크게 어긋났을 것이다.
[확인] 이번 실패는 가중치가 너무 작아서 생긴 것이 아니다 —
weighted gradient 가 설계대로 legacy 의 20/15/10/5% 로 들어갔고,
그 결과 synthetic loss 는 실제로 내려갔다(mass 1.27 → 0.56).
