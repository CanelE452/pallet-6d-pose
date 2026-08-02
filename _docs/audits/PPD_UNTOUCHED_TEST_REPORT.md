# PPD untouched synthetic test (5,916 frames)

checkpoint 는 validation 에서 고정(L0 ep4 / M0 ep8 / M1 ep20).  untouched 결과로
checkpoint 를 바꾸지 않았다.  unsigned baseline indexed reproj = **98.14 px**.

```
arm  availability      conditional  inversion    indexed    macroF1  maskIoU  gate
                       polarity acc              reproj
────────────────────────────────────────────────────────────────────────────────────
L0   3716/5916 (0.628)   0.951      183/3716    1.16 px    0.311    0.097   PASS
M0   3716/5916 (0.628)   0.950      187/3716    1.16 px    0.344    0.706   FAIL
M1   3716/5916 (0.628)   0.952      180/3716    1.15 px    0.409    0.746   PASS
```

- [확인] **availability 와 conditional accuracy 는 분리해 읽어야 한다.**
  end-to-end success = (3716-inv)/5916 ≈ 0.60 이며, 나머지 2200 프레임은
  candidate pair 자체가 없어 polarity 질문이 성립하지 않는다.
- [확인] validation(754/1045 = 0.721) 대비 availability 가 0.628 로 **떨어졌다**.
  이는 selection 문제가 아니라 **upstream SAI availability** 문제다.
- [확인] M0 는 conditional 0.9497 로 기준 0.95 에 **0.0003 미달**하여 FAIL.
  임계 근처이므로 "M0 가 명확히 나쁘다" 로 읽으면 안 된다.
- [확인] indexed reproj 는 98.14 → 1.15~1.16 px (98.8% 감소).

[판정] untouched gate: **L0 PASS, M1 PASS, M0 FAIL(경계)** → L0/M1 만 real 로 진행.
