# PPD validation (group-disjoint, 1045 frames)

checkpoint selection 은 **이 표만** 사용했다.  untouched·real 미참조(테스트로 강제).

```
arm  best_epoch  candidate_pair  conditional  inversion  indexed    macroF1  maskIoU  gate
                                 polarity acc            reproj
────────────────────────────────────────────────────────────────────────────────────────
L0         4      754/1045      0.980       15/754   0.89 px   0.317    0.097   PASS
M0         8      754/1045      0.980       15/754   0.89 px   0.350    0.714   PASS
M1        20      754/1045      0.977       17/754   0.89 px   0.417    0.754   PASS
unsigned baseline indexed reproj = 105.58 px
```

- [확인] 세 arm 모두 20 epoch 완주, early stop 없음, 동일 init·동일 sampler permutation.
- [확인] **32-frame 암기가 아니다**: 학습에 쓰지 않은 754 candidate-pair 프레임에서 0.977~0.980.
- [확인] availability 754/1045 = 0.721 은 conditional accuracy 와 **분리해 읽어야 한다**.
- [확인] pixel 지표(macro F1 0.31~0.41)는 여전히 낮다 = H2 historical FAIL 유지.
  "선은 두껍지만 polarity 는 맞다" 는 overfit 단계 패턴이 held-out 에서도 재현된다.
