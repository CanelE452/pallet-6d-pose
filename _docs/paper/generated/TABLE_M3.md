# Table M3 — Core self-training component ablation

```text
Configuration                                   corner↓    det↑   AUROC↑   FPR95↓   R med↓    yaw↓
────────────────────────────────────────────────────────────────────────────────────────────────
Base                                              4.420   0.975   0.9921   0.0417        —       —
Source-only continuation                          4.352   0.966   0.9872   0.0573        —       —
+ self-training (no filter)                       4.335   0.981   0.9913   0.0558        —       —
+ Confidence filtering                            4.242   0.987   0.9923   0.0469        —       —
+ Keypoint-removal reprojection consistency       4.313   0.987   0.9911   0.0502        —       —
+ Horizontal-flip keypoint consistency            4.180   0.984   0.9953   0.0283        —       —
```

## 단계별 차이

```text
R0               -> R0_CONT             4.420 -> 4.352    Δ   -0.068   추가 최적화 자체 (real pseudo-label 없음)
R0_CONT          -> R1_NAIVE            4.352 -> 4.335    Δ   -0.016   real pseudo-label 로 학습한다는 것 자체
R1_NAIVE         -> R2_CONF             4.335 -> 4.242    Δ   -0.094   confidence filtering
R2_CONF          -> R4_CONF_REMOVE      4.242 -> 4.313    Δ   +0.071   keypoint-removal reprojection consistency
R4_CONF_REMOVE   -> R5_PROPOSED         4.313 -> 4.180    Δ   -0.132   horizontal-flip keypoint consistency
```

`R0 -> R1` 을 곧바로 self-training 효과라고 부르지 않는다.
그 차이에는 추가 최적화 자체의 몫이 섞여 있고, 그 몫이 `R0 -> R0-CONT` 다.
