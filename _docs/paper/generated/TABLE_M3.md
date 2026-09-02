# Table M3 — Core self-training component ablation

```text
Configuration                                   corner↓    det↑   AUROC↑   FPR95↓   R med↓    yaw↓
────────────────────────────────────────────────────────────────────────────────────────────────
Base                                              6.501   0.975   0.9921   0.0417        —       —
Source-only continuation                          6.763   0.966   0.9872   0.0573        —       —
+ self-training (no filter)                       7.027   0.981   0.9913   0.0558        —       —
+ Confidence filtering                            6.899   0.987   0.9923   0.0469        —       —
+ Keypoint-removal reprojection consistency       6.826   0.987   0.9911   0.0502        —       —
+ Horizontal-flip keypoint consistency            7.057   0.984   0.9953   0.0283        —       —
```

## 단계별 차이

```text
R0               -> R0_CONT             6.501 -> 6.763    Δ   +0.263   추가 최적화 자체 (real pseudo-label 없음)
R0_CONT          -> R1_NAIVE            6.763 -> 7.027    Δ   +0.264   real pseudo-label 로 학습한다는 것 자체
R1_NAIVE         -> R2_CONF             7.027 -> 6.899    Δ   -0.128   confidence filtering
R2_CONF          -> R4_CONF_REMOVE      6.899 -> 6.826    Δ   -0.074   keypoint-removal reprojection consistency
R4_CONF_REMOVE   -> R5_PROPOSED         6.826 -> 7.057    Δ   +0.231   horizontal-flip keypoint consistency
```

`R0 -> R1` 을 곧바로 self-training 효과라고 부르지 않는다.
그 차이에는 추가 최적화 자체의 몫이 섞여 있고, 그 몫이 `R0 -> R0-CONT` 다.

## 각 기하 필터의 단독 기여

위 누적 표만으로는 flip 의 **단독** 기여를 못 뽑는다.  `R4 -> R5` 는
keypoint-removal 이 이미 걸린 상태에서 flip 을 더한 값이기 때문이다.
그래서 각 필터를 단독으로 학습했다.  모두 같은 confidence 전처리 위에서,
같은 exposure·update·init 으로 돌았고 replicate 3 회씩이다.

```text
Configuration                 unique PL  corner↓ mean     std  AUROC↑ mean  FPR95↓ mean
────────────────────────────────────────────────────────────────────────────────────────
neither (Confidence only)           272         6.974   0.089       0.9917       0.0450
+ Reprojection only                 251         6.976   0.072       0.9920       0.0490
+ Keypoint-removal only             267         7.001   0.129       0.9912       0.0538
+ Horizontal-flip only              263         7.074   0.097       0.9936       0.0341
+ both (Proposed)                   259         7.127   0.049       0.9938       0.0352
```

### 단독 기여와 상호작용

```text
keypoint-removal 단독   6.974 -> 7.001   Δ +0.027
horizontal-flip 단독    6.974 -> 7.074   Δ +0.100
둘 다 (Proposed)        6.974 -> 7.127   Δ +0.153

단독 합                 Δ +0.127
실제 조합               Δ +0.153
상호작용                Δ +0.025
```

상호작용 항이 음수면 두 필터가 서로를 보완하고, 양수면 겹치는 일을 한다.
replicate 3 회의 std 를 함께 보고 산포보다 큰 차이만 주장한다.
