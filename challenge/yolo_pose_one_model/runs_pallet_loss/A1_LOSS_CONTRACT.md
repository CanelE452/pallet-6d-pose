# A1 — Symmetry-Aware Keypoint Objective (계약)

작성 2026-08-22. **결과가 나오기 전에** 확정한다. 학습 후 이 문서를 고치지 않는다.

## 0. 변수는 LOSS 하나뿐

A0 = `V1_FIXED_MATCHED10K` Standard seed42. A1 = **동일** data / init / seed / 60ep
+ symmetry-aware keypoint objective. `A2 projective term 은 포함하지 않는다` —
코드에서 `pspc.enabled=False`, `lambda_pc=0.0` 로 강제하고 T5 가 실증한다
(`projective_loss` 호출 0회).

```
init      challenge/weights/pretrained_yolo/yolo26n-pose.pt   (A0 와 동일)
data      datasets/v1_fixed_matched10k                        (train 9,867 / val 133)
seed      42     epochs 60    batch 32   imgsz 640   SGD lr0 0.01 lrf 0.01
          cos_lr True   close_mosaic 10   pose 12.0   kobj 1.0   patience 0
```

## 1. 기호

`d_id(i)`  = instance i 의 per-instance keypoint loss (KeypointLoss 와 **같은 수식**,
마지막 `.mean()` 만 instance 축으로 남김 — T3 이 12자리 일치 확인).
`d_180(i)` = GT 를 `P180` 로 재배열한 뒤 같은 수식.

```
P180 = (5,4,7,6,1,0,3,2)          centroid 8 고정
```
fixed corner 기준 실제 Rz(π) 에서 유도했고 involution·bijection 이며 4 asset·800
프레임에서 실패 0 (`FIXED_P180_PERMUTATIONS.json`). **camera-facing perm_v4 가 아니다.**

validity mask 도 함께 재배열한다 — 점만 옮기고 mask 를 두면 화면 밖 점이 유효로
둔갑한다 (T11).

## 2. 분기는 contract 가 정한다 — 코드에 굳히지 않았다

`SYMMETRY_MANIFEST` 의 asset→class 가 분기를 고른다. UNRESOLVED 가 하나라도 있으면
build 단계에서 막는다.

```
CASE 1  4 asset 전부 SYM180        L = mean_i  min(d_id, d_180)
CASE 2  혼재                        L = mean_i [ SYM: min(d_id,d_180) | ASYM: d_id ]
                                      + λ_role · mean_{ASYM} relu(m + d_id − d_180)
CASE 3  UNRESOLVED 잔존             BLOCK — 임의 분류 금지
```

class 배정은 image→asset 이다. `v1_fixed_matched10k` 는 **10,000/10,000 이 정확히
객체 1개**임을 전수 확인했으므로 image-level 메타 대입이 유효하다 (T12).

role 항은 epoch 0–5 = 0, 5–20 선형 상승, 이후 1.0 (`role_ramp`). 초기에 아직
아무것도 못 맞추는 상태에서 role 을 걸면 잡음을 학습한다.

## 3. 미리 잰 것 — 표적의 크기

`A1_PRECALIB.json` (A0 수렴 체크포인트, train 2,048 프레임 / 20,438 anchor).

```
asset                                     n   d_id p50  d180 p50  sep p50   flip%
──────────────────────────────────────────────────────────────────────────────────
eur_pallet_bk_cc0.glb                  5443    0.0310   0.8893   0.8545    2.39%
scene.usd                              4600    0.0232   0.8892   0.8636    2.59%
scene_1.usd                            4949    0.0242   0.8891   0.8626    2.89%
woodpallet_block_jtoastie_ccby.glb     5446    0.0289   0.8892   0.8585    1.67%

frame 단위 (다수 anchor 뒤집힘) : 49 / 2048  =  2.39%
```

읽는 법 두 가지.

1. **min() 이 발화하는 비율은 수렴 시점 2.4% 다.** 학습 도중에는 더 높으므로 이 값은
   **하한**이다. 상한은 측정 불가 — 재려면 학습을 돌려야 하는데 contract 전 학습
   금지다. 따라서 "A1 은 효과가 없다" 를 이 표로 주장하지 않는다.
2. **role 항은 어떤 정상적 margin 에서도 거의 죽어 있다.** sep 분포가 양극단
   (2.4% 는 음수, 나머지는 ≈0.86)이라 hinge 활성 비율이

   ```
   m     0.0   0.02   0.05   0.10   0.20   0.40   0.60   0.85   1.00
   활성  2.4%   2.4%   2.6%   2.8%   3.1%   4.4%   7.9%  41.7%  100%
   ```

   m 을 0.85 까지 올려야 활성이 의미 있게 늘지만, 그건 **이미 올바른 instance 까지
   벌주는** 값이라 채택할 수 없다. ⇒ CASE 2 라 해도 A1 의 신호는 사실상 SYM min()
   가지에서만 나온다. 이 사실을 결과가 나온 뒤에 발견하면 사후해석이 되므로 여기
   적어 둔다.

**real 표적 크기는 측정하지 못했다.** realdev161 의 fixed-object GT 가 아직 없고
(`11_realdev161_fixed_gt` 미생성), 그걸 만드는 것 자체가 axis contract 문제
(`CASE C GT_DEPENDENT_AXIS_LEAK_PRESENT`)에 묶여 있다. `_cc_raw_dump.json` 캐시는
camera-facing 학습본 예측이라 쓸 수 없다.

## 4. 검정력 — val 133 으로는 못 잰다

base rate 2.4% × 133 프레임 ≈ **3 프레임**. synthetic val 로 flip 감소를 재면
검정력이 없다. 따라서 진단셋을 분리한다.

```
A1_DIAG_SET = broad40k 중 v1_fixed_matched10k train 9,867 stem 을 제외한 슬라이스
              목표 ≥ 4,000 프레임 (base rate 2.4% → 기대 flip ≈ 96 프레임)
```
train 에 안 들어간 프레임만 쓴다. **이 정의는 학습 전에 고정하고 결과를 보고 바꾸지
않는다.**
