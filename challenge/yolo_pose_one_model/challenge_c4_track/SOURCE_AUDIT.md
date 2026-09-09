# SOURCE_AUDIT — 과제용 C4 track

2026-09-05.  기억이나 지시문의 경로를 믿지 않고 **디스크에서 직접** 확인한 결과다.
기계 판독용 값은 `SOURCE_AUDIT.json` 에 있고, 여기에는 판단과 단서를 적는다.

```
git HEAD    985201c2297e8408fce72d72e3d4ca8e790bbdb8
            "livegt: 09-03·09-04 촬영분 반입 + flip/noise 증강 + v4 FT"
branch      main (origin/main 과 동기)
ultralytics 8.4.60      torch 2.1.1+cu118      CUDA 11.8
GPU         NVIDIA GeForce RTX 3080 (10,240 MiB)      python 3.10.20
```

작업트리에 다른 세션의 미커밋 변경 154 항목이 있다.  이번 track 은 새 폴더만 쓰고
기존 산출물을 덮지 않는다.

---

## 1. Stage A 후보 — `runs_paper/yolo26n_paper_generic_v1_seed42`

`args.yaml` 에서 읽은 실제 실행 recipe (기억으로 재구성하지 않았다):

```
model         challenge/weights/pretrained_yolo/yolo26n-pose.pt   (COCO pretrained)
data          datasets/broad40k/data.yaml
epochs 60     batch 32      imgsz 640      seed 42      patience 0
optimizer SGD lr0 0.01  lrf 0.01  cos_lr  warmup 3.0  momentum 0.937  wd 0.0005
pose 12.0     kobj 1.0     box 7.5   cls 0.5   dfl 1.5   rle 1.0
mosaic 0.3    close_mosaic 10   scale 0.25   translate 0.1
fliplr 0.0    flipud 0.0   degrees 0   shear 0   perspective 0
hsv 0.015 / 0.5 / 0.35     erasing 0.4     auto_augment randaugment
single_cls true   deterministic true   workers 4   save_period 10
```

`results.csv` 는 헤더 + **60 epoch 행**, 결번 없음 — 정상 완주 `[확인]`.

### checkpoint

```
last.pt   6a40a4d430fd205a427e38a1927aad2a0a0bef20b984484e0b77318e7bd355ea   6,552,679 B
best.pt   632b1248911c6cc4629eb97b601d1833c5251d53eba26145354f3c3dd5fa46ce
```

`last.pt` 의 SHA 는 `dimension_conditioning_probe/SOURCE_AUDIT.md` 에 이미 기록돼
있던 값과 **정확히 일치**한다 `[확인]`.  provenance 불명이 아니므로 §13-2 중단 사유에
해당하지 않는다.

### ★ best.pt 를 쓰면 안 되는 이유

`datasets/broad40k/_build.json` 과 `paper_generic_v1_manifest.json` 이 명시한다 —
val 500 장은 **train 과 같은 pool 에서 strided 로 뽑은 것**이지 holdout 이 아니다
(`stems[::len(stems)//500]`).  그래서 이 run 의 `results.csv` pose mAP50-95 0.93188 은
in-distribution 값이고, checkpoint 선택에 쓰면 안 된다.  기록들이 반복해서
"**last.pt 를 쓰라**" 고 적어 두었다.  이 track 도 `last.pt` 를 INIT 으로 쓴다.

---

## 2. broad40k provenance — 요구한 네 가지 전부 확인

```
train 39,500 / val 500        image·label 짝 불일치 0
symlink 0 (전부 실파일)        깨진 링크 0
kpt_shape [9,3]               라벨 40,000 개 전 줄이 정확히 32 토큰
flip_idx [1,0,3,2,5,4,7,6,8]  (단 fliplr 0.0 이라 이 run 에서는 사용되지 않음)
```

### v1/v2 palletobj 혼입 — 없다

파일명 패턴이 아니라 **원본 라벨 40,000 개의 내용**을 전수 조사했다.

```
source_asset 히스토그램 (40,000 개 전수)
  10,182  eur_pallet_bk_cc0.glb
  10,099  woodpallet_block_jtoastie_ccby.glb
  10,095  scene_1.usd
   9,624  scene.usd

"palletobj" 포함        0
"pallet_full" 포함      0
"scan" 포함            28  → 전부 occluder_asset: hand_truck_scan_02 (가림물, 타깃 아님)
```

### real 혼입 — 없다

파일명은 전부 `f<숫자>.png` 한 패턴으로 붕괴한다(`manual_gt` / `capture_` /
`forklift_` / `live_capture` / `real_` / `night` / `outside` / `pallet0` 전부 0 건).
라벨에는 camera intrinsics · `pose_transform` · `euler_angles` · `background_asset` ·
`exposure_ev` · `scene_preset` 이 프레임마다 들어 있고 `gt_source` 는 `(none)` 이다 —
전부 렌더된 synthetic `[확인]`.

### `g38_legacy_v1v2_p0_tex20k` 와 혼동하지 않았다

```
broad40k                    39,500 / 500     실파일, f####.png
g38_legacy_v1v2_p0_tex20k   55,980 / 4,020   100% symlink, G38__/TEX__/P__ 세 shard
train stem 교집합            0
```

v1/v2 재료(`P__shard`, `TEX__shard`)는 저쪽에만 있고 broad40k 에는 하나도 없다.

### 남는 한계 두 가지 — 기록에 남긴다

1. **mesh-hash 대조는 이 머신에서 불가능하다.** `TARGET_ASSET_EXCLUSION_AUDIT.md` 가
   `NOT_POSSIBLE_LOCALLY` 로 적어 두었다 (4 자산 중 0 개만 hash 가능 — `.usd` 는 `pxr`
   없이 파싱 불가, `.glb` 두 개는 이 머신에 없음).  따라서 배제 근거는 **asset 문자열
   전수 + `dimensions_m` 기하 signature** 다.  "메시 단위로 증명했다" 고 쓰면 과장이다.
2. **두께 비 coverage 구멍.** 과제 타깃의 두께비는 0.0923 인데 broad40k 에서 그보다
   얇은 프레임은 117/40,000 (0.29%) 뿐이다 (`PAPER_GENERIC_PURPOSE.md` 기록).

### keypoint 규약

broad40k 라벨은 `camera_dynamic_0123_v4` — **camera-facing** 이다.  같은 RGB 를 쓰는
object-fixed relabel 은 `broad40k_fixed` 로 따로 있다.  이번 track 의 real GT 도
camera-facing 0123 이므로 **broad40k 쪽이 맞다** `[확인]`.

---

## 3. Stage A 재사용 판정 — 재학습하지 않는다

지시문 §2-1 의 조건을 하나씩 대조했다.

```
broad40k membership 일치     ✓  (위 감사)
COCO yolo26n-pose init       ✓  args.yaml model
60 epoch 정상 완주            ✓  results.csv 60 행
batch 32                     ✓
seed 42                      ✓
stock PoseLoss26             ✓  custom loss 흔적 없음
real supervision 없음         ✓  라벨 전수 조사
checkpoint 파일 정상          ✓  SHA 가 기존 기록과 일치
results.csv epoch 수 정상     ✓
```

전부 충족하므로 **60 epoch 를 다시 돌리지 않고 `last.pt` 를 INIT 으로 재사용**한다.
기존 run 은 덮어쓰지 않는다.

---

## 4. square real FT 데이터 — `datasets/live_gt_v4`

추측한 이름을 쓰지 않고 `runs_live_gt/ft_live_gt_v4/args.yaml` 의 `data:` 를 따라갔다.

```
train 3,087        원본 696 · truncation crop 999 · flip 696 · noise 696
val     155        원본만 (crop/aug 0 개)
split              interleave, val_every 6
kpt_shape [9,3]    flip_idx [1,0,3,2,5,4,7,6,8]
라벨 3,242 개 전부 1 줄 · 32 토큰
촬영 그룹          handheld_20260902 344 · forklift_v4_20260901 67
                   forklift_v4_20260903 51 · forklift_v4_20260904 389
```

### val-derived leakage = 0 (선언이 아니라 계산)

파생본 이름을 정규화해(접두어 `crop__`/`aug__` 제거, 꼬리 `_t<n>`/`_f`/`_n` 제거,
`manual_gt__` 토큰 제거) 원본 stem 을 복원한 뒤 교집합을 실제로 셌다.

```
train 파생본이 유래한 원본 stem   696
val 원본 stem                     155
교집합                              0      ← leakage 없음
정규화 규칙 검증                  696/696  train 원본과 완전 일치 (가짜 0 이 아님)
```

수치 교차검증도 맞는다: `flip_noise_aug_livegt` 1,702 = 851×2, 사용 1,392 = 696×2,
제외 310 = 155×2.  `truncation_crops_livegt` 1,206 = 402×3, 사용 999 = 333×3,
제외 207 = 69×3.

### ★ 교란변수로 기록해 둘 것

* **crop 은 train 원본의 48%(333/696)만 덮는다.**  crop 소스가 402 장짜리 구세대
  촬영분에서만 만들어졌고 09-03·09-04 신규분에는 재생성되지 않았다.  이번 F0/F1 은
  둘 다 같은 데이터를 쓰므로 arm 간 비교는 오염되지 않지만, 절대 성능을 해석할 때는
  이 불균형을 감안해야 한다.
* **`erasing 0.4` · `auto_augment randaugment` 는 통제되지 않은 잔여 증강이다.**
  `BASE_CONTRACT_AUG` 에 없는 ultralytics 기본값이라 "base contract" 라는 이름과 달리
  꺼져 있지 않다.  F0/F1 양쪽에 똑같이 적용되므로 비교에는 영향이 없다.
* `live_gt_v4` 데이터셋과 `ft_live_gt_v4` run 은 **git 에 없다**.  지금 이 디스크가
  유일한 사본이다.

---

## 5. 이번 track 이 v4 와 다른 점

v4 는 `pallet_yolo26n_pose_ft.pt`(real 161 + negative + synthetic 으로 FT 된 것)에서
출발했다.  이번 track 은 설계상 **PAPER_GENERIC Stage A checkpoint** 에서 출발한다.
recipe(epoch/batch/optimizer/lr/augment/seed)만 v4 artifact 에서 읽어 그대로 쓴다.

```
INIT   runs_paper/yolo26n_paper_generic_v1_seed42/weights/last.pt
       sha256 6a40a4d430fd205a427e38a1927aad2a0a0bef20b984484e0b77318e7bd355ea
       F0 / F1 동일
```

---

## 6. 중단 기준 점검 (§13)

```
1  broad40k real/v1v2 혼입 배제 실패      해당 없음 — 라벨 전수 조사로 배제
2  Stage A provenance 불명               해당 없음 — SHA 가 기존 기록과 일치
```

나머지 항목(permutation 구조, parity, one2many/one2one, NaN, stale cache,
init SHA 동일, data membership 동일)은 `TEST_RESULTS.txt` 와 실행 로그에서 다룬다.
