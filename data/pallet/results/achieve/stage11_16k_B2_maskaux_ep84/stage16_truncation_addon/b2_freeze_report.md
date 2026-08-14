# B2 Freeze Report — STAGE16 Truncation Add-on Baseline

목적: STAGE16(V<8 truncation 데이터 add-on 트랙)의 **고정 baseline** 으로 B2 를 동결한다.
이 문서 이후 STAGE16 의 모든 비교는 여기 명시된 B2 checkpoint·recipe·eval 설정을 기준으로 한다.

작성일: 2026-06-29. 출처 태그: `[확인]`=파일/로그에서 직접 인용, `[추정]`=주석/명명에서 추론(미검증).

---

## 1. Checkpoint (frozen)

- **B2 = `weights/stage11_16k_B2_maskaux/final_net_epoch_0084.pth`** `[확인]`
  - = `net_epoch_0084.pth` 와 동일 내용 (size 219,330,070 vs 219,328,798; `final_*` 는 동일 epoch save). 학습 추적 epoch = 0084.
  - SHA 미기록(필요시 별도 산출).
- 계보: `challenge0123 baseline` → `B3_replay`(replay-mix anti-forgetting) → **B2_maskaux**(mask-aux finetune).
  - B2 init = `weights/stage11_16k_B3_replay/net_epoch_0074.pth` `[확인 header.txt: net_path]`

---

## 2. Training Recipe

출처: `weights/stage11_16k_B2_maskaux/header.txt` (저장된 argparse Namespace) `[확인]` +
`data/pallet/eval_results/stage11_16k/run_train_B2.sh` `[확인]`.

```
optimizer / schedule
  batchsize        8
  imagesize        448
  lr               1e-4
  sigma            4.0        (belief Gaussian; <1 gradient-vanish 회피 — 프로젝트 규칙)
  epochs           84         (cumulative: init ep0074 → +10 → ep0084)
  save_every       1
  manualseed       2657
  workers          6

init / finetune
  net_path         weights/stage11_16k_B3_replay/net_epoch_0074.pth  (B3_final)
  (train.py finetune = net_path epoch 이어받고 --epochs 누적목표 해석;
   memory dope-finetune-cumulative-epoch)

discriminative LR / freeze
  encoder_lr_scale     0.1    (encoder=VGG backbone LR ×0.1)
  encoder_freeze_steps 750    (초반 750 step encoder freeze)
  BN freeze            명시 플래그 없음 → train.py 기본거동 [추정]

balance / replay (anti-forgetting)
  balance_groups   "mixed_v8_train|/v1/|/v2/:2,  /v3/:1,  addon_v1_train:1"
  → group ratio = base(mixed_v8 + v1 + v2) : v3 : addon = 2 : 1 : 1
    (B3 와 동일 replay-mix → forgetting 방지 유지)

mask auxiliary (B2 의 핵심 추가; TRAINING-ONLY)
  mask_aux         True
  mask_weight      0.01       (spec 0.05 였으나 belief loss ~0.01-0.02 vs mask BCE
                               floor ~0.2-0.4 → 0.05 면 co-dominant 위반.
                               0.01 로 mask term ~0.003 subordinate. run_train_B2.sh NOTE)
  mask_warmup      0
  mask GT source   JSON mask_rle decode ONLY. old base(mask_rle 없음)→valid=0,
                   mask loss 미적용. v3/addon frame 만 mask BCE 기여. [확인 주석]
  inference 영향   없음 — belief-peak decode 그대로, mask hard-gate 안 씀.
                   heatmap=main, mask=aux. [확인 run_train_B2.sh 헤더]

off (사용 안 한 loss)
  truncation_aug_prob 0.0   ★ B2 는 truncation 증강 OFF (STAGE16 가 메우려는 빈틈)
  symmetric_loss / geo_loss / vis_coord_loss / rel_loss / struct_loss = 모두 False
```

---

## 3. Dataset Manifest

출처: header.txt `data=[...]` `[확인]` + 실제 디렉토리 json count `[확인]`.

```
group           dir                                              n(json)   replay weight
──────────────────────────────────────────────────────────────────────────────────────
base  mixed_v8_train  data/pallet/training_data/mixed_v8_train     9,000  ┐
base  v1              challenge/data/02_synthetic/training/v1                    9,997  ├ 2
base  v2              challenge/data/02_synthetic/training/v2                    9,994  ┘
v3    batch_000..008  challenge/data/02_synthetic/training/v3/batch_00[0-8]     18,000    1   (2000×9)
addon addon_v1_train  challenge/data/02_synthetic/training/addon_v1_train        5,400    1
──────────────────────────────────────────────────────────────────────────────────────
                                                       원본 합계 ≈ 52,391 frames
```
- replay weight 는 epoch 당 sampling 비율(2:1:1)이지 원본 장수 비율이 아님.
- v3 = challenge 트랙 Blender 합성(mask_rle 보유). addon_v1_train = palletobj_v1 add-on.
- mixed_v8_train = v4 교정본(camera-facing 0123). v8 object-frame 폐기본 아님
  (memory mixed-v8-train-corrected-usable). v1/v2 = challenge 전용(논문 금지).
- ⚠ 트랙: B2 = **challenge 트랙**(v1/v2/mixed_v8 포함). 논문 트랙 아님.

---

## 4. Eval Setup (frozen for STAGE16 comparisons)

출처: `data/pallet/eval_results/stage11_16k/eval_stage11.py` `[확인]`.

```
real eval set    collect_val_frames()  (filter-val outside+night)
               + collect_manual()      (manual GT 36)
               = 123 real frames        (V=8: 106,  V<8: 17)
               SEAL guard: FINAL_TEST_SESSIONS leak assert (capturepallet07/09 등 제외)

preprocessing    aspect-only, NO padding (min-side→400, /8 격자정렬). diag 에서 base 최선.
                 MEAN/STD = ImageNet. (eval_pvnet_heads.preprocess)
decode           heatmap-only (belief peak). threshold = 0.3.
                 sigma = 4.0 (학습과 동일).

metrics (order-free)
  corner_med     hungarian overall median px (split_metrics overall, orig px)
  good%          corner < 10px
  det%           n_det >= 6  (N_DET_MIN=6)
  worst2         hungarian 매칭 코너 거리 top-2 평균의 frame-median
  pnp%           solve_pose(9kp, per-frame K, GT dims) 성공률
  pnp_rep        성공 frame 의 reproj px median
  honest full-8  (STAGE16 추가) solve_pose 후 8 corner projection vs GT projected_cuboid
                 (화면밖 코너 포함) median — gate-only false-accept 노출용
  V split        occ = in-frame corner 수(num_corners_unoccluded 또는 GT 화면안 count)
                 V=8 / V<8(0<=occ<8). size/depth 보강축은 GT 분포 33/66 percentile.
```

---

## 5. Eval Numbers Recap (baseline / B3 / B2)

출처: `data/pallet/eval_results/stage11_16k/eval_report.txt`(baseline, B3_ep74) `[확인]`
+ `_docs/history/2026-06-27.md` STAGE13 표(B2 maskaux) `[확인]`.
real 123 frames (V=8:106 / V<8:17). ⚠ **소표본** — 특히 V<8 N=17 은 단일 수치 과신 금지.

```
V=8 (N=106)
model         good%   corner_med   det%    worst2   pnp%
──────────────────────────────────────────────────────────
challenge0123  32.4     11.3       67.0     26.8    67.9
B3_replay      44.7     10.2       71.7     24.0    72.6
B2_maskaux     55.3★     9.6★      71.7     22.2★   77.4★    ← frozen baseline

V<8 (N=17)
model         det%    corner_med   pnp%     note
──────────────────────────────────────────────────────────
challenge0123  5.9      14.8        5.9
B3_replay      0.0      —           0.0
B2_maskaux    11.8       —          —       (=2/17 검출. 여전히 약점)

ALL (N=123)  — eval_report.txt
challenge0123  good 31.9  corner 11.4  det 58.5  pnp 59.3
B3_ep74        good 44.7  corner 10.2  det 61.8  pnp 62.6
(B2 ALL 은 STAGE13 당시 미기록 — V=8/V<8 분해만 있음. STAGE16 Step1 재평가로 보강.)
```

누적 효과(challenge0123 → B2, V=8): good% +22.9%p, worst2 26.8→22.2, pnp 67.9→77.4.
레버 = **데이터-side**(replay-mix → mask-aux). 표현 변경(벡터/voting/offset/subset-PnP)은
모두 heatmap 에 패배(memory pvnet-dense-vector-voting-negative-result, diag 진단).

---

## 6. B2 의 남은 약점 = STAGE16 의 표적

- V=8 은 B2 로 양호(good 55%, det 72%)하나 **V<8(truncation) 은 det 11.8%** 로 거의 붕괴.
- 원인(diag·STAGE12): truncation 은 postprocess(subset-PnP decode)로 회복 불가 →
  **학습/증강 레버** 필요. B2 는 `truncation_aug_prob=0.0` 로 truncation 노출이 사실상 없음.
- STAGE16 = real V<8 실패 분포를 측정(Step1) → 그 분포를 따라가는 truncation add-on
  합성데이터 생성(다음 단계). 블라인드 V<8 양산 방지.

미상/추정 항목: BN freeze 거동(train.py 기본), checkpoint SHA(미산출).
