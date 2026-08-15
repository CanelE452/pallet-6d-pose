# PAPER_STAGE_A · PART 2 — mask auxiliary readiness 감사 (S1 게이트)

**목적**: paper_base_v2 학습 데이터(19308장)에 **visible object mask** 가 있는지 소스별로
실제 파일을 뒤져 판별 → Paper-S1(mask auxiliary FT) 가능/불가 결정.

**결론 (한 줄): paper 트랙에 valid visible mask 가 전무. Paper-S1 = BLOCKED.**

---

## 감사 대상 (paper_base_v2 header.txt 확인)
```
data = [ mixed_v8_train, v4_split_base, aug_squash_v2, aug_trunc_v2, aug_scale_v2 ]
truncation_aug_prob=0.0, mask_aux 없음 (heatmap only)
```
경로: `data/pallet/training_data/{...}`

## 소스별 결과 (전수 스캔)
```
source          n(json)  mask obj-key   frames_with_mask   seg PNG   결론
──────────────────────────────────────────────────────────────────────────────
mixed_v8_train    9000   []             0                  없음      mask 없음
v4_split_base     4000   []             0                  없음      mask 없음
aug_squash_v2     2212   []             0                  없음      mask 없음
aug_trunc_v2      2971   []             0                  없음      mask 없음
aug_scale_v2      1125   []             0                  없음      mask 없음
──────────────────────────────────────────────────────────────────────────────
합계             19308    —             0                   —        전무
```

### 세부 판정 (task 하위 질문별)
- **(a) 원본 옆 seg PNG / mask 파일 따로?** — 없음. training_data 하위 어디에도 `*mask*`/`*seg*`/
  `*instance*` 디렉토리나 PNG 없음(`find` 결과 0건). 각 프레임은 `{id}.png`(RGB) + `{id}.json` 뿐.
- **(b) mixed_v8 의 Isaac 생성분 instance seg 가 저장됐나?** — **아니오.** mixed_v8_train 9000장 전부
  JSON obj-key = `[class, visibility, location, quaternion_xyzw, euler_angles, pose_transform,
  projected_cuboid_centroid, projected_cuboid, cuboid]` — mask 필드 0. `.json.orig`(7205장)도 동일,
  mask 없음. Isaac instance seg 는 "무료 제공 가능"했지만 **생성 시 저장 안 됨**.
- **(c) aug류(squash/trunc/scale) mask?** — 없음. 이들은 v4_split_base 의 2D warp 파생본이고
  base 자체가 mask 없음 → 상속할 원천이 없음. JSON obj-key 에 mask 계열 0.

### 대조군 (valid mask 가 어떻게 생겼는지 — challenge 트랙, paper 금지)
```
challenge/data/02_synthetic/training/addon_v1_train : mask_rle=True,
    obj-keys 에 [visible_mask, mask_bbox_xywh, mask_area_px, mask_rle] 존재  ← 진짜 visible mask
challenge/data/02_synthetic/training/v3/*           : mask_rle 보유 (memory v3-mask-use-rle-not-png 확인)
```
→ B2(`stage11_16k_B2_maskaux`)의 mask_aux 는 **v3 + addon_v1_train 의 mask_rle** 에서 왔음
(B2 header 의 data 목록에 v1/v2/v3/addon 포함). **이 소스들은 challenge 전용 → paper 트랙 사용 금지**
(memory v1v2-challenge-only, paper purity).

### bbox/hull 가짜 mask 배제 확인
paper 소스엔 `projected_cuboid`/`cuboid` 는 있으나, 이걸로 만든 mask 는 **cuboid-hull 과포함
confound**(memory pvnet-dense-vector-voting-negative: 박스마스크 과포함이 real 도메인갭 유발).
→ hull/bbox 가짜 mask 는 **valid 아님**. 사용 금지.

---

## S1 실행 메커니즘 확인 (왜 config 를 못 쓰나)
`Deep_Object_Pose/common/utils_dataset.py` (mask_aux 경로) [확인]:
- `mask_aux=True` → `pvnet_mask_rle=True` → `__getitem__` 에서 `obj0["mask_rle"]` 를
  `decode_mask_rle` 로 디코드.
- **mask_rle 부재 시 `pvnet_mask_valid=0.0`** → trainer 가 해당 프레임 mask loss 를 **마스크아웃**.
- 즉 paper 소스(mask_rle 0건)로 mask_aux 를 켜면 **모든 프레임 valid=0 → seg head 가 학습 신호 0**.
  = mask auxiliary 가 no-op. **가짜 S1**(실제 mask 정규화 효과 없음).

→ 그러므로 **PART 3 config 를 쓰지 않는다** (억지 config = 무의미/오해 소지).

---

## PART 3 판정: Paper-S1 = BLOCKED

**블로커 = paper 트랙 데이터에 visible mask(mask_rle) 부재.** config 미작성.

### 대안 경로 (mask 를 정직하게 확보하는 방법)
1. **[정공법] Isaac Sim 재렌더 + instance/semantic segmentation annotator.**
   paper_base_v2 의 mixed_v8_train/v4_split_base 는 절차적(procedural) Isaac/Replicator 생성물이므로,
   동일 씬 구성으로 재생성하되 Replicator 의 `instance_segmentation`/`semantic_segmentation` 출력을
   켜서 프레임별 occlusion-aware visible mask → `mask_rle` 로 저장. (판자 사이 구멍·가림 보존됨,
   무료 GT). ★ Blender 환경 블로커 주의(memory STAGE16: Step2 생성=Windows FoundationPose 필요).
2. **[차선] Blender 재렌더** (동일 자산 사용 가능 시) instance pass 로 visible mask 추출.
3. **[근사, 위험] SAM/그 계열로 RGB→visible mask 유사생성** 후 occlusion-aware 검증 통과분만 사용.
   단 pseudo-mask 품질 confound 리스크 → 정량 주장에는 부적합, 정성/보조만.
4. **하지 말 것**: cuboid-hull/bbox 투영 mask(과포함 confound, memory pvnet-negative).

### mask 확보되면 즉시 쓸 S1 recipe (B2 기준, 대기)
mask valid 확보 후에만 아래로 `Deep_Object_Pose/train/train.py` 실행 (config 는 그때 작성):
```
net_path            = weights/paper_base/paper_base_v2/final_net_epoch_0060.pth   (paper ckpt에서 이어받기)
data                = paper 소스 5종(+ 새로 mask 붙은 재렌더분)만. v3/addon/B2 절대 금지
mask_aux            = True
mask_weight         = 0.01      (B2 값)
mask_warmup         = 0
encoder_lr_scale    = 0.1       (encoder 저LR)
encoder_freeze_steps= 750
sigma               = 4.0
truncation_aug_prob = 0.0       (paper_base_v2 parity 유지)
epochs              = base+quick (누적 목표; memory dope-finetune-cumulative-epoch)
outf                = weights/paper_S1_maskaux
```
heatmap = main task 유지, dense vector field 미구축(heatmap+mask aux only).
```
```
