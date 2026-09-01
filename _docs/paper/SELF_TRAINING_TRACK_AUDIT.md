# YOLO26 geometry-filtered self-training track — PHASE 0 audit

작성 2026-09-01. 학습 0회 시점의 감사 기록이다. 여기 있는 수치는 전부 실측이고,
결과를 본 뒤 고치지 않는다.

`HEAD` = `dfb8cd71dafd440c4a66c1a550c23fc31de6e12f` (main).

## 0. 트랙 방향 (고정)

paper pipeline 은 **YOLO26 기반으로 고정**한다. DOPE 는 비교 baseline 및
same-data backbone control 로만 쓴다. `scripts/self_training/self_train.py` 는
DOPE 전용이므로 MAIN Proposed 학습에 사용하지 않는다.

```text
Ours = Confidence + LOO + Flip
```

confidence 는 contribution 이 아니라 **standard pre-filter** 다.
methodological contribution 은 LOO + Flip 두 기하 일관성 필터다.

## 1. Base checkpoint

```text
role                    path                                                                sha256
────────────────────────────────────────────────────────────────────────────────────────────────────
YOLO26n synthetic-only  challenge/yolo_pose_one_model/spatial_concat_scratch/runs/           970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7
                        YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt
DOPE same-data          weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth          0de80490cb3b4f9b11565db7a4aea6338f64edb8f9614910bfb52bf03ce0dc3f
```

YOLO26 checkpoint 는 `conda activate pallet-yolo26` 에서만 로드된다.

```text
env pallet-yolo26   python 3.10.20   ultralytics 8.4.60   torch 2.1.1+cu118 (cuda 11.8)
```

`pallet-pose` env 는 ultralytics 8.0.120 이라 C3k2 가 없어 `torch.load` 에서 죽는다.

## 2. PAPER_EVAL — paper-facing source of truth

`PAPER_EVAL` = SHA256-deduplicated union(DEV_EVAL, NEW_EVAL).
`scripts/evaluation/eval_workspace.py::evaluation_population_views` 가 유일한 산출부다.
숫자를 문서에 복붙하지 말고 여기서 재계산한다.

```text
population              N
──────────────────────────
positive (combined)   319
  plastic             194
  wood                125
negative (unique)    2688
paper_domain daytime   70   plastic, PREFERRED_READY
paper_domain nighttime  50   plastic, READY
paper_domain none      199
```

`FINAL_EVAL` 은 frozen DEV alias(173행 고정)다. **paper 표에 쓰지 않는다** —
쓰면 새로 어노테이션한 프레임이 영원히 보이지 않는다.

### 2.1 구성 — 왜 evaluator closure 가 MAIN 의 하드 blocker 인가

```text
source                              N    storage_mode        gt 위치
──────────────────────────────────────────────────────────────────────────────────
dev_existing/  (legacy DEV)       173    independent_copy    data/evaluation/pallet_eval_v1/dev_existing/annotations/
final/positive/ (신규 어노)        146    workspace_native    data/evaluation/pallet_eval_v1/final/positive/annotations/
```

신규 146 = plastic_day_01 44 + plastic_night_01 22 + wood_day_01 24 + wood_night_01 56.

MAIN 의 daytime 70 / nighttime 50 은 legacy 와 신규에 **걸쳐 있다**.
따라서 evaluator 를 PAPER_EVAL 로 열지 않으면 M2 자체를 계산할 수 없다.
이건 선택이 아니라 선행 조건이다.

## 3. Evaluator closure — 실측 판정

`challenge/evaluation_v2` 는 legacy 173 DEV population 에 묶여 있다
(`COMMON_DEV_PLASTIC_POS128` + `DEV_WOOD_POS45` + `DEV_NEG2689`).

`paper_real_eval.py` 의 하드 게이트는 `schema_version == "real_pallet_gt_v2"` 이고
전체 검증기 `validate_gt_v2` 를 통과해야 한다. 실제로 돌려봤다.

```text
set                                 validate_gt_v2   schema_version
────────────────────────────────────────────────────────────────────
NEW  final/positive (146)           PASS 146/146     real_pallet_gt_v2
LEGACY migrated_gt (140)            PASS 140/140     real_pallet_gt_v2
LEGACY migrated_gt_wood (45)        PASS  45/45      real_pallet_gt_v2
workspace dev_existing (185)        PASS 185/185     real_pallet_gt_v2
```

legacy 173 의 workspace 사본은 `challenge/real_gt_v2/migrated_gt*` 와 GT 핵심 필드가
**173/173 동일**하다 (pose_transform, projected_cuboid, manual_kps,
keypoint_annotations, physical_dimensions_m). workspace 를 단일 source 로 써도 손실이 없다.

**판정: 2D metric 과 ranking metric(AP / AUROC / FPR95)은 형식 변환 없이 연결 가능하다.**

### 3.1 필요한 코드 변경 (frozen contract 확장)

`challenge/evaluation_v2/real_dataset_contract.py` 는 population 을 닫힌 enum 으로 관리한다.
PAPER_EVAL 을 추가하려면 다음 7개 지점을 함께 고쳐야 한다.

```text
PopulationId                            신규 id 3개 추가
EXPECTED_POPULATIONS                    count/kind/role
POPULATION_OBJECT_TYPES                 manifest-level object scope
_OBJECT_AWARE_AVAILABLE_POPULATIONS     object-aware 판정 대상
_OBJECT_AWARE_SOURCE_TYPES              source→object_type
_OBJECT_AWARE_SOURCE_ROLES              source→role
validate_evaluation_pair                허용 pair
```

manifest 는 `pallet_pose_population_manifest_v1` 형식으로 생성하고
`membership_sha256` 은 `real_dataset_contract.membership_sha256` 으로 계산한다.

### 3.2 POSE_METRICS_STATUS = BLOCKED

```text
신규 146 장   pose_status                UNCONFIRMED_SIGNED_AXIS    146/146
              migration_status           MANUAL_REVIEW_REQUIRED     146/146
              camera_facing_pnp.axis_assignment_confirmed  False    146/146
기존 gate     PLASTIC  selector FAIL (83/140, NIGHT 13/28); symmetry FROZEN
              WOOD     selector NOT_RUN; signed pose/symmetry UNREVIEWED
              ALL      Restricted ADD-S / rotation / translation / yaw BLOCKED → null
```

blocker 를 축별로 분리하면:

```text
축                    상태        원인
──────────────────────────────────────────────────────────────────────────
Plastic axis selector BLOCKED     prediction-only W/D hypothesis 선택 실패
Wood symmetry         BLOCKED     symmetry_status UNREVIEWED (registry)
신규 146 signed axis  BLOCKED     axis_assignment 미확정 (어노 단계 산출물)
intrinsics            OK          camera_data.intrinsics 전 프레임 존재
```

→ M1/M2/M3/M5 의 `R med / Yaw / t / IoU3D / AUC` 는 **채우지 않는다.**
2D 성능이 좋아도 "6D pose improved" 라고 쓰지 않는다.
self-training 자체는 2D pseudo-keypoint 기반이므로 이 blocker 와 무관하게 진행 가능하다.

## 4. 재사용 판정 — 기존 수학적 필터

`scripts/data_prep/canonical_filters.py` 는 무차원 비율 기반이라 해상도/거리 불변이다.

```text
filter_C(kps_orig, pnp_solver, R, t, tau_C=0.05, min_kps=5)
    s_C = median(leave-one-keypoint-out reprojection error) / projected_diagonal
    입력이 순수 2D keypoint → YOLO 에 그대로 재사용한다.

filter_A(model, img_bgr, pred_kps_belief, device, pnp_solver, R, t, tau_A=0.05)
    DOPE 전용. belief map forward + 400x400 학습입력 + 448 하드코딩 스케일에 묶여 있다.
    → YOLO 용으로 재구현한다. 수식만 이식하고 절대 px threshold 는 만들지 않는다.
    s_A = median || p_i - unflip(swap(p_i^flip)) || / projected_diagonal

FLIP_PAIRS = [(0,1), (3,2), (4,5), (7,6)]
```

R0 의 `data.yaml` 은 `flip_idx: [1,0,3,2,5,4,7,6,8]` 이다 —
(0,1)(2,3)(4,5)(6,7) 스왑에 8 고정으로, `FLIP_PAIRS` 와 **동일한 페어링**이다. 일치 확인됨.

tau 는 양쪽 다 canonical default `0.05` 를 frozen constant 로 쓴다.
YOLO 640 이라는 이유로 10px / 20px 같은 새 절대 threshold 를 만들지 않는다.

기존 `scripts/self_training/self_train_pseudo.py::_flip_score` 의 absolute 10px gate 는
**사용하지 않는다** (DOPE 해상도 전제).

## 5. R0 학습 레시피 (args.yaml 실측 — 새로 지어내지 않는다)

```text
model      yolo26n-pose.pt (COCO pretrained)      epochs 60    batch 32    imgsz 640
optimizer  SGD    lr0 0.01   lrf 0.01   cos_lr    momentum 0.937   weight_decay 0.0005
warmup     3.0    seed 42    deterministic true   patience 0       close_mosaic 10
loss       box 7.5   cls 0.5   dfl 1.5   pose 12.0   kobj 1.0
aug        mosaic 0.3   mixup 0   copy_paste 0   erasing 0.4   scale 0.25   translate 0.1
           hsv 0.015/0.5/0.35   degrees 0   fliplr 0.0   flipud 0.0
data       challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k/data.yaml
           nc 1   kpt_shape [9,3]   flip_idx [1,0,3,2,5,4,7,6,8]   names {0: pallet}
```

`fliplr = 0.0` 이 중요하다. R0 는 좌우반전 augmentation 없이 학습됐으므로
flip consistency 는 학습으로 강제된 항등식이 아니다 — **독립 신호로서 유효**하다.
학생 arm 도 같은 값을 유지해야 arm 간 비교가 성립한다.

## 6. Teacher inference 계약 (재사용 — 이중 threshold 금지)

`scripts/stage0/model_compare/mc_dump_yolo.py` 가 release README 배포 계약을 그대로 쓴다.

```text
PAD 100   BORDER_REFLECT_101   imgsz 640   conf 0.4   최고 box confidence 인스턴스   좌표 -PAD
```

teacher cache 는 이 계약을 그대로 따른다. `conf 0.4` 는 inference floor 이고
TAU_BOX 후보 0.70~0.85 보다 낮으므로 quality gate 와 겹치지 않는다.
pseudo-label gate 는 cache 를 만든 **뒤에** 적용한다.

confidence score 는 `result.boxes.conf` 하나만 쓴다. 논문 명칭은
**YOLO detection confidence** 이고 calibrated probability 라고 부르지 않는다.
`box_conf * kp_conf` 같은 post-hoc score 를 새로 만들지 않는다.
`keypoints.conf` 는 valid corner 판정과 diagnostic 에만 쓴다.

## 7. Adaptation pool 실현성 (PHASE 5)

```text
pool                                imgs      sessions
────────────────────────────────────────────────────────
data/pallet/raw_data/outside      21,644           15
data/pallet/raw_data/night        18,268           10
```

Daytime 500 + Nighttime 500 = `U_MAIN` 1000 unique 는 충분히 가능하다.
membership 은 deterministic SHA sampling 으로 고정하고, freeze 전에 반드시 검증한다.

```text
adapt session ∩ eval session  = 0
exact image SHA overlap       = 0
```

MAIN Proposed 는 **모델 하나**다. Day model / Night model 을 따로 MAIN 으로 만들지 않는다.
M2 는 같은 Proposed 모델을 Daytime / Nighttime subgroup 으로 나눠 평가한다.
domain-specific model 이 필요하면 Appendix A8 에서만 한다.

## 8. EXPERIMENTS.md stale (PHASE 1 대상)

현재 M1/M5/§7 이 폐기된 `FINAL_EVAL` population 을 노출하고 있다.

```text
위치                     현재 (stale)                        교체 대상
──────────────────────────────────────────────────────────────────────────────
L129-134  M1 결과표      FINAL_EVAL / PLASTIC 128 / 173 2689  PAPER_EVAL / 194 / 319 2688
L137-139  M1 각주        positive 173행, denominator 128       PAPER_EVAL 재계산값
L448-454  M5 조건표      FINAL_EVAL Plastic 128 Wood 45        PAPER_EVAL 194 / 125
                         DAY 100 NIGHT 28                      DAY 168 NIGHT 106
L910-927  §7 구성표      FINAL_EVAL 173 / 2688                 PAPER_EVAL 319 / 2688
```

숫자를 손으로 복붙하지 않는다. `evaluation_population_views` 에서 재계산해
표 metadata 를 생성하고, M1/M2/M3/M5 가 모두 같은 PAPER_EVAL source 를 보게 한다.

## 9. 실행 순서 (PHASE 28 — 이 순서를 지킨다)

```text
      단계                                      상태
────────────────────────────────────────────────────────────
P0-0  repository 감사                           DONE (이 문서)
P0-1  PAPER_EVAL evaluator closure              NEXT
P0-2  R0 YOLO / DOPE baseline re-eval
P0-3  Day500 + Night500 adaptation manifest freeze
P0-4  R0 teacher inference cache (1회만)
P0-5  confidence threshold unlabeled-only lock
P0-6  YOLO LOO / normalized flip 구현 + unit test
P0-7  M4 filter-quality diagnostic
P0-8  F0/F1/F2/F3/F4 pseudo-label manifest freeze
P0-9  5-arm smoke
P0-10 5-arm full seed42 training
P0-11 PAPER_EVAL full evaluation
P0-12 M1/M2/M3/M4/M5 표 생성
```

그 뒤에만 A2 quantity-matched / A3 rounds·seeds / PVNet·SingleShotPose 를 한다.

첫 MAIN 실험은 **ONE-ROUND STATIC TEACHER** 로 고정한다. 학생 5개는 전부 같은 R0
init, 같은 teacher cache, 같은 U_MAIN, 같은 synthetic replay, 같은 exposure/epoch/seed/aug 를
쓰고 **pseudo-label selection rule 만 다르다**. teacher drift 를 필터 비교에 섞지 않기 위해서다.

## 10. 이 트랙에서 하지 않을 것

```text
DOPE self_train.py 를 YOLO proposed 학습에 사용
PAPER_EVAL GT 로 confidence threshold tuning
결과를 본 뒤 threshold / epoch 수정
box_conf 를 calibrated probability 로 호칭
box_conf * kp_conf 같은 post-hoc score 생성
DOPE flip 10px absolute threshold 를 YOLO 640 에 그대로 사용
LOO 에서 GT W/D axis 사용 (registry hypothesis 둘의 min 으로만 판단)
pseudo 6D pose 를 GT 처럼 저장 (2D box + 2D keypoints + visibility 만 저장)
Daytime/Nighttime eval frame 을 adaptation pool 에 포함
arm 마다 다른 teacher / epoch / step / exposure
pose gate BLOCKED 인데 6D metric 생성
evaluation data 추가 annotation 을 자동 요구
```
