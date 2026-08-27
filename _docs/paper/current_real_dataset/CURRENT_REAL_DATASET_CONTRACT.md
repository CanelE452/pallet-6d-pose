# Current Real Dataset — Contract

작성 2026-08-27. 수치는 전부 실제 artifact 재계산이다(`DATASET_AUDIT.json`).

## 1. artifact 실제 위치

```
REVIEWED_CLEAN_REALDEV_V2
  /home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/
  REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json          (+ .sha256)
  ★repo 밖이다. 사본이 Downloads/PALLET_WINDOWS_EXPERIMENT_DATA_20260825/
    eval_positive_real128/provenance/ 에도 있다.

FT_EVAL_LEAK.json
  challenge/yolo_pose_one_model/runs_camera_facing_loss/
  ubuntu_cf_loss_queue_20260823T0930/FT_EVAL_LEAK.json

evaluator
  같은 폴더의 cf_real_eval.py  /  neg_eval_one.py

real negative
  data/pallet/raw_data/negative_real_20260823/rgb/

GT / annotation
  challenge/data/01_real/eval_canonical/{_outside_eval_manual_gt,
      capturepalletcad_manual_gt, capture0403noapril_manual_gt}/
  challenge/data/01_real/manual_gt/{capturepallet07,capturepallet09,
      capturenight08,capturenight09}_manual_gt/

annotation tool
  scripts/annotate/annotate.py (+ annotate_draw.py / annotate_pnp.py / annotate_io.py)

GT-QA 산출물
  위 REAL_GT_QA_20260821T133405Z/ 아래 REAL_GT_REVIEW_PACKET(62 files) · review_queue(1)
  ★reviewed_gt / fixed_gt / logs 는 **0 files** — 비어 있다.
```

## 2. population 표

```
population                 raw    excluded              current   DAY  NIGHT  role
──────────────────────────────────────────────────────────────────────────────────────
reviewed positive          161    21 (GT-QA)              140     112    28   DEVELOPMENT
common comparison positive 140    12 (FT overlap)         128     100    28   DEVELOPMENT
real negative                —     —                    2,689       —     —   DEVELOPMENT

population                 used for training?   used for model selection?
──────────────────────────────────────────────────────────────────────────
reviewed positive 140      No                   ★Yes (반복 사용)
common positive 128        No                   ★Yes (반복 사용)
real negative 2,689        No *                 ★Yes (반복 사용)
```

\* challenge FT 계열은 별도의 negative 259장을 학습에 썼고, 그 셋과 이 2,689 는
교집합 0 이다(memory `real-negative-set-20260823-for-ap`). 이 2,689 자체는 학습 미사용.

### GT-QA exclusion 내역 (실측)

```
RED_GT_QA_EXCLUDED   19
AMBER_EXCLUDE         2
────────────────────────
excluded_unique      21      161 - 21 = 140  ✔ manifest set_checks 와 일치
```

### 세션 구성 (140 기준, 실측)

```
set              n    directory
────────────────────────────────────────────────────────────────────────
eval_cad        18    challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt
eval_night08    12    challenge/data/01_real/manual_gt/capturenight08_manual_gt
eval_night09    16    challenge/data/01_real/manual_gt/capturenight09_manual_gt
eval_noapril    12    challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt
eval_outside    22    challenge/data/01_real/eval_canonical/_outside_eval_manual_gt
eval_pallet07   27    challenge/data/01_real/manual_gt/capturepallet07_manual_gt
eval_pallet09   33    challenge/data/01_real/manual_gt/capturepallet09_manual_gt
```

## 3. ★ 128 의 provenance

```
140  reviewed clean
-12  FT overlap (FT_EVAL_LEAK.json)
────
128  common comparison DEV     DAY 100 / NIGHT 28
```

제외된 12장은 **전부 `eval_outside`** 이고 **전부 DAY** 다(NIGHT 0). 실측으로 확인했다.

FT_EVAL_LEAK.json 의 사유를 그대로 옮긴다:

> `eval_outside` 는 별도 촬영이 아니라 capturepallet02/03/04/05/08 에서 뽑아 모은 셋 —
> 디렉토리명만 다르고 같은 프레임이다. **세션 디렉토리 비교만으로는 안 잡힌다.**

> impact: FT 의 DAY 포함 real 지표는 낙관 편향. NIGHT 지표는 영향 없음.

이 제외는 **FT/adaptation 으로 real 을 학습한 모델**에만 의미가 있다. real 을 한 장도
학습하지 않은 모델(합성 전용 arm)에게는 누수가 아니다. 그럼에도 **모집단을 섞지 않기
위해** 128 을 그대로 유지한다.

## 4. ★ evaluator 별 모집단 — 섞지 말 것

```
evaluator            positive population      negative
─────────────────────────────────────────────────────────
cf_real_eval.py      140  (leak 미제외)         —
neg_eval_one.py      128  (leak 제외)          2,689
```

`neg_eval_one.py` 는 `LEAK = FT_EVAL_LEAK.json["leaked_frame_ids"]` 를 읽어
POS 구성에서 뺀다. `cf_real_eval.py` 는 그 제외를 하지 않는다.

따라서:

- detection recall · correct-box recall · corner error → **140 모집단**
- positive confidence 분포 · AP/AUROC/FPR · matched-recall FP → **128 + 2,689 모집단**

두 수치를 한 행에 나란히 놓을 때는 어느 모집단인지 반드시 병기한다.

두 evaluator 공통 추론 recipe(코드 실측):

```
pad = 100, cv2.BORDER_REFLECT_101, imgsz 640, top-1 by box conf
cf_real_eval  conf 0.001 (threshold-free) + conf_deploy 기본 0.40
neg_eval_one  conf 0.001 (threshold-free)
```

## 5. annotation contract (실측)

라벨 JSON 구조 — 140/140 전수 동일:

```
top            camera_data, objects
camera_data    width, height, intrinsics{fx, fy, cx, cy}
objects[0]     class("pallet") · name("real_pallet") · visibility · pose_transform
               projected_cuboid(8점) · projected_cuboid_centroid
               dimensions_m{width,height,depth} · gt_source("manual") · split("eval")
               manual_kps(9점) · reproj_error_px
```

```
확인된 것
  gt_source                       "manual"  140/140
  objects 개수                    1  140/140
  해상도                          640x480  140/140
  projected_cuboid                8점 · manual_kps 9점  140/140
  camera intrinsics               존재 140/140 (fx,fy,cx,cy)
  prediction-assisted annotation  없음 — annotate.py 에 모델/torch/.pt 참조 0건
  keypoint convention             camera_dynamic_0123_v4 (KP_NAMES 실측)
  reproj_error_px                 median 1.245 · p90 2.530 · max 4.481 px

★ 확인된 것 — 주의가 필요한 항목
  per-keypoint visible/occluded flag   ★없음. `visibility` 는 object-level 스칼라이고
                                        140/140 전부 값 1 이다. 개별 코너의 가시성은
                                        라벨에 없다.
  object dimensions_m                  ★두 변종. (1.1, 0.11, 1.3) 81장 /
                                        (1.3, 0.11, 1.1) 59장 — 프레임별 W/D 스왑.
                                        ADD·translation 계열이 dims 에 민감하다.

NOT RECORDED (라벨/매니페스트에 없음 — 추측하지 않는다)
  annotator 식별자 / 인원 수
  annotation 작성·수정 타임스탬프 이력
  per-frame QC 판정 근거 (RED/AMBER 사유는 frame_id 목록만 있고 사유 텍스트 없음)
  occlusion / truncation 라벨
  capture 일시 · 조명 조건 · 카메라 설정 metadata
```

TODO: GT-QA 의 `reviewed_gt` / `fixed_gt` / `logs` 폴더가 **비어 있다**. 리뷰 과정의
중간 산출물이 남아 있지 않아 "어떤 기준으로 21장을 뺐는가" 를 문서에서 재구성할 수 없다.
`REAL_GT_REVIEW_PACKET`(62 files)에 무엇이 있는지 별도 확인이 필요하다.
