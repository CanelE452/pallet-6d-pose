# 정확도 실패 계층 분해 — 재사용 가능한 artifact · 모듈 인벤토리 (v1)

작성 2026-09-06 · HEAD `2e5ec0e` · 이 문서는 **읽기 전용 조사** 결과다 (학습·추론 0, 코드 수정 0).

목적: detection / candidate selection / keypoint role / keypoint localisation / PnP 증폭
다섯 계층으로 실패를 분해하기 전에, **재추론 없이 쓸 수 있는 것**과 **없어서 새로 만들어야 하는 것**을
파일·스키마 수준으로 확정한다.

모든 문장에 `[확인]`(파일을 실제로 열어 값을 봄) / `[추정]`(이름·주석·문서에서 추론) 태그를 붙인다.

---

## 요약 블록

| 항목 | 값 |
|---|---|
| MULTI_CANDIDATE_PREDICTION_CACHE_AVAILABLE | **YES** — `challenge/yolo_pose_one_model/analysis_pre_v2/_cc_raw_dump.json` (주), `challenge/yolo_pose_one_model/analysis_pre_v2/_d1d2_raw.json` (부, top-5 절단) [확인] |
| MULTITEACHER_PER_FRAME_PREDICTIONS_AVAILABLE | **YES** — `data/pallet/results/multiteacher_corner_distill_v1/predictions/{T0..T6,C0,C1}.json`, 교사 7 + 특화 2 = 9 arm × 319 프레임. 단 **arm 당 top-1 1세트만** [확인] |
| REUSABLE_POSE_EVALUATOR | **트랙 B(최신·권장)** `scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py` : `symmetry_aware_add_m` / `pose_auc` / `rotation_error_degrees` / `yaw_error_degrees` / `translation_components_m` / `model_diameter_m` / `cuboid_model_points`, 묶음은 `pose_evaluation_paths.py:score_pose_against_gt` · `evaluate_frame`. **트랙 A(게이트형)** `challenge/evaluation_v2/pose_metrics.py` : `add_error_m` / `pose_auc` / `summarize_pose_errors` / `build_pose_metric_gate`. **공통 단일 구현** `challenge/evaluation_v2/oriented_iou3d.py:oriented_iou_3d`. 2D AP 는 `challenge/evaluation_v2/paper_real_eval.py:_average_precision_at_iou` / `evaluate_2d_with_predictor`. 진단 계열은 `scripts/stage0/real_eval/re_metrics.py` [확인] |
| REUSABLE_PNP_WRAPPER | 정본 = `scripts/paper/pose_metric_closure_v1/pose_evaluation_paths.py:predict_pose_without_gt` (내부 `challenge/evaluation_v2/pnp_selector.py:select_pnp_hypotheses`) 와 `:predict_pose_with_oracle_axis`. 저수준은 `cv2.SOLVEPNP_SQPNP` + `cv2.solvePnPRefineLM` 을 `run_pose_evaluation.py:solve` 가 감쌈. 어노테이션 계열은 `scripts/annotate/annotate_pnp.py:solve_pose` / `solve_pose_candidates` [확인] |
| REUSABLE_GEOMETRY_REGISTRY | `challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json` — sha256 `0c7a10729b6db18cbe47fa4adb158e2f26ec7a7c9458f59ee60d023c282f0627` (실측). 파생 = `challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json` (실측 sha256 `e923b44880b031a7d3a9e2fffb5a6bd287cfa0de758133eeb5a73770137eba86`). pose 평가 전용 추가 계약 = `data/pallet/results/paper_pose_metric_closure_v1/POSE_EVAL_OBJECT_CONTRACT.json` (sha256 `a4c2918b4b0e9c97f94332d2e7e35132a8cbe0e738db25d92ea55e0d81210dbd`, 파일 안에 기록). **셋 다 편집 금지** [확인] |
| RE_INFERENCE_REQUIRED_FOR_CANDIDATE_ORACLE | **NO (조건부)** — `yolo26n_paper_generic_v1_seed42/weights/last.pt` 한 모델에 대해서는 재추론 0 으로 후보 oracle 계산 가능하고, 이미 계산된 결과도 있다(`_rr_cands.json` / `RERANK_ORACLE.json`). **그 외 모든 체크포인트(R0~R5, T0~T6, C0/C1, G38 계열, DOPE)는 top-1 만 저장돼 있어 재추론이 필요하다.** [확인] |

### 요약 블록의 단서 조항 (읽지 않고 인용하지 말 것)

- `_cc_raw_dump.json` / `_d1d2_raw.json` 은 **`.gitignore` 246·247 줄에 등재된 untracked 파일**이다 [확인].
  로컬 디스크에만 있고 새로 clone 하면 없다. 분해 작업의 입력으로 쓸 거면 먼저 백업하거나
  추적 대상으로 옮길지 결정해야 한다.
- 이 두 덤프의 population 은 **GT QA 이전의 161 장**이다. 현재 정본 manifest 는 140 장이고,
  차집합 21 장은 `challenge/real_gt_v2/INVALID_GT_QUARANTINE.json` (entries 23) 로 격리된
  `RED_GT_QA_EXCLUDED` 프레임이다 — 실측으로 `_cc_raw_dump.json` positive 161 중
  현재 manifest 에 없는 것이 정확히 21 장이고, 반대 방향 차집합은 0 이다 [확인].
  **모델 예측 자체는 유효**하지만, `_rr_cands.json` 의 GT 파생 스칼라(iou / R / t / s5 / corner)는
  그 21 장에서 격리된 GT 를 기준으로 계산된 값이다.
- 후보 oracle 을 **다른 체크포인트로** 확장하려면 재추론이 필요하다. 필요한 변경은 작다:
  `scripts/paper/pose_metric_closure_v1/run_frozen_arm_inference.py` 84~99 줄이
  `best = int(np.argmax(scores))` 로 top-1 만 남기므로 [확인], 전 후보를 남기는 덤프는
  `challenge/yolo_pose_one_model/analysis_pre_v2/cc_dump.py:boxes_of()` 방식을 그대로 쓰면 된다 [확인].
  (이 문서는 조사만 하므로 코드는 건드리지 않았다.)

---

## 0. population 지도 — 세 개가 섞여 있다

계층 분해에서 수치를 합치기 전에 반드시 구분해야 한다 [확인].

| population id | N | 구성 | 어디에 쓰였나 |
|---|---|---|---|
| `PAPER_EVAL_ALL_POS` | 319 (plastic 194 + wood 125) | 13 세션: `wood_night_01` 56, `plastic_day_01` 44, `eval_pallet09` 33, `eval_pallet07` 27, `wood_183705` 25, `wood_day_01` 24, `plastic_night_01` 22, `wood_184309` 20, `eval_cad` 18, `eval_night09` 16, `eval_night08` 12, `eval_noapril` 12, `eval_outside` 10 | `paper_eval_v1~v5`, `paper_pose_metric_closure_v1`, `multiteacher_corner_distill_v1` |
| `DEV_NEG2689` | 2,689 | negative only (session 필드 공란) | 위와 같은 CSV 의 NEGATIVE 행 |
| 구 canonical 161 / 현 140 | 161 → 140 | `eval_pallet09` 36, `eval_pallet07` 27, `eval_night09` 25, `eval_outside` 22, `eval_cad` 22, `eval_night08` 17, `eval_noapril` 12 (161 기준) | `analysis_pre_v2`, `model_compare`, `audit_20260821T*` |

- 319 population 의 manifest = `challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json` (items 319) [확인].
  세션 이름이 `wood_*` / `plastic_*` 로 시작하는 항목은 `data/evaluation/pallet_eval_v1/` 아래에 있고,
  `eval_*` 로 시작하는 항목은 `challenge/data/01_real/` 계열이다 [확인].
- 161/140 population 의 GT 소스 = `challenge/yolo_pose_one_model/paper_generic_pipeline/eval_manifest.json`
  (`manifest: PAPER_YOLO_EVAL_DEV_140_GT_QA_CLEAN`, `n_total: 140`,
  `gt_qa.raw_total_before_quarantine: 161`, `gt_qa.excluded: 21`) [확인].
- 두 population 은 **세션 구성이 다르다** (319 에는 wood_* / plastic_* 가 있고 161 에는 없다).
  한 표에 섞으면 안 된다 [확인].

---

## 1. ★ 다중 후보 예측 캐시 — YES

### 1-1. `challenge/yolo_pose_one_model/analysis_pre_v2/_cc_raw_dump.json` (2.4 MB, untracked)

전 후보를 절단 없이 남긴 유일한 덤프다 [확인].

- 모델: `challenge/yolo_pose_one_model/runs_paper/yolo26n_paper_generic_v1_seed42/weights/last.pt`
  (파일 안 `weights` 필드. SHA 는 기록 없음) [확인]
- recipe: `{"pad": 100, "imgsz": 640, "dump_conf": 0.001, "border": "BORDER_REFLECT_101",
  "selection": "top-1 by box conf among survivors"}` [확인].
  `selection` 문자열은 **오해를 부른다** — 실제 저장 배열은 전 후보다.
  생성기 `analysis_pre_v2/cc_dump.py:boxes_of()` 가 `for i in range(len(conf))` 로 전수를 담고
  conf 내림차순 정렬만 한다 [확인].
- population: `positive` 161 (구 canonical), `negative` 259 (`forklift_raw_20260528_163408` 시퀀스) [확인]
- 후보 수: positive 총 **865** detection (프레임당 1~19, 분포 `1:27, 2:19, 3:20, 4:15, 5:14, 6:21, 7:8, 8:7, 9:4, 10:4, 11:6, 12:4, 13:4, 14:2, 15:1, 16:2, 18:2, 19:1`),
  negative 총 **1,427** (0 후보 33 프레임 포함) [확인]
- 실측 스키마

```
{ "weights": str,
  "recipe": {"pad","imgsz","dump_conf","border","selection"},
  "positive": [ {"fid": str, "set": str, "population": str,
                 "boxes": [ {"conf": float,
                             "xyxy": [x1,y1,x2,y2],
                             "kps": [[x,y] x 9],          # 8 코너 + centroid
                             "kp_conf": [float x 9]} ] } ],   # conf 내림차순
  "negative": [ {"frame": str, "boxes": [...동일...] } ] }
```

- 좌표계: **원본 이미지 픽셀** [확인]. `cc_dump.py` 가 `xyxy - PAD`, `kps - PAD` 로 pad 100 을 되뺀다.
  실측 범위 x ∈ [-100.0, 740.0], y ∈ [-99.90, 580.0] — 640×480 원본에 reflect pad 100 을 얹은
  좌표계 그대로다(음수 = 프레임 밖).
- 신뢰도 필드: box `conf`, keypoint `kp_conf` 9개 모두 있음 [확인]
- GT / 오차 필드: 없음. `fid` 로 `eval_manifest.json` 과 조인해야 한다 [확인]

### 1-2. `challenge/yolo_pose_one_model/analysis_pre_v2/_d1d2_raw.json` (2.0 MB, untracked)

해상도 축(640/960/1280)이 붙은 다중 후보 덤프. **conf 내림차순 상위 5개로 절단** [확인].

- 모델: 1-1 과 동일 체크포인트 [확인]
- `confs: [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]`, `sizes: [640, 960, 1280]`, `pad: 100` [확인]
- population: `frames` = dict, key = frame_id, **161** 개. `sealed` 필드로 105(True) / 56(False) 구분 [확인]
- 후보 총수: 640 → 585, 960 → 751, 1280 → 793 [확인]
- 실측 스키마

```
frames[fid] = { "set": str, "sealed": bool,
                "gt_bbox": [x1,y1,x2,y2],                       # GT 있음
                "runs": { "640"|"960"|"1280": {
                    "n_candidates_at_min_conf": int,
                    "candidates": [ {"box_conf": float, "bbox": [4],
                                     "iou_with_gt": float,       # 오차 필드 있음
                                     "kp_conf_mean": float,
                                     "kps": [[x,y] x 8]} ],      # ★ 8개, centroid 없음
                    "n_above": {conf: n, ...} } } }
```

- 좌표계: 원본 픽셀 (x ∈ [-100.0, 740.0]) [확인]

### 1-3. 이미 계산된 후보 oracle — `_rr_cands.json` · `RERANK_ORACLE.json` · `_rr_detail.json`

`_cc_raw_dump.json` 을 재추론 0 으로 소비해 후보별 GT 진단치를 붙여 둔 결과가 이미 있다 [확인].
생성기 = `challenge/yolo_pose_one_model/analysis_pre_v2/rr_oracle.py`.

- `_rr_cands.json` (git 추적됨): list 161, 각 원소
  `{"fid","set","population","cands":[{"rank","conf","iou","R","t","s5","corner","reproj",
  "depth_ok","cuboid_ok","kp_conf_mean","kp_conf_min","box_area","box_diag"}]}`,
  후보 총 **865** (= `_cc_raw_dump.json` positive 총수와 일치) [확인].
  `R`·`t` 는 후보별 PnP 결과 대비 GT 오차, `corner` 는 8 코너 median px 오차,
  `iou` 는 GT bbox 와의 IoU. **좌표는 없다** — 좌표가 필요하면 `_cc_raw_dump.json` 과 `rank` 로 조인.
- `RERANK_ORACLE.json`: `phase1_topk_oracle` 이 K=1,2,3,5 에서
  `availability / correct_recall / R_median / R_p90 / t_median / t_p90 /
  success_5cm5_oracleR / success_5cm5_any` 를 담는다.
  실측 K=1 correct_recall 0.7081 → K=5 로 갈수록 상승, `success_5cm5_any` 는 0.3043 → 0.3106 [확인].
- `_rr_detail.json`: 프레임 단위 실패 분류(§7 참조).

**결론**: `yolo26n_paper_generic_v1` 한 모델에 대한 candidate-selection 계층 분해는
**추론 0 · 새 evaluator 0** 으로 즉시 가능하다 [확인].

---

## 2. top-1 만 저장된 예측 캐시 (좌표 있음)

| path | model / checkpoint | population (N, 세션) | top1/multi | 실측 필드 | 좌표계 | 신뢰도 |
|---|---|---|---|---|---|---|
| `data/pallet/results/paper_pose_metric_closure_v1/predictions/{R0,R0_CONT,R1_NAIVE,R2_CONF,R3_CONF_REPROJ,R4_CONF_REMOVE,R5_PROPOSED,A8_DAY_ONLY}.json` | arm 별 checkpoint + sha256 (파일 안 `checkpoint`,`checkpoint_sha256`; 전체 목록은 `POSE_ARM_CHECKPOINT_LOCK.json`) | `PAPER_EVAL_ALL_POS` 319 | **top1** (`argmax(box conf)`) | `schema_version:"frozen_arm_prediction_v1"`, `recipe`, `recipe_lock_sha256`, `population_frame_order_sha256`, `n_frames`, `no_detection`, `frames[fid] = {status, box_xyxy[4], box_conf, keypoints_xy[9][2], keypoints_conf[9]}` | 원본 픽셀 (pad 100 을 뺀 값) | `box_conf`, `keypoints_conf` |
| `data/pallet/results/multiteacher_corner_distill_v1/predictions/T0_R0_YOLO26N_G38LEGACY.json` 외 T1~T6 | 교사 7종, 각 파일에 `checkpoint` + `checkpoint_sha256` + `architecture` | 같은 319, `population_frame_order_sha256` = `72f83f6f0682209beec6134d8ab935ff1f56348ba816691133882a9d38d8c15f` (closure 계열과 동일) | **top1** | 위와 동일 + `detections`(후보 **개수**), `box_source` | 원본 픽셀 | 있음 |
| 같은 폴더 `C0.json`, `C1.json` | `specialist_C0/C1_last.pt` + sha256, `base_arm: T0_R0_YOLO26N_G38LEGACY` | 같은 319 | top1 | 위 + `refined_by` | 원본 픽셀 | 있음 |
| `data/pallet/results/paper_eval_v1/baselines/DOPE_R0_PREDICTIONS.json` | DOPE `weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth`, sha256 `0de80490...` | 3,008 프레임 키(=319 positive + 2,689 negative), `n_detected: 407` | 형식은 리스트지만 실측 **길이 max 1** (1:407, 0:2601) → 사실상 top1 | `schema_version:"paper_cached_predictions_v1"`, `frames[image_path] = [{score, box_xyxy, keypoints_xy(null 허용), n_detected_corners}]` | 원본 픽셀 | `score` (belief peak, YOLO box conf 와 스케일 다름 — 파일 안에 명시) |
| `data/pallet/results/model_compare/kps_*.json` (15개) | 모델별 `weights` 문자열 (sha 없음). 예: `kps_yolo26n_ft`, `kps_yolo26m_ft`, `kps_yolo26n_paper_generic_v1`, `kps_FINAL40K_seed1*`, `kps_LV3_*`, `kps_Y0E`, `kps_YN`, `kps_G38_ONLY_60EP`, `kps_JOINT_G38_LEGACY_TEX`, `kps_yolo26n_synth`, `kps_yolo26n_broad40k_5ep` | 140 또는 161 (파일마다 `n_frames`) | top1 | `{model, weights, recipe, n_frames, n_detected, frames:[{set, sealed, fid, image, kps[9][2], kp_conf[9], box_conf}]}`; FINAL40K 계열만 추가로 `kps_argmax, score_4kp, n_det, line_theta, line_rho` | 원본 픽셀 | `box_conf`, `kp_conf` |
| `data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json` | `.../YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt`, sha256 `970a0913...` | `entries` 1,000 (unlabeled pool, `capture_session` 필드) | top1 + **flip top1** | `schema_version:"paper_teacher_prediction_cache_v1"`, `entries[i] = {image_path, image_sha256, capture_session, paper_condition, image_width, image_height, camera_matrix[3][3], n_instances, n_flip_instances, top1{box_xyxy, box_conf, keypoints_xy, keypoints_conf}, flip_top1{...}}` | 원본 픽셀 | 있음 |
| `data/pallet/results/paper_fast_teacher_v1/R0_TTA960_CACHE.json` | 파일 안에 checkpoint 기록 없음. 폴더명으로 R0 계열 TTA 960 [추정] | 319 (키 = `eval_cad:...` 형식, 13 세션 분포는 319 population 과 동일) | top1 + flip_top1 | `{fid: {image_width, image_height, top1{keypoints_xy, keypoints_conf, box_conf}, flip_top1{...}}}` | 원본 픽셀 | 있음 |
| `data/pallet/results/paper_eval_v1/visual_audit/AUDIT_PREDICTIONS.json` | R0 · R5 두 arm | **99** 프레임 (319 의 육안검수 서브셋; 11 세션) [확인] | top1 | `{frame_id: {image_path, R0{keypoints[9][2], valid[9], box_conf}, R5{...}}}` | 원본 픽셀 | `box_conf` |
| `data/pallet/results/paper_eval_v1/visual_audit/MIRROR_PREDICTIONS.json` | R0 · R2_CONF · R5_PROPOSED | **957** = 3 arm × 319, 키 = `"{arm}|{frame_id}"` [확인] | top1 | `{keypoints[9][2], box_conf}` | 원본 픽셀 | `box_conf` |
| `challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_a_oracle/PREDICTION_CACHE.npz` | `.../OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt`, sha256 `1a806ca497fde5175334e908540ebae22a45deeb08862f014ac8460f8d43ac3c` (metadata_json 안) | 185 = `DEV_PLASTIC_POS140` 140 + `DEV_WOOD_POS45` 45 | top1, **후보 개수만** `detection_counts` (0~15) | `frame_ids, object_types, images, labels, sessions, domains, detection_counts, confidences, boxes_xyxy(185,4), keypoints_xy(185,9,2), camera_intrinsics(185,3,3), metadata_json` | 원본 픽셀 (max 1264) | `confidences` |
| `challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_c_probe/FEATURE_CACHE.npz` | 동일 sha `1a806ca4...` | real 185 + synth 40,000 | top1 (파일 안 `candidate_scope` 가 하위 후보 폐기를 명시) | `real_keypoints_xy(185,9,2)`, `real_boxes_xyxy`, `real_detection_counts`, `synth_keypoints_xy(40000,9,2)`, `synth_*` | 원본 픽셀 | — |
| `data/pallet/eval_results/dope_vs_paper/pred_yolo.json`, `pred_dope.json` | `weights` 문자열 | `fids` 4 장 (스팟 체크) | top1 | `{weights, fids, pred:{fid:{sess, pred8[8][2], pred_c, gt8, ...}}}` | 원본 픽셀 | [추정] |

**공통 recipe** (여러 파일에 같은 값이 기록돼 있다 [확인]):
`imgsz=640`, `pad_px=100`, `border=BORDER_REFLECT_101`,
`confidence_floor=0.001`, `top_detection_rule="highest box confidence per frame"`,
`unpad_rule="subtract pad from box and keypoint coordinates after inference"`.
정본 출처 = `data/pallet/results/paper_pose_metric_closure_v1/INFERENCE_REPLAY_LOCK.json`
(`schema_version: inference_replay_lock_v1`, `status: FROZEN`).

**패리티 증거**: `INFERENCE_PARITY_R0.json` 에서 `paper_eval_v1/arms/R0_per_frame.csv` 캐시와
새로 replay 한 `predictions/R0.json` 이 319/319 detection 일치,
box 좌표 delta median·max 모두 0.0, keypoint 오차 delta max 5e-7 px 로 재현됐다 [확인].
→ **같은 recipe 로 재추론하면 기존 캐시와 비트 수준에 가깝게 일치**한다는 뜻이므로,
새 arm 을 추가해도 기존 캐시와 섞어 비교할 수 있다.

---

## 3. 후보 개수만 남고 좌표는 버려진 artifact (candidate oracle 불가)

계층 분해에서 "후보가 몇 개 있었나"는 알 수 있지만 "2등 후보가 정답이었나"는 알 수 없는 파일들 [확인].

| path | 개수 필드 | population |
|---|---|---|
| `data/pallet/results/paper_eval_v{1,2,3,5}/arms/*_per_frame.csv` | `candidate_count` | 3,008 행 = `PAPER_EVAL_ALL_POS` 319 + `DEV_NEG2689` 2,689 |
| `data/pallet/results/multiteacher_corner_distill_v1/predictions/T*.json`, `C*.json` | `detections` | 319 |
| `challenge/evaluation_v2/dev_results/YOLO26_G38_PER_FRAME.csv` | `candidate_count` | 2,817 = `COMMON_DEV_POS128` 128 + `DEV_NEG2689` 2,689 |
| `challenge/evaluation_v2/selector_diagnostic/SELECTOR_DIAGNOSTIC.json`, `SELECTOR_PER_FRAME.csv` | `detection_count` | `DEV_POS140` 140 |
| `challenge/yolo_pose_one_model/dimension_conditioning_probe/phase_a_oracle/{PREDICTION_CACHE.npz, ORACLE_PER_FRAME.csv}` | `detection_counts` / `detection_count` | 185 / 700 (5 arm × 140) |

`paper_eval_v*/arms/*_per_frame.csv` 실측 헤더 (22 컬럼) [확인]:

```
population_id, role, kind, frame_id, object_type, session_id, source_set, domain, image,
candidate_count, top_score, top_box_x1, top_box_y1, top_box_x2, top_box_y2,
top_target_iou, top_iou50_match, top_keypoints_shape_valid, supervised_keypoint_count,
top_keypoint_supervised_errors_px, top_keypoint_supervised_error_median_px,
top_keypoint_all_annotated_errors_px
```

- `top_keypoint_supervised_errors_px` 는 `;` 로 구분된 가변 길이 리스트다 (visibility 감독 마스크 통과분만).
  `top_keypoint_all_annotated_errors_px` 는 annotated 전부. 두 통계는 **정의가 다르므로 섞지 말 것** —
  생성 코드 `challenge/evaluation_v2/paper_real_eval.py` 의 `PER_FRAME_FIELDS` 주석이 이를 명시한다 [확인].
- 실측 `candidate_count` 분포(3,008 행): `0:1150, 1:733, 2:416, 3:242, 4:166, 5:96, 6:79, 7:55, 8:24, 9:22, ...` [확인]
- 이 CSV 에는 **keypoint 좌표가 없다**. 좌표는 `paper_pose_metric_closure_v1/predictions/<ARM>.json` 에 있다 [확인].

---

## 4. 재사용할 evaluator

### 4-0. ★ pose metric 트랙이 **둘** 있다 — 섞으면 조용히 틀린다

| 트랙 | 위치 | 성격 |
|---|---|---|
| **A. 게이트형** | `challenge/evaluation_v2/pose_metrics.py` (+ `paper_real_eval.py`, `real_dataset_contract.py`) | manifest·canonical migration·symmetry·selector 4중 게이트를 통과해야만 숫자를 뱉는다. 2D(AP·keypoint px)와 pose 를 한 CLI 로 함께 낸다 |
| **B. pose metric closure v1 (최신)** | `scripts/paper/pose_metric_closure_v1/` | `S = {I, Ry(180)}` 를 함수 내부에 박아 둔 대칭 인식 metric. IoU3D · lateral/depth 분해 · MAIN/DIAGNOSTIC/ORACLE 3경로 |

B 는 A 를 지우지 않고 **A 의 부품을 재사용한다** — `run_pose_evaluation.py` 가
`challenge.evaluation_v2.oriented_iou3d.oriented_iou_3d` 와
`challenge.evaluation_v2.pnp_selector.select_pnp_hypotheses` 를 import 한다 [확인].
→ `oriented_iou3d` 와 `pnp_selector` 는 **양 트랙 공통 단일 구현**이고,
ADD/AUC/rotation/yaw 만 트랙별로 정의가 다르다.
논문 최신 표(`PAPER_CANONICAL_NUMBER_SOURCES.json`)는 **B** 를 가리킨다 [확인].

★ 함정 3가지 (전부 실제 코드로 확인) [확인]:

1. **`pose_auc` 시그니처가 다르다.**
   A `pose_auc(normalized_errors, *, max_fraction=0.1)` — **이미 diameter 로 나눈** 값을 받는다.
   B `pose_auc(errors_m, diameter_m, *, max_fraction=0.1, points=1001)` — **raw 미터 + diameter 별도**.
   잘못 넘기면 예외 없이 틀린 값이 나온다.
2. **`rotation_error_degrees` / `yaw_error_degrees` 가 두 모듈에 같은 이름으로 있고 의미가 다르다.**
   A 는 대칭 처리를 하지 않는다(호출부 `paper_real_eval.py` 가 `equivalent_rotations` 로 min 을 취한다), yaw 범위 0..180.
   B 는 함수 안에서 S 에 대해 min 을 취하고, yaw 를 **0..90 으로 접는다**.
3. **A 의 `adds_error_m` 은 논문 ADD-S 가 아니다.** docstring 원문:
   "Unrestricted nearest-neighbour distance (not paper symmetry policy). The paper evaluator does not
   call this helper: on a cuboid keypoint set it could silently grant pitch/roll symmetries."
   논문 ADD-S = frozen symmetry contract 의 proper rotation 위 **대응점 ADD 최소값**
   (= B 의 `symmetry_aware_add_m`, 또는 A 트랙에서는 `paper_real_eval` 호출부의 min 루프).

### 4-1. pose (ADD/ADD-S AUC, 회전·병진, yaw) — 트랙 B (권장)

`scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py` [확인]

| 함수 | 입력 계약 | 출력 |
|---|---|---|
| `symmetry_aware_add_m(model_points, R_pred, t_pred, R_gt, t_gt)` | `model_points` (N,3) m 단위 object frame, R 은 (3,3) proper rotation, t 는 (3,) m | float, `min over S of mean_i ||T_pred X_i − T_gt S X_i||`. 대칭군 `S = {I, Ry(180)}` 를 **내부에서** 처리 |
| `pose_auc(errors_m, diameter_m, *, ...)` | 오차 리스트(m), 모델 지름(m) | [0,1] 정규화 AUC. τ ∈ [0, 0.1·diameter], 적분점 `AUC_INTEGRATION_POINTS = 1001`, `AUC_MAX_FRACTION = 0.1` (상수 동결) |
| `rotation_error_degrees(R_pred, R_gt)` | (3,3) ×2 | 대칭군 최소 geodesic (deg) |
| `yaw_error_degrees(R_pred, R_gt)` | (3,3) ×2 | mod 180 후 0~90 으로 접은 yaw 오차 |
| `translation_error_m` / `translation_components_m(t_pred, t_gt)` | (3,) ×2 | `{total_m, lateral_m, depth_m}` |
| `model_diameter_m(model_points)` | (N,3) | 최대 점간 거리(m) |
| `cuboid_model_points(extents)` | `(across, height, along)` m | (8,3) 코너 |

★ 대칭 규약: **180도는 흡수, 90도는 흡수하지 않는다.** 모듈 docstring 에 명시돼 있고
`SYMMETRY_GROUP = (I, diag(-1,1,-1))` 로 하드코딩돼 있다 [확인].
"long/short 축을 잘못 고르면 약 90도 오차로 남아야 한다"는 것이 설계 의도다.

### 4-1b. pose — 트랙 A (게이트형)

`challenge/evaluation_v2/pose_metrics.py` 공개 API [확인]:
`build_pose_metric_gate`, `blocked_pose_metrics`, `rotation_error_degrees`, `yaw_error_degrees`,
`translation_error_m`, `transformed_model_points`, `add_error_m`, `adds_error_m`,
`model_diameter_m`, `pose_auc`, `summarize_pose_errors`, `summarize_multishape_pose_errors`.
데이터클래스 `PoseMetricGate`, `PoseErrorRecord`. 상수 `POSE_METRIC_FIELDS`.

- 입력은 전부 **R (3,3) + t (3,) 분리 — 4×4 pose 를 받지 않는다** [확인].
- `build_pose_metric_gate(*, canonical_migration_status, selector_report, symmetry_status, final_manifest_frozen)`
  네 조건이 모두 통과해야 숫자가 나온다. 실패하면 `summarize_pose_errors` 가 records 를
  순회하기 전에 BLOCKED 를 반환한다 (의도적 fail-closed) [확인].
- 계층 분해 같은 **진단 용도에는 이 게이트가 방해가 된다**. 게이트 없이 쓰려면 트랙 B 를 쓸 것.

### 4-2. 3D IoU — 양 트랙 공통 단일 구현

`challenge/evaluation_v2/oriented_iou3d.py:oriented_iou_3d(rotation_a, translation_a, extents_a,
rotation_b, translation_b, extents_b) -> float` [확인]

- 입력 계약: R 은 (3,3) 직교(atol 1e-6; **det 는 검사하지 않는다**), t 는 **박스 중심** (3,) m,
  `extents` 는 **full extents 3개(half-extent 아님), 전부 양수** [확인].
- 12 반평면 교차의 볼록 껍질 부피. 샘플링·복셀·축정렬 근사 없음(docstring 이 `metric_split_lock.md` §2.3 인용).
  SciPy `ConvexHull` 필수 — 없으면 `RuntimeError` [확인].
- public 보조: `intersection_volume(...)`, `box_volume(extents)` [확인].
- **복제 금지** — 저장소 유일 구현이다.

### 4-3. 2D detection AP / keypoint 오차

`challenge/evaluation_v2/paper_real_eval.py` [확인]
- `evaluate_2d_with_predictor(pair, predictor) -> dict` — **공개 진입점**.
  `predictor.predict(image_path) -> list[(score, box_xyxy(4,), keypoints_xy(9,2)|None)]` 라는
  duck-typing 인터페이스만 만족하면 임의 모델을 꽂을 수 있다. 좌표는 **이미 원본 픽셀계**여야 한다.
- `_average_precision_at_iou(candidates, n_positive, iou_threshold)` — COCO 방식 101-point
  interpolated AP, 프레임당 1 매칭. IoU 0.50:0.05:0.95 (10개)로
  `box_ap50_95`, `box_ap50`, `box_ap_by_iou` 를 만든다.
- `_evaluate_2d_collected(pair, targets, candidates, top_by_frame)` — top-1 매칭(IoU≥0.5) 후
  keypoint 픽셀 오차 분포를 `keypoint_all_labeled` / `keypoint_all_annotated_unknown_visibility` /
  `keypoint_visibility_1` / `keypoint_visibility_2` 로 분리해 낸다.
- `_per_frame_rows(...)` — §3 의 CSV 를 만든다.
- `evaluate_pose_records(targets, top_by_frame, pose_context)` — GT 는 `PositiveTarget`
  (내부에 4×4 `canonical_pose_transform` 또는 `canonical_pose_candidate_transforms`,
  `camera_intrinsics`, `physical_dimensions`), 예측은 **(9,2) finite keypoints** 필요.
  shape 불일치면 전 필드 +inf 인 실패 레코드가 된다 [확인].
- `_UltralyticsPredictor.predict` + 상수 `INFERENCE_PAD = 100`, `INFERENCE_IMGSZ = 640`,
  `INFERENCE_CONFIDENCE_FLOOR = 0.001` — 트랙 A 의 추론 recipe 정본 [확인].

**PCK 는 frozen contract 2.2 계층에 이름이 올라 있으나(`METRIC_NAMING_LOCK.md`), 두 정본 evaluator
트랙 어디에도 구현이 없다** [확인]. 저장소에서 살아 있는 유일한 구현은
`scripts/data_prep/eval/evaluate_on_val.py:254 compute_pck(pred_kps, gt_kps, threshold_px=10) -> (correct, total)`
(음수 좌표는 skip). 정본 evaluator 는 PCK 대신 keypoint median/p90 px 를 쓴다.

### 4-4. 별도 계열 — `scripts/stage0/real_eval/re_metrics.py`

`analysis_pre_v2` · `model_compare` · `audit_20260821T*` 가 쓰는 evaluator [확인].
공개 함수: `add`, `add_s`, `pose_error`, `success_5cm5deg`, `model_diameter`, `pose_auc`,
`yaw_error`, `box_corners`, `box_halfspaces`, `intersection_volume`, `iou_3d`,
`precision_recall`, `average_precision`.

★ **정본 판정**: 논문 수치의 출처는 `_docs/paper/final/PAPER_CANONICAL_NUMBER_SOURCES.json` 이고,
거기에 등재된 pose 파일은 전부 `paper_pose_metric_closure_v1/POSE_EVALUATION_*.json` 계열이다 [확인].
따라서 **논문에 들어갈 pose 수치는 4-1/4-2, 진단·후보 분석은 4-4** 로 쓰는 것이 기존 관례와 일치한다.
두 계열은 대칭 처리가 다르다 — `re_metrics.success_5cm5deg` 는 5cm5° 이고,
현재 지표 규약 `_docs/paper/final/METRIC_NAMING_LOCK.md` 는 `metric_split_lock.md` §2 의 4계층을
선언하고 "`px` 는 2D localisation 지표이지 6D pose 지표가 아니다" 를 못 박는다 [확인].
★ 같은 문서가 pose 계층(2.3)을 `BLOCKED` 로 적어 두었으나, 이후
`PAPER_CANONICAL_SYNC_20260904` 가 이를 REPORTABLE 로 동기화했다 — **두 문서가 어긋나 있다.**
pose 수치를 인용할 때 어느 쪽을 따를지 먼저 확정할 것.
**두 계열의 수치를 한 표에 섞지 말 것.**

### 4-5. legacy — 쓰지 말 것 [확인]

- `scripts/self_training/metrics.py` : `compute_ADD`, `compute_ADD_S`(무제약 NN — 정사각 footprint 에서
  90도 swap 을 0 으로 만든다), `compute_auc`(thresholds 100점 + strict `<` — 정본은 1001점 + `<=`),
  `compute_5cm_5deg`, `compute_reproj_error`, `class PoseEvaluator`.
  `metric_split_lock.md` 가 5cm5°/10cm10° 를 보고 지표에서 내리고 AUC·AP 를 올렸다.
- `Deep_Object_Pose/evaluate/add_compute.py` — visii 렌더러 의존, 순수 함수 아님, 재사용 불가.
  `evaluate.py` 는 `def`/`class` 가 하나도 없는 스크립트, `kpd_compute.py` 는 `get_all_entries` 하나뿐.
  `utils_eval.py:calculate_auc*` 만 순수 numpy 지만 정본 아님.

### 4-6. 새 evaluator 복제가 필요한가 — 필요 없다

- ADD/ADD-S AUC · 3D IoU · yaw · 회전 · 병진 성분 · AP · keypoint px 오차 — 전부 위 모듈에 있다 [확인].
- PCK 만 정본 트랙에 없으나, `paper_real_eval._distribution` 이 뱉는 픽셀 오차 리스트에
  임계값을 걸면 그대로 나온다 — **새 evaluator 가 아니라 집계 한 줄**이다.
- 계층 분해에 필요한 새 코드는 evaluator 가 아니라 **조인 로직**(후보 ↔ GT ↔ 실패 라벨)뿐이다.
- import 난이도: `symmetry_aware_pose_metrics.py` 는 **numpy 만 의존**하고 게이트가 없어 가장 쉽다.
  단 패키지가 아니라 스크립트 디렉터리이므로 `sys.path` 에
  `scripts/paper/pose_metric_closure_v1` 를 넣어야 한다 (`run_pose_evaluation.py` 가 그렇게 한다) [확인].

---

## 5. PnP 래퍼 · geometry · intrinsics

### 5-1. PnP 경로 (세 갈래를 절대 섞지 않는 설계)

`scripts/paper/pose_metric_closure_v1/pose_evaluation_paths.py` [확인]

| 함수 | 입력 | 출력 | 성격 |
|---|---|---|---|
| `predict_pose_without_gt(predicted_keypoints, camera_intrinsics, physical_long_m, physical_short_m, physical_height_m, selector_config=None)` | 예측 9점 (앞 8개만 사용), K (3,3), 물리 치수 3개(m) | `{"mode":"main", "is_oracle":False, "selector_result":PnPSelectionResult, "gt_consulted":False}` | **MAIN — 배포 가능**. GT 파라미터가 시그니처에 없다(계약 테스트로 강제) |
| `predict_pose_with_oracle_axis(predicted_keypoints, camera_intrinsics, reviewed_long_axis, long_m, short_m, height_m)` | 위 + 사람이 검수한 long axis (`CF_WIDTH`/`CF_DEPTH`) | `{"mode":"oracle","is_oracle":True,"status","rotation"(3,3),"translation"(3,)}` | **ORACLE — 상한**. `SOLVEPNP_SQPNP` → `solvePnPRefineLM`, 유효 점 6개 미만이면 `POSE_UNRESOLVED` |
| `score_pose_against_gt(model_points, R_pred, t_pred, R_gt, t_gt, extents)` | 위 결과 + GT | `{rotation_error_deg, yaw_error_deg, translation_error_m/cm, lateral_error_cm, depth_error_cm, iou3d, symmetry_aware_add_m, symmetry_aware_add_normalized, model_diameter_m}` | 예측이 끝난 **뒤에만** 호출 |
| `load_pose_object_contract(path)` / `object_spec(contract, object_type)` | 계약 JSON 경로 / object_type 문자열 | `{long_m, short_m, height_m, ...}` | 계약 sha 를 함께 기록 |

- 저수준 selector: `challenge/evaluation_v2/pnp_selector.py:select_pnp_hypotheses(predicted_keypoints,
  camera_intrinsics, physical_dimensions, config=None) -> PnPSelectionResult` [확인].
  **양 트랙 공통 단일 정본**이다 (트랙 A `paper_real_eval.py`, 트랙 B `run_pose_evaluation.py` ·
  `pose_evaluation_paths.predict_pose_without_gt` 셋 다 이것을 호출) [확인].
  - 입력 계약: `predicted_keypoints` (9,2) float64, **finite 전부 필수**(NaN 하나면 ValueError).
    `camera_intrinsics` (3,3) — fx>0, fy>0, 마지막 행 `[0,0,1]` 강제.
    `physical_dimensions` 는 `PhysicalDimensionsXYZ` 또는 `{"x","y","z"}` mapping —
    **(W,D,H) 튜플을 주면 TypeError** [확인].
  - **confidence 인자가 없다** — conf 필터를 안 한다. object points 도 인자가 아니라
    내부에서 `geometry.camera_facing_keypoints_3d(assignment, physical_dimensions)` 로 생성한다 [확인].
  - **centroid(index 8)는 PnP 에 쓰지 않는다** — `obj[:8]`, `pts[:8]` 로 8점만 풀고
    centroid 는 재투영 진단에만 쓴다 [확인].
  - 내부: YAW_0 / YAW_90 두 parity 각각 `cv2.solvePnP(..., SOLVEPNP_SQPNP)` → `solvePnPRefineLM`
    → `validate_proper_rotation` → reprojection RMSE + cheirality + invariant + upright + degeneracy
    가중합 score [확인].
  - **GT·label·session prior 를 받지 않는다**(docstring 명시). 출력
    `PnPSelectionResult(status: SELECTED|AMBIGUOUS|FAILED, selected_hypothesis, hypotheses(2),
    canonical_candidates, ambiguity)`.
    ★ `canonical_candidates` 는 **항상 2개(부호 미결) 또는 4개(parity tie)**다 —
    `canonical_pose` 단수 필드가 의도적으로 없다. `ambiguity` 값은
    `SIGNED_AXIS_UNRESOLVED_TWO_CANDIDATES` / `WD_PARITY_TIED_AND_SIGNED_AXIS_UNRESOLVED` /
    `NO_VALID_WD_HYPOTHESIS` [확인].
  - cv2 는 lazy import — dry-run·계약 테스트는 OpenCV 없이 통과한다 [확인].
- DIAGNOSTIC 경로(최소 재투영 선택)는 `run_pose_evaluation.py` 안에 인라인돼 있다 [확인].
- 어노테이션 전용(**평가에 쓰지 말 것**): `scripts/annotate/annotate_pnp.py`
  - `solve_pose(kps_2d, K, dims=None, extrapolated_mask=None, img_shape=None, keypoint_weights=None,
    keypoint_uncertainties=None, auto_swap_dims=True, ..., physical_dimensions=None,
    camera_facing_hypothesis_override=None)` → 선택된 pose dict 또는 `None`.
    `kps_2d` 는 **9 슬롯 리스트이며 None 을 허용**(결측 표현). `dims=(W,D,H)` legacy 와
    `physical_dimensions` 는 **둘 중 하나만** (둘 다 주면 ValueError) [확인].
  - `solve_pose_safe(..., min_corners=7, reject_wd_ambiguity=True, ...)` → **항상 dict**,
    `{"accepted": bool, "reason", "reasons", "pose", ...}`. 호출자가 `accepted` 를 반드시 확인해야 한다 [확인].
  - `solve_pose_candidates(...)` → as-given + W/D-swapped 전체 후보 리스트
  - `make_pallet_keypoints_3d_diagram(width=1.1, depth=1.3, height=0.11)` → (9,3),
    camera-facing local frame (X=right, Y=down, Z=forward), 0~3 = near face, 8 = centroid [확인]
  - `assess_keypoint_topology(...)`, `parallelogram_extrapolate(kps_2d, missing_idx)` [확인]
  - 상수 `PALLET_DIMS = (1.1, 1.3, 0.11)` (width, depth, height) — **mutable 모듈 전역 legacy shim** [확인].
    `solve_pose_candidates` docstring 이 "새 논문 GT 호출자는 이 mutable 전역에서 치수를 얻지 않는다" 고 명시.
- self-training 계열(**정본 아님, 새로 붙이지 말 것**): `scripts/self_training/pnp_solver.py` —
  `PalletPnPSolver(camera_matrix, dist_coeffs=None, pallet_dims=(1.1, 1.1, 0.15), use_ransac=True, ...)`
  의 `.solve(keypoints_2d, sigmas=None, w_min=0.3, w_max=1.8)`, `.solve_reproj_guided(...)`,
  `.solve_adaptive(...)` [확인]. **conf 를 받는 유일한 솔버**지만, 기본 dims 가 `(1.1, 1.1, 0.15)` 로
  정본 plastic (1.1×1.3×0.11) 과 다르고 corner 순서 규약도 NDDS/Isaac 로 갈린다.
- solver 교체 실험은 이미 끝나 있다: `data/pallet/results/paper_pose_metric_closure_v1/solver_swap_v1/SOLVER_SWAP_RESULTS.json`
  — S0_SQPNP_LM / D1_GN_LS / D2_GN_HUBER / D3_SQPNP_GN / D4_GN_HUBER_CONF 5종,
  `new_training: 0, new_inference: 0`, R0 기준 rotation median 2.262 → 2.296 deg 로
  **solver 를 바꿔도 거의 변하지 않는다** [확인]. PnP 증폭 계층을 볼 때 이 결과를 먼저 읽을 것.

### 5-2. geometry registry (읽기 전용)

| 파일 | sha256 (실측) | 내용 |
|---|---|---|
| `challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json` | `0c7a10729b6db18cbe47fa4adb158e2f26ec7a7c9458f59ee60d023c282f0627` | `plastic_standard_110x130x11` = x 1.1 / y 0.11 / z 1.3 m, `wood_small_80x59x14` = x 0.8 / y 0.14 / z 0.59 m. 둘 다 `geometry_status: FROZEN`, `symmetry_contract: challenge/real_gt_v2/SYMMETRY_CONTRACT.json` |
| `challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json` | `e923b44880b031a7d3a9e2fffb5a6bd287cfa0de758133eeb5a73770137eba86` | challenge 전용. `_note` 에 "위 registry 를 verbatim 복사했고 여기서 편집 금지" 라고 적혀 있다. `default_object_type: plastic_standard_110x110x15` |
| `data/pallet/results/paper_pose_metric_closure_v1/POSE_EVAL_OBJECT_CONTRACT.json` | `a4c2918b4b0e9c97f94332d2e7e35132a8cbe0e738db25d92ea55e0d81210dbd` (파일 안 참조에 기록) | pose 평가 전용 **가산 계약**. `orientation_definition: {equivalent_deg:[0,180], distinct_deg:[90,270]}`. 파일 안에 "기존 registry 의 sha 는 62개 파일 198곳에 핀돼 있어 편집하지 않았다"고 명시 |

★ **셋 다 편집 금지** [확인]. 계층 분해에서 치수가 필요하면 아래 로더로 읽기만 한다.
sha 는 JSON 안의 필드가 아니라 **파일 바이트 전체의 sha256** 이며,
`load_object_geometry_registry` 가 로드 시점에 계산해 `ObjectGeometryRegistry.sha256` 에 넣는다 [확인].
`challenge/tests/test_object_geometry_registry.py` 가 `0c7a1072...` 를 하드코딩 assert 한다 [확인].

등록 객체 (challenge registry 기준, `default_object_type = "plastic_standard_110x110x15"`) [확인]:

| object_type | x_m | y_m | z_m | geometry / symmetry |
|---|---|---|---|---|
| `plastic_standard_110x130x11` | 1.1 | 0.11 | 1.3 | FROZEN / **FROZEN** (`challenge/real_gt_v2/SYMMETRY_CONTRACT.json`) |
| `wood_small_80x59x14` | 0.8 | 0.14 | 0.59 | FROZEN / **UNREVIEWED** (contract null) |
| `plastic_standard_110x110x15` | 1.1 | 0.15 | 1.1 | 과제 전용, 논문 계약과 무관 |

축 의미: `y` = **top-to-bottom 높이, +Y 가 아래(down)**, `x`/`z` = footprint 두 축.
`source_measurement_order = "width_depth_height"` [확인].

registry 로더 — `scripts/annotate/object_geometry_registry.py` [확인]:
`load_object_geometry_registry(path=DEFAULT_REGISTRY_PATH) -> ObjectGeometryRegistry`,
`get_geometry_spec(object_type, *, registry_path=...) -> ObjectGeometrySpec`,
`ObjectGeometryRegistry.resolve(object_type)` (alias 해석 — `"plastic"`, `"PLASTIC_STANDARD"`,
`"real_pallet"`, `"wood"` 등), `ObjectGeometrySpec.physical_dimensions` /
`.physical_dimensions_m` / `.legacy_wdh_tuple`(= `(x, z, y)`).
상수 `PLASTIC_OBJECT_TYPE`, `WOOD_OBJECT_TYPE`, `PLASTIC_SQUARE_OBJECT_TYPE`,
`SCHEMA_VERSION`, `DEFAULT_REGISTRY_PATH`.

`scripts/annotate/pallet_geometry.py` 공개 API [확인]:
`canonical_dimensions()`, `physical_dimensions_xyz(...)`, `canonical_keypoints_3d(...)`,
`camera_facing_dimensions(...)`, `camera_facing_keypoints_3d(...)`,
`canonical_to_camera_facing_transform(...)`,
`canonical_to_camera_facing_keypoint_permutation(...)`,
`axis_assignment_candidates_from_camera_facing_dimensions(...)`,
`camera_facing_hypothesis_name(...)`, `camera_facing_to_canonical_pose(...)`,
`make_pose_transform(rotation, translation)`, `validate_proper_rotation(...)`.
상수: `CANONICAL_X_M = 1.10`, `CANONICAL_Y_M = 0.11`, `CANONICAL_Z_M = 1.30`.
데이터클래스: `PhysicalDimensionsXYZ`, `CameraFacingDimensionsWHD`, enum `AxisAssignment`.

프레임 계약 (모듈 docstring) [확인]: canonical physical frame = centroid 원점, 우수좌표계,
`+X` = 물리 1.10 m 축, `+Y` = top→bottom(down) 0.11 m, `+Z` = 물리 1.30 m 축.
`P_cf[i] = A @ P_canonical[p[i]]`, `R_canonical = R_cf @ A`, `t_canonical = t_cf`
(양 프레임이 keypoint 8 을 원점으로 공유하기 때문).
`canonical_to_camera_facing_transform` 은 **정확한 정수 quarter-turn**
(`matrix_power([[0,0,1],[0,1,0],[-1,0,0]], yaw//90)`) — 부동소수 회전이 아니다.

★ `pallet_geometry._diagram_points` 의 corner 순서는
`symmetry_aware_pose_metrics.cuboid_model_points` 와 **정확히 같다** [확인] →
두 트랙의 `model_points` 를 서로 바꿔 넣어도 안전하다.

### 5-3. camera intrinsics 로더 — 단일 정본 없음, 네 갈래

| 경로 | 함수 / 위치 | 입력 → 출력 |
|---|---|---|
| ① GT JSON 내장 (트랙 A) | `challenge/evaluation_v2/paper_real_eval.py` 의 `_legacy_forbidden_target` 내부 (별도 함수 아님) | GT v2 payload `camera_data.intrinsics.{fx,fy,cx,cy}` → (3,3) float64. finite·fx>0·fy>0 검증. 없으면 `GT_V2_CAMERA_INTRINSICS_REQUIRED` [확인] |
| ② intrinsics manifest | `challenge/evaluation_v2/selector_diagnostic.py:_load_intrinsics_manifest(path, manifest)` | `challenge/real_gt_v2/manifests/DEV_POS140_INTRINSICS.json`, 레코드 필드 `{frame_id, fx, fy, cx, cy, source_label_sha256}` → `{frame_id: (3,3)}`. ★ **현재 이 파일의 `records` 는 비어 있다(len 0)** [확인] |
| ③ annotation JSON 직접 파싱 (트랙 B) | `scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py` 인라인 | `payload["camera_data"]["intrinsics"]` → (3,3). 경로는 `AXIS_REVIEW_MANIFEST.json` 의 `frames_list[i]["annotation"]` [확인] |
| ④ 세션 단위 `cam_K.txt` | `scripts/self_training_yolo/dump_teacher_predictions.py:_camera_matrix(image_path)` | `image_path.parent.parent / "cam_K.txt"`, 공백 구분 9 값 → 3×3. 없으면 None [확인] |

이미 K 가 artifact 안에 들어 있어 다시 열 필요가 없는 경우 [확인]:
- 161/140 population — `paper_generic_pipeline/eval_manifest.json` 의 `items[i]["K"]`
- `phase_a_oracle/PREDICTION_CACHE.npz` 의 `camera_intrinsics(185,3,3)`
- `R0_TEACHER_CACHE.json` 의 `entries[i]["camera_matrix"]`

★ `PositiveTarget.intrinsics_quality` 가 `"ESTIMATED_HFOV"` 면 `evaluate_pose_records` 가
`POSE_INTRINSICS_ESTIMATED_HFOV_NOT_APPROVED` 로 **거부**한다 [확인].

---

## 6. keypoint 좌표계 변환

정본이라 부를 만한 **단일 유틸 함수는 없다** — YOLO 경로는 어디서나 인라인이다 [확인].

- 정방향: `cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)` → `model.predict(padded, imgsz=640)`
- 역방향: `boxes.xyxy - PAD`, `keypoints.xy - PAD` (스칼라 100 뺄셈이 전부)
- 대표 구현 [확인]:
  - `challenge/evaluation_v2/paper_real_eval.py:_UltralyticsPredictor.predict` — **트랙 A 정본**.
    상수 `INFERENCE_PAD = 100`, `INFERENCE_IMGSZ = 640`, `INFERENCE_CONFIDENCE_FLOOR = 0.001`
  - `scripts/paper/pose_metric_closure_v1/run_frozen_arm_inference.py` (78~99줄) — **트랙 B 정본**.
    pad/imgsz/conf 를 `INFERENCE_REPLAY_LOCK.json` 에서 읽는다
  - `challenge/yolo_pose_one_model/analysis_pre_v2/cc_dump.py:boxes_of()` (110~111줄) — 전 후보 버전
  - `scripts/self_training_yolo/dump_teacher_predictions.py:_extract(result, index)` —
    재사용 가능한 최소 단위. `{box_xyxy, box_conf, keypoints_xy, keypoints_conf}` 원본 픽셀계
  - 같은 패턴이 `challenge/yolo_pose_one_model/` 아래 20곳 이상에 복제돼 있다
- ⚠️ **border 모드가 갈리는 곳이 있다**: `challenge/scripts/evaluate/eval_ab_crop.py:predict` 와
  `challenge/pallet_jetson_deploy/infer_fps.py:predict` 는 `BORDER_REFLECT`(101 아님)를 쓴다 [확인].
  이 두 경로에서 나온 좌표를 정본 캐시와 섞지 말 것.
- 규칙의 정본 문서 = `INFERENCE_REPLAY_LOCK.json` 의
  `"unpad_rule": "subtract pad from box and keypoint coordinates after inference"` [확인]
- letterbox 는 ultralytics 내부에서 처리되고 결과 좌표는 이미 padded-image 좌표로 돌아온다.
  따라서 저장된 좌표는 전부 **원본 픽셀** 이며, pad 영역에 걸친 점은 음수 또는 W/H 초과가 된다 [확인].
- DOPE 계열은 별도 함수가 있다: `scripts/stage0/selftrain/s16_real_eval.py:belief_to_orig_pad(bx, by, bw, bh, nw, nh, sc, pad, W, H)`
  와 `pad_frame(img, pad)` [확인]. `challenge/scripts/infer/dope_predict_mp4_pad.py:pad_frame(img, pad, mode)` 도 있다.
- ★ DOPE 추론은 reflect padding 이 필수다 — `DOPE_R0_PREDICTIONS.json` 의 `recipe.reflect_padding_note`
  가 "plain squash 는 잘림·근접에서 체계적으로 과소검출한다"고 기록하고 있다 [확인].

`challenge/yolo_pose_one_model/p26_tal_target_audit/ta_core.py` 의
`letterbox_params(padded_shape) -> (r, dw, dh, (h,w))` 와 `gt_to_input(gt_xyxy, padded_shape)` 가
저장소에서 letterbox `r/dw/dh` 를 명시적으로 계산하는 유일한 함수다
(`r = min(h/h0, w/w0)`, `dw = (w-nw)/2`, `dh = (h-nh)/2`).
단 **정변환(original → input)만 있고 역변환은 없다** [확인].
평가 경로에서는 ultralytics 가 letterbox 역변환을 내부에서 끝내고 padded-image 좌표로 돌려주므로,
남는 일은 `- 100` 등방 shift 뿐이다 [확인].

---

## 6-bis. population loader API

### 6b-1. `challenge/data_paths.py` [확인]

- `EVAL_CANONICAL: dict[str, str]` — 7 키 → **repo-상대 디렉터리 경로**. 함수가 아니라 단순 dict.
- `EVAL_CANONICAL_TOTAL = 140`, `EVAL_CANONICAL_RAW_TOTAL_BEFORE_GT_QA = 161`,
  `EVAL_CANONICAL_GT_QA_EXCLUDED = 21`.
  ★ **CLAUDE.md 의 "161장" 표기와 코드의 140 이 다르다** — GT QA 로 21장이 격리된 뒤 값이다.
  계층 분해에서는 코드 값(140)을 따르고, 161 짜리 옛 artifact 를 쓸 때만 차이를 명시할 것.
- `FINAL_TEST = ("eval_pallet07","eval_pallet09","eval_night08","eval_night09")` —
  주석에 "2026-08-20 부로 봉인 해제, 재봉인 불가. 이름만 하위호환" 이라고 적혀 있다 [확인].
  **threshold 튜닝·모델 선택 금지.**
- `FORBIDDEN_EVAL_SOURCES = ("_eval_sets/outside_combined", "_eval_sets/night_combined")`
- 함수: `get(name, absolute=False)`, `invalid_gt_source_paths()`, `missing()`.
  CLI: `python challenge/data_paths.py --get eval_cad [--abs] | --list | --check`

### 6b-2. `challenge/evaluation_v2/real_dataset_contract.py` (트랙 A) [확인]

`load_population_manifest(path, *, validate_files=True)`,
`load_repo_population(population_id, *, validate_files=True)`,
`manifest_path(population_id)`, `membership_sha256(items)`,
`validate_evaluation_pair(positive, negative, ...)`,
`validate_repo_population_contract(*, validate_files=True)`.

- manifest 디렉터리 `challenge/real_gt_v2/manifests`, schema `pallet_pose_population_manifest_v1`.
- **eager fail-closed**: 멤버 파일이 하나라도 없으면 분모를 줄이지 않고 에러.
  `expected_count` / `object_types` / `kind` / `role` 이 하드코딩 기대치와 정확히 일치해야 한다.
- `ManifestItem.canonical_record()` 가 **두 세대 스키마**로 분기한다:
  신형 `{frame_id, object_type, session_id, image_path, gt_v2_path, population_role, source_population, domain}`,
  구형 `{frame_id, image, label, source_set, domain}` (membership sha 를 byte 단위로 보존하려고 유지).
  접근자 `.image_path ≡ .image`, `.gt_v2_path ≡ .label`. 경로는 항상 repo-상대.
- GT JSON 여는 법: `(REPO_ROOT / item.label)` → `json.loads` →
  `schema_version == "real_pallet_gt_v2"` 확인 →
  `scripts/annotate/real_gt_v2_schema.validate_gt_v2(payload)` 전체 검증 → `PositiveTarget`.

### 6b-3. `scripts/evaluation/eval_workspace.py` (트랙 B) [확인]

`load_frames(root: Path) -> list[dict]` — `root/manifests/frames.csv` (CSV).
`evaluation_population_views(frames) -> dict[str, list[Mapping]]` —
키 `DEV_PLASTIC_AUDITED140`, `DEV_EVAL_POSITIVE/NEGATIVE`, `FINAL_EVAL_*`(legacy alias),
**`PAPER_EVAL_POSITIVE/NEGATIVE`(정식)**, `ALL_AVAILABLE_*`.
workspace root = `data/evaluation/pallet_eval_v1`.
경로 규약 `sessions/{session_id}/rgb/{stem}.png`, `annotations/{session_id}/{stem}.json`.
소스 주석: "`PAPER_EVAL` = SHA-dedup union(DEV_EVAL, NEW_EVAL), `held_out_final = false`.
진짜 untouched test 를 나중에 만들면 `HELDOUT_EVAL` 이라는 별도 이름을 쓴다."

### 6b-4. ★ frame ID 형식이 세 가지 — 혼용 금지 [확인]

| 형식 | 예 | 쓰이는 곳 |
|---|---|---|
| 순수 타임스탬프 | `1778651519235162880` | `challenge/real_gt_v2/manifests/*.json` 의 `items[].frame_id`, `eval_manifest.json`, `_cc_raw_dump.json`, `model_compare/kps_*.json` |
| 세션 접두 복합 | `eval_cad__1778653003088339968` | `paper_pose_metric_closure_v1/*`, `multiteacher_corner_distill_v1/predictions/*` |
| 콜론 구분 | `eval_cad:1778653003088339968` | `paper_eval_v*/arms/*_per_frame.csv`, `AXIS_FAILURES.json`, `R0_TTA960_CACHE.json` |

생성 규칙: `scripts/evaluation/eval_workspace.py` 의 `f"{session_id}__{safe_component(stem)}"` [확인].
**세 형식을 잇는 정규화 함수는 없다** — 조인 스크립트가 직접 처리해야 한다.

---

## 7. failure label artifact — 프레임 단위 실패 분류가 이미 있다

### 7-1. candidate selection 계층 — `challenge/yolo_pose_one_model/analysis_pre_v2/_rr_detail.json` (git 추적)

161 프레임, 각 원소
`{fid, set, population, cls, n_cand, top1_iou, top1_R, top1_t, best_correct_rank, best_correct_R, best_correct_t}` [확인].

실측 `cls` 분포 [확인]:

| cls | n | 뜻 |
|---|---|---|
| `TOP1_ALREADY_GOOD` | 65 | top-1 이 이미 정답 |
| `B_CORRECT_BOX_BAD_KP` | 59 | 박스는 맞는데 keypoint 가 틀림 |
| `C_NO_CORRECT_CANDIDATE` | 28 | 후보 리스트 전체에 정답이 없음 |
| `A_GOOD_CANDIDATE_MISRANKED` | 9 | 정답 후보가 있는데 랭킹이 놓침 |

사전 고정 정의(결과 보기 전 고정) [확인]: `IOU_MATCH = 0.5`, `USABLE = R ≤ 10° AND t ≤ 0.10 m`,
oracle = top-K 중 R 오차 최소.

→ 이 표 하나로 **detection / candidate-selection / keypoint 계층의 몫이 이미 분리돼 있다**.
계층 분해의 출발점으로 그대로 쓸 수 있다. 단 population 은 161(구본)이고 21장이 GT 격리 대상이다.

### 7-2. axis permutation 계층 — `data/pallet/results/paper_eval_v1/AXIS_FAILURES.json`

`schema_version: mirror_failure_diagnosis_v1` [확인]. 319 프레임 × 3 arm (`R0`, `R2_CONF`, `R5_PROPOSED`).

프레임 엔트리 실측 키:
`identity_median_px, identity_max_px, permutation_max_px{identity, yaw90, yaw180, yaw270, mirror,
mirror_yaw90, mirror_yaw180, mirror_yaw270}, best_permutation, best_permutation_max_px,
hungarian_max_px, hungarian_assignment_gt_from_pred, centroid_delta_px, verdict, box_conf,
paper_domain, object_type` [확인].

판정 기준(파일 안 `criterion`): `axis_absolute_px = 25.0`, `axis_ratio = 0.5`,
통계 = 감독 keypoint 최대 코너 오차. 순열 8종의 인덱스 배열이 파일에 그대로 들어 있다 [확인].

실측 `verdict` 분포 [확인]:

| arm | OK | MISLOCATED | AXIS_PERMUTED |
|---|---|---|---|
| R0 | 221 | 83 | 15 |
| R2_CONF | 203 | 99 | 17 |
| R5_PROPOSED | 199 | 104 | 16 |

★ 파일 안 note: "median 은 90도 순열의 이봉분포에 속는다 — 최대 오차로 볼 것" [확인].

### 7-3. pose 계층 per-frame — `data/pallet/results/paper_pose_metric_closure_v1/POSE_PER_FRAME_BY_ARM.json`

`schema_version: pose_per_frame_by_arm_v1`, `all_reproduced_exactly: true` [확인].
`per_frame[arm]` = list, 원소
`{frame_id, session_id, object_type, paper_domain, axis_correct, rotation_error_deg,
yaw_error_deg, translation_error_cm, iou3d, add_sym_m, diameter_m}` [확인].

### 7-4. keypoint role 계층 — `challenge/yolo_pose_one_model/analysis_pre_v2/KEYPOINT_PERMUTATION_AUDIT.json`

top-level: `{note, gross_threshold_R_deg, n_solved, n_gross, best_permutation_counts,
identity_best_rate, role_confusion_rate, rows}` [확인].
`rows` **116**, 각 행 실측 키 [확인]:
`fid, set, population, identity_R, identity_corner, lr_swap_R, lr_swap_corner,
near_far_swap_R, near_far_swap_corner, near_far_lr_R, near_far_lr_corner,
top_bottom_R, top_bottom_corner, rot180_face_R, rot180_face_corner`.

### 7-4b. PnP 증폭 계층 — `challenge/yolo_pose_one_model/analysis_pre_v2/KEYPOINT_SUBSET_PNP.json`

top-level `{note, populations, rows}`, `rows` **116**, 실측 키 [확인]:
`fid, set, population, P0_all8_R, P0_all8_t, top7_R/t, top6_R/t, top5_R/t, top4_R/t,
loo0_R … loo7_R, near_only_R, far_only_R, top_only_R, bottom_only_R`.
→ **부분집합 PnP · leave-one-out 이 이미 계산돼 있다.** 재추론 0.

★ 두 파일 모두 `rows` 가 **116** 이다 — 161 population 중 PnP 가 풀린 프레임만이다 [확인].
분모를 161 이나 140 으로 착각하지 말 것.

### 7-5. 계층별 판정이 이미 문서화된 곳 — `challenge/yolo_pose_one_model/analysis_pre_v2/PRE_V2_DECISION.json`

실측 내용 [확인]:

| 축 | verdict | 근거(파일 원문) |
|---|---|---|
| D1_NO_BOX | `CONFIDENCE_CALIBRATION_ISSUE` | NO_BOX 45 중 30 (66.7%) 이 conf 0.001 에서 IoU≥0.5 후보를 낸다. availability 잠재 이득 18.6 pp |
| D2_RESOLUTION | `NOT_A_BOTTLENECK` | 960 은 availability +4.7pp 이나 IoU≥0.5 −1.9pp, 1280 은 붕괴 |
| D3_SEMANTICS | `ROLE_CONFUSION_PRESENT` | gross(R>10°) 24 장 중 identity 최선 11 (45.8%), near_far_swap 최선 11 (45.8%) |
| D4_SUBSET | `NO_SINGLE_BAD_KEYPOINT` | top7 이 전체와 사실상 동일, top5 이하부터 급격 악화 |
| D5_SOLVER | `SOLVER_IS_FINE` | GT 점에서 네 solver 모두 0.00~0.04°, 예측 점에서만 3.3~3.5° |
| D6_GT_CALIBRATION | `EVALUATION_NOISE_FLOOR_ACCEPTABLE` | GT 재투영 median 0.42~2.46 px (night09 p90 6.83 px 최대) |

→ **다섯 계층 중 detection(D1) · keypoint role(D3) 두 개는 이미 "크다"로 판정돼 있고,
해상도·부분집합·solver 는 "아니다"로 판정돼 있다.** 새 실험을 설계하기 전에 이 판정부터 읽을 것.

### 7-6. 랭킹 feature 의 판별력 — `RERANK_FEATURES.json`

`phase3_single_feature` 에 feature 별 AUC / MRR 이 있다. 실측 일부 [확인]:
`box_conf` box-AUC 0.7248 / usable-AUC 0.8182, `kp_conf_mean` 0.6767 / 0.8215,
`kp_conf_min` 0.6058 / 0.8147, `box_area` box-AUC 0.98 (단 n_auc_frames = 8 — **표본 8장**).
사전 고정: `no_learned_reranker: true`, `no_feature_combination: "결과 보고 조합하지 않는다"`.

★ `box_area` AUC 0.98 을 인용할 때는 반드시 `n_auc_frames = 8` 을 함께 적을 것 [확인].

### 7-7. tail / regression

- `data/pallet/results/paper_eval_v1/TAIL_HIGH_ERROR_FRAMES.csv` (314 행) — 헤더 실측:
  `image, session_id, object_type, paper_domain, occlusion, truncation, distance_bin, error_source,
  proposed_corner_median_px, proposed_corner_max_px, baseline_corner_median_px, gross_keypoints,
  in_worst_decile` [확인]
- `data/pallet/results/paper_eval_v1/REGRESSION_DIAGNOSIS.json` (`schema_version:
  selftraining_regression_diagnosis_v1`) — R0 vs R5 를 domain 별로
  `detection_crosstab {BOTH_DETECTED, R0_ONLY, R5_ONLY, BOTH_MISSED}` + 프레임 ID 리스트로 분해 [확인]

---

## 8. 교사 예측 (multi-teacher) — 상세

`data/pallet/results/multiteacher_corner_distill_v1/` [확인]

- `TEACHER_REGISTRY.json` (`status: FROZEN_BEFORE_MEASUREMENT`) — 교사 7종의 checkpoint + sha256 +
  아키텍처 + 학습 소스. 편입 규칙: real keypoint/pose 감독 0, pseudo-label 감독 0,
  camera-facing 0123 v4, `kpt_shape [9,3]`, `flip_idx [1,0,3,2,5,4,7,6,8]`, PAPER_EVAL 미사용.
- `predictions/T{0..6}.json` + `C0.json` + `C1.json` — 9 arm × 319 프레임, arm 당 top-1.
  전 arm 이 같은 `population_frame_order_sha256 = 72f83f6f...` 를 공유한다 [확인] →
  **arm 간 프레임 단위 짝지음이 보장된다.**
- `no_detection` 실측: T0 0, T1 2, T2 0, T3(DOPE) 22, T4 2, T5 0, T6 0 [확인]
- `gate_a_teacher_headroom/GATE_A_RESULT.json` — `population: "DEV_EVAL 319, supervised keypoints"`,
  `n_rows_all 2818 / n_rows_corner 2499 / n_rows_visible_corner 1594`, `new_training 0, new_inference 0`.
  교사별로 `ALL_supervised / corners / visible_corners / centroid / by_corner`에
  `{n, median_px, p90_px, mean_px, gross20, gross40}` [확인].
- `gate_c_local_specialist/TARGET_TEACHER_CACHE.json` (`schema_version: mtcd_target_teacher_cache_v1`) —
  unlabeled pool `data/evaluation/pallet_eval_v1/adaptation/MAIN_UNLABELED_BALANCED.csv` 에 대한
  교사 합의 캐시. `tau_consensus_normalised`, `n_usable`, `usable_rate` 포함 [확인].

★ 기존 판정(memory): 여러 교사 합의는 **신뢰도를 주지 라벨을 주지 않는다**.
oracle p90 43.9 → 13.3 은 실재하지만 GT 없이 못 고른다.
계층 분해에서 이 캐시는 "keypoint localisation 의 상한"을 재는 용도로만 쓸 것.

---

## 9. 계층별로 무엇이 있고 무엇이 없나

| 계층 | 재사용 가능한 입력 | 재추론 필요? |
|---|---|---|
| **detection** (박스가 아예 안 나옴) | `_cc_raw_dump.json` 의 conf 0.001 전 후보 + `PRE_V2_DECISION.json:D1` + `paper_eval_v*/arms/*_per_frame.csv` 의 `candidate_count` + `DEV_NEG2689` negative 2,689 | **NO** (yolo26n_paper_generic_v1 한정). 다른 체크포인트는 YES |
| **candidate selection** (후보 중 잘못 고름) | `_cc_raw_dump.json` + `_rr_cands.json`(후보별 R/t/iou/s5) + `RERANK_ORACLE.json`(K=1,2,3,5 상한) + `_rr_detail.json`(4-class 라벨) + `RERANK_FEATURES.json` | **NO** (같은 한정) |
| **keypoint role** (순열·역할 혼동) | `AXIS_FAILURES.json`(319×3arm, 8순열 max px) + `KEYPOINT_PERMUTATION_AUDIT.json`(116행) + `PRE_V2_DECISION.json:D3` | **NO** |
| **keypoint localisation** (좌표 정밀도) | `paper_pose_metric_closure_v1/predictions/*.json`(8 arm×319) + `multiteacher/predictions/*.json`(9 arm×319) + `GATE_A_RESULT.json`(교사별 코너 오차 분포) + `paper_eval_v*/arms/*_per_frame.csv`(감독 keypoint px 오차 리스트) | **NO** |
| **PnP 증폭** (같은 2D → pose 오차) | `solver_swap_v1/SOLVER_SWAP_RESULTS.json`(5 solver × 7 arm) + `KEYPOINT_SUBSET_PNP.json`(부분집합/LOO) + `PRE_V2_DECISION.json:D5` + `pose_evaluation_paths.py` 의 MAIN/DIAGNOSTIC/ORACLE 3경로 | **NO** |

**빠져 있는 것 (새로 만들어야 함)** [확인]:

1. `paper_eval_v*` / `pose_metric_closure_v1` / `multiteacher` 가 쓰는 **319 population 에 대한
   다중 후보 덤프가 없다.** 다중 후보는 161 population + 다른 체크포인트에만 있다.
   두 계열을 잇는 후보 oracle 을 원하면 재추론이 필요하다.
2. `_cc_raw_dump.json` 의 population 은 GT QA 이전 161 이라, 21 장을 **버리고 140 으로 재집계**하는
   작업이 필요하다(예측은 재사용, GT 파생 스칼라만 재계산).
3. 계층을 하나의 표로 잇는 **조인 스크립트**. 재사용할 evaluator·PnP·geometry 는 다 있으므로
   새로 짤 것은 조인과 집계뿐이다.

---

## 10. 절대 하지 말 것

- `challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json`, `challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json`,
  `POSE_EVAL_OBJECT_CONTRACT.json` 편집 — sha 가 다수 artifact 에 핀돼 있고 테스트가 하드코딩 assert 한다 [확인].
- `oriented_iou3d.py` / `pnp_selector.py` 를 복제 — 저장소 유일 구현이고 양 트랙이 공유한다 [확인].
- **`pose_auc` 를 트랙 간에 바꿔 호출** — A 는 정규화된 값, B 는 raw m + diameter. 예외 없이 틀린다 [확인].
- **`rotation_error_degrees` / `yaw_error_degrees` 를 트랙 구분 없이 import** —
  A 는 대칭 미처리 · yaw 0..180, B 는 대칭 처리 · yaw 0..90 [확인].
- **`pose_metrics.adds_error_m` 을 ADD-S 로 보고** — docstring 이 직접 "paper evaluator 는 호출하지 않는다"고 경고 [확인].
- `scripts/self_training/metrics.py` · `pnp_solver.py` · `Deep_Object_Pose/evaluate/` 를 새로 붙이기 —
  전부 legacy 이고 dims·corner 순서·AUC 적분점이 정본과 다르다 [확인].
- `re_metrics` 계열 수치와 `symmetry_aware_pose_metrics` 계열 수치를 한 표에 섞기 [확인].
- 319 population 과 161/140 population 을 합산 — 세션 구성이 다르다 [확인].
- frame_id 세 형식(`ts` / `set__ts` / `set:ts`)을 정규화 없이 조인 [확인].
- `_cc_raw_dump.json`(yolo26n_paper_generic_v1)과 G38 계열(`OLD_ROOT_G38_..._SEED42`, sha `1a806ca4...`)
  artifact 를 같은 모델처럼 비교 — **다른 체크포인트다** [확인].
- `box_area` AUC 0.98 을 표본수 없이 인용 — `n_auc_frames = 8` [확인].
- `KEYPOINT_PERMUTATION_AUDIT` / `KEYPOINT_SUBSET_PNP` 의 분모를 161·140 으로 착각 — `rows` 는 **116** [확인].
- `_rr_cands.json` 의 GT 파생 값을 21 장 격리 프레임까지 포함해 그대로 쓰기 [확인].
- `FINAL_TEST` 4 세션(`eval_pallet07/09`, `eval_night08/09`)으로 threshold 튜닝·모델 선택 —
  봉인이 소진됐고 재봉인 불가다 [확인].
