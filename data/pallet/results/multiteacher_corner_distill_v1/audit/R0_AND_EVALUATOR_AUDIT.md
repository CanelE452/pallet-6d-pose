# 기준 모델(R0) · 평가기 사실 감사

모두 저장소 실측이다. 이 트랙은 아래 어느 파일도 수정하지 않는다.

## 기준 모델 R0

```
arm            R0  (synthetic-only)
checkpoint     challenge/yolo_pose_one_model/spatial_concat_scratch/runs/
               YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt
sha256         970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7   [확인] 재계산 일치
size           6,552,807 B
lock           data/pallet/results/paper_pose_metric_closure_v1/POSE_ARM_CHECKPOINT_LOCK.json
학습           yolo26n-pose 에서 60 epoch, batch 32, imgsz 640, SGD lr0 0.01, seed 42,
               cos_lr, close_mosaic 10, single_cls, mosaic 0.3, scale 0.25, flip 0.0
               pose 12.0 / kobj 1.0 / rle 1.0 / angle 1.0
데이터         challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k/data.yaml
               train 55,980 / val 4,020, 전부 렌더 합성. kpt_shape [9,3],
               flip_idx [1,0,3,2,5,4,7,6,8]
real GT        사용 0 — synthetic only
```

### 불일치 1 — lock 문서의 "정본" 과 실제 사용 파일이 다르다

`spatial_concat_scratch/BASE_CHECKPOINT_LOCK.json` 은 `checkpoint_rule = "fixed-budget
epoch-60 last.pt"` 라 쓰고 best.pt 를 `best_pt_comparison_only` 로 분류한다.
그런데 논문 arm R0 로 동결된 것은 **best.pt** 다 (`paper_eval_v1/arms/R0.json` 의
선언 sha 가 best.pt 의 sha 다). `TRAINING_CONTRACT.json` 은 이 run 자체를
`"scope": "challenge-performance exploratory; not paper-track evidence"` 라 적었다.

**이 트랙의 처리**: R0 = best.pt 를 그대로 쓴다. 이유는 공개된 R0 수치(detection 0.975,
keypoint median 6.6157, IoU3D 0.60318 …)가 전부 best.pt 로 산출됐고, 이 트랙의 모든
비교는 그 R0 대비이기 때문이다. lock 문서의 문구 불일치는 **기존 논문 트랙의 사안**이고
여기서 고치지 않는다. 다만 R0 를 "합성 전용 60epoch 정본" 이라고 부를 때
last.pt 가 아니라는 사실을 같이 적는다.

## 추론 recipe (그대로 재사용)

`INFERENCE_REPLAY_LOCK.json` — status FROZEN, sha256 `2c1bedd7…389ef25c` `[확인] 재계산 일치`
```
input_size 640 · pad_px 100 · BORDER_REFLECT_101 · 추론 후 pad 만큼 빼기
confidence_floor 0.001 · top-1 = 최고 box confidence · augment false · half false · device 0
frame_order_sha256 72f83f6f…d8c15f  (R0.json 의 값과 일치)
```
reflect padding 은 memory `dope-inference-needs-reflect-padding` 의 교훈이 반영된 부분이다.
이 트랙의 모든 teacher 추론은 **같은 recipe** 로 돈다 — 그래야 상보성이 recipe 차이가 아니다.

## 6D 평가기 (그대로 재사용, 새 지표 발명 0)

```
드라이버   scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py
지표       scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py
IoU3D      challenge/evaluation_v2/oriented_iou3d.oriented_iou_3d
solver     cv2.SOLVEPNP_SQPNP → cv2.solvePnPRefineLM, distCoeffs None
대칭군     (I, diag(-1,1,-1)) — 180 도만. 90 도는 다른 pose 로 센다
물체       POSE_EVAL_OBJECT_CONTRACT.json  sha256 a4c2918b…10dbd
           plastic 1.30/1.10/0.11 (194) · wood 0.80/0.59/0.14 (125)
```

세 경로 MAIN / DIAGNOSTIC / ORACLE 을 낸다.
- MAIN: `select_pnp_hypotheses` — GT 인자 없음. 배포 가능한 축 선택.
- DIAGNOSTIC: 최소 재투영 선택. 사후 진단.
- ORACLE: GT 축 공급. 상한.

### 불일치 2 — "MAIN 은 GT-free" 를 어디까지 읽어야 하나

세 경로 모두 `extents` 와 `model_points` 를 `truth["physical_dimensions_m"]`,
즉 GT 가 해소한 축 정렬 치수에서 만든다 (`run_pose_evaluation.py:141-146`).
이걸 "누수" 로 부르면 과하다 — ADD·IoU3D 는 원래 **참 물체 모델**로 채점하는 지표이고,
pred/GT 양쪽에 같은 extents 가 들어간다. 축 *선택* 에 GT 가 들어가는 경로는 ORACLE 뿐이다.
다만 known-size 가정이 평가에 들어가 있다는 사실은 남는다.

그리고 실제로 MAIN 의 R,t 는 selector 가 푼 pose 가 아니라
`fits[chosen]`(DIAGNOSTIC 이 푼 solve 결과)을 재사용한다 — 그래서
`POSE_EVALUATION_R0.json` 에서 MAIN 과 DIAGNOSTIC 의 t median·IoU3D 가 소수점까지 같다.
selector 는 라벨(CF_WIDTH/CF_DEPTH)만 고르고 pose 는 공통 solve 를 쓴다. `[확인]`

**이 트랙의 처리**: 평가기를 한 줄도 고치지 않고 MAIN 을 primary 로 보고한다.
ORACLE 은 항상 oracle 로 표시한다. 절대값을 배포 성능이라 부르지 않는다.

### 축 선택기가 이미 실패 상태라는 사실

`POSE_CLOSURE_STATUS.json`
```
selector_accuracy_pooled 0.6014   constant_guess_baseline 0.5725   gain +0.0289
gates overall/night/min_session  전부 FAIL
cost_when_wrong  rotation 85.3도 · yaw 85.27도 · translation 0.219 m (맞을 때 0.065 m)
```
따라서 이 트랙의 6D 수치 차이는 **축 선택기의 성패에 크게 지배된다**.
keypoint 가 좋아져 축 선택이 좋아지면 6D 가 따라오는 것이 이 트랙의 인과 경로이고,
그렇지 않으면 6D 는 2D 개선을 반영하지 못할 수 있다. 이 점을 결과 해석에 먼저 쓴다.

## 2D 지표 — 정의 출처와 이 트랙의 확장

median/p90 정의는 `challenge/evaluation_v2/paper_real_eval.py::_distribution`
(`np.median`, `np.percentile(...,90)`, keypoint 를 프레임 넘어 pooled).
오차는 `||pred_kp - gt_kp||`, supervised = `xy 존재 & visibility>0`.

`gross20` / `gross40` 은 pose closure 파이프라인에 **없다**. 다른 트랙
(`runs_pallet_loss/real_equiv_eval.py:146`, `a1_measure.py:132`, `p26_driver.py:67`)에
`(e > 20).mean()` / `(e > 40).mean()` 로 존재한다. 이 트랙은 그 정의를 그대로
`mtcd_common.error_stats` 에 옮겨 쓰고 METHOD_LOCK 에 고정한다.

### R0 의 2D 기준선 (이 트랙 정의로 재계산) `[확인]`

```
집합                     n      median    p90      gross20   gross40
────────────────────────────────────────────────────────────────────
supervised 0..8        2818     6.837    52.03      0.190     0.116
supervised corners 0..7 2499     6.971    60.31      0.199     0.124
visible only (vis==2)  1594     6.360    43.89      0.157     0.102
centroid 8              319     5.871    22.46      0.119     0.053

corner 별 (supervised)
  kp0 n=317 med 6.23 p90 69.30 g20 0.164   kp4 n=319 med 5.63 p90 63.37 g20 0.204
  kp1 n=301 med 7.16 p90 33.13 g20 0.146   kp5 n=316 med 6.96 p90 61.86 g20 0.225
  kp2 n=301 med 6.58 p90 35.09 g20 0.163   kp6 n=316 med 8.79 p90 62.29 g20 0.250
  kp3 n=310 med 6.74 p90 68.06 g20 0.177   kp7 n=319 med 7.85 p90 63.39 g20 0.260
```

### 논문 공표값과의 parity — 자릿수까지 재현됨 `[확인]`

논문은 `median 6.6157 / p90 38.670` 을 싣는다. 위 표(319 프레임 전부)는 6.837 / 52.03 이다.
차이의 원인은 **IoU50 매칭 게이트**다. GT 코너 8개의 축정렬 외접상자와 arm 의 top-1 상자가
IoU >= 0.5 인 프레임만 남기면 311/319 이 남고, 그 위에서

```
matched-only  n 2756  median 6.615677617706521  p90 38.67003769171306
공표값                 median 6.6157             p90 38.670
```

즉 이 트랙의 계산 경로는 논문 evaluator 와 **같은 수를 낸다**. 정의를 새로 만든 게 아니다.

### 이 트랙의 2D 보고 규칙 (METHOD_LOCK 에 고정)

```
POOLED_ALL        해당 arm 이 검출에 성공한 모든 프레임. 꼬리를 숨기지 않는다.
POOLED_MATCHED50  그 arm 의 top-1 상자가 GT 코너 외접상자와 IoU>=0.5 인 프레임.
                  논문 헤드라인과 직접 비교 가능한 정의.
PAIRED_COMMON     두 arm 을 짝지어 비교할 때는 양쪽 다 예측이 있는 프레임만 쓰고
                  arm 마다 다른 매칭 게이트를 적용하지 않는다
                  (게이트를 적용하면 arm 마다 모집단이 달라져 짝짓기가 깨진다).
```

중앙값 6.8 px 대비 p90 52 px, gross20 19% — **꼬리가 두껍다**.
이 트랙이 겨냥하는 실패는 바로 이 꼬리다.
