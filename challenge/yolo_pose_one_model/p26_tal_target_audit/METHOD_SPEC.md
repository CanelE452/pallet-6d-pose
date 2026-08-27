# METHOD SPEC — P26 TAL target audit (training-0)

## 질문

Y0 의 classification score 오정렬 원인이 T1 assignment / T2 target quality /
T3 within-image ranking objective / T4 real generalization / T5 level calibration
중 무엇인가.

## 고정한 것 (결과 보기 전)

```
checkpoint  representation audit 과 동일한 Y0 (sha 37f904b9…)
inference   imgsz 640 / conf 0.001 / iou 0.7 / max_det 300 / one2one E2E / pad100 REFLECT_101
금지         backward · optimizer · model.train() · fuse · 가중치 변경 · threshold tuning
CASE        T1~T6 정의는 결과 후 변경 없음
```

## assigner 를 진단 모드로 실행하는 방법

`utils/loss.py:398-429` 의 `get_assigned_targets_and_loss` 경로를 그대로 재조합한다.
loss 는 계산하지 않고 assigner 만 부른다.

```
pred_scores/pred_distri = preds["one2one"]["scores"/"boxes"]
anchor_points, stride   = make_anchors(preds["one2one"]["feats"], stride, 0.5)
pred_bboxes             = criterion.one2one.bbox_decode(anchor_points, pred_distri)
targets                 = criterion.one2one.preprocess(GT_xywh_norm, 1, scale_tensor=imgsz[[1,0,1,0]])
assigner(pred_scores.sigmoid(), pred_bboxes*stride, anchor_points*stride, gt_labels, gt_bboxes, mask_gt)
```

GT 는 padded 이미지 좌표 → letterbox 파라미터(r, dw, dh)로 model input 좌표로 옮긴다.
`align_metric` / `overlaps` 는 `get_box_metrics` 를 감싸 읽는다(반환값 불변).

## provenance

`get_topk_index` 래퍼로 anchor flat index 만 기록. 검증: 매핑된 anchor 의
`sigmoid(class_logit)` 이 final confidence 와 일치해야 한다.

## 그룹 (REAL/SYNTH 동일 정의)

```
R+ / S+   IoU>=0.5 후보 중 conf 최고
RW / SW   같은 프레임의 IoU<0.5 후보 중 conf 최고
RANKFAIL  correct 가 있는데 wrong 의 logit 이 더 큰 프레임
ASSIGNED  fg_mask=1 인 anchor 중 target 최대 (assigner 가 실제로 고른 anchor)
```

## 주의

P3/P4/P5 는 순차 layer 가 아니다 — level 별로 분리해 보고 level 끼리는 "어디가 더 나쁜가"만 본다.
oracle level offset 은 real DEV 에 fit 한 **진단 상한**이며 배포 성능이 아니다.
