# SOURCE AUDIT — 설치본 ultralytics 8.4.60 의 실제 TAL

경로: `utils/tal.py`(TaskAlignedAssigner), `utils/loss.py`(PoseLoss26 / E2ELoss / assigner 생성).
아래 식은 **설치본에서 그대로 옮긴 것**이며 공식 문서·기억에서 가져오지 않았다 `[확인]`.

## assigner 생성

```
utils/loss.py:1174-1175  E2ELoss
    one2many = loss_fn(model, tal_topk=10)
    one2one  = loss_fn(model, tal_topk=7, tal_topk2=1)

utils/loss.py:360-367    v8DetectionLoss.__init__
    TaskAlignedAssigner(topk=tal_topk, num_classes=nc, alpha=0.5, beta=6.0,
                        stride=..., topk2=tal_topk2)
```

## alignment metric (tal.py:159-190)

```
bbox_scores[b,g,a] = pd_scores[b,a,cls_g]          # sigmoid 확률 (assigner 에 sigmoid 후 전달)
overlaps[b,g,a]    = bbox_iou(gt, pd, xywh=False, CIoU=True).clamp_(0)
align_metric       = bbox_scores^alpha * overlaps^beta        alpha=0.5, beta=6.0
```

## positive 선택 (tal.py:133-157, 317-355)

```
mask_in_gts = anchor center 가 GT box 내부
mask_topk   = align_metric 상위 topk
mask_pos    = mask_topk * mask_in_gts * mask_gt
if topk2 != topk:                                   # one2one 은 topk=7, topk2=1
    mask_pos *= (align_metric 상위 topk2 위치)       # -> GT 당 anchor 1 개
fg_mask     = mask_pos.sum(-2)
```

## target score (tal.py:265-287, 122-129)

```
target_scores = one_hot(target_label)               # 0/1
target_scores = where(fg_mask > 0, target_scores, 0)
pos_align_metrics = (align_metric*mask_pos).amax(-1, keepdim)
pos_overlaps      = (overlaps*mask_pos).amax(-1, keepdim)
norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + eps)).amax(-2)
target_scores    *= norm_align_metric
```

즉 **assigned anchor 의 target = norm_align_metric ∈ [0,1]**, 나머지는 정확히 0.

## classification loss normalization (loss.py:431-437, 454)

```
target_scores_sum = max(target_scores.sum(), 1)
loss[1] = bce(pred_scores, target_scores).sum() / target_scores_sum
loss[1] *= hyp.cls
```

anchor 전체(P3+P4+P5 concat)에 대한 **단일 합/단일 분모**다. level 별 항이 없다.

## 여섯 질문에 대한 답

```
1. one2one 의 classification target 은?
   assigned anchor 에서 norm_align_metric (align_metric·pos_overlaps/pos_align_metrics 의 amax),
   그 외 anchor 는 0.                                                        [확인]

2. box IoU 가 target score 에 직접 들어가는가?
   YES. align_metric = score^0.5 * CIoU^6.0 이고 normalization 에도 pos_overlaps 가 곱해진다.
   지수 6.0 이므로 IoU 가 지배적이다.                                        [확인]

3. keypoint quality 가 classification target 에 들어가는가?
   NO. TAL 은 pd_scores 와 box CIoU 만 본다. keypoint 는 loss[1](pose)·loss[2](kobj)·
   loss[5](rle) 로 따로 학습되며 target_scores 에 어떤 경로로도 들어가지 않는다. [확인]

4. one2one 에서 GT 당 몇 anchor 가 positive 인가?
   topk=7 로 후보를 고른 뒤 topk2=1 로 다시 걸러 **정확히 1 개**.              [확인]

5. non-assigned high-confidence candidate 의 cls target 은?
   정확히 0 (`torch.where(fg_scores_mask > 0, target_scores, 0)`).           [확인]

6. P3/P4/P5 가 동일 cls objective / normalization 을 공유하는가?
   YES. assigner 는 concat 된 전체 anchor 위에서 한 번 돌고, BCE 는 전체 합을
   `target_scores_sum` 하나로 나눈다. level 별 가중·정규화가 없다.            [확인]
```

## 이 audit 이 하지 않는 것

`model.train()` 호출 안 함(BN state 불변), optimizer 생성 안 함, backward 안 함,
fuse 안 함. assigner 는 **진단 목적으로만** GT 를 넣어 실행한다.
