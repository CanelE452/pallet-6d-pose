# LOSS PATH AUDIT — local ultralytics 8.4.60

경로: `.../site-packages/ultralytics/utils/loss.py`, `.../nn/tasks.py`, `.../nn/modules/head.py`.
아래 이름은 **실제 local source 에서 확인한 것**이며 추측이 아니다 `[확인]`.

## 실제 텐서 이름과 위치

```
classification logits      pred_scores          loss.py:403-406  (b, A, nc)  permute 후
classification target      target_scores        loss.py:422      assigner 반환
target_scores_sum          target_scores_sum    loss.py:431      = max(target_scores.sum(), 1)
표준 cls loss              loss[1]              loss.py:434-437  bce(pred_scores, target_scores).sum()/tss
cls gain 적용 위치         loss.py:454          loss[1] *= self.hyp.cls
foreground mask            fg_mask              loss.py:422      (b, A) bool
matched GT index           target_gt_idx        loss.py:422
predicted keypoints        pred_kpts            loss.py:820,834  (b, A, nkpt, dim)  kpts_decode 후
GT keypoints (선택된 것)   selected_keypoints   loss.py:950 -> gt_kpt = selected[masks]
keypoint validity          kpt_mask             loss.py:958      gt_kpt[...,2] != 0
Pose26 RLE sigma           preds["kpts_sigma"]  loss.py:835-838  -> pred_kpt[..., -2:]
one2many / one2one         E2ELoss              loss.py:1169-1191
                           one2many = loss_fn(model, tal_topk=10)
                           one2one  = loss_fn(model, tal_topk=7, tal_topk2=1)
criterion 생성             PoseModel.init_criterion  tasks.py:695-697
                           return E2ELoss(self, PoseLoss26)
```

## normalized error `e` — 새로 만들지 않는다

`KeypointLoss.forward` (loss.py:327-330):

```
d = (pred_x - gt_x)^2 + (pred_y - gt_y)^2
e = d / ((2 * sigmas)^2 * (area + 1e-9) * 2)          # cocoeval 형태
loss = mean( kpt_loss_factor * (1 - exp(-e)) * kpt_mask )
```

`area` 는 `xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1)` (loss.py:955), stride 로 나눈 뒤다.
`sigmas` 는 stock `self.keypoint_loss.sigmas` 를 그대로 쓴다 — **임의 sigma table 없음**.

## Y1 구현 방식 — stock 코드를 복사하지 않는다

`pallet_yolo_loss/posecls.py` 의 `PoseAwareClsLoss26(PoseLoss26)`:

1. `get_assigned_targets_and_loss` 안에서 `self.bce` 를 일시적으로 감싸 **stock 이 실제로
   넘기는** `pred_scores` / `target_scores` 를 포착한다 (loss.py:434 에서 정확히 한 번 호출).
   반환값은 그대로 통과시키므로 표준 cls loss 는 한 글자도 안 바뀐다.
2. `calculate_keypoints_loss` 안에서 `self.keypoint_loss` 를 감싸 **stock 이 실제로 넘기는**
   `pred_kpt / gt_kpt / kpt_mask / area` 로 위 식의 `e` 를 다시 계산하고
   `q_pose = mean_valid(exp(-e))` 를 `no_grad` + `.detach()` 로 저장한다.
   반환값은 stock 그대로라 keypoint loss 도 안 바뀐다.
3. `loss()` 에서 super 를 호출한 뒤 alignment 항을 **cls 칸(loss[3])에만** 더한다.

이 방식이면 upstream 이 바뀌어도 수식이 갈라지지 않는다 — stock 이 쓰는 바로 그 텐서를
읽기 때문이다. `mh_arms.heads_from_f50` 이 wiring test 로 고정된 것과 같은 원칙이다.

## scale 을 두 번 곱하지 않는다

표준 cls 는 `loss[1] *= self.hyp.cls` 로 이미 곱해져서 `loss[3]` 자리에 들어온다.
alignment 항은 `self.hyp.cls * LAMBDA * L_align` 로 **같은 classification scale 한 번만**
곱해 더한다. global loss 에는 추가 weight 를 곱하지 않는다.
분모는 표준 cls 와 같은 `target_scores_sum` 을 쓴다 (표준은 전체 anchor 합, alignment 는
foreground 합 — 분모를 공유해야 두 항의 상대 크기가 안정적이다).

## one2many / one2one

`E2ELoss` 가 `loss_fn` 을 **두 번 독립 생성**하므로, 각 criterion 인스턴스가 자기
`fg_mask / target_gt_idx / target_scores / pred_kpts` 만 본다. 복사·공유 없음 `[확인]`.
`GRADIENT_ROUTING.json` 에서 두 branch 모두 `reached: true` 로 확인했다.
