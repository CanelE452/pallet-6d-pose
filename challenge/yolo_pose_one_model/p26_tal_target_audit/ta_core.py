"""TAL diagnostic — assigner 를 GT 와 함께 실행해 target_score 를 읽는다.

★ backward 0 · optimizer 0 · model.train() 0 · fuse 0 · 가중치 변경 0.
"""
from __future__ import annotations
import json, os

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/p26_tal_target_audit"
W = f"{Y}/runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt"
PAD, CONF, IOU_NMS, MAX_DET, IMGSZ = 100, 0.001, 0.7, 640, 640
MAX_DET = 300
LEVEL = {0: "P3", 1: "P4", 2: "P5"}


class TALProbe:
    def __init__(self):
        from ultralytics import YOLO
        from ultralytics.nn.modules.head import Detect, Pose, Pose26
        for c in (Detect, Pose, Pose26):
            c.fuse = lambda self: None
        self.yolo = YOLO(W, task="pose")
        self.model = self.yolo.model
        self.model.eval()                      # train() 호출 안 함 — BN state 불변
        self.head = self.model.model[-1]
        self.cap = {}
        self._hook = self.model.register_forward_hook(self._grab)
        self._topk_idx = None
        orig = self.head.get_topk_index

        def wrapped(scores, max_det):
            out = orig(scores, max_det)
            self._topk_idx = out[2].detach()
            return out
        self.head.get_topk_index = wrapped
        self.crit = None
        self.align = {}

    def _grab(self, mod, inp, out):
        self.cap["x"] = inp[0].detach()
        if isinstance(out, tuple) and isinstance(out[1], dict):
            self.cap["preds"] = out[1]

    def criterion(self):
        if self.crit is None:
            self.crit = self.model.init_criterion()          # E2ELoss(PoseLoss26)
            a = self.crit.one2one.assigner
            orig = a.get_box_metrics

            def wrapped(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
                am, ov = orig(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)
                self.align["align_metric"] = am.detach()
                self.align["overlaps"] = ov.detach()
                return am, ov
            a.get_box_metrics = wrapped
        return self.crit

    # ---------------------------------------------------------------- infer
    def predict(self, img_bgr, pad=PAD):
        import cv2
        p = (cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
             if pad else img_bgr)
        self.cap.clear(); self._topk_idx = None; self.align.clear()
        r = self.yolo.predict(p, conf=CONF, imgsz=IMGSZ, iou=IOU_NMS, max_det=MAX_DET,
                              device=0, verbose=False)[0]
        return r, p

    def letterbox_params(self, padded_shape):
        """captured input tensor 와 padded 원본 shape 로 r/dw/dh 복원."""
        h, w = self.cap["x"].shape[-2:]
        h0, w0 = padded_shape[:2]
        r = min(h / h0, w / w0)
        nw, nh = round(w0 * r), round(h0 * r)
        return r, (w - nw) / 2.0, (h - nh) / 2.0, (h, w)

    def gt_to_input(self, gt_xyxy, padded_shape):
        r, dw, dh, _ = self.letterbox_params(padded_shape)
        g = np.asarray(gt_xyxy, float)
        return np.array([g[0]*r + dw, g[1]*r + dh, g[2]*r + dw, g[3]*r + dh])

    # ---------------------------------------------------------------- TAL
    @torch.no_grad()
    def assign(self, gt_xyxy_input):
        """assigner 를 diagnostic 으로 실행. 반환 target_scores/fg_mask/align/overlaps."""
        from ultralytics.utils.tal import make_anchors
        c = self.criterion().one2one
        pd = self.cap["preds"]["one2one"]
        pred_distri = pd["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = pd["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(pd["feats"], c.stride, 0.5)
        imgsz = torch.tensor(pd["feats"][0].shape[2:], device=pred_scores.device,
                             dtype=pred_scores.dtype) * c.stride[0]
        g = torch.tensor(gt_xyxy_input, dtype=pred_scores.dtype, device=pred_scores.device)
        cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        bw, bh = (g[2] - g[0]), (g[3] - g[1])
        norm = torch.tensor([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], device=g.device,
                            dtype=g.dtype)
        tgt = torch.tensor([[0.0, 0.0, 0, 0, 0, 0]], device=g.device, dtype=g.dtype)
        tgt[0, 2:] = torch.stack([cx, cy, bw, bh]) / norm
        targets = c.preprocess(tgt, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = c.bbox_decode(anchor_points, pred_distri)
        _, tb, target_scores, fg_mask, target_gt_idx = c.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        return {"target_scores": target_scores[0, :, 0].cpu().numpy(),
                "fg_mask": fg_mask[0].cpu().numpy(),
                "target_gt_idx": target_gt_idx[0].cpu().numpy(),
                "align_metric": self.align["align_metric"][0, 0].cpu().numpy(),
                "overlaps": self.align["overlaps"][0, 0].cpu().numpy(),
                "logits": pd["scores"][0, 0].cpu().numpy(),
                "n_anchors": int(pred_scores.shape[1])}

    def level_of(self, flat):
        sizes = [(int(t.shape[-2]), int(t.shape[-1]))
                 for t in [self.cap["preds"]["one2one"]["feats"][i] for i in range(3)]]
        off = 0
        for i, (h, w) in enumerate(sizes):
            if flat < off + h * w:
                return LEVEL[i], i
            off += h * w
        return None, None

    def final_flat(self, n):
        if self._topk_idx is None:
            return None
        f = self._topk_idx[0, :, 0].cpu().numpy()
        return f[:n].astype(int) if n <= len(f) else None


def iou_xyxy(a, b):
    xx = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    yy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = xx * yy
    return i / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i, 1e-9)
