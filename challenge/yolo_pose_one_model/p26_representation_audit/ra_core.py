"""Y0 frozen instrumentation — candidate provenance + one2one classification path taps.

★ fuse 하지 않는다 (Pose26.fuse 가 module 을 지운다).
★ 값은 한 개도 바꾸지 않는다 — forward hook 으로 읽기만 하고, get_topk_index 는 결과를
  그대로 통과시키며 idx 만 기록한다.  M0 parity 로 검증한다.
"""
from __future__ import annotations
import os

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/p26_representation_audit"
W = f"{Y}/runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt"
PAD, CONF, IOU_NMS, MAX_DET, IMGSZ = 100, 0.001, 0.7, 300, 640
LEVEL_NAME = {0: "P3", 1: "P4", 2: "P5"}


class Instrumented:
    """YOLO(W) 를 fuse 없이 로드하고 one2one cls/pose tower 에 hook 을 건다."""

    def __init__(self, hooks: bool = True):
        from ultralytics import YOLO
        from ultralytics.nn.modules.head import Detect, Pose, Pose26
        for cls in (Detect, Pose, Pose26):          # fuse 로 module 이 사라지지 않게
            cls.fuse = lambda self: None
        self.yolo = YOLO(W, task="pose")
        self.model = self.yolo.model
        self.head = self.model.model[-1]
        self.nl = self.head.nl
        self.cap = {}
        self.idx = None
        self.topk_out = None
        self._handles = []
        if hooks:
            self._install()
        self._wrap_topk()

    # -- hooks ------------------------------------------------------------
    def _install(self):
        def mk(name):
            def fn(mod, inp, out):
                self.cap[name] = out.detach()
                if name.endswith("cls1"):
                    self.cap[name.replace("cls1", "neck_in")] = inp[0].detach()
            return fn
        for i in range(self.nl):
            lv = LEVEL_NAME[i]
            self._handles.append(self.head.one2one_cv3[i][0].register_forward_hook(mk(f"{lv}_cls1")))
            self._handles.append(self.head.one2one_cv3[i][1].register_forward_hook(mk(f"{lv}_cls_pen")))
            self._handles.append(self.head.one2one_cv3[i][2].register_forward_hook(mk(f"{lv}_logit")))
            self._handles.append(self.head.one2one_cv4[i].register_forward_hook(mk(f"{lv}_pose_pen")))

    def _wrap_topk(self):
        head = self.head
        orig = head.get_topk_index

        def wrapped(scores, max_det):
            out = orig(scores, max_det)
            self.idx = out[2].detach()          # (B, k, 1) anchor flat index — 값 변경 없음
            return out
        head.get_topk_index = wrapped
        self._orig_topk = orig

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- geometry ---------------------------------------------------------
    def level_sizes(self):
        """flat index -> (level, y, x) 를 풀기 위한 각 level 의 (H, W)."""
        out = []
        for i in range(self.nl):
            t = self.cap.get(f"{LEVEL_NAME[i]}_logit")
            out.append((int(t.shape[-2]), int(t.shape[-1])))
        return out

    def decode_flat(self, flat: int):
        sizes = self.level_sizes()
        off = 0
        for i, (h, w) in enumerate(sizes):
            n = h * w
            if flat < off + n:
                r = flat - off
                return i, r // w, r % w
            off += n
        raise IndexError(flat)

    def vectors(self, flat: int):
        """해당 source cell 의 tap vector 들. neck_in 차원은 level 마다 다르다."""
        lv_i, y, x = self.decode_flat(flat)
        lv = LEVEL_NAME[lv_i]
        out = {"level": lv, "level_idx": lv_i, "y": int(y), "x": int(x)}
        for tap in ("neck_in", "cls1", "cls_pen", "logit", "pose_pen"):
            t = self.cap.get(f"{lv}_{tap}")
            out[tap] = t[0, :, y, x].float().cpu().numpy() if t is not None else None
        return out

    # -- inference --------------------------------------------------------
    def predict(self, img_bgr, pad=PAD):
        import cv2
        p = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101) if pad else img_bgr
        self.cap.clear(); self.idx = None
        r = self.yolo.predict(p, conf=CONF, imgsz=IMGSZ, iou=IOU_NMS, max_det=MAX_DET,
                              device=0, verbose=False)[0]
        return r


def iou_xyxy(a, b):
    xx = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    yy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = xx * yy
    return i / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i, 1e-9)


def map_final_to_flat(inst, conf_arr):
    """final detection 행 -> anchor flat index.

    end2end NMS 는 postprocess 출력(top-k, score 내림차순)에서 conf 필터 + max_det 만 한다.
    따라서 살아남은 행의 순서가 보존된다 — conf 배열 일치로 검증한다.
    """
    if inst.idx is None:
        return None
    flat = inst.idx[0, :, 0].cpu().numpy()          # (k,) score 내림차순
    n = len(conf_arr)
    if n == 0:
        return np.array([], dtype=int)
    if n > len(flat):
        return None
    return flat[:n].astype(int)
