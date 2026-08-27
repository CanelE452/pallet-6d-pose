"""M0/M1/M2 inference-path patch — 학습 0, 가중치 불변.

stock source 를 복사하지 않는다. Pose26.forward 를 대체하되 **stock 이 쓰는 바로 그
메서드**(`forward_head` / `_inference` / `postprocess`)만 다시 조합한다.

M0  P26_E2E      one2one -> _inference -> postprocess            (정본과 동일)
M1  P26_O2M_RAW  one2many -> _inference -> postprocess           (같은 top-k rule, NMS 없음)
M2  P26_O2M_NMS  one2many -> _inference -> xyxy2xywh -> stock NMS(end2end=False)
"""
from __future__ import annotations

import torch

MODE = None
_INSTALLED = {"n": 0}
CALLS = {"o2m_forward": 0, "nms_forced": 0}


def install(mode: str):
    """MODE 를 걸고 필요한 최소 지점만 patch 한다."""
    global MODE
    assert mode in ("E2E", "O2M_RAW", "O2M_NMS"), mode
    MODE = mode
    from ultralytics.nn.modules.head import Detect, Pose, Pose26
    from ultralytics.utils import ops
    from ultralytics.utils import nms as NMS

    # (1) fuse 가 one2many head 를 지우지 않게 한다. Conv+BN fusion 은 그대로 일어난다.
    for cls in (Detect, Pose, Pose26):
        cls.fuse = lambda self: None

    # (2) forward — stock 메서드만 재조합
    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        if MODE == "E2E":
            src = preds["one2one"] if self.end2end else preds
            y = self._inference(src)
            if self.end2end:
                y = self.postprocess(y.permute(0, 2, 1))
            return y if self.export else (y, preds)
        # --- one2many 경로 ---
        CALLS["o2m_forward"] += 1
        src = preds["one2many"] if self.end2end else preds
        y = self._inference(src)                       # (B, 4+nc+nk, A), box 는 xyxy
        if MODE == "O2M_RAW":
            return self.postprocess(y.permute(0, 2, 1))
        # O2M_NMS: stock NMS 가 xywh 를 기대하므로 정확히 역변환해 넘긴다
        y = y.clone()
        y[:, :4] = ops.xyxy2xywh(y[:, :4].permute(0, 2, 1)).permute(0, 2, 1)
        return y

    Pose26.forward = forward

    # (3) O2M_NMS 만 stock NMS 의 end2end 플래그를 False 로 강제한다.
    #     conf / iou / max_det / nc 는 predictor args 그대로.
    if MODE == "O2M_NMS":
        orig = NMS.non_max_suppression

        def patched(prediction, *a, **kw):
            if kw.get("end2end", False):
                kw["end2end"] = False
                CALLS["nms_forced"] += 1
            return orig(prediction, *a, **kw)

        NMS.non_max_suppression = patched
        import ultralytics.models.yolo.detect.predict as DP
        DP.nms.non_max_suppression = patched
    _INSTALLED["n"] += 1
    return {"mode": MODE, "installed": _INSTALLED["n"]}
