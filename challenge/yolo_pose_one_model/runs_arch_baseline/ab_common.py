"""공통 — pretrained 감사 / synthetic 9kp / efficiency.  새 solver·threshold 0."""
from __future__ import annotations
import hashlib, json, os, sys, time

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/runs_arch_baseline"
DS = f"{Y}/datasets/g38_generic_only"
PRE = f"{ROOT}/challenge/weights/pretrained_yolo"

MODELS = {
    "B8":  {"label": "YOLOv8n-Pose",  "init": f"{PRE}/yolov8n-pose.pt",
            "run": "AB_G38_B8_YOLOV8N_30EP_SEED42"},
    "B11": {"label": "YOLO11n-Pose",  "init": f"{PRE}/yolo11n-pose.pt",
            "run": "AB_G38_B11_YOLO11N_30EP_SEED42"},
    "Y0":  {"label": "YOLO26n-Pose",  "init": f"{PRE}/yolo26n-pose.pt",
            "run": None,
            "weights": f"{Y}/runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt"},
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def model_stats(weights):
    """params / GFLOPs — ultralytics 자체 계산을 쓴다 (새 정의 안 만듦)."""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    m = YOLO(weights, task="pose")
    params = int(sum(p.numel() for p in m.model.parameters()))
    try:
        gflops = float(get_flops(m.model, imgsz=640))
    except Exception:
        gflops = None
    del m
    return params, gflops


def latency(weights, n_warm=30, n_run=200):
    """batch1 / imgsz640 / RTX3080, warmup 후 median·p90·FPS."""
    import torch, cv2
    from ultralytics import YOLO
    m = YOLO(weights, task="pose")
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(n_warm):
        m.predict(img, imgsz=640, device=0, verbose=False)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        m.predict(img, imgsz=640, device=0, verbose=False)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    del m
    torch.cuda.empty_cache()
    a = np.array(ts)
    return {"n_warmup": n_warm, "n_run": n_run,
            "latency_ms_median": float(np.median(a)),
            "latency_ms_p90": float(np.percentile(a, 90)),
            "FPS_from_median": float(1000.0 / np.median(a)),
            "note": "end-to-end predict() 왕복 (전처리+추론+postprocess). "
                    "v8/11 은 NMS 포함, 26 은 end2end — 이 차이가 latency 에 들어있다."}


def load_gt(label_path, w, h):
    """YOLO pose label -> (box_xyxy_px, kpts_px(9,2), vis(9,))"""
    line = open(label_path).read().strip().split("\n")[0].split()
    v = [float(x) for x in line]
    cx, cy, bw, bh = v[1:5]
    box = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    k = np.array(v[5:]).reshape(-1, 3)
    return box, np.stack([k[:, 0] * w, k[:, 1] * h], 1), k[:, 2]


def iou_xyxy(a, b):
    xx = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    yy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = xx * yy
    return i / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i, 1e-9)


def synth_9kp(weights, tag, limit=None):
    """G38 val 1,998 에서 real 과 **같은 정의**의 9kp median/p90.

    real 쪽과 다른 점은 padding 뿐 — synthetic val 은 이미 PAD100 캔버스라 추가 pad 없음.
    """
    out_p = f"{NS}/SYNTH9KP_{tag}.json"
    if os.path.exists(out_p):
        return json.load(open(out_p))
    import cv2, torch
    from ultralytics import YOLO
    m = YOLO(weights, task="pose")
    files = sorted(os.listdir(f"{DS}/images/val"))
    if limit:
        files = files[:limit]
    errs, ncb, n = [], 0, 0
    for f in files:
        ip = f"{DS}/images/val/{f}"
        lp = f"{DS}/labels/val/{os.path.splitext(f)[0]}.txt"
        im = cv2.imread(ip)
        if im is None or not os.path.exists(lp):
            continue
        h, w = im.shape[:2]
        gtb, gtk, vis = load_gt(lp, w, h)
        r = m.predict(im, conf=0.001, imgsz=640, device=0, verbose=False)[0]
        n += 1
        if r.boxes is None or len(r.boxes) == 0:
            continue
        cf = r.boxes.conf.cpu().numpy()
        i = int(np.argmax(cf))                       # top-1 by box conf, real 과 동일
        box = r.boxes.xyxy.cpu().numpy()[i]
        if iou_xyxy(box, gtb) < 0.5:                 # correct_box, real 과 동일
            continue
        ncb += 1
        kp = r.keypoints.xy.cpu().numpy()[i]
        d = np.linalg.norm(kp - gtk, axis=1)
        errs.append(d[vis > 0] if (vis > 0).any() else d)
    del m
    torch.cuda.empty_cache()
    e = np.concatenate(errs) if errs else np.array([])
    out = {"tag": tag, "weights": os.path.relpath(weights, ROOT), "n_frames": n,
           "n_correct_box": ncb, "cbox": ncb / max(n, 1),
           "kp_median": float(np.median(e)) if e.size else None,
           "kp_p90": float(np.percentile(e, 90)) if e.size else None,
           "gross20": float((e > 20).mean()) if e.size else None,
           "definition": ("real evaluator 와 동일: top-1 by box conf, correct_box IoU>=0.5, "
                          "keypoint L2 px. synthetic val 은 이미 PAD100 캔버스라 추가 pad 없음."),
           "n_kpt_used": int(e.size)}
    json.dump(out, open(out_p, "w"), indent=2, ensure_ascii=False)
    return out
