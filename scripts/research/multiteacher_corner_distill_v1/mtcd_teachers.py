"""동결 teacher 추론 — 모든 teacher 가 같은 recipe 를 탄다.

recipe 는 `INFERENCE_REPLAY_LOCK.json` 을 그대로 따른다:
    PAD 100 BORDER_REFLECT_101 -> imgsz 640 -> 추론 -> 좌표에서 PAD 를 뺀다

DOPE 는 640 대신 448 을 쓴다(학습 imagesize).  PAD 100 은 동일하다 —
DOPE 학습셋 PNG 가 이미 PAD 100 이 적용된 캔버스이기 때문이다
(`backbone_dope_final_v1/bd_convert.py`: 720x480 --pad100--> 920x680,
 라벨 = projected_cuboid + 100).  real 이미지를 PAD 없이 넣으면 학습 때와
프레이밍이 달라진다 — memory `dope-inference-needs-reflect-padding` 의 교훈.

출력 스키마는 `frozen_arm_prediction_v1` 과 같다.  그래야 기존 pose evaluator 를
한 줄도 고치지 않고 쓸 수 있다.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import mtcd_common as M

PAD_PX = 100
PAD_BORDER = cv2.BORDER_REFLECT_101
YOLO_IMGSZ = 640
YOLO_CONF = 0.001
DOPE_IMGSZ = 448
DOPE_THRESHOLD = 0.3
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def pad_image(image: np.ndarray) -> np.ndarray:
    return cv2.copyMakeBorder(image, PAD_PX, PAD_PX, PAD_PX, PAD_PX, PAD_BORDER)


# ------------------------------------------------------------------- YOLO ---
def load_yolo(weights: Path):
    from ultralytics import YOLO
    return YOLO(str(weights), task="pose")


def infer_yolo(model, image: np.ndarray, device: str = "0",
               already_padded: bool = False) -> dict:
    """already_padded=True 는 합성 학습셋 PNG 처럼 캔버스에 PAD 가 구워진 경우다."""
    padded = image if already_padded else pad_image(image)
    shift = 0.0 if already_padded else float(PAD_PX)
    result = model.predict(padded, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, augment=False,
                           half=False, device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return {"status": "NO_DETECTION"}
    scores = result.boxes.conf.detach().cpu().numpy()
    best = int(np.argmax(scores))
    box = result.boxes.xyxy.detach().cpu().numpy()[best] - shift
    keypoints = keypoint_conf = None
    if result.keypoints is not None:
        keypoints = result.keypoints.xy.detach().cpu().numpy()[best] - shift
        if result.keypoints.conf is not None:
            keypoint_conf = result.keypoints.conf.detach().cpu().numpy()[best]
    return {
        "status": "OK",
        "box_xyxy": box.tolist(),
        "box_conf": float(scores[best]),
        "keypoints_xy": keypoints.tolist() if keypoints is not None else None,
        "keypoints_conf": keypoint_conf.tolist() if keypoint_conf is not None else None,
        "detections": int(len(scores)),
        "box_source": "detector",
    }


# ------------------------------------------------------------------- DOPE ---
def load_dope(weights: Path, device: str = "cuda:0"):
    import sys
    sys.path.insert(0, str(M.REPO_ROOT / "Deep_Object_Pose" / "common"))
    import torch
    from models import DopeNetwork
    model = DopeNetwork()
    state = torch.load(str(weights), map_location=device)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def _belief_peaks(belief: np.ndarray, threshold: float = DOPE_THRESHOLD):
    """filter_pr_camfacing.extract_keypoints_from_belief 와 같은 정의."""
    from scipy.ndimage import gaussian_filter
    offset, ran = 0.4395, 5
    out = []
    for i in range(belief.shape[0]):
        bmap = belief[i]
        if bmap.max() < threshold:
            out.append((-1.0, -1.0, float(bmap.max())))
            continue
        sm = gaussian_filter(bmap, sigma=2)
        pl = np.zeros_like(sm); pl[1:, :] = sm[:-1, :]
        pr = np.zeros_like(sm); pr[:-1, :] = sm[1:, :]
        pu = np.zeros_like(sm); pu[:, 1:] = sm[:, :-1]
        pd = np.zeros_like(sm); pd[:, :-1] = sm[:, 1:]
        peaks = (sm >= pl) & (sm >= pr) & (sm >= pu) & (sm >= pd) & (sm > threshold)
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            out.append((-1.0, -1.0, float(bmap.max())))
            continue
        best = int(np.argmax([bmap[y, x] for y, x in zip(ys, xs)]))
        px, py = int(xs[best]), int(ys[best])
        y0, y1 = max(0, py - ran), min(bmap.shape[0], py + ran + 1)
        x0, x1 = max(0, px - ran), min(bmap.shape[1], px + ran + 1)
        patch = bmap[y0:y1, x0:x1]
        if patch.sum() > 0:
            xg, yg = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            wx = float(np.average(xg, weights=patch)) + offset
            wy = float(np.average(yg, weights=patch)) + offset
        else:
            wx, wy = float(px), float(py)
        out.append((wx, wy, float(bmap.max())))
    return out


def infer_dope(model, image: np.ndarray, device: str = "cuda:0",
               already_padded: bool = False) -> dict:
    """belief peak 를 원본(=PAD 제거) 이미지 픽셀로 되돌린다.

    already_padded=True 는 합성 학습셋 PNG 처럼 캔버스에 PAD 가 이미 구워진 경우다.
    그 경우 PAD 를 더하지도 빼지도 않는다.
    """
    import torch
    padded = image if already_padded else pad_image(image)
    height, width = padded.shape[:2]
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (DOPE_IMGSZ, DOPE_IMGSZ)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(((resized - IMAGENET_MEAN) / IMAGENET_STD)
                              .transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        belief_stages, _ = model(tensor)
    belief = belief_stages[-1][0].detach().cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    sx, sy = bw / width, bh / height          # belief 격자 <- padded 캔버스
    shift = 0.0 if already_padded else float(PAD_PX)

    peaks = _belief_peaks(belief)
    keypoints, confs = [], []
    for wx, wy, conf in peaks:
        if wx < 0:
            keypoints.append([float("nan"), float("nan")])
        else:
            keypoints.append([wx / sx - shift, wy / sy - shift])
        confs.append(float(conf))
    corners = np.asarray(keypoints[:8], dtype=np.float64)
    usable = np.isfinite(corners).all(axis=1)
    if usable.sum() < 4:
        return {"status": "NO_DETECTION", "keypoints_conf": confs}
    valid = corners[usable]
    box = [float(valid[:, 0].min()), float(valid[:, 1].min()),
           float(valid[:, 0].max()), float(valid[:, 1].max())]
    return {
        "status": "OK",
        "box_xyxy": box,
        "box_conf": float(np.mean(confs[:8])),
        "keypoints_xy": keypoints,
        "keypoints_conf": confs,
        "detections": 1,
        "box_source": "corner_hull",       # DOPE 에는 detector box 가 없다
    }
