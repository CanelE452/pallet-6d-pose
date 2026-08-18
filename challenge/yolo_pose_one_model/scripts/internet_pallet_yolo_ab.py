"""internet_pallet_data 에 YOLO26n base vs finetune A/B — 처음 본 팔레트 일반화 확인.

`scripts/stage0/internet_pallet_infer.py`(s2 전용) 의 두 로직을 그대로 따른다:
  - 파일명에서 치수(mm) 파싱 → PnP dims
  - K 미지(인터넷 사진) → HFOV 스윕으로 reproj 최소 focal 채택
    (고정 HFOV 가정은 PnP 가 K 오차를 포즈 기울임으로 흡수해 큐보이드가 깔때기로 왜곡됨)
그 스크립트는 DOPE(pallet-pose env) 전용이라 import 하지 못해 두 함수만 복제했다.

★ 누수 없음: 이 10 장은 base(합성 전용)·ft(real 400+neg 259) 어느 쪽 학습셋에도 없다
  (runs_ft/PURPOSE.md 포함 목록에 internet 없음).
★ 절대 스케일/깊이는 여전히 미지 — 정성 비교용이다. GT 가 없으므로 kp 오차도 못 낸다.

사용: <pallet-yolo26 python> challenge/yolo_pose_one_model/scripts/internet_pallet_yolo_ab.py
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "annotate"))
import annotate_pnp as APNP  # noqa: E402

SRC = os.path.join(ROOT, "data/pallet/raw_data/internet_pallet_data")
OUT = os.path.join(ROOT, "data/pallet/eval_results/internet_pallet_yolo_ab")
PAD = 100
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
MODELS = [
    ("base(synth)", "challenge/yolo_pose_one_model/final/"
                    "pallet_yolo26n_pose_640_b32_final.pt"),
    ("ft(real+neg)", "challenge/yolo_pose_one_model/runs_ft/"
                     "ft_a_real157_neg259_synth12k/weights/best.pt"),
]


def parse_dims_m(name):
    """파일명 -> ((W,D,H) m, h_default). 무게(kg, 2자리)는 범위로 자동 제외."""
    nums = [int(n) for n in re.findall(r"\d+", name)]
    dims = [n for n in nums if 100 <= n <= 2500]
    if len(dims) < 2:
        return None
    W, D = dims[0], dims[1]
    if len(dims) >= 3:
        H, hdef = dims[2], False
    else:
        H, hdef = 120, True
    return (W / 1000.0, D / 1000.0, H / 1000.0), hdef


def K_from_hfov(w, h, hfov):
    fx = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1]], np.float64)


def predict(model, img, conf=0.25):
    inp = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    r = model.predict(inp, verbose=False, conf=conf, imgsz=640)[0]
    if r.boxes is None or not len(r.boxes):
        return None, None, 0.0
    b = int(np.argmax(r.boxes.conf.cpu().numpy()))
    kp = r.keypoints.data.cpu().numpy()[b].copy()
    kp[:, 0] -= PAD
    kp[:, 1] -= PAD
    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        if kp[i, 2] >= 0.5:
            pred8[i] = kp[i, :2]
    pred_c = [float(kp[8, 0]), float(kp[8, 1])] if kp[8, 2] >= 0.5 else None
    return pred8, pred_c, float(r.boxes.conf[b])


def solve_with_fsearch(pred8, pred_c, dims, shape):
    """HFOV 스윕 → reproj 최소 focal 채택. 반환 (pose, hfov, reproj)."""
    kps9 = [None if not np.isfinite(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps9.append(pred_c)
    if sum(1 for k in kps9[:8] if k is not None) < 6:
        return None, None, None
    h, w = shape[:2]
    best = (None, None, np.inf)
    for hfov in np.arange(30.0, 121.0, 2.5):
        K = K_from_hfov(w, h, hfov)
        try:
            pose = APNP.solve_pose(kps9, K, dims=dims, img_shape=shape)
        except Exception:
            continue
        if pose is None:
            continue
        e = float(pose["reproj_error_px"])
        if e < best[2]:
            best = (pose, float(hfov), e)
    return best


def draw(img, pred8, pred_c, pose, dims, hfov, tag, conf, dims_txt):
    vis = img.copy()
    h, w = img.shape[:2]
    if pose is not None:
        K = K_from_hfov(w, h, hfov)
        # solve_pose(auto_swap_dims=True) 는 as-given 과 W/D swap 을 모두 풀고 reproj 가
        # 낮은 쪽을 고른다. 그 (R,t) 는 pose["dims"] 에만 맞으므로 파일명 dims 로 상자를
        # 다시 만들면 pose 와 다른 모양을 그리게 된다. 2026-08-18 진단에서 이 이미지의
        # 큐보이드가 39px(최대 82px) 어긋난 원인이 이것이었다 — 헤더의 reproj 5.5px 는
        # 풀린 pose 값이라 그림과 숫자가 따로 놀았다.
        X = APNP.make_pallet_keypoints_3d(*pose["dims"])[:8]
        R = np.asarray(pose["R"], float)
        t = np.asarray(pose["t"], float).ravel()
        P = (R @ X.T).T + t
        if (P[:, 2] > 1e-6).all():
            uv = np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2],
                           K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]], 1)
            Q = [(int(round(u)), int(round(v))) for u, v in uv]
            th = max(2, w // 400)
            for a, b in EDGES:
                cv2.line(vis, Q[a], Q[b], (0, 0, 255), th, cv2.LINE_AA)
    r = max(4, w // 220)
    for i in range(8):
        if np.isfinite(pred8[i, 0]):
            cv2.circle(vis, (int(round(pred8[i, 0])), int(round(pred8[i, 1]))),
                       r, (255, 0, 0), -1, cv2.LINE_AA)
    if pred_c:
        cv2.circle(vis, (int(pred_c[0]), int(pred_c[1])), r, (255, 255, 255),
                   -1, cv2.LINE_AA)
    n = int(np.isfinite(pred8[:, 0]).sum())
    ok = pose is not None
    l1 = (f"{tag}   det {n}/8   conf {conf:.2f}   "
          + (f"PnP hfov{hfov:.0f} reproj{pose['reproj_error_px']:.1f}px"
             if ok else "PnP FAIL"))
    hh = max(30, w // 22)
    bar = np.zeros((hh * 2, w, 3), np.uint8)
    fs = w / 1400.0
    cv2.putText(bar, l1, (6, int(hh * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, 255, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, dims_txt, (6, int(hh * 1.6)), cv2.FONT_HERSHEY_SIMPLEX,
                fs * 0.8, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def main():
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(SRC, "*"))
                   if os.path.splitext(f)[1].lower() in
                   (".jpg", ".jpeg", ".png", ".webp"))
    models = [(t, YOLO(os.path.join(ROOT, w), task="pose")) for t, w in MODELS]
    print(f"{'file':<46}{'model':<14}{'det':>5}{'conf':>7}{'hfov':>7}{'reproj':>9}")
    print("-" * 90)
    for fp in files:
        name = os.path.basename(fp)
        img = cv2.imread(fp)
        if img is None:
            continue
        pd_ = parse_dims_m(name)
        if pd_ is None:
            print(f"{name[:44]:<46}  치수 파싱 실패 — 건너뜀")
            continue
        dims, hdef = pd_
        dims_txt = (f"dims from filename = {dims[0]:.3f} x {dims[1]:.3f} x "
                    f"{dims[2]:.3f} m" + ("  (H 기본값)" if hdef else "")
                    + "   |  blue=keypoint  red=PnP cuboid  (K 추정, 절대깊이 미지)")
        tiles = []
        for tag, m in models:
            p8, pc, conf = predict(m, img)
            if p8 is None:
                p8, pc = np.full((8, 2), np.nan), None
            pose, hfov, rep = solve_with_fsearch(p8, pc, dims, img.shape)
            tiles.append(draw(img, p8, pc, pose, dims, hfov, tag, conf, dims_txt))
            n = int(np.isfinite(p8[:, 0]).sum())
            print(f"{name[:44]:<46}{tag:<14}{n:>3}/8{conf:>7.2f}"
                  + (f"{hfov:>7.0f}{rep:>9.2f}" if pose is not None
                     else f"{'-':>7}{'FAIL':>9}"))
        H = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
                 for t in tiles]
        stem = re.sub(r"[^\w가-힣]+", "_", os.path.splitext(name)[0])[:60]
        cv2.imwrite(os.path.join(OUT, f"{stem}.jpg"), np.hstack(tiles),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
