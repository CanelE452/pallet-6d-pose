"""paper_s1_dump_predictions.py — Paper-S1 belief-peak 예측 키포인트를 JSON 으로 덤프.

paper_s1_allframes_overlays.py 는 예측점(pred8/centroid/peak)을 계산해 jpg 에만
그리고 좌표는 저장하지 않는다. 이 스크립트는 동일 모델·전처리·추출 로직을
재사용(overlay 모듈 import)해 프레임별 예측 키포인트 픽셀좌표를 JSON 으로 남긴다.

★ overlay 와 완전히 동일: weights net_epoch_0065, reflect-pad100, THRESH 0.3,
  belief-peak(PnP 없음), 원본 이미지 픽셀좌표.
출력: data/pallet/eval_results/s1_allframes_overlays/s1_predictions.json
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import time

import numpy as np
import cv2
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
OV_PATH = os.path.join(ROOT, "scripts/stage0/paper_s1_allframes_overlays.py")

_spec = importlib.util.spec_from_file_location("ov", OV_PATH)
ov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ov)   # top-level: sys.path 세팅 + E 모듈 로드

OUT_JSON = os.path.join(ov.OUT_ROOT, "s1_predictions.json")


def infer_kps(model, ip, device):
    """raw 이미지 -> (kps8 list[[x,y]|None], centroid [x,y]|None, peaks list[float|None])."""
    img = cv2.imread(ip)
    if img is None:
        return None
    H, W = img.shape[:2]
    proc = ov.E.pad_frame(img, ov.PAD)
    tensor, nw, nh, sc = ov.preprocess(proc)
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps = ov.extract_keypoints_from_belief(belief, ov.THRESH)  # 9 x (wx,wy,peak)

    pts8, peaks = [], []
    for k in kps[:8]:
        if k[0] < 0:
            pts8.append(None)
            peaks.append(None)
        else:
            xy = ov.E.belief_to_orig_pad(k[0], k[1], bw, bh, nw, nh, sc,
                                         ov.PAD, W, H)
            pts8.append([float(xy[0]), float(xy[1])])
            peaks.append(float(k[2]))
    centroid = None
    if kps[8][0] >= 0:
        c = ov.E.belief_to_orig_pad(kps[8][0], kps[8][1], bw, bh, nw, nh, sc,
                                    ov.PAD, W, H)
        centroid = [float(c[0]), float(c[1])]
    return pts8, centroid, peaks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ov.E.load_model(ov.WEIGHTS, device)
    print(f"[model] {ov.WEIGHTS}  device={device}", flush=True)

    frames = ov.gather()
    total = sum(len(v) for v in frames.values())
    print(f"[set] total={total}", flush=True)

    predictions = {}
    t0 = time.time()
    done = 0
    bad = []
    for sess, paths in frames.items():
        sd = {}
        for ip in paths:
            fid = os.path.splitext(os.path.basename(ip))[0]
            done += 1
            r = infer_kps(model, ip, device)
            if r is None:
                bad.append(f"{sess}/{fid}")
                continue
            pts8, centroid, peaks = r
            n_det = sum(p is not None for p in pts8)
            mean_peak = (float(np.mean([p for p in peaks if p is not None]))
                         if n_det else 0.0)
            sd[fid] = {
                "kps8": pts8,          # 원본 이미지 픽셀좌표, 미검출=null
                "centroid": centroid,  # index 8, 미검출=null
                "peaks": peaks,        # 코너별 belief peak, 미검출=null
                "n_det": n_det,
                "mean_peak": round(mean_peak, 4),
            }
            if done % 500 == 0:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (total - done) / rate if rate else 0
                print(f"[{done}/{total}] {rate:.1f}f/s ETA={eta/60:.1f}m",
                      flush=True)
        predictions[sess] = sd
        print(f"[done] {sess}: {len(sd)} frames", flush=True)

    out = {
        "meta": {
            "weights": ov.WEIGHTS,
            "preprocess": f"reflect-pad{ov.PAD}",
            "thresh": ov.THRESH,
            "method": "belief-peak (no PnP), original-image pixel coords",
            "convention": "camera-facing 0123: front 0-3, rear 4-7, "
                          "{0,1,4,5}=top / {2,3,6,7}=bottom, centroid=index8",
            "coords": "kps8/centroid = [x,y] in original raw image pixels; "
                      "null = not detected",
            "note": "GT 없음(raw). 예측점만. paper_s1_allframes_overlays.py 의 "
                    "jpg 오버레이와 동일 모델·전처리로 산출.",
            "n_frames": total,
        },
        "predictions": predictions,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f)
    print(f"\n[save] {OUT_JSON}", flush=True)
    print(f"  frames={total} bad={len(bad)} "
          f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
