"""_diag_kfov_overlay.py — DIAGNOSTIC ONLY. 현재 HFOV60(깔때기) vs 최적 HFOV(교정) 큐보이드 비교 저장.

Usage: conda activate pallet-pose; python scripts/stage0/_diag_kfov_overlay.py
산출: data/pallet/eval_results/paper_s2_scratch_diffpnp/_diag_kfov_overlay_{tag}.jpg
"""
from __future__ import annotations
import math
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

import paper_s2_testset17_9filters as T   # noqa: E402
import internet_pallet_infer as IPI       # noqa: E402
import annotate_pnp as APNP               # noqa: E402
import cv2                                # noqa: E402
import torch                              # noqa: E402

M = T.M
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EDGES = M.EDGES
OUT = os.path.join(ROOT, "data/pallet/eval_results/paper_s2_scratch_diffpnp")
TARGETS = {"09": ("중고파렛트(18kg) [1100*1100*150mm] .png", 35),
           "08": ("수출용파렛트(1000×1000×120)4.jpg", 45)}


def K_for(w, h, hfov):
    fx = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1]], np.float64)


def draw(img, pred8, pred_c, pose, hfov, reproj):
    im = img.copy()
    pa = np.array(pose["projected_all"], float)[:8]
    bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
    pa[bad] = np.nan
    for a, b in EDGES:
        if not (np.isnan(pa[a, 0]) or np.isnan(pa[b, 0])):
            cv2.line(im, (int(pa[a, 0]), int(pa[a, 1])), (int(pa[b, 0]), int(pa[b, 1])),
                     (0, 0, 255), 2, cv2.LINE_AA)
    for i in range(8):
        if not np.isnan(pred8[i, 0]):
            p = (int(pred8[i, 0]), int(pred8[i, 1]))
            cv2.circle(im, p, 5, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(im, str(i), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
    im = cv2.copyMakeBorder(im, 46, 4, 4, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(im, f"HFOV={hfov}deg  reproj={reproj:.1f}px  tilt={pose['_v8_tilt']:.2f}",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def main():
    model = T.E.load_model(T.WEIGHTS, DEV)
    for tag, (name, best_hfov) in TARGETS.items():
        fp = os.path.join(IPI.SRC, name)
        img = cv2.imread(fp)
        dims, _ = IPI.parse_dims_m(name)
        APNP.PALLET_DIMS = dims
        belief, geom, wh = M.infer_belief(model, img, DEV, IPI.PAD)
        pred8, pred_c, _, _ = M.belief_to_pred(belief, geom, wh, IPI.PAD, IPI.THRESH)
        kps9 = [None if np.isnan(pred8[i, 0]) else [float(pred8[i, 0]), float(pred8[i, 1])]
                for i in range(8)]
        kps9.append(list(pred_c) if pred_c is not None else None)
        panels = []
        for hfov in (60, best_hfov):
            K = K_for(img.shape[1], img.shape[0], hfov)
            pose = APNP.solve_pose(kps9, K, dims=dims, img_shape=img.shape)
            panels.append(draw(img, pred8, pred_c, pose, hfov, pose["reproj_error_px"]))
        h = max(p.shape[0] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 8, cv2.BORDER_CONSTANT,
                                     value=(20, 20, 20)) for p in panels]
        out = os.path.join(OUT, f"_diag_kfov_overlay_{tag}.jpg")
        cv2.imwrite(out, np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[{tag}] saved {out}")


if __name__ == "__main__":
    main()
