"""diag3_photometric.py — 트랙B 진단 3: 동일프레임 photometric intervention(인과 강).
outside(현재 검출되는) 프레임을 그대로 두고 photometric만 night-like로 단계 변환
(P0 원본 → P1 노출 → P2 +shot/read noise → P3 +blur/WB → P4 +glare/clip),
재추론해 어느 열화에서 night식 no-response가 재현되나 본다. geometry/배경/GT 고정.
+ 저조도 지표(real night vs outside ROI)로 P1 노출 타깃 보정.
학습 없음. base=challenge0123. diag2의 classify/inference 재사용.
"""
import os, sys, json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from eval_pvnet_heads import load_pvnet_model, preprocess, belief_to_orig  # noqa
from four_arm_pl_compare import collect_val_frames  # noqa
from diag2_raw_decode_stages import (classify, gt_to_belief, disk_max,  # noqa
                                     GT_DISK, TAU0, N_DET_MIN)

WEIGHTS = os.path.join(ROOT, "weights/challenge0123/final_net_epoch_0060.pth")
OUT = os.path.join(ROOT, "data/pallet/eval_results/stage9_diag")


# ── sRGB <-> linear ──
def srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def roi_box(gt8, H, W, pad=4):
    x0 = max(0, int(gt8[:, 0].min()) - pad); x1 = min(W, int(gt8[:, 0].max()) + pad)
    y0 = max(0, int(gt8[:, 1].min()) - pad); y1 = min(H, int(gt8[:, 1].max()) + pad)
    return x0, y0, x1, y1


def luma(bgr):
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


def lowlight_metrics(img, gt8):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = roi_box(gt8, H, W)
    if x1 <= x0 or y1 <= y0:
        return None
    f = img.astype(np.float32) / 255.0
    Y = luma(f)
    roi = Y[y0:y1, x0:x1]
    # surrounding ring
    rx0, ry0 = max(0, x0 - (x1 - x0) // 2), max(0, y0 - (y1 - y0) // 2)
    rx1, ry1 = min(W, x1 + (x1 - x0) // 2), min(H, y1 + (y1 - y0) // 2)
    ring = Y[ry0:ry1, rx0:rx1]
    gray = (roi * 255).astype(np.uint8)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var() if gray.size else 0.0
    return {"luma_p10": float(np.percentile(roi, 10)),
            "luma_p50": float(np.percentile(roi, 50)),
            "dark_ratio": float(np.mean(roi < 0.1)),
            "obj_bg_contrast": float(np.median(roi) - np.median(ring)),
            "grad_energy": float(lap),
            "highlight_clip": float(np.mean(roi > 0.95))}


# ── photometric degradation pipeline (linear-RGB) ──
def degrade(img, level, exp, rng):
    """img BGR uint8 -> degraded BGR uint8. level 0..4."""
    if level == 0:
        return img.copy()
    f = img.astype(np.float32) / 255.0
    lin = srgb2lin(f)
    lin = lin * exp                                   # P1 exposure ↓
    if level >= 2:                                    # shot + read noise (linear)
        sigma = np.sqrt(np.maximum(0.0008 * lin, 0) + 0.002 ** 2)
        lin = lin + rng.normal(0, 1, lin.shape).astype(np.float32) * sigma
    if level >= 3:                                    # blur + WB shift
        lin = np.clip(lin, 0, None)
        out = lin2srgb(lin)
        out = cv2.GaussianBlur((out * 255).astype(np.uint8), (0, 0), 1.2).astype(np.float32) / 255.0
        lin = srgb2lin(out)
        lin[..., 0] *= 1.15; lin[..., 2] *= 0.9      # cooler (B up, R down) BGR
    if level >= 4:                                    # local glare + highlight clip
        H, W = lin.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = int(W * 0.7), int(H * 0.4)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.12 * W) ** 2))
        lin = lin + (g[..., None] * 0.8).astype(np.float32)
    return (lin2srgb(lin) * 255).clip(0, 255).astype(np.uint8)


def infer_classify(model, device, img, gt8, gt_ctr):
    import torch
    t, nw, nh, sc = preprocess(img); t = t.to(device)
    with torch.no_grad():
        beliefs, _ = model(t)
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    gts = list(gt8) + [gt_ctr]
    raw_pred = np.full((9, 2), np.nan); score = np.zeros(9); gtlocal = np.zeros(9)
    margins = []
    for k in range(9):
        ch = belief[k]
        yi, xi = np.unravel_index(int(ch.argmax()), ch.shape)
        score[k] = float(ch[yi, xi])
        raw_pred[k] = belief_to_orig(xi, yi, bw, bh, nw, nh, sc)
        gbx, gby = gt_to_belief(gts[k], sc, bw, bh, nw, nh)
        gl = disk_max(ch, gbx, gby, GT_DISK); gtlocal[k] = gl
        ch2 = ch.copy()
        bx0, bx1 = max(0, int(gbx) - GT_DISK), min(bw, int(gbx) + GT_DISK + 1)
        by0, by1 = max(0, int(gby) - GT_DISK), min(bh, int(gby) + GT_DISK + 1)
        ch2[by0:by1, bx0:bx1] = 0
        if k < 8:
            margins.append(gl - float(ch2.max()))     # GT-local - best off-GT
    corner_err = np.array([np.linalg.norm(raw_pred[i] - gt8[i]) for i in range(8)])
    r = {"score": score, "gtlocal": gtlocal,
         "corner_err": corner_err, "center_err": float(np.linalg.norm(raw_pred[8] - gt_ctr))}
    cls, n_final = classify(r)
    return cls, n_final, float(np.median(corner_err)), float(np.median(margins))


def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_pvnet_model(WEIGHTS, device, numVec=0, numSeg=0)
    frames = collect_val_frames()

    # 1) 저조도 지표: night vs outside ROI
    ll = {"outside": [], "night": []}
    store = {"outside": [], "night": []}
    for dom, fid, jp, ip in frames:
        if dom not in ll:
            continue
        d = json.load(open(jp)); o = d["objects"][0]
        gt8 = np.array(o["projected_cuboid"], float)[:8]
        gt_ctr = np.array(o["projected_cuboid_centroid"], float)
        img = cv2.imread(ip)
        if img is None:
            continue
        m = lowlight_metrics(img, gt8)
        if m:
            ll[dom].append(m)
        store[dom].append((ip, gt8, gt_ctr, img))

    L = ["TRACK-B DIAG 3 — 저조도 지표 + 동일프레임 photometric intervention",
         f"weights={WEIGHTS}"]
    L.append("\n[저조도 지표 ROI(GT bbox) median — night vs outside]")
    keys = ["luma_p10", "luma_p50", "dark_ratio", "obj_bg_contrast",
            "grad_energy", "highlight_clip"]
    L.append(f"  {'metric':<16}{'outside':>10}{'night':>10}")
    med = {}
    for dom in ("outside", "night"):
        med[dom] = {k: float(np.median([x[k] for x in ll[dom]])) for k in keys}
    for k in keys:
        L.append(f"  {k:<16}{med['outside'][k]:>10.3f}{med['night'][k]:>10.3f}")

    # exposure factor: night/outside luma_p50 in linear
    exp = float(srgb2lin(np.array(med['night']['luma_p50']))
                / max(1e-4, srgb2lin(np.array(med['outside']['luma_p50']))))
    L.append(f"\n  → P1 exposure factor (linear night/outside luma_p50) = {exp:.3f}")

    # 2) 동일프레임 intervention: outside 중 P0 검출되는 프레임만
    rng = np.random.RandomState(0)
    levels = [0, 1, 2, 3, 4]
    agg = {lv: {"n": 0, "noresp": 0, "detected": 0, "errs": [], "margins": []}
           for lv in levels}
    used = 0
    for ip, gt8, gt_ctr, img in store["outside"]:
        cls0, nf0, _, _ = infer_classify(model, device, img, gt8, gt_ctr)
        if cls0 != "detected":
            continue                      # P0서 검출되는 것만(열화 가시화)
        used += 1
        for lv in levels:
            di = degrade(img, lv, exp, rng)
            cls, nf, em, mg = infer_classify(model, device, di, gt8, gt_ctr)
            a = agg[lv]; a["n"] += 1
            a["noresp"] += (cls == "no_response")
            a["detected"] += (cls == "detected")
            a["errs"].append(em); a["margins"].append(mg)
    L.append(f"\n[동일프레임 night-like 변환] outside P0-detected {used}장")
    L.append(f"  {'level':<22}{'det%':>7}{'noResp%':>9}{'rawErrMed':>11}{'GTmargin':>10}")
    names = {0: "P0 원본", 1: "P1 노출↓", 2: "P2 +noise",
             3: "P3 +blur/WB", 4: "P4 +glare/clip"}
    for lv in levels:
        a = agg[lv]
        if not a["n"]:
            continue
        L.append(f"  {names[lv]:<22}{100*a['detected']/a['n']:>6.0f}%"
                 f"{100*a['noresp']/a['n']:>8.0f}%"
                 f"{np.median(a['errs']):>11.1f}{np.median(a['margins']):>10.3f}")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(OUT, "diag3_photometric.txt"), "w").write(txt)
    print(f"\n[save] {OUT}/diag3_photometric.txt")


if __name__ == "__main__":
    main()
