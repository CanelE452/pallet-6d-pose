"""diag4_geometry_matching.py — 트랙B 진단 4.
GT geometry(조명무관)만 매칭변수로 night↔outside 1:1 NN 매칭 후, geometry 통제
상태에서도 night raw-response gap이 남는지 검정. 학습 X.

매칭변수: A=log(hull/HW), S=median|b-f|/sqrt(hull), T=truncation, V=in-frame kp,
         P=log(front_face_area/back_face_area).  (photometric은 매칭변수 금지)
outcome(연속, 프레임단위, in-frame corner만): local_score_med/q25, margin_med, raw_corner_med
        (보조 binary: no_response/competing/final_det)
common-support→robust-norm→1:1 NN(no replacement)→SMD before/after→Wilcoxon/McNemar.
"""
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os, sys, json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from eval_pvnet_heads import load_pvnet_model, preprocess, belief_to_orig  # noqa
from four_arm_pl_compare import collect_val_frames  # noqa
from diag2_raw_decode_stages import classify, gt_to_belief, disk_max, GT_DISK  # noqa

WEIGHTS = os.path.join(ROOT, "weights/challenge0123/final_net_epoch_0060.pth")
OUT = os.path.join(ROOT, "data/pallet/eval_results/stage9_diag")
DEPTH_PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]
FRONT = [0, 1, 2, 3]; BACK = [4, 5, 6, 7]
CALIPER = 2.0     # 정규화공간 nearest 거리 caliper (초과 night 제외)
MATCH_VARS = ["A", "S", "T", "V", "P"]


def poly_area(pts):
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))


def features(gt8, gt_ctr, H, W):
    hull = max(poly_area(gt8), 1.0)
    seps = [np.linalg.norm(gt8[b] - gt8[f]) for f, b in DEPTH_PAIRS]
    allpts = np.vstack([gt8, gt_ctr])
    inb = ((allpts[:, 0] >= 0) & (allpts[:, 0] < W) &
           (allpts[:, 1] >= 0) & (allpts[:, 1] < H))
    fa = max(poly_area(gt8[FRONT]), 1.0); ba = max(poly_area(gt8[BACK]), 1.0)
    return {"A": float(np.log(hull / (H * W))),
            "S": float(np.median(seps) / np.sqrt(hull)),
            "T": float(1.0 - inb.sum() / 9.0),
            "V": float(inb[:8].sum()),
            "P": float(np.log(fa / ba))}


def outcomes(model, device, ip, gt8, gt_ctr, H, W):
    import torch
    img = cv2.imread(ip)
    if img is None:
        return None
    t, nw, nh, sc = preprocess(img); t = t.to(device)
    with torch.no_grad():
        beliefs, _ = model(t)
    belief = beliefs[-1][0].cpu().numpy(); bh, bw = belief.shape[1], belief.shape[2]
    gts = list(gt8) + [gt_ctr]
    score = np.zeros(9); gtlocal = np.zeros(9); offgt = np.zeros(9)
    raw = np.full((9, 2), np.nan)
    inframe = np.zeros(8, bool)
    for k in range(9):
        ch = belief[k]
        yi, xi = np.unravel_index(int(ch.argmax()), ch.shape)
        score[k] = float(ch[yi, xi]); raw[k] = belief_to_orig(xi, yi, bw, bh, nw, nh, sc)
        gbx, gby = gt_to_belief(gts[k], sc, bw, bh, nw, nh)
        gtlocal[k] = disk_max(ch, gbx, gby, GT_DISK)
        ch2 = ch.copy()
        bx0, bx1 = max(0, int(gbx) - GT_DISK), min(bw, int(gbx) + GT_DISK + 1)
        by0, by1 = max(0, int(gby) - GT_DISK), min(bh, int(gby) + GT_DISK + 1)
        ch2[by0:by1, bx0:bx1] = 0
        offgt[k] = float(ch2.max())
        if k < 8:
            inframe[k] = (0 <= gts[k][0] < W) and (0 <= gts[k][1] < H)
    cerr = np.array([np.linalg.norm(raw[i] - gt8[i]) for i in range(8)])
    idx = np.where(inframe)[0]
    if len(idx) == 0:
        return None
    cls, _ = classify({"score": score, "gtlocal": gtlocal,
                       "corner_err": cerr, "center_err": float(np.linalg.norm(raw[8] - gt_ctr))})
    gl = gtlocal[idx]; mg = (gtlocal - offgt)[idx]
    return {"local_med": float(np.median(gl)),
            "local_q25": float(np.percentile(gl, 25)),
            "margin_med": float(np.median(mg)),
            "raw_corner_med": float(np.median(cerr[idx])),
            "no_response": int(cls == "no_response"),
            "competing": int(cls == "competing_peak"),
            "final_det": int(cls == "detected")}


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = np.sqrt((a.var() + b.var()) / 2) + 1e-9
    return (a.mean() - b.mean()) / sd


def main():
    import torch
    from scipy.stats import wilcoxon
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_pvnet_model(WEIGHTS, device, numVec=0, numSeg=0)

    recs = {"outside": [], "night": []}
    for dom, fid, jp, ip in collect_val_frames():
        if dom not in recs:
            continue
        d = json.load(open(jp)); o = d["objects"][0]
        gt8 = np.array(o["projected_cuboid"], float)[:8]
        gt_ctr = np.array(o["projected_cuboid_centroid"], float)
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        feat = features(gt8, gt_ctr, H, W)
        oc = outcomes(model, device, ip, gt8, gt_ctr, H, W)
        if oc is None:
            continue
        recs[dom].append({"fid": fid, **feat, **oc})

    L = ["TRACK-B DIAG 4 — geometry-matched night vs outside",
         f"weights={WEIGHTS}  n_outside={len(recs['outside'])} n_night={len(recs['night'])}",
         f"매칭변수={MATCH_VARS} (photometric 제외)  caliper={CALIPER}"]

    # robust-normalize over pooled
    pooled = recs["outside"] + recs["night"]
    norm = {}
    for v in MATCH_VARS:
        vals = np.array([r[v] for r in pooled])
        md = np.median(vals); iqr = (np.percentile(vals, 75) - np.percentile(vals, 25)) or 1.0
        norm[v] = (md, iqr)
    def vec(r):
        return np.array([(r[v] - norm[v][0]) / norm[v][1] for v in MATCH_VARS])

    # SMD before (전체)
    L.append("\n[balance SMD]   (|SMD|<0.2 목표)")
    L.append(f"  {'var':<4}{'night_med':>10}{'out_med':>9}{'SMD_before':>12}{'SMD_after':>11}")

    # 1:1 NN without replacement (greedy by nearest dist)
    O = recs["outside"]; Nn = recs["night"]
    Ovec = [vec(r) for r in O]
    pairs = []; used = set()
    cand = []
    for ni, nr in enumerate(Nn):
        d = [np.linalg.norm(vec(nr) - ov) for ov in Ovec]
        cand.append((min(d), ni, int(np.argmin(d))))
    for dist, ni, _ in sorted(cand):
        if dist > CALIPER:
            continue
        nv = vec(Nn[ni])
        order = np.argsort([np.linalg.norm(nv - Ovec[oi]) for oi in range(len(O))])
        for oi in order:
            if oi not in used and np.linalg.norm(nv - Ovec[oi]) <= CALIPER:
                used.add(oi); pairs.append((ni, oi)); break

    for v in MATCH_VARS:
        nb = [r[v] for r in Nn]; ob = [r[v] for r in O]
        if pairs:
            na = [Nn[ni][v] for ni, _ in pairs]; oa = [O[oi][v] for _, oi in pairs]
            sa = smd(na, oa)
        else:
            sa = float("nan")
        L.append(f"  {v:<4}{np.median(nb):>10.2f}{np.median(ob):>9.2f}"
                 f"{smd(nb, ob):>12.2f}{sa:>11.2f}")

    L.append(f"\n[matching] night usable={len(Nn)} → matched pairs={len(pairs)} "
             f"(caliper {CALIPER} 내)")
    if len(pairs) < 10:
        L.append("  ⚠ matched night < 10 → common support 부족, domain effect 식별 불가.")
    # matched outcome 비교
    L.append("\n[matched outcome] night vs outside (paired, night-outside Δ, Wilcoxon p)")
    L.append(f"  {'outcome':<16}{'night_med':>10}{'out_med':>9}{'Δmed':>8}{'p':>8}")
    for key in ("local_med", "local_q25", "margin_med", "raw_corner_med"):
        nv = np.array([Nn[ni][key] for ni, _ in pairs])
        ov = np.array([O[oi][key] for _, oi in pairs])
        try:
            p = wilcoxon(nv, ov).pvalue if len(pairs) >= 6 else float("nan")
        except Exception:
            p = float("nan")
        L.append(f"  {key:<16}{np.median(nv):>10.3f}{np.median(ov):>9.3f}"
                 f"{np.median(nv - ov):>8.3f}{p:>8.3f}")
    # binary McNemar (no_response, final_det)
    L.append("\n[matched binary] night vs outside (discordant b/c, McNemar p)")
    for key in ("no_response", "competing", "final_det"):
        nv = [Nn[ni][key] for ni, _ in pairs]; ov = [O[oi][key] for _, oi in pairs]
        b = sum(1 for x, y in zip(nv, ov) if x == 1 and y == 0)  # night yes, out no
        c = sum(1 for x, y in zip(nv, ov) if x == 0 and y == 1)
        from scipy.stats import binomtest
        p = binomtest(b, b + c).pvalue if (b + c) > 0 else float("nan")
        L.append(f"  {key:<14} night_rate={np.mean(nv):.2f} out_rate={np.mean(ov):.2f}"
                 f"  b(night+/out-)={b} c={c}  p={p:.3f}")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(OUT, "diag4_geometry_match.txt"), "w").write(txt)
    json.dump({"pairs": pairs, "outside": O, "night": Nn},
              open(os.path.join(OUT, "diag4_records.json"), "w"), indent=2, default=str)
    print(f"\n[save] {OUT}/diag4_geometry_match.txt")


if __name__ == "__main__":
    main()
