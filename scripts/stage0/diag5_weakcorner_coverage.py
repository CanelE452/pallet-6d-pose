"""diag5_weakcorner_coverage.py — 트랙B: weak-corner failure attribution + G1/G2/G3.
diag4의 28%p 잔존 detection gap을 (1) matched-pair failure-type 분해,
(2) weak-corner 통계(weakest id front/back, weak_count), (3) 제외 night(G2) vs
matched night(G1) vs outside(G3) 로 분해. 1 inference. diag4 features/매칭 + diag2 classify 재사용.
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
from diag2_raw_decode_stages import classify, gt_to_belief, disk_max, GT_DISK  # noqa
from diag4_geometry_matching import features, MATCH_VARS, CALIPER  # noqa

WEIGHTS = os.path.join(ROOT, "weights/challenge0123/final_net_epoch_0060.pth")
OUT = os.path.join(ROOT, "data/pallet/eval_results/stage9_diag")
BACK = [4, 5, 6, 7]
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def per_frame(model, device, ip, gt8, gt_ctr):
    import torch
    img = cv2.imread(ip)
    if img is None:
        return None
    H, W = img.shape[:2]
    t, nw, nh, sc = preprocess(img); t = t.to(device)
    with torch.no_grad():
        beliefs, _ = model(t)
    belief = beliefs[-1][0].cpu().numpy(); bh, bw = belief.shape[1], belief.shape[2]
    gts = list(gt8) + [gt_ctr]
    score = np.zeros(9); gtlocal = np.zeros(9); raw = np.full((9, 2), np.nan)
    inframe = np.zeros(8, bool)
    for k in range(9):
        ch = belief[k]; yi, xi = np.unravel_index(int(ch.argmax()), ch.shape)
        score[k] = float(ch[yi, xi]); raw[k] = belief_to_orig(xi, yi, bw, bh, nw, nh, sc)
        gbx, gby = gt_to_belief(gts[k], sc, bw, bh, nw, nh)
        gtlocal[k] = disk_max(ch, gbx, gby, GT_DISK)
        if k < 8:
            inframe[k] = (0 <= gts[k][0] < W) and (0 <= gts[k][1] < H)
    cerr = np.array([np.linalg.norm(raw[i] - gt8[i]) for i in range(8)])
    cls, _ = classify({"score": score, "gtlocal": gtlocal, "corner_err": cerr,
                       "center_err": float(np.linalg.norm(raw[8] - gt_ctr))})
    feat = features(gt8, gt_ctr, H, W)
    idx = np.where(inframe)[0]
    # border distance (in-frame corners) + shortest edge
    bdist = min((min(gts[i][0], W - gts[i][0], gts[i][1], H - gts[i][1]) for i in idx),
                default=0.0)
    emin = min(np.linalg.norm(gt8[a] - gt8[b]) for a, b in EDGES)
    gl_in = gtlocal[idx]
    weakest = idx[int(np.argmin(gl_in))] if len(idx) else -1
    return {**feat, "cls": cls, "inframe_idx": idx.tolist(),
            "gtlocal_in": gl_in.tolist(), "B": float(bdist), "Emin": float(emin),
            "weakest_id": int(weakest), "weakest_back": int(weakest in BACK),
            "final_det": int(cls == "detected"),
            "no_response": int(cls == "no_response"),
            "competing": int(cls == "competing_peak"),
            "corner_loc": int(cls == "corner_localization")}


def vecs(rs, norm):
    return [np.array([(r[v] - norm[v][0]) / norm[v][1] for v in MATCH_VARS]) for r in rs]


def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_pvnet_model(WEIGHTS, device, numVec=0, numSeg=0)
    R = {"outside": [], "night": []}
    for dom, fid, jp, ip in collect_val_frames():
        if dom not in R:
            continue
        d = json.load(open(jp)); o = d["objects"][0]
        gt8 = np.array(o["projected_cuboid"], float)[:8]
        gt_ctr = np.array(o["projected_cuboid_centroid"], float)
        r = per_frame(model, device, ip, gt8, gt_ctr)
        if r:
            r["fid"] = fid; R[dom].append(r)

    # weak threshold = detected frames in-frame gtlocal p5
    detsc = [g for r in (R["outside"] + R["night"]) if r["final_det"]
             for g in r["gtlocal_in"]]
    wt = float(np.percentile(detsc, 5)) if detsc else 0.3
    for r in R["outside"] + R["night"]:
        r["weak_count"] = int(np.sum(np.array(r["gtlocal_in"]) < wt))

    # 매칭 (diag4 동일)
    pooled = R["outside"] + R["night"]; norm = {}
    for v in MATCH_VARS:
        vals = np.array([r[v] for r in pooled]); md = np.median(vals)
        iqr = (np.percentile(vals, 75) - np.percentile(vals, 25)) or 1.0
        norm[v] = (md, iqr)
    O, Nn = R["outside"], R["night"]; Ov = vecs(O, norm)
    used = set(); pairs = []
    cand = sorted((min(np.linalg.norm(nv - ov) for ov in Ov), ni)
                  for ni, nv in enumerate(vecs(Nn, norm)))
    Nv = vecs(Nn, norm)
    for dist, ni in cand:
        if dist > CALIPER:
            continue
        order = np.argsort([np.linalg.norm(Nv[ni] - Ov[oi]) for oi in range(len(O))])
        for oi in order:
            if oi not in used and np.linalg.norm(Nv[ni] - Ov[oi]) <= CALIPER:
                used.add(oi); pairs.append((ni, oi)); break
    matched_ni = {ni for ni, _ in pairs}
    G1 = [Nn[ni] for ni in matched_ni]
    G2 = [Nn[ni] for ni in range(len(Nn)) if ni not in matched_ni]
    G3 = O

    L = [f"TRACK-B DIAG 5 — weak-corner attribution + G1/G2/G3  (weak_thr=p5={wt:.3f})",
         f"n_out={len(O)} n_night={len(Nn)} matched_pairs={len(pairs)}"]
    cats = ["detected", "no_response", "competing", "corner_loc"]
    # (1) matched-pair failure-type risk difference
    L.append("\n[① matched 22쌍 failure-type paired risk diff (night-outside %p)]")
    for c in cats:
        nk = "final_det" if c == "detected" else c
        nn = 100*np.mean([Nn[ni][nk] for ni, _ in pairs])
        oo = 100*np.mean([O[oi][nk] for _, oi in pairs])
        L.append(f"  {c:<14} night {nn:>5.0f}%  out {oo:>5.0f}%  Δ {nn-oo:>+5.0f}%p")
    # (2) G1/G2/G3
    L.append("\n[② G1 matched-night / G2 excluded-night / G3 outside]")
    L.append(f"  {'grp':<14}{'N':>4}{'det%':>6}{'noResp%':>8}{'comp%':>7}"
             f"{'cornLoc%':>9}{'medT':>6}{'medA':>7}{'wkCnt':>6}{'wkBack%':>8}")
    for name, g in [("G1 matched-N", G1), ("G2 excluded-N", G2), ("G3 outside", G3)]:
        if not g:
            continue
        L.append(f"  {name:<14}{len(g):>4}{100*np.mean([r['final_det'] for r in g]):>5.0f}%"
                 f"{100*np.mean([r['no_response'] for r in g]):>7.0f}%"
                 f"{100*np.mean([r['competing'] for r in g]):>6.0f}%"
                 f"{100*np.mean([r['corner_loc'] for r in g]):>8.0f}%"
                 f"{np.median([r['T'] for r in g]):>6.2f}{np.median([r['A'] for r in g]):>7.2f}"
                 f"{np.median([r['weak_count'] for r in g]):>6.0f}"
                 f"{100*np.mean([r['weakest_back'] for r in g]):>7.0f}%")
    # (3) weakest corner front/back + final_det vs weak_count
    L.append("\n[③ weakest corner & weak_count vs detection]")
    for name, g in [("night-all", Nn), ("outside", O)]:
        det = [r for r in g if r["final_det"]]; nd = [r for r in g if not r["final_det"]]
        L.append(f"  {name:<10} weakBack%={100*np.mean([r['weakest_back'] for r in g]):.0f}"
                 f"  weakCnt(det)={np.mean([r['weak_count'] for r in det]) if det else float('nan'):.1f}"
                 f"  weakCnt(miss)={np.mean([r['weak_count'] for r in nd]) if nd else float('nan'):.1f}")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(OUT, "diag5_weakcorner.txt"), "w").write(txt)
    print(f"\n[save] {OUT}/diag5_weakcorner.txt")


if __name__ == "__main__":
    main()
