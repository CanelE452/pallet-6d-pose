"""diag6_synthetic_coverage.py — 트랙B: H1(coverage 부족) vs H2(sim2real) vs H3(model limit).
real G1/G2/G3 의 geometry 가 synthetic TRAIN(4.9만 sample) support 안에 있나(d20, threshold는
synthetic held-out val 분포로 보정) + 그 geometry 에서 synthetic held-out 이 실제로 잘 되나
(local detection). 0) validity audit(corners off-frame=task-def). 학습 X.
synthetic train = mixed_v8+v1+v2 (challenge0123 학습셋) → coverage only(추론 안 함=암기회피).
synthetic held-out = training_data/val (challenge0123 미학습) → 추론 local 성능.
"""
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os, sys, json, glob
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
from eval_pvnet_heads import load_pvnet_model, preprocess, belief_to_orig  # noqa
from four_arm_pl_compare import collect_val_frames  # noqa
from diag2_raw_decode_stages import classify, gt_to_belief, disk_max, GT_DISK  # noqa
from diag4_geometry_matching import MATCH_VARS, CALIPER  # noqa

WEIGHTS = os.path.join(ROOT, "weights/challenge0123/final_net_epoch_0060.pth")
OUT = os.path.join(ROOT, "data/pallet/eval_results/stage9_diag")
TRAIN_DIRS = [os.path.join(ROOT, "data/pallet/training_data/mixed_v8_train"),
              os.path.join(ROOT, "challenge/data/02_synthetic/training/v1"),
              os.path.join(ROOT, "challenge/data/02_synthetic/training/v2")]
VAL_DIR = os.path.join(ROOT, "data/pallet/training_data/val")
DEPTH = [(0, 4), (1, 5), (2, 6), (3, 7)]; FRONT = [0, 1, 2, 3]; BACK = [4, 5, 6, 7]
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
GVEC = ["A", "S", "P", "Bn", "Emin"]   # 연속 coverage feature
N_TRAIN = 8000


def parea(p):
    return float(cv2.contourArea(cv2.convexHull(p.astype(np.float32)))) if len(p) >= 3 else 0.0


def feat(gt8, ctr, H, W):
    hull = max(parea(gt8), 1.0)
    seps = [np.linalg.norm(gt8[b] - gt8[f]) for f, b in DEPTH]
    allp = np.vstack([gt8, ctr])
    inb = ((allp[:, 0] >= 0) & (allp[:, 0] < W) & (allp[:, 1] >= 0) & (allp[:, 1] < H))
    fa = max(parea(gt8[FRONT]), 1.0); ba = max(parea(gt8[BACK]), 1.0)
    idx = np.where(inb[:8])[0]
    bd = min((min(gt8[i, 0], W - gt8[i, 0], gt8[i, 1], H - gt8[i, 1]) for i in idx), default=0.0)
    emin = min(np.linalg.norm(gt8[a] - gt8[b]) for a, b in EDGES)
    sq = np.sqrt(hull)
    return {"A": float(np.log(hull / (H * W))), "S": float(np.median(seps) / sq),
            "P": float(np.log(fa / ba)), "Bn": float(bd / sq), "Emin": float(emin / sq),
            "T": float(1 - inb.sum() / 9), "V": int(inb[:8].sum())}


def load_feats(dirs, n):
    js = []
    for d in dirs:
        js += glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
    js.sort()
    if n and n < len(js):
        js = [js[int(i)] for i in np.linspace(0, len(js) - 1, n).round().astype(int)]
    out = []
    for jp in js:
        try:
            d = json.load(open(jp)); o = d["objects"][0]
            g8 = np.array(o["projected_cuboid"], float)[:8]
            ct = np.array(o["projected_cuboid_centroid"], float)
            cd = d.get("camera_data", {})
            W = cd.get("width", 640); H = cd.get("height", 480)
            out.append(feat(g8, ct, H, W))
        except Exception:
            continue
    return out


def real_infer(model, device, ip, gt8, ctr, H, W):
    import torch
    img = cv2.imread(ip)
    if img is None:
        return None
    t, nw, nh, sc = preprocess(img); t = t.to(device)
    with torch.no_grad():
        beliefs, _ = model(t)
    b = beliefs[-1][0].cpu().numpy(); bh, bw = b.shape[1], b.shape[2]
    gts = list(gt8) + [ctr]; score = np.zeros(9); gl = np.zeros(9); raw = np.full((9, 2), np.nan)
    for k in range(9):
        ch = b[k]; yi, xi = np.unravel_index(int(ch.argmax()), ch.shape)
        score[k] = float(ch[yi, xi]); raw[k] = belief_to_orig(xi, yi, bw, bh, nw, nh, sc)
        gbx, gby = gt_to_belief(gts[k], sc, bw, bh, nw, nh); gl[k] = disk_max(ch, gbx, gby, GT_DISK)
    cerr = np.array([np.linalg.norm(raw[i] - gt8[i]) for i in range(8)])
    cls, _ = classify({"score": score, "gtlocal": gl, "corner_err": cerr,
                       "center_err": float(np.linalg.norm(raw[8] - ctr))})
    f = feat(gt8, ctr, H, W)
    return {**f, "cls": cls, "det": int(cls == "detected"),
            "noresp": int(cls == "no_response")}


def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_pvnet_model(WEIGHTS, dev, numVec=0, numSeg=0)

    # real
    R = {"outside": [], "night": []}
    for dom, fid, jp, ip in collect_val_frames():
        if dom not in R:
            continue
        d = json.load(open(jp)); o = d["objects"][0]
        g8 = np.array(o["projected_cuboid"], float)[:8]; ct = np.array(o["projected_cuboid_centroid"], float)
        im = cv2.imread(ip)
        if im is None:
            continue
        r = real_infer(model, dev, ip, g8, ct, im.shape[0], im.shape[1])
        if r:
            r["fid"] = fid; R[dom].append(r)
    # 매칭 → G1/G2
    pooled = R["outside"] + R["night"]; nrm = {}
    for v in MATCH_VARS:
        a = np.array([r[v] for r in pooled]); nrm[v] = (np.median(a),
            (np.percentile(a, 75) - np.percentile(a, 25)) or 1.0)
    def mvec(r):
        return np.array([(r[v] - nrm[v][0]) / nrm[v][1] for v in MATCH_VARS])
    O = R["outside"]; Nn = R["night"]; Ov = [mvec(r) for r in O]; used = set(); pn = set()
    for dist, ni in sorted((min(np.linalg.norm(mvec(Nn[i]) - ov) for ov in Ov), i)
                           for i in range(len(Nn))):
        if dist > CALIPER:
            continue
        nv = mvec(Nn[ni]); order = np.argsort([np.linalg.norm(nv - Ov[oi]) for oi in range(len(O))])
        for oi in order:
            if oi not in used and np.linalg.norm(nv - Ov[oi]) <= CALIPER:
                used.add(oi); pn.add(ni); break
    G1 = [Nn[i] for i in pn]; G2 = [Nn[i] for i in range(len(Nn)) if i not in pn]; G3 = O

    # synthetic train(coverage) + held-out val(추론)
    syn_tr = load_feats(TRAIN_DIRS, N_TRAIN)
    val_js = sorted(glob.glob(os.path.join(VAL_DIR, "*.json")))
    syn_val = []
    for jp in val_js:
        ip = jp[:-5] + ".png"
        if not os.path.exists(ip):
            continue
        d = json.load(open(jp)); o = d["objects"][0]
        g8 = np.array(o["projected_cuboid"], float)[:8]; ct = np.array(o["projected_cuboid_centroid"], float)
        im = cv2.imread(ip)
        if im is None:
            continue
        r = real_infer(model, dev, ip, g8, ct, im.shape[0], im.shape[1])
        if r:
            syn_val.append(r)

    # robust scale by synth-train
    gnrm = {}
    for v in GVEC:
        a = np.array([s[v] for s in syn_tr]); gnrm[v] = (np.median(a),
            (np.percentile(a, 75) - np.percentile(a, 25)) or 1.0)
    def gv(r):
        return np.array([(r[v] - gnrm[v][0]) / gnrm[v][1] for v in GVEC])
    TR = np.array([gv(s) for s in syn_tr])
    def d20(r):
        d = np.linalg.norm(TR - gv(r), axis=1); return float(np.partition(d, 20)[20])
    # threshold from synth-val d20
    val_d20 = np.array([d20(s) for s in syn_val])
    thr95, thr99 = np.percentile(val_d20, 95), np.percentile(val_d20, 99)

    def supp(dv):
        return "in" if dv <= thr95 else ("border" if dv <= thr99 else "OOS")

    # synth-val local detection (real → K nearest synth-val)
    VV = np.array([gv(s) for s in syn_val]); valdet = np.array([s["det"] for s in syn_val])
    def local_syn_det(r, K=20):
        d = np.linalg.norm(VV - gv(r), axis=1); idx = np.argsort(d)[:K]
        w = np.exp(-(d[idx] ** 2) / (np.median(d[idx]) ** 2 + 1e-9))
        return float(np.sum(w * valdet[idx]) / (w.sum() + 1e-9))

    L = [f"TRACK-B DIAG 6 — synthetic coverage(H1) vs sim2real(H2) vs model-limit(H3)",
         f"syn_train={len(syn_tr)} syn_val={len(syn_val)} (val held-out)  GVEC={GVEC}",
         f"support thr(synth-val d20): in<={thr95:.2f} border<={thr99:.2f} OOS>"]
    # 0) validity
    L.append("\n[0 validity] num in-frame corners <8 비율 (close-up off-frame=task-def)")
    for nm, g in [("G1 matchedN", G1), ("G2 excludedN", G2), ("G3 outside", G3)]:
        L.append(f"  {nm:<13} N={len(g):>3}  V<8 비율={100*np.mean([r['V']<8 for r in g]):>4.0f}%  "
                 f"medV={np.median([r['V'] for r in g]):.0f} medT={np.median([r['T'] for r in g]):.2f}")
    # 1) coverage
    L.append("\n[1 coverage] real→synth-train d20 support")
    L.append(f"  {'grp':<13}{'N':>4}{'d20med':>8}{'in%':>6}{'border%':>9}{'OOS%':>6}")
    cov = {}
    for nm, g in [("G1 matchedN", G1), ("G2 excludedN", G2), ("G3 outside", G3),
                  ("syn_val", syn_val)]:
        ds = [d20(r) for r in g]; cls = [supp(x) for x in ds]; cov[nm] = cls
        L.append(f"  {nm:<13}{len(g):>4}{np.median(ds):>8.2f}"
                 f"{100*np.mean([c=='in' for c in cls]):>5.0f}%"
                 f"{100*np.mean([c=='border' for c in cls]):>8.0f}%"
                 f"{100*np.mean([c=='OOS' for c in cls]):>5.0f}%")
    # 2) real vs local-synth
    L.append("\n[2 real vs local-synth-val] (그 geometry에서 synthetic은 되나)")
    L.append(f"  {'grp':<13}{'realDet%':>9}{'realNoR%':>9}{'locSynDet%':>11}")
    for nm, g in [("G1 matchedN", G1), ("G2 excludedN", G2), ("G3 outside", G3)]:
        lsd = np.mean([local_syn_det(r) for r in g])
        L.append(f"  {nm:<13}{100*np.mean([r['det'] for r in g]):>8.0f}%"
                 f"{100*np.mean([r['noresp'] for r in g]):>8.0f}%{100*lsd:>10.0f}%")
    L.append(f"\n[핵심] ΔOOS = P(OOS|G2)-P(OOS|G1) = "
             f"{100*(np.mean([c=='OOS' for c in cov['G2 excludedN']])-np.mean([c=='OOS' for c in cov['G1 matchedN']])):.0f}%p")
    txt = "\n".join(L); print(txt)
    open(os.path.join(OUT, "diag6_coverage.txt"), "w").write(txt)
    print(f"\n[save] {OUT}/diag6_coverage.txt")


if __name__ == "__main__":
    main()
