"""diag6b_vstrat.py — diag6 정밀화: V=8(full) vs V<8(truncated) 분리 + same-V 이웃검색.
G2의 real-syn 갭이 truncation confound인지(same-V로 비교), full-view에서도 sim2real 남는지.
diag6 함수 재사용. 추론 경량.
"""
import os, sys, json, glob
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
from eval_pvnet_heads import load_pvnet_model  # noqa
from four_arm_pl_compare import collect_val_frames  # noqa
from diag4_geometry_matching import MATCH_VARS, CALIPER  # noqa
from diag6_synthetic_coverage import (feat, real_infer, load_feats, GVEC,  # noqa
                                      TRAIN_DIRS, VAL_DIR, N_TRAIN, WEIGHTS, OUT)


def main():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_pvnet_model(WEIGHTS, dev, numVec=0, numSeg=0)

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
            R[dom].append(r)
    # 매칭 G1/G2
    pooled = R["outside"] + R["night"]; nrm = {}
    for v in MATCH_VARS:
        a = np.array([r[v] for r in pooled]); nrm[v] = (np.median(a),
            (np.percentile(a, 75) - np.percentile(a, 25)) or 1.0)
    mvec = lambda r: np.array([(r[v]-nrm[v][0])/nrm[v][1] for v in MATCH_VARS])
    O = R["outside"]; Nn = R["night"]; Ov = [mvec(r) for r in O]; used=set(); pn=set()
    for dist, ni in sorted((min(np.linalg.norm(mvec(Nn[i])-ov) for ov in Ov), i) for i in range(len(Nn))):
        if dist > CALIPER:
            continue
        nv = mvec(Nn[ni]); order = np.argsort([np.linalg.norm(nv-Ov[oi]) for oi in range(len(O))])
        for oi in order:
            if oi not in used and np.linalg.norm(nv-Ov[oi]) <= CALIPER:
                used.add(oi); pn.add(ni); break
    G = {"G1 matchedN": [Nn[i] for i in pn],
         "G2 excludedN": [Nn[i] for i in range(len(Nn)) if i not in pn],
         "G3 outside": O}

    # synthetic train(coverage) + held-out val(추론), V 보존
    syn_tr = load_feats(TRAIN_DIRS, N_TRAIN)   # has V
    syn_val = []
    for jp in sorted(glob.glob(os.path.join(VAL_DIR, "*.json"))):
        ip = jp[:-5]+".png"
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

    gnrm = {}
    for v in GVEC:
        a = np.array([s[v] for s in syn_tr]); gnrm[v] = (np.median(a),
            (np.percentile(a, 75)-np.percentile(a, 25)) or 1.0)
    gv = lambda r: np.array([(r[v]-gnrm[v][0])/gnrm[v][1] for v in GVEC])

    def vbin(r):
        return "V=8" if r["V"] >= 8 else "V<8"
    # same-V 이웃 풀
    TRb = {b: np.array([gv(s) for s in syn_tr if vbin(s) == b] or [[1e9]*len(GVEC)]) for b in ("V=8","V<8")}
    VLb = {b: [s for s in syn_val if vbin(s) == b] for b in ("V=8","V<8")}
    VLv = {b: np.array([gv(s) for s in VLb[b]] or [[1e9]*len(GVEC)]) for b in ("V=8","V<8")}
    VLdet = {b: np.array([s["det"] for s in VLb[b]] or [0]) for b in ("V=8","V<8")}

    def d20_sameV(r):
        b = vbin(r); D = np.linalg.norm(TRb[b]-gv(r), axis=1)
        return float(np.partition(D, min(20, len(D)-1))[min(20, len(D)-1)])
    def locsyn_sameV(r, K=20):
        b = vbin(r); V = VLv[b]
        if len(V) < 3:
            return float("nan")
        D = np.linalg.norm(V-gv(r), axis=1); idx = np.argsort(D)[:K]
        w = np.exp(-(D[idx]**2)/(np.median(D[idx])**2+1e-9))
        return float(np.sum(w*VLdet[b][idx])/(w.sum()+1e-9))

    L = ["TRACK-B DIAG 6b — V=8/V<8 분리 + same-V 이웃검색",
         f"syn_train={len(syn_tr)}(V=8:{sum(vbin(s)=='V=8' for s in syn_tr)} V<8:{sum(vbin(s)=='V<8' for s in syn_tr)}) "
         f"syn_val={len(syn_val)}(V=8:{sum(vbin(s)=='V=8' for s in syn_val)} V<8:{sum(vbin(s)=='V<8' for s in syn_val)})"]
    L.append(f"\n  {'group':<13}{'Vbin':<6}{'N':>4}{'realDet%':>9}{'realNoR%':>9}"
             f"{'locSynDet%':>11}{'gap%p':>7}{'d20med(sameV)':>14}")
    for nm, g in G.items():
        for b in ("V=8", "V<8"):
            sub = [r for r in g if vbin(r) == b]
            if not sub:
                continue
            rd = 100*np.mean([r["det"] for r in sub])
            rn = 100*np.mean([r["noresp"] for r in sub])
            ls = np.nanmean([locsyn_sameV(r) for r in sub])
            d20 = np.median([d20_sameV(r) for r in sub])
            L.append(f"  {nm:<13}{b:<6}{len(sub):>4}{rd:>8.0f}%{rn:>8.0f}%"
                     f"{100*ls:>10.0f}%{100*ls-rd:>6.0f}{d20:>14.2f}")
    txt = "\n".join(L); print(txt)
    open(os.path.join(OUT, "diag6b_vstrat.txt"), "w").write(txt)
    print(f"\n[save] {OUT}/diag6b_vstrat.txt")


if __name__ == "__main__":
    main()
