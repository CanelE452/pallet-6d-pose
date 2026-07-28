"""diag6c_montage.py — G2 V=8 12장 montage: real + nearest synth-train/val 3장씩.
coarse geometry NN이 실제 projection topology(visible face/yaw/skew/border)까지 같은지
눈검증. 같으면 sim2real 결론 유지, 다르면 geometry descriptor 보강.
"""
import os, sys, json, glob
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
from four_arm_pl_compare import collect_val_frames  # noqa
from diag4_geometry_matching import MATCH_VARS, CALIPER  # noqa
from diag6_synthetic_coverage import feat, GVEC, TRAIN_DIRS, VAL_DIR, N_TRAIN  # noqa

OUT = os.path.join(ROOT, "data/pallet/eval_results/stage9_diag/montage")
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
TILE = 300


def load_with_path(dirs, n):
    js = []
    for d in dirs:
        js += glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
    js.sort()
    if n and n < len(js):
        js = [js[int(i)] for i in np.linspace(0, len(js)-1, n).round().astype(int)]
    out = []
    for jp in js:
        ip = jp[:-5] + ".png"
        if not os.path.exists(ip):
            continue
        try:
            d = json.load(open(jp)); o = d["objects"][0]
            g8 = np.array(o["projected_cuboid"], float)[:8]
            ct = np.array(o["projected_cuboid_centroid"], float)
            cd = d.get("camera_data", {}); W = cd.get("width", 640); H = cd.get("height", 480)
            f = feat(g8, ct, H, W)
            out.append((f, ip, g8))
        except Exception:
            continue
    return out


def tile(ip, g8, label):
    img = cv2.imread(ip)
    if img is None:
        img = np.zeros((TILE, TILE, 3), np.uint8)
    H, W = img.shape[:2]
    for a, b in EDGES:
        pa, pb = g8[a], g8[b]
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 255, 255), 2)
    for i, p in enumerate(g8):
        cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255) if i < 4 else (255, 100, 0), -1)
    img = cv2.resize(img, (TILE, int(TILE * H / W)))
    bar = np.zeros((22, img.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return np.vstack([bar, img])


def main():
    os.makedirs(OUT, exist_ok=True)
    # G2 V=8 식별
    R = {"outside": [], "night": []}
    for dom, fid, jp, ip in collect_val_frames():
        if dom not in R:
            continue
        d = json.load(open(jp)); o = d["objects"][0]
        g8 = np.array(o["projected_cuboid"], float)[:8]; ct = np.array(o["projected_cuboid_centroid"], float)
        im = cv2.imread(ip)
        if im is None:
            continue
        f = feat(g8, ct, im.shape[0], im.shape[1])
        R[dom].append({**f, "ip": ip, "g8": g8, "fid": fid})
    pooled = R["outside"] + R["night"]; nrm = {}
    for v in MATCH_VARS:
        a = np.array([r[v] for r in pooled]); nrm[v] = (np.median(a),
            (np.percentile(a, 75)-np.percentile(a, 25)) or 1.0)
    mvec = lambda r: np.array([(r[v]-nrm[v][0])/nrm[v][1] for v in MATCH_VARS])
    O = R["outside"]; Nn = R["night"]; Ov = [mvec(r) for r in O]; used=set(); pn=set()
    for dist, ni in sorted((min(np.linalg.norm(mvec(Nn[i])-ov) for ov in Ov), i) for i in range(len(Nn))):
        if dist > CALIPER:
            continue
        nv = mvec(Nn[ni]); order = np.argsort([np.linalg.norm(nv-Ov[oi]) for oi in range(len(O))])
        for oi in order:
            if oi not in used and np.linalg.norm(nv-Ov[oi]) <= CALIPER:
                used.add(oi); pn.add(ni); break
    G2 = [Nn[i] for i in range(len(Nn)) if i not in pn and Nn[i]["V"] >= 8]
    print(f"[G2 V=8] {len(G2)}장")

    syn_tr = load_with_path(TRAIN_DIRS, N_TRAIN)
    syn_val = load_with_path([VAL_DIR], 0)
    gnrm = {}
    for v in GVEC:
        a = np.array([s[0][v] for s in syn_tr]); gnrm[v] = (np.median(a),
            (np.percentile(a, 75)-np.percentile(a, 25)) or 1.0)
    gv = lambda f: np.array([(f[v]-gnrm[v][0])/gnrm[v][1] for v in GVEC])
    TRv = np.array([gv(s[0]) for s in syn_tr]); VLv = np.array([gv(s[0]) for s in syn_val])

    for k, r in enumerate(G2):
        q = gv(r)
        ti = np.argsort(np.linalg.norm(TRv - q, axis=1))[:3]
        vi = np.argsort(np.linalg.norm(VLv - q, axis=1))[:3]
        lab = f"REAL g2 A{r['A']:.1f} S{r['S']:.2f} P{r['P']:.2f} Bn{r['Bn']:.2f}"
        row = [tile(r["ip"], r["g8"], lab)]
        for j in ti:
            f, ip, g8 = syn_tr[j]
            row.append(tile(ip, g8, f"synTR A{f['A']:.1f} S{f['S']:.2f} P{f['P']:.2f}"))
        for j in vi:
            f, ip, g8 = syn_val[j]
            row.append(tile(ip, g8, f"synVAL A{f['A']:.1f} S{f['S']:.2f} P{f['P']:.2f}"))
        hh = max(t.shape[0] for t in row)
        row = [np.vstack([t, np.zeros((hh-t.shape[0], t.shape[1], 3), np.uint8)]) for t in row]
        cv2.imwrite(os.path.join(OUT, f"montage_{k:02d}_{r['fid']}.jpg"),
                    cv2.hconcat(row), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[save] {OUT}  ({len(G2)} montages: real | 3 synTR | 3 synVAL)")


if __name__ == "__main__":
    main()
