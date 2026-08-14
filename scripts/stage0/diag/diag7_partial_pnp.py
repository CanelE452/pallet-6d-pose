"""diag7_partial_pnp.py — Track B 무학습 probe.
truncation(V<8) 프레임에서 (1) GT in-frame corner 만으로 pose 풀리나(Oracle partial-PnP),
(2) 현재 모델 예측 in-frame corner 로 풀리나(predicted). V bin(4~8)별 성공률/reproj.
→ Oracle 성공+predicted 실패 = border supervision/confidence selection으로 해결가능,
  Oracle도 실패 = keypoint 배치/conditioning 문제(label 전에 feasibility 재검토). 학습 X.
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
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))
sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
from four_arm_pl_compare import collect_val_frames  # noqa
from eval_pvnet_heads import load_pvnet_model, preprocess, belief_to_orig, collect_manual  # noqa
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa
from annotate_pnp import solve_pose, make_pallet_keypoints_3d, project_3d  # noqa

WEIGHTS = os.path.join(ROOT, "weights/challenge0123/final_net_epoch_0060.pth")
OUT = os.path.join(ROOT, "data/pallet/eval_results/stage9_diag")


def K_of(d):
    it = d.get("camera_data", {}).get("intrinsics")
    if it:
        return np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]], float)
    return np.array([[614.18, 0, 329.28], [0, 614.31, 0, ][0:3] if False else [0, 614.31, 234.53],
                     [0, 0, 1]], float)


def reproj_full(pose, gt8, K):
    kp3d = make_pallet_keypoints_3d(*pose["dims"])
    pr = project_3d(kp3d, pose["R"], pose["t"], K)[:8]
    return float(np.median(np.linalg.norm(pr - gt8, axis=1)))


def posdepth_ok(pose, K):
    kp3d = make_pallet_keypoints_3d(*pose["dims"])
    z = (kp3d @ pose["R"].T + pose["t"])[:, 2]
    return bool((z > 0).all() and pose["t"][2] > 0)


def main():
    import torch, argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", default=WEIGHTS)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_pvnet_model(args.weights, dev, numVec=0, numSeg=0)
    frames = list(collect_val_frames()) + [("manual", f, jp, ip) for _, f, jp, ip in collect_manual(0)]

    recs = []
    for dom, fid, jp, ip in frames:
        d = json.load(open(jp)); o = d["objects"][0]
        gt8 = np.array(o["projected_cuboid"], float)[:8]
        ctr = np.array(o["projected_cuboid_centroid"], float)
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]; K = K_of(d)
        inb = [(0 <= gt8[i, 0] < W) and (0 <= gt8[i, 1] < H) for i in range(8)]
        V = int(sum(inb))
        cin = (0 <= ctr[0] < W) and (0 <= ctr[1] < H)
        # reference pose (모든 GT, 화면밖 좌표 포함=참 pose)
        try:
            ref = solve_pose([list(gt8[i]) for i in range(8)] + [list(ctr)], K, img_shape=(H, W))
        except Exception:
            ref = None
        # oracle partial: in-frame GT corner만
        kps_o = [list(gt8[i]) if inb[i] else None for i in range(8)] + [list(ctr) if cin else None]
        try:
            po = solve_pose(kps_o, K, img_shape=(H, W))
        except Exception:
            po = None
        # predicted partial: 모델 in-frame 검출 corner
        t, nw, nh, sc = preprocess(img); t = t.to(dev)
        with torch.no_grad():
            beliefs, _ = model(t)
        bel = beliefs[-1][0].cpu().numpy(); bh, bw = bel.shape[1], bel.shape[2]
        kpb = extract_keypoints_from_belief(bel, 0.3)
        kps_p = []
        for i in range(9):
            if kpb[i][0] >= 0:
                ox, oy = belief_to_orig(kpb[i][0], kpb[i][1], bw, bh, nw, nh, sc)
                kps_p.append([ox, oy] if (0 <= ox < W and 0 <= oy < H) else None)
            else:
                kps_p.append(None)
        try:
            pp = solve_pose(kps_p, K, img_shape=(H, W)) if sum(x is not None for x in kps_p[:8]) >= 4 else None
        except Exception:
            pp = None
        recs.append({"dom": dom, "V": V,
                     "o_ok": po is not None, "o_reproj": (reproj_full(po, gt8, K) if po else np.nan),
                     "o_depth": (posdepth_ok(po, K) if po else False),
                     "p_ok": pp is not None, "p_reproj": (reproj_full(pp, gt8, K) if pp else np.nan),
                     "n_pred_in": sum(x is not None for x in kps_p[:8])})

    L = ["TRACK-B DIAG 7 — partial-PnP feasibility (학습 X)",
         f"weights={WEIGHTS}  N={len(recs)}",
         "Oracle=GT in-frame corner / Predicted=모델 in-frame 검출 corner. reproj=full 8 GT median px"]
    L.append(f"\n  {'Vbin':<6}{'N':>4}{'oraclePnP%':>11}{'oReprojMed':>11}{'oDepthOK%':>10}"
             f"{'predPnP%':>9}{'pReprojMed':>11}")
    for vb in [8, 7, 6, 5, 4, 3]:
        sub = [r for r in recs if r["V"] == vb]
        if not sub:
            continue
        oo = [r for r in sub if r["o_ok"]]
        orj = [r["o_reproj"] for r in oo if np.isfinite(r["o_reproj"])]
        pp = [r for r in sub if r["p_ok"]]
        prj = [r["p_reproj"] for r in pp if np.isfinite(r["p_reproj"])]
        L.append(f"  V={vb:<4}{len(sub):>4}{100*len(oo)/len(sub):>10.0f}%"
                 f"{(np.median(orj) if orj else float('nan')):>11.1f}"
                 f"{100*np.mean([r['o_depth'] for r in oo])if oo else float('nan'):>9.0f}%"
                 f"{100*len(pp)/len(sub):>8.0f}%"
                 f"{(np.median(prj) if prj else float('nan')):>11.1f}")
    # V<8 요약
    v8 = [r for r in recs if r["V"] < 8]
    if v8:
        oo = [r for r in v8 if r["o_ok"] and r["o_depth"]]
        L.append(f"\n[V<8 요약] N={len(v8)}  Oracle(valid-depth)성공={100*len(oo)/len(v8):.0f}%  "
                 f"reproj통과(<10px)={100*np.mean([np.isfinite(r['o_reproj']) and r['o_reproj']<10 for r in v8]):.0f}%")
    txt = "\n".join(L); print(txt)
    open(os.path.join(OUT, "diag7_partial_pnp.txt"), "w").write(txt)
    print(f"\n[save] {OUT}/diag7_partial_pnp.txt")


if __name__ == "__main__":
    main()
