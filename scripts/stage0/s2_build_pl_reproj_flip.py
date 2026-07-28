"""s2_build_pl_reproj_flip.py — reproj-gate ∧ flip 필터로 full-pool PL 빌드.

필터: reproj(PnP dims-known) <= REPROJ_TAU  AND  f3_flip <= FLIP_TAU
GT 어노 프레임(전체) 홀드아웃(누수0), wood 제외.
출력: data/pallet/training_data/<out_name>/  {fid}.json(NDDS pseudo) + {fid}.png

Usage: python -u scripts/stage0/s2_build_pl_reproj_flip.py <infer_dir> <out_name> [reproj_tau] [flip_tau]
"""
from __future__ import annotations
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

import paper_s2_filterval_9filters as F   # noqa: E402,F401
import paper_s2_testset17_9filters as T   # noqa: E402
import s2_fullpool_infer as INF           # noqa: E402
import s2_fullpool_build_pl as B          # noqa: E402  (collect_eval_fids/convert/ndds 재사용)
import annotate_pnp as APNP               # noqa: E402
import cv2                                # noqa: E402

NMIN = T.N_DET_MIN
DIMS = (1.1, 1.3, 0.12)
MISSING = [-100.0, -100.0]
EXCLUDE = {"wood_indoor", "wood_outdoor"}
SESS2DIR = {os.path.basename(s): s for dom, seqs in INF.DOMAINS.items() for s in seqs}


def reproj_of(rec, K):
    kps = [list(p) if p is not None else None for p in rec["pred8"]]
    kps.append(rec["pred_c"])
    try:
        pose = APNP.solve_pose(kps, K, dims=DIMS)
        return pose.get("reproj_error_px", 999.0) if pose else 999.0
    except Exception:
        return 999.0


def main():
    infer_dir = sys.argv[1] if len(sys.argv) > 1 else "paper_s2_fullpool_infer"
    out_name = sys.argv[2] if len(sys.argv) > 2 else "paper_s2_pl_reproj_flip"
    rtau = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    ftau = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    only_dom = sys.argv[5] if len(sys.argv) > 5 else None
    src = os.path.join(ROOT, "data/pallet/results", infer_dir)
    out = os.path.join(ROOT, "data/pallet/training_data", out_name)
    os.makedirs(out, exist_ok=True)
    eval_fids = B.collect_eval_fids()
    print(f"[leak-guard] GT held-out: {len(eval_fids)}   filter: reproj<={rtau} & flip<={ftau}")

    n = 0
    per = {}
    for jf in sorted(glob.glob(os.path.join(src, "*.jsonl"))):
        dom = os.path.splitext(os.path.basename(jf))[0]
        if dom in EXCLUDE:
            continue
        if only_dom and dom != only_dom:
            continue
        for line in open(jf):
            rec = json.loads(line)
            if rec["n_det"] < NMIN or rec["fid"] in eval_fids:
                continue
            f3 = rec["scores"].get("f3_flip")
            if f3 is None or f3 > ftau:
                continue
            seq = SESS2DIR.get(rec["session"])
            ip = os.path.join(seq, "rgb", rec["fid"] + ".png") if seq else None
            if not ip or not os.path.isfile(ip):
                continue
            K = np.loadtxt(os.path.join(seq, "cam_K.txt")).reshape(3, 3) \
                if os.path.isfile(os.path.join(seq, "cam_K.txt")) else INF.DEFAULT_K
            if reproj_of(rec, K) > rtau:
                continue
            img = cv2.imread(ip)
            if img is None:
                continue
            H, W = img.shape[:2]
            pts, valid, conf = B.convert(rec["pred8"], rec["pred_c"], rec["peaks"])
            ann = {
                "camera_data": {"width": W, "height": H,
                                "intrinsics": {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                                               "cx": float(K[0, 2]), "cy": float(K[1, 2])}},
                "source_model": "paper_s2_stageB_ep57", "source_rgb": os.path.relpath(ip, ROOT),
                "pseudo_keypoint_valid": valid,
                "objects": [{"class": "pallet", "name": "real_pallet", "visibility": 1,
                             "gt_source": "pseudo", "projected_cuboid": pts[:8],
                             "projected_cuboid_centroid": pts[8], "pseudo_keypoint_valid": valid,
                             "pseudo_keypoint_confidence": conf,
                             "pseudo_provenance": {"selection": f"reproj<={rtau}&flip<={ftau}",
                                                   "domain": dom, "gt_annotations_used": False}}],
            }
            with open(os.path.join(out, f"{rec['fid']}.json"), "w") as f:
                json.dump(ann, f)
            op = os.path.join(out, f"{rec['fid']}.png")
            if not os.path.exists(op):
                try:
                    os.symlink(os.path.abspath(ip), op)
                except OSError:
                    import shutil
                    shutil.copy2(ip, op)
            n += 1
            per[dom] = per.get(dom, 0) + 1
    print(f"[build] {n} PL frames -> {out}")
    print("  per-domain:", per)


if __name__ == "__main__":
    main()
