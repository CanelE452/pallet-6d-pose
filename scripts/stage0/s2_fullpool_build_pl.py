"""s2_fullpool_build_pl.py — full-pool 추론 jsonl + FULL7 필터 pass 를 DOPE PL 데이터셋으로.

입력: data/pallet/results/<infer_dir>/{domain}.jsonl  (pred8/pred_c/peaks)
필터: FULL7 (f3 포함). pass 프레임만 PL 로. wood 제외(16:9 aspect 문제).
출력: data/pallet/training_data/<out_name>/  {fid}.json (NDDS pseudo) + {fid}.png(symlink)

Usage: python -u scripts/stage0/s2_fullpool_build_pl.py <infer_dir> <out_name>
  예: python -u scripts/stage0/s2_fullpool_build_pl.py paper_s2_fullpool_infer paper_s2_fullpool_r1
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

import paper_s2_filterval_9filters as F   # noqa: E402,F401
import paper_s2_testset17_9filters as T   # noqa: E402
import s2_fullpool_infer as INF           # noqa: E402
import cv2                                # noqa: E402

M, TAU, NMIN = T.M, T.TAU, T.N_DET_MIN
FULL7 = ["f1_peak", "f2_peak_ratio", "f3_flip", "f4_tta_stab",
         "f5_rear_conf", "f6_frsep", "f7_posdepth"]
ENV = {"size_lo": 0.0, "size_hi": 1.0, "asp_lo": 0.0, "asp_hi": 10.0}
MISSING = [-100.0, -100.0]
EXCLUDE_DOMAINS = {"wood_indoor", "wood_outdoor"}   # 16:9 aspect 문제로 self-train 제외

SESS2DIR = {os.path.basename(seq): seq
            for dom, seqs in INF.DOMAINS.items() for seq in seqs}

# ★ 누수 방지: 모든 GT 어노 프레임(평가셋)은 PL(학습)에서 제외 (split 무관, 전체 홀드아웃).
EVAL_GT_GLOBS = [
    "challenge/data/01_real/eval_canonical/_outside_eval_manual_gt", "challenge/data/01_real/manual_gt/_night_eval_manual_gt",
    "challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt", "challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt",
    "challenge/data/01_real/manual_gt/capturepallet0*_manual_gt",
    "challenge/data/01_real/manual_gt/capturenight0*_manual_gt",
]


def collect_eval_fids():
    fids = set()
    for g in EVAL_GT_GLOBS:
        for fo in glob.glob(os.path.join(ROOT, g)):
            for jf in glob.glob(os.path.join(fo, "*.json")):
                try:
                    if "projected_cuboid" in json.load(open(jf))["objects"][0]:
                        fids.add(os.path.splitext(os.path.basename(jf))[0])
                except Exception:
                    pass
    return fids


def passes(rec):
    if rec["n_det"] < NMIN:
        return False
    row = {"n_det": rec["n_det"], "scores": rec["scores"],
           "f7_posdepth": rec["f7_posdepth"], "pred_sr": rec["pred_sr"],
           "pred_asp": rec["pred_asp"]}
    return all(M.apply_filter(f, row, TAU.get(f), ENV) for f in FULL7)


def convert(pred8, pred_c, peaks):
    pts, valid = [], []
    for i in range(8):
        p = pred8[i]
        ok = p is not None and math.isfinite(p[0]) and math.isfinite(p[1])
        valid.append(ok)
        pts.append([float(p[0]), float(p[1])] if ok else list(MISSING))
    ok = pred_c is not None and math.isfinite(pred_c[0])
    valid.append(ok)
    pts.append([float(pred_c[0]), float(pred_c[1])] if ok else list(MISSING))
    return pts, valid, [float(x) for x in peaks]


def main():
    infer_dir = sys.argv[1] if len(sys.argv) > 1 else "paper_s2_fullpool_infer"
    out_name = sys.argv[2] if len(sys.argv) > 2 else "paper_s2_fullpool_r1"
    only_dom = sys.argv[3] if len(sys.argv) > 3 else None   # 도메인 필터 (per-domain PL)
    src = os.path.join(ROOT, "data/pallet/results", infer_dir)
    out = os.path.join(ROOT, "data/pallet/training_data", out_name)
    os.makedirs(out, exist_ok=True)

    eval_fids = collect_eval_fids()
    print(f"[leak-guard] eval-marked frames excluded from PL: {len(eval_fids)}")

    n_written = 0
    n_leak_skip = 0
    per_dom = {}
    for jf in sorted(glob.glob(os.path.join(src, "*.jsonl"))):
        dom = os.path.splitext(os.path.basename(jf))[0]
        if dom in EXCLUDE_DOMAINS:
            continue
        if only_dom and dom != only_dom:      # per-domain PL 필터
            continue
        for line in open(jf):
            rec = json.loads(line)
            if not passes(rec):
                continue
            if rec["fid"] in eval_fids:          # 누수 방지: 평가셋 제외
                n_leak_skip += 1
                continue
            seq = SESS2DIR.get(rec["session"])
            ip = os.path.join(seq, "rgb", rec["fid"] + ".png") if seq else None
            if not ip or not os.path.isfile(ip):
                continue
            img = cv2.imread(ip)
            if img is None:
                continue
            H, W = img.shape[:2]
            K = np.loadtxt(os.path.join(seq, "cam_K.txt")).reshape(3, 3) \
                if os.path.isfile(os.path.join(seq, "cam_K.txt")) else INF.DEFAULT_K
            pts, valid, conf = convert(rec["pred8"], rec["pred_c"], rec["peaks"])
            ann = {
                "camera_data": {
                    "width": W, "height": H,
                    "intrinsics": {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                                   "cx": float(K[0, 2]), "cy": float(K[1, 2])},
                },
                "source_model": "paper_s2_stageB_ep57",
                "source_rgb": os.path.relpath(ip, ROOT),
                "pseudo_keypoint_valid": valid,
                "objects": [{
                    "class": "pallet", "name": "real_pallet", "visibility": 1,
                    "gt_source": "pseudo", "source_model": "paper_s2_stageB_ep57",
                    "projected_cuboid": pts[:8],
                    "projected_cuboid_centroid": pts[8],
                    "pseudo_keypoint_valid": valid,
                    "pseudo_keypoint_confidence": conf,
                    "pseudo_provenance": {
                        "inference_preprocess": "squash_to_400x400",
                        "label_geometry": "raw_heatmap_peaks_no_pnp",
                        "selection": "FULL7", "domain": dom, "gt_annotations_used": False,
                    },
                }],
            }
            fid = rec["fid"]
            with open(os.path.join(out, f"{fid}.json"), "w") as f:
                json.dump(ann, f)
            op = os.path.join(out, f"{fid}.png")
            if not os.path.exists(op):
                try:
                    os.symlink(os.path.abspath(ip), op)
                except OSError:
                    import shutil
                    shutil.copy2(ip, op)
            n_written += 1
            per_dom[dom] = per_dom.get(dom, 0) + 1
    print(f"[build] {n_written} PL frames -> {out}  (eval-leak skipped: {n_leak_skip})")
    print("  per-domain:", per_dom)


if __name__ == "__main__":
    main()
