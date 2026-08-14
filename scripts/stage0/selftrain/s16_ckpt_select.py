"""s16_ckpt_select.py — STAGE16-3 ckpt 선택 (synthetic val 기준, NOT real).

과거 교훈: real 로 ckpt 고르면 과적합. synthetic val 로 고른 뒤 real 은 판정용.
  - Val-Old  = data/pallet/training_data/val (held-out, mixed_v8 conv)
  - Val-v3   = challenge/data/02_synthetic/training/v3/batch_009 (held-out, camera-facing v4)
  - (Val-trunc = truncation_addon_v1 은 전량 학습에 사용됨 -> 진짜 holdout 없음.
     in-domain fit 참고용으로만 소량 평가, 선택 기준 아님.)

선택 기준: Val-Old 붕괴 없음(corner_med/det B2 대비 유지) + V=8 유지 + Val-v3 개선.
지표 order-free: Hungarian corner median/good%, det%(n_det>=6), honest full-8 reproj.
전처리: aspect-only (synthetic 은 clean full-frame -> pad 불필요/유해, memory).

출력: data/pallet/eval_results/stage16_truncation_addon/stage16_eval/ckpt_select.{txt,json}
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import glob, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))
sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
sys.path.insert(0, os.path.join(ROOT, "data", "pallet", "eval_results", "stage11_16k"))

import cv2, torch
from filter_pr_camfacing import extract_keypoints_from_belief
from eval_pvnet_heads import (preprocess, belief_to_orig, heatmap_pred8,
                              split_metrics)
from eval_stage11 import load_model, K_from_json, dims_from_json, hungarian_match
import annotate_pnp as APNP

THRESHOLD, N_DET_MIN, GOOD_PX = 0.3, 6, 10.0
OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/stage16_truncation_addon/stage16_eval")

VAL_OLD = os.path.join(ROOT, "data/pallet/training_data/val")
VAL_V3 = os.path.join(ROOT, "challenge/data/02_synthetic/training/v3/batch_009")
VAL_TRUNC = os.path.join(ROOT, "challenge/data/02_synthetic/training/truncation_addon_v1")  # NOT heldout


def collect(dirpath, n):
    out = []
    for jp in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        if jp.endswith(".face_centers.json"):
            continue
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(dirpath, fid + ".png")
        if os.path.exists(ip):
            out.append((jp, ip))
    if 0 < n < len(out):
        idx = np.linspace(0, len(out) - 1, n).round().astype(int)
        out = [out[int(i)] for i in sorted(set(idx))]
    return out


def eval_frame(model, jp, ip, device):
    img = cv2.imread(ip)
    if img is None:
        return None
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    h0, w0 = img.shape[:2]
    inb = ((gt8[:, 0] >= 0) & (gt8[:, 0] < w0)
           & (gt8[:, 1] >= 0) & (gt8[:, 1] < h0))
    occ = int(inb.sum())

    tensor, nw, nh, sc = preprocess(img)
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps_bel = extract_keypoints_from_belief(belief, THRESHOLD)
    pred8 = heatmap_pred8(kps_bel, bw, bh, nw, nh, sc)

    m = split_metrics(pred8, gt8)
    dists, _ = hungarian_match(pred8, gt8)

    # honest full-8 reproj (order-free)
    K = K_from_json(d)
    dims = dims_from_json(d)
    kps9 = [None if np.isnan(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    if kps_bel[8][0] >= 0:
        cx, cy = belief_to_orig(kps_bel[8][0], kps_bel[8][1], bw, bh, nw, nh, sc)
        kps9.append([float(cx), float(cy)])
    else:
        kps9.append(None)
    honest8, pnp_ok = np.inf, 0
    if sum(1 for k in kps9 if k is not None) >= N_DET_MIN:
        try:
            pose = APNP.solve_pose(kps9, K, dims=dims, img_shape=img.shape)
            if pose is not None:
                pnp_ok = 1
                pa = np.array(pose["projected_all"], float)[:8]
                bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
                pa[bad] = np.nan
                hd, _ = hungarian_match(pa, gt8)
                if hd is not None:
                    honest8 = float(np.mean(hd))
        except Exception:
            pass

    return {"occ": occ, "corner": m["overall"], "n_det": m["n_det"],
            "det": 1 if m["n_det"] >= N_DET_MIN else 0,
            "honest8": honest8, "pnp_ok": pnp_ok}


def agg(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    cor = [r["corner"] for r in rows if np.isfinite(r["corner"])]
    h8 = [r["honest8"] for r in rows if r["pnp_ok"] and np.isfinite(r["honest8"])]
    return {
        "n": n,
        "det_pct": round(100 * np.mean([r["det"] for r in rows]), 1),
        "corner_med": round(float(np.median(cor)), 1) if cor else None,
        "good_pct": round(100 * np.mean([c < GOOD_PX for c in cor]), 1) if cor else None,
        "honest8_med": round(float(np.median(h8)), 1) if h8 else None,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sets = {"Val-Old": collect(VAL_OLD, 300),
            "Val-v3": collect(VAL_V3, 300),
            "Val-trunc(train,not-heldout)": collect(VAL_TRUNC, 200)}
    for k, v in sets.items():
        print(f"[set] {k}: {len(v)} frames")

    ckpts = {"B2_ep84": os.path.join(ROOT, "weights/stage11_16k_B2_maskaux/final_net_epoch_0084.pth")}
    for ep in range(85, 95):
        p = os.path.join(ROOT, f"weights/stage16_trunc_replay/net_epoch_{ep:04d}.pth")
        if ep == 94:
            p = os.path.join(ROOT, "weights/stage16_trunc_replay/final_net_epoch_0094.pth")
        if os.path.exists(p):
            ckpts[f"S16_ep{ep}"] = p

    out = {}
    txt = ["# STAGE16-3 ckpt selection (synthetic val; real is judgment-only)",
           "# metric order-free: corner_med(px) good%(<10px) det%(n>=6) honest8(full-8 reproj px)",
           "# Val-trunc = trained-on (NOT held-out) -> in-domain fit only, not a selection criterion",
           ""]
    for name, path in ckpts.items():
        model = load_model(path, device)
        row = {}
        line = [f"{name:<14}"]
        for sk, frames in sets.items():
            rows = [eval_frame(model, jp, ip, device) for jp, ip in frames]
            rows = [r for r in rows if r is not None]
            a_all = agg(rows)
            a_v8 = agg([r for r in rows if r["occ"] == 8])
            row[sk] = {"all": a_all, "V8": a_v8}
            line.append(f"| {sk[:9]:>9} cm={str(a_all['corner_med']):>5} "
                        f"g%={str(a_all['good_pct']):>5} d%={str(a_all['det_pct']):>5} "
                        f"h8={str(a_all['honest8_med']):>5} "
                        f"[V8 cm={str(a_v8.get('corner_med')):>5} d%={str(a_v8.get('det_pct')):>5}]")
        out[name] = row
        txt.append("  ".join(line))
        print("  ".join(line))
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, "ckpt_select.txt"), "w") as f:
        f.write("\n".join(txt) + "\n")
    with open(os.path.join(OUT_DIR, "ckpt_select.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)
    print(f"\nsaved: {OUT_DIR}/ckpt_select.txt , ckpt_select.json")


if __name__ == "__main__":
    main()
