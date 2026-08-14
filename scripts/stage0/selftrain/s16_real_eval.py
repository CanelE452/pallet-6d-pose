"""s16_real_eval.py — STAGE16-3 최종 real 판정 (B2 vs S16, order-free, reflect-pad).

가설(STAGE17/18): 진짜 병목 = rear 코너(4-7) depth 붕괴 @ 저앙각(≲8deg). trunc_addon
replay-mix 로 저앙각 데이터를 주입해 이를 완화하려 함. 이 스크립트는 그 실제 판정.

real = collect_val_frames(outside+night filter-val) + collect_manual(36) = 123장.
  ★ final-test(p07/p09,n08/n09) 절대 미포함 (collect_val_frames SEAL guard).
전처리 = reflect-pad 100 (B2 near/truncation eval 관례; memory dope-inference-needs-reflect-padding).

지표 (order-free):
  - Hungarian corner: overall/front(0-3)/rear(4-7) median px  [order-free]
  - det% (n_det>=6), good%(corner<10px), worst2 (top-2 corner med)
  - honest full-8 reproj: solve_pose(9kp,per-frame K,GT dims) 8코너 투영 vs GT
    projected_cuboid, Hungarian order-free (화면밖 코너 포함) -> 정직한 pose 오차
  - PnP success%
bins: ALL / V=8 / V<8 / V7 / V6 / V5 / V4 / near-large / border /
      ★ elevation(<3/3-8/8-15/15-25/25+, STAGE18 elev_from_pose) + corner front/rear.

출력: data/pallet/eval_results/stage16_truncation_addon/stage16_eval/
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import argparse, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "data", "pallet", "eval_results", "stage11_16k"))

import cv2, torch
from filter_pr_camfacing import extract_keypoints_from_belief
from eval_pvnet_heads import preprocess, split_metrics
from eval_stage11 import load_model, K_from_json, dims_from_json
from tau_calibrate import collect_val_frames
from eval_pvnet_heads import collect_manual
from stage18_elevation_threshold import elev_from_pose
import annotate_pnp as APNP

THRESHOLD, N_DET_MIN, GOOD_PX, PAD = 0.3, 6, 10.0, 100
FRONT, REAR = [0, 1, 2, 3], [4, 5, 6, 7]
SPIKE_PX = 15.0
OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/stage16_truncation_addon/stage16_eval")
ELEV_BINS = [(-90, 3), (3, 8), (8, 15), (15, 25), (25, 90)]
ELEV_LBL = ["<3", "3-8", "8-15", "15-25", "25+"]


def pad_frame(img, pad):
    if pad <= 0:
        return img
    h, w = img.shape[:2]
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    return cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR)


def belief_to_orig_pad(bx, by, bw, bh, nw, nh, sc, pad, W, H):
    ux, uy = nw / bw, nh / bh
    cx = (bx * ux) / sc
    cy = (by * uy) / sc
    if pad > 0:
        return cx * (W + 2 * pad) / W - pad, cy * (H + 2 * pad) / H - pad
    return cx, cy


def hungarian(pred, gt):
    valid = ~np.isnan(pred[:, 0])
    if valid.sum() < N_DET_MIN:
        return None, None, None
    from scipy.optimize import linear_sum_assignment
    P = pred[valid]
    validi = np.where(valid)[0]
    cost = np.linalg.norm(P[:, None, :] - gt[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return cost[ri, ci], ci, validi[ri]


def eval_frame(model, jp, ip, device):
    img = cv2.imread(ip)
    if img is None:
        return None
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    H, W = img.shape[:2]
    K = K_from_json(d)
    dims = dims_from_json(d)

    inb = ((gt8[:, 0] >= 0) & (gt8[:, 0] < W) & (gt8[:, 1] >= 0) & (gt8[:, 1] < H))
    V = int(inb.sum())
    proj_size = (float(np.hypot(gt8[inb, 0].ptp(), gt8[inb, 1].ptp()))
                 if inb.sum() >= 2 else np.nan)
    depth = np.nan
    elev = None
    pt = o.get("pose_transform")
    if pt:
        T = np.array(pt, float)
        if T.shape == (4, 4):
            depth = float(T[2, 3])
            elev = elev_from_pose(T)
    border = bool(V < 8)  # any GT corner off-frame

    proc = pad_frame(img, PAD)
    tensor, nw, nh, sc = preprocess(proc)
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps_bel = extract_keypoints_from_belief(belief, THRESHOLD)

    pred8 = np.full((8, 2), np.nan)
    for i, k in enumerate(kps_bel[:8]):
        if k[0] < 0:
            continue
        pred8[i] = belief_to_orig_pad(k[0], k[1], bw, bh, nw, nh, sc, PAD, W, H)
    pred_c = None
    if kps_bel[8][0] >= 0:
        pred_c = list(belief_to_orig_pad(kps_bel[8][0], kps_bel[8][1],
                                         bw, bh, nw, nh, sc, PAD, W, H))

    m = split_metrics(pred8, gt8)
    dists, ci, _ = hungarian(pred8, gt8)
    worst2 = (float(np.mean(np.sort(dists)[-2:]))
              if dists is not None and len(dists) >= 2 else np.inf)
    # per-GT-corner matched error (order-free) -> front/rear split
    e_corner = [None] * 8
    if dists is not None:
        for dd, gi in zip(dists, ci):
            e_corner[int(gi)] = float(dd)
    ef = [e_corner[i] for i in FRONT if e_corner[i] is not None]
    er = [e_corner[i] for i in REAR if e_corner[i] is not None]
    e_front = float(np.mean(ef)) if ef else np.nan
    e_rear = float(np.mean(er)) if er else np.nan

    # honest full-8 reproj (order-free)
    kps9 = [None if np.isnan(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps9.append(pred_c)
    honest8, pnp_ok, sel_reproj = np.inf, 0, np.inf
    if sum(1 for k in kps9 if k is not None) >= N_DET_MIN:
        try:
            pose = APNP.solve_pose(kps9, K, dims=dims, img_shape=img.shape)
            if pose is not None:
                pnp_ok = 1
                sel_reproj = float(pose["reproj_error_px"])
                pa = np.array(pose["projected_all"], float)[:8]
                bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
                pa[bad] = np.nan
                hd, _, _ = hungarian(pa, gt8)
                if hd is not None:
                    honest8 = float(np.mean(hd))
        except Exception:
            pass

    return {"occ": V, "proj_size": proj_size, "depth": depth,
            "elev": (round(float(elev), 2) if elev is not None else None),
            "border": border, "corner": m["overall"], "n_det": m["n_det"],
            "det": 1 if m["n_det"] >= N_DET_MIN else 0, "worst2": worst2,
            "e_front": e_front, "e_rear": e_rear,
            "n_front_det": len(ef), "n_rear_det": len(er),
            "honest8": honest8, "sel_reproj": sel_reproj, "pnp_ok": pnp_ok,
            "gt8": gt8.tolist(), "pred8": pred8.tolist(), "ip": ip}


def agg(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    cor = [r["corner"] for r in rows if np.isfinite(r["corner"])]
    ef = [r["e_front"] for r in rows if np.isfinite(r["e_front"])]
    er = [r["e_rear"] for r in rows if np.isfinite(r["e_rear"])]
    w2 = [r["worst2"] for r in rows if np.isfinite(r["worst2"])]
    h8 = [r["honest8"] for r in rows if r["pnp_ok"] and np.isfinite(r["honest8"])]
    rear_all = [r["e_rear"] for r in rows if np.isfinite(r["e_rear"])]
    return {
        "n": n,
        "det_pct": round(100 * np.mean([r["det"] for r in rows]), 1),
        "corner_med": round(float(np.median(cor)), 1) if cor else None,
        "good_pct": round(100 * np.mean([c < GOOD_PX for c in cor]), 1) if cor else None,
        "front_med": round(float(np.median(ef)), 1) if ef else None,
        "rear_med": round(float(np.median(er)), 1) if er else None,
        "rear_spike_pct": (round(100 * np.mean([e > SPIKE_PX for e in rear_all]), 1)
                           if rear_all else None),
        "worst2_med": round(float(np.median(w2)), 1) if w2 else None,
        "pnp_ok_pct": round(100 * np.mean([r["pnp_ok"] for r in rows]), 1),
        "honest8_med": round(float(np.median(h8)), 1) if h8 else None,
    }


def line(label, s):
    return (f"{label:<12} n={s.get('n', 0):<4} "
            f"det%={str(s.get('det_pct')):<6} "
            f"good%={str(s.get('good_pct')):<6} "
            f"cor={str(s.get('corner_med')):<6} "
            f"front={str(s.get('front_med')):<6} "
            f"rear={str(s.get('rear_med')):<6} "
            f"rspk%={str(s.get('rear_spike_pct')):<6} "
            f"w2={str(s.get('worst2_med')):<6} "
            f"pnp%={str(s.get('pnp_ok_pct')):<6} "
            f"h8={str(s.get('honest8_med')):<6}")


def overlay_pair(rowB2, rowS16, out_path):
    def draw(row, tag):
        img = cv2.imread(row["ip"])
        if img is None:
            return None
        gt8 = np.array(row["gt8"], float)
        pr8 = np.array(row["pred8"], float)
        EDG = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
               (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in EDG:
            cv2.line(img, tuple(gt8[a].astype(int)), tuple(gt8[b].astype(int)),
                     (60, 200, 60), 1)
        for i in range(8):
            col = (0, 0, 255) if i in FRONT else (255, 128, 0)
            if not np.isnan(pr8[i, 0]):
                cv2.circle(img, (int(pr8[i, 0]), int(pr8[i, 1])), 4, col, -1)
                cv2.line(img, (int(pr8[i, 0]), int(pr8[i, 1])),
                         tuple(gt8[i].astype(int)), col, 1)
        el = row.get("elev")
        txt = (f"{tag} V={row['occ']} el={el} rear={row['e_rear']:.0f} "
               f"h8={row['honest8']:.0f}")
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        return img
    a = draw(rowB2, "B2")
    b = draw(rowS16, "S16")
    if a is None or b is None:
        return
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1] + 4, 3), np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1] + 4:] = b
    cv2.imwrite(out_path, canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = collect_val_frames() + collect_manual()
    print(f"[real] eval frames = {len(frames)} (filter-val outside+night + manual; final-test SEALED)")

    results = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        print(f"=== {name} : {path} ===")
        model = load_model(path, device)
        rows = []
        for dom, fid, jp, ip in frames:
            r = eval_frame(model, jp, ip, device)
            if r is not None:
                r["dom"] = dom
                r["fid"] = fid
                rows.append(r)
        results[name] = rows
        del model
        torch.cuda.empty_cache()

    names = list(results.keys())

    def vsel(rows, occ):
        return [r for r in rows if r["occ"] == occ]

    bins = {}
    bins["ALL"] = lambda rows: rows
    bins["V=8"] = lambda rows: [r for r in rows if r["occ"] == 8]
    bins["V<8"] = lambda rows: [r for r in rows if 0 <= r["occ"] < 8]
    for v in (7, 6, 5, 4):
        bins[f"V{v}"] = (lambda rows, vv=v: [r for r in rows if r["occ"] == vv])
    bins["near-large"] = (lambda rows: [r for r in rows
                          if np.isfinite(r["depth"]) and r["depth"] < 3.24
                          and np.isfinite(r["proj_size"]) and r["proj_size"] >= 286.7])
    bins["border"] = lambda rows: [r for r in rows if r["border"]]

    out = {"names": names, "n_frames": len(frames), "pad": PAD,
           "tables": {}, "per_frame": {}}
    txt = [f"# STAGE16-3 real eval — {len(frames)} frames (filter-val outside+night + manual GT)",
           f"# reflect-pad={PAD}. order-free Hungarian corner + honest full-8 reproj.",
           "# cor=overall corner med px | front=corner0-3 | rear=corner4-7 | "
           "rspk%=rear>15px | h8=honest full-8 reproj (pose error) | det%=n>=6",
           "# ★ V=8 회귀 없어야(det/good 유지) + V<8 det↑ + 저앙각 rear↓ 가 성공.",
           ""]

    for bk, fn in bins.items():
        txt.append(f"## {bk}")
        tab = {}
        for nm in names:
            s = agg(fn(results[nm]))
            tab[nm] = s
            txt.append(line(nm, s))
        out["tables"][bk] = tab
        txt.append("")

    # ---- elevation bins x front/rear (the direct target) ----
    txt.append("## ★ ELEVATION bins (STAGE18 elev_from_pose; edge-on~0 top-down~90)")
    txt.append("#   rear = corner 4-7 median err (the injection target)")
    elev_tab = {}
    for lbl, (lo, hi) in zip(ELEV_LBL, ELEV_BINS):
        txt.append(f"-- elev {lbl} deg --")
        row = {}
        # bin n from first model (GT-only)
        for nm in names:
            sel = [r for r in results[nm]
                   if r["elev"] is not None and lo <= r["elev"] < hi]
            s = agg(sel)
            row[nm] = s
            txt.append(line(nm, s))
        elev_tab[lbl] = row
        txt.append("")
    out["tables"]["elevation"] = elev_tab

    # elevation coverage (real)
    elevs = [r["elev"] for r in results[names[0]] if r["elev"] is not None]
    out["real_elev_available"] = len(elevs)
    txt.append(f"# real frames with elevation: {len(elevs)}/{len(frames)}  "
               f"range=[{min(elevs):.1f},{max(elevs):.1f}]" if elevs else "# no elev")

    # per-frame dump (compact) for both models keyed by fid
    for nm in names:
        out["per_frame"][nm] = [
            {k: r[k] for k in ("dom", "fid", "occ", "elev", "depth", "corner",
                               "e_front", "e_rear", "det", "honest8", "pnp_ok",
                               "worst2")}
            for r in results[nm]]

    report = "\n".join(txt)
    print("\n" + report)
    with open(os.path.join(OUT_DIR, "real_eval.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(OUT_DIR, "real_eval.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)

    # ---- overlays: B2 vs S16 on low-elevation frames (<8 deg), by fid ----
    if len(names) >= 2:
        b2n, s16n = names[0], names[1]
        by_fid = {(r["dom"], r["fid"]): r for r in results[b2n]}
        low = [r for r in results[s16n]
               if r["elev"] is not None and r["elev"] < 8]
        low.sort(key=lambda r: (r["elev"] if r["elev"] is not None else 99))
        ovdir = os.path.join(OUT_DIR, "overlays_lowelev")
        os.makedirs(ovdir, exist_ok=True)
        nov = 0
        for r16 in low[:14]:
            rB2 = by_fid.get((r16["dom"], r16["fid"]))
            if rB2 is None:
                continue
            op = os.path.join(ovdir, f"el{r16['elev']:.0f}_{r16['dom']}_{r16['fid']}.jpg")
            overlay_pair(rB2, r16, op)
            nov += 1
        print(f"[overlays] {nov} low-elev B2|S16 pairs -> {ovdir}")

    print(f"\nsaved: {OUT_DIR}/real_eval.txt , real_eval.json")


if __name__ == "__main__":
    main()
