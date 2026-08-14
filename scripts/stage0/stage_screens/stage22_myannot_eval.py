"""stage22_myannot_eval.py — re-eval STAGE19(mixup)/STAGE20(cut-paste) pilots on
USER'S hand-annotated sets (capturepalletcad 22 + capture0403noapril 6).

목적: filter-val 에서 mixup(front+1.3)/cutpaste(front+22) 악화가 손 어노 셋에서도
재현되는지 정직 확인. B2 절대 기준 + paired same-frame(control vs treat).

★ N=cad22 + noapril6 = 28 (noapril 6장 극소) → 정성/예비, 방향성만. 단정 금지.

인프라 재사용: eval_capturecad_b2.eval_frame (order-free Hungarian corner,
solve_pose order-free W/D, honest full-8 reproj, reflect-pad100, per-frame K).
dims: solve_pose 는 PALLET_DIMS(1.1/1.3/0.11)+W/D swap auto → cad(w1.1/d1.3)·
noapril(w1.3/d1.1) 둘 다 magnitude {1.1,1.3} 이라 네이티브 커버. 프레임 GT
projected_cuboid 로 honest8 비교(order-free) → dims 섞임 무관.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
ECAD_PATH = os.path.join(
    ROOT, "data/pallet/eval_results/stage16_truncation_addon/"
    "capturecad_b2_eval/eval_capturecad_b2.py")
OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/stage22_myannot_eval")

_spec = importlib.util.spec_from_file_location("ecad", ECAD_PATH)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

GOOD_PX, GROSS_PX = 10.0, 20.0

DATASETS = {
    "cad": os.path.join(ROOT, "challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt"),
    "noapril": os.path.join(ROOT, "challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt"),
}
MODELS = {
    "B2":     "weights/stage11_16k_B2_maskaux/final_net_epoch_0084.pth",
    "s19_control": "weights/stage19_mixup_pilot/control/final_net.pth",
    "s19_mixup":   "weights/stage19_mixup_pilot/mixup/final_net.pth",
    "s20_control": "weights/stage20_cutpaste_pilot/control/final_net.pth",
    "s20_cutpaste": "weights/stage20_cutpaste_pilot/cutpaste/final_net.pth",
}
PAIRS = [("s19_control", "s19_mixup"), ("s20_control", "s20_cutpaste")]


def frames_of(dirp):
    out = []
    for jp in sorted(glob.glob(os.path.join(dirp, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(dirp, fid + ".png")
        if os.path.exists(ip):
            out.append((jp, ip))
    return out


def per_corner_dists(row):
    """order-free Hungarian per-corner dists (matched GT-idx bucket)."""
    pr = np.array(row["pred8"], float)
    gt = np.array(row["gt8"], float)
    d, ci = E.hungarian(pr, gt)
    return d, ci


def med(a):
    a = [x for x in a if np.isfinite(x)]
    return round(float(np.median(a)), 1) if a else None


def summarize(rows):
    """rows = eval_frame outputs (pad100). Aggregate honest metrics."""
    n = len(rows)
    det = [r for r in rows if r["det"]]
    front = [r["front"] for r in rows if np.isfinite(r["front"])]
    rear = [r["back"] for r in rows if np.isfinite(r["back"])]
    corner = [r["corner"] for r in rows if np.isfinite(r["corner"])]
    worst2 = [r["worst2"] for r in rows if np.isfinite(r["worst2"])]
    honest = [r["pnp_honest8"] for r in rows
              if r["pnp_ok"] and np.isfinite(r["pnp_honest8"])]
    # per-corner good/gross over ALL matched corners
    good = gross = ncorner = 0
    for r in rows:
        d, _ = per_corner_dists(r)
        if d is None:
            continue
        for x in d:
            ncorner += 1
            if x < GOOD_PX:
                good += 1
            if x > GROSS_PX:
                gross += 1
    return {
        "n": n,
        "det_pct": round(100 * len(det) / n, 0) if n else None,
        "front_med": med(front),
        "rear_med": med(rear),
        "corner_med": med(corner),
        "worst2_med": med(worst2),
        "pnp_pct": round(100 * np.mean([r["pnp_ok"] for r in rows]), 0) if n else None,
        "honest8_med": med(honest),
        "good": good, "gross": gross, "ncorner": ncorner,
    }


def main():
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # results[model][dset] = list of eval_frame rows (pad100)
    results = {m: {} for m in MODELS}
    vdist = {}   # vdist[dset][v] = count
    for mname, wrel in MODELS.items():
        mdl = E.load_model(os.path.join(ROOT, wrel), device)
        for dname, dirp in DATASETS.items():
            rows = []
            for jp, ip in frames_of(dirp):
                r = E.eval_frame(mdl, jp, ip, device, 0.3, 100)
                if r is not None:
                    rows.append(r)
            results[mname][dname] = rows
        del mdl
        torch.cuda.empty_cache() if device == "cuda" else None
        print(f"[done] {mname}")

    # V_geom dist (GT-only, take from B2 rows)
    for dname in DATASETS:
        vdist[dname] = {}
        for r in results["B2"][dname]:
            vdist[dname][r["v_geom"]] = vdist[dname].get(r["v_geom"], 0) + 1

    # ---- report ----
    L = []
    L.append("# STAGE22 — mixup/cut-paste re-eval on USER hand-annot sets")
    L.append("# eval: order-free Hungarian corner + solve_pose(order-free W/D) +")
    L.append("#       honest full-8 reproj(GT projected_cuboid, order-free) +")
    L.append("#       reflect-pad100 + per-frame K.  front/rear = matched-GT-idx bucket.")
    L.append("# ★ N=cad22 + noapril6 → 정성/예비. 단정 금지.")
    L.append("")
    for dname in DATASETS:
        vd = vdist[dname]
        L.append(f"# {dname} V_geom(in-frame GT corners): "
                 + ", ".join(f"V{v}:{vd[v]}" for v in sorted(vd)))
    L.append("")

    def hdr():
        return (f"{'model':<14}{'n':>3}{'det%':>6}{'front':>7}{'rear':>7}"
                f"{'corner':>8}{'worst2':>8}{'pnp%':>6}{'honest8':>9}"
                f"{'good':>6}{'gross':>7}")

    out = {"datasets": {}, "paired": {}, "v_geom": {
        k: {str(a): b for a, b in v.items()} for k, v in vdist.items()}}

    # ---- absolute per-model per-dataset ----
    for dname in DATASETS:
        L.append(f"## {dname} (N={len(results['B2'][dname])}) — ABSOLUTE (all frames, pad100)")
        L.append(hdr()); L.append("-" * len(hdr()))
        out["datasets"][dname] = {}
        for mname in MODELS:
            s = summarize(results[mname][dname])
            out["datasets"][dname][mname] = s
            L.append(f"{mname:<14}{s['n']:>3}{str(s['det_pct']):>6}"
                     f"{str(s['front_med']):>7}{str(s['rear_med']):>7}"
                     f"{str(s['corner_med']):>8}{str(s['worst2_med']):>8}"
                     f"{str(s['pnp_pct']):>6}{str(s['honest8_med']):>9}"
                     f"{s['good']:>6}{s['gross']:>7}")
        L.append("")

    # ---- paired same-frame (both arms detect n_det>=6) ----
    L.append("## PAIRED same-frame (both arms det>=6; front+1.3/+22 재현 확인)")
    for a, b in PAIRS:
        for dname in DATASETS:
            ra = {r["fid"]: r for r in results[a][dname]}
            rb = {r["fid"]: r for r in results[b][dname]}
            fids = [f for f in ra if ra[f]["det"] and rb[f].get("det")]
            sa = summarize([ra[f] for f in fids])
            sb = summarize([rb[f] for f in fids])
            key = f"{a}_vs_{b}__{dname}"
            out["paired"][key] = {"npair": len(fids), a: sa, b: sb}
            L.append(f"\n[{a} vs {b}] {dname}  paired n={len(fids)}")
            L.append(hdr()); L.append("-" * len(hdr()))
            for nm, s in ((a, sa), (b, sb)):
                L.append(f"{nm:<14}{s['n']:>3}{str(s['det_pct']):>6}"
                         f"{str(s['front_med']):>7}{str(s['rear_med']):>7}"
                         f"{str(s['corner_med']):>8}{str(s['worst2_med']):>8}"
                         f"{str(s['pnp_pct']):>6}{str(s['honest8_med']):>9}"
                         f"{s['good']:>6}{s['gross']:>7}")
            # deltas (treat - control) on paired
            df = (sb['front_med'] - sa['front_med']
                  if sa['front_med'] and sb['front_med'] else None)
            dr = (sb['rear_med'] - sa['rear_med']
                  if sa['rear_med'] and sb['rear_med'] else None)
            dh = (sb['honest8_med'] - sa['honest8_med']
                  if sa['honest8_med'] and sb['honest8_med'] else None)
            L.append(f"  Δ(treat-control): front={df} rear={dr} honest8={dh} "
                     f"(양수=악화)")
            out["paired"][key]["delta"] = {"front": df, "rear": dr, "honest8": dh}

    report = "\n".join(L)
    print("\n" + report)
    with open(os.path.join(OUT_DIR, "myannot_paired.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(OUT_DIR, "myannot_eval.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)

    # ---- overlays: same-frame B2 vs s20_cutpaste vs s19_mixup ----
    ovdir = os.path.join(OUT_DIR, "overlays")
    os.makedirs(ovdir, exist_ok=True)
    import cv2
    ov_models = ["B2", "s20_cutpaste", "s19_mixup"]
    colors = {"B2": (0, 255, 255), "s20_cutpaste": (0, 0, 255),
              "s19_mixup": (255, 0, 255)}
    n_saved = 0
    for dname in DATASETS:
        # pick up to 3 frames spread by B2 honest8
        b2rows = [r for r in results["B2"][dname]
                  if r["pnp_ok"] and np.isfinite(r["pnp_honest8"])]
        b2rows.sort(key=lambda r: r["pnp_honest8"])
        picks = []
        if b2rows:
            idxs = sorted(set([0, len(b2rows) // 2, len(b2rows) - 1]))
            picks = [b2rows[i]["fid"] for i in idxs]
        for fid in picks:
            base = {r["fid"]: r for r in results["B2"][dname]}[fid]
            img = cv2.imread(base["ip"])
            if img is None:
                continue
            gt8 = np.array(base["gt8"], float)
            for p in gt8:
                cv2.circle(img, (int(p[0]), int(p[1])), 6, (0, 255, 0), 2)
            for mi, mname in enumerate(ov_models):
                rr = {r["fid"]: r for r in results[mname][dname]}.get(fid)
                if rr is None:
                    continue
                pr = np.array(rr["pred8"], float)
                for p in pr:
                    if np.isnan(p[0]):
                        continue
                    cv2.circle(img, (int(p[0]), int(p[1])), 3,
                               colors[mname], -1)
                cv2.putText(img, f"{mname} h8={rr['pnp_honest8']:.1f}",
                            (8, 20 + 18 * mi), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colors[mname], 2)
            cv2.putText(img, "GT=green", (8, 20 + 18 * len(ov_models)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(ovdir, f"{dname}_{fid}.jpg"), img)
            n_saved += 1

    print(f"\n[save] {OUT_DIR}/myannot_paired.txt , myannot_eval.json")
    print(f"[save] overlays: {ovdir} ({n_saved} imgs)")


if __name__ == "__main__":
    main()
