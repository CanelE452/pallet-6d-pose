"""stage22_filter_funnel.py — does best model produce CLEAN pseudo-labels on the
USER's hand-annotated GT sets (cad 22 + noapril 6)?

목적: best 모델(B2, +참고 s19_mixup) 예측이 우리 self-training 기하 필터
     (diag / flip / loo(=ransac_loo) / diag∧flip)를 통과하는가 + ★통과분이
     GT 대비 실제로 정확(clean)한가. 6/18 funnel(clean≈0)·STAGE19-A funnel의
     연장선, 이번엔 GT 있는 손 어노 셋.

인프라 재사용(새 기하 0):
  eval_capturecad_b2.eval_frame  : pad100 pred8/pred_c, order-free Hungarian
      corner/front/back, honest full-8 reproj(solve_pose ITERATIVE, order-free W/D).
  filter_pr_camfacing            : filt_diag, canonical_kp3d, ransac_consensus,
      loo_stability, extract_keypoints_from_belief, belief 매핑.
  filter_flip_consistency        : FLIP_PAIRS, flip_consistency_score (그러나
      flip 추론은 pad100 로 재구현 — kpA(pad100)와 같은 regime 유지가 핵심).

Threshold (★ sweep 아님, 관례값 고정 — 의심 §5):
  diag_resid : 0.02 (task 지정) + 0.05 (code/deployed anchor) 둘 다.
  flip       : 8 px (task) + 10 px (deployed FLIP_TAU) 둘 다.
  loo        : ransac consensus c>=6 (EPNP, tau=5px) AND loo_stability tau=0.05
               (filter_pr_camfacing 관례 = ransac_loo). ★loo 는 필터용 EPNP —
               eval/거리추정 pose 는 solve_pose(ITERATIVE, honest8)로 별도.

Precision(통과분): clean = frame order-free 8-corner median < 10px (GOOD_PX).
  또 front(0-3)/rear(4-7) 분리 median, honest8 median, per-corner good/gross.
  ★통과=정확 성립? 특히 통과했는데 rear 틀린 confidently-wrong 집계.

★ N=cad22 + noapril6 = 28 극소표본 → 정성/예비. 단정 금지.
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
OUT_DIR = os.path.join(
    ROOT, "data/pallet/eval_results/stage22_myannot_eval/filter_funnel")

sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

_spec = importlib.util.spec_from_file_location("ecad", ECAD_PATH)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

from filter_pr_camfacing import (  # noqa: E402
    filt_diag, canonical_kp3d, ransac_consensus, loo_stability,
)
from filter_flip_consistency import FLIP_PAIRS  # noqa: E402

GOOD_PX, GROSS_PX = 10.0, 20.0
FRONT, BACK = [0, 1, 2, 3], [4, 5, 6, 7]

# per-dataset dims (width, depth, height) — GT known
DATASETS = {
    "cad": {
        "dir": os.path.join(ROOT, "challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt"),
        "dims": (1.1, 1.3, 0.11)},   # w, d, h
    "noapril": {
        "dir": os.path.join(ROOT, "challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt"),
        "dims": (1.3, 1.1, 0.11)},
}
MODELS = {
    "B2":       "weights/stage11_16k_B2_maskaux/final_net_epoch_0084.pth",
    "s19_mixup": "weights/stage19_mixup_pilot/mixup/final_net.pth",
}

# threshold pairs (primary = task-specified; alt = deployed/code)
DIAG_TAUS = [0.02, 0.05]
FLIP_TAUS = [8.0, 10.0]
LOO_TAU = 0.05
PRIMARY = {"diag": 0.02, "flip": 8.0}


def frames_of(dirp):
    out = []
    for jp in sorted(glob.glob(os.path.join(dirp, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(dirp, fid + ".png")
        if os.path.exists(ip):
            out.append((jp, ip))
    return out


def build_swap():
    swap = list(range(9))
    for a, b in FLIP_PAIRS:
        swap[a], swap[b] = b, a
    return swap


_SWAP = build_swap()


def flip_kp_pad100(model, img, device, threshold, pad):
    """Flip inference on the SAME pad100 regime as eval_frame → 9 kp in original
    px, un-flipped(x -> W-x) + camera-facing symmetric swap. Keeps kpA/kpB on the
    same coordinate + detection regime (memory: pad100 essential for near/trunc)."""
    import cv2
    import torch
    H, W = img.shape[:2]
    img_flip = cv2.flip(img, 1)
    proc = E.pad_frame(img_flip, pad)
    tensor, nw, nh, sc = E.preprocess(proc)
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps_bel = E.extract_keypoints_from_belief(belief, threshold)
    flip_px = [None] * 9
    for i, k in enumerate(kps_bel[:9]):
        if k[0] < 0:
            continue
        u, v = E.belief_to_orig_pad(k[0], k[1], bw, bh, nw, nh, sc, pad, W, H)
        flip_px[i] = (W - u, v)   # un-flip x back to original orientation
    out = [None] * 9
    for i in range(9):
        out[i] = flip_px[_SWAP[i]]
    return out


def flip_score(kpA, kpB):
    d = []
    for i in range(9):
        if kpA[i] is not None and kpB[i] is not None:
            d.append(np.linalg.norm(np.asarray(kpA[i], float)
                                    - np.asarray(kpB[i], float)))
    if len(d) < 4:
        return None
    return float(np.mean(d))


def kp9_from_row(r):
    """pred8(list, NaN) + pred_c -> length-9 list of (x,y) or None."""
    pr = np.array(r["pred8"], float)
    kp = [None if np.isnan(pr[i, 0]) else (float(pr[i, 0]), float(pr[i, 1]))
          for i in range(8)]
    kp.append(tuple(r["pred_c"]) if r["pred_c"] is not None else None)
    return kp


def per_corner(row):
    pr = np.array(row["pred8"], float)
    gt = np.array(row["gt8"], float)
    d, ci = E.hungarian(pr, gt)
    return d, ci


def precision_of(rows):
    """Precision metrics of a passing subset (order-free, GT known)."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    corner = [r["corner"] for r in rows if np.isfinite(r["corner"])]
    front = [r["front"] for r in rows if np.isfinite(r["front"])]
    rear = [r["back"] for r in rows if np.isfinite(r["back"])]
    honest = [r["pnp_honest8"] for r in rows
              if r["pnp_ok"] and np.isfinite(r["pnp_honest8"])]
    # frame-level clean = order-free 8-corner median < GOOD_PX
    clean = [r for r in rows if np.isfinite(r["corner"]) and r["corner"] < GOOD_PX]
    gross_fr = [r for r in rows if np.isfinite(r["corner"]) and r["corner"] > GROSS_PX]
    # confidently-wrong: passed but rear median > GROSS (front可以 fine)
    conf_wrong = [r for r in rows
                  if np.isfinite(r["back"]) and r["back"] > GROSS_PX]
    # per-corner good/gross over matched corners
    good = gross = ncorner = 0
    for r in rows:
        d, _ = per_corner(r)
        if d is None:
            continue
        for x in d:
            ncorner += 1
            good += int(x < GOOD_PX)
            gross += int(x > GROSS_PX)

    def med(a):
        a = [x for x in a if np.isfinite(x)]
        return round(float(np.median(a)), 1) if a else None

    return {
        "n": n,
        "clean_n": len(clean),
        "clean_pct": round(100 * len(clean) / n, 0),
        "gross_frame_n": len(gross_fr),
        "conf_wrong_n": len(conf_wrong),   # passed but rear>20px
        "corner_med": med(corner),
        "front_med": med(front),
        "rear_med": med(rear),
        "honest8_med": med(honest),
        "percorner_good": good, "percorner_gross": gross, "ncorner": ncorner,
    }


def main():
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # results[model][dset] = list of dict rows (eval_frame + filter scores)
    results = {m: {} for m in MODELS}
    for mname, wrel in MODELS.items():
        mdl = E.load_model(os.path.join(ROOT, wrel), device)
        for dname, dd in DATASETS.items():
            width, depth, height = dd["dims"]
            kp3d = canonical_kp3d(width, depth, height)
            K_dist = np.zeros((5, 1))
            rows = []
            for jp, ip in frames_of(dd["dir"]):
                r = E.eval_frame(mdl, jp, ip, device, 0.3, 100)
                if r is None:
                    continue
                d = json.load(open(jp))
                K = E.K_from_json(d)
                kp9 = kp9_from_row(r)
                # diag (score once, threshold later)
                _, diag_sc = filt_diag(kp9)
                r["diag_score"] = float(diag_sc) if np.isfinite(diag_sc) else None
                # flip (pad100 consistent)
                import cv2
                img = cv2.imread(ip)
                kpB = flip_kp_pad100(mdl, img, device, 0.3, 100)
                r["flip_score"] = flip_score(kp9, kpB)
                # loo = ransac consensus(c>=6, EPNP) AND loo_stability(tau=0.05)
                n_cons, R_rs, t_rs = ransac_consensus(kp9, kp3d, K, K_dist)
                ransac_pass = n_cons >= 6
                loo_ok = (loo_stability(kp9, kp3d, K, K_dist, R_rs, t_rs,
                                        tau=LOO_TAU)
                          if (ransac_pass and R_rs is not None) else False)
                r["ransac_cons"] = int(n_cons)
                r["loo_pass"] = bool(ransac_pass and loo_ok)
                rows.append(r)
            results[mname][dname] = rows
        del mdl
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[done] {mname}")

    # ---- funnel + precision ----
    def diag_pass(r, tau):
        return r["diag_score"] is not None and r["diag_score"] < tau

    def flip_pass(r, tau):
        return r["flip_score"] is not None and r["flip_score"] <= tau

    out = {"good_px": GOOD_PX, "gross_px": GROSS_PX, "loo_tau": LOO_TAU,
           "diag_taus": DIAG_TAUS, "flip_taus": FLIP_TAUS,
           "primary": PRIMARY, "sets": {}}
    L = []
    L.append("# STAGE22 FILTER FUNNEL — best model clean-PL check on USER hand-annot")
    L.append("# funnel: detect -> diag / flip / loo(=ransac_loo) / diag&flip")
    L.append("# ★통과분 precision(GT known): clean=order-free 8corner med<10px;")
    L.append("#   front/rear=matched-GT-idx bucket; honest8=solve_pose(ITERATIVE)")
    L.append("#   8corner(GTdim) vs GT projected_cuboid, order-free.")
    L.append(f"# thresholds(고정,NOT sweep): diag<{DIAG_TAUS} flip<={FLIP_TAUS} "
             f"loo(ransac c>=6 & loo_stab tau={LOO_TAU})")
    L.append(f"# PRIMARY = diag<{PRIMARY['diag']} flip<={PRIMARY['flip']}")
    L.append("# ★ N=cad22+noapril6 극소표본 → 정성/예비. 단정 금지.")
    L.append("")

    filters = []
    for dt in DIAG_TAUS:
        filters.append((f"diag<{dt}", lambda r, dt=dt: diag_pass(r, dt)))
    for ft in FLIP_TAUS:
        filters.append((f"flip<={ft:.0f}", lambda r, ft=ft: flip_pass(r, ft)))
    filters.append(("loo(ransac_loo)", lambda r: r["loo_pass"]))
    # diag&flip at primary + alt
    filters.append((f"diag<{PRIMARY['diag']}&flip<={PRIMARY['flip']:.0f}",
                    lambda r: diag_pass(r, PRIMARY["diag"])
                    and flip_pass(r, PRIMARY["flip"])))
    filters.append(("diag<0.05&flip<=10",
                    lambda r: diag_pass(r, 0.05) and flip_pass(r, 10.0)))

    def hdr():
        return (f"{'filter':<24}{'pass':>5}{'clean':>6}{'clean%':>7}"
                f"{'grossF':>7}{'confWr':>7}{'cor':>6}{'front':>6}{'rear':>6}"
                f"{'hon8':>6}")

    for mname in MODELS:
        L.append(f"{'='*90}")
        L.append(f"## MODEL = {mname}")
        out["sets"][mname] = {}
        for dname in DATASETS:
            rows = results[mname][dname]
            det = [r for r in rows if r["det"]]
            # baseline detect precision
            base = precision_of(det)
            L.append(f"\n### {dname}  (N={len(rows)}, detected={len(det)})")
            # V_geom dist
            vd = {}
            for r in rows:
                vd[r["v_geom"]] = vd.get(r["v_geom"], 0) + 1
            L.append("   V_geom(in-frame GT corners): "
                     + ", ".join(f"V{v}:{vd[v]}" for v in sorted(vd)))
            L.append(hdr())
            L.append("-" * len(hdr()))

            def line(label, s):
                if s["n"] == 0:
                    return f"{label:<24}{0:>5}{'--':>6}{'--':>7}{'--':>7}{'--':>7}{'--':>6}{'--':>6}{'--':>6}{'--':>6}"
                return (f"{label:<24}{s['n']:>5}{s['clean_n']:>6}"
                        f"{s['clean_pct']:>6.0f}%{s['gross_frame_n']:>7}"
                        f"{s['conf_wrong_n']:>7}{str(s['corner_med']):>6}"
                        f"{str(s['front_med']):>6}{str(s['rear_med']):>6}"
                        f"{str(s['honest8_med']):>6}")

            L.append(line("detect(baseline)", base))
            dset_out = {"n": len(rows), "detected": len(det),
                        "v_geom": {str(k): v for k, v in vd.items()},
                        "detect_baseline": base, "filters": {}}
            for label, fn in filters:
                passed = [r for r in det if fn(r)]
                s = precision_of(passed)
                dset_out["filters"][label] = s
                L.append(line(label, s))
            out["sets"][mname][dname] = dset_out
        L.append("")

    report = "\n".join(L)
    print("\n" + report)
    with open(os.path.join(OUT_DIR, "filter_funnel.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(OUT_DIR, "filter_funnel.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)
    print(f"\n[save] {OUT_DIR}/filter_funnel.txt , filter_funnel.json")


if __name__ == "__main__":
    main()
