"""ralph_eval_3d_reliability.py — re-judge ralph self-training (ransac_loo strict PL)
by (A) 3D-space error via pseudo-GT PnP and (B) heatmap confidence "reliability score",
instead of 2D px only.

WHY: 2D corner_med (ralph_eval_all) can under/over-state pose quality (order-free px on a
flat pallet hides depth collapse). Here two orthogonal axes:
  (A) 3D error: GT 2D 8-corner + per-frame dims + K -> solve_pose = pseudo-GT 6D pose.
      pred 2D 8-corner + same dims + K -> pred 6D pose. Report ADD(m), |t|(m), rot(deg).
      ★ pseudo-GT is PnP-from-2D, NOT metrology; absolute 6D limited. Same dims/K for
        GT and pred so the DELTA between models is fair.
  (B) reliability = per-corner belief peak signed by correctness (user-defined):
      correct_i = 1 if ||pred_i - GT_i|| < tau_px (same-index; DOPE belief map i == corner i,
      camera-facing 0123, convention verified: split_metrics front/back ~= hungarian corner).
      w_i = peaks[i] (belief peak 0..1). reliability_i = w_i*(2*correct_i-1).
      frame score = mean_i reliability_i (i=0..7), range [-1,1]. Higher = confidence aligned
      with accuracy. undetected corner -> peak<THRESH ~ small weight ~= 0 contribution.

REUSE (no re-impl): ralph_eval_all.DOMAINS (same held-out eval GT folders, same dedup),
  paper_s2_testset17_9filters.infer_squash / T.E (load_model, hungarian, K_from_json),
  annotate_pnp.solve_pose (camera-facing 0123, auto W/D swap), make_pallet_keypoints_3d.
All models are squash-parity (s2 lineage) — same infer_squash as ralph_eval_all.

Usage: conda activate pallet-pose; python -u scripts/stage0/ralph_eval_3d_reliability.py
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import ralph_eval_all as RA               # noqa: E402  (DOMAINS — same held-out eval sets)
import paper_s2_testset17_9filters as T   # noqa: E402  (infer_squash, E, THRESH, N_DET_MIN)
import annotate_pnp as APNP               # noqa: E402  (solve_pose, make_pallet_keypoints_3d)
import cv2                                # noqa: E402
import torch                              # noqa: E402
import matplotlib                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402

DOMAINS = RA.DOMAINS
NMIN = T.N_DET_MIN
TAU_PX = 20.0          # confident-wrong tracker threshold (cor_pk/wro_pk only)
TAU_ADD = [0.10, 0.15]  # m; usable-pose rate thresholds (user default, approved)
GROSS_PX = 20.0        # 9kp-median px -> gross frame
CATAS_PX = 40.0        # 9kp-median px -> catastrophic frame


def grade(err):
    """user value fn: <=10px great(+1), 10..40 lenient linear +1->0,
    40..80 linear 0->-1, >80px bad(-1). rewards 'roughly right', punishes
    only confident-huge-wrong."""
    if not np.isfinite(err):
        return -1.0
    if err <= 10.0:
        return 1.0
    if err <= 40.0:
        return 1.0 - (err - 10.0) / 30.0
    if err <= 80.0:
        return -(err - 40.0) / 40.0
    return -1.0

OUT_DIR = os.path.join(ROOT, "data", "pallet", "results", "ralph_selftrain")
SB = os.path.join(ROOT, "data", "pallet", "results", "ralph_selftrain")

# label -> weights (ROOT-relative). R0 = original s2; ctrl = synthetic-only (PL=0);
# self-domain per-domain; combined = best.
MODELS = {
    "R0":        "weights/paper_s2_stageB/net_epoch_0057_noseg.pth",
    "ctrl":      f"{SB}/ctrl_s2_synonly/round_02.pth",
    "st_outside": f"{SB}/h6_s2_outside/round_02.pth",
    "st_night":   f"{SB}/h3_s2_night/round_02.pth",
    "st_noapril": f"{SB}/h7_s2_noapril/round_02.pth",
    "combined":   f"{SB}/h4_s2_combined/round_02.pth",
}
# per-domain "self-train (self-domain)" model; cad has no domain-specific self-train.
SELF_OF = {"outside": "st_outside", "night": "st_night",
           "noapril": "st_noapril", "cad": None}
# models shown in per-domain table (in order)
TABLE_MODELS = ["R0", "ctrl", "SELF", "combined"]

KP3D8 = APNP.make_pallet_keypoints_3d()[:8]   # canonical 8 corners (camera-facing 0123)


def load_eval():
    """Same held-out eval frames as ralph_eval_all + per-frame K/dims/centroid."""
    frames = {}
    for dom, fos in DOMAINS.items():
        seen = {}
        for fo in fos:
            for jf in sorted(glob.glob(os.path.join(ROOT, fo, "*.json"))):
                fid = os.path.splitext(os.path.basename(jf))[0]
                if fid in seen:
                    continue
                ip = jf[:-5] + ".png"
                if not os.path.isfile(ip):
                    continue
                try:
                    d = json.load(open(jf))
                    o = d["objects"][0]
                    gt8 = np.array(o["projected_cuboid"], float)[:8]
                    gtc = o.get("projected_cuboid_centroid")
                    K = T.E.K_from_json(d)
                    dm = o.get("dimensions_m")
                    if isinstance(dm, dict):
                        dims = (float(dm["width"]), float(dm["depth"]),
                                float(dm["height"]))
                    else:
                        dims = (1.1, 1.3, 0.12)
                    split = str(o.get("split", "?"))
                except Exception:
                    continue
                seen[fid] = {"fid": fid, "ip": ip, "gt8": gt8, "gtc": gtc,
                             "K": K, "dims": dims, "split": split}
        frames[dom] = list(seen.values())
    return frames


def gt_pose(fr):
    """pseudo-GT 6D from GT 2D 8-corner (+centroid) + dims + K. Model-independent."""
    gt9 = [[float(p[0]), float(p[1])] for p in fr["gt8"]]
    gt9.append(list(fr["gtc"]) if fr["gtc"] is not None else None)
    try:
        pose = APNP.solve_pose(gt9, fr["K"], dims=fr["dims"])
    except Exception:
        return None
    if pose is None:
        return None
    return pose["R"], pose["t"]


def pred_pose(pred8, pred_c, K, dims):
    kps = [None if np.isnan(pred8[i, 0]) else
           [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps.append(list(pred_c) if pred_c is not None else None)
    try:
        pose = APNP.solve_pose(kps, K, dims=dims)
    except Exception:
        return None
    if pose is None:
        return None
    return pose["R"], pose["t"]


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def add_error(Rg, tg, Rp, tp):
    """same-index ADD over 8 corners (m). indices aligned by convention."""
    Xg = (Rg @ KP3D8.T).T + tg
    Xp = (Rp @ KP3D8.T).T + tp
    return float(np.mean(np.linalg.norm(Xp - Xg, axis=1)))


def reliability_frame(pred8, gt8, peaks):
    """per-corner belief peak weighted by graded correctness (user value fn).
    reliability_i = peak_i * grade(err_i); undetected kp -> 0 contribution.
    returns (frame_score, list[(correct_bool@TAU_PX, peak, err_px)])."""
    rels, detail = [], []
    for i in range(8):
        w = float(peaks[i])
        if np.isnan(pred8[i, 0]):
            err = np.inf
            rels.append(0.0)              # undetected -> 0 contribution (spec)
        else:
            err = float(np.hypot(pred8[i, 0] - gt8[i, 0], pred8[i, 1] - gt8[i, 1]))
            rels.append(w * grade(err))
        correct = 1 if np.isfinite(err) and err < TAU_PX else 0
        detail.append((correct, w, err))
    return float(np.mean(rels)), detail


def kp9_median_px(pred8, pred_c, gt8, gtc):
    """per-frame 9-keypoint (8 corners + centroid) median px error, same-index.
    undetected kp -> inf so it counts against the median."""
    errs = []
    for i in range(8):
        if np.isnan(pred8[i, 0]):
            errs.append(np.inf)
        else:
            errs.append(float(np.hypot(pred8[i, 0] - gt8[i, 0],
                                       pred8[i, 1] - gt8[i, 1])))
    if gtc is not None:
        if pred_c is None:
            errs.append(np.inf)
        else:
            errs.append(float(np.hypot(pred_c[0] - gtc[0], pred_c[1] - gtc[1])))
    return float(np.median(errs))


def eval_model(label, wp, frames, gtposes, device):
    model = T.E.load_model(wp, device)
    per_dom = {}
    for dom, fs in frames.items():
        adds, trans, rots = [], [], []
        rel_scores = []
        kp9_meds = []              # 9kp median px, DETECTED frames only (gross/catas)
        cpk, wpk = [], []          # peaks of correct / wrong corners (all frames)
        det = 0
        n_gtpnp = 0
        n_3d = 0
        usable = {t: 0 for t in TAU_ADD}   # detected AND ADD<t, over ALL frames
        for fr in fs:
            img = cv2.imread(fr["ip"])
            _, pred8, pred_c, peaks, _ = T.infer_squash(model, img, device)
            n_det = int(np.sum(~np.isnan(pred8[:, 0])))
            detected = n_det >= NMIN
            if detected:
                det += 1
                kp9_meds.append(kp9_median_px(pred8, pred_c, fr["gt8"], fr["gtc"]))
            # (B) reliability — over ALL frames (undetected -> 0 contribution)
            rs, detail = reliability_frame(pred8, fr["gt8"], peaks)
            rel_scores.append(rs)
            for correct, w, err in detail:
                if not np.isfinite(err):
                    continue          # undetected: no peak-at-location judgement
                (cpk if correct else wpk).append(w)
            # (A) 3D — needs detection + both PnP ok
            gp = gtposes.get(fr["fid"])
            if gp is not None:
                n_gtpnp += 1
            if detected and gp is not None:
                pp = pred_pose(pred8, pred_c, fr["K"], fr["dims"])
                if pp is not None:
                    Rg, tg = gp
                    Rp, tp = pp
                    a = add_error(Rg, tg, Rp, tp)
                    adds.append(a)
                    trans.append(float(np.linalg.norm(tp - tg)))
                    rots.append(rot_err_deg(Rg, Rp))
                    n_3d += 1
                    for t in TAU_ADD:
                        if a < t:
                            usable[t] += 1
        n = len(fs)

        def med(a):
            return round(float(np.median(a)), 4) if a else None
        n_gross = sum(1 for m in kp9_meds if m > GROSS_PX)
        n_catas = sum(1 for m in kp9_meds if m > CATAS_PX)
        per_dom[dom] = {
            "n": n, "det": det,
            "det_pct": round(100 * det / n, 1) if n else None,
            "n_3d": n_3d, "n_gtpnp": n_gtpnp,
            "add_med_m": med(adds), "trans_med_m": med(trans), "rot_med_deg": med(rots),
            # ★ PRIMARY: usable-pose rate over ALL frames (miss=not usable)
            "usable_010": round(100 * usable[0.10] / n, 1) if n else None,
            "usable_015": round(100 * usable[0.15] / n, 1) if n else None,
            "n_usable_010": usable[0.10], "n_usable_015": usable[0.15],
            # gross/catas: 9kp-median px, DETECTED frames as denominator
            "gross_pct": round(100 * n_gross / det, 1) if det else None,
            "catas_pct": round(100 * n_catas / det, 1) if det else None,
            "n_det_denom": det,
            "reliability": round(float(np.mean(rel_scores)), 4) if rel_scores else None,
            "correct_peak_mean": round(float(np.mean(cpk)), 4) if cpk else None,
            "wrong_peak_mean": round(float(np.mean(wpk)), 4) if wpk else None,
            "n_correct_corners": len(cpk), "n_wrong_corners": len(wpk),
        }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return per_dom


def sel_label(dom, m):
    return SELF_OF[dom] if m == "SELF" else m


def _cell(v, fmt):
    return "  n/a " if v is None else fmt.format(v)


def build_table(results):
    L = []
    hdr = (f"{'domain':<9}{'model':<12}{'N':>4}{'det%':>6}"
           f"{'ADD(m)':>8}{'n3D':>5}"
           f"{'usbl.10':>8}{'usbl.15':>8}"
           f"{'gross%':>8}{'catas%':>8}{'relia':>8}")
    for dom in DOMAINS:
        L.append("-" * len(hdr))
        for m in TABLE_MODELS:
            lab = sel_label(dom, m)
            if lab is None:
                L.append(f"{dom:<9}{'self(n/a)':<12}"
                         f"{'(no cad-specific self-train model)'}")
                continue
            r = results[lab][dom]
            tag = m if m != "SELF" else f"self={lab}"
            L.append(
                f"{dom if m==TABLE_MODELS[0] else '':<9}{tag:<12}"
                f"{r['n']:>4}{_cell(r['det_pct'],'{:>5.1f}'):>6}"
                f"{_cell(r['add_med_m'],'{:>7.3f}'):>8}"
                f"{r['n_3d']:>5}"
                f"{_cell(r['usable_010'],'{:>7.1f}'):>8}"
                f"{_cell(r['usable_015'],'{:>7.1f}'):>8}"
                f"{_cell(r['gross_pct'],'{:>7.1f}'):>8}"
                f"{_cell(r['catas_pct'],'{:>7.1f}'):>8}"
                f"{_cell(r['reliability'],'{:>7.3f}'):>8}")
    return hdr, L


def make_figures(results):
    doms = list(DOMAINS.keys())
    groups = ["R0", "ctrl", "SELF", "combined"]
    glabels = ["R0", "ctrl(syn-only)", "self-domain", "combined"]
    colors = ["#888888", "#4C9BE0", "#E0574C", "#5CB85C"]

    def val(dom, g, key):
        lab = sel_label(dom, g)
        if lab is None:
            return None
        return results[lab][dom][key]

    for key, ylab, fname, title in [
        ("usable_015", "usable-pose rate %% (ADD<0.15m)  [higher=better]",
         "eval3d_usable_bars.png",
         "PRIMARY: usable-pose rate = %det AND ADD<0.15m over ALL frames"),
        ("add_med_m", "ADD (m)  [lower=better]", "eval3d_add_bars.png",
         "3D ADD via pseudo-GT PnP (survivorship: detected+PnP only)"),
        ("reliability", "reliability score  [higher=better]",
         "eval3d_reliability_bars.png",
         "Confidence reliability = mean_i peak_i * grade(err_i)")]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(doms))
        w = 0.2
        for gi, (g, gl, c) in enumerate(zip(groups, glabels, colors)):
            vals = [val(d, g, key) for d in doms]
            xs = x + (gi - 1.5) * w
            plotv = [0 if v is None else v for v in vals]
            bars = ax.bar(xs, plotv, w, label=gl, color=c,
                          edgecolor="k" if g == "SELF" else "none",
                          linewidth=1.5 if g == "SELF" else 0)
            for xi, v, b in zip(xs, vals, bars):
                if v is None:
                    ax.text(xi, 0.001, "n/a", ha="center", va="bottom",
                            fontsize=7, rotation=90)
                else:
                    ax.text(xi, b.get_height(), f"{v:.3f}", ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}\n(N={results['R0'][d]['n']})" for d in doms])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(fontsize=8, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout()
        fp = os.path.join(OUT_DIR, fname)
        fig.savefig(fp, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {fp}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    APNP.PALLET_DIMS = (1.1, 1.3, 0.11)   # default; per-frame dims passed explicitly
    frames = load_eval()
    print("[eval GT sizes]", {d: len(v) for d, v in frames.items()})
    print("[split composition]")
    for d, fs in frames.items():
        comp = {}
        for fr in fs:
            comp[fr["split"]] = comp.get(fr["split"], 0) + 1
        print(f"   {d}: {comp}")

    # pseudo-GT poses (model-independent) — compute once
    gtposes = {}
    ngtfail = {d: 0 for d in DOMAINS}
    for d, fs in frames.items():
        for fr in fs:
            gp = gt_pose(fr)
            if gp is None:
                ngtfail[d] += 1
            else:
                gtposes[fr["fid"]] = gp
    print("[pseudo-GT PnP fail per domain]", ngtfail)

    results = {}
    for label, wp in MODELS.items():
        wpp = wp if os.path.isabs(wp) else os.path.join(ROOT, wp)
        if not os.path.isfile(wpp):
            print("[skip missing]", label, wp)
            continue
        print(f"[eval] {label} ...", flush=True)
        results[label] = eval_model(label, wpp, frames, gtposes, device)

    # save json
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "eval3d_reliability.json"), "w") as f:
        json.dump({"tau_px": TAU_PX, "tau_add": TAU_ADD,
                   "gross_px": GROSS_PX, "catas_px": CATAS_PX,
                   "self_of": SELF_OF, "models": MODELS,
                   "gt_pnp_fail": ngtfail, "results": results}, f, indent=2)

    hdr, rows = build_table(results)
    L = ["# ralph self-training — usable-pose re-eval (value-aligned)", "",
         "- ★PRIMARY usbl.10/usbl.15 = usable-pose rate = %(det AND ADD<0.10/0.15m) over",
         "      ALL eval frames (miss/PnP-fail = not usable). rewards miss->roughly-right (recall).",
         "- ADD(m) = same-index 8-corner 3D dist via pseudo-GT PnP; MEDIAN over det+PnP-ok",
         "      frames (n3D). ★survivorship: coverage separate; pseudo-GT=PnP-from-2D not metrology.",
         f"- gross% / catas% = %(9kp-median px > {GROSS_PX:.0f} / {CATAS_PX:.0f}), DETECTED frames as denom.",
         "- relia = mean_i peak_i*grade(err_i); grade: <=10px +1, 10..40 +1->0, 40..80 0->-1,",
         "      >80px -1 (lenient on small err, punishes only confident-huge-wrong). undet kp->0.",
         "- squash-parity infer (all s2 lineage); same held-out eval GT as ralph_eval_all.",
         "- self-domain = per-domain self-train (outside=h6,night=h3,noapril=h7); cad has none.",
         "  ctrl=synthetic-only (PL=0) isolates PL-specific effect. combined=h4.", "",
         "```", hdr] + rows + ["```", ""]
    with open(os.path.join(OUT_DIR, "eval3d_reliability.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join([hdr] + rows))
    make_figures(results)
    print(f"\n[save] {OUT_DIR}/eval3d_reliability.json | .md | *_bars.png")


if __name__ == "__main__":
    main()
