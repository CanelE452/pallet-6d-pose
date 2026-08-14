"""paper_s2_real_eval.py — PAPER_S2 Phase 5 FINAL real judgement.

Q: Does DiffPnP3D (paper_s2 Stage B, λ0.005) improve REAL rear / low-angle pose
   vs the previous best (paper_s1 mask-aux)? Or is the synthetic-val win synth-only
   (STAGE16 precedent)?

★ EVAL-PARITY (memory gotcha — non-negotiable):
  - paper_s2 (Stage A/B) trained with ANISOTROPIC squash (640x480->400x400). Eval with
    squash preprocess + belief(50)->orig x(W/50, H/50). The shared aspect-preserving
    preprocess is OFF-distribution for these models -> would be invalid.
  - paper_s1 / paper_base_v2 trained no-pad aspect-preserving. Eval with E.eval_frame
    PAD=0 (A_nopad = official train/infer parity, same as eval_paper_s1_paired.py).
  Only the belief-decode PREPROCESS differs per model. The downstream metric+PnP path
  is IDENTICAL for all models (order-free Hungarian corner + solve_pose order-free W/D
  + honest full-8 reproj), so the comparison is fair.

Real sets (shared, same frames per model):
  - filterval N=123 (outside44 + night43 + manual36) = LOW-ANGLE regime (elev -7.7~10.4,
    46 frames <3deg, 67 in 3-8deg) -> the PRIMARY rear/low-angle transfer test.
  - handannot17 N=17 = HIGH-ANGLE (25+deg 13/17). memory: NOT representative of real.
    Qualitative only, no rear claim.
  final-test sessions SEALED out by stage25 split-lock.

dims: solve_pose uses PALLET_DIMS=(1.1,1.3,0.11), order-free W/D swap. Real measured
  = 1.1x1.3x0.12m (commit 0baa6df); H 0.11 vs 0.12 negligible. Same dims for ALL models.
elevation: GT pose_transform (real-data convention -> meaningful, unlike synthetic).

Usage: python scripts/stage0/paper_s2/paper_s2_real_eval.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

import cv2  # noqa: E402
import torch  # noqa: E402
import stage25_paperbase_eval as S  # noqa: E402  (SETS, elev_of, summarize, per_corner_dists)
from stage25_paperbase_eval import E  # noqa: E402  (load_model, eval_frame, hungarian)
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from eval_pvnet_heads import split_metrics  # noqa: E402
import annotate_pnp as APNP  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/paper_s2_scratch_diffpnp")
THRESH, PAD_ASPECT, N_DET_MIN = 0.3, 0, 6
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# (name, weights, preprocess-mode)
MODELS = [
    ("paper_s2_stageB", "weights/paper_s2_stageB/net_epoch_0057.pth", "squash"),
    ("paper_s1",        "weights/paper_s1_maskaux/net_epoch_0065.pth", "aspect"),
    ("paper_base_v2",   "weights/paper_base_v2/final_net_epoch_0060.pth", "aspect"),
    ("paper_s2_stageA", "weights/paper_s2_stageA/net_epoch_0042.pth", "squash"),
]
BASELINE = "paper_s1"   # Delta reference
CHALLENGER = "paper_s2_stageB"

# elevation bins tuned to real low-angle regime (rear-collapse <~8deg)
ELEV_BINS = [(-90, 3), (3, 8), (8, 15), (15, 90)]
ELEV_LBL = ["<3", "3-8", "8-15", "15+"]
METRICS = ["det_pct", "front_med", "rear_med", "corner_med", "honest8_med",
           "good_pct", "gross_pct", "pnp_pct"]
BETTER = {"det_pct": +1, "front_med": -1, "rear_med": -1, "corner_med": -1,
          "honest8_med": -1, "good_pct": +1, "gross_pct": -1, "pnp_pct": +1}


def preprocess_squash(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (400, 400), interpolation=cv2.INTER_LINEAR)
    t = (r.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)


def eval_frame_squash(model, jp, ip, device):
    """s2 squash-parity decode -> SAME row schema as E.eval_frame (order-free,
    solve_pose PALLET_DIMS honest8). Belief(50)->orig via (W/50, H/50)."""
    img = cv2.imread(ip)
    if img is None:
        return None
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    gtc = np.array(o["projected_cuboid_centroid"], float)
    H, W = img.shape[:2]
    K = E.K_from_json(d)
    inb = ((gt8[:, 0] >= 0) & (gt8[:, 0] < W) & (gt8[:, 1] >= 0) & (gt8[:, 1] < H))
    v_geom = int(inb.sum())
    sx, sy = W / 50.0, H / 50.0

    with torch.no_grad():
        beliefs, _ = model(preprocess_squash(img).to(device))
    belief = beliefs[-1][0].cpu().numpy()
    kps_bel = extract_keypoints_from_belief(belief, THRESH)

    pred8 = np.full((8, 2), np.nan)
    for i, k in enumerate(kps_bel[:8]):
        if k[0] < 0:
            continue
        pred8[i] = [k[0] * sx, k[1] * sy]
    pred_c = None
    if kps_bel[8][0] >= 0:
        pred_c = [kps_bel[8][0] * sx, kps_bel[8][1] * sy]

    m = split_metrics(pred8, gt8)
    dists, _ = E.hungarian(pred8, gt8)
    worst2 = (float(np.mean(np.sort(dists)[-2:]))
              if dists is not None and len(dists) >= 2 else np.inf)

    kps9 = [None if np.isnan(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps9.append(pred_c)
    pnp_reproj_click, pnp_honest8, pnp_ok = np.inf, np.inf, 0
    proj_all = None
    if sum(1 for k in kps9 if k is not None) >= N_DET_MIN:
        try:
            pose = APNP.solve_pose(kps9, K, dims=APNP.PALLET_DIMS, img_shape=img.shape)
            if pose is not None:
                pnp_ok = 1
                pnp_reproj_click = float(pose["reproj_error_px"])
                proj_all = np.array(pose["projected_all"], float)[:8]
                pa = proj_all.copy()
                bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
                pa[bad] = np.nan
                hd, _ = E.hungarian(pa, gt8)
                if hd is not None:
                    pnp_honest8 = float(np.mean(hd))
        except Exception:
            pass

    return {"fid": os.path.splitext(os.path.basename(jp))[0],
            "v_geom": v_geom, "n_det": m["n_det"],
            "det": 1 if m["n_det"] >= N_DET_MIN else 0,
            "corner": m["overall"], "front": m["front"], "back": m["back"],
            "worst2": worst2, "pnp_ok": pnp_ok,
            "pnp_reproj_click": pnp_reproj_click, "pnp_honest8": pnp_honest8,
            "gt8": gt8.tolist(), "gtc": gtc.tolist(),
            "pred8": pred8.tolist(), "pred_c": pred_c,
            "proj_all": (proj_all.tolist() if proj_all is not None else None),
            "ip": ip}


def elev_bin_breakdown(rows, elev_map):
    eb = {}
    for lbl, (lo, hi) in zip(ELEV_LBL, ELEV_BINS):
        sel = [r for r in rows if (elev_map.get(r["fid"]) is not None
                                   and lo <= elev_map[r["fid"]] < hi)]
        if sel:
            eb[lbl] = S.summarize(sel)
    return eb


def _fmt(v):
    return "  -  " if v is None else f"{v}"


def paired_table(a_s, b_s, aname, bname, title):
    L = [f"### {title}",
         f"{'metric':<11}{aname:>15}{bname:>15}{'d(B-A)':>12}"]
    L.append("-" * len(L[-1]))
    for m in METRICS:
        a, b = a_s.get(m), b_s.get(m)
        d = ""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = round(b - a, 2)
            arrow = ""
            if delta != 0:
                improved = (delta > 0) == (BETTER[m] > 0)
                arrow = " +good" if improved else " -bad"
            d = f"{delta:+g}{arrow}"
        L.append(f"{m:<11}{_fmt(a):>15}{_fmt(b):>15}{d:>12}")
    return "\n".join(L)


def draw(row, tag, elev):
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
        col = (0, 0, 255) if i < 4 else (255, 128, 0)  # front red / rear blue
        if not np.isnan(pr8[i, 0]):
            cv2.circle(img, (int(pr8[i, 0]), int(pr8[i, 1])), 4, col, -1)
            cv2.line(img, (int(pr8[i, 0]), int(pr8[i, 1])),
                     tuple(gt8[i].astype(int)), col, 1)
    rear = row["back"] if np.isfinite(row["back"]) else -1
    h8 = row["pnp_honest8"] if np.isfinite(row["pnp_honest8"]) else -1
    txt = f"{tag} V={row['v_geom']} el={elev:.0f} rear={rear:.0f} h8={h8:.0f}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fl = {s: fn() for s, fn in S.SETS.items()}
    elev = {s: {fid: S.elev_of(jp) for _, fid, jp, _ in fl[s]} for s in fl}
    for s in fl:
        by = {}
        for dom, *_ in fl[s]:
            by[dom] = by.get(dom, 0) + 1
        print(f"[set] {s}: N={len(fl[s])} {by}")

    results = {m: {s: [] for s in S.SETS} for m, _, _ in MODELS}
    for mname, wrel, mode in MODELS:
        wp = os.path.join(ROOT, wrel)
        mdl = E.load_model(wp, device)
        for s in S.SETS:
            for dom, fid, jp, ip in fl[s]:
                if mode == "squash":
                    r = eval_frame_squash(mdl, jp, ip, device)
                else:
                    r = E.eval_frame(mdl, jp, ip, device, THRESH, PAD_ASPECT)
                if r is not None:
                    r["dom"] = dom
                    results[mname][s].append(r)
        del mdl
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[done] {mname} ({mode})")

    # ---- aggregate ----
    out = {"parity": {m: mode for m, _, mode in MODELS},
           "weights": {m: w for m, w, _ in MODELS},
           "pad_aspect": PAD_ASPECT, "thresh": THRESH,
           "dims": APNP.PALLET_DIMS, "sets": {}}
    for s in S.SETS:
        out["sets"][s] = {}
        for mname, _, _ in MODELS:
            rows = results[mname][s]
            out["sets"][s][mname] = {
                "overall": S.summarize(rows),
                "V8": S.summarize([r for r in rows if r["v_geom"] == 8]),
                "Vlt8": S.summarize([r for r in rows if r["v_geom"] < 8]),
                "by_domain": {dm: S.summarize([r for r in rows if r["dom"] == dm])
                              for dm in sorted(set(r["dom"] for r in rows))},
                "elev_bins": elev_bin_breakdown(rows, elev[s]),
            }
    with open(os.path.join(OUT_DIR, "real_eval.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)
    print(f"[save] {OUT_DIR}/real_eval.json")

    write_summary(out, elev, results)
    make_overlays(results, elev)


def write_summary(out, elev, results):
    L = ["# PAPER_S2 Phase 5 — REAL eval (final transfer judgement)", "",
         "Q: DiffPnP3D (Stage B λ0.005) rear/low-angle improvement TRANSFER to real?",
         "",
         "## Protocol / parity",
         "```",
         "eval-parity (each model = its own training preprocess; downstream metric IDENTICAL):",
         f"  paper_s2_stageA/B : squash 640x480->400x400 aniso, belief(50)->orig x(W/50,H/50)",
         f"  paper_s1 / base   : aspect-preserving no-pad (PAD={out['pad_aspect']}, official A_nopad)",
         "metric: order-free Hungarian corner + split_metrics front/back + solve_pose",
         f"  (order-free W/D, dims={out['dims']} ~= measured 1.1x1.3x0.12) + honest full-8 reproj.",
         "good%<10px gross%>20px per-corner. det%=n_det>=6. elevation=GT pose_transform.",
         "```",
         "",
         "## Sets",
         "```",
         "filterval  N=123 (outside44+night43+manual36) = LOW-ANGLE (elev -7.7~10.4;",
         "                  46 frames <3deg, 67 in 3-8deg) -> PRIMARY rear/low-angle test.",
         "handannot17 N=17 = HIGH-ANGLE (25+deg 13/17). memory: NOT real-representative;",
         "                  qualitative only, NO rear claim.",
         "final-test sessions SEALED (split-lock).",
         "```", ""]

    order = [m for m, _, _ in MODELS]
    for s in S.SETS:
        L.append(f"## {s}")
        # elevation dist
        els = [elev[s][fid] for fid in elev[s] if elev[s][fid] is not None]
        edist = []
        for lbl, (lo, hi) in zip(ELEV_LBL, ELEV_BINS):
            edist.append(f"{lbl}:{sum(1 for e in els if lo <= e < hi)}")
        L.append("elev dist(deg): " + ", ".join(edist))
        L.append("")
        L.append("### overall (all models)")
        L.append("```")
        L.append(_hdr())
        L.append("-" * len(_hdr()))
        for m in order:
            L.append(_row(m, out["sets"][s][m]["overall"]))
        L.append("```")
        L.append("")
        # V split
        L.append("### V=8 (full-view) vs V<8 (truncated)")
        L.append("```")
        L.append(_hdr())
        L.append("-" * len(_hdr()))
        for m in order:
            L.append(_row(m + " V8", out["sets"][s][m]["V8"]))
            L.append(_row(m + " V<8", out["sets"][s][m]["Vlt8"]))
        L.append("```")
        L.append("")
        # paired Stage B vs paper_s1
        L.append(f"### PAIRED: {CHALLENGER} vs {BASELINE} (Delta = challenger - baseline)")
        L.append("```")
        L.append(paired_table(out["sets"][s][BASELINE]["overall"],
                              out["sets"][s][CHALLENGER]["overall"],
                              BASELINE, CHALLENGER, f"{s} overall"))
        L.append("")
        L.append(paired_table(out["sets"][s][BASELINE]["V8"],
                              out["sets"][s][CHALLENGER]["V8"],
                              BASELINE, CHALLENGER, f"{s} V=8"))
        L.append("```")
        L.append("")
        # elevation bins paired (filterval primary)
        if s == "filterval":
            L.append("### elevation bins — PAIRED (challenger vs baseline)")
            L.append("```")
            for lbl in ELEV_LBL:
                ba = out["sets"][s][BASELINE]["elev_bins"].get(lbl)
                ch = out["sets"][s][CHALLENGER]["elev_bins"].get(lbl)
                if ba and ch:
                    L.append(paired_table(ba, ch, BASELINE, CHALLENGER,
                                          f"elev {lbl} deg (n={ba['n']})"))
                    L.append("")
            L.append("```")
            L.append("")
            # all-model elevation-bin rear/corner table
            L.append("### elevation bins — rear_med (all models)")
            L.append("```")
            L.append(f"{'elev':<8}" + "".join(f"{m:>18}" for m in order))
            for lbl in ELEV_LBL:
                cells = []
                for m in order:
                    eb = out["sets"][s][m]["elev_bins"].get(lbl, {})
                    cells.append(f"{str(eb.get('rear_med')):>18}")
                L.append(f"{lbl:<8}" + "".join(cells))
            L.append("```")
            L.append("")

    # ---- VERDICT ----
    L.append("## VERDICT — transfer to real (filterval PRIMARY)")
    L.append("```")
    ba = out["sets"]["filterval"][BASELINE]["overall"]
    ch = out["sets"]["filterval"][CHALLENGER]["overall"]

    def d(k):
        a, b = ba.get(k), ch.get(k)
        return None if a is None or b is None else round(b - a, 2)
    rear_d = d("rear_med")
    h8_d = d("honest8_med")
    cor_d = d("corner_med")
    front_d = d("front_med")
    det_d = d("det_pct")
    gross_d = d("gross_pct")
    L.append(f"  rear_med   : {BASELINE}={ba.get('rear_med')} -> {CHALLENGER}={ch.get('rear_med')}  d={rear_d}")
    L.append(f"  honest8    : {BASELINE}={ba.get('honest8_med')} -> {CHALLENGER}={ch.get('honest8_med')}  d={h8_d}")
    L.append(f"  corner_med : {BASELINE}={ba.get('corner_med')} -> {CHALLENGER}={ch.get('corner_med')}  d={cor_d}")
    L.append(f"  front_med  : {BASELINE}={ba.get('front_med')} -> {CHALLENGER}={ch.get('front_med')}  d={front_d}")
    L.append(f"  det%       : {BASELINE}={ba.get('det_pct')} -> {CHALLENGER}={ch.get('det_pct')}  d={det_d}")
    L.append(f"  gross%     : {BASELINE}={ba.get('gross_pct')} -> {CHALLENGER}={ch.get('gross_pct')}  d={gross_d}")
    # low-angle (3-8 + <3) combined rear delta = the injection target
    L.append("")
    for lbl in ["<3", "3-8"]:
        b_eb = out["sets"]["filterval"][BASELINE]["elev_bins"].get(lbl, {})
        c_eb = out["sets"]["filterval"][CHALLENGER]["elev_bins"].get(lbl, {})
        if b_eb and c_eb:
            rb, rc = b_eb.get("rear_med"), c_eb.get("rear_med")
            rd = None if rb is None or rc is None else round(rc - rb, 2)
            L.append(f"  low-angle rear elev {lbl} (n={b_eb.get('n')}): "
                     f"{rb} -> {rc}  d={rd}")
    L.append("```")
    L.append("")
    L.append("★ Small-sample + convention caveats: filterval N=123 (domain-mixed, low-angle),")
    L.append("  handannot17 N=17 high-angle (not real-representative). ckpt = synthetic-val best")
    L.append("  (NOT real-selected). Judgement only. rear improvement lower=better; a negative or")
    L.append("  ~zero d(rear) on low-angle bins = synth-only (STAGE16 precedent), not transfer.")
    with open(os.path.join(OUT_DIR, "real_eval_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[save] {OUT_DIR}/real_eval_summary.md")


def _hdr():
    return (f"{'model':<18}{'n':>4}{'det%':>6}{'front':>7}{'rear':>7}"
            f"{'corner':>8}{'worst2':>8}{'pnp%':>6}{'honest8':>9}"
            f"{'good%':>7}{'gross%':>7}")


def _row(label, s):
    return (f"{label:<18}{str(s.get('n')):>4}{str(s.get('det_pct')):>6}"
            f"{str(s.get('front_med')):>7}{str(s.get('rear_med')):>7}"
            f"{str(s.get('corner_med')):>8}{str(s.get('worst2_med')):>8}"
            f"{str(s.get('pnp_pct')):>6}{str(s.get('honest8_med')):>9}"
            f"{str(s.get('good_pct')):>7}{str(s.get('gross_pct')):>7}")


def make_overlays(results, elev):
    """Stage B vs paper_s1 side-by-side on low-angle filterval frames (rear focus)."""
    ovdir = os.path.join(OUT_DIR, "real_overlays")
    os.makedirs(ovdir, exist_ok=True)
    s = "filterval"
    by_s1 = {r["fid"]: r for r in results[BASELINE][s]}
    chr_rows = results[CHALLENGER][s]
    # rank by (challenger rear improvement vs baseline) to show both improve & regress
    pairs = []
    for rc in chr_rows:
        rb = by_s1.get(rc["fid"])
        el = elev[s].get(rc["fid"])
        if rb is None or el is None:
            continue
        if not (np.isfinite(rc["back"]) and np.isfinite(rb["back"])):
            continue
        pairs.append((rc["back"] - rb["back"], el, rc, rb))
    pairs.sort(key=lambda x: x[0])  # most-improved (neg) first
    picks = pairs[:5] + pairs[-5:]  # 5 improved + 5 regressed
    n = 0
    for delta, el, rc, rb in picks:
        a = draw(rb, "s1", el)
        b = draw(rc, "stageB", el)
        if a is None or b is None:
            continue
        h = max(a.shape[0], b.shape[0])
        canvas = np.zeros((h, a.shape[1] + b.shape[1] + 4, 3), np.uint8)
        canvas[:a.shape[0], :a.shape[1]] = a
        canvas[:b.shape[0], a.shape[1] + 4:] = b
        tag = "impr" if delta < 0 else "regr"
        cv2.imwrite(os.path.join(
            ovdir, f"{tag}_d{delta:+.0f}_el{el:.0f}_{rc['fid']}.jpg"), canvas)
        n += 1
    print(f"[overlays] {n} low-angle s1|stageB pairs -> {ovdir}")


if __name__ == "__main__":
    main()
