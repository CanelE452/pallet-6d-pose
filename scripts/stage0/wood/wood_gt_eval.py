"""wood_gt_eval.py — PAPER_S2 Stage B best on wood pallet WITH manual GT (45 frames).

이전 wood_infer_filter.py 는 GT 없이 self-consistency 만 봤다. 이제 손어노 GT 45장이
생겼으므로 (1) Stage B 의 wood GT 대비 진짜 정확도, (2) 필터별 pass/fail,
(3) ★필터 선택성 = 필터 accept 가 GT-good(정확) 프레임과 일치하는가(P/R) 를 측정한다.

재사용:
  - wood_infer_filter (wf): decode_squash / flip_infer_squash / tta_stab_squash /
    solve_wood / WOOD_DIMS / model / squash-parity 전처리.
  - s1_cad_9filters (s1): 필터 스코어러(f1~f9), bbox/size/excursion, EDGES, TAU.
  - eval_pvnet_heads.split_metrics: order-free Hungarian corner_med(front/back).
  - annotate_wood.K_for_resolution: RealSense 1080p 16:9 선형유도 K.

dims: (0.8, 0.59, 0.14) m (사용자 지정). solve_pose W/D swap 자동 (APNP.PALLET_DIMS override).

★ caveat: N=45 소표본, wood=UNSEEN(치수·텍스처 학습에 없음), 2세션, 필터 임계 heuristic.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
import importlib.util  # noqa: E402


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


wf = _load("wf", os.path.join(ROOT, "scripts", "stage0", "wood", "wood_infer_filter.py"))
s1 = wf.s1
import torch  # noqa: E402
from eval_pvnet_heads import split_metrics  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
import annotate_pnp as APNP  # noqa: E402
from annotate_wood import K_for_resolution  # noqa: E402

APNP.PALLET_DIMS = wf.WOOD_DIMS  # ensure wood dims for solve_pose auto-swap

GOOD_PX = 10.0
GROSS_PX = 20.0
N_DET_MIN = wf.N_DET_MIN
WOOD_DIMS = wf.WOOD_DIMS

GT_DIRS = {
    "pallet_20260618_183705":
        os.path.join(ROOT, "challenge", "data", "wood_pallet_20260618_183705_manual_gt"),
    "pallet_20260618_184309":
        os.path.join(ROOT, "challenge", "data", "wood_pallet_20260618_184309_manual_gt"),
}
FRAME_ROOT = os.path.join(ROOT, "data", "pallet", "raw_data", "wood", "selected")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "paper_s2_scratch_diffpnp")
OV_DIR = os.path.join(OUT_DIR, "wood_gt_overlays")
MD_PATH = os.path.join(OUT_DIR, "wood_gt_eval.md")

# filters: f1~f9 + reproj self-consistency gate
FILTERS = ["f1_peak", "f2_peak_ratio", "f3_flip", "f4_tta_stab", "f5_rear_conf",
           "f6_frsep", "f7_posdepth", "f8_size_env", "f9_bbox_iou", "reproj_selfcons"]
TAU = dict(s1.TAU)
REPROJ_TAU = wf.REPROJ_TAU  # 10.0


# ── order-free Hungarian median (pred/proj possibly with NaN) vs full GT8 ──
def hungarian_med(A, B):
    """A (8,2) may contain NaN; B (8,2) full GT. median matched dist, None if <6."""
    valid = ~np.isnan(A[:, 0])
    if valid.sum() < N_DET_MIN:
        return None
    P = A[valid]
    cost = np.linalg.norm(P[:, None, :] - B[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return float(np.median(cost[ri, ci]))


def collect():
    frames = []
    for ses, gtdir in GT_DIRS.items():
        for jp in sorted(glob.glob(os.path.join(gtdir, "*.json"))):
            fid = os.path.splitext(os.path.basename(jp))[0]
            ip = os.path.join(FRAME_ROOT, ses, fid + ".jpg")
            if os.path.exists(ip):
                frames.append((ses, fid, jp, ip))
    return frames


def load_gt(jp):
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    gtc = np.array(o["projected_cuboid_centroid"], float)
    return gt8, gtc


def process(model, ses, fid, jp, ip, K, device):
    img = cv2.imread(ip)
    H, W = img.shape[:2]
    img_diag = float(np.hypot(W, H))
    gt8, gtc = load_gt(jp)

    pred8, pred_c, peaks, ratios = wf.decode_squash(model, img, device)
    n_det = int((~np.isnan(pred8[:, 0])).sum())
    det_ok = n_det >= N_DET_MIN

    # ── GT accuracy ──────────────────────────────────────────────────────
    m = split_metrics(pred8, gt8)  # order-free Hungarian
    corner_med = m["overall"] if np.isfinite(m["overall"]) else None
    front = m["front"] if np.isfinite(m["front"]) else None
    rear = m["back"] if np.isfinite(m["back"]) else None

    # ── filter scores (self-sup f1~f7 + reproj) + GT-aware f8/f9 ─────────
    det8 = [i for i in range(8) if not np.isnan(pred8[i, 0])]
    rear_idx = [i for i in (4, 5, 6, 7) if not np.isnan(pred8[i, 0])]
    f1 = float(min(peaks[i] for i in det8)) if len(det8) >= N_DET_MIN else None
    f2 = float(min(ratios[i] for i in det8)) if len(det8) >= N_DET_MIN else None
    f5 = float(min(peaks[i] for i in rear_idx)) if len(rear_idx) == 4 else None
    f6 = s1.frsep_frac(pred8)

    f3 = f4 = None
    pose = None
    posdepth = False
    reproj = None
    honest8 = None
    proj_all = None
    if det_ok:
        kpB = wf.flip_infer_squash(model, img, device)
        f3 = s1.flip_score(pred8, pred_c, kpB)
        f4 = wf.tta_stab_squash(model, img, device)
        pose, posdepth, reproj = wf.solve_wood(pred8, pred_c, K, img.shape)
        if pose is not None:
            pa = np.array(pose["projected_all"], float)[:8]
            bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
            pa[bad] = np.nan
            proj_all = pa
            honest8 = hungarian_med(pa, gt8)  # PnP reproj vs GT (order-free)

    # f8 size_env (per-frame pred vs GT-envelope built later), f9 bbox_iou vs GT bbox
    pred_sr, pred_asp = s1.size_aspect(pred8, img_diag)
    gt_sr, gt_asp = s1.size_aspect(gt8, img_diag)
    f9 = s1.bbox_iou(s1.bbox_of(pred8), s1.bbox_of(gt8)) if det_ok else None

    scores = {"f1_peak": f1, "f2_peak_ratio": f2, "f3_flip": f3, "f4_tta_stab": f4,
              "f5_rear_conf": f5, "f6_frsep": f6, "f7_posdepth": posdepth,
              "f9_bbox_iou": f9, "reproj_selfcons": reproj}
    return {
        "ses": ses, "fid": fid, "ip": ip, "n_det": n_det, "det_ok": det_ok,
        "gt8": gt8, "gtc": gtc, "pred8": pred8, "pred_c": pred_c,
        "proj_all": proj_all,
        "corner_med": corner_med, "front": front, "rear": rear, "honest8": honest8,
        "gt_good": bool(corner_med is not None and corner_med < GOOD_PX),
        "gt_gross": bool(corner_med is None or corner_med > GROSS_PX),
        "scores": scores, "pred_sr": pred_sr, "pred_asp": pred_asp,
        "gt_sr": gt_sr, "gt_asp": gt_asp,
    }


def build_envelope(recs):
    """f8 GT envelope: [p2.5,p97.5] of GT size-ratio & aspect over 45 GT frames."""
    srs = np.array([r["gt_sr"] for r in recs if r["gt_sr"] is not None])
    asps = np.array([r["gt_asp"] for r in recs if r["gt_asp"] is not None])
    return {
        "size_lo": float(np.percentile(srs, 2.5)), "size_hi": float(np.percentile(srs, 97.5)),
        "asp_lo": float(np.percentile(asps, 2.5)), "asp_hi": float(np.percentile(asps, 97.5)),
    }


def passes(fname, rec, env):
    if not rec["det_ok"]:
        return False
    s = rec["scores"]
    if fname == "f7_posdepth":
        return bool(s["f7_posdepth"])
    if fname == "f8_size_env":
        if rec["pred_sr"] is None or rec["pred_asp"] is None:
            return False
        es = s1.excursion(rec["pred_sr"], env["size_lo"], env["size_hi"])
        ea = s1.excursion(rec["pred_asp"], env["asp_lo"], env["asp_hi"])
        return max(es, ea) == 0.0
    v = s[fname]
    if v is None:
        return False
    if fname in ("f1_peak", "f2_peak_ratio", "f5_rear_conf", "f6_frsep", "f9_bbox_iou"):
        return v >= TAU[fname]
    if fname in ("f3_flip", "f4_tta_stab"):
        return v <= TAU[fname]
    if fname == "reproj_selfcons":
        return v <= REPROJ_TAU
    return False


# ── selectivity: confusion of filter-accept vs GT-good ───────────────────
def confusion(recs, accept_fn):
    tp = fp = fn = tn = 0
    for r in recs:
        acc = accept_fn(r)
        good = r["gt_good"]
        if acc and good:
            tp += 1
        elif acc and not good:
            fp += 1
        elif not acc and good:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=prec, recall=rec, n_acc=tp + fp)


# ── overlays: GT green + pred blue + PnP-reproj red ──────────────────────
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
EDGES = s1.EDGES


def draw_edges(img, pts, col, th):
    ok = ~np.isnan(pts[:, 0])
    for a, b in EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, (int(pts[a, 0]), int(pts[a, 1])),
                     (int(pts[b, 0]), int(pts[b, 1])), col, th, cv2.LINE_AA)


def save_overlay(rec, tag, path, accept, env):
    img = cv2.imread(rec["ip"])
    draw_edges(img, rec["gt8"], GREEN, 2)               # GT green
    if rec["proj_all"] is not None:
        draw_edges(img, rec["proj_all"], RED, 2)        # PnP reproj red
    p8 = rec["pred8"]
    for i in range(8):
        if not np.isnan(p8[i, 0]):
            p = (int(p8[i, 0]), int(p8[i, 1]))
            cv2.circle(img, p, 5, BLUE, -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, cv2.LINE_AA)
    if rec["pred_c"] is not None:
        cv2.drawMarker(img, (int(rec["pred_c"][0]), int(rec["pred_c"][1])),
                       BLUE, cv2.MARKER_CROSS, 12, 2)
    cm = f"{rec['corner_med']:.1f}" if rec["corner_med"] is not None else "n/a"
    h8 = f"{rec['honest8']:.1f}" if rec["honest8"] is not None else "n/a"
    rp = rec["scores"]["reproj_selfcons"]
    rps = f"{rp:.1f}" if rp is not None else "n/a"
    npass = sum(1 for f in FILTERS if passes(f, rec, env))
    h1 = (f"{tag}  det={rec['n_det']}/8  corner_med={cm}px  honest8={h8}px  "
          f"reproj={rps}px  #pass={npass}/10  accept={'Y' if accept else 'n'}")
    h2 = "wood GT (0.8x0.59x0.14) | GT=green pred=blue PnP-reproj=red | GT-good=corner_med<10px"
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, h1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, h2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def main():
    os.makedirs(OV_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(OV_DIR, "*.jpg")):
        os.remove(old)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = s1.E.load_model(wf.WEIGHTS, device)

    frames = collect()
    img0 = cv2.imread(frames[0][3])
    H, W = img0.shape[:2]
    K, ksrc = K_for_resolution(W, H)
    print(f"[wood-GT] N={len(frames)}  res={W}x{H}  K={ksrc}  dims={WOOD_DIMS}")

    recs = []
    for i, (ses, fid, jp, ip) in enumerate(frames):
        recs.append(process(model, ses, fid, jp, ip, K, device))
        if (i + 1) % 15 == 0 or i == 0:
            r = recs[-1]
            print(f"  [{i+1}/{len(frames)}] {fid} det={r['n_det']} "
                  f"corner_med={r['corner_med']}")

    env = build_envelope(recs)

    # ── accuracy aggregate ───────────────────────────────────────────────
    N = len(recs)
    dets = [r for r in recs if r["det_ok"]]
    goods = [r for r in dets if r["gt_good"]]
    grosses = [r for r in recs if r["gt_gross"]]

    # ── selectivity per filter + combos ──────────────────────────────────
    def single(f):
        return lambda r: passes(f, r, env)

    STRONG = ["f1_peak", "f4_tta_stab", "f5_rear_conf", "reproj_selfcons"]
    ALLR = FILTERS
    combos = {
        "COMBO strong(f1&f4&f5&reproj)": lambda r: all(passes(f, r, env) for f in STRONG),
        "COMBO all10": lambda r: all(passes(f, r, env) for f in ALLR),
        "reproj-gate only": lambda r: passes("reproj_selfcons", r, env),
        "f9_bbox_iou only": lambda r: passes("f9_bbox_iou", r, env),
    }
    conf_single = {f: confusion(recs, single(f)) for f in FILTERS}
    conf_combo = {name: confusion(recs, fn) for name, fn in combos.items()}

    # headline accept = strong combo
    accept_fn = combos["COMBO strong(f1&f4&f5&reproj)"]
    for r in recs:
        r["accept"] = accept_fn(r)

    # ── overlays: TP/FP/FN buckets ───────────────────────────────────────
    tp = [r for r in recs if r["accept"] and r["gt_good"]]
    fp = [r for r in recs if r["accept"] and not r["gt_good"]]
    fn = [r for r in recs if not r["accept"] and r["gt_good"]]
    tp.sort(key=lambda r: r["corner_med"] or 1e9)
    fp.sort(key=lambda r: -(r["corner_med"] or 0))
    fn.sort(key=lambda r: r["corner_med"] or 1e9)
    n_ov = 0
    for bucket, tag in [(tp, "TP_acc-good"), (fp, "FP_acc-bad"), (fn, "FN_rej-good")]:
        for r in bucket[:6]:
            save_overlay(r, tag, os.path.join(
                OV_DIR, f"{tag}_{r['ses'][-6:]}_{r['fid']}.jpg"), r["accept"], env)
            n_ov += 1

    write_md(recs, N, dets, goods, grosses, env, conf_single, conf_combo,
             len(tp), len(fp), len(fn), n_ov, K, ksrc, W, H)
    print(f"\n[save] {MD_PATH}")
    print(f"[save] {n_ov} overlays -> {OV_DIR}")


def _c(v, fmt="{:.2f}"):
    return "  -  " if v is None else fmt.format(v)


def write_md(recs, N, dets, goods, grosses, env, conf_single, conf_combo,
             n_tp, n_fp, n_fn, n_ov, K, ksrc, W, H):
    L = []
    A = L.append
    A("# Stage B best on wood pallet WITH manual GT (selectivity)")
    A("")
    A("- weights: `weights/paper_s2_stageB/net_epoch_0057.pth` (PAPER_S2 Stage B best)")
    A(f"- data: wood RealSense D435I manual-GT, N={N} (2 sessions), {W}x{H}")
    A(f"- dims (unseen, user-given): {WOOD_DIMS} m | K: {ksrc}")
    A(f"        fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    A("- preprocess: **SQUASH** (Stage B training parity, eval_frame_squash)")
    A(f"- GT-good = corner_med(order-free Hungarian) < {GOOD_PX:.0f}px ; "
      f"GT-gross = corner_med > {GROSS_PX:.0f}px or no-det")
    A("- ★ N=45 소표본, wood=UNSEEN(치수·텍스처), 2세션, 필터 임계 heuristic.")
    A("")
    # accuracy
    n_det = len(dets)
    n_good = len(goods)
    n_gross = len(grosses)
    A("## Stage B wood 정확도 (GT 45 기준)")
    A("```")
    A(f"det (n_det>=6)     : {n_det}/{N}  ({100*n_det/N:.0f}%)")
    A(f"corner_med  median : {_c(med([r['corner_med'] for r in dets]), '{:.1f}')} px  (over det frames)")
    A(f"honest8     median : {_c(med([r['honest8'] for r in dets]), '{:.1f}')} px  (PnP reproj vs GT)")
    A(f"front       median : {_c(med([r['front'] for r in dets]), '{:.1f}')} px")
    A(f"rear        median : {_c(med([r['rear'] for r in dets]), '{:.1f}')} px")
    A(f"GT-good (<{GOOD_PX:.0f}px)   : {n_good}/{N}  ({100*n_good/N:.0f}%)")
    A(f"GT-gross(>{GROSS_PX:.0f}px)  : {n_gross}/{N}  ({100*n_gross/N:.0f}%)")
    A("```")
    A("")
    # selectivity
    A("## ★ 필터 선택성 (accept vs GT-good confusion, N=45)")
    A("```")
    A(f"{'filter/combo':<32}{'n_acc':>6}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}"
      f"{'Prec':>7}{'Recall':>8}")
    A("-" * 73)
    for f in FILTERS:
        c = conf_single[f]
        A(f"{f:<32}{c['n_acc']:>6}{c['tp']:>4}{c['fp']:>4}{c['fn']:>4}{c['tn']:>4}"
          f"{c['prec']:>7.3f}{c['recall']:>8.3f}")
    A("-" * 73)
    for name, c in conf_combo.items():
        A(f"{name:<32}{c['n_acc']:>6}{c['tp']:>4}{c['fp']:>4}{c['fn']:>4}{c['tn']:>4}"
          f"{c['prec']:>7.3f}{c['recall']:>8.3f}")
    A("```")
    A(f"(TP=accept&good FP=accept&bad FN=reject&good TN=reject&bad ; total GT-good={len(goods)})")
    A("")
    A("## per-frame")
    A("```")
    hdr = (f"{'frame':<20}{'det':>4}{'cmed':>6}{'hon8':>6}{'frnt':>5}{'rear':>5}"
           f"{'peak':>5}{'flip':>5}{'tta':>5}{'reproj':>7}{'f9iou':>6}"
           f"{'#p':>3} acc GTg")
    A(hdr)
    A("-" * len(hdr))
    for r in recs:
        s = r["scores"]
        npass = sum(1 for f in FILTERS if passes(f, r, env))
        A(f"{r['ses'][-6:]+'/'+r['fid']:<20}{r['n_det']:>4}"
          f"{_c(r['corner_med'],'{:.1f}'):>6}{_c(r['honest8'],'{:.1f}'):>6}"
          f"{_c(r['front'],'{:.1f}'):>5}{_c(r['rear'],'{:.1f}'):>5}"
          f"{_c(s['f1_peak']):>5}{_c(s['f3_flip'],'{:.0f}'):>5}"
          f"{_c(s['f4_tta_stab'],'{:.1f}'):>5}{_c(s['reproj_selfcons'],'{:.1f}'):>7}"
          f"{_c(s['f9_bbox_iou'],'{:.2f}'):>6}{npass:>3}"
          f"{'  Y' if r['accept'] else '  n'}{'  Y' if r['gt_good'] else '  n'}")
    A("```")
    A("")
    A(f"overlays: TP(accept&good)={n_tp} FP(accept&bad)={n_fp} FN(reject&good)={n_fn}, "
      f"saved {n_ov} -> `wood_gt_overlays/`")
    A(f"f8 GT-envelope: size[{env['size_lo']:.3f},{env['size_hi']:.3f}] "
      f"asp[{env['asp_lo']:.2f},{env['asp_hi']:.2f}]")
    A("")
    A("★ caveat: N=45 소표본, wood UNSEEN, 2세션, 필터 임계 heuristic. 절대수치 과해석 금지.")
    with open(MD_PATH, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
