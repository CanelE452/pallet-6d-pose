#!/usr/bin/env python3
"""STAGE22 PART C eval — coord vs control paired ablation readout.

1) ckpt 선택: synthetic val (order-free full-8 median + det). arm 당 best.
2) real 페어 eval: front/rear 분리 median + elevation bin + good/gross + honest full-8(order-free) + V=8 회귀.
3) 병리진단: coord arm rear 채널 hard-argmax peak ↔ soft-argmax μ 거리(belief px). 크면=전역 soft-argmax 원거리 false-peak 견인.

real inference = reflect-pad(pad=100) (validated real-eval 전처리, corner01/STAGE16-4 정합).
GT = projected_cuboid[:8] (camera-facing 0123). real same-idx per-corner(corner01 검증). syn=order-free만.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os, sys, json, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
from tau_calibrate import collect_val_frames  # noqa
from eval_pvnet_heads import collect_manual, collect_syn, load_pvnet_model  # noqa
from stage18_elevation_threshold import elev_from_pose  # noqa

CAD_DIR = os.path.join(ROOT, "challenge", "data", "capturepalletcad_manual_gt")
OUT = os.path.join(ROOT, "data", "pallet", "eval_results", "stage22_coord_diag", "partC")
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
PAD = 100; THRESHOLD = 0.3
FRONT = [0, 1, 2, 3]; REAR = [4, 5, 6, 7]
GOOD, GROSS, SPIKE = 10.0, 20.0, 15.0
BINS = [(-90, 3), (3, 8), (8, 90)]; BIN_LBL = ["<3", "3-8", "8+"]


def collect_cad():
    out = []
    for jp in sorted(glob.glob(os.path.join(CAD_DIR, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(CAD_DIR, fid + ".png")
        if os.path.exists(ip):
            out.append(("cad", fid, jp, ip))
    return out


def load_gt(jp):
    d = json.load(open(jp)); o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    pose = np.array(o["pose_transform"], float) if "pose_transform" in o else None
    return gt8, pose


def make_infer(model, device):
    import cv2, torch
    from filter_pr_camfacing import extract_keypoints_from_belief
    from dope_predict_mp4_pad import pad_frame

    def infer(img, pad=PAD, want_belief=False):
        h0, w0 = img.shape[:2]
        proc = pad_frame(img, pad, "reflect")
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        ph, pw = proc.shape[:2]
        sc = 400.0 / min(ph, pw)
        nw = max(8, int(round(pw * sc)) & ~7); nh = max(8, int(round(ph * sc)) & ~7)
        t = (cv2.resize(rgb, (nw, nh)).astype(np.float32) / 255.0 - MEAN) / STD
        tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
        belief = out[0][-1][0].cpu().numpy()
        kps = extract_keypoints_from_belief(belief, THRESHOLD)
        bh, bw = belief.shape[1], belief.shape[2]
        ux, uy = nw / bw, nh / bh
        pred8 = np.full((8, 2), np.nan); n = 0
        for i, k in enumerate(kps[:8]):
            if k[0] < 0:
                continue
            cx = (k[0] * ux) / sc; cy = (k[1] * uy) / sc
            pred8[i] = (cx * (w0 + 2 * pad) / w0 - pad, cy * (h0 + 2 * pad) / h0 - pad)
            n += 1
        return (n, pred8, belief) if want_belief else (n, pred8)
    return infer


def per_corner_err(pred8, gt8):
    e = np.full(8, np.nan)
    for i in range(8):
        if np.isfinite(pred8[i, 0]):
            e[i] = float(np.linalg.norm(pred8[i] - gt8[i]))
    return e


def hungarian8(pred8, gt8):
    det = [i for i in range(8) if np.isfinite(pred8[i, 0])]
    if len(det) < 8:
        return None
    from scipy.optimize import linear_sum_assignment
    C = np.linalg.norm(pred8[:, None, :] - gt8[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(C)
    return float(np.mean([C[r, c] for r, c in zip(ri, ci)]))


def facemed(e, idx):
    v = [e[i] for i in idx if np.isfinite(e[i])]
    return float(np.mean(v)) if v else None


def bin_of(e):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= e < hi:
            return i
    return len(BINS) - 1


def agg(v):
    a = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return None
    return round(float(np.median(a)), 2)


# ---------- syn val ckpt selection ----------
def eval_syn(infer, syn_frames):
    dets, full8 = 0, []
    for dom, fid, jp, ip in syn_frames:
        import cv2
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8, _ = load_gt(jp)
        n, pred8 = infer(img)
        if n >= 8:
            dets += 1
            h = hungarian8(pred8, gt8)
            if h is not None:
                full8.append(h)
    return {"n": len(syn_frames), "det8": dets,
            "det8_rate": round(dets / max(len(syn_frames), 1), 3),
            "full8_med": agg(full8)}


# ---------- real eval ----------
def eval_real(infer, real_frames, want_belief_diag=False):
    import cv2
    recs = []
    for dom, fid, jp, ip in real_frames:
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8, pose = load_gt(jp)
        if want_belief_diag:
            n, pred8, belief = infer(img, want_belief=True)
        else:
            n, pred8 = infer(img); belief = None
        if n < 6:
            recs.append({"dom": dom, "fid": str(fid), "n_det": int(n),
                         "elev": round(float(elev_from_pose(pose)), 2) if pose is not None else None,
                         "front": None, "rear": None, "full8": None, "detected": False,
                         "V_inframe": None})
            continue
        e = per_corner_err(pred8, gt8)
        h0, w0 = img.shape[:2]
        inb = ((gt8[:, 0] >= 0) & (gt8[:, 0] < w0) & (gt8[:, 1] >= 0) & (gt8[:, 1] < h0))
        rec = {"dom": dom, "fid": str(fid), "n_det": int(n), "detected": True,
               "V_inframe": int(inb.sum()),
               "elev": round(float(elev_from_pose(pose)), 2) if pose is not None else None,
               "front": facemed(e, FRONT), "rear": facemed(e, REAR),
               "full8": hungarian8(pred8, gt8)}
        if want_belief_diag and belief is not None:
            rec["mu_argmax_rear"] = mu_argmax_dist(belief, REAR)
            rec["mu_argmax_front"] = mu_argmax_dist(belief, FRONT)
        recs.append(rec)
    return recs


def mu_argmax_dist(belief, channels):
    """belief[c] hard-argmax peak ↔ soft-argmax μ 거리(belief px). 평균 over channels(peak>thr)."""
    import torch
    ds = []
    for c in channels:
        bm = belief[c]
        if bm.max() < THRESHOLD:
            continue
        py, px = np.unravel_index(np.argmax(bm), bm.shape)   # hard argmax
        b = torch.from_numpy(bm).float().clamp(min=0)
        s = b.sum()
        if s <= 0:
            continue
        H, W = bm.shape
        ys = torch.arange(H).float(); xs = torch.arange(W).float()
        mu_y = float((b.sum(1) * ys).sum() / s)              # global soft-argmax
        mu_x = float((b.sum(0) * xs).sum() / s)
        ds.append(float(np.hypot(mu_x - px, mu_y - py)))
    return round(float(np.mean(ds)), 2) if ds else None


def summarize_real(recs):
    det = [r for r in recs if r["detected"]]
    def stat(key, sub=None):
        rr = det if sub is None else [r for r in det if sub(r)]
        return {"n": len([r for r in rr if r[key] is not None]),
                "med": agg([r[key] for r in rr])}
    out = {
        "n_frames": len(recs), "n_detected": len(det),
        "det_rate": round(len(det) / max(len(recs), 1), 3),
        "front": stat("front"), "rear": stat("rear"), "full8": stat("full8"),
        "rear_good_rate": _rate(det, "rear", GOOD, True),
        "rear_gross_rate": _rate(det, "rear", GROSS, False),
        "V8": {"front": stat("front", lambda r: r["V_inframe"] == 8),
               "rear": stat("rear", lambda r: r["V_inframe"] == 8),
               "n": len([r for r in det if r["V_inframe"] == 8])},
        "elev_bins": {},
    }
    for i, lbl in enumerate(BIN_LBL):
        g = [r for r in det if r["elev"] is not None and bin_of(r["elev"]) == i]
        out["elev_bins"][lbl] = {"n": len(g),
                                 "front": agg([r["front"] for r in g]),
                                 "rear": agg([r["rear"] for r in g])}
    return out


def _rate(det, key, thr, lt):
    v = [r[key] for r in det if r[key] is not None]
    if not v:
        return None
    return round(sum(1 for x in v if (x < thr if lt else x > thr)) / len(v), 3)


def paired(recs_a, recs_b, key):
    """same-frame pairs (both detected & key not None)."""
    ma = {r["fid"]: r for r in recs_a}
    pairs = []
    for rb in recs_b:
        ra = ma.get(rb["fid"])
        if ra and ra.get(key) is not None and rb.get(key) is not None:
            pairs.append((ra[key], rb[key], ra.get("elev")))
    if not pairs:
        return {"n": 0}
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    return {"n": len(pairs), "control_med": round(float(np.median(a)), 2),
            "coord_med": round(float(np.median(b)), 2),
            "delta_med": round(float(np.median(b - a)), 2),
            "n_improve": int((b < a - 1).sum()), "n_worse": int((b > a + 1).sum())}


def list_ckpts(arm_dir):
    cks = sorted(glob.glob(os.path.join(arm_dir, "net_epoch_*.pth")))
    return cks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control_dir", default=os.path.join(ROOT, "weights/stage22_coord_pilot/control"))
    ap.add_argument("--coord_dir", default=os.path.join(ROOT, "weights/stage22_coord_pilot/coord"))
    ap.add_argument("--n_syn", type=int, default=150)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_frames = collect_val_frames() + collect_manual() + collect_cad()
    syn_frames = [("syn_val", f, j, i) for _, f, j, i in collect_syn(args.n_syn)]
    print(f"[frames] real={len(real_frames)} syn={len(syn_frames)}")

    result = {"arms": {}}
    best = {}
    for arm, adir in [("control", args.control_dir), ("coord", args.coord_dir)]:
        cks = list_ckpts(adir)
        print(f"[{arm}] {len(cks)} ckpts")
        sel = []
        for ck in cks:
            model, _, _ = load_pvnet_model(ck, device)
            infer = make_infer(model, device)
            s = eval_syn(infer, syn_frames)
            sel.append({"ckpt": os.path.basename(ck), **s})
            print(f"  {os.path.basename(ck)} syn det8={s['det8_rate']} full8_med={s['full8_med']}")
            del model
            torch.cuda.empty_cache()
        # select: max det8_rate, tie-break min full8_med
        cand = [c for c in sel if c["full8_med"] is not None]
        pick = max(cand, key=lambda c: (c["det8_rate"], -(c["full8_med"] or 1e9)))
        best[arm] = os.path.join(adir, pick["ckpt"])
        result["arms"][arm] = {"ckpt_select": sel, "best": pick["ckpt"]}
        print(f"[{arm}] BEST = {pick['ckpt']}")

    # real eval on best ckpts (coord with belief diag)
    recs = {}
    for arm in ("control", "coord"):
        model, _, _ = load_pvnet_model(best[arm], device)
        infer = make_infer(model, device)
        recs[arm] = eval_real(infer, real_frames, want_belief_diag=(arm == "coord"))
        result["arms"][arm]["real_summary"] = summarize_real(recs[arm])
        del model
        torch.cuda.empty_cache()

    # paired
    result["paired"] = {
        "front": paired(recs["control"], recs["coord"], "front"),
        "rear": paired(recs["control"], recs["coord"], "rear"),
        "full8": paired(recs["control"], recs["coord"], "full8"),
    }
    # paired by elev bin (rear)
    result["paired_rear_by_elev"] = {}
    ctrl_by = {r["fid"]: r for r in recs["control"]}
    for i, lbl in enumerate(BIN_LBL):
        sub_a = [r for r in recs["control"] if r["elev"] is not None and bin_of(r["elev"]) == i]
        sub_b = [r for r in recs["coord"] if r["elev"] is not None and bin_of(r["elev"]) == i]
        result["paired_rear_by_elev"][lbl] = paired(sub_a, sub_b, "rear")

    # mu-argmax diagnostic (coord model)
    mu_rear = [r.get("mu_argmax_rear") for r in recs["coord"] if r.get("mu_argmax_rear") is not None]
    mu_front = [r.get("mu_argmax_front") for r in recs["coord"] if r.get("mu_argmax_front") is not None]
    result["mu_argmax_diag_coord"] = {
        "rear_med_beliefpx": agg(mu_rear), "rear_n": len(mu_rear),
        "front_med_beliefpx": agg(mu_front), "front_n": len(mu_front),
        "note": "hard-argmax peak↔soft-argmax μ 거리(belief px). rear>>front 면 soft-argmax가 rear에서 원거리 false-peak 견인.",
    }
    result["real_records"] = {"control": recs["control"], "coord": recs["coord"]}
    json.dump(result, open(os.path.join(OUT, "partC_eval.json"), "w"), indent=1)
    write_summary(result, best)
    print(f"[partC] eval done -> {OUT}")


def write_summary(res, best):
    L = ["# STAGE22 PART C — coord vs control paired ablation (pilot, B2 init +8ep, epoch_size6000)\n"]
    L.append(f"best ckpt: control={os.path.basename(best['control'])} coord={os.path.basename(best['coord'])}")
    L.append("real inference=reflect-pad(pad=100). GT=projected_cuboid[:8]. real same-idx per-corner.\n")

    for arm in ("control", "coord"):
        s = res["arms"][arm]["real_summary"]
        L.append(f"## {arm} real summary (N={s['n_frames']}, det={s['n_detected']} rate={s['det_rate']})")
        L.append("```")
        L.append(f"front_med={s['front']['med']}(n{s['front']['n']})  rear_med={s['rear']['med']}(n{s['rear']['n']})  "
                 f"full8_med={s['full8']['med']}(n{s['full8']['n']})")
        L.append(f"rear good(<10)={s['rear_good_rate']}  rear gross(>20)={s['rear_gross_rate']}")
        L.append(f"V=8(n{s['V8']['n']}): front={s['V8']['front']['med']} rear={s['V8']['rear']['med']}")
        eb = s["elev_bins"]
        L.append("elev bins (front/rear med):")
        for lbl in BIN_LBL:
            b = eb[lbl]
            L.append(f"  {lbl:>5} n={b['n']:>3} front={b['front']} rear={b['rear']}")
        L.append("```")

    L.append("\n## PAIRED (same-frame control vs coord; delta<0 = coord 개선)")
    L.append("```")
    for lbl in ("front", "rear", "full8"):
        p = res["paired"][lbl]
        if p.get("n", 0) == 0:
            L.append(f"{lbl:>7}: no pairs"); continue
        L.append(f"{lbl:>7}: n={p['n']} control={p['control_med']} coord={p['coord_med']} "
                 f"delta={p['delta_med']:+} improve={p['n_improve']} worse={p['n_worse']}")
    L.append("")
    L.append("rear paired by elev bin:")
    for lbl in BIN_LBL:
        p = res["paired_rear_by_elev"][lbl]
        if p.get("n", 0) == 0:
            L.append(f"  {lbl:>5}: no pairs"); continue
        L.append(f"  {lbl:>5}: n={p['n']} control={p['control_med']} coord={p['coord_med']} "
                 f"delta={p['delta_med']:+} imp={p['n_improve']} wrs={p['n_worse']}")
    L.append("```")

    m = res["mu_argmax_diag_coord"]
    L.append("\n## 병리진단 — coord rear 채널 hard-argmax↔soft-argmax μ 거리 (belief px)")
    L.append("```")
    L.append(f"rear  med={m['rear_med_beliefpx']} (n{m['rear_n']})")
    L.append(f"front med={m['front_med_beliefpx']} (n{m['front_n']})")
    L.append(f"{m['note']}")
    L.append("```")

    # verdict
    pr = res["paired"]["rear"]
    L.append("\n## 판정 (task 3-way)")
    if pr.get("n", 0) >= 5:
        d = pr["delta_med"]
        if d < -1:
            L.append(f"- (i) rear 페어 개선 delta={d:+.1f}px → signal. full-argmax + 국소 5×5 변형 제안. "
                     "μ↔argmax 진단으로 국소화 필요성 확인.")
        elif d > 1:
            L.append(f"- (iii) rear 회귀 delta={d:+.1f}px → coord 폐기 근거.")
        else:
            L.append(f"- (ii) rear 무변화 delta={d:+.1f}px → 슬라이드 '표준기법 채택(Integral Pose "
                     "Regression 계열)'으로 낮출 근거 + ablation 숫자 확보.")
    else:
        L.append(f"- rear 페어 N={pr.get('n')} 소표본 → 예비.")
    L.append("\n★ 페어 N 명시. real 소표본=예비. loss-ratio proxy(≠gradient) 로 λ 캘리브(0.24, ~7.5%).")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
