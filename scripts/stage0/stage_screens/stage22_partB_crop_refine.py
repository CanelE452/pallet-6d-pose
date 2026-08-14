#!/usr/bin/env python3
"""STAGE22 PART B: crop-and-refine 2단 추론 (학습 X).

질문: 유효 해상도(short-side 400 → belief short-side 50)가 rear 정밀도 병목인가?
방법: pass1 표준추론(aspect-only, ★패딩 금지) → bbox(pred kp min/max+margin20%) →
       crop → aspect-resize → pass2 재추론 → 좌표 역변환(offset). same-frame 페어 비교.

★해석가이드(보고서 그대로):
  rear 개선 → 유효해상도 병목 신호(stride/2-stage 근거)
  rear 악화 → 스케일 분포 이탈 confound 가능(near-large 전례), "해상도 기각" 단정금지(confounded 표기)

재사용: corner01_diagnosis.infer machinery (load_pvnet_model, extract_keypoints_from_belief,
        aspect-only 좌표역매핑). GT = projected_cuboid[:8] (camera-facing 0123).
  real: same-idx per-corner valid(corner01 검증). syn: mixed_v8 convention → order-free(Hungarian) full-8 만.
Convention: 0-3=front, 4-7=rear.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
from tau_calibrate import collect_val_frames  # noqa
from eval_pvnet_heads import collect_manual, collect_syn  # noqa
from stage18_elevation_threshold import elev_from_pose  # noqa

CAD_DIR = os.path.join(ROOT, "challenge", "data", "capturepalletcad_manual_gt")
WEIGHTS = os.path.join(ROOT, "weights", "stage11_16k_B2_maskaux", "final_net_epoch_0084.pth")
OUT = os.path.join(ROOT, "data", "pallet", "eval_results", "stage22_coord_diag", "partB")
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
THRESHOLD = 0.3
MARGIN = 0.20
FRONT = [0, 1, 2, 3]
REAR = [4, 5, 6, 7]
GOOD, GROSS = 10.0, 20.0
BINS = [(-90, 3), (3, 8), (8, 90)]
BIN_LBL = ["<3", "3-8", "8+"]


def collect_cad():
    out = []
    for jp in sorted(glob.glob(os.path.join(CAD_DIR, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(CAD_DIR, fid + ".png")
        if os.path.exists(ip):
            out.append(("cad", fid, jp, ip))
    return out


def load_gt(jp):
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    dims = o.get("dimensions_m")
    pose = np.array(o["pose_transform"], float) if "pose_transform" in o else None
    cam = d.get("camera_data", {}).get("intrinsics")
    return gt8, dims, pose, cam


def per_corner_err(pred8, gt8):
    e = np.full(8, np.nan)
    for i in range(8):
        if np.isfinite(pred8[i, 0]):
            e[i] = float(np.linalg.norm(pred8[i] - gt8[i]))
    return e


def hungarian_full8(pred8, gt8):
    """order-free full-8 mean err (both faces detected). None if <8 det."""
    det = [i for i in range(8) if np.isfinite(pred8[i, 0])]
    if len(det) < 8:
        return None
    from scipy.optimize import linear_sum_assignment
    C = np.linalg.norm(pred8[:, None, :] - gt8[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(C)
    return float(np.mean([C[r, c] for r, c in zip(ri, ci)]))


def bin_of(e):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= e < hi:
            return i
    return len(BINS) - 1


def agg(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return {"n": 0, "median": None, "p90": None}
    return {"n": int(a.size), "median": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2)}


def build_infer(device):
    import cv2, torch
    from filter_pr_camfacing import extract_keypoints_from_belief
    from eval_pvnet_heads import load_pvnet_model
    model, numVec, numSeg = load_pvnet_model(WEIGHTS, device)
    print(f"[model] numVec={numVec} numSeg={numSeg}")

    def infer(img):
        """aspect-only(패딩X) 추론. pred8 in img-pixel coords (channel-order 보존)."""
        h0, w0 = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sc = 400.0 / min(h0, w0)
        nw = max(8, int(round(w0 * sc)) & ~7)
        nh = max(8, int(round(h0 * sc)) & ~7)
        t = (cv2.resize(rgb, (nw, nh)).astype(np.float32) / 255.0 - MEAN) / STD
        tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
        belief = out[0][-1][0].cpu().numpy()
        kps = extract_keypoints_from_belief(belief, THRESHOLD)
        bh, bw = belief.shape[1], belief.shape[2]
        ux, uy = nw / bw, nh / bh
        pred8 = np.full((8, 2), np.nan)
        n = 0
        for i, k in enumerate(kps[:8]):
            if k[0] < 0:
                continue
            pred8[i] = (k[0] * ux / sc, k[1] * uy / sc)
            n += 1
        return n, pred8
    return infer


def crop_bbox(pred8, w0, h0):
    det = pred8[np.isfinite(pred8[:, 0])]
    if len(det) < 2:
        return None
    x0, y0 = det[:, 0].min(), det[:, 1].min()
    x1, y1 = det[:, 0].max(), det[:, 1].max()
    mw, mh = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
    x0 = int(max(0, np.floor(x0 - mw)))
    y0 = int(max(0, np.floor(y0 - mh)))
    x1 = int(min(w0, np.ceil(x1 + mw)))
    y1 = int(min(h0, np.ceil(y1 + mh)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return x0, y0, x1, y1


def pnp_reproj(pred8, dims, cam):
    """SOLVEPNP_ITERATIVE(평면물체) full-8 reproj median px. real(dims known)만."""
    if dims is None or cam is None:
        return None
    det = [i for i in range(8) if np.isfinite(pred8[i, 0])]
    if len(det) < 6:
        return None
    import cv2
    from filter_pr_camfacing import canonical_kp3d
    obj = canonical_kp3d(dims["width"], dims["depth"], dims["height"])[:8]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], float)
    op = obj[det].astype(np.float64)
    ip = pred8[det].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(op, ip, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj.astype(np.float64), rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)
    errs = [np.linalg.norm(proj[i] - pred8[i]) for i in det]
    return float(np.median(errs))


def main():
    import cv2, torch
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {WEIGHTS} ({device})")
    infer = build_infer(device)

    real = collect_val_frames() + collect_manual() + collect_cad()
    syn = [("syn_val", f, j, i) for _, f, j, i in collect_syn(200)]
    frames = [(True, *r) for r in real] + [(False, *s) for s in syn]
    print(f"[frames] real={len(real)} syn={len(syn)}")

    recs = []
    skip_pass1 = 0
    skip_pass2 = 0
    for n, (is_real, dom, fid, jp, ip) in enumerate(frames):
        img = cv2.imread(ip)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        gt8, dims, pose, cam = load_gt(jp)
        elev = elev_from_pose(pose) if (is_real and pose is not None) else None

        n1, pred1 = infer(img)
        if n1 < 6:
            skip_pass1 += 1
            continue
        bb = crop_bbox(pred1, w0, h0)
        if bb is None:
            skip_pass2 += 1
            continue
        x0, y0, x1, y1 = bb
        crop = img[y0:y1, x0:x1]
        n2, pred2c = infer(crop)
        pred2 = pred2c.copy()
        pred2[:, 0] += x0
        pred2[:, 1] += y0
        # undetected in pass2 stay nan
        for i in range(8):
            if not np.isfinite(pred2c[i, 0]):
                pred2[i] = (np.nan, np.nan)

        e1 = per_corner_err(pred1, gt8)
        e2 = per_corner_err(pred2, gt8)

        def face_med(e, idx):
            v = [e[i] for i in idx if np.isfinite(e[i])]
            return float(np.mean(v)) if v else None

        rec = {
            "is_real": is_real, "dom": dom, "fid": str(fid),
            "elev": round(float(elev), 2) if elev is not None else None,
            "n1": int(n1), "n2": int(n2),
            "crop_wh": [x1 - x0, y1 - y0], "img_wh": [w0, h0],
            "p1_front": face_med(e1, FRONT), "p1_rear": face_med(e1, REAR),
            "p2_front": face_med(e2, FRONT), "p2_rear": face_med(e2, REAR),
            "p1_full8_hung": hungarian_full8(pred1, gt8),
            "p2_full8_hung": hungarian_full8(pred2, gt8),
            "p1_reproj": pnp_reproj(pred1, dims, cam),
            "p2_reproj": pnp_reproj(pred2, dims, cam),
        }
        recs.append(rec)
        if (n + 1) % 60 == 0:
            print(f"  [{n+1}/{len(frames)}] recs={len(recs)}")

    result = summarize(recs, skip_pass1, skip_pass2)
    json.dump({"records": recs, **result},
              open(os.path.join(OUT, "partB.json"), "w"), indent=1)
    write_summary(result, recs, skip_pass1, skip_pass2)
    print(f"[partB] done -> {OUT}  (recs={len(recs)}, skip_pass1={skip_pass1}, skip_pass2={skip_pass2})")


def summarize(recs, skip1, skip2):
    real = [r for r in recs if r["is_real"]]
    syn = [r for r in recs if not r["is_real"]]

    def paired(recs_, k1, k2):
        """same-frame pairs where both defined."""
        p = [(r[k1], r[k2]) for r in recs_ if r[k1] is not None and r[k2] is not None]
        if not p:
            return {"n": 0}
        a = np.array([x[0] for x in p]); b = np.array([x[1] for x in p])
        return {"n": len(p),
                "pass1_med": round(float(np.median(a)), 2),
                "refine_med": round(float(np.median(b)), 2),
                "delta_med": round(float(np.median(b - a)), 2),
                "n_improve": int((b < a - 1).sum()),
                "n_worse": int((b > a + 1).sum())}

    out = {"real_pairs": {}, "syn_pairs": {}, "real_elev_bins": {}}
    for lbl, (k1, k2) in [("front", ("p1_front", "p2_front")),
                          ("rear", ("p1_rear", "p2_rear")),
                          ("full8_hung", ("p1_full8_hung", "p2_full8_hung")),
                          ("reproj", ("p1_reproj", "p2_reproj"))]:
        out["real_pairs"][lbl] = paired(real, k1, k2)
        out["syn_pairs"][lbl] = paired(syn, k1, k2)

    # good/gross rates (rear, paired)
    def rate(recs_, key, thr, lt=True):
        v = [r[key] for r in recs_ if r[key] is not None]
        if not v:
            return None
        return round(sum(1 for x in v if (x < thr if lt else x > thr)) / len(v), 3)
    out["real_rear_good_rate"] = {"pass1": rate(real, "p1_rear", GOOD),
                                   "refine": rate(real, "p2_rear", GOOD)}
    out["real_rear_gross_rate"] = {"pass1": rate(real, "p1_rear", GROSS, lt=False),
                                    "refine": rate(real, "p2_rear", GROSS, lt=False)}

    # elevation bin decomposition (real, rear paired)
    for i, blbl in enumerate(BIN_LBL):
        g = [r for r in real if r["elev"] is not None and bin_of(r["elev"]) == i]
        out["real_elev_bins"][blbl] = {
            "n": len(g),
            "rear": paired(g, "p1_rear", "p2_rear"),
            "front": paired(g, "p1_front", "p2_front"),
        }
    return out


def write_summary(res, recs, skip1, skip2):
    L = []
    L.append("# STAGE22 PART B — crop-and-refine 2단 추론 (B2, 학습 X)\n")
    L.append(f"records={len(recs)}  skip_pass1(det<6)={skip1}  skip_pass2(no-bbox)={skip2}")
    L.append("pass1=aspect-only(패딩X). refine=pred bbox+margin20% crop→aspect400→재추론→offset역변환.")
    L.append("GT=projected_cuboid[:8]. real=same-idx per-corner(corner01 검증). "
             "syn=mixed_v8 convention→order-free(Hungarian) full-8 만 신뢰.\n")

    def ptbl(title, pd):
        L.append(f"## {title}")
        L.append("```")
        L.append(f"{'metric':>12} {'n':>4} {'pass1_med':>10} {'refine_med':>11} {'delta':>7} {'improve':>8} {'worse':>7}")
        L.append("-" * 66)
        for lbl in ["front", "rear", "full8_hung", "reproj"]:
            p = pd[lbl]
            if p.get("n", 0) == 0:
                L.append(f"{lbl:>12} {'0':>4}  (no pairs)")
                continue
            L.append(f"{lbl:>12} {p['n']:>4} {p['pass1_med']:>10} {p['refine_med']:>11} "
                     f"{p['delta_med']:>+7} {p['n_improve']:>8} {p['n_worse']:>7}")
        L.append("```\n")
    ptbl("REAL same-frame pairs (delta<0 = refine 개선)", res["real_pairs"])
    ptbl("SYN (in-domain control; front/rear convention-confounded, full8_hung/reproj 신뢰)", res["syn_pairs"])

    L.append("## REAL rear good/gross rate")
    L.append("```")
    g = res["real_rear_good_rate"]; gr = res["real_rear_gross_rate"]
    L.append(f"rear good(<10px)  pass1={g['pass1']}  refine={g['refine']}")
    L.append(f"rear gross(>20px) pass1={gr['pass1']}  refine={gr['refine']}")
    L.append("```\n")

    L.append("## REAL elevation-bin (rear paired)")
    L.append("```")
    L.append(f"{'elev':>6} {'n':>4} {'rear_p1':>8} {'rear_ref':>9} {'delta':>7} {'imp':>4} {'wrs':>4}")
    for blbl in BIN_LBL:
        b = res["real_elev_bins"][blbl]; p = b["rear"]
        if p.get("n", 0) == 0:
            L.append(f"{blbl:>6} {b['n']:>4}  (no rear pairs)")
            continue
        L.append(f"{blbl:>6} {p['n']:>4} {p['pass1_med']:>8} {p['refine_med']:>9} "
                 f"{p['delta_med']:>+7} {p['n_improve']:>4} {p['n_worse']:>4}")
    L.append("```\n")

    # verdict guide
    rear = res["real_pairs"]["rear"]
    L.append("## 해석 가이드 (task 지정)")
    if rear.get("n", 0) >= 5:
        d = rear["delta_med"]
        if d < -1:
            L.append(f"- REAL rear delta={d:+.1f}px (개선) → 유효해상도 병목 신호(stride/2-stage 근거). "
                     "★단 crop=near-large 스케일 이동이므로 스케일 confound 잔존, 예비.")
        elif d > 1:
            L.append(f"- REAL rear delta={d:+.1f}px (악화) → 해상도 기각 단정 금지. "
                     "스케일 분포 이탈 confound 가능(near-large 전례) — CONFOUNDED 표기.")
        else:
            L.append(f"- REAL rear delta={d:+.1f}px (무변화) → 유효해상도가 rear 병목 아님(주 신호). "
                     "rear 붕괴는 해상도보다 flat-view depth 성분(PART A 정합).")
    else:
        L.append(f"- REAL rear 페어 N={rear.get('n')} 소표본 → 예비, 판정 보류.")
    L.append(f"\n★ 페어 N 명시: real rear N={res['real_pairs']['rear'].get('n')}, "
             f"syn full8 N={res['syn_pairs']['full8_hung'].get('n')}.")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
