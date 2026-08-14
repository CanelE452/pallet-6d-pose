"""stage16_v8lt_failure.py — STAGE16 Step1: B2_final 의 real V<8(truncation) 실패 분해.

목적: 비싼 truncation 합성데이터를 블라인드로 양산하기 전에, real V<8 가 "어떻게"
실패하는지(어느 코너가 어느 border 로 잘리고, 화면안인데 못맞추나 vs 화면밖 NA)
프레임별 구조속성으로 측정 → add-on 생성 분포 처방.

학습 없음. B2_final 추론 1회 + GT 기하 분석.

재사용:
  - eval_stage11.load_model / K_from_json / dims_from_json / hungarian_match
  - eval_pvnet_heads.preprocess / belief_to_orig / heatmap_pred8 / collect_manual
  - filter_pr_camfacing.extract_keypoints_from_belief
  - tau_calibrate.collect_val_frames
  - annotate_pnp.solve_pose / project_with_pose   (honest full-8 reproj)

출력:
  data/pallet/results/stage16_truncation_addon/v8lt_failure_report.json
  data/pallet/results/stage16_truncation_addon/v8lt_failure_montage/*.jpg
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))
sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
sys.path.insert(0, os.path.join(ROOT, "data", "pallet", "eval_results", "stage11_16k"))

import cv2  # noqa: E402
import torch  # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from eval_pvnet_heads import (preprocess, belief_to_orig, heatmap_pred8,  # noqa: E402
                              collect_manual)
from tau_calibrate import collect_val_frames  # noqa: E402
from eval_stage11 import (load_model, K_from_json, dims_from_json,  # noqa: E402
                          hungarian_match)
import annotate_pnp as APNP  # noqa: E402

WEIGHTS = os.path.join(ROOT, "weights", "stage11_16k_B2_maskaux",
                       "final_net_epoch_0084.pth")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "results",
                       "stage16_truncation_addon")
MONTAGE_DIR = os.path.join(OUT_DIR, "v8lt_failure_montage")
THRESHOLD = 0.3
N_DET_MIN = 6
GOOD_PX = 10.0
HONEST_OK_PX = 15.0   # honest full-8 reproj 성공 기준 (STAGE12 정합)

# corner color (BGR) for montage. 0-3 front, 4-7 back.
CORNER_COL = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 128),
              (255, 0, 0), (255, 128, 0), (255, 255, 0), (255, 0, 128)]
CNAME = ["nTL", "nTR", "nBR", "nBL", "fTL", "fTR", "fBR", "fBL"]


def in_frame_mask(gt8, w, h):
    return ((gt8[:, 0] >= 0) & (gt8[:, 0] < w)
            & (gt8[:, 1] >= 0) & (gt8[:, 1] < h))


def border_of(pt, w, h):
    """화면밖 코너가 어느 경계를 넘었나. 다중 가능(예: L+T) -> set."""
    bs = set()
    if pt[0] < 0:
        bs.add("L")
    if pt[0] >= w:
        bs.add("R")
    if pt[1] < 0:
        bs.add("T")
    if pt[1] >= h:
        bs.add("B")
    return bs


def analyze_frame(model, dom, fid, jp, ip, device):
    img = cv2.imread(ip)
    if img is None:
        return None
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    h0, w0 = img.shape[:2]
    inb = in_frame_mask(gt8, w0, h0)
    V = int(inb.sum())

    # in/out bitmask (corner 0..7), out corner ids, border dirs
    in_bits = "".join("1" if inb[i] else "0" for i in range(8))
    out_ids = [i for i in range(8) if not inb[i]]
    borders = {}
    border_count = {"L": 0, "R": 0, "T": 0, "B": 0}
    for i in out_ids:
        bs = border_of(gt8[i], w0, h0)
        borders[i] = sorted(bs)
        for b in bs:
            border_count[b] += 1

    # projected size (in-frame bbox diag) + bbox area
    if inb.sum() >= 2:
        proj_size = float(np.hypot(gt8[inb, 0].ptp(), gt8[inb, 1].ptp()))
        bbox_area = float(gt8[inb, 0].ptp() * gt8[inb, 1].ptp())
    else:
        proj_size, bbox_area = np.nan, np.nan

    # camera elevation / azimuth from pose_transform (cam-frame).
    # elevation = angle of camera optical axis vs object up; we report from R.
    elev = azim = np.nan
    depth = np.nan
    pt = o.get("pose_transform")
    if pt:
        T = np.array(pt, float)
        if T.shape == (4, 4):
            R = T[:3, :3]
            depth = float(T[2, 3])
            # object Y axis (height/up, local Y=down) in cam frame -> tilt
            # elevation proxy: angle between cam -Z (toward obj) projected.
            # Use object up = -R[:,1] (local +Y=down). elevation = asin of its
            # cam-Y component is ambiguous; report camera viewing angles instead:
            #   azim = atan2(t_x, t_z) of object center in cam frame (deg)
            #   elev = atan2(-t_y, hypot(t_x,t_z))  (object above/below optical)
            tx, ty, tz = T[0, 3], T[1, 3], T[2, 3]
            azim = float(np.degrees(np.arctan2(tx, tz)))
            elev = float(np.degrees(np.arctan2(-ty, np.hypot(tx, tz))))

    near = bool(np.isfinite(depth) and depth < 3.24)   # eval_stage11 near edge
    large = bool(np.isfinite(proj_size) and proj_size >= 286.7)  # large edge

    # ── inference ──
    tensor, nw, nh, sc = preprocess(img)
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps_bel = extract_keypoints_from_belief(belief, THRESHOLD)
    pred8 = heatmap_pred8(kps_bel, bw, bh, nw, nh, sc)

    # heatmap peak score per corner (raw, threshold 무관: bmap.max())
    peak_scores = [float(kps_bel[i][2]) for i in range(8)]

    # heatmap raw top-1 error (threshold 無): use raw argmax of each belief map
    # for IN-FRAME corners only. = "화면안인데 못맞추나(appearance)" 신호.
    raw_err = {}
    for i in range(8):
        bm = belief[i]
        py, px = np.unravel_index(int(np.argmax(bm)), bm.shape)
        ox, oy = belief_to_orig(px, py, bw, bh, nw, nh, sc)
        raw_err[i] = float(np.hypot(ox - gt8[i, 0], oy - gt8[i, 1]))
    raw_err_inframe = [raw_err[i] for i in range(8) if inb[i]]
    raw_err_inframe_med = (float(np.median(raw_err_inframe))
                           if raw_err_inframe else np.nan)

    # detection (n_det>=6) + hungarian matched corners -> B2 failed corner ids
    dists, ci = hungarian_match(pred8, gt8)
    n_det = int(np.sum(~np.isnan(pred8[:, 0])))
    det_ok = n_det >= N_DET_MIN
    failed_corners = []   # GT corner ids matched but error>GOOD_PX
    corner_med = np.nan
    if dists is not None:
        corner_med = float(np.median(dists))
        for m, gi in enumerate(ci):
            if dists[m] > GOOD_PX:
                failed_corners.append(int(gi))

    # ── PnP (per-frame K, GT dims) + honest full-8 reproj ──
    K = K_from_json(d)
    dims = dims_from_json(d)
    kps9 = []
    for i in range(8):
        kps9.append(None if np.isnan(pred8[i, 0])
                    else [float(pred8[i, 0]), float(pred8[i, 1])])
    if kps_bel[8][0] >= 0:
        cx, cy = belief_to_orig(kps_bel[8][0], kps_bel[8][1], bw, bh, nw, nh, sc)
        kps9.append([float(cx), float(cy)])
    else:
        kps9.append(None)

    pnp_ok = 0
    sel_reproj = np.nan      # selected-keypoint reproj (gate signal, no GT)
    honest_reproj = np.nan   # full-8 vs GT projected_cuboid (incl. off-screen)
    pos_depth = False
    pose = None
    nvalid = sum(1 for k in kps9 if k is not None)
    if nvalid >= N_DET_MIN:
        try:
            pose = APNP.solve_pose(kps9, K, dims=dims, img_shape=img.shape)
        except Exception:
            pose = None
    if pose is not None:
        pnp_ok = 1
        sel_reproj = float(pose["reproj_error_px"])
        Rp, tp = pose["R"], pose["t"]
        pos_depth = bool(tp[2] > 0)
        proj_all = APNP.project_with_pose(Rp, tp, K, pose["dims"])
        errs = []
        for i in range(8):
            u, v = proj_all[i]
            if u == -1.0 and v == -1.0:
                errs.append(1e6)
            else:
                errs.append(float(np.hypot(u - gt8[i, 0], v - gt8[i, 1])))
        honest_reproj = float(np.median(errs))

    rec = {
        "frame_id": fid, "dom": dom, "V_geom": V,
        "in_frame_bitmask": in_bits, "out_of_frame_corners": out_ids,
        "out_corner_names": [CNAME[i] for i in out_ids],
        "border_per_out_corner": borders,
        "border_count": border_count,
        "proj_size_px": round(proj_size, 1) if np.isfinite(proj_size) else None,
        "bbox_area_px2": round(bbox_area, 0) if np.isfinite(bbox_area) else None,
        "depth_m": round(depth, 2) if np.isfinite(depth) else None,
        "elev_deg": round(elev, 1) if np.isfinite(elev) else None,
        "azim_deg": round(azim, 1) if np.isfinite(azim) else None,
        "near": near, "large": large,
        "n_det": n_det, "det_ok": bool(det_ok),
        "corner_med_px": round(corner_med, 1) if np.isfinite(corner_med) else None,
        "B2_failed_corner_ids": failed_corners,
        "peak_scores": [round(s, 3) for s in peak_scores],
        "raw_err_inframe_med_px": (round(raw_err_inframe_med, 1)
                                   if np.isfinite(raw_err_inframe_med) else None),
        "raw_err_per_corner_px": {i: round(raw_err[i], 1) for i in range(8)},
        "honest_full8_reproj_px": (round(honest_reproj, 1)
                                   if np.isfinite(honest_reproj) else None),
        "sel_kp_reproj_px": (round(sel_reproj, 1)
                             if np.isfinite(sel_reproj) else None),
        "pnp_ok": bool(pnp_ok), "pnp_positive_depth": pos_depth,
        "honest_success": bool(pnp_ok and pos_depth
                               and np.isfinite(honest_reproj)
                               and honest_reproj < HONEST_OK_PX),
    }
    return rec, img, gt8, pred8, kps9, pose, K, w0, h0


def draw_montage(rec, img, gt8, pred8, w0, h0, out_path):
    vis = img.copy()
    # frame border (drawn inside for visibility of which side is cut)
    # GT cuboid edges
    EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    def clippt(p):
        return (int(np.clip(p[0], -5, w0 + 5)), int(np.clip(p[1], -5, h0 + 5)))
    for a, b in EDGES:
        cv2.line(vis, clippt(gt8[a]), clippt(gt8[b]), (60, 200, 60), 1)
    # GT corners (in/out) + B2 pred
    inb = in_frame_mask(gt8, w0, h0)
    for i in range(8):
        gp = clippt(gt8[i])
        col = CORNER_COL[i]
        if inb[i]:
            cv2.circle(vis, gp, 5, col, -1)
        else:
            cv2.drawMarker(vis, gp, col, cv2.MARKER_TILTED_CROSS, 12, 2)
        if not np.isnan(pred8[i, 0]):
            pp = (int(pred8[i, 0]), int(pred8[i, 1]))
            cv2.circle(vis, pp, 4, col, 2)
            cv2.line(vis, gp, pp, col, 1)
    # border-cut annotation (which sides crossed)
    crossed = [b for b, c in rec["border_count"].items() if c > 0]
    txt1 = (f"{rec['frame_id'][:10]} {rec['dom']} V={rec['V_geom']} "
            f"out={rec['out_corner_names']} cut={crossed}")
    txt2 = (f"det={'Y' if rec['det_ok'] else 'N'}({rec['n_det']}) "
            f"honest={rec['honest_full8_reproj_px']} "
            f"sel={rec['sel_kp_reproj_px']} "
            f"rawIF={rec['raw_err_inframe_med_px']} "
            f"posD={'Y' if rec['pnp_positive_depth'] else 'N'}")
    cv2.rectangle(vis, (0, 0), (w0, 34), (0, 0, 0), -1)
    cv2.putText(vis, txt1, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1)
    cv2.putText(vis, txt2, (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (180, 220, 255), 1)
    cv2.imwrite(out_path, vis)


def aggregate(v8lt):
    """집계·처방 보고 텍스트."""
    n = len(v8lt)
    L = [f"# STAGE16 Step1 — real V<8 failure aggregate (N={n})", ""]
    # (1) V_geom 분포
    from collections import Counter
    vc = Counter(r["V_geom"] for r in v8lt)
    L.append("## (1) V_geom 분포")
    for v in sorted(vc):
        L.append(f"  V={v}: {vc[v]}")
    # (2) bitmask top-N
    bm = Counter(r["in_frame_bitmask"] for r in v8lt)
    L.append("\n## (2) in-frame bitmask 빈도 top (1=in,0=out; idx0..7=nTL nTR nBR nBL fTL fTR fBR fBL)")
    for mask, c in bm.most_common(10):
        outc = [CNAME[i] for i in range(8) if mask[i] == "0"]
        L.append(f"  {mask}  x{c}   out={outc}")
    # out-corner frequency
    oc = Counter()
    for r in v8lt:
        for i in r["out_of_frame_corners"]:
            oc[i] += 1
    L.append("\n## (2b) out-of-frame 코너 개별 빈도")
    for i in range(8):
        L.append(f"  {i} {CNAME[i]}: {oc.get(i,0)}")
    # (3) border 방향 분포
    bc = Counter()
    for r in v8lt:
        for b, c in r["border_count"].items():
            if c > 0:
                bc[b] += 1   # frame-level: this frame crossed border b
    L.append("\n## (3) border 방향 분포 (frame-level: 해당 border 로 잘린 프레임 수)")
    for b in ["L", "R", "T", "B"]:
        L.append(f"  {b}: {bc.get(b,0)}")
    # corner-level border crossings
    bcc = Counter()
    for r in v8lt:
        for b, c in r["border_count"].items():
            bcc[b] += c
    L.append("  (corner-level 누적 crossing): "
             + ", ".join(f"{b}={bcc.get(b,0)}" for b in ["L", "R", "T", "B"]))
    # (4) size/elev/azim/depth 분포
    def stat(key):
        vals = [r[key] for r in v8lt if r.get(key) is not None]
        if not vals:
            return "n/a"
        a = np.array(vals)
        return (f"min={a.min():.1f} med={np.median(a):.1f} max={a.max():.1f}")
    L.append("\n## (4) size / depth / elev / azim 분포 (V<8)")
    L.append(f"  proj_size_px : {stat('proj_size_px')}")
    L.append(f"  depth_m      : {stat('depth_m')}")
    L.append(f"  elev_deg     : {stat('elev_deg')}")
    L.append(f"  azim_deg     : {stat('azim_deg')}")
    nnear = sum(1 for r in v8lt if r["near"])
    nlarge = sum(1 for r in v8lt if r["large"])
    L.append(f"  near(<3.24m): {nnear}/{n}   large(size>=286.7px): {nlarge}/{n}")
    # (5) raw error: in-frame 못맞춤(appearance) vs 화면밖 NA
    det = [r for r in v8lt if r["det_ok"]]
    L.append("\n## (5) 실패 원인 분리: 화면안 코너 못맞춤(appearance) vs 화면밖(NA)")
    L.append(f"  det_ok(n_det>=6): {len(det)}/{n}")
    rif = [r["raw_err_inframe_med_px"] for r in v8lt
           if r["raw_err_inframe_med_px"] is not None]
    if rif:
        a = np.array(rif)
        L.append(f"  raw_err in-frame med (threshold無): "
                 f"min={a.min():.1f} med={np.median(a):.1f} max={a.max():.1f}")
        L.append(f"    -> in-frame 코너도 raw_err>{GOOD_PX}px 인 프레임: "
                 f"{int((a>GOOD_PX).sum())}/{len(rif)} "
                 f"(=화면안인데 못맞춤=appearance gap 신호)")
    hs = sum(1 for r in v8lt if r["honest_success"])
    L.append(f"  honest_success(pos-depth & full8<{HONEST_OK_PX}px): {hs}/{n}")
    # sel vs honest gap (gate-only false-accept 노출)
    pairs = [(r["sel_kp_reproj_px"], r["honest_full8_reproj_px"]) for r in v8lt
             if r["sel_kp_reproj_px"] is not None
             and r["honest_full8_reproj_px"] is not None]
    if pairs:
        L.append("  sel_reproj vs honest_full8 (gate 과대평가 노출):")
        for s, ho in sorted(pairs, key=lambda x: -x[1])[:8]:
            L.append(f"    sel={s:6.1f}  honest={ho:7.1f}")
    return "\n".join(L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MONTAGE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(WEIGHTS, device)

    frames = collect_val_frames() + collect_manual()
    print(f"[stage16] eval frames: {len(frames)}  weights={WEIGHTS}")

    all_recs = []
    v8lt = []
    for dom, fid, jp, ip in frames:
        res = analyze_frame(model, dom, fid, jp, ip, device)
        if res is None:
            continue
        rec, img, gt8, pred8, kps9, pose, K, w0, h0 = res
        all_recs.append(rec)
        if 0 <= rec["V_geom"] < 8:
            v8lt.append(rec)
            mp = os.path.join(MONTAGE_DIR, f"V{rec['V_geom']}_{rec['dom']}_{fid}.jpg")
            draw_montage(rec, img, gt8, pred8, w0, h0, mp)

    print(f"[stage16] V<8 frames: {len(v8lt)} / {len(all_recs)}")
    report = aggregate(v8lt)
    print("\n" + report)

    out = {
        "weights": WEIGHTS, "threshold": THRESHOLD,
        "n_frames_total": len(all_recs),
        "n_V8": sum(1 for r in all_recs if r["V_geom"] == 8),
        "n_Vlt8": len(v8lt),
        "honest_ok_px": HONEST_OK_PX,
        "corner_index_legend": {i: CNAME[i] for i in range(8)},
        "v8lt_frames": v8lt,
        "all_frames_summary": [
            {"frame_id": r["frame_id"], "dom": r["dom"], "V_geom": r["V_geom"],
             "det_ok": r["det_ok"]} for r in all_recs],
    }
    jp_out = os.path.join(OUT_DIR, "v8lt_failure_report.json")
    with open(jp_out, "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, "v8lt_failure_aggregate.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nsaved: {jp_out}")
    print(f"saved: {OUT_DIR}/v8lt_failure_aggregate.txt")
    print(f"montage: {MONTAGE_DIR}/  ({len(v8lt)} imgs)")


if __name__ == "__main__":
    main()
