"""diag2_raw_decode_stages.py — 트랙B 진단 2단계(학습 X, 1 inference).
night 등 미검출을 어느 단계에서 깨지는지 분해:
  per-channel raw top-1(global argmax+score, threshold/grouping 우회)
  + GT-local response(GT 주변 disk max) + second peak
  → 실패 4분류: no_response / competing_peak / corner_localization / gate_postprocess
  + threshold sweep(coverage / conditional-acc / end-to-end)
  + ρ_raw_all (threshold 없는 argmax 전체) vs ρ_detected (survivor bias 점검)

도메인: outside/night(filter-val) + manual. base=challenge0123.
"""
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from eval_pvnet_heads import (load_pvnet_model, preprocess, belief_to_orig,  # noqa
                              collect_manual, load_gt8)
from four_arm_pl_compare import collect_val_frames  # noqa

WEIGHTS = os.path.join(ROOT, "weights/challenge0123/final_net_epoch_0060.pth")
OUT = os.path.join(ROOT, "data/pallet/eval_results/stage9_diag")
TAU0 = 0.3                 # 현재 confidence threshold
N_DET_MIN = 6
GOOD_PX = 10.0
DEPTH_PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]
GT_DISK = 3               # GT-local response disk radius (belief px)


def gt_to_belief(gt_xy, sc, bw, bh, nw, nh):
    # inverse of belief_to_orig: ox=(bx*nw/bw)/sc -> bx = ox*sc*bw/nw
    return gt_xy[0] * sc * bw / nw, gt_xy[1] * sc * bh / nh


def disk_max(chan, cx, cy, r):
    H, W = chan.shape
    x0, x1 = max(0, int(cx - r)), min(W, int(cx + r + 1))
    y0, y1 = max(0, int(cy - r)), min(H, int(cy + r + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(chan[y0:y1, x0:x1].max())


def per_frame(model, device, ip, gt8, gt_ctr):
    import cv2, torch
    img = cv2.imread(ip)
    if img is None:
        return None
    t, nw, nh, sc = preprocess(img); t = t.to(device)
    with torch.no_grad():
        beliefs, _ = model(t)
    belief = beliefs[-1][0].cpu().numpy()   # (9,bh,bw)
    bh, bw = belief.shape[1], belief.shape[2]
    gts = list(gt8) + [gt_ctr]
    raw_pred = np.full((9, 2), np.nan)
    score = np.zeros(9); gtlocal = np.zeros(9); second = np.zeros(9)
    for k in range(9):
        ch = belief[k]
        yi, xi = np.unravel_index(int(ch.argmax()), ch.shape)
        score[k] = float(ch[yi, xi])
        raw_pred[k] = belief_to_orig(xi, yi, bw, bh, nw, nh, sc)
        gbx, gby = gt_to_belief(gts[k], sc, bw, bh, nw, nh)
        gtlocal[k] = disk_max(ch, gbx, gby, GT_DISK)
        # second peak: suppress disk around global argmax
        ch2 = ch.copy()
        x0, x1 = max(0, xi - 4), min(bw, xi + 5)
        y0, y1 = max(0, yi - 4), min(bh, yi + 5)
        ch2[y0:y1, x0:x1] = 0
        second[k] = float(ch2.max())
    # errors (orig px)
    corner_err = np.array([np.linalg.norm(raw_pred[i] - gt8[i]) for i in range(8)])
    center_err = float(np.linalg.norm(raw_pred[8] - gt_ctr))
    return {"raw_pred": raw_pred, "score": score, "gtlocal": gtlocal,
            "second": second, "corner_err": corner_err, "center_err": center_err,
            "gt8": gt8}


def classify(r):
    """미검출 단계 분류 (final = 현 decoder: corner score>=TAU0 가 6+개)."""
    sc = r["score"][:8]
    n_final = int((sc >= TAU0).sum())
    final_det = n_final >= N_DET_MIN
    raw_corner_good = float(np.median(r["corner_err"])) < GOOD_PX
    raw_center_good = r["center_err"] < GOOD_PX
    # GT 주변엔 반응 있는데(상위) global argmax는 딴 데(=competing) 판단:
    #   GT-local score 가 높은 코너 비율
    gt_resp_high = float(np.mean(r["gtlocal"][:8] >= TAU0))
    if final_det:
        return "detected", n_final
    if raw_corner_good and raw_center_good:
        return "gate_postprocess", n_final     # raw 좌표는 맞는데 gate가 reject
    if raw_center_good and not raw_corner_good:
        return "corner_localization", n_final
    if gt_resp_high >= 0.5:
        return "competing_peak", n_final       # GT엔 반응 있는데 argmax는 딴 peak
    return "no_response", n_final


def rho_of(pred8, gt8):
    out = []
    for f, b in DEPTH_PAIRS:
        gv = gt8[b] - gt8[f]; gn = np.linalg.norm(gv)
        if gn < 1e-6 or np.isnan(pred8[f, 0]) or np.isnan(pred8[b, 0]):
            continue
        out.append(float(np.dot(pred8[b] - pred8[f], gv / gn) / gn))
    return out


def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_pvnet_model(WEIGHTS, device, numVec=0, numSeg=0)

    frames = [(d, fid, jp, ip) for d, fid, jp, ip in collect_val_frames()]
    frames += [("manual", fid, jp, ip) for d, fid, jp, ip in collect_manual(0)]

    recs = []
    for dom, fid, jp, ip in frames:
        d = json.load(open(jp)); o = d["objects"][0]
        gt8 = np.array(o["projected_cuboid"], float)[:8]
        gt_ctr = np.array(o["projected_cuboid_centroid"], float)
        r = per_frame(model, device, ip, gt8, gt_ctr)
        if r is None:
            continue
        cls, n_final = classify(r)
        recs.append({"dom": dom, "fid": fid, "cls": cls, "n_final": n_final,
                     "raw_corner_med": float(np.median(r["corner_err"])),
                     "center_err": r["center_err"],
                     "score8": r["score"][:8].tolist(),
                     "gtlocal8": r["gtlocal"][:8].tolist(),
                     "rho_raw": rho_of(r["raw_pred"], gt8)})

    doms = ["outside", "night", "manual"]
    L = ["TRACK-B DIAG 2 — 미검출 단계 분해 (raw top-1, threshold/grouping 우회)",
         f"weights={WEIGHTS}  TAU0={TAU0}  GOOD<{GOOD_PX}px  gate=corner score>=TAU0 가 {N_DET_MIN}+"]
    # 1) 실패 분류표
    cats = ["detected", "gate_postprocess", "corner_localization",
            "competing_peak", "no_response"]
    L.append("\n[단계 분류] (N | 각 분류 %)")
    L.append(f"  {'dom':<9}{'N':>4}  " + "".join(f"{c[:10]:>13}" for c in cats))
    for dom in doms:
        sub = [r for r in recs if r["dom"] == dom]
        if not sub:
            continue
        row = f"  {dom:<9}{len(sub):>4}  "
        for c in cats:
            row += f"{100*sum(1 for r in sub if r['cls']==c)/len(sub):>12.0f}%"
        L.append(row)
    # 2) raw vs detected accuracy + ρ survivor bias
    L.append("\n[raw argmax 정확도 + ρ survivor-bias 점검]")
    L.append(f"  {'dom':<9}{'rawCornMed':>11}{'rawCtrErr':>10}"
             f"{'ρ_rawAll':>9}{'ρ_detOnly':>10}{'flip%raw':>9}")
    for dom in doms:
        sub = [r for r in recs if r["dom"] == dom]
        if not sub:
            continue
        rc = np.median([r["raw_corner_med"] for r in sub])
        ce = np.median([r["center_err"] for r in sub])
        rho_all = [x for r in sub for x in r["rho_raw"]]
        rho_det = [x for r in sub if r["cls"] == "detected" for x in r["rho_raw"]]
        fa = (100*np.mean([x < 0 for x in rho_all]) if rho_all else float("nan"))
        L.append(f"  {dom:<9}{rc:>11.1f}{ce:>10.1f}"
                 f"{(np.median(rho_all) if rho_all else float('nan')):>9.2f}"
                 f"{(np.median(rho_det) if rho_det else float('nan')):>10.2f}{fa:>8.0f}%")
    # 3) threshold sweep
    L.append("\n[threshold sweep] (corner score>=τ 가 6+ = accept)")
    L.append(f"  {'dom':<9}{'τ':>6}{'cover%':>8}{'condAcc%':>10}{'e2e%':>7}")
    taus = [round(TAU0 * f, 3) for f in (0, 0.33, 0.67, 1.0, 1.2)]
    for dom in doms:
        sub = [r for r in recs if r["dom"] == dom]
        if not sub:
            continue
        for tau in taus:
            acc = [r for r in sub if sum(1 for s in r["score8"] if s >= tau) >= N_DET_MIN]
            good = [r for r in acc if r["raw_corner_med"] < GOOD_PX]
            cov = 100*len(acc)/len(sub)
            ca = (100*len(good)/len(acc) if acc else float("nan"))
            e2e = 100*len(good)/len(sub)
            L.append(f"  {dom:<9}{tau:>6}{cov:>8.0f}{ca:>10.0f}{e2e:>7.0f}")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(OUT, "diag2_raw_stages.txt"), "w").write(txt)
    json.dump(recs, open(os.path.join(OUT, "diag2_records.json"), "w"),
              indent=2, default=str)
    print(f"\n[save] {OUT}/diag2_raw_stages.txt")


if __name__ == "__main__":
    main()
