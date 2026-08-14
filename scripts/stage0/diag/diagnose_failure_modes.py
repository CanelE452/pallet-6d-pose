"""diagnose_failure_modes.py — 트랙B 진단 1단계.
heatmap base(challenge0123)의 실패를 `도메인 × projected front-back separation`로
분해하고 flip/flatten 을 ρ 로 정량화. 거리·카메라높이 직접 분해 안 함(confound).

ρ_j = (pred_b - pred_f)·u_j / ||gt_b - gt_f||,  u_j = GT 깊이방향 단위벡터
  ρ≈1 정상깊이 / ρ≈0 flatten / ρ<0 flip / ρ>1 과장.
front=0-3, back=4-7, depth pair=(0,4)(1,5)(2,6)(3,7).
corner err = identity-matched(채널 i=corner i, flip 보이게) median px.
"""
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from eval_pvnet_heads import (load_pvnet_model, preprocess, belief_to_orig,  # noqa
                              collect_manual, load_gt8)
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa
from four_arm_pl_compare import collect_val_frames  # noqa

WEIGHTS = os.path.join(ROOT, "weights/challenge0123/final_net_epoch_0060.pth")
OUT = os.path.join(ROOT, "data/pallet/eval_results/stage9_diag")
DEPTH_PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]


def pred_corners_orig(model, device, ip):
    import cv2, torch
    img = cv2.imread(ip)
    if img is None:
        return None
    t, nw, nh, sc = preprocess(img); t = t.to(device)
    with torch.no_grad():
        beliefs, _ = model(t)
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps = extract_keypoints_from_belief(belief, 0.3)
    out = np.full((8, 2), np.nan)
    for i in range(8):
        if kps[i][0] >= 0:
            out[i] = belief_to_orig(kps[i][0], kps[i][1], bw, bh, nw, nh, sc)
    return out


def analyze(gt8, pred8):
    """returns dict: separation(px), rho_med, corner_err_med, n_det."""
    seps = [np.linalg.norm(gt8[b] - gt8[f]) for f, b in DEPTH_PAIRS]
    sep = float(np.median(seps))
    rhos = []
    for f, b in DEPTH_PAIRS:
        gv = gt8[b] - gt8[f]; gn = np.linalg.norm(gv)
        if gn < 1e-6 or np.isnan(pred8[f, 0]) or np.isnan(pred8[b, 0]):
            continue
        u = gv / gn
        rhos.append(float(np.dot(pred8[b] - pred8[f], u) / gn))
    det = ~np.isnan(pred8[:, 0])
    errs = [np.linalg.norm(pred8[i] - gt8[i]) for i in range(8) if det[i]]
    return {"sep": sep,
            "rho_med": (float(np.median(rhos)) if rhos else np.nan),
            "rhos": rhos,
            "err_med": (float(np.median(errs)) if errs else np.nan),
            "n_det": int(det.sum())}


def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_pvnet_model(WEIGHTS, device, numVec=0, numSeg=0)

    frames = [(dom, fid, jp, ip) for dom, fid, jp, ip in collect_val_frames()]
    frames += [("manual", fid, jp, ip) for dom, fid, jp, ip in collect_manual(0)]

    recs = []
    for dom, fid, jp, ip in frames:
        gt8 = load_gt8(jp)
        pred8 = pred_corners_orig(model, device, ip)
        if pred8 is None:
            continue
        a = analyze(gt8, pred8); a["dom"] = dom; a["fid"] = fid
        recs.append(a)

    # separation tertile bins (전체 GT 기준)
    allsep = np.array([r["sep"] for r in recs])
    q1, q2 = np.percentile(allsep, [33.3, 66.6])
    def binof(s):
        return "lo(flat)" if s < q1 else ("hi(deep)" if s >= q2 else "mid")
    for r in recs:
        r["bin"] = binof(r["sep"])

    lines = ["TRACK-B DIAG 1 — domain × front-back separation × flip/flatten",
             f"weights={WEIGHTS}",
             f"separation tertiles (px): lo<{q1:.0f}  mid  hi>={q2:.0f}",
             "ρ: ≈1 정상 / ≈0 flatten / <0 flip.  corner err = identity-matched median px",
             "det% = n_det>=6 비율 (검출). err/ρ는 검출분만."]
    doms = ["outside", "night", "manual"]
    bins = ["hi(deep)", "mid", "lo(flat)"]
    hdr = (f"  {'domain':<9}{'sepbin':<10}{'N':>4}{'det%':>6}{'errMed':>8}"
           f"{'ρmed':>7}{'flip%':>7}{'flat%':>7}")
    lines.append("\n" + hdr); lines.append("  " + "-" * (len(hdr) - 2))
    for dom in doms:
        for b in bins:
            sub = [r for r in recs if r["dom"] == dom and r["bin"] == b]
            if not sub:
                continue
            det = [r for r in sub if r["n_det"] >= 6]
            errs = [r["err_med"] for r in det if not np.isnan(r["err_med"])]
            allrho = [x for r in det for x in r["rhos"]]
            detpct = 100 * len(det) / len(sub)
            em = (np.median(errs) if errs else float("nan"))
            rm = (np.median(allrho) if allrho else float("nan"))
            flip = (100 * np.mean([x < 0 for x in allrho]) if allrho else float("nan"))
            flat = (100 * np.mean([0 <= x < 0.5 for x in allrho]) if allrho else float("nan"))
            lines.append(f"  {dom:<9}{b:<10}{len(sub):>4}{detpct:>6.0f}"
                         f"{em:>8.1f}{rm:>7.2f}{flip:>7.0f}{flat:>7.0f}")
    # 도메인별 separation 분포
    lines.append("\n[도메인별 separation 중앙값(px) — 작을수록 flat/edge-on]")
    for dom in doms:
        s = [r["sep"] for r in recs if r["dom"] == dom]
        if s:
            lines.append(f"  {dom:<9} median_sep={np.median(s):6.0f}  "
                         f"(min {np.min(s):.0f} / max {np.max(s):.0f}, n={len(s)})")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "diag1_domain_separation.txt"), "w").write(txt)
    import json
    json.dump(recs, open(os.path.join(OUT, "diag1_records.json"), "w"),
              indent=2, default=str)
    print(f"\n[save] {OUT}/diag1_domain_separation.txt")


if __name__ == "__main__":
    main()
