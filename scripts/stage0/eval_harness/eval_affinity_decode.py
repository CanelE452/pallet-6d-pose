"""eval_affinity_decode.py — STEP6 affinity-association decode vs belief-peak decode.

마지막 미시험 조각: 이 프로젝트는 affinity map 을 학습하지만 추론(키포인트 위치
추출)엔 belief peak 만 쓴다. NVIDIA 정식 추론은 affinity 로 greedy association 을
한다. affinity 를 추론에 쓰면 belief-peak decode 보다 corner 정확도가 나아지나?
(특히 real back 코너.)

★ 학습 절대 안 함 — affinity 는 이미 학습된 가중치를 그대로 쓴다. decode(추론)만 비교.

두 decode (같은 forward 출력에 적용):
  belief-peak : extract_keypoints_from_belief(belief, thr)  (현재 방식, 대조군)
  affinity    : ObjectDetector.find_objects(belief, aff, config, scale_factor=1)
                (NVIDIA greedy association: 각 corner-peak 를 centroid 로 affinity
                 단위벡터 각도 매칭. 실험군)

convention 정합 [확인]:
  GT 생성 GenerateMapAffinity: 채널 2*i,2*i+1 = corner i → centroid 단위벡터.
  detector find_objects: aff[i_lists*2], aff[i_lists*2+1] 를 corner i 위치에서 읽어
    v_center = centroid - corner 와 비교 → 동일한 "corner→centroid" 방향. 정합 일치.
  corner 순서: belief 채널 i ↔ aff 채널 2i 로 1:1, 둘 다 camera-facing 0123. 재배선 불필요.

좌표계: find_objects 를 scale_factor=1 로 호출 → corner/centroid 모두 belief 격자(50px)
  좌표. belief-peak 와 동일 격자. 이후 kp_to_pred8 로 원본 px 매핑(eval_pvnet_heads 방식).

평가: same-frame 페어(두 decode 모두 n_det>=6 인 프레임)만 비교. split_metrics
  (order-free hungarian, overall/front/back) + good<10 / gross>20 + per-frame win.
  평가셋: filter-val(real, 주) / manual GT / synthetic.

출력: data/pallet/eval_results/stage6_affinity/
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

import cv2  # noqa: E402
import torch  # noqa: E402

from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from eval_pvnet_heads import (  # noqa: E402
    load_pvnet_model, preprocess, heatmap_pred8, kp_to_pred8, split_metrics,
    collect_manual, collect_syn, load_gt8, GOOD_PX, GROSS_PX, N_DET_MIN,
)
from tau_calibrate import collect_val_frames  # noqa: E402
from detector import ObjectDetector  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "stage6_affinity")


class DetCfg:
    """NVIDIA detector config (config_pose.yaml 기본값)."""
    def __init__(self, sigma=3.0, thresh_map=0.01, thresh_points=0.1,
                 threshold=0.3, thresh_angle=0.5):
        self.sigma = sigma
        self.thresh_map = thresh_map
        self.thresh_points = thresh_points
        self.threshold = threshold
        self.thresh_angle = thresh_angle


def affinity_decode_pred8(belief_t, aff_t, cfg, bw, bh, nw, nh, sc):
    """NVIDIA greedy association → (8,2) 원본 px (centroid 제외, NaN 보존).

    belief_t/aff_t : torch (9,H,W)/(16,H,W) on device.
    scale_factor=1 → corner 좌표는 belief 격자 단위. 가장 confident 한 object 선택.
    """
    objects, _ = ObjectDetector.find_objects(
        belief_t, aff_t, cfg, numvertex=8, run_sampling=False, scale_factor=1)
    pred8 = np.full((8, 2), np.nan)
    if not objects:
        return pred8
    # 가장 confident centroid 의 object 선택 (단일 팔레트 가정이지만 안전)
    best = max(objects, key=lambda o: o[3])
    corners = best[1]  # length-8 list of (x,y) belief-grid or None
    for i in range(8):
        c = corners[i]
        if c is None:
            continue
        ox, oy = _belief_to_orig(c[0], c[1], bw, bh, nw, nh, sc)
        pred8[i] = (ox, oy)
    return pred8


def _belief_to_orig(bx, by, bw, bh, nw, nh, sc):
    ux, uy = nw / bw, nh / bh
    return (bx * ux) / sc, (by * uy) / sc


def collect(evalset, n_frames):
    if evalset == "manual":
        return collect_manual(n_frames)
    if evalset == "synthetic":
        return collect_syn(n_frames)
    # filter-val: real, 주 평가셋. outside+night 모두.
    frames = collect_val_frames()
    if n_frames > 0:
        frames = frames[:n_frames]
    return frames


def bucket(vals):
    v = [x for x in vals if np.isfinite(x)]
    if not v:
        return {"N": 0, "median": None, "good": 0, "gross": 0}
    a = np.array(v)
    return {"N": int(a.size), "median": round(float(np.median(a)), 1),
            "good": int((a < GOOD_PX).sum()), "gross": int((a > GROSS_PX).sum())}


def run(model, frames, cfg, threshold, device, sanity_dir=None, n_sanity=0):
    """same-frame 페어로 belief-peak vs affinity-assoc 비교.

    Returns dict: paired metrics (overall/back/front) per decode + per-frame win.
    """
    pairs = []      # (belief_metric, aff_metric) — 둘 다 n_det>=6 인 프레임만
    n_total = n_bel_det = n_aff_det = 0
    sanity_done = 0
    for dom, fid, jp, ip in frames:
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8 = load_gt8(jp)
        tensor, nw, nh, sc = preprocess(img)
        tensor = tensor.to(device)
        with torch.no_grad():
            out = model(tensor)
        beliefs, affs = out[0], out[1]
        belief_t = beliefs[-1][0]          # torch (9,H,W)
        aff_t = affs[-1][0]                # torch (16,H,W)
        belief = belief_t.cpu().numpy()
        bh, bw = belief.shape[1], belief.shape[2]
        n_total += 1

        # belief-peak decode (대조군)
        kps_bel = extract_keypoints_from_belief(belief, threshold)
        ph8 = heatmap_pred8(kps_bel, bw, bh, nw, nh, sc)
        m_bel = split_metrics(ph8, gt8)

        # affinity-assoc decode (실험군)
        pa8 = affinity_decode_pred8(belief_t, aff_t, cfg, bw, bh, nw, nh, sc)
        m_aff = split_metrics(pa8, gt8)

        if m_bel["n_det"] >= N_DET_MIN:
            n_bel_det += 1
        if m_aff["n_det"] >= N_DET_MIN:
            n_aff_det += 1

        # sanity overlay (synthetic 첫 몇 장): GT vs aff-assoc 꼭짓점
        if sanity_dir and sanity_done < n_sanity:
            _sanity_overlay(img, ph8, pa8, gt8, fid, dom, sanity_dir)
            sanity_done += 1

        # same-frame 페어: 둘 다 검출(overall 유한)
        if np.isfinite(m_bel["overall"]) and np.isfinite(m_aff["overall"]):
            pairs.append((m_bel, m_aff, dom, fid))

    return _aggregate(pairs, n_total, n_bel_det, n_aff_det)


def _aggregate(pairs, n_total, n_bel_det, n_aff_det):
    res = {"n_total": n_total, "n_bel_det": n_bel_det, "n_aff_det": n_aff_det,
           "n_paired": len(pairs)}
    for key in ("overall", "front", "back"):
        bel_vals = [p[0][key] for p in pairs]
        aff_vals = [p[1][key] for p in pairs]
        res[key] = {"belief": bucket(bel_vals), "affinity": bucket(aff_vals)}
        # per-frame win (affinity strictly better by >0.5px)
        wins = ties = losses = 0
        for bv, av in zip(bel_vals, aff_vals):
            if not (np.isfinite(bv) and np.isfinite(av)):
                continue
            if av < bv - 0.5:
                wins += 1
            elif av > bv + 0.5:
                losses += 1
            else:
                ties += 1
        res[key]["aff_win"] = wins
        res[key]["tie"] = ties
        res[key]["aff_loss"] = losses
    return res


def _sanity_overlay(img, ph8, pa8, gt8, fid, dom, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vis = img.copy()
    for i in range(8):
        g = gt8[i]
        cv2.circle(vis, (int(g[0]), int(g[1])), 6, (0, 255, 0), 2)   # GT green
        if np.isfinite(ph8[i, 0]):
            cv2.circle(vis, (int(ph8[i, 0]), int(ph8[i, 1])), 4,
                       (255, 0, 0), -1)                              # belief blue
        if np.isfinite(pa8[i, 0]):
            cv2.drawMarker(vis, (int(pa8[i, 0]), int(pa8[i, 1])),
                           (0, 0, 255), cv2.MARKER_CROSS, 12, 2)     # aff red x
    cv2.putText(vis, "GT=green belief=blue aff=red-x", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(out_dir, f"sanity_{dom}_{fid}.jpg"), vis)


def fmt_table(title, res):
    lines = [f"# {title}",
             f"# frames={res['n_total']} bel_det>={N_DET_MIN}:{res['n_bel_det']} "
             f"aff_det>={N_DET_MIN}:{res['n_aff_det']} paired={res['n_paired']} "
             f"(good<{GOOD_PX} gross>{GROSS_PX})"]
    hdr = (f"{'subset':<9}{'decode':<10}{'N':>5}{'med':>8}{'good':>7}{'gross':>7}"
           f"{'aff_win':>9}{'tie':>6}{'aff_loss':>10}")
    lines.append(hdr)
    lines.append("─" * len(hdr))
    for key in ("overall", "back", "front"):
        d = res[key]
        b = d["belief"]
        a = d["affinity"]
        lines.append(
            f"{key:<9}{'belief':<10}{b['N']:>5}"
            f"{('-' if b['median'] is None else b['median']):>8}"
            f"{b['good']:>7}{b['gross']:>7}"
            f"{'':>9}{'':>6}{'':>10}")
        lines.append(
            f"{'':<9}{'affinity':<10}{a['N']:>5}"
            f"{('-' if a['median'] is None else a['median']):>8}"
            f"{a['good']:>7}{a['gross']:>7}"
            f"{d['aff_win']:>9}{d['tie']:>6}{d['aff_loss']:>10}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "weights", "challenge0123", "final_net_epoch_0060.pth"))
    ap.add_argument("--evalset", nargs="+",
                    default=["filter-val", "manual", "synthetic"],
                    choices=["filter-val", "manual", "synthetic"])
    ap.add_argument("--n_frames", type=int, default=0, help="0=전체")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="belief-peak decode 임계 (대조군)")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--thresh_map", type=float, default=0.01)
    ap.add_argument("--thresh_points", type=float, default=0.1)
    ap.add_argument("--thresh_angle", type=float, default=0.5)
    ap.add_argument("--n_syn_frames", type=int, default=200,
                    help="synthetic 평가셋 프레임 수")
    ap.add_argument("--n_sanity", type=int, default=3,
                    help="synthetic sanity overlay 장수")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, numVec, numSeg = load_pvnet_model(args.weights, device)
    print(f"[model] {args.weights} numVec={numVec} numSeg={numSeg} device={device}")
    assert numVec == 0 and numSeg == 0, \
        "expected pure NVIDIA DOPE (belief+affinity, no vec/seg heads)"

    cfg = DetCfg(args.sigma, args.thresh_map, args.thresh_points,
                 args.threshold, args.thresh_angle)
    print(f"[cfg] sigma={cfg.sigma} thresh_map={cfg.thresh_map} "
          f"thresh_points={cfg.thresh_points} thresh_angle={cfg.thresh_angle}")

    all_res = {}
    for es in args.evalset:
        nf = args.n_syn_frames if es == "synthetic" else args.n_frames
        frames = collect(es, nf)
        print(f"\n[eval] evalset={es} frames={len(frames)}")
        sanity_dir = (os.path.join(OUT_DIR, "sanity")
                      if es == "synthetic" else None)
        res = run(model, frames, cfg, args.threshold, device,
                  sanity_dir=sanity_dir, n_sanity=args.n_sanity)
        title = f"STEP6 affinity-assoc vs belief-peak — evalset={es}"
        table = fmt_table(title, res)
        print("\n" + table + "\n")
        all_res[es] = res
        with open(os.path.join(OUT_DIR, f"eval_{es}.txt"), "w") as f:
            f.write(table + "\n")

    json.dump({"weights": args.weights,
               "good_px": GOOD_PX, "gross_px": GROSS_PX,
               "config": {"sigma": cfg.sigma, "thresh_map": cfg.thresh_map,
                          "thresh_points": cfg.thresh_points,
                          "thresh_angle": cfg.thresh_angle,
                          "belief_threshold": args.threshold},
               "results": all_res},
              open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    print(f"[save] {OUT_DIR}/summary.json + eval_*.txt")


if __name__ == "__main__":
    main()
