"""paper_s1_improvement_overlays.py — filter-val 에서 PaperBase-v2 -> Paper-S1
corner 오차가 크게 준 프레임 8~10장의 개별 오버레이(cherry-pick 시연용).

목적: S1(mask-aux finetune)이 base 대비 개선한 것을 눈으로 보여주기.
      per-frame corner_med(order-free Hungarian vs GT8) 감소량 top 프레임 선택.
★ 전체 평균 아님 — base 가 나쁜 프레임 위주로 고른 시연(honest disclosure).

전처리: pad=0 (no-pad / aspect-only = official protocol). reflect-pad 금지.
데이터: stage25.frames_filterval() = collect_val_frames(outside+night split-lock)
        + manual_gt. final-test(pallet09/07, night09/08) SEAL.
인프라: eval_capturecad_b2.eval_frame (order-free Hungarian, solve_pose, honest8),
        stage22 draw_cuboid (sentinel [-1,-1] + nan skip 수정본).

선택: base_cm - s1_cm 큰 순. GT 정합 확실 gate = s1 honest8 finite & < HONEST_MAX
      (order-free 매칭 아티팩트 배제), 양 모델 det>=6.
색: GT=green, PaperBase-v2=red, Paper-S1=blue. front 0-3 / rear 4-7 작은 번호.
"""
from __future__ import annotations
import importlib.util
import os
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

ECAD_PATH = os.path.join(
    ROOT, "data/pallet/eval_results/stage16_truncation_addon/"
    "capturecad_b2_eval/eval_capturecad_b2.py")
_spec = importlib.util.spec_from_file_location("ecad", ECAD_PATH)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

import stage25_paperbase_eval as S25  # noqa: E402  (frames_filterval, split-lock)

OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/paper_s1/improvement_overlays")
MODELS = {
    "base": "weights/paper_base/paper_base_v2/final_net_epoch_0060.pth",
    "s1":   "weights/paper_s1/paper_s1_maskaux/net_epoch_0065.pth",
}
PAD = 0            # no-pad / aspect-only (official protocol)
THRESH = 0.3
N_DET_MIN = 6
HONEST_MAX = 30.0  # s1 honest8 gate: geometrically valid pose (아티팩트 배제)
N_PICK = 10

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),        # front loop
         (4, 5), (5, 6), (6, 7), (7, 4),        # rear loop
         (0, 4), (1, 5), (2, 6), (3, 7)]        # near->far connectors
GT_COL = (0, 255, 0)      # green
BASE_COL = (0, 0, 255)    # red   (PaperBase-v2)
S1_COL = (255, 60, 0)     # blue  (Paper-S1)
FRONT_TXT = (255, 255, 0)
REAR_TXT = (0, 200, 255)


def _valid_mask(pts):
    x, y = pts[:, 0], pts[:, 1]
    sentinel = (x == -1.0) & (y == -1.0)
    return ~(np.isnan(x) | sentinel)


def _is_valid_pt(p):
    if p is None:
        return False
    x, y = float(p[0]), float(p[1])
    if np.isnan(x):
        return False
    return not (x == -1.0 and y == -1.0)


def corner_med_orderfree(pred8, gt8):
    """per-frame order-free Hungarian corner median, GT sentinel/nan 마스킹."""
    pred8 = np.asarray(pred8, float)
    gt8 = np.asarray(gt8, float)
    pv = _valid_mask(pred8)
    gv = _valid_mask(gt8)
    if pv.sum() < N_DET_MIN or gv.sum() < N_DET_MIN:
        return None
    P, G = pred8[pv], gt8[gv]
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return float(np.median(cost[ri, ci]))


def draw_cuboid(img, pts8, centroid, col, r, dot_thick, label_off, num=True):
    pts = np.asarray(pts8, float)
    ok = _valid_mask(pts)
    for a, b in EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, (int(pts[a, 0]), int(pts[a, 1])),
                     (int(pts[b, 0]), int(pts[b, 1])), col, 1, cv2.LINE_AA)
    for i in range(8):
        if not ok[i]:
            continue
        p = (int(pts[i, 0]), int(pts[i, 1]))
        cv2.circle(img, p, r, col, dot_thick, cv2.LINE_AA)
        if num:
            tc = FRONT_TXT if i < 4 else REAR_TXT
            cv2.putText(img, str(i), (p[0] + label_off, p[1] - label_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, tc, 1, cv2.LINE_AA)
    if _is_valid_pt(centroid):
        c = (int(centroid[0]), int(centroid[1]))
        cv2.drawMarker(img, c, col, cv2.MARKER_CROSS, 10, 2)


def main():
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = S25.frames_filterval()
    by = {}
    for dom, *_ in frames:
        by[dom] = by.get(dom, 0) + 1
    print(f"[set] filterval N={len(frames)} {by}")

    # run both models over all frames
    res = {m: {} for m in MODELS}
    for mname, wrel in MODELS.items():
        mdl = E.load_model(os.path.join(ROOT, wrel), device)
        for dom, fid, jp, ip in frames:
            r = E.eval_frame(mdl, jp, ip, device, THRESH, PAD)
            if r is not None:
                res[mname][fid] = r
        del mdl
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[done] {mname}")

    # per-frame improvement table
    meta = {fid: (dom, jp, ip) for dom, fid, jp, ip in frames}
    rows = []
    for fid in res["base"]:
        rb, rs = res["base"].get(fid), res["s1"].get(fid)
        if rb is None or rs is None:
            continue
        cm_b = corner_med_orderfree(rb["pred8"], rb["gt8"])
        cm_s = corner_med_orderfree(rs["pred8"], rs["gt8"])
        if cm_b is None or cm_s is None:      # 양 모델 det>=6 필요
            continue
        dom = meta[fid][0]
        rows.append({
            "fid": fid, "dom": dom,
            "v_geom": rb["v_geom"],
            "cm_b": cm_b, "cm_s": cm_s, "impr": cm_b - cm_s,
            "s1_honest8": rs["pnp_honest8"], "base_honest8": rb["pnp_honest8"],
        })

    rows.sort(key=lambda x: x["impr"], reverse=True)

    # honest8 gate (GT 정합 확실 = s1 pose geometrically valid, 아티팩트 배제)
    picks = [x for x in rows
             if np.isfinite(x["s1_honest8"]) and x["s1_honest8"] < HONEST_MAX
             and x["impr"] > 0][:N_PICK]

    # ---- overlays ----
    index = ["# paper_s1 improvement overlays (cherry-pick 시연, no-pad/aspect)",
             "# GT=green  PaperBase-v2=red  Paper-S1=blue",
             "# corner_med = per-frame order-free Hungarian vs GT8 (px)",
             f"# honest8 gate: s1 honest8 finite & < {HONEST_MAX}px",
             "#",
             f"# {'file':<40} {'dom':<7} {'V':>2} {'base_cm':>8} "
             f"{'s1_cm':>7} {'impr':>6} {'s1_h8':>7} {'base_h8':>8}"]
    for x in picks:
        dom, jp, ip = meta[x["fid"]]
        rb, rs = res["base"][x["fid"]], res["s1"][x["fid"]]
        img = cv2.imread(ip)
        if img is None:
            continue
        draw_cuboid(img, rb["gt8"], rb["gtc"], GT_COL, 6, 2, 7, num=False)  # GT
        draw_cuboid(img, rb["pred8"], rb["pred_c"], BASE_COL, 4, -1, -9)    # base
        draw_cuboid(img, rs["pred8"], rs["pred_c"], S1_COL, 3, -1, 6)       # s1

        b, s = round(x["cm_b"], 1), round(x["cm_s"], 1)

        def h(v):
            return f"{v:.1f}" if np.isfinite(v) else "inf"
        hdr1 = (f"{dom} {x['fid']}  V={x['v_geom']}  "
                f"corner_med: base {b} -> S1 {s}  (down {round(x['impr'],1)}px)")
        hdr2 = (f"GT=green  base=red  S1=blue  |  honest8 base={h(x['base_honest8'])} "
                f"S1={h(x['s1_honest8'])}")
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(img, hdr1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, hdr2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        fname = f"{dom}_{x['fid']}_base{int(round(b))}_s1{int(round(s))}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, fname), img)
        index.append(f"  {fname:<40} {dom:<7} {x['v_geom']:>2} {b:>8} "
                     f"{s:>7} {round(x['impr'],1):>6} {h(x['s1_honest8']):>7} "
                     f"{h(x['base_honest8']):>8}")

    # also dump full sorted table (top 25) for transparency
    index.append("")
    index.append("# --- full sorted-by-improvement table (top 25, before gate) ---")
    for x in rows[:25]:
        def h(v):
            return f"{v:.1f}" if np.isfinite(v) else "inf"
        gate = "PICK" if x in picks else ""
        index.append(f"  {x['dom']:<7} {x['fid']:<24} V={x['v_geom']} "
                     f"base_cm={round(x['cm_b'],1):>6} s1_cm={round(x['cm_s'],1):>6} "
                     f"impr={round(x['impr'],1):>6} s1_h8={h(x['s1_honest8']):>7} {gate}")

    with open(os.path.join(OUT_DIR, "index.txt"), "w") as f:
        f.write("\n".join(index) + "\n")

    print(f"\n[save] {OUT_DIR}  ({len(picks)} overlays + index.txt)")
    print(f"[picks] {len(picks)} frames:")
    for x in picks:
        print(f"  {x['dom']}/{x['fid']} V={x['v_geom']} "
              f"base_cm={round(x['cm_b'],1)} -> s1_cm={round(x['cm_s'],1)} "
              f"(down {round(x['impr'],1)}px) s1_h8={round(x['s1_honest8'],1)}")


if __name__ == "__main__":
    main()
