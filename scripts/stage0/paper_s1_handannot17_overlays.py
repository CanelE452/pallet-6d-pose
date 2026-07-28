"""paper_s1_handannot17_overlays.py — 손 어노 테스트셋 17장 각각에 대해
PaperBase-v2 vs Paper-S1 예측 + GT 오버레이 (개별 이미지 1장씩, 합본 금지).

목적: 고앙각 손라벨 셋 17장에서 base -> S1 변화를 눈으로 확인.
      ★ 이 셋은 고앙각이라 base 가 이미 잘함 -> 개선 미미 예상. 정직하게 전부 출력
      (cherry-pick 금지, 개선/무변화/악화 전부 포함).

데이터: stage22 testset_full8_manifest.txt (cad11 + noapril6, 8/8 완전 어노).
        dims: solve_pose 가 (1.1,1.3) / (1.3,1.1) 둘 다 시도 -> cad/noapril 모두 OK.
        corner_med(main metric) 는 2D order-free 라 dims 무관.
전처리: pad=0 (no-pad / aspect-only = official protocol).
인프라: eval_capturecad_b2.eval_frame (order-free Hungarian corner, solve_pose honest8).
색: GT=green (큐보이드+8코너+centroid), PaperBase-v2=red, Paper-S1=blue.
    front 0-3 / rear 4-7 작은 번호. 미검출 코너 skip.
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

MANIFEST = os.path.join(
    ROOT, "data/pallet/eval_results/stage22_myannot_eval/testset_full8_manifest.txt")
OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/paper_s1/handannot17_overlays")
MODELS = {
    "base": "weights/paper_base_v2/final_net_epoch_0060.pth",
    "s1":   "weights/paper_s1_maskaux/net_epoch_0065.pth",
}
PAD = 0            # no-pad / aspect-only (official protocol)
THRESH = 0.3
N_DET_MIN = 6
EPS = 1.0          # |impr| < EPS px -> "무변화(no-change)"

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


def load_manifest():
    frames = []
    with open(MANIFEST) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            dom, fid, jp, ip = ln.split()
            frames.append((dom, fid, os.path.join(ROOT, jp), os.path.join(ROOT, ip)))
    return frames


def main():
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = load_manifest()
    by = {}
    for dom, *_ in frames:
        by[dom] = by.get(dom, 0) + 1
    print(f"[set] handannot17 N={len(frames)} {by}")

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

    index = [
        "# paper_s1 handannot17 overlays — PaperBase-v2 vs Paper-S1 (개별 17장)",
        "# GT=green(cuboid+8corner+centroid)  PaperBase-v2=red  Paper-S1=blue",
        "# no-pad / aspect-only (official).  corner_med=per-frame order-free Hungarian vs GT8 (px)",
        "# V=in-frame GT corner count(고앙각셋).  det=n_det(>=6 valid)",
        f"# impr = base_cm - s1_cm  (양수=개선, |impr|<{EPS}px=무변화, 음수=악화)",
        "#",
        f"# {'idx':>3} {'domain':<8} {'fid':<20} {'V':>2} "
        f"{'base_det':>8} {'s1_det':>7} {'base_cm':>8} {'s1_cm':>7} "
        f"{'impr':>7} {'verdict':<9}",
    ]

    n_impr = n_same = n_worse = n_skip = 0
    saved = []

    for idx, (dom, fid, jp, ip) in enumerate(frames):
        rb, rs = res["base"].get(fid), res["s1"].get(fid)
        img = cv2.imread(ip)
        if img is None or rb is None or rs is None:
            index.append(f"# SKIP idx={idx} {dom} {fid} (load/infer fail)")
            n_skip += 1
            continue

        cm_b = corner_med_orderfree(rb["pred8"], rb["gt8"])
        cm_s = corner_med_orderfree(rs["pred8"], rs["gt8"])
        b_det, s_det = rb["n_det"], rs["n_det"]
        v = rb["v_geom"]

        # verdict (both detected enough for corner_med)
        if cm_b is None or cm_s is None:
            impr = None
            verdict = "under-det"
        else:
            impr = cm_b - cm_s
            if impr > EPS:
                verdict = "IMPROVE"
                n_impr += 1
            elif impr < -EPS:
                verdict = "WORSE"
                n_worse += 1
            else:
                verdict = "same"
                n_same += 1

        # ---- draw ----
        draw_cuboid(img, rb["gt8"], rb["gtc"], GT_COL, 6, 2, 7, num=False)   # GT
        draw_cuboid(img, rb["pred8"], rb["pred_c"], BASE_COL, 4, -1, -9)     # base
        draw_cuboid(img, rs["pred8"], rs["pred_c"], S1_COL, 3, -1, 6)        # s1

        def s(v):
            return f"{v:.1f}" if v is not None else "n/a"

        def hh(v):
            return f"{v:.1f}" if np.isfinite(v) else "inf"

        imp_s = f"{impr:+.1f}" if impr is not None else "n/a"
        hdr1 = (f"[{idx}] {dom} {fid}  V={v}  det base={b_det}/8 S1={s_det}/8")
        hdr2 = (f"corner_med: base {s(cm_b)} -> S1 {s(cm_s)}  ({imp_s}px, {verdict})  "
                f"| GT=green base=red S1=blue")
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(img, hdr1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, hdr2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        bx = int(round(cm_b)) if cm_b is not None else -1
        sx = int(round(cm_s)) if cm_s is not None else -1
        fname = f"{dom}_{idx:02d}_{fid}_base{bx}_s1{sx}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, fname), img)
        saved.append(fname)

        index.append(
            f"  {idx:>3} {dom:<8} {fid:<20} {v:>2} "
            f"{b_det:>8} {s_det:>7} {s(cm_b):>8} {s(cm_s):>7} "
            f"{imp_s:>7} {verdict:<9}")

    index.append("")
    index.append(f"# summary (EPS={EPS}px): IMPROVE={n_impr}  same={n_same}  "
                 f"WORSE={n_worse}  under-det/skip={n_skip}  total={len(frames)}")

    with open(os.path.join(OUT_DIR, "index.txt"), "w") as f:
        f.write("\n".join(index) + "\n")

    print(f"\n[save] {OUT_DIR}  ({len(saved)} overlays + index.txt)")
    print(f"[summary] IMPROVE={n_impr} same={n_same} WORSE={n_worse} skip={n_skip}")


if __name__ == "__main__":
    main()
