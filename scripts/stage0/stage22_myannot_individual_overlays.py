"""stage22_myannot_individual_overlays.py — per-frame GT vs B2-pred overlays.

목적: 손 어노 2 셋(capturepalletcad 22 + capture0403noapril 6 = 28장)의 각
이미지마다 개별 오버레이 1장(총 28장). GT(초록) + B2 예측(빨강) 을 원본 위에.
★ 합본(montage) 만들지 않음 — 프레임당 독립 파일.

인프라 재사용: eval_capturecad_b2.eval_frame (reflect-pad100, belief peak decode,
per-frame K, order-free 메트릭). 예측 코너 = belief peak decode 후 pad 역변환한
pred8/pred_c (PnP proj_all 아님 — raw belief 예측을 그림).

convention camera_dynamic_0123_v4 (annotate_pnp.make_pallet_keypoints_3d):
  0 near-top-L, 1 near-top-R, 2 near-bot-R, 3 near-bot-L   (FRONT, -Z)
  4 far-top-L,  5 far-top-R,  6 far-bot-R,  7 far-bot-L    (REAR,  +Z)
  8 centroid
edges = front loop(0-1-2-3-0) + rear loop(4-5-6-7-4) + connectors(0-4,1-5,2-6,3-7).

★ 그리기 버그 수정(2026-07-04): GT projected_cuboid 는 미어노 코너를 sentinel
  [-1,-1] 로 저장(null->sentinel). 기존엔 nan 만 skip 해서 (-1,-1) 로 edge 를 그려
  화면 좌상단으로 향하는 X자 대각선이 생겼다(cad_17 의 4,7). 이제 양 끝점이 모두
  유효(nan 아니고 sentinel 아님)한 edge 만 그린다. ★ 평가 수치(det/corner/honest8)
  는 점 위치 기반이라 이 edge 버그와 무관 = STAGE22 결론 불변.
"""
from __future__ import annotations
import glob
import importlib.util
import os

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
ECAD_PATH = os.path.join(
    ROOT, "data/pallet/eval_results/stage16_truncation_addon/"
    "capturecad_b2_eval/eval_capturecad_b2.py")
OUT_DIR = os.path.join(
    ROOT, "data/pallet/eval_results/stage22_myannot_eval/overlays_individual")

_spec = importlib.util.spec_from_file_location("ecad", ECAD_PATH)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

WEIGHTS = os.path.join(ROOT, "weights/stage11_16k_B2_maskaux/final_net_epoch_0084.pth")
DATASETS = {
    "cad": os.path.join(ROOT, "challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt"),
    "noapril": os.path.join(ROOT, "challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt"),
}
PAD = 100
THRESH = 0.3

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),        # front loop
         (4, 5), (5, 6), (6, 7), (7, 4),        # rear loop
         (0, 4), (1, 5), (2, 6), (3, 7)]        # near->far connectors
GT_COL = (0, 255, 0)        # green
PR_COL = (0, 0, 255)        # red
FRONT_TXT = (255, 255, 0)   # cyan-ish (front corner idx)
REAR_TXT = (0, 200, 255)    # orange   (rear corner idx)


def frames_of(dirp):
    out = []
    for jp in sorted(glob.glob(os.path.join(dirp, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(dirp, fid + ".png")
        if os.path.exists(ip):
            out.append((jp, ip))
    return out


def _valid_mask(pts):
    """유효 코너 = nan 아니고 sentinel (-1,-1) 아닌 것.
    GT 는 projected_cuboid 에서 미어노 코너를 [-1,-1] 로 저장(null->sentinel).
    pred 는 미검출을 nan 으로 저장. 둘 다 '없는 점' 으로 취급해야 edge 안 그림.
    (eval_frame 도 proj_all 에서 동일하게 (-1,-1)->제외 처리: line 150)"""
    x, y = pts[:, 0], pts[:, 1]
    sentinel = (x == -1.0) & (y == -1.0)
    return ~(np.isnan(x) | sentinel)


def _is_valid_pt(p):
    if p is None:
        return False
    x, y = float(p[0]), float(p[1])
    if np.isnan(x):
        return False
    if x == -1.0 and y == -1.0:
        return False
    return True


def draw_cuboid(img, pts8, centroid, col, r, dot_thick, label_off):
    """pts8: (8,2) w/ nan 또는 sentinel(-1,-1) for missing. centroid: [x,y]/None.
    ★ 양 끝점이 모두 유효한 edge 만 그린다 (미어노/미검출 코너로 이어진
    엉뚱한 대각선 방지 — cad_17 의 4,7 =[-1,-1] X자 꼬임 원인)."""
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
        tc = FRONT_TXT if i < 4 else REAR_TXT
        cv2.putText(img, str(i), (p[0] + label_off, p[1] - label_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, tc, 1, cv2.LINE_AA)
    if _is_valid_pt(centroid):
        c = (int(centroid[0]), int(centroid[1]))
        cv2.drawMarker(img, c, col, cv2.MARKER_CROSS, 10, 2)
        cv2.putText(img, "8", (c[0] + label_off, c[1] - label_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)


def main():
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E.load_model(WEIGHTS, device)

    index = ["# file                          domain fid                    "
             "V det corner_med worst2 h8_reproj"]
    n_saved = 0
    _sanity_done = [False]
    good_frames, bad_frames = [], []
    for dname, dirp in DATASETS.items():
        frames = frames_of(dirp)
        for i, (jp, ip) in enumerate(frames):
            r = E.eval_frame(model, jp, ip, device, THRESH, PAD)
            if r is None:
                continue
            img = cv2.imread(ip)
            gt8 = np.array(r["gt8"], float)
            gtc = r["gtc"]
            pr8 = np.array(r["pred8"], float)
            prc = r["pred_c"]

            # sanity: 첫 8/8-full-valid GT 프레임에서 edge 가 sentinel 로
            # 안 이어지는지(꼬임 없는 육면체) 로그로 검산
            gt_ok = _valid_mask(np.asarray(gt8, float))
            if gt_ok.sum() == 8 and not _sanity_done[0]:
                drawn = [(a, b) for a, b in EDGES if gt_ok[a] and gt_ok[b]]
                print(f"[sanity] {dname}_{i:02d} {r['fid']}: GT 8/8 valid, "
                      f"{len(drawn)}/12 edges drawn (expect 12, no sentinel edge)")
                _sanity_done[0] = True

            draw_cuboid(img, gt8, gtc, GT_COL, 5, 2, 6)      # GT green (outer)
            draw_cuboid(img, pr8, prc, PR_COL, 3, -1, -8)    # pred red (filled)

            cor = r["corner"]
            w2 = r["worst2"]
            h8 = r["pnp_honest8"]
            hdr = (f"{dname} V={r['v_geom']} det={r['n_det']}/8 "
                   f"cor={cor:.1f} w2={w2:.1f} h8={h8:.1f}  GT=green pred=red")
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(img, hdr, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            fid = r["fid"]
            fname = f"{dname}_{i:02d}_{fid}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, fname), img)
            n_saved += 1

            def s(x):
                return f"{x:.1f}" if np.isfinite(x) else "inf"
            index.append(f"{fname:<32} {dname:<7} {fid:<22} "
                         f"{r['v_geom']} {r['n_det']}/8 {s(cor):>6} "
                         f"{s(w2):>6} {s(h8):>6}")
            # crude qualitative bucket for the summary
            if r["det"] and np.isfinite(cor) and cor < 8:
                good_frames.append((dname, fid, cor))
            if (not r["det"]) or (np.isfinite(cor) and cor > 20) \
                    or (not np.isfinite(cor)):
                bad_frames.append((dname, fid, cor))

    with open(os.path.join(OUT_DIR, "index.txt"), "w") as f:
        f.write("\n".join(index) + "\n")

    print(f"[save] {OUT_DIR}  ({n_saved} overlays + index.txt)")
    print(f"[good] n={len(good_frames)}: "
          + ", ".join(f"{d}/{fid[:6]}({c:.1f})" for d, fid, c in good_frames[:8]))
    print(f"[bad ] n={len(bad_frames)}: "
          + ", ".join(f"{d}/{fid[:6]}({c})" for d, fid, c in bad_frames[:8]))


if __name__ == "__main__":
    main()
