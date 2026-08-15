"""stage22_myannot_overlays_challenge0123.py — per-frame GT vs challenge0123-pred.

목적: 확정 8/8 완전 어노 테스트셋 17장(cad 11 + noapril 6)의 각 이미지마다
개별 오버레이 1장(합본 금지). GT(초록) + challenge0123 예측(빨강) 을 원본 위에.

★ 모델 = challenge0123/final_net_epoch_0060.pth (최초 camera_dynamic_0123_v4
  baseline, squash 무패딩 학습). 이 모델은 squash 학습이라 추론도 squash(무패딩,
  PAD=0)로 돌려 순수 상태를 본다. reflect-pad 쓰지 않음. 극근접 cad 에서 squash
  과소검출이 나오면 = 모델 특성(버그 아님), 검출 안 되면 그대로 표기.

인프라 재사용: eval_capturecad_b2.eval_frame (belief peak decode, per-frame K,
order-free 메트릭). PAD=0 이면 pad_frame/belief_to_orig_pad 가 squash 경로.
예측 코너 = belief peak decode(raw belief 예측, PnP proj 아님).

프레임 소스: testset_full8_manifest.txt (domain fid json_path img_path).
경로는 repo 루트 기준 상대경로.

convention camera_dynamic_0123_v4:
  0 near-top-L, 1 near-top-R, 2 near-bot-R, 3 near-bot-L   (FRONT, -Z)
  4 far-top-L,  5 far-top-R,  6 far-bot-R,  7 far-bot-L    (REAR,  +Z)
  8 centroid
edges = front loop(0-1-2-3-0) + rear loop(4-5-6-7-4) + connectors(0-4,1-5,2-6,3-7).

★ 유효 코너: GT 는 미어노 코너를 sentinel [-1,-1], pred 는 미검출을 nan 으로
  저장. 둘 다 '없는 점' 으로 취급 → 양 끝점이 모두 유효한 edge 만 그린다.
"""
from __future__ import annotations
import importlib.util
import os

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
ECAD_PATH = os.path.join(
    ROOT, "data/pallet/eval_results/stage16_truncation_addon/"
    "capturecad_b2_eval/eval_capturecad_b2.py")
MANIFEST = os.path.join(
    ROOT, "data/pallet/eval_results/stage22_myannot_eval/"
    "testset_full8_manifest.txt")
OUT_DIR = os.path.join(
    ROOT, "data/pallet/eval_results/stage22_myannot_eval/overlays_challenge0123")

_spec = importlib.util.spec_from_file_location("ecad", ECAD_PATH)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

WEIGHTS = os.path.join(ROOT, "weights/challenge_track/challenge0123/final_net_epoch_0060.pth")
PAD = 0          # ★ squash 추론 (무패딩) — 모델 순수 상태
THRESH = 0.3

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),        # front loop
         (4, 5), (5, 6), (6, 7), (7, 4),        # rear loop
         (0, 4), (1, 5), (2, 6), (3, 7)]        # near->far connectors
GT_COL = (0, 255, 0)        # green
PR_COL = (0, 0, 255)        # red
FRONT_TXT = (255, 255, 0)   # cyan-ish (front corner idx)
REAR_TXT = (0, 200, 255)    # orange   (rear corner idx)


def read_manifest(path):
    """returns list of (domain, fid, json_abs, img_abs)."""
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue
            dom, fid, jrel, irel = parts[0], parts[1], parts[2], parts[3]
            out.append((dom, fid,
                        os.path.join(ROOT, jrel), os.path.join(ROOT, irel)))
    return out


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
    if x == -1.0 and y == -1.0:
        return False
    return True


def draw_cuboid(img, pts8, centroid, col, r, dot_thick, label_off):
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

    frames = read_manifest(MANIFEST)
    print(f"[manifest] {len(frames)} frames "
          f"(cad {sum(1 for f in frames if f[0]=='cad')} + "
          f"noapril {sum(1 for f in frames if f[0]=='noapril')})")

    index = ["# challenge0123 (squash/no-pad) pred vs GT — 8/8 full-anno testset",
             f"# weights: {WEIGHTS}",
             "# file                          domain fid                    "
             "V det corner_med worst2 honest8"]
    n_saved, n_nodet = 0, 0
    for dom, fid, jp, ip in frames:
        if not (os.path.exists(jp) and os.path.exists(ip)):
            print(f"[skip] missing: {dom} {fid}")
            continue
        r = E.eval_frame(model, jp, ip, device, THRESH, PAD)
        if r is None:
            print(f"[skip] eval None: {dom} {fid}")
            continue
        img = cv2.imread(ip)
        gt8 = np.array(r["gt8"], float)
        gtc = r["gtc"]
        pr8 = np.array(r["pred8"], float)
        prc = r["pred_c"]

        draw_cuboid(img, gt8, gtc, GT_COL, 5, 2, 6)      # GT green (outer)
        draw_cuboid(img, pr8, prc, PR_COL, 3, -1, -8)    # pred red (filled)

        cor = r["corner"]
        w2 = r["worst2"]
        h8 = r["pnp_honest8"]

        def s(x):
            return f"{x:.1f}" if np.isfinite(x) else "inf"

        hdr = (f"{dom} V={r['v_geom']} det={r['n_det']}/8 "
               f"corner_med={s(cor)} w2={s(w2)} honest8={s(h8)}  "
               f"GT=green pred=red (squash)")
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, hdr, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        fname = f"{dom}_{fid}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, fname), img)
        n_saved += 1
        if r["n_det"] < 6:
            n_nodet += 1

        index.append(f"{fname:<32} {dom:<7} {fid:<22} "
                     f"{r['v_geom']} {r['n_det']}/8 {s(cor):>6} "
                     f"{s(w2):>6} {s(h8):>6}")

    with open(os.path.join(OUT_DIR, "index.txt"), "w") as f:
        f.write("\n".join(index) + "\n")

    print(f"[save] {OUT_DIR}  ({n_saved} overlays + index.txt)")
    print(f"[squash] frames with <6 det (under-detected): {n_nodet}/{n_saved}")


if __name__ == "__main__":
    main()
