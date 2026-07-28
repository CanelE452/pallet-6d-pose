"""paper_s1_allframes_overlays.py — Paper-S1 belief-peak 오버레이 (raw 전체, ★PnP 없음).

목적: 대상 세션의 모든 raw rgb (~4343장) 각각에 Paper-S1(net_epoch_0065) 예측
      키포인트를 오버레이해 프레임당 1장 저장(합본 금지) → 사용자가 전체
      unlabeled 풀을 브라우징하며 "정답(good PL)" 프레임을 눈으로 고르는 용도.

★ PnP / solve_pose / posdepth 계산·표시 없음 (사용자 지시).
  belief-peak 로 검출한 키포인트만 그리고, cuboid edge 는 검출 kp 직접 연결.
★ GT 없음(raw) → 정확도 표시 불가. det(n_kp) / mean belief peak 만 판단 프록시.
★ idempotent: 이미 있는 출력 파일은 skip, 없는 것만 생성.

전처리: reflect-pad100 (near-field 검출 확보용, official 아님).
색: 검출 코너=blue, edge=blue, centroid=cross, front 0-3 / rear 4-7 번호.
"""
from __future__ import annotations
import argparse
import csv
import glob
import importlib.util
import os
import sys
import time

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

import torch
from models import DopeNetwork
from filter_pr_camfacing import extract_keypoints_from_belief
from eval_pvnet_heads import preprocess

# eval_capturecad_b2 재사용: load_model / pad_frame / belief_to_orig_pad
ECAD_PATH = os.path.join(
    ROOT, "data/pallet/eval_results/stage16_truncation_addon/"
    "capturecad_b2_eval/eval_capturecad_b2.py")
_spec = importlib.util.spec_from_file_location("ecad", ECAD_PATH)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

WEIGHTS = os.path.join(ROOT, "weights/paper_s1_maskaux/net_epoch_0065.pth")
OUT_ROOT = os.path.join(ROOT, "data/pallet/eval_results/s1_allframes_overlays")
PAD = 100
THRESH = 0.3
N_DET_MIN = 6

# session -> raw rgb dir
SESSIONS = {
    "capturepalletcad":  "data/pallet/raw_data/outside/capturepalletcad/rgb",
    "capture0403noapril": "data/pallet/raw_data/capture0403noapril/rgb",
    "capturepallet02":   "data/pallet/raw_data/outside/capturepallet02/rgb",
    "capturepallet03":   "data/pallet/raw_data/outside/capturepallet03/rgb",
    "capturepallet04":   "data/pallet/raw_data/outside/capturepallet04/rgb",
    "capturepallet05":   "data/pallet/raw_data/outside/capturepallet05/rgb",
    "capturepallet08":   "data/pallet/raw_data/outside/capturepallet08/rgb",
}

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),        # front loop
         (4, 5), (5, 6), (6, 7), (7, 4),        # rear loop
         (0, 4), (1, 5), (2, 6), (3, 7)]        # near->far connectors
S1_COL = (255, 60, 0)     # blue (Paper-S1 pred)
FRONT_TXT = (255, 255, 0)
REAR_TXT = (0, 200, 255)


def _valid_mask(pts):
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    sentinel = (x == -1.0) & (y == -1.0)
    return ~(np.isnan(x) | sentinel)


def _is_valid_pt(p):
    if p is None:
        return False
    x = float(p[0])
    return not (np.isnan(x) or (x == -1.0 and float(p[1]) == -1.0))


def draw_pred(img, pts8, centroid):
    pts = np.asarray(pts8, float)
    ok = _valid_mask(pts)
    for a, b in EDGES:                       # cuboid edge = 검출 kp 직접 연결
        if ok[a] and ok[b]:
            cv2.line(img, (int(pts[a, 0]), int(pts[a, 1])),
                     (int(pts[b, 0]), int(pts[b, 1])), S1_COL, 1, cv2.LINE_AA)
    for i in range(8):
        if not ok[i]:
            continue
        p = (int(pts[i, 0]), int(pts[i, 1]))
        cv2.circle(img, p, 4, S1_COL, -1, cv2.LINE_AA)
        tc = FRONT_TXT if i < 4 else REAR_TXT
        cv2.putText(img, str(i), (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, tc, 1, cv2.LINE_AA)
    if _is_valid_pt(centroid):
        c = (int(centroid[0]), int(centroid[1]))
        cv2.drawMarker(img, c, S1_COL, cv2.MARKER_CROSS, 10, 2)


def infer_frame(model, ip, device):
    """raw 이미지 -> (pred8 (8,2) nan-fill, pred_c or None, n_det, mean_peak).

    ★ PnP 없음. belief peak 키포인트만. mean_peak = 검출 코너 belief peak 평균.
    """
    img = cv2.imread(ip)
    if img is None:
        return None
    H, W = img.shape[:2]
    proc = E.pad_frame(img, PAD)
    tensor, nw, nh, sc = preprocess(proc)
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps = extract_keypoints_from_belief(belief, THRESH)   # 9 x (wx,wy,peak)

    pred8 = np.full((8, 2), np.nan)
    peaks = []
    for i, k in enumerate(kps[:8]):
        if k[0] < 0:
            continue
        pred8[i] = E.belief_to_orig_pad(k[0], k[1], bw, bh, nw, nh, sc,
                                        PAD, W, H)
        peaks.append(float(k[2]))
    pred_c = None
    if kps[8][0] >= 0:
        pred_c = list(E.belief_to_orig_pad(kps[8][0], kps[8][1], bw, bh,
                                           nw, nh, sc, PAD, W, H))
    n_det = int(np.sum(_valid_mask(pred8)))
    mean_peak = float(np.mean(peaks)) if peaks else 0.0
    return img, pred8, pred_c, n_det, mean_peak


def gather():
    out = {}
    for sess, rel in SESSIONS.items():
        paths = sorted(glob.glob(os.path.join(ROOT, rel, "*.png")))
        out[sess] = paths
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="세션당 최대 프레임(0=전체, 테스트용)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_ROOT, exist_ok=True)
    model = E.load_model(WEIGHTS, device)
    print(f"[model] {WEIGHTS}  device={device}", flush=True)

    frames = gather()
    total = sum(len(v) for v in frames.values())
    print(f"[set] sessions={ {s: len(v) for s, v in frames.items()} } "
          f"total={total}", flush=True)

    index_rows = []          # (session, fid, det, mean_peak)
    summ = {}
    bad = []
    t0 = time.time()
    done = 0
    n_new = 0
    n_skip = 0

    for sess, paths in frames.items():
        sdir = os.path.join(OUT_ROOT, sess)
        os.makedirs(sdir, exist_ok=True)
        if args.limit:
            paths = paths[:args.limit]
        dets, peaks_all, n_detok = [], [], 0

        for ip in paths:
            fid = os.path.splitext(os.path.basename(ip))[0]
            done += 1

            # idempotent: 이미 생성된 출력이 있으면 skip (glob prefix)
            existing = glob.glob(os.path.join(sdir, f"{sess}_{fid}_det*.jpg"))
            if existing:
                n_skip += 1
                # index/summary 위해 파일명에서 det/pk 복구
                bn = os.path.basename(existing[0])
                try:
                    det = int(bn.split("_det")[1].split("_")[0])
                    pk = float(bn.split("_pk")[1].rsplit(".jpg", 1)[0])
                except Exception:
                    det, pk = 0, 0.0
                index_rows.append((sess, fid, det, pk))
                dets.append(det)
                peaks_all.append(pk)
                if det >= N_DET_MIN:
                    n_detok += 1
                continue

            res = infer_frame(model, ip, device)
            if res is None:
                bad.append(f"{sess}/{fid}")
                continue
            img, pred8, pred_c, n_det, mean_peak = res
            draw_pred(img, pred8, pred_c)

            hdr = (f"{sess} {fid}  det={n_det}/8  peak={mean_peak:.2f}")
            cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(img, hdr, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            fname = f"{sess}_{fid}_det{n_det}_pk{mean_peak:.2f}.jpg"
            cv2.imwrite(os.path.join(sdir, fname), img)
            n_new += 1

            index_rows.append((sess, fid, n_det, round(mean_peak, 3)))
            dets.append(n_det)
            peaks_all.append(mean_peak)
            if n_det >= N_DET_MIN:
                n_detok += 1

            if done % 500 == 0:
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"[{done}/{total}] new={n_new} skip={n_skip} "
                      f"bad={len(bad)} {rate:.1f}f/s ETA={eta/60:.1f}m",
                      flush=True)

        summ[sess] = {
            "n": len(paths),
            "det_ge6": n_detok,
            "det_pct": round(100 * n_detok / len(paths), 1) if paths else 0.0,
            "det_mean": round(float(np.mean(dets)), 2) if dets else 0.0,
            "peak_med": round(float(np.median(peaks_all)), 3)
            if peaks_all else 0.0,
        }
        print(f"[done] {sess}: n={summ[sess]['n']} "
              f"det>=6={n_detok} ({summ[sess]['det_pct']}%) "
              f"peak_med={summ[sess]['peak_med']}", flush=True)

    # index.csv
    with open(os.path.join(OUT_ROOT, "index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "fid", "det", "mean_peak"])
        w.writerows(index_rows)

    # bad_frames.txt
    if bad:
        with open(os.path.join(OUT_ROOT, "bad_frames.txt"), "w") as f:
            f.write("\n".join(bad) + "\n")

    # summary.md
    lines = ["# Paper-S1 all-frames overlays — 세션별 검출/신뢰도 요약", "",
             f"weights: `{WEIGHTS}`  ",
             "전처리: reflect-pad100 (near-field, official 아님)  ",
             "★ PnP 없음 — belief-peak 키포인트만. GT 없음(raw)  ",
             "★ 정확도 표시 불가 → det(검출 코너수) / peak(belief peak) 프록시  ",
             "★ challenge/palletobj 계열은 S1(paper-track)엔 unseen → 도메인갭  ",
             "★ cad near-field 미검출 다수 예상 [추정]", "",
             "```",
             f"{'session':<20} {'N':>5} {'det>=6':>7} {'det%':>6} "
             f"{'det_mean':>9} {'peak_med':>9}",
             "-" * 62]
    tot_n = tot_ok = 0
    for sess in SESSIONS:
        s = summ.get(sess)
        if not s:
            continue
        lines.append(f"{sess:<20} {s['n']:>5} {s['det_ge6']:>7} "
                     f"{s['det_pct']:>6} {s['det_mean']:>9} {s['peak_med']:>9}")
        tot_n += s["n"]
        tot_ok += s["det_ge6"]
    lines.append("-" * 62)
    lines.append(f"{'TOTAL':<20} {tot_n:>5} {tot_ok:>7} "
                 f"{round(100*tot_ok/tot_n,1) if tot_n else 0:>6}")
    lines.append("```")
    lines.append("")
    lines.append(f"new={n_new}  skip(existing)={n_skip}  bad={len(bad)}  "
                 f"elapsed={ (time.time()-t0)/60:.1f}m")
    lines.append("컬럼: det>=6=검출성공(코너 6+) 프레임수, det%=검출률, "
                 "det_mean=평균 검출 코너수(/8), peak_med=mean belief peak 중앙값.")
    with open(os.path.join(OUT_ROOT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[save] {OUT_ROOT}", flush=True)
    print(f"  new={n_new} skip={n_skip} bad={len(bad)} "
          f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
