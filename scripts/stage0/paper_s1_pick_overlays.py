"""paper_s1_pick_overlays.py — Paper-S1 예측 vs GT 개별 오버레이 (프레임당 1장).

목적: noapril(6) + outside/capturepallet(44) + cad(22) ≈ 72 프레임 각각에 대해
      Paper-S1 예측 + GT 오버레이를 1장씩 저장(합본 금지) → 사용자가 눈으로
      "어느 프레임이 정답(good PL)인가"를 골라 라벨 정의(사람 판단)를 세우는 용도.

★ 판정/분석 최소. 이미지 + 판단정보(코너별 오차) 제공이 목적.

데이터: GT 있는 dev 셋만 (★final-test 봉인: capturepallet07/09 제외).
전처리: reflect-pad100 (near-field 검출 확보용, 사용자가 다 보게. official 아님).
        도메인 통일 (cad near-field 대응).
인프라: eval_capturecad_b2.eval_frame (order-free Hungarian corner, per-frame K).
        corner metric 은 2D order-free 라 dims 무관 → domain dims 차이 무영향.
색: GT=green(cuboid edge + 8corner + centroid, sentinel[-1,-1]/nan skip)
    Paper-S1 pred=blue(검출 코너만, 미검출 skip). front 0-3 / rear 4-7 작은 번호.

★ challenge/palletobj 는 S1(paper-track)엔 unseen → 도메인갭 존재 (명시).
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

ECAD_PATH = os.path.join(
    ROOT, "data/pallet/eval_results/stage16_truncation_addon/"
    "capturecad_b2_eval/eval_capturecad_b2.py")
_spec = importlib.util.spec_from_file_location("ecad", ECAD_PATH)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

WEIGHTS = os.path.join(ROOT, "weights/paper_s1_maskaux/net_epoch_0065.pth")
OUT_ROOT = os.path.join(ROOT, "data/pallet/eval_results/s1_pick_overlays")
PAD = 100          # reflect-pad (near-field 검출 확보. official 아님)
THRESH = 0.3
N_DET_MIN = 6

# 도메인 -> manual_gt 디렉토리 (★07/09 = final-test 봉인 제외)
DOMAINS = {
    "noapril": ["capture0403noapril_manual_gt"],
    "outside": ["capturepallet02_manual_gt", "capturepallet03_manual_gt",
                "capturepallet04_manual_gt", "capturepallet05_manual_gt",
                "capturepallet08_manual_gt"],
    "cad":     ["capturepalletcad_manual_gt"],
}
DATA_ROOT = os.path.join(ROOT, "challenge", "data")

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),        # front loop
         (4, 5), (5, 6), (6, 7), (7, 4),        # rear loop
         (0, 4), (1, 5), (2, 6), (3, 7)]        # near->far connectors
GT_COL = (0, 255, 0)      # green
S1_COL = (255, 60, 0)     # blue  (Paper-S1)
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
    x, y = float(p[0]), float(p[1])
    if np.isnan(x):
        return False
    return not (x == -1.0 and y == -1.0)


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


def gather_frames():
    """도메인별 (jp, ip) 리스트 (정렬)."""
    out = {}
    for dom, dirs in DOMAINS.items():
        frames = []
        for d in dirs:
            base = os.path.join(DATA_ROOT, d)
            for jp in sorted(__import__("glob").glob(os.path.join(base, "*.json"))):
                fid = os.path.splitext(os.path.basename(jp))[0]
                ip = os.path.join(base, fid + ".png")
                if os.path.exists(ip):
                    frames.append((fid, jp, ip))
        out[dom] = frames
    return out


def fmt(v):
    return f"{v:.1f}" if (v is not None and np.isfinite(v)) else "n/a"


# ═══════════════════════════════════════════════════════════════════════
#  RAW mode — GT 없는 raw 프레임 전체에 Paper-S1 예측만 오버레이 (프레임당 1장).
#  사용자가 unlabeled 풀을 브라우징하며 good PL 을 눈으로 고르는 용도.
#  GT 없음 → 초록 GT/corner_med 없음. 판단 프록시 = det + mean belief peak + posdepth.
# ═══════════════════════════════════════════════════════════════════════

RAW_OUT_ROOT = os.path.join(ROOT, "data/pallet/eval_results/s1_allframes_overlays")

# session name -> raw rgb 디렉토리 (★사용자 지정 5세션만: cad/noapril/02/03/04/05/08)
RAW_SESSIONS = {
    "capturepalletcad":   "data/pallet/raw_data/outside/capturepalletcad",
    "capture0403noapril": "data/pallet/raw_data/capture0403noapril",
    "capturepallet02":    "data/pallet/raw_data/outside/capturepallet02",
    "capturepallet03":    "data/pallet/raw_data/outside/capturepallet03",
    "capturepallet04":    "data/pallet/raw_data/outside/capturepallet04",
    "capturepallet05":    "data/pallet/raw_data/outside/capturepallet05",
    "capturepallet08":    "data/pallet/raw_data/outside/capturepallet08",
}
# cam_K.txt 없을 때 기본 K (모든 세션 동일값 — RealSense D435i 촬영)
DEFAULT_K = np.array([[614.1838989257812, 0.0, 329.2803955078125],
                      [0.0, 614.3132934570312, 234.5288085937500],
                      [0.0, 0.0, 1.0]], float)


def load_K(session_dir):
    kp = os.path.join(session_dir, "cam_K.txt")
    if os.path.exists(kp):
        try:
            K = np.loadtxt(kp).reshape(3, 3)
            return K.astype(float)
        except Exception:
            pass
    return DEFAULT_K.copy()


def infer_raw(model, ip, K, device, threshold, pad):
    """GT 없는 raw 프레임 1장 추론. reflect-pad100 경로 재사용(eval_capturecad_b2).

    반환: pred8(8,2 nan=미검출), pred_c([u,v] or None), n_det, mean_peak,
          posdepth(bool), reproj(float or None). None if 이미지 로드 실패."""
    img = cv2.imread(ip)
    if img is None:
        return None
    H, W = img.shape[:2]

    proc = E.pad_frame(img, pad)
    tensor, nw, nh, sc = E.preprocess(proc)
    import torch
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps_bel = E.extract_keypoints_from_belief(belief, threshold)

    peaks = [float(k[2]) for k in kps_bel]          # 9 map 별 belief peak(신뢰도)
    mean_peak = float(np.mean(peaks)) if peaks else 0.0

    pred8 = np.full((8, 2), np.nan)
    for i, k in enumerate(kps_bel[:8]):
        if k[0] < 0:
            continue
        pred8[i] = E.belief_to_orig_pad(k[0], k[1], bw, bh, nw, nh, sc, pad, W, H)
    pred_c = None
    if kps_bel[8][0] >= 0:
        pred_c = list(E.belief_to_orig_pad(kps_bel[8][0], kps_bel[8][1],
                                           bw, bh, nw, nh, sc, pad, W, H))
    n_det = int(np.sum(~np.isnan(pred8[:, 0])))

    # PnP(치수 known dims, order-free W/D) — posdepth = 카메라 앞(Z>0) 여부
    posdepth, reproj = False, None
    kps9 = [None if np.isnan(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps9.append(pred_c)
    if sum(1 for k in kps9 if k is not None) >= 6:
        try:
            pose = E.APNP.solve_pose(kps9, K, dims=E.APNP.PALLET_DIMS,
                                     img_shape=img.shape)
            if pose is not None:
                t = np.asarray(pose["t"], float).reshape(-1)
                posdepth = bool(t[2] > 0)
                reproj = float(pose["reproj_error_px"])
        except Exception:
            pass

    return {"img": img, "pred8": pred8, "pred_c": pred_c, "n_det": n_det,
            "mean_peak": mean_peak, "posdepth": posdepth, "reproj": reproj}


def main_raw():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RAW_OUT_ROOT, exist_ok=True)

    model = E.load_model(WEIGHTS, device)
    print(f"[raw] model loaded on {device}: {WEIGHTS}")

    # 세션별 프레임 수집
    sessions = {}
    total = 0
    for sess, rel in RAW_SESSIONS.items():
        rgb_dir = os.path.join(ROOT, rel, "rgb")
        frames = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        sessions[sess] = (os.path.join(ROOT, rel), frames)
        total += len(frames)
        print(f"[raw] {sess}: {len(frames)} frames  (K from cam_K.txt)")
    print(f"[raw] TOTAL = {total} frames")

    idx_rows = []                       # index.csv
    bad_frames = []                     # 로드/추론 실패
    summ = {}                           # summary 집계
    done = 0
    t0 = time.time()

    for sess, (sess_dir, frames) in sessions.items():
        out_dir = os.path.join(RAW_OUT_ROOT, sess)
        os.makedirs(out_dir, exist_ok=True)
        K = load_K(sess_dir)
        n_det_ok, n_posdepth, peak_sum = 0, 0, 0.0

        for fp in frames:
            fid = os.path.splitext(os.path.basename(fp))[0]
            r = infer_raw(model, fp, K, device, THRESH, PAD)
            done += 1
            if r is None:
                bad_frames.append(f"{sess}/{fid}  (image load fail)")
                if done % 500 == 0:
                    print(f"[raw] {done}/{total}  ({time.time()-t0:.0f}s)")
                continue

            img = r["img"]
            n_det = r["n_det"]
            peak = r["mean_peak"]
            posdepth = r["posdepth"]
            reproj = r["reproj"]

            # ---- draw: Paper-S1 pred 파랑 (큐보이드 + 코너 + centroid + 번호) ----
            draw_cuboid(img, r["pred8"], r["pred_c"], S1_COL, 4, -1, 6, num=True)

            # ---- header (판단 정보: GT 없어 corner_med 생략) ----
            rp = f"{reproj:.1f}" if reproj is not None else "n/a"
            hdr1 = (f"{sess} {fid}  det={n_det}/8  mean_peak={peak:.3f}  "
                    f"posdepth={posdepth}  reproj={rp}px")
            hdr2 = "Paper-S1 pred = blue (front 0-3 / rear 4-7)   [raw, no GT]"
            cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
            cv2.putText(img, hdr1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, hdr2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)

            pk = int(round(peak * 100))
            fname = f"{sess}_{fid}_det{n_det}_pk{pk:03d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), img)

            idx_rows.append({"session": sess, "fid": fid, "det": n_det,
                             "mean_peak": round(peak, 4),
                             "posdepth": int(posdepth), "file": fname})
            if n_det >= 6:
                n_det_ok += 1
            if posdepth:
                n_posdepth += 1
            peak_sum += peak

            if done % 500 == 0:
                print(f"[raw] {done}/{total}  ({time.time()-t0:.0f}s)  "
                      f"last={sess}/{fid} det={n_det}")

        n = len(frames)
        summ[sess] = {"n": n, "det_ok": n_det_ok, "posdepth": n_posdepth,
                      "peak_mean": (peak_sum / n if n else 0.0)}

    # ---- index.csv ----
    with open(os.path.join(RAW_OUT_ROOT, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["session", "fid", "det",
                                          "mean_peak", "posdepth", "file"])
        w.writeheader()
        w.writerows(idx_rows)

    # ---- bad_frames.txt ----
    with open(os.path.join(RAW_OUT_ROOT, "bad_frames.txt"), "w") as f:
        f.write("\n".join(bad_frames) + ("\n" if bad_frames else ""))

    # ---- summary.md ----
    lines = ["# Paper-S1 all-frames overlays — 세션별 검출/신뢰도 요약 (raw, GT 없음)",
             "",
             f"weights: `{WEIGHTS}`  ",
             "전처리: reflect-pad100 (near-field 검출 확보. official 아님)  ",
             "판단 프록시: det(검출 코너수) + mean_peak(belief 신뢰도) + posdepth(PnP Z>0)  ",
             "★ GT 없음 → corner_med(정확도) 계산 불가. 예측 신뢰도만 표시  ",
             "★ challenge/palletobj 도메인갭 · cad near-field 미검출 다수 예상  ", "",
             "```",
             f"{'session':<20} {'N':>5} {'det>=6':>7} {'det%':>6} "
             f"{'posdepth':>8} {'peak_mean':>9}",
             "─" * 62]
    for sess in RAW_SESSIONS:
        s = summ.get(sess, {"n": 0, "det_ok": 0, "posdepth": 0, "peak_mean": 0})
        pct = (100 * s["det_ok"] / s["n"]) if s["n"] else 0.0
        lines.append(f"{sess:<20} {s['n']:>5} {s['det_ok']:>7} {pct:>5.1f}% "
                     f"{s['posdepth']:>8} {s['peak_mean']:>9.3f}")
    tot_n = sum(summ[s]["n"] for s in summ)
    tot_ok = sum(summ[s]["det_ok"] for s in summ)
    tot_pd = sum(summ[s]["posdepth"] for s in summ)
    pct = (100 * tot_ok / tot_n) if tot_n else 0.0
    lines += ["─" * 62,
              f"{'TOTAL':<20} {tot_n:>5} {tot_ok:>7} {pct:>5.1f}% {tot_pd:>8}",
              "```", "",
              f"생성 이미지: {len(idx_rows)}장  ·  로드 실패: {len(bad_frames)}장  ",
              "index.csv 컬럼: session/fid/det/mean_peak/posdepth/file "
              "(정렬·필터해서 good PL 후보 고르기용)  ",
              "파일명: `{session}_{fid}_det{N}_pk{peak*100}.jpg`  "]
    with open(os.path.join(RAW_OUT_ROOT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[raw][done] {len(idx_rows)} overlays, {len(bad_frames)} bad "
          f"({time.time()-t0:.0f}s)")
    print(f"[raw][save] {RAW_OUT_ROOT}")
    print(f"  index.csv, summary.md, bad_frames.txt + {list(RAW_SESSIONS)} 하위폴더")
    for sess in RAW_SESSIONS:
        s = summ[sess]
        pct = (100 * s["det_ok"] / s["n"]) if s["n"] else 0.0
        print(f"  {sess}: N={s['n']} det>=6={s['det_ok']} ({pct:.1f}%) "
              f"posdepth={s['posdepth']} peak={s['peak_mean']:.3f}")


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_ROOT, exist_ok=True)

    frames_by_dom = gather_frames()
    counts = {d: len(f) for d, f in frames_by_dom.items()}
    print(f"[set] frames per domain = {counts}  total={sum(counts.values())}")

    model = E.load_model(WEIGHTS, device)

    index = [
        "# paper_s1_pick_overlays — Paper-S1 예측 vs GT (프레임당 1장, 합본 금지)",
        f"# weights: {WEIGHTS}",
        "# 전처리: reflect-pad100 (near-field 검출 확보용, official 아님)",
        "# GT=green(cuboid+8corner+centroid)  Paper-S1 pred=blue(검출 코너만)",
        "# corner_med/front_med/rear_med = per-frame order-free Hungarian vs GT8 (px)",
        "# det=n_det(>=6 valid belief corner)  V=in-frame GT corner count",
        "# missing=미검출 belief 코너 인덱스(0-3=front,4-7=rear)  centroid=det 미포함",
        "#",
        f"# {'domain':<8} {'idx':>3} {'fid':<20} {'V':>2} {'det':>4} "
        f"{'corner_med':>10} {'front_med':>9} {'rear_med':>9} {'missing':<14}",
    ]

    # summary 집계용
    summ = {}

    for dom, frames in frames_by_dom.items():
        dom_dir = os.path.join(OUT_ROOT, dom)
        os.makedirs(dom_dir, exist_ok=True)
        dets, cms, n_nodet = [], [], 0

        for idx, (fid, jp, ip) in enumerate(frames):
            r = E.eval_frame(model, jp, ip, device, THRESH, PAD)
            img = cv2.imread(ip)
            if img is None or r is None:
                index.append(f"# SKIP {dom} idx={idx} {fid} (load/infer fail)")
                continue

            gt8 = r["gt8"]
            gtc = r["gtc"]
            pred8 = np.array(r["pred8"], float)
            pred_c = r["pred_c"]
            n_det = r["n_det"]
            v_geom = r["v_geom"]
            cm = r["corner"]        # overall order-free median (inf if under-det)
            fm = r["front"]
            rm = r["back"]

            pv = _valid_mask(pred8)
            missing = [i for i in range(8) if not pv[i]]
            miss_s = ",".join(str(i) for i in missing) if missing else "none"

            # ---- draw (GT 먼저, pred 위에) ----
            draw_cuboid(img, gt8, gtc, GT_COL, 6, 2, 7, num=False)          # GT
            draw_cuboid(img, pred8, pred_c, S1_COL, 4, -1, 6, num=True)     # S1

            # ---- header (판단 정보) ----
            cm_disp = fmt(cm)
            hdr1 = (f"{dom} [{idx}] {fid}  V={v_geom} det={n_det}/8  "
                    f"corner_med={cm_disp}px")
            hdr2 = (f"front_med={fmt(fm)}  rear_med={fmt(rm)}  "
                    f"missing=[{miss_s}]   GT=green  S1-pred=blue")
            cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
            cv2.putText(img, hdr1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, hdr2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)

            cx = int(round(cm)) if np.isfinite(cm) else -1
            fname = f"{dom}_{idx:02d}_{fid}_cm{cx}_det{n_det}.jpg"
            cv2.imwrite(os.path.join(dom_dir, fname), img)

            dets.append(n_det)
            if np.isfinite(cm):
                cms.append(cm)
            if n_det < N_DET_MIN:
                n_nodet += 1

            index.append(
                f"  {dom:<8} {idx:>3} {fid:<20} {v_geom:>2} {n_det:>4} "
                f"{cm_disp:>10} {fmt(fm):>9} {fmt(rm):>9} {'['+miss_s+']':<14}")

        summ[dom] = {
            "n": len(frames),
            "det_ge6": sum(1 for d in dets if d >= N_DET_MIN),
            "n_nodet": n_nodet,
            "det_mean": (float(np.mean(dets)) if dets else 0.0),
            "cm_n": len(cms),
            "cm_med": (float(np.median(cms)) if cms else None),
            "cm_good10": (sum(1 for c in cms if c < 10.0)),
        }

    # ---- index.txt ----
    with open(os.path.join(OUT_ROOT, "index.txt"), "w") as f:
        f.write("\n".join(index) + "\n")

    # ---- summary.md ----
    lines = ["# Paper-S1 pick overlays — 도메인별 검출/오차 요약", "",
             f"weights: `{WEIGHTS}`  ", "전처리: reflect-pad100 (official 아님)  ",
             "corner_med = per-frame order-free Hungarian vs GT8 (px)  ",
             "★ challenge/palletobj 는 S1(paper-track)엔 unseen → 도메인갭 존재  ",
             "★ N 소표본(도메인당 6~44) → 과결론 금지", "",
             "```",
             f"{'domain':<8} {'N':>3} {'det>=6':>7} {'no-det':>7} "
             f"{'det_mean':>9} {'cm_med':>7} {'cm<10px':>8}",
             "─" * 60]
    for dom in DOMAINS:
        s = summ[dom]
        cm_med_s = f"{s['cm_med']:.1f}" if s["cm_med"] is not None else "n/a"
        good_s = f"{s['cm_good10']}/{s['cm_n']}"
        lines.append(
            f"{dom:<8} {s['n']:>3} {s['det_ge6']:>7} {s['n_nodet']:>7} "
            f"{s['det_mean']:>9.2f} {cm_med_s:>7} {good_s:>8}")
    tot_n = sum(summ[d]["n"] for d in DOMAINS)
    tot_det = sum(summ[d]["det_ge6"] for d in DOMAINS)
    lines.append("─" * 60)
    lines.append(f"{'TOTAL':<8} {tot_n:>3} {tot_det:>7}")
    lines.append("```")
    lines.append("")
    lines.append("컬럼: det>=6=검출성공 프레임수, no-det=미검출(<6)프레임수, "
                 "det_mean=평균 검출 코너수(/8), cm_med=corner_med 중앙값, "
                 "cm<10px=corner_med<10px 프레임/유효프레임.")
    with open(os.path.join(OUT_ROOT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[save] {OUT_ROOT}")
    print(f"  index.txt, summary.md + {list(DOMAINS)} 하위폴더")
    for dom in DOMAINS:
        s = summ[dom]
        print(f"  {dom}: N={s['n']} det>=6={s['det_ge6']} "
              f"cm_med={s['cm_med']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gt", "raw"], default="raw",
                    help="gt=GT 있는 dev 셋(기존) / raw=GT 없는 raw 프레임 전체")
    args = ap.parse_args()
    if args.mode == "raw":
        main_raw()
    else:
        main()
