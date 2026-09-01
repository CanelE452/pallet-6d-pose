"""같은 데이터로 학습한 DOPE vs YOLO26n — 정본에서 대략 비교.

두 모델은 **같은 55,980 프레임**으로 60ep 학습했다.  그래서 차이를 백본 탓으로 읽을 수
있다 (데이터·epoch·real 감독 0 이 모두 같다).

★내는 지표를 고른 이유
    검출률 · corner px      GT 의 2D keypoint 만 쓴다 -> axis leak 무관, 안전
    R/t/AUC/5cm5           ★내지 않는다.  PnP 3D 모델을 GT `dimensions_m` 에서
                           만들기 때문에 평가가 90도 yaw 구분을 대신 풀어준다
                           (memory: evaluator-receives-gt-per-frame-axis-assignment,
                            REPRESENTATION_BLOCK=True).  이 상태의 pose 수치는
                           모델 비교 근거로 못 쓴다.

corner 오차는 order-free Hungarian median 이다 — 기존 정의를 그대로 쓰고 새로
만들지 않는다.

env 가 갈리므로 `--part` 로 나눠 두 번 돌린 뒤 `--part report` 로 합친다.
    DOPE   pallet-pose
    YOLO   pallet-yolo26
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/scripts/stage0")
sys.path.insert(0, f"{ROOT}/scripts/stage0/paper_s2")
sys.path.insert(0, f"{ROOT}/scripts/annotate")

PAD = 100
OUT = f"{ROOT}/data/pallet/eval_results/dope_vs_yolo_canonical"
DOPE_W = f"{ROOT}/weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth"
YOLO_W = (f"{ROOT}/challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
          "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")
N_DET_MIN = 6


def canonical():
    spec = importlib.util.spec_from_file_location(
        "dp", f"{ROOT}/challenge/data_paths.py")
    dp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dp)
    out = []
    for name, d in dp.EVAL_CANONICAL.items():
        for jp in sorted(glob.glob(os.path.join(str(d), "*.json"))):
            ip = jp[:-5] + ".png"
            if os.path.exists(ip):
                out.append((name, jp, ip))
    return out


def gt8(jp):
    o = json.load(open(jp))["objects"][0]
    if o.get("split") != "eval":
        return None
    k = o.get("projected_cuboid")
    return np.asarray(k[:8], float) if k and len(k) >= 8 else None


def hungarian_median(pred, gt):
    from scipy.optimize import linear_sum_assignment
    ok = np.isfinite(pred[:, 0])
    if ok.sum() < 4:
        return float("inf")
    c = np.linalg.norm(pred[ok][:, None, :] - gt[None, :, :], axis=2)
    r, cc = linear_sum_assignment(c)
    return float(np.median(c[r, cc]))


def run_dope(frames):
    import cv2
    import paper_s2_testset17_9filters as T
    E, M = T.E, T.M
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = E.load_model(DOPE_W, dev)
    rows = []
    for sess, jp, ip in frames:
        g = gt8(jp)
        if g is None:
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        belief, geom, wh = M.infer_belief(model, img, dev, PAD)
        pred8, _, peaks, _ = M.belief_to_pred(belief, geom, wh, PAD, T.THRESH)
        n = int((~np.isnan(pred8[:, 0])).sum())
        det = [i for i in range(8) if not np.isnan(pred8[i, 0])]
        rows.append({"sess": sess, "fid": os.path.basename(jp)[:-5],
                     "n_det": n, "kp_err": hungarian_median(pred8, g),
                     "conf": float(max(peaks[i] for i in det)) if det else 0.0})
        if len(rows) % 40 == 0:
            print(f"  dope {len(rows)}…", flush=True)
    return rows


def run_yolo(frames):
    import cv2
    from ultralytics import YOLO
    model = YOLO(YOLO_W, task="pose")
    rows = []
    for sess, jp, ip in frames:
        g = gt8(jp)
        if g is None:
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        inp = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        r = model.predict(inp, verbose=False, conf=0.25, imgsz=640)[0]
        pred8, conf = np.full((8, 2), np.nan), 0.0
        if r.boxes is not None and len(r.boxes):
            b = int(np.argmax(r.boxes.conf.cpu().numpy()))
            conf = float(r.boxes.conf.cpu().numpy()[b])
            kp = r.keypoints.data.cpu().numpy()[b].copy()
            kp[:, 0] -= PAD
            kp[:, 1] -= PAD
            for i in range(8):
                if kp[i, 2] >= 0.5:
                    pred8[i] = kp[i, :2]
        rows.append({"sess": sess, "fid": os.path.basename(jp)[:-5],
                     "n_det": int(np.isfinite(pred8[:, 0]).sum()),
                     "kp_err": hungarian_median(pred8, g), "conf": conf})
        if len(rows) % 40 == 0:
            print(f"  yolo {len(rows)}…", flush=True)
    return rows


def summarize(rows):
    n = len(rows)
    det = np.array([r["n_det"] for r in rows])
    err = np.array([r["kp_err"] for r in rows], float)
    fin = np.isfinite(err)
    return {
        "n": n,
        "det8_rate": float((det == 8).mean()),
        "det6plus_rate": float((det >= N_DET_MIN).mean()),
        "mean_n_det": float(det.mean()),
        "corner_median": float(np.median(err[fin])) if fin.any() else None,
        "corner_p90": float(np.percentile(err[fin], 90)) if fin.any() else None,
        "corner_le10_rate": float((err <= 10).mean()),
        "corner_le5_rate": float((err <= 5).mean()),
    }


def report():
    d = json.load(open(f"{OUT}/rows_dope.json"))["rows"]
    y = json.load(open(f"{OUT}/rows_yolo.json"))["rows"]
    dm = {r["fid"]: r for r in d}
    ym = {r["fid"]: r for r in y}
    common = [f for f in dm if f in ym]
    d = [dm[f] for f in common]
    y = [ym[f] for f in common]
    sd, sy = summarize(d), summarize(y)

    print(f"\n공통 프레임 {len(common)}장 (정본, split=='eval')")
    print(f"학습 데이터 동일 55,980 · 60ep · real 감독 0\n")
    print(f"{'지표':26}{'DOPE':>12}{'YOLO26n':>12}{'우세':>8}")
    print("-" * 58)
    rows_ = [
        ("검출 8/8 비율", "det8_rate", "up", "{:.3f}"),
        ("검출 >=6 비율", "det6plus_rate", "up", "{:.3f}"),
        ("평균 검출 코너수", "mean_n_det", "up", "{:.2f}"),
        ("corner median (px)", "corner_median", "down", "{:.2f}"),
        ("corner p90 (px)", "corner_p90", "down", "{:.2f}"),
        ("corner <=10px 비율", "corner_le10_rate", "up", "{:.3f}"),
        ("corner <=5px 비율", "corner_le5_rate", "up", "{:.3f}"),
    ]
    for lab, k, dirn, fmt in rows_:
        a, b = sd[k], sy[k]
        if a is None or b is None:
            win = "-"
        else:
            win = ("YOLO" if b > a else "DOPE") if dirn == "up" else \
                  ("YOLO" if b < a else "DOPE")
        print(f"{lab:26}{fmt.format(a):>12}{fmt.format(b):>12}{win:>8}")

    print("\n세션별 corner median (px)")
    print(f"{'session':22}{'n':>5}{'DOPE':>10}{'YOLO26n':>10}")
    print("-" * 47)
    for s in sorted({r["sess"] for r in d}):
        di = [r["kp_err"] for r in d if r["sess"] == s and np.isfinite(r["kp_err"])]
        yi = [r["kp_err"] for r in y if r["sess"] == s and np.isfinite(r["kp_err"])]
        ns = sum(1 for r in d if r["sess"] == s)
        print(f"{s:22}{ns:>5}"
              f"{(np.median(di) if di else float('nan')):>10.2f}"
              f"{(np.median(yi) if yi else float('nan')):>10.2f}")

    print("\n★R/t/AUC/5cm5 는 내지 않았다 — PnP 3D 모델을 GT dimensions_m 에서")
    print("  만들어 평가가 축 배정을 대신 풀어준다 (REPRESENTATION_BLOCK).")
    json.dump({"n_common": len(common), "dope": sd, "yolo": sy},
              open(f"{OUT}/SUMMARY.json", "w"), indent=1)
    print(f"\n-> {OUT}/SUMMARY.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["dope", "yolo", "report"], required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.part == "report":
        return report()
    frames = canonical()
    print(f"정본 스캔 {len(frames)}장", flush=True)
    rows = run_dope(frames) if a.part == "dope" else run_yolo(frames)
    w = DOPE_W if a.part == "dope" else YOLO_W
    json.dump({"weights": os.path.relpath(w, ROOT), "rows": rows},
              open(f"{OUT}/rows_{a.part}.json", "w"), indent=1)
    print(f"  {a.part}: {len(rows)}장 -> {OUT}/rows_{a.part}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
