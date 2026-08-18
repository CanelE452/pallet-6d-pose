"""FT 판정 — runs_ft/PURPOSE.md 의 지표 3종 중 계산 가능한 둘을 base 와 나란히 잰다.

지표 1 (주지표) FP율 : forklift_raw 의 팔레트 없는 259 프레임에서 conf>=thr 검출이
                      나오는 비율. 낮을수록 좋다. negative 는 학습에 들어갔으므로
                      이 수치는 "학습한 것을 다시 맞히는" in-sample 이다 — 과신 금지.
                      그래서 학습에 쓰지 않은 나머지 652 프레임의 검출률도 같이 낸다
                      (팔레트가 있는 구간이 대부분이라 여기선 검출이 유지돼야 정상).
지표 2 eval 정본     : 학습에서 제외한 유일한 셋. 검출률과 keypoint 오차가 base 대비
                      떨어지지 않아야 한다(회귀 감시).

지표 3(영상 육안)은 사람이 봐야 하므로 여기서 하지 않는다.

사용:
  python .../eval_ft_fp.py --weights <a.pt> [<b.pt> ...] --conf 0.4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "challenge/yolo_pose_one_model"
NEG_SEQ = REPO / "data/pallet/raw_data/outside/forklift_raw_20260528_163408/rgb"
PAD = 100

EVAL_DIRS = [
    "challenge/data/01_real/eval_canonical/_outside_eval_manual_gt",
    "challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt",
    "challenge/data/01_real/eval_canonical/capturepalletcad_manual_gt",
    "challenge/data/01_real/manual_gt/capturepallet07_manual_gt",
    "challenge/data/01_real/manual_gt/capturepallet09_manual_gt",
    "challenge/data/01_real/manual_gt/capturenight08_manual_gt",
    "challenge/data/01_real/manual_gt/capturenight09_manual_gt",
]


def predict_batch(model, paths, conf, bs=32):
    """각 이미지의 (max_conf, kps(9,2) or None). 패딩 좌표계 그대로 반환."""
    out = []
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        imgs = []
        keep = []
        for p in chunk:
            im = cv2.imread(str(p))
            if im is None:
                out.append((0.0, None))
                continue
            imgs.append(cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101))
            keep.append(p)
        if not imgs:
            continue
        for r in model.predict(imgs, verbose=False, conf=conf, imgsz=640):
            if r.boxes is None or not len(r.boxes):
                out.append((0.0, None))
                continue
            c = r.boxes.conf.cpu().numpy()
            b = int(np.argmax(c))
            out.append((float(c[b]), r.keypoints.xy.cpu().numpy()[b] - PAD))
    return out


def gt_kps(ann, eval_only=True):
    """정본 규칙: eval 은 objects[0].split == "eval" 인 프레임만(최상위 아님).
    이 필터를 빼면 train/무표시가 섞여 161 이 아니라 185 가 된다(실제로 겪음)."""
    try:
        o = json.load(open(ann, encoding="utf-8"))["objects"][0]
    except Exception:
        return None
    if eval_only and o.get("split") != "eval":
        return None
    proj = o.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None
    cen = o.get("projected_cuboid_centroid") or [-1.0, -1.0]
    return np.array([list(map(float, p)) for p in proj[:8]] + [list(map(float, cen))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--neg-json", default=str(OUT_ROOT / "runs_ft/forklift_raw_conf.json"))
    ap.add_argument("--neg-conf-max", type=float, default=0.20)
    args = ap.parse_args()

    rows = json.load(open(args.neg_json, encoding="utf-8"))
    neg_frames = {r["frame"] for r in rows if r["max_conf"] < args.neg_conf_max}
    all_frames = sorted(int(os.path.splitext(os.path.basename(p))[0])
                        for p in glob.glob(str(NEG_SEQ / "*.png")))
    neg_paths = [NEG_SEQ / f"{f:06d}.png" for f in sorted(neg_frames)]
    rest_paths = [NEG_SEQ / f"{f:06d}.png" for f in all_frames if f not in neg_frames]

    ev = []
    for d in EVAL_DIRS:
        for a in sorted(glob.glob(str(REPO / d / "*.json"))):
            g = gt_kps(a)
            p = os.path.splitext(a)[0] + ".png"
            if g is not None and os.path.exists(p):
                ev.append((p, g, os.path.basename(os.path.dirname(a))))

    print(f"negative(팔레트 없음, 육안검수) {len(neg_paths)} | 나머지 forklift "
          f"{len(rest_paths)} | eval 정본 {len(ev)}\n")
    print("※ negative 는 base 의 conf<0.2 로 후보를 뽑아 육안 확인한 셋이라, 단일 임계"
          " 0.4 로 FP율을 재면\n   base 가 정의상 0% 가 된다(순환). 그래서 임계를 훑어"
          " 곡선으로 비교한다.\n")
    hdr = (f"{'weights':<34}{'FP율':>8}{'나머지검출':>10}"
           f"{'eval검출':>9}{'kp med':>9}{'kp p90':>9}")
    print(hdr)
    print("-" * len(hdr.encode('utf-8').decode('utf-8')) if False else "-" * 80)

    THRS = [0.05, 0.10, 0.25, 0.40]
    results = {}
    for w in args.weights:
        m = YOLO(w)
        neg_c = [c for c, _ in predict_batch(m, neg_paths, 0.01)]
        rest_c = [c for c, _ in predict_batch(m, rest_paths, 0.01)]
        fp_curve = {t: sum(1 for c in neg_c if c >= t) / len(neg_c) for t in THRS}
        rest_curve = {t: sum(1 for c in rest_c if c >= t) / len(rest_c) for t in THRS}
        fp = sum(1 for c in neg_c if c >= args.conf)
        rest = sum(1 for c in rest_c if c >= args.conf)
        det, errs = 0, []
        for (p, g, _), (c, k) in zip(ev, predict_batch(m, [e[0] for e in ev], args.conf)):
            if c < args.conf or k is None:
                continue
            det += 1
            ok = [i for i in range(9) if not (g[i][0] == -1 and g[i][1] == -1)]
            if ok:
                errs.append(float(np.median(np.linalg.norm(k[ok] - g[ok], axis=1))))
        name = os.path.relpath(w, OUT_ROOT) if str(w).startswith(str(OUT_ROOT)) else w
        name = name[-33:]
        med = np.median(errs) if errs else float("nan")
        p90 = np.percentile(errs, 90) if errs else float("nan")
        print(f"{name:<34}{fp/len(neg_paths)*100:>7.1f}%{rest/len(rest_paths)*100:>9.1f}%"
              f"{det/len(ev)*100:>8.1f}%{med:>9.2f}{p90:>9.2f}")
        print(f"{'':>34}  FP율@thr " + "  ".join(f"{t:.2f}:{fp_curve[t]*100:5.1f}%" for t in THRS))
        print(f"{'':>34}  나머지@thr " + "  ".join(f"{t:.2f}:{rest_curve[t]*100:5.1f}%" for t in THRS))
        results[str(w)] = {"fp_curve": fp_curve, "rest_curve": rest_curve,
                           "fp_rate": fp / len(neg_paths), "fp_n": fp,
                           "rest_det": rest / len(rest_paths),
                           "eval_det": det / len(ev), "kp_med": med, "kp_p90": p90}

    print("\nFP율 = 팔레트 없는 프레임에서 검출이 난 비율(낮을수록 좋음, in-sample 주의)")
    print("나머지검출 = 학습에 안 쓴 652 프레임 검출률(팔레트 있는 구간이 대부분 -> 유지돼야 정상)")
    print("eval검출/kp = 학습에서 제외한 정본 161장 (회귀 감시)")
    json.dump(results, open(OUT_ROOT / "runs_ft/_eval_ft.json", "w", encoding="utf-8"),
              indent=2, default=float)


if __name__ == "__main__":
    main()
