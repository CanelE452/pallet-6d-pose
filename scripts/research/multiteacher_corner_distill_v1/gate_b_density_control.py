"""GATE B 부속 진단 — 후보 근접률이 '의미' 인가 '밀도' 인가.

patch 당 후보가 48 개면 반경 12 (약 625 px^2) 안에서 무작위로 뿌려도
GT 3px 이내에 하나쯤은 들어간다.  같은 patch·같은 후보 수로 균등난수를 뿌려
근접률을 다시 재고, 고전 CV 후보가 그보다 나은지 본다.

임계를 바꾸지 않는다.  Gate B 판정을 어떻게 읽어야 하는지만 정한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
from gate_b_corner_evidence import FAMILIES, KEY_MAP, REF


def main() -> int:
    lock = json.loads(M.METHOD_LOCK_PATH.read_text())
    radius = int(lock["gate_b"]["search_radius_px"])
    cfg = lock["gate_b"]["candidate_generators_frozen"]
    rng = np.random.default_rng(20260905)

    gts = [M.load_gt(f) for f in M.dev_eval_frames()]
    preds = M.load_prediction_file(M.PREDICTIONS / f"{REF}.json")

    real = {3: 0, 5: 0, 10: 0}
    rand = {3: 0, 5: 0, 10: 0}
    real_min, rand_min, counts, total = [], [], [], 0
    for gt in gts:
        pred = M.prediction_keypoints(preds.get(gt["frame_id"]))
        if pred is None:
            continue
        image = cv2.imread(str(M.REPO_ROOT / gt["image"]))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        for k in range(8):
            if not gt["visible"][k] or not np.isfinite(pred[k]).all():
                continue
            cx, cy = int(round(pred[k][0])), int(round(pred[k][1]))
            x0, y0 = max(0, cx - radius), max(0, cy - radius)
            x1, y1 = min(w, cx + radius + 1), min(h, cy + radius + 1)
            if x1 - x0 < 5 or y1 - y0 < 5:
                continue
            patch = gray[y0:y1, x0:x1]
            pts = []
            for family, func in FAMILIES.items():
                try:
                    pts += list(func(patch, **cfg[KEY_MAP[family]]))
                except Exception:
                    pass
            n = len(pts)
            total += 1
            counts.append(n)
            if n == 0:
                continue
            P = np.asarray(pts, float) + np.array([x0, y0], float)
            dr = float(np.linalg.norm(P - gt["xy"][k], axis=1).min())
            real_min.append(dr)
            Q = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])
            dq = float(np.linalg.norm(Q - gt["xy"][k], axis=1).min())
            rand_min.append(dq)
            for thr in (3, 5, 10):
                real[thr] += dr <= thr
                rand[thr] += dq <= thr

    n = len(real_min)
    report = {
        "n_patches": total, "n_with_candidates": n,
        "candidates_per_patch": {"median": float(np.median(counts)),
                                 "mean": float(np.mean(counts)),
                                 "p95": float(np.percentile(counts, 95))},
        "coverage_classical": {str(t): real[t] / n for t in (3, 5, 10)},
        "coverage_uniform_random_same_count": {str(t): rand[t] / n for t in (3, 5, 10)},
        "lift": {str(t): (real[t] / n) - (rand[t] / n) for t in (3, 5, 10)},
        "min_distance_median_px": {"classical": float(np.median(real_min)),
                                   "uniform_random": float(np.median(rand_min))},
        "interpretation_rule":
            "lift 가 작으면 근접률은 semantic corner 검출이 아니라 후보 밀도의 산물이다. "
            "그 경우 Gate B 의 oracle headroom 은 '국소 RGB 에 코너 증거가 있다' 가 아니라 "
            "'반경 12 안을 촘촘히 뒤지면 GT 근처 점이 있다' 로 읽어야 한다.",
    }
    out = M.GATE_B / "GATE_B_DENSITY_CONTROL.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"patch {total}  후보/patch 중앙 {report['candidates_per_patch']['median']:.0f}  "
          f"평균 {report['candidates_per_patch']['mean']:.1f}")
    print(f"{'thr':>5}{'고전 CV':>10}{'균등난수':>10}{'lift':>9}")
    for t in (3, 5, 10):
        print(f"{t:5d}{report['coverage_classical'][str(t)]:10.3f}"
              f"{report['coverage_uniform_random_same_count'][str(t)]:10.3f}"
              f"{report['lift'][str(t)]:+9.3f}")
    print(f"최근접 거리 중앙값  고전 {report['min_distance_median_px']['classical']:.2f} px  "
          f"난수 {report['min_distance_median_px']['uniform_random']:.2f} px")
    print(f"-> {out.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
