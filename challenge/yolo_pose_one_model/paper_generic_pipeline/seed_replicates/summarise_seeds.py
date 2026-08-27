"""PHASE 8 — seed42/43/44 요약.  가장 좋은 seed 하나만 고르지 않는다."""
from __future__ import annotations
import json, os, sys
import numpy as np
ROOT = "/home/minjae/Documents/github/pallet-pose"
EVAL = os.path.join(ROOT, "challenge/yolo_pose_one_model/evaluation")
KEYS = [("availability", None), ("corner_px", "median"), ("corner_px", "p90"),
        ("R_deg", "median"), ("R_deg", "p90"), ("t_m", "median"), ("t_m", "p90"),
        ("ADD_S", "median"), ("IoU3D", "median"), ("success_5cm5deg", None)]


def main(seeds=(42, 43, 44)):
    got = {}
    for s in seeds:
        p = os.path.join(EVAL, f"PAPER_YOLO_REAL_DEV_RESULT_seed{s}.json")
        if s == 42:
            p = os.path.join(EVAL, "PAPER_YOLO_REAL_DEV_RESULT.json")
        if os.path.exists(p):
            got[s] = json.load(open(p))["models"][
                f"yolo26n_paper_generic_v1"]
    if len(got) < 2:
        print(f"  seed 결과 {len(got)} 개뿐 — 요약하지 않는다 "
              f"(3 seed 전부 나온 뒤 보고한다)")
        return
    out = {}
    for pop in ("OPEN_56", "REAL_CHALLENGE_DEV_105"):
        block = {}
        for key, sub in KEYS:
            vals = []
            for s, m in got.items():
                v = m[pop][key]
                vals.append(v[sub] if sub and isinstance(v, dict) else v)
            vals = [v for v in vals if v is not None]
            name = f"{key}.{sub}" if sub else key
            block[name] = {"per_seed": dict(zip(got, vals)),
                           "mean": round(float(np.mean(vals)), 4),
                           "std": round(float(np.std(vals, ddof=1)), 4)
                           if len(vals) > 1 else None,
                           "min": round(float(np.min(vals)), 4),
                           "max": round(float(np.max(vals)), 4)}
        out[pop] = block
    out["note"] = ("세 seed 전체를 논문에 보고한다. DEV 결과를 보고 가장 좋은 "
                   "seed 하나만 고르지 않는다.")
    json.dump(out, open(os.path.join(EVAL, "PAPER_YOLO_SEED_SUMMARY.json"), "w"),
              indent=1)
    print(f"  seed {sorted(got)} 요약 -> PAPER_YOLO_SEED_SUMMARY.json")


if __name__ == "__main__":
    main()
