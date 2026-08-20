"""Reporting only -- the gate in mh_presence_verdict.py is not touched.

The brief fixes the threshold by "among thresholds with Recall >= 95%, minimise
negative FP/image", and separately asks the gate for "Recall drop <= 2pp".
Minimising FP always walks the threshold down to the 95% floor, so the two
clauses can only hold together when FP/image is already tied there.  This file
reports what FP/image actually is at the gate's own recall (>= 98%), so the
inconsistency can be judged on numbers rather than on argument.
"""
from __future__ import annotations

import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_data as MD     # noqa: E402
import mh_presence as PR  # noqa: E402

OUT = MD.OUT
TARGETS = (0.98, 0.95)


def point(pos, neg, target):
    """Largest threshold whose realised recall still satisfies the constraint.

    np.quantile lands between samples, so it returned recall 0.97997 for a
    0.98 target -- a 2.003pp drop, which fails a "<= 2pp" constraint by
    discretisation rather than by performance.  Selecting from the observed
    positive scores makes the constraint exact.  This is the FP-minimising
    choice subject to `recall >= target`, so it is the frozen rule, and it is
    the conservative direction: the threshold moves down and FP goes up.
    """
    ordered = np.sort(pos)                      # ascending
    misses = int(np.floor(len(pos) * (1.0 - target)))
    thr = float(ordered[misses]) if misses < len(ordered) else float(ordered[-1])
    recall = float((pos >= thr).mean())
    while recall < target and misses > 0:       # ties can push recall under
        misses -= 1
        thr = float(ordered[misses])
        recall = float((pos >= thr).mean())
    return {"recall_target": target,
            "threshold": thr,
            "recall": round(recall, 5),
            "fp_per_image": round(float((neg >= thr).mean()), 5)}


def main():
    report = {"note": "reporting only; gate unchanged", "seeds": {}}
    for seed in (1, 2):
        c = np.load(OUT / f"presence_z_cache_seed{seed}.npz", allow_pickle=True)
        layer, _ = PR.fit_linear(c["pos_train"], c["neg_train"], seed)
        arms = {
            "P1_SCORE4KP": (c["pos_dev"][:, 3], c["neg_dev"][:, 3]),
            "P2_DETACHED_LINEAR": (PR.apply_linear(layer, c["pos_dev"]),
                                   PR.apply_linear(layer, c["neg_dev"])),
        }
        block = {}
        for arm, (pos, neg) in arms.items():
            block[arm] = [point(pos, neg, t) for t in TARGETS]
            for e in block[arm]:
                red = 100.0 * (1.0 - e["fp_per_image"])
                drop = 100.0 * (1.0 - e["recall"])
                e["fp_reduction_pct"] = round(red, 2)
                e["recall_drop_pp"] = round(drop, 3)
                print(f"  seed{seed} {arm:<20} recall {e['recall']:.4f} "
                      f"(drop {drop:+.2f}pp)  FP/img {e['fp_per_image']:.4f} "
                      f"({red:+.1f}%)", flush=True)
        report["seeds"][f"seed{seed}"] = block
    (OUT / "presence_recall_fp_curve.json").write_text(json.dumps(report, indent=1))
    print("-> presence_recall_fp_curve.json")


if __name__ == "__main__":
    main()
