"""P2 vs P1 at a fixed operating point, paired over the same negative images.

Uses the same constraint-exact threshold rule as mh_presence_curve.point, so
the comparison is made where the frozen protocol actually operates.
"""
from __future__ import annotations

import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_data as MD             # noqa: E402
import mh_presence as PR         # noqa: E402
import mh_presence_curve as PC   # noqa: E402

OUT = MD.OUT
B = 10000
SEED = 20260902


def main():
    rng = np.random.default_rng(SEED)
    out = {"note": "paired bootstrap over the SAME negative dev images",
           "B": B, "threshold_rule": "constraint-exact, see mh_presence_curve",
           "seeds": {}}
    for seed in (1, 2):
        c = np.load(OUT / f"presence_z_cache_seed{seed}.npz", allow_pickle=True)
        layer, _ = PR.fit_linear(c["pos_train"], c["neg_train"], seed)
        arms = {"P1": (c["pos_dev"][:, 3], c["neg_dev"][:, 3]),
                "P2": (PR.apply_linear(layer, c["pos_dev"]),
                       PR.apply_linear(layer, c["neg_dev"]))}
        block = {}
        for target in PC.TARGETS:
            thr = {k: PC.point(p, n, target)["threshold"]
                   for k, (p, n) in arms.items()}
            f1 = (arms["P1"][1] >= thr["P1"]).astype(float)
            f2 = (arms["P2"][1] >= thr["P2"]).astype(float)
            idx = rng.integers(0, len(f1), (B, len(f1)))
            d = f2[idx].mean(1) - f1[idx].mean(1)   # negative = P2 better
            lo, hi = (float(v) for v in np.quantile(d, [0.025, 0.975]))
            block[f"recall_{target}"] = {
                "P1_fp": round(float(f1.mean()), 5),
                "P2_fp": round(float(f2.mean()), 5),
                "delta_P2_minus_P1": round(float(d.mean()), 5),
                "CI95": [round(lo, 5), round(hi, 5)],
                "P2_better_excludes_zero": bool(hi < 0)}
            print(f"  seed{seed} recall{target}  P1 {f1.mean():.4f}  "
                  f"P2 {f2.mean():.4f}  delta {d.mean():+.4f}  "
                  f"CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"{'P2 better' if hi < 0 else '0 포함 -- 미확립'}", flush=True)
        out["seeds"][f"seed{seed}"] = block
    (OUT / "presence_p2_vs_p1_bootstrap.json").write_text(json.dumps(out, indent=1))
    print("-> presence_p2_vs_p1_bootstrap.json")


if __name__ == "__main__":
    main()
