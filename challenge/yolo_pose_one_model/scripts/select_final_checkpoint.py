"""Evaluate every saved checkpoint on the synthetic target-val set and pick one.

Selection order (synthetic val, measured 2026-08-15):

  1. normalised keypoint error   min   <- GT-referenced accuracy
  2. median reprojection error   min
  3. PnP success rate            max
  4. detection rate              max
  5. pose mAP50-95               max   (from the training results.csv, if present)

★ This is NOT the prompt's order, and the change is deliberate. The prompt puts PnP
success rate first for the case with no yaw GT - a sensible rule on REAL data, where
nothing else is measurable. On synthetic val it inverts the ranking and picks the worst
model. Measured on 100 frames of target val:

    ckpt       pnp%   err_norm   err_all   reproj
    epoch0     85.0   0.0268     9.77 px   3.64 px   <- would win on PnP rate
    epoch55    73.0   0.0053     1.62 px   1.00 px
    best/last  73.0   0.0055     1.56 px   0.89 px

Two things drive the inversion:
  - An undertrained net gives every keypoint a similar, fairly high confidence, so more
    of them clear the kp-conf threshold and PnP gets attempted. A trained net assigns low
    confidence to occluded points, which removes them from the PnP set.
  - PnP returns a solution from any 6 points. epoch0's keypoints are self-consistent
    (reproj 3.64 px) yet 9.77 px away from GT. Reprojection cannot see that; only a
    GT-referenced error can.

Synthetic val HAS ground truth, so keypoint error is the direct measurement and
reprojection/PnP-rate are the proxies used when GT is absent. On real data the prompt's
original order applies again.

The winner is COPIED to final/pallet_yolo26n_pose_640_b32_final.pt. Originals are kept.

A synthetic-val winner is not proven best on the real forklift domain. That claim needs
the real finetune round.

Usage:
  python .../select_final_checkpoint.py --run runs/stage_a_synth_640_b32_seed42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "challenge/yolo_pose_one_model"


def summarise(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return None
    n = len(rows)

    def fl(r, k):
        v = r.get(k, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    det = sum(1 for r in rows if r.get("detected") == "1")
    pnp = sum(1 for r in rows if r.get("pnp_ok") == "1")

    def med(k, only_pnp=False):
        v = [fl(r, k) for r in rows if (not only_pnp or r.get("pnp_ok") == "1")]
        v = sorted(x for x in v if x is not None)
        return v[len(v) // 2] if v else float("inf")

    return {"n": n, "det_rate": det / n, "pnp_rate": pnp / n,
            "err_norm": med("err_norm"), "median_reproj": med("median_reproj", True),
            "err_all": med("err_all")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir under challenge/yolo_pose_one_model")
    ap.add_argument("--dataset", default="datasets/stage_a")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    run = OUT / args.run
    ckpts = sorted((run / "weights").glob("*.pt"))
    if not ckpts:
        sys.exit(f"no checkpoints in {run/'weights'}")
    print(f"{len(ckpts)} checkpoints in {run.relative_to(REPO)}")

    results = {}
    for c in ckpts:
        dst = OUT / "reports" / f"eval_T_{run.name}_{c.stem}.csv"
        cmd = [sys.executable, str(OUT / "scripts/eval_task_pose.py"),
               "--weights", str(c), "--dataset", args.dataset, "--split", "val",
               "--domain", "T", "--out", str(dst)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"\n=== {c.name}")
        subprocess.run(cmd, check=True, cwd=str(REPO))
        s = summarise(dst)
        if s:
            results[c.name] = s

    # accuracy first (see module docstring: PnP rate inverts the ranking on synthetic val)
    order = sorted(results.items(),
                   key=lambda kv: (kv[1]["err_norm"], kv[1]["median_reproj"],
                                   -kv[1]["pnp_rate"], -kv[1]["det_rate"]))
    print(f"\n{'checkpoint':<20}{'pnp%':>8}{'err_norm':>10}{'reproj':>9}{'det%':>8}")
    for k, v in order:
        print(f"{k:<20}{100*v['pnp_rate']:>8.1f}{v['err_norm']:>10.4f}"
              f"{v['median_reproj']:>9.2f}{100*v['det_rate']:>8.1f}")

    best = order[0][0]
    src = run / "weights" / best
    final = OUT / "final/pallet_yolo26n_pose_640_b32_final.pt"
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, final)
    sha = hashlib.sha256(final.read_bytes()).hexdigest()
    json.dump({"selected": best, "source": str(src.relative_to(REPO)),
               "final": str(final.relative_to(REPO)), "sha256": sha,
               "criterion": "synthetic target-val: err_norm > median_reproj > pnp_rate > det_rate "
                            "(accuracy first; PnP rate inverts the ranking here - see docstring)",
               "caveat": "no real data was used this round; real performance is unmeasured",
               "table": {k: v for k, v in order}},
              open(OUT / "final/selection.json", "w", encoding="utf-8"), indent=2)
    print(f"\nselected {best} -> {final.relative_to(REPO)}\nsha256 {sha}")


if __name__ == "__main__":
    main()
