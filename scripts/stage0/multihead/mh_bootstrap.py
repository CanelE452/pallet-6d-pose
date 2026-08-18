"""PHASE 11 -- paired frame bootstrap for every comparison in the diagnosis.

A median is one number and says nothing about whether the difference between two
arms would survive a different draw of frames.  Every comparison here is paired:
the same frame contributes to both arms in every resample, so the frame-to-frame
spread -- which is enormous, from easy full-view boxes to badly truncated ones --
cancels instead of drowning the effect.

Seeds are never pooled.  Two seeds is n=2 and stays n=2; the bootstrap is over
frames within a seed, and the seeds are reported side by side.  Pooling them
would turn n=2 into a confident-looking n=1024 that is not there.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_data as MD                                            # noqa: E402

OUT = MD.OUT
RESAMPLES = 10_000
SEEDS = (1, 2)
RNG_SEED = 20260817


def _paired_median_ratio(a, b, resamples=RESAMPLES, seed=RNG_SEED):
    """Percent improvement of b over a, with a paired frame bootstrap CI.

    Positive means b is better (smaller).  Frames are resampled with replacement
    as pairs; the statistic is recomputed from the resampled medians rather than
    averaged, because a ratio of medians is not the median of ratios.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 8:
        return {"n": int(len(a))}
    point = 100.0 * (np.median(a) - np.median(b)) / abs(np.median(a))
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(a), size=(resamples, len(a)))
    ma = np.median(a[index], axis=1)
    mb = np.median(b[index], axis=1)
    draws = 100.0 * (ma - mb) / np.abs(ma)
    return {"n": int(len(a)),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "improvement_pct": float(point),
            "ci95_low": float(np.percentile(draws, 2.5)),
            "ci95_high": float(np.percentile(draws, 97.5)),
            "p_b_better": float((draws > 0).mean())}


def _rows(arm, seed, label="long25k", step="25000"):
    path = OUT / f"mh_screen_{arm}_{label}_seed{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())[step]["D2_MH_DEV512"]["rows"]


def _per_frame(rows, metric):
    """One number per frame, so the pairing is by frame not by role."""
    out = []
    for row in rows:
        if metric == "angle":
            values = [a for a, s in zip(row["angle"], row["support"]) if s]
        elif metric == "offset":
            values = [o for o, s in zip(row["offset"], row["support"]) if s]
        elif metric == "cigm":
            values = row["cigm_cell"]
        elif metric == "direct":
            values = row.get("direct_cell", [])
        elif metric == "R_C":
            block = row.get("pose_C")
            values = [block["R_deg"]] if block and block.get("solved") else []
        elif metric == "R_L":
            block = row.get("pose_L")
            values = [block["R_deg"]] if block and block.get("solved") else []
        else:
            raise KeyError(metric)
        out.append(float(np.median(values)) if values else np.nan)
    return np.asarray(out)


def _align(rows_a, rows_b):
    """Compare only frames both arms scored, matched by stem."""
    index_b = {row["stem"]: row for row in rows_b}
    pairs = [(a, index_b[a["stem"]]) for a in rows_a if a["stem"] in index_b]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def screen_comparisons():
    result = {}
    pairs = (("A1_vs_A0", "A0_LINE_ONLY", "A1_CORNER_LINE"),
             ("A2_vs_A1", "A1_CORNER_LINE", "A2_CORNER_LINE_MASK"),
             ("A2_vs_A0", "A0_LINE_ONLY", "A2_CORNER_LINE_MASK"))
    metrics = ("angle", "offset", "cigm", "direct", "R_C", "R_L")
    for name, base, candidate in pairs:
        for seed in SEEDS:
            rows_a, rows_b = _rows(base, seed), _rows(candidate, seed)
            if rows_a is None or rows_b is None:
                continue
            rows_a, rows_b = _align(rows_a, rows_b)
            for metric in metrics:
                a, b = _per_frame(rows_a, metric), _per_frame(rows_b, metric)
                if not np.isfinite(a).any() or not np.isfinite(b).any():
                    continue
                result[f"{name}|seed{seed}|{metric}"] = _paired_median_ratio(a, b)
    return result


def stopgrad_comparisons():
    arms = ("E0_CONTINUE_LINE", "E1_SHARED_CORNER_LINE", "E2_STOPGRAD_CORNER")
    loaded = {}
    for seed in SEEDS:
        for arm in arms:
            path = OUT / f"stopgrad_{arm}_seed{seed}.json"
            if path.exists():
                history = json.loads(path.read_text())
                marks = [k for k in history if k.isdigit()]
                if marks:
                    last = max(marks, key=int)
                    block = history[last].get("D2_MH_DEV512", {})
                    if "rows" in block:
                        loaded[(arm, seed)] = (last, block["rows"])
    result = {}
    for name, base, candidate in (("E1_vs_E0", arms[0], arms[1]),
                                  ("E2_vs_E1", arms[1], arms[2]),
                                  ("E2_vs_E0", arms[0], arms[2])):
        for seed in SEEDS:
            if (base, seed) not in loaded or (candidate, seed) not in loaded:
                continue
            step_a, rows_a = loaded[(base, seed)]
            step_b, rows_b = loaded[(candidate, seed)]
            if step_a != step_b:
                continue
            rows_a, rows_b = _align(rows_a, rows_b)
            for metric in ("angle", "offset", "cigm", "direct"):
                a, b = _per_frame(rows_a, metric), _per_frame(rows_b, metric)
                if not np.isfinite(a).any() or not np.isfinite(b).any():
                    continue
                result[f"{name}|seed{seed}|{metric}|step{step_a}"] = \
                    _paired_median_ratio(a, b)
    return result


def _print(title, block):
    if not block:
        print(f"\n{title}: nothing to compare yet")
        return
    print(f"\n{title}   (positive = second arm better, 10,000 paired resamples)")
    print(f"{'comparison':<38}{'n':>6}{'a med':>9}{'b med':>9}"
          f"{'improve':>10}{'95% CI':>20}{'P(better)':>11}")
    for key, value in block.items():
        if "improvement_pct" not in value:
            continue
        ci = f"[{value['ci95_low']:+.2f}, {value['ci95_high']:+.2f}]"
        print(f"{key:<38}{value['n']:>6}{value['median_a']:>9.4f}"
              f"{value['median_b']:>9.4f}{value['improvement_pct']:>+9.2f}%"
              f"{ci:>20}{value['p_b_better']:>11.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.parse_args()
    result = {"resamples": RESAMPLES, "pairing": "by frame, seeds never pooled",
              "screen": screen_comparisons(), "stopgrad": stopgrad_comparisons()}
    (OUT / "mh_bootstrap.json").write_text(json.dumps(result, indent=1))
    _print("screen @25,000", result["screen"])
    _print("stop-grad continuation", result["stopgrad"])
    print(f"\n-> {OUT / 'mh_bootstrap.json'}")


if __name__ == "__main__":
    main()
