"""PHASE 1-2 -- learning-curve and parameter-drift audit.  No training.

Evaluates every surviving checkpoint mark of C0 and C1_RESCUE on the same
canonical cells, so the question "does the Y30 enrichment move the model in a
consistent direction" is asked along the whole continuation, not only at 3,000.

NON_LA is subsampled to a fixed 600 frames -- it is 5,753 of the 6,242 dev
frames and would dominate the cost while being the cell least in question.  The
three LA cells are used whole.  The subsample is drawn once, seeded, and written
to the manifest before any metric is read.

Classification is defined here, before the curves exist:

    LC_A  the two seeds disagree in direction over most of the trajectory
    LC_B  they agree early and one or both reverse late
    LC_C  neither -- the trajectory oscillates around zero
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_cigm as CG            # noqa: E402
import mh_curriculum as CU      # noqa: E402
import mh_curriculum_report as CR  # noqa: E402
import mh_data as MD            # noqa: E402
import mh_diagnose as DG        # noqa: E402
import mh_r0r1 as RR            # noqa: E402
import mh_regime as RG          # noqa: E402
import mh_screen as MS          # noqa: E402
import mh_splitlate as SL       # noqa: E402

OUT = MD.OUT
SEEDS = (1, 2)
ARMS = ("C0", "C1_RESCUE")
MARKS = (0, 250, 500, 1000, 2000, 3000)
NON_LA_SAMPLE = 600
SAMPLE_SEED = 20260829
# improvement convention: positive = RESCUE better than C0 (error went down)
EARLY = (250, 500, 1000)
LATE = (2000, 3000)


def log(m):
    print(m, flush=True)


def population():
    path = OUT / "lcurve_population_manifest.json"
    meta = RR.dev_index()
    if path.exists():
        keep = json.loads(path.read_text())["stems"]
        return [(MD.DATA, s) for s in keep], meta
    by = {}
    for s, v in meta.items():
        by.setdefault(v["cell"], []).append(s)
    rng = np.random.default_rng(SAMPLE_SEED)
    keep = sorted(by["LA_FRONTAL"] + by["LA_EASY"] + by["LA_HARD"])
    non = sorted(by["NON_LA"])
    keep += sorted(rng.choice(non, size=NON_LA_SAMPLE, replace=False).tolist())
    keep = sorted(keep)
    path.write_text(json.dumps(
        {"n": len(keep), "non_la_sample": NON_LA_SAMPLE,
         "sample_seed": SAMPLE_SEED,
         "by_cell": {c: sum(1 for s in keep if meta[s]["cell"] == c)
                     for c in RR.CELLS},
         "stems": keep}, indent=1))
    return [(MD.DATA, s) for s in keep], meta


def load(arm, seed, step):
    p = MS.CKPT / f"curriculum_{arm}_seed{seed}" / f"step_{step:05d}.pth"
    if not p.exists():
        return None, None
    state = torch.load(p, map_location=MD.DEV, weights_only=False)
    model = SL.SplitLate("A1_CORNER_LINE")
    model.load_state_dict(state["model"])
    model.to(MD.DEV).eval()
    return model, state["model"]


def drift(a, b, prefix):
    keys = [k for k in a if k.startswith(prefix)]
    if not keys:
        return None
    d = torch.cat([(a[k] - b[k]).reshape(-1).float() for k in keys])
    base = torch.cat([b[k].reshape(-1).float() for k in keys])
    return {"l2": round(float(d.norm()), 6),
            "relative_l2": round(float(d.norm() / base.norm().clamp(min=1e-12)), 8),
            "max_abs": float(d.abs().max())}


def run(_a):
    MS.deterministic()
    items, meta = population()
    log(f"population {len(items)}  " + str(
        {c: sum(1 for _, s in items if meta[s]['cell'] == c) for c in RR.CELLS}))
    curves, drifts = {}, {}
    for seed in SEEDS:
        prev_dir = {}
        for step in MARKS:
            per_arm, raw = {}, {}
            for arm in ARMS:
                model, sd = load(arm, seed, step)
                if model is None:
                    log(f"  missing {arm} seed{seed} @{step}")
                    continue
                rows = RR.measure(model, items, meta)
                per_arm[arm] = {c: RR.agg([r for r in rows if r["cell"] == c])
                                for c in RR.CELLS}
                raw[arm] = sd
            if len(per_arm) < 2:
                continue
            for c in RR.CELLS:
                a, b = per_arm["C0"][c], per_arm["C1_RESCUE"][c]
                for m in ("obs_rms", "front_rear_shift", "affine_scale_gap",
                          "centroid_shift", "nonaffine_rms", "R_median",
                          "R_p90", "t_median", "t_p90"):
                    curves.setdefault(f"seed{seed}|{c}|{m}", {})[str(step)] = \
                        round(RR.gain(a[m], b[m]), 3) if a[m] else None
                curves.setdefault(
                    f"seed{seed}|{c}|success_5cm5deg", {})[str(step)] = round(
                    100.0 * (b["success_5cm5deg"] - a["success_5cm5deg"]), 3)
            for name, prefix in (("corner_late", "corner_late"),
                                 ("belief_head", "net."),
                                 ("line_late", "line_late")):
                d = drift(raw["C1_RESCUE"], raw["C0"], prefix)
                if d:
                    drifts.setdefault(f"seed{seed}|{name}", {})[str(step)] = d
            log(f"seed{seed} @{step:5d}  " + "  ".join(
                f"{c.split('_')[-1]} rms={curves[f'seed{seed}|{c}|obs_rms'][str(step)]:+.1f}"
                f" R={curves[f'seed{seed}|{c}|R_median'][str(step)]:+.1f}"
                for c in RR.CELLS))

    def classify(key_fmt):
        out = {}
        for c in RR.CELLS:
            for m in ("obs_rms", "R_median", "t_median"):
                s1 = curves.get(f"seed1|{c}|{m}", {})
                s2 = curves.get(f"seed2|{c}|{m}", {})
                steps = [s for s in map(str, MARKS[1:]) if s in s1 and s in s2]
                if not steps:
                    continue
                disagree = sum(1 for s in steps
                               if np.sign(s1[s]) != np.sign(s2[s]))
                early = [s for s in steps if int(s) in EARLY]
                late = [s for s in steps if int(s) in LATE]

                def mean(d, ks):
                    return float(np.mean([d[k] for k in ks])) if ks else 0.0
                flip = (np.sign(mean(s1, early)) != np.sign(mean(s1, late))
                        or np.sign(mean(s2, early)) != np.sign(mean(s2, late)))
                agree_early = (np.sign(mean(s1, early)) ==
                               np.sign(mean(s2, early)))
                if disagree >= 0.6 * len(steps):
                    label = "LC_A"
                elif agree_early and flip:
                    label = "LC_B"
                else:
                    label = "LC_C"
                out[f"{c}|{m}"] = {"label": label,
                                   "disagree_marks": f"{disagree}/{len(steps)}",
                                   "seed1_final": s1[steps[-1]],
                                   "seed2_final": s2[steps[-1]]}
        return out

    verdict = classify(None)
    payload = {"convention": "positive = C1_RESCUE better than C0",
               "marks": list(MARKS), "curves": curves,
               "classification": verdict,
               "classification_rule": {
                   "LC_A": "sign disagrees on >=60% of marks",
                   "LC_B": "seeds agree early (250-1000) and a seed flips late (2000-3000)",
                   "LC_C": "neither"}}
    (OUT / "learning_curve_metrics.json").write_text(json.dumps(payload, indent=1))
    (OUT / "parameter_drift.json").write_text(json.dumps(drifts, indent=1))
    log("\n=== classification ===")
    for k, v in verdict.items():
        log(f"  {k:<26} {v['label']}  disagree {v['disagree_marks']}  "
            f"final s1 {v['seed1_final']:+.1f} s2 {v['seed2_final']:+.1f}")
    log("-> learning_curve_metrics.json / parameter_drift.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.parse_args()
    run(None)
