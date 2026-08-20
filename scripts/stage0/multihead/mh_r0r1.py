"""PHASE 1B + R0 + R1 -- canonical failure map, C0/C1 re-evaluation, exposure audit.

No training.  Everything reads existing checkpoints, labels and the deterministic
samplers the two arms were built from.

Canonical yaw is the dataset release contract, `45 - facing_margin`.  The old
bucket names turn out to agree with it exactly (Y15_30 -> 100% canonical 15-30,
Y30_PLUS -> 100% canonical >=30, zero leakage), but that was verified rather than
assumed -- the mismatch was only ever between canonical and an earlier
pose-derived yaw.

Cells:  LA_FRONTAL  e<8, yaw<15      LA_EASY  e<8, 15<=yaw<30
        LA_HARD     e<8, yaw>=30     NON_LA   e>=8
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
import mh_regime as RG          # noqa: E402
import mh_screen as MS          # noqa: E402
import mh_splitlate as SL       # noqa: E402

OUT = MD.OUT
SEEDS = (1, 2)
ARMS_EVAL = ("C0", "C1", "C1_RESCUE")
BOOTSTRAP = 10_000
BOOT_SEED = 20260827
CELLS = ("LA_FRONTAL", "LA_EASY", "LA_HARD", "NON_LA")


def log(m):
    print(m, flush=True)


def cell_of(elev, yaw):
    if elev >= 8.0:
        return "NON_LA"
    return "LA_FRONTAL" if yaw < 15 else ("LA_EASY" if yaw < 30 else "LA_HARD")


def dev_index():
    """Every MH_DEV frame with its canonical cell and V_vis."""
    idx = RG.load()
    yaw = np.load(OUT / "regime_yaw_canonical.npy")
    split = np.array([str(x) for x in idx["split"]])
    stems = np.array([str(s) for s in idx["stem"]])
    keep = split == "MH_DEV"
    return {s: {"cell": cell_of(float(e), float(y)), "vvis": int(v)}
            for s, e, y, v in zip(stems[keep], idx["elev_actual"][keep],
                                  yaw[keep], idx["V_vis_actual"][keep])}


def measure(model, items, meta):
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), MS.BATCH):
            chunk = items[start:start + MS.BATCH]
            pack = CU.load_pack_items(chunk)
            beliefs = CU.corner_forward(model, pack["images"])
            peaks = MS._decode_peaks(beliefs[-1][:, :9])
            for i, (root, stem) in enumerate(chunk):
                label = CU.read_label_from(root, stem)
                obj = label["objects"][0]
                w, h = pack["resolution"][i]
                X, K = CG.object_points(label), CG.intrinsics(label)
                R, t = CG.gt_pose(label)
                tg = pack["grid"][i][:8]
                pr = peaks[i][:8]
                tpx = np.asarray(obj["projected_cuboid"], float)[:8]
                ppx = CG.grid_to_pixels(pr, w, h)
                m = CR.observable_mask(X, R, t, tpx, w, h)
                err = np.linalg.norm(ppx - tpx, axis=1)
                fit = DG._affine_fit(tg, pr)
                pe = CG.pose_error(CG.solve(X, ppx, K), R, t)
                rows.append({
                    "stem": stem, "cell": meta[stem]["cell"],
                    "vvis": meta[stem]["vvis"],
                    "obs_rms": float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else np.nan,
                    "front_rear_shift": float(np.linalg.norm(
                        (pr[list(DG.FRONT)].mean(0) - pr[list(DG.REAR)].mean(0))
                        - (tg[list(DG.FRONT)].mean(0) - tg[list(DG.REAR)].mean(0)))),
                    "affine_scale_gap": abs(fit["scale_isotropic"] - 1.0),
                    "centroid_shift": float(np.linalg.norm(pr.mean(0) - tg.mean(0))),
                    "nonaffine_rms": float(fit["nonaffine_rms"]),
                    "R": pe[0] if pe else np.nan, "t": pe[1] if pe else np.nan})
    return rows


KEYS = ("obs_rms", "front_rear_shift", "affine_scale_gap",
        "centroid_shift", "nonaffine_rms")


def agg(rows):
    R = np.array([r["R"] for r in rows], float)
    t = np.array([r["t"] for r in rows], float)
    g = np.isfinite(R) & np.isfinite(t)
    e = {"n": len(rows)}
    for k in KEYS:
        v = np.array([r[k] for r in rows], float); v = v[np.isfinite(v)]
        e[k] = round(float(np.median(v)), 5) if len(v) else None
    for nm, arr, q in (("R_median", R, 50), ("R_p90", R, 90),
                       ("t_median", t, 50), ("t_p90", t, 90)):
        e[nm] = round(float(np.percentile(arr[g], q)), 5) if g.any() else None
    e["success_5cm5deg"] = round(float(
        ((R <= 5.0) & (t <= 0.05) & g).sum() / max(len(rows), 1)), 4)
    return e


def gain(a, b):
    return 100.0 * (a - b) / abs(a) if a else 0.0


def run(a):
    MS.deterministic()
    meta = dev_index()
    stems = sorted(meta)
    items = [(MD.DATA, s) for s in stems]
    log(f"MH_DEV {len(stems)} frames; cells " + str(
        {c: sum(1 for v in meta.values() if v['cell'] == c) for c in CELLS}))

    frames, report = {}, {"cells": CELLS, "yaw": "45 - facing_margin",
                          "population": "MH_DEV (all 6242)", "seeds": {}}
    for seed in SEEDS:
        block = {}
        for arm in ARMS_EVAL:
            p = MS.CKPT / f"curriculum_{arm}_seed{seed}" / "step_03000.pth"
            st = torch.load(p, map_location=MD.DEV, weights_only=False)
            model = SL.SplitLate("A1_CORNER_LINE")
            model.load_state_dict(st["model"]); model.to(MD.DEV).eval()
            if arm != "C0":
                base = torch.load(MS.CKPT / f"curriculum_C0_seed{seed}"
                                  / "step_03000.pth", map_location=MD.DEV,
                                  weights_only=False)["model"]
                diff = max(float((state_v - base[k]).abs().max())
                           for k, state_v in st["model"].items()
                           if k.startswith("line_late"))
                report.setdefault("line_parity", {})[f"{arm}_seed{seed}"] = diff
                log(f"  line param max|diff| C0 vs {arm} seed{seed} = {diff:.3e}")
            rows = measure(model, items, meta)
            frames[(arm, seed)] = rows
            by = {c: agg([r for r in rows if r["cell"] == c]) for c in CELLS}
            for c in CELLS:                       # V_vis stratification
                for v in (4, 5, 6, 7):
                    sub = [r for r in rows if r["cell"] == c and r["vvis"] == v]
                    if len(sub) >= 15:
                        by[f"{c}|Vvis{v}"] = agg(sub)
            block[arm] = by
            log(f"seed{seed} {arm}  " + "  ".join(
                f"{c}: n={by[c]['n']} rms={by[c]['obs_rms']:.2f} R={by[c]['R_median']:.2f}"
                f" t={by[c]['t_median']:.4f}" for c in CELLS))
        report["seeds"][f"seed{seed}"] = block

    boot = {"resamples": BOOTSTRAP, "seeds": {}}
    for seed in SEEDS:
        c0 = frames[("C0", seed)]
        order = {r["stem"]: i for i, r in enumerate(c0)}
        rng = np.random.default_rng(BOOT_SEED + seed)
        entry = {}
        for arm in ARMS_EVAL[1:]:
          pairs = [(c0[order[r["stem"]]], r) for r in frames[(arm, seed)]]
          for c in CELLS:
            sub = [q for q in pairs if q[0]["cell"] == c]
            if len(sub) < 20:
                continue
            for met in ("obs_rms", "front_rear_shift", "affine_scale_gap", "R", "t"):
                x = np.array([q[0][met] for q in sub], float)
                y = np.array([q[1][met] for q in sub], float)
                ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
                if len(ok) < 20:
                    continue
                d = np.empty(BOOTSTRAP)
                for j in range(BOOTSTRAP):
                    pk = ok[rng.integers(0, len(ok), len(ok))]
                    d[j] = gain(float(np.median(x[pk])), float(np.median(y[pk])))
                entry[f"{arm}#{c}#{met}"] = {
                    "n": int(len(ok)),
                    "effect_pct": round(float(np.median(d)), 2),
                    "ci95": [round(float(np.percentile(d, 2.5)), 2),
                             round(float(np.percentile(d, 97.5)), 2)],
                    "P_better": round(float((d > 0).mean()), 4)}
        boot["seeds"][f"seed{seed}"] = entry
    (OUT / "R2_FULL_REEVAL.json").write_text(json.dumps(report, indent=1))
    (OUT / "C1_RESCUE_bootstrap.json").write_text(json.dumps(boot, indent=1))
    log("-> R2_FULL_REEVAL.json / C1_RESCUE_bootstrap.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.parse_args()
    run(None)
