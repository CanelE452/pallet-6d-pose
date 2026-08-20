"""Re-judge C0 vs C1 without retraining.

Two things changed since the first judgement and neither needs a new run:

  * the yaw convention the dataset was designed with is the canonical one, and
    the earlier risk map used a different one.  Recomputed under the canonical
    convention, `E0<8 x Y15_30` -- half of what the new 5K targets -- is a
    near-baseline cell (9.62 px, x1.14) while `E0<8 x Yc>=30` is the real
    failure (30.60 px, x3.63) and low-angle *frontal* is second (13.01, x1.54).
  * the first judgement ran on 35-51 frames per target.  MH_DEV was never
    trained on and only 1,536 of its 6,242 frames had been read, so a larger
    confirmation population costs nothing but bookkeeping.

Train frames are deliberately not used as "unseen": both arms start from
E3 @18k, which was trained on the whole train pool, so no train frame is unseen
with respect to the model under test.
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_cigm as CG          # noqa: E402
import mh_curriculum as CU    # noqa: E402
import mh_curriculum_report as CR   # noqa: E402
import mh_data as MD          # noqa: E402
import mh_diagnose as DG      # noqa: E402
import mh_screen as MS        # noqa: E402
import mh_splitlate as SL     # noqa: E402

OUT = MD.OUT
SEEDS = (1, 2)
BOOTSTRAP = 10_000
BOOT_SEED = 20260826


def log(m):
    print(m, flush=True)


def populations():
    """(name, [(root, stem), ...], label_fn) for each population."""
    out = {}
    d5 = json.loads((OUT / "d5_target_unseen_manifest.json").read_text())
    by = {}
    for target, stems in d5["targets"].items():
        for s in stems:
            by[s] = target
    out["BROAD_UNSEEN"] = ([(MD.DATA, s) for s in d5["stems"]],
                           lambda st, _by=by: _by[st])
    small = []
    for name in ("d2_mh_dev512", "d3_mh_conf512", "d4_theta_confirm512"):
        small += json.loads((OUT / f"{name}_manifest.json").read_text())["stems"]
    yawc = np.load(OUT / "regime_yaw_canonical.npy")
    import mh_regime as RG
    idx = RG.load()
    pos = {str(s): i for i, s in enumerate(idx["stem"])}

    def small_label(stem):
        i = pos[stem]
        e, y = float(idx["elev_actual"][i]), float(yawc[i])
        if e < 8.0:
            return "T2" if y >= 30 else ("T1" if y >= 15 else "T3")
        return "T4"
    out["DEV_SMALL"] = ([(MD.DATA, s) for s in small], small_label)
    new = json.loads((OUT / "new_unseen_manifest.json").read_text())
    items = [(CU.LA_ROOT / x.split("/")[0], x.split("/")[1])
             for x in new["items"]]
    bucket = {x.split("/")[1] + "|" + x.split("/")[0]: x.split("/")[0]
              for x in new["items"]}
    out["NEW_UNSEEN"] = (items,
                         lambda st, _b=bucket: "N1" if "y15_30" in
                         [k for k in _b if k.startswith(st + "|")][0] else "N2")
    return out


def evaluate(model, items, label_fn):
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
                X = CG.object_points(label)
                K = CG.intrinsics(label)
                R, t = CG.gt_pose(label)
                truth_g = pack["grid"][i][:8]
                pred = peaks[i][:8]
                truth_px = np.asarray(obj["projected_cuboid"], float)[:8]
                pred_px = CG.grid_to_pixels(pred, w, h)
                mask = CR.observable_mask(X, R, t, truth_px, w, h)
                err = np.linalg.norm(pred_px - truth_px, axis=1)
                fit = DG._affine_fit(truth_g, pred)
                pose = CG.solve(X, pred_px, K)
                pe = CG.pose_error(pose, R, t)
                rows.append({
                    "stem": stem, "target": label_fn(stem),
                    "obs_rms": float(np.sqrt(np.mean(err[mask] ** 2)))
                    if mask.any() else np.nan,
                    "front_rear_shift": float(np.linalg.norm(
                        (pred[list(DG.FRONT)].mean(0) - pred[list(DG.REAR)].mean(0))
                        - (truth_g[list(DG.FRONT)].mean(0)
                           - truth_g[list(DG.REAR)].mean(0)))),
                    "affine_scale_gap": abs(fit["scale_isotropic"] - 1.0),
                    "R": pe[0] if pe else np.nan,
                    "t": pe[1] if pe else np.nan})
    return rows


def summarise(rows):
    R = np.array([r["R"] for r in rows], float)
    t = np.array([r["t"] for r in rows], float)
    good = np.isfinite(R) & np.isfinite(t)
    e = {"n": len(rows)}
    for k in ("obs_rms", "front_rear_shift", "affine_scale_gap"):
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        e[k] = round(float(np.median(v)), 5) if len(v) else None
    e["R_median"] = round(float(np.median(R[good])), 4) if good.any() else None
    e["t_median"] = round(float(np.median(t[good])), 5) if good.any() else None
    e["success_5cm5deg"] = round(float(
        ((R <= 5.0) & (t <= 0.05) & good).sum() / max(len(rows), 1)), 4)
    return e


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", type=int, default=3000)
    a = p.parse_args()
    MS.deterministic()
    pops = populations()
    result = {"step": a.step, "populations": {k: len(v[0]) for k, v in pops.items()},
              "yaw_convention": "45 - facing_margin (canonical)", "seeds": {}}
    frames = {}
    for seed in SEEDS:
        block = {}
        for arm in ("C0", "C1"):
            path = MS.CKPT / f"curriculum_{arm}_seed{seed}" / f"step_{a.step:05d}.pth"
            state = torch.load(path, map_location=MD.DEV, weights_only=False)
            model = SL.SplitLate("A1_CORNER_LINE")
            model.load_state_dict(state["model"])
            model.to(MD.DEV).eval()
            for pop, (items, fn) in pops.items():
                rows = evaluate(model, items, fn)
                frames[(pop, arm, seed)] = rows
                agg = {}
                for tgt in sorted({r["target"] for r in rows}):
                    agg[tgt] = summarise([r for r in rows if r["target"] == tgt])
                agg["ALL"] = summarise(rows)
                block.setdefault(pop, {})[arm] = agg
                log(f"seed{seed} {arm} {pop:<13} " + "  ".join(
                    f"{k} n={v['n']} rms={v['obs_rms']:.2f} R={v['R_median']:.2f}"
                    for k, v in agg.items() if k != "ALL"))
        result["seeds"][f"seed{seed}"] = block

    def gain(x, y):
        return 100.0 * (x - y) / abs(x) if x else 0.0
    boot = {"resamples": BOOTSTRAP, "seeds": {}}
    for seed in SEEDS:
        entry = {}
        for pop in pops:
            c = frames[(pop, "C0", seed)]
            k = frames[(pop, "C1", seed)]
            order = {r["stem"]: i for i, r in enumerate(c)}
            pairs = [(c[order[r["stem"]]], r) for r in k if r["stem"] in order]
            rng = np.random.default_rng(BOOT_SEED + seed)
            for tgt in sorted({r["target"] for r in c}) + ["ALL"]:
                sub = [q for q in pairs if tgt == "ALL" or q[0]["target"] == tgt]
                if len(sub) < 20:
                    continue
                for met in ("obs_rms", "front_rear_shift", "R", "t"):
                    x = np.array([q[0][met] for q in sub], float)
                    y = np.array([q[1][met] for q in sub], float)
                    ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
                    if len(ok) < 20:
                        continue
                    d = np.empty(BOOTSTRAP)
                    for j in range(BOOTSTRAP):
                        pick = ok[rng.integers(0, len(ok), len(ok))]
                        d[j] = gain(float(np.median(x[pick])), float(np.median(y[pick])))
                    entry[f"{pop}#{tgt}#{met}"] = {
                        "n": int(len(ok)),
                        "effect_pct": round(float(np.median(d)), 2),
                        "ci95": [round(float(np.percentile(d, 2.5)), 2),
                                 round(float(np.percentile(d, 97.5)), 2)],
                        "P_better": round(float((d > 0).mean()), 4)}
        boot["seeds"][f"seed{seed}"] = entry
    (OUT / "recheck_report.json").write_text(json.dumps(result, indent=1))
    (OUT / "recheck_bootstrap.json").write_text(json.dumps(boot, indent=1))
    log("-> recheck_report.json / recheck_bootstrap.json")


if __name__ == "__main__":
    main()
