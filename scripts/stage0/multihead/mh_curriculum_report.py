"""PHASE 7-10 -- evaluate C0 vs C1 on the target regimes and judge the gate.

Targets use the dataset's own yaw convention, `abs_frontal_yaw = 45 -
facing_margin`, because that is what CORNER_LA_OBLIQUE_V1 was targeted with.
Recomputing BROAD's target cells with it reproduces the release note's counts
exactly (1120 / 1116), which is the check that settles which convention to use.

    T1  elevation < 8 deg,  15 <= |yaw| < 30      primary
    T2  elevation < 8 deg,       |yaw| >= 30      primary
    T3  elevation < 8 deg,       |yaw| <  15      low-angle frontal safety
    T4  everything else                            safety

The evaluation population is D2 + D3 + D4.  It was used while designing the
dataset, so this is a short-screen development population and is labelled as one
-- not a paper-final independent confirmation.

The line branch is checked for *exact* parity, not a percentage band: with
disjoint late blocks and a frozen trunk it is a wiring invariant, and any drift
means the isolation is broken rather than that the data helped.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_cigm as CG                                             # noqa: E402
import mh_curriculum as CU                                       # noqa: E402
import mh_data as MD                                             # noqa: E402
import mh_diagnose as DG                                         # noqa: E402
import mh_screen as MS                                           # noqa: E402
import mh_splitlate as SL                                        # noqa: E402

OUT = MD.OUT
CKPT = MS.CKPT
POPULATIONS = ("D2_MH_DEV512", "D3_MH_CONF512", "D4_THETA_CONFIRM512")
SEEDS = (1, 2)
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260824

GATE = {"corner_rms_gain_pct": 10.0, "geometry_gain_pct": 10.0,
        "R_gain_pct": 5.0, "t_gain_pct": 5.0,
        "safety_degrade_pct": 5.0}

FACES = {"front": (0, 1, 2, 3), "rear": (4, 5, 6, 7), "top": (0, 1, 5, 4),
         "bottom": (3, 2, 6, 7), "left": (0, 3, 7, 4), "right": (1, 2, 6, 5)}
INCIDENT = {c: [f for f, v in FACES.items() if c in v] for c in range(8)}


def log(message):
    print(message, flush=True)


def observable_mask(X, rotation, translation, pixels, width, height):
    camera = (rotation @ X.T).T + translation
    centre = camera.mean(0)
    facing = {}
    for name, corners in FACES.items():
        face = camera[list(corners)]
        normal = np.cross(face[1] - face[0], face[3] - face[0])
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        if normal @ (face.mean(0) - centre) < 0:
            normal = -normal
        facing[name] = (normal @ (-face.mean(0))) > 0
    self_visible = np.array([any(facing[f] for f in INCIDENT[c])
                             for c in range(8)])
    inframe = np.array([(0 <= x < width and 0 <= y < height)
                        for x, y in pixels])
    return self_visible & inframe


def target_of(elevation, yaw):
    if elevation < 8.0:
        if yaw >= 30.0:
            return "T2 lo|y30+"
        if yaw >= 15.0:
            return "T1 lo|y15-30"
        return "T3 lo|y<15"
    return "T4 other"


def evaluate_checkpoint(path, features, grid_theta, grid_rho, valid, stems_by):
    state = torch.load(path, map_location=MD.DEV, weights_only=False)
    model = SL.SplitLate("A1_CORNER_LINE")
    model.load_state_dict(state["model"])
    model.to(MD.DEV).eval()
    rows = []
    with torch.no_grad():
        for population, stems in stems_by.items():
            for start in range(0, len(stems), MS.BATCH):
                chunk = stems[start:start + MS.BATCH]
                pack = MD.load_pack(chunk)
                beliefs = CU.corner_forward(model, pack["images"])
                peaks = MS._decode_peaks(beliefs[-1][:, :9])
                for index, stem in enumerate(chunk):
                    label = MD.read_label(stem)
                    obj = label["objects"][0]
                    width, height = pack["resolution"][index]
                    X = CG.object_points(label)
                    K = CG.intrinsics(label)
                    rotation, translation = CG.gt_pose(label)
                    truth_grid = pack["grid"][index][:8]
                    predicted = peaks[index][:8]
                    truth_px = np.asarray(obj["projected_cuboid"], float)[:8]
                    predicted_px = CG.grid_to_pixels(predicted, width, height)
                    mask = observable_mask(X, rotation, translation, truth_px,
                                           width, height)
                    error = np.linalg.norm(predicted_px - truth_px, axis=1)
                    fit = DG._affine_fit(truth_grid, predicted)
                    pose = CG.solve(X, predicted_px, K)
                    err = CG.pose_error(pose, rotation, translation)
                    elevation = float((obj.get("v2_labels") or {})
                                      .get("elevation_deg_actual", np.nan))
                    yaw = 45.0 - float(obj.get("facing_margin", np.nan))
                    rows.append({
                        "stem": stem, "target": target_of(elevation, yaw),
                        "obs_rms": float(np.sqrt(np.mean(error[mask] ** 2)))
                        if mask.any() else np.nan,
                        "front_rear_shift": float(np.linalg.norm(
                            (predicted[list(DG.FRONT)].mean(0)
                             - predicted[list(DG.REAR)].mean(0))
                            - (truth_grid[list(DG.FRONT)].mean(0)
                               - truth_grid[list(DG.REAR)].mean(0)))),
                        "affine_scale_gap": abs(fit["scale_isotropic"] - 1.0),
                        "centroid_shift": float(np.linalg.norm(
                            predicted.mean(0) - truth_grid.mean(0))),
                        "nonaffine_rms": float(fit["nonaffine_rms"]),
                        "R": err[0] if err else np.nan,
                        "t": err[1] if err else np.nan})
    line_params = torch.cat([p.detach().reshape(-1)
                             for p in model.line_late.parameters()])
    return rows, line_params.float().cpu().numpy()


MEASURES = ("obs_rms", "front_rear_shift", "affine_scale_gap",
            "centroid_shift", "nonaffine_rms")


def summarise(rows):
    R = np.array([r["R"] for r in rows], float)
    t = np.array([r["t"] for r in rows], float)
    good = np.isfinite(R) & np.isfinite(t)
    entry = {"n": len(rows)}
    for key in MEASURES:
        values = np.array([r[key] for r in rows], float)
        values = values[np.isfinite(values)]
        entry[key] = round(float(np.median(values)), 5) if len(values) else None
    entry["R_median"] = round(float(np.median(R[good])), 4) if good.any() else None
    entry["R_p90"] = round(float(np.percentile(R[good], 90)), 4) if good.any() else None
    entry["t_median"] = round(float(np.median(t[good])), 5) if good.any() else None
    entry["t_p90"] = round(float(np.percentile(t[good], 90)), 5) if good.any() else None
    entry["success_5cm5deg"] = round(float(
        ((R <= 5.0) & (t <= 0.05) & good).sum() / max(len(rows), 1)), 4)
    return entry


def gain(reference, value):
    if reference is None or value is None or not reference:
        return 0.0
    return 100.0 * (reference - value) / abs(reference)


def run(arguments):
    MS.deterministic()
    grid_theta, grid_rho, valid, features = MS.lattice()
    stems_by = {p: json.loads(
        (OUT / f"{p.lower()}_manifest.json").read_text())["stems"]
        for p in POPULATIONS}
    arms = arguments.arms.split(",")
    result = {"step": arguments.step, "gate": GATE, "arms": arms,
              "population": "D2+D3+D4 (short-screen development population)",
              "yaw_convention": "45 - facing_margin (dataset definition)",
              "seeds": {}}
    frames = {}
    for seed in SEEDS:
        block, line_params = {}, {}
        for arm in arms:
            path = (CKPT / f"curriculum_{arm}_seed{seed}"
                    / f"step_{arguments.step:05d}.pth")
            if not path.exists():
                log(f"skip {arm} seed{seed}: {path.name} absent")
                continue
            rows, params = evaluate_checkpoint(
                path, features, grid_theta, grid_rho, valid, stems_by)
            frames[(arm, seed)] = rows
            line_params[arm] = params
            by_target = {}
            for name in CU.TARGETS:
                subset = [r for r in rows if r["target"] == name]
                if subset:
                    by_target[name] = summarise(subset)
            by_target["ALL"] = summarise(rows)
            block[arm] = by_target
            log(f"seed{seed} {arm:<9} "
                + "  ".join(f"{n.split()[0]} n={by_target[n]['n']}"
                            f" rms={by_target[n]['obs_rms']:.2f}"
                            f" R={by_target[n]['R_median']:.2f}"
                            for n in CU.TARGETS if n in by_target))
        if "C0" in line_params:
            for arm in arms:
                if arm == "C0" or arm not in line_params:
                    continue
                diff = float(np.abs(line_params["C0"]
                                    - line_params[arm]).max())
                block.setdefault("line_parity", {})[arm] = diff
                log(f"seed{seed} line param max|diff| C0 vs {arm} = {diff:.3e}")
        result["seeds"][f"seed{seed}"] = block

    # ---------------------------------------------------------- gate
    for arm in arms:
        if arm == "C0":
            continue
        verdict, per_seed = [], {}
        for seed in SEEDS:
            block = result["seeds"].get(f"seed{seed}", {})
            if "C0" not in block or arm not in block:
                continue
            control, test = block["C0"], block[arm]
            checks = {}
            ok = True
            for name in ("T1 lo|y15-30", "T2 lo|y30+"):
                if name not in control:
                    continue
                a, b = control[name], test[name]
                item = {
                    "n": b["n"],
                    "obs_rms_gain": round(gain(a["obs_rms"], b["obs_rms"]), 2),
                    "frs_gain": round(gain(a["front_rear_shift"],
                                           b["front_rear_shift"]), 2),
                    "scale_gain": round(gain(a["affine_scale_gap"],
                                             b["affine_scale_gap"]), 2),
                    "R_gain": round(gain(a["R_median"], b["R_median"]), 2),
                    "t_gain": round(gain(a["t_median"], b["t_median"]), 2),
                    "success_delta_pp": round(
                        100.0 * (b["success_5cm5deg"]
                                 - a["success_5cm5deg"]), 2)}
                item["PASS"] = bool(
                    item["obs_rms_gain"] >= GATE["corner_rms_gain_pct"]
                    and max(item["frs_gain"], item["scale_gain"])
                    >= GATE["geometry_gain_pct"]
                    and item["R_gain"] >= GATE["R_gain_pct"]
                    and item["t_gain"] >= GATE["t_gain_pct"]
                    and item["success_delta_pp"] >= 0.0)
                ok = ok and item["PASS"]
                checks[name] = item
            for name in ("T3 lo|y<15", "T4 other"):
                if name not in control:
                    continue
                a, b = control[name], test[name]
                item = {"n": b["n"],
                        "R_gain": round(gain(a["R_median"], b["R_median"]), 2),
                        "t_gain": round(gain(a["t_median"], b["t_median"]), 2)}
                item["SAFE"] = bool(
                    item["R_gain"] >= -GATE["safety_degrade_pct"]
                    and item["t_gain"] >= -GATE["safety_degrade_pct"])
                ok = ok and item["SAFE"]
                checks[name] = item
            parity = block.get("line_parity", {}).get(arm)
            checks["line_param_max_diff"] = parity
            checks["LINE_ISOLATION_EXACT"] = bool(parity == 0.0)
            ok = ok and checks["LINE_ISOLATION_EXACT"]
            checks["SEED_PASS"] = bool(ok)
            per_seed[f"seed{seed}"] = checks
            verdict.append(ok)
        result[f"{arm}_gate"] = {"per_seed": per_seed,
                                 "PASS": bool(verdict and all(verdict))}
        log(f"\n{arm}: PASS = {result[f'{arm}_gate']['PASS']}")
        for name, checks in per_seed.items():
            for target in ("T1 lo|y15-30", "T2 lo|y30+"):
                if target in checks:
                    c = checks[target]
                    log(f"  {name} {target:<14} rms {c['obs_rms_gain']:+6.2f}% "
                        f"frs {c['frs_gain']:+6.2f}% scale {c['scale_gain']:+6.2f}% "
                        f"R {c['R_gain']:+6.2f}% t {c['t_gain']:+6.2f}% "
                        f"5cm5 {c['success_delta_pp']:+5.2f}pp -> "
                        f"{'PASS' if c['PASS'] else 'FAIL'}")

    # ---------------------------------------------------------- bootstrap
    boot = {"resamples": BOOTSTRAP, "seeds": {}}
    for seed in SEEDS:
        entry = {}
        for arm in arms:
            if arm == "C0" or (arm, seed) not in frames:
                continue
            control = frames[("C0", seed)]
            test = frames[(arm, seed)]
            order = {r["stem"]: i for i, r in enumerate(control)}
            pairs = [(control[order[r["stem"]]], r) for r in test
                     if r["stem"] in order]
            rng = np.random.default_rng(BOOTSTRAP_SEED + seed)
            for name in CU.TARGETS + ("ALL",):
                subset = [p for p in pairs
                          if name == "ALL" or p[0]["target"] == name]
                if len(subset) < 20:
                    continue
                for metric in ("obs_rms", "front_rear_shift", "R", "t"):
                    a = np.array([p[0][metric] for p in subset], float)
                    b = np.array([p[1][metric] for p in subset], float)
                    good = np.flatnonzero(np.isfinite(a) & np.isfinite(b))
                    if len(good) < 20:
                        continue
                    draws = np.empty(BOOTSTRAP)
                    for k in range(BOOTSTRAP):
                        pick = good[rng.integers(0, len(good), len(good))]
                        draws[k] = gain(float(np.median(a[pick])),
                                        float(np.median(b[pick])))
                    entry[f"{arm}|{name}|{metric}"] = {
                        "n": int(len(good)),
                        "effect_pct": round(float(np.median(draws)), 2),
                        "ci95": [round(float(np.percentile(draws, 2.5)), 2),
                                 round(float(np.percentile(draws, 97.5)), 2)],
                        "P_better": round(float((draws > 0).mean()), 4)}
        boot["seeds"][f"seed{seed}"] = entry
    (OUT / "branch_curriculum_bootstrap.json").write_text(
        json.dumps(boot, indent=1))
    (OUT / f"branch_curriculum_report_step{arguments.step}.json").write_text(
        json.dumps(result, indent=1))
    log(f"\n-> branch_curriculum_report_step{arguments.step}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=3000)
    parser.add_argument("--arms", default="C0,C1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
