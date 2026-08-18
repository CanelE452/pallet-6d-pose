"""PHASE 6-9 -- pose-aware corner supervision on top of E3, P0 against P1.

Opened only because the theta-only screen failed its two-seed gate.  The premise
is that what is left is not the solver's `rho` but the corner configuration's own
pose geometry: the scale oracle recovers 31-33% of translation, no predictor
reaches it, and the residual is systematic rather than noisy.

Nothing new is invented here.  The audit found a validated differentiable PnP in
the repo and this reuses it:

    Deep_Object_Pose/train/diffpnp3d_loss.py
        LocalSoftArgmax2D   differentiable sub-pixel keypoint from a belief map
        DiffPnP3DLoss       GT-seeded unrolled Gauss-Newton, then Huber on the
                            3D corner displacement / object diagonal

and the pose triple comes from `mh_cigm`, which is the same triple every pose
number in this study was computed from.  It was re-checked numerically before
being used: projecting `object_points` with `gt_pose` through `intrinsics`
reproduces the labelled corners to 1e-4 px.

## Arms

    P0   E3 continuation, unchanged:   L = L_line + lambda_corner * L_corner
    P1   plus pose supervision:        L = ... + lambda_pose * L_pose

`L_pose` is a function of the belief maps only, and in `SplitLate` the belief
stages read `corner_late`, a module the line branch does not share.  The early
trunk is frozen and detached.  So the line path in P1 is the same computation as
in P0 by construction, and `test_pose_gradient_does_not_reach_line` pins it.

## Coordinate handling

`LocalSoftArgmax2D` is constructed with `orig_size = belief_size = (50, 50)` so
it returns grid coordinates, and the per-frame scale to pixels is applied here as
`x * width / 50, y * height / 50` -- character for character what
`mh_cigm.grid_to_pixels` does.  Resolution varies frame to frame, which is why
the scale cannot live inside the module.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]
                       / "Deep_Object_Pose" / "train"))

import mh_arms as MH                                             # noqa: E402
import mh_cigm as CG                                             # noqa: E402
import mh_data as MD                                             # noqa: E402
import mh_diagnose as DG                                         # noqa: E402
import mh_screen as MS                                           # noqa: E402
import mh_splitlate as SL                                        # noqa: E402
from mh_arms import CAP, DH, V2                                  # noqa: E402

OUT = MD.OUT
CKPT = MS.CKPT
SOURCE_RUN = "screen_A1_CORNER_LINE_e3confirm25k"
SOURCE_STEP = 18000
STEPS = 3000
MARKS = (250, 500, 1000, 2000, 3000)

# PHASE 7B, second registration.  The first grid was (0.01, 0.1, 1.0), chosen by
# comparing loss *values*: `L_pose` was about 325x the weighted corner term, so
# 0.01 looked like a mild 3x.  That reasoning was wrong.  Measured on the belief
# maps the model actually trains, the gradient ratio is 253-297x at lambda 1.0,
# so 0.01 already puts about 70x the weighted corner gradient into the branch:
#
#     |dL_corner/d(belief)| 5.3e-05   x lambda_corner 0.035  ->  1.9e-06
#     |dL_pose  /d(belief)| 1.35e-02  at lambda_pose 1.0
#
# All three destroyed the model and their ordering was not monotone, which is
# what diverged runs look like -- the first calibration measured nothing about
# pose supervision, so it is kept on file as a failed measurement rather than a
# result.  The mechanism is understood and is not a defect: `corner_loss` is an
# MSE spread over 9x50x50 cells while `L_pose` reaches the maps through a
# soft-argmax that concentrates all sensitivity in a 7x7 window per corner.
#
# This grid is anchored so the pose gradient runs from well below the corner
# gradient to about twice it.  It is still a fixed pre-registered grid, and the
# anchor is a measurement of scale, not a rule that sets lambda -- setting
# lambda from a step-0 gradient ratio is what went wrong with lambda_corner.
LAMBDA_POSE_CANDIDATES = (1e-5, 3e-5, 1e-4, 3e-4)
LAMBDA_POSE_CANDIDATES_REJECTED_V1 = (0.01, 0.1, 1.0)
GRADIENT_ANCHOR = {"corner_grad": 5.3e-05, "weighted_corner_grad": 1.9e-06,
                   "pose_grad_at_lambda_1": 1.35e-02,
                   "measured_on": "E3 @18k seed1, 4 batches of 8"}

# DiffPnP3D contract, taken from the validated call sites rather than retuned.
SOFTARGMAX_WINDOW = 7
SOFTARGMAX_TEMPERATURE = 0.1
GN_STEPS = 4

# PHASE 9 gate, locked before P1 ran.
GATE = {"t_gain_pct": 10.0, "R_degrade_pct": 3.0,
        "geometry_gain_pct": 10.0, "line_degrade_pct": 0.5}

# Calibration, locked before the first short run.  D0 only, seed 1 only.
CALIBRATION_STEPS = 500
CALIBRATION_CORNER_DEGRADE_PCT = 10.0     # safety filter
CALIBRATION_T_DEGRADE_PCT = 0.0           # safety filter
# The theta-only screen failed because "best on the selection metric among the
# survivors" walked seed 1 to the edge of its locked grid.  Same shape of rule
# here, so the same guard: a candidate has to be clearly better, not marginally
# better, to justify a larger weight.
CALIBRATION_TIE_MARGIN_PCT = 20.0


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------ targets

def pose_targets(chunk, pack):
    """Per-frame X, K, R_gt, t_gt, diag, mask -- all from the validated triple."""
    X, K, R, t, diag, mask = [], [], [], [], [], []
    for index, stem in enumerate(chunk):
        label = MD.read_label(stem)
        points = CG.object_points(label)
        rotation, translation = CG.gt_pose(label)
        X.append(points)
        K.append(CG.intrinsics(label))
        R.append(rotation)
        t.append(translation)
        diag.append(float(np.linalg.norm(points.max(0) - points.min(0))))
        camera = (rotation @ points.T).T + translation
        mask.append(bool(pack["belief_valid"][index, :8].all())
                    and bool((camera[:, 2] > 1e-3).all()))
    device = MH.DEV
    to = lambda array, dtype=torch.float32: torch.as_tensor(  # noqa: E731
        np.asarray(array), dtype=dtype, device=device)
    return {"X": to(X), "K": to(K), "R_gt": to(R), "t_gt": to(t),
            "diag": to(diag), "mask": to(mask, torch.bool)}


def corner_pixels(beliefs, resolution, sampler):
    """Differentiable 2D corners in image pixels, grid semantics preserved."""
    coords, _ = sampler(beliefs[:, :8])
    width = torch.as_tensor([r[0] for r in resolution], dtype=coords.dtype,
                            device=coords.device)
    height = torch.as_tensor([r[1] for r in resolution], dtype=coords.dtype,
                             device=coords.device)
    scale = torch.stack([width, height], -1)[:, None, :] / MD.GRID
    return coords * scale


def build_pose_loss():
    from diffpnp3d_loss import DiffPnP3DLoss, LocalSoftArgmax2D
    sampler = LocalSoftArgmax2D(window=SOFTARGMAX_WINDOW,
                                temperature=SOFTARGMAX_TEMPERATURE,
                                orig_size=(MD.GRID, MD.GRID),
                                belief_size=(MD.GRID, MD.GRID)).to(MH.DEV)
    loss = DiffPnP3DLoss(n_gn=GN_STEPS, temperature=SOFTARGMAX_TEMPERATURE)
    return sampler, loss


def build_model(seed):
    source = CKPT / f"{SOURCE_RUN}_seed{seed}" / f"step_{SOURCE_STEP:05d}.pth"
    if not source.exists():
        raise SystemExit(f"source checkpoint missing: {source}")
    state = torch.load(source, map_location=MH.DEV, weights_only=False)
    torch.manual_seed(CAP.SEED)
    np.random.seed(CAP.SEED)
    model = SL.SplitLate("A1_CORNER_LINE")
    model.load_state_dict(state["model"])
    model.to(MH.DEV)
    return model, str(source)


# ------------------------------------------------------------------ audit

def run_audit(_arguments):
    """PHASE 7B: measure the loss scales before choosing lambda_pose."""
    MS.deterministic()
    weights = MS.lambdas()
    pool, _ = MD.pools()
    grid_theta, grid_rho, valid, features = MS.lattice()
    sampler, pose_loss = build_pose_loss()
    result = {"candidates": list(LAMBDA_POSE_CANDIDATES),
              "lambda_corner": weights["corner"], "seeds": {}}
    for seed in (1, 2):
        model, source = build_model(seed)
        model.train()
        rows = []
        stream = V2.step_schedule(list(pool), 8, MS.BATCH)
        for chunk, _ in stream:
            pack = MD.load_pack(chunk)
            out = model(pack["images"], features)
            theta_c, rho_c, support = DH.batch_rows(pack, CG.EDGES)
            target = DH.target_distribution(
                theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho,
                valid).reshape(*theta_c.shape, -1)
            line = DH.cross_entropy(out["line_scores"], target, support, valid)
            corner = MH.corner_loss(out["beliefs"], pack["belief"],
                                    pack["belief_valid"])
            targets = pose_targets(chunk, pack)
            pixels = corner_pixels(out["beliefs"][-1], pack["resolution"],
                                   sampler)
            value, info = pose_loss(pixels, targets["X"], targets["K"],
                                    targets["R_gt"], targets["t_gt"],
                                    targets["diag"], targets["mask"])
            rows.append({"line": float(line), "corner": float(corner),
                         "pose": float(value),
                         "usable": int(targets["mask"].sum()),
                         "n": len(chunk)})
        summary = {k: round(float(np.mean([r[k] for r in rows])), 6)
                   for k in ("line", "corner", "pose")}
        summary["usable_fraction"] = round(
            sum(r["usable"] for r in rows) / max(sum(r["n"] for r in rows), 1), 4)
        summary["weighted_corner"] = round(
            summary["corner"] * weights["corner"], 6)
        summary["pose_over_weighted_corner"] = round(
            summary["pose"] / max(summary["weighted_corner"], 1e-12), 4)
        summary["source"] = source
        result["seeds"][f"seed{seed}"] = summary
        log(f"seed{seed}  line {summary['line']:.5f}  corner {summary['corner']:.6f}"
            f"  (x lambda = {summary['weighted_corner']:.6f})"
            f"  pose {summary['pose']:.6f}"
            f"  usable {summary['usable_fraction']:.3f}")
    path = OUT / "pose_aware_loss_audit.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


# ------------------------------------------------------------------ train

def run_train(arguments):
    MS.deterministic()
    weights = MS.lambdas()
    pool, populations = MD.pools()
    seed = arguments.seed
    if seed != 1:
        import random
        pool = list(pool)
        random.Random(seed).shuffle(pool)
    grid_theta, grid_rho, valid, features = MS.lattice()
    sampler, pose_loss = build_pose_loss()

    arm = arguments.arm
    lambda_pose = float(arguments.lambda_pose) if arm == "P1" else 0.0
    model, source = build_model(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=CAP.LR, weight_decay=CAP.WD)
    history = {"arm": arm, "seed": seed, "source_checkpoint": source,
               "source_step": SOURCE_STEP, "steps": STEPS, "marks": list(MARKS),
               "CONTINUATION_OPTIMIZER": "FRESH",
               "lambda_corner": weights["corner"], "lambda_pose": lambda_pose,
               "trainable_params": int(sum(p.numel() for p in trainable))}
    path = OUT / f"pose_aware_{arm}_seed{seed}.json"
    directory = CKPT / f"poseaware_{arm}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    log(f"{arm} seed {seed}  lambda_pose {lambda_pose}  from {SOURCE_STEP}")

    def mark(step):
        entry = {"step": step}
        for label, stems in populations.items():
            entry[label] = MS.evaluate(model, stems, features, grid_theta,
                                       grid_rho, valid)
            line = entry[label]["line"]
            corner = entry[label]["corner"]
            log(f"  {arm} s{seed} @{step:5d} {label:<14} angle "
                f"{line['angle_median']:7.4f} offset {line['offset_median']:7.4f}"
                f" | cornerC {corner.get('direct_cell_median', '-')}")
        history[str(step)] = entry
        MS.write_json(path, history)
        torch.save({"arm": arm, "seed": seed, "step": step, "source": source,
                    "lambda_pose": lambda_pose, "model": model.state_dict()},
                   directory / f"step_{step:05d}.pth")

    mark(0)
    done = 0
    for chunk, _ in V2.step_schedule(pool, STEPS, MS.BATCH):
        model.train()
        pack = MD.load_pack(chunk)
        out = model(pack["images"], features)
        theta_c, rho_c, support = DH.batch_rows(pack, CG.EDGES)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho,
            valid).reshape(*theta_c.shape, -1)
        loss = DH.cross_entropy(out["line_scores"], target, support, valid)
        corner = MH.corner_loss(out["beliefs"], pack["belief"],
                                pack["belief_valid"])
        loss = loss + weights["corner"] * corner
        if lambda_pose:
            targets = pose_targets(chunk, pack)
            pixels = corner_pixels(out["beliefs"][-1], pack["resolution"],
                                   sampler)
            value, _ = pose_loss(pixels, targets["X"], targets["K"],
                                 targets["R_gt"], targets["t_gt"],
                                 targets["diag"], targets["mask"])
            loss = loss + lambda_pose * value
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        done += 1
        if done in MARKS:
            mark(done)
    log(f"-> {path}")


# ------------------------------------------------------------------ report

def geometry_and_pose(model, stems, features, grid_theta, grid_rho, valid):
    """PHASE 9 primaries: pose geometry first, corner median second."""
    rows, poses = [], []
    with torch.no_grad():
        for start in range(0, len(stems), MS.BATCH):
            chunk = stems[start:start + MS.BATCH]
            pack = MD.load_pack(chunk)
            out = model(pack["images"], features)
            peaks = MS._decode_peaks(out["beliefs"][-1][:, :9])
            for index, stem in enumerate(chunk):
                label = MD.read_label(stem)
                truth = pack["grid"][index][:8]
                predicted = peaks[index][:8]
                width, height = pack["resolution"][index]
                fit = DG._affine_fit(truth, predicted)
                pose = CG.solve(CG.object_points(label),
                                CG.grid_to_pixels(predicted, width, height),
                                CG.intrinsics(label))
                rotation, translation = CG.gt_pose(label)
                error = CG.pose_error(pose, rotation, translation)
                poses.append(error if error else (np.nan, np.nan))
                rows.append({
                    "affine_scale_isotropic": fit["scale_isotropic"],
                    "front_rear_shift": float(np.linalg.norm(
                        (predicted[list(DG.FRONT)].mean(0)
                         - predicted[list(DG.REAR)].mean(0))
                        - (truth[list(DG.FRONT)].mean(0)
                           - truth[list(DG.REAR)].mean(0)))),
                    "centroid_shift": float(np.linalg.norm(
                        predicted.mean(0) - truth.mean(0))),
                    "nonaffine_rms": fit["nonaffine_rms"],
                    "cornerC": float(np.linalg.norm(
                        predicted - truth, axis=1).mean()),
                })
    poses = np.asarray(poses)
    good = np.isfinite(poses).all(1)
    summary = {k: round(float(np.median([r[k] for r in rows])), 5)
               for k in rows[0]}
    summary["R_median"] = round(float(np.median(poses[good, 0])), 4)
    summary["R_p90"] = round(float(np.percentile(poses[good, 0], 90)), 4)
    summary["t_median"] = round(float(np.median(poses[good, 1])), 5)
    summary["t_p90"] = round(float(np.percentile(poses[good, 1], 90)), 5)
    summary["success_5cm5deg"] = round(float(
        ((poses[:, 0] <= 5.0) & (poses[:, 1] <= 0.05) & good).sum()
        / max(len(poses), 1)), 4)
    frames = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    frames["R"] = poses[:, 0]
    frames["t"] = poses[:, 1]
    return summary, frames


def run_report(arguments):
    MS.deterministic()
    grid_theta, grid_rho, valid, features = MS.lattice()
    populations = [p for p in arguments.populations.split(",") if p]
    result = {"step": arguments.step, "seeds": {}}
    for seed in (1, 2):
        block = {}
        for arm in ("P0", "P1"):
            checkpoint = (CKPT / f"poseaware_{arm}_seed{seed}"
                          / f"step_{arguments.step:05d}.pth")
            if not checkpoint.exists():
                log(f"skip {arm} seed{seed}: {checkpoint.name} absent")
                continue
            state = torch.load(checkpoint, map_location=MH.DEV,
                               weights_only=False)
            model = SL.SplitLate("A1_CORNER_LINE")
            model.load_state_dict(state["model"])
            model.to(MH.DEV).eval()
            block[arm] = {}
            for population in populations:
                stems = json.loads(
                    (OUT / f"{population.lower()}_manifest.json").read_text()
                )["stems"]
                summary, frames = geometry_and_pose(model, stems, features,
                                                    grid_theta, grid_rho, valid)
                block[arm][population] = summary
                np.savez_compressed(
                    OUT / f"pose_aware_frames_{arm}_seed{seed}_{population}.npz",
                    **frames)
                log(f"seed{seed} {arm} {population:<15} "
                    f"R {summary['R_median']:7.3f} t {summary['t_median']:.4f} "
                    f"5cm5 {summary['success_5cm5deg']:.4f} "
                    f"scale {summary['affine_scale_isotropic']:.4f} "
                    f"frs {summary['front_rear_shift']:.4f} "
                    f"cornerC {summary['cornerC']:.4f}")
        result["seeds"][f"seed{seed}"] = block
    path = OUT / f"pose_aware_report_step{arguments.step}.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


def run_calibrate(_arguments):
    """PHASE 7B: one short trajectory per candidate, selected on D0 alone.

    Selection metric is `|affine_scale_isotropic - 1|`, which is one of the
    gate's own terms and the quantity the scale oracle named as the translation
    lever -- selecting on an axis the gate does not score is the mistake the
    lambda audit found in the full-line solver.
    """
    MS.deterministic()
    grid_theta, grid_rho, valid, features = MS.lattice()
    stems = json.loads(
        (OUT / "d0_mh_seen512_manifest.json").read_text())["stems"]
    seed = 1
    result = {"steps": CALIBRATION_STEPS, "seed": seed,
              "candidates": list(LAMBDA_POSE_CANDIDATES),
              "population": "D0_MH_SEEN512 only",
              "rule": "reject if cornerC degrades > 10% or t degrades at all; "
                      "among survivors take the largest reduction of "
                      "|affine_scale_isotropic - 1|, and prefer the smaller "
                      "lambda unless the larger one is better by > 20%",
              "arms": {}}

    def short_run(lambda_pose):
        import random
        pool, _ = MD.pools()
        pool = list(pool)
        sampler, pose_loss = build_pose_loss()
        weights = MS.lambdas()
        model, _ = build_model(seed)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(trainable, lr=CAP.LR,
                                      weight_decay=CAP.WD)
        done = 0
        for chunk, _ in V2.step_schedule(pool, CALIBRATION_STEPS, MS.BATCH):
            model.train()
            pack = MD.load_pack(chunk)
            out = model(pack["images"], features)
            theta_c, rho_c, support = DH.batch_rows(pack, CG.EDGES)
            target = DH.target_distribution(
                theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho,
                valid).reshape(*theta_c.shape, -1)
            loss = DH.cross_entropy(out["line_scores"], target, support, valid)
            loss = loss + weights["corner"] * MH.corner_loss(
                out["beliefs"], pack["belief"], pack["belief_valid"])
            if lambda_pose:
                targets = pose_targets(chunk, pack)
                pixels = corner_pixels(out["beliefs"][-1], pack["resolution"],
                                       sampler)
                value, _ = pose_loss(pixels, targets["X"], targets["K"],
                                     targets["R_gt"], targets["t_gt"],
                                     targets["diag"], targets["mask"])
                loss = loss + lambda_pose * value
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            done += 1
        model.eval()
        summary, _ = geometry_and_pose(model, stems, features, grid_theta,
                                       grid_rho, valid)
        return summary

    base = short_run(0.0)
    result["arms"]["P0"] = base
    log(f"P0  scale {base['affine_scale_isotropic']:.4f}  "
        f"t {base['t_median']:.5f}  cornerC {base['cornerC']:.4f}")
    base_gap = abs(base["affine_scale_isotropic"] - 1.0)
    survivors = []
    for lambda_pose in LAMBDA_POSE_CANDIDATES:
        entry = short_run(lambda_pose)
        gap = abs(entry["affine_scale_isotropic"] - 1.0)
        entry["scale_gap"] = round(gap, 5)
        entry["scale_gap_gain_pct"] = round(
            100.0 * (base_gap - gap) / max(base_gap, 1e-9), 2)
        entry["cornerC_gain_pct"] = round(
            100.0 * (base["cornerC"] - entry["cornerC"]) / base["cornerC"], 2)
        entry["t_gain_pct"] = round(
            100.0 * (base["t_median"] - entry["t_median"]) / base["t_median"], 2)
        reasons = []
        if entry["cornerC_gain_pct"] < -CALIBRATION_CORNER_DEGRADE_PCT:
            reasons.append("cornerC")
        if entry["t_gain_pct"] < -CALIBRATION_T_DEGRADE_PCT:
            reasons.append("t")
        entry["rejected_for"] = reasons
        result["arms"][f"P1_lambda{lambda_pose:g}"] = entry
        if not reasons:
            survivors.append((entry["scale_gap_gain_pct"], lambda_pose))
        log(f"lam {lambda_pose:<5g} scale {entry['affine_scale_isotropic']:.4f} "
            f"(gap {entry['scale_gap_gain_pct']:+6.2f}%)  "
            f"t {entry['t_gain_pct']:+6.2f}%  "
            f"cornerC {entry['cornerC_gain_pct']:+6.2f}%  "
            f"{'OK' if not reasons else 'reject:' + ','.join(reasons)}")
    if survivors:
        survivors.sort(key=lambda pair: pair[1])          # smallest lambda first
        best = survivors[0]
        for score, lambda_pose in survivors[1:]:
            if score > best[0] * (1.0 + CALIBRATION_TIE_MARGIN_PCT / 100.0):
                best = (score, lambda_pose)
        result["selected_lambda_pose"] = best[1]
        result["NO_SAFE_LAMBDA"] = False
    else:
        result["selected_lambda_pose"] = None
        result["NO_SAFE_LAMBDA"] = True
    log(f"-> selected lambda_pose = {result['selected_lambda_pose']}")
    path = OUT / "pose_aware_calibration.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["audit", "calibrate", "train", "report"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arm", default="P1", choices=["P0", "P1"])
    parser.add_argument("--lambda-pose", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=3000)
    parser.add_argument("--populations",
                        default="D2_MH_DEV512,D3_MH_CONF512")
    arguments = parser.parse_args()
    {"audit": run_audit, "calibrate": run_calibrate, "train": run_train,
     "report": run_report}[arguments.command](arguments)


if __name__ == "__main__":
    main()
