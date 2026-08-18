"""PHASE 10-11 -- causal screen on the data axis, inside the existing 40k.

No new rendering.  The risk map located the damage; this asks whether changing
*what the model sees*, with everything else held fixed, moves it.

## Why low-angle and not partial visibility

Counted on `MH_TRAIN` before the arms were defined:

    <8 deg      2,587 frames    7.66%      real captures are about 94%
    V<=6        7,621 frames   22.58%
    <8 x V<=6     867 frames    2.57%

`V<=6` is the worse regime in the risk map -- 5cm5deg between 0.000 and 0.023 at
every elevation -- but it is already 22.6% of training.  A resampling screen
cannot test a coverage hypothesis on something that is not scarce, and the
earlier truncation diagnosis already pointed at label/task policy rather than
volume for that regime.  Low-angle is the only axis where the training
distribution is actually deficient against the deployment one, so that is the
single target, fixed here before either arm ran.

`<8 x V<=6` would be the sharpest cell but holds 867 frames; over a 24,000-frame
budget that is 28 repeats of the same images, which tests memorisation more than
coverage.

## Why the control is retrained rather than reused

3,000 steps at batch 8 consumes 24,000 frames from a 33,758-frame pool, so the
run never completes one pass and the *prefix* of the pool is the exposure
distribution.  The existing P0 arm streamed the pool in its natural order.
Building the treatment arm by a weighted draw would therefore change both the
weighting and the ordering.  Both arms are built by the same draw procedure here,
differing only in the weights.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_arms as MH                                             # noqa: E402
import mh_cigm as CG                                             # noqa: E402
import mh_data as MD                                             # noqa: E402
import mh_diagnose as DG                                         # noqa: E402
import mh_poseaware as PA                                        # noqa: E402
import mh_screen as MS                                           # noqa: E402
import mh_splitlate as SL                                        # noqa: E402
from mh_arms import CAP, DH, V2                                  # noqa: E402

OUT = MD.OUT
CKPT = MS.CKPT
SOURCE_RUN = PA.SOURCE_RUN
SOURCE_STEP = PA.SOURCE_STEP
STEPS = PA.STEPS
MARKS = PA.MARKS
EXPOSURES = STEPS * CAP.BATCH

ARMS = ("DCTRL", "DANGLE")
LOW_ANGLE_DEG = 8.0
# Locked before either arm ran.  Four times the natural 7.66%, which is a real
# intervention while still leaving the high-elevation regime the majority.
DANGLE_TARGET_SHARE = 0.30
DRAW_SEED = 20260819

GATE = {"geometry_gain_pct": 10.0, "fullview_degrade_pct": 5.0,
        "line_degrade_pct": 5.0}
TARGET = "low-angle (<8 deg)"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def draw_pool(arm, seed):
    """24,000 exposures, same procedure for both arms, weights differ."""
    rows = [r for r in MD.load_split() if r["split"] == "MH_TRAIN"]
    low = [r["stem"] for r in rows if r["elev"] < LOW_ANGLE_DEG]
    high = [r["stem"] for r in rows if r["elev"] >= LOW_ANGLE_DEG]
    rng = random.Random(DRAW_SEED + seed)
    if arm == "DCTRL":
        share = len(low) / len(rows)
    else:
        share = DANGLE_TARGET_SHARE
    n_low = int(round(EXPOSURES * share))
    n_high = EXPOSURES - n_low

    def take(source, count):
        out = []
        while len(out) < count:
            block = list(source)
            rng.shuffle(block)
            out.extend(block[:count - len(out)])
        return out

    pool = take(low, n_low) + take(high, n_high)
    rng.shuffle(pool)
    return pool, {"share_low_target": round(share, 4),
                  "n_low": n_low, "n_high": n_high,
                  "low_available": len(low), "high_available": len(high),
                  "low_repeats": round(n_low / max(len(low), 1), 3),
                  "high_repeats": round(n_high / max(len(high), 1), 3)}


def audit_pool(pool):
    """Realised composition, so any imbalance the draw introduced is visible."""
    meta = {r["stem"]: r for r in MD.load_split()}
    rows = [meta[s] for s in pool if s in meta]
    total = max(len(rows), 1)

    def share(key, fn):
        counts = {}
        for row in rows:
            counts[fn(row)] = counts.get(fn(row), 0) + 1
        return {str(k): round(v / total, 4) for k, v in sorted(counts.items())}

    return {
        "n_exposures": len(rows),
        "unique_frames": len({s for s in pool}),
        "elev_bins": share("elev", lambda r: (
            "<8" if r["elev"] < 8 else "8-15" if r["elev"] < 15
            else "15-30" if r["elev"] < 30 else ">=30")),
        "v_bins": share("v", lambda r: (
            "V=8" if r["v"] == 8 else "V=7" if r["v"] == 7 else "V<=6")),
        "pallet_type": share("pallet_type", lambda r: r["pallet_type"]),
        "size_bins": share("size", lambda r: (
            "small" if r["size"] < 0.25 else
            "mid" if r["size"] < 0.40 else "near/large")),
    }


def run_train(arguments):
    MS.deterministic()
    weights = MS.lambdas()
    _, populations = MD.pools()
    grid_theta, grid_rho, valid, features = MS.lattice()
    arm, seed = arguments.arm, arguments.seed
    pool, draw = draw_pool(arm, seed)
    composition = audit_pool(pool)
    model, source = PA.build_model(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=CAP.LR, weight_decay=CAP.WD)
    history = {"arm": arm, "seed": seed, "source_checkpoint": source,
               "source_step": SOURCE_STEP, "steps": STEPS,
               "marks": list(MARKS), "CONTINUATION_OPTIMIZER": "FRESH",
               "target": TARGET, "draw": draw, "composition": composition,
               "lambda_corner": weights["corner"]}
    path = OUT / f"data_resampling_{arm}_seed{seed}.json"
    directory = CKPT / f"resample_{arm}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    log(f"{arm} seed{seed}  low share "
        f"{composition['elev_bins'].get('<8', 0)}  "
        f"unique {composition['unique_frames']}")

    def mark(step):
        entry = {"step": step}
        for label, stems in populations.items():
            entry[label] = MS.evaluate(model, stems, features, grid_theta,
                                       grid_rho, valid)
        history[str(step)] = entry
        MS.write_json(path, history)
        torch.save({"arm": arm, "seed": seed, "step": step, "source": source,
                    "model": model.state_dict()},
                   directory / f"step_{step:05d}.pth")
        line = entry["D2_MH_DEV512"]["line"]
        log(f"  {arm} s{seed} @{step:5d}  angle {line['angle_median']:7.4f} "
            f"offset {line['offset_median']:7.4f}")

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
        loss = loss + weights["corner"] * MH.corner_loss(
            out["beliefs"], pack["belief"], pack["belief_valid"])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        done += 1
        if done in MARKS:
            mark(done)
    log(f"-> {path}")


def cell_metrics(model, stems, features, grid_theta, grid_rho, valid, meta):
    """Per-frame geometry and pose, kept as arrays so cells can be sliced."""
    rows = []
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
                info = meta[stem]
                rows.append({
                    "stem": stem, "elev": float(info["elev"]),
                    "v": int(info["v"]),
                    "front_rear_shift": float(np.linalg.norm(
                        (predicted[list(DG.FRONT)].mean(0)
                         - predicted[list(DG.REAR)].mean(0))
                        - (truth[list(DG.FRONT)].mean(0)
                           - truth[list(DG.REAR)].mean(0)))),
                    "affine_scale_gap": abs(fit["scale_isotropic"] - 1.0),
                    "nonaffine_rms": float(fit["nonaffine_rms"]),
                    "corner_error": float(np.linalg.norm(
                        predicted - truth, axis=1).mean()),
                    "R": error[0] if error else np.nan,
                    "t": error[1] if error else np.nan})
    return rows


def run_report(arguments):
    MS.deterministic()
    grid_theta, grid_rho, valid, features = MS.lattice()
    meta = {r["stem"]: r for r in MD.load_split()}
    populations = [p for p in arguments.populations.split(",") if p]
    result = {"step": arguments.step, "target": TARGET, "gate": GATE,
              "seeds": {}}
    for seed in (1, 2):
        block = {}
        for arm in ARMS:
            checkpoint = (CKPT / f"resample_{arm}_seed{seed}"
                          / f"step_{arguments.step:05d}.pth")
            if not checkpoint.exists():
                log(f"skip {arm} seed{seed}: {checkpoint.name} absent")
                continue
            state = torch.load(checkpoint, map_location=MH.DEV,
                               weights_only=False)
            model = SL.SplitLate("A1_CORNER_LINE")
            model.load_state_dict(state["model"])
            model.to(MH.DEV).eval()
            rows = []
            for population in populations:
                stems = json.loads(
                    (OUT / f"{population.lower()}_manifest.json").read_text()
                )["stems"]
                rows.extend(cell_metrics(model, stems, features, grid_theta,
                                         grid_rho, valid, meta))
            np.savez_compressed(
                OUT / f"data_resampling_frames_{arm}_seed{seed}.npz",
                **{k: np.array([r[k] for r in rows])
                   for k in rows[0] if k != "stem"},
                stem=np.array([r["stem"] for r in rows]))
            subsets = {
                "ALL": np.ones(len(rows), bool),
                "low-angle": np.array([r["elev"] < LOW_ANGLE_DEG
                                       for r in rows]),
                "full-view (V=8)": np.array([r["v"] == 8 for r in rows]),
                "V<=6": np.array([r["v"] <= 6 for r in rows]),
                "low-angle x V<=6": np.array(
                    [r["elev"] < LOW_ANGLE_DEG and r["v"] <= 6 for r in rows]),
            }
            entry = {}
            for name, mask in subsets.items():
                selected = [r for r, ok in zip(rows, mask) if ok]
                if not selected:
                    continue
                R = np.array([r["R"] for r in selected], float)
                t = np.array([r["t"] for r in selected], float)
                good = np.isfinite(R) & np.isfinite(t)
                entry[name] = {
                    "n": len(selected),
                    "front_rear_shift": round(float(np.median(
                        [r["front_rear_shift"] for r in selected])), 5),
                    "affine_scale_gap": round(float(np.median(
                        [r["affine_scale_gap"] for r in selected])), 5),
                    "corner_error": round(float(np.median(
                        [r["corner_error"] for r in selected])), 5),
                    "R_median": round(float(np.median(R[good])), 4)
                    if good.any() else None,
                    "t_median": round(float(np.median(t[good])), 5)
                    if good.any() else None,
                    "success_5cm5deg": round(float(
                        ((R <= 5.0) & (t <= 0.05) & good).sum()
                        / max(len(selected), 1)), 4)}
            trained = json.loads(
                (OUT / f"data_resampling_{arm}_seed{seed}.json").read_text())
            final = trained[str(arguments.step)]["D2_MH_DEV512"]["line"]
            entry["line"] = {"angle_median": final["angle_median"],
                             "offset_median": final["offset_median"]}
            block[arm] = entry
            log(f"seed{seed} {arm:<7} low-angle frs "
                f"{entry['low-angle']['front_rear_shift']:.4f} "
                f"scale {entry['low-angle']['affine_scale_gap']:.4f} "
                f"R {entry['low-angle']['R_median']:7.3f} "
                f"t {entry['low-angle']['t_median']:.4f} | "
                f"line {final['angle_median']:.4f}")
        result["seeds"][f"seed{seed}"] = block
    path = OUT / f"data_resampling_report_step{arguments.step}.json"
    path.write_text(json.dumps(result, indent=1))
    log(f"-> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "report", "pool"])
    parser.add_argument("--arm", default="DANGLE", choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step", type=int, default=3000)
    parser.add_argument("--populations",
                        default="D2_MH_DEV512,D3_MH_CONF512,"
                                "D4_THETA_CONFIRM512")
    arguments = parser.parse_args()
    if arguments.command == "pool":
        for arm in ARMS:
            pool, draw = draw_pool(arm, arguments.seed)
            composition = audit_pool(pool)
            log(f"{arm}: {draw}")
            log(f"  elev {composition['elev_bins']}")
            log(f"  V    {composition['v_bins']}")
            log(f"  size {composition['size_bins']}")
            log(f"  unique frames {composition['unique_frames']}")
        return
    {"train": run_train, "report": run_report}[arguments.command](arguments)


if __name__ == "__main__":
    main()
