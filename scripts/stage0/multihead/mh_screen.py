"""32-frame overfit, then the A0/A1/A2 quick screen, then the evaluation.

Nothing about the three arms differs except the loss.  Same initialisation, same
freeze boundary, same optimiser, learning rate, weight decay, batch, step count,
data order and seed -- `mh_wiring.T3` shows the line branch is bit-identical
across arms when the extra weights are zero, which is the strongest form of that
claim available.

Two prediction paths are scored separately and never averaged together:

    PATH-C   belief peak -> 8 corners -> PnP
    PATH-L   Hough argmax -> 12 lines -> CIGM -> 8 corners -> PnP

and a third number, `min(error_direct, error_CIGM)` per corner, is computed with
ground truth purely to answer whether the two paths fail on different corners.
It is a diagnostic.  It is not an inference result and is never reported as one.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_arms as MH                                            # noqa: E402
import mh_cigm as CG                                            # noqa: E402
import mh_data as MD                                            # noqa: E402
from mh_arms import CAP, DEV, DH, V2                            # noqa: E402

OUT = MD.OUT
CKPT = MD.ROOT / "weights/paper_s2/paper_s2_multihead"
BATCH = CAP.BATCH
OVERFIT_FRAMES = 32
OVERFIT_STEPS = 1500
OVERFIT_MARKS = (250, 500, 1000, 1500)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _json_default(value):
    """Coerce the numpy scalars that leak out of comparisons, and nothing else.

    A blanket `default=str` would let a stray tensor through as its repr and
    produce a file that parses but says nothing.  This converts the four numpy
    types that can legitimately appear and still raises on anything unexpected.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unexpected {type(value).__name__} in results")


def write_json(path, payload):
    """Write results as soon as they exist.

    The first screen run trained A0 for all 6,000 steps, evaluated every mark,
    and then lost the lot to a `np.bool_` in the final `json.dumps`.  Marks are
    now flushed as they complete, so a serialisation fault can cost at most the
    mark being written.
    """
    path.write_text(json.dumps(payload, indent=1, default=_json_default))


def deterministic():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = MH.LATE.DETERMINISTIC_WORKSPACE
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def lattice():
    grid_theta, grid_rho, valid = DH.lattice()
    return grid_theta, grid_rho, valid, DH.hypothesis_features(grid_theta, grid_rho)


def lambdas() -> dict:
    calibration = json.loads((OUT / "mh_wiring.json").read_text())["T4_LAMBDA"]
    return {"corner": calibration["lambda_corner"],
            "mask": calibration["lambda_mask"],
            "measured_ratio_corner": calibration["ratio_corner_over_line"],
            "measured_ratio_mask": calibration["ratio_mask_over_line"]}


def overfit_frames() -> list[str]:
    """Four frames from each of the eight reported strata.

    Section 14 asks for easy full-view, low-angle, truncation, partial-edge,
    small and large -- which is exactly the cross of the three axes the split
    already stratifies on, so the same definition serves both and there is no
    second, hand-picked notion of "hard" anywhere in this screen.
    """
    rows = [r for r in MD.load_split() if r["split"] == "MH_TRAIN"]
    buckets: dict[str, list[str]] = {}
    for row in sorted(rows, key=lambda r: r["stem"]):
        buckets.setdefault(row["stratum"], []).append(row["stem"])
    picked: list[str] = []
    for stratum in sorted(buckets):
        picked.extend(buckets[stratum][:OVERFIT_FRAMES // len(buckets)])
    return sorted(picked)


# --------------------------------------------------------------------------
# training


def total_loss(model, pack, features, grid_theta, grid_rho, valid, weights,
               ramp=1.0):
    out = model(pack["images"], features)
    theta_c, rho_c, support = DH.batch_rows(pack, CG.EDGES)
    target = DH.target_distribution(theta_c.reshape(-1), rho_c.reshape(-1),
                                    grid_theta, grid_rho,
                                    valid).reshape(*theta_c.shape, -1)
    parts = {"line": DH.cross_entropy(out["line_scores"], target, support, valid)}
    if "beliefs" in out:
        parts["corner"] = MH.corner_loss(out["beliefs"], pack["belief"],
                                         pack["belief_valid"])
    if "segments" in out:
        parts["mask"] = MH.mask_loss(out["segments"], pack["mask"])
    total = parts["line"]
    for name in ("corner", "mask"):
        if name in parts:
            total = total + ramp * weights[name] * parts[name]
    return total, parts, out


def train_arm(arm, pool, marks, weights, populations, tag, ramp_steps=0,
              evaluate_every_mark=True, flush_to=None, seed=None,
              split_late=False):
    grid_theta, grid_rho, valid, features = lattice()
    seed = CAP.SEED if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if split_late:
        # Long confirmation of E3 on the same 25k budget as A0/A1/A2, so the
        # candidate is judged from scratch rather than as a continuation.
        import mh_splitlate as SL
        model = SL.SplitLate(arm)
    else:
        model = MH.MultiHeadModel(arm)
    optimiser = torch.optim.AdamW(model.trainable_parameters(), lr=CAP.LR,
                                  weight_decay=CAP.WD)
    history, running, done = {}, {"line": [], "corner": [], "mask": []}, 0
    started = time.time()
    for chunk, _ in V2.step_schedule(pool, max(marks), BATCH):
        model.train()
        pack = MD.load_pack(chunk)
        ramp = 1.0 if ramp_steps <= 0 else min(1.0, (done + 1) / ramp_steps)
        loss, parts, _ = total_loss(model, pack, features, grid_theta, grid_rho,
                                    valid, weights, ramp)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        for name, value in parts.items():
            running[name].append(float(value.detach()))
        done += 1
        if done in marks:
            entry = {"step": done, "ramp": ramp,
                     "elapsed_s": round(time.time() - started, 1)}
            for name, values in running.items():
                if values:
                    entry[f"train_{name}_last250"] = float(np.mean(values[-250:]))
            if evaluate_every_mark:
                for label, stems in populations.items():
                    entry[label] = evaluate(model, stems, features, grid_theta,
                                            grid_rho, valid)
                    line = entry[label]["line"]
                    corner = entry[label]["corner"]
                    log(f"  {tag} @{done:5d} {label:<14} angle med "
                        f"{line['angle_median']:7.4f} p90 {line['angle_p90']:7.3f}"
                        f" | offset med {line['offset_median']:7.4f}"
                        f" | cornerC {corner.get('direct_cell_median', '-')}"
                        f" cornerL {corner['cigm_cell_median']}")
            directory = CKPT / tag
            directory.mkdir(parents=True, exist_ok=True)
            torch.save({"arm": arm, "tag": tag, "step": done,
                        "weights": weights, "seed": seed,
                        "model": model.state_dict()},
                       directory / f"step_{done:05d}.pth")
            history[str(done)] = entry
            if flush_to is not None:
                write_json(flush_to, history)
    return history, model


# --------------------------------------------------------------------------
# evaluation


def _decode_peaks(belief: torch.Tensor) -> np.ndarray:
    """argmax cell refined by its 3x3 neighbourhood, per channel, in 50-grid."""
    maps = belief.detach().float().cpu().numpy()
    batch, channels, height, width = maps.shape
    out = np.zeros((batch, channels, 2))
    for b in range(batch):
        for c in range(channels):
            plane = maps[b, c]
            iy, ix = np.unravel_index(int(np.argmax(plane)), plane.shape)
            y0, y1 = max(0, iy - 1), min(height, iy + 2)
            x0, x1 = max(0, ix - 1), min(width, ix + 2)
            patch = np.clip(plane[y0:y1, x0:x1], 0, None)
            total = patch.sum()
            if total <= 1e-9:
                out[b, c] = (ix, iy)
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            out[b, c] = ((patch * xs).sum() / total, (patch * ys).sum() / total)
    return out


def _peak_confidence(belief: torch.Tensor) -> np.ndarray:
    flat = belief.detach().float().flatten(2)
    top = flat.topk(2, dim=-1).values
    return torch.stack([top[..., 0], top[..., 0] - top[..., 1]], -1).cpu().numpy()


def _hough_confidence(scores: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    masked = scores.masked_fill(~valid[None, None], -float("inf")).float()
    probability = torch.softmax(masked, -1)
    top = probability.topk(2, dim=-1).values
    entropy = -(probability.clamp_min(1e-12).log() * probability).sum(-1)
    return torch.stack([top[..., 0], top[..., 0] - top[..., 1], entropy],
                       -1).cpu().numpy()


@torch.no_grad()
def evaluate(model, stems, features, grid_theta, grid_rho, valid) -> dict:
    model.eval()
    rows = []
    for start in range(0, len(stems), BATCH):
        chunk = stems[start:start + BATCH]
        if not chunk:
            continue
        pack = MD.load_pack(chunk)
        out = model(pack["images"], features)
        theta_hat, rho_hat = DH.decode(out["line_scores"], grid_theta, grid_rho,
                                       valid)
        theta_gt, rho_gt, support = DH.batch_rows(pack, CG.EDGES)
        corners_l, residual, condition = CG.cigm_corners(theta_hat, rho_hat)
        corners_l = corners_l.detach().cpu().numpy()
        hough_conf = _hough_confidence(out["line_scores"], valid)
        direct = peak = None
        if "beliefs" in out:
            direct = _decode_peaks(out["beliefs"][-1][:, :9])
            peak = _peak_confidence(out["beliefs"][-1][:, :9])
        mask_iou = None
        if "segments" in out:
            predicted = torch.sigmoid(out["segments"][-1]) > 0.5
            truth = pack["mask"] > 0.5
            union = (predicted | truth).flatten(1).sum(-1).float()
            intersection = (predicted & truth).flatten(1).sum(-1).float()
            mask_iou = (intersection / union.clamp_min(1.0)).cpu().numpy()
        for index, stem in enumerate(chunk):
            angle, offset = DH.measure(theta_hat[index], rho_hat[index],
                                       theta_gt[index], rho_gt[index])
            rows.append(_frame_row(stem, pack, index, angle, offset,
                                   support[index].cpu().numpy(), corners_l[index],
                                   None if direct is None else direct[index],
                                   None if peak is None else peak[index],
                                   hough_conf[index], residual[index].cpu().numpy(),
                                   condition[index].cpu().numpy(),
                                   None if mask_iou is None else float(mask_iou[index])))
    return _aggregate(rows)


def _frame_row(stem, pack, index, angle, offset, support, corners_l, direct,
               peak, hough_conf, residual, condition, mask_iou) -> dict:
    label = MD.read_label(stem)
    width, height = pack["resolution"][index]
    truth = pack["grid"][index]
    model_points = CG.object_points(label)
    camera = CG.intrinsics(label)
    rotation_gt, translation_gt = CG.gt_pose(label)
    pixels_gt = np.asarray(label["objects"][0]["projected_cuboid"], float)

    error_l = np.linalg.norm(corners_l - truth[:8], axis=1)
    # Whether each GT corner falls inside the belief grid at all.  Without this
    # the corner medians pool two populations that mean different things: a
    # heatmap cannot place a peak outside its own grid, so scoring those corners
    # against PATH-C measures an impossibility, while CIGM reaches them by
    # intersection.  That difference is the complementarity hypothesis, so it has
    # to be separable rather than averaged away.
    in_grid = [bool(0 <= x < MD.GRID and 0 <= y < MD.GRID) for x, y in truth[:8]]
    row = {"stem": stem,
           "angle": angle.tolist(), "offset": offset.tolist(),
           "support": support.astype(bool).tolist(),
           "in_grid": in_grid,
           "cigm_cell": error_l.tolist(),
           "cigm_residual": residual.tolist(),
           "cigm_condition": condition.tolist(),
           "hough_peak": hough_conf[:, 0].tolist(),
           "hough_margin": hough_conf[:, 1].tolist(),
           "hough_entropy": hough_conf[:, 2].tolist(),
           "mask_iou": mask_iou}

    pose_l = CG.solve(model_points, CG.grid_to_pixels(corners_l, width, height),
                      camera)
    row["pose_L"] = _pose_row(pose_l, model_points, camera, pixels_gt,
                              rotation_gt, translation_gt)
    if direct is not None:
        error_c = np.linalg.norm(direct[:8] - truth[:8], axis=1)
        row["direct_cell"] = error_c.tolist()
        row["direct_centroid_cell"] = float(
            np.linalg.norm(direct[8] - truth[8]))
        row["peak"] = peak[:, 0].tolist()
        row["peak_margin"] = peak[:, 1].tolist()
        pose_c = CG.solve(model_points,
                          CG.grid_to_pixels(direct[:8], width, height), camera)
        row["pose_C"] = _pose_row(pose_c, model_points, camera, pixels_gt,
                                  rotation_gt, translation_gt)
        row["oracle_cell"] = np.minimum(error_c, error_l).tolist()
        row["direct_wins"] = (error_c < error_l).tolist()
    return row


def _pose_row(pose, model_points, camera, pixels_gt, rotation_gt, translation_gt):
    if pose is None:
        return {"solved": False}
    error = CG.pose_error(pose, rotation_gt, translation_gt)
    return {"solved": True, "R_deg": error[0], "t_m": error[1],
            "reproj_px": CG.reprojection(model_points, pose, camera, pixels_gt)}


def _quantiles(values, keys=("median", "p90")):
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                       float)
    if array.size == 0:
        return {k: None for k in keys}
    out = {}
    if "median" in keys:
        out["median"] = float(np.median(array))
    if "p90" in keys:
        out["p90"] = float(np.percentile(array, 90))
    return out


def _aggregate(rows) -> dict:
    """Per-role numbers are pooled over supported roles only, as the loss is."""
    angles, offsets = [], []
    for row in rows:
        for role, supported in enumerate(row["support"]):
            if supported:
                angles.append(row["angle"][role])
                offsets.append(row["offset"][role])
    cigm = [v for row in rows for v in row["cigm_cell"]]
    inside = [v for row in rows
              for v, ok in zip(row["cigm_cell"], row["in_grid"]) if ok]
    outside = [v for row in rows
               for v, ok in zip(row["cigm_cell"], row["in_grid"]) if not ok]
    result = {
        "n_frames": len(rows),
        "line": {"angle_median": _quantiles(angles)["median"],
                 "angle_p90": _quantiles(angles)["p90"],
                 "offset_median": _quantiles(offsets)["median"],
                 "offset_p90": _quantiles(offsets)["p90"],
                 "supported_roles": len(angles)},
        "corner": {"cigm_cell_median": _round(_quantiles(cigm)["median"]),
                   "cigm_cell_p90": _round(_quantiles(cigm)["p90"]),
                   "cigm_in_grid_median": _round(_quantiles(inside)["median"]),
                   "cigm_off_grid_median": _round(_quantiles(outside)["median"]),
                   "corners_off_grid": len(outside),
                   "corners_in_grid": len(inside)},
        "pose_L": _pose_summary(rows, "pose_L"),
    }
    if rows and "direct_cell" in rows[0]:
        direct = [v for row in rows for v in row["direct_cell"]]
        oracle = [v for row in rows for v in row["oracle_cell"]]
        wins = [w for row in rows for w in row["direct_wins"]]
        direct_in = [v for row in rows
                     for v, ok in zip(row["direct_cell"], row["in_grid"]) if ok]
        direct_out = [v for row in rows
                      for v, ok in zip(row["direct_cell"], row["in_grid"]) if not ok]
        result["corner"]["direct_cell_median"] = _round(_quantiles(direct)["median"])
        result["corner"]["direct_cell_p90"] = _round(_quantiles(direct)["p90"])
        result["corner"]["direct_in_grid_median"] = _round(
            _quantiles(direct_in)["median"])
        result["corner"]["direct_off_grid_median"] = _round(
            _quantiles(direct_out)["median"])
        result["corner"]["oracle_min_cell_median"] = _round(
            _quantiles(oracle)["median"])
        result["corner"]["direct_win_fraction"] = round(float(np.mean(wins)), 4)
        result["corner"]["centroid_cell_median"] = _round(_quantiles(
            [row["direct_centroid_cell"] for row in rows])["median"])
        result["pose_C"] = _pose_summary(rows, "pose_C")
    ious = [row["mask_iou"] for row in rows if row.get("mask_iou") is not None]
    if ious:
        result["mask"] = {"iou_median": _round(_quantiles(ious)["median"]),
                          "iou_p10": _round(float(np.percentile(ious, 10)))}
    result["rows"] = rows
    return result


def _pose_summary(rows, key) -> dict:
    entries = [row[key] for row in rows if key in row]
    solved = [e for e in entries if e.get("solved")]
    if not entries:
        return {}
    # Success is counted over every frame, not over the frames that solved.  A
    # method that refuses half the frames and nails the rest must not read as
    # better than one that answers all of them.
    hits = sum(1 for e in solved if e["R_deg"] <= 5.0 and e["t_m"] <= 0.05)
    return {"solve_rate": round(len(solved) / len(entries), 4),
            "R_deg_median": _round(_quantiles([e["R_deg"] for e in solved])["median"]),
            "R_deg_p90": _round(_quantiles([e["R_deg"] for e in solved])["p90"]),
            "t_m_median": _round(_quantiles([e["t_m"] for e in solved])["median"]),
            "reproj_px_median": _round(
                _quantiles([e["reproj_px"] for e in solved])["median"]),
            "success_5cm5deg": round(hits / len(entries), 4)}


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


# --------------------------------------------------------------------------


def run_overfit(arguments):
    deterministic()
    weights = lambdas()
    pool = overfit_frames()
    populations = {"OVERFIT32": pool}
    results = {"frames": pool, "lambdas": weights, "marks": list(OVERFIT_MARKS)}
    for arm in MH.ARMS:
        log(f"overfit {arm}  n={len(pool)}")
        history, _ = train_arm(arm, pool, OVERFIT_MARKS, weights, populations,
                               f"overfit_{arm}")
        for entry in history.values():
            entry["OVERFIT32"].pop("rows", None)
        results[arm] = history
    write_json(OUT / "mh_overfit32.json", results)
    log(f"-> {OUT / 'mh_overfit32.json'}")


def run_reference(arguments):
    """Where the checkpoint already is, before a single optimiser step.

    Needed because the line head starts at chance and its gradient reshapes the
    shared block within a few hundred steps, which drags the warm-started corner
    and mask heads down before they recover.  Without a step-0 row, "A1's corner
    head reaches X" cannot be read against anything, and an arm that ended worse
    than its own initialisation would look like a result.
    """
    deterministic()
    grid_theta, grid_rho, valid, features = lattice()
    _, populations = MD.pools()
    torch.manual_seed(CAP.SEED)
    np.random.seed(CAP.SEED)
    model = MH.MultiHeadModel("A2_CORNER_LINE_MASK")
    entry = {"step": 0, "note": "no optimiser step; heads as loaded"}
    for label, stems in populations.items():
        entry[label] = evaluate(model, stems, features, grid_theta, grid_rho, valid)
        entry[label].pop("rows", None)
        corner = entry[label]["corner"]
        log(f"  reference @    0 {label:<14} cornerC {corner['direct_cell_median']}"
            f" (in-grid {corner['direct_in_grid_median']})"
            f" cornerL {corner['cigm_cell_median']}"
            f" | mask IoU {entry[label].get('mask', {}).get('iou_median')}")
    write_json(OUT / "mh_reference_step0.json", entry)
    log(f"-> {OUT / 'mh_reference_step0.json'}")


def run_screen(arguments):
    deterministic()
    weights = lambdas()
    train, populations = MD.pools()
    if arguments.final_pool:
        # FINAL_SYNTH_TRAIN_V1: the whole 40,000, MH_DEV folded back in.  This
        # is only legitimate because the final claim moved to REAL IN-HOUSE
        # DEV/TEST, so there is no synthetic holdout left to protect.  Any
        # evaluation on MH_DEV from such a checkpoint is in-train, and
        # `--no-eval` is expected alongside this flag for that reason.
        train = [r["stem"] for r in MD.load_split()]
        populations = {}
    pool = train[:arguments.pool] if arguments.pool else train
    if arguments.seed != CAP.SEED:
        # A second seed varies the initialisation and the data order together.
        # Varying only the initialisation would answer a narrower question than
        # the one that matters, which is whether A1's margin survives a
        # different draw of everything the run is free to draw.
        import random as _random
        pool = list(pool)
        _random.Random(arguments.seed).shuffle(pool)
    marks = tuple(int(m) for m in arguments.marks.split(","))
    results = {"lambdas": weights, "pool": len(pool), "marks": list(marks),
               "pool_source": ("FINAL_SYNTH_TRAIN_V1 (BROAD 40,000)"
                               if arguments.final_pool else "MH_TRAIN"),
               "evaluated_every_mark": not arguments.no_eval,
               "batch": BATCH, "lr": CAP.LR, "weight_decay": CAP.WD,
               "seed": arguments.seed, "ramp_steps": arguments.ramp,
               "split_sha256": json.loads(
                   MD.CONTRACT_JSON.read_text())["split_sha256"]}
    for arm in (MH.ARMS if arguments.arm == "all" else (arguments.arm,)):
        log(f"screen {arm}  pool={len(pool)} marks={marks}")
        suffix = "" if arguments.seed == CAP.SEED else f"_seed{arguments.seed}"
        if arguments.label:
            suffix = f"_{arguments.label}_seed{arguments.seed}"
        path = OUT / f"mh_screen_{arm}{suffix}.json"
        history, _ = train_arm(arm, pool, marks, weights, populations,
                               f"screen_{arm}{suffix}", ramp_steps=arguments.ramp,
                               evaluate_every_mark=not arguments.no_eval,
                               flush_to=path, seed=arguments.seed,
                               split_late=arguments.split_late)
        # The per-frame rows live in the per-arm file; meta stays small enough to
        # read, and records only what was held constant across the arms.
        results.setdefault("arms", {})[arm] = {
            mark: {key: ({k: v for k, v in value.items() if k != "rows"}
                         if isinstance(value, dict) else value)
                   for key, value in entry.items()}
            for mark, entry in history.items()}
        log(f"-> {path}")
    suffix = "" if arguments.seed == CAP.SEED else f"_seed{arguments.seed}"
    if arguments.label:
        suffix = f"_{arguments.label}_seed{arguments.seed}"
    write_json(OUT / f"mh_screen_meta{suffix}.json", results)
    log(f"-> {OUT / f'mh_screen_meta{suffix}.json'}")


def run_evaluate(arguments):
    """Score an arm from its saved checkpoints instead of retraining it.

    Training is bit-reproducible here, so re-running would give the same weights
    -- this exists so a lost results file costs minutes rather than the hour the
    arm took, and so any mark can be re-scored later with a changed metric
    without touching the trajectory that produced it.
    """
    deterministic()
    grid_theta, grid_rho, valid, features = lattice()
    _, populations = MD.pools()
    arm = arguments.arm
    directory = CKPT / f"screen_{arm}"
    checkpoints = sorted(directory.glob("step_*.pth"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints under {directory}")
    path = OUT / f"mh_screen_{arm}.json"
    history = json.loads(path.read_text()) if path.exists() else {}
    for checkpoint in checkpoints:
        step = int(checkpoint.stem.split("_")[1])
        state = torch.load(checkpoint, map_location=DEV, weights_only=False)
        torch.manual_seed(CAP.SEED)
        np.random.seed(CAP.SEED)
        model = MH.MultiHeadModel(arm)
        model.load_state_dict(state["model"])
        entry = {"step": step, "from_checkpoint": checkpoint.name,
                 "weights": state.get("weights")}
        for label, stems in populations.items():
            entry[label] = evaluate(model, stems, features, grid_theta,
                                    grid_rho, valid)
            line = entry[label]["line"]
            corner = entry[label]["corner"]
            log(f"  {arm} @{step:5d} {label:<14} angle med "
                f"{line['angle_median']:7.4f} | offset med "
                f"{line['offset_median']:7.4f} | cornerC "
                f"{corner.get('direct_cell_median', '-')} cornerL "
                f"{corner['cigm_cell_median']}")
        history[str(step)] = entry
        write_json(path, history)
        del model
    log(f"-> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["reference", "overfit", "screen", "evaluate"])
    parser.add_argument("--arm", default="all")
    parser.add_argument("--pool", type=int, default=0, help="0 = whole train")
    parser.add_argument("--marks", default="500,1000,2000,3000,4000")
    parser.add_argument("--ramp", type=int, default=500)
    parser.add_argument("--seed", type=int, default=CAP.SEED)
    parser.add_argument("--split-late", action="store_true",
                        help="two late blocks, one per branch (E3 architecture)")
    parser.add_argument("--final-pool", action="store_true",
                        help="train on all 40,000 BROAD frames "
                             "(FINAL_SYNTH_TRAIN_V1). Off by default so every "
                             "historical run reproduces unchanged.")
    parser.add_argument("--no-eval", action="store_true",
                        help="skip the per-mark population evaluation. "
                             "Required with --final-pool: those populations "
                             "are inside the training pool.")
    parser.add_argument("--label", default="",
                        help="suffix for outputs, so a longer schedule cannot "
                             "overwrite the 6,000-step screen it is testing")
    arguments = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"reference": run_reference, "overfit": run_overfit,
     "screen": run_screen, "evaluate": run_evaluate}[arguments.command](arguments)


if __name__ == "__main__":
    main()
