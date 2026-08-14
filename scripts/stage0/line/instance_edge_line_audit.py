"""Learned 12-edge field localization mechanism audit.  Read-only: no training,
no optimizer, no checkpoint written.

The learned head reaches 44% of corners within 20px on synthetic and 2.5% on
eval56, while the same decoder on ground-truth geometry reaches 98.7%.  Channel
identity is already known to be intact, so the question left is where the
localization is lost: whether the correct physical line is present in the
predicted field and merely not selected, whether only its extent is wrong, or
whether it is not there at all -- and whether that differs between synthetic and
real.

    python scripts/stage0/line/instance_edge_line_audit.py all
    python scripts/stage0/line/instance_edge_line_audit.py status
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
STAGE0 = ROOT / "scripts/stage0"
for _extra in (STAGE0, ROOT / "Deep_Object_Pose/common", ROOT / "Deep_Object_Pose/train",
               ROOT / "challenge/scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

RUNNER_PATH = STAGE0 / "line" / "instance_edge_learnability.py"
_spec = importlib.util.spec_from_file_location("IEL", RUNNER_PATH)
IEL = importlib.util.module_from_spec(_spec)
sys.modules["IEL"] = IEL
_spec.loader.exec_module(IEL)

import instance_edge_hypotheses as IEH        # noqa: E402
import instance_edge_topology as IET          # noqa: E402

BASE = IEL.RESULT_ROOT
OUT = BASE / "line_hypothesis_audit"
REPORTS = (ROOT / "_docs/audits/eval56_summary/canonical_corner_audit"
           / "instance_edge_learnability/line_hypothesis_audit")

# ---------------------------------------------------------------------------
# Frozen before any result.  None of these may move in response to an outcome.
# ---------------------------------------------------------------------------
AUDIT_ARMS = ("L12-F50", "L12-MS")          # L5 is structurally non-generative
SETS = ("val", "untouched", "eval56", "wood")
UNTOUCHED_STRIDE = 4        # declared coverage bound, never a silent cap
GRID = IEL.GRID_12
CORNER_OK_PX = IEL.CORNER_OK_PX
PHASES = ("lock", "profile", "extract", "policy", "match", "triplet",
          "selection", "taxonomy", "decide", "report")

# Phase J gates, all fixed up front.
GATE_J1 = {"edge_top5_strict_line": 0.80, "edge_top5_strict_segment": 0.65,
           "corner_triplet_top5": 0.60, "oracle_corner_le20": 0.60}
GATE_J2 = {"edge_top5_strict_line": 0.50, "edge_top5_strict_segment": 0.35,
           "corner_triplet_top5": 0.25, "oracle_corner_le20": 0.25}
GATE_J3 = {"oracle_over_s0": 0.20, "s1_over_s0": 0.10}
GATE_J4 = {"top5_strict_line": 0.60, "top5_strict_segment": 0.35, "l1_share": 0.40}
GATE_J5 = {"synthetic_strict": 0.70, "canonical_strict": 0.30,
           "synthetic_oracle": 0.50, "canonical_oracle": 0.15}
GATE_J6 = {"edge_top5_strict_line": 0.60, "corner_triplet_top5": 0.40,
           "oracle_corner_le20": 0.40}

log = IEL.log
atomic_write = IEL.atomic_write
device = IEL.device


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# frames and predictions
# ---------------------------------------------------------------------------
def load_frames(topology: dict[str, Any], label: str) -> list:
    if label in ("val", "untouched"):
        manifest = json.loads(
            (IEL.PPD_ROOT / f"ppd_{label}_manifest.json").read_text("utf-8"))
        files = [f["file"] for f in manifest["frames"]]
        if label == "untouched":
            files = files[::UNTOUCHED_STRIDE]
        return [f for f in (IEL.load_synthetic(name, topology, want5=False)
                            for name in files) if f is not None]
    return IEL.load_canonical(topology, label)


def gt_segments(frame, topology) -> list:
    width, height = frame.size
    segments, _ = IET.clipped_edges_in_grid(frame.corners, topology,
                                            width, height, GRID)
    return segments


@torch.no_grad()
def predict_maps(model, encoder, frames, batch: int = 16) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch]
        images = IEL.normalise(np.stack([f.image for f in chunk])).to(device)
        high, low = encoder.taps(images)
        output.append(torch.sigmoid(model(high, low)).float().cpu().numpy())
    return np.concatenate(output)


# ---------------------------------------------------------------------------
# Phase A/B -- locks
# ---------------------------------------------------------------------------
def phase_lock(state) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    complete = json.loads((BASE / "COMPLETE").read_text("utf-8"))
    synthetic = json.loads((BASE / "synthetic_results.json").read_text("utf-8"))
    previous = json.loads((BASE / "input_lock.json").read_text("utf-8"))

    checkpoints = {}
    for arm in AUDIT_ARMS:
        entry = synthetic["arms"][arm]
        for seed, block in entry["seeds"].items():
            path = ROOT / block["checkpoint"]
            digest = IEL.sha256_file(path)
            if digest != block["checkpoint_sha256"]:
                raise RuntimeError(f"BLOCKED: checkpoint changed since selection: {path}")
            checkpoints[f"{arm}|{seed}"] = {
                "path": block["checkpoint"], "sha256": digest,
                "selected_epoch": block["selected_epoch"],
                "selection_basis": "synthetic validation only"}
        checkpoints[f"{arm}|selected"] = {"seed": entry["best_seed"]}

    a1_now = IEL.sha256_file(IEL.A1_CKPT)
    lock = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "head": git("rev-parse", "HEAD"),
        "origin_main": git("rev-parse", "origin/main"),
        "git_status": git("status", "--porcelain"),
        "audit_type": "read-only mechanism audit",
        "training_runs": 0, "optimizers_created": 0, "checkpoints_written": 0,
        "state_all_done": all(v == "DONE" for v in complete["phases"].values()),
        "complete_decision": complete["final_decision"],
        "mechanism_labels": complete["mechanism_labels"],
        "tests": complete["tests"]["full_tests"],
        "a1_sha256": a1_now,
        "a1_sha256_unchanged": a1_now == previous["a1_sha256"],
        "a1_training_steps": 0, "a1_parameter_delta": 0,
        "final_test_open_count": 0,
        "sealed_tokens": list(IEL.SEALED),
        "arms": list(AUDIT_ARMS),
        "l5_excluded": "the five-class representation is structurally non-generative "
                       "(O5C collapses eight corners onto two points)",
        "sets": list(SETS),
        "checkpoints": checkpoints,
    }
    atomic_write(OUT / "input_lock.json", json.dumps(lock, indent=1))
    atomic_write(OUT / "checkpoint_lock.json", json.dumps(checkpoints, indent=1))
    log(f"[lock] state all DONE {lock['state_all_done']}  A1 unchanged "
        f"{lock['a1_sha256_unchanged']}  arms {AUDIT_ARMS}")
    if not (lock["state_all_done"] and lock["a1_sha256_unchanged"]):
        raise RuntimeError("BLOCKED: prior run integrity failed")
    return lock


# ---------------------------------------------------------------------------
# Phase C -- raw field profile
# ---------------------------------------------------------------------------
def phase_profile(state) -> dict[str, Any]:
    topology = IEL.load_topology()
    encoder = IEL.FrozenEncoder().to(device)
    synthetic = json.loads((BASE / "synthetic_results.json").read_text("utf-8"))
    rows = []
    for arm in AUDIT_ARMS:
        seed = int(synthetic["arms"][arm]["best_seed"])
        model, _, _ = IEL.load_selected(arm, seed, encoder)
        for label in SETS:
            frames = load_frames(topology, label)
            maps = predict_maps(model, encoder, frames)
            for index, frame in enumerate(frames):
                segments = gt_segments(frame, topology)
                fields, available = IET.distance_fields_from_segments(segments, GRID)
                for channel in range(12):
                    probability = maps[index, channel]
                    distance = fields[channel]
                    gt_band = distance <= IEH.GT_BAND_CELLS
                    near = ((distance >= IEH.NEAR_BACKGROUND_CELLS[0])
                            & (distance <= IEH.NEAR_BACKGROUND_CELLS[1]))
                    far = distance > IEH.NEAR_BACKGROUND_CELLS[1]
                    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
                    entropy = float(-(clipped * np.log(clipped)
                                      + (1 - clipped) * np.log(1 - clipped)).mean())
                    row = {
                        "arm": arm, "seed": seed, "set": label,
                        "frame": index, "channel": channel,
                        "edge_available": bool(available[channel]),
                        "p_min": float(probability.min()),
                        "p_mean": float(probability.mean()),
                        "p_max": float(probability.max()),
                        "entropy": entropy,
                        "gt_band_mean": float(probability[gt_band].mean())
                        if gt_band.any() else None,
                        "near_bg_mean": float(probability[near].mean())
                        if near.any() else None,
                        "far_bg_mean": float(probability[far].mean())
                        if far.any() else None,
                    }
                    for threshold in (0.3, 0.5, 0.7, 0.9):
                        row[f"area_{threshold}"] = int((probability >= threshold).sum())
                    if row["gt_band_mean"] is not None and row["far_bg_mean"]:
                        row["gt_over_bg"] = row["gt_band_mean"] / max(row["far_bg_mean"], 1e-9)
                    else:
                        row["gt_over_bg"] = None
                    rows.append(row)
            log(f"[profile] {arm} s{seed} {label}: {len(frames)} frames")
            del frames, maps
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "field_profile.parquet")
    summary = (frame[frame.edge_available]
               .groupby(["arm", "set"])
               .agg(p_mean=("p_mean", "mean"), gt_band=("gt_band_mean", "mean"),
                    near_bg=("near_bg_mean", "mean"), far_bg=("far_bg_mean", "mean"),
                    gt_over_bg=("gt_over_bg", "median"), entropy=("entropy", "mean"),
                    area_05=("area_0.5", "mean"), area_03=("area_0.3", "mean"))
               .reset_index())
    summary.to_csv(OUT / "field_profile.csv", index=False)
    log("[profile]\n" + summary.to_string(index=False))
    return {"rows": len(rows)}


# ---------------------------------------------------------------------------
# Phase D/E -- extraction and policy
# ---------------------------------------------------------------------------
def candidates_for_frame(probability_frame: np.ndarray, name: str,
                         parameter: float) -> list[list[dict[str, Any]]]:
    return [IEH.extract(name, parameter, probability_frame[channel])
            for channel in range(12)]


def score_policy(frames, maps, topology, name, parameter) -> dict[str, Any]:
    incidence = {int(k): v for k, v in topology["corner_edge_incidence"].items()}
    edge_top5 = edge_total = 0
    triplet_top5 = triplet_total = 0
    oracle_good = oracle_total = 0
    for index, frame in enumerate(frames):
        segments = gt_segments(frame, topology)
        candidates = candidates_for_frame(maps[index], name, parameter)
        best_match: list[Optional[dict[str, Any]]] = [None] * 12
        strict_available = [False] * 12
        for channel in range(12):
            if segments[channel] is None:
                continue
            edge_total += 1
            a, b = segments[channel]
            best = None
            for candidate in candidates[channel]:
                metrics = IEH.match_metrics(candidate, a, b)
                if metrics is None:
                    continue
                key = (not metrics["strict_line"], metrics["angle_err_deg"]
                       + metrics["offset_cells"])
                if best is None or key < best[0]:
                    best = (key, candidate, metrics)
            if best is not None and best[2]["strict_line"]:
                edge_top5 += 1
                strict_available[channel] = True
                best_match[channel] = best[1]
        width, height = frame.size
        for corner in range(8):
            truth = frame.corners[corner]
            if truth is None:
                continue
            triplet_total += 1
            incident = incidence[corner]
            if not all(strict_available[k] for k in incident):
                continue
            triplet_top5 += 1
            lines = [(best_match[k]["theta"], best_match[k]["rho"]) for k in incident]
            solution = IEH.intersect_lines(lines)
            oracle_total += 1
            if solution is None:
                continue
            x = solution["point"][0] * width / GRID
            y = solution["point"][1] * height / GRID
            if np.hypot(x - truth[0], y - truth[1]) <= CORNER_OK_PX:
                oracle_good += 1
    return {"extractor": name, "parameter": parameter,
            "edge_top5_strict_line": edge_top5 / max(edge_total, 1),
            "corner_triplet_top5": triplet_top5 / max(triplet_total, 1),
            "oracle_corner_le20": oracle_good / max(oracle_total, 1),
            "edges": edge_total, "corners": triplet_total}


def phase_policy(state) -> dict[str, Any]:
    """Extractor and threshold are chosen on synthetic validation only."""
    topology = IEL.load_topology()
    encoder = IEL.FrozenEncoder().to(device)
    synthetic = json.loads((BASE / "synthetic_results.json").read_text("utf-8"))
    arm = "L12-MS"
    seed = int(synthetic["arms"][arm]["best_seed"])
    model, _, _ = IEL.load_selected(arm, seed, encoder)
    frames = load_frames(topology, "val")
    limit = int(os.environ.get("AUDIT_POLICY_FRAMES", "300"))
    frames = frames[:limit]
    maps = predict_maps(model, encoder, frames)
    log(f"[policy] selection on synthetic validation only: {arm} s{seed}, "
        f"{len(frames)} frames (declared bound, not a silent cap)")

    rows = []
    for name, (_, parameters) in IEH.EXTRACTORS.items():
        for parameter in parameters:
            began = time.time()
            row = score_policy(frames, maps, topology, name, parameter)
            row["seconds"] = round(time.time() - began, 1)
            rows.append(row)
            log(f"[policy] {name:18s} {parameter:>5}  top5-strict-line "
                f"{100*row['edge_top5_strict_line']:5.1f}%  triplet "
                f"{100*row['corner_triplet_top5']:5.1f}%  oracle<=20px "
                f"{100*row['oracle_corner_le20']:5.1f}%  ({row['seconds']}s)")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "policy_selection.csv", index=False)

    order = {"E1_COMPONENT_TLS": 0, "E2_WEIGHTED_HOUGH": 1, "E3_TOP_MASS_TLS": 2}
    threshold_order = {0.5: 0, 0.7: 1, 0.3: 2, 0.9: 3}
    best = min(rows, key=lambda r: (
        -round(r["edge_top5_strict_line"], 6),
        -round(r["corner_triplet_top5"], 6),
        -round(r["oracle_corner_le20"], 6),
        order[r["extractor"]],
        threshold_order.get(r["parameter"], 9)))
    policy = {
        "extractor": best["extractor"], "parameter": best["parameter"],
        "top_k": IEH.TOP_K,
        "selection_set": "synthetic validation", "selection_arm": f"{arm} seed{seed}",
        "selection_frames": len(frames),
        "primary_metric": "edge top-5 strict infinite-line availability",
        "tie_breaks": ["corner triplet availability", "oracle-selected corner <=20px",
                       "simplest extractor E1 > E2 > E3",
                       "threshold 0.5 > 0.7 > 0.3 > 0.9"],
        "untouched_stride": UNTOUCHED_STRIDE,
        "constants": {"strict_angle_deg": IEH.STRICT_ANGLE_DEG,
                      "strict_offset_cells": IEH.STRICT_OFFSET_CELLS,
                      "loose_angle_deg": IEH.LOOSE_ANGLE_DEG,
                      "loose_offset_cells": IEH.LOOSE_OFFSET_CELLS,
                      "strict_overlap": IEH.STRICT_OVERLAP,
                      "loose_overlap": IEH.LOOSE_OVERLAP,
                      "hough_support_cells": IEH.HOUGH_SUPPORT_CELLS,
                      "condition_max": IEH.CONDITION_MAX},
        "all_arms": rows,
    }
    policy["policy_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in policy.items() if k != "all_arms"},
                   sort_keys=True).encode()).hexdigest()
    atomic_write(OUT / "extraction_policy.json", json.dumps(policy, indent=1))
    log(f"[policy] LOCKED {policy['extractor']} @ {policy['parameter']} "
        f"sha {policy['policy_sha256'][:12]}")
    return policy


# ---------------------------------------------------------------------------
# vectorised triplet selection
# ---------------------------------------------------------------------------
def solve_triplets(normals: np.ndarray, rhos: np.ndarray) -> tuple:
    """Least-squares intersection for a batch of three-line combinations.

    ``normals`` is (M, 3, 2) and ``rhos`` is (M, 3).  Solving the 2x2 normal
    equations in closed form keeps this vectorised; the condition number of the
    3x2 design matrix is the square root of the condition number of A^T A.
    """
    ata = np.einsum("mij,mik->mjk", normals, normals)
    atb = np.einsum("mij,mi->mj", normals, rhos)
    trace = ata[:, 0, 0] + ata[:, 1, 1]
    determinant = ata[:, 0, 0] * ata[:, 1, 1] - ata[:, 0, 1] * ata[:, 1, 0]
    discriminant = np.maximum(trace ** 2 - 4 * determinant, 0.0)
    root = np.sqrt(discriminant)
    high = 0.5 * (trace + root)
    low = 0.5 * (trace - root)
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = np.sqrt(np.where(low > 1e-12, high / low, np.inf))
        inverse = np.stack([
            np.stack([ata[:, 1, 1], -ata[:, 0, 1]], axis=1),
            np.stack([-ata[:, 1, 0], ata[:, 0, 0]], axis=1)], axis=1)
        point = np.einsum("mij,mj->mi", inverse, atb) / determinant[:, None]
    residual = np.linalg.norm(np.einsum("mij,mj->mi", normals, point) - rhos, axis=1)
    valid = (np.isfinite(condition) & (condition <= IEH.CONDITION_MAX)
             & np.isfinite(point).all(axis=1))
    return point, residual, condition, valid


def choose_triplet(candidate_lists: list[list[dict[str, Any]]]) -> Optional[dict]:
    """S1: fixed lexicographic choice over the top-5 cubed combinations.

    Residual first, then condition number, then the sum of candidate scores.
    No weighted objective and no tuned coefficient -- a weighted sum would be a
    hyperparameter fitted to the answer.
    """
    if any(len(c) == 0 for c in candidate_lists):
        return None
    counts = [len(c) for c in candidate_lists]
    index = np.stack(np.meshgrid(*[np.arange(n) for n in counts], indexing="ij"), -1)
    index = index.reshape(-1, 3)
    normals = np.stack([
        np.stack([[np.cos(candidate_lists[axis][i]["theta"]),
                   np.sin(candidate_lists[axis][i]["theta"])]
                  for i in index[:, axis]]) for axis in range(3)], axis=1)
    rhos = np.stack([
        np.array([candidate_lists[axis][i]["rho"] for i in index[:, axis]])
        for axis in range(3)], axis=1)
    scores = np.stack([
        np.array([candidate_lists[axis][i]["score"] for i in index[:, axis]])
        for axis in range(3)], axis=1).sum(axis=1)
    point, residual, condition, valid = solve_triplets(normals, rhos)
    if not valid.any():
        return None
    order = np.lexsort((-scores[valid], np.round(condition[valid], 6),
                        np.round(residual[valid], 6)))
    pick = np.nonzero(valid)[0][order[0]]
    return {"point": [float(point[pick, 0]), float(point[pick, 1])],
            "residual": float(residual[pick]), "condition": float(condition[pick])}


# ---------------------------------------------------------------------------
# Phase F/G/H/I -- one pass per arm/set
# ---------------------------------------------------------------------------
def solver_spec(frame):
    """Intrinsics and dimensions, wherever the frame type keeps them.

    Synthetic frames carry them directly; canonical frames keep them on the
    EvalFrame that also owns the manual ground truth.
    """
    evaluation = getattr(frame, "eval_frame", None)
    if evaluation is not None:
        return evaluation.K, evaluation.dims
    return frame.K, frame.dims


def audit_set(arm: str, seed: int, label: str, frames, maps, topology,
              policy) -> dict[str, Any]:
    incidence = {int(k): v for k, v in topology["corner_edge_incidence"].items()}
    name, parameter = policy["extractor"], policy["parameter"]
    edge_rows, corner_rows, task_rows = [], [], {"oracle": [], "s0": [], "s1": []}
    points_by_mode = {"oracle": [], "s0": [], "s1": []}

    for index, frame in enumerate(frames):
        width, height = frame.size
        segments = gt_segments(frame, topology)
        candidates = candidates_for_frame(maps[index], name, parameter)
        oracle_pick: list[Optional[dict]] = [None] * 12
        per_channel: list[dict[str, Any]] = []
        for channel in range(12):
            record = {"arm": arm, "seed": seed, "set": label, "frame": index,
                      "channel": channel, "state": frame.visibility[channel],
                      "n_candidates": len(candidates[channel]),
                      "available": segments[channel] is not None}
            if segments[channel] is None:
                per_channel.append(record)
                edge_rows.append(record)
                continue
            a, b = segments[channel]
            metrics = [IEH.match_metrics(c, a, b) for c in candidates[channel]]
            metrics = [m for m in metrics if m is not None]
            for key in ("strict_line", "loose_line", "strict_segment", "loose_segment"):
                record[f"top1_{key}"] = bool(metrics[0][key]) if metrics else False
                record[f"top3_{key}"] = any(m[key] for m in metrics[:3])
                record[f"top5_{key}"] = any(m[key] for m in metrics)
            if metrics:
                record["top1_angle"] = metrics[0]["angle_err_deg"]
                record["top1_offset"] = metrics[0]["offset_cells"]
                record["top1_overlap"] = metrics[0]["overlap_ratio"]
                best = min(range(len(metrics)),
                           key=lambda i: (not metrics[i]["strict_line"],
                                          metrics[i]["angle_err_deg"]
                                          + metrics[i]["offset_cells"]))
                record["best_angle"] = metrics[best]["angle_err_deg"]
                record["best_offset"] = metrics[best]["offset_cells"]
                record["best_overlap"] = metrics[best]["overlap_ratio"]
                if metrics[best]["strict_line"]:
                    oracle_pick[channel] = candidates[channel][best]
            # Phase I edge-level taxonomy
            if not record.get("top5_loose_line"):
                record["taxonomy"] = "L0_NO_LINE_HYPOTHESIS"
            elif record.get("top5_strict_line") and not record.get("top5_strict_segment"):
                record["taxonomy"] = "L1_INFINITE_LINE_ONLY"
            elif record.get("top5_strict_segment") and not record.get("top1_strict_segment"):
                record["taxonomy"] = "L2_SEGMENT_PRESENT_NOT_TOP1"
            elif record.get("top1_strict_segment"):
                record["taxonomy"] = "L6_STABLE_LINE_HYPOTHESIS"
            else:
                record["taxonomy"] = "L0_NO_LINE_HYPOTHESIS"
            per_channel.append(record)
            edge_rows.append(record)

        scale = np.array([width / GRID, height / GRID])
        frame_points = {"oracle": [], "s0": [], "s1": []}
        for corner in range(8):
            incident = incidence[corner]
            truth = frame.corners[corner]
            row = {"arm": arm, "seed": seed, "set": label, "frame": index,
                   "corner": corner, "near": corner < 4,
                   "states": "|".join(frame.visibility[k] for k in incident)}
            row["triplet_top1"] = all(
                per_channel[k].get("top1_strict_line", False) for k in incident)
            row["triplet_top5"] = all(
                per_channel[k].get("top5_strict_line", False) for k in incident)
            row["triplet_loose_top5"] = all(
                per_channel[k].get("top5_loose_line", False) for k in incident)

            def place(mode, solution):
                if solution is None:
                    frame_points[mode].append(None)
                    row[f"{mode}_err"] = None
                    return
                point = np.asarray(solution["point"]) * scale
                frame_points[mode].append([float(point[0]), float(point[1])])
                row[f"{mode}_err"] = (None if truth is None else
                                      float(np.hypot(point[0] - truth[0],
                                                     point[1] - truth[1])))
                row[f"{mode}_residual"] = solution["residual"]
                row[f"{mode}_condition"] = solution["condition"]

            picks = [oracle_pick[k] for k in incident]
            place("oracle", None if any(p is None for p in picks) else
                  IEH.intersect_lines([(p["theta"], p["rho"]) for p in picks]))
            tops = [candidates[k][:1] for k in incident]
            place("s0", None if any(len(t) == 0 for t in tops) else
                  IEH.intersect_lines([(t[0]["theta"], t[0]["rho"]) for t in tops]))
            place("s1", choose_triplet([candidates[k] for k in incident]))
            corner_rows.append(row)
        for mode in points_by_mode:
            points_by_mode[mode].append(frame_points[mode])

    # PnP for the three modes
    pnp = {}
    for mode, per_frame in points_by_mode.items():
        tasks = []
        keep = []
        for index, points in enumerate(per_frame):
            if any(p is None for p in points):
                continue
            frame = frames[index]
            keep.append(index)
            intrinsics, dims = solver_spec(frame)
            tasks.append((points + [None], intrinsics, dims,
                          (frame.size[1], frame.size[0], 3),
                          list(frame.corners) + [None]))
        solved = IEL.solve_many(tasks)
        finite = sum(1 for ok, _ in solved if ok)
        reproj = [v for _, v in solved if v is not None]
        pnp[mode] = {"complete_frames": len(keep), "frames": len(frames),
                     "finite_pnp": finite,
                     "reproj_median": float(np.median(reproj)) if reproj else None}
    return {"edges": edge_rows, "corners": corner_rows, "pnp": pnp}


def phase_match(state) -> dict[str, Any]:
    topology = IEL.load_topology()
    policy = json.loads((OUT / "extraction_policy.json").read_text("utf-8"))
    encoder = IEL.FrozenEncoder().to(device)
    synthetic = json.loads((BASE / "synthetic_results.json").read_text("utf-8"))
    IEL.start_pool()
    edges, corners, pnp_rows = [], [], []
    for arm in AUDIT_ARMS:
        for seed_key in synthetic["arms"][arm]["seeds"]:
            seed = int(seed_key)
            model, _, _ = IEL.load_selected(arm, seed, encoder)
            for label in SETS:
                frames = load_frames(topology, label)
                maps = predict_maps(model, encoder, frames)
                began = time.time()
                result = audit_set(arm, seed, label, frames, maps, topology, policy)
                edges.extend(result["edges"])
                corners.extend(result["corners"])
                for mode, block in result["pnp"].items():
                    pnp_rows.append({"arm": arm, "seed": seed, "set": label,
                                     "mode": mode, **block})
                available = [e for e in result["edges"] if e["available"]]
                top5 = np.mean([e.get("top5_strict_line", False) for e in available]) \
                    if available else 0.0
                triplet = np.mean([c["triplet_top5"] for c in result["corners"]])
                oracle = [c["oracle_err"] for c in result["corners"]
                          if c.get("oracle_err") is not None]
                log(f"[match] {arm} s{seed} {label:10s} n={len(frames):5d} "
                    f"top5-strict-line {100*top5:5.1f}%  triplet {100*triplet:5.1f}%  "
                    f"oracle<=20px "
                    f"{100*np.mean(np.array(oracle) <= CORNER_OK_PX) if oracle else 0:5.1f}%"
                    f"  ({time.time()-began:.0f}s)")
                del frames, maps
    pd.DataFrame(edges).to_parquet(OUT / "line_candidates.parquet")
    pd.DataFrame(corners).to_parquet(OUT / "corner_triplet_raw.parquet")
    pd.DataFrame(pnp_rows).to_csv(OUT / "pnp_metrics.csv", index=False)
    log(f"[match] {len(edges)} edge rows, {len(corners)} corner rows")
    return {"edge_rows": len(edges), "corner_rows": len(corners)}


# ---------------------------------------------------------------------------
# Phase J/K -- aggregation, gates, decision
# ---------------------------------------------------------------------------
def aggregate(edges: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    available = edges[edges["available"]]
    rows = []
    for (arm, seed, label), block in available.groupby(["arm", "seed", "set"]):
        corner_block = corners[(corners.arm == arm) & (corners.seed == seed)
                               & (corners["set"] == label)]
        oracle = corner_block["oracle_err"].dropna()
        s0 = corner_block["s0_err"].dropna()
        s1 = corner_block["s1_err"].dropna()
        rows.append({
            "arm": arm, "seed": seed, "set": label,
            "edges": len(block), "corners": len(corner_block),
            "edge_top1_strict_line": block["top1_strict_line"].mean(),
            "edge_top5_strict_line": block["top5_strict_line"].mean(),
            "edge_top5_loose_line": block["top5_loose_line"].mean(),
            "edge_top1_strict_segment": block["top1_strict_segment"].mean(),
            "edge_top5_strict_segment": block["top5_strict_segment"].mean(),
            "edge_top5_loose_segment": block["top5_loose_segment"].mean(),
            "corner_triplet_top1": corner_block["triplet_top1"].mean(),
            "corner_triplet_top5": corner_block["triplet_top5"].mean(),
            "corner_triplet_loose_top5": corner_block["triplet_loose_top5"].mean(),
            "oracle_n": len(oracle),
            "oracle_le20": float((oracle <= CORNER_OK_PX).mean()) if len(oracle) else 0.0,
            "oracle_median": float(oracle.median()) if len(oracle) else None,
            "s0_n": len(s0),
            "s0_le20": float((s0 <= CORNER_OK_PX).mean()) if len(s0) else 0.0,
            "s0_median": float(s0.median()) if len(s0) else None,
            "s1_n": len(s1),
            "s1_le20": float((s1 <= CORNER_OK_PX).mean()) if len(s1) else 0.0,
            "s1_median": float(s1.median()) if len(s1) else None,
        })
    return pd.DataFrame(rows)


def phase_decide(state) -> dict[str, Any]:
    edges = pd.read_parquet(OUT / "line_candidates.parquet")
    corners = pd.read_parquet(OUT / "corner_triplet_raw.parquet")
    summary = aggregate(edges, corners)
    summary.to_csv(OUT / "edge_availability.csv", index=False)

    taxonomy = (edges[edges["available"]]
                .groupby(["arm", "set", "taxonomy"]).size()
                .rename("n").reset_index())
    taxonomy.to_csv(OUT / "failure_taxonomy.csv", index=False)

    # corner-level taxonomy L3/L4
    corner_taxonomy = []
    for (arm, label), block in corners.groupby(["arm", "set"]):
        l3 = block[(block.triplet_top1) & (block.s0_err > CORNER_OK_PX)]
        l4 = block[(block.oracle_err <= CORNER_OK_PX)
                   & (block.s0_err > CORNER_OK_PX)
                   & ((block.s1_err > CORNER_OK_PX) | block.s1_err.isna())]
        corner_taxonomy.append({
            "arm": arm, "set": label, "corners": len(block),
            "L3_TOP1_LINE_CORRECT_CORNER_BAD": len(l3),
            "L4_ORACLE_TRIPLET_GOOD_FIXED_BAD": len(l4)})
    pd.DataFrame(corner_taxonomy).to_csv(OUT / "corner_taxonomy.csv", index=False)

    def best(arm: str, label: str) -> dict[str, Any]:
        block = summary[(summary.arm == arm) & (summary["set"] == label)]
        return block.sort_values("edge_top5_strict_line", ascending=False).iloc[0].to_dict()

    primary = max(AUDIT_ARMS,
                  key=lambda a: best(a, "untouched")["edge_top5_strict_line"])
    untouched = best(primary, "untouched")
    canonical = {label: best(primary, label) for label in ("eval56", "wood")}
    canonical_mean = {k: float(np.mean([canonical["eval56"][k], canonical["wood"][k]]))
                      for k in ("edge_top5_strict_line", "edge_top5_strict_segment",
                                "corner_triplet_top5", "oracle_le20", "s0_le20",
                                "s1_le20")}

    j1 = {k: untouched[{"oracle_corner_le20": "oracle_le20"}.get(k, k)] >= v
          for k, v in GATE_J1.items()}
    j2 = {k: all(canonical[s][{"oracle_corner_le20": "oracle_le20"}.get(k, k)] >= v
                 for s in ("eval56", "wood")) for k, v in GATE_J2.items()}
    hypothesis_present_synthetic = all(j1.values())
    hypothesis_present_canonical = all(j2.values())

    j3 = {
        "oracle_over_s0>=20pp": (untouched["oracle_le20"] - untouched["s0_le20"]
                                 >= GATE_J3["oracle_over_s0"]),
        "s1_over_s0>=10pp": (untouched["s1_le20"] - untouched["s0_le20"]
                             >= GATE_J3["s1_over_s0"]),
        "precondition": hypothesis_present_synthetic or hypothesis_present_canonical,
    }
    selection_failure = j3["precondition"] and (j3["oracle_over_s0>=20pp"]
                                                or j3["s1_over_s0>=10pp"])
    all_taxonomy = taxonomy[taxonomy.arm == primary]
    failures = all_taxonomy[all_taxonomy.taxonomy != "L6_STABLE_LINE_HYPOTHESIS"]["n"].sum()
    l1 = all_taxonomy[all_taxonomy.taxonomy == "L1_INFINITE_LINE_ONLY"]["n"].sum()
    j4 = {
        "top5_strict_line>=60%": untouched["edge_top5_strict_line"] >= GATE_J4["top5_strict_line"],
        "top5_strict_segment<35%": untouched["edge_top5_strict_segment"] < GATE_J4["top5_strict_segment"],
        "l1_share>=40%": (l1 / max(failures, 1)) >= GATE_J4["l1_share"],
    }
    endpoint_failure = all(j4.values())

    j5 = {
        "synthetic_strict>=70%": untouched["edge_top5_strict_line"] >= GATE_J5["synthetic_strict"],
        "canonical_strict<30%": canonical_mean["edge_top5_strict_line"] < GATE_J5["canonical_strict"],
        "synthetic_oracle>=50%": untouched["oracle_le20"] >= GATE_J5["synthetic_oracle"],
        "canonical_oracle<15%": canonical_mean["oracle_le20"] < GATE_J5["canonical_oracle"],
    }
    transfer_collapse = ((j5["synthetic_strict>=70%"] and j5["canonical_strict<30%"])
                         or (j5["synthetic_oracle>=50%"] and j5["canonical_oracle<15%"]))

    j6 = {
        "top5_strict_line<60%": untouched["edge_top5_strict_line"] < GATE_J6["edge_top5_strict_line"],
        "triplet_top5<40%": untouched["corner_triplet_top5"] < GATE_J6["corner_triplet_top5"],
        "oracle_le20<40%": untouched["oracle_le20"] < GATE_J6["oracle_corner_le20"],
    }
    dense_absent = any(j6.values())

    if dense_absent and transfer_collapse:
        decision = "MIXED_LINE_FAILURE"
    elif dense_absent:
        decision = "DENSE_INSTANCE_LINE_BRANCH_STOP"
    elif transfer_collapse:
        decision = "INSTANCE_EDGE_DOMAIN_TRANSFER_FIRST"
    elif endpoint_failure:
        decision = "ENDPOINT_SEGMENT_REPRESENTATION_REQUIRED"
    elif selection_failure and j3["s1_over_s0>=10pp"]:
        decision = "PARAMETRIC_INSTANCE_LINE_FITTING_FIRST"
    elif selection_failure:
        decision = "TOPOLOGY_AWARE_LINE_SELECTION_FIRST"
    else:
        decision = "BLOCKED"

    architecture = {
        "twelve_edge_representation": "VALID",
        "dense_predictor": "STOP" if dense_absent or transfer_collapse else "GO",
        "parametric_extractor": "GO" if selection_failure and not dense_absent else "STOP",
        "CIGM": "GO" if (hypothesis_present_synthetic and not dense_absent) else "STOP",
        "line_only_branch": "STOP" if decision in (
            "DENSE_INSTANCE_LINE_BRANCH_STOP", "MIXED_LINE_FAILURE") else "GO",
        "fusion": "STOP",
        "spatial_hcrm": "NEXT" if decision in (
            "DENSE_INSTANCE_LINE_BRANCH_STOP", "MIXED_LINE_FAILURE") else "DEFER",
    }
    payload = {
        "decision": decision, "primary_arm": primary,
        "gates": {"J1": j1, "J2": j2, "J3": j3, "J4": j4, "J5": j5, "J6": j6},
        "verdicts": {
            "HYPOTHESIS_PRESENT_SYNTHETIC": hypothesis_present_synthetic,
            "HYPOTHESIS_PRESENT_CANONICAL": hypothesis_present_canonical,
            "SELECTION_FAILURE": selection_failure,
            "ENDPOINT_SEGMENT_FAILURE": endpoint_failure,
            "HYPOTHESIS_TRANSFER_COLLAPSE": transfer_collapse,
            "DENSE_LOCALIZATION_ABSENT": dense_absent},
        "untouched": untouched, "canonical": canonical,
        "canonical_mean": canonical_mean,
        "architecture": architecture,
        "note": "O12 is an oracle on ground-truth geometry; oracle-selected corners "
                "here are a diagnostic upper bound using GT to pick among top-5 and "
                "are not a deployable method.  Finite PnP is not accurate pose.",
    }
    atomic_write(OUT / "final_decision.json", json.dumps(payload, indent=1, default=str))
    atomic_write(OUT / "architecture_decision.json",
                 json.dumps(architecture, indent=1))
    log(f"[decide] {decision}  primary {primary}")
    for name, value in payload["verdicts"].items():
        log(f"         {name:32s} {value}")
    return payload


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def phase_report(state) -> dict[str, Any]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    lock = json.loads((OUT / "input_lock.json").read_text("utf-8"))
    policy = json.loads((OUT / "extraction_policy.json").read_text("utf-8"))
    decision = json.loads((OUT / "final_decision.json").read_text("utf-8"))
    summary = pd.read_csv(OUT / "edge_availability.csv")
    profile = pd.read_csv(OUT / "field_profile.csv")
    taxonomy = pd.read_csv(OUT / "failure_taxonomy.csv")
    corner_taxonomy = pd.read_csv(OUT / "corner_taxonomy.csv")
    pnp = pd.read_csv(OUT / "pnp_metrics.csv")
    written = []

    def write(name, body):
        atomic_write(REPORTS / name, body)
        written.append(name)

    write("LINE_HYPOTHESIS_INPUT_LOCK.md",
          "# Audit input lock\n\nRead-only: no training run, no optimizer, no "
          "checkpoint written.\n\n```\n" + json.dumps(lock, indent=1) + "\n```\n")
    write("RAW_FIELD_LOCALIZATION_PROFILE.md",
          "# Raw learned field profile\n\nGT band is within 1.5 belief cells of the "
          "projected edge; near background is 5-10 cells; far background is beyond "
          "10.  Bands were fixed before the run.\n\n```\n"
          + profile.to_string(index=False) + "\n```\n")
    write("LINE_HYPOTHESIS_EXTRACTION.md",
          "# Deterministic hypothesis extraction\n\nThe policy was chosen on "
          "synthetic validation alone and then frozen; canonical results never "
          "touched it.\n\n```\n" + json.dumps(
              {k: v for k, v in policy.items() if k != "all_arms"}, indent=1)
          + "\n```\n\n## every predeclared arm\n\n```\n"
          + pd.DataFrame(policy["all_arms"]).to_string(index=False) + "\n```\n")
    write("EDGE_TOPK_AVAILABILITY.md",
          "# Edge top-K availability\n\n```\n"
          + summary[["arm", "seed", "set", "edges", "edge_top1_strict_line",
                     "edge_top5_strict_line", "edge_top5_loose_line",
                     "edge_top1_strict_segment", "edge_top5_strict_segment"]]
          .to_string(index=False) + "\n```\n")
    write("CORNER_TRIPLET_AVAILABILITY.md",
          "# Corner triplet availability and oracle selection\n\nOracle selection "
          "uses ground truth to pick among the top-5 per channel.  It is a "
          "diagnostic upper bound, not a deployable method.\n\n```\n"
          + summary[["arm", "seed", "set", "corners", "corner_triplet_top1",
                     "corner_triplet_top5", "oracle_le20", "oracle_median"]]
          .to_string(index=False) + "\n```\n")
    write("FIXED_TOPOLOGY_SELECTION.md",
          "# Fixed, ground-truth-free selection\n\nS0 takes each channel's top-1. "
          "S1 searches top-5 cubed per corner and picks lexicographically by "
          "intersection residual, then condition number, then candidate score sum. "
          "No weighted objective, no tuned coefficient.\n\n```\n"
          + summary[["arm", "seed", "set", "s0_le20", "s0_median", "s1_le20",
                     "s1_median", "oracle_le20"]].to_string(index=False)
          + "\n```\n\n## finite PnP (not pose accuracy)\n\n```\n"
          + pnp.to_string(index=False) + "\n```\n")
    write("ENDPOINT_SEGMENT_ANALYSIS.md",
          "# Infinite line against segment extent\n\n```\n"
          + summary[["arm", "seed", "set", "edge_top5_strict_line",
                     "edge_top5_strict_segment", "edge_top5_loose_segment"]]
          .to_string(index=False) + "\n```\n")
    write("SYNTHETIC_TO_CANONICAL_TRANSFER.md",
          "# Synthetic to canonical transfer\n\n```\n"
          + json.dumps({"untouched": decision["untouched"],
                        "canonical": decision["canonical"],
                        "canonical_mean": decision["canonical_mean"],
                        "J5": decision["gates"]["J5"]}, indent=1, default=str)
          + "\n```\n")
    write("LINE_FAILURE_TAXONOMY.md",
          "# Failure taxonomy\n\n## edge level\n\n```\n"
          + taxonomy.to_string(index=False)
          + "\n```\n\n## corner level\n\n```\n"
          + corner_taxonomy.to_string(index=False) + "\n```\n")
    write("LINE_ARCHITECTURE_DECISION.md",
          f"# Decision: {decision['decision']}\n\n```\n"
          + json.dumps({"decision": decision["decision"],
                        "primary_arm": decision["primary_arm"],
                        "verdicts": decision["verdicts"],
                        "gates": decision["gates"],
                        "architecture": decision["architecture"]},
                       indent=1, default=str) + "\n```\n")
    write("NEXT_ADMISSIBLE_EXPERIMENT.md", next_experiment(decision))
    write("LINE_HYPOTHESIS_PROVENANCE.md",
          "# Provenance\n\n```\n" + json.dumps({
              "runner": "scripts/stage0/line/instance_edge_line_audit.py",
              "module": "Deep_Object_Pose/common/instance_edge_hypotheses.py",
              "tests": "challenge/tests/test_instance_edge_line_audit.py",
              "result_root": str(OUT.relative_to(ROOT)),
              "head": lock["head"], "policy_sha256": policy["policy_sha256"],
              "training_runs": 0, "final_test_open_count": 0,
              "untouched_stride": UNTOUCHED_STRIDE,
          }, indent=1) + "\n```\n")
    for name in ("input_lock.json", "extraction_policy.json", "final_decision.json",
                 "architecture_decision.json", "edge_availability.csv",
                 "failure_taxonomy.csv", "corner_taxonomy.csv",
                 "field_profile.csv", "pnp_metrics.csv", "policy_selection.csv"):
        source = OUT / name
        if source.is_file():
            atomic_write(REPORTS / name, source.read_text("utf-8"))
            written.append(name)
    return {"written": written}


def next_experiment(decision: dict[str, Any]) -> str:
    case = decision["decision"]
    lines = [f"# Next admissible experiment: {case}", "",
             "Nothing here is trained in this audit.", ""]
    if case == "MIXED_LINE_FAILURE":
        lines += [
            "Dense line branch STOPS.  Synthetic localization and canonical transfer",
            "fail together, so neither a better selector nor a domain-adaptation pass",
            "would reach a candidate that is not there.", "",
            "Primary next: A1 + Spatial HCRM provisional screen.",
            "Line work resumes only behind a new output representation --",
            "endpoint heatmaps, endpoint-conditioned segments, or a parametric",
            "segment head -- designed and screened separately."]
    elif case == "DENSE_INSTANCE_LINE_BRANCH_STOP":
        lines += ["The correct line is not in the top-5 even on synthetic.",
                  "Stop the dense field head; move to A1 + Spatial HCRM."]
    elif case == "INSTANCE_EDGE_DOMAIN_TRANSFER_FIRST":
        lines += ["Candidates exist on synthetic and vanish on real.  Keep the",
                  "twelve-edge target; change predictor and domain: real edge",
                  "pseudo-labels, domain-adaptive features, appearance augmentation,",
                  "object-conditioned decoding."]
    elif case == "ENDPOINT_SEGMENT_REPRESENTATION_REQUIRED":
        lines += ["Infinite lines are right and extents are wrong.  Move to endpoint",
                  "heatmaps or a parametric segment representation."]
    elif case == "TOPOLOGY_AWARE_LINE_SELECTION_FIRST":
        lines += ["Correct candidates are present and the fixed selector misses them.",
                  "Next: topology-consistent selection, global cuboid consistency,",
                  "confidence-calibrated incident-edge selection."]
    elif case == "PARAMETRIC_INSTANCE_LINE_FITTING_FIRST":
        lines += ["Fixed topology selection already recovers much of the gap.",
                  "Next: deterministic hypotheses + CIGM, no learned fusion."]
    else:
        lines += ["Blocked."]
    return "\n".join(lines) + "\n"


def phase_test(state) -> dict[str, Any]:
    new = subprocess.run([sys.executable, "-m", "pytest", "-q",
                          "challenge/tests/test_instance_edge_line_audit.py"],
                         cwd=ROOT, capture_output=True, text=True)
    full = subprocess.run([sys.executable, "-m", "pytest", "-q", "challenge/tests"],
                          cwd=ROOT, capture_output=True, text=True)
    payload = {"new_command": "python -m pytest -q challenge/tests/test_instance_edge_line_audit.py",
               "full_command": "python -m pytest -q challenge/tests",
               "new_returncode": new.returncode,
               "new_tail": new.stdout.strip().splitlines()[-1:],
               "full_returncode": full.returncode,
               "full_tail": full.stdout.strip().splitlines()[-1:]}
    atomic_write(OUT / "tests.json", json.dumps(payload, indent=1))
    log(f"[test] new {payload['new_tail']}  full {payload['full_tail']}")
    return payload


DRIVERS = {"lock": phase_lock, "profile": phase_profile, "policy": phase_policy,
           "match": phase_match, "decide": phase_decide, "report": phase_report,
           "test": phase_test}
ORDER = ("lock", "profile", "policy", "match", "decide", "report", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(DRIVERS) + ["all", "status"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from-phase", default=None)
    arguments = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state = IEL.State(OUT / "state.json")
    if arguments.command == "status":
        for phase in ORDER:
            print(f"{phase:10s} {state.get(phase)}")
        return
    phases = list(ORDER)
    if arguments.command != "all":
        phases = [arguments.command]
    elif arguments.from_phase:
        phases = phases[phases.index(arguments.from_phase):]
    for phase in phases:
        if state.get(phase) == "DONE" and not arguments.force:
            log(f"[{phase}] already DONE -- skipping")
            continue
        state.set(phase, "RUNNING")
        began = time.time()
        try:
            DRIVERS[phase](state)
        except Exception as error:
            state.set(phase, "FAILED", error=repr(error))
            raise
        state.set(phase, "DONE", seconds=round(time.time() - began, 1))
    log("[audit] done")


if __name__ == "__main__":
    main()
