"""Reaggregate the reusable N3-v3 evidence from raw local predictions.

This module implements contracts A/B/C/D/E/H/I in the execution handoff.  It
never imports a previously aggregated metric as a result: every numeric result
is recomputed from raw coordinates and the current locked DEV319 truth.  Old
receipts are read only for provenance/exposure checks.

The large ignored artifacts normally live in the primary checkout while this
code may run from an isolated worktree.  ``resolve_source_root`` therefore
finds a git worktree which owns the required raw cache; outputs always stay in
the current worktree's :mod:`common` DOC/RAW roots.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts.research.pallet_n3_completion_v3 import common as C
from scripts.research.pallet_n3_completion_v3 import evaluation as E
from scripts.research.pallet_n3_completion_v3 import metrics as M


SCHEMA = "pallet_n3_completion_v3_reuse_v1"
BASELINE_REL = Path("data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json")
DCP_REL = Path("data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV")
CAP_REL = Path("data/pallet/results/pallet_refinement_cap_sensitivity_v1/predictions")
OCCLUSION_REL = Path(
    "data/pallet/results/pallet_eval_occlusion_severity_v1/direct_review/"
    "OCCLUSION_SEVERITY_MANIFEST.json")
SPLIT_REL = Path("_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json")
ST_ROOT = Path("data/pallet/results/pallet_type_selftrain_v1/selftrain_recovery_v1/pose_only")
ST_AUDIT_REL = Path(
    "_docs/experiments/pallet_selftraining_paper_closure_v1/CORE_COMPARABILITY_AUDIT.json")
ST_PROTOCOL_REL = Path(
    "_docs/experiments/pallet_selftraining_paper_closure_v1/PAPER_CLOSURE_PROTOCOL.json")

DCP_ARMS = ("N0_BASE_REPLAY", "N1_SYM_ONLY", "N2_DIM_ONLY", "N3_DIM_SYM")
CORE_ARMS = ("OLD_P", "N2_DIM_ONLY", "N3_DIM_SYM")
CAP_ARMS = ("N2_DIM_ONLY", "N3_DIM_SYM")
CAP_VARIANTS = {
    "cap1pct": .01,
    "cap2pct": .02,
    "no_output_cap": None,
}
EXPECTED_FIXED = {
    "R0": {
        "matched_pooled_corner8_median_px": 6.720674617532823,
        "matched_pooled_corner8_P90_px": 43.890018716816925,
        "full_PCK10_fraction": .6342537014805922,
        "E_sym": .04952389936997251,
        "pose_translation_cm_median": 7.896851501830845,
        "pose_rotation_deg_median": 2.5388775343237744,
    },
    "N3_DIM_SYM": {
        "matched_pooled_corner8_median_px": 5.778160604002257,
        "matched_pooled_corner8_P90_px": 42.133753550391525,
        "full_PCK10_fraction": .6858743497398959,
        "E_sym": .04841908679319606,
        "pose_translation_cm_median": 7.0676496641524365,
        "pose_rotation_deg_median": 2.0703933055782517,
    },
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path, source_root: Path) -> dict:
    """Bind an input, retaining which checkout owns an ignored artifact."""
    resolved = path.resolve()
    try:
        shown = str(resolved.relative_to(source_root.resolve()))
    except ValueError:
        shown = str(resolved)
    return {
        "path": shown,
        "source_root": str(source_root.resolve()),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _verify_declared(entry: Mapping[str, Any], source_root: Path) -> dict:
    """Resolve and verify a path/hash/size receipt before reporting reuse."""
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = source_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = bind(path, source_root)
    if actual["sha256"] != entry.get("sha256"):
        raise ValueError(f"declared SHA-256 changed: {path}")
    if entry.get("bytes") is not None and actual["bytes"] != entry["bytes"]:
        raise ValueError(f"declared byte size changed: {path}")
    return actual


def _has_raw_root(root: Path) -> bool:
    return (root / BASELINE_REL).is_file() and (root / DCP_REL).is_dir()


def resolve_source_root(explicit: Path | str | None = None) -> Path:
    """Resolve the checkout holding ignored raw evidence, without guessing paths."""
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not _has_raw_root(root):
            raise FileNotFoundError(f"source root lacks baseline/DCP raw artifacts: {root}")
        return root
    if _has_raw_root(C.ROOT):
        return C.ROOT.resolve()
    try:
        listing = subprocess.check_output(
            ["git", "-C", str(C.ROOT), "worktree", "list", "--porcelain"],
            text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FileNotFoundError("cannot enumerate git worktrees for raw evidence") from exc
    candidates = [Path(line[9:]).resolve() for line in listing.splitlines()
                  if line.startswith("worktree ")]
    owners = [root for root in candidates if _has_raw_root(root)]
    if len(owners) != 1:
        raise FileNotFoundError(
            f"expected one worktree with raw baseline/DCP artifacts, found {owners}")
    return owners[0]


def _selected(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["score"]))


def _record_candidate(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    index = record.get("selected_index")
    if index is None:
        return None
    candidates = record.get("candidates")
    if (not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes))
            or not isinstance(index, int) or not 0 <= index < len(candidates)):
        raise ValueError(f"{record.get('id')}: invalid selected candidate")
    return candidates[index]


def _points(candidate: Mapping[str, Any] | None) -> Any:
    return None if candidate is None else candidate["keypoints_xy"]


def _preserved(base: Mapping[str, Any] | None,
               candidate: Mapping[str, Any] | None) -> bool:
    if base is None or candidate is None:
        return base is candidate
    return (float(base["score"]) == float(candidate["score"])
            and np.array_equal(np.asarray(base["box_xyxy"], float),
                               np.asarray(candidate["box_xyxy"], float))
            and np.array_equal(np.asarray(base["keypoints_xy"], float)[8],
                               np.asarray(candidate["keypoints_xy"], float)[8]))


def _attach(rows: Sequence[dict], method: str,
            candidates: Mapping[str, Mapping[str, Any] | None],
            *, preserve_from: str | None = None) -> None:
    expected = {row["id"] for row in rows}
    if set(candidates) != expected:
        raise ValueError(f"{method}: prediction identity differs from truth")
    for row in rows:
        candidate = candidates[row["id"]]
        if preserve_from is not None:
            base = row["_candidates"][preserve_from]
            if not _preserved(base, candidate):
                raise ValueError(f"{row['id']}: {method} changed score/box/center8")
        detected = candidate is not None
        matched = detected and E._iou(candidate["box_xyxy"], row["box"]) >= E.MATCH_IOU
        row.setdefault("predictions", {})[method] = _points(candidate)
        row.setdefault("detected", {})[method] = detected
        row.setdefault("matched", {})[method] = matched
        row.setdefault("_candidates", {})[method] = candidate


def load_dev_context(source_root: Path, *, include_pose: bool = True) -> tuple[list[dict], dict]:
    """Join DEV319 truth, R0 raw candidates, and direct occlusion review."""
    rows = E.load_dev319_truth(include_pose=include_pose)
    manifest = _read(C.DEV)
    items = {item["frame_id"]: item for item in manifest["items"]}
    if set(items) != {row["id"] for row in rows}:
        raise ValueError("manifest/truth identity mismatch")
    baseline_path = source_root / BASELINE_REL
    baseline = _read(baseline_path)
    if baseline.get("complete") is not True or len(baseline.get("frames", {})) != 3008:
        raise ValueError("baseline raw cache is incomplete")
    base_candidates = {
        row["id"]: _selected(baseline["frames"][items[row["id"]]["image_path"]])
        for row in rows
    }

    occ_path = source_root / OCCLUSION_REL
    occurrence = _read(occ_path)
    labels = {}
    mapping = {
        "CLEAN": "clean",
        "MODERATE_OCCLUSION": "moderate",
        "SEVERE_OCCLUSION": "severe",
    }
    for value in occurrence["rows"]:
        if value["frame_id"] in items:
            severity = mapping.get(value["severity"])
            if severity is None:
                raise ValueError(f"unknown direct severity {value['severity']!r}")
            labels[value["frame_id"]] = severity
    for row in rows:
        row["occlusion"] = labels.get(row["id"], M.UNCLASSIFIED)
        row["_image_path"] = items[row["id"]]["image_path"]
    counts = Counter(row["occlusion"] for row in rows)
    expected_counts = {"clean": 29, "moderate": 20, "severe": 79,
                       M.UNCLASSIFIED: 191}
    if dict(counts) != expected_counts:
        raise ValueError(f"direct-review join changed: {dict(counts)}")
    _attach(rows, "R0", base_candidates)
    sources = {
        "manifest": bind(C.DEV, C.ROOT),
        "baseline_candidates": bind(baseline_path, source_root),
        "occlusion_direct_review": bind(occ_path, source_root),
    }
    return rows, sources


def attach_dcp(rows: Sequence[dict], source_root: Path,
               arm: str, seed: int) -> tuple[str, dict]:
    path = source_root / DCP_REL / f"{arm}_seed{seed}.json"
    payload = _read(path)
    if payload.get("complete") is not True or payload.get("GT_input") is not False:
        raise ValueError(f"invalid/incomplete DCP raw artifact: {path}")
    records = payload.get("records", [])
    by_id = {record["id"]: _record_candidate(record) for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"duplicate DCP prediction id: {path}")
    method = f"{arm}_seed{seed}"
    _attach(rows, method, by_id, preserve_from="R0")
    checkpoint = _verify_declared(payload["checkpoint"], source_root)
    return method, {"prediction": bind(path, source_root), "checkpoint": checkpoint}


def _headline(result: Mapping[str, Any]) -> dict:
    corner = result["corner"]
    values = {
        **corner["metrics"],
        **corner["denominators"],
    }
    if "pose" in result:
        pose = result["pose"]
        values.update(
            pose_translation_cm_median=pose["translation_cm"]["median"],
            pose_translation_cm_P90=pose["translation_cm"]["P90"],
            pose_rotation_deg_median=pose["rotation_deg"]["median"],
            pose_rotation_deg_P90=pose["rotation_deg"]["P90"],
            pose_yaw_deg_median=pose["yaw_deg"]["median"],
            pose_yaw_deg_P90=pose["yaw_deg"]["P90"],
            pose_IoU3D_median=pose["IoU3D"]["median"],
            pose_ADDsym_AUC_full=pose["ADDsym_AUC_full"],
            pose_available_frames=pose["denominators"]["available_frames"],
            pose_coverage=pose["denominators"]["coverage"],
        )
    return values


def _compact(result: Mapping[str, Any]) -> dict:
    output = {
        "status": "COMPLETE",
        "headline": _headline(result),
        "corner": result["corner"],
        "subgroups": result["subgroups"],
    }
    if "pose" in result:
        output["pose"] = result["pose"]
        output["pose_subgroups"] = result["pose_subgroups"]
    return output


def _mean_dict(values: Sequence[Mapping[str, Any]]) -> dict:
    """Arithmetic mean of compatible scalar leaves; identical counts stay ints."""
    if not values:
        return {}
    output = {}
    keys = set.intersection(*(set(value) for value in values))
    for key in sorted(keys):
        items = [value[key] for value in values]
        if all(isinstance(item, Mapping) for item in items):
            output[key] = _mean_dict(items)  # type: ignore[arg-type]
        elif all(isinstance(item, (int, float)) and not isinstance(item, bool)
                 and np.isfinite(item) for item in items):
            if all(isinstance(item, int) for item in items) and len(set(items)) == 1:
                output[key] = int(items[0])
            else:
                output[key] = float(np.mean(items))
        elif all(item == items[0] for item in items):
            output[key] = items[0]
    return output


def _group_headlines(result: Mapping[str, Any], field: str) -> dict:
    groups = result["subgroups"][field]
    pose_groups = result.get("pose_subgroups", {}).get(field, {})
    output = {}
    for label, corner in groups.items():
        temporary = {"corner": corner}
        if label in pose_groups:
            temporary["pose"] = pose_groups[label]
        output[label] = _headline(temporary)
    return output


def _seed_panel(evaluated: Mapping[str, Mapping[str, Any]], arm: str,
                *, fields: Sequence[str] = ("material", "occlusion")) -> dict:
    names = [f"{arm}_seed{seed}" for seed in C.SEEDS]
    results = [evaluated[name] for name in names]
    panel = {
        "status": "COMPLETE",
        "aggregation": "arithmetic mean of per-seed statistics; predictions are never averaged",
        "per_seed": {str(seed): _compact(evaluated[f"{arm}_seed{seed}"])
                     for seed in C.SEEDS},
        "mean": {"headline": _mean_dict([_headline(result) for result in results])},
    }
    for field in fields:
        by_seed = [_group_headlines(result, field) for result in results]
        labels = set.intersection(*(set(value) for value in by_seed))
        panel["mean"][field] = {
            label: _mean_dict([value[label] for value in by_seed])
            for label in sorted(labels)
        }
    return panel


def _single_panel(result: Mapping[str, Any]) -> dict:
    return {
        "status": "COMPLETE",
        "result": _compact(result),
        "mean": {
            "headline": _headline(result),
            "material": _group_headlines(result, "material"),
            "occlusion": _group_headlines(result, "occlusion"),
        },
    }


def _regression(observed: Mapping[str, Any], expected: Mapping[str, float],
                tolerance: float = 1e-12) -> dict:
    checks = {}
    for key, wanted in expected.items():
        got = observed.get(key)
        error = None if got is None else abs(float(got) - wanted)
        checks[key] = {"observed": got, "expected": wanted,
                       "absolute_error": error,
                       "pass": error is not None and error <= tolerance}
    return {"status": "PASS" if all(row["pass"] for row in checks.values())
            else "BLOCKED_INTEGRITY", "tolerance": tolerance, "checks": checks}


def build_core(source_root: Path, *, include_pose: bool = True) -> tuple[dict, dict]:
    """Build A/B/C/D once, retaining raw scored rows for downstream E/output."""
    rows, sources = load_dev_context(source_root, include_pose=include_pose)
    bindings = {}
    arms = ("OLD_P",) + DCP_ARMS
    for arm in arms:
        for seed in C.SEEDS:
            _, bindings[f"{arm}_seed{seed}"] = attach_dcp(rows, source_root, arm, seed)

    evaluated = {"R0": M.evaluate_method(rows, "R0", include_pose=include_pose)}
    for arm in arms:
        for seed in C.SEEDS:
            method = f"{arm}_seed{seed}"
            need_pose = include_pose and arm in CORE_ARMS
            evaluated[method] = M.evaluate_method(rows, method, include_pose=need_pose)

    r0 = _single_panel(evaluated["R0"])
    panels = {arm: _seed_panel(evaluated, arm) for arm in arms}
    n3_mean = panels["N3_DIM_SYM"]["mean"]["headline"]
    a = {
        "status": "COMPLETE",
        "population": "DEV319",
        "methods": {
            "R0": r0,
            "OLD_P": panels["OLD_P"],
            "N2_DIM_ONLY": panels["N2_DIM_ONLY"],
            "N3_DIM_SYM": panels["N3_DIM_SYM"],
        },
        "fixed_number_regression": {
            "R0": _regression(r0["mean"]["headline"], EXPECTED_FIXED["R0"]),
            "N3_DIM_SYM": _regression(n3_mean, EXPECTED_FIXED["N3_DIM_SYM"]),
        },
        "new_pose_P90": {
            method: {key: panel["mean"]["headline"][key] for key in (
                "pose_translation_cm_P90", "pose_rotation_deg_P90", "pose_yaw_deg_P90")}
            for method, panel in (("R0", r0), ("OLD_P", panels["OLD_P"]),
                                  ("N2_DIM_ONLY", panels["N2_DIM_ONLY"]),
                                  ("N3_DIM_SYM", panels["N3_DIM_SYM"]))
        } if include_pose else None,
    }
    if any(value["status"] != "PASS" for value in a["fixed_number_regression"].values()):
        a["status"] = "BLOCKED_INTEGRITY"

    b_panels = {arm: panels[arm] for arm in DCP_ARMS}
    keys = ("matched_pooled_corner8_median_px",
            "matched_pooled_corner8_P90_px", "full_PCK10_fraction", "E_sym")
    deltas = {}
    for name, left, right in (
            ("N2_minus_N0", "N2_DIM_ONLY", "N0_BASE_REPLAY"),
            ("N1_minus_N0", "N1_SYM_ONLY", "N0_BASE_REPLAY"),
            ("N3_minus_N2", "N3_DIM_SYM", "N2_DIM_ONLY"),
            ("N3_minus_N1", "N3_DIM_SYM", "N1_SYM_ONLY")):
        lhs = b_panels[left]["mean"]["headline"]
        rhs = b_panels[right]["mean"]["headline"]
        deltas[name] = {key: lhs[key] - rhs[key] for key in keys}
    b = {"status": "COMPLETE", "population": "DEV319",
         "contract": "same raw candidates; locked whole-object symmetry; corners0-7",
         "methods": b_panels, "mean_metric_deltas": deltas}

    core_for_groups = {
        "R0": r0,
        "OLD_P": panels["OLD_P"],
        "N2_DIM_ONLY": panels["N2_DIM_ONLY"],
        "N3_DIM_SYM": panels["N3_DIM_SYM"],
    }
    c = {
        "status": "COMPLETE",
        "population": "DEV319 rectangular",
        "frame_counts": {"plastic": 194, "wood": 125},
        "methods": {name: {"material": value["mean"]["material"]}
                    for name, value in core_for_groups.items()},
    }
    d = {
        "status": "PARTIAL_LABELS",
        "population": "DEV319",
        "frame_counts": {"clean": 29, "moderate": 20, "severe": 79,
                         "unclassified": 191},
        "classification": {
            "source": sources["occlusion_direct_review"],
            "method": "direct human three-choice",
            "prior_prediction_exposure": "NOT_COLLECTED",
        },
        "methods": {name: {"occlusion": value["mean"]["occlusion"]}
                    for name, value in core_for_groups.items()},
        "corner_visibility": {
            "status": "BLOCKED_LABEL",
            "result": None,
            "reason": "direct review explicitly records corner_visibility_collected=false",
            "known_visible_corners": None,
            "known_occluded_corners": None,
            "unknown_supervised_corners": 2499,
        },
    }
    return {"A": a, "B": b, "C": c, "D": d,
            "sources": {**sources, "dcp_predictions": bindings}}, {
                "rows": rows, "evaluated": evaluated,
            }


def fixed_branch_damage(rows: Sequence[Mapping[str, Any]], base_method: str,
                        candidate_method: str,
                        cap_fraction: float | None) -> tuple[dict, list[dict]]:
    """Damage/recovery on the whole-object branch selected by the base only."""
    tolerance = M.NO_CHANGE_TOLERANCE_PX
    # Saved coordinates passed through float32 affine/cap arithmetic.  The
    # largest observed round-trip excess is <3.4e-5 px, so 1e-4 px separates
    # serialization error from a meaningful cap violation.
    cap_numeric_tolerance = 1e-4
    corner_before, corner_after, movements = [], [], []
    frame_rows = []
    cap_violations = []
    bound_violations = []
    for row in rows:
        base = np.asarray(row["predictions"][base_method], dtype=float)
        after = np.asarray(row["predictions"][candidate_method], dtype=float)
        if base.shape != (9, 2) or after.shape != (9, 2):
            continue
        matched = bool(row["matched"][base_method] and row["matched"][candidate_method])
        if not matched:
            continue
        base_measure = M.score_corner_rows([row], base_method)[0]
        branch = int(base_measure["branch"])
        permutation = np.asarray(row["permutations"], int)[branch]
        gt = np.asarray(row["gt"], float)[permutation]
        valid = np.asarray(row["valid"], bool)[permutation]
        finite = (np.isfinite(base).all(-1) & np.isfinite(after).all(-1)
                  & np.isfinite(gt).all(-1))
        support = valid[:8] & finite[:8]
        if not support.any():
            continue
        before = np.linalg.norm(base[:8] - gt[:8], axis=-1)[support]
        final = np.linalg.norm(after[:8] - gt[:8], axis=-1)[support]
        move = np.linalg.norm(after[:8] - base[:8], axis=-1)[support]
        cap_px = (None if cap_fraction is None else
                  float(cap_fraction * math.hypot(*row["hw"])))
        if cap_px is not None:
            bad_move = np.flatnonzero(move > cap_px + cap_numeric_tolerance)
            bad_bound = np.flatnonzero(
                final + cap_numeric_tolerance < np.maximum(0., before - cap_px))
            if len(bad_move):
                cap_violations.append({"id": row["id"], "count": int(len(bad_move)),
                                       "max_movement_px": float(move.max()),
                                       "cap_px": cap_px})
            if len(bad_bound):
                bound_violations.append({"id": row["id"], "count": int(len(bad_bound))})
        before_mean, after_mean = float(before.mean()), float(final.mean())
        frame_rows.append({
            "id": row["id"], "session": row["session"], "base_branch": branch,
            "base_mean_px": before_mean, "candidate_mean_px": after_mean,
            "delta_px": after_mean - before_mean, "supported_corners": int(support.sum()),
            "cap_px": cap_px,
        })
        corner_before.extend(before.tolist())
        corner_after.extend(final.tolist())
        movements.extend(move.tolist())
    before = np.asarray(corner_before, float)
    after = np.asarray(corner_after, float)
    movement = np.asarray(movements, float)
    corner_delta = after - before
    frame_delta = np.asarray([row["delta_px"] for row in frame_rows], float)
    if cap_fraction is None:
        inside = outside = None
        cap_hits = None
    else:
        # Each frame has a different cap; recover it in the same frame/corner order.
        caps = np.concatenate([
            np.full(row["supported_corners"], row["cap_px"], dtype=float)
            for row in frame_rows
        ]) if frame_rows else np.array([], float)
        inside = int((before <= caps).sum())
        outside = int((before > caps).sum())
        cap_hits = int(np.isclose(
            movement, caps, rtol=0., atol=cap_numeric_tolerance).sum())
    result = {
        "status": "COMPLETE" if not cap_violations and not bound_violations else "BLOCKED_INTEGRITY",
        "base_branch_fixed_for_candidate": True,
        "matched_comparable_frames": len(frame_rows),
        "observed_comparable_corners": int(len(before)),
        "no_change_tolerance_px": tolerance,
        "cap_numeric_tolerance_px": cap_numeric_tolerance,
        "output_cap_fraction": cap_fraction,
        "initial_error_inside_cap_corners": inside,
        "initial_error_outside_cap_corners": outside,
        "cap_hit_corners": cap_hits,
        "corner_change": {
            "improved": int((corner_delta < -tolerance).sum()),
            "no_change": int((np.abs(corner_delta) <= tolerance).sum()),
            "worsened": int((corner_delta > tolerance).sum()),
        },
        "frame_change": {
            "improved": int((frame_delta < -tolerance).sum()),
            "no_change": int((np.abs(frame_delta) <= tolerance).sum()),
            "worsened": int((frame_delta > tolerance).sum()),
        },
        "good5_to_bad10_corners": int(((before < 5) & (after > 10)).sum()),
        "bad20_to_good10_corners": int(((before > 20) & (after < 10)).sum()),
        "movement": {
            "median_px": float(np.median(movement)) if len(movement) else None,
            "P90_px": float(np.quantile(movement, .9)) if len(movement) else None,
            "max_px": float(movement.max()) if len(movement) else None,
            "cap_violations": len(cap_violations),
            "cap_violation_examples": cap_violations[:20],
        },
        "triangle_lower_bound_violations": len(bound_violations),
        "triangle_lower_bound_examples": bound_violations[:20],
    }
    return result, frame_rows


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        result[order[index:end]] = (index + end - 1) / 2 + 1
        index = end
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> dict:
    first, second = np.asarray(left, float), np.asarray(right, float)
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return {"n": int(len(first)), "pearson": None, "spearman": None}
    return {
        "n": int(len(first)),
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "spearman": float(np.corrcoef(_rankdata(first), _rankdata(second))[0, 1]),
    }


def paired_2d_pose(frame_rows: Sequence[Mapping[str, Any]],
                   base_pose: Sequence[Mapping[str, Any]],
                   candidate_pose: Sequence[Mapping[str, Any]]) -> dict:
    """Pair fixed-branch 2-D changes with pose changes on the same frames."""
    base = {row["id"]: row for row in base_pose}
    candidate = {row["id"]: row for row in candidate_pose}
    if set(base) != set(candidate):
        raise ValueError("pose rows differ between base and candidate")
    new_failures = [key for key in base if base[key].get("available")
                    and not candidate[key].get("available")]
    recoveries = [key for key in base if not base[key].get("available")
                  and candidate[key].get("available")]
    paired = []
    for frame in frame_rows:
        left, right = base[frame["id"]], candidate[frame["id"]]
        if not left.get("available") or not right.get("available"):
            continue
        paired.append({
            "id": frame["id"], "delta_2d_px": float(frame["delta_px"]),
            "delta_translation_cm": float(right["translation_cm"] - left["translation_cm"]),
            "delta_rotation_deg": float(right["rotation_deg"] - left["rotation_deg"]),
            "delta_yaw_deg": float(right["yaw_deg"] - left["yaw_deg"]),
        })
    tolerance = 1e-9
    output = {
        "pose_population_frames": len(base),
        "paired_with_fixed_branch_2d_frames": len(paired),
        "new_pose_failures": len(new_failures),
        "pose_recoveries": len(recoveries),
        "new_pose_failure_ids": new_failures,
        "pose_recovery_ids": recoveries,
        "lower_is_better": True,
        "no_change_tolerance": tolerance,
    }
    two_d = [row["delta_2d_px"] for row in paired]
    for field in ("translation_cm", "rotation_deg", "yaw_deg"):
        values = [row[f"delta_{field}"] for row in paired]
        array = np.asarray(values, float)
        output[field] = {
            "median_delta": float(np.median(array)) if len(array) else None,
            "improved": int((array < -tolerance).sum()),
            "no_change": int((np.abs(array) <= tolerance).sum()),
            "worsened": int((array > tolerance).sum()),
            "correlation_with_2d_delta": _correlation(two_d, values),
        }
    output["joint_direction_counts"] = dict(Counter(
        ("2d_improved" if row["delta_2d_px"] < -tolerance else
         "2d_worsened" if row["delta_2d_px"] > tolerance else "2d_same") + "__" +
        ("translation_improved" if row["delta_translation_cm"] < -tolerance else
         "translation_worsened" if row["delta_translation_cm"] > tolerance
         else "translation_same") for row in paired))
    return output


def build_cap(source_root: Path, state: Mapping[str, Any]) -> tuple[dict, dict]:
    rows = state["rows"]
    evaluated = state["evaluated"]
    sources, raw_frames = {}, {}
    panels = {}
    base_pose = evaluated["R0"]["pose_rows"]
    for arm in CAP_ARMS:
        panels[arm] = {}
        for variant, fraction in CAP_VARIANTS.items():
            per_seed = {}
            for seed in C.SEEDS:
                path = source_root / CAP_REL / f"{arm}_seed{seed}__{variant}.json"
                payload = _read(path)
                if payload.get("GT_input") is not False or not payload.get(
                        "baseline_candidates_preserved"):
                    raise ValueError(f"invalid cap artifact contract: {path}")
                records = payload.get("records", [])
                by_id = {}
                for record in records:
                    base = next(row for row in rows if row["id"] == record["id"])[
                        "_candidates"]["R0"]
                    index = record.get("selected_index")
                    if index is None:
                        by_id[record["id"]] = None
                    else:
                        if base is None:
                            raise ValueError("cap prediction exists without R0 candidate")
                        value = dict(base)
                        value["keypoints_xy"] = record["points"]
                        by_id[record["id"]] = value
                method = f"E_{arm}_seed{seed}_{variant}"
                _attach(rows, method, by_id, preserve_from="R0")
                result = M.evaluate_method(rows, method, include_pose=True)
                damage, frame_rows = fixed_branch_damage(rows, "R0", method, fraction)
                paired = paired_2d_pose(frame_rows, base_pose, result["pose_rows"])
                per_seed[str(seed)] = {
                    "metrics": _compact(result),
                    "fixed_base_branch_damage": damage,
                    "paired_2d_pose": paired,
                }
                raw_frames[method] = {
                    "fixed_branch": frame_rows,
                    "corner_scores": result["corner_rows"],
                    "pose_scores": result["pose_rows"],
                }
                sources[method] = bind(path, source_root)
            panels[arm][variant] = {
                "status": "COMPLETE" if all(
                    value["fixed_base_branch_damage"]["status"] == "COMPLETE"
                    for value in per_seed.values()) else "BLOCKED_INTEGRITY",
                "output_cap_fraction": fraction,
                "per_seed": per_seed,
                "mean_headline": _mean_dict([
                    value["metrics"]["headline"] for value in per_seed.values()]),
                "mean_damage_counts": _mean_dict([
                    value["fixed_base_branch_damage"] for value in per_seed.values()]),
            }
    return {
        "status": "COMPLETE" if all(
            value["status"] == "COMPLETE" for arm in panels.values()
            for value in arm.values()) else "BLOCKED_INTEGRITY",
        "base_method": "R0",
        "contract": "base-selected whole-object GT branch fixed after correction; strict 5/10/20 px thresholds",
        "methods": panels,
        "sources": sources,
    }, raw_frames


def _attach_candidate_file(rows: Sequence[dict], path: Path, method: str,
                           source_root: Path) -> dict:
    payload = _read(path)
    if payload.get("complete") is not True or not isinstance(payload.get("frames"), Mapping):
        raise ValueError(f"incomplete comparator artifact: {path}")
    by_id = {row["id"]: _selected(payload["frames"][row["_image_path"]]) for row in rows}
    _attach(rows, method, by_id, preserve_from="R0")
    return bind(path, source_root)


def _attach_posefix(rows: Sequence[dict], path: Path, method: str,
                    source_root: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        ids = [str(value) for value in payload["ids"]]
        points = np.asarray(payload["points"], dtype=float)
        if points.shape != (319, 2, 4, 9, 2):
            raise ValueError(f"unexpected PoseFix diagnosis shape: {points.shape}")
        # chain0/pass1 is the first uncapped canonical PoseFix application.
        predicted = {frame_id: value for frame_id, value in zip(ids, points[:, 0, 1])}
        checkpoint_sha = str(payload["checkpoint_sha256"].item())
    by_id = {}
    for row in rows:
        base = row["_candidates"]["R0"]
        if base is None:
            by_id[row["id"]] = None
        else:
            candidate = dict(base)
            candidate["keypoints_xy"] = predicted[row["id"]].tolist()
            by_id[row["id"]] = candidate
    _attach(rows, method, by_id, preserve_from="R0")
    entry = bind(path, source_root)
    entry.update(checkpoint_sha256=checkpoint_sha,
                 selected_tensor="points[:,0,1]",
                 meaning="RAW chain, first pass; uncapped")
    return entry


def build_comparators(source_root: Path, state: Mapping[str, Any]) -> tuple[dict, dict]:
    rows = state["rows"]
    bindings, evaluated = {}, {}
    for seed in C.SEEDS:
        d = (source_root / "data/pallet/results/pallet_sensors_refinement_closeout_v1/"
             f"evaluation/D{seed}/PREDICTIONS.json")
        l = (source_root / "data/pallet/results/pallet_line_pose_v1/"
             f"evaluation/image_line_only_seed{seed}/PREDICTIONS.json")
        p = (source_root / "data/pallet/results/pallet_posefix_replay_diagnosis_v1/"
             f"predictions/seed{seed}_REAL_DEV.npz")
        for family, path, attach in (("D", d, _attach_candidate_file),
                                     ("L", l, _attach_candidate_file),
                                     ("PoseFix", p, _attach_posefix)):
            method = f"H_{family}_seed{seed}"
            bindings[method] = attach(rows, path, method, source_root)
            evaluated[method] = M.evaluate_method(rows, method, include_pose=True)
        d_receipt = (source_root / "_docs/experiments/pallet_sensors_refinement_closeout_v1/"
                     f"D{seed}_COMPLETION.json")
        l_receipt = (source_root / "data/pallet/results/pallet_line_pose_v1/runs/"
                     f"image_line_only_seed{seed}/COMPLETION.json")
        posefix_receipt = (source_root / "_docs/experiments/pallet_sensors_submission_v1/"
                           f"PRIOR{seed}_COMPLETE.json")
        bindings[f"D_seed{seed}_receipt"] = bind(d_receipt, source_root)
        bindings[f"L_seed{seed}_receipt"] = bind(l_receipt, source_root)
        bindings[f"PoseFix_seed{seed}_receipt"] = bind(posefix_receipt, source_root)
    bindings["D_protocol"] = bind(
        source_root / "_docs/experiments/pallet_sensors_refinement_closeout_v1/"
                      "D_PROTOCOL_LOCK.json", source_root)
    bindings["L_selection"] = bind(
        source_root / "data/pallet/results/pallet_line_pose_v1/SELECTION.json", source_root)
    bindings["PoseFix_protocol"] = bind(
        source_root / "_docs/experiments/pallet_posefix_replay_diagnosis_v1/"
                      "REPLAY_PROTOCOL.json", source_root)
    methods = {}
    for family in ("D", "L", "PoseFix"):
        renamed = {f"{family}_seed{seed}": evaluated[f"H_{family}_seed{seed}"]
                   for seed in C.SEEDS}
        methods[family] = _seed_panel(renamed, family)
    methods["D"]["provenance"] = {
        "role": "direct coordinate residual regression",
        "supervision": "synthetic source only; normalized residual L1",
        "training": "3 seeds x 6000 updates x batch16; final step only",
        "dimensions": False, "symmetry_supervision": False,
    }
    methods["L"]["provenance"] = {
        "role": "image-conditioned line-structure correction",
        "supervision": "synthetic source only",
        "training": "3 seeds x 6000 updates x batch16; synthetic selection",
        "dimensions": False,
    }
    methods["PoseFix"]["provenance"] = {
        "role": "PoseFix-derived pallet9 RGB correction",
        "variant": "canonical synthetic-only PRIOR1/2/3 last6000; one raw pass",
        "input": "RGB crop plus initial R0 points/box; ResNet152",
        "not_used": "later real+synthetic Replay last300",
        "dimensions": False, "output_cap": None,
    }
    raw = {name: {"corner_scores": value["corner_rows"],
                  "pose_scores": value["pose_rows"]}
           for name, value in evaluated.items()}
    return {"status": "COMPLETE", "population": "DEV319",
            "methods": methods, "sources": bindings}, raw


def _student_candidates(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any] | None]:
    output = {}
    for record in payload.get("records", []):
        prediction = record.get("prediction", record)
        output[record["id"]] = _record_candidate(prediction)
    return output


def build_student_audit(source_root: Path, state: Mapping[str, Any]) -> tuple[dict, dict]:
    """Audit full319 exposure and rescore only the predeclared safe128 panel."""
    rows = state["rows"]
    by_id = {row["id"]: row for row in rows}
    split_path = source_root / SPLIT_REL
    audit_path = source_root / ST_AUDIT_REL
    protocol_path = source_root / ST_PROTOCOL_REL
    split = _read(split_path)
    audit = _read(audit_path)
    protocol = _read(protocol_path)
    safe = split.get("heldout", [])
    safe_ids = [record["id"] for record in safe]
    if len(safe_ids) != 128 or not set(safe_ids) <= set(by_id):
        raise ValueError("locked HELDOUT128 is not a DEV319 subset")
    if (audit.get("train_eval_RGB_overlap") != [] or
            audit.get("train_eval_recording_overlap") != [] or
            audit.get("teacher_eval_exact_ID_overlap") != []):
        raise ValueError("historical safe-cohort exposure audit no longer passes")
    safe_sessions = {by_id[frame_id]["session"] for frame_id in safe_ids}
    if safe_sessions & set(audit.get("teacher_sessions", [])):
        raise ValueError("teacher session overlaps HELDOUT128")

    sources = {
        "split_lock": bind(split_path, source_root),
        "comparability_audit": bind(audit_path, source_root),
        "closure_protocol": bind(protocol_path, source_root),
    }
    arms = ("SYN_LR5", "RAW_LR5", "REF_LR5")
    candidates = {}
    checkpoints = {}
    for arm in arms:
        path = source_root / ST_ROOT / f"EVAL_PREDICTIONS_{arm}.json"
        payload = _read(path)
        if payload.get("complete") is not True:
            raise ValueError(f"incomplete student prediction artifact: {path}")
        values = _student_candidates(payload)
        if len(values) != 194 or not set(safe_ids) <= set(values):
            raise ValueError(f"{arm} does not cover locked plastic194/HELDOUT128")
        candidates[arm] = values
        sources[f"predictions_{arm}"] = bind(path, source_root)
        checkpoint = _verify_declared(payload["checkpoint"], source_root)
        fit_path = (source_root / "_docs/experiments/pallet_type_selftrain_v1/"
                    f"selftrain_recovery_v1/pose_only/FIT_{arm}.json")
        fit = _read(fit_path)
        if fit.get("complete") is not True or fit.get("optimizer_steps") != 320:
            raise ValueError(f"incomplete student fit receipt: {fit_path}")
        fit_checkpoint = _verify_declared(fit["checkpoint"], source_root)
        if fit_checkpoint["sha256"] != checkpoint["sha256"]:
            raise ValueError(f"prediction and fit checkpoint differ: {arm}")
        sources[f"fit_{arm}"] = bind(fit_path, source_root)
        sources[f"checkpoint_{arm}"] = checkpoint
        checkpoints[arm] = checkpoint
    safe_rows = [dict(by_id[frame_id]) for frame_id in safe_ids]
    # Remove method maps but retain the truth/pose fields and direct labels.
    for row in safe_rows:
        row["predictions"] = {"R0": by_id[row["id"]]["predictions"]["R0"]}
        row["detected"] = {"R0": by_id[row["id"]]["detected"]["R0"]}
        row["matched"] = {"R0": by_id[row["id"]]["matched"]["R0"]}
        row["_candidates"] = {"R0": by_id[row["id"]]["_candidates"]["R0"]}
    for arm in arms:
        _attach(safe_rows, arm, {frame_id: candidates[arm][frame_id]
                                for frame_id in safe_ids})
    # N3 is a three-seed method; keep seeds separate and average metrics.
    for seed in C.SEEDS:
        method = f"N3_DIM_SYM_seed{seed}"
        for row in safe_rows:
            source = by_id[row["id"]]
            row["predictions"][method] = source["predictions"][method]
            row["detected"][method] = source["detected"][method]
            row["matched"][method] = source["matched"][method]
            row["_candidates"][method] = source["_candidates"][method]

    evaluated = {name: M.evaluate_method(safe_rows, name, include_pose=True)
                 for name in ("R0",) + arms}
    for seed in C.SEEDS:
        name = f"N3_DIM_SYM_seed{seed}"
        evaluated[name] = M.evaluate_method(safe_rows, name, include_pose=True)
    safe_result = {
        "status": "COMPLETE_REUSED_DEV",
        "population": "predeclared HELDOUT128 plastic; 7 recording groups; reused DEV, not independent test",
        "exposure": {
            "student_train_RGB_overlap": 0,
            "student_train_recording_overlap": 0,
            "teacher_exact_ID_overlap": 0,
            "teacher_session_overlap": 0,
            "teacher_manual_budget": {"images": audit["teacher_manual_images"],
                                      "corners": audit["teacher_manual_corners"]},
            "independent_test": False,
            "LR5_selection_history": "historically selected on reused plastic194 DEV",
        },
        "methods": {
            "R0": _single_panel(evaluated["R0"]),
            "source_only_update": _single_panel(evaluated["SYN_LR5"]),
            "raw_pseudo_student": _single_panel(evaluated["RAW_LR5"]),
            "corrected_pseudo_student": _single_panel(evaluated["REF_LR5"]),
            "R0_plus_N3": _seed_panel(evaluated, "N3_DIM_SYM"),
        },
        "checkpoints": checkpoints,
    }
    main = {
        "status": "BLOCKED_CONTRACT",
        "result": None,
        "reason": (
            "No common non-exposed DEV319 panel exists: the three locked student raw artifacts "
            "cover plastic194 only, and 66/194 are outside the predeclared recording-disjoint "
            "HELDOUT128. Wood predictions would require different material-specific students."),
        "table_XVI_cells": {
            "source_only_update": None,
            "raw_pseudo_student": None,
            "corrected_pseudo_student": None,
        },
    }
    raw = {name: {"corner_scores": value["corner_rows"],
                  "pose_scores": value["pose_rows"]}
           for name, value in evaluated.items()}
    return {"status": "PARTIAL_SAFE_COHORT", "DEV319_table": main,
            "safe_common_cohort": safe_result, "sources": sources}, raw


def _raw_core(state: Mapping[str, Any]) -> dict:
    output = {}
    for name, result in state["evaluated"].items():
        output[name] = {"corner_scores": result["corner_rows"]}
        if "pose_rows" in result:
            output[name]["pose_scores"] = result["pose_rows"]
    return output


def build(source_root: Path | str | None = None,
          *, contracts: Iterable[str] = ("A", "B", "C", "D", "E", "H", "I")) -> tuple[dict, dict]:
    root = resolve_source_root(source_root)
    wanted = {str(value).upper() for value in contracts}
    unknown = wanted - {"A", "B", "C", "D", "E", "H", "I"}
    if unknown:
        raise ValueError(f"unsupported reuse contracts: {sorted(unknown)}")
    core, state = build_core(root, include_pose=True)
    result = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "aggregation": "raw coordinates rescored; per-seed metric then arithmetic mean",
        "contracts": {key: core[key] for key in ("A", "B", "C", "D") if key in wanted},
        "source_bindings": {"core": core["sources"]},
    }
    raw = {"core": _raw_core(state)}
    if "E" in wanted:
        value, details = build_cap(root, state)
        result["contracts"]["E"] = value
        result["source_bindings"]["cap"] = value["sources"]
        raw["cap"] = details
    if "H" in wanted:
        value, details = build_comparators(root, state)
        result["contracts"]["H"] = value
        result["source_bindings"]["comparators"] = value["sources"]
        raw["comparators"] = details
    if "I" in wanted:
        value, details = build_student_audit(root, state)
        result["contracts"]["I"] = value
        result["source_bindings"]["students"] = value["sources"]
        raw["students_safe128"] = details
    statuses = {key: value.get("status") for key, value in result["contracts"].items()}
    result["status"] = "COMPLETE_WITH_DECLARED_BLOCKS" if all(
        value not in {"BLOCKED_INTEGRITY"} for value in statuses.values()) else "BLOCKED_INTEGRITY"
    result["contract_statuses"] = statuses
    return result, raw


def _csv_rows(result: Mapping[str, Any]) -> list[dict]:
    rows = []
    for contract in ("A", "B", "H"):
        section = result.get("contracts", {}).get(contract, {})
        for method, panel in section.get("methods", {}).items():
            headline = panel.get("mean", {}).get("headline")
            if headline is None:
                continue
            rows.append({"contract": contract, "method": method, **{
                key: headline.get(key) for key in (
                    "matched_pooled_corner8_median_px",
                    "matched_pooled_corner8_P90_px", "full_PCK10_fraction", "E_sym",
                    "pose_translation_cm_median", "pose_translation_cm_P90",
                    "pose_rotation_deg_median", "pose_rotation_deg_P90",
                    "matched_frames", "observed_corners")}})
    return rows


def write_outputs(result: Mapping[str, Any], raw: Mapping[str, Any]) -> dict:
    raw_dir = C.RAW / "reuse"
    doc_json = C.DOC / "REUSE_RESULTS.json"
    raw_json = raw_dir / "PER_FRAME_SCORES.json"
    bindings_json = raw_dir / "SOURCE_BINDINGS.json"
    csv_path = C.DOC / "REUSE_RESULTS.csv"
    C.write(doc_json, result)
    C.write(raw_json, raw)
    C.write(bindings_json, result["source_bindings"])
    csv_rows = _csv_rows(result)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(csv_rows[0]) if csv_rows else ["contract", "method"]
    pending = csv_path.with_name(csv_path.name + ".pending")
    with pending.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    pending.replace(csv_path)
    return {"summary": str(doc_json), "csv": str(csv_path),
            "per_frame": str(raw_json), "bindings": str(bindings_json)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--contracts", default="A,B,C,D,E,H,I",
                        help="comma-separated subset of A,B,C,D,E,H,I")
    parser.add_argument("--no-write", action="store_true")
    arguments = parser.parse_args(argv)
    contracts = [value.strip().upper() for value in arguments.contracts.split(",")
                 if value.strip()]
    result, raw = build(arguments.source_root, contracts=contracts)
    paths = None if arguments.no_write else write_outputs(result, raw)
    print(json.dumps({"status": result["status"],
                      "contract_statuses": result["contract_statuses"],
                      "source_root": result["source_root"], "outputs": paths},
                     ensure_ascii=False, indent=2))
    return 0 if result["status"] != "BLOCKED_INTEGRITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
