"""Pure N3-v3 aggregation over explicit per-frame records.

The 2-D definitions are delegated to the locked
``pallet_dim_conditioned_p_v1.eval_math`` implementation.  Optional pose
evaluation is delegated to that experiment's prediction-only ``pose.py``
contract; GT matching never gates PnP.

Input rows are mappings with these required keys::

    id, session, hw, gt, valid, permutations,
    predictions[method], matched[method], detected[method]

``material`` and ``occlusion`` are optional.  Missing/empty labels remain
``unclassified``; they are never inferred from images, predictions, or error.
For pose evaluation a row additionally contains ``pose`` with ``K``, ``xyz``
and ``truth`` in the schema consumed by the locked pose module.

All functions return in-memory dictionaries and write no files.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys

import numpy as np

from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as _eval
from scripts.research.pallet_n3_completion_v3 import common as _common


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260917
NO_CHANGE_TOLERANCE_PX = 1e-9  # locked by eval_math.damage
UNCLASSIFIED = "unclassified"
POSE_SOURCE = (_common.ROOT /
               "scripts/research/pallet_dim_conditioned_p_v1/pose.py")


def _label(value: Any) -> str:
    """Preserve an explicit label and keep absent labels unclassified."""
    if value is None:
        return UNCLASSIFIED
    text = str(value).strip()
    return text if text else UNCLASSIFIED


def _method_value(row: Mapping[str, Any], field: str, method: str) -> Any:
    if field not in row:
        raise ValueError(f"{row.get('id', '<unknown>')}: missing {field}")
    value = row[field]
    if isinstance(value, Mapping):
        if method not in value:
            raise ValueError(f"{row.get('id', '<unknown>')}: {field}[{method!r}] missing")
        return value[method]
    return value


def _validate_input_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = list(rows)
    ids = [row.get("id") for row in rows]
    if any(value is None for value in ids):
        raise ValueError("Every row requires a non-null id")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate frame id")
    for row in rows:
        if row.get("session") is None or str(row["session"]).strip() == "":
            raise ValueError(f"{row['id']}: session is required for session bootstrap")
        hw = np.asarray(row.get("hw"), dtype=np.float64)
        if hw.shape != (2,) or not np.isfinite(hw).all() or np.any(hw <= 0):
            raise ValueError(f"{row['id']}: hw must be two positive finite values")
        gt = np.asarray(row.get("gt"), dtype=np.float64)
        valid = np.asarray(row.get("valid"), dtype=bool)
        permutations = np.asarray(row.get("permutations"), dtype=np.int64)
        if gt.shape != (9, 2) or valid.shape != (9,):
            raise ValueError(f"{row['id']}: expected nine GT points and validity entries")
        if permutations.ndim != 2 or permutations.shape[1] != 9 or len(permutations) == 0:
            raise ValueError(f"{row['id']}: permutations must have shape [G,9]")
        if np.any(permutations < 0) or np.any(permutations > 8):
            raise ValueError(f"{row['id']}: permutation index outside 0..8")
        if (not all(np.array_equal(np.sort(value), np.arange(9))
                    for value in permutations) or
                np.any(permutations[:, 8] != 8)):
            raise ValueError(
                f"{row['id']}: each whole-object permutation must preserve center 8")
    return rows


def _prediction(row: Mapping[str, Any], method: str) -> np.ndarray:
    value = _method_value(row, "predictions", method)
    if value is None:
        return np.full((9, 2), np.nan, dtype=np.float64)
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (9, 2) or np.isinf(points).any():
        raise ValueError(f"{row['id']}: {method} prediction must be finite/NaN [9,2]")
    finite = np.isfinite(points)
    if not np.array_equal(finite[:, 0], finite[:, 1]):
        raise ValueError(f"{row['id']}: partly missing x/y pair")
    return points


def score_corner_rows(rows: Sequence[Mapping[str, Any]], method: str) -> list[dict]:
    """Score one method with the locked whole-object symmetry evaluator."""
    source = _validate_input_rows(rows)
    scored: list[dict] = []
    for row in source:
        matched = bool(_method_value(row, "matched", method))
        detected = bool(_method_value(row, "detected", method))
        if matched and not detected:
            raise ValueError(f"{row['id']}: matched frame cannot be undetected")
        measured = _eval.measure(
            _prediction(row, method), row["gt"], row["valid"],
            row["permutations"], row["hw"], matched=matched,
            detected=detected)
        measured.update(
            id=row["id"],
            session=str(row["session"]),
            material=_label(row.get("material")),
            occlusion=_label(row.get("occlusion")),
        )
        scored.append(measured)
    return scored


def _corner_summary(scored: Sequence[Mapping[str, Any]]) -> dict:
    raw = _eval.summary(list(scored))
    pck = {str(key): raw["PCK"][str(key)] for key in (5, 10, 20)}
    return {
        "metrics": {
            "matched_pooled_corner8_median_px": raw["matched_pooled_corner8_median_px"],
            "matched_pooled_corner8_P90_px": raw["matched_pooled_corner8_P90_px"],
            "full_PCK": pck,
            "full_PCK10_fraction": pck["10"],
            "E_sym": raw["E_sym"],
            "E_fixed": raw["E_fixed"],
            "full_penalty_median_px": raw["full_penalty_median_px"],
            "full_penalty_P90_px": raw["full_penalty_P90_px"],
            "gross20_fraction": raw["gross20"],
        },
        "denominators": {
            "frames": raw["total_frames"],
            "evaluable_frames": raw["evaluable_frames"],
            "matched_frames": raw["matched"],
            "detected_frames": raw["detected"],
            "missing_detection_frames": raw["missing"],
            "full_supervised_corners": raw["corners"],
            "observed_corners": raw["observed_corners"],
            "match_coverage": raw["coverage"],
            "non_evaluable_ids": raw["non_evaluable_ids"],
        },
        "contract": {
            "points": "corners 0-7 only; center 8 excluded from pooled corner metrics",
            "symmetry": "one approved whole-object permutation per frame",
            "conditional": "median/P90 pool finite predicted supervised corners on matched frames",
            "full_population": "PCK/E_sym/full-penalty retain misses with image-diagonal penalty",
        },
    }


def _group_summaries(scored: Sequence[Mapping[str, Any]], field: str) -> dict:
    labels = sorted({str(row[field]) for row in scored})
    return {
        label: _corner_summary([row for row in scored if row[field] == label])
        for label in labels
    }


def _cross_group_summaries(scored: Sequence[Mapping[str, Any]]) -> dict:
    keys = sorted({(str(row["material"]), str(row["occlusion"])) for row in scored})
    return {
        f"{material}::{occlusion}": _corner_summary([
            row for row in scored
            if row["material"] == material and row["occlusion"] == occlusion
        ])
        for material, occlusion in keys
    }


@lru_cache(maxsize=1)
def _pose_contract():
    """Load the locked pose module lazily so 2-D aggregation stays lightweight."""
    source_dir = str(POSE_SOURCE.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    return _common.local_module("locked_dim_conditioned_pose", POSE_SOURCE)


def score_pose_rows(rows: Sequence[Mapping[str, Any]], method: str) -> list[dict]:
    """Run prediction-only PnP/pose scoring without a GT match gate."""
    source = _validate_input_rows(rows)
    module = _pose_contract()
    output: list[dict] = []
    for row in source:
        if "pose" not in row:
            raise ValueError(f"{row['id']}: missing pose contract")
        spec = row["pose"]
        for key in ("K", "xyz", "truth"):
            if key not in spec:
                raise ValueError(f"{row['id']}: pose.{key} missing")
        detected = bool(_method_value(row, "detected", method))
        # Deliberately do not consult `matched`: pose.py is prediction-only.
        points = _prediction(row, method) if detected else None
        inferred = module.infer(
            points, np.asarray(spec["K"], dtype=np.float64),
            np.asarray(spec["xyz"], dtype=np.float64),
            source=bool(spec.get("source", False)))
        metric = dict(module.metric((row["id"], inferred, spec["truth"])))
        metric.update(
            session=str(row["session"]),
            material=_label(row.get("material")),
            occlusion=_label(row.get("occlusion")),
        )
        output.append(metric)
    return output


def _pose_summary(scored: Sequence[Mapping[str, Any]], module=None) -> dict:
    module = _pose_contract() if module is None else module
    rows = list(scored)
    available = [row for row in rows if row.get("available")]

    def stats(field: str) -> dict:
        values = np.asarray([row[field] for row in available], dtype=np.float64)
        return {
            "median": float(np.median(values)) if len(values) else None,
            "P90": float(np.quantile(values, .9)) if len(values) else None,
        }

    normalized = [row["ADDsym_normalized"] if row.get("available") else float("inf")
                  for row in rows]
    conditional = [row["ADDsym_normalized"] for row in available]
    return {
        "denominators": {
            "frames": len(rows),
            "available_frames": len(available),
            "failed_frames": len(rows) - len(available),
            "coverage": len(available) / len(rows) if rows else None,
        },
        "translation_cm": stats("translation_cm"),
        "rotation_deg": stats("rotation_deg"),
        "yaw_deg": stats("yaw_deg"),
        "IoU3D": stats("IoU3D"),
        "ADDsym_AUC_full": float(module.pose_auc(normalized, 1.0)) if rows else None,
        "ADDsym_AUC_conditional": (
            float(module.pose_auc(conditional, 1.0)) if conditional else None),
        "contract": "prediction-only selector/SQPnP/LM; no GT match gate; failures stay in full AUC/coverage",
    }


def _pose_groups(scored: Sequence[Mapping[str, Any]], field: str, module) -> dict:
    labels = sorted({str(row[field]) for row in scored})
    return {
        label: _pose_summary([row for row in scored if row[field] == label], module)
        for label in labels
    }


def evaluate_method(rows: Sequence[Mapping[str, Any]], method: str,
                    *, include_pose: bool = False) -> dict:
    """Return complete overall and subgroup metrics for one frozen path."""
    corner_rows = score_corner_rows(rows, method)
    result = {
        "method": method,
        "corner": _corner_summary(corner_rows),
        "subgroups": {
            "material": _group_summaries(corner_rows, "material"),
            "occlusion": _group_summaries(corner_rows, "occlusion"),
            "material_x_occlusion": _cross_group_summaries(corner_rows),
        },
        "corner_rows": corner_rows,
    }
    if include_pose:
        module = _pose_contract()
        pose_rows = score_pose_rows(rows, method)
        result["pose"] = _pose_summary(pose_rows, module)
        result["pose_subgroups"] = {
            "material": _pose_groups(pose_rows, "material", module),
            "occlusion": _pose_groups(pose_rows, "occlusion", module),
        }
        result["pose_rows"] = pose_rows
    return result


def _bootstrap_stats(scored: Sequence[Mapping[str, Any]],
                     session_weights: Mapping[str, int] | None = None) -> dict:
    observed: list[float] = []
    full: list[float] = []
    esym: list[float] = []
    for row in scored:
        weight = 1 if session_weights is None else int(session_weights.get(row["session"], 0))
        if weight <= 0 or not row.get("evaluable"):
            continue
        observed.extend(row["observed_errors"] * weight)
        full.extend(row["errors"] * weight)
        esym.extend([row["E_sym"]] * weight)
    observed_array = np.asarray(observed, dtype=np.float64)
    full_array = np.asarray(full, dtype=np.float64)
    return {
        "matched_pooled_corner8_median_px": (
            float(np.quantile(observed_array, .5)) if len(observed_array) else None),
        "matched_pooled_corner8_P90_px": (
            float(np.quantile(observed_array, .9)) if len(observed_array) else None),
        "full_PCK10_fraction": (
            float(np.mean(full_array <= 10)) if len(full_array) else None),
        "E_sym": float(np.mean(esym)) if esym else None,
        "full_penalty_median_px": (
            float(np.quantile(full_array, .5)) if len(full_array) else None),
        "full_penalty_P90_px": (
            float(np.quantile(full_array, .9)) if len(full_array) else None),
    }


def _bootstrap_arrays(scored: Sequence[Mapping[str, Any]],
                      session_index: Mapping[str, int]) -> dict:
    """Prepare values once; bootstrap draws only change integer session weights."""
    values = {"observed": [], "full": [], "esym": []}
    groups = {"observed": [], "full": [], "esym": []}
    for row in scored:
        if not row.get("evaluable"):
            continue
        group = session_index[row["session"]]
        for key, source in (("observed", row["observed_errors"]),
                            ("full", row["errors"]),
                            ("esym", [row["E_sym"]])):
            values[key].extend(source)
            groups[key].extend([group] * len(source))
    output = {}
    for key in values:
        array = np.asarray(values[key], dtype=np.float64)
        index = np.asarray(groups[key], dtype=np.int64)
        order = np.argsort(array, kind="stable")
        output[key] = {"values": array, "groups": index,
                       "sorted_values": array[order],
                       "sorted_groups": index[order]}
    return output


def _integer_weighted_quantile(prepared: Mapping[str, np.ndarray],
                               session_counts: np.ndarray, q: float) -> float | None:
    """Match NumPy's default linear quantile on integer-replicated samples."""
    values = prepared["sorted_values"]
    if not len(values):
        return None
    weights = session_counts[prepared["sorted_groups"]]
    cumulative = np.cumsum(weights, dtype=np.int64)
    total = int(cumulative[-1])
    if total == 0:
        return None
    position = (total - 1) * q
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    lower = values[np.searchsorted(cumulative, lower_rank, side="right")]
    upper = values[np.searchsorted(cumulative, upper_rank, side="right")]
    return float(lower + (upper - lower) * (position - lower_rank))


def _prepared_bootstrap_stats(prepared: Mapping[str, Mapping[str, np.ndarray]],
                              session_counts: np.ndarray) -> dict:
    observed = prepared["observed"]
    full = prepared["full"]
    esym = prepared["esym"]
    full_weights = session_counts[full["groups"]]
    esym_weights = session_counts[esym["groups"]]
    full_count = int(full_weights.sum())
    esym_count = int(esym_weights.sum())
    return {
        "matched_pooled_corner8_median_px":
            _integer_weighted_quantile(observed, session_counts, .5),
        "matched_pooled_corner8_P90_px":
            _integer_weighted_quantile(observed, session_counts, .9),
        "full_PCK10_fraction": (
            float(np.dot(full_weights, full["values"] <= 10) / full_count)
            if full_count else None),
        "E_sym": (
            float(np.dot(esym_weights, esym["values"]) / esym_count)
            if esym_count else None),
        "full_penalty_median_px":
            _integer_weighted_quantile(full, session_counts, .5),
        "full_penalty_P90_px":
            _integer_weighted_quantile(full, session_counts, .9),
    }


def session_bootstrap(base_rows: Sequence[Mapping[str, Any]],
                      candidate_rows: Sequence[Mapping[str, Any]],
                      *, resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED) -> dict:
    """Whole-session paired bootstrap, recalculating each target statistic.

    The reported direction is ``candidate - base``.  Undefined conditional
    support is retained as an invalid draw; it is never dropped to make a CI.
    """
    if resamples != BOOTSTRAP_RESAMPLES or seed != BOOTSTRAP_SEED:
        raise ValueError("N3 v3 locks 10,000 resamples and seed 20260917")
    base_rows, candidate_rows = list(base_rows), list(candidate_rows)
    if [row["id"] for row in base_rows] != [row["id"] for row in candidate_rows]:
        raise ValueError("Paired rows must have identical frame order")
    base_sessions = [row["session"] for row in base_rows]
    if base_sessions != [row["session"] for row in candidate_rows]:
        raise ValueError("Paired rows must have identical session labels")
    sessions = sorted(set(base_sessions))
    if not sessions:
        raise ValueError("Session bootstrap requires at least one frame")
    observed_base = _bootstrap_stats(base_rows)
    observed_candidate = _bootstrap_stats(candidate_rows)
    observed = {
        key: (None if observed_base[key] is None or observed_candidate[key] is None
              else observed_candidate[key] - observed_base[key])
        for key in observed_base
    }
    rng = np.random.default_rng(seed)
    lookup = {session: index for index, session in enumerate(sessions)}
    prepared_base = _bootstrap_arrays(base_rows, lookup)
    prepared_candidate = _bootstrap_arrays(candidate_rows, lookup)
    counts = rng.multinomial(
        len(sessions), np.full(len(sessions), 1 / len(sessions)), size=resamples)
    draws = {key: [] for key in observed}
    for count in counts:
        base = _prepared_bootstrap_stats(prepared_base, count)
        candidate = _prepared_bootstrap_stats(prepared_candidate, count)
        for key in draws:
            value = (None if base[key] is None or candidate[key] is None
                     else candidate[key] - base[key])
            draws[key].append(value)
    metrics = {}
    for key, values in draws.items():
        valid = [value for value in values if value is not None and np.isfinite(value)]
        invalid = len(values) - len(valid)
        entry = {
            "observed_delta": observed[key],
            "invalid_draws": invalid,
            "better_direction": "higher" if key == "full_PCK10_fraction" else "lower",
        }
        if observed[key] is None:
            entry["status"] = "NO_OBSERVED_SUPPORT"
        elif invalid:
            entry["status"] = "UNDEFINED_RESAMPLED_SUPPORT_NO_CI"
        else:
            array = np.asarray(valid, dtype=np.float64)
            entry.update(status="COMPLETE",
                         CI95=[float(np.quantile(array, .025)),
                               float(np.quantile(array, .975))])
        metrics[key] = entry
    return {
        "direction": "candidate_minus_base",
        "unit": "session",
        "sessions": len(sessions),
        "resamples": resamples,
        "seed": seed,
        "paired_sampling": True,
        "recalculates_each_statistic": True,
        "multiplicity_adjusted": False,
        "metrics": metrics,
    }


def _paired_counts(base_rows: Sequence[Mapping[str, Any]],
                   candidate_rows: Sequence[Mapping[str, Any]]) -> dict:
    raw = _eval.damage(list(base_rows), list(candidate_rows))
    comparable = (raw["improved_frames"] + raw["unchanged_frames"] +
                  raw["harmed_frames"])
    return {
        "comparable_frames": comparable,
        "improved_frames": raw["improved_frames"],
        "no_change_frames": raw["unchanged_frames"],
        "worsened_frames": raw["harmed_frames"],
        "no_change_tolerance_px": NO_CHANGE_TOLERANCE_PX,
        "good5_to_bad10_corners": raw["good5_to_bad10"],
        "bad20_to_good10_corners": raw["bad20_to_good10"],
        "reverse_good5_to_bad10_corners": raw["reverse_good5_to_bad10"],
        "evaluation_branch_changed_frames": raw["evaluation_branch_changed"],
        "canonical_GT_identity_aligned": raw["canonical_GT_identity_aligned"],
    }


def paired_corner_analysis(base_rows: Sequence[Mapping[str, Any]],
                           candidate_rows: Sequence[Mapping[str, Any]]) -> dict:
    """Report all favorable, unchanged, and adverse paired frame outcomes."""
    base_rows, candidate_rows = list(base_rows), list(candidate_rows)
    if [row["id"] for row in base_rows] != [row["id"] for row in candidate_rows]:
        raise ValueError("Paired rows must have identical frame order")
    for base, candidate in zip(base_rows, candidate_rows):
        if (base["material"], base["occlusion"]) != (
                candidate["material"], candidate["occlusion"]):
            raise ValueError(f"{base['id']}: subgroup labels differ between methods")
        if (base["detected"], base["matched"]) != (
                candidate["detected"], candidate["matched"]):
            raise ValueError(
                f"{base['id']}: N3 must preserve base detection and match state")
    by_field = {}
    for field in ("material", "occlusion"):
        by_field[field] = {}
        for label in sorted({row[field] for row in base_rows}):
            indices = [i for i, row in enumerate(base_rows) if row[field] == label]
            by_field[field][label] = _paired_counts(
                [base_rows[i] for i in indices],
                [candidate_rows[i] for i in indices])
    return {
        "frame_change": _paired_counts(base_rows, candidate_rows),
        "subgroups": by_field,
        "bootstrap": session_bootstrap(base_rows, candidate_rows),
        "direction": "candidate minus base; lower error is favorable",
    }


def evaluate_comparison(rows: Sequence[Mapping[str, Any]], base_method: str,
                        candidate_method: str, *, include_pose: bool = False) -> dict:
    """Evaluate frozen base and N3 paths on one identical ordered population."""
    rows = _validate_input_rows(rows)
    base = evaluate_method(rows, base_method, include_pose=include_pose)
    candidate = evaluate_method(rows, candidate_method, include_pose=include_pose)
    return {
        "population_frames": len(rows),
        "base_method": base_method,
        "candidate_method": candidate_method,
        "base": base,
        "candidate": candidate,
        "paired": paired_corner_analysis(base["corner_rows"], candidate["corner_rows"]),
        "contract": "same ordered frames/GT/permutations; explicit method-specific detection and match state",
    }
