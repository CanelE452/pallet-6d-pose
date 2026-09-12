"""Evaluate synthetic predictions and apply the frozen progression gates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .data import resolve_record_path
from .util import immutable_json, read_json, require_new, sha256


def _prediction_map(path: Path) -> tuple[dict, dict]:
    value = read_json(path)
    if value.get("schema") != "symdht_local_predictions_v1" or value.get("GT_opened") is not False:
        raise ValueError("wrong prediction schema or GT-free provenance")
    records = {record["frame_id"]: record for record in value["records"]}
    if len(records) != len(value["records"]):
        raise ValueError("duplicate prediction frame ID")
    return value, records


def _errors(points: np.ndarray, prediction_valid: np.ndarray, target: np.ndarray,
            target_valid: np.ndarray, permutations: list[list[int]], diagonal: float):
    best = None
    for permutation_index, permutation in enumerate(permutations):
        gt, valid = target[np.asarray(permutation)], target_valid[np.asarray(permutation)]
        valid = valid[:8]
        if not valid.any():
            candidate = (diagonal, np.full(8, diagonal), permutation_index, valid)
        else:
            error = np.full(8, np.nan)
            usable = valid & prediction_valid[:8] & np.isfinite(points[:8]).all(-1)
            error[valid] = diagonal
            error[usable] = np.linalg.norm(points[:8][usable] - gt[:8][usable], axis=-1)
            candidate = (float(np.mean(error[valid])), error, permutation_index, valid)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def evaluate(manifest_path: Path, prediction_path: Path, output_path: Path) -> dict:
    manifest = read_json(manifest_path)
    prediction, predicted = _prediction_map(prediction_path)
    manifest_ids = [record["frame_id"] for record in manifest["records"]]
    if set(manifest_ids) != set(predicted):
        raise ValueError("prediction/manifest frame ID sets differ; missing output is not dropped")
    rows, pooled, base_pooled = [], [], []
    good_count = harm_count = 0
    catastrophic = []
    corrections, reliabilities, p_nonnulls, concentrations, nulls = [], [], [], [], []
    role_reliability = [[] for _ in range(12)]
    tangent_force_max = 0.0
    for record in manifest["records"]:
        target_data = torch.load(resolve_record_path(manifest_path, record["supervision"]), map_location="cpu")
        target = target_data["points"][0].numpy().astype(np.float64)
        target_valid = target_data["valid"][0].numpy().astype(bool)
        obs = torch.load(resolve_record_path(manifest_path, record["observation"]), map_location="cpu")
        base = obs["base_points"][0].numpy().astype(np.float64)
        base_valid = obs["point_valid"][0].numpy().astype(bool)
        image_hw = obs["image_hw"][0].numpy().astype(np.float64)
        diagonal = float(np.linalg.norm(image_hw[::-1]))
        item = predicted[record["frame_id"]]
        points = np.asarray(item["points"], np.float64)
        point_valid = np.asarray(item["point_valid"], bool)
        base_result = _errors(base, base_valid, target, target_valid,
                              record["symmetry_permutations"], diagonal)
        method_result = _errors(points, point_valid, target, target_valid,
                                record["symmetry_permutations"], diagonal)
        base_mean, _, base_choice, _ = base_result
        method_mean, method_error, method_choice, method_mask = method_result
        pooled.extend(method_error[method_mask].tolist() if method_mask.any() else [diagonal] * 8)
        # Damage uses one whole-object assignment selected by the baseline for both arms.
        permutation = np.asarray(record["symmetry_permutations"][base_choice])
        gt, valid = target[permutation], target_valid[permutation]
        b_error = np.full(8, diagonal); m_error = np.full(8, diagonal)
        b_ok = valid[:8] & base_valid[:8] & np.isfinite(base[:8]).all(-1)
        m_ok = valid[:8] & point_valid[:8] & np.isfinite(points[:8]).all(-1)
        b_error[b_ok] = np.linalg.norm(base[:8][b_ok] - gt[:8][b_ok], axis=-1)
        m_error[m_ok] = np.linalg.norm(points[:8][m_ok] - gt[:8][m_ok], axis=-1)
        evaluable = valid[:8]
        base_pooled.extend(b_error[evaluable].tolist())
        good = evaluable & (b_error <= 5)
        good_count += int(good.sum()); harm_count += int((good & (m_error > 10)).sum())
        if base_mean <= 10 and method_mean > 50:
            catastrophic.append(record["frame_id"])
        row = {
            "frame_id": record["frame_id"], "raw_diagonal_px": diagonal,
            "base_frame_mean_px": base_mean, "method_frame_mean_px": method_mean,
            "method_frame_normalized": method_mean / diagonal,
            "base_symmetry_choice": base_choice, "method_symmetry_choice": method_choice,
        }
        if "correction" in item:
            correction = np.asarray(item["correction"], np.float64)
            magnitude = np.linalg.norm(correction[:8], axis=-1)
            row.update(correction_mean_px=float(magnitude.mean()), correction_max_px=float(magnitude.max()))
            corrections.extend(magnitude.tolist())
            reliability = np.asarray(item["reliability"], np.float64)
            p_nonnull = np.asarray(item["p_nonnull"], np.float64)
            concentration = np.asarray(item["concentration"], np.float64)
            null_proxy = np.asarray(item["null_proxy"], np.float64)
            reliabilities.extend(reliability.tolist()); p_nonnulls.extend(p_nonnull.tolist())
            concentrations.extend(concentration.tolist()); nulls.extend(null_proxy.tolist())
            for role, value in enumerate(reliability): role_reliability[role].append(float(value))
            # Every actual per-mode WLS force is scalar*n; its tangent component must be zero.
            lines = np.asarray(item["raw_lines"], np.float64)
            weights = np.asarray(item["absolute_mode_weight"], np.float64)
            normals = lines[..., :2]
            tangents = np.stack((-normals[..., 1], normals[..., 0]), -1)
            base_edges = np.asarray(((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)))
            edge_mid = base[base_edges].mean(1)
            signed = (normals * edge_mid[:, None]).sum(-1) + lines[..., 2]
            force = -weights[..., None] * signed[..., None] * normals
            tangent_force_max = max(tangent_force_max, float(np.abs((force * tangents).sum(-1)).max()))
        rows.append(row)
    frame_normalized = np.asarray([row["method_frame_normalized"] for row in rows])
    pooled_array = np.asarray(pooled)
    result = {
        "schema": "symdht_local_evaluation_v1", "arm": prediction["arm"],
        "seed": prediction.get("seed"), "manifest_sha256": sha256(manifest_path),
        "prediction_sha256": sha256(prediction_path), "n_frames": len(rows),
        "primary_frame_mean_over_raw_diagonal": float(frame_normalized.mean()),
        "pooled_symmetric_median_px": float(np.median(pooled_array)),
        "pooled_symmetric_p90_px": float(np.percentile(pooled_array, 90)),
        "good_point_count_base_le_5px": good_count,
        "good_point_damage_count_method_gt_10px": harm_count,
        "good_point_damage_rate": harm_count / max(good_count, 1),
        "new_catastrophic_frame_count": len(catastrophic),
        "new_catastrophic_frame_ids": catastrophic,
        "correction_magnitude_px": None if not corrections else {
            "mean": float(np.mean(corrections)), "median": float(np.median(corrections)),
            "p90": float(np.percentile(corrections, 90)), "max": float(np.max(corrections))},
        "reliability_proxy": None if not reliabilities else {
            "mean": float(np.mean(reliabilities)), "per_role_mean": [float(np.mean(v)) for v in role_reliability]},
        "p_nonnull_proxy_mean": None if not p_nonnulls else float(np.mean(p_nonnulls)),
        "null_proxy_mean": None if not nulls else float(np.mean(nulls)),
        "concentration_mean": None if not concentrations else float(np.mean(concentrations)),
        "normal_direction_QA": None if not corrections else {
            "weighted_force_tangent_max_abs": tangent_force_max,
            "PASS": tangent_force_max <= 1e-7,
            "scope": "actual saved modes; verifies each WLS line-force contribution, not total multi-line delta"},
        "records": rows,
    }
    immutable_json(require_new(output_path), result)
    return result


def compare(point_path: Path, direct_paths: list[Path], hough_paths: list[Path], output_path: Path) -> dict:
    if len(direct_paths) != 3 or len(hough_paths) != 3:
        raise ValueError("the frozen pilot requires exactly three Direct and three Hough seeds")
    point = read_json(point_path); direct = [read_json(path) for path in direct_paths]
    hough = [read_json(path) for path in hough_paths]
    p_primary = point["primary_frame_mean_over_raw_diagonal"]
    seed_gates = []
    for value in hough:
        checks = {
            "relative_primary_improvement_ge_1pct": (p_primary - value["primary_frame_mean_over_raw_diagonal"]) / p_primary >= .01,
            "median_nonworse": value["pooled_symmetric_median_px"] <= point["pooled_symmetric_median_px"],
            "p90_nonworse": value["pooled_symmetric_p90_px"] <= point["pooled_symmetric_p90_px"],
            "new_catastrophic_zero": value["new_catastrophic_frame_count"] == 0,
            "good_point_damage_le_0p5pct": value["good_point_damage_rate"] <= .005,
        }
        seed_gates.append({"seed": value["seed"], "checks": checks, "PASS": all(checks.values())})
    local_go = all(value["PASS"] for value in seed_gates)
    h_better_count = 0; paired_means = []
    for d_value, h_value in zip(direct, hough):
        if h_value["primary_frame_mean_over_raw_diagonal"] < d_value["primary_frame_mean_over_raw_diagonal"]:
            h_better_count += 1
        d_rows = {row["frame_id"]: row for row in d_value["records"]}
        differences = [row["method_frame_normalized"] - d_rows[row["frame_id"]]["method_frame_normalized"]
                       for row in h_value["records"]]
        paired_means.append(float(np.mean(differences)))
    incremental = local_go and h_better_count >= 2 and float(np.mean(paired_means)) < 0
    if incremental:
        verdict = "DHT_INCREMENTAL_GO"
    elif local_go:
        verdict = "LINE_FUSION_GO_DHT_SPECIFIC_NOT_ESTABLISHED"
    else:
        verdict = "NO_SYNTHETIC_LOCAL_FUSION_SIGNAL"
    result = {
        "schema": "symdht_local_comparison_v1", "scientific_verdict": verdict,
        "synthetic_local_fusion_go": local_go, "dht_incremental_go": incremental,
        "point": {key: point[key] for key in ("primary_frame_mean_over_raw_diagonal",
                                                "pooled_symmetric_median_px", "pooled_symmetric_p90_px")},
        "direct": [{key: value[key] for key in ("seed", "primary_frame_mean_over_raw_diagonal",
                                                  "pooled_symmetric_median_px", "pooled_symmetric_p90_px",
                                                  "good_point_damage_rate", "new_catastrophic_frame_count")}
                   for value in direct],
        "hough": [{key: value[key] for key in ("seed", "primary_frame_mean_over_raw_diagonal",
                                                 "pooled_symmetric_median_px", "pooled_symmetric_p90_px",
                                                 "good_point_damage_rate", "new_catastrophic_frame_count")}
                  for value in hough],
        "hough_seed_gates": seed_gates,
        "hough_better_than_direct_seed_count": h_better_count,
        "hough_minus_direct_paired_frame_mean_by_seed": paired_means,
        "hough_minus_direct_paired_frame_mean_seed_average": float(np.mean(paired_means)),
        "real_dev_allowed": local_go,
    }
    immutable_json(require_new(output_path), result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--manifest", required=True, type=Path)
    evaluation.add_argument("--prediction", required=True, type=Path)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.set_defaults(function=lambda a: evaluate(a.manifest.resolve(), a.prediction.resolve(), a.output.resolve()))
    comparison = sub.add_parser("compare")
    comparison.add_argument("--point", required=True, type=Path)
    comparison.add_argument("--direct", required=True, nargs=3, type=Path)
    comparison.add_argument("--hough", required=True, nargs=3, type=Path)
    comparison.add_argument("--output", required=True, type=Path)
    comparison.set_defaults(function=lambda a: compare(a.point.resolve(), [p.resolve() for p in a.direct],
                                                        [p.resolve() for p in a.hough], a.output.resolve()))
    return result


def main() -> None:
    args = parser().parse_args(); value = args.function(args)
    print(value.get("scientific_verdict", "PASS"))


if __name__ == "__main__":
    main()
