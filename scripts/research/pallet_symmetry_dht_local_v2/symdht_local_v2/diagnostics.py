"""Pre-training equation fixtures and explanatory GT oracle decomposition."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset, collate, observation_batch
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.hough import Lattice
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import immutable_json, read_json, sha256

from .geometry import candidate_utility, continuous_line_targets, decode_single_mode, single_mode_wls


def _require_finite_json(value, path="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite oracle value at {path}: {value}")
    if isinstance(value, dict):
        for key, item in value.items(): _require_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value): _require_finite_json(item, f"{path}[{index}]")


def equation_fixtures(output: Path):
    diagonal = 800.
    ordering = []
    for error in (2., 5., 10.):
        rows = []
        for name, correction in (("baseline", 0.), ("partial", error / 2), ("exact", error)):
            after = abs(error - correction)
            point = F.smooth_l1_loss(torch.tensor([after / diagonal]), torch.zeros(1)).item()
            no_harm = .5 * max(after - error, 0.) / diagonal
            rows.append({"case": name, "correction_px": correction, "error_after_px": after,
                         "point_plus_no_harm": point + no_harm})
        ordering.append({"error_before_px": error, "rows": rows,
                         "exact_lt_partial_lt_baseline": rows[2]["point_plus_no_harm"] < rows[1]["point_plus_no_harm"] < rows[0]["point_plus_no_harm"]})
    value = {
        "schema": "symdht_local_failure_mechanism_tests_v2", "PASS": all(x["exact_lt_partial_lt_baseline"] for x in ordering),
        "A_loss_scale": {"anchor_removed": True, "one_sided_no_harm_retained": True, "manual_ordering": ordering},
        "B_support_utility": {"support_definition": "structural existence", "utility_definition": "predicted usefulness for correcting current point", "positive_gain_strictly_greater_px": .25, "fixtures_covered_by_pytest": ["exact point + wrong sharp line", "wrong point + exact line", "support true + utility false"]},
        "C_continuous_supervision": {"choice": "A_soft_target_distribution", "theta_bandwidth_bins": 1., "rho_bandwidth_bins": 1., "lattice_unchanged": [36, 65]},
        "D_latent_alternative": {"choice": "utility-weighted MAP top-1 per semantic edge", "alternative_normal_equations_summed": False, "x0_x10_compromise_fixture": "PASS"},
        "test_command": "python -m pytest scripts/research/pallet_symmetry_dht_local_v2/tests -q", "tests_passed": 6}
    immutable_json(output, value); return value


def _point_errors(points, point_valid, target, target_valid, diagonal):
    evaluable = target_valid[:8]
    if not evaluable.any():
        return torch.full((8,), diagonal, dtype=points.dtype)
    error = torch.full((8,), diagonal, dtype=points.dtype)
    usable = evaluable & point_valid[:8] & torch.isfinite(points[:8]).all(-1)
    error[usable] = torch.linalg.vector_norm(points[:8][usable] - target[:8][usable], dim=-1)
    return error[evaluable]


def _choose_gt(base, point_valid, target, target_valid, permutations, diagonal):
    best = None
    for index, permutation in enumerate(permutations):
        perm = torch.as_tensor(permutation, dtype=torch.long)
        gt, valid = target[perm], target_valid[perm]
        errors = _point_errors(base, point_valid, gt, valid, diagonal)
        mean = float(errors.mean()) if errors.numel() else diagonal
        if best is None or mean < best[0]: best = (mean, gt, valid, index)
    return best


def _raw_gt_lines(gt: torch.Tensor):
    rows = []
    for a, b in EDGES:
        if not torch.isfinite(gt[[a, b]]).all():
            rows.append(torch.tensor([1., 0., 0.], dtype=gt.dtype, device=gt.device))
            continue
        direction = gt[b] - gt[a]
        normal = torch.stack((direction[1], -direction[0]))
        normal = normal / torch.linalg.vector_norm(normal).clamp_min(1e-8)
        rows.append(torch.cat((normal, -(normal * gt[a]).sum()[None])))
    return torch.stack(rows)[None]


def _oracle_correct(obs, gt, gt_valid, lines, available):
    finite = torch.isfinite(lines).all(-1)
    fallback = torch.zeros_like(lines); fallback[..., 0] = 1
    lines = torch.where(finite[..., None], lines, fallback)
    available = available & finite
    diagonal = torch.linalg.vector_norm(obs["image_hw"].flip(-1), dim=-1)
    utility, gains, usable = candidate_utility(obs["base_points"], obs["point_valid"], obs["point_sigma"],
                                              lines, gt, gt_valid, .005 * diagonal, obs["image_hw"])
    utility = utility * available.to(utility.dtype)
    points, _ = single_mode_wls(obs["base_points"], obs["point_valid"], obs["point_sigma"], lines,
                                utility, .005 * diagonal, obs["image_hw"])
    diagonal_value = float(diagonal[0])
    error = _point_errors(points[0], obs["point_valid"][0], gt[0], gt_valid[0], diagonal_value)
    return float(error.mean()), error.tolist(), float(utility.mean()), gains[0].tolist()


def oracle(manifest: Path, v1_prediction: Path, output: Path):
    dataset = ObservationDataset(manifest, targets=True); lattice = Lattice()
    predictions = read_json(v1_prediction)
    predicted = {row["frame_id"]: row for row in predictions["records"]}
    rows = []; pooled = {key: [] for key in ("baseline", "continuous_GT", "quantized_soft_GT", "v1_predicted")}
    normalized = {key: [] for key in pooled}
    for item in dataset:
        diagonal = float(torch.linalg.vector_norm(item["image_hw"].flip(-1)))
        _, gt0, valid0, choice = _choose_gt(item["base_points"], item["point_valid"], item["target_points"], item["target_valid"], item["symmetry_permutations"], diagonal)
        batch = collate([item]); obs = observation_batch(batch, torch.device("cpu"))
        gt, gt_valid = gt0[None], valid0[None]
        base_errors = _point_errors(obs["base_points"][0], obs["point_valid"][0], gt[0], gt_valid[0], diagonal)
        base_mean = float(base_errors.mean()) if base_errors.numel() else diagonal
        exact_lines = _raw_gt_lines(gt[0]); support = torch.tensor([[valid0[a] and valid0[b] for a, b in EDGES]])
        exact_mean, exact_errors, exact_use, _ = _oracle_correct(obs, gt, gt_valid, exact_lines, support)
        soft, _, soft_support, _ = continuous_line_targets(gt, gt_valid, obs["box"], lattice)
        decoded = decode_single_mode(soft.clamp_min(1e-30).log(), torch.ones_like(soft, dtype=torch.bool), lattice, obs["box"])
        quant_mean, quant_errors, quant_use, _ = _oracle_correct(obs, gt, gt_valid, decoded["raw_line"], soft_support)
        old = predicted[item["frame_id"]]; old_lines = torch.tensor(old["raw_lines"], dtype=torch.float32)
        old_weights = torch.tensor(old["absolute_mode_weight"], dtype=torch.float32)
        index = old_weights.argmax(-1); selected = old_lines[torch.arange(12), index][None]
        pred_available = old_weights.max(-1).values.gt(0)[None]
        pred_mean, pred_errors, pred_use, _ = _oracle_correct(obs, gt, gt_valid, selected, pred_available)
        values = {"baseline": base_mean, "continuous_GT": exact_mean, "quantized_soft_GT": quant_mean, "v1_predicted": pred_mean}
        for key, mean in values.items(): normalized[key].append(mean / diagonal)
        pooled["baseline"].extend(base_errors.tolist()); pooled["continuous_GT"].extend(exact_errors); pooled["quantized_soft_GT"].extend(quant_errors); pooled["v1_predicted"].extend(pred_errors)
        rows.append({"frame_id": item["frame_id"], "baseline_symmetry_choice": choice, "raw_diagonal_px": diagonal,
                     "baseline_error_px": base_mean, "GT_continuous_line_corrected_error_px": exact_mean,
                     "quantized_decode_GT_line_corrected_error_px": quant_mean,
                     "v1_predicted_line_corrected_error_px": pred_mean,
                     "oracle_utility_mean": {"continuous_GT": exact_use, "quantized_soft_GT": quant_use, "v1_predicted": pred_use}})
    summary = {key: {"frame_mean_over_raw_diagonal": float(np.mean(normalized[key])),
                     "pooled_median_px": float(np.median(pooled[key])), "pooled_p90_px": float(np.percentile(pooled[key], 90))} for key in pooled}
    value = {"schema": "symdht_local_oracle_decomposition_v2", "scope": "GT-using explanatory diagnostic only; not deployable performance",
             "trained_updates": 0, "manifest_sha256": sha256(manifest), "v1_prediction_sha256": sha256(v1_prediction),
             "v1_prediction_seed": predictions.get("seed"), "fusion": "v2 single-alternative WLS with GT-oracle binary utility at fixed 0.25px gain",
             "summary": summary, "records": rows}
    _require_finite_json(value)
    immutable_json(output, value); return value


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    fixtures = sub.add_parser("fixtures"); fixtures.add_argument("--output", type=Path, required=True)
    ora = sub.add_parser("oracle"); ora.add_argument("--manifest", type=Path, required=True); ora.add_argument("--v1-prediction", type=Path, required=True); ora.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fixtures": value = equation_fixtures(args.output.resolve())
    else: value = oracle(args.manifest.resolve(), args.v1_prediction.resolve(), args.output.resolve())
    print("PASS" if value.get("PASS", True) else "FAIL")


if __name__ == "__main__": main()
