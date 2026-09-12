"""Evaluate v2 predictions, utility calibration, and the frozen close/go gate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.hough import Lattice
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import immutable_json, read_json, sha256

from .geometry import candidate_utility, continuous_line_targets


def _errors(points, point_valid, target, target_valid, permutations, diagonal):
    best = None
    for choice, permutation in enumerate(permutations):
        gt = target[np.asarray(permutation)]; valid = target_valid[np.asarray(permutation)][:8]
        if not valid.any():
            candidate = (diagonal, np.full(8, diagonal), choice, valid)
        else:
            error = np.full(8, np.nan); usable = valid & point_valid[:8] & np.isfinite(points[:8]).all(-1)
            error[valid] = diagonal; error[usable] = np.linalg.norm(points[:8][usable] - gt[:8][usable], axis=-1)
            candidate = (float(error[valid].mean()), error, choice, valid)
        if best is None or candidate[0] < best[0]: best = candidate
    return best


def _average_ranks(values):
    order = np.argsort(values, kind="mergesort"); ranks = np.empty(len(values), float); start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]: end += 1
        ranks[order[start:end]] = .5 * (start + end - 1) + 1; start = end
    return ranks


def _calibration(scores, gains, labels, ambiguity):
    scores, gains, labels, ambiguity = map(np.asarray, (scores, gains, labels, ambiguity))
    positives, negatives = int(labels.sum()), int((~labels.astype(bool)).sum())
    if positives and negatives:
        ranks = _average_ranks(scores); auroc = float((ranks[labels.astype(bool)].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    else: auroc = None
    order = np.argsort(-scores, kind="mergesort"); sorted_labels = labels[order].astype(bool)
    auprc = float((np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1))[sorted_labels].mean()) if positives else None
    score_rank, gain_rank = _average_ranks(scores), _average_ranks(gains)
    spearman = float(np.corrcoef(score_rank, gain_rank)[0, 1]) if len(scores) > 1 and score_rank.std() and gain_rank.std() else None
    def quartiles(by, name):
        result = []
        for q, indices in enumerate(np.array_split(np.argsort(by, kind="mergesort"), 4), 1):
            result.append({"quartile": q, f"mean_{name}": float(by[indices].mean()),
                           "mean_actual_candidate_gain_px": float(gains[indices].mean()),
                           "positive_utility_rate": float(labels[indices].mean()), "count": len(indices)})
        return result
    return {"definition": "utility predicts whether the selected line helps correct the current point; not visibility",
            "positive_gain_strictly_greater_px": .25, "count": len(scores), "positive_count": positives,
            "AUROC": auroc, "AUPRC": auprc, "spearman_utility_vs_gain": spearman,
            "utility_quartiles": quartiles(scores, "predicted_utility"),
            "ambiguity_quartiles": quartiles(ambiguity, "mode_ambiguity")}


def evaluate(manifest: Path, prediction_path: Path, output: Path):
    dataset = ObservationDataset(manifest, targets=True); prediction = read_json(prediction_path)
    if prediction.get("schema") != "symdht_local_predictions_v2" or prediction.get("GT_opened") is not False:
        raise ValueError("wrong prediction schema or GT-free provenance")
    predicted = {row["frame_id"]: row for row in prediction["records"]}
    if set(predicted) != {item["frame_id"] for item in dataset}: raise ValueError("frame ID mismatch")
    rows = []; pooled = []; base_pooled = []; corrections = []
    utility_scores = []; utility_gains = []; utility_labels = []; ambiguities = []
    line_errors = []; bin_correct = bin_total = 0; good_count = harm_count = 0; catastrophic = []
    covered = base_covered = target_count = 0; lattice = Lattice()
    for item in dataset:
        target = item["target_points"].numpy().astype(float); target_valid = item["target_valid"].numpy().astype(bool)
        base = item["base_points"].numpy().astype(float); base_valid = item["point_valid"].numpy().astype(bool)
        diagonal = float(torch.linalg.vector_norm(item["image_hw"].flip(-1)))
        pred = predicted[item["frame_id"]]; points = np.asarray(pred["points"], float); point_valid = np.asarray(pred["point_valid"], bool)
        permutations = item["symmetry_permutations"].tolist()
        base_result = _errors(base, base_valid, target, target_valid, permutations, diagonal)
        method_result = _errors(points, point_valid, target, target_valid, permutations, diagonal)
        base_mean, base_error, base_choice, base_mask = base_result
        method_mean, method_error, method_choice, method_mask = method_result
        pooled.extend(method_error[method_mask].tolist() if method_mask.any() else [diagonal] * 8)
        base_pooled.extend(base_error[base_mask].tolist() if base_mask.any() else [diagonal] * 8)
        permutation = np.asarray(permutations[base_choice]); gt = target[permutation]; gt_valid = target_valid[permutation]
        evaluable = gt_valid[:8]; target_count += int(evaluable.sum())
        b_ok = evaluable & base_valid[:8] & np.isfinite(base[:8]).all(-1); m_ok = evaluable & point_valid[:8] & np.isfinite(points[:8]).all(-1)
        base_covered += int(b_ok.sum()); covered += int(m_ok.sum())
        b_assigned = np.full(8, diagonal); m_assigned = np.full(8, diagonal)
        b_assigned[b_ok] = np.linalg.norm(base[:8][b_ok] - gt[:8][b_ok], axis=-1)
        m_assigned[m_ok] = np.linalg.norm(points[:8][m_ok] - gt[:8][m_ok], axis=-1)
        good = evaluable & (b_assigned <= 5); good_count += int(good.sum()); harm_count += int((good & (m_assigned > 10)).sum())
        if base_mean <= 10 and method_mean > 50: catastrophic.append(item["frame_id"])
        row = {"frame_id": item["frame_id"], "arm": prediction["arm"], "seed": prediction.get("seed"),
               "raw_diagonal_px": diagonal, "base_frame_mean_px": base_mean, "method_frame_mean_px": method_mean,
               "method_frame_normalized": method_mean / diagonal, "gain_frame_px": base_mean - method_mean,
               "base_symmetry_choice": base_choice, "method_symmetry_choice": method_choice}
        if prediction["arm"] != "point":
            correction = np.linalg.norm(np.asarray(pred["correction"], float)[:8], axis=-1); corrections.extend(correction.tolist())
            row["correction_mean_px"] = float(correction.mean()); row["correction_max_px"] = float(correction.max())
            lines = torch.tensor(pred["raw_line"], dtype=torch.float32)[None]
            finite = torch.isfinite(lines).all(-1); fallback = torch.zeros_like(lines); fallback[..., 0] = 1
            lines = torch.where(finite[..., None], lines, fallback)
            gt_t = torch.tensor(gt, dtype=torch.float32)[None]; valid_t = torch.tensor(gt_valid)[None]
            base_t = item["base_points"][None]; pv_t = item["point_valid"][None]; ps_t = item["point_sigma"][None]; hw_t = item["image_hw"][None]
            labels, gains, usable = candidate_utility(base_t, pv_t, ps_t, lines, gt_t, valid_t, torch.tensor([.005 * diagonal]), hw_t)
            utilities = np.asarray(pred["utility"], float); ambiguity = np.asarray(pred["ambiguity"], float)
            active = (usable & finite)[0].numpy().astype(bool)
            utility_scores.extend(utilities[active]); utility_gains.extend(gains[0].numpy()[active]); utility_labels.extend(labels[0].numpy()[active].astype(bool)); ambiguities.extend(ambiguity[active])
            soft, hard, support, _ = continuous_line_targets(gt_t, valid_t, item["box"][None], lattice)
            anchors = np.asarray(pred["anchor_index"], int); sup = support[0].numpy().astype(bool)
            bin_total += int(sup.sum()); bin_correct += int((anchors[sup] == hard[0].numpy()[sup]).sum())
            for edge_index, (a, b) in enumerate(EDGES):
                if sup[edge_index] and finite[0, edge_index]:
                    line = lines[0, edge_index]; endpoints = gt_t[0, [a, b]]
                    line_errors.append(float((endpoints @ line[:2] + line[2]).abs().mean()))
            row["utility_mean"] = float(utilities.mean()); row["actual_candidate_gain_mean_px"] = float(gains[0][active].mean()) if active.any() else 0.; row["mode_ambiguity_mean"] = float(ambiguity.mean())
        rows.append(row)
    frame_norm = np.asarray([row["method_frame_normalized"] for row in rows]); pooled_a = np.asarray(pooled)
    result = {"schema": "symdht_local_evaluation_v2", "arm": prediction["arm"], "seed": prediction.get("seed"),
              "manifest_sha256": sha256(manifest), "prediction_sha256": sha256(prediction_path), "n_frames": len(rows),
              "primary_frame_mean_over_raw_diagonal": float(frame_norm.mean()), "pooled_symmetric_median_px": float(np.median(pooled_a)),
              "pooled_symmetric_p90_px": float(np.percentile(pooled_a, 90)), "coverage_count": covered,
              "baseline_coverage_count": base_covered, "target_point_count": target_count, "coverage_rate": covered / max(target_count, 1),
              "good_point_count_base_le_5px": good_count, "good_point_damage_count_method_gt_10px": harm_count,
              "good_point_damage_rate": harm_count / max(good_count, 1), "new_catastrophic_frame_count": len(catastrophic),
              "new_catastrophic_frame_ids": catastrophic,
              "correction_magnitude_px": None if not corrections else {"mean": float(np.mean(corrections)), "median": float(np.median(corrections)), "p90": float(np.percentile(corrections, 90)), "max": float(np.max(corrections))},
              "continuous_line_endpoint_distance_px": None if not line_errors else {"mean": float(np.mean(line_errors)), "median": float(np.median(line_errors)), "p90": float(np.percentile(line_errors, 90))},
              "hard_bin_accuracy_secondary": None if not bin_total else {"correct": bin_correct, "total": bin_total, "accuracy": bin_correct / bin_total},
              "utility_calibration": None if not utility_scores else _calibration(utility_scores, utility_gains, utility_labels, ambiguities), "records": rows}
    immutable_json(output, result); return result


def publish(point_path: Path, direct_paths, hough_paths, docs: Path):
    point = read_json(point_path); direct = [read_json(p) for p in direct_paths]; hough = [read_json(p) for p in hough_paths]
    p_primary = point["primary_frame_mean_over_raw_diagonal"]; gates = []
    for value in hough:
        checks = {"relative_primary_improvement_ge_1pct": (p_primary - value["primary_frame_mean_over_raw_diagonal"]) / p_primary >= .01,
                  "median_nonworse": value["pooled_symmetric_median_px"] <= point["pooled_symmetric_median_px"],
                  "p90_nonworse": value["pooled_symmetric_p90_px"] <= point["pooled_symmetric_p90_px"],
                  "coverage_nonworse": value["coverage_count"] >= point["coverage_count"],
                  "good_point_damage_le_0p5pct": value["good_point_damage_rate"] <= .005,
                  "new_catastrophic_zero": value["new_catastrophic_frame_count"] == 0}
        gates.append({"seed": value["seed"], "checks": checks, "PASS": all(checks.values())})
    passed = all(row["PASS"] for row in gates); verdict = "DHT_LOCAL_SYNTHETIC_GATE_PASSED" if passed else "DHT_LOCAL_TRACK_CLOSED"
    keys = ("seed", "primary_frame_mean_over_raw_diagonal", "pooled_symmetric_median_px", "pooled_symmetric_p90_px", "coverage_count", "good_point_damage_rate", "new_catastrophic_frame_count", "correction_magnitude_px", "continuous_line_endpoint_distance_px", "hard_bin_accuracy_secondary", "utility_calibration")
    compact = lambda value: {key: value.get(key) for key in keys}
    per_seed = {"schema": "symdht_local_synth_per_seed_v2", "point": {k: point[k] for k in ("primary_frame_mean_over_raw_diagonal", "pooled_symmetric_median_px", "pooled_symmetric_p90_px", "coverage_count")}, "direct": [compact(v) for v in direct], "hough": [compact(v) for v in hough], "hough_seed_gates": gates}
    summary = {"schema": "symdht_local_result_summary_v2", "scientific_verdict": verdict, "DHT_track_CLOSED": not passed,
               "synthetic_gate_passed": passed, "real_DEV_executed": False, "real_DEV_allowed": passed,
               "FINAL_opened": False, "C4_status": "C4_NOT_EVALUATED", "hough_seed_gates": gates,
               "point": per_seed["point"], "direct": [compact(v) for v in direct], "hough": [compact(v) for v in hough]}
    immutable_json(docs / "SYNTH_PER_SEED.json", per_seed); immutable_json(docs / "RESULT_SUMMARY.json", summary)
    fields = ["frame_id", "arm", "seed", "raw_diagonal_px", "base_frame_mean_px", "method_frame_mean_px", "method_frame_normalized", "gain_frame_px", "correction_mean_px", "correction_max_px", "utility_mean", "actual_candidate_gain_mean_px", "mode_ambiguity_mean"]
    csv_path = docs / "SYNTH_PER_FRAME.csv"
    if csv_path.exists(): raise FileExistsError(csv_path)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for value in [point, *direct, *hough]:
            for row in value["records"]: writer.writerow({key: row.get(key) for key in fields})
    return summary


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate"); ev.add_argument("--manifest", type=Path, required=True); ev.add_argument("--prediction", type=Path, required=True); ev.add_argument("--output", type=Path, required=True)
    pub = sub.add_parser("publish"); pub.add_argument("--point", type=Path, required=True); pub.add_argument("--direct", type=Path, nargs=3, required=True); pub.add_argument("--hough", type=Path, nargs=3, required=True); pub.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate": value = evaluate(args.manifest.resolve(), args.prediction.resolve(), args.output.resolve()); print("PASS")
    else: value = publish(args.point.resolve(), [p.resolve() for p in args.direct], [p.resolve() for p in args.hough], args.docs.resolve()); print(value["scientific_verdict"])


if __name__ == "__main__": main()
