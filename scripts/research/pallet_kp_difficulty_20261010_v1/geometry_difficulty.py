"""Decompose the sealed original experiment; never infer, fit, or render.

Uses Python's standard library only. Accuracy means agreement within the
original eight-pixel threshold with the stored geometric reference, not
agreement with independently measured physical ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


OLD = "pallet_observation_refiner_20261009_v1"
NEW = "pallet_kp_difficulty_20261010_v1"
IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
SWAP = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
C2 = [[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]
PRIMARY = ("BASE_GEOM_NOSELF_ROBUST", "N3_SUBPIX_GEOM_NOSELF_ROBUST")
READ_FILES = ("INPUTS.json", "PREDICTIONS.jsonl.gz",
              "LEARNED_PREDICTIONS.jsonl.gz", "REAL_MASK_STRESS.jsonl.gz",
              "REAL_CORRESPONDENCE_ROWS.jsonl.gz", "GEOMETRY_STRESS.jsonl.gz")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def close(a, b, tolerance=1e-9):
    return a is not None and len(a) == len(b) and all(
        abs(x-y) <= tolerance for x, y in zip(a, b))


def dimension(extents, xyz):
    # Historical text labels describe size ordering; they do not index WD/DW.
    if extents is None:
        return None
    if close(extents, xyz):
        return 0
    if close(extents, [xyz[2], xyz[1], xyz[0]]):
        return 1
    raise AssertionError("stored extents match neither registered dimension")


def mm(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def transpose(a):
    return [list(x) for x in zip(*a)]


def physical(rotation, extents, xyz):
    return mm(rotation, SWAP if dimension(extents, xyz) == 1 else IDENTITY)


def rotation_gap(a, b):
    product = mm(transpose(a), b)
    cosine = (sum(product[i][i] for i in range(3))-1.)/2.
    return math.degrees(math.acos(max(-1., min(1., cosine))))


def pose_separation(a, b):
    raw = rotation_gap(a["R_physical"], b["R_physical"])
    c2 = min(raw, rotation_gap(a["R_physical"], mm(b["R_physical"], C2)))
    return {"centroid_cm": 100.*math.dist(a["centroid"], b["centroid"]),
            "physical_SO3_deg": raw, "physical_proper_C2_deg": c2}


def corners(extents):
    a, b, c = [x/2. for x in extents]
    return [[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
            [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]]


def project(rotation, centroid, extents, calibration):
    result = []
    for point in corners(extents):
        camera = [sum(rotation[i][j]*point[j] for j in range(3))+centroid[i]
                  for i in range(3)]
        homogeneous = [sum(calibration[i][j]*camera[j] for j in range(3))
                       for i in range(3)]
        assert homogeneous[2] > 0., "saved alternative violates positive depth"
        result.append([homogeneous[i]/homogeneous[2] for i in (0, 1)])
    return result


def image_shape(points):
    """Two-dimensional centered scatter spectrum; no pose fitting."""
    if not points:
        return {"singular_values": [], "relative_singular_values": [], "numerical_rank": 0}
    center = [statistics.mean(p[i] for p in points) for i in (0, 1)]
    x = [p[0]-center[0] for p in points]
    y = [p[1]-center[1] for p in points]
    xx, yy, xy = math.fsum(t*t for t in x), math.fsum(t*t for t in y), math.fsum(a*b for a,b in zip(x,y))
    delta = math.hypot(xx-yy, 2.*xy)
    eigen = [max(0., (xx+yy+delta)/2.), max(0., (xx+yy-delta)/2.)]
    values = [math.sqrt(t) for t in eigen]
    relative = [s/values[0] if values[0] else 0. for s in values]
    return {"singular_values": values, "relative_singular_values": relative,
            "numerical_rank": sum(s>1e-10 for s in relative)}


def distributions(values):
    v = sorted(float(x) for x in values if x is not None and math.isfinite(x))
    if not v:
        return {"n": 0, "mean": None, "sample_variance": None, "sample_sd": None,
                "median": None, "P90": None, "min": None, "max": None}
    index = .9*(len(v)-1)
    low, high = math.floor(index), math.ceil(index)
    return {"n": len(v), "mean": statistics.mean(v),
            "sample_variance": statistics.variance(v) if len(v)>1 else None,
            "sample_sd": statistics.stdev(v) if len(v)>1 else None,
            "median": statistics.median(v),
            "P90": v[low]+(index-low)*(v[high]-v[low]),
            "min": v[0], "max": v[-1]}


def histogram(rows, field):
    return {str(k): v for k, v in sorted(Counter(r[field] for r in rows).items(),
                                        key=lambda item: str(item[0]))}


def outcomes(rows):
    return dict(Counter(r["paired_pose_outcome"] for r in rows))


def metric_summary(rows):
    return {"frames": len(rows), "new_pose": sum(r["new_pose_estimated"] for r in rows),
            "fallback": sum(r["fallback_used"] for r in rows),
            "no_pose": sum(r["no_pose"] for r in rows),
            "paired_outcomes": outcomes(rows),
            "translation_cm": distributions(r["translation_cm"] for r in rows),
            "rotation_deg": distributions(r["rotation_deg"] for r in rows),
            "translation_delta_cm": distributions(r["translation_delta_cm"] for r in rows),
            "rotation_delta_deg": distributions(r["rotation_delta_deg"] for r in rows)}


def paired_outcome(audit):
    if audit["fallback_used"]:
        return "fallback"
    if audit["no_pose"]:
        return "no_pose"
    t, r = audit["translation_delta_cm"], audit["rotation_delta_deg"]
    if t is None or r is None:
        return "pair_unavailable"
    if t < 0. and r < 0.:
        return "both_improved"
    if t > 0. and r > 0.:
        return "both_worsened"
    return "mixed_or_equal"


def mask_role(method):
    if "ORACLE" in method or "DROP_ONE" in method or "KEEP_ONE" in method:
        return "oracle_or_injected_diagnostic"
    if "NO_MASK" in method:
        return "mask_not_applied; known-state mismatch is not classifier error"
    return "predicted_geometry_mask"


def alternatives(raw, checks):
    solver = raw["solver"]
    if not raw["new_pose_estimated"] or not solver.get("R_cf"):
        return {"stored_alternative_count": 0, "coverage": "no_new_point_pose",
                "nearest": None, "best_stored_opposing_dimension": None,
                "all_stored": []}
    xyz = raw["xyz"]
    chosen_dimension = dimension(solver["cf_extents"], xyz)
    chosen = {"R_physical": physical(solver["R_cf"], solver["cf_extents"], xyz),
              "centroid": solver["centroid"]}
    initial = raw.get("initial_pose")
    u = solver.get("used", [])
    inliers = solver.get("final_inliers", solver.get("inliers", []))
    stored = []
    for i, alt in enumerate(solver.get("alternatives", [])):
        projected = project(alt["R_cf"], alt["centroid"], alt["dimensions"], raw["K"])
        residuals = [math.dist(projected[j], raw["input_points"][j]) for j in u]
        alt_inliers = [j for j, residual in zip(u, residuals) if residual <= 8.]
        sse = sum(min(residual*residual, 64.) for residual in residuals)
        assert len(alt_inliers) == alt["inlier_count"]
        assert math.isclose(sse, alt["truncated_sse_px2"], rel_tol=1e-8, abs_tol=1e-6)
        alt_pose = {"R_physical": physical(alt["R_cf"], alt["dimensions"], xyz),
                    "centroid": alt["centroid"]}
        index = dimension(alt["dimensions"], xyz)
        entry = {"stored_index": i, "dimension_index": index,
                 "dimensions_m": alt["dimensions"], "same_dimension": index == chosen_dimension,
                 "generator": alt["generator"], "generator_ids": alt["generator_ids"],
                 "inlier_count": alt["inlier_count"], "inlier_ids_reprojected": alt_inliers,
                 "same_inlier_ids_as_chosen": alt_inliers == inliers,
                 "inlier_gap": len(inliers)-alt["inlier_count"],
                 "truncated_sse_px2": alt["truncated_sse_px2"],
                 "sse_margin_px2": alt["truncated_sse_px2"]-solver["truncated_sse_px2"],
                 "separation_from_chosen": pose_separation(chosen, alt_pose),
                 "separation_from_initial": pose_separation(initial, alt_pose)
                 if initial and initial.get("R_physical") else None}
        stored.append(entry)
        checks["stored_alternatives_reprojected_without_fit"] += 1
    opposing = next((a for a in stored if not a["same_dimension"]), None)
    return {"stored_alternative_count": len(stored),
            "coverage": "opposing_dimension_recorded" if opposing else
            "opposing_dimension_not_in_at_most_two_stored_alternatives",
            "nearest": stored[0] if stored else None,
            "best_stored_opposing_dimension": opposing, "all_stored": stored}


def decompose(audit, raw, checks):
    for key in ("id", "method", "new_pose_estimated", "fallback_used", "no_pose"):
        assert audit[key] == raw[key], (key, audit["id"], audit["method"])
    pool, correct, wrong, unknown = [set(audit[key]) for key in
        ("pool_ids", "correct_pool_ids", "wrong_pool_ids", "unknown_pool_ids")]
    assert correct | wrong | unknown == pool
    assert not (correct & wrong or correct & unknown or wrong & unknown)
    final = set(audit["final_inlier_ids"])
    assert set(audit["correct_final_inlier_ids"]) == final & correct
    assert set(audit["wrong_final_inlier_ids"]) == final & wrong
    assert set(audit["unknown_final_inlier_ids"]) == final & unknown
    for index in pool:
        error = audit["reference_error_input_native_px"][index]
        expected = unknown if error is None else (correct if error<=audit["accurate_threshold_px"] else wrong)
        assert index in expected, (audit["method"], audit["id"], index)
    checks["truth_partition_rows_checked"] += 1
    initial = raw.get("initial_pose") or {}
    actual = raw.get("actual_pose") or {}
    solver = raw["solver"]
    initial_dim = dimension(initial.get("cf_extents"), raw["xyz"])
    selected_dim = dimension(actual.get("cf_extents"), raw["xyz"])
    local = bool(audit["local_point_line_refinement"])
    geometry = solver.get("geometry", {})
    point_inliers = bool(audit["point_PnP_inliers_applicable"])
    if local:
        assert not point_inliers, "point-line fit must not invent point PnP inliers"
    row = {"schema": "geometry_difficulty_row_v1", "id": audit["id"],
           "session": raw["session"], "method": audit["method"],
           "condition": audit.get("condition"), "reference_kind": "GEOMETRIC_PROXY",
           "coordinate_phase": audit["oracle_phase"],
           "reference_matched": audit["reference_matched"],
           "accuracy_threshold_px": audit["accurate_threshold_px"],
           "pool_count": len(pool), "pool_ids": audit["pool_ids"],
           "accurate_pool_count": len(correct), "accurate_pool_ids": audit["correct_pool_ids"],
           "inaccurate_pool_ids": audit["wrong_pool_ids"], "unknown_pool_ids": audit["unknown_pool_ids"],
           "reference_error_input_native_px": audit["reference_error_input_native_px"],
           "accurate_pool_ge4": len(correct) >= 4,
           "accurate_pool_layout_m": audit["correct_pool_layout_m"],
           "accurate_pool_shape_rank": audit["correct_pool_shape_rank"],
           "accurate_pool_shape_singular_values": audit["correct_pool_shape_singular_values"],
           "accurate_pool_image_layout": image_shape([raw["input_points"][i] for i in sorted(correct)]),
           "accurate_pool_relative_third_singular_value":
               (audit["correct_pool_shape_singular_values"][2]/audit["correct_pool_shape_singular_values"][0]
                if len(audit["correct_pool_shape_singular_values"]) >= 3 and
                audit["correct_pool_shape_singular_values"][0] else None),
           "final_inlier_count": len(final), "final_inlier_ids": audit["final_inlier_ids"],
           "accurate_final_inlier_count": audit["correct_final_inlier_count"],
           "accurate_final_inlier_ids": audit["correct_final_inlier_ids"],
           "inaccurate_final_inlier_ids": audit["wrong_final_inlier_ids"],
           "unknown_final_inlier_ids": audit["unknown_final_inlier_ids"],
           "point_PnP_inliers_applicable": point_inliers,
           "all_final_point_inliers_reference_accurate": bool(final) and point_inliers and final <= correct,
           "human_direct_pool_ids": audit["human_direct_pool_ids"],
           "human_direct_accurate_pool_ids": audit["human_direct_correct_pool_ids"],
           "false_excluded_direct_ids": audit["false_excluded_direct_ids"],
           "false_excluded_accurate_ids": audit["false_excluded_accurate_ids"],
           "false_retained_human_self_ids": audit["false_retained_human_self_ids"],
           "mask_role": mask_role(audit["method"]),
           "mask_wrong_on_known": audit["mask_wrong_on_known"],
           "paired_pose_outcome": paired_outcome(audit),
           "new_pose_estimated": audit["new_pose_estimated"], "fallback_used": audit["fallback_used"],
           "no_pose": audit["no_pose"], "output_status": raw["output_status"],
           "solver_reason": solver.get("reason"),
           "translation_cm": audit["translation_cm"], "rotation_deg": audit["rotation_deg"],
           "translation_delta_cm": audit["translation_delta_cm"],
           "rotation_delta_deg": audit["rotation_delta_deg"],
           "ADDsym_m": (raw.get("pose") or {}).get("ADDsym_m"),
           "initial_dimension_index": initial_dim, "selected_dimension_index": selected_dim,
           "initial_cf_extents_m": initial.get("cf_extents"), "selected_cf_extents_m": actual.get("cf_extents"),
           "initial_historical_label": initial.get("selected_hypothesis"),
           "selected_hypothesis_label": actual.get("selected_hypothesis"),
           "dimension_switched_on_new_pose": bool(audit["new_pose_estimated"] and
               initial_dim is not None and selected_dim is not None and initial_dim != selected_dim),
           "solver_image_layout": geometry.get("image"),
           "solver_object_layout": geometry.get("object"),
           "solver_local_jacobian": geometry.get("jacobian"),
           "weak_four_point_consensus": solver.get("weak_four_point_consensus", False),
           "local_point_line_refinement": local, "point_line_rank": audit["point_line_rank"],
           "normalized_jacobian_condition": audit["normalized_jacobian_condition"],
           "stored_candidates": alternatives(raw, checks)}
    return row


def arm_summary(rows):
    ge4 = [r for r in rows if r["accurate_pool_ge4"]]
    point = [r for r in rows if r["point_PnP_inliers_applicable"]]
    summary = metric_summary(rows)
    summary.update(accurate_pool_histogram=histogram(rows, "accurate_pool_count"),
                   accurate_pool_lt4=len(rows)-len(ge4), accurate_pool_ge4=len(ge4),
                   accurate_pool_ge5=sum(r["accurate_pool_count"] >= 5 for r in rows),
                   accurate_pool_ge6=sum(r["accurate_pool_count"] >= 6 for r in rows),
                   accurate_pool_ge4_shape_rank=histogram(ge4, "accurate_pool_shape_rank"),
                   accurate_pool_ge4_image_rank=dict(Counter(str(r["accurate_pool_image_layout"]["numerical_rank"]) for r in ge4)),
                   frames_with_unknown_pool=sum(bool(r["unknown_pool_ids"]) for r in rows),
                   point_inlier_applicable_frames=len(point),
                   accurate_final_point_inliers_ge4=sum(r["accurate_final_inlier_count"] >= 4 for r in point),
                   accurate_pool_ge4_but_accurate_final_inliers_lt4=sum(
                       r["accurate_pool_ge4"] and r["accurate_final_inlier_count"] < 4 for r in point),
                   accurate_pool_metrics={str(k): metric_summary([r for r in rows if r["accurate_pool_count"]==k])
                                         for k in sorted({r["accurate_pool_count"] for r in rows})},
                   final_inlier_metrics={str(k): metric_summary([r for r in point if r["final_inlier_count"]==k])
                                         for k in sorted({r["final_inlier_count"] for r in point})},
                   all_reference_accurate_final_point_inlier_metrics={str(k): metric_summary([
                       r for r in point if r["final_inlier_count"]==k and
                       r["new_pose_estimated"] and r["all_final_point_inliers_reference_accurate"]])
                       for k in sorted({r["final_inlier_count"] for r in point})})
    # No-mask mismatch is deliberately kept out of a classifier-error table.
    if rows[0]["mask_role"] == "predicted_geometry_mask":
        summary["mask_known_relation"] = {
            "wrong": metric_summary([r for r in rows if r["mask_wrong_on_known"]]),
            "matching_known_human_self_states": metric_summary([r for r in rows if not r["mask_wrong_on_known"]]),
            "wrong_with_accurate_pool_ge4": sum(r["mask_wrong_on_known"] and r["accurate_pool_ge4"] for r in rows),
            "matching_with_accurate_pool_ge4": sum(not r["mask_wrong_on_known"] and r["accurate_pool_ge4"] for r in rows)}
    new = [r for r in rows if r["new_pose_estimated"]]
    changed = [r for r in new if r["dimension_switched_on_new_pose"]]
    unchanged = [r for r in new if not r["dimension_switched_on_new_pose"]]
    summary["initial_vs_selected_dimension"] = {
        "historical_label_warning": "short/long-face-front labels do not equal registry WD/DW indices",
        "transition_counts_new_pose_only": dict(Counter(
            f'{r["initial_dimension_index"]}->{r["selected_dimension_index"]}' for r in new)),
        "switched": metric_summary(changed), "same_dimension": metric_summary(unchanged)}
    stored = [r for r in new if r["stored_candidates"]["coverage"] != "no_new_point_pose"]
    opposing = [r for r in stored if r["stored_candidates"]["best_stored_opposing_dimension"]]
    opposing_entries = [r["stored_candidates"]["best_stored_opposing_dimension"] for r in opposing]
    equal = [a for a in opposing_entries if a["inlier_gap"] == 0]
    nearest = [r["stored_candidates"]["nearest"] for r in stored if r["stored_candidates"]["nearest"]]
    summary["stored_alternative_coverage"] = {
        "new_point_pose_frames": len(stored), "opposing_dimension_recorded": len(opposing),
        "opposing_dimension_not_recorded": len(stored)-len(opposing),
        "nearest_alternative_same_dimension": sum(a["same_dimension"] for a in nearest),
        "nearest_alternative_different_dimension": sum(not a["same_dimension"] for a in nearest),
        "nearest_alternative_inlier_gap_counts": dict(Counter(str(a["inlier_gap"]) for a in nearest)),
        "nearest_alternative_sse_margin_px2": distributions(a["sse_margin_px2"] for a in nearest),
        "equal_inlier_count_nearest_alternative_sse_margin_px2": distributions(
            a["sse_margin_px2"] for a in nearest if a["inlier_gap"]==0),
        "opposing_inlier_gap_counts": dict(Counter(str(a["inlier_gap"]) for a in opposing_entries)),
        "opposing_same_inlier_ids": sum(a["same_inlier_ids_as_chosen"] for a in opposing_entries),
        "equal_inlier_count_opposing_sse_margin_px2": distributions(a["sse_margin_px2"] for a in equal),
        "opposing_physical_SO3_separation_deg": distributions(a["separation_from_chosen"]["physical_SO3_deg"] for a in opposing_entries),
        "opposing_proper_C2_separation_deg": distributions(a["separation_from_chosen"]["physical_proper_C2_deg"] for a in opposing_entries),
        "opposing_centroid_separation_cm": distributions(a["separation_from_chosen"]["centroid_cm"] for a in opposing_entries),
        "opposing_coverage_by_final_inlier_count": dict(Counter(str(r["final_inlier_count"]) for r in opposing)),
        "equal_inlier_count_opposing_and_dimension_switched": sum(
            r["dimension_switched_on_new_pose"] and r["stored_candidates"]["best_stored_opposing_dimension"]["inlier_gap"]==0
            for r in opposing),
        "equal_count_sse_margin_under_1e_minus_8_ids_descriptive_only": [
            r["id"] for r in opposing if r["stored_candidates"]["best_stored_opposing_dimension"]["inlier_gap"]==0
            and abs(r["stored_candidates"]["best_stored_opposing_dimension"]["sse_margin_px2"]) < 1e-8],
        "limitation": "At most two alternatives were saved. Absence of an opposing dimension is missing coverage, not evidence of uniqueness. Same dimension does not prove same basin. Margins are lexicographic: inlier gap first; SSE margin is comparable only at equal inlier count."}
    return summary


def protected_check(root, snapshot):
    if snapshot is None:
        return {"checked": False, "reason": "no --protected-state supplied"}
    state = json.loads(snapshot.read_text())
    files = state["protected_files"]
    for record in files:
        path = root / record["path"]
        assert path.is_file() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], record["path"]
    return {"checked": True, "files": len(files), "all_unchanged": True,
            "baseline_commit": state["baseline_commit"], "snapshot_sha256": sha(snapshot)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--protected-state", type=Path)
    args = p.parse_args()
    started = time.monotonic()
    root = args.root.resolve()
    old = root / "_docs/experiments" / OLD
    output = args.output_dir or root / "_docs/experiments" / NEW
    outputs = [output / name for name in ("GEOMETRY_DIFFICULTY.json", "GEOMETRY_DIFFICULTY_ROWS.jsonl.gz")]
    if any(path.exists() for path in outputs):
        p.error("refusing to overwrite completed artifacts; use a fresh --output-dir")
    before = protected_check(root, args.protected_state)
    bindings = [{"path": str((old/name).relative_to(root)), "sha256": sha(old/name),
                 "bytes": (old/name).stat().st_size} for name in READ_FILES]
    inputs = json.loads((old/"INPUTS.json").read_text())["frames"]
    ids = {r["id"] for r in inputs}
    assert len(inputs) == len(ids) == 319
    predictions = {}
    for name in ("PREDICTIONS.jsonl.gz", "LEARNED_PREDICTIONS.jsonl.gz", "REAL_MASK_STRESS.jsonl.gz"):
        for row in read_rows(old/name):
            key = (row["method"], row["id"])
            assert key not in predictions
            predictions[key] = row
    audit = read_rows(old/"REAL_CORRESPONDENCE_ROWS.jsonl.gz")
    assert len(audit) == len(predictions) == 6699
    assert len({(a["method"], a["id"]) for a in audit}) == 6699
    checks = Counter()
    rows = [decompose(a, predictions[(a["method"], a["id"])], checks) for a in audit]
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    for method, group in groups.items():
        assert len(group) == 319 and {r["id"] for r in group} == ids, method
    assert len(groups) == 21
    summaries = {method: arm_summary(group) for method, group in groups.items()}
    selected_examples = [("N3_SUBPIX_GEOM_NOSELF_ROBUST", "plastic_night_01:038630"),
                         ("N3_SUBPIX_GEOM_NOSELF_ROBUST", "plastic_day_01:011497"),
                         ("N3_SUBPIX_GEOM_NOSELF_ROBUST", "eval_pallet09:1778653806958839552")]
    lookup = {(r["method"], r["id"]): r for r in rows}
    examples = [lookup[key] for key in selected_examples]
    analytic = read_rows(old/"GEOMETRY_STRESS.jsonl.gz")
    assert len(analytic) == 2816
    near = next(r for r in analytic if r["scene_id"] == "ANALYTIC_000" and
                r["condition"] == "NEAR_COLLINEAR" and r["solver"] == "ROBUST")
    analytic_example = {"scene_id": near["scene_id"], "condition": near["condition"],
                        "solver": near["solver"], "reference_kind": "EXACT_ANALYTIC_TRUTH",
                        "translation_cm": near["translation_cm"], "rotation_deg": near["rotation_deg"],
                        "correct_remaining_count": near["correct_remaining_count"],
                        "inlier_correct_count": near["inlier_correct_count"],
                        "inlier_wrong_count": near["inlier_wrong_count"],
                        "geometry": near["result"].get("geometry"),
                        "inlier_ids": near["result"].get("final_inliers"),
                        "reason": near["result"].get("reason"),
                        "new_fit_in_this_diagnostic": False}
    assert summaries[PRIMARY[0]]["accurate_pool_ge4"] == 169
    assert summaries[PRIMARY[1]]["accurate_pool_ge4"] == 192
    assert summaries[PRIMARY[0]]["initial_vs_selected_dimension"]["switched"]["frames"] == 35
    assert summaries[PRIMARY[1]]["initial_vs_selected_dimension"]["switched"]["frames"] == 31
    assert examples[0]["accurate_final_inlier_count"] == 4 and examples[0]["dimension_switched_on_new_pose"]
    assert examples[0]["solver_local_jacobian"]["numerical_rank"] == 6
    assert examples[0]["stored_candidates"]["nearest"]["same_dimension"]
    assert not examples[0]["stored_candidates"]["best_stored_opposing_dimension"]["same_dimension"]
    after = protected_check(root, args.protected_state)
    output.mkdir(parents=True, exist_ok=True)
    with outputs[1].open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as zipped:
            for row in rows:
                zipped.write((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n").encode())
    summary = {"schema": "geometry_difficulty_v1", "complete": True,
        "scope": "Read-only decomposition of original sealed real/analytic results. No original result is replaced.",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_bindings": bindings,
        "reference_contract": {
            "real_reference_kind": "GEOMETRIC_PROXY",
            "input_accuracy": "Finite, valid, matched reference and input; native coordinate distance <=8 px in the original frozen baseline corner phase.",
            "unknown": "Missing/unmatched/invalid reference coordinates remain unknown, not inaccurate.",
            "human_visibility": "DIRECT_VISIBLE labels do not guarantee accurate input coordinates; matching self masks do not guarantee sufficient accurate correspondence geometry.",
            "hidden_reference_limit": "Some hidden reference coordinates are pose-derived; they are not independent physical observations.",
            "metric_units": "translation cm, rotation deg, ADDsym m; stored original metric values reused, not recalculated against new GT.",
            "sufficiency": "Accurate pool >=4 is a nominal diagnostic at a fixed pixel tolerance, not a necessary or sufficient guarantee of low pose error or unique pose. A four-point hypothesis requires four allowed correspondences; their true accuracy is a separate property. Shape rank and local Jacobian rank do not establish global uniqueness.",
            "rotation_separation": "Stored-pose separation reports raw SO(3) and minimum proper C2 yaw-pi separation separately; neither is a GT error.",
            "dimension_index": "0=registry (W,H,D); 1=(D,H,W). Compare extents, not historical short/long labels.",
            "mask_metrics": "Classifier-like wrong/matching counts reported only where a predicted mask was applied; no-mask mismatches are not classifier failures.",
            "sample_variance": "ddof=1, P90 linear interpolation. Dispersion is error variation, not uncertainty of the mean."},
        "populations": {"real_frames": 319, "sessions": len({r["session"] for r in inputs}),
                        "arms": 21, "rows": len(rows), "original_analytic_rows_read": len(analytic)},
        "by_method": summaries, "examples": examples,
        "analytic_false_consensus_example": analytic_example,
        "checks": {**dict(checks), "all_21_arms_contain_same_319_ids": True,
                   "accurate_inaccurate_unknown_partitions_verified": True,
                   "four_and_five_point_examples_from_original_rows": True,
                   "before_protected_state": before, "after_protected_state": after},
        "execution": {"new_detector_calls": 0, "new_head_calls": 0, "new_training_updates": 0,
                      "new_PnP_or_optimizer_calls": 0, "new_RGB_or_mask_renders": 0,
                      "algebraic_reprojections_of_stored_poses_only": True,
                      "wall_seconds": time.monotonic()-started,
                      "script_sha256": sha(Path(__file__))},
        "next_action_evidence": {
            "global_ambiguity": "A full-rank, all-four-reference-accurate consensus can switch dimensions and greatly worsen both reference pose errors. Nearest alternative margins commonly describe the same dimension while the competing dimension is missing from the saved top two.",
            "diagnostic": "Record the best candidate per dimension and its inlier-count/SSE tuple, including physical pose separation; preserve nominal pool and unknown-reference distinctions.",
            "causal_ablation": "A separate new-investigation wrapper can freeze the dimension chosen by the same-input initial pose while refitting six pose DOF on unchanged masked observations. It tests the effect of dimension preservation, not observational disambiguation or an independent physical truth gate.",
            "caution": "Initial dimension can itself be wrong; helpful and harmful original branch switches must both be counted. No correspondence absence should be filled with initial corners without an independent observation-validity gate."},
        "output_rows": {"path": str(outputs[1].relative_to(root)) if outputs[1].is_relative_to(root) else outputs[1].name,
                        "sha256": sha(outputs[1]), "bytes": outputs[1].stat().st_size}}
    outputs[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"complete": True, "rows": len(rows), "arms": len(groups),
                      "checks": dict(checks), "wall_seconds": summary["execution"]["wall_seconds"],
                      "protected_files": after.get("files"), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
