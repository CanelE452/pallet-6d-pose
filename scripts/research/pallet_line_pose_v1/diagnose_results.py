"""Read-only posthoc geometry diagnosis of all nine completed paper evaluations.

No inference, calibration, rule selection, or source mutation occurs here. GT
defines explanatory strata only. This file is independent of the frozen driver.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import paper_evaluation as PE


ARMS = ("image_joint", "geometry_joint", "image_line_only")
SEEDS = (1, 2, 3)
EDGES = ((1, 2), (3, 0), (5, 6), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
ROLE_NAMES = ("front_right_height", "front_left_height", "rear_right_height",
              "rear_left_height", "left_top_depth", "right_top_depth",
              "right_bottom_depth", "left_bottom_depth")
BINS = ("easy_le5", "moderate_5to10", "hard_gt10", "unavailable")
MATERIALS = {"plastic_standard_110x130x11": "plastic", "wood_small_80x59x14": "wood"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def check(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Input SHA mismatch: {path}")
    return actual


def distribution(values):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p90=float(np.quantile(values, .9)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def difficulty(value):
    return ("unavailable" if not np.isfinite(value) else "easy_le5" if value <= 5
            else "moderate_5to10" if value <= 10 else "hard_gt10")


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else np.nan


def points_of(candidate):
    if candidate is None or candidate.get("keypoints_xy") is None:
        return None
    points = np.asarray(candidate["keypoints_xy"], dtype=float)
    return points if points.shape == (9, 2) and np.isfinite(points).all() else None


def line_through(first, second):
    delta = second - first
    length = np.linalg.norm(delta)
    if length < 1e-9:
        return None
    normal = np.array([-delta[1], delta[0]]) / length
    return np.r_[normal, -normal @ first]


def endpoint_distance(h, endpoints):
    return float(np.abs(endpoints @ h[:2] + h[2]).mean() / np.linalg.norm(h[:2]))


def paired_summary(baseline, after, mask):
    """Frame is the sampling unit; three seed-specific errors are averaged."""
    mask = np.asarray(mask, bool)
    paired = mask & np.isfinite(baseline) & np.isfinite(after).all(0)
    left, right = baseline[paired], after[:, paired]
    average = right.mean(0)
    delta = average - left
    return dict(population_frames=int(mask.sum()), paired_frames=int(paired.sum()),
        unavailable_frames=int((mask & ~paired).sum()),
        baseline_frame_mean_error_px=distribution(left),
        refined_seedmean_frame_mean_error_px=distribution(average),
        delta_frame_mean_error_px=distribution(delta),
        improved_frames=int((delta < -1e-9).sum()), worsened_frames=int((delta > 1e-9).sum()),
        unchanged_frames=int((np.abs(delta) <= 1e-9).sum()),
        improved_fraction=float((delta < -1e-9).mean()) if len(delta) else None,
        per_seed={str(seed): dict(refined_mean_px=float(right[i].mean()) if len(left) else None,
                  mean_delta_px=float((right[i] - left).mean()) if len(left) else None,
                  improved_frames=int((right[i] < left - 1e-9).sum()))
                  for i, seed in enumerate(SEEDS)})


def line_summary(before, after, mask=None):
    # Same frame/role support in all three seeds; do not inflate n by three.
    valid = np.isfinite(before) & np.isfinite(after).all(0)
    if mask is not None:
        valid &= mask
    left, right = before[valid], after[:, valid]
    delta = right.mean(0) - left
    frame_mask = valid.any(-1) if valid.ndim == 2 else valid
    return dict(paired_frame_roles=int(valid.sum()), paired_frames=int(frame_mask.sum()),
        baseline_endpoint_distance_px=distribution(left),
        learned_seedmean_endpoint_distance_px=distribution(right.mean(0)),
        delta_endpoint_distance_px=distribution(delta),
        improved_fraction=float((delta < -1e-9).mean()) if len(delta) else None,
        per_seed_mean_distance_px={str(seed): finite_mean(right[i]) for i, seed in enumerate(SEEDS)})


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def self_test():
    assert [difficulty(x) for x in (5, 5.01, 10, 10.01, np.nan)] == list(BINS[:2]) + [BINS[1], BINS[2], BINS[3]]
    h = line_through(np.array([2., 3.]), np.array([2., 9.]))
    assert endpoint_distance(h, np.array([[4., 1.], [4., 8.]])) == 2
    p, q, gt = np.array([2., 3.]), np.array([4., 8.]), np.array([1., 9.])
    e, d = p-gt, q-p
    assert abs(np.sum((q-gt)**2)-np.sum(e**2)-(2*e@d+d@d)) < 1e-12
    result = paired_summary(np.array([2., 4., np.nan]), np.array([[1., 8., 0.]]*3), np.ones(3, bool))
    assert result["paired_frames"] == 2 and result["improved_frames"] == 1
    assert result["delta_frame_mean_error_px"]["mean"] == 1.5


def diagnose(run_dir):
    marker_path = run_dir / "REAL_EVALUATION_COMPLETE.json"
    marker = read(marker_path)
    if not marker.get("complete") or not marker.get("PASS") or marker.get("expected_runs") != 9:
        raise ValueError("All nine actual evaluations must complete before diagnosis")
    if {(r["arm"], r["seed"]) for r in marker["runs"]} != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("Incomplete or unexpected arm/seed population")
    selection_path = run_dir / "SELECTION.json"
    selection_sha = check(selection_path, marker["selection_sha256"])
    baseline_path = run_dir / "baseline/FULL_CANDIDATES.json"
    check(baseline_path, marker["baseline_cache_sha256"])
    protocol_path = run_dir / "BASELINE_PROTOCOL.json"
    protocol = read(protocol_path)
    source_sha = {str(p.resolve()): sha(p) for p in (Path(__file__), marker_path, selection_path,
                                                   baseline_path, protocol_path, PE.POS)}
    for name, digest in protocol["source_sha256"].items():
        source_sha[name] = check(name, digest)
    for name, digest in marker["source_sha256"].items():
        source_sha[name] = check(name, digest)
    baseline = read(baseline_path)
    pair = PE.population()
    items = list(pair.positive.items)
    targets = [PE.E._legacy_forbidden_target(item) for item in items]
    n = len(items)
    assert n == 319
    keys = [PE.canonical_key(item.image) for item in items]
    gt = np.stack([target.keypoints_xy for target in targets])
    gt_mask = np.stack([target.keypoint_supervision_mask for target in targets])
    p = np.full((n, 9, 2), np.nan)
    base_error = np.full((n, 9), np.nan)
    base_line_error = np.full((n, 8), np.nan)
    matched = np.zeros(n, bool)
    frame_meta = []
    for j, (item, target, key) in enumerate(zip(items, targets, keys)):
        candidates = baseline["frames"][key]
        top = candidates[int(np.argmax([c["score"] for c in candidates]))] if candidates else None
        points = points_of(top)
        if points is not None:
            p[j] = points
        matched[j] = (points is not None and PE.E._box_iou(np.array(top["box_xyxy"]), target.box_xyxy) >= .5)
        if matched[j]:
            base_error[j, gt_mask[j]] = np.linalg.norm(points-gt[j], axis=1)[gt_mask[j]]
            for role, (a, b) in enumerate(EDGES):
                h = line_through(points[a], points[b])
                if gt_mask[j, a] and gt_mask[j, b] and np.linalg.norm(gt[j, a]-gt[j, b]) >= 2 and h is not None:
                    base_line_error[j, role] = endpoint_distance(h, gt[j, [a, b]])
        frame_meta.append(dict(frame_id=item.frame_id, image_key=key, session_id=item.session_id,
            object_type=item.object_type, material=MATERIALS[item.object_type],
            domain=getattr(item, "domain", None) or "UNKNOWN",
            gt_supervised_points=int(gt_mask[j].sum()), iou50_matched=bool(matched[j])))
    base_frame_error = np.array([finite_mean(v) for v in base_error])
    for j, value in enumerate(base_frame_error):
        frame_meta[j].update(baseline_frame_mean_error_px=value, difficulty=difficulty(value))
    arrays = {}
    audit = dict(csv_max_abs_error_delta_px=0., paper_median_p90_max_abs_delta_px=0.,
                 center_max_abs_delta_px=0., squared_error_identity_max_abs_delta_px2=0.,
                 line_affine_max_abs_coefficient_delta_px=0., line_unit_normal_max_abs_delta=0.,
                 baseline_candidate_parity=True, all_nine_complete=True)
    for arm in ARMS:
        arrays[arm] = dict(error=np.full((3, n, 9), np.nan), movement=np.full((3, n, 9), np.nan),
            line_error=np.full((3, n, 8), np.nan), null=np.full((3, n, 8), np.nan),
            entropy=np.full((3, n, 8), np.nan), head_used=np.zeros((3, n), bool),
            e_dot_d=np.full((3, n, 9), np.nan), move_squared=np.full((3, n, 9), np.nan))
        out = arrays[arm]
        for si, seed in enumerate(SEEDS):
            receipt = next(r for r in marker["runs"] if r["arm"] == arm and r["seed"] == seed)
            directory = Path(receipt["output_dir"])
            for filename, field in (("COMPLETION.json", "completion_sha256"), ("RESULTS.json", "result_sha256"),
                                    ("INFERENCE_AUDIT.json", "inference_audit_sha256"), ("IDENTITY_AUDIT.json", "identity_audit_sha256")):
                source_sha[str(directory/filename)] = check(directory/filename, receipt[field])
            inference_audit = read(directory/"INFERENCE_AUDIT.json")
            assert inference_audit["complete"] and inference_audit["PASS"]
            pred_path = directory/"IMAGE_PREDICTIONS.json"
            source_sha[str(pred_path)] = check(pred_path, inference_audit["image_predictions_sha256"])
            csv_path = directory/"PAPER_2D_per_frame.csv"
            source_sha[str(csv_path)] = sha(csv_path)
            csv_rows = {r["frame_id"]: r for r in csv.DictReader(csv_path.open()) if r["kind"] == "POSITIVE"}
            saved = read(pred_path)
            assert saved["complete"] and saved["bindings"]["selection_sha256"] == selection_sha
            records = {r["image_key"]: r for r in saved["records"]}
            assert set(records) == set(keys) and len(records) == n
            for j, key in enumerate(keys):
                pred = records[key]["prediction"]
                assert pred["arm"] == arm and pred["seed"] == seed and not pred["already_padded"]
                assert pred["coordinate_frame"] == "supplied_image_pixels"
                source_sha[str(PE.REPO/key)] = check(PE.REPO/key, records[key]["image_sha256"])
                selected = pred["selected_index"]
                if selected is None:
                    assert not np.isfinite(p[j]).any()
                    continue
                assert np.array_equal(np.asarray(pred["baseline_selected"]["keypoints_xy"]), p[j])
                q = points_of(pred["candidates"][selected])
                if q is None:
                    assert not matched[j]
                    continue
                d = q-p[j]
                out["movement"][si, j] = np.linalg.norm(d, axis=1)
                audit["center_max_abs_delta_px"] = max(audit["center_max_abs_delta_px"], float(np.abs(d[8]).max()))
                out["head_used"][si, j] = pred["head_used"]
                if matched[j]:
                    e = p[j]-gt[j]
                    out["error"][si, j, gt_mask[j]] = np.linalg.norm(q-gt[j], axis=1)[gt_mask[j]]
                    out["e_dot_d"][si, j, gt_mask[j]] = (e*d).sum(-1)[gt_mask[j]]
                    out["move_squared"][si, j, gt_mask[j]] = (d*d).sum(-1)[gt_mask[j]]
                    discrepancy = (np.linalg.norm(q-gt[j], axis=1)**2-np.linalg.norm(e, axis=1)**2
                                   -2*(e*d).sum(-1)-(d*d).sum(-1))
                    audit["squared_error_identity_max_abs_delta_px2"] = max(audit["squared_error_identity_max_abs_delta_px2"],
                        float(np.abs(discrepancy[gt_mask[j]]).max(initial=0)))
                csv_error = np.array([float(x) for x in csv_rows[items[j].frame_id]["top_keypoint_supervised_errors_px"].split(";") if x])
                actual_error = out["error"][si, j, np.isfinite(out["error"][si, j])]
                assert len(csv_error) == len(actual_error)
                audit["csv_max_abs_error_delta_px"] = max(audit["csv_max_abs_error_delta_px"], float(np.abs(csv_error-actual_error).max(initial=0)))
                diag = pred.get("diagnostics")
                if diag is None:
                    continue
                assert np.allclose(diag["move_px"], out["movement"][si, j], atol=1e-9, rtol=1e-9)
                h = np.asarray(diag["line_h_supplied_image"], float)
                hin = np.asarray(diag["line_h_input"], float)
                line_valid = np.asarray(diag["line_valid"], bool)
                ih, iw = diag["input_shape_hw"]
                rh, rw = pred["raw_shape_hw"]
                gain = min(640/(rh+200), 640/(rw+200))
                offset = np.array([round((iw-round((rw+200)*gain))/2-.1), round((ih-round((rh+200)*gain))/2-.1)])
                assert abs(gain-diag["gain"]) < 1e-12
                rebuilt = hin.copy()
                rebuilt[:, 2] = (hin[:, 2]+hin[:, :2]@(offset+100*gain))/gain
                audit["line_affine_max_abs_coefficient_delta_px"] = max(audit["line_affine_max_abs_coefficient_delta_px"], float(np.abs(rebuilt-h).max()))
                audit["line_unit_normal_max_abs_delta"] = max(audit["line_unit_normal_max_abs_delta"], float(np.abs(np.linalg.norm(h[line_valid, :2], axis=1)-1).max(initial=0)))
                out["null"][si, j, line_valid] = np.asarray(diag["null_probability"])[line_valid]
                out["entropy"][si, j, line_valid] = np.asarray(diag["entropy_normalized"])[line_valid]
                for role, (a, b) in enumerate(EDGES):
                    if line_valid[role] and np.isfinite(base_line_error[j, role]):
                        out["line_error"][si, j, role] = endpoint_distance(h[role], gt[j, [a, b]])
            flat = out["error"][si][np.isfinite(out["error"][si])]
            result = read(directory/"RESULTS.json")["two_d"]
            for quantile, name in ((.5, "keypoint_location_median_px"), (.9, "keypoint_location_p90_px")):
                delta = abs(float(np.quantile(flat, quantile))-result[name])
                audit["paper_median_p90_max_abs_delta_px"] = max(audit["paper_median_p90_max_abs_delta_px"], delta)
        out["frame_error"] = np.array([[finite_mean(v) for v in row] for row in out["error"]])
    assert audit["csv_max_abs_error_delta_px"] <= 5.1e-7
    assert audit["paper_median_p90_max_abs_delta_px"] <= 1e-9
    assert audit["center_max_abs_delta_px"] == 0
    assert audit["squared_error_identity_max_abs_delta_px2"] <= 1e-7
    assert audit["line_affine_max_abs_coefficient_delta_px"] <= 1e-9
    assert audit["line_unit_normal_max_abs_delta"] <= 2e-6
    groups = {"difficulty": list(BINS), "domain": ["DAY", "NIGHT", "UNKNOWN"],
              "material": ["plastic", "wood"], "session_id": sorted({r["session_id"] for r in frame_meta})}
    summaries = {}
    for arm, arr in arrays.items():
        def describe(mask):
            summary = paired_summary(base_frame_error, arr["frame_error"], mask)
            for key in ("movement", "null", "entropy"):
                # Each point/role is first averaged across the same three seeds.
                values = arr[key][:, mask, :8] if key == "movement" else arr[key][:, mask]
                summary[key + ("_px" if key == "movement" else "")] = distribution(values.mean(0))
            summary["line_geometry"] = line_summary(base_line_error, arr["line_error"], mask[:, None])
            ed, dd = arr["e_dot_d"][:, mask, :8], arr["move_squared"][:, mask, :8]
            valid = np.isfinite(ed) & np.isfinite(dd) & (dd > 1e-12)
            summary["displacement_geometry"] = dict(
                evaluated_moving_corner_seed_instances=int(valid.sum()),
                scope="Supported matched corners0..7; repeated seeds are descriptive instances, not independent samples",
                wrong_initial_direction_fraction=float((ed[valid] > 0).mean()) if valid.any() else None,
                mean_2e_dot_d_px2=finite_mean(2*ed), mean_move_squared_px2=finite_mean(dd),
                mean_squared_error_delta_px2=finite_mean(2*ed+dd))
            return summary
        all_mask = np.ones(n, bool)
        summary = dict(all_frames=describe(all_mask), per_seed_head_used_frames=arr["head_used"].sum(1).tolist(),
                       selected_rule=read(selection_path)["selected_rules"][arm])
        for field, labels in groups.items():
            summary["by_"+field] = {label: describe(np.array([r[field] == label for r in frame_meta])) for label in labels}
        summary["line_roles"] = {ROLE_NAMES[r]: dict(role=r, endpoints=list(EDGES[r]),
            **line_summary(base_line_error[:, r], arr["line_error"][:, :, r])) for r in range(8)}
        summary["line_families"] = {family: line_summary(base_line_error[:, sl], arr["line_error"][:, :, sl])
                                    for family, sl in (("height", slice(0, 4)), ("depth", slice(4, 8)))}
        summaries[arm] = summary
    comparisons = {}
    for other in ARMS[1:]:
        left = arrays[other]["frame_error"].mean(0)
        comparisons["image_joint_minus_"+other] = dict(
            all_frames=paired_summary(left, arrays["image_joint"]["frame_error"], np.ones(n, bool)),
            by_difficulty={label: paired_summary(left, arrays["image_joint"]["frame_error"],
                np.array([r["difficulty"] == label for r in frame_meta])) for label in BINS},
            note="Comparison of complete synthetically selected pipelines; selected displacement caps may differ, so this is not an isolated one-factor loss effect.")
    for j, row in enumerate(frame_meta):
        row["arms"] = {arm: dict(per_seed_frame_mean_error_px=arr["frame_error"][:, j],
            seedmean_frame_mean_error_px=finite_mean(arr["frame_error"][:, j]),
            seedmean_delta_frame_mean_error_px=finite_mean(arr["frame_error"][:, j])-base_frame_error[j],
            per_seed_corner_mean_movement_px=[finite_mean(v) for v in arr["movement"][:, j, :8]],
            per_seed_mean_null=[finite_mean(v) for v in arr["null"][:, j]],
            per_seed_mean_entropy=[finite_mean(v) for v in arr["entropy"][:, j]],
            per_seed_line_endpoint_distance_px=arr["line_error"][:, j]) for arm, arr in arrays.items()}
        row["baseline_line_endpoint_distance_px"] = base_line_error[j]
    value = dict(schema="pallet_line_pose_posthoc_diagnosis_v1", complete=True, PASS=True,
        primary_arm="image_joint", seeds=list(SEEDS), source_sha256=source_sha,
        population=dict(positive_frames=n, matched_frames=int(matched.sum()),
            gt_supervised_points=int(gt_mask.sum()), scored_points=int(np.isfinite(base_error).sum()),
            role="DEV", held_out_final=False, negative_frames_not_rediagnosed=2689),
        method=dict(point_metric="Per-frame arithmetic mean of official GT-supervised visibility>0 points0..8 (including unchanged centroid), on frozen top-confidence IoU>=0.5 matched detection; original-image pixels",
            difficulty="Baseline frame mean <=5, (5,10], >10 px; unavailable kept as its own stratum",
            posthoc_GT_strata_not_deployment_gate=True, no_inference=True, no_real_selection=True,
            aggregation="Average the same three seed errors per frame before frame-level descriptive summaries and benefit counts; seed count does not multiply population size. No new hypothesis test or success definition.",
            line_metric="Mean absolute perpendicular distance of both GT endpoints to the unit-normal infinite line. Both corners require official visibility>0 and GT segment length>=2 original pixels; original predicted endpoints must be nondegenerate; learned line_valid must hold in all three seeds. No null/entropy threshold is applied.",
            line_semantics="Eight camera-facing-convention amodal cuboid height/depth support lines; supervision is not evidence of physical edge visibility. Width-axis four lines are not predicted.",
            null_semantics="Null class probability in 222-way tempered distribution; fusion reduces line precision by1-null",
            entropy_semantics="Conditional non-null distribution entropy divided by log(221); it is not a calibrated probability of physical correctness",
            missing_policy="All319 remain in each applicable group denominator. Conditional matched point/line means explicitly report their available count; missing cases are not assigned zero error.",
            material_mapping=MATERIALS, unknown_domain="45 frames lack a declared domain; no DAY/NIGHT inference from filenames"),
        arms=summaries, comparisons=comparisons, frames=frame_meta, geometry_audit=audit,
        observations=[], hypotheses=[
            "An improved learned support line can still worsen an already accurate corner: two incident line constraints and the point anchor jointly determine its movement; a line constrains normal error but does not identify tangent position by itself.",
            "The signed 2e·d term measures whether a displacement initially approaches GT. Even a negative term can be outweighed by ||d||² (overshoot). These GT-based decompositions explain saved outputs and cannot be used as an inference gate.",
            "Small changes in 2D points can change PnP rotation/translation differently. This diagnostic measures 2D geometry and does not override the frozen 6D evaluation or establish a causal image-feature effect."],
        limits=["Reused319-frame DEV population across13 sessions; subgroup counts can be small and correlated. These are descriptive comparisons, not an independent final-test claim.",
                "Difficulty is defined after observing GT; real difficulty thresholds, rules, temperatures and seeds were not selected here.",
                "Saved mean lines are conditional on a local candidate grid around YOLO-derived lines. They are amodal structural predictions and do not prove the model attended to a visible pallet edge.",
                "The two controls have their own synthetic-selected readout rules. Pipeline differences cannot be attributed to a single component alone."])
    for label in BINS[:3]:
        v = summaries["image_joint"]["by_difficulty"][label]
        value["observations"].append(dict(kind="measured", stratum=label, frames=v["paired_frames"],
            baseline_frame_mean_px=v["baseline_frame_mean_error_px"]["mean"],
            refined_frame_mean_px=v["refined_seedmean_frame_mean_error_px"]["mean"],
            delta_px=v["delta_frame_mean_error_px"]["mean"], improved_frames=v["improved_frames"]))
    # Verify inputs still match after reading; this script cannot silently mix revisions.
    for name, digest in source_sha.items():
        check(name, digest)
    target = run_dir/"DIAGNOSIS.json"
    temporary = target.with_suffix(".pending.json")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(target)
    print(json.dumps(dict(path=str(target), sha256=sha(target), observations=value["observations"], geometry_audit=audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Geometry/denominator self-tests PASS")
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required unless --self-test")
        diagnose(args.run_dir.resolve())
