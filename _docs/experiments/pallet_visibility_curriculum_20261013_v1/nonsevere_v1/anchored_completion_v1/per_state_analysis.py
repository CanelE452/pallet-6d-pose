"""Post-seal state diagnostics; no model selection, prediction or tuning."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261013
REPEATS = 10000


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, float)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
        median=float(np.median(values)) if len(values) else None,
        P90=float(np.quantile(values, .9)) if len(values) else None,
        maximum=float(values.max()) if len(values) else None)


def paired(entries):
    if not entries:
        return dict(corners=0, frames=0, sessions=0, delta_px=distribution([]), CI95=None)
    differences = np.asarray([item[2] for item in entries], float)
    sessions = sorted({item[1] for item in entries})
    summed = np.array([sum(item[2] for item in entries if item[1] == s) for s in sessions])
    count = np.array([sum(item[1] == s for item in entries) for s in sessions])
    sampled = np.random.default_rng(SEED).integers(0, len(sessions), (REPEATS, len(sessions)))
    boot = summed[sampled].sum(1) / count[sampled].sum(1)
    by_frame = defaultdict(list)
    for fid, _, delta in entries:
        by_frame[fid].append(delta)
    return dict(corners=len(entries), frames=len(by_frame), sessions=len(sessions),
        delta_px=distribution(differences), CI95=np.quantile(boot, [.025, .975]).tolist() if len(sessions) > 1 else None,
        session_interval_informative=len(sessions) > 1,
        degenerate_single_session_resample_quantiles=np.quantile(boot, [.025, .975]).tolist() if len(sessions) == 1 else None,
        corner_improved=int((differences < -1e-6).sum()),
        corner_worsened=int((differences > 1e-6).sum()),
        corner_unchanged=int((np.abs(differences) <= 1e-6).sum()),
        frame_equal_weighted_mean_delta_px=float(np.mean([np.mean(v) for v in by_frame.values()])),
        improved_frame_ids=sorted(fid for fid, v in by_frame.items() if np.mean(v) < -1e-6),
        worsened_frame_ids=sorted(fid for fid, v in by_frame.items() if np.mean(v) > 1e-6),
        bootstrap="Whole recording-session resampling; point-weighted mean; shared corners/frames retained; few sessions, exploratory")


def main():
    output = HERE / "PER_STATE_FINDINGS_V1.json"
    if output.exists():
        raise FileExistsError("Preserve analysis")
    selection = read(HERE / "SELECTION_CAL.json")
    group_reports, bindings = {}, {"SELECTION_CAL.json": sha(HERE / "SELECTION_CAL.json")}
    for group in ("CAL_N3", "TEST_N3", "TEST_PF"):
        directory = HERE / group
        seal, protocol = read(directory / "PREDICTION_SEAL.json"), read(directory / "PROTOCOL.json")
        assert seal["complete"] and seal["before_scoring"]
        assert seal["predictions_sha256"] == sha(directory / "PREDICTIONS.jsonl.gz")
        assert seal["protocol_sha256"] == sha(directory / "PROTOCOL.json")
        assert seal["observations_sha256"] == sha(directory / "OBSERVATIONS.jsonl.gz")
        results = read(directory / "RESULTS.json")
        assert results["predictions_sha256"] == seal["predictions_sha256"]
        chosen = selection["selected"] if group == "CAL_N3" else "SELECTED"
        coordinates = rows(directory / "COORDINATE_ROWS.jsonl.gz")
        predictions = {(r["id"], r["method"]): r for r in rows(directory / "PREDICTIONS.jsonl.gz")}
        evidence = {r["id"]: r["observation"] for r in rows(directory / "OBSERVATIONS.jsonl.gz")}
        coordinate_index = {(r["method"], r["id"], r["corner"]): r for r in coordinates}
        selected = [r for r in coordinates if r["method"] == chosen]
        label_path = HERE.parent / ("OOF_pose_v1/ISOLATED_CAL_SCORING.json" if group == "CAL_N3" else "test_reused_v1/ISOLATED_TEST_SCORING.json")
        assert sha(label_path) == results["labels_sha256"]
        truth = read(label_path)["rows"]
        assert set(truth) == set(protocol["frame_ids"])
        assert len(truth) == (46 if group == "CAL_N3" else 49)
        bindings[group] = {name: sha(directory / name) for name in ("PROTOCOL.json", "PREDICTION_SEAL.json", "RESULTS.json", "COORDINATE_ROWS.jsonl.gz", "PREDICTIONS.jsonl.gz")}
        bindings[group]["isolated_labels_sha256"] = sha(label_path)
        categories = sorted({r["category"] for r in selected})
        state_stats = {}
        covariance_rows, measurement_rows, used_state = [], [], Counter()
        for state in categories:
            points = [r for r in selected if r["category"] == state]
            raw_valid = [r for r in points if r["raw_prediction_error_px"] is not None]
            comparisons = {"completed_minus_raw_same_valid_corners": paired([(r["id"], r["session"], r["model_error_px"] - r["raw_prediction_error_px"]) for r in raw_valid]),
                "completed_minus_initial_rigid_same_corners": paired([(r["id"], r["session"], r["model_error_px"] - r["initial_rigid_error_px"]) for r in points])}
            for control in ("OLD_PROGRESSIVE", "BOUNDARY_ONLY", "UNPROTECTED", "STRICT_POOL"):
                entries = []
                for row in points:
                    baseline = coordinate_index.get((control, row["id"], row["corner"]))
                    if baseline is not None:
                        assert baseline["category"] == state
                        entries.append((row["id"], row["session"], row["model_error_px"] - baseline["model_error_px"]))
                if entries:
                    comparisons["selected_minus_" + control] = paired(entries)
            state_stats[state] = dict(corners=len(points), frames=len({r["id"] for r in points}),
                sessions=len({r["session"] for r in points}), raw_valid_corners=len(raw_valid),
                raw_same_valid_px=distribution([r["raw_prediction_error_px"] for r in raw_valid]),
                completion_same_raw_valid_px=distribution([r["model_error_px"] for r in raw_valid]),
                completion_all_GT_valid_px=distribution([r["model_error_px"] for r in points]),
                initial_rigid_same_GT_valid_px=distribution([r["initial_rigid_error_px"] for r in points]),
                comparisons=comparisons)
        for row in selected:
            fid, k = row["id"], row["corner"]
            prediction = predictions[fid, chosen]
            solution = prediction["solution"]
            mapping = np.asarray(truth[fid]["metadata"]["native_to_reference_corner_ids"], int)
            assert sorted(mapping.tolist()) == list(range(8))
            target = np.asarray(truth[fid]["target"]["gt"], float)[:8][mapping]
            error = np.asarray(prediction["completed_points"], float)[k] - target[k]
            assert abs(np.linalg.norm(error) - row["model_error_px"]) < 1e-6
            if k in solution["used_ids"]:
                used_state[row["human_state"]] += 1
            measured = np.asarray(prediction["measured_points"], float)[k]
            original = np.asarray(prediction["original_points"], float)[k]
            if np.isfinite(measured).all() and np.isfinite(original).all() and not np.all(original == -1):
                displacement = float(np.linalg.norm(measured-original))
                if displacement > 1e-8:
                    measurement_rows.append(dict(id=fid, session=row["session"], corner=k, category=row["category"],
                        measurement_role=solution["keypoint_status"][k]["measurement_role"],
                        raw_error_px=float(np.linalg.norm(original-target[k])),
                        measured_error_px=float(np.linalg.norm(measured-target[k])),
                        completed_error_px=row["model_error_px"], applied_shift_px=displacement,
                        used_for_pose=k in solution["used_ids"]))
            covariance = solution["uncertainty"]["projected_covariance_px2"]
            if covariance is not None and covariance[k] is not None:
                matrix = np.asarray(covariance[k], float)
                if np.isfinite(matrix).all() and np.linalg.eigvalsh(matrix).min() > 1e-12:
                    distance2 = float(error @ np.linalg.solve(matrix, error))
                    covariance_rows.append(dict(id=fid, session=row["session"], corner=k, category=row["category"],
                        mahalanobis_squared=distance2, inside_nominal_95ellipse=distance2 <= 5.991464547107979,
                        estimated_dof=solution["estimated_dof"], used_for_pose=k in solution["used_ids"]))
        measurement_stats = {}
        for role in sorted({r["measurement_role"] for r in measurement_rows}):
            subset = [r for r in measurement_rows if r["measurement_role"] == role]
            measurement_stats[role] = dict(points=len(subset), frames=len({r["id"] for r in subset}),
                raw_error_px=distribution([r["raw_error_px"] for r in subset]),
                measured_error_px=distribution([r["measured_error_px"] for r in subset]),
                completed_error_px=distribution([r["completed_error_px"] for r in subset]),
                delta_measured_minus_raw=paired([(r["id"], r["session"], r["measured_error_px"] - r["raw_error_px"]) for r in subset]))
        covariance_stats = {}
        for state in sorted({r["category"] for r in covariance_rows}):
            subset = [r for r in covariance_rows if r["category"] == state]
            covariance_stats[state] = dict(points=len(subset), frames=len({r["id"] for r in subset}),
                empirical_inside_fraction=float(np.mean([r["inside_nominal_95ellipse"] for r in subset])),
                mahalanobis_squared=distribution([r["mahalanobis_squared"] for r in subset]))
        selected_predictions = [p for (fid, method), p in predictions.items() if method == chosen]
        selection_modes = Counter("BOUNDARY_SEED_EXCEPTION" if p["solution"]["final_guard"]["independent_boundary_seed_exception"]
            else "ACCEPTED_FULL_POOL" if p["solution"]["accepted"] else "EXACT_INITIAL" for p in selected_predictions)
        changed = [p for p in selected_predictions if not np.allclose(np.asarray(p["completed_points"])[:8], np.asarray(predictions[p["id"], "INITIAL"]["completed_points"])[:8], atol=1e-8, rtol=0)]
        group_reports[group] = dict(selected_method=chosen, whole_pose_frames=len(protocol["frame_ids"]),
            pose_summary={method: value for method, value in results["summary"].items() if method in (chosen, "INITIAL", "BOUNDARY_ONLY", "OLD_PROGRESSIVE", "OLD_JOINT", "UNPROTECTED", "STRICT_POOL")},
            coordinate_frames=results["coordinate_phase_compatible_frames"],
            coordinate_incompatible_ids=results["coordinate_phase_incompatible_ids"], categories=state_stats,
            absent_categories=[name for name in ("TRUNCATED", "FULLY_HIDDEN", "OCCLUDED") if name not in categories],
            decision_modes=dict(selection_modes), final_pose_changed_frames=len(changed),
            RGB_junction_measurement_frames=sum(bool(p["solution"]["RGB_junction_measurement_ids"]) for p in selected_predictions),
            RGB_junction_used_frames=sum(bool(set(p["solution"]["RGB_junction_measurement_ids"]) & set(p["solution"]["used_ids"])) for p in selected_predictions),
            final_used_human_states=dict(used_state), changed_measurement_summary=measurement_stats,
            changed_measurement_rows=measurement_rows,
            conditional_covariance_diagnostic=dict(coverage_by_category=covariance_stats,
                finite_positive_covariance_corners=len(covariance_rows), all_coordinate_corners=len(selected),
                empirical_inside_fraction=float(np.mean([r["inside_nominal_95ellipse"] for r in covariance_rows])) if covariance_rows else None,
                nominal_level=.95, chi2_two_dimensional_cutoff=5.991464547107979,
                confidence_calibrated=False, covariance_is_weighted_L2_surrogate=True,
                conditions_on_initial_rotation_in_three_DOF=True, fallback_covariance_absent=True,
                omitted_errors=["corner_semantic_bias", "camera_and_dimension_error", "shared_edge_correlation", "upstream_fixed_rotation_error", "robust_influence", "global_pose_or_correspondence_multimodality"],
                rows=covariance_rows))
    result = dict(schema="POST_SEAL_INDEPENDENT_PER_STATE_FINDINGS_V1", group_reports=group_reports,
        analysis_script_sha256=sha(Path(__file__)),
        bindings=bindings, bootstrap=dict(seed=SEED, repetitions=REPEATS, unit="recording_session", selection_or_tuning=False),
        scope=dict(CAL46_and_TEST49_isolated_only=True, all_seals_verified_before_labels=True,
            labels_used_for_scoring_diagnostics_only=True, unseen_TEST=False, FINAL_reads=0,
            severe_frames=0, models_or_hyperparameters_changed=False, new_predictions_or_training=0),
        caveats=["Whole-cohort pose includes all46/49; coordinate diagnostics are44/46 CAL and42/49 TEST after a fixed model/reference phase compatibility check",
            "GT poses and hidden corner targets are historical annotation reconstructions, not independent physical measurements",
            "Raw and final coordinate comparisons use the exact same finite original raw corners; initial-rigid and old-progressive comparisons use exact matching frame/native slots",
            "DIRECT_OUTLINE/MODEL_INTERIOR categories use the initial geometric outline, not certified actual background contact",
            "UNANNOTATED coordinates are separate diagnostics and are not evidence of semantic hidden/visible correction",
            "Human truncation and fully-hidden labels absent from this eligible-coordinate cohort; no performance conclusion for those categories",
            "Completion versus raw includes the benefit already present in initial rigid projection; it is not solely new refinement contribution",
            "Nominal covariance ellipse coverage is an exploratory failure diagnostic, not a claim of calibration",
            "No variant or threshold is selected from these TEST/state/coverage outcomes",
            "Single-recording categories cannot estimate between-recording uncertainty; their degenerate bootstrap quantiles are not labelled as an informative95%CI."])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output, sha(output), flush=True)
    for group, report in group_reports.items():
        print(group, "decisions", report["decision_modes"], "changed", report["final_pose_changed_frames"], "RGB used", report["RGB_junction_used_frames"])
        print("covariance", report["conditional_covariance_diagnostic"]["empirical_inside_fraction"], report["conditional_covariance_diagnostic"]["finite_positive_covariance_corners"])
        for state, stats in report["categories"].items():
            print(state, stats["corners"], round(stats["completion_all_GT_valid_px"]["mean"], 4),
                {name: [round(value["delta_px"]["mean"], 4), [round(v, 4) for v in value["CI95"]] if value["CI95"] else None] for name, value in stats["comparisons"].items()})


if __name__ == "__main__":
    main()
