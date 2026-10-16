"""Read-only postseal invariant/objective/coordinate audit of seven fixed arms."""
from collections import Counter, defaultdict
from pathlib import Path
import difflib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric, distribution

HERE = Path(__file__).resolve().parent
NS = HERE.parent


def main():
    destination = HERE / "INFERENCE_AUDIT.json"
    if destination.exists():
        raise FileExistsError("Preserve completed independent audit")
    protocol = C.read(HERE / "PROTOCOL.json")
    seal = C.read(HERE / "PREDICTION_SEAL.json")
    raw = list(C.rows(HERE / "PREDICTIONS.jsonl.gz"))
    observations = {r["id"]: r["observation"] for r in C.rows(HERE / "OBSERVATIONS.jsonl.gz")}
    frames = {r["id"]: r for r in C.read(NS / "PREDICTION_INPUTS.json")["frames"] if r["partition"] == "TEST"}
    cohort = C.read(NS / "COHORT_PROTOCOL.json")
    previous = {(r["id"], r["method"]): r for r in C.rows(NS / "test_reused_v1/PREDICTIONS.jsonl.gz")}
    boundary = C.read(NS / "test_reused_v1/GEOMETRIC_BOUNDARY_VISIBILITY_AUDIT_V1.json")
    boundary_ids = {r["id"]: r["selected_native_ids"] for r in boundary["frame_selection_records"]}
    checks, failures, max_error = Counter(), [], defaultdict(float)

    def check(name, value, detail=None):
        checks[name] += 1
        if not bool(value):
            failures.append(dict(check=name, detail=detail))

    def equal(name, left, right, detail=None, tolerance=1e-10):
        left, right = np.asarray(left, float), np.asarray(right, float)
        check(name, left.shape == right.shape and np.allclose(left, right, rtol=0, atol=tolerance, equal_nan=True), detail)
        if left.shape == right.shape and left.size:
            delta = np.abs(left - right)
            finite = delta[np.isfinite(delta)]
            if finite.size:
                max_error[name] = max(max_error[name], float(finite.max()))

    check("complete343_seal", seal["complete"] and seal["before_scoring"] and seal["rows"] == len(raw) == 343)
    check("prediction_binding", C.sha(HERE / "PREDICTIONS.jsonl.gz") == seal["predictions_sha256"])
    check("observations_binding", C.sha(HERE / "OBSERVATIONS.jsonl.gz") == seal["observations_sha256"])
    check("protocol_binding", C.sha(HERE / "PROTOCOL.json") == seal["protocol_sha256"])
    check("annotation_canary", seal["GT_read_canary"] == "PASS")
    check("exact49_no_severe", set(frames) == set(observations) == set(protocol["frame_ids"]) and len(frames) == 49
          and not set(frames) & set(cohort["excluded_severe_ids"]))
    counts = Counter(r["method"] for r in raw)
    check("seven_arms49_each", counts == {r["name"]: 49 for r in protocol["arms"]})
    scoring_amendment = C.read(HERE / "SCORING_CODE_AMENDMENT.json")
    changes = {}
    for source, digest in protocol["source_sha256"].items():
        path = Path(source)
        if path.suffix == ".py":
            snapshot = HERE / "code_snapshot" / path.relative_to(ROOT)
            check("immutable_source_snapshot", C.sha(snapshot) == digest, source)
        current = C.sha(path)
        if current != digest:
            changes[source] = dict(frozen=digest, current=current)
        if path == HERE / "replay.py":
            check("scoring_only_amendment_binding", digest == scoring_amendment["inference_source_sha256"]
                  and current == scoring_amendment["scoring_source_sha256"] and scoring_amendment["before_scoring"], source)
        else:
            check("model_config_population_code_binding", current == digest, source)
    snapshot_runner = HERE / "code_snapshot" / (HERE / "replay.py").relative_to(ROOT)
    runner_diff = list(difflib.unified_diff(snapshot_runner.read_text().splitlines(),
                                         (HERE / "replay.py").read_text().splitlines(), lineterm=""))
    check("only_runner_counter_amended", set(changes) == {str(HERE / "replay.py")})

    by = {(r["id"], r["method"]): r for r in raw}
    checkpoint_counts, guard_fallbacks, unknown_counts, support_counts = Counter(), Counter(), Counter(), Counter()
    stage_records, coordinate_records, guard_records, parity_records = [], [], [], []
    maximum_shift, corrected_count, boundary_selected = 0., 0, 0

    def objective(xyz, measurement, K, R, t, R0, t0, ids, weights, cfg):
        ids = np.asarray(ids, int)
        residual = np.linalg.norm(C.project(xyz, R, t, K)[ids] - measurement[ids], axis=1)
        radius = cfg["robust_px"]
        loss = np.where(residual <= radius, residual ** 2, 2 * radius * residual - radius ** 2)
        data_sum = float(np.sum(loss * weights[ids]))
        scale = max(np.linalg.norm(t0), np.linalg.norm(np.ptp(xyz, axis=0)), 1e-6)
        rotation = cv2.Rodrigues(np.asarray(R) @ R0.T)[0].reshape(3)
        translation = (np.asarray(t) - t0) / scale
        prior = float(cfg["prior_strength_px"] ** 2 * (
            np.dot(rotation, rotation) / np.deg2rad(cfg["rotation_prior_deg"]) ** 2
            + np.dot(translation, translation) / cfg["translation_prior_fraction"] ** 2))
        total = max(float(weights[ids].sum()), 1e-12)
        return dict(objective=(data_sum + prior) / total, data_objective=data_sum / total,
                    prior_cost_sum=prior, frozen_weight_sum=total)

    def compare_objective(name, recomputed, stored, detail):
        equal(name, [recomputed[k] for k in recomputed], [stored[k] for k in recomputed], detail, 1e-8)

    for fid, frame in frames.items():
        q = np.asarray(frame["points"]["N3_SUBPIX"], float)
        K = np.asarray(frame["K"], float)
        initial = frame["initial_poses"]["N3_SUBPIX"]
        R0, t0 = np.asarray(initial["R_cf"], float), np.asarray(initial["centroid"], float)
        xyz = C.cuboid(*initial["cf_extents"])
        Q = R0.T @ np.asarray(initial["R_physical"], float)
        H = C.self_hidden(xyz, R0, t0)
        h, w = frame["raw_hw"]
        valid = np.isfinite(q[:8]).all(1) & ~np.all(q[:8] == -1, axis=1)
        inside = valid & (q[:8, 0] >= 0) & (q[:8, 0] < w) & (q[:8, 1] >= 0) & (q[:8, 1] < h)
        o = observations[fid]
        fused = np.asarray(o["fused_points"], float)
        observed = np.asarray(o["observed_points"], float)
        check("DEV_TEST_only", frame["population_role"] == "DEV" and frame["partition"] == "TEST", fid)
        check("actual_image_SHA", C.sha(ROOT / frame["image"]) == frame["image_sha256"], fid)
        equal("observation_raw_order_center", o["raw_points"], q, fid, 0)
        equal("observation_fixed_H", o["self_hidden"], H, fid, 0)
        equal("hidden_missing_outside_fusion_preserved", fused[:8][H | ~inside], q[:8][H | ~inside], fid, 0)
        equal("observed_and_fused_center", np.vstack([fused[8:], observed[8:]]), np.vstack([q[8:], q[8:]]), fid, 0)
        shift = np.linalg.norm(fused[:8][valid] - q[:8][valid], axis=1)
        maximum_shift = max(maximum_shift, float(shift.max(initial=0)))
        corrected_count += int((shift > 1e-12).sum())
        check("fusion_maximum6px_and_canvas", np.all(shift <= 6 + 1e-8) and np.all(fused[:8][inside, 0] >= 0)
              and np.all(fused[:8][inside, 0] < w) and np.all(fused[:8][inside, 1] >= 0) and np.all(fused[:8][inside, 1] < h), fid)
        check("one_fresh_BOX_ONLY_mask", o["mask"]["mask_forward_calls"] == 1 and o["mask"]["policy"] == "BOX_ONLY"
              and not o["mask"]["points"] and not o["mask"]["labels"], fid)
        quality = np.asarray(o["mask"]["quality"])
        check("mask_quality_selection_first_argmax", o["mask"]["selected_index"] == int(np.argmax(quality)), fid)
        expected_box = np.r_[np.maximum(q[:8][valid].min(0), [0, 0]), np.minimum(q[:8][valid].max(0), [w - 1, h - 1])]
        equal("BOX_ONLY_original_raw8_not_projection", o["mask"]["box"], expected_box, fid, 0)
        app = o["visibility_appearance"]
        check("fresh_full_RGB_appearance_previous49_exact", app == previous[fid, "SEGMENTED_RGB_CURRICULUM"]["appearance"], fid)
        selected = o["boundary_visibility"]["selected_corner_ids"]
        boundary_selected += len(selected)
        check("geometric_boundary_selection_previous_exact", selected == boundary_ids[fid], fid)
        for k in selected:
            equal("boundary_p_previous_exact", o["boundary_visibility"]["p_visible"][k], app["visibility_score"][k], [fid, k], 0)
        for r in raw:
            if r["id"] != fid:
                continue
            method, pose = r["method"], r["actual_pose"]
            equal("all343_original_native_q", r["original_points"], q, [fid, method], 0)
            equal("registered_original_family_preserved", pose["cf_extents"], initial["cf_extents"], [fid, method], 0)
            check("family_registered", np.allclose(pose["cf_extents"], frame["xyz"], atol=1e-9)
                  or np.allclose(pose["cf_extents"], np.asarray(frame["xyz"])[[2, 1, 0]], atol=1e-9), [fid, method])
            equal("physicalQ_preserved", pose["R_physical"], np.asarray(pose["R_cf"]) @ Q, [fid, method], 1e-12)
            if method == "INITIAL":
                check("frozen_initial_pose", pose == initial, fid)
                continue
            sol = r["solution"]
            measurement = q[:8] if method.startswith("RAW_") else fused[:8]
            equal("arm_measurement_binding", r["observed_points"], q if method.startswith("RAW_") else fused, [fid, method], 0)
            equal("solver_no_projection_measurement", sol["observed_points"], measurement, [fid, method], 0)
            equal("all294_fixed_H", r["self_hidden"], H, [fid, method], 0)
            equal("canvasK_reprojection", r["projected_points"], C.project(xyz, pose["R_cf"], pose["centroid"], K), [fid, method])
            check("immutable_appearance_all_arms", r["appearance"] == app, [fid, method])
            check("final_H_fit_exclusion", not any(H[k] or not inside[k] for k in sol["used_ids"]), [fid, method])
            completed = q.copy()
            if sol["accepted"]:
                completed[:8] = measurement
                completed[np.flatnonzero(H)] = np.asarray(r["projected_points"])[H]
            else:
                check("fallback_exact_initial", pose == initial, [fid, method])
            equal("completion_only_after_fit_and_center", r["completed_points"], completed, [fid, method], 0)
            cfg = sol["configuration"]
            states = np.asarray(app["states"])
            visibility = np.clip(np.asarray(app["visibility_score"]), 0, 1)
            accuracy = np.clip(np.asarray(app["local_accuracy_score"]), 0, 1)
            independent = np.asarray(app.get("independent", [True] * 8)) & inside & ~H
            excludes = np.isin(states, ["SELF_OCCLUDED", "SELF_HIDDEN", "OUT_OF_FRAME", "TRUNCATED", "OCCLUSION_SUSPECTED", "EXTERNAL_OCCLUDED"])
            eligible = valid & independent & ~H & ~excludes
            visible = eligible & np.isin(states, ["VISIBLE_BOUNDARY", "VISIBLE_INTERIOR"]) & (visibility >= cfg["visible_threshold"]) & (accuracy >= cfg["minimum_accuracy"])
            unknown = eligible & (states == "UNKNOWN")
            weights = np.zeros(8)
            weights[visible] = np.maximum(visibility[visible], .05) * np.maximum(accuracy[visible], .05)
            weights[visible & (states == "VISIBLE_INTERIOR")] *= cfg["interior_weight"]
            weights[unknown] = cfg["unknown_weight"] * np.maximum(accuracy[unknown], .05)
            visible_ids = np.flatnonzero(visible).tolist()
            check("frozen_visible_pool", sol["visible_ids"] == visible_ids, [fid, method])
            for index, stage in enumerate(sol["stages"]):
                check("all_stages_H_outside_excluded", not any(H[k] or not inside[k] for k in stage["used_ids"]), [fid, method, index])
                if stage.get("weights") is not None:
                    equal("frozen_stage_weights", stage["weights"], weights[stage["used_ids"]], [fid, method, index], 0)
                if stage["name"].startswith("add_consistent_unknown_"):
                    unknown_counts[method + ("_accepted" if stage["accepted"] else "_rejected")] += 1
                    if stage.get("intermediate_accept_discarded_by_final_guard"):
                        unknown_counts[method + "_restored_away"] += 1
                if not cfg["progressive_active_support"]:
                    continue
                if "active_objective_before" in stage:
                    before = objective(xyz, measurement, K, stage["pose_before_R"], stage["pose_before_t"], R0, t0, stage["used_ids"], weights, cfg)
                    compare_objective("active_SUM_prior_before", before, stage["active_objective_before"], [fid, method, index])
                if "active_objective_after" in stage:
                    after = objective(xyz, measurement, K, stage["candidate_R"], stage["candidate_t"], R0, t0, stage["used_ids"], weights, cfg)
                    compare_objective("active_SUM_prior_after", after, stage["active_objective_after"], [fid, method, index])
                    policy = "legacy_unknown_visible_guards" if stage["name"].startswith("add_consistent_unknown_") else "progressive_active_support"
                    check("unknown_legacy_validation_preserved", stage["validation_policy"] == policy, [fid, method, index])
                    if stage["accepted"] and policy == "progressive_active_support":
                        check("accepted_active_gain", before["objective"] - after["objective"] > cfg["minimum_gain_px2"], [fid, method, index])
            if not cfg["progressive_active_support"]:
                continue
            equal("frozen_weights8_exact", sol["frozen_weights_by_id"], weights, [fid, method], 0)
            guard = sol["final_guard"]
            check("all_omitted_still_final_guard_pool", guard["all_visible_ids"] == guard["all_visible_ids"] == visible_ids
                  and guard["all_omitted_visible_ids"] == sorted(set(visible_ids) - set(sol["used_ids"])), [fid, method])
            reference = objective(xyz, measurement, K, R0, t0, R0, t0, visible_ids, weights, cfg)
            compare_objective("final_initial_SUM_prior", reference, guard["initial"], [fid, method])
            accepted_indices = [i for i, s in enumerate(sol["stages"]) if i and s["accepted"] and s.get("fit_performed", True)]
            check("every_accepted_fit_checkpoint_present", [c["stage_index"] for c in guard["checkpoints"]] == accepted_indices, [fid, method])
            passing = []
            for checkpoint in guard["checkpoints"]:
                stage_index = checkpoint["stage_index"]
                stage = sol["stages"][stage_index]
                full = objective(xyz, measurement, K, stage["R"], stage["t"], R0, t0, visible_ids, weights, cfg)
                compare_objective("checkpoint_all_visible_SUM_prior", full, checkpoint["full_visible"], [fid, method, stage_index])
                gain = reference["objective"] - full["objective"]
                equal("checkpoint_gain", gain, checkpoint["initial_objective_gain"], [fid, method, stage_index], 1e-8)
                check("checkpoint_pass_threshold", checkpoint["passes_initial_all_visible_guard"] == (gain > cfg["minimum_gain_px2"]), [fid, method, stage_index])
                if gain > cfg["minimum_gain_px2"]:
                    passing.append(stage_index)
            retained = passing[-1] if passing else None
            check("latest_temporal_passing_checkpoint", guard["retained_checkpoint_index"] == retained, [fid, method])
            guard_fallbacks[method + ":" + guard["fallback_source"]] += 1
            if retained is None:
                check("no_passing_restores_initial", not sol["accepted"] and not sol["used_ids"] and pose == initial, [fid, method])
            else:
                stage = sol["stages"][retained]
                equal("retained_checkpoint_output_R", sol["R"], stage["R"], [fid, method], 0)
                equal("retained_checkpoint_output_t", sol["t"], stage["t"], [fid, method], 0)
                check("retained_checkpoint_output_support", sol["used_ids"] == stage["used_ids"], [fid, method])
            output = objective(xyz, measurement, K, sol["R"], sol["t"], R0, t0, visible_ids, weights, cfg)
            compare_objective("output_SUM_prior", output, guard["output_objective"], [fid, method])
            check("final_initial_objective_guarantee", output["objective"] <= reference["objective"] + 1e-9, [fid, method])
            check("no_batch_or_GT_checkpoint_selection", not guard["final_batch_refit_performed"] and not guard["GT_stage_selection"], [fid, method])
            support_counts[method + "_omitted_visible"] += len(guard["all_omitted_visible_ids"])
            guard_records.append(dict(id=fid, method=method, retained=retained, omitted=guard["all_omitted_visible_ids"],
                                      fallback=guard["fallback_source"], initial=reference, output=output))
        for prefix in ("RAW", "FUSED"):
            active, together = by[fid, prefix + "_ACTIVE"], by[fid, prefix + "_TOGETHER"]
            d = np.linalg.norm(np.asarray(active["projected_points"]) - np.asarray(together["projected_points"]), axis=1)
            parity_records.append(dict(id=fid, coordinates=prefix, maximum_projection_difference_px=float(d.max()),
                                       same_used_ids=active["solution"]["used_ids"] == together["solution"]["used_ids"],
                                       same_accepted=active["solution"]["accepted"] == together["solution"]["accepted"]))
    check("271_boundary_subset_unchanged", boundary_selected == 271)
    # Only this permitted isolated49 truth file is accessed, after seal checks.
    truth = C.read(NS / "test_reused_v1/ISOLATED_TEST_SCORING.json")["rows"]
    check("truth_exact49", set(truth) == set(frames))
    scored = [dict(r, metric=pose_metric(r["actual_pose"], truth[r["id"]]["target"]["truth"])) for r in raw]
    lookup = {(r["id"], r["method"]): r for r in scored}
    summary, rescues = {}, {}
    results = C.read(HERE / "RESULTS.json")
    for method in counts:
        group = [r for r in scored if r["method"] == method]
        summary[method] = dict(frames=len(group), accepted=sum(r["solution"]["accepted"] for r in group),
            success=sum(r["metric"]["success_5cm_5deg"] for r in group),
            metrics={k: distribution([r["metric"][k] for r in group]) for k in ("translation_cm", "rotation_deg", "ADDsym_cm")})
        check("pose_success_reproduced", summary[method]["success"] == results["summary"][method]["success"], method)
        for metric in summary[method]["metrics"]:
            equal("pose_mean_reproduced", summary[method]["metrics"][metric]["mean"], results["summary"][method]["metrics"][metric]["mean"], [method, metric], 1e-12)
        rescues[method] = dict(rescued_ids=[r["id"] for r in group if r["metric"]["success_5cm_5deg"] and not lookup[r["id"], "INITIAL"]["metric"]["success_5cm_5deg"]],
            lost_ids=[r["id"] for r in group if not r["metric"]["success_5cm_5deg"] and lookup[r["id"], "INITIAL"]["metric"]["success_5cm_5deg"]])
    for fid, frame in frames.items():
        meta = truth[fid]["metadata"]
        target = truth[fid]["target"]
        gt = np.asarray(target["gt"])[meta["native_to_reference_corner_ids"]]
        valid = np.asarray(target["valid"])[meta["native_to_reference_corner_ids"]]
        q = np.asarray(observations[fid]["raw_points"])[:8]
        fused = np.asarray(observations[fid]["fused_points"])[:8]
        for k, state in enumerate(meta["human_states_native"]):
            if state == "DIRECT_VISIBLE" and valid[k] and np.isfinite(gt[k]).all():
                coordinate_records.append(dict(id=fid, session=frame["session"], corner=k,
                    raw_error_px=float(np.linalg.norm(q[k] - gt[k])), fused_error_px=float(np.linalg.norm(fused[k] - gt[k])),
                    changed=bool(np.linalg.norm(q[k] - fused[k]) > 1e-12)))
    def coordinate_summary(records):
        before = np.asarray([r["raw_error_px"] for r in records])
        after = np.asarray([r["fused_error_px"] for r in records])
        return dict(n=len(records), raw=distribution(before), fused=distribution(after),
            improved=int((after < before - 1e-10).sum()), worsened=int((after > before + 1e-10).sum()),
            unchanged=int((np.abs(after - before) <= 1e-10).sum()),
            within_px={str(t): dict(raw=int((before <= t).sum()), fused=int((after <= t).sum())) for t in (2, 4, 8, 16)})
    report = dict(schema="INDEPENDENT_REUSED_TEST49_PROGRESSIVE_AUDIT_V1", status="PASS" if not failures else "FAIL",
        checks_executed=dict(checks), failures=failures, maximum_numeric_errors=dict(max_error),
        rows=343, frames=49, severity="CLEAN/MODERATE only; original95cohort bound", no_model_promotion=True,
        source_bindings=dict(immutable_snapshot_verified=True, current_changes=changes,
                             scoring_counter_amendment=scoring_amendment, exact_runner_diff=runner_diff),
        observations=dict(fresh_SAM_calls=49, fused_changed_points=corrected_count, maximum_fused_shift_px=maximum_shift,
                          boundary_subset271_and_RGB_probabilities_exact_previous=True),
        final_guards=dict(fallback_counts=dict(guard_fallbacks), omitted_counts=dict(support_counts),
                          unknown_attempt_counts=dict(unknown_counts), records=guard_records,
                          limitation="Current49 has no earlier-passing fallback or accepted-unknown rollback; mechanism is not empirically exercised here."),
        active_vs_together=dict(records=parity_records, maximum_projection_difference_px=max(r["maximum_projection_difference_px"] for r in parity_records),
            interpretation="Active support allows more intermediate stages; final poses converge numerically to the batch/together control. No material ordering performance advantage established."),
        independently_reproduced_pose=summary, rescue_loss_vs_initial=rescues,
        DIRECT_coordinate=dict(all=coordinate_summary(coordinate_records), actually_changed=coordinate_summary([r for r in coordinate_records if r["changed"]]), records=coordinate_records),
        read_scope=dict(TEST_GT_files=[str(NS / "test_reused_v1/ISOLATED_TEST_SCORING.json")], TEST_RGB_decodes=0, TEST_image_SHA_reads=49,
                        severe_or_FINAL_image_annotation_reads=0, new_inference_or_training=0),
        limitations=["Previously observedTEST49 with4sessions, not independent confirmation. All primary49frames retained.",
            "An initial-reference corner phase mismatch can confound point errors; the existing fixed phase is preserved, not relabelled here.",
            "Line covariance is conditional and not calibratedsemantic accuracy; detector sigma2/fusion alpha are fixed assumptions.",
            "Final guard protects frozen visible weightedSUM plus initialprior, not every individual residual or groundtruth pose accuracy.",
            "Sourcecodediff changes only accepted-fit diagnostic counter afterseal and before scoring; predictions/models/inference snapshots are unchanged."])
    C.write(destination, report)
    print(json.dumps(dict(status=report["status"], checks=sum(checks.values()), failures=failures[:12],
        path=str(destination), fused_active=summary["FUSED_ACTIVE"], rescues=rescues["FUSED_ACTIVE"],
        coordinates=report["DIRECT_coordinate"]["actually_changed"], parity_max=report["active_vs_together"]["maximum_projection_difference_px"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
