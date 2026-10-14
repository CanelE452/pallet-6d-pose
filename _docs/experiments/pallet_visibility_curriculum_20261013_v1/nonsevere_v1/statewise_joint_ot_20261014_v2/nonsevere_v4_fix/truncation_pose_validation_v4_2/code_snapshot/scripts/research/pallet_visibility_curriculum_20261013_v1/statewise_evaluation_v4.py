"""Fresh98 RGB/full-API v4 evaluation, prediction seal before separate scoring.

Predict has no annotation access. Controls reuse the same immutable image B/I,
CAL effective covariance, nonS scale and initial registered geometry. Reuse of
a solver result requires identical mathematical-input hashes. Journals retain
every distinct feasible sample while compacting repeated candidate audits.
"""
from __future__ import annotations

import os
for _setting in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_setting] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from . import common as C
from .scoring import distribution, pose_metric
from .statewise_joint_experiment_v2 import _frames, crop_bindings, COHORT, STUDY, _model_compatible
from .statewise_observations_v4 import bindings as observation_source_bindings
from .statewise_joint_api_v4 import StatewiseJointV4, finish_observations
from .statewise_uncertainty_calibration_v4 import load_uncertainty_model, solver_config_from_model
from .statewise_uncertain_solver_v4 import solve_uncertain_joint_ot_pnp, _configuration
from .state_visibility_v2 import load_state_model
from .statewise_masks_v2 import StatewiseMaskProvider

FIX = STUDY / "nonsevere_v4_fix"
OUTPUT = FIX / "evaluation_v1"
SELECTION = FIX / "cal_selection_v1/SELECTED_CONFIG.json"
FRESH = FIX / "fresh_observations_v2"
COMPARATORS = STUDY / "diagnosis_compare_v1/comparator"
ARM_ALL = "SELECTED_V4_ALL_OT"
ARM_NO_OT = "SELECTED_V4_NO_OT"
ARM_NO_PRIOR = "SELECTED_V4_NO_PRIOR"
ARMS = (ARM_ALL, ARM_NO_OT, ARM_NO_PRIOR)
METRICS = ("translation_cm", "rotation_deg", "ADDsym_cm", "success_5cm_5deg")
PARITY = {"states_exact": True, "native_ids_exact": True,
          "BI_xy_atol_px": 1e-6, "BI_observed_covariance_atol_px2": 1e-6,
          "OT_domain_numeric_atol": 1e-6, "nonS_scale_atol_px": 1e-9,
          "K_native_geometry_initial_pose_atol": 1e-12,
          "rtol": 0.0, "mismatch_policy": "abort without a scoreable prediction seal"}
_CONTEXT = None


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(C.ROOT)), "sha256": C.sha(path), "bytes": path.stat().st_size}


def _resolve(binding):
    return C.ROOT / (binding["path"] if isinstance(binding, dict) else binding)


def _verify_binding(binding):
    path = _resolve(binding)
    if C.sha(path) != binding["sha256"]:
        raise ValueError("Frozen binding changed: " + str(path))
    return path


def _digest(value):
    data = json.dumps(C.finite(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def source_bindings():
    result = observation_source_bindings()
    for name in ("statewise_uncertain_solver_v4.py", "statewise_uncertainty_calibration_v4.py", "statewise_evaluation_v4.py"):
        path = Path(__file__).parent / name
        result[str(path.relative_to(C.ROOT))] = C.sha(path)
    return result


def _verify_sources(bindings):
    mismatch = [p for p, sha in bindings.items() if C.sha(C.ROOT / p) != sha]
    if mismatch:
        raise ValueError("Source/input bindings changed: " + ", ".join(mismatch))


def _selection(path):
    path = Path(path).resolve()
    seal_path = path.parent / "SELECTION_SEAL.json"
    if not seal_path.is_file():
        raise FileNotFoundError("CAL selection seal is required before reading SELECTED_CONFIG")
    seal = C.read(seal_path)
    if not seal.get("before_any_TEST_prediction_or_score") or C.sha(path) != seal["selected_config_sha256"]:
        raise ValueError("CAL selection not sealed before TEST evaluation")
    selected = C.read(path)
    if selected.get("TEST_files_opened") is not False or seal.get("TEST_files_opened") is not False:
        raise ValueError("Selection read TEST files")
    if not selected.get("selected_from_LOGO_CAL46_only"):
        raise ValueError("CAL-only conditional LOGO selection required")
    if selected["source_sha256"] != seal["source_sha256"]:
        raise ValueError("CAL selection source seal mismatch")
    _verify_sources(selected["source_sha256"])
    model_path = _verify_binding(selected["calibration_model"])
    expected_model = FIX / "calibration/final_adapter_CAL46/MODEL.json"
    if model_path.resolve() != expected_model.resolve():
        raise ValueError("Final fresh-CAL46 uncertainty model required")
    cal_seal = _verify_binding(selected["CAL_prediction_seal"])
    if C.sha(cal_seal) != seal["CAL_prediction_seal_sha256"]:
        raise ValueError("CAL prediction seal changed")
    if selected["calibration_model"] != seal["calibration_model"]:
        raise ValueError("CAL selected model/seal disagree")
    if C.sha(path.parent / "PROTOCOL.json") != selected["protocol_sha256"]:
        raise ValueError("CAL selection protocol changed")
    score_sha = C.sha(path.parent / "CAL_SELECTION_SCORE.json")
    if score_sha != selected["CAL_SELECTION_SCORE_sha256"] or score_sha != seal["CAL_score_sha256"]:
        raise ValueError("CAL-only selection score changed")
    candidate = C.read(cal_seal)
    if (not candidate.get("before_CAL_scorer_target_read") or not candidate.get("CAL_only")
            or candidate.get("TEST_files_opened") is not False or candidate["rows"] != 184
            or candidate["protocol_sha256"] != selected["protocol_sha256"]
            or C.sha(path.parent / "CANDIDATE_PREDICTIONS.jsonl.gz") != candidate["predictions_sha256"]):
        raise ValueError("CAL184 candidate prediction seal invalid")
    if float(selected["selected_prior_weight"]) not in (0., .25, 1., 4.):
        raise ValueError("Prior outside predeclared CAL-only grid")
    if float(selected["selected_prior_weight"]) != float(seal["selected_prior_weight"]):
        raise ValueError("Selected weight seal mismatch")
    model = load_uncertainty_model(model_path)
    config = _configuration(selected["solver_configuration"])
    if config["prior_weight"] != float(selected["selected_prior_weight"]):
        raise ValueError("Selected prior weight/config mismatch")
    exported = solver_config_from_model(model, config["prior_weight"])
    for name in exported:
        if _digest(config[name]) != _digest(exported[name]):
            raise ValueError("Selected solver/model calibration disagreement: " + name)
    return selected, model, config, _binding(path), _binding(seal_path)


def _runtime_models():
    path = STUDY / "RUNTIME_CONFIG_V3.json"
    config = C.read(path)
    descriptors = {name: STUDY / config[name] for name in ("state_model", "OT_model", "mask_descriptor", "boundary_mask_config")}
    bindings = {str(path.relative_to(C.ROOT)): C.sha(path)}
    for name, p in descriptors.items():
        if C.sha(p) != config["bindings_sha256"][name]:
            raise ValueError("Frozen runtime descriptor changed: " + name)
        bindings[str(p.resolve().relative_to(C.ROOT))] = C.sha(p)
    for name in ("state_model", "mask_descriptor"):
        p = descriptors[name]
        descriptor = C.read(p)
        arrays = p.parent / descriptor["arrays_file"]
        if C.sha(arrays) != descriptor["arrays_sha256"]:
            raise ValueError("Frozen state arrays changed")
        bindings[str(arrays.relative_to(C.ROOT))] = C.sha(arrays)
        for key in ("backbone", "pallet_segmenter"):
            weight = C.ROOT / descriptor[key]["weights_repo_relative"]
            if C.sha(weight) != descriptor[key]["weights_sha256"]:
                raise ValueError("Frozen backbone/segmenter weight changed")
            bindings[str(weight.relative_to(C.ROOT))] = C.sha(weight)
    sam = C.read(descriptors["boundary_mask_config"])
    weight = (descriptors["boundary_mask_config"].resolve().parent / sam["mobile_sam_checkpoint"]).resolve()
    if C.sha(weight) != sam["model_sha256"]["mobile_sam_checkpoint"]:
        raise ValueError("MobileSAM weight changed")
    # Absolute checkpoints are permitted; preserve the actual binding path.
    bindings[str(weight.relative_to(C.ROOT)) if weight.is_relative_to(C.ROOT) else str(weight)] = C.sha(weight)
    return config, descriptors, bindings


def _fresh_cache(frames):
    seal_path = FRESH / "OBSERVATION_SEAL.json"
    seal = C.read(seal_path)
    protocol_path = FRESH / "PROTOCOL.json"
    prediction_path = FRESH / "OBSERVATIONS.jsonl.gz"
    if seal["annotations_read"] or not seal["before_new_uncertainty_fitting"]:
        raise ValueError("Reference observation cache accessed annotations")
    if C.sha(prediction_path) != seal["observations_sha256"] or C.sha(protocol_path) != seal["protocol_sha256"]:
        raise ValueError("Fresh observation cache seal mismatch")
    # A cache is bound to its executed snapshot. A later metadata-only API fix
    # need not invalidate its observations; fresh actual-runtime parity below
    # still checks the current observation outputs, whose source is separately
    # frozen in this evaluation's protocol.
    for relative, digest in seal["source_sha256"].items():
        if C.sha(FRESH / "code_snapshot" / relative) != digest:
            raise ValueError("Fresh cache executed source snapshot changed: " + relative)
    current_differences = [relative for relative, digest in seal["source_sha256"].items()
                           if C.sha(C.ROOT / relative) != digest]
    rows = list(C.rows(prediction_path))
    if len(rows) != 98 or [r["id"] for r in rows] != [r["id"] for r in frames]:
        raise ValueError("Complete fresh-cache/real98 input identity mismatch")
    return {r["id"]: r for r in rows}, {"observations": _binding(prediction_path), "seal": _binding(seal_path),
        "protocol": _binding(protocol_path), "executed_snapshot_source_hashes": seal["source_sha256"],
        "current_source_differences_after_cache": current_differences,
        "current_semantics_checked_by_actual98_observation_parity": True}


def _maxdiff(a, b):
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("inf")
    return float(np.max(np.abs(x - y))) if x.size else 0.


def _recursive_equal(a, b, atol=1e-6):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_recursive_equal(a[k], b[k], atol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_recursive_equal(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= atol)
    return a == b


def observation_parity(current, cached):
    a, b = current, cached["runtime"]
    am, bm = a["frozen_visible_measurements"], b["frozen_visible_measurements"]
    ids_a, ids_b = [m["native_corner"] for m in am], [m["native_corner"] for m in bm]
    xy = max((_maxdiff(x["xy"], y["xy"]) for x, y in zip(am, bm)), default=0.)
    cov = max((_maxdiff(x["covariance_px2"], y["covariance_px2"]) for x, y in zip(am, bm)), default=0.)
    scale = abs(a["ot_domains"]["nonself_prediction_scale_px"] - b["ot_domains"]["nonself_prediction_scale_px"])
    checks = {"states_exact": a["states"] == b["states"], "native_BI_ids_exact": ids_a == ids_b,
              "BI_xy_max_abs_px": xy, "BI_covariance_max_abs_px2": cov,
              "BI_sources_exact": [m.get("source") for m in am] == [m.get("source") for m in bm],
              "OT_domains_equal_with_declared_tolerance": _recursive_equal(a["ot_domains"]["domains"], b["ot_domains"]["domains"], PARITY["OT_domain_numeric_atol"]),
              "OT_domains_hash_exact": a["ot_domains"]["domains_sha256"] == b["ot_domains"]["domains_sha256"],
              "nonS_scale_max_abs_px": scale,
              "original_raw_points_and_coarse_exact": a["original_raw_points"] == b["original_raw_points"] and a["original_coarse_pose"] == b["original_coarse_pose"],
              "RGB_image_pixel_hash_exact": a["visible_correction"]["image_pixel_sha256"] == b["visible_correction"]["image_pixel_sha256"]}
    geom_diff = max(_maxdiff(a["initial_geometry"][name], b["initial_geometry"][name]) for name in ("K", "native_xyz", "R_cf", "centroid", "physical_registration_Q"))
    checks["initial_geometry_max_abs_difference"] = geom_diff
    checks["passed"] = bool(checks["states_exact"] and checks["native_BI_ids_exact"] and checks["BI_sources_exact"]
        and checks["OT_domains_equal_with_declared_tolerance"] and checks["original_raw_points_and_coarse_exact"]
        and checks["RGB_image_pixel_hash_exact"] and xy <= PARITY["BI_xy_atol_px"]
        and cov <= PARITY["BI_observed_covariance_atol_px2"] and scale <= PARITY["nonS_scale_atol_px"]
        and geom_diff <= PARITY["K_native_geometry_initial_pose_atol"])
    return checks


def S_contract_audit(runtime):
    hidden = {i for i, flag in enumerate(runtime["initial_geometry"]["initial_self_hidden"]) if flag}
    observed = runtime["state_observation"]
    correction = runtime["visible_correction"]
    masks = runtime["mask_audit"]
    domains = runtime["ot_domains"]
    checks = {"initial_S_fixed": hidden == {i for i, s in enumerate(runtime["states"]) if s == "S"},
              "S_state_RGB_queries_zero": not hidden.intersection(observed.get("rgb_queried_ids", []))
                 and all(not r.get("rgb_evaluated", False) for r in observed["records"] if r["native_id"] in hidden),
              "S_mask_prompts_zero": not hidden.intersection(masks.get("prompt_native_ids", [])) and masks["S_excluded_before_prompt"],
              "S_correction_queries_zero": not hidden.intersection(correction.get("queried_native_ids", [])) and not correction["provenance"]["S_queried"],
              "S_BI_measurements_zero": all(m["native_corner"] not in hidden for m in runtime["frozen_visible_measurements"]),
              "S_regions_zero": all(d["native_corner"] not in hidden for d in domains["domains"]) and domains["S_image_features_queried"] == 0,
              "S_effective_fit_measurements_zero": all(m["native_corner"] not in hidden for m in runtime["calibrated_visible_measurements"]),
              "S_pose_projection_only_readout": all(runtime["keypoint_readout"]["point_sources"][i] == "FINAL_PNP_S_PROJECTION" for i in hidden),
              "S_image_terms_zero_in_solver": runtime["joint_pose"]["provenance"]["S_image_terms"] == 0,
              "S_region_terms_zero_in_solver": runtime["joint_pose"]["provenance"]["S_region_terms"] == 0}
    return {"initial_S_ids": sorted(hidden), "checks": checks, "passed": all(checks.values())}


def _physics_spec(observation, fit, config, states=None, domains=None):
    g = observation["initial_geometry"]
    return {"xyz": g["native_xyz"], "K": g["K"], "states": states or observation["states"],
            "visible": deepcopy(fit), "domains": deepcopy(observation["ot_domains"]["domains"] if domains is None else domains),
            "image_hw": g["image_hw"], "coarse_pose": {"R": g["R_cf"], "t": g["centroid"], "physical_registration_Q": g["physical_registration_Q"]},
            "initializations": None, "config": deepcopy(config)}


def _solve_spec(spec):
    return solve_uncertain_joint_ot_pnp(spec["xyz"], spec["K"], spec["states"], spec["visible"],
        spec["domains"], spec["image_hw"], spec["coarse_pose"], initializations=spec["initializations"], config=spec["config"])


def control_from_observations(observation, fit, model, config, arm, known_results=None):
    states, domains, cfg = list(observation["states"]), deepcopy(observation["ot_domains"]["domains"]), deepcopy(config)
    if arm == ARM_NO_OT:
        states = ["U" if s in ("O", "T") else s for s in states]
        domains = []
    elif arm == ARM_NO_PRIOR:
        cfg["prior_weight"] = 0.
    elif arm != ARM_ALL:
        raise ValueError("Unknown fixed control arm")
    spec = _physics_spec(observation, fit, cfg, states, domains)
    key = _digest(spec)
    cached = (known_results or {}).get(key)
    solution = deepcopy(cached) if cached is not None else _solve_spec(spec)
    control_observation = deepcopy(observation)
    control_observation["states"] = states
    control_observation["ot_domains"]["domains"] = domains
    control_observation["ot_domains"]["domains_sha256"] = _digest(domains)
    result = finish_observations(control_observation, solution, fit, model)
    if result["frozen_visible_measurements"] != observation["frozen_visible_measurements"] or result["calibrated_visible_measurements"] != fit:
        raise ValueError("Control changed immutable observed/effective B/I")
    return result, key, {"mathematical_input_sha256": key, "cached_identical_input_result": cached is not None,
                         "changes": "O/T states to U and domains empty" if arm == ARM_NO_OT else "prior_weight=0" if arm == ARM_NO_PRIOR else "none",
                         "same_BI_xy_observed_covariance_fit_covariance_reliability_and_scale": True,
                         "same_initial_registered_geometry": True}, solution


def _compact_solution(solution):
    result = deepcopy(solution)
    for record in result.get("candidates", []):
        for name in ("visible_measurement_audit", "domain_audit"):
            audit = record.pop(name, None)
            if audit is not None:
                record[name + "_count"] = len(audit) if isinstance(audit, list) else len(audit.keys())
    # Every distinct feasible sample, selected audit and optimizer outcome stays.
    if result.get("corrected_BI_seed") == result.get("BI_prior_seed"):
        result.pop("corrected_BI_seed", None)
        result["corrected_BI_seed_alias"] = "BI_prior_seed"
    result["journal_compaction"] = {"per_candidate_repeated_visible_and_domain_audits_omitted": True,
        "all_candidate_optimizer_outcomes_retained": True, "all_distinct_feasible_pose_samples_retained": True,
        "selected_pose_full_constraint_and_visible_audits_retained": True, "solver_decisions_changed": False}
    return result


def _compact_runtime(runtime):
    value = deepcopy(runtime)
    value["joint_pose"] = _compact_solution(value["joint_pose"])
    value.pop("solution", None)
    value["solution_alias"] = "joint_pose"
    return value


def _init_worker(selected, model, config, device):
    global _CONTEXT
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime_descriptor, paths, _ = _runtime_models()
    head = load_state_model(str(paths["state_model"]), device)
    masks = StatewiseMaskProvider.from_configs(C.read(paths["mask_descriptor"]), paths["boundary_mask_config"], device)
    _CONTEXT = (head, C.read(paths["OT_model"]), masks, model, config, device)


def _run_frame(task):
    frame, cached = task
    head, ot, masks, model, config, device = _CONTEXT
    path = C.ROOT / frame["image"]
    file_sha = C.sha(path)
    if frame["image_sha256"] and file_sha != frame["image_sha256"]:
        raise ValueError("Actual RGB file changed: " + frame["id"])
    if cached.get("image_sha256") and file_sha != cached["image_sha256"]:
        raise ValueError("Cached RGB binding changed")
    image = cv2.imread(str(path))
    if image is None or list(image.shape[:2]) != frame["raw_hw"]:
        raise ValueError("Actual RGB canvas mismatch")
    api = StatewiseJointV4(head, ot, masks, uncertainty_model=model,
        config={"solver": config, "prediction_source": "N3_SUBPIX", "record_id": frame["id"], "device": device})
    # Explicit full API; every frame receives a fresh current RGB forward.
    result = api.refine(image, frame["raw_points"], frame["K"], C.cuboid(*frame["initial_pose"]["cf_extents"]),
                        frame["initial_pose"], crop_transform=frame["crop_transform"])
    parity = observation_parity(result, cached)
    S_audit = S_contract_audit(result)
    if not parity["passed"] or not S_audit["passed"]:
        raise ValueError("Fresh API parity/S contract failed: " + frame["id"] + " " + json.dumps([parity, S_audit]))
    fit = deepcopy(result["calibrated_visible_measurements"])
    spec = _physics_spec(result, fit, config)
    key = _digest(spec)
    known = {key: deepcopy(result["joint_pose"])}
    controls, control_audits = {}, {}
    for arm in (ARM_NO_OT, ARM_NO_PRIOR):
        control, ckey, audit, solution = control_from_observations(result, fit, model, config, arm, known)
        controls[arm] = {"actual_pose": control["actual_pose"], "states": control["states"],
                         "accepted": control["accepted"], "solution": _compact_solution(solution),
                         "keypoint_readout": control["keypoint_readout"], "ambiguity_audit": control["ambiguity_audit"],
                         "observation_alias": "runtime.frozen_visible_measurements",
                         "calibrated_measurements_alias": "runtime.calibrated_visible_measurements"}
        known[ckey] = solution
        control_audits[arm] = audit
    return {k: deepcopy(frame[k]) for k in ("id", "source_record_id", "partition", "session", "cohort", "image", "K", "raw_hw", "crop_transform", "initial_pose")} | {
        "image_sha256": file_sha, "runtime": _compact_runtime(result), "controls": controls,
        "control_input_audit": control_audits, "fresh_cache_parity": parity, "S_contract_audit": S_audit,
        "all_OT_mathematical_input_sha256": key,
        "solver_calls_this_frame": 1 + sum(not a["cached_identical_input_result"] for a in control_audits.values()),
        "fresh_current_RGB_full_API": True, "annotation_targets_read": False}


def predict(output=OUTPUT, selection=SELECTION, workers=3, device="cuda"):
    output, selection = Path(output).resolve(), Path(selection).resolve()
    if output.exists():
        raise FileExistsError("Preserve complete/partial evaluations: " + str(output))
    if workers not in (1, 2, 3) or not str(device).startswith("cuda"):
        raise ValueError("One to three spawned workers with CUDA required")
    selected, model, config, selection_binding, selection_seal = _selection(selection)
    frames = _frames()
    if len(frames) != 98:
        raise ValueError("Exact95 primary+3 real crops required")
    cached, cache_bindings = _fresh_cache(frames)
    runtime_descriptor, model_paths, model_bindings = _runtime_models()
    fixed = C.read(FIX / "FIX_PROTOCOL_V1.json")
    protected = fixed["protected_user_sha256"]
    _verify_sources(protected)
    source = {**source_bindings(), **selected["source_sha256"]}
    _verify_sources(source)
    inputs = {str((COHORT / "PREDICTION_INPUTS.json").relative_to(C.ROOT)): C.sha(COHORT / "PREDICTION_INPUTS.json"),
              str((COHORT / "COHORT_PROTOCOL.json").relative_to(C.ROOT)): C.sha(COHORT / "COHORT_PROTOCOL.json"),
              str((FIX / "FIX_PROTOCOL_V1.json").relative_to(C.ROOT)): C.sha(FIX / "FIX_PROTOCOL_V1.json"),
              **crop_bindings(), **model_bindings}
    output.mkdir(parents=True)
    protocol = {"schema": "STATEWISE_FRESH_RGB_UNCERTAINTY_EVALUATION_V4_1",
        "ids": [f["id"] for f in frames], "primary_population": {"CAL": 46, "TEST": 49, "severe": 0},
        "actual_RGB_crop_count": 3, "crops_are_primary": False, "arms": list(ARMS),
        "all_frames_run_full_refine_API": True, "frozen_upstream": "N3_SUBPIX same native q/K/Q and registered geometry",
        "state_mask_BI_OT_models_unchanged": True, "new_RGB_current_mask_state_and_BI_forwards": True,
        "selected_config": selection_binding, "selection_seal": selection_seal,
        "calibration_model": selected["calibration_model"], "solver_configuration": config,
        "selected_prior_weight": config["prior_weight"], "prior_selected_on_CAL_only": True,
        "controls_reuse_same_immutable_BI_fit_covariance_nonS_scale_and_initial_geometry": True,
        "NO_OT_changes": "O/T to U plus no domains", "NO_PRIOR_changes": "prior_weight=0",
        "solver_cache_reuse": "identical complete mathematical-input SHA256 only; fullAPI ALL_OT always executed",
        "source_sha256": source, "input_and_model_sha256": inputs, "protected_user_sha256": protected,
        "fresh_observation_cache": cache_bindings, "parity_tolerances": PARITY,
        "S_image_queries_allowed": 0, "S_region_terms_allowed": 0, "annotation_targets_accessed": False,
        "TEST_status": "REUSED_EXPLORATORY_NOT_NEW_UNSEEN", "workers": workers, "device": device,
        "CPU_threads_per_worker": 1, "required_target": fixed["required_primary_target"],
        "journal_compaction": "drop duplicate solution alias/per-candidate repeated audits; retain all distinct samples and selected audits",
        "maximum_gzip_bytes": 100_000_000}
    C.write(output / "PROTOCOL.json", protocol)
    for relative, sha in source.items():
        destination = output / "code_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / relative, destination)
        if C.sha(destination) != sha:
            raise ValueError("Executed source snapshot mismatch")
    start, by_id = time.monotonic(), {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker,
                             initargs=(selected, model, config, device)) as pool:
        futures = {pool.submit(_run_frame, (frame, cached[frame["id"]])): frame["id"] for frame in frames}
        for future in as_completed(futures):
            row = future.result()
            if row["id"] != futures[future] or row["id"] in by_id:
                raise ValueError("Worker ID/native binding mismatch")
            by_id[row["id"]] = row
            C.write(output / "LIVE_PROGRESS.json", {"completed": len(by_id), "total": 98, "last_id": row["id"],
                "elapsed_seconds": time.monotonic() - start, "solver_calls": sum(r["solver_calls_this_frame"] for r in by_id.values())})
            print(f"{len(by_id)}/98 {row['id']} states={''.join(row['runtime']['states'])} accepted={row['runtime']['accepted']}", flush=True)
    ordered = [by_id[f["id"]] for f in frames]
    _verify_sources(source)
    _verify_sources(inputs)
    _verify_sources(protected)
    _selection(selection)
    _fresh_cache(frames)
    C.save_rows(output / "PREDICTIONS.jsonl.gz", ordered)
    if (output / "PREDICTIONS.jsonl.gz").stat().st_size >= protocol["maximum_gzip_bytes"]:
        raise ValueError("Dense prediction journal exceeds preregistered100MB cap; no seal issued")
    C.write(output / "CONTRACT_CHECKS.json", {"rows": 98, "fresh_cache_parity_pass": all(r["fresh_cache_parity"]["passed"] for r in ordered),
        "S_contract_checks_pass": all(r["S_contract_audit"]["passed"] for r in ordered),
        "protected6_pre_and_post_unchanged": True, "sources_and_inputs_pre_and_post_unchanged": True,
        "solver_calls": sum(r["solver_calls_this_frame"] for r in ordered),
        "all98_pose_fallbacks_preserved": True, "TEST_target_access_before_prediction_seal": False})
    C.write(output / "PREDICTION_SEAL.json", {"schema": "STATEWISE_FRESH_RGB_PREDICTION_SEAL_V4_1",
        "before_scoring": True, "predictions_sha256": C.sha(output / "PREDICTIONS.jsonl.gz"),
        "protocol_sha256": C.sha(output / "PROTOCOL.json"), "contract_checks_sha256": C.sha(output / "CONTRACT_CHECKS.json"),
        "source_sha256": source, "rows": 98, "primary_rows": 95, "crop_rows": 3, "severe_rows": 0,
        "all3arms_complete": True, "annotation_targets_accessed": False,
        "elapsed_seconds": time.monotonic() - start, "workers": workers})


def _verify_prediction_seal(output):
    seal = C.read(output / "PREDICTION_SEAL.json")
    protocol = C.read(output / "PROTOCOL.json")
    if not seal["before_scoring"] or seal["annotation_targets_accessed"] or seal["rows"] != 98 or not seal["all3arms_complete"]:
        raise ValueError("Incomplete/annotation-tainted prediction seal")
    for name, key in (("PREDICTIONS.jsonl.gz", "predictions_sha256"), ("PROTOCOL.json", "protocol_sha256"), ("CONTRACT_CHECKS.json", "contract_checks_sha256")):
        if C.sha(output / name) != seal[key]:
            raise ValueError("Prediction seal mismatch: " + name)
    _verify_sources(seal["source_sha256"])
    _verify_sources(protocol["input_and_model_sha256"])
    _verify_sources(protocol["protected_user_sha256"])
    for binding in (protocol["selected_config"], protocol["selection_seal"], protocol["calibration_model"],
                    *[b for b in protocol["fresh_observation_cache"].values() if isinstance(b, dict) and "path" in b]):
        _verify_binding(binding)
    for relative, digest in protocol["fresh_observation_cache"]["executed_snapshot_source_hashes"].items():
        if C.sha(FRESH / "code_snapshot" / relative) != digest:
            raise ValueError("Fresh cache snapshot changed after prediction")
    rows = list(C.rows(output / "PREDICTIONS.jsonl.gz"))
    if [r["id"] for r in rows] != protocol["ids"] or len(rows) != 98:
        raise ValueError("Complete ordered98 population mismatch")
    if not all(r["fresh_cache_parity"]["passed"] and r["S_contract_audit"]["passed"] for r in rows):
        raise ValueError("Prediction parity/S contract not passed")
    return rows, protocol, seal


def _old_comparator_poses(primary_rows):
    """Verify exact N3 inputs and raw sealed poses BEFORE label access."""
    directory = COHORT / "posefix_sub_comparison_v1"
    seal = C.read(directory / "PREDICTION_SEAL.json")
    path = directory / "PREDICTIONS.jsonl.gz"
    if not seal["before_scoring"] or C.sha(path) != seal.get("predictions_sha256", seal.get("sha256")):
        raise ValueError("Old comparator prediction seal mismatch")
    if seal.get("protocol_sha256") and C.sha(directory / "PROTOCOL.json") != seal["protocol_sha256"]:
        raise ValueError("Old comparator protocol mismatch")
    data = list(C.rows(path))
    if len(data) != seal["rows"]:
        raise ValueError("Old comparator sealed count mismatch")
    canonical = {r["id"]: r for r in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"]}
    baseline = {r["frame"]["id"]: r["frame"] for r in C.read(directory / "BASELINE_INPUTS.json")["frames"]}
    current = {r["id"]: r for r in primary_rows}
    if set(canonical) != set(current) or set(baseline) != set(current):
        raise ValueError("95 primary comparator ID mismatch")
    result = {"OLD_RGB_SINGLE_FAMILY": {}, "BOUNDARY_PROGRESSIVE": {}}
    methods = {"JOINT_SINGLE_FAMILY": "OLD_RGB_SINGLE_FAMILY", "FUSED_ACTIVE": "BOUNDARY_PROGRESSIVE"}
    initials = {}
    for r in data:
        method = r.get("method", r.get("arm"))
        if method == "INITIAL_N3_SUBPIX":
            initials[r["id"]] = r["actual_pose"]
        if method in methods:
            dst = result[methods[method]]
            if r["id"] in dst:
                raise ValueError("Duplicate comparator pose")
            dst[r["id"]] = r["actual_pose"]
    for arm in result:
        if set(result[arm]) != set(current):
            raise ValueError("Comparator must retain every95 pose including fallback")
    if set(initials) != set(current):
        raise ValueError("Comparator initial poses incomplete")
    for i, r in current.items():
        f, b = canonical[i], baseline[i]
        if not (r["image"] == f["image"] == b["image"] and r["K"] == f["K"] == b["K"] and r["raw_hw"] == f["raw_hw"] == b["raw_hw"]
                and r["initial_pose"] == f["initial_poses"]["N3_SUBPIX"] == initials[i]
                and _maxdiff(r["runtime"]["original_raw_points"], f["points"]["N3_SUBPIX"]) == 0
                and _maxdiff(b["points"]["N3_SUBPIX"], f["points"]["N3_SUBPIX"]) == 0):
            raise ValueError("Comparator RGB/K/native input mismatch: " + i)
        coarse = r["initial_pose"]
        Q0 = np.asarray(coarse["R_cf"]).T @ np.asarray(coarse["R_physical"])
        for arm in result:
            pose = result[arm][i]
            Q = np.asarray(pose["R_cf"]).T @ np.asarray(pose["R_physical"])
            if pose["cf_extents"] != coarse["cf_extents"] or _maxdiff(Q, Q0) > 1e-7:
                raise ValueError("Comparator native registration changed")
    v3dir = STUDY / "evaluation_v3"
    v3seal = C.read(v3dir / "PREDICTION_SEAL.json")
    if not v3seal["before_scoring"] or C.sha(v3dir / "PREDICTIONS.jsonl.gz") != v3seal["predictions_sha256"] or C.sha(v3dir / "PROTOCOL.json") != v3seal["protocol_sha256"]:
        raise ValueError("Frozen v3 comparator seal changed")
    _verify_sources(v3seal["source_sha256"])
    v3 = {r["id"]: r for r in C.rows(v3dir / "PREDICTIONS.jsonl.gz") if r["cohort"] == "PRIMARY_NONSEVERE"}
    if set(v3) != set(current):
        raise ValueError("Frozen v3 primary IDs mismatch")
    for i, r in v3.items():
        cur = current[i]
        if not (r["initial_pose"] == cur["initial_pose"] and r["K"] == cur["K"] and r["image"] == cur["image"]
                and r["runtime"]["original_raw_points"] == cur["runtime"]["original_raw_points"]):
            raise ValueError("Frozen v3 original input mismatch")
    result["FROZEN_STATEWISE_V3"] = {i: r["runtime"]["actual_pose"] for i, r in v3.items()}
    return result, {"predictions": _binding(path), "seal": _binding(directory / "PREDICTION_SEAL.json"),
                   "protocol": _binding(directory / "PROTOCOL.json"), "baseline_inputs": _binding(directory / "BASELINE_INPUTS.json"),
                   "frozen_v3_predictions": _binding(v3dir / "PREDICTIONS.jsonl.gz"), "frozen_v3_seal": _binding(v3dir / "PREDICTION_SEAL.json"),
                   "frozen_v3_protocol": _binding(v3dir / "PROTOCOL.json"),
                   "same95_RGB_K_native_q_initial_pose_and_Q": True}


def _summary(values):
    return {"frames": len(values), "metrics": {k: distribution([float(v[k]) for v in values]) for k in METRICS},
            "success_count": sum(bool(v["success_5cm_5deg"]) for v in values), "all_fallbacks_included": True}


def score(output=OUTPUT):
    output = Path(output).resolve()
    if (output / "POSTSEAL_SCORING.json").exists():
        raise FileExistsError("Preserve completed scoring")
    rows, protocol, seal = _verify_prediction_seal(output)
    primary = [r for r in rows if r["cohort"] == "PRIMARY_NONSEVERE"]
    old, comparator_binding = _old_comparator_poses(primary)
    # Annotation-coordinate access begins ONLY after all98/3arms seals/bindings.
    label_paths = {"CAL": COHORT / "OOF_pose_v1/ISOLATED_CAL_SCORING.json", "TEST": COHORT / "test_reused_v1/ISOLATED_TEST_SCORING.json"}
    labels = {p: C.read(path)["rows"] for p, path in label_paths.items()}
    for part, count in (("CAL", 46), ("TEST", 49)):
        if len(labels[part]) != count or set(labels[part]) != {r["id"] for r in primary if r["partition"] == part}:
            raise ValueError("Isolated physical reference/population mismatch")
    values, crops, confusion, compatible_count = [], [], defaultdict(Counter), Counter()
    comparator_scores = {name: {} for name in old}
    for row in rows:
        label = labels[row["partition"]][row["source_record_id"]]
        truth, metadata = label["target"]["truth"], label["metadata"]
        if truth["order"] != 2:
            raise ValueError("Fixed physical C2 scorer contract changed")
        poses = {"ORIGINAL_N3_PNP": row["initial_pose"], ARM_ALL: row["runtime"]["actual_pose"],
                 **{arm: row["controls"][arm]["actual_pose"] for arm in (ARM_NO_OT, ARM_NO_PRIOR)}}
        if row["cohort"] == "PRIMARY_NONSEVERE":
            poses.update({arm: data[row["id"]] for arm, data in old.items()})
        metrics = {arm: pose_metric(pose, truth) for arm, pose in poses.items()}
        compatible, gap = _model_compatible(row["initial_pose"], truth, metadata["native_to_reference_corner_ids"])
        if row["cohort"] == "PRIMARY_NONSEVERE":
            compatible_count[row["partition"]] += compatible
            for arm in old:
                comparator_scores[arm][row["id"]] = metrics[arm]
            for human, state in zip(metadata["human_states_native"], row["runtime"]["states"]):
                confusion[row["partition"]][(human, state)] += 1
        all_pose, no_ot = row["runtime"]["actual_pose"], row["controls"][ARM_NO_OT]["actual_pose"]
        pose_delta = {"translation_m": float(np.linalg.norm(np.asarray(all_pose["centroid"]) - no_ot["centroid"])),
                      "rotation_matrix_norm": float(np.linalg.norm(np.asarray(all_pose["R_cf"]) - no_ot["R_cf"]))}
        values.append({"id": row["id"], "partition": row["partition"], "session": row["session"], "cohort": row["cohort"],
            "metrics": metrics, "states": row["runtime"]["states"], "physical_native_compatible": compatible, "physical_native_gap_m": gap,
            "BI_count": len(row["runtime"]["frozen_visible_measurements"]), "OT_count": len(row["runtime"]["ot_domains"]["domains"]),
            "joint_accepted": row["runtime"]["accepted"], "material_ambiguity": row["runtime"]["ambiguity_audit"]["material_ambiguity"],
            "actual_pose_source": all_pose["source"], "actual_pose_fallback": all_pose["fallback"],
            "OT_changes_pose_vs_no_OT": pose_delta["translation_m"] > 1e-7 or pose_delta["rotation_matrix_norm"] > 1e-6,
            "pose_delta_all_minus_no_OT": pose_delta, "human_states_native": metadata["human_states_native"]})
        if row["crop_transform"]:
            mapping = np.asarray(metadata["native_to_reference_corner_ids"], int)
            gt = np.asarray(label["target"]["gt"], float)[:8][mapping]
            valid = np.asarray(label["target"]["valid"], bool)[:8][mapping]
            x0, y0, _, _ = row["crop_transform"]["crop_xyxy"]
            gt -= np.array([x0, y0])
            h, w = row["raw_hw"]
            actual_T = [i for i in range(8) if valid[i] and metadata["human_states_native"][i] == "DIRECT_VISIBLE"
                        and (gt[i, 0] < 0 or gt[i, 0] >= w or gt[i, 1] < 0 or gt[i, 1] >= h)]
            crops.append({"id": row["id"], "actual_transformed_visible_T_ids": actual_T,
                "predicted_T_ids": [i for i, s in enumerate(row["runtime"]["states"]) if s == "T"],
                "states": row["runtime"]["states"], "native_gt": gt.tolist(), "native_gt_valid": valid.tolist(),
                "joint_accepted": row["runtime"]["accepted"], "material_ambiguity": row["runtime"]["ambiguity_audit"]["material_ambiguity"],
                "conditional_T_is_not_physical_truncation_certificate": True, "source_visible_annotations_not_independent_pose_metrology": True})
    summary = {}
    for part in ("CAL", "TEST"):
        group = [v for v in values if v["partition"] == part and v["cohort"] == "PRIMARY_NONSEVERE"]
        summary[part] = {"frames": len(group), "pose": {arm: _summary([v["metrics"][arm] for v in group]) for arm in group[0]["metrics"]},
            "compatible_model_frames": compatible_count[part], "incompatible_model_frames": len(group) - compatible_count[part],
            "joint_accepted": sum(v["joint_accepted"] for v in group), "fallbacks": sum(v["actual_pose_fallback"] for v in group),
            "material_ambiguity": sum(v["material_ambiguity"] for v in group),
            "state_confusion": [{"actual": a, "predicted": p, "n": n} for (a, p), n in sorted(confusion[part].items())],
            "OT_information": {"BI_ge3": {"frames_with_OT": sum(v["BI_count"] >= 3 and v["OT_count"] > 0 for v in group),
                "pose_changed_vs_NO_OT": sum(v["BI_count"] >= 3 and v["OT_count"] > 0 and v["OT_changes_pose_vs_no_OT"] for v in group)},
                "BI2": {"frames_with_OT": sum(v["BI_count"] == 2 and v["OT_count"] > 0 for v in group),
                        "joint_accepted_family": sum(v["BI_count"] == 2 and v["OT_count"] > 0 and v["joint_accepted"] for v in group)}}}
    # Independently recompute comparator values, then compare to prior sealed JSON.
    benchmark_path = COMPARATORS / "BENCHMARK.json"
    benchmark = C.read(benchmark_path)
    for part, path in label_paths.items():
        if benchmark["label_bindings"][part]["sha256"] != C.sha(path):
            raise ValueError("Comparator/reference JSON bindings differ")
    old_parity = {}
    for arm, records in comparator_scores.items():
        reference = benchmark["per_frame_scores"]["CURRENT_JOINT_STATEWISE_V3" if arm == "FROZEN_STATEWISE_V3" else arm]
        maximum = max(abs(float(metric[k]) - float(reference[i]["metrics"][k])) for i, metric in records.items() for k in METRICS)
        if maximum > 1e-12:
            raise ValueError("Old comparator same-pose re-score disagrees: " + arm)
        old_parity[arm] = {"frames": 95, "max_metric_difference": maximum, "physical_same_pose_rescored": True}
    test = summary["TEST"]["pose"][ARM_ALL]
    target = protocol["required_target"]
    achieved = {"translation": test["metrics"]["translation_cm"]["mean"] <= target["TEST_translation_cm_mean_at_most"],
                "rotation": test["metrics"]["rotation_deg"]["mean"] <= target["TEST_rotation_deg_mean_at_most"],
                "success": test["success_count"] >= target["TEST_success_5cm_5deg_at_least"]}
    _verify_prediction_seal(output)
    for binding in comparator_binding.values():
        if isinstance(binding, dict) and "path" in binding:
            _verify_binding(binding)
    C.write(output / "POSTSEAL_SCORING.json", {"schema": "STATEWISE_V4_PHYSICAL_C2_POSTSEAL_SCORING_V1",
        "rows": values, "crops": crops, "summary": summary, "reference_symmetry_order": 2,
        "all95_primary_fallbacks_included": True, "primary_severe_rows": 0, "crops_excluded_from_primary": 3,
        "label_bindings": {part: _binding(path) for part, path in label_paths.items()},
        "comparator_bindings": comparator_binding, "comparator_existing_JSON_metric_parity": old_parity,
        "comparator_existing_benchmark": _binding(benchmark_path), "prediction_seal": _binding(output / "PREDICTION_SEAL.json"),
        "required_target": target, "target_checks": achieved, "target_achieved": all(achieved.values()),
        "all_three_predeclared_numeric_targets_met": all(achieved.values()),
        "statistical_or_general_superiority_proved": False,
        "independent_generalization_or_metrology_claim_allowed": False,
        "no_TEST_parameter_selection": True, "protected6_unchanged": True,
        "reference_limitation": "Historical annotation-derived physical/C2 reference; TEST reused exploratory, not independent physical metrology or unseen holdout."})
    print(json.dumps({"TEST": test, "target_checks": achieved, "target_achieved": all(achieved.values())}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("predict", "score"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase == "predict":
        predict(args.output, args.selection, args.workers, args.device)
    else:
        score(args.output)


if __name__ == "__main__":
    main()
