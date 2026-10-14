"""Fresh98 full-RGB evaluation of the independently CAL-selected prior mean.

The preserved v4 evaluator/solver/files are imported without modification.
Final and NO_OT execute the new bias API/solver on current RGB. NO_PRIOR is
physically independent of the prior mean and may reuse a sealed v4 solution
only after its exact full mathematical-input hash matches. No prediction seal
is issued until all98 frames and all3 arms are complete. Score is explicit.
"""
from __future__ import annotations

import os
for _setting in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_setting] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from .statewise_bias_api_v4_1 import StatewiseBiasV4_1
from . import statewise_uncertain_solver_v4_bias as B

OUTPUT = E.FIX / "evaluation_bias_v4_1"
SELECTION = E.FIX / "bias_selection_v4_1/SELECTED_CONFIG.json"
_BASELINE_CONFIG = None


def load_bias_selection(path=SELECTION):
    """Return selected, fullCAL model, full bias config, config/seal bindings.

    Seals and CAL-only metadata must pass before the selection/model is used.
    Hashing score/prediction artifacts does not open annotation coordinates.
    This guard is also suitable for the opt-in one-image runtime CLI.
    """
    path = Path(path).resolve()
    seal_path = path.parent / "SELECTION_SEAL.json"
    if not seal_path.is_file():
        raise FileNotFoundError("Bias CAL selection seal required before SELECTED_CONFIG")
    seal = C.read(seal_path)
    if not seal.get("before_new_TEST_score") or seal.get("TEST_files_opened") is not False or C.sha(path) != seal["selected_config_sha256"]:
        raise ValueError("Bias selection seal is not CAL-only/pre-TEST-score")
    selected = C.read(path)
    if selected.get("TEST_files_opened") is not False or not selected.get("selected_from_LOGO_CAL46_only"):
        raise ValueError("CAL-only LOGO bias selection required")
    if selected["source_sha256"] != seal["source_sha256"]:
        raise ValueError("Bias selected source/seal mismatch")
    E._verify_sources(selected["source_sha256"])
    model_path = E._verify_binding(selected["calibration_model"])
    if model_path.resolve() != (E.FIX / "calibration/final_adapter_CAL46/MODEL.json").resolve() or selected["calibration_model"] != seal["calibration_model"]:
        raise ValueError("Bias must use the sealed final fresh-CAL46 model")
    model = E.load_uncertainty_model(model_path)
    stage1_path = E._verify_binding(selected["stage1_selected_config"])
    stage1, stage1_model, base_config, _, _ = E._selection(stage1_path)
    if stage1["calibration_model"] != selected["calibration_model"]:
        raise ValueError("Bias and stage1 uncertainty models differ")
    config = B._configuration(selected["solver_configuration"])
    strength = float(selected["selected_bias_strength"])
    if strength not in (0., .5, 1.) or strength != float(seal["selected_bias_strength"]) or config["bias_strength"] != strength:
        raise ValueError("Bias strength differs from frozen CAL grid/choice")
    if bool(selected["bias_adopted"]) != (strength > 0) or selected["bias_adopted"] != seal["bias_adopted"]:
        raise ValueError("Bias adoption metadata disagrees")
    if config["prior_weight"] != stage1["selected_prior_weight"] or config["prior_weight"] != selected["selected_prior_weight"] or config["prior_weight"] != seal["selected_prior_weight"]:
        raise ValueError("Stage1 prior weight changed in bias selection")
    mean = model["prior_fit"]["median_bias6"]
    if not np.array_equal(config["prior_bias6"], mean):
        raise ValueError("Prior mean must equal the supplied CAL model median")
    for name in base_config:
        if E._digest(config[name]) != E._digest(base_config[name]):
            raise ValueError("Bias changed a frozen stage1 solver parameter: " + name)
    if C.sha(path.parent / "PROTOCOL.json") != selected["protocol_sha256"]:
        raise ValueError("Bias selection protocol changed")
    score_sha = C.sha(path.parent / "CAL_SELECTION_SCORE.json")
    if score_sha != selected["CAL_SELECTION_SCORE_sha256"] or score_sha != seal["CAL_score_sha256"]:
        raise ValueError("Bias CAL-only score changed")
    pred_seal_path = E._verify_binding(selected["CAL_prediction_seal"])
    if C.sha(pred_seal_path) != seal["CAL_prediction_seal_sha256"]:
        raise ValueError("Bias CAL prediction seal changed")
    pred_seal = C.read(pred_seal_path)
    if (not pred_seal["before_CAL_scorer_target_read"] or pred_seal["TEST_files_opened"] is not False
            or pred_seal["rows"] != 138 or pred_seal["protocol_sha256"] != selected["protocol_sha256"]
            or C.sha(path.parent / "CANDIDATE_PREDICTIONS.jsonl.gz") != pred_seal["predictions_sha256"]):
        raise ValueError("Bias138 prediction seal invalid")
    return selected, model, config, E._binding(path), E._binding(seal_path)


def source_bindings():
    values = E.source_bindings()
    for name in ("statewise_bias_api_v4_1.py", "statewise_uncertain_solver_v4_bias.py", "evaluation_bias_v4_1.py"):
        path = Path(__file__).parent / name
        values[str(path.relative_to(C.ROOT))] = C.sha(path)
    return values


def _init_worker(selected, model, config, base_config, device):
    global _BASELINE_CONFIG
    E._init_worker(selected, model, config, device)
    _BASELINE_CONFIG = deepcopy(base_config)


def _no_OT(result, fit, model, config):
    states = ["U" if s in ("O", "T") else s for s in result["states"]]
    spec = E._physics_spec(result, fit, config, states, [])
    key = E._digest(spec)
    original_key = E._digest(E._physics_spec(result, fit, config))
    if key == original_key:
        solution, reused = deepcopy(result["joint_pose"]), True
    else:
        solution = B.solve_uncertain_joint_ot_pnp(**spec)
        reused = False
    source = deepcopy(result)
    source["states"] = states
    source["ot_domains"]["domains"] = []
    source["ot_domains"]["domains_sha256"] = E._digest([])
    control = E.finish_observations(source, solution, fit, model)
    if control["frozen_visible_measurements"] != result["frozen_visible_measurements"] or control["calibrated_visible_measurements"] != fit:
        raise ValueError("Bias NO_OT changed image/effective B/I measurements")
    return {"actual_pose": control["actual_pose"], "states": states, "accepted": control["accepted"],
            "solution": E._compact_solution(solution), "keypoint_readout": control["keypoint_readout"],
            "ambiguity_audit": control["ambiguity_audit"], "observation_alias": "runtime.frozen_visible_measurements",
            "calibrated_measurements_alias": "runtime.calibrated_visible_measurements"}, {
            "mathematical_input_sha256": key, "cached_identical_input_result": reused,
            "changes": "O/T states to U and domains empty; selected prior weight/median bias unchanged",
            "same_BI_xy_observed_covariance_fit_covariance_reliability_and_scale": True,
            "same_initial_registered_geometry": True}


def _run_frame(task):
    frame, cached = task
    head, ot, masks, model, config, device = E._CONTEXT
    path = C.ROOT / frame["image"]
    sha = C.sha(path)
    if (frame["image_sha256"] and sha != frame["image_sha256"]) or (cached.get("image_sha256") and sha != cached["image_sha256"]):
        raise ValueError("Bias actual RGB file binding changed")
    image = cv2.imread(str(path))
    if image is None or list(image.shape[:2]) != frame["raw_hw"]:
        raise ValueError("Bias actual RGB canvas mismatch")
    api = StatewiseBiasV4_1(head, ot, masks, uncertainty_model=model, config={"solver": config,
        "prediction_source": "N3_SUBPIX", "record_id": frame["id"], "device": device})
    result = api.refine(image, frame["raw_points"], frame["K"], C.cuboid(*frame["initial_pose"]["cf_extents"]),
                        frame["initial_pose"], crop_transform=frame["crop_transform"])
    parity, S_audit = E.observation_parity(result, cached), E.S_contract_audit(result)
    if not parity["passed"] or not S_audit["passed"]:
        raise ValueError("Bias fresh98 observation/S contract failed: " + frame["id"])
    fit = deepcopy(result["calibrated_visible_measurements"])
    no_ot, audit = _no_OT(result, fit, model, config)
    # Exactly the original v4 NO_PRIOR mathematical input. The prior mean has
    # no physical effect at weight0; coarse initialization stays unchanged.
    base = deepcopy(_BASELINE_CONFIG)
    base["prior_weight"] = 0.
    no_prior_spec = E._physics_spec(result, fit, base)
    row = {k: deepcopy(frame[k]) for k in ("id", "source_record_id", "partition", "session", "cohort", "image", "K", "raw_hw", "crop_transform", "initial_pose")}
    row.update(image_sha256=sha, runtime=E._compact_runtime(result), controls={E.ARM_NO_OT: no_ot},
        control_input_audit={E.ARM_NO_OT: audit}, fresh_cache_parity=parity, S_contract_audit=S_audit,
        all_OT_mathematical_input_sha256=E._digest(E._physics_spec(result, fit, config)),
        expected_original_v4_no_prior_sha256=E._digest(no_prior_spec),
        solver_calls_this_frame=1 + int(not audit["cached_identical_input_result"]),
        fresh_current_RGB_full_API=True, annotation_targets_read=False,
        prior_mean_from_CAL_only=True, bias_strength=config["bias_strength"],
        NO_PRIOR_deferred_until_original_v4_prediction_seal=True)
    return row


def _hydrate_no_prior(rows, base_config, baseline=E.OUTPUT):
    baseline = Path(baseline).resolve()
    old, _, _ = E._verify_prediction_seal(baseline)
    old = {r["id"]: r for r in old}
    model = E.load_uncertainty_model(E.FIX / "calibration/final_adapter_CAL46/MODEL.json")
    binding = E._binding(baseline / "PREDICTION_SEAL.json")
    reused, fresh = 0, 0
    for row in rows:
        source = old[row["id"]]
        expected = row["expected_original_v4_no_prior_sha256"]
        prior_inactive = deepcopy(base_config)
        prior_inactive["prior_weight"] = 0.
        spec = E._physics_spec(row["runtime"], row["runtime"]["calibrated_visible_measurements"], prior_inactive)
        if expected != E._digest(spec):
            raise ValueError("Deferred NO_PRIOR mathematical input changed")
        exact = expected == source["control_input_audit"][E.ARM_NO_PRIOR]["mathematical_input_sha256"]
        if exact:
            control = deepcopy(source["controls"][E.ARM_NO_PRIOR])
            reused += 1
        else:
            # Tolerated observation parity is insufficient for cache reuse.
            # A different exact mathematical input receives a new pure solve.
            solution = E._solve_spec(spec)
            done = E.finish_observations(row["runtime"], solution, row["runtime"]["calibrated_visible_measurements"], model)
            control = {"actual_pose": done["actual_pose"], "states": done["states"], "accepted": done["accepted"],
                "solution": E._compact_solution(solution), "keypoint_readout": done["keypoint_readout"],
                "ambiguity_audit": done["ambiguity_audit"], "observation_alias": "runtime.frozen_visible_measurements",
                "calibrated_measurements_alias": "runtime.calibrated_visible_measurements"}
            row["solver_calls_this_frame"] += 1
            fresh += 1
        control["prior_mean_physically_inactive_at_weight0"] = True
        row["controls"][E.ARM_NO_PRIOR] = control
        row["control_input_audit"][E.ARM_NO_PRIOR] = {"mathematical_input_sha256": expected,
            "cached_identical_input_result": exact, "changes": "prior_weight=0; median mean term physically inactive",
            "same_BI_xy_observed_covariance_fit_covariance_reliability_and_scale": True,
            "same_initial_registered_geometry": True, "reuse_prediction_seal": binding if exact else None,
            "mean_free_v4_solver_is_same_NO_PRIOR_objective_guards_and_initializations": True}
        row["NO_PRIOR_deferred_until_original_v4_prediction_seal"] = False
    return rows, {"original_v4_prediction_seal": binding, "original_v4_predictions": E._binding(baseline / "PREDICTIONS.jsonl.gz"),
                  "exact_NO_PRIOR_cache_reuses": reused, "new_NO_PRIOR_solves_for_exact_input_mismatches": fresh,
                  "prior_mean_inactive_at_weight0": True, "all98_controls_complete_before_seal": True}


def predict(output=OUTPUT, selection=SELECTION, workers=3, device="cuda"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous bias evaluation: " + str(output))
    if workers not in (1, 2, 3) or not str(device).startswith("cuda"):
        raise ValueError("One to three spawn workers with CUDA required")
    selected, model, config, selected_binding, selection_seal = load_bias_selection(selection)
    if not selected["bias_adopted"]:
        raise ValueError("No CAL bias adoption; preserve original v4 evaluation instead")
    stage1, _, base_config, _, _ = E._selection(E._resolve(selected["stage1_selected_config"]))
    frames = E._frames()
    cached, cache_bindings = E._fresh_cache(frames)
    _, _, model_bindings = E._runtime_models()
    fixed = C.read(E.FIX / "FIX_PROTOCOL_V1.json")
    protected = fixed["protected_user_sha256"]
    E._verify_sources(protected)
    sources = {**source_bindings(), **selected["source_sha256"]}
    E._verify_sources(sources)
    inputs = {str((E.COHORT / name).relative_to(C.ROOT)): C.sha(E.COHORT / name) for name in ("PREDICTION_INPUTS.json", "COHORT_PROTOCOL.json")}
    inputs.update({str((E.FIX / name).relative_to(C.ROOT)): C.sha(E.FIX / name) for name in
                   ("FIX_PROTOCOL_V1.json", "CAL_MEDIAN_BIAS_AMENDMENT_V1.json")})
    inputs.update(E.crop_bindings())
    inputs.update(model_bindings)
    output.mkdir(parents=True)
    protocol = {"schema": "STATEWISE_FRESH_RGB_CAL_MEDIAN_BIAS_EVALUATION_V4_1",
        "ids": [f["id"] for f in frames], "primary_population": {"CAL": 46, "TEST": 49, "severe": 0},
        "actual_RGB_crop_count": 3, "crops_are_primary": False, "arms": list(E.ARMS),
        "arm_labels_use_shared_v4_scorer_version_is_defined_by_this_protocol": True,
        "all_frames_run_full_refine_API": True, "actual_API": "StatewiseBiasV4_1 inherited observe/refine",
        "selected_config": selected_binding, "selection_seal": selection_seal,
        "stage1_selected_config": selected["stage1_selected_config"], "calibration_model": selected["calibration_model"],
        "solver_configuration": config, "selected_prior_weight": config["prior_weight"],
        "selected_bias_strength": config["bias_strength"], "prior_bias6_from_fullCAL_model": config["prior_bias6"],
        "prior_and_bias_selected_on_CAL_only": True, "frozen_upstream": "N3_SUBPIX same native q/K/Q/model",
        "controls_reuse_same_immutable_BI_fit_covariance_nonS_scale_and_initial_geometry": True,
        "NO_OT_changes": "O/T to U and empty regions; selected median prior retained",
        "NO_PRIOR_changes": "weight0 disables mean and prior loss; same observations/OT/initialization",
        "NO_PRIOR_reuse": "deferred sealed-v4 exact full mathematical-input hash comparison; recompute mismatches",
        "source_sha256": sources, "input_and_model_sha256": inputs, "protected_user_sha256": protected,
        "fresh_observation_cache": cache_bindings, "parity_tolerances": E.PARITY,
        "S_image_queries_allowed": 0, "S_region_terms_allowed": 0, "annotation_targets_accessed": False,
        "TEST_status": "REUSED_EXPLORATORY_NOT_NEW_UNSEEN", "workers": workers, "device": device,
        "CPU_threads_per_worker": 1, "required_target": fixed["required_primary_target"], "maximum_gzip_bytes": 100_000_000}
    C.write(output / "PROTOCOL.json", protocol)
    for relative, sha in sources.items():
        destination = output / "code_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / relative, destination)
        if C.sha(destination) != sha:
            raise ValueError("Bias executed snapshot mismatch")
    started, by_id = time.monotonic(), {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker,
        initargs=(selected, model, config, base_config, device)) as pool:
        futures = {pool.submit(_run_frame, (f, cached[f["id"]])): f["id"] for f in frames}
        for future in as_completed(futures):
            row = future.result()
            if row["id"] != futures[future] or row["id"] in by_id:
                raise ValueError("Bias worker/native ID mismatch")
            by_id[row["id"]] = row
            C.write(output / "LIVE_PROGRESS.json", {"completed": len(by_id), "total": 98, "last_id": row["id"],
                "stage": "fresh_RGB_final_and_NO_OT", "elapsed_seconds": time.monotonic() - started})
            print(f"{len(by_id)}/98 {row['id']} bias={config['bias_strength']} accepted={row['runtime']['accepted']}", flush=True)
    ordered = [by_id[f["id"]] for f in frames]
    C.save_rows(output / "FRESH_TWO_ARM_PREDICTIONS.jsonl.gz", ordered)
    C.write(output / "FRESH_TWO_ARM_SEAL.json", {"complete_primary_and_NO_OT_frames": 98,
        "before_any_scorer_target_read": True, "NO_PRIOR_pending": True, "scoreable_prediction_seal": False,
        "predictions_sha256": C.sha(output / "FRESH_TWO_ARM_PREDICTIONS.jsonl.gz"), "protocol_sha256": C.sha(output / "PROTOCOL.json")})
    while not (E.OUTPUT / "PREDICTION_SEAL.json").is_file():
        E._verify_sources(sources)
        E._verify_sources(protected)
        C.write(output / "LIVE_PROGRESS.json", {"completed": 98, "total": 98,
            "stage": "await_original_v4_NO_PRIOR_prediction_seal", "elapsed_seconds": time.monotonic() - started,
            "annotations_read": False})
        time.sleep(15)
    ordered, reuse_audit = _hydrate_no_prior(ordered, base_config)
    E._verify_sources(sources)
    E._verify_sources(inputs)
    E._verify_sources(protected)
    load_bias_selection(selection)
    E._fresh_cache(frames)
    C.save_rows(output / "PREDICTIONS.jsonl.gz", ordered)
    if (output / "PREDICTIONS.jsonl.gz").stat().st_size >= protocol["maximum_gzip_bytes"]:
        raise ValueError("Bias dense journal exceeds100MB; no complete seal")
    C.write(output / "NO_PRIOR_REUSE_AUDIT.json", reuse_audit)
    C.write(output / "CONTRACT_CHECKS.json", {"rows": 98,
        "fresh_cache_parity_pass": all(r["fresh_cache_parity"]["passed"] for r in ordered),
        "S_contract_checks_pass": all(r["S_contract_audit"]["passed"] for r in ordered),
        "all98_all3arms_complete": True, "protected6_pre_and_post_unchanged": True,
        "sources_and_inputs_pre_and_post_unchanged": True, "solver_calls": sum(r["solver_calls_this_frame"] for r in ordered),
        "NO_PRIOR_exact_cache_reuses": reuse_audit["exact_NO_PRIOR_cache_reuses"],
        "all98_pose_fallbacks_preserved": True, "TEST_target_access_before_prediction_seal": False})
    C.write(output / "PREDICTION_SEAL.json", {"schema": "STATEWISE_FRESH_RGB_BIAS_PREDICTION_SEAL_V4_1",
        "before_scoring": True, "predictions_sha256": C.sha(output / "PREDICTIONS.jsonl.gz"),
        "protocol_sha256": C.sha(output / "PROTOCOL.json"), "contract_checks_sha256": C.sha(output / "CONTRACT_CHECKS.json"),
        "NO_PRIOR_reuse_audit_sha256": C.sha(output / "NO_PRIOR_REUSE_AUDIT.json"),
        "source_sha256": sources, "rows": 98, "primary_rows": 95, "crop_rows": 3, "severe_rows": 0,
        "all3arms_complete": True, "annotation_targets_accessed": False,
        "selected_bias_strength": config["bias_strength"], "elapsed_seconds": time.monotonic() - started})


def score(output=OUTPUT, selection=SELECTION):
    """Explicit post-seal scoring only; root decides when this phase runs."""
    output = Path(output).resolve()
    selected, _, config, _, _ = load_bias_selection(selection)
    rows, protocol, seal = E._verify_prediction_seal(output)
    if config != protocol["solver_configuration"]:
        raise ValueError("Bias prediction/selected configuration mismatch")
    if C.sha(output / "NO_PRIOR_REUSE_AUDIT.json") != seal["NO_PRIOR_reuse_audit_sha256"]:
        raise ValueError("Bias NO_PRIOR alias evidence changed")
    reuse = C.read(output / "NO_PRIOR_REUSE_AUDIT.json")
    E._verify_binding(reuse["original_v4_prediction_seal"])
    E._verify_binding(reuse["original_v4_predictions"])
    E._verify_prediction_seal(E.OUTPUT)
    E.score(output)
    C.write(output / "BIAS_POSTSEAL_BINDING.json", {"selected_bias_strength": config["bias_strength"],
        "selected_prior_weight": config["prior_weight"], "selected_config": E._binding(Path(selection)),
        "postseal_scoring": E._binding(output / "POSTSEAL_SCORING.json"),
        "shared_physical_C2_scorer": True, "TEST_parameters_selected_on_CAL_only": True})


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
        score(args.output, args.selection)


if __name__ == "__main__":
    main()
