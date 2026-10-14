"""CAL-only conditional LOGO prior selection on sealed fresh observations.

No RGB or TEST file is opened. Every candidate pose is sealed before the CAL
scorer opens targets. Upstream state/BI/OT predictors are frozen all-CAL fits;
the uncertainty model alone is fitted without each held session. This is not
a full-pipeline out-of-fold generalization estimate.
"""
from __future__ import annotations

import os
for _setting in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_setting] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits

from . import common as C
from . import statewise_uncertain_solver_v4 as S
from .statewise_uncertainty_calibration_v4 import apply_calibrated_measurements, solver_config_from_model
from .statewise_joint_api_v4 import finish_observations
from .scoring import pose_metric, distribution


STUDY = C.DOC / "nonsevere_v1/statewise_joint_ot_20261014_v2"
FIX = STUDY / "nonsevere_v4_fix"
OUTPUT = FIX / "cal_selection_v1"
WEIGHTS = [0., .25, 1., 4.]
METRICS = ("translation_cm", "rotation_deg", "ADDsym_cm")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    encoded = json.dumps(C.finite(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(C.finite(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def binding(path):
    path = Path(path)
    return {"path": str(path.relative_to(C.ROOT)), "sha256": C.sha(path), "bytes": path.stat().st_size}


def read_lines(path):
    with gzip.open(path, "rt") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_lines(path, rows):
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(C.finite(row), separators=(",", ":"), allow_nan=False) + "\n").encode())
        raw.flush()
        os.fsync(raw.fileno())


def append_checkpoint(path, row):
    # One gzip member per result is closed and fsynced before it is considered
    # durable. Concatenated members remain readable by Python's gzip reader.
    with Path(path).open("ab") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            zipped.write((json.dumps(C.finite(row), separators=(",", ":"), allow_nan=False) + "\n").encode())
        raw.flush()
        os.fsync(raw.fileno())


def load_inputs():
    observation_path = FIX / "fresh_observations_v2/CAL_OBSERVATIONS.jsonl.gz"
    observation_seal_path = observation_path.parent / "OBSERVATION_SEAL.json"
    model_dir = FIX / "calibration/final_adapter_CAL46"
    seal = C.read(observation_seal_path)
    assert C.sha(observation_path) == seal["CAL_observations_sha256"]
    calibration_seal = C.read(model_dir / "SEAL.json")
    assert all(C.sha(model_dir / name) == expected for name, expected in calibration_seal["files"].items())
    assert C.sha(Path(__file__).parent / "statewise_uncertainty_calibration_v4.py") == calibration_seal["source_sha256"]
    rows = read_lines(observation_path)
    assert len(rows) == 46 and len({r["id"] for r in rows}) == 46
    assert all(r["partition"] == "CAL" and r["cohort"] == "PRIMARY_NONSEVERE" for r in rows)
    report = C.read(model_dir / "CAL_LOGO_REPORT.json")
    full_model = C.read(model_dir / "MODEL.json")
    calibration_protocol = C.read(model_dir / "PROTOCOL.json")
    ids = {r["id"] for r in rows}
    assert set(full_model["fit_ids"]) == ids and not full_model["TEST_labels_read"]
    fold_by_id = {}
    for fold in report["LOGO_folds"]:
        train, held = set(fold["train_ids"]), set(fold["heldout_ids"])
        assert not train & held and train | held == ids
        assert set(fold["model"]["fit_ids"]) == train
        assert not fold["model"]["TEST_labels_read"]
        for frame_id in held:
            assert frame_id not in fold_by_id
            fold_by_id[frame_id] = fold
    assert set(fold_by_id) == ids
    assert all(fold_by_id[r["id"]]["heldout_session"] == r["session"] for r in rows)
    return rows, fold_by_id, full_model, calibration_protocol


def build_spec(row, model, weight, arm="LOGO_CANDIDATE"):
    observation = deepcopy(row["runtime"])
    original = deepcopy(observation["frozen_visible_measurements"])
    original_domains = deepcopy(observation["ot_domains"]["domains"])
    scale = observation["ot_domains"]["nonself_prediction_scale_px"]
    fit = apply_calibrated_measurements(original, model, states=observation["states"], nonself_scale_px=scale)
    removed = []
    if arm == "SELECTED_V4_NO_OT":
        removed = [i for i, state in enumerate(observation["states"]) if state in ("O", "T")]
        observation["states"] = ["U" if state in ("O", "T") else state for state in observation["states"]]
        observation["control_original_ot_domains"] = deepcopy(observation["ot_domains"])
        observation["ot_domains"] = {"domains": [], "nonself_prediction_scale_px": scale,
                                      "control": "ALL_OT_TO_U_ONLY", "removed_native_ids": removed}
    elif arm == "SELECTED_V4_NO_PRIOR":
        weight = 0.
    elif arm != "LOGO_CANDIDATE":
        raise ValueError("Unknown selection arm")
    config = {**deepcopy(S._DEFAULTS), **solver_config_from_model(model, prior_weight=weight)}
    # Explicitly retain every frozen finite initialization/union limit.
    assert config["max_p3p_triples"] == 56 and config["global_rotation_starts"] == 24
    assert config["maximum_union_branches"] == 256 and config["max_joint_iterations"] == 250
    geometry = observation["initial_geometry"]
    spec = {"xyz": deepcopy(geometry["native_xyz"]), "K": deepcopy(geometry["K"]),
            "states": deepcopy(observation["states"]), "visible": deepcopy(fit),
            "domains": deepcopy(observation["ot_domains"]["domains"]),
            "image_hw": deepcopy(geometry["image_hw"]),
            "coarse_pose": {"R": deepcopy(geometry["R_cf"]), "t": deepcopy(geometry["centroid"]),
                            "physical_registration_Q": deepcopy(geometry["physical_registration_Q"])},
            "config": config}
    assert original == observation["frozen_visible_measurements"]
    for a, b in zip(original, fit):
        assert a["xy"] == b["xy"] and a["covariance_px2"] == b["covariance_px2"]
    if arm != "SELECTED_V4_NO_OT":
        assert spec["domains"] == original_domains
    return {"id": row["id"], "session": row["session"], "partition": "CAL", "arm": arm,
            "prior_weight": float(weight), "observation": observation, "model": deepcopy(model),
            "solver_input": spec, "solver_input_sha256": physics_key(spec),
            "model_sha256": digest(model), "removed_OT_ids": removed,
            "original_BI_sha256": digest(original), "calibrated_BI_sha256": digest(fit),
            "original_OT_sha256": digest(original_domains), "nonself_prediction_scale_px": scale,
            "source_row_sha256": digest(row)}


def physics_key(spec):
    visible_keys = ("native_corner", "xy", "covariance_px2", "protection_px",
                    "fit_covariance_px2", "fit_reliability_weight")
    domain_keys = ("native_corner", "native_id", "state", "shape", "center_xy", "ellipse_center",
                   "half_axes_px", "half_axes", "angle_deg", "angle", "bounds_xyxy", "box_xyxy",
                   "halfplanes", "normal", "offset", "outside_canvas_side", "outside_margin_px", "canvas_hw")
    physics = deepcopy(spec)
    physics["visible"] = [{k: r[k] for k in visible_keys if k in r} for r in spec["visible"]]
    physics["domains"] = [{k: r[k] for k in domain_keys if k in r} for r in spec["domains"]]
    return digest(physics)


def fixture_checks(rows, fold_by_id):
    row = rows[0]
    fold = fold_by_id[row["id"]]
    current = build_spec(row, fold["model"], 1.)
    no_ot = build_spec(row, fold["model"], 1., "SELECTED_V4_NO_OT")
    no_prior = build_spec(row, fold["model"], 1., "SELECTED_V4_NO_PRIOR")
    assert current["solver_input"]["visible"] == no_ot["solver_input"]["visible"] == no_prior["solver_input"]["visible"]
    assert current["solver_input"]["coarse_pose"] == no_ot["solver_input"]["coarse_pose"] == no_prior["solver_input"]["coarse_pose"]
    assert current["solver_input"]["domains"] == no_prior["solver_input"]["domains"]
    assert no_ot["solver_input"]["domains"] == []
    assert current["solver_input"]["config"]["prior_weight"] == no_ot["solver_input"]["config"]["prior_weight"] == 1.
    assert no_prior["solver_input"]["config"]["prior_weight"] == 0.
    assert all(a == b if a not in ("O", "T") else b == "U" for a, b in zip(current["solver_input"]["states"], no_ot["solver_input"]["states"]))
    assert row["id"] not in fold["model"]["fit_ids"]
    # Repeat on the sparse natural O case to avoid a vacuous no-domain proof.
    domain_row = next(r for r in rows if r["runtime"]["ot_domains"]["domains"])
    c = build_spec(domain_row, fold_by_id[domain_row["id"]]["model"], .25)
    d = build_spec(domain_row, fold_by_id[domain_row["id"]]["model"], .25, "SELECTED_V4_NO_OT")
    assert c["solver_input"]["domains"] and not d["solver_input"]["domains"]
    assert c["solver_input"]["visible"] == d["solver_input"]["visible"]
    assert c["nonself_prediction_scale_px"] == d["nonself_prediction_scale_px"]
    return {"all_pass": True, "fixtures": 4, "LOGO_train_heldout_disjoint": True,
            "NO_OT_changes_only_OT_states_and_domains": True, "NO_PRIOR_changes_only_weight": True,
            "observed_BI_xy_covariance_unchanged": True, "nonself_scale_held_fixed": True,
            "TEST_files_opened": False, "RGB_reads": 0, "pose_solver_calls": 0}


def verify_bindings(protocol):
    for path, expected in protocol["source_sha256"].items():
        assert C.sha(C.ROOT / path) == expected, "Source changed: " + path
    for record in protocol["input_bindings"].values():
        assert C.sha(C.ROOT / record["path"]) == record["sha256"], record["path"]
    for record in protocol["preserved_model_bindings"]:
        assert C.sha(C.ROOT / record["path"]) == record["sha256"], record["path"]
    for path, expected in protocol["protected_user_sha256"].items():
        assert C.sha(C.ROOT / path) == expected, path


def prepare(output=OUTPUT):
    output = Path(output)
    if output.resolve() != OUTPUT.resolve():
        raise ValueError("Only task-owned cal_selection_v1 output is allowed")
    rows, folds, full, cal_protocol = load_inputs()
    plan = C.read(FIX / "FIX_PROTOCOL_V1.json")
    assert plan["prior_weight_candidates"] == WEIGHTS
    fixtures = fixture_checks(rows, folds)
    model_dir = FIX / "calibration/final_adapter_CAL46"
    inputs = {"fix_plan": FIX / "FIX_PROTOCOL_V1.json", "CAL_observations": FIX / "fresh_observations_v2/CAL_OBSERVATIONS.jsonl.gz",
              "observation_seal": FIX / "fresh_observations_v2/OBSERVATION_SEAL.json",
              "uncertainty_MODEL": model_dir / "MODEL.json", "CAL_LOGO_REPORT": model_dir / "CAL_LOGO_REPORT.json",
              "calibration_protocol": model_dir / "PROTOCOL.json", "calibration_seal": model_dir / "SEAL.json",
              "API_metadata_amendment": FIX / "API_SOLVE_METADATA_AMENDMENT_V4_1.json",
              "recipe": FIX / "calibration/RECIPE_V1.json"}
    source_paths = set(C.read(STUDY / "evaluation_v3/PREDICTION_SEAL.json")["source_sha256"])
    source_paths.update(C.read(FIX / "fresh_observations_v2/OBSERVATION_SEAL.json")["source_sha256"])
    source_paths.update(str(Path(__file__).parent.joinpath(name).relative_to(C.ROOT)) for name in (
        "statewise_uncertain_solver_v4.py", "statewise_uncertainty_calibration_v4.py", "statewise_joint_api_v4.py",
        "statewise_selection_v4.py", "test_statewise_uncertain_solver_v4.py"))
    frozen19 = C.read(STUDY / "evaluation_v3/PREDICTION_SEAL.json")["source_sha256"]
    assert all(C.sha(C.ROOT / path) == expected for path, expected in frozen19.items())
    original_model_checks = C.read(STUDY / "diagnosis_compare_v1/counterfactual/MODEL_PRESERVATION_CHECK.json")["checks"]
    # Expected target binding is inherited from CAL calibration metadata. No
    # target bytes are opened by this prediction preparation phase.
    target_path = Path(cal_protocol["input_binding"]["CAL_annotations"])
    assert target_path.name == "ISOLATED_CAL_SCORING.json"
    protocol = {"schema": "CAL_LOGO_PRIOR_SELECTION_V4", "created_UTC": now(),
                "prior_weight_candidates": WEIGHTS, "CAL_ids": [r["id"] for r in rows],
                "sessions": dict(Counter(r["session"] for r in rows)), "candidate_rows": 184,
                "selection_order": ["maximize_all46_success", "minimize_mean_translation_cm", "minimize_mean_rotation_deg", "fixed_grid_order"],
                "conditional_subsystem_LOGO_not_full_pipeline_OOF": True,
                "upstream_models_are_frozen_all_CAL_fits": True, "TEST_files_opened": False, "RGB_reads": 0,
                "prediction_seal_required_before_CAL_scorer_target_read": True,
                "source_sha256": {p: C.sha(C.ROOT / p) for p in sorted(source_paths)},
                "input_bindings": {k: binding(p) for k, p in inputs.items()},
                "preserved_model_bindings": original_model_checks,
                "protected_user_sha256": plan["protected_user_sha256"],
                "CAL_target": {"path": str(target_path.relative_to(C.ROOT)),
                               "sha256": cal_protocol["input_binding"]["CAL_annotations_sha256"],
                               "binding_inherited_without_target_read": True},
                "solver_limits": {k: S._DEFAULTS[k] for k in ("max_p3p_triples", "global_rotation_starts", "max_joint_iterations", "maximum_union_branches")},
                "workers": 3, "threads_per_worker": 1, "native_GT_incompatible_cases_kept_in_scoring": True,
                "post_choice_controls": ["SELECTED_V4_NO_OT", "SELECTED_V4_NO_PRIOR"],
                "controls_use_same_held_session_model_and_fit_covariance": True,
                "final_production_model_uses_full_CAL_fit": True}
    verify_bindings(protocol)
    tasks = [build_spec(row, folds[row["id"]]["model"], weight) for weight in WEIGHTS for row in rows]
    task_inputs = [{k: t[k] for k in ("id", "session", "arm", "prior_weight", "solver_input", "solver_input_sha256", "model_sha256",
                                     "source_row_sha256", "original_BI_sha256", "calibrated_BI_sha256", "original_OT_sha256", "nonself_prediction_scale_px")} for t in tasks]
    if (output / "PROTOCOL.json").exists():
        old = C.read(output / "PROTOCOL.json")
        comparable = lambda p: {k: v for k, v in p.items() if k != "created_UTC"}
        assert comparable(old) == comparable(protocol), "Resume protocol mismatch"
        verify_bindings(old)
    else:
        output.mkdir(parents=True, exist_ok=True)
        write(output / "PROTOCOL.json", protocol)
        write(output / "CONTRACT_FIXTURES.json", fixtures)
        for path, expected in protocol["source_sha256"].items():
            destination = output / "code_snapshot" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(C.ROOT / path, destination)
            assert C.sha(destination) == expected
        write_lines(output / "CANDIDATE_INPUTS.jsonl.gz", task_inputs)
        write(output / "INPUT_SEAL.json", {"before_pose_fitting": True, "CAL_rows": 46, "candidate_rows": 184,
              "candidate_inputs_sha256": C.sha(output / "CANDIDATE_INPUTS.jsonl.gz"),
              "protocol_sha256": C.sha(output / "PROTOCOL.json"), "unique_solver_inputs": len({t["solver_input_sha256"] for t in tasks}),
              "CAL_ids_sha256": digest(protocol["CAL_ids"]), "TEST_files_opened": False, "scorer_target_reads": 0})
    return tasks


def solve_task(task):
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        solution = S.solve_uncertain_joint_ot_pnp(**task["solver_input"])
    solution["selection_prediction_context"] = {"CAL_heldout_session_model": True, "runtime_target_reads": 0, "TEST_reads": 0}
    return {"solver_input_sha256": task["solver_input_sha256"], "solution": solution,
            "solve_seconds": time.perf_counter() - started}


def render(task, solved):
    runtime = finish_observations(task["observation"], deepcopy(solved["solution"]),
                                   task["solver_input"]["visible"], task["model"])
    assert runtime["frozen_visible_measurements"] == task["observation"]["frozen_visible_measurements"]
    for m in runtime["frozen_visible_measurements"]:
        assert runtime["keypoint_readout"]["reported_points"][m["native_corner"]] == m["xy"]
    return {"id": task["id"], "partition": "CAL", "session": task["session"], "arm": task["arm"],
            "prior_weight": task["prior_weight"], "solver_input_sha256": task["solver_input_sha256"],
            "uncertainty_fold_model_sha256": task["model_sha256"], "runtime": runtime,
            "actual_pose": runtime["actual_pose"], "solve_seconds": solved["solve_seconds"],
            "original_BI_sha256": task["original_BI_sha256"], "calibrated_BI_sha256": task["calibrated_BI_sha256"],
            "original_OT_sha256": task["original_OT_sha256"], "removed_OT_ids": task["removed_OT_ids"],
            "nonself_prediction_scale_px": task["nonself_prediction_scale_px"],
            "no_RGB_forward": True, "no_TEST_reads": True, "prediction_contains_no_score": True}


def run_tasks(tasks, output, filename, sealname):
    output = Path(output)
    protocol = C.read(output / "PROTOCOL.json")
    verify_bindings(protocol)
    if (output / sealname).exists():
        seal = C.read(output / sealname)
        assert C.sha(output / filename) == seal["predictions_sha256"]
        return read_lines(output / filename)
    checkpoint = output / "SOLVER_CHECKPOINTS.jsonl.gz"
    cache = {r["solver_input_sha256"]: r for r in read_lines(checkpoint)} if checkpoint.exists() else {}
    unique = {t["solver_input_sha256"]: t for t in tasks}
    pending = {key: task for key, task in unique.items() if key not in cache}
    started = time.perf_counter()
    write(output / "LIVE_PROGRESS.json", {"stage": filename, "declared_rows": len(tasks), "unique_inputs": len(unique),
                                          "cached_inputs": len(unique) - len(pending), "pending_inputs": len(pending), "completed_new": 0})
    print(json.dumps({"stage": filename, "rows": len(tasks), "unique": len(unique), "cached": len(unique) - len(pending), "new": len(pending)}), flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as executor:
            futures = {executor.submit(solve_task, task): key for key, task in pending.items()}
            completed = 0
            for future in as_completed(futures):
                key = futures[future]
                solved = future.result()
                assert solved["solver_input_sha256"] == key
                append_checkpoint(checkpoint, solved)
                cache[key] = solved
                completed += 1
                progress = {"stage": filename, "declared_rows": len(tasks), "unique_inputs": len(unique), "cached_inputs": len(unique) - len(pending),
                            "new_inputs": len(pending), "completed_new": completed, "elapsed_seconds": time.perf_counter() - started,
                            "last_solve_seconds": solved["solve_seconds"], "last_id": pending[key]["id"], "last_weight": pending[key]["prior_weight"]}
                write(output / "LIVE_PROGRESS.json", progress)
                if completed % 5 == 0 or completed == len(pending):
                    print(json.dumps(progress), flush=True)
    predictions = [render(t, cache[t["solver_input_sha256"]]) for t in tasks]
    write_lines(output / filename, predictions)
    verify_bindings(protocol)
    write(output / sealname, {"schema": "CAL_SELECTION_PREDICTION_SEAL_V4", "created_UTC": now(),
          "before_CAL_scorer_target_read": filename == "CANDIDATE_PREDICTIONS.jsonl.gz", "CAL_only": True,
          "rows": len(predictions), "predictions_sha256": C.sha(output / filename), "protocol_sha256": C.sha(output / "PROTOCOL.json"),
          "source_sha256": protocol["source_sha256"], "input_bindings_unchanged": True,
          "unique_solver_inputs": len(unique), "cached_inputs_reused": len(unique) - len(pending),
          "elapsed_seconds": time.perf_counter() - started, "TEST_files_opened": False})
    return predictions


def read_CAL_targets_after_seal(output, sealname):
    output = Path(output)
    seal = C.read(output / sealname)
    protocol = C.read(output / "PROTOCOL.json")
    verify_bindings(protocol)
    assert seal["protocol_sha256"] == C.sha(output / "PROTOCOL.json")
    target = protocol["CAL_target"]
    assert C.sha(C.ROOT / target["path"]) == target["sha256"]
    labels = C.read(C.ROOT / target["path"])["rows"]
    assert set(protocol["CAL_ids"]) <= set(labels)
    return labels


def summarize(rows):
    return {"frames": len(rows), "success_count": sum(bool(r["metrics"]["success_5cm_5deg"]) for r in rows),
            "metrics": {k: distribution([r["metrics"][k] for r in rows]) for k in METRICS},
            "joint_accepted": sum(r["runtime"]["accepted"] for r in rows),
            "BI2_pose_family": sum((r["runtime"]["joint_pose"].get("ambiguity") or {}).get("pose_family_from_two_BI", False) for r in rows),
            "fallback_count": sum(not r["runtime"]["accepted"] for r in rows)}


def score_select(output=OUTPUT):
    output = Path(output)
    seal = C.read(output / "CANDIDATE_PREDICTION_SEAL.json")
    assert seal["before_CAL_scorer_target_read"]
    assert C.sha(output / "CANDIDATE_PREDICTIONS.jsonl.gz") == seal["predictions_sha256"]
    labels = read_CAL_targets_after_seal(output, "CANDIDATE_PREDICTION_SEAL.json")
    rows = read_lines(output / "CANDIDATE_PREDICTIONS.jsonl.gz")
    assert len(rows) == 184
    groups, sessions = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for row in rows:
        row["metrics"] = pose_metric(row["actual_pose"], labels[row["id"]]["target"]["truth"])
        groups[row["prior_weight"]].append(row)
        sessions[row["prior_weight"]][row["session"]].append(row)
    summary = {str(weight): summarize(groups[weight]) for weight in WEIGHTS}
    assert all(v["frames"] == 46 for v in summary.values())
    rank_key = lambda weight: (-summary[str(weight)]["success_count"],
                               summary[str(weight)]["metrics"]["translation_cm"]["mean"],
                               summary[str(weight)]["metrics"]["rotation_deg"]["mean"], WEIGHTS.index(weight))
    selected = min(WEIGHTS, key=rank_key)
    score = {"schema": "CAL_CONDITIONAL_LOGO_PRIOR_SELECTION_SCORE_V4", "created_UTC": now(),
             "prediction_seal_sha256": C.sha(output / "CANDIDATE_PREDICTION_SEAL.json"), "target_reads_after_prediction_seal": True,
             "summary": summary, "sessions": {str(w): {s: summarize(r) for s, r in sessions[w].items()} for w in WEIGHTS},
             "selected_prior_weight": selected, "selection_key": list(rank_key(selected)), "TEST_files_opened": False,
             "all46_scored_without_native_compatibility_filter": True, "conditional_subsystem_LOGO_not_full_pipeline_OOF": True}
    write_lines(output / "CANDIDATE_SCORED_ROWS.jsonl.gz", rows)
    write(output / "CAL_SELECTION_SCORE.json", score)
    _, _, full, _ = load_inputs()
    config = {**deepcopy(S._DEFAULTS), **solver_config_from_model(full, selected)}
    selected_record = {"schema": "FROZEN_CAL_LOGO_SELECTED_UNCERTAIN_V4_CONFIG", "created_UTC": now(),
                       "selected_prior_weight": selected, "prior_weight": selected, "solver_configuration": config,
                       "calibration_model": binding(FIX / "calibration/final_adapter_CAL46/MODEL.json"),
                       "selection_rule": ["maximize_all46_success", "minimize_mean_translation_cm", "minimize_mean_rotation_deg", "fixed_grid_order"],
                       "selected_from_LOGO_CAL46_only": True, "TEST_files_opened": False,
                       "conditional_subsystem_LOGO_not_full_pipeline_OOF": True,
                       "CAL_SELECTION_SCORE_sha256": C.sha(output / "CAL_SELECTION_SCORE.json"),
                       "candidate_prediction_seal_sha256": C.sha(output / "CANDIDATE_PREDICTION_SEAL.json"),
                       "CAL_prediction_seal": binding(output / "CANDIDATE_PREDICTION_SEAL.json"),
                       "source_sha256": C.read(output / "PROTOCOL.json")["source_sha256"],
                       "input_bindings": C.read(output / "PROTOCOL.json")["input_bindings"],
                       "protocol_sha256": C.sha(output / "PROTOCOL.json"), "fixed_grid_tie_order": WEIGHTS}
    if (output / "SELECTED_CONFIG.json").exists():
        old = C.read(output / "SELECTED_CONFIG.json")
        assert old["prior_weight"] == selected and old["solver_configuration"] == config
    else:
        write(output / "SELECTED_CONFIG.json", selected_record)
    write(output / "SELECTION_SEAL.json", {"schema": "CAL_ONLY_SELECTION_SEAL_V4", "created_UTC": now(),
          "selected_config_sha256": C.sha(output / "SELECTED_CONFIG.json"), "selected_prior_weight": selected,
          "CAL_prediction_seal_sha256": C.sha(output / "CANDIDATE_PREDICTION_SEAL.json"),
          "CAL_score_sha256": C.sha(output / "CAL_SELECTION_SCORE.json"),
          "calibration_model": selected_record["calibration_model"], "source_sha256": selected_record["source_sha256"],
          "TEST_files_opened": False, "before_any_TEST_prediction_or_score": True})
    print(json.dumps({"selected_prior_weight": selected, "summary": summary}), flush=True)
    return selected


def controls(output=OUTPUT):
    output = Path(output)
    choice = C.read(output / "SELECTED_CONFIG.json")
    rows, folds, _, _ = load_inputs()
    tasks = [build_spec(row, folds[row["id"]]["model"], choice["prior_weight"], arm)
             for arm in ("SELECTED_V4_NO_OT", "SELECTED_V4_NO_PRIOR") for row in rows]
    write_lines(output / "CONTROL_INPUTS.jsonl.gz", [{k: t[k] for k in ("id", "arm", "solver_input", "solver_input_sha256", "model_sha256", "removed_OT_ids",
                                                                           "original_BI_sha256", "calibrated_BI_sha256", "nonself_prediction_scale_px")} for t in tasks])
    write(output / "CONTROL_INPUT_SEAL.json", {"after_CAL_choice": True, "selected_config_sha256": C.sha(output / "SELECTED_CONFIG.json"),
          "control_inputs_sha256": C.sha(output / "CONTROL_INPUTS.jsonl.gz"), "rows": 92,
          "same_LOGO_fit_covariances_as_selected_candidates": True, "TEST_files_opened": False})
    predictions = run_tasks(tasks, output, "CONTROL_PREDICTIONS.jsonl.gz", "CONTROL_PREDICTION_SEAL.json")
    labels = read_CAL_targets_after_seal(output, "CONTROL_PREDICTION_SEAL.json")
    groups = defaultdict(list)
    for row in predictions:
        row["metrics"] = pose_metric(row["actual_pose"], labels[row["id"]]["target"]["truth"])
        groups[row["arm"]].append(row)
    write_lines(output / "CONTROL_SCORED_ROWS.jsonl.gz", predictions)
    write(output / "CAL_CONTROL_SCORE.json", {"schema": "POST_CHOICE_CAL_LOGO_CONTROLS_V4", "after_choice": True,
          "summary": {arm: summarize(group) for arm, group in groups.items()}, "TEST_files_opened": False,
          "BI_ge3_and_BI2_admission_cases_reported_separately": True,
          "by_BI_count": {arm: {label: summarize([r for r in group if (len(r["runtime"]["frozen_visible_measurements"]) >= 3) == flag])
                                for label, flag in (("BI_ge3", True), ("BI_lt3", False))} for arm, group in groups.items()}})
    protocol = C.read(output / "PROTOCOL.json")
    verify_bindings(protocol)
    write(output / "COMPLETION_RECEIPT.json", {"created_UTC": now(), "CAL_candidate_rows": 184, "post_choice_control_rows": 92,
          "selected_prior_weight": choice["prior_weight"], "all_bindings_match": True, "TEST_files_opened": False, "RGB_reads": 0,
          "artifacts": {name: binding(output / name) for name in ("PROTOCOL.json", "INPUT_SEAL.json", "CANDIDATE_PREDICTIONS.jsonl.gz",
              "CANDIDATE_PREDICTION_SEAL.json", "CAL_SELECTION_SCORE.json", "SELECTED_CONFIG.json", "SELECTION_SEAL.json", "CONTROL_INPUT_SEAL.json",
              "CONTROL_PREDICTIONS.jsonl.gz", "CONTROL_PREDICTION_SEAL.json", "CAL_CONTROL_SCORE.json")}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("check", "prepare", "predict", "score", "controls", "all"))
    args = parser.parse_args()
    if args.stage == "check":
        rows, folds, _, _ = load_inputs()
        print(json.dumps(fixture_checks(rows, folds)), flush=True)
        return
    if args.stage in ("prepare", "predict", "all"):
        tasks = prepare()
        if args.stage != "prepare":
            run_tasks(tasks, OUTPUT, "CANDIDATE_PREDICTIONS.jsonl.gz", "CANDIDATE_PREDICTION_SEAL.json")
    if args.stage in ("score", "all"):
        score_select()
    if args.stage in ("controls", "all"):
        controls()


if __name__ == "__main__":
    main()
