"""CAL-only second-stage median-prior-bias selection; preserved v4 is read-only.

Lambda stays at its sealed CAL choice of 1. Bias strengths 0/.5/1 use the same
held-session models and all other numerical inputs. Strength zero reuses the
exact sealed v4 poses after mathematical input equality checks. No TEST/RGB
file is opened. CAL targets are opened only after 138 predictions are sealed.
"""
from __future__ import annotations

import os
for _setting in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_setting] = "1"

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

from . import common as C
from . import joint_ot_pnp as P
from . import statewise_uncertain_solver_v4 as V4
from . import statewise_uncertain_solver_v4_bias as B
from . import statewise_selection_v4 as D
from .statewise_joint_api_v4 import finish_observations
from .scoring import pose_metric

OUTPUT = D.FIX / "bias_selection_v4_1"
GRID = [0., .5, 1.]


def tests():
    from .test_statewise_uncertain_solver_v4 import fixture
    bias = np.array([.012, -.025, .003, .0015, -.0006, .014])
    for phi in ([1e-9, -2e-9, 3e-9], [.2, -.7, .4], [3.13, .01, -.02]):
        p = np.r_[phi, [.12, -.08, 2.7]]
        center = np.array([.03, -.05, .04])
        R0 = Rotation.from_rotvec([-.05, .08, -.02]).as_matrix()
        c0 = np.array([.11, -.07, 2.3])
        W = np.diag([4., 5., 3., 20., 25., 8.])
        original, old_jacobian = V4._prior_residual_and_jacobian(p, .9, center, R0, c0, W)
        fun = lambda x: B._prior_residual_and_jacobian(x, .9, center, R0, c0, W, bias)
        residual, jacobian = fun(p)
        assert np.allclose(residual, original - W @ bias, atol=1e-12)
        assert np.array_equal(jacobian, old_jacobian)
        assert np.allclose(jacobian, P._jacobian(lambda x: fun(x)[0], p), atol=2e-6, rtol=2e-6)
    args = fixture()
    original = V4.solve_uncertain_joint_ot_pnp(*args[:6], args[6], config=args[7])
    corrected = B.solve_uncertain_joint_ot_pnp(*args[:6], args[6], config={**args[7], "prior_bias6": bias.tolist(), "bias_strength": 0.})
    assert original["R"] == corrected["R"] and original["t"] == corrected["t"]
    assert original["objective"] == corrected["objective"]
    assert original["visible_observations"] == corrected["visible_observations"]
    zero_a = B.solve_uncertain_joint_ot_pnp(*args[:6], args[6], config={**args[7], "prior_weight": 0., "prior_bias6": bias.tolist(), "bias_strength": 0.})
    zero_b = B.solve_uncertain_joint_ot_pnp(*args[:6], args[6], config={**args[7], "prior_weight": 0., "prior_bias6": bias.tolist(), "bias_strength": 1.})
    assert zero_a["R"] == zero_b["R"] and zero_a["t"] == zero_b["t"] and zero_a["objective"] == zero_b["objective"]
    assert not zero_b["prior_audit"]["bias_active_in_loss"]
    moved = B.solve_uncertain_joint_ot_pnp(*args[:6], args[6], config={**args[7], "prior_bias6": bias.tolist(), "bias_strength": 1.})
    assert moved["accepted"] and moved["prior_audit"]["bias_active_in_loss"]
    assert np.linalg.norm(np.asarray(moved["t"]) - original["t"]) > 1e-5
    assert moved["visible_observations"] == original["visible_observations"]
    for bad in ([0.] * 5, [0., 0., float("inf"), 0., 0., 0.]):
        try:
            B._configuration({"prior_bias6": bad})
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid finite bias accepted")
    # Full robust objective gradient, including constant-biased prior.
    xyz, K, states, visible, domains, hw, coarse, _ = args
    cfg = B._configuration({"bias_strength": .5, "prior_bias6": bias.tolist()})
    xyz, K, _, m, _, _ = P._inputs(xyz, K, states, visible, domains, hw, B._helper_config(cfg))
    for item in m:
        item["fit_reliability_weight"] = .7
    scale = np.linalg.norm(np.ptp(xyz, axis=0))
    R0, t0 = np.asarray(coarse["R"]), np.asarray(coarse["t"])
    p = P._pose_vector(R0, t0 + [.006, -.002, .01], scale)
    prior = {"R": R0, "center": R0 @ xyz.mean(axis=0) + t0, "whitening": np.diag([4., 5., 3., 20., 25., 8.])}
    value, gradient, _ = B._objective_pair(p, xyz, K, m, scale, prior, cfg)
    numerical = P._jacobian(lambda x: np.array([B._objective_pair(x, xyz, K, m, scale, prior, cfg)[0]]), p)[0]
    assert np.isfinite(value) and np.allclose(gradient, numerical, atol=2e-5, rtol=2e-6)
    return {"all_pass": True, "prior_Jacobian_cases": 3, "unchanged_Jacobian_after_constant_bias": True,
            "bias0_exact_pose_objective_parity": True, "lambda0_bias_has_no_loss_pose_effect": True,
            "finite_bias_validation": True, "full_biased_objective_gradient": True,
            "bias_can_move_pose_without_changing_BI": True, "dataset_GT_RGB_TEST_reads": 0}


def build(row, fold, strength):
    task = D.build_spec(row, fold["model"], 1.)
    task["arm"] = "CAL_LOGO_MEDIAN_BIAS"
    task["bias_strength"] = strength
    task["solver_input"]["config"].update(prior_bias6=deepcopy(fold["model"]["prior_fit"]["median_bias6"]), bias_strength=strength)
    task["solver_input_sha256"] = D.physics_key(task["solver_input"])
    baseline = deepcopy(task["solver_input"])
    baseline["config"].pop("prior_bias6")
    baseline["config"].pop("bias_strength")
    task["v4_baseline_input_sha256"] = D.physics_key(baseline)
    return task


def prepare():
    rows, folds, full, _ = D.load_inputs()
    original_protocol = C.read(D.OUTPUT / "PROTOCOL.json")
    D.verify_bindings(original_protocol)
    original_choice = C.read(D.OUTPUT / "SELECTED_CONFIG.json")
    original_seal = C.read(D.OUTPUT / "SELECTION_SEAL.json")
    assert original_choice["selected_prior_weight"] == 1.
    assert C.sha(D.OUTPUT / "SELECTED_CONFIG.json") == original_seal["selected_config_sha256"]
    amendment = C.read(D.FIX / "CAL_MEDIAN_BIAS_AMENDMENT_V1.json")
    assert amendment["bias_strength_candidates"] == GRID
    assert amendment["conditional_CAL_selection_stage1_prior_weight"] == 1.
    assert amendment["stage1_selection_sha256"] == C.sha(D.OUTPUT / "SELECTED_CONFIG.json")
    assert amendment["CAL_full_median_bias6"] == full["prior_fit"]["median_bias6"]
    fixture_receipt = tests()
    tasks = [build(row, folds[row["id"]], strength) for strength in GRID for row in rows]
    original_inputs = {t["solver_input_sha256"]: t for t in D.read_lines(D.OUTPUT / "CANDIDATE_INPUTS.jsonl.gz")}
    for task in tasks:
        baseline = deepcopy(task["solver_input"])
        baseline["config"].pop("prior_bias6")
        baseline["config"].pop("bias_strength")
        assert baseline == original_inputs[task["v4_baseline_input_sha256"]]["solver_input"]
    sources = deepcopy(original_protocol["source_sha256"])
    for name in ("statewise_uncertain_solver_v4_bias.py", "statewise_bias_selection_v4_1.py"):
        path = Path(__file__).parent / name
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    inputs = deepcopy(original_protocol["input_bindings"])
    for name, path in {"median_bias_amendment": D.FIX / "CAL_MEDIAN_BIAS_AMENDMENT_V1.json",
                       "stage1_protocol": D.OUTPUT / "PROTOCOL.json", "stage1_selection": D.OUTPUT / "SELECTED_CONFIG.json",
                       "stage1_selection_seal": D.OUTPUT / "SELECTION_SEAL.json",
                       "stage1_candidate_inputs": D.OUTPUT / "CANDIDATE_INPUTS.jsonl.gz",
                       "stage1_candidate_predictions": D.OUTPUT / "CANDIDATE_PREDICTIONS.jsonl.gz",
                       "stage1_candidate_prediction_seal": D.OUTPUT / "CANDIDATE_PREDICTION_SEAL.json"}.items():
        inputs[name] = D.binding(path)
    protocol = {**deepcopy(original_protocol), "schema": "CAL_LOGO_MEDIAN_PRIOR_BIAS_SELECTION_V4_1", "created_UTC": D.now(),
                "candidate_rows": 138, "bias_strength_candidates": GRID, "fixed_prior_weight": 1.,
                "prior_weight_candidates": [1.], "source_sha256": sources, "input_bindings": inputs,
                "mathematical_change": "subtract strength*fold_training_median_bias6 before prior whitening only",
                "adopt_only_if_CAL_lexicographic_order_improves": True,
                "beta_zero_exact_v4_lambda1_math_cache_reuse": True, "post_choice_controls": []}
    D.verify_bindings(protocol)
    if (OUTPUT / "PROTOCOL.json").exists():
        old = C.read(OUTPUT / "PROTOCOL.json")
        assert {k: v for k, v in old.items() if k != "created_UTC"} == {k: v for k, v in protocol.items() if k != "created_UTC"}
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        D.write(OUTPUT / "PROTOCOL.json", protocol)
        D.write(OUTPUT / "BIAS_FIXTURE_TESTS.json", fixture_receipt)
        for path, expected in sources.items():
            target = OUTPUT / "code_snapshot" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(C.ROOT / path, target)
            assert C.sha(target) == expected
        entries = [{k: t[k] for k in ("id", "session", "prior_weight", "bias_strength", "solver_input", "solver_input_sha256",
                                     "v4_baseline_input_sha256", "model_sha256", "original_BI_sha256", "calibrated_BI_sha256", "original_OT_sha256")} for t in tasks]
        D.write_lines(OUTPUT / "CANDIDATE_INPUTS.jsonl.gz", entries)
        D.write(OUTPUT / "INPUT_SEAL.json", {"created_UTC": D.now(), "before_CAL_pose_fitting": True, "candidate_rows": 138,
                "candidate_inputs_sha256": C.sha(OUTPUT / "CANDIDATE_INPUTS.jsonl.gz"), "protocol_sha256": C.sha(OUTPUT / "PROTOCOL.json"),
                "v4_beta_zero_reused": 46, "new_full_solver_tasks": 92, "TEST_files_opened": False, "CAL_scorer_target_reads": 0})
    return tasks


def worker(task):
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        solution = B.solve_uncertain_joint_ot_pnp(**task["solver_input"])
    return {"solver_input_sha256": task["solver_input_sha256"], "solution": solution,
            "solve_seconds": time.perf_counter() - started, "bias_zero_cache_reused": False}


def predict(tasks):
    protocol = C.read(OUTPUT / "PROTOCOL.json")
    D.verify_bindings(protocol)
    if (OUTPUT / "PREDICTION_SEAL.json").exists():
        assert C.sha(OUTPUT / "CANDIDATE_PREDICTIONS.jsonl.gz") == C.read(OUTPUT / "PREDICTION_SEAL.json")["predictions_sha256"]
        return
    checkpoint = OUTPUT / "SOLVER_CHECKPOINTS.jsonl.gz"
    cache = {r["solver_input_sha256"]: r for r in D.read_lines(checkpoint)} if checkpoint.exists() else {}
    old_predictions = {r["solver_input_sha256"]: r for r in D.read_lines(D.OUTPUT / "CANDIDATE_PREDICTIONS.jsonl.gz") if r["prior_weight"] == 1.}
    old_seal = C.read(D.OUTPUT / "CANDIDATE_PREDICTION_SEAL.json")
    assert C.sha(D.OUTPUT / "CANDIDATE_PREDICTIONS.jsonl.gz") == old_seal["predictions_sha256"]
    for task in tasks:
        if task["bias_strength"] != 0. or task["solver_input_sha256"] in cache:
            continue
        prior_row = old_predictions[task["v4_baseline_input_sha256"]]
        solution = deepcopy(prior_row["runtime"]["joint_pose"])
        solution["schema"] = "statewise_uncertain_solver_v4_bias"
        solution["prior_audit"].update(prior_bias6=deepcopy(task["solver_input"]["config"]["prior_bias6"]), bias_strength=0.,
            effective_prior_bias6=[0.] * 6, bias_subtracted_before_whitening=True, bias_active_in_loss=False)
        solution["bias_extension_cache_audit"] = {"beta_zero_exact_v4_pose_reuse": True,
            "original_prediction_seal_sha256": C.sha(D.OUTPUT / "CANDIDATE_PREDICTION_SEAL.json"),
            "original_solver_input_sha256": task["v4_baseline_input_sha256"],
            "new_solver_math_bias0_parity_fixture_passed": True}
        result = {"solver_input_sha256": task["solver_input_sha256"], "solution": solution,
                  "solve_seconds": 0., "bias_zero_cache_reused": True}
        D.append_checkpoint(checkpoint, result)
        cache[task["solver_input_sha256"]] = result
    pending = [t for t in tasks if t["solver_input_sha256"] not in cache]
    started = time.perf_counter()
    print({"declared": 138, "beta0_reused": 46, "new_pending": len(pending)}, flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as executor:
            futures = {executor.submit(worker, t): t for t in pending}
            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                assert result["solver_input_sha256"] == task["solver_input_sha256"]
                D.append_checkpoint(checkpoint, result)
                cache[result["solver_input_sha256"]] = result
                completed += 1
                progress = {"new_completed": completed, "new_pending_initial": len(pending), "beta0_reused": 46,
                            "elapsed_seconds": time.perf_counter() - started, "last_id": task["id"], "last_bias_strength": task["bias_strength"],
                            "last_solve_seconds": result["solve_seconds"]}
                D.write(OUTPUT / "LIVE_PROGRESS.json", progress)
                if completed % 5 == 0 or completed == len(pending):
                    print(progress, flush=True)
    rows = []
    for task in tasks:
        result = cache[task["solver_input_sha256"]]
        runtime = finish_observations(task["observation"], deepcopy(result["solution"]), task["solver_input"]["visible"], task["model"])
        assert runtime["frozen_visible_measurements"] == task["observation"]["frozen_visible_measurements"]
        rows.append({"id": task["id"], "partition": "CAL", "session": task["session"], "arm": task["arm"],
            "prior_weight": 1., "bias_strength": task["bias_strength"], "runtime": runtime, "actual_pose": runtime["actual_pose"],
            "solver_input_sha256": task["solver_input_sha256"], "original_BI_sha256": task["original_BI_sha256"],
            "calibrated_BI_sha256": task["calibrated_BI_sha256"], "original_OT_sha256": task["original_OT_sha256"],
            "v4_baseline_input_sha256": task["v4_baseline_input_sha256"], "bias_zero_cache_reused": result["bias_zero_cache_reused"],
            "solve_seconds": result["solve_seconds"], "prediction_contains_no_score": True, "TEST_files_opened": False})
    D.write_lines(OUTPUT / "CANDIDATE_PREDICTIONS.jsonl.gz", rows)
    D.verify_bindings(protocol)
    D.write(OUTPUT / "PREDICTION_SEAL.json", {"schema": "CAL_MEDIAN_BIAS_CANDIDATE_SEAL_V4_1", "created_UTC": D.now(),
        "rows": 138, "predictions_sha256": C.sha(OUTPUT / "CANDIDATE_PREDICTIONS.jsonl.gz"),
        "protocol_sha256": C.sha(OUTPUT / "PROTOCOL.json"), "source_sha256": protocol["source_sha256"],
        "before_CAL_scorer_target_read": True, "beta0_exact_original_pose_reuse": 46, "new_full_solver_tasks": 92,
        "elapsed_seconds": time.perf_counter() - started, "TEST_files_opened": False})


def score():
    seal = C.read(OUTPUT / "PREDICTION_SEAL.json")
    assert C.sha(OUTPUT / "CANDIDATE_PREDICTIONS.jsonl.gz") == seal["predictions_sha256"]
    labels = D.read_CAL_targets_after_seal(OUTPUT, "PREDICTION_SEAL.json")
    rows = D.read_lines(OUTPUT / "CANDIDATE_PREDICTIONS.jsonl.gz")
    groups = defaultdict(list)
    for row in rows:
        row["metrics"] = pose_metric(row["actual_pose"], labels[row["id"]]["target"]["truth"])
        groups[row["bias_strength"]].append(row)
    summaries = {str(strength): D.summarize(groups[strength]) for strength in GRID}
    assert all(s["frames"] == 46 for s in summaries.values())
    metric_key = lambda strength: (-summaries[str(strength)]["success_count"],
                                  summaries[str(strength)]["metrics"]["translation_cm"]["mean"],
                                  summaries[str(strength)]["metrics"]["rotation_deg"]["mean"])
    selected = min(GRID, key=lambda strength: (*metric_key(strength), GRID.index(strength)))
    adopted = selected != 0. and metric_key(selected) < metric_key(0.)
    if not adopted:
        selected = 0.
    D.write_lines(OUTPUT / "CANDIDATE_SCORED_ROWS.jsonl.gz", rows)
    result = {"schema": "CAL_MEDIAN_BIAS_SELECTION_SCORE_V4_1", "created_UTC": D.now(), "summary": summaries,
              "selected_bias_strength": selected, "bias_adopted_by_CAL_rule": adopted,
              "fixed_prior_weight": 1., "selection_rule": ["max_all46_success", "min_T", "min_R", "grid_order"],
              "prediction_seal_sha256": C.sha(OUTPUT / "PREDICTION_SEAL.json"), "CAL_target_read_after_prediction_seal": True,
              "native_incompatible2_kept_in_scoring": True, "TEST_files_opened": False}
    D.write(OUTPUT / "CAL_SELECTION_SCORE.json", result)
    _, _, model, _ = D.load_inputs()
    config = {**deepcopy(B._DEFAULTS), **D.solver_config_from_model(model, 1.),
              "prior_bias6": deepcopy(model["prior_fit"]["median_bias6"]), "bias_strength": selected}
    protocol = C.read(OUTPUT / "PROTOCOL.json")
    selected_record = {"schema": "FROZEN_CAL_ONLY_MEDIAN_BIAS_CONFIG_V4_1", "created_UTC": D.now(),
        "selected_prior_weight": 1., "selected_bias_strength": selected, "bias_adopted": adopted,
        "solver_configuration": config, "calibration_model": D.binding(D.FIX / "calibration/final_adapter_CAL46/MODEL.json"),
        "selection_rule": result["selection_rule"], "CAL_prediction_seal": D.binding(OUTPUT / "PREDICTION_SEAL.json"),
        "CAL_SELECTION_SCORE_sha256": C.sha(OUTPUT / "CAL_SELECTION_SCORE.json"), "protocol_sha256": C.sha(OUTPUT / "PROTOCOL.json"),
        "source_sha256": protocol["source_sha256"], "input_bindings": protocol["input_bindings"],
        "stage1_selected_config": D.binding(D.OUTPUT / "SELECTED_CONFIG.json"),
        "selected_from_LOGO_CAL46_only": True, "conditional_subsystem_LOGO_not_full_pipeline_OOF": True,
        "TEST_files_opened": False, "fixed_grid_tie_order": GRID}
    if (OUTPUT / "SELECTED_CONFIG.json").exists():
        assert C.read(OUTPUT / "SELECTED_CONFIG.json")["solver_configuration"] == config
    else:
        D.write(OUTPUT / "SELECTED_CONFIG.json", selected_record)
    D.verify_bindings(protocol)
    D.write(OUTPUT / "SELECTION_SEAL.json", {"schema": "CAL_MEDIAN_BIAS_SELECTION_SEAL_V4_1", "created_UTC": D.now(),
        "selected_config_sha256": C.sha(OUTPUT / "SELECTED_CONFIG.json"), "selected_bias_strength": selected,
        "selected_prior_weight": 1., "bias_adopted": adopted, "CAL_score_sha256": C.sha(OUTPUT / "CAL_SELECTION_SCORE.json"),
        "CAL_prediction_seal_sha256": C.sha(OUTPUT / "PREDICTION_SEAL.json"), "calibration_model": selected_record["calibration_model"],
        "source_sha256": protocol["source_sha256"], "TEST_files_opened": False, "before_new_TEST_score": True})
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("test", "prepare", "predict", "score", "all"))
    args = parser.parse_args()
    if args.stage == "test":
        print(tests(), flush=True)
        return
    if args.stage in ("prepare", "predict", "all"):
        tasks = prepare()
        if args.stage != "prepare":
            predict(tasks)
    if args.stage in ("score", "all"):
        score()


if __name__ == "__main__":
    main()
