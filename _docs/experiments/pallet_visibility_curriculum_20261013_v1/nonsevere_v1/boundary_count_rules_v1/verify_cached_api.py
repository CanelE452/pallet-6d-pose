"""Prediction-only CAL46 closeout: frozen helper and live cached-observer API."""
from copy import deepcopy
import builtins
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CORE = ROOT / "scripts/research/pallet_visibility_curriculum_20261013_v1"
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_evidence import summarize_boundary_evidence
from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_rule_api import BoundaryRuleCorrector


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in gzip.open(path, "rt")]


class CachedObserver:
    def __init__(self, observation):
        self.observation = deepcopy(observation)

    def assess(self, image, points, K, initial):
        # Cached evidence is bound to the sealed CAL46 observation, not imagezeros.
        return deepcopy(self.observation)


def main():
    out = HERE / "CACHED_API_VERIFICATION.json"
    amendment = HERE / "POSTSEAL_CANVAS_VALIDATION_AMENDMENT.json"
    if out.exists() or amendment.exists():
        raise FileExistsError("Preserve completed verification")
    # No labels, retired/new TEST inputs, severe rows or model forwards are needed.
    original = builtins.open, io.open, Path.open
    def wrap(fn):
        def guard(path, *args, **kwargs):
            name = str(path)
            if any(s in name for s in ("SCORING", "SCORED_ROWS", "TARGETS", "RESULTS.json",
                                      "/annotations/", "test_reused", "FINAL")):
                raise AssertionError("Forbidden verification input: " + name)
            return fn(path, *args, **kwargs)
        return guard
    builtins.open, io.open, Path.open = map(wrap, original)
    started = time.perf_counter()
    seal = read(HERE / "PREDICTION_SEAL.json")
    assert seal["before_scoring"] and sha(HERE / "PREDICTIONS.jsonl.gz") == seal["predictions_sha256"]
    assert sha(HERE / "PROTOCOL.json") == seal["protocol_sha256"]
    assert sha(HERE / "OBSERVATIONS.jsonl.gz") == seal["observations_sha256"]
    for filename in ("boundary_rules.py", "boundary_evidence.py", "curriculum.py", "test_boundary_rules.py"):
        assert sha(HERE / "code_snapshot" / filename) == seal["code_bindings"][str(CORE / filename)]
    for filename in ("boundary_rules.py", "curriculum.py", "test_boundary_rules.py"):
        assert sha(CORE / filename) == seal["code_bindings"][str(CORE / filename)]
    name = "_count_verification_frozen"
    spec = importlib.util.spec_from_file_location(name, HERE / "code_snapshot/__init__.py",
        submodule_search_locations=[str(HERE / "code_snapshot")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    frozen_helper = importlib.import_module(name + ".boundary_evidence")
    frames = read(HERE.parent / "CAL_INPUTS.json")["frames"]
    observations = {r["id"]: r for r in rows(HERE / "OBSERVATIONS.jsonl.gz")}
    predictions = {(r["id"], r["arm"]): r for r in rows(HERE / "PREDICTIONS.jsonl.gz")}
    assert len(frames) == len(observations) == 46 and len(predictions) == 322
    cv2.setNumThreads(1)
    counts = dict(frames=0, helper_full_output_exact=0, API_full_solution_exact=0,
        API_actual_pose_exact=0, API_3DOF_both_rotations_exact=0,
        API_initial_fallback_exact=0, projected_completion_only=0, center_exact=0)
    checks = []
    image_bindings = []
    live_before = {str(CORE / f): sha(CORE / f) for f in
        ("boundary_rules.py", "boundary_evidence.py", "boundary_rule_api.py", "curriculum.py", "common.py", "joint_api.py")}
    for frame in frames:
        assert frame["partition"] == "CAL"
        fid = frame["id"]
        observation = observations[fid]["observation"]
        h, w = frame["raw_hw"]
        current = summarize_boundary_evidence(observation, image_hw=(h, w))
        frozen = frozen_helper.summarize_boundary_evidence(observation, image_hw=(h, w))
        assert current == frozen == observations[fid]["evidence"]
        counts["helper_full_output_exact"] += 1
        initial = frame["initial_poses"]["N3_SUBPIX"]
        raw = np.asarray(observation["raw_points"], float)
        snapshot = deepcopy(observation)
        for coord, arm in (("RAW", "RAW_BOUNDARY_COUNT_RULE"), ("FUSED", "FUSED_SIGMA2_BOUNDARY_COUNT_RULE")):
            expected = predictions[fid, arm]
            actual = BoundaryRuleCorrector(CachedObserver(observation), {"coordinate_source": coord}).refine(
                np.zeros((h, w, 3), np.uint8), raw, frame["K"], initial, frame["xyz"])
            assert actual["solution"] == expected["solution"]
            assert actual["actual_pose"] == expected["actual_pose"]
            assert actual["self_hidden"] == expected["fixed_H"]
            assert actual["observed_points"] == observation["raw_points"]
            assert actual["corrected_measurements"] == expected["observed_points"]
            assert actual["completed_points"][8:] == observation["raw_points"][8:]
            assert actual["projected_points"] == expected["solution"]["projected"]
            solution = actual["solution"]
            assert set(solution["used_ids"]) <= set(current["strong_ids"])
            assert all(not observation["self_hidden"][k] for stage in solution["stages"] for k in stage["used_ids"])
            assert not actual["provenance"]["projected_completion_used_for_fit"]
            assert not actual["provenance"]["projections_used_as_measurements"]
            assert solution["unknown_added_ids"] == solution["interior_supplement_ids"] == []
            if solution["accepted"]:
                ids = sorted(set(range(8)) - set(solution["used_ids"]))
                assert actual["projected_completion_ids"] == ids
                np.testing.assert_array_equal(np.asarray(actual["completed_points"])[ids],
                                              np.asarray(actual["projected_points"])[ids])
                assert solution["aggregate_before"]["objective"] - solution["aggregate_after"]["objective"] > 1e-7
                if solution["estimated_dof"] == 3:
                    assert actual["actual_pose"]["R_cf"] == initial["R_cf"]
                    assert actual["actual_pose"]["R_physical"] == initial["R_physical"]
                    counts["API_3DOF_both_rotations_exact"] += 1
            else:
                assert actual["actual_pose"] == initial and solution["used_ids"] == []
                assert actual["projected_completion_ids"] == []
                assert actual["completed_points"] == observation["raw_points"]
                counts["API_initial_fallback_exact"] += 1
            counts["API_full_solution_exact"] += 1
            counts["API_actual_pose_exact"] += 1
            counts["projected_completion_only"] += 1
            counts["center_exact"] += 1
            checks.append(dict(id=fid, coordinate_source=coord, branch=solution["branch"],
                count=solution["reliable_boundary_count"], accepted=solution["accepted"], used_ids=solution["used_ids"],
                solution_exact=True, actual_pose_exact=True, completed_ids=actual["projected_completion_ids"]))
        assert observation == snapshot
        image_bindings.append(dict(id=fid, image=frame["image"], image_sha256=frame["image_sha256"],
            role="sealed source identity only; cached observer verification does not decode image"))
        counts["frames"] += 1
    for path, digest in live_before.items():
        assert sha(path) == digest
    assert counts["API_full_solution_exact"] == 92 and counts["API_initial_fallback_exact"] == 64
    assert counts["API_3DOF_both_rotations_exact"] == 18
    builtins.open, io.open, Path.open = original
    helper_change = dict(schema="postseal_canvas_validation_amendment_v1", after_CAL_scoring=True,
        frozen_protocol_and_predictions_preserved=True,
        frozen_helper_sha256=sha(HERE / "code_snapshot/boundary_evidence.py"),
        current_helper_sha256=sha(CORE / "boundary_evidence.py"),
        change="Reject fractional image_hw instead of silently converting to integers; actual CAL46 canvases are unchanged integers.",
        threshold_policy_model_coordinate_changes=False, full_helper_output_exact_frames=46,
        exact_cached_API_outputs=92, evidence_strong_points=sum(r["evidence"]["strong_count"] for r in observations.values()))
    C.write(amendment, helper_change)
    C.write(out, dict(schema="count_rule_CAL_cached_API_verification_v1", PASS=True,
        scope="CAL46 prediction-only cached observer; software/API parity, not new accuracy evaluation or realRGB forwards",
        severe_inputs=0, TEST_inputs=0, FINAL_inputs=0, GT_inputs=0, image_or_model_forwards=0,
        cached_image_zeros_use_actual_canvas=True, counts=counts,
        maximum_R_t_solution_difference=0., source_hashes=live_before,
        frozen_predictions_sha256=seal["predictions_sha256"], protocol_sha256=seal["protocol_sha256"],
        observations_sha256=seal["observations_sha256"], verifier_sha256=sha(__file__),
        helper_amendment_sha256=sha(amendment), image_source_bindings=image_bindings,
        per_frame_checks=checks, seconds=time.perf_counter()-started))
    print("CAL_CACHED_API_VERIFIED", json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
