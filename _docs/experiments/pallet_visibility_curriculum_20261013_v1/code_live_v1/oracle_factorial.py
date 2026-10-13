"""CAL-only oracle diagnosis; never part of the deployable inference pipeline.

The 2x2 intervention fixes the predicted initial pose/model/self-occlusion mask
and cached RGB evidence, then replaces coordinates and/or visibility. Annotated
coordinates include historical geometric completion of virtual/hidden corners.
Model/physical-basis consistency is audited separately, since a frozen incorrect
W/D hypothesis cannot be repaired by changing coordinate precision alone.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from .curriculum import solve_curriculum
from .run import simple_pose
from .scoring import distribution, paired, pose_metric, rotation_y


def oracle_appearance(cached, human_states):
    result = deepcopy(cached)
    states, visibility, accuracy = [], [], []
    for index, state in enumerate(human_states):
        if state == "DIRECT_VISIBLE":
            states.append("VISIBLE_BOUNDARY" if cached["boundary_score"][index] >= .35 else "VISIBLE_INTERIOR")
            visibility.append(1.)
            accuracy.append(1.)
        else:
            states.append({"EXTERNAL_OCCLUDED": "OCCLUSION_SUSPECTED",
                           "SELF_OCCLUDED": "SELF_OCCLUDED",
                           "UNANNOTATED": "UNKNOWN"}.get(state, "UNKNOWN"))
            visibility.append(0.)
            accuracy.append(0.)
    result.update(states=states, visibility_score=visibility, local_accuracy_score=accuracy)
    result["oracle_visibility"] = True
    result["oracle_boundary_or_interior"] = False
    return result


def model_phase_audit(frame, initial):
    mapping = np.asarray(frame["scoring_metadata"]["native_to_reference_corner_ids"], int)
    if sorted(mapping.tolist()) != list(range(8)):
        raise ValueError("native_to_reference_corner_ids must be a permutation of 0..7")
    Q = np.asarray(initial["R_cf"]).T @ np.asarray(initial["R_physical"])
    physical_points = C.cuboid(*initial["cf_extents"]) @ Q.T
    truth = frame["target"]["truth"]
    # target.gt is expressed in the target bbox/body corner order, whose W/D
    # basis may differ from the registry physical model used by pose scoring.
    physical_from_reference = np.asarray(truth["body_R"]).T @ np.asarray(truth["R"])
    reference_physical_points = C.cuboid(*truth["body_xyz"]) @ physical_from_reference.T
    mapped_reference_points = reference_physical_points[mapping]
    gaps = [float(np.max(np.linalg.norm(physical_points @ rotation_y(j * 2 * np.pi / truth["order"]).T - mapped_reference_points,
                                        axis=1))) for j in range(truth["order"])]
    return dict(consistent=bool(min(gaps) < 1e-6), maximum_3d_corner_gap_m=min(gaps),
                native_to_reference_corner_ids=mapping.tolist(),
                initial_cf_extents=initial["cf_extents"], truth_physical_extents=truth["xyz"],
                truth_reference_extents=truth["body_xyz"],
                physical_from_reference_body=physical_from_reference.tolist(), physical_from_cf=Q.tolist())


def prediction_from_solution(initial, solution):
    Q = np.asarray(initial["R_cf"]).T @ np.asarray(initial["R_physical"])
    return dict(available=True, R_cf=solution["R"],
                R_physical=(np.asarray(solution["R"]) @ Q).tolist(),
                centroid=solution["t"], cf_extents=initial["cf_extents"],
                selected_hypothesis=initial["selected_hypothesis"])


def summary(rows):
    if not rows:
        return dict(frames=0)
    return dict(frames=len(rows), sessions=len({r["session"] for r in rows}),
                accepted=sum(r["solution"]["accepted"] for r in rows),
                reasons=dict(Counter(r["solution"].get("reason", "unspecified") for r in rows)),
                metrics={name: distribution([r["metric"][name] for r in rows])
                         for name in ("translation_cm", "rotation_deg", "ADDsym_cm")},
                success_5cm_5deg=float(np.mean([r["metric"]["success_5cm_5deg"] for r in rows])))


def run(bundle_path, prediction_directory, output, include_reference_control=False):
    bundle = C.read(bundle_path)
    frames = bundle["frames"]
    if not frames or any(f["partition"] != "CAL" or f["scoring_metadata"]["partition"] != "CAL" for f in frames):
        raise ValueError("This diagnostic accepts an explicitly CAL-only input bundle")
    protocol = C.read(prediction_directory / "PROTOCOL.json")
    if protocol["partition"] != "calibration" or set(protocol["ids"]) != {f["id"] for f in frames}:
        raise ValueError("CAL-only predictions must match the complete input bundle population")
    input_rows = list(C.rows(prediction_directory / "PREDICTIONS.jsonl.gz"))
    by_method = {(r["id"], r["method"]): r for r in input_rows if r["arm"] == "N3_SUBPIX"}
    if any(r["id"] not in protocol["ids"] for r in input_rows):
        raise ValueError("Prediction artifact contains a non-CAL frame")
    output.mkdir(parents=True, exist_ok=False)
    settings = protocol["settings"].get("curriculum") or {}
    C.write(output / "PROTOCOL.json", dict(schema="CAL_ORACLE_FACTORIAL_V1", diagnostic_only=True,
        deployable=False, oracle_data_used=True, TEST_accessed=False, FINAL_accessed=False,
        frozen_initial_pose=True, frozen_initial_model_and_physical_basis=True,
        frozen_initial_self_mask=True, estimated_visibility_source="cached original RGB evidence and trained CAL head",
        oracle_coordinate_source="target.gt indexed by fixed native_to_reference_corner_ids",
        boundary_topology_source="cached RGB boundary score, shared across visibility intervention",
        oracle_coordinates_include_geometric_completion=True,
        reference_control=bool(include_reference_control), reference_corner_basis="truth.body_xyz/body_R, mapped to truth physical R",
        settings=settings, ids=protocol["ids"],
        bundle_sha256=C.sha(bundle_path), predictions_sha256=C.sha(prediction_directory / "PREDICTIONS.jsonl.gz"),
        source_sha256={name: C.sha(Path(__file__).parent / name) for name in
                       ("oracle_factorial.py", "curriculum.py", "run.py", "scoring.py")}))
    results, audits = [], []
    start = time.monotonic()
    for frame in frames:
        ident = frame["id"]
        cached = by_method[(ident, "RGB_CURRICULUM")]
        initial = frame["initial_poses"]["N3_SUBPIX"]
        frozen = by_method[(ident, "INITIAL")]["actual_pose"]
        if not np.allclose(initial["R_cf"], frozen["R_cf"]) or not np.allclose(initial["centroid"], frozen["centroid"]):
            raise ValueError("Initial pose binding changed: " + ident)
        xyz = C.cuboid(*initial["cf_extents"])
        K = np.asarray(frame["K"], float)
        R, t = np.asarray(initial["R_cf"]), np.asarray(initial["centroid"])
        self_hidden = C.self_hidden(xyz, R, t)
        if not np.array_equal(self_hidden, cached["self_hidden"]):
            raise ValueError("Initial self-occlusion binding changed: " + ident)
        phase = model_phase_audit(frame, initial)
        audits.append(dict(id=ident, session=frame["session"], **phase))
        q = np.asarray(cached["original_points"], float)
        mapping = np.asarray(phase["native_to_reference_corner_ids"], int)
        oracle_q = np.asarray(frame["target"]["gt"], float)[:8][mapping]
        estimated = cached["appearance"]
        oracle_state = oracle_appearance(estimated, frame["scoring_metadata"]["human_states_native"])
        solutions = {}
        for coord_name, coords in (("PRED_COORD", q), ("ORACLE_COORD", oracle_q)):
            for state_name, appearance in (("EST_STATE", estimated), ("ORACLE_STATE", oracle_state)):
                name = coord_name + "_" + state_name
                solutions[name] = solve_curriculum(coords, xyz, K, R, t, self_hidden, appearance, settings)
        # Independent unregularized controls determine whether a frozen local
        # prior, a wrong object hypothesis, or correspondences cap this method.
        solutions["PRED_ALL_STANDARD"] = simple_pose(q, xyz, K, initial, np.ones(8, bool))
        solutions["ORACLE_ALL_STANDARD"] = simple_pose(oracle_q, xyz, K, initial, np.ones(8, bool))
        if include_reference_control:
            truth = frame["target"]["truth"]
            Q = np.asarray(initial["R_cf"]).T @ np.asarray(initial["R_physical"])
            # This is a geometric oracle, not manual corner annotation.  It
            # projects the exact target physical pose in the frozen model basis.
            geometric_oracle_q = C.project(xyz, np.asarray(truth["R"]) @ Q.T, truth["t"], K)
            solutions["ORACLE_GEOMETRIC_PROJECTION_STANDARD"] = simple_pose(
                geometric_oracle_q, xyz, K, initial, np.ones(8, bool))
        for name, solution in solutions.items():
            actual_pose = prediction_from_solution(initial, solution)
            results.append(dict(id=ident, session=frame["session"], variant=name,
                diagnosis="ORACLE_DIAGNOSTIC" if name.startswith("ORACLE_") or "ORACLE_STATE" in name else "PREDICTION_DIAGNOSTIC",
                oracle_coordinate_kind="exact_physical_pose_projection" if name == "ORACLE_GEOMETRIC_PROJECTION_STANDARD" else
                                       ("historical_annotation_with_completion" if name.startswith("ORACLE_") else None),
                initial_self_hidden=self_hidden.tolist(), model_phase_consistent=phase["consistent"],
                solution=solution, actual_pose=actual_pose,
                metric=pose_metric(actual_pose, frame["target"]["truth"])))
        if include_reference_control:
            reference_xyz = C.cuboid(*frame["target"]["truth"]["body_xyz"])
            reference_q = np.asarray(frame["target"]["gt"], float)[:8]
            truth = frame["target"]["truth"]
            reference_initial = dict(R_cf=truth["body_R"], R_physical=truth["R"], centroid=truth["t"],
                                     cf_extents=truth["body_xyz"], selected_hypothesis="ORACLE_REFERENCE")
            solution = simple_pose(reference_q, reference_xyz, np.asarray(frame["target"]["K"]),
                                   reference_initial, np.ones(8, bool))
            actual_pose = prediction_from_solution(reference_initial, solution)
            results.append(dict(id=ident, session=frame["session"], variant="ORACLE_REFERENCE_STANDARD",
                diagnosis="ORACLE_DIAGNOSTIC", initial_self_hidden=None,
                model_phase_consistent=phase["consistent"], reference_model_consistent=True,
                initial_model_replaced_for_reference_control=True,
                solution=solution, actual_pose=actual_pose, metric=pose_metric(actual_pose, truth)))
    C.save_rows(output / "ROWS.jsonl.gz", results)
    C.write(output / "MODEL_PHASE_AUDIT.json", dict(frames=audits,
        consistent=sum(r["consistent"] for r in audits), inconsistent=sum(not r["consistent"] for r in audits)))
    groups = defaultdict(list)
    for row in results:
        groups[row["variant"]].append(row)
    aggregate = {variant: summary(rows) for variant, rows in groups.items()}
    subsets = {}
    for label, predicate in (("PHYSICAL_MODEL_PHASE_CONSISTENT", lambda r:r["model_phase_consistent"]),
                             ("FROZEN_MODEL_PHASE_INCONSISTENT", lambda r:not r["model_phase_consistent"])):
        subsets[label] = {variant: summary([r for r in rows if predicate(r)]) for variant, rows in groups.items()}
    initial_reference = {f["id"]:dict(id=f["id"], session=f["session"],
        metric=pose_metric(f["initial_poses"]["N3_SUBPIX"], f["target"]["truth"])) for f in frames}
    comparisons = {variant + "-FROZEN_INITIAL": paired(rows, initial_reference) for variant, rows in groups.items()}
    comparisons["PRED_COORD_ORACLE_STATE-PRED_COORD_EST_STATE"] = paired(groups["PRED_COORD_ORACLE_STATE"], {r["id"]:r for r in groups["PRED_COORD_EST_STATE"]})
    comparisons["ORACLE_COORD_EST_STATE-PRED_COORD_EST_STATE"] = paired(groups["ORACLE_COORD_EST_STATE"], {r["id"]:r for r in groups["PRED_COORD_EST_STATE"]})
    C.write(output / "RESULTS.json", dict(schema="CAL_ORACLE_FACTORIAL_V1", diagnostic_only=True,
        frames=len(frames), variants=len(groups), rows=len(results), seconds=time.monotonic()-start,
        summary=aggregate, subsets=subsets, paired_comparisons=comparisons,
        frozen_initial_summary=summary([dict(**r, solution=dict(accepted=False, reason="frozen_initial")) for r in initial_reference.values()]),
        TEST_accessed=False, FINAL_accessed=False, rows_sha256=C.sha(output / "ROWS.jsonl.gz"),
        limitations=["CAL revisited; no estimate of unseen generalization",
                     "Historical pose reference reconstructed from labels, not an independent physical measurement",
                     "GT coordinates include completed hidden/virtual corners",
                     "Physical W/D model branch remains frozen in the factorial",
                     "Cached RGB state is held fixed when coordinates change; no oracle RGB resampling",
                     "Oracle visible states remove self/external occlusion as well as unknown states; initial hard self mask remains"] ))
    for variant, item in aggregate.items():
        print(variant, "accepted", item["accepted"], "T_mean", round(item["metrics"]["translation_cm"]["mean"], 5),
              "T_median", round(item["metrics"]["translation_cm"]["median"], 5),
              "R_mean", round(item["metrics"]["rotation_deg"]["mean"], 5),
              "success", round(item["success_5cm_5deg"], 5), flush=True)
    print("MODEL_PHASE", "consistent", sum(r["consistent"] for r in audits),
          "inconsistent", sum(not r["consistent"] for r in audits), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-reference-control", action="store_true")
    args = parser.parse_args()
    run(args.bundle, args.predictions, args.output, args.include_reference_control)
