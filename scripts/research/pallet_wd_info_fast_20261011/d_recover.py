"""Recover only a failed final solve using sealed, prediction-only selector fits.

The initial adapter passed a JSON list as K to the frozen OpenCV solver. Preserve
that attempt and its solver charges; complete its final8-corner fit with ndarray
K. No refitting, new candidate selection, data changes or hyperparameter changes.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import d_local as D
import d_evaluate as E


def cached_selection(row, tolerance):
    xyz = np.asarray(row["fixed_metadata"]["dimensions_pnp_WH_D_m"], float)
    candidates = []
    for name, record in row["candidate_selector_values"].items():
        dims = dict(width=float(max(xyz[0], xyz[2]) if name=="long-face-front" else min(xyz[0], xyz[2])),
                    height=float(xyz[1]), depth=float(min(xyz[0], xyz[2]) if name=="long-face-front" else max(xyz[0], xyz[2])))
        dimensions = SimpleNamespace(as_dict=lambda dims=dims: dict(dims))
        candidates.append(SimpleNamespace(name=name, success=record["success"], score=record["score"],
                                          camera_facing_dimensions=dimensions))
    valid = [candidate for candidate in candidates if candidate.success]
    if len(valid)==1:
        selected = valid[0].name
    elif len(valid)==2 and abs(valid[0].score-valid[1].score)>tolerance:
        selected = min(valid, key=lambda candidate: float(candidate.score)).name
    else:
        selected = None
    return SimpleNamespace(hypotheses=candidates, selected_hypothesis=selected)


def run(args):
    env, refiner, data_type, protocol, checkpoint_path, base, heads = D.setup(args, "cpu")
    common = E.new_common(); common.lock_ready()
    public = Path(args.public_dir); private = Path(args.private_dir)
    first_seal = D.read(public/"D_POSE_SEAL.json")
    assert D.file_sha(private/first_seal["private_namespace"]) == first_seal["sha256"]
    first_calls = sum(first_seal["actual_cv_calls"].values())
    assert first_calls == 6055
    old = list(common.rows(private/first_seal["private_namespace"]))
    assert len(old)==1211 and all(not row["actual_pose"]["available"] for row in old)
    pose = E.module("wd_D_recovered_pose", env.HERE/"pose.py")
    from challenge.evaluation_v2.pnp_selector import SelectorConfig
    tolerance = SelectorConfig().parity_tie_tolerance
    lock = dict(status="LOCKED_BEFORE_RECOVERY_SOLVERS", cause="K decoded from JSON was a list, accepted by selector validation but rejected by final cv2.solvePnP",
                first_attempt="D_POSE_SEAL.json", first_attempt_cv_calls=first_calls,
                first_source_sha256=D.file_sha(Path(__file__).with_name("d_evaluate.py")),
                recovery_code_sha256=D.file_sha(__file__),
                unchanged=["training checkpoints", "qFinal", "K numeric entries", "registered dimensions", "original selector scores", "original min-score/tie rule", "original infer/solve/metric"],
                changed="Convert K to float64 ndarray; reuse sealed original selector results and perform only original final8-corner solve",
                planned_remaining_cv_calls=2422+18, maximum_cumulative_D_cv_calls=8495,
                verification="Three fixed first frames: cached selector+final solve vs fresh complete original infer; all pose fields must agree at 1e-7")
    common.write(public/"D_REPAIR_LOCK.json", lock)
    recovered = []; checks = []
    with common.pnp_counter("D", 8500) as calls:
        guarded = {}
        for name in calls:
            original_cv = getattr(cv2, name); guarded[name] = original_cv
            def cumulative_guard(*a, _fn=original_cv, **kw):
                assert first_calls+sum(calls.values()) < 8500, "Cumulative D solver budget exhausted"
                return _fn(*a, **kw)
            setattr(cv2, name, cumulative_guard)
        for row in old:
            saved_selector = cached_selection(row, tolerance)
            assert saved_selector.selected_hypothesis is not None
            original = pose.select_pnp_hypotheses
            pose.select_pnp_hypotheses = lambda *a, **kw: saved_selector
            try:
                actual = pose.infer(np.asarray(row["qFinal"], float), np.asarray(row["fixed_metadata"]["K"], float),
                                    np.asarray(row["fixed_metadata"]["dimensions_pnp_WH_D_m"], float), False)
            finally:
                pose.select_pnp_hypotheses = original
            assert actual["available"], row["id"]
            recovered.append(dict(row, actual_pose=actual, correction="K ndarray; sealed original selector fit reused"))
        for row in recovered[:3]:
            fresh = pose.infer(np.asarray(row["qFinal"], float), np.asarray(row["fixed_metadata"]["K"], float),
                               np.asarray(row["fixed_metadata"]["dimensions_pnp_WH_D_m"], float), False)
            cached = row["actual_pose"]
            assert fresh["available"] == cached["available"] and fresh["selected_hypothesis"] == cached["selected_hypothesis"]
            deltas = {key: float(np.max(np.abs(np.asarray(fresh[key])-np.asarray(cached[key]))))
                      for key in ("R_cf", "R_physical", "centroid", "cf_extents", "reprojection_px")}
            assert max(deltas.values()) <= 1e-7
            checks.append(dict(id=row["id"], method=row["method"], deltas=deltas, PASS=True))
        for name, original_cv in guarded.items():
            setattr(cv2, name, original_cv)
    assert first_calls+sum(calls.values()) <= 8500
    destination = private/"D/REPAIRED_POSES.jsonl.gz"
    E.write_rows(destination, recovered, common)
    common.write(public/"D_REPAIRED_POSE_SEAL.json", dict(status="SEALED_BEFORE_CORRECTED_REFERENCE_SCORING",
                 private_namespace="D/REPAIRED_POSES.jsonl.gz", sha256=D.file_sha(destination),
                 recovery_cv_calls=dict(calls), cumulative_D_cv_calls=first_calls+sum(calls.values()),
                 fresh_complete_original_F_checks=checks, initial_attempt_preserved=True,
                 qFinal_unchanged=True, reference_inputs_in_F=False))
    allowed = D.read(public/"INPUT_CONTRACT.json"); ids = set(allowed["allowed_ids"])
    geometry = E.module("wd_D_recovered_geometry", Path(__file__).with_name("geometry_analysis.py"))
    reference_rows = {r["id"]: r for r in E.published_rows(args, "_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz")
                      if r["id"] in ids and r["method"]=="BASE" and r["seed"]==1}
    truth = geometry.truth_real(env.ROOT, reference_rows)
    first_rows = list(common.rows(public/"D_ROWS.jsonl.gz"))
    old_by = {(r["id"], r["method"]): r for r in first_rows}
    scored = []
    for row in recovered:
        fid = row["id"]; record = dict(old_by[fid, row["method"]])
        metric = pose.metric((fid, row["actual_pose"], truth[fid]))
        hypothesis = row["actual_pose"]["selected_hypothesis"]
        unclear = fid in allowed["unclear_allowed_ids"]
        record.update(pose=metric, hypothesis=hypothesis, axis_wrong=None if unclear else int(hypothesis!=reference_rows[fid]["hypOracle"]),
                      axis_status="UNCLEAR" if unclear else "EVALUABLE", strict_success=geometry.good(metric),
                      adapter_recovery=True, first_attempt_pose_invalid=True)
        scored.append(record)
    baseline = [dict(r) for r in first_rows if r["method"] in ("OLD_P", "N3_DIM_SYM", "N3_THEN_SUBPIX")]
    historical = {(r["id"], "OLD_P" if r["method"]=="P" else r["method"]): r["pose"]
                  for r in E.published_rows(args, "_docs/experiments/pallet_paper_tables_245_20261011/RECT_ROWS.jsonl.gz")
                  if r["id"] in ids and r["backbone"]=="YOLO" and r["seed"]==1 and r["method"] in ("P", "N3_DIM_SYM", "N3_THEN_SUBPIX")}
    for r in baseline:
        r["pose"] = historical[r["id"], r["method"]]
    scored.extend(baseline)
    E.write_rows(public/"D_REPAIRED_ROWS.jsonl.gz", scored, common)
    grouped = {name: [r for r in scored if r["method"]==name] for name in sorted({r["method"] for r in scored})}
    aggregates = {}
    for population, subset in [("ALLOWED173", ids), ("RECT245_INTERSECTION110", set(allowed["rect245_intersection_ids"]))]:
        aggregates[population] = {}
        for name, records in grouped.items():
            records = [r for r in records if r["id"] in subset]
            value = E.summary(records)
            value["ADDsym_AUC_full"] = pose.pose_auc([r["pose"]["ADDsym_normalized"] if r["pose"]["available"] else float("inf") for r in records], 1.0)
            aggregates[population][name] = value
    changes = []
    for suffix in ("", "_SUBPIX"):
        control = {r["id"]: r for r in grouped["D_CONTROL"+suffix]}
        joint = {r["id"]: r for r in grouped["D_JOINT_DIM"+suffix]}
        for fid in sorted(ids):
            a, b = control[fid], joint[fid]
            changes.append(dict(id=fid, session=a["session"], route="Stage1" if not suffix else "SubPix",
                                control_hypothesis=a["hypothesis"], joint_hypothesis=b["hypothesis"],
                                control_axis_wrong=a["axis_wrong"], joint_axis_wrong=b["axis_wrong"],
                                control_success=a["strict_success"], joint_success=b["strict_success"],
                                control_T_cm=a["pose"]["translation_cm"], joint_T_cm=b["pose"]["translation_cm"],
                                control_R_deg=a["pose"]["rotation_deg"], joint_R_deg=b["pose"]["rotation_deg"],
                                corner_delta_px=b["corner"]["frame_mean_px"]-a["corner"]["frame_mean_px"]))
    with (public/"D_ALL_PAIRED_TRANSITIONS.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0])); writer.writeheader(); writer.writerows(changes)
    sensitivity = D.read(public/"D_SENSITIVITY.json")
    scored_by = {(r["id"], r["method"]): r for r in scored}
    for row in sensitivity:
        a, b = scored_by[row["id"], "D_JOINT_DIM"], scored_by[row["id"], row["intervention"]]
        row.update(selection_switched=a["hypothesis"]!=b["hypothesis"],
                   strict_success_change=int(b["strict_success"])-int(a["strict_success"]),
                   T_change_cm=b["pose"]["translation_cm"]-a["pose"]["translation_cm"],
                   R_change_deg=b["pose"]["rotation_deg"]-a["pose"]["rotation_deg"])
    common.write(public/"D_REPAIRED_SENSITIVITY.json", sensitivity)
    common.write(public/"D_REPAIRED_RESULTS.json", dict(status="DONE_WITH_EXPLICIT_ADAPTER_RECOVERY", exploratory=True,
                 authoritative_rows="D_REPAIRED_ROWS.jsonl.gz", initial_results="D_RESULTS.json INVALID_K_LIST_ADAPTER",
                 initial_attempt_not_a_method_failure=True, frames=173, seed=1, aggregates=aggregates,
                 corner_damage=D.read(public/"D_RESULTS.json")["corner_damage"],
                 cumulative_cv_calls=first_calls+sum(calls.values()), checkpoints_refitted=False,
                 all_recovery_damage_transitions="D_ALL_PAIRED_TRANSITIONS.csv", source_generation_calls=1211,
                 same_split_within_D=True, cross_task_C_D_split_mismatch="D_CROSS_TASK_EXPOSURE.json; no C-vs-D independent synthetic holdout comparison"))
    print("D_REPAIR_COMPLETE", json.dumps(aggregates["ALLOWED173"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-root", "published-root", "private-dir", "public-dir"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    run(args)


if __name__=="__main__":
    main()
