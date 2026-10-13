"""Five declared CAL-only pose arms using frozen source local RGB evidence.

Source scores are read from their prediction-only sealed cache. The fixed
upstream original N3 measurements and source-segmented RGB appearance are
reused; no projected location becomes an observation. Pose truth is read only
after all arm predictions have been sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from .curriculum import solve_curriculum, CurriculumConfig
from .local_occlusion_cal import binding, no_scoring_reads, confusion
from .scoring import pose_metric, distribution, paired


ARMS = {
    "SOURCE_T0P2": .2,
    "SOURCE_T0P35": .35,
    "SOURCE_T0P5": .5,
    "SOURCE_T0P65": .65,
    "RGB_SOURCE_VETO_T0P2": None,
}
REFERENCE = "SEGMENTED_RGB_CURRICULUM"


def local_appearance(frozen, probability, threshold):
    app = deepcopy(frozen)
    for k, state in enumerate(app["states"]):
        if state in ("SELF_OCCLUDED", "OUT_OF_FRAME"):
            continue
        p = float(probability[k])
        if threshold is None:
            if state in ("VISIBLE_BOUNDARY", "VISIBLE_INTERIOR") and p < .2:
                app["states"][k] = "UNKNOWN"
                app["visibility_score"][k] = 0.
            continue
        app["visibility_score"][k] = p
        if p >= threshold:
            app["states"][k] = "VISIBLE_BOUNDARY" if app["boundary_score"][k] >= .35 else "VISIBLE_INTERIOR"
            app["local_accuracy_score"][k] = max(.12, app["local_accuracy_score"][k])
        else:
            app["states"][k] = "UNKNOWN"
    app["diagnostics"]["source_local_support_pilot"] = dict(
        threshold=threshold, local_support_scores=[float(x) for x in probability],
        veto_threshold=.2 if threshold is None else None,
        target="local source body support proxy under synthetic opacity",
        physical3D_visibility_certified=False, RGB_input_only=True,
        entropy_not_used_as_visibility_certificate=True)
    return app


def infer(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    frames = C.read(args.inputs)["frames"]
    if len(frames) != 76 or any(row["partition"] != "CAL" for row in frames):
        raise AssertionError("CAL76 only")
    upstream_seal = C.read(args.upstream / "PREDICTION_SEAL.json")
    if not upstream_seal["before_scoring"] or C.sha(args.upstream / "PREDICTIONS.jsonl.gz") != upstream_seal["sha256"]:
        raise AssertionError("Primary v1 prediction seal differs")
    feature_seal = C.read(args.source_cache / "FEATURES_SEAL.json")
    if not feature_seal["complete"] or binding(args.source_cache / "PREDICTION_FEATURES.jsonl.gz") != feature_seal["feature_cache"]:
        raise AssertionError("Source prediction-only cache seal differs")
    upstream = {row["id"]: row for row in C.rows(args.upstream / "PREDICTIONS.jsonl.gz") if row["method"] == REFERENCE}
    cached = {row["id"]: row for row in C.rows(args.source_cache / "PREDICTION_FEATURES.jsonl.gz")}
    if set(upstream) != {row["id"] for row in frames} or set(cached) != set(upstream):
        raise AssertionError("Exact all-frame upstream/cache IDs required")
    dependencies = [Path(__file__), args.inputs, args.upstream / "PREDICTIONS.jsonl.gz",
                    args.source_cache / "PREDICTION_FEATURES.jsonl.gz",
                    Path(__file__).with_name("curriculum.py"), Path(__file__).with_name("common.py")]
    protocol = dict(schema="frozen_source_support_five_pose_arms_CAL_v1", frames=76, partition="CAL",
        arms=ARMS, primary=dict(upstream=str(args.upstream), method=REFERENCE,
                              expected_translation_cm=116.29147334738165,
                              expected_rotation_deg=19.290009317834865, expected_success_count=37),
        arm_definitions="Four raw-source thresholds .2,.35,.5,.65; one unchanged RGB appearance with p<.2 veto to UNKNOWN and score0",
        original_q_unchanged=True, boundary_coordinate_corrections=False, source_projected_probe_as_fit_point=False,
        frozen_H_OOB_from_primary=True, initial_upstream_family_preserved=True,
        boundary_scores="frozen v1 source-segmented appearance", visible_accuracy_floor=.12,
        source_arm_curriculum="visible_threshold=tau, easy_visibility=max(.45,tau), every other CurriculumConfig default unchanged",
        veto_curriculum="primary v1 visible_threshold=.35, every other default unchanged; existing UNKNOWN stage remains enabled",
        declared_before_new_pose_scoring=True, labels_opened_only_postseal=True,
        limits=["Source head certifies local RGB support proxy, not physical3D corner observability.",
                "Source domain is graphite; real material and natural occluder domain gap persists.",
                "Unknown stage can later admit low-weight independent measurements; veto is not a final hard exclusion.",
                "Prior CAL/root TEST summaries motivated this development pilot; future reused TEST remains non-independent."],
        selection="Finite CAL comparison of five declared arms; no automatic deployment or TEST replay",
        new_TEST_reads=0, FINAL_reads=0, source_head=feature_seal["source_head"],
        bindings=[binding(path) for path in dependencies])
    C.write(out / "PROTOCOL.json", protocol)
    records = []
    begin = time.monotonic()
    with no_scoring_reads():
        for index, frame in enumerate(frames):
            base, source = upstream[frame["id"]], cached[frame["id"]]
            q = np.asarray(frame["points"]["N3_SUBPIX"], float)[:8]
            initial = frame["initial_poses"]["N3_SUBPIX"]
            xyz = C.cuboid(*initial["cf_extents"])
            K = np.asarray(frame["K"]); R = np.asarray(initial["R_cf"]); t = np.asarray(initial["centroid"])
            H = C.self_hidden(xyz, R, t)
            if not np.array_equal(q, np.asarray(base["original_points"])) or not np.array_equal(q, np.asarray(source["frozen_q"])):
                raise AssertionError("Frozen original q differs")
            if not np.array_equal(H, base["self_hidden"]) or not np.array_equal(H, source["frozen_self_hidden"]):
                raise AssertionError("Fixed primary/source H differs")
            probability = source["frozen_source_query"]["local_support_score"]
            # Primary reference is retained verbatim rather than recomputed.
            reference = deepcopy(base); reference["method"] = "PRIMARY_V1"
            records.append(reference)
            for name, threshold in ARMS.items():
                app = local_appearance(base["appearance"], probability, threshold)
                config = dict(visible_threshold=.35) if threshold is None else dict(
                    visible_threshold=threshold, easy_visibility=max(.45, threshold))
                solution = solve_curriculum(q, xyz, K, R, t, H, app, config)
                Q = R.T @ np.asarray(initial["R_physical"])
                actual_pose = dict(available=True, R_cf=solution["R"],
                    R_physical=(np.asarray(solution["R"]) @ Q).tolist(), centroid=solution["t"],
                    cf_extents=initial["cf_extents"], selected_hypothesis=initial["selected_hypothesis"])
                projected = C.project(xyz, solution["R"], solution["t"], K)
                native = np.asarray(frame["points"]["N3_SUBPIX"], float).copy()
                if solution["accepted"]:
                    native[np.flatnonzero(H)] = projected[H]
                for stage in solution.get("stages", []):
                    if stage.get("name") != "initial" and any(H[k] for k in stage["used_ids"]):
                        raise AssertionError("Known self-hidden point became a fit observation")
                records.append(dict(id=frame["id"], session=frame["session"], arm="N3_SUBPIX", method=name,
                    actual_pose=actual_pose, solution=solution, self_hidden=H.tolist(),
                    appearance=app, original_points=q.tolist(), native_points=native.tolist(),
                    projected_points=projected.tolist(), correction=None, source_head_frozen=True,
                    fixed_source_support_probability=probability, projected_probe_used_as_observation=False))
            if (index + 1) % 16 == 0:
                print("SOURCE_SUPPORT_POSES", index + 1, round(time.monotonic() - begin, 2), flush=True)
    if [binding(path) for path in dependencies] != protocol["bindings"]:
        raise AssertionError("Frozen pose inference dependency changed")
    C.save_rows(out / "PREDICTIONS.jsonl.gz", records)
    C.write(out / "PREDICTION_SEAL.json", dict(complete=True, before_scoring=True, rows=len(records),
        sha256=C.sha(out / "PREDICTIONS.jsonl.gz"), methods=dict(Counter(row["method"] for row in records)),
        labels_read_during_inference=0, new_TEST_reads=0, FINAL_reads=0,
        projected_points_used_as_observations=False, seconds=time.monotonic() - begin))


def state_metrics(rows, metadata):
    y, p, predicted, eligible = [], [], [], []
    for row in rows:
        states = metadata[row["id"]]["scoring_metadata"]["human_states_native"]
        q = np.asarray(row["original_points"])
        h, w = metadata[row["id"]]["raw_hw"]
        for k, state in enumerate(states):
            if state not in ("DIRECT_VISIBLE", "EXTERNAL_OCCLUDED"):
                continue
            if row["self_hidden"][k] or not (0 <= q[k, 0] < w and 0 <= q[k, 1] < h):
                continue
            y.append(int(state == "DIRECT_VISIBLE")); p.append(row["appearance"]["visibility_score"][k])
            predicted.append(int(row["appearance"]["states"][k] in ("VISIBLE_BOUNDARY", "VISIBLE_INTERIOR")))
            eligible.append([row["id"], k])
    y, p, predicted = np.asarray(y), np.asarray(p), np.asarray(predicted)
    from sklearn.metrics import roc_auc_score
    return dict(eligible_points=len(y), AUC=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                confusion=confusion(y, predicted, .5), keys=eligible)


def pose_summary(rows):
    return dict(frames=len(rows), accepted=sum(row["solution"]["accepted"] for row in rows),
        success_count=sum(row["metric"]["success_5cm_5deg"] for row in rows),
        reasons=dict(Counter(row["solution"]["reason"] for row in rows)),
        metrics={name: distribution([row["metric"][name] for row in rows])
                 for name in ("translation_cm", "rotation_deg", "ADDsym_cm")},
        visibility_availability=dict(
            frames_with4_state_visible=sum(sum(s in ("VISIBLE_BOUNDARY", "VISIBLE_INTERIOR") for s in row["appearance"]["states"]) >= 4 for row in rows),
            frames_with4_final_fit_points=sum(len(row["solution"].get("used_ids", [])) >= 4 for row in rows),
            final_fit_counts=dict(Counter(len(row["solution"].get("used_ids", [])) for row in rows)),
            state_visible_counts=dict(Counter(sum(s in ("VISIBLE_BOUNDARY", "VISIBLE_INTERIOR") for s in row["appearance"]["states"]) for row in rows))))


def score(args):
    out = Path(args.output)
    if (out / "RESULTS.json").exists():
        raise FileExistsError("Preserve completed pose experiment")
    seal = C.read(out / "PREDICTION_SEAL.json")
    if not seal["before_scoring"] or C.sha(out / "PREDICTIONS.jsonl.gz") != seal["sha256"]:
        raise AssertionError("Pose predictions must be sealed")
    labels = C.read(args.scoring)["frames"]
    if len(labels) != 76 or any(row["partition"] != "CAL" for row in labels):
        raise AssertionError("Isolated CAL76 pose truth only")
    metadata = {row["id"]: row for row in labels}
    rows = list(C.rows(out / "PREDICTIONS.jsonl.gz"))
    groups = {}
    for row in rows:
        row["metric"] = pose_metric(row["actual_pose"], metadata[row["id"]]["target"]["truth"])
        groups.setdefault(row["method"], []).append(row)
    if any(len(group) != 76 for group in groups.values()):
        raise AssertionError("All76 frames required for every arm")
    baseline = pose_summary(groups["PRIMARY_V1"])
    if (abs(baseline["metrics"]["translation_cm"]["mean"] - 116.29147334738165) > 1e-7
            or abs(baseline["metrics"]["rotation_deg"]["mean"] - 19.290009317834865) > 1e-7
            or baseline["success_count"] != 37):
        raise AssertionError("Primary v1 reference mismatch")
    sessions = sorted({row["session"] for row in rows})
    summaries, per_session, states, comparisons = {}, {}, {}, {}
    reference = {row["id"]: row for row in groups["PRIMARY_V1"]}
    for name, group in groups.items():
        summaries[name] = pose_summary(group)
        states[name] = state_metrics(group, metadata)
        per_session[name] = {session: dict(pose=pose_summary([row for row in group if row["session"] == session]),
                                          visibility=state_metrics([row for row in group if row["session"] == session], metadata))
                             for session in sessions}
        if name != "PRIMARY_V1":
            comparisons[name] = paired(group, reference)
    result = dict(complete=True, schema="frozen_source_local_support_pose_CAL_results_v1", frames=76,
        sessions=sessions, arms=ARMS, summary=summaries, per_session=per_session,
        visibility=states, paired_against_primary=comparisons, labels_source=binding(args.scoring),
        predictions=binding(out / "PREDICTIONS.jsonl.gz"), protocol=binding(out / "PROTOCOL.json"),
        new_TEST_reads=0, FINAL_reads=0, integration_performed=False,
        source_head_physically_certified=False, projected_probe_used_as_point_observation=False,
        selection_note="Finite CAL development comparison; no new head training, no TEST inference. Existing unknown stages can admit independently measured points at low weight.")
    C.save_rows(out / "SCORED_ROWS.jsonl.gz", rows)
    C.write(out / "RESULTS.json", result)
    for name, item in summaries.items():
        v = states[name]["confusion"]
        print(name, "T", round(item["metrics"]["translation_cm"]["mean"], 4),
              "R", round(item["metrics"]["rotation_deg"]["mean"], 4), "success", item["success_count"],
              "accepted", item["accepted"], "directRecall", round(v["direct_recall"], 4),
              "externalRecall", round(v["external_recall"], 4), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer", "score"))
    parser.add_argument("--inputs", type=Path, default=C.SCRATCH / "CAL_INPUTS.json")
    parser.add_argument("--upstream", type=Path, default=C.SCRATCH / "cal_segmentation_v2")
    parser.add_argument("--source-cache", type=Path, default=C.SCRATCH / "local_occlusion_real_cal_v1")
    parser.add_argument("--scoring", type=Path, default=C.SCRATCH / "ORACLE_CAL_INPUTS.json")
    parser.add_argument("--output", type=Path, default=C.SCRATCH / "local_support_pose_v1")
    args = parser.parse_args()
    (infer if args.stage == "infer" else score)(args)


if __name__ == "__main__":
    main()
