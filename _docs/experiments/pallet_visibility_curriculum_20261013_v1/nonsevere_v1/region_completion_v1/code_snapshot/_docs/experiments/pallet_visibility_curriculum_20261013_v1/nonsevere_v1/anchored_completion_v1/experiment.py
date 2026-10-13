"""CAL-only strategy selection, then fixed reused TEST and upstream transfer."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import builtins
import io
import json
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.anchored_completion_api import AnchoredCompletion

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent
BASELINES = COHORT / "posefix_sub_comparison_v1"
GROUPS = {"CAL_N3": ("CAL", "N3"), "TEST_N3": ("TEST", "N3"), "TEST_PF": ("TEST", "PF")}


def options():
    result = {}
    for pool in ("interior_only", "all_visible"):
        for accuracy in (.12, .25):
            for rgb in (False, True):
                name = "PROTECTED_" + ("INTERIOR" if pool == "interior_only" else "VISIBLE")
                name += "_" + ("RGB" if rgb else "DETECTOR") + "_A" + str(round(accuracy*100))
                result[name] = dict(coordinate_source="FUSED", anchors=True, supplement=True,
                    supplemental_pool=pool, anchor_protection=True, supplemental_visibility=.65,
                    supplemental_accuracy=accuracy, anchor_max_projection_shift_px=2.,
                    anchor_max_huber_increase_px2=1., use_supplemental_observations=rgb,
                    allow_detector_supplements=True, final_guard_policy="anchor_priority")
    return result


def arms(group):
    if group == "CAL_N3":
        return options()
    chosen = C.read(HERE / "SELECTION_CAL.json")["rules"]
    return dict(SELECTED=chosen, UNPROTECTED=dict(chosen, anchor_protection=False),
        STRICT_POOL=dict(chosen, final_guard_policy="strict_eligible_pool"),
        NO_ANCHOR_ROLE=dict(chosen, anchors=False), BOUNDARY_RAW=dict(chosen, coordinate_source="RAW"))


@contextmanager
def no_labels():
    original = builtins.open, io.open, Path.open
    def wrap(fn):
        def guard(path, *args, **kwargs):
            if any(token in str(path) for token in ("SCORING", "SCORED_ROWS", "ORACLE", "TARGETS", "RESULTS.json", "/annotations/", "FINAL")):
                raise AssertionError("Annotation/outcome access during prediction: " + str(path))
            return fn(path, *args, **kwargs)
        return guard
    builtins.open, io.open, Path.open = map(wrap, original)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = original


class Cached:
    def __init__(self, observation):
        self.observation = observation

    def assess(self, *args):
        return deepcopy(self.observation)


def prepare(group):
    partition, family = GROUPS[group]
    target = HERE / group
    target.mkdir(exist_ok=True)
    if (target / "PROTOCOL.json").exists():
        raise FileExistsError("Preserve protocol")
    runtime = ROOT / "scripts/research/pallet_visibility_curriculum_20261013_v1"
    names = ["anchored_completion.py", "anchored_completion_api.py", "interior_observation.py",
        "boundary_rules.py", "boundary_evidence.py", "curriculum.py", "common.py", "joint_api.py",
        "branch_curriculum.py", "scoring.py", "boundary_refine.py"]
    files = [runtime / name for name in names]
    files += [ROOT / "scripts/research/pallet_foundation_mask_20261013_v1/uncertainty.py"]
    files += [Path(__file__), COHORT / "PREDICTION_INPUTS.json", COHORT / "COHORT_PROTOCOL.json",
        BASELINES / "PREDICTIONS.jsonl.gz", BASELINES / "PREDICTION_SEAL.json"]
    if group == "CAL_N3":
        files += [COHORT / "boundary_count_rules_v1" / name for name in ("OBSERVATIONS.jsonl.gz", "PREDICTION_SEAL.json")]
        files += [HERE / name for name in ("CAL_RGB_INTERIOR_OBSERVATIONS.jsonl.gz",
            "CAL_RGB_INTERIOR_PROTOCOL.json", "CAL_RGB_INTERIOR_PREDICTION_SEAL.json")]
        amendment = HERE / "CAL_RGB_INTERIOR_VALIDATION_AMENDMENT_V2.json"
        if amendment.exists():
            files.append(amendment)
    elif group == "TEST_N3":
        files += [COHORT / "boundary_progressive_replay_v1" / name for name in ("OBSERVATIONS.jsonl.gz", "PREDICTION_SEAL.json")]
    else:
        files += [COHORT / "boundary_rule_replay_v1" / name for name in ("OBSERVATIONS.jsonl.gz", "PREDICTION_SEAL.json")]
    if group != "CAL_N3":
        files += [HERE / "SELECTION_CAL.json"]
    frames = [f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"] if f["partition"] == partition]
    assert len(frames) == (46 if partition == "CAL" else 49)
    copied = ["INITIAL", "BOUNDARY_ONLY"] + (["OLD_PROGRESSIVE", "OLD_JOINT"] if family == "N3" else [])
    snapshot = target / "code_snapshot"
    for file in files:
        if file.suffix == ".py":
            dst = snapshot / file.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dst)
    C.write(target / "PROTOCOL.json", dict(schema="ANCHORED_COMPLETION_FROZEN_COMPARISON_V1", group=group,
        partition=partition, family=family, before_this_inference_and_scoring=True,
        frame_ids=[f["id"] for f in frames], source_sha256={str(f): C.sha(f) for f in files},
        rules=arms(group), copied_or_fixed_controls=copied,
        selection="CAL46 only: maximum whole-frame 5cm5deg success, then minimum mean translation, then mean rotation, then lexicographic option name. No photo-specific oracle or TEST choice.",
        CAL_options_rationale="Fixed 2x2x2 experiment: model interior/all-visible, detector/actual RGB junction, legacy .12 accuracy floor/.25 cutoff; observed score floor motivates ablation before coordinate/pose labels.",
        anchors="Actual two-line and RGB-supported independent observations, fixed native slots/initial H; per-point seed projection drift<=2px and Huber loss increase<=1px2, not exact image lock or correctness guarantee",
        missing="Hidden/self/truncated/unknown coordinates never fit; all eight final corners are one rigid model projection, separated from immutable image measurements",
        seed_exception="Anchor-priority can retain independently accepted boundary seed when weaker full-pool veto rejects expansions; strict-pool is explicit matched control, no claim of full-pool gain on seed exception",
        RGB="CAL fresh RGB junction extraction previously sealed without labels; TEST one new fixed-profile junction extraction per image; all Tiny/SAM inputs reused from their exact sealed studies",
        pose_metrics="Whole46/49 including fallback/phase-incompatible; proper physical symmetry and no GT stage/native correspondence selection",
        coordinate_metrics="Post-seal fixed locked native mapping; matched model-compatible frames only for coordinate errors, per human state plus direct outline/interior. No pixel clipping or GT permutation. Raw detector vs initial rigid vs final rigid completion separately.",
        uncertainty="Conditional weighted local information surrogate, not calibrated probability; shared edge dependence, semantic/model bias, K/dimension noise and fixed initial rotation error omitted",
        bootstrap="10000 draws by recording session, seed20261013; frame/corner dependence retained",
        previously_seen_TEST=partition == "TEST", independent_confirmation=False,
        new_training=0, severe_frames=0, FINAL_reads=0, default_model_promoted=False,
        stop_rule="These fixed CAL options and selected matched TEST controls once; preserve all outcomes; no TEST threshold/radius/anchor-budget/strategy tuning"))
    print("ANCHOR_PROTOCOL", group, C.sha(target / "PROTOCOL.json"), flush=True)


def infer(group):
    from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_rule_api import BoundaryRuleCorrector
    from scripts.research.pallet_visibility_curriculum_20261013_v1.interior_observation import infer_interior_observations
    partition, family = GROUPS[group]
    target = HERE / group
    protocol = C.read(target / "PROTOCOL.json")
    if (target / "PREDICTIONS.jsonl.gz").exists():
        raise FileExistsError("Preserve predictions")
    for path, digest in protocol["source_sha256"].items():
        assert C.sha(path) == digest, path
    cv2.setNumThreads(1)
    started = time.perf_counter()
    predictions, observations = [], []
    with no_labels():
        try:
            C.read(COHORT / "test_reused_v1/ISOLATED_TEST_SCORING.json")
        except AssertionError:
            pass
        else:
            raise AssertionError("GT canary failed")
        frames = [f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"] if f["partition"] == partition]
        assert [f["id"] for f in frames] == protocol["frame_ids"]
        baseline_seal = C.read(BASELINES / "PREDICTION_SEAL.json")
        assert baseline_seal["before_scoring"] and baseline_seal["predictions_sha256"] == C.sha(BASELINES / "PREDICTIONS.jsonl.gz")
        baseline = {(r["id"], r["method"]): r for r in C.rows(BASELINES / "PREDICTIONS.jsonl.gz") if r["partition"] == partition}
        if family == "PF":
            base_obs = {r["id"]: r["observation"] for r in C.rows(COHORT / "boundary_rule_replay_v1/OBSERVATIONS.jsonl.gz") if r["family"] == "PF_SUB"}
        else:
            path = COHORT / ("boundary_count_rules_v1" if partition == "CAL" else "boundary_progressive_replay_v1") / "OBSERVATIONS.jsonl.gz"
            base_obs = {r["id"]: r["observation"] for r in C.rows(path)}
        cal_rgb = {r["id"]: r["interior_observations"] for r in C.rows(HERE / "CAL_RGB_INTERIOR_OBSERVATIONS.jsonl.gz")} if partition == "CAL" else {}
        for index, frame in enumerate(frames):
            image_path = ROOT / frame["image"]
            assert C.sha(image_path) == frame["image_sha256"]
            image = cv2.imread(str(image_path))
            assert image is not None and list(image.shape[:2]) == frame["raw_hw"]
            initial = frame["initial_poses"]["N3_SUBPIX"] if family == "N3" else baseline[frame["id"], "POSEFIX_SUBPIX_PF_ANCHOR"]["actual_pose"]
            q = np.asarray(frame["points"]["N3_SUBPIX"] if family == "N3" else baseline[frame["id"], "POSEFIX_SUBPIX_PF_ANCHOR"]["native_points"], float)
            obs = deepcopy(base_obs[frame["id"]])
            np.testing.assert_array_equal(obs["raw_points"], q)
            obs["image_hw"] = frame["raw_hw"]
            obs["interior_observations"] = cal_rgb[frame["id"]] if partition == "CAL" else infer_interior_observations(image, q, frame["K"], initial, obs)
            observations.append(dict(id=frame["id"], family=family, observation=obs))
            for name, config in protocol["rules"].items():
                output = AnchoredCompletion(Cached(obs), config).refine(image, q, frame["K"], initial, frame["xyz"])
                if not output["solution"]["accepted"]:
                    assert output["actual_pose"] == initial
                if output["solution"]["estimated_dof"] == 3:
                    assert output["actual_pose"]["R_cf"] == initial["R_cf"] and output["actual_pose"]["R_physical"] == initial["R_physical"]
                predictions.append(dict(id=frame["id"], session=frame["session"], method=name,
                    actual_pose=output["actual_pose"], solution=output["solution"],
                    original_points=q.tolist(), measured_points=output["measured_points"],
                    completed_points=output["completed_points"], GT_inputs=False))
            fixed = BoundaryRuleCorrector(Cached(base_obs[frame["id"]]), dict(coordinate_source="FUSED")).refine(
                image, q, frame["K"], initial, frame["xyz"])
            controls = {"INITIAL": initial, "BOUNDARY_ONLY": fixed["actual_pose"]}
            if family == "N3":
                controls.update(OLD_PROGRESSIVE=baseline[frame["id"], "FUSED_ACTIVE"]["actual_pose"],
                                OLD_JOINT=baseline[frame["id"], "JOINT_SINGLE_FAMILY"]["actual_pose"])
            for name, pose in controls.items():
                p = C.project(C.cuboid(*pose["cf_extents"]), pose["R_cf"], pose["centroid"], frame["K"])
                control_solution = fixed["solution"] if name == "BOUNDARY_ONLY" else None
                if name.startswith("OLD_"):
                    baseline_name = "FUSED_ACTIVE" if name == "OLD_PROGRESSIVE" else "JOINT_SINGLE_FAMILY"
                    control_solution = baseline[frame["id"], baseline_name]["solution"]
                predictions.append(dict(id=frame["id"], session=frame["session"], method=name,
                    actual_pose=pose, solution=control_solution,
                    original_points=q.tolist(), completed_points=np.vstack([p, q[8:]]).tolist(),
                    source="sealed existing prediction" if name.startswith("OLD_") else "fixed control", GT_inputs=False))
            if (index+1) % 10 == 0:
                print("ANCHOR_INFER", group, index+1, round(time.perf_counter()-started, 2), flush=True)
    for path, digest in protocol["source_sha256"].items():
        assert C.sha(path) == digest, path
    expected = len(protocol["frame_ids"])*(len(protocol["rules"])+len(protocol["copied_or_fixed_controls"]))
    assert len(predictions) == expected
    C.save_rows(target / "PREDICTIONS.jsonl.gz", predictions)
    C.save_rows(target / "OBSERVATIONS.jsonl.gz", observations)
    C.write(target / "PREDICTION_SEAL.json", dict(before_scoring=True, complete=True, rows=expected,
        predictions_sha256=C.sha(target / "PREDICTIONS.jsonl.gz"), observations_sha256=C.sha(target / "OBSERVATIONS.jsonl.gz"),
        protocol_sha256=C.sha(target / "PROTOCOL.json"), GT_canary="PASS", fresh_Tiny_or_SAM_calls=0,
        new_RGB_junction_calls=0 if partition == "CAL" else len(frames), reused_CAL_RGB_calls=len(frames) if partition == "CAL" else 0,
        new_training=0, severe_frames=0, FINAL_reads=0, seconds=time.perf_counter()-started))
    print("ANCHOR_SEALED", group, expected, flush=True)


def dist(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
        median=float(np.median(x)) if len(x) else None, P90=float(np.quantile(x, .9)) if len(x) else None)


def compatible(initial, truth, mapping):
    Q = np.asarray(initial["R_cf"]).T @ np.asarray(initial["R_physical"])
    target_Q = np.asarray(truth["body_R"]).T @ np.asarray(truth["R"])
    own = C.cuboid(*initial["cf_extents"]) @ Q
    target = (C.cuboid(*truth["body_xyz"]) @ target_Q)[mapping]
    gaps = []
    for index in range(truth["order"]):
        angle = index*2*np.pi/truth["order"]
        c, s = np.cos(angle), np.sin(angle)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        gaps.append(np.linalg.norm(own @ Ry.T-target, axis=1).max())
    return min(gaps) < 1e-6


def score(group):
    from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric, paired
    target = HERE / group
    if (target / "RESULTS.json").exists():
        raise FileExistsError("Preserve results")
    protocol, seal = C.read(target / "PROTOCOL.json"), C.read(target / "PREDICTION_SEAL.json")
    assert seal["before_scoring"] and seal["predictions_sha256"] == C.sha(target / "PREDICTIONS.jsonl.gz")
    labels_path = COHORT / ("OOF_pose_v1/ISOLATED_CAL_SCORING.json" if group == "CAL_N3" else "test_reused_v1/ISOLATED_TEST_SCORING.json")
    truth = C.read(labels_path)["rows"]
    assert set(truth) == set(protocol["frame_ids"])
    frames = {f["id"]: f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"]}
    rows = list(C.rows(target / "PREDICTIONS.jsonl.gz"))
    obs = {r["id"]: r["observation"] for r in C.rows(target / "OBSERVATIONS.jsonl.gz")}
    groups, coords, consistency = defaultdict(list), defaultdict(list), []
    initial = {r["id"]: r for r in rows if r["method"] == "INITIAL"}
    phase = {}
    for fid in protocol["frame_ids"]:
        phase[fid] = compatible(initial[fid]["actual_pose"], truth[fid]["target"]["truth"],
                               np.asarray(truth[fid]["metadata"]["native_to_reference_corner_ids"], int))
    for row in rows:
        fid = row["id"]
        row["metric"] = pose_metric(row["actual_pose"], truth[fid]["target"]["truth"])
        groups[row["method"]].append(row)
        model = C.project(C.cuboid(*row["actual_pose"]["cf_extents"]), row["actual_pose"]["R_cf"],
                          row["actual_pose"]["centroid"], frames[fid]["K"])
        np.testing.assert_allclose(np.asarray(row["completed_points"])[:8], model, atol=1e-10, rtol=0)
        if row.get("solution") and "anchor_reference" in row["solution"]:
            sol = row["solution"]
            anchors = sol["anchor_ids"]
            measured = np.asarray(sol["measured_points"])
            consistency.append(dict(id=fid, method=row["method"], anchor_count=len(anchors),
                used_ids=sol["used_ids"], accepted=sol["accepted"], estimated_dof=sol["estimated_dof"],
                completion_consistent_with_one_rigid_pose=True,
                exact_output_anchor_clamp_difference_px=dist(np.linalg.norm(model[anchors]-measured[anchors], axis=1)),
                actual_anchor_guard=sol["final_anchor_guard"], final_guard=sol["final_guard"],
                uncertainty=sol["uncertainty"]))
        if not phase[fid] or row["method"] == "OLD_JOINT":
            continue
        mapping = np.asarray(truth[fid]["metadata"]["native_to_reference_corner_ids"], int)
        gt = np.asarray(truth[fid]["target"]["gt"], float)[:8][mapping]
        raw = np.asarray(row["original_points"], float)[:8]
        initial_pose = initial[fid]["actual_pose"]
        initial_model = C.project(C.cuboid(*initial_pose["cf_extents"]), initial_pose["R_cf"],
                                  initial_pose["centroid"], frames[fid]["K"])
        human = truth[fid]["metadata"]["human_states_native"]
        outline = np.asarray(obs[fid]["boundary_visibility"]["expected_outline"], bool)
        for k in range(8):
            if not np.isfinite(gt[k]).all() or np.all(gt[k] == -1):
                continue
            category = ("DIRECT_OUTLINE" if outline[k] else "DIRECT_MODEL_INTERIOR") if human[k] == "DIRECT_VISIBLE" else human[k]
            raw_error = float(np.linalg.norm(raw[k]-gt[k])) if np.isfinite(raw[k]).all() and not np.all(raw[k] == -1) else None
            coords[row["method"], category].append(dict(id=fid, session=row["session"], corner=k,
                model_error_px=float(np.linalg.norm(model[k]-gt[k])), raw_prediction_error_px=raw_error,
                initial_rigid_error_px=float(np.linalg.norm(initial_model[k]-gt[k])),
                human_state=human[k], initial_outline=bool(outline[k])))
    summary = {}
    for name, group_rows in groups.items():
        solutions = [r["solution"] for r in group_rows if r.get("solution")]
        summary[name] = dict(frames=len(group_rows), success=sum(r["metric"]["success_5cm_5deg"] for r in group_rows),
            accepted=sum(s["accepted"] for s in solutions),
            dof=dict(Counter(str(s.get("estimated_dof")) for s in solutions)),
            metrics={k: dist([r["metric"][k] for r in group_rows]) for k in ("translation_cm", "rotation_deg", "ADDsym_cm")})
        print("ANCHOR_SCORE", group, name, summary[name]["success"],
            round(summary[name]["metrics"]["translation_cm"]["mean"], 4),
            round(summary[name]["metrics"]["rotation_deg"]["mean"], 4), flush=True)
    contrasts = {}
    for name, values in groups.items():
        if name != "INITIAL":
            contrasts[name+"-INITIAL"] = paired(values, initial, repeats=10000)
    if group != "CAL_N3":
        selected = {r["id"]: r for r in groups["SELECTED"]}
        for name in ("UNPROTECTED", "STRICT_POOL", "NO_ANCHOR_ROLE", "BOUNDARY_RAW", "BOUNDARY_ONLY", "OLD_PROGRESSIVE", "OLD_JOINT"):
            if name in groups:
                contrasts[name+"-SELECTED"] = paired(groups[name], selected, repeats=10000)
    coordinate_summary = {name+"/"+category: dict(corners=len(values), frames=len({r["id"] for r in values}),
        model_error_px=dist([r["model_error_px"] for r in values]),
        initial_rigid_error_px=dist([r["initial_rigid_error_px"] for r in values]),
        raw_prediction_error_px=dist([r["raw_prediction_error_px"] for r in values if r["raw_prediction_error_px"] is not None]),
        paired_raw_valid_model_error_px=dist([r["model_error_px"] for r in values if r["raw_prediction_error_px"] is not None]))
        for (name, category), values in coords.items()}
    C.save_rows(target / "SCORED_ROWS.jsonl.gz", rows)
    C.save_rows(target / "COORDINATE_ROWS.jsonl.gz", [dict(method=n, category=s, **v) for (n, s), values in coords.items() for v in values])
    C.save_rows(target / "CONSISTENCY_DIAGNOSTICS.jsonl.gz", consistency)
    C.write(target / "RESULTS.json", dict(summary=summary, paired=contrasts, coordinate_summary=coordinate_summary,
        coordinate_phase_compatible_frames=sum(phase.values()), coordinate_phase_incompatible_ids=[fid for fid, ok in phase.items() if not ok],
        labels_sha256=C.sha(labels_path), predictions_sha256=seal["predictions_sha256"],
        pose_primary_uses_all_frames=True, pixel_metrics_fixed_correspondence=True,
        TEST_already_observed=group != "CAL_N3", independent_confirmation=False,
        severe_frames=0, FINAL_reads=0, selected_checkpoint_or_correspondence_from_GT=False,
        limits=["Pose and hidden image reference comes from historical annotation reconstruction, not independently measured physical pose",
               "Coordinate diagnostics exclude known model/reference phase incompatibility without remapping predictions",
               "Conditional information surrogate is not calibrated semantic or coordinate confidence",
               "CAL strategy selection is in-sample; reused TEST is exploratory, not unseen generalization"] ))


def select():
    path = HERE / "SELECTION_CAL.json"
    if path.exists():
        raise FileExistsError("Preserve CAL selection")
    results = C.read(HERE / "CAL_N3/RESULTS.json")
    choices = options()
    ranking = sorted(choices, key=lambda name: (-results["summary"][name]["success"],
        results["summary"][name]["metrics"]["translation_cm"]["mean"],
        results["summary"][name]["metrics"]["rotation_deg"]["mean"], name))
    C.write(path, dict(selected=ranking[0], rules=choices[ranking[0]], ranked_options=ranking,
        criterion="Maximum CAL46 whole-frame5cm5deg success then minimum meanT then meanR then lexicographic name",
        source_results_sha256=C.sha(HERE / "CAL_N3/RESULTS.json"),
        source_prediction_seal_sha256=C.sha(HERE / "CAL_N3/PREDICTION_SEAL.json"),
        selection_before_new_TEST_predictions=True, no_TEST_metric_or_annotation_reads=True,
        further_TEST_tuning=False, automatic_default_promotion=False))
    print("ANCHOR_CAL_SELECTED", ranking[0], flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "infer", "score", "select"))
    parser.add_argument("--group", choices=GROUPS, default="CAL_N3")
    args = parser.parse_args()
    if args.phase == "select":
        select()
    else:
        {"prepare": prepare, "infer": infer, "score": score}[args.phase](args.group)
