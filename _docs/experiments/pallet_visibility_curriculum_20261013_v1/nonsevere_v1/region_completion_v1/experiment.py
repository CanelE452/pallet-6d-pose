"""Frozen finite-pose comparison with CAL-only empirical coarse regions."""
from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.empirical_regions import prediction_scale, region_queries, build_empirical_regions
from scripts.research.pallet_visibility_curriculum_20261013_v1.protected_completion_api import pose_proposal
from scripts.research.pallet_visibility_curriculum_20261013_v1.protected_proposals import select_protected_pose, state_keypoint_readout
from scripts.research.pallet_visibility_curriculum_20261013_v1.region_completion import select_region_pose
from scripts.research.pallet_visibility_curriculum_20261013_v1.keypoint_regions import image_pixel_sha256

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent
ANCHOR = COHORT / "anchored_completion_v1"
BASELINES = COHORT / "posefix_sub_comparison_v1"
spec = importlib.util.spec_from_file_location("anchor_study", ANCHOR / "experiment.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
GROUPS = ("CAL_N3", "TEST_N3", "TEST_PF")


def prepare():
    if (HERE / "PROTOCOL.json").exists():
        raise FileExistsError("Preserve frozen study")
    runtime = ROOT / "scripts/research/pallet_visibility_curriculum_20261013_v1"
    sources = [Path(__file__), ANCHOR / "experiment.py", COHORT / "PREDICTION_INPUTS.json",
        COHORT / "COHORT_PROTOCOL.json", COHORT / "BOUNDARY_PROGRESSIVE_CONFIG.json",
        BASELINES / "PREDICTIONS.jsonl.gz", BASELINES / "PREDICTION_SEAL.json", ANCHOR / "SELECTION_CAL.json"]
    sources += list(runtime.glob("*.py"))
    for group in GROUPS:
        sources += [ANCHOR / group / name for name in ("PROTOCOL.json", "PREDICTIONS.jsonl.gz", "PREDICTION_SEAL.json", "OBSERVATIONS.jsonl.gz")]
    for file in sources:
        if file.suffix == ".py":
            target = HERE / "code_snapshot" / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    C.write(HERE / "PROTOCOL.json", dict(schema="STATEWISE_REGION_FINITE_COMPARISON_V1",
        before_radius_fitting_and_this_inference=True,
        source_sha256={str(p): C.sha(p) for p in sources},
        groups=list(GROUPS), frames={"CAL_N3": 46, "TEST_N3": 49, "TEST_PF": 49},
        methods=["INITIAL", "BASE", "PROPOSALS", "FINITE_VISIBLE", "FINITE_REGION"],
        region_model=dict(CAL_only=True, target_empirical_point_quantile=.90,
            categories=["INITIAL_MODEL_SELF", "OTHER_UNOBSERVED"], minimum_category_samples=20,
            normalization="finite raw native eight-point bounding diagonal, minimum16px",
            quantile="ceil((n+1)*.90)-th order statistic capped at n; positive floor1e-6",
            calibrated_joint_coverage=False, independent_CAL_calibration=False),
        region_policy=dict(weight=.10, precise_correspondence_count_added=0, hard_pruning=False,
            radius_source="raw image-regressed predictions and CAL error tolerance, never pose covariance",
            exclude_precise_pool=True, statistically_independent_from_initial_PnP=False),
        finite_pool="BASE + same-image legacy progressive and frozen joint poses (N3 only for joint) + at most4 P3P roots of one max-spread precise-anchor triple",
        paired_region_control="FINITE_VISIBLE and FINITE_REGION share identical candidate pool, anchor, geometry, depth, trust and visible objective tolerance; only region_weight0 versus.10 differs",
        P3P_policy="Three actual image anchors may estimate full6DOF via finite P3P roots; distinct opt-in policy from frozenV1 exact-R three-point translation. Need rank6 and originalcondition<=15; no hidden point correspondence",
        guards="Same fixed accepted anchor reference2px and per-anchorHuber increase1px2; when no accepted reference exists, explicitly freeze initialpose as2px/1px2 reference. Visible full-pool objective must be <=current+.10 and<=initial+.10; combined score must strictly improve; originaltrust bounds unchanged. Tolerance guard is separately recorded from originalstrict initialgain",
        proposal_policy="PROPOSALS retainsV1 support/DOF and requires strict dual initial/current objective gains. FINITE controls have identical tolerance policy and includeP3P",
        output="Mixed reported2D points preserve real anchors and used real measurements; separate one-pose rigid corners; neither output is recycled toPnP",
        previously_seen_CAL_and_TEST=True, independent_confirmation=False, selection_from_TEST=False,
        global_arm_selection="None: all fixed arms reported once; optional region configuration fixed before this inference, no outcome winner promotion",
        stop_rule="Fit one empirical model, run these five arms once; no TEST radius/weight/threshold/native permutation tuning",
        new_training_severe_frames=0, evaluation_severe_frames=0, FINAL_reads=0))
    print("REGION_PROTOCOL", C.sha(HERE / "PROTOCOL.json"), flush=True)


def verify():
    protocol = C.read(HERE / "PROTOCOL.json")
    for path, digest in protocol["source_sha256"].items():
        assert C.sha(path) == digest, path
    return protocol


def anchor_rows(group):
    path = ANCHOR / group
    seal = C.read(path / "PREDICTION_SEAL.json")
    assert seal["before_scoring"] and seal["predictions_sha256"] == C.sha(path / "PREDICTIONS.jsonl.gz")
    method = C.read(ANCHOR / "SELECTION_CAL.json")["selected"] if group == "CAL_N3" else "SELECTED"
    return {r["id"]: r for r in C.rows(path / "PREDICTIONS.jsonl.gz") if r["method"] == method}


def fit():
    protocol = verify()
    if (HERE / "REGION_MODEL.json").exists():
        raise FileExistsError("Preserve empirical CAL model")
    labels_path = COHORT / "OOF_pose_v1/ISOLATED_CAL_SCORING.json"
    truth = C.read(labels_path)["rows"]
    frames = {f["id"]: f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"] if f["partition"] == "CAL"}
    base = anchor_rows("CAL_N3")
    errors, used, incompatible = defaultdict(list), [], []
    for fid, row in base.items():
        frame, target = frames[fid], truth[fid]
        mapping = np.asarray(target["metadata"]["native_to_reference_corner_ids"], int)
        initial = frame["initial_poses"]["N3_SUBPIX"]
        if not previous.compatible(initial, target["target"]["truth"], mapping):
            incompatible.append(fid)
            continue
        gt = np.asarray(target["target"]["gt"], float)[:8][mapping]
        q = np.asarray(row["solution"]["raw_points"], float)[:8]
        scale = prediction_scale(q)
        for k, category in region_queries(row["solution"]).items():
            if not np.isfinite(gt[k]).all() or np.all(gt[k] == -1):
                continue
            error = float(np.linalg.norm(q[k]-gt[k])/scale)
            errors[category].append(error)
            used.append(dict(id=fid, session=frame["session"], corner=k, category=category,
                             normalized_error=error, human_state=target["metadata"]["human_states_native"][k]))
    pooled = [v for values in errors.values() for v in values]
    def radius(values):
        index = min(len(values)-1, int(np.ceil((len(values)+1)*.90))-1)
        return max(float(np.sort(values)[index]), 1e-6)
    model = dict(schema="empirical_CAL_coarse_keypoint_regions_v1", categories={},
        pooled_normalized_radius=radius(pooled), pooled_samples=len(pooled),
        calibrated_joint_coverage=False, calibrated_probability=False,
        radius_rule=protocol["region_model"], labels_sha256=C.sha(labels_path),
        CAL_prediction_seal_sha256=C.sha(ANCHOR / "CAL_N3/PREDICTION_SEAL.json"),
        model_reference_incompatible_CAL_ids=incompatible, new_severe_frames=0,
        limitations=["CAL already used for upstream heads and method selection; empirical in-sample tolerance",
            "Frames and corners within recording sessions are correlated; no exchangeability or joint coverage guarantee",
            "Registered virtual corner and reconstructed pose reference can have native phase/semantic bias",
            "Predictions may already contribute to initialPnP; not an independent likelihood"])
    for category in protocol["region_model"]["categories"]:
        values = errors[category]
        model["categories"][category] = dict(samples=len(values), sessions=len({r["session"] for r in used if r["category"] == category}),
            normalized_radius=radius(values) if len(values) >= 20 else model["pooled_normalized_radius"],
            radius_source="category_CAL_quantile" if len(values) >= 20 else "pooled_CAL_fallback")
    C.write(HERE / "REGION_MODEL.json", model)
    C.save_rows(HERE / "CAL_RADIUS_FIT_ROWS.jsonl.gz", used)
    print("REGION_CAL_FIT", model["categories"], flush=True)


def infer(group):
    from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_curriculum_api import BoundaryCurriculum
    protocol = verify()
    target = HERE / group
    target.mkdir(exist_ok=True)
    if (target / "PREDICTIONS.jsonl.gz").exists():
        raise FileExistsError("Preserve region predictions")
    started = time.perf_counter()
    model_sha = C.sha(HERE / "REGION_MODEL.json")
    cv2.setNumThreads(1)
    rows = []
    with previous.no_labels():
        model = C.read(HERE / "REGION_MODEL.json")
        base = anchor_rows(group)
        frames = {f["id"]: f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"]}
        baseline = {(r["id"], r["method"]): r for r in C.rows(BASELINES / "PREDICTIONS.jsonl.gz")}
        observations = {r["id"]: r["observation"] for r in C.rows(ANCHOR / group / "OBSERVATIONS.jsonl.gz")}
        progressive = C.read(COHORT / "BOUNDARY_PROGRESSIVE_CONFIG.json")["curriculum"]
        for index, (fid, record) in enumerate(base.items()):
            frame = frames[fid]
            image_path = ROOT / frame["image"]
            assert C.sha(image_path) == frame["image_sha256"]
            image = cv2.imread(str(image_path))
            assert image is not None
            initial = frame["initial_poses"]["N3_SUBPIX"] if group != "TEST_PF" else baseline[fid, "POSEFIX_SUBPIX_PF_ANCHOR"]["actual_pose"]
            xyz, K = C.cuboid(*initial["cf_extents"]), frame["K"]
            solution = record["solution"]
            q = solution["raw_points"]
            proposals = []
            if group != "TEST_PF":
                for name, source in (("LEGACY_PROGRESSIVE", "FUSED_ACTIVE"), ("LEARNED_JOINT", "JOINT_SINGLE_FAMILY")):
                    proposals.append(pose_proposal(name, baseline[fid, source]["actual_pose"], initial))
            else:
                pose = BoundaryCurriculum(previous.Cached(observations[fid]), progressive).refine(image, q, K, initial, frame["xyz"])["actual_pose"]
                proposals.append(pose_proposal("LEGACY_PROGRESSIVE", pose, initial))
            regions = build_empirical_regions(solution, image, K, xyz, model, model_sha256=model_sha)
            pixel_sha = image_pixel_sha256(image)
            outputs = dict(BASE=deepcopy(solution),
                PROPOSALS=select_protected_pose(solution, xyz, K, initial["R_cf"], initial["centroid"], proposals),
                FINITE_VISIBLE=select_region_pose(solution, xyz, K, initial["R_cf"], initial["centroid"], regions, proposals, {"region_weight": 0.}, image_sha256=pixel_sha),
                FINITE_REGION=select_region_pose(solution, xyz, K, initial["R_cf"], initial["centroid"], regions, proposals, {"region_weight": .10}, image_sha256=pixel_sha))
            for name, output in outputs.items():
                readout = state_keypoint_readout(output)
                R, t = np.asarray(output["R"]), output["t"]
                R0 = np.asarray(initial["R_cf"])
                pose = dict(initial, R_cf=output["R"], centroid=t,
                            R_physical=(R @ (R0.T @ np.asarray(initial["R_physical"]))).tolist())
                if np.array_equal(R, R0):
                    pose["R_cf"], pose["R_physical"] = deepcopy(initial["R_cf"]), deepcopy(initial["R_physical"])
                if not output["accepted"]:
                    pose = deepcopy(initial)
                rows.append(dict(id=fid, session=frame["session"], method=name, actual_pose=pose,
                    solution=output, keypoint_readout=readout, coarse_regions=regions if name.startswith("FINITE_") else [],
                    initial_pose=initial, original_points=q, GT_inputs=False))
            projected = C.project(xyz, initial["R_cf"], initial["centroid"], K)
            rows.append(dict(id=fid, session=frame["session"], method="INITIAL", actual_pose=initial,
                solution=None, keypoint_readout=dict(reported_points=np.vstack([projected, np.asarray(q)[8:]]).tolist(),
                    rigid_model_corners=projected.tolist(), point_sources=["INITIAL_RIGID_MODEL"]*8),
                coarse_regions=[], initial_pose=initial, original_points=q, GT_inputs=False))
            if (index+1) % 10 == 0:
                print("REGION_INFER", group, index+1, round(time.perf_counter()-started, 2), flush=True)
    verify()
    assert C.sha(HERE / "REGION_MODEL.json") == model_sha
    assert len(rows) == protocol["frames"][group]*5
    C.save_rows(target / "PREDICTIONS.jsonl.gz", rows)
    C.write(target / "PREDICTION_SEAL.json", dict(before_scoring=True, rows=len(rows),
        predictions_sha256=C.sha(target / "PREDICTIONS.jsonl.gz"), protocol_sha256=C.sha(HERE / "PROTOCOL.json"),
        region_model_sha256=model_sha, source_bindings_verified_before_and_after=True,
        fresh_Tiny_or_SAM_calls=0, fresh_joint_calls=0, fresh_junction_calls=0,
        pure_PF_progressive_calls=49 if group == "TEST_PF" else 0,
        new_neural_training=0, CAL_empirical_region_fit=1, severe_frames=0, FINAL_reads=0, seconds=time.perf_counter()-started))
    print("REGION_SEALED", group, len(rows), flush=True)


def score(group):
    from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric, paired
    target = HERE / group
    if (target / "RESULTS.json").exists():
        raise FileExistsError("Preserve fixed results")
    seal = C.read(target / "PREDICTION_SEAL.json")
    assert seal["before_scoring"] and seal["predictions_sha256"] == C.sha(target / "PREDICTIONS.jsonl.gz")
    labels_path = COHORT / ("OOF_pose_v1/ISOLATED_CAL_SCORING.json" if group == "CAL_N3" else "test_reused_v1/ISOLATED_TEST_SCORING.json")
    truth = C.read(labels_path)["rows"]
    rows = list(C.rows(target / "PREDICTIONS.jsonl.gz"))
    frames = {f["id"]: f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"]}
    summaries, coords, diagnostics = defaultdict(list), [], []
    for row in rows:
        fid, method, pose = row["id"], row["method"], row["actual_pose"]
        frame, label = frames[fid], truth[fid]
        row["metric"] = pose_metric(pose, label["target"]["truth"])
        summaries[method].append(row)
        model_points = C.project(C.cuboid(*pose["cf_extents"]), pose["R_cf"], pose["centroid"], frame["K"])
        np.testing.assert_allclose(row["keypoint_readout"]["rigid_model_corners"], model_points, atol=1e-10, rtol=0)
        mapping = np.asarray(label["metadata"]["native_to_reference_corner_ids"], int)
        compatible = previous.compatible(row["initial_pose"], label["target"]["truth"], mapping)
        if compatible:
            gt = np.asarray(label["target"]["gt"], float)[:8][mapping]
            raw, mixed = np.asarray(row["original_points"], float)[:8], np.asarray(row["keypoint_readout"]["reported_points"], float)[:8]
            for k in range(8):
                if not np.isfinite(gt[k]).all() or np.all(gt[k] == -1):
                    continue
                coords.append(dict(id=fid, session=frame["session"], method=method, corner=k,
                    human_state=label["metadata"]["human_states_native"][k], source=row["keypoint_readout"]["point_sources"][k],
                    mixed_error_px=float(np.linalg.norm(mixed[k]-gt[k])), rigid_error_px=float(np.linalg.norm(model_points[k]-gt[k])),
                    raw_error_px=float(np.linalg.norm(raw[k]-gt[k])) if np.isfinite(raw[k]).all() and not np.all(raw[k] == -1) else None))
        if method in ("FINITE_VISIBLE", "FINITE_REGION"):
            audit = row["solution"]["region_audit"]
            values = audit["candidates"]
            region_hits = []
            if compatible:
                gt = np.asarray(label["target"]["gt"], float)[:8][mapping]
                for region in row["coarse_regions"]:
                    k = region["native_corner"]
                    if np.isfinite(gt[k]).all() and not np.all(gt[k] == -1):
                        center = np.asarray(region["center_xy"])
                        scale = np.asarray(region["half_axes_px"])
                        region_hits.append(dict(corner=k, covered=bool(np.sum(((gt[k]-center)/scale)**2) <= 1.),
                                                state=region["state"], radius_px=float(scale.max())))
            diagnostics.append(dict(id=fid, method=method, selected=audit["selected"],
                roots=audit["generation"].get("roots", 0), candidates=len(values),
                positive_region_penalty_candidates=sum(v["region"]["normalized_penalty"] > 0 for v in values),
                admissible_candidates=sum(v.get("usable", False) for v in values),
                compatible=compatible, region_reference_coverage=region_hits))
    summary = {name: dict(frames=len(values), success=sum(v["metric"]["success_5cm_5deg"] for v in values),
        translation_cm=previous.dist([v["metric"]["translation_cm"] for v in values]),
        rotation_deg=previous.dist([v["metric"]["rotation_deg"] for v in values]),
        selected_pose_sources=dict(Counter(v["solution"].get("selected_pose_source", "V1_BASE") for v in values if v["solution"])))
        for name, values in summaries.items()}
    for name, result in summary.items():
        print("REGION_SCORE", group, name, result["success"], round(result["translation_cm"]["mean"], 4),
              round(result["rotation_deg"]["mean"], 4), flush=True)
    zero = {r["id"]: r for r in summaries["FINITE_VISIBLE"]}
    contrast = paired(summaries["FINITE_REGION"], zero, repeats=10000)
    C.save_rows(target / "SCORED_ROWS.jsonl.gz", rows)
    C.save_rows(target / "COORDINATE_ROWS.jsonl.gz", coords)
    C.save_rows(target / "REGION_DIAGNOSTICS.jsonl.gz", diagnostics)
    C.write(target / "RESULTS.json", dict(summary=summary, region_minus_same_pool_visible=contrast,
        predictions_sha256=seal["predictions_sha256"], labels_sha256=C.sha(labels_path),
        CAL_in_sample=group == "CAL_N3", TEST_reused=group != "CAL_N3", independent_confirmation=False,
        pose_primary_all_frames=True, native_mapping_locked=True, no_outcome_selected_checkpoint=True,
        region_hard_pruning=False, no_joint_coverage_or_pose_probability_claim=True, severe_frames=0))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "fit", "infer", "score"))
    parser.add_argument("--group", choices=GROUPS, default="CAL_N3")
    args = parser.parse_args()
    if args.phase in ("prepare", "fit"):
        {"prepare": prepare, "fit": fit}[args.phase]()
    else:
        {"infer": infer, "score": score}[args.phase](args.group)
