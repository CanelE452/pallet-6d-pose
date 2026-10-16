"""Frozen count rules attached to N3+Sub and literal PoseFix+Sub on reused49."""
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
from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_rule_api import BoundaryRuleCorrector
from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_observation import BoundaryObservation

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent
PRIOR = COHORT / "boundary_progressive_replay_v1"
BASELINE = COHORT / "posefix_sub_comparison_v1"
ARMS = [family + "_COUNT_" + coordinates for family in ("N3_SUB", "PF_SUB") for coordinates in ("RAW", "FUSED")]


@contextmanager
def no_labels():
    original = builtins.open, io.open, Path.open
    def wrap(fn):
        def guard(path, *args, **kwargs):
            if any(x in str(path) for x in ("SCORING", "SCORED_ROWS", "ORACLE", "TARGETS", "RESULTS.json", "COMPARISON", "/annotations/", "FINAL")):
                raise AssertionError("Count replay accessed labels/outcomes before sealing: " + str(path))
            return fn(path, *args, **kwargs)
        return guard
    builtins.open, io.open, Path.open = map(wrap, original)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = original


def prepare():
    if (HERE / "PROTOCOL.json").exists():
        raise FileExistsError("Preserve count replay preregistration")
    scripts = ROOT / "scripts/research/pallet_visibility_curriculum_20261013_v1"
    files = [COHORT / n for n in ("PREDICTION_INPUTS.json", "COHORT_PROTOCOL.json", "BOUNDARY_RULE_CONFIG.json",
        "BOUNDARY_OBSERVATION_CONFIG.json", "CAL_VISIBILITY_CONFIG.json", "visibility_logistic_CAL46_v2.json",
        "clean_source_segmentation_v1/best_source_CAL.pt")]
    files += [PRIOR / n for n in ("PREDICTIONS.jsonl.gz", "OBSERVATIONS.jsonl.gz", "PREDICTION_SEAL.json")]
    files += [BASELINE / n for n in ("PREDICTIONS.jsonl.gz", "PREDICTION_SEAL.json")]
    files += [scripts / n for n in ("boundary_rule_api.py", "boundary_rules.py", "boundary_evidence.py", "boundary_observation.py",
        "curriculum.py", "common.py", "joint_api.py", "branch_curriculum.py", "boundary_visibility.py", "geometric_boundary_audit.py",
        "calibration.py", "segmentation.py", "segmented_appearance.py", "appearance.py", "boundary_refine.py")]
    files += [ROOT / "scripts/research/pallet_foundation_mask_20261013_v1" / n for n in ("foundation.py", "uncertainty.py")]
    vendor = COHORT.parent / "foundation_mask_v1/vendor/MobileSAM"
    files += [vendor / "weights/mobile_sam.pt"] + sorted((vendor / "mobile_sam").rglob("*.py"))
    files += [Path(__file__)]
    snapshot = HERE / "code_snapshot"
    snapshot.mkdir()
    for file in files:
        if file.suffix == ".py":
            target = snapshot / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    frames = [f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"] if f["partition"] == "TEST"]
    assert len(frames) == 49
    C.write(HERE / "PROTOCOL.json", dict(schema="REUSED_TEST49_COUNT_RULES_TWO_UPSTREAM_BASELINES_V1",
        before_count_replay_and_scoring=True, frame_ids=[f["id"] for f in frames],
        source_sha256={str(f): C.sha(f) for f in files},
        arms=ARMS + ["N3_SUB_INITIAL", "PF_SUB_INITIAL", "N3_FUSED_PROGRESSIVE"],
        count_rules="Same CAL-preregistered >=5/4 full6DOF,3 fixedR translation,<=2 exactinitial; no interior/unknown supplement; actual two-image-line support and p>=.65",
        N3_observations="Already sealed actualTEST49 rawN3+Sub initial pose/RGB/Tiny/BOX_ONLYSAM and sigma2 blend; no fresh N3 masks",
        PoseFix_observations="New currentRGB/Tiny/BOX_ONLYSAM queried at already sealed literalPoseFix+Sub points and its own existingF pose; original nativeorder/H/canvas/body basis",
        PoseFix_Sub_definition="PF-local 1% cap retains learnedPoseFix point beforeSub; chosen operationally to attach a postprocessor to its caller's inputs, not chosen for metric superiority. ExistingYOLO-totalcap variant remains reported in separate matchedcomparison.",
        fresh_SAM_forwards=49, new_training=0, severe_frames=0, FINAL_reads=0,
        TEST_already_observed=True, independent_confirmation=False, TEST_tuning=False,
        final_completion="On acceptance, only selectedindependent measuredpoints fit; remaining8slots are explicit output-only projections, ninthcenter unchanged. <=2/failure retains original points and pose.",
        evaluation="All49 primary includingfallbacks; paired4session10000bootstrapseed20261013; DOF/count/fallback distributions and baseline rescues/losses, no oraclestage/nativephase remap",
        stop_rule="Execute these seven frozen arms once; no outcome-dependent thresholds/countgate/anchor changes"))
    print("COUNT_TEST49_PREREGISTERED", C.sha(HERE / "PROTOCOL.json"), flush=True)


class Cached:
    def __init__(self, observation):
        self.observation = observation

    def assess(self, *args):
        return deepcopy(self.observation)


def infer():
    import torch
    p = C.read(HERE / "PROTOCOL.json")
    if (HERE / "PREDICTIONS.jsonl.gz").exists():
        raise FileExistsError("Preserve count predictions")
    for name, digest in p["source_sha256"].items():
        assert C.sha(name) == digest, name
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    output, observations = [], []
    with no_labels(), torch.inference_mode():
        try:
            C.read(COHORT / "test_reused_v1/ISOLATED_TEST_SCORING.json")
        except AssertionError:
            pass
        else:
            raise AssertionError("Label canary failed")
        seal = C.read(PRIOR / "PREDICTION_SEAL.json")
        assert seal["before_scoring"] and seal["predictions_sha256"] == C.sha(PRIOR / "PREDICTIONS.jsonl.gz")
        baseline_seal = C.read(BASELINE / "PREDICTION_SEAL.json")
        assert baseline_seal["before_scoring"]
        n3obs = {r["id"]: r["observation"] for r in C.rows(PRIOR / "OBSERVATIONS.jsonl.gz")}
        prior = {r["id"]: r for r in C.rows(PRIOR / "PREDICTIONS.jsonl.gz") if r["method"] == "FUSED_ACTIVE"}
        pf = {r["id"]: r for r in C.rows(BASELINE / "PREDICTIONS.jsonl.gz") if r["method"] == "POSEFIX_SUBPIX_PF_ANCHOR" and r["partition"] == "TEST"}
        frames = [f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"] if f["partition"] == "TEST"]
        assert [f["id"] for f in frames] == p["frame_ids"] and set(pf) == set(p["frame_ids"])
        observer = BoundaryObservation.from_config(COHORT / "BOUNDARY_OBSERVATION_CONFIG.json", "cuda")
        for index, frame in enumerate(frames):
            image_path = ROOT / frame["image"]
            assert C.sha(image_path) == frame["image_sha256"]
            image = cv2.imread(str(image_path))
            n3initial = frame["initial_poses"]["N3_SUBPIX"]
            pf_initial = pf[frame["id"]]["actual_pose"]
            pf_q = np.asarray(pf[frame["id"]]["native_points"], float)
            new_pf = observer.assess(image, pf_q, frame["K"], pf_initial)
            for family, q, initial, observation in (
                ("N3_SUB", np.asarray(frame["points"]["N3_SUBPIX"], float), n3initial, n3obs[frame["id"]]),
                ("PF_SUB", pf_q, pf_initial, new_pf)):
                observations.append(dict(id=frame["id"], family=family, observation=observation))
                for coord in ("RAW", "FUSED"):
                    engine = BoundaryRuleCorrector(Cached(observation), dict(coordinate_source=coord, strong_visibility_threshold=.65))
                    result = engine.refine(image, q, frame["K"], initial, frame["xyz"])
                    assert result["actual_pose"]["cf_extents"] == initial["cf_extents"]
                    if not result["decision"]["accepted"]:
                        assert result["actual_pose"] == initial
                        np.testing.assert_array_equal(result["completed_points"], q)
                    if result["decision"]["estimated_dof"] == 3:
                        assert result["actual_pose"]["R_cf"] == initial["R_cf"] and result["actual_pose"]["R_physical"] == initial["R_physical"]
                    output.append(dict(id=frame["id"], session=frame["session"], method=family + "_COUNT_" + coord,
                        actual_pose=result["actual_pose"], solution=result["solution"], decision=result["decision"],
                        original_points=q.tolist(), corrected_measurements=result["corrected_measurements"],
                        completed_points=result["completed_points"], projected_points=result["projected_points"],
                        projected_completion_ids=result["projected_completion_ids"], self_hidden=result["self_hidden"]))
                output.append(dict(id=frame["id"], session=frame["session"], method=family + "_INITIAL",
                    actual_pose=initial, solution=dict(accepted=False), original_points=q.tolist()))
            stored = prior[frame["id"]]
            output.append(dict(id=frame["id"], session=frame["session"], method="N3_FUSED_PROGRESSIVE",
                actual_pose=stored["actual_pose"], solution=stored["solution"], original_points=stored["original_points"]))
            if (index + 1) % 10 == 0:
                print("COUNT_TEST49_INFER", index + 1, round(time.perf_counter() - started, 2), flush=True)
    for name, digest in p["source_sha256"].items():
        assert C.sha(name) == digest, name
    assert len(output) == 343 and len(observations) == 98
    C.save_rows(HERE / "PREDICTIONS.jsonl.gz", output)
    C.save_rows(HERE / "OBSERVATIONS.jsonl.gz", observations)
    C.write(HERE / "PREDICTION_SEAL.json", dict(complete=True, before_scoring=True, rows=343,
        predictions_sha256=C.sha(HERE / "PREDICTIONS.jsonl.gz"), observations_sha256=C.sha(HERE / "OBSERVATIONS.jsonl.gz"),
        protocol_sha256=C.sha(HERE / "PROTOCOL.json"), GT_canary="PASS", fresh_PoseFix_SAM_calls=49,
        N3_masks_reused_from_exact_frozen_RGB_study=49, TEST_already_observed=True, independent_confirmation=False,
        severe_frames=0, FINAL_reads=0, seconds=time.perf_counter() - started))
    print("COUNT_TEST49_SEALED", 343, flush=True)


def distribution(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), median=float(np.median(x)), P90=float(np.quantile(x, .9)))


def score():
    from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric
    if (HERE / "RESULTS.json").exists():
        raise FileExistsError("Preserve count results")
    p = C.read(HERE / "PROTOCOL.json")
    seal = C.read(HERE / "PREDICTION_SEAL.json")
    assert seal["before_scoring"] and seal["predictions_sha256"] == C.sha(HERE / "PREDICTIONS.jsonl.gz")
    truth = C.read(COHORT / "test_reused_v1/ISOLATED_TEST_SCORING.json")["rows"]
    assert len(truth) == 49 and set(truth) == set(p["frame_ids"])
    rows = list(C.rows(HERE / "PREDICTIONS.jsonl.gz"))
    groups = defaultdict(list)
    for row in rows:
        row["metric"] = pose_metric(row["actual_pose"], truth[row["id"]]["target"]["truth"])
        groups[row["method"]].append(row)
    summary = {}
    for method, group in groups.items():
        counts = Counter(r.get("decision", {}).get("image_supported_boundary_count") for r in group)
        by_count = {}
        if "COUNT" in method:
            for count in counts:
                selected = [r for r in group if r["decision"]["image_supported_boundary_count"] == count]
                by_count[str(count)] = dict(frames=len(selected), accepted=sum(r["solution"]["accepted"] for r in selected),
                    success=sum(r["metric"]["success_5cm_5deg"] for r in selected),
                    metrics={k: distribution([r["metric"][k] for r in selected]) for k in ("translation_cm", "rotation_deg")})
        summary[method] = dict(frames=len(group), success=sum(r["metric"]["success_5cm_5deg"] for r in group),
            accepted=sum(r["solution"]["accepted"] for r in group),
            metrics={k: distribution([r["metric"][k] for r in group]) for k in ("translation_cm", "rotation_deg", "ADDsym_cm")},
            count_histogram={str(k): v for k, v in counts.items()}, by_count=by_count,
            fallbacks=dict(Counter(r["solution"].get("fallback_reason") for r in group)))
        print(method, "T", summary[method]["metrics"]["translation_cm"]["mean"], "R",
            summary[method]["metrics"]["rotation_deg"]["mean"], "success", summary[method]["success"],
            "accepted", summary[method]["accepted"], flush=True)
    lookup = {(r["id"], r["method"]): r for r in rows}
    sessions = sorted({r["session"] for r in rows})
    rng = np.random.default_rng(20261013)
    draw = rng.integers(0, len(sessions), (10000, len(sessions)))
    paired, rescued = {}, {}
    contrasts = [(family + "_COUNT_" + coord, family + "_INITIAL") for family in ("N3_SUB", "PF_SUB") for coord in ("RAW", "FUSED")]
    contrasts += [("N3_SUB_COUNT_FUSED", "N3_FUSED_PROGRESSIVE"), ("PF_SUB_COUNT_FUSED", "N3_SUB_COUNT_FUSED")]
    for a, b in contrasts:
        group = groups[a]
        report = {}
        for metric in ("translation_cm", "rotation_deg", "ADDsym_cm", "success_5cm_5deg"):
            delta = np.array([float(r["metric"][metric]) - float(lookup[r["id"], b]["metric"][metric]) for r in group])
            sums = np.array([sum(d for d, r in zip(delta, group) if r["session"] == s) for s in sessions])
            counts = np.array([sum(r["session"] == s for r in group) for s in sessions])
            boot = sums[draw].sum(1) / counts[draw].sum(1)
            report[metric] = dict(mean_difference=float(delta.mean()), CI95=np.quantile(boot, [.025, .975]).tolist())
        paired[a + "-" + b] = report
        rescued[a + "-" + b] = dict(rescued=[r["id"] for r in group if r["metric"]["success_5cm_5deg"] and not lookup[r["id"], b]["metric"]["success_5cm_5deg"]],
            lost=[r["id"] for r in group if not r["metric"]["success_5cm_5deg"] and lookup[r["id"], b]["metric"]["success_5cm_5deg"]])
    C.save_rows(HERE / "SCORED_ROWS.jsonl.gz", rows)
    C.write(HERE / "RESULTS.json", dict(schema=p["schema"], summary=summary, paired=paired,
        rescued_lost=rescued, policy_or_anchor_tuning=False, TEST_already_observed=True,
        independent_confirmation=False, severe_frames=0, FINAL_reads=0,
        comparison_role="Attaching fixed count-based postprocessor to each upstream pipeline; guarded completion, not an all-condition accuracy guarantee"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "infer", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "infer": infer, "score": score}[args.phase]()
