"""One frozen, exploratory replay on previously observed nonsevere TEST49."""
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
from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_curriculum_api import BoundaryCurriculum
from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_observation import BoundaryObservation

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent
ARMS = [dict(name=c + "_" + p, coordinates=c, staged=p != "TOGETHER", active=p == "ACTIVE")
        for c in ("RAW", "FUSED") for p in ("LEGACY", "ACTIVE", "TOGETHER")]


@contextmanager
def no_labels():
    originals = builtins.open, io.open, Path.open
    def wrap(fn):
        def guard(path, *args, **kwargs):
            if any(x in str(path) for x in ("SCORING", "ORACLE", "SCORED_ROWS", "RESULTS.json", "TARGETS", "/annotations/", "FINAL")):
                raise AssertionError("Forbidden exploratory inference label read: " + str(path))
            return fn(path, *args, **kwargs)
        return guard
    builtins.open, io.open, Path.open = map(wrap, originals)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = originals


def prepare():
    if (HERE / "PROTOCOL.json").exists():
        raise FileExistsError("Preserve frozen exploratory replay")
    from scripts.research.pallet_visibility_curriculum_20261013_v1.curriculum import CurriculumConfig
    assert CurriculumConfig(progressive_active_support=True).progressive_active_support
    frames = C.read(COHORT / "PREDICTION_INPUTS.json")["frames"]
    selected = [f for f in frames if f["partition"] == "TEST"]
    assert len(frames) == 95 and len(selected) == 49 and len({f["id"] for f in selected}) == 49
    scripts = ROOT / "scripts/research/pallet_visibility_curriculum_20261013_v1"
    files = [COHORT / n for n in ("PREDICTION_INPUTS.json", "SPLIT.json", "COHORT_PROTOCOL.json",
        "BOUNDARY_PROGRESSIVE_CONFIG.json", "BOUNDARY_CURRICULUM_CONFIG.json", "BOUNDARY_OBSERVATION_CONFIG.json",
        "CAL_VISIBILITY_CONFIG.json", "visibility_logistic_CAL46_v2.json", "clean_source_segmentation_v1/best_source_CAL.pt")]
    files += [scripts / n for n in ("boundary_curriculum_api.py", "boundary_observation.py", "curriculum.py", "common.py",
        "joint_api.py", "branch_curriculum.py", "boundary_visibility.py", "geometric_boundary_audit.py", "calibration.py",
        "segmentation.py", "segmented_appearance.py", "appearance.py", "boundary_refine.py")]
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
    C.write(HERE / "PROTOCOL.json", dict(schema="REUSED_NONSEVERE_TEST49_BOUNDARY_PROGRESSIVE_V1",
        before_new_TEST_image_inference_and_scoring=True, TEST_already_observed=True,
        independent_confirmation=False, user_severe_training_and_primary_eval_exclusion=True,
        frames=49, frame_ids=[f["id"] for f in selected], arms=ARMS + [dict(name="INITIAL")],
        source_sha256={str(f): C.sha(f) for f in files},
        appearance="Frozen clean-source Tiny426/72 and CAL46 RGB visibility_v2 at raw queries; fixed for every arm",
        observations="One fresh BOX_ONLY MobileSAM/current image and two image transition lines; fixed sigma2 alpha<=.5 blend",
        stages="Initial geometric-outline information seed; legacy versus CAL-preregistered active support; same q/weights/eligibility/unknown rules",
        final_active_guard="Latest accepted checkpoint passing full immutable visible pool + fixed initial prior against INITIAL; no GT/minimum stage or batch fallback",
        controls="Same RAW/FUSED RGB evidence together, original model family and physicalQ, center/H/native order unchanged",
        tuning_or_selection_from_TEST=False, new_training=False, FINAL_reads=0, severe_frames=0,
        metrics="All49 including fallbacks/phase-incompatible frames, physical reference translation/rotation/ADDsym+5cm5deg; paired4sessionbootstrap10000",
        stop_rule="Run declared finite7arms once; no TEST-dependent parameter/stage/head adjustment; report regressions and absence of ordering advantage"))
    print("REUSED_TEST49_BOUNDARY_PREREGISTERED", C.sha(HERE / "PROTOCOL.json"), flush=True)


class CachedObservation:
    def __init__(self, evidence, coordinates, observer):
        self.evidence = evidence
        self.coordinates = coordinates
        self.boundary_profile = observer.boundary_profile
        self.fusion = observer.fusion

    def assess(self, image, points, K, initial):
        result = deepcopy(self.evidence)
        if self.coordinates == "RAW":
            result["fused_points"] = deepcopy(result["raw_points"])
        return result


def infer():
    import torch
    p = C.read(HERE / "PROTOCOL.json")
    if (HERE / "PREDICTIONS.jsonl.gz").exists():
        raise FileExistsError("Preserve sealed exploratory TEST predictions")
    for name, digest in p["source_sha256"].items():
        assert C.sha(name) == digest, name
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    rows, observations = [], []
    with no_labels(), torch.inference_mode():
        try:
            C.read(COHORT / "test_reused_v1/ISOLATED_TEST_SCORING.json")
        except AssertionError:
            pass
        else:
            raise AssertionError("Label canary failed")
        frames = [f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"] if f["partition"] == "TEST"]
        assert [f["id"] for f in frames] == p["frame_ids"]
        observer = BoundaryObservation.from_config(COHORT / "BOUNDARY_OBSERVATION_CONFIG.json", "cuda")
        for index, frame in enumerate(frames):
            path = ROOT / frame["image"]
            assert C.sha(path) == frame["image_sha256"]
            image = cv2.imread(str(path))
            initial = frame["initial_poses"]["N3_SUBPIX"]
            q = np.asarray(frame["points"]["N3_SUBPIX"], float)
            observation = observer.assess(image, q, frame["K"], initial)
            observations.append(dict(id=frame["id"], observation=observation))
            H = np.asarray(observation["self_hidden"], bool)
            for arm in ARMS:
                config = dict(visible_threshold=.2, geometric_outline_priority=arm["staged"],
                              staged_visible=arm["staged"], progressive_active_support=arm["active"])
                engine = BoundaryCurriculum(CachedObservation(observation, arm["coordinates"], observer), config)
                result = engine.refine(image, q, frame["K"], initial, frame["xyz"])
                solution = result["solution"]
                assert result["self_hidden"] == observation["self_hidden"]
                assert np.array_equal(result["corrected_measurements"][8:], q[8:])
                assert all(not H[k] for s in solution["stages"] for k in s["used_ids"])
                assert result["actual_pose"]["cf_extents"] == initial["cf_extents"]
                if not solution["accepted"]:
                    assert result["actual_pose"] == initial and np.array_equal(result["completed_points"], q)
                if arm["active"] and solution["accepted"]:
                    guard = solution["final_guard"]
                    assert guard["retained_checkpoint_index"] is not None
                rows.append(dict(id=frame["id"], session=frame["session"], method=arm["name"],
                    actual_pose=result["actual_pose"], solution=solution,
                    self_hidden=result["self_hidden"], appearance=result["visibility_assessment"],
                    original_points=q.tolist(), observed_points=result["corrected_measurements"],
                    projected_points=result["projected_points"], completed_points=result["completed_points"],
                    original_family_fixed=True, new_labels_in_inference=False))
            rows.append(dict(id=frame["id"], session=frame["session"], method="INITIAL", actual_pose=initial,
                             solution=dict(accepted=False), original_points=q.tolist()))
            if (index + 1) % 10 == 0:
                print("REUSED_TEST49_BOUNDARY_INFER", index + 1, round(time.perf_counter() - started, 2), flush=True)
    for name, digest in p["source_sha256"].items():
        assert C.sha(name) == digest, name
    assert len(rows) == 49 * 7
    C.save_rows(HERE / "OBSERVATIONS.jsonl.gz", observations)
    C.save_rows(HERE / "PREDICTIONS.jsonl.gz", rows)
    C.write(HERE / "PREDICTION_SEAL.json", dict(before_scoring=True, complete=True, rows=len(rows),
        predictions_sha256=C.sha(HERE / "PREDICTIONS.jsonl.gz"), observations_sha256=C.sha(HERE / "OBSERVATIONS.jsonl.gz"),
        protocol_sha256=C.sha(HERE / "PROTOCOL.json"), GT_read_canary="PASS", fresh_MobileSAM_calls=49,
        severe_frames=0, FINAL_reads=0, previously_observed_TEST=True, independent_confirmation=False,
        seconds=time.perf_counter() - started))
    print("REUSED_TEST49_BOUNDARY_SEALED", len(rows), flush=True)


def dist(values):
    a = np.asarray(values, float)
    return dict(mean=float(a.mean()), median=float(np.median(a)), P90=float(np.quantile(a, .9)))


def score():
    from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric
    if (HERE / "RESULTS.json").exists():
        raise FileExistsError("Preserve completed exploratory scoring")
    p = C.read(HERE / "PROTOCOL.json")
    seal = C.read(HERE / "PREDICTION_SEAL.json")
    assert seal["before_scoring"] and seal["predictions_sha256"] == C.sha(HERE / "PREDICTIONS.jsonl.gz")
    truth = C.read(COHORT / "test_reused_v1/ISOLATED_TEST_SCORING.json")["rows"]
    assert set(truth) == set(p["frame_ids"]) and len(truth) == 49
    rows = list(C.rows(HERE / "PREDICTIONS.jsonl.gz"))
    groups = defaultdict(list)
    for row in rows:
        row["metric"] = pose_metric(row["actual_pose"], truth[row["id"]]["target"]["truth"])
        groups[row["method"]].append(row)
    summary = {}
    for method, group in groups.items():
        summary[method] = dict(frames=len(group), accepted=sum(r["solution"]["accepted"] for r in group),
            success=sum(r["metric"]["success_5cm_5deg"] for r in group),
            metrics={k: dist([r["metric"][k] for r in group]) for k in ("translation_cm", "rotation_deg", "ADDsym_cm")},
            stage_rejections=dict(Counter(s["reason"] for r in group for s in r["solution"].get("stages", [])[1:] if not s["accepted"])))
        print(method, "T", summary[method]["metrics"]["translation_cm"]["mean"],
              "R", summary[method]["metrics"]["rotation_deg"]["mean"], "success", summary[method]["success"], flush=True)
    lookup = {(r["id"], r["method"]): r for r in rows}
    sessions = sorted({r["session"] for r in rows})
    rng = np.random.default_rng(20261013)
    draw = rng.integers(0, len(sessions), (10000, len(sessions)))
    paired = {}
    contrasts = [("RAW_ACTIVE", "RAW_LEGACY"), ("RAW_ACTIVE", "RAW_TOGETHER"),
        ("FUSED_ACTIVE", "FUSED_LEGACY"), ("FUSED_ACTIVE", "FUSED_TOGETHER"),
        ("FUSED_ACTIVE", "RAW_ACTIVE"), ("FUSED_ACTIVE", "INITIAL"), ("FUSED_TOGETHER", "RAW_TOGETHER")]
    for a, b in contrasts:
        group = groups[a]
        comparison = {}
        for metric in ("translation_cm", "rotation_deg", "ADDsym_cm", "success_5cm_5deg"):
            delta = np.array([float(r["metric"][metric]) - float(lookup[r["id"], b]["metric"][metric]) for r in group])
            sums = np.array([sum(d for d, r in zip(delta, group) if r["session"] == s) for s in sessions])
            counts = np.array([sum(r["session"] == s for r in group) for s in sessions])
            boot = sums[draw].sum(1) / counts[draw].sum(1)
            comparison[metric] = dict(mean_difference=float(delta.mean()), CI95=np.quantile(boot, [.025, .975]).tolist())
        paired[a + "-" + b] = comparison
    support = {}
    for method in ("RAW_LEGACY", "RAW_ACTIVE", "FUSED_LEGACY", "FUSED_ACTIVE"):
        group = groups[method]
        support[method] = dict(first_seed_accepted=sum(bool(r["solution"]["stages"][1]["accepted"]) for r in group),
            accepted_fit_stages=sum(sum(s["accepted"] for s in r["solution"]["stages"][1:] if s["name"] != "final_initial_visible_safeguard") for r in group),
            omitted_visible_points=sum(len(set(r["solution"]["visible_ids"]) - set(r["solution"]["used_ids"])) for r in group),
            final_guard_fallbacks=dict(Counter(r["solution"].get("final_guard", {}).get("fallback_source", "legacy") for r in group)))
    C.save_rows(HERE / "SCORED_ROWS.jsonl.gz", rows)
    C.write(HERE / "RESULTS.json", dict(schema=p["schema"], summary=summary, paired=paired, support=support,
        TEST_already_observed=True, independent_confirmation=False, tuning_after_scoring=False,
        severe_frames=0, FINAL_reads=0, original_family_fixed=True,
        claim_limit="Exploratory original-family transfer; separates coordinate fusion from ordering/guard effects. No unseen or physical-measurement confirmation.",
        prediction_sha256=seal["predictions_sha256"]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "infer", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "infer": infer, "score": score}[args.phase]()
