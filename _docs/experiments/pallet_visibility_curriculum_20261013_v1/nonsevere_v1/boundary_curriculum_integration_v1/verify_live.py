"""Actual-RGB CAL-only callable integration against already sealed evidence."""
from contextlib import contextmanager
from pathlib import Path
import builtins
import hashlib
import io
import json
import sys
import time
import shutil

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.boundary_curriculum_api import BoundaryCurriculum

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent
SOURCE = COHORT / "SAM_boundary_uncertainty_v1"
OUTLINE = COHORT / "geometric_outline_fusion_v1"


@contextmanager
def prediction_only():
    originals = builtins.open, io.open, Path.open
    forbidden = ("SCORING", "ORACLE", "SCORED_ROWS", "RESULTS.json", "TARGETS", "/annotations/", "test_reused", "FINAL")
    def wrap(fn):
        def guard(path, *args, **kwargs):
            if any(x in str(path) for x in forbidden):
                raise AssertionError("Forbidden integration label/evaluation access: " + str(path))
            return fn(path, *args, **kwargs)
        return guard
    builtins.open, io.open, Path.open = map(wrap, originals)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = originals


def prepare():
    if (HERE / "PROTOCOL.json").exists():
        raise FileExistsError("Preserve frozen integration protocol")
    files = [COHORT / "CAL_INPUTS.json", COHORT / "COHORT_PROTOCOL.json",
             COHORT / "BOUNDARY_CURRICULUM_CONFIG.json", COHORT / "BOUNDARY_OBSERVATION_CONFIG.json",
             COHORT / "CAL_VISIBILITY_CONFIG.json", COHORT / "visibility_logistic_CAL46_v2.json",
             COHORT / "clean_source_segmentation_v1/best_source_CAL.pt",
             SOURCE / "PREDICTIONS.jsonl.gz", SOURCE / "PREDICTION_SEAL.json",
             SOURCE / "OBSERVED_LINE_UNCERTAINTY.json", OUTLINE / "PREDICTIONS.jsonl.gz",
             OUTLINE / "PREDICTION_SEAL.json"]
    scripts = ROOT / "scripts/research/pallet_visibility_curriculum_20261013_v1"
    files += [scripts / name for name in ("boundary_curriculum_api.py", "boundary_observation.py", "curriculum.py",
              "common.py", "joint_api.py", "boundary_visibility.py", "geometric_boundary_audit.py", "calibration.py",
              "boundary_refine.py", "appearance.py", "segmentation.py", "segmented_appearance.py")]
    files += [ROOT / "scripts/research/pallet_foundation_mask_20261013_v1" / n for n in ("foundation.py", "uncertainty.py")]
    vendor = COHORT.parent / "foundation_mask_v1/vendor/MobileSAM"
    files += [vendor / "weights/mobile_sam.pt"] + sorted((vendor / "mobile_sam").rglob("*.py"))
    files += [Path(__file__)]
    snapshot = HERE / "code_snapshot"
    snapshot.mkdir()
    for file in files:
        if file.suffix == ".py":
            relative = file.relative_to(ROOT)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    C.write(HERE / "PROTOCOL.json", dict(schema="CAL46_BOUNDARY_CURRICULUM_CALLABLE_INTEGRATION_V1",
        before_current_RGB_inference=True, files_sha256={str(f): C.sha(f) for f in files},
        frames=46, CAL_only=True, severe_rows=0, TEST_reads=0, FINAL_reads=0,
        purpose="Implementation/evidence parity; no new accuracy claim or policy selection",
        observer="One new full RGB Tiny/head and BOX_ONLY MobileSAM per image; no reused masks",
        comparison="Already sealed SAM observations/lines/fusion and geometric-outline staged poses; no GT labels",
        tolerances=dict(coordinate_px=1e-10, visibility_probability=5e-7, R_matrix=1e-5, translation_m=1e-5),
        exact_requirements="Original input/order/center/H/RGB states/fusion candidate IDs/accepted/used IDs/mask index+BOX",
        labels_and_pose_fit_projections_used_for_measurements=False))


def infer():
    import torch
    protocol = C.read(HERE / "PROTOCOL.json")
    if (HERE / "PREDICTIONS.jsonl.gz").exists():
        raise FileExistsError("Preserve completed current-RGB integration")
    for name, digest in protocol["files_sha256"].items():
        assert C.sha(name) == digest, name
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    output = []
    max_errors = dict(coordinate_px=0., visibility_probability=0., R_matrix=0., translation_m=0.)
    with prediction_only(), torch.inference_mode():
        try:
            C.read(COHORT / "CAL_SCORING_METADATA.json")
        except AssertionError:
            pass
        else:
            raise AssertionError("Annotation canary failed")
        for folder, field in ((SOURCE, "predictions_sha256"), (OUTLINE, "sha256")):
            seal = C.read(folder / "PREDICTION_SEAL.json")
            assert seal["before_scoring"] and seal[field] == C.sha(folder / "PREDICTIONS.jsonl.gz")
        frames = C.read(COHORT / "CAL_INPUTS.json")["frames"]
        assert len(frames) == 46 and all(f["partition"] == "CAL" for f in frames)
        source = {r["id"]: r for r in C.rows(SOURCE / "PREDICTIONS.jsonl.gz") if r["method"] == "FUSED_SIGMA2_STAGED"}
        poses = {r["id"]: r for r in C.rows(OUTLINE / "PREDICTIONS.jsonl.gz") if r["arm"] == "FUSED_SIGMA2_GEOMETRY_INFO_STAGED"}
        evidence = {r["id"]: r for r in C.read(SOURCE / "OBSERVED_LINE_UNCERTAINTY.json")}
        engine = BoundaryCurriculum.from_config(COHORT / "BOUNDARY_CURRICULUM_CONFIG.json", "cuda")
        for index, frame in enumerate(frames):
            image_path = ROOT / frame["image"]
            assert C.sha(image_path) == frame["image_sha256"]
            image = cv2.imread(str(image_path))
            initial = frame["initial_poses"]["N3_SUBPIX"]
            result = engine.refine(image, frame["points"]["N3_SUBPIX"], frame["K"], initial, frame["xyz"])
            prior = source[frame["id"]]
            old_pose = poses[frame["id"]]
            obs = result["observation_result"]
            observed = evidence[frame["id"]]
            coordinate_error = float(np.max(np.abs(np.asarray(obs["fused_points"]) - prior["observed_points"])))
            assert np.allclose(obs["observed_points"], observed["correction"]["points"], rtol=0, atol=1e-10, equal_nan=True)
            assert obs["correction"]["observed_refined_ids"] == observed["correction"]["observed_refined_ids"]
            assert obs["visibility_appearance"]["states"] == prior["appearance"]["states"]
            assert result["self_hidden"] == prior["self_hidden"]
            assert result["solution"]["accepted"] == old_pose["solution"]["accepted"]
            assert result["solution"]["used_ids"] == old_pose["solution"]["used_ids"]
            errors = dict(coordinate_px=coordinate_error,
                visibility_probability=float(np.max(np.abs(np.asarray(obs["visibility_appearance"]["visibility_score"]) - prior["appearance"]["visibility_score"]))),
                R_matrix=float(np.max(np.abs(np.asarray(result["actual_pose"]["R_cf"]) - old_pose["actual_pose"]["R_cf"]))),
                translation_m=float(np.max(np.abs(np.asarray(result["actual_pose"]["centroid"]) - old_pose["actual_pose"]["centroid"]))))
            for key, value in errors.items():
                assert value <= protocol["tolerances"][key], (frame["id"], key, value)
                max_errors[key] = max(max_errors[key], value)
            output.append(dict(id=frame["id"], session=frame["session"], result=result, parity_errors=errors))
            if (index + 1) % 10 == 0:
                print("BOUNDARY_CALLABLE_RGB_CAL46", index + 1, round(time.perf_counter() - started, 2), flush=True)
    for name, digest in protocol["files_sha256"].items():
        assert C.sha(name) == digest, name
    C.save_rows(HERE / "PREDICTIONS.jsonl.gz", output)
    C.write(HERE / "PREDICTION_SEAL.json", dict(complete=True, before_scoring=True, frames=46,
        sha256=C.sha(HERE / "PREDICTIONS.jsonl.gz"), protocol_sha256=C.sha(HERE / "PROTOCOL.json"),
        actual_RGB_forward_frames=46, fresh_MobileSAM_forwards=46, annotation_canary="PASS",
        severe_rows=0, TEST_reads=0, FINAL_reads=0, seconds=time.perf_counter() - started))
    C.write(HERE / "QA.json", dict(PASS=True, frames=46, max_errors=max_errors,
        pose_acceptance_and_used_ids_exact=True, H_center_input_channel_identity_exact=True,
        intersection_candidates_and_coordinates_matched=True, labels_read=0, no_new_performance_claim=True,
        scope="Actual RGB callable inference parity with sealed CAL evidence and original-family staged poses"))
    print("BOUNDARY_CALLABLE_CAL46_PASS", max_errors, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "infer"))
    args = parser.parse_args()
    prepare() if args.phase == "prepare" else infer()
