"""Read-only input audit; publish relative paths/hashes and keep checkout snapshots private.

Run with the existing pallet-pose interpreter. No detector, learned model,
training, synthetic data generation, or pose evaluator is executed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def snapshot(root, private, label):
    status = git(root, "status", "--porcelain=v1", "-z")
    diff = git(root, "diff", "--binary", "HEAD")
    tracked = git(root, "ls-files", "-m", "-d", "-z").decode().split("\0")
    hashes = {p: sha(root / p) if (root / p).is_file() else "MISSING"
              for p in tracked if p}
    private.mkdir(parents=True, exist_ok=True)
    (private / (label + "_status.bin")).write_bytes(status)
    (private / (label + "_tracked.diff")).write_bytes(diff)
    value = dict(head=git(root, "rev-parse", "HEAD").decode().strip(),
                 branch=git(root, "branch", "--show-current").decode().strip(),
                 status_sha256=hashlib.sha256(status).hexdigest(),
                 tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                 changed_tracked_sha256=hashes)
    write(private / (label + "_snapshot.json"), value)
    return value


def audit(root, private):
    import cv2
    import numpy as np
    import torch

    start = time.monotonic()
    before = snapshot(root, private, "input_audit_before")
    doc = root / "_docs/experiments/pallet_dim_conditioned_p_v1"
    protocol = read(doc / "TRAIN_PROTOCOL_LOCK.json")
    trained = read(doc / "TRAINING_COMPLETE.json")
    dev = read(doc / "DEV_INFERENCE_COMPLETE.json")
    calibration = read(doc / "CALIBRATION_AND_SELECTION.json")
    assert trained["complete"] and dev["complete"] and calibration["complete"]
    assert protocol["seeds"] == [1, 2, 3]
    assert protocol["selected_rule"] == calibration["rule"] == {
        "lam": 1.0, "max_move_image_diagonal_fraction": 0.01}
    assert calibration["real_access"] is False
    assert calibration["new_lambda_cap_sweep"] is False
    prior_path = root / "_docs/experiments/pallet_n3_subpix_20261008_v1/PROTOCOL.json"
    prior = read(prior_path)
    source_path = root / "_docs/experiments/pallet_training_free_compare_20261007_v1/PROTOCOL.json"
    source = read(source_path)
    assert digest(prior["input_manifest"]) == prior["input_manifest_sha256"]
    assert digest(source["input_manifest"]) == source["input_manifest_sha256"]
    expected_ids = prior["population"]["frame_ids"]
    assert len(expected_ids) == len(set(expected_ids)) == 319
    source_inputs = {v["id"]: v for v in source["input_manifest"]}
    prior_inputs = {v["id"]: v for v in prior["input_manifest"]}
    assert set(source_inputs) == set(prior_inputs) == set(expected_ids)

    bindings = []
    def binding(relative, expected=None):
        path = root / relative
        actual = sha(path)
        if expected is not None:
            assert actual == expected, "HASH_MISMATCH: " + relative
        value = dict(path=relative, sha256=actual, bytes=path.stat().st_size,
                     expected_sha256=expected, verified=True)
        bindings.append(value)
        return value

    for relative in ["TRAIN_PROTOCOL_LOCK.json", "TRAINING_COMPLETE.json",
                     "DEV_INFERENCE_COMPLETE.json", "CALIBRATION_AND_SELECTION.json",
                     "DIM_NORMALIZATION_LOCK.json"]:
        binding("_docs/experiments/pallet_dim_conditioned_p_v1/" + relative)
    binding(str(prior_path.relative_to(root)))
    binding(str(source_path.relative_to(root)))
    for b in source["code"] + source["dependencies"] + prior["code"]:
        binding(b["path"], b["sha256"])
    for b in prior["dependencies"]:
        # Prior labels omit immutable scorer directories and baseline namespace.
        if b["path"] in ("pose.py", "eval_math.py"):
            relative = "scripts/research/pallet_dim_conditioned_p_v1/" + b["path"]
        elif b["path"].startswith("immutable_a22/"):
            continue
        else:
            relative = b["path"]
        binding(relative, b["sha256"])
    for relative in ["RESULT_KO.md", "PREDICTIONS.jsonl.gz", "VERIFICATION.json", "RUNTIME.json"]:
        binding("_docs/experiments/pallet_n3_subpix_20261008_v1/" + relative)
    binding("scripts/research/pallet_dim_conditioned_p_v1/inference.py")
    binding("scripts/research/pallet_dim_conditioned_p_v1/refiner.py")
    binding("scripts/research/pallet_training_free_compare_20261007_v1/methods.py")
    registry_relative = "challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json"
    registry = {r["object_type"]: r for r in read(root / registry_relative)["objects"]}
    binding(registry_relative)
    # Locate the inference-owned contract path from dcp_env rather than guess it.
    env_text = (root / "scripts/research/pallet_dim_conditioned_p_v1/dcp_env.py").read_text()
    sym_match = re.search(r"SYM_DOC=ROOT/'([^']+)'", env_text)
    assert sym_match is not None, "Unable to resolve existing inference symmetry contract"
    symmetry_relative = sym_match.group(1) + "/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json"
    symmetry = {r["object_type"]: r for r in read(root / symmetry_relative)["objects"]}
    binding(symmetry_relative)
    baseline_path = Path(source["baseline"]["path"])
    assert sha(baseline_path) == source["baseline"]["sha256"], "Immutable a22 baseline hash mismatch"
    bindings.append(dict(path="immutable_a22/A_REAL_DEV_BASELINES.json", sha256=sha(baseline_path),
                         bytes=baseline_path.stat().st_size, expected_sha256=source["baseline"]["sha256"], verified=True))

    seed_records = {}
    models = []
    for seed in (1, 2, 3):
        arm = f"N3_DIM_SYM_seed{seed}"
        checkpoint = next(v for v in trained["checkpoints"] if f"{arm}/" in v["path"])
        ck_binding = binding(checkpoint["path"], checkpoint["sha256"])
        assert ck_binding["bytes"] == checkpoint["bytes"]
        calibration_row = calibration["temperatures"][arm]
        assert calibration_row["checkpoint"] == checkpoint
        ck = torch.load(root / checkpoint["path"], map_location="cpu", weights_only=False)
        assert ck["complete"] and ck["step"] == 6000
        assert ck["baseline_checkpoint_sha256"] == trained["R0_hash"]
        pred_binding = binding(dev["files"][arm]["path"], dev["files"][arm]["sha256"])
        packet = read(root / pred_binding["path"])
        assert packet["complete"] and packet["GT_input"] is False
        assert packet["checkpoint"] == checkpoint
        records = {r["id"]: r for r in packet["records"]}
        assert len(records) == len(packet["records"]) == 319
        assert set(records) == set(expected_ids)
        seed_records[seed] = records
        models.append(dict(seed=seed, arm=arm, checkpoint=ck_binding, prediction=pred_binding,
                           temperature=calibration_row["temperature"], calibration_real_access=False,
                           temperature_lock_sha256=sha(doc / "CALIBRATION_AND_SELECTION.json"),
                           checkpoint_step=ck["step"], baseline_checkpoint_sha256=ck["baseline_checkpoint_sha256"],
                           config=ck["config"], prediction_GT_input=False))

    grade_relative = "_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/static/LABEL_PROVENANCE_AUDIT.json"
    grade_packet = read(root / grade_relative)
    grade_rows = [r for r in grade_packet["rows"] if r["population"] == "DEV319"]
    grades = {r["id"]: r for r in grade_rows}
    assert len(grades) == len(grade_rows) == 319 and set(grades) == set(expected_ids)
    assert dict(Counter(v["severity"] for v in grades.values())) == dict(clean=153, moderate=92, severe=74)
    grade_binding = binding(grade_relative)

    axis_path = root / "data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json"
    # Axis-manifest frame_id uses a different delimiter than prediction IDs.
    # Join through the stored image key instead of fabricating ID conversion.
    axes = {r["image"]: r for r in read(axis_path)["frames_list"]}
    targets = read(root / "data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json")
    all_bindings = []
    overshoot = {}
    for seed in (1, 2, 3):
        overshoot[str(seed)] = []
    for fid in expected_ids:
        b = source_inputs[fid]
        p = prior_inputs[fid]
        assert b["session"] == p["session"] == grades[fid]["session"]
        for old_key, new_key in (("initial_points", "q0"), ("prediction_support", "prediction_support"),
                                 ("K", "K"), ("dimensions_whd_m", "dimensions_whd_m"), ("raw_hw", "raw_hw")):
            assert digest(b[old_key]) == digest(p[new_key]), (fid, old_key)
        image = root / b["image"]
        cache = root / b["cache"]
        assert sha(image) == b["image_sha256"] == p["image_sha256"] == grades[fid]["image"]["sha256"], fid
        assert sha(cache) == b["cache_sha256"] == p["cache_sha256"], fid
        assert b["image"] == grades[fid]["image"]["path"] == axes[b["image"]]["image"], fid
        annotation_path = root / axes[b["image"]]["annotation"]
        assert sha(annotation_path) == grades[fid]["annotation"]["sha256"], fid
        annotation = read(annotation_path)
        raw = annotation["camera_data"]["intrinsics"]
        K = np.array([[raw["fx"], 0, raw["cx"]], [0, raw["fy"], raw["cy"]], [0, 0, 1.]])
        assert np.array_equal(K, np.asarray(b["K"])), fid
        image_check = cv2.imread(str(image), cv2.IMREAD_COLOR)
        assert image_check is not None and list(image_check.shape[:2]) == b["raw_hw"], fid
        captured = torch.load(cache, map_location="cpu", weights_only=False)
        native = captured["captured"]
        selected = native["selected_index"]
        assert selected == b["selected_index"] == p["selected_index"], fid
        q0 = np.asarray(native["candidates"][selected]["keypoints_xy"], dtype=np.float64)
        support = np.isfinite(q0).all(-1) & ~np.all(q0 == -1, axis=-1)
        assert np.array_equal(q0, np.asarray(b["initial_points"]), equal_nan=True), fid
        assert np.array_equal(support, np.asarray(b["prediction_support"])), fid
        assert np.array_equal(np.asarray(captured["dimensions"])[[0, 2, 1]], np.asarray(b["dimensions_whd_m"])), fid
        assert list(captured["raw_hw"]) == b["raw_hw"], fid
        object_type = captured["object_type"]
        d = registry[object_type]["physical_dimensions_m"]
        canonical_wdh = np.asarray([d["x"], d["z"], d["y"]])
        assert np.allclose(canonical_wdh, captured["dimensions"], rtol=0, atol=1e-12), fid
        assert symmetry[object_type]["group_order"] == captured["order"], fid
        base_metadata = [{k: v for k, v in c.items() if k != "keypoints_xy"} for c in native["candidates"]]
        base_metadata = [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in c.items()} for c in base_metadata]
        for seed in (1, 2, 3):
            record = seed_records[seed][fid]
            assert record["key"] == b["image"] and record["selected_index"] == selected, (seed, fid)
            assert len(record["candidates"]) == len(native["candidates"]), (seed, fid)
            metadata = [{k: v for k, v in c.items() if k != "keypoints_xy"} for c in record["candidates"]]
            assert digest(metadata) == digest(base_metadata), (seed, fid, "candidate metadata")
            for index, (base_candidate, new_candidate) in enumerate(zip(native["candidates"], record["candidates"])):
                if index != selected:
                    assert np.array_equal(np.asarray(base_candidate["keypoints_xy"]), np.asarray(new_candidate["keypoints_xy"]), equal_nan=True), (seed, fid, index)
            qn = np.asarray(record["candidates"][selected]["keypoints_xy"], dtype=np.float64)
            assert qn.shape == (9, 2) and np.isfinite(qn[support]).all(), (seed, fid)
            assert np.array_equal(qn[8], q0[8], equal_nan=True), (seed, fid, "center")
            assert np.array_equal(qn[~support], q0[~support], equal_nan=True), (seed, fid, "unsupported")
            if seed == 1:
                assert np.array_equal(qn, np.asarray(p["qN"]), equal_nan=True), fid
            limit = .01 * np.hypot(*b["raw_hw"])
            excess = np.linalg.norm(qn[:8] - q0[:8], axis=-1) - limit
            overshoot[str(seed)].extend(excess[excess > 1e-10].tolist())
            assert np.max(excess, initial=0) <= 1e-4, (seed, fid, "historic float32 cap")
        all_bindings.append(dict(id=fid, session=b["session"], grade=grades[fid]["severity"],
                                 image=dict(path=b["image"], sha256=b["image_sha256"]),
                                 detector_feature_cache=dict(path=b["cache"], sha256=b["cache_sha256"]),
                                 annotation=dict(path=str(annotation_path.relative_to(root)), sha256=sha(annotation_path)),
                                 raw_hw=b["raw_hw"], K=b["K"], dimensions_wdh_m=canonical_wdh.tolist(),
                                 PnP_xyz_whd_m=b["dimensions_whd_m"], object_type=object_type,
                                 symmetry_order=int(captured["order"]), selected_index=selected,
                                 prediction_support=b["prediction_support"], base_points=b["initial_points"]))

    with gzip.open(root / "_docs/experiments/pallet_n3_subpix_20261008_v1/PREDICTIONS.jsonl.gz", "rt") as stream:
        previous_rows = [json.loads(line) for line in stream]
    grouped = {arm: [v for v in previous_rows if v["method"] == arm] for arm in prior["methods"]}
    for arm, rows in grouped.items():
        assert len(rows) == 319 and {v["id"] for v in rows} == set(expected_ids), arm
        assert sum(v["corner"]["matched"] for v in rows) == 311
        assert sum(len(v["corner"]["errors"]) for v in rows) == 2499
        assert sum(len(v["corner"]["observed_errors"]) for v in rows) == 2445
    assert sum(sum(targets[fid]["valid"][:8]) for fid in expected_ids) == 2499
    assert sum(targets[fid]["matched"] for fid in expected_ids) == 311
    runtime = read(root / "_docs/experiments/pallet_n3_subpix_20261008_v1/RUNTIME.json")
    binding(runtime["raw_rows"]["path"], runtime["raw_rows"]["sha256"])
    with gzip.open(root / runtime["raw_rows"]["path"], "rt") as stream:
        runtime_rows = [json.loads(line) for line in stream]
    assert runtime["status"] == "DONE" and runtime["statistics_official"] is True
    assert runtime["frames"] == 26 and runtime["panel_sessions"] == 13
    runtime_verified = {}
    for arm, summary in runtime["summaries"].items():
        values = np.asarray([r["full_ms"] for r in runtime_rows if r["phase"] == "measured" and r["arm"] == arm])
        assert len(values) == 130
        computed = dict(n=len(values), mean_ms=float(np.mean(values)), sample_variance_ms2=float(np.var(values, ddof=1)),
                        sample_std_ms=float(np.std(values, ddof=1)), median_ms=float(np.median(values)),
                        p90_ms=float(np.quantile(values, .9)), max_ms=float(np.max(values)))
        for key, value in computed.items():
            if key in summary["full_pipeline"]:
                assert np.isclose(summary["full_pipeline"][key], value, rtol=0, atol=1e-12), (arm, key)
        runtime_verified[arm] = computed
    after = snapshot(root, private, "input_audit_after")
    assert before == after, "Original checkout changed during read-only input audit"
    return dict(schema="pallet_n3_subpix_final_input_audit_v1", status="PASS", complete=True,
                population=dict(images=319, sessions=13, matched_images=311,
                                reference_corners=2499, observed_corners=2445,
                                frame_ids=expected_ids),
                models=models, fixed_rule=calibration["rule"], inputs=all_bindings,
                source_bindings=bindings, input_manifest_sha256=digest(all_bindings),
                grades=dict(source=grade_binding, counts=dict(clean=153, moderate=92, severe=74),
                            classification_semantics="NOT_CONFIRMED", inference_use=False),
                prediction_schema=dict(root_keys=list(packet), record_keys=list(packet["records"][0]),
                                       candidate_keys=list(packet["records"][0]["candidates"][0])),
                numeric=dict(N3_original_float32_cap_overshoot={k: dict(corners=len(v), max_px=max(v, default=0)) for k, v in overshoot.items()},
                             N3_historical_cap_tolerance_px=1e-4,
                             combined_final_cap_tolerance_px=1e-10),
                checks=dict(all_seed_ID_sets_identical=True, all_319_image_cache_hashes_match=True,
                            all_319_annotation_hashes_match=True, K_dimensions_symmetry_support_match=True,
                            selected_candidate_and_metadata_match=True, center_and_unsupported_preserved=True,
                            existing_reference_denominators_match=True,
                            source_checkout_preserved=True),
                original_checkout_snapshot_hashes=dict(status_sha256=before["status_sha256"],
                    tracked_diff_sha256=before["tracked_diff_sha256"],
                    snapshot_record_sha256=digest(before), private_snapshot_published=False),
                environment=dict(interpreter_environment="existing pallet-pose", python=sys.version.split()[0],
                                 torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__),
                runtime_reuse=dict(status="PASS", source="_docs/experiments/pallet_n3_subpix_20261008_v1/RUNTIME.json",
                                   attribution="existing 2026-10-08 seed1 timing; no seed2/3 timing claim",
                                   frames=26, sessions=13, warmup_each=20, repeats=5,
                                   independent_raw_row_statistics=runtime_verified, new_timing_calls=0),
                execution=dict(elapsed_seconds=time.monotonic()-start, model_calls=0, F_calls=0,
                               optimizer_updates=0, source_writes=0, new_data=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, default=Path("/dev/shm/pallet-n3-subpix-final-private"))
    args = parser.parse_args()
    try:
        result = audit(args.source_root.resolve(), args.private_dir)
    except Exception as error:
        result = dict(schema="pallet_n3_subpix_final_input_audit_v1", status="BLOCKED", complete=False,
                      error_type=type(error).__name__, error=str(error).replace(str(args.source_root.resolve()), "<SOURCE_ROOT>"))
        write(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        raise
    write(args.output, result)
    print(json.dumps(dict(status=result["status"], population={k: v for k, v in result["population"].items() if k != "frame_ids"},
                         checks=result["checks"], numeric=result["numeric"], execution=result["execution"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
