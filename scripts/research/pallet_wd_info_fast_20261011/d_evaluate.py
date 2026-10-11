"""Calibrate and seal matched residual predictions before permitted GT scoring."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

import d_local as D


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def new_common():
    # Historical dcp_env imports its historical "common". Keep the new helper
    # under a distinct module name so neither environment can shadow the other.
    return module("wd_D_new_common", Path(__file__).with_name("common.py"))


def write_rows(path, rows, common):
    assert not Path(path).exists()
    with gzip.open(path, "xt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(common.clean(row), ensure_ascii=False, allow_nan=False) + "\n")


def load_heads(args, base, heads):
    receipt = D.read(Path(args.public_dir) / "D_TRAINING_RECEIPT.json")
    for arm, head in heads.items():
        path = Path(args.private_dir) / "D" / (arm + ".pt")
        assert D.file_sha(path) == receipt["checkpoints"][arm]["sha256"]
        checkpoint = torch.load(path, map_location="cpu")
        assert checkpoint["step"] == receipt["updates_per_arm"][arm]
        head.load_state_dict(checkpoint["state_dict"])
        head.eval().requires_grad_(False)
    return receipt


@torch.no_grad()
def calibrate(args):
    env, refiner, data_type, protocol, path, base, heads = D.setup(args, "cuda")
    common = new_common(); common.lock_ready()
    receipt = load_heads(args, base, heads)
    data = data_type(); frozen = D.FrozenCandidates(base)
    rows = np.flatnonzero(data.partitions == "calibration")
    assert len(rows) == 1004
    grid = protocol["temperature_grid"]
    totals = {arm: np.zeros(len(grid)) for arm in D.ARMS}
    frames = 0; corners = 0
    begin = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(rows), 16):
        batch = data.batch(rows[start:start+16], "N3_DIM_SYM")
        output, features = frozen.forward(batch)
        target = refiner.targets(output, batch["gt_points"], batch["gt_valid"])
        mask = target["support"]; count = mask.sum(-1); used = count > 0
        frames += int(used.sum()); corners += int(count.sum())
        for arm in D.ARMS:
            metadata = batch["context"] if arm == "D_JOINT_DIM" else torch.zeros_like(batch["context"])
            modified = D.altered_output(output, features, heads[arm], metadata)
            for i, temperature in enumerate(grid):
                ce = -(target["distribution"].double() *
                       (modified["logits"].double()/temperature).log_softmax(-1)).sum(-1)
                totals[arm][i] += float(((ce*mask).sum(-1)/count.clamp_min(1))[used].sum())
    torch.cuda.synchronize()
    selected = {}
    for arm in D.ARMS:
        candidates = [dict(temperature=t, score=float(v/frames)) for t, v in zip(grid, totals[arm])]
        best = min(candidates, key=lambda row: (row["score"], abs(math.log(row["temperature"])), row["temperature"]))
        selected[arm] = dict(**best, candidates=candidates)
    result = dict(status="DONE", population="all original synthetic calibration", frames=len(rows),
                  supported_frames=frames, supported_corners=corners, temperatures=selected,
                  objective="Original fixed-index Gaussian candidate CE; no symmetry phase during calibration",
                  rule=protocol["selected_rule"], real_selection=False, lambda_cap_search=False,
                  common_updates=receipt["updates_per_arm"], actual_device_seconds=time.monotonic()-begin,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), new_PnP_calls=0,
                  raw_RGB_feature_extractions=0, calibration_row_indices_sha256=
                  __import__("hashlib").sha256(rows.astype("<i8").tobytes()).hexdigest())
    common.write(Path(args.public_dir)/"D_CALIBRATION.json", result)
    print("D_CALIBRATION", json.dumps(result), flush=True)


def published_rows(args, name):
    with gzip.open(Path(args.published_root)/name, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


@torch.no_grad()
def predict(args):
    env, refiner, data_type, protocol, path, base, heads = D.setup(args, "cuda")
    common = new_common(); common.lock_ready(); load_heads(args, base, heads)
    allowed = D.read(Path(args.public_dir)/"INPUT_CONTRACT.json")
    ids = allowed["allowed_ids"]; assert len(ids) == 173
    cache = D.read(env.DOC/"DEV_CACHE_COMPLETE.json")
    by = {row["id"]: row for row in cache["records"] if row["id"] in set(ids)}
    assert set(by) == set(ids)
    calibration = D.read(Path(args.public_dir)/"D_CALIBRATION.json")
    normalization = D.read(env.DOC/"DIM_NORMALIZATION_LOCK.json")
    from point_inference import replace_selected
    image_features = env.old("features")
    frozen = D.FrozenCandidates(base)
    side = np.load(env.RAW/"DIMENSION_SIDECAR.npz")
    data = data_type()
    original_train = np.flatnonzero(data.partitions == "train")
    permutation = np.random.default_rng(20261011).permutation(original_train)
    shuffle_dims = {fid: side["dimensions"][int(index)] for fid, index in zip(ids, permutation[:len(ids)])}
    # This loader reads only whitelisted detector metadata/coordinates, never GT.
    n3_inputs = {}
    for row in published_rows(args, "_docs/experiments/pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz"):
        if row["id"] in by and row["method"] == "BASE" and row["seed"] == 1:
            n3_inputs[row["id"]] = {key: row[key] for key in
                                    ("id", "session", "grade", "q0", "raw_hw", "prediction_support", "fixed_metadata")}
    assert set(n3_inputs) == set(ids)
    methods = module("wd_D_subpix", Path(args.published_root)/"scripts/research/pallet_training_free_compare_20261007_v1/methods.py")
    methods.ROOT = env.ROOT
    baseline = D.read(env.RAW/"predictions/REAL_DEV/OLD_P_seed1.json")
    baseline_by = {row["id"]: row for row in baseline["records"] if row["id"] in by}
    predictions = []; sensitivity = []; pixels = Counter()
    begin = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for index, fid in enumerate(ids):
        entry = by[fid]; assert D.file_sha(env.ROOT/entry["path"]) == entry["sha256"]
        cached = torch.load(env.ROOT/entry["path"], map_location="cpu")
        assert cached["GT_input"] is False and cached["id"] == fid
        captured = cached["captured"]
        inputs = image_features.branch_inputs(captured)
        meta = n3_inputs[fid]; q0 = np.array(meta["q0"], float)
        assert np.array_equal(q0, np.asarray(captured["candidates"][captured["selected_index"]]["keypoints_xy"]), equal_nan=True)
        raw = cv2.imread(str(env.ROOT/cached["key"]), cv2.IMREAD_COLOR)
        assert raw is not None and list(raw.shape[:2]) == list(cached["raw_hw"])
        assert D.file_sha(env.ROOT/cached["key"]) == cached["image_sha256"]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY); pixels["gray_decodes"] += 1
        support = np.array(meta["prediction_support"], bool)
        if inputs is None:
            raise ValueError("Explicit empty-detection contract required; do not silently remove frame")
        batch = {key: torch.as_tensor(inputs[key], device="cuda")[None]
                 for key in ("points", "boxes", "point_valid", "input_shape")}
        batch.update(p3=captured["p3"].cuda(), p4=captured["p4"].cuda())
        output, features = frozen.forward(batch)
        correct = torch.as_tensor(refiner.context(np.asarray(cached["dimensions"])[None], [cached["order"]], normalization, False), device="cuda")
        shuffled = torch.as_tensor(refiner.context(np.asarray(shuffle_dims[fid])[None], [cached["order"]], normalization, False), device="cuda")
        contexts = {"D_CONTROL": torch.zeros_like(correct), "D_JOINT_DIM": correct,
                    "D_JOINT_MEAN": torch.zeros_like(correct), "D_JOINT_SHUFFLE": shuffled}
        cap = protocol["selected_rule"]["max_move_image_diagonal_fraction"]*math.hypot(*cached["raw_hw"])*inputs["gain"]
        lam = protocol["selected_rule"]["lam"]
        variants = {}
        all_logs = {}
        for arm, metadata in contexts.items():
            owner = "D_CONTROL" if arm == "D_CONTROL" else "D_JOINT_DIM"
            modified = D.altered_output(output, features, heads[owner], metadata)
            temperature = calibration["temperatures"][owner]["temperature"]
            decoded = refiner.decode(modified, temperature, lam, cap)[0].cpu().numpy()
            restored = replace_selected(captured["candidates"], captured["selected_index"], inputs["points"], decoded, inputs["gain"], lam)
            variants[arm] = np.asarray(restored[captured["selected_index"]]["keypoints_xy"], float)
            all_logs[arm] = modified["logits"][0].cpu().numpy()
        original_temperature = D.read(env.C.B/"P_SELECTION.json")["temperatures"]["1"]["temperature"]
        decoded = refiner.decode(output, original_temperature, lam, cap)[0].cpu().numpy()
        restored = replace_selected(captured["candidates"], captured["selected_index"], inputs["points"], decoded, inputs["gain"], lam)
        p_only = np.asarray(restored[captured["selected_index"]]["keypoints_xy"], float)
        existing = baseline_by[fid]
        assert np.array_equal(p_only, np.asarray(existing["candidates"][existing["selected_index"]]["keypoints_xy"]), equal_nan=True), fid
        serial = {"D_CONTROL_SUBPIX": variants["D_CONTROL"], "D_JOINT_DIM_SUBPIX": variants["D_JOINT_DIM"],
                  "OLD_P_SUBPIX": p_only}
        for arm, start_points in serial.items():
            native, diagnostic = methods.correct(gray, start_points, support, "SUBPIX")
            variants[arm] = methods.cap_points(q0, native, raw.shape[1], raw.shape[0], support)
            pixels[arm+"_cornerSubPix_calls"] += diagnostic["algorithm_corner_calls"]
        for arm, q in variants.items():
            assert np.array_equal(q[8], q0[8], equal_nan=True)
            assert np.array_equal(q[~support], q0[~support], equal_nan=True)
            predictions.append(dict(**meta, method=arm, seed=1, backbone="YOLO", qFinal=q,
                                    source_flag=False, predictor_reference_inputs=False,
                                    PnP_dimensions="original correct canonical dimensions, invariant under metadata intervention"))
        for arm in ("D_JOINT_MEAN", "D_JOINT_SHUFFLE"):
            a = all_logs[arm]-all_logs["D_JOINT_DIM"]
            probability = torch.as_tensor(all_logs[arm]/calibration["temperatures"]["D_JOINT_DIM"]["temperature"]).softmax(-1).numpy()
            reference = torch.as_tensor(all_logs["D_JOINT_DIM"]/calibration["temperatures"]["D_JOINT_DIM"]["temperature"]).softmax(-1).numpy()
            sensitivity.append(dict(id=fid, intervention=arm,
                                     centered_logit_maxabs=float(np.max(np.abs(a-a.mean(-1, keepdims=True)))),
                                     probability_L1_mean=float(np.abs(probability-reference).sum(-1).mean()),
                                     coordinate_delta_px=np.linalg.norm(variants[arm][:8]-variants["D_JOINT_DIM"][:8], axis=-1),
                                     argmax_changed=int((all_logs[arm].argmax(-1)!=all_logs["D_JOINT_DIM"].argmax(-1)).sum()),
                                     correct_PnP_dimensions_held_fixed=True))
        if (index+1) % 50 == 0:
            print("D_PREDICTION", index+1, len(ids), flush=True)
    torch.cuda.synchronize()
    destination = Path(args.private_dir)/"D/PREDICTIONS.jsonl.gz"
    write_rows(destination, predictions, common)
    common.write(Path(args.public_dir)/"D_PREDICTION_SEAL.json",
                 dict(status="SEALED_BEFORE_REFERENCE_SCORING", frames=len(ids), rows=len(predictions),
                      methods=sorted(set(r["method"] for r in predictions)),
                      private_namespace="D/PREDICTIONS.jsonl.gz", sha256=D.file_sha(destination),
                      reference_inputs=False, protected_feature_reads=0, raw_RGB_feature_extractions=0,
                      feature_cache="matching current R0 captured P3/P4 reuse", source_checkpoint_sha256=env.R0_SHA,
                      original_P_coordinates_exact=True, counts=dict(pixels),
                      shuffle="TRAIN-supported canonical dimensions; fixed RNG20261011 permutation, assigned to sorted allowed IDs",
                      actual_device_seconds=time.monotonic()-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    common.write(Path(args.private_dir)/"D/SENSITIVITY.json", sensitivity)
    print("D_PREDICTIONS_SEALED", len(predictions), flush=True)


def summary(rows):
    available = [r for r in rows if r["pose"]["available"]]
    axis = [r for r in rows if r["axis_wrong"] is not None]
    value = dict(frames=len(rows), pose_available=len(available), pose_coverage=len(available)/len(rows),
                 success_count=sum(r["strict_success"] for r in rows),
                 strict_success_rate=sum(r["strict_success"] for r in rows)/len(rows),
                 axis_evaluable=len(axis), wrong_axis=sum(r["axis_wrong"] for r in axis),
                 confusion_rate=sum(r["axis_wrong"] for r in axis)/len(axis) if axis else None,
                 unclear_count=sum(r["axis_status"]=="UNCLEAR" for r in rows),
                 missing_hypothesis=sum(r["hypothesis"] is None for r in rows))
    for key in ("translation_cm", "rotation_deg", "ADDsym_m", "IoU3D"):
        a = np.array([r["pose"][key] for r in available], float)
        value[key] = dict(mean=float(a.mean()), SD=float(a.std(ddof=1)), median=float(np.median(a)),
                          P90=float(np.quantile(a, .9)), P95=float(np.quantile(a, .95))) if len(a) else None
    return value


def evaluate(args):
    env, refiner, data_type, protocol, path, base, heads = D.setup(args, "cpu")
    common = new_common(); common.lock_ready()
    seal = D.read(Path(args.public_dir)/"D_PREDICTION_SEAL.json")
    pred_path = Path(args.private_dir)/seal["private_namespace"]
    assert D.file_sha(pred_path) == seal["sha256"]
    allowed = D.read(Path(args.public_dir)/"INPUT_CONTRACT.json")
    ids = set(allowed["allowed_ids"])
    pose = module("wd_D_pose", env.HERE/"pose.py")
    predictions = list(common.rows(pred_path)); inferred = []
    with common.pnp_counter("D", 8500) as calls:
        for index, row in enumerate(predictions):
            trace = []; original = pose.select_pnp_hypotheses
            def traced(*a, **kw):
                result = original(*a, **kw); trace.append(result); return result
            pose.select_pnp_hypotheses = traced
            try:
                actual = pose.infer(row["qFinal"], row["fixed_metadata"]["K"],
                                    row["fixed_metadata"]["dimensions_pnp_WH_D_m"], False)
            finally:
                pose.select_pnp_hypotheses = original
            candidates = {}
            if trace:
                for h in trace[0].hypotheses:
                    candidates[h.name] = dict(success=h.success, score=h.score, score_components=dict(h.score_components),
                                              R_cf=h.rotation_camera_facing, t=h.translation_camera_facing,
                                              semantics="original selector fit; not the additional final8-corner pose fit")
            inferred.append(dict(row, actual_pose=actual, candidate_selector_values=candidates))
            if (index+1) % 250 == 0:
                print("D_PRODUCTION_F", index+1, len(predictions), sum(calls.values()), flush=True)
    pose_path = Path(args.private_dir)/"D/POSES.jsonl.gz"
    write_rows(pose_path, inferred, common)
    common.write(Path(args.public_dir)/"D_POSE_SEAL.json", dict(status="SEALED_BEFORE_REFERENCE_SCORING",
                 private_namespace="D/POSES.jsonl.gz", sha256=D.file_sha(pose_path), rows=len(inferred),
                 original_F_calls=len(inferred), actual_cv_calls=dict(calls), reference_inputs_in_F=False))
    # References are first materialized below both persisted prediction seals.
    geometry = module("wd_D_geometry", Path(__file__).with_name("geometry_analysis.py"))
    wd_reference = {r["id"]: r for r in published_rows(args, "_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz")
                    if r["id"] in ids and r["method"] == "BASE" and r["seed"] == 1}
    truth = geometry.truth_real(env.ROOT, wd_reference)
    assert set(truth) == ids
    targets = D.read(env.ROOT/"data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json")
    corner_math = module("wd_D_corner_math", env.HERE/"eval_math.py")
    scored = []
    for row in inferred:
        fid = row["id"]; metric = pose.metric((fid, row["actual_pose"], truth[fid]))
        hypothesis = row["actual_pose"].get("selected_hypothesis")
        unclear = fid in allowed["unclear_allowed_ids"]
        wrong = None if unclear or hypothesis is None else int(hypothesis != wd_reference[fid]["hypOracle"])
        target = targets[fid]
        corners = corner_math.measure(row["qFinal"], target["gt"], target["valid"], target["permutations"],
                                      row["raw_hw"], target["matched"], row["fixed_metadata"]["selected_index"] is not None)
        corners["id"] = fid
        scored.append(dict(id=fid, session=row["session"], grade=row["grade"], method=row["method"], seed=1,
                           backbone="YOLO", pose=metric, corner=corners, hypothesis=hypothesis,
                           axis_wrong=wrong, axis_status="UNCLEAR" if unclear else "MISSING" if hypothesis is None else "EVALUABLE",
                           strict_success=geometry.good(metric), candidate_selector_values=row["candidate_selector_values"],
                           prediction_reference_separated=True))
    # Immutable OLD_P/N3 Stage1 and N3 serial baselines: no new solver calls.
    baseline_rows = []
    for row in published_rows(args, "_docs/experiments/pallet_paper_tables_245_20261011/RECT_ROWS.jsonl.gz"):
        if row["id"] not in ids or row["backbone"] != "YOLO" or row["seed"] != 1 or row["method"] not in ("P", "N3_DIM_SYM", "N3_THEN_SUBPIX"):
            continue
        fid = row["id"]; name = row["actual_pose"].get("selected_hypothesis")
        old = row["pose"]
        metric = dict(id=fid, available=old["available"])
        if old["available"]:
            # The public paper row stores original metrics under short labels.
            metric.update(translation_cm=old["translation_cm"] if "translation_cm" in old else old["T_cm"],
                          rotation_deg=old["rotation_deg"] if "rotation_deg" in old else old["R_deg"],
                          ADDsym_m=old["ADDsym_m"], IoU3D=old["IoU3D"])
        unclear = fid in allowed["unclear_allowed_ids"]
        baseline_rows.append(dict(id=fid, session=row["session"], grade=row["grade"], method="OLD_P" if row["method"]=="P" else row["method"],
                                  seed=1, backbone="YOLO", pose=metric, corner=dict(row["corner"], id=fid), hypothesis=name,
                                  axis_wrong=None if unclear or name is None else int(name!=wd_reference[fid]["hypOracle"]),
                                  axis_status="UNCLEAR" if unclear else "MISSING" if name is None else "EVALUABLE",
                                  strict_success=geometry.good(metric), reused_historical_pose=True))
    assert len(baseline_rows) == 3*173
    scored.extend(baseline_rows)
    write_rows(Path(args.public_dir)/"D_ROWS.jsonl.gz", scored, common)
    grouped = {name: [r for r in scored if r["method"]==name] for name in sorted(set(r["method"] for r in scored))}
    aggregates = {population: {name: summary([r for r in rows if r["id"] in subset]) for name, rows in grouped.items()}
                  for population, subset in [("ALLOWED173", ids), ("RECT245_INTERSECTION110", set(allowed["rect245_intersection_ids"]))]}
    damage = {}
    for suffix in ("", "_SUBPIX"):
        control = sorted(grouped["D_CONTROL"+suffix], key=lambda r: r["id"])
        joint = sorted(grouped["D_JOINT_DIM"+suffix], key=lambda r: r["id"])
        damage["JOINT_minus_CONTROL"+suffix] = corner_math.damage([r["corner"] for r in control], [r["corner"] for r in joint])
    sensitivities = D.read(Path(args.private_dir)/"D/SENSITIVITY.json")
    by = {(r["id"], r["method"]): r for r in scored}
    for row in sensitivities:
        original = by[row["id"], "D_JOINT_DIM"]; intervention = by[row["id"], row["intervention"]]
        row.update(selection_switched=original["hypothesis"]!=intervention["hypothesis"],
                   strict_success_change=int(intervention["strict_success"])-int(original["strict_success"]),
                   T_change_cm=None if not original["pose"]["available"] or not intervention["pose"]["available"] else
                   intervention["pose"]["translation_cm"]-original["pose"]["translation_cm"],
                   R_change_deg=None if not original["pose"]["available"] or not intervention["pose"]["available"] else
                   intervention["pose"]["rotation_deg"]-original["pose"]["rotation_deg"])
    common.write(Path(args.public_dir)/"D_SENSITIVITY.json", sensitivities)
    common.write(Path(args.public_dir)/"D_RESULTS.json", dict(status="DONE", exploratory=True, seed=1,
                 frames=173, independent_training_seeds=1, existing_N3_seeds="All three remain frozen; matched D comparison uses seed1",
                 aggregates=aggregates, corner_damage=damage, baseline_reused=True,
                 new_cv_calls=sum(calls.values()), original_F_calls=len(inferred),
                 wrong_axis_denominator="Reference-evaluable and pose-hypothesis available; UNCLEAR and missing separately counted",
                 full_success_denominator=173, candidate_values_semantics="Cached selector fit before original final8-corner solve; never a best-of-two result"))
    print("D_EVALUATION_COMPLETE", json.dumps(aggregates["ALLOWED173"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["calibrate", "predict", "evaluate"])
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--published-root", required=True)
    parser.add_argument("--private-dir", required=True)
    parser.add_argument("--public-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
