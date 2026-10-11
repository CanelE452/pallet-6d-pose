"""Narrow 110-frame reuse, fixed CVRank and frozen production-F evaluation.

Old coordinates/metrics are authenticated and reused without new model forwards.
Every new coordinate and prediction-only pose is sealed before reference scoring.
All writes are exclusive to this namespace or its registered private directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_nonsevere_precision_20261011 import contract as C
from scripts.research.pallet_nonsevere_precision_20261011 import evaluate as E

DOC = ROOT / "_docs/experiments/pallet_rgb_only_cvrank_20261011"
PRIVATE = Path(os.environ["PALLET_RGB_ONLY_PRIVATE"]).resolve()
PRIOR = C.PRIVATE
OLD_DOC = ROOT / "_docs/experiments/pallet_nonsevere_precision_20261011"
TF_DOC = ROOT / "_docs/experiments/pallet_training_free_compare_20261007_v1"
SEEDS = (1, 2)
REUSED_METHODS = ("BASE", "SUBPIX", "OLD_P", "OLD_P_SUBPIX", "M1", "M1_SUBPIX",
                  "M3", "M3_SUBPIX", "POSEFIX", "POSEFIX_SUBPIX")
CV_METHODS = ("CVRANK_NATIVE", "CVRANK_CAP1", "CVRANK_THEN_SUBPIX")
RGB_METHODS = ("RGB_ONLY", "RGB_ONLY_SUBPIX")


def binding(path):
    path = Path(path).resolve()
    for owner, base in (("new_worktree", ROOT), ("private_new", PRIVATE),
                        ("prior_private", PRIOR), ("historical_source", C.SOURCE)):
        if path.is_relative_to(base):
            return dict(owner=owner, path=str(path.relative_to(base)), sha256=C.sha(path),
                        bytes=path.stat().st_size)
    raise ValueError("Unregistered input owner")


def verify(value):
    bases = dict(new_worktree=ROOT, private_new=PRIVATE, prior_private=PRIOR,
                 historical_source=C.SOURCE)
    path = bases[value["owner"]] / value["path"]
    assert C.sha(path) == value["sha256"], ("Frozen input changed", value["path"])
    return path


def write(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(PRIVATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(C.clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path, records):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(PRIVATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="xb", mtime=0) as stream:
        for row in records:
            stream.write((json.dumps(C.clean(row), separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False) + "\n").encode())


def source_methods():
    # The existing implementation resolves its immutable calibration beside its
    # own source checkout. No old evaluator, training or GT-loader is called.
    gate = "scripts/research/multiteacher_corner_distill_v1/gate_b_corner_evidence.py"
    C.module("scripts.research.multiteacher_corner_distill_v1.gate_b_corner_evidence",
             C.SOURCE / gate)
    return C.module("rgb_cvrank_frozen_source_methods", C.SOURCE /
                    "scripts/research/pallet_training_free_compare_20261007_v1/methods.py")


def locked():
    path = DOC / "PLAN_AND_CONTRACT.json"
    assert path.exists(), "Parent PLAN must be locked before any new correction/F/scoring"
    contract = C.read(DOC / "INPUT_REUSE_CONTRACT.json")
    assert contract["code"]["sha256"] == C.sha(Path(__file__))
    for value in contract["bindings"]:
        verify(value)
    return binding(path)


def prior_predictions():
    return [r for r in C.rows(PRIOR / "NEW_PREDICTIONS_SEALED.jsonl.gz")
            if r["method"] in REUSED_METHODS and int(r["seed"]) in SEEDS]


def common():
    rows = prior_predictions()
    by = {(r["method"], int(r["seed"]), r["id"]): r for r in rows}
    return rows, {(fid, sd): by[("BASE", sd, fid)] for _, sd, fid in by
                  if ("BASE", sd, fid) in by}


def public_configuration(config):
    result = copy.deepcopy(config)
    for key in ("method_lock", "source_calibration"):
        path = Path(result[key]["path"])
        result[key]["path"] = str(path.relative_to(C.SOURCE))
        result[key]["owner"] = "historical_source"
    for item in result["reused_code"]:
        item["path"] = str(Path(item["path"]).relative_to(C.SOURCE))
        item["owner"] = "historical_source"
    return result


def prepare():
    begun = time.monotonic()
    old = C.read(OLD_DOC / "DATA_CONTRACT.json")
    ids = sorted(old["allowed_ids"])
    assert len(ids) == 110 and old["grade_counts"] == dict(clean=68, moderate=42)
    predictions, bases = common()
    assert len(predictions) == 2200 and len(bases) == 220
    expected = {(method, sd, fid) for method in REUSED_METHODS for sd in SEEDS for fid in ids}
    assert {(r["method"], int(r["seed"]), r["id"]) for r in predictions} == expected
    prior_seal = C.read(OLD_DOC / "NEW_PREDICTION_SEAL.json")
    assert C.sha(PRIOR / prior_seal["predictions"]["path"]) == prior_seal["predictions"]["sha256"]
    input_lock = C.read(OLD_DOC / "NEW_PREDICTION_INPUT_LOCK.json")
    assert input_lock["code"]["sha256"] == C.sha(ROOT / input_lock["code"]["path"])
    old_pose = C.read(OLD_DOC / "NEW_POSE_SEAL.json")
    assert C.sha(PRIOR / old_pose["poses"]["path"]) == old_pose["poses"]["sha256"]
    precision = C.read(OLD_DOC / "PRECISION_PREDICTION_SEAL.json")
    assert precision["FP32"] and not any(precision[k] for k in ("AMP", "matmul_TF32", "cudnn_TF32"))
    assert C.sha(PRIOR / "RAW_NEW_PREDICTIONS.npz") == precision["predictions_sha256"]
    assert C.sha(PRIOR / "CURRENT_DETECTOR_CAPTURES.json") == precision["detector_captures_sha256"]
    match = C.read(OLD_DOC / "CURRENT_DETECTOR_MATCH_PARITY.json")
    assert match["status"] == "MATCH_PARITY_PASS" and match["changed_match_rows"] == 0
    assert match["current_matches"]["sha256"] == C.sha(PRIOR / "CURRENT_DETECTOR_MATCHES.json")
    assert old["private_reference"]["sha256"] == C.sha(PRIOR / "REFERENCE.json")
    captures = {(r["id"], int(r["seed"])): r for r in C.read(PRIOR / "CURRENT_DETECTOR_CAPTURES.json")
                if int(r["seed"]) in SEEDS}
    input_parity = []
    for fid in ids:
        first = bases[(fid, 1)]
        assert C.sha(C.SOURCE / first["image"]["path"]) == first["image"]["sha256"]
        for sd in SEEDS:
            b = bases[(fid, sd)]
            cp = captures[(fid, sd)]
            index = cp["selected_index"]
            assert index == int(np.argmax([x["score"] for x in cp["candidates"]]))
            for method in REUSED_METHODS:
                r = next(x for x in predictions if x["id"] == fid and x["seed"] == sd and x["method"] == method)
                assert C.digest(r["q0"]) == C.digest(b["qFinal"])
                for key in ("prediction_support", "raw_hw", "fixed_metadata", "image", "detected"):
                    assert C.digest(r[key]) == C.digest(b[key]), ("Common input mismatch", method, key)
                assert np.array_equal(r["qFinal"][8], b["qFinal"][8], equal_nan=True)
            assert C.digest(b["qFinal"]) == C.digest(first["qFinal"]), "Rule seed dedup requires exact Base"
            assert C.digest(b["prediction_support"]) == C.digest(first["prediction_support"])
            assert C.digest(cp) == C.digest(dict(captures[(fid, 1)], seed=sd))
        input_parity.append(dict(id=fid, same_seed_all10_methods=True,
                                 seed1_seed2_q0_support_object_exact=True))
    # Raw training-free coordinates exist, but current initial-coordinate parity
    # is checked before deciding reuse. Old summary/pose fields are never used.
    tf = C.read(TF_DOC / "PROTOCOL.json")
    manifests = {r["id"]: r for r in tf["input_manifest"] if r["id"] in set(ids)}
    raw = [r for r in C.rows(TF_DOC / "PREDICTIONS.jsonl.gz", set(ids))
           if r["method"] in ("CVRANK_NATIVE", "CVRANK_CAP1")]
    assert len(raw) == 220 and set(manifests) == set(ids)
    cache_by = {(r["id"], r["method"]): r for r in raw}
    cache_checks = []
    for fid in ids:
        m, b = manifests[fid], bases[(fid, 1)]
        delta = np.max(np.abs(np.asarray(m["initial_points"], float) - np.asarray(b["qFinal"], float)))
        native = cache_by[(fid, "CVRANK_NATIVE")]
        assert np.asarray(native["native_points"]).shape == (9, 2)
        assert C.digest(native["RAW_native_points"]) == C.digest(m["initial_points"])
        assert C.digest(native["prediction_support"]) == C.digest(m["prediction_support"])
        image_equal = m["image_sha256"] == b["image"]["sha256"]
        shape_equal = m["raw_hw"] == b["raw_hw"]
        K_dims_equal = np.array_equal(m["K"], b["fixed_metadata"]["K"]) and np.array_equal(
            m["dimensions_whd_m"], b["fixed_metadata"]["dimensions_pnp_WH_D_m"])
        support_equal = m["prediction_support"] == b["prediction_support"]
        cache_checks.append(dict(id=fid, q0_exact=bool(delta == 0), q0_max_delta_px=float(delta),
            image_hash_equal=image_equal, raw_shape_equal=shape_equal, K_dims_equal=bool(K_dims_equal),
            support_equal=support_equal, selected_index_equal=m["selected_index"] == captures[(fid, 1)]["selected_index"],
            raw_cache_reusable=bool(delta == 0 and image_equal and shape_equal and K_dims_equal and support_equal)))
    methods = source_methods()
    config = methods.method_configuration()
    assert C.sha(Path(methods.__file__)) == C.sha(ROOT / "scripts/research/pallet_training_free_compare_20261007_v1/methods.py")
    for key in ("SUBPIX", "CVRANK", "CAP1"):
        assert C.digest(config[key]) == C.digest(tf["methods"][key])
    reusable = all(r["raw_cache_reusable"] for r in cache_checks)
    # The current detector uses exactly the same raw-resolution/border/letterbox
    # recipe as the authenticated old cache. Read one authoritative metadata
    # record per raw shape, never pass its neck tensors or old predictions to the
    # RGB-only model. Numerical detector points/boxes remain the current capture.
    import torch
    transforms = {}
    transform_probes = []
    for fid in ids:
        hw = tuple(bases[(fid, 1)]["raw_hw"])
        if hw in transforms:
            continue
        m = manifests[fid]
        path = C.SOURCE / m["cache"]
        assert C.sha(path) == m["cache_sha256"]
        meta = torch.load(path, map_location="cpu", weights_only=False)["captured"]
        canvas, shape, pad = tuple(meta["canvas_shape"]), tuple(meta["input_shape"]), int(meta["added_border"])
        assert pad == 100 and canvas == (hw[0] + 200, hw[1] + 200)
        gain = min(640 / canvas[0], 640 / canvas[1])
        resized = (round(canvas[0] * gain), round(canvas[1] * gain))
        assert shape == tuple(int(np.ceil(x / 32)) * 32 for x in resized)
        offset = np.array([round((shape[1]-resized[1])/2-.1), round((shape[0]-resized[0])/2-.1)])
        transforms[hw] = dict(input_shape=shape, canvas_shape=canvas, added_border=pad, gain=gain, offset=offset)
        transform_probes.append(dict(raw_hw=list(hw), input_shape=list(shape), canvas_shape=list(canvas),
            added_border=pad, gain=gain, offset_xy=offset, authenticated_metadata_cache=binding(path),
            cached_neck_or_old_predictions_used_as_new_model_input=False))
        del meta
    write_rows(PRIVATE / "REUSED_PREDICTIONS_SEALED.jsonl.gz", predictions)
    with (PRIVATE / "COMMON_INPUTS.npz").open("xb") as stream:
        np.savez_compressed(stream, ids=np.array(ids),
            q0=np.array([bases[(fid, 1)]["qFinal"] for fid in ids]),
            support=np.array([bases[(fid, 1)]["prediction_support"] for fid in ids], bool),
            raw_hw=np.array([bases[(fid, 1)]["raw_hw"] for fid in ids]),
            K=np.array([bases[(fid, 1)]["fixed_metadata"]["K"] for fid in ids]),
            xyz=np.array([bases[(fid, 1)]["fixed_metadata"]["dimensions_pnp_WH_D_m"] for fid in ids]),
            boxes=np.array([captures[(fid, 1)]["candidates"][captures[(fid, 1)]["selected_index"]]["box_xyxy"] for fid in ids]),
            selected_index=np.array([captures[(fid, 1)]["selected_index"] for fid in ids]),
            input_shape=np.array([transforms[tuple(bases[(fid, 1)]["raw_hw"])]["input_shape"] for fid in ids]),
            canvas_shape=np.array([transforms[tuple(bases[(fid, 1)]["raw_hw"])]["canvas_shape"] for fid in ids]),
            added_border=np.array([transforms[tuple(bases[(fid, 1)]["raw_hw"])]["added_border"] for fid in ids]),
            gain=np.array([transforms[tuple(bases[(fid, 1)]["raw_hw"])]["gain"] for fid in ids]),
            offset_xy=np.array([transforms[tuple(bases[(fid, 1)]["raw_hw"])]["offset"] for fid in ids]),
            image_paths=np.array([bases[(fid, 1)]["image"]["path"] for fid in ids]),
            image_sha256=np.array([bases[(fid, 1)]["image"]["sha256"] for fid in ids]))
    paths = [OLD_DOC / name for name in ("DATA_CONTRACT.json", "ROW_METRICS.jsonl.gz",
        "NEW_PREDICTION_SEAL.json", "NEW_POSE_SEAL.json", "NEW_PREDICTION_INPUT_LOCK.json",
        "PRECISION_PREDICTION_SEAL.json", "CURRENT_DETECTOR_MATCH_PARITY.json")]
    paths += [PRIOR / name for name in ("NEW_PREDICTIONS_SEALED.jsonl.gz", "NEW_POSES_SEALED.jsonl.gz",
        "RAW_NEW_PREDICTIONS.npz", "REFERENCE.json", "CURRENT_DETECTOR_CAPTURES.json", "CURRENT_DETECTOR_MATCHES.json")]
    paths += [TF_DOC / "PROTOCOL.json", TF_DOC / "PREDICTIONS.jsonl.gz",
        C.SOURCE / "scripts/research/pallet_dim_conditioned_p_v1/pose.py",
        C.SOURCE / "scripts/research/pallet_dim_conditioned_p_v1/eval_math.py",
        ROOT / "scripts/research/pallet_nonsevere_precision_20261011/evaluate.py"]
    paths += [C.SOURCE / x["path"] for x in public_configuration(config)["reused_code"]]
    paths += [C.SOURCE / public_configuration(config)[key]["path"] for key in ("method_lock", "source_calibration")]
    write(DOC / "INPUT_REUSE_CONTRACT.json", dict(status="PASS_NARROW_COMMON110_REUSE_CONTRACT",
        code=binding(Path(__file__)), frames=110, sessions=9, seeds=list(SEEDS), grade_counts=old["grade_counts"],
        id_sha256=old["id_sha256"], allowed_ids=ids, bindings=[binding(p) for p in paths],
        reused_methods=list(REUSED_METHODS), reused_rows=2200, expected_main_rows=3300,
        common_inputs=binding(PRIVATE / "COMMON_INPUTS.npz"),
        native_to_input_metadata_probes=transform_probes,
        reused_predictions=binding(PRIVATE / "REUSED_PREDICTIONS_SEALED.jsonl.gz"),
        reference_kind=old["reference_kind"], axis_evaluable_frames=109,
        prior_source_solver_and_corner_metric_unchanged=True, precision=dict(FP32=True, AMP=False,
            matmul_TF32=False, cudnn_TF32=False, inherited_neck="FP16 rounded then FP32 sampled; RGB_ONLY skips neck"),
        common_input_parity=input_parity, CV_cache_checks=cache_checks, CV_raw_cache_reusable=reusable,
        CV_action="REUSE_AUTHENTICATED_NATIVE_COORDINATES" if reusable else "RECOMPUTE110_CPU_NATIVE_FROM_CURRENT_COMMON_BASE",
        CV_old_q0_max_delta_px=max(r["q0_max_delta_px"] for r in cache_checks),
        CV_configuration=public_configuration(config),
        new_algorithms_or_tuning=0, new_GPU_inference=0, new_training_updates=0, new_F_calls=0,
        expected_new_unique_F_calls=dict(CVRANK=330, RGB_ONLY=440, total=770),
        expected_new_maximum_OpenCV_solve_refine_calls=4620,
        old_reference_data_reused_without_regeneration=True, protected_or_Severe_RGB_reads=0,
        historical_protocol="Read original prediction-only manifest metadata; only allowed110 records enter computations",
        wall_seconds=time.monotonic()-begun))
    print("INPUT_REUSE_READY", len(predictions), "CV_CACHE", reusable, flush=True)


def infer_seal(records, name, seed_dedup=False):
    pose = C.pose_api()
    output = []
    cache = {}
    begun = time.monotonic()
    with E.counter() as calls:
        for row in records:
            key = row["method"], row["id"]
            if seed_dedup and key in cache:
                previous = cache[key]
                for field in ("qFinal", "prediction_support", "fixed_metadata", "raw_hw", "detected"):
                    assert C.digest(row[field]) == C.digest(previous[field])
                actual = previous["actual_pose"]
            else:
                metadata = row["fixed_metadata"]
                q = np.asarray(row["qFinal"], float) if row.get("detected", True) else None
                actual = pose.infer(q, np.asarray(metadata["K"], float),
                    np.asarray(metadata["dimensions_pnp_WH_D_m"], float), False)
            result = dict(row, actual_pose=actual)
            output.append(result)
            cache[key] = result
    write_rows(PRIVATE / (name + "_POSES_SEALED.jsonl.gz"), output)
    unique_F = len(cache) if seed_dedup else len(records)
    write(DOC / (name + "_POSE_SEAL.json"), dict(status="SEALED_PREDICTION_ONLY_POSES_BEFORE_REFERENCE_SCORING",
        poses=binding(PRIVATE / (name + "_POSES_SEALED.jsonl.gz")), rows=len(output),
        actual_F_calls=unique_F, reused_identical_seed_pose_rows=len(output)-unique_F,
        OpenCV_calls=dict(calls), total_OpenCV_calls=sum(calls.values()),
        inference_reference_inputs=False, wall_seconds=time.monotonic()-begun))
    return output, dict(calls), unique_F


def score(records):
    # Called only after inference/seals; reference stays outside predictor/F.
    reference = C.read(PRIOR / "REFERENCE.json")
    matches = C.read(PRIOR / "CURRENT_DETECTOR_MATCHES.json")
    return [E.metric_row(r, reference[r["id"]], matches[f'{r["seed"]}:{r["id"]}']) for r in records]


def cvrank():
    begun = time.monotonic()
    plan = locked()
    contract = C.read(DOC / "INPUT_REUSE_CONTRACT.json")
    _, bases = common()
    methods = source_methods()
    assert C.digest(public_configuration(methods.method_configuration())) == C.digest(contract["CV_configuration"])
    raw_cache = {(r["id"], r["method"]): r for r in C.rows(TF_DOC / "PREDICTIONS.jsonl.gz", set(contract["allowed_ids"]))
                 if r["method"] == "CVRANK_NATIVE"}
    predictions = []
    counts = Counter()
    diagnostics = []
    for fid in contract["allowed_ids"]:
        base = bases[(fid, 1)]
        assert C.sha(C.SOURCE / base["image"]["path"]) == base["image"]["sha256"]
        image = cv2.imread(str(C.SOURCE / base["image"]["path"]), cv2.IMREAD_COLOR)
        assert image is not None and list(image.shape[:2]) == base["raw_hw"]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        counts["RGB_decodes"] += 1
        q0, support = np.asarray(base["qFinal"], float), np.asarray(base["prediction_support"], bool)
        if contract["CV_raw_cache_reusable"]:
            native = np.asarray(raw_cache[(fid, "CVRANK_NATIVE")]["native_points"], float)
            diagnostic = dict(reused_authenticated_native=True, algorithm_corner_calls=0)
            counts["reused_native_coordinates"] += 1
        else:
            native, diagnostic = methods.correct(gray, q0, support, "CVRANK")
            counts["CVRANK_image_calls"] += 1
            counts["CVRANK_corner_calls"] += diagnostic["algorithm_corner_calls"]
        h, w = gray.shape
        capped = methods.cap_points(q0, native, w, h, support)
        then_subpix, subpix_diag = E.subpix(gray, q0, native, support)
        counts["cornerSubPix_calls"] += subpix_diag["algorithm_corner_calls"]
        counts["SP_final_cap_active"] += subpix_diag["final_cap_active"]
        diagnostics.append(dict(id=fid, CVRANK=diagnostic, SUBPIX=subpix_diag))
        for sd in SEEDS:
            for name, points in zip(CV_METHODS, (native, capped, then_subpix)):
                predictions.append(dict(bases[(fid, sd)], method=name, qFinal=points,
                    subpix_input=native, status="OK", precision_basis="Shared authenticated FP32/TF32-off BASE; frozen source CVRank CPU; no new detector forward"))
    write_rows(PRIVATE / "CVRANK_DIAGNOSTICS.jsonl.gz", diagnostics)
    write_rows(PRIVATE / "CVRANK_PREDICTIONS_SEALED.jsonl.gz", predictions)
    write(DOC / "CVRANK_PREDICTION_SEAL.json", dict(status="SEALED_BEFORE_PRODUCTION_F_OR_REFERENCE_SCORING",
        predictions=binding(PRIVATE / "CVRANK_PREDICTIONS_SEALED.jsonl.gz"), rows=660,
        frames=110, seeds=list(SEEDS), seed_coordinates_exact_identical=True,
        algorithm_computed_once_per_frame=True, plan=plan, code=binding(Path(__file__)),
        diagnostics=binding(PRIVATE / "CVRANK_DIAGNOSTICS.jsonl.gz"), actual_image_calls=dict(counts),
        GT_or_visibility_or_teacher_inputs=False, source_settings_changed=False))
    inferred, calls, actual_F = infer_seal(predictions, "CVRANK", seed_dedup=True)
    scored = score(inferred)
    write_rows(DOC / "CVRANK_ROW_METRICS.jsonl.gz", scored)
    write(DOC / "CVRANK_COMPUTE_RECEIPT.json", dict(status="COMPLETE", rows=660, frames=110,
        native_cache_reused=contract["CV_raw_cache_reusable"], counts=dict(counts), actual_F_calls=actual_F,
        OpenCV_calls=calls, total_OpenCV_calls=sum(calls.values()), new_GPU_forwards=0,
        new_optimizer_updates=0, wall_seconds=time.monotonic()-begun,
        failures=sum(not r["available"] for r in scored), methods=list(CV_METHODS),
        duplicate_seed_rows="Alignment only; not independent rule training or independent images"))
    print("CVRANK_DONE", len(scored), actual_F, sum(calls.values()), flush=True)


def evaluate_rgb(path):
    begun = time.monotonic()
    locked()
    assert (DOC / "SYNTHETIC_SELECTION.json").exists(), "Synthetic-only selection must precede REAL scoring"
    _, bases = common()
    with np.load(path, allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    assert all(k in a for k in ("ids", "seeds", "points", "subpix_input"))
    assert a["points"].shape == a["subpix_input"].shape == (220, 9, 2)
    assert {(str(fid), int(sd)) for fid, sd in zip(a["ids"], a["seeds"])} == set(bases)
    seal = C.read(DOC / "RGB_ONLY_PREDICTION_SEAL.json")
    assert seal["predictions_sha256"] == C.sha(path), "RAW RGB-only points must be sealed before scoring"
    predictions = []
    for i, fid in enumerate(a["ids"]):
        b = bases[(str(fid), int(a["seeds"][i]))]
        if "q0" in a:
            assert np.array_equal(a["q0"][i], b["qFinal"], equal_nan=True)
        if "support" in a:
            assert np.array_equal(a["support"][i], b["prediction_support"])
        if "boxes" in a:
            cp = {(r["id"], r["seed"]): r for r in C.read(PRIOR / "CURRENT_DETECTOR_CAPTURES.json")}
            cap = cp[(str(fid), int(a["seeds"][i]))]
            assert np.array_equal(a["boxes"][i], cap["candidates"][cap["selected_index"]]["box_xyxy"])
        for array in (a["points"][i], a["subpix_input"][i]):
            assert np.array_equal(array[8], b["qFinal"][8], equal_nan=True)
            unusable = ~np.asarray(b["prediction_support"], bool)
            assert np.array_equal(array[unusable], np.asarray(b["qFinal"])[unusable], equal_nan=True)
        predictions.append(dict(b, method="RGB_ONLY", qFinal=a["points"][i], subpix_input=a["subpix_input"][i],
            detected=bool(a["detected"][i]) if "detected" in a else b["detected"], status="OK",
            precision_basis="RGB_ONLY FP32 AMP/TF32-off; identical frozen common BASE points/box/support; no p3/p4 input"))
    # Reuse existing image/coordinate algorithm, with exclusive new output writes.
    corrections, image_counts = E.generate_subpix(predictions, {"RGB_ONLY"})
    all_predictions = predictions + corrections
    write_rows(PRIVATE / "RGB_ONLY_ALL_PREDICTIONS_SEALED.jsonl.gz", all_predictions)
    write(DOC / "RGB_ONLY_FINAL_COORDINATE_SEAL.json", dict(status="SEALED_NATIVE_AND_SUBPIX_BEFORE_F_SCORING",
        predictions=binding(PRIVATE / "RGB_ONLY_ALL_PREDICTIONS_SEALED.jsonl.gz"), rows=440,
        input=binding(path), input_seal=binding(DOC / "RGB_ONLY_PREDICTION_SEAL.json"),
        RGB_counts=image_counts, subpix_start="uncapped RGB_ONLY expectation", final_cap="once; original BASE raw diagonal1%"))
    inferred, calls, actual_F = infer_seal(all_predictions, "RGB_ONLY")
    scored = score(inferred)
    write_rows(DOC / "RGB_ONLY_ROW_METRICS.jsonl.gz", scored)
    write(DOC / "RGB_ONLY_EVALUATION_RECEIPT.json", dict(status="COMPLETE", rows=440,
        actual_F_calls=actual_F, OpenCV_calls=calls, total_OpenCV_calls=sum(calls.values()),
        image_counts=image_counts, new_GPU_forwards=0, new_optimizer_updates=0,
        wall_seconds=time.monotonic()-begun, failures=sum(not r["available"] for r in scored)))
    print("RGB_ONLY_EVALUATION_DONE", len(scored), actual_F, sum(calls.values()), flush=True)


def finalize():
    begun = time.monotonic()
    locked()
    reused = [r for r in C.rows(OLD_DOC / "ROW_METRICS.jsonl.gz")
              if r["method"] in REUSED_METHODS and int(r["seed"]) in SEEDS]
    poses = [r for r in C.rows(PRIOR / "NEW_POSES_SEALED.jsonl.gz")
             if r["method"] in REUSED_METHODS and int(r["seed"]) in SEEDS]
    assert len(reused) == len(poses) == 2200
    # Independent metric replay uses already-saved actual poses: zero new F.
    recalculated = score(poses)
    by = {(r["method"], r["seed"], r["id"]): r for r in reused}
    assert len(by) == 2200
    checks = []
    for r in recalculated:
        before = by[(r["method"], r["seed"], r["id"])]
        checks.append(dict(method=r["method"], seed=r["seed"], id=r["id"],
                           all_metric_fields_exact=C.digest(before) == C.digest(r)))
    assert all(r["all_metric_fields_exact"] for r in checks), "Existing metric reuse parity failed"
    cv = list(C.rows(DOC / "CVRANK_ROW_METRICS.jsonl.gz"))
    rgb = list(C.rows(DOC / "RGB_ONLY_ROW_METRICS.jsonl.gz"))
    result = sorted(reused + cv + rgb, key=lambda r: (r["method"], r["seed"], r["id"]))
    assert len(result) == 3300
    methods = (*REUSED_METHODS, *CV_METHODS, *RGB_METHODS)
    assert len(methods) == 15
    ids = set(C.read(DOC / "INPUT_REUSE_CONTRACT.json")["allowed_ids"])
    assert {(r["method"], r["seed"], r["id"]) for r in result} == {(m, sd, fid) for m in methods for sd in SEEDS for fid in ids}
    write_rows(DOC / "ROW_METRICS.jsonl.gz", result)
    write(DOC / "EVALUATION_REUSE_VERIFICATION.json", dict(status="PASS", reused_rows=2200,
        independent_saved_pose_metric_recalculations=2200, all_metric_fields_exact=True,
        new_F_calls_for_reuse_verification=0, checks=checks, new_GPU_forwards=0,
        source_rows=binding(OLD_DOC / "ROW_METRICS.jsonl.gz"), output=binding(DOC / "ROW_METRICS.jsonl.gz"),
        total_rows=3300, methods=list(methods), seeds=list(SEEDS), independent_frames=110, sessions=9,
        new_coordinates_rows=1100, failures=sum(not r["available"] for r in result),
        per_route_denominator=220, reference_old_methods="N3/M0/M2/M4 remain in unchanged prior report only",
        wall_seconds=time.monotonic()-begun))
    combined = poses + list(C.rows(PRIVATE / "CVRANK_POSES_SEALED.jsonl.gz")) + list(C.rows(PRIVATE / "RGB_ONLY_POSES_SEALED.jsonl.gz"))
    write_rows(PRIVATE / "MAIN_POSES_SEALED.jsonl.gz", combined)
    print("MAIN_ROWS_DONE", len(result), len(methods), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "cvrank", "rgb", "finalize"))
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "prepare":
        prepare()
    elif args.phase == "cvrank":
        cvrank()
    elif args.phase == "rgb":
        assert args.predictions is not None
        evaluate_rgb(args.predictions.resolve())
    else:
        finalize()
