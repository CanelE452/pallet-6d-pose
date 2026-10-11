"""Frozen production F and identical final SubPix for non-Severe native points.

Coordinates and poses are sealed before current metric scoring. Public rows
contain errors and outcomes, never RGB, predicted/reference coordinates or R/t.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_nonsevere_precision_20261011 import contract as C


@contextmanager
def counter():
    calls = Counter()
    original = {}
    for name in ("solvePnP", "solvePnPGeneric", "solvePnPRansac", "solvePnPRefineLM", "solvePnPRefineVVS"):
        if not hasattr(cv2, name):
            continue
        fn = getattr(cv2, name)
        original[name] = fn
        def counted(*args, _name=name, _fn=fn, **kwargs):
            calls[_name] += 1
            return _fn(*args, **kwargs)
        setattr(cv2, name, counted)
    try:
        yield calls
    finally:
        for name, fn in original.items():
            setattr(cv2, name, fn)


def subpix(gray, q0, initial, support):
    """Same fixed native algorithm and one Base-anchored final cap."""
    q0 = np.asarray(q0, float)
    q = np.array(initial, float, copy=True)
    support = np.asarray(support, bool)
    h, w = gray.shape
    algorithm_calls = 0
    outside = unsupported = exceptions = 0
    for k in range(8):
        if not support[k] or not np.isfinite(q[k]).all() or (q[k] == -1).all():
            unsupported += 1
            continue
        if not (0 <= q[k, 0] < w and 0 <= q[k, 1] < h) or min(h, w) < 15:
            outside += 1
            continue
        algorithm_calls += 1
        try:
            refined = cv2.cornerSubPix(gray, q[k].astype(np.float32).reshape(1, 1, 2).copy(),
                (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .001))
            refined = np.asarray(refined, float).reshape(2)
            if np.isfinite(refined).all() and 0 <= refined[0] < w and 0 <= refined[1] < h:
                q[k] = refined
        except (cv2.error, ValueError):
            exceptions += 1
    usable = support[:8] & np.isfinite(q0[:8]).all(-1) & ~(q0[:8] == -1).all(-1)
    assert np.isfinite(q[:8][usable]).all(), "Nonfinite supported correction"
    delta = q[:8] - q0[:8]
    length = np.linalg.norm(delta, axis=-1)
    factor = np.ones(8)
    moving = length > 0
    factor[moving] = np.minimum(1, .01 * math.hypot(h, w) / length[moving])
    final = q0.copy()
    final[:8][usable] = q0[:8][usable] + delta[usable] * factor[usable, None]
    final[8] = q0[8]
    return final, dict(algorithm_corner_calls=algorithm_calls, outside_initial=outside,
        unsupported=unsupported, exceptions=exceptions, final_cap_active=int((usable & (factor < 1)).sum()),
        cap_once=True, cap_reference="BASE", raw_subpix=q)


def runtime_inputs():
    with np.load(C.PRIVATE / "BASELINE_INPUTS.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return {str(fid): dict(id=str(fid), session=str(arrays["sessions"][i]), grade=str(arrays["grades"][i]),
        q0=arrays["q0"][i], prediction_support=arrays["support"][i], raw_hw=arrays["raw_hw"][i],
        fixed_metadata=dict(K=arrays["K"][i], dimensions_pnp_WH_D_m=arrays["xyz"][i]),
        image=dict(path=str(arrays["image_paths"][i]), sha256=str(arrays["image_sha256"][i])))
        for i, fid in enumerate(arrays["ids"])}


def generate_subpix(predictions, methods):
    grouped = {}
    for row in predictions:
        if row["method"] in methods:
            grouped.setdefault(row["id"], []).append(row)
    result = []
    counts = Counter()
    for fid, records in sorted(grouped.items()):
        image_meta = records[0]["image"]
        path = C.SOURCE / image_meta["path"]
        assert C.sha(path) == image_meta["sha256"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None and tuple(image.shape[:2]) == tuple(records[0]["raw_hw"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        counts["raw_RGB_decodes"] += 1
        for row in records:
            if row.get("detected", True):
                # Current common route starts from uncapped expectation. The
                # historical B0 rows have no such field and remain unchanged.
                start = row.get("subpix_input", row["qFinal"])
                q, diagnostic = subpix(gray, row["q0"], start, row["prediction_support"])
            else:
                q = np.asarray(row["qFinal"], float)
                diagnostic = dict(algorithm_corner_calls=0, missing_detection=True)
            counts["cornerSubPix_calls"] += diagnostic["algorithm_corner_calls"]
            counts["final_cap_active"] += diagnostic.get("final_cap_active", 0)
            name = {"BASE": "SUBPIX", "N3_DIM_SYM": "N3_THEN_SUBPIX"}.get(row["method"], row["method"] + "_SUBPIX")
            result.append(dict(row, method=name, qFinal=q, subpix_diagnostic=diagnostic))
    return result, dict(counts)


def infer_and_seal(predictions, phase):
    pose = C.pose_api()
    inferred = []
    begun = time.monotonic()
    with counter() as calls:
        for row in predictions:
            q = np.asarray(row["qFinal"], float) if row.get("detected", True) else None
            metadata = row["fixed_metadata"]
            actual = pose.infer(q, np.asarray(metadata["K"], float),
                np.asarray(metadata["dimensions_pnp_WH_D_m"], float), False)
            inferred.append(dict(row, actual_pose=actual))
    path = C.PRIVATE / (phase + "_POSES_SEALED.jsonl.gz")
    C.write_rows(path, inferred)
    C.write(C.DOC / (phase + "_POSE_SEAL.json"), dict(status="ALL_PREDICTION_ONLY_F_POSES_SEALED_BEFORE_CURRENT_METRIC_SCORING",
        rows=len(inferred), poses=C.binding(path), actual_F_calls=len(predictions), OpenCV_calls=dict(calls),
        total_OpenCV_calls=sum(calls.values()), wall_seconds=time.monotonic()-begun,
        K_array_type="float64 ndarray; numeric values unchanged", predictor_or_F_reference_inputs=False))
    return inferred, dict(calls)


def metric_row(row, reference, matched_override=None):
    truth = {key: (int(value) if key == "order" else np.asarray(value, float)) for key, value in reference["truth"].items()}
    metric = C.pose_api().metric((row["id"], row["actual_pose"], truth))
    matched = bool((reference["matched"] if matched_override is None else matched_override)
        and row.get("detected", True))
    corner = C.math_api().measure(row["qFinal"], reference["gt"], reference["valid"], reference["permutations"],
        row["raw_hw"], matched, bool(row.get("detected", True)))
    hyp = row["actual_pose"].get("selected_hypothesis")
    evaluable = reference["axis_status"] == "EVALUABLE"
    axis_wrong = None if not evaluable or hyp is None else int(hyp != reference["reference_hypothesis"])
    success = bool(metric.get("available") and metric["translation_cm"] < 5 and metric["rotation_deg"] < 5)
    confusion = bool(metric.get("available") and metric["rotation_deg"] > 45 and abs(metric["yaw_deg"]) >= 60)
    errors = corner.get("canonical_errors", [None] * 8)
    valid = corner.get("canonical_valid", [False] * 8)
    assert len(errors) == len(valid) == 8
    return dict(method=row["method"], id=row["id"], seed=int(row["seed"]), session=row["session"], grade=row["grade"],
        available=bool(metric.get("available")), pose=metric, strict_success=success, axis_wrong=axis_wrong,
        axis_reference_evaluable=evaluable, axis_status=reference["axis_status"], wd_confusion=confusion,
        corner_errors_px=errors, corner_valid=valid, corner_evaluable=bool(corner.get("evaluable")),
        corner_mean_px=corner.get("frame_mean_px"), corner_evaluation_branch=corner.get("branch"),
        matched=matched, detected=bool(row.get("detected", True)), selected_hyp=hyp,
        reference_kind=reference["reference_kind"], inference_reference_inputs=False,
        prediction_status=row.get("status", "OK"),
        precision_basis=row.get("precision_basis", "Authenticated historical frozen coordinates; original GPU settings; current CPU F replay"))


def parity(scored, inferred):
    old = {(r["method"], r["seed"], r["id"]): r for r in inferred if "historical_metric" in r}
    checks = []
    for row in scored:
        key = row["method"], row["seed"], row["id"]
        if key not in old:
            continue
        previous = old[key]
        a = row["pose"]
        b = previous["historical_metric"]
        deltas = {name: abs(a[name] - b[name]) for name in
            ("translation_cm", "rotation_deg", "yaw_deg", "IoU3D", "ADDsym_m", "ADDsym_normalized") if a.get("available") and b.get("available")}
        corner_before = previous["historical_corner"].get("canonical_errors", [None] * 8)
        corner_delta = max([abs(x-y) for x, y in zip(row["corner_errors_px"], corner_before) if x is not None and y is not None] or [0.])
        equal = a.get("available") == b.get("available") and row["selected_hyp"] == previous["historical_actual_pose"].get("selected_hypothesis")
        checks.append(dict(method=row["method"], seed=row["seed"], id=row["id"], availability_hypothesis_equal=equal,
            max_metric_delta=max(deltas.values(), default=0.), corner_max_delta_px=corner_delta,
            PASS=equal and max(deltas.values(), default=0.) <= 1e-7 and corner_delta <= 1e-7))
    assert len(checks) == 2310
    return checks


def baseline():
    begun = time.monotonic()
    C.verify_inputs()
    C.write(C.DOC / "BASELINE_EVALUATION_LOCK.json", dict(status="LOCKED_BEFORE_ANY_NEW_BASELINE_CORRECTION_OR_F",
        code=C.binding(Path(__file__)), contract=C.binding(C.DOC / "DATA_CONTRACT.json"),
        cached_predictions=C.binding(C.PRIVATE / "BASELINE_PREDICTIONS_SEALED.jsonl.gz"),
        methods=list(C.BASELINE_MAP.values()) + ["OLD_P_SUBPIX"], seeds=list(C.SEEDS), frames=110,
        expected_F_calls=2640, maximum_expected_OpenCV_calls=15840,
        cached_GPU_precision="Preserved original source settings; no claim of new TF32=False image-forward parity",
        current_CPU_F_exact=True, new_GPU_forwards=0, new_training_updates=0))
    predictions = list(C.rows(C.PRIVATE / "BASELINE_PREDICTIONS_SEALED.jsonl.gz"))
    correction, image_counts = generate_subpix(predictions, {"OLD_P"})
    assert len(correction) == 330
    # OLD_P_SUBPIX is a new route; old bare-method metrics must not be its parity target.
    for row in correction:
        for key in ("historical_actual_pose", "historical_metric", "historical_corner"):
            row.pop(key, None)
    all_predictions = predictions + correction
    C.write_rows(C.PRIVATE / "BASELINE_ALL_COORDINATES_SEALED.jsonl.gz", all_predictions)
    C.write(C.DOC / "BASELINE_SUBPIX_COORDINATE_SEAL.json", dict(status="SEALED_BEFORE_CURRENT_F_AND_SCORING",
        predictions=C.binding(C.PRIVATE / "BASELINE_ALL_COORDINATES_SEALED.jsonl.gz"), rows=len(all_predictions),
        reused_rows=2310, new_OLD_P_SUBPIX_rows=330, raw_RGB_counts=image_counts,
        whole_population=110, new_GPU_forwards=0, new_training_updates=0))
    inferred, calls = infer_and_seal(all_predictions, "BASELINE")
    reference = C.read(C.PRIVATE / "REFERENCE.json")
    scored = [metric_row(row, reference[row["id"]]) for row in inferred]
    checks = parity(scored, inferred)
    C.write(C.DOC / "BASELINE_PARITY.json", dict(status="PASS" if all(r["PASS"] for r in checks) else "FAIL_STOP",
        rows=len(checks), absolute_tolerance=1e-7, checks=checks,
        original_all_metric_max_delta=max(r["max_metric_delta"] for r in checks),
        original_corner_max_delta_px=max(r["corner_max_delta_px"] for r in checks),
        scope="Same saved nativecoordinates/currentfrozenCPU F and metric; not precision-change detector/head parity"))
    assert all(r["PASS"] for r in checks), "Frozen baseline path parity gate failed"
    C.write_rows(C.DOC / "B0_ROW_METRICS.jsonl.gz", scored)
    C.write(C.DOC / "BASELINE_COMPUTE_RECEIPT.json", dict(status="COMPLETE_CPU_BASELINE_REPLAY", rows=len(scored), frames=110,
        methods=sorted(set(r["method"] for r in scored)), seeds=list(C.SEEDS), actual_F_calls=len(inferred),
        OpenCV_calls=calls, total_OpenCV_calls=sum(calls.values()), new_GPU_forwards=0, new_training_updates=0,
        image_counts=image_counts, wall_seconds=time.monotonic()-begun, CPU_peak_memory="NOT_MEASURED",
        original_source_bindings_unchanged=True, precision_change_GPU_parity="Parent runtime phase required; separate from this CPU path check"))
    C.verify_inputs()
    print("BASELINE_EVALUATION_DONE", len(scored), sum(calls.values()), flush=True)


def load_new(path):
    runtime = runtime_inputs()
    with np.load(path, allow_pickle=False) as archive:
        a = {key: archive[key] for key in archive.files}
    qkey = "points" if "points" in a else "qFinal"
    assert all(key in a for key in ("ids", "seeds", "methods", qkey, "subpix_input"))
    assert a[qkey].shape == (len(a["ids"]), 9, 2)
    assert a["subpix_input"].shape == a[qkey].shape
    old_methods = ("BASE", "OLD_P", "N3_DIM_SYM", "POSEFIX")
    new_methods = ("M0", "M1", "M2", "M3", "M4")
    raw_base = {(str(fid), int(seed)): np.asarray(q, float)
        for fid, seed, method, q in zip(a["ids"], a["seeds"], a["methods"], a[qkey]) if str(method) == "BASE"}
    assert set(raw_base) == {(fid, seed) for fid in runtime for seed in C.SEEDS}
    result = []
    for i, raw_id in enumerate(a["ids"]):
        fid = str(raw_id)
        assert fid in runtime
        method = str(a["methods"][i])
        seed = int(a["seeds"][i])
        assert method in (*old_methods, *new_methods)
        assert seed in (C.SEEDS if method in old_methods else C.PRIMARY_SEEDS)
        q0 = raw_base[(fid, seed)]
        support = np.isfinite(q0).all(-1) & ~(q0 == -1).all(-1)
        r = dict(runtime[fid], q0=q0, prediction_support=support)
        q = np.asarray(a[qkey][i], float)
        uncapped = np.asarray(a["subpix_input"][i], float)
        assert np.array_equal(q[8], r["q0"][8], equal_nan=True), "Center changed"
        assert np.array_equal(q[~r["prediction_support"]], r["q0"][~r["prediction_support"]], equal_nan=True), "Missing support changed"
        assert np.array_equal(uncapped[8], r["q0"][8], equal_nan=True), "Uncapped center changed"
        assert np.array_equal(uncapped[~support], q0[~support], equal_nan=True), "Uncapped missing support changed"
        result.append(dict(r, method=method, seed=seed, qFinal=q,
            subpix_input=uncapped,
            detected=bool(a["detected"][i]) if "detected" in a else bool(support[:8].any()),
            status=str(a["status"][i]) if "status" in a else "OK",
            precision_basis="Fresh shared frozen YOLO FP32 inference; matmulTF32=False,cudnnTF32=False; same native source and precision for all accuracy routes"))
    keys = {(r["method"], r["seed"], r["id"]) for r in result}
    assert len(keys) == len(result)
    for method in sorted(set(r["method"] for r in result)):
        for seed in (C.SEEDS if method in old_methods else C.PRIMARY_SEEDS):
            assert {r["id"] for r in result if r["method"] == method and r["seed"] == seed} == set(runtime)
    assert set(old_methods) <= set(r["method"] for r in result)
    return sorted(result, key=lambda r: (r["method"], r["seed"], r["id"]))


def evaluate(path):
    begun = time.monotonic()
    C.verify_inputs()
    assert C.read(C.DOC / "BASELINE_PARITY.json")["status"] == "PASS"
    selection_path = C.DOC / "SYNTHETIC_SELECTION.json"
    assert selection_path.exists(), "Synthetic-only model selection must precede current REAL scoring"
    precision_path = C.DOC / "PRECISION_PREDICTION_SEAL.json"
    precision = C.read(precision_path)
    assert precision["predictions_sha256"] == C.sha(path)
    assert precision["FP32"] and not precision["AMP"] and not precision["matmul_TF32"] and not precision["cudnn_TF32"]
    match_path = C.DOC / "CURRENT_DETECTOR_MATCH_PARITY.json"
    match_gate = C.read(match_path)
    assert match_gate["status"] in ("MATCH_PARITY_PASS", "CURRENT_FROZEN_MATCH_POLICY_APPLIED_GLOBALLY")
    assert match_gate["rows"] == 330 and not match_gate["source_match_policy_changed"]
    assert match_gate["precision_prediction_seal"]["sha256"] == C.sha(precision_path)
    C.verify_binding(match_gate["current_capture"])
    seal_path = C.DOC / "NEW_PREDICTION_INPUT_LOCK.json"
    C.write(seal_path, dict(status="LOCKED_NEW_NATIVE_COORDINATES_BEFORE_CURRENT_REAL_CORRECTION_AND_F",
        code=C.binding(Path(__file__)), input_predictions=C.binding(path),
        frames=110, new_seeds=list(C.PRIMARY_SEEDS), existing_seeds=list(C.SEEDS),
        synthetic_selection=C.binding(selection_path), common_precision_seal=C.binding(precision_path),
        current_object_match_gate=C.binding(match_path),
        input_contract=C.binding(C.DOC / "DATA_CONTRACT.json"),
        no_reference_based_prediction_choice=True))
    predictions = load_new(path)
    correction, image_counts = generate_subpix(predictions, {r["method"] for r in predictions})
    all_predictions = predictions + correction
    C.write_rows(C.PRIVATE / "NEW_PREDICTIONS_SEALED.jsonl.gz", all_predictions)
    C.write(C.DOC / "NEW_PREDICTION_SEAL.json", dict(status="SEALED_ALL_NEW_NATIVE_AND_SUBPIX_ROUTES_BEFORE_F_SCORING",
        predictions=C.binding(C.PRIVATE / "NEW_PREDICTIONS_SEALED.jsonl.gz"), rows=len(all_predictions),
        models=sorted(set(r["method"] for r in predictions)), raw_RGB_counts=image_counts, no_GT_inputs=True,
        initial_corners="Fresh BASE for the same image/seed; same precision; shared by all methods",
        B0_historical_rows="Preserved separately; not merged into current common precision headline"))
    inferred, calls = infer_and_seal(all_predictions, "NEW")
    # Current matching remains a separate reference-side scoring input. It is
    # never carried by corrected coordinates or passed to production F.
    current_matches = C.read(C.verify_binding(match_gate["current_matches"]))
    reference = C.read(C.PRIVATE / "REFERENCE.json")
    scored = [metric_row(row, reference[row["id"]], current_matches[f'{row["seed"]}:{row["id"]}'])
        for row in inferred]
    C.write_rows(C.DOC / "ROW_METRICS.jsonl.gz", scored)
    C.write(C.DOC / "NEW_EVALUATION_RECEIPT.json", dict(status="COMPLETE_ALL_EXECUTED_ARMS_AND_SEEDS", rows=len(scored),
        frames=110, new_seeds=list(C.PRIMARY_SEEDS), existing_seeds=list(C.SEEDS), methods=sorted(set(r["method"] for r in scored)),
        actual_F_calls=len(inferred), OpenCV_calls=calls, total_OpenCV_calls=sum(calls.values()), image_counts=image_counts,
        new_GPU_forwards=0, new_training_updates=0, wall_seconds=time.monotonic()-begun,
        failed_pose_rows=sum(not r["available"] for r in scored), supported_only_case_filter=False,
        current_object_match_gate=C.binding(match_path), historical_match_reused_without_verification=False,
        model_selection="Parent synthetic VAL only; every executed arm reported here; REAL cannot pick newmodel"))
    C.verify_inputs()
    print("NEW_EVALUATION_DONE", len(scored), sum(calls.values()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("baseline", "new"))
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "baseline":
        baseline()
    else:
        assert args.predictions is not None
        evaluate(args.predictions.resolve())
