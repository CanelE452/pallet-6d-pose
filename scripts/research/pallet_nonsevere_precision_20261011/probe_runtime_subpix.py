"""CPU-only diagnosis of saved PoseFix runtime/accuracy coordinate differences.

Uses already sealed native predictions. No GPU forward, PnP, training, choice
of settings, or alteration of accuracy/runtime records occurs here.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_nonsevere_precision_20261011 import contract as C
from scripts.research.pallet_nonsevere_precision_20261011.evaluate import subpix


def main():
    begin = time.monotonic()
    cv2.setNumThreads(1)
    # Resolve the frozen method's namespace dependencies from the same source
    # tree used by InferenceEngine. This import does not construct a network.
    sys.path.insert(0, str(C.SOURCE))
    source = C.module("nonsevere_runtime_probe_native_methods",
        C.SOURCE / "scripts/research/pallet_training_free_compare_20261007_v1/methods.py")
    configuration = source.method_configuration()
    assert configuration["SUBPIX"]["winSize"] == [5, 5]
    assert configuration["SUBPIX"]["zeroZone"] == [-1, -1]
    assert configuration["SUBPIX"]["criteria"]["maxCount"] == 40
    assert configuration["SUBPIX"]["criteria"]["epsilon"] == .001
    assert configuration["CAP1"]["fraction_raw_image_diagonal"] == .01
    predictions = {(r["method"], r["id"]): r
        for r in C.rows(C.PRIVATE / "NEW_PREDICTIONS_SEALED.jsonl.gz")
        if r["seed"] == 1 and r["method"] in ("POSEFIX", "POSEFIX_SUBPIX")}
    latency_raw = list(C.rows(C.PRIVATE / "LATENCY_POSEFIX_CAPTURES.jsonl.gz"))
    latency_final = list(C.rows(C.PRIVATE / "LATENCY_POSEFIX_SUBPIX_CAPTURES.jsonl.gz"))
    assert len(latency_raw) == len(latency_final) == 90
    allowed = set(C.read(C.DOC / "DATA_CONTRACT.json")["allowed_ids"])
    assert {r["id"] for r in latency_raw} <= allowed
    images = {}
    for fid in sorted({r["id"] for r in latency_raw}):
        row = predictions["POSEFIX", fid]
        path = C.SOURCE / row["image"]["path"]
        assert C.sha(path) == row["image"]["sha256"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None and tuple(image.shape[:2]) == tuple(row["raw_hw"])
        images[fid] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    counter = Counter()
    original = cv2.cornerSubPix
    def counted(*args, **kwargs):
        counter["cornerSubPix"] += 1
        return original(*args, **kwargs)
    cv2.cornerSubPix = counted
    rows = []
    accuracy_replay = {}
    try:
        for fid, gray in images.items():
            r = predictions["POSEFIX", fid]
            native, _ = source.correct(gray, r["qFinal"], r["prediction_support"], "SUBPIX")
            final = source.cap_points(r["q0"], native, gray.shape[1], gray.shape[0], r["prediction_support"])
            accuracy_replay[fid] = final
        for captured in latency_raw:
            fid = captured["id"]
            r = predictions["POSEFIX", fid]
            gray = images[fid]
            initial = np.asarray(captured["points"], float)
            native, diagnostic = source.correct(gray, initial, r["prediction_support"], "SUBPIX")
            native_final = source.cap_points(r["q0"], native, gray.shape[1], gray.shape[0], r["prediction_support"])
            evaluator_final, _ = subpix(gray, r["q0"], initial, r["prediction_support"])
            same_input_difference = float(np.max(np.abs(native_final-evaluator_final)))
            assert same_input_difference <= 1e-9
            rows.append(dict(id=fid, repetition=captured["repetition"],
                same_input_runtime_vs_evaluator_max_abs_px=same_input_difference,
                saved_bare_PoseFix_latency_vs_accuracy_max_abs_px=float(np.max(np.abs(initial-np.asarray(r["qFinal"])))),
                saved_bare_input_CPU_SP_vs_accuracy_SP_max_abs_px=float(np.max(np.abs(native_final-np.asarray(predictions["POSEFIX_SUBPIX", fid]["qFinal"])))),
                source_attempted_corners=diagnostic["algorithm_corner_calls"]))
    finally:
        cv2.cornerSubPix = original
    replay_delta = max(float(np.max(np.abs(q-np.asarray(predictions["POSEFIX_SUBPIX", fid]["qFinal"]))))
        for fid, q in accuracy_replay.items())
    recorded_final_delta = max(float(np.max(np.abs(np.asarray(r["points"])
        - np.asarray(predictions["POSEFIX_SUBPIX", r["id"]]["qFinal"])))) for r in latency_final)
    assert replay_delta <= 1e-9
    C.write(C.DOC / "RUNTIME_SUBPIX_PARITY_PROBE.json", dict(
        status="SAME_INPUT_FUNCTION_PARITY_PASS_WITH_RECORDED_PIPELINE_NONEXACTNESS",
        code=C.binding(Path(__file__)), evaluator=C.binding(ROOT / C.PREFIX / "evaluate.py"),
        frozen_source_methods=C.binding(C.SOURCE / "scripts/research/pallet_training_free_compare_20261007_v1/methods.py"),
        saved_latency_raw=C.binding(C.PRIVATE / "LATENCY_POSEFIX_CAPTURES.jsonl.gz"),
        saved_latency_final=C.binding(C.PRIVATE / "LATENCY_POSEFIX_SUBPIX_CAPTURES.jsonl.gz"),
        accuracy_prediction_seal=C.binding(C.DOC / "NEW_PREDICTION_SEAL.json"),
        images=18, saved_bare_PoseFix_calls=90, saved_final_PoseFix_SP_calls=90,
        identical_native_settings=dict(winSize=[5,5], zeroZone=[-1,-1], COUNT=40, EPS=.001,
            float32_OpenCV_initial_points=True, center_and_missing_support_preserved=True,
            final_cap_fraction_raw_diagonal=.01, final_cap_once=True, final_cap_anchor="same BASE"),
        same_input_runtime_vs_evaluator_max_abs_px=max(r["same_input_runtime_vs_evaluator_max_abs_px"] for r in rows),
        same_accuracy_input_CPU_replay_max_abs_px=replay_delta,
        saved_bare_PoseFix_latency_vs_accuracy_max_abs_px=max(r["saved_bare_PoseFix_latency_vs_accuracy_max_abs_px"] for r in rows),
        saved_bare_input_CPU_SP_vs_accuracy_SP_max_abs_px=max(r["saved_bare_input_CPU_SP_vs_accuracy_SP_max_abs_px"] for r in rows),
        recorded_final_PoseFix_SP_latency_vs_accuracy_max_abs_px=recorded_final_delta,
        interpretation="Same native input gives identical source-runtime/evaluator SubPix+cap outputs. The saved small bare PoseFix coordinate differences can be amplified by the unchanged SubPix algorithm. This does not establish the origin of the GPU coordinate difference.",
        attribution_limit="Latency POSEFIX_SUBPIX captures contain final points only, not its exact pre-SubPix network output. CPU replay uses separately saved bare POSEFIX captures. CUDA nondeterminism is an unverified possible source, not a demonstrated cause.",
        accuracy_runtime_bit_exact_parity_claim=False, altered_existing_outputs=False,
        native_coordinates_or_RGB_public=False, raw_RGB_decodes=18,
        actual_cornerSubPix_calls=counter["cornerSubPix"], new_GPU_forwards=0,
        production_F_calls=0, training_updates=0, changed_hyperparameters=0,
        wall_seconds=time.monotonic()-begin, checks=rows))
    print("RUNTIME_SUBPIX_CPU_PROBE_DONE", replay_delta, recorded_final_delta,
        max(r["saved_bare_input_CPU_SP_vs_accuracy_SP_max_abs_px"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
