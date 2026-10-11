"""Authenticate the confidence-selected object, then apply the frozen IoU policy.

Prediction capture is sealed before current reference matching. No reference
value changes the selected detector candidate, keypoints, or production F.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_nonsevere_precision_20261011 import contract as C


def selected_array(path, allowed):
    """Decode only allowed records from the historical pretty-JSON input list."""
    result = {}
    buffer = None
    with Path(path).open() as stream:
        for line in stream:
            if line == "  {\n":
                buffer = [line]
            elif buffer is not None:
                buffer.append(line)
                if line in ("  },\n", "  }\n"):
                    text = "".join(buffer).strip().rstrip(",")
                    match = C.ID_PATTERN.search(text)
                    if match is not None and match.group(1) in allowed:
                        value = json.loads(text)
                        assert value["id"] not in result
                        result[value["id"]] = value
                    buffer = None
    assert set(result) == set(allowed)
    return result


def iou(a, b):
    """Exact frozen dev_evaluate.iou arithmetic, not a new match rule."""
    a, b = np.array(a), np.array(b)
    inter = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    return float(inter / max(np.maximum(a[2:] - a[:2], 0).prod()
        + np.maximum(b[2:] - b[:2], 0).prod() - inter, 1e-12))


def confidence_selection(value):
    candidates = value["candidates"]
    scores = np.array([r["score"] for r in candidates], float)
    assert np.isfinite(scores).all()
    expected = int(np.argmax(scores)) if candidates else None
    assert value["selected_index"] == expected, "Selected candidate is not the confidence argmax"
    if expected is None:
        return None
    candidate = candidates[expected]
    assert np.asarray(candidate["box_xyxy"]).shape == (4,)
    assert np.isfinite(candidate["box_xyxy"]).all()
    return candidate


def prepare():
    begun = time.monotonic()
    ids = C.read(C.DOC / "DATA_CONTRACT.json")["allowed_ids"]
    allowed = set(ids)
    historical_path = C.SOURCE / "data/pallet/results/pallet_posefix_replay_diagnosis_v1/INPUTS.json"
    inputs = selected_array(historical_path, allowed)
    reference = C.read(C.PRIVATE / "REFERENCE.json")
    baseline = {r["id"]: r for r in C.rows(C.PRIVATE / "BASELINE_PREDICTIONS_SEALED.jsonl.gz")
        if r["method"] == "BASE" and r["seed"] == 1}
    audit_path = C.PUBLIC / "_docs/experiments/pallet_paper_tables_245_20261011/INPUT_AUDIT.json"
    audit = {r["id"]: r for r in C.read(audit_path)["images"] if r["id"] in allowed}
    assert set(inputs) == set(reference) == set(baseline) == set(audit) == allowed
    old_selection = []
    match_reference = {}
    annotation_bindings = []
    for fid in ids:
        old = inputs[fid]
        assert old["split"] == "REAL_DEV" and old["pad"] == 0
        selected = confidence_selection(old["prediction"])
        assert np.array_equal(np.asarray(old["points"]), np.asarray(baseline[fid]["q0"]), equal_nan=True)
        if selected is not None:
            assert np.array_equal(np.asarray(old["box"]), np.asarray(selected["box_xyxy"]))
            assert np.array_equal(np.asarray(old["points"]), np.asarray(selected["keypoints_xy"]))
        ann = audit[fid]["annotation"]
        ann_path = C.SOURCE / ann["path"]
        assert C.sha(ann_path) == ann["sha256"]
        payload = C.read(ann_path)
        assert payload["schema_version"] == "real_pallet_gt_v2" and len(payload["objects"]) == 1
        xy = np.array([a["xy"] for a in payload["objects"][0]["keypoint_annotations"][:8]], float)
        assert xy.shape == (8, 2) and np.isfinite(xy).all()
        assert np.array_equal(xy, np.asarray(reference[fid]["gt"], float)[:8])
        box = np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
        overlap = None if selected is None else iou(selected["box_xyxy"], box)
        matched = selected is not None and overlap >= .5
        assert matched == reference[fid]["matched"]
        old_selection.append(dict(id=fid, selected_index=old["prediction"]["selected_index"],
            candidates=[dict(score=r["score"], box_xyxy=r["box_xyxy"]) for r in old["prediction"]["candidates"]],
            image_path=old["image"], image_sha256=old["image_sha256"],
            selection_rule="np.argmax detector confidence; no reference inputs"))
        match_reference[fid] = dict(gt_box_xyxy=box, old_matched=matched, old_selected_IoU=overlap,
            reference_box="GT-v2 all8 annotated xy envelope, including nonsupervised stored locations")
        annotation_bindings.append(dict(path=ann["path"], sha256=ann["sha256"]))
    C.write(C.PRIVATE / "EXPECTED_DETECTOR_SELECTION.json", old_selection)
    C.write(C.PRIVATE / "DETECTOR_MATCH_REFERENCE.json", match_reference)
    C.write(C.DOC / "DETECTOR_MATCH_REFERENCE_LOCK.json", dict(status="OLD_CONFIDENCE_SELECTION_AND_OBJECT_MATCH_AUTHENTICATED",
        code=C.binding(Path(__file__)), frames=110, old_selection=C.binding(C.PRIVATE / "EXPECTED_DETECTOR_SELECTION.json"),
        private_match_reference=C.binding(C.PRIVATE / "DETECTOR_MATCH_REFERENCE.json"),
        original_input_binding=C.binding(historical_path), annotation_bindings=annotation_bindings,
        source_policy=[C.binding(C.SOURCE / p) for p in (
            "scripts/research/pallet_posefix_replay_diagnosis_v1/run.py",
            "scripts/research/pallet_dim_conditioned_p_v1/dev_evaluate.py",
            "challenge/evaluation_v2/paper_real_eval.py")],
        source_lines=dict(confidence_selection="run.py:83-96", match="dev_evaluate.py:66-69 and run.py:96",
            GT_box="paper_real_eval.py:1634-1650"),
        selection_inputs="Detector confidence only; reference box is used only after prediction sealing for 2D scoring",
        match_rule="detected and exact frozen IoU(selected native box, GT all8 envelope)>=0.5",
        old_matched_frames=sum(r["old_matched"] for r in match_reference.values()),
        GT_boxes_public=False, protected_per_ID_access=0, GPU_forwards=0, production_F_calls=0,
        wall_seconds=time.monotonic()-begun))
    print("DETECTOR_EXPECTED_SELECTION_READY", len(old_selection), flush=True)


def validate():
    begun = time.monotonic()
    lock = C.read(C.DOC / "DETECTOR_MATCH_REFERENCE_LOCK.json")
    for name in ("old_selection", "private_match_reference"):
        C.verify_binding(lock[name])
    precision = C.read(C.DOC / "PRECISION_PREDICTION_SEAL.json")
    current_path = C.PRIVATE / "CURRENT_DETECTOR_CAPTURES.json"
    assert precision["detector_captures_sha256"] == C.sha(current_path)
    assert precision["predictions_sha256"] == C.sha(C.PRIVATE / "RAW_NEW_PREDICTIONS.npz")
    assert (C.DOC / "SYNTHETIC_SELECTION.json").exists()
    old = {r["id"]: r for r in C.read(C.PRIVATE / "EXPECTED_DETECTOR_SELECTION.json")}
    reference = C.read(C.PRIVATE / "DETECTOR_MATCH_REFERENCE.json")
    current = C.read(current_path)
    assert {(r["id"], int(r["seed"])) for r in current} == {(fid, seed) for fid in old for seed in C.SEEDS}
    assert len(current) == 330
    checks = []
    matches = {}
    for row in current:
        fid, seed = row["id"], int(row["seed"])
        selected = confidence_selection(row)
        before = confidence_selection(old[fid])
        overlap = None if selected is None else iou(selected["box_xyxy"], reference[fid]["gt_box_xyxy"])
        matched = selected is not None and overlap >= .5
        matches[f"{seed}:{fid}"] = matched
        checks.append(dict(id=fid, seed=seed, old_candidates=len(old[fid]["candidates"]),
            current_candidates=len(row["candidates"]), old_selected_index=old[fid]["selected_index"],
            current_selected_index=row["selected_index"], selected_index_unchanged=row["selected_index"] == old[fid]["selected_index"],
            confidence_argmax_verified=True,
            selected_box_max_abs_delta_px=None if selected is None or before is None else float(np.max(np.abs(np.asarray(selected["box_xyxy"])-before["box_xyxy"]))),
            selected_score_abs_delta=None if selected is None or before is None else abs(selected["score"]-before["score"]),
            old_selected_IoU=reference[fid]["old_selected_IoU"], current_selected_IoU=overlap,
            old_matched=reference[fid]["old_matched"], current_matched=matched,
            match_unchanged=matched == reference[fid]["old_matched"]))
    C.write(C.PRIVATE / "CURRENT_DETECTOR_MATCHES.json", matches)
    changes = sum(not r["match_unchanged"] for r in checks)
    C.write(C.DOC / "CURRENT_DETECTOR_MATCH_PARITY.json", dict(
        status="MATCH_PARITY_PASS" if changes == 0 else "CURRENT_FROZEN_MATCH_POLICY_APPLIED_GLOBALLY",
        rows=330, frames=110, code=C.binding(Path(__file__)), precision_prediction_seal=C.binding(C.DOC / "PRECISION_PREDICTION_SEAL.json"),
        reference_lock=C.binding(C.DOC / "DETECTOR_MATCH_REFERENCE_LOCK.json"), current_capture=C.binding(current_path),
        current_matches=C.binding(C.PRIVATE / "CURRENT_DETECTOR_MATCHES.json"),
        source_match_policy_changed=False, changed_match_rows=changes,
        changed_selected_index_rows=sum(not r["selected_index_unchanged"] for r in checks),
        current_matched_per_seed=dict(Counter(r["seed"] for r in checks if r["current_matched"])),
        all_accuracy_routes_share_same_image_seed_match=True, prediction_or_F_GT_inputs=False,
        current_prediction_seal_preceded_current_reference_matching=True,
        checks=checks, protected_per_ID_access=0, GPU_forwards=0, production_F_calls=0,
        wall_seconds=time.monotonic()-begun))
    print("CURRENT_DETECTOR_MATCH_DONE", changes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "validate"))
    args = parser.parse_args()
    prepare() if args.phase == "prepare" else validate()
