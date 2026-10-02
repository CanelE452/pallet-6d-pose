"""Separate 2D evaluation of N3 on the frozen GREEN0918_119 square set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluation.green_saved_labels_v1 import annotation_arrays

from . import common as C
from . import evaluation as E


SNAPSHOT = C.ROOT / "_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json"
SYMMETRY = C.ROOT / "_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json"
OBJECT_TYPE = "plastic_standard_110x110x15"


def truth_rows(*, in_frame_only: bool = False) -> list[dict]:
    snapshot = C.read(SNAPSHOT)
    symmetry = C.read(SYMMETRY)
    groups = [row for row in symmetry["objects"] if row["object_type"] == OBJECT_TYPE]
    if len(groups) != 1 or groups[0]["group_order"] != 4:
        raise RuntimeError("Square C4 symmetry contract drift")
    permutations = groups[0]["permutations"]
    rows = []
    for record in snapshot.get("records", []):
        annotation_path = C.ROOT / record["annotation"]["path"]
        if C.sha256(annotation_path) != record["annotation"]["sha256"]:
            raise RuntimeError(f"Square annotation hash drift: {record['id']}")
        annotation = C.read(annotation_path)
        gt, all_known = annotation_arrays(annotation)
        _, manual = annotation_arrays(annotation, True)
        h, w = record["original_hw"]
        inside = (np.isfinite(gt).all(-1)
                  & (gt[:, 0] >= 0) & (gt[:, 0] < w)
                  & (gt[:, 1] >= 0) & (gt[:, 1] < h))
        valid = manual & inside if in_frame_only else manual
        match_points = all_known & inside
        if not match_points.any():
            raise RuntimeError(f"No square matching points: {record['id']}")
        box = np.r_[gt[match_points].min(0), gt[match_points].max(0)]
        rows.append({
            "id": record["id"], "session": record["session"],
            "hw": [h, w], "gt": gt.tolist(), "valid": valid.tolist(),
            "box": box.tolist(), "permutations": permutations,
            "material": "plastic", "occlusion": "unclassified",
        })
    if len(rows) != 119 or len({row["id"] for row in rows}) != 119:
        raise RuntimeError("Square119 membership drift")
    expected = 600 if in_frame_only else 602
    if sum(np.asarray(row["valid"], bool)[:8].sum() for row in rows) != expected:
        raise RuntimeError("Square manual-corner denominator drift")
    return rows


def evaluate(backbone: str) -> dict:
    prediction_path = C.RAW / "predictions" / f"{backbone}_GREEN0918_119.json"
    payload = C.read(prediction_path)
    modes = {}
    for name, in_frame_only in (("manual_declared", False), ("manual_in_frame", True)):
        modes[name] = E.evaluate_payloads(
            [(None, payload)], truth_rows(in_frame_only=in_frame_only),
            backbone=backbone, include_pose=False, expected_frames=None)
    output = {
        "schema": "pallet_n3_completion_v3_square_evaluation_v1",
        "complete": all(value["complete"] for value in modes.values()),
        "backbone": backbone, "dataset": "GREEN0918_119",
        "dataset_status": "reused development 2D audit; not independent TEST",
        "frames": 119, "manual_declared_corners": 602,
        "manual_in_frame_corners": 600,
        "dimensions_wdh_m": [1.1, 1.1, .15], "material": "plastic",
        "physical_pose_reference": False, "pose_metrics": None,
        "dimension_effect_identifiable": False,
        "dimension_effect_note": (
            "All frames share one W,D,H vector; transfer of the full trained N3 "
            "package is measurable, but causal benefit of per-frame dimension variation is not."),
        "predictions": C.binding(prediction_path),
        "snapshot": C.binding(SNAPSHOT), "symmetry": C.binding(SYMMETRY),
        "modes": modes,
    }
    C.write(C.DOC / f"SQUARE_{backbone.upper()}_RESULTS.json", output, freeze=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=tuple(C.CONFIGS))
    arguments = parser.parse_args()
    result = evaluate(arguments.backbone)
    print(json.dumps({
        "complete": result["complete"], "backbone": result["backbone"],
        "frames": result["frames"],
        "manual_declared": {
            method: row["headline"] for method, row in
            result["modes"]["manual_declared"]["methods"].items()},
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
