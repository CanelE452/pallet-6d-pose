"""Build a prediction-blind static severity and corner-visibility review manifest."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.research.pallet_n3_completion_v3 import common as C
from scripts.research.pallet_n3_completion_v3 import reuse as R
from .audit import DOC, RAW, ROOT, GREEN_REL, bind, read, sha256, write_json


VISIBILITY_SOURCE_REL = Path(
    "data/pallet/results/pallet_verified_anchor_v1/metadata_conflict_qa/"
    "VERIFIED_LABELS_FINAL_PRIVATE.json")


def _overlay_for_annotation(path: Path) -> Path:
    return path.parent / "_overlays" / f"{path.stem}.png"


def build(source: Path) -> dict:
    dev_rows, _ = R.load_dev_context(source, include_pose=False)
    dev_by_id = {row["id"]: row for row in dev_rows}
    dev_manifest = read(C.DEV)
    completed = read(RAW / "COMPLETED319_LABEL_SIDECAR.json")
    labels = {row["id"]: row for row in completed["rows"]}
    visibility_path = source / VISIBILITY_SOURCE_REL
    visibility = read(visibility_path)
    known = {}
    for frame_index, corner_id in visibility["review_queue"]:
        frame = visibility["frames"][frame_index]
        corner = frame["corners"][corner_id]
        status = corner.get("status")
        if status in {"DIRECT_VISIBLE", "EXTERNAL_OCCLUDED"}:
            known[(frame["frame_id"], int(corner_id))] = {
                "status": status,
                "source": visibility["reference_version"],
                "annotator": frame.get("annotator", visibility.get("annotator")),
                "prior_prediction_exposure": "PNP_ASSISTED_HISTORY_UNKNOWN",
            }

    cases: list[dict[str, Any]] = []
    # Square severity is the only remaining frame-level severity task, so it is first.
    green_path = source / GREEN_REL
    green = read(green_path)
    for record in green["records"]:
        image = source / record["image"]["path"]
        annotation_path = source / record["annotation"]["path"]
        overlay = source / record["overlay"]["path"]
        annotation = read(annotation_path)["objects"][0]
        points = annotation["keypoint_annotations"][:8]
        cases.append({
            "case_id": f"GREEN0918::{record['id']}",
            "population": "GREEN0918",
            "frame_id": record["id"],
            "session": record["session"],
            "material": "plastic",
            "shape": "square",
            "image": bind(image, root=source),
            "overlay": bind(overlay, root=source),
            "annotation": bind(annotation_path, root=source),
            "frame_severity": {"status": "UNREVIEWED", "locked": False,
                               "source": None},
            "corners": [{
                "corner_id": index,
                "metric_reference": point.get("source") == "manual_click",
                "reference_source": point.get("source", "unknown"),
                "reference_in_frame": bool(point.get("in_frame", False)),
                "status": "UNREVIEWED",
                "locked": False,
                "source": None,
            } for index, point in enumerate(points)],
            "reference_overlay_exposure": "recorded by UI when opened",
        })

    for item in dev_manifest["items"]:
        frame_id = item["frame_id"]
        row = dev_by_id[frame_id]
        image = source / item["image_path"]
        annotation_path = source / item["gt_v2_path"]
        overlay = _overlay_for_annotation(annotation_path)
        if not overlay.is_file():
            raise FileNotFoundError(overlay)
        points = []
        for corner_id, present in enumerate(row["valid"][:8]):
            prior = known.get((frame_id, corner_id))
            points.append({
                "corner_id": corner_id,
                "metric_reference": bool(present),
                "reference_source": "locked_DEV_coordinate" if present else "NO_REFERENCE",
                "reference_in_frame": None,
                "status": prior["status"] if prior else "UNREVIEWED",
                "locked": prior is not None,
                "source": prior,
            })
        label = labels[frame_id]
        cases.append({
            "case_id": f"DEV319::{frame_id}",
            "population": "DEV319",
            "frame_id": frame_id,
            "session": row["session"],
            "material": row["material"],
            "shape": "rectangular",
            "image": bind(image, root=source),
            "overlay": bind(overlay, root=source),
            "annotation": bind(annotation_path, root=source),
            "frame_severity": {"status": label["severity"], "locked": True,
                               "source": "HUMAN_DIRECT_CLASS",
                               "review_id": label["review_id"],
                               "exported_at": label["exported_at"]},
            "corners": points,
            "reference_overlay_exposure": "recorded by UI when opened",
        })

    if len(cases) != 438 or len({case["case_id"] for case in cases}) != 438:
        raise ValueError("expected 438 unique static review cases")
    referenced = [point for case in cases for point in case["corners"]
                  if point["metric_reference"]]
    counts = Counter(point["status"] for point in referenced)
    manifest = {
        "schema": "pallet_static_registry_review_manifest_v1",
        "title": "Static pallet severity and corner visibility review",
        "prediction_blind": True,
        "model_outputs_included": False,
        "error_values_included": False,
        "severity_definitions": {
            "clean": "No externally blocking object; use UNKNOWN if the frame cannot be judged.",
            "moderate": "Human judgment under the recovered three-class protocol; no new numeric area threshold.",
            "severe": "Human judgment under the recovered three-class protocol; no new numeric area threshold.",
            "unknown": "Insufficient evidence or ambiguous cause.",
        },
        "visibility_states": ["DIRECT_VISIBLE", "EXTERNAL_OCCLUDED", "SELF_OCCLUDED",
                              "OUT_OF_FRAME", "OBJECT_ABSENT", "UNKNOWN"],
        "counts": {
            "cases": len(cases), "square_frame_severity_pending": 119,
            "DEV_frame_severity_complete": 319,
            "metric_reference_points": len(referenced),
            "visibility_preapproved": counts["DIRECT_VISIBLE"] + counts["EXTERNAL_OCCLUDED"],
            "visibility_pending": counts["UNREVIEWED"],
            "nonmetric_slots_optional": 438 * 8 - len(referenced),
        },
        "review_rules": [
            "Judge frame severity from the original RGB before opening the reference overlay.",
            "The overlay contains reference identities only; it contains no Base/N3 prediction or error.",
            "Opening the overlay is recorded as reference exposure.",
            "Do not infer visibility from coordinate presence, PnP projection, or validity flags.",
            "Existing 71 statuses remain provenance-locked; a new opinion must be a separate review revision.",
            "Missing-reference slots may be annotated, but do not enter the existing 2D denominator.",
        ],
        "source_root": str(source),
        "source_bindings": [bind(green_path, root=source), bind(C.DEV, root=ROOT),
                            bind(visibility_path, root=source),
                            bind(RAW / "COMPLETED319_LABEL_SIDECAR.json", root=ROOT)],
        "cases": cases,
    }
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path,
                        default=Path("/home/minjae/Documents/github/pallet-pose"))
    args = parser.parse_args(argv)
    manifest = build(args.source_root.expanduser().resolve())
    review = DOC / "review"
    write_json(review / "STATIC_REVIEW_MANIFEST.json", manifest)
    manifest_hash = sha256(review / "STATIC_REVIEW_MANIFEST.json")
    write_json(RAW / "STATIC_REVIEW_TEMPLATE.json", {
        "schema": "pallet_static_registry_review_response_v1",
        "manifest_sha256": manifest_hash,
        "reviewer": "",
        "review_status": "DRAFT_NOT_HUMAN_REVIEWED",
        "created_at": None,
        "updated_at": None,
        "responses": {},
        "note": "The CLI intentionally leaves all new human responses empty.",
    })
    print(json.dumps({"status": "READY_FOR_HUMAN_REVIEW",
                      "cases": manifest["counts"]["cases"],
                      "severity_pending": manifest["counts"]["square_frame_severity_pending"],
                      "visibility_pending": manifest["counts"]["visibility_pending"],
                      "manifest_sha256": manifest_hash}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
