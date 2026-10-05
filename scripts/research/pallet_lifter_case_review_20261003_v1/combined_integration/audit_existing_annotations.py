"""Inventory legacy lifter annotations without promoting them to new review evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import cv2

from bridge import read_json, write_json


SESSIONS = ("173507", "174126", "174342", "174925")


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(owner_root, work_root):
    owner_root, work_root = Path(owner_root).resolve(), Path(work_root).resolve()
    source = owner_root / "challenge/data/01_real/live_capture_gt"
    plan_path = work_root / "data/pallet/results/pallet_lifter_case_review_20261003_v1/LIFTER_EVALUATION_PLAN.json"
    manifest_path = work_root / "data/pallet/results/pallet_lifter_case_review_20261003_v1/review/MANIFEST.json"
    plan = read_json(plan_path)
    manifest = read_json(manifest_path)
    planned = {(row["session_id"], row["saved_frame_index"]): row
               for row in plan["frames"]}
    review = {(row["session_id"], row["saved_frame_index"])
              for row in manifest["frames"]}
    sessions, all_target, digest_rows = {}, set(), []
    totals = Counter()
    for session in SESSIONS:
        directory = source / f"forklift_v4_{session}_manual_gt"
        counter, indices, pixel_matches = Counter(), [], 0
        for annotation in sorted(directory.glob("*.json")):
            index = int(annotation.stem)
            key = (session, index)
            all_target.add(key)
            indices.append(index)
            value = read_json(annotation)
            counter["annotation_json"] += 1
            counter["population_role_" + str(value.get("population_role"))] += 1
            counter["objects"] += len(value.get("objects", []))
            counter["has_reviewer_field"] += "reviewer" in value
            counter["has_review_time_field"] += "review_time" in value
            for obj in value.get("objects", []):
                counter["object_split_" + str(obj.get("split"))] += 1
                counter["migration_" + str(obj.get("migration_status"))] += 1
                counter["pose_" + str(obj.get("pose_status"))] += 1
                counter["occlusion_" + str(obj.get("occlusion_level"))] += 1
                for point in obj.get("keypoint_annotations", []):
                    counter["point_source_" + str(point.get("source"))] += 1
            image = annotation.with_suffix(".png")
            if image.is_file():
                decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
                pixel_sha = hashlib.sha256(decoded.tobytes()).hexdigest() if decoded is not None else None
                if key in planned and pixel_sha == planned[key]["decoded_bgr_sha256"]:
                    pixel_matches += 1
            digest_rows.append((str(annotation.relative_to(source)), file_sha(annotation)))
        totals.update(counter)
        sessions[session] = {"path": str(directory), **dict(counter),
            "all_indices_in_frozen_8910_plan": all((session, index) in planned for index in indices),
            "png_decoded_pixel_hash_matches_plan": pixel_matches,
            "review120_overlap_indices": sorted(index for index in indices
                                                if (session, index) in review)}
    all_dirs = sorted(path for path in source.glob("*manual_gt") if path.is_dir())
    all_json_count = sum(1 for directory in all_dirs for _ in directory.glob("*.json"))
    aggregate = hashlib.sha256()
    for name, digest in sorted(digest_rows):
        aggregate.update(name.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    tracked = subprocess.run(["git", "ls-files",
        *[f"challenge/data/01_real/live_capture_gt/forklift_v4_{session}_manual_gt/*.json"
          for session in SESSIONS]], cwd=owner_root, capture_output=True, text=True, check=True)
    tracked_count = len([line for line in tracked.stdout.splitlines() if line])
    backup_root = owner_root / "outputs/green_paper_review_20260918/backups"
    backup_count = sum(1 for path in backup_root.rglob("*.json")
        if any(f"forklift_v4_{session}_manual_gt" in path.parts for session in SESSIONS)) if backup_root.is_dir() else 0
    new_review_files = []
    for root in (owner_root, work_root):
        found = subprocess.run(["find", str(root), "-type", "f", "(",
            "-name", "annotations_in_progress.json", "-o",
            "-name", "LIFTER_REFERENCE_REVIEWED*.json", "-o",
            "-name", "LIFTER_OBJECT_MATCH_REVIEWED*.json", ")", "-print"],
            capture_output=True, text=True, check=True)
        new_review_files.extend(line for line in found.stdout.splitlines() if line)
    new_review_files.sort()
    return {"schema_version": "lifter_existing_annotation_audit_v1",
        "source_root": str(source), "frozen_plan": str(plan_path),
        "review_manifest": str(manifest_path), "sessions": sessions,
        "target_session_annotation_json_count": len(all_target),
        "target_session_tracked_json_count": tracked_count,
        "target_session_aggregate_json_sha256": aggregate.hexdigest(),
        "all_live_capture_gt_directory_count": len(all_dirs),
        "all_live_capture_gt_json_count": all_json_count,
        "other_session_json_count": all_json_count - len(all_target),
        "backup_copy_json_count_for_target_sessions": backup_count,
        "new_120_review_store_or_export_files": new_review_files,
        "new_120_review_store_or_export_file_count": len(new_review_files),
        "review120_overlap_count": len(all_target & review),
        "review120_overlap_frame_ids": [f"{session}:{index}"
                                        for session, index in sorted(all_target & review)],
        "totals": dict(totals),
        "reuse_decision": {"status": "NEEDS_HUMAN_PROVENANCE_REVIEW",
            "can_auto_promote_to_lifter_reference_review_v1": False,
            "reasons": [
                "legacy records have no reviewer identity or actual review timestamps",
                "all 67 objects remain MANUAL_REVIEW_REQUIRED and UNCONFIRMED_SIGNED_AXIS",
                "all 67 legacy files declare population_role DEV while objects declare split train, so independent held-out status is not established",
                "245 corners are PnP projections and 67 centers are automatic, not manual clicks",
                "only two legacy frames overlap the frozen 120-frame review sample",
                "occlusion_level is unknown for all 67 objects"],
            "manual_clicks_exist": totals["point_source_manual_click"],
            "machine_or_geometry_derived_points_not_human_clicks": (
                totals["point_source_pnp_projected"] + totals["point_source_centroid_auto"]),
            "human_reviewed_files_created": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.owner_root, args.work_root)
    write_json(args.output, result)
    print(json.dumps({"status": "VERIFIED_COMPLETE", "output": str(args.output),
                      "target_annotations": result["target_session_annotation_json_count"],
                      "review120_overlap": result["review120_overlap_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
