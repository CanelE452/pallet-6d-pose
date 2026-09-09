from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.evaluation.eval_dataset_status import refresh_after_annotation
from scripts.evaluation.eval_workspace import (
    WorkspaceError,
    compute_progress,
    load_frame_tag_overrides,
    load_frames,
    scaffold_workspace,
    sha256_file,
    update_frame_tags_csv_many,
)
from scripts.evaluation.incoming_promotion import promote_incoming_annotation


CAMERA_MATRIX = "608 0 326\n0 607 239\n0 0 1\n"


@dataclass(frozen=True)
class IncomingFixture:
    root: Path
    material: str
    lighting: str
    source_dir: Path
    selected_stem: str
    unselected_stem: str
    selected_image: Path
    unselected_image: Path
    selected_annotation: Path
    unselected_annotation: Path
    annotation_dir: Path
    destination_session: str
    destination_dir: Path
    destination_annotations: Path


def _annotation_payload(material: str, *, revision: int = 1) -> dict[str, object]:
    object_type = (
        "plastic_standard_110x130x11"
        if material == "plastic"
        else "wood_small_80x59x14"
    )
    return {
        "schema_version": "pallet_pose_test_annotation_v1",
        "revision": revision,
        "objects": [
            {
                "object_type": object_type,
                "projected_cuboid": [
                    [float(index + revision), float(index * 2 + revision)]
                    for index in range(8)
                ],
                "occlusion_level": "none",
                "truncation": {
                    "is_truncated": False,
                    "bbox_outside_fraction": 0.0,
                },
                "keypoint_annotations": [],
            }
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _incoming_fixture(
    tmp_path: Path,
    *,
    material: str,
    lighting: str,
    image_suffix: str = ".png",
) -> IncomingFixture:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)

    source_session = f"capture_{material}_{lighting}"
    source_dir = root / "incoming/sessions" / source_session
    source_rgb = source_dir / "rgb"
    source_rgb.mkdir(parents=True)
    _write_json(
        source_dir / "session.json",
        {
            "schema_version": "pallet_eval_incoming_capture_v1",
            "session_id": source_session,
            "workspace_scope": "INCOMING_UNREVIEWED",
            "population_role": None,
            "active_evaluation_member": False,
            "review_status": "UNREVIEWED_MIXED_CAPTURE",
            "object_type": "unknown",
            "lighting": lighting,
            "capture_protocol": "test_capture",
            "default_tags": {
                "occlusion": "unknown",
                "truncation": "unknown",
                "distance_bin": "unknown",
                "size_bin": "unknown",
                "elevation_bin": "unknown",
                "view_bin": "unknown",
            },
            "resolution": {"width": 640, "height": 480},
            "camera": {
                "intrinsics_quality": "PROVIDED_UNVERIFIED",
                "intrinsics_source": "test camera_info.json",
                "K": [
                    [608.0, 0.0, 326.0],
                    [0.0, 607.0, 239.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        },
    )
    (source_dir / "cam_K.txt").write_text(CAMERA_MATRIX, encoding="utf-8")

    selected_stem = "000010"
    unselected_stem = "000011"
    selected_image = source_rgb / f"{selected_stem}{image_suffix}"
    unselected_image = source_rgb / f"{unselected_stem}{image_suffix}"
    selected_image.write_bytes(
        f"selected-{material}-{lighting}-{image_suffix}".encode("ascii")
    )
    unselected_image.write_bytes(
        f"unselected-{material}-{lighting}-{image_suffix}".encode("ascii")
    )

    annotation_dir = root / "incoming/annotations" / f"{source_session}__{material}"
    selected_annotation = annotation_dir / f"{selected_stem}.json"
    unselected_annotation = annotation_dir / f"{unselected_stem}.json"
    _write_json(selected_annotation, _annotation_payload(material))
    # This is deliberately complete.  A targeted save hook must still leave it
    # in staging until this exact annotation is saved/reviewed.
    _write_json(unselected_annotation, _annotation_payload(material))
    update_frame_tags_csv_many(
        annotation_dir,
        {
            selected_image.name: {
                "distance_bin": "far",
                "elevation_bin": "high",
                "view_bin": "front",
                "occlusion": "medium",
                "truncation": "mild",
            },
            unselected_image.name: {
                "distance_bin": "near",
                "elevation_bin": "low",
                "view_bin": "rear",
                "occlusion": "none",
                "truncation": "none",
            },
        },
    )

    destination_session = f"{material}_{lighting}_01"
    destination_dir = root / "final/positive/sessions" / destination_session
    destination_annotations = (
        root / "final/positive/annotations" / destination_session
    )
    return IncomingFixture(
        root=root,
        material=material,
        lighting=lighting,
        source_dir=source_dir,
        selected_stem=selected_stem,
        unselected_stem=unselected_stem,
        selected_image=selected_image,
        unselected_image=unselected_image,
        selected_annotation=selected_annotation,
        unselected_annotation=unselected_annotation,
        annotation_dir=annotation_dir,
        destination_session=destination_session,
        destination_dir=destination_dir,
        destination_annotations=destination_annotations,
    )


@pytest.mark.parametrize(
    ("material", "lighting", "image_suffix"),
    [
        ("plastic", "day", ".png"),
        ("plastic", "night", ".jpg"),
        ("wood", "day", ".png"),
        ("wood", "night", ".png"),
    ],
)
def test_incoming_save_promotes_only_saved_stem_and_refreshes_progress(
    tmp_path: Path,
    material: str,
    lighting: str,
    image_suffix: str,
) -> None:
    fixture = _incoming_fixture(
        tmp_path,
        material=material,
        lighting=lighting,
        image_suffix=image_suffix,
    )
    root = fixture.root
    source_image = fixture.selected_image
    source_annotation = fixture.selected_annotation
    destination_dir = fixture.destination_dir
    destination_annotations = fixture.destination_annotations
    selected_stem = fixture.selected_stem
    unselected_stem = fixture.unselected_stem
    destination_session = fixture.destination_session

    source_before = {
        path: (sha256_file(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in (source_image, source_annotation)
    }
    line = refresh_after_annotation(root, source_annotation, source_image)

    destination_image = destination_dir / "rgb" / source_image.name
    destination_annotation = destination_annotations / source_annotation.name
    assert destination_image.read_bytes() == source_image.read_bytes()
    assert destination_annotation.read_bytes() == source_annotation.read_bytes()
    assert not destination_image.is_symlink()
    assert not destination_annotation.is_symlink()
    assert destination_image.stat().st_ino != source_image.stat().st_ino
    assert destination_annotation.stat().st_ino != source_annotation.stat().st_ino
    assert source_before == {
        path: (sha256_file(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in (source_image, source_annotation)
    }

    # A complete but not-currently-saved staging annotation must not be swept
    # into evaluation as a side effect of this save.
    assert not (destination_dir / "rgb" / f"{unselected_stem}{image_suffix}").exists()
    assert not (destination_annotations / f"{unselected_stem}.json").exists()

    tag_rows = load_frame_tag_overrides(destination_dir)
    assert set(tag_rows) == {selected_stem}
    assert tag_rows[selected_stem]["distance_bin"] == "far"
    assert tag_rows[selected_stem]["elevation_bin"] == "high"
    assert tag_rows[selected_stem]["view_bin"] == "front"
    assert tag_rows[selected_stem]["occlusion"] == "medium"
    assert tag_rows[selected_stem]["truncation"] == "mild"

    matching_rows = [
        row
        for row in load_frames(root)
        if row["session_id"] == destination_session
        and row["frame_id"].endswith(f"__{selected_stem}")
    ]
    assert len(matching_rows) == 1
    row = matching_rows[0]
    assert row["object_type"] == material
    assert row["lighting"] == lighting
    assert row["distance_bin"] == "far"
    assert row["elevation_bin"] == "high"
    assert row["occlusion"] == "medium"
    assert row["truncation"] == "mild"
    assert row["is_annotated"] == "true"
    assert row["image_path"].endswith(f"/{selected_stem}{image_suffix}")

    progress = compute_progress(load_frames(root))
    assert progress.positive_total == 1
    assert getattr(progress, material) == 1
    assert getattr(progress, lighting) == 1
    assert progress.far == 1
    assert progress.high == 1
    assert progress.occlusion == 1
    assert progress.truncation == 1
    assert "combined target 1/300 positive" in line
    report = (root / "reports/ANNOTATION_PROGRESS.md").read_text(encoding="utf-8")
    assert re.search(r"Positive total\s+1 / 300", report)

    destination_metadata = json.loads(
        (destination_dir / "session.json").read_text(encoding="utf-8")
    )
    assert destination_metadata["resolution"] == [640, 480]
    assert destination_metadata["promoted_from_sessions"] == [
        fixture.source_dir.name
    ]
    assert (destination_dir / "cam_K.txt").read_text(encoding="utf-8") == CAMERA_MATRIX


def test_resaving_existing_destination_updates_annotation_tags_and_manifest(
    tmp_path: Path,
) -> None:
    fixture = _incoming_fixture(
        tmp_path,
        material="plastic",
        lighting="day",
    )
    root = fixture.root
    source_image = fixture.selected_image
    source_annotation = fixture.selected_annotation
    annotation_dir = fixture.annotation_dir
    destination_dir = fixture.destination_dir
    destination_annotations = fixture.destination_annotations
    selected_stem = fixture.selected_stem
    destination_session = fixture.destination_session

    refresh_after_annotation(root, source_annotation, source_image)
    destination_annotation = destination_annotations / source_annotation.name
    first_hash = sha256_file(destination_annotation)

    _write_json(source_annotation, _annotation_payload("plastic", revision=2))
    update_frame_tags_csv_many(
        annotation_dir,
        {
            source_image.name: {
                "distance_bin": "near",
                "elevation_bin": "low",
                "view_bin": "left",
                "occlusion": "none",
                "truncation": "none",
            }
        },
    )
    line = refresh_after_annotation(root, source_annotation, source_image)

    assert sha256_file(destination_annotation) != first_hash
    assert destination_annotation.read_bytes() == source_annotation.read_bytes()
    tag_rows = load_frame_tag_overrides(destination_dir)
    assert tag_rows[selected_stem]["distance_bin"] == "near"
    assert tag_rows[selected_stem]["elevation_bin"] == "low"
    assert tag_rows[selected_stem]["view_bin"] == "left"
    assert tag_rows[selected_stem]["occlusion"] == "none"
    assert tag_rows[selected_stem]["truncation"] == "none"

    matching_rows = [
        row
        for row in load_frames(root)
        if row["session_id"] == destination_session
        and row["frame_id"].endswith(f"__{selected_stem}")
    ]
    assert len(matching_rows) == 1
    row = matching_rows[0]
    assert row["distance_bin"] == "near"
    assert row["elevation_bin"] == "low"
    assert row["view_bin"] == "left"
    assert row["occlusion"] == "none"
    assert row["truncation"] == "none"
    assert compute_progress(load_frames(root)).positive_total == 1
    assert "combined target 1/300 positive" in line


def test_deleting_incoming_annotation_disables_final_copy_and_decrements_progress(
    tmp_path: Path,
) -> None:
    fixture = _incoming_fixture(
        tmp_path,
        material="wood",
        lighting="night",
    )
    refresh_after_annotation(
        fixture.root,
        fixture.selected_annotation,
        fixture.selected_image,
    )
    destination_annotation = (
        fixture.destination_annotations / fixture.selected_annotation.name
    )
    destination_image = (
        fixture.destination_dir / "rgb" / fixture.selected_image.name
    )
    promoted_bytes = destination_annotation.read_bytes()
    assert compute_progress(load_frames(fixture.root)).positive_total == 1

    deleted_source = fixture.selected_annotation.with_suffix(".json.deleted")
    fixture.selected_annotation.rename(deleted_source)
    line = refresh_after_annotation(
        fixture.root,
        fixture.selected_annotation,
        fixture.selected_image,
        deleted=True,
    )

    deleted_destination = destination_annotation.with_suffix(".json.deleted")
    assert not destination_annotation.exists()
    assert deleted_destination.read_bytes() == promoted_bytes
    assert destination_image.is_file()
    assert deleted_source.is_file()
    assert compute_progress(load_frames(fixture.root)).positive_total == 0
    final_row = next(
        row
        for row in load_frames(fixture.root)
        if row["session_id"] == fixture.destination_session
        and row["frame_id"].endswith(f"__{fixture.selected_stem}")
    )
    assert final_row["is_annotated"] == "false"
    assert "combined target 0/300 positive" in line
    assert "incoming: 1 removed" in line
    report = (
        fixture.root / "reports/ANNOTATION_PROGRESS.md"
    ).read_text(encoding="utf-8")
    assert re.search(r"Positive total\s+0 / 300", report)


def test_same_stem_image_sha_collision_is_reported_without_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _incoming_fixture(
        tmp_path,
        material="plastic",
        lighting="day",
    )
    refresh_after_annotation(
        fixture.root,
        fixture.selected_annotation,
        fixture.selected_image,
    )
    destination_image = (
        fixture.destination_dir / "rgb" / fixture.selected_image.name
    )
    destination_annotation = (
        fixture.destination_annotations / fixture.selected_annotation.name
    )
    destination_image.write_bytes(b"different-pixels-already-at-destination")
    collision_sha = sha256_file(destination_image)
    old_annotation_sha = sha256_file(destination_annotation)
    _write_json(
        fixture.selected_annotation,
        _annotation_payload("plastic", revision=2),
    )

    with pytest.raises(WorkspaceError, match="image collision.*same frame stem"):
        promote_incoming_annotation(
            fixture.root,
            fixture.selected_annotation,
        )
    assert sha256_file(destination_image) == collision_sha
    assert sha256_file(destination_annotation) == old_annotation_sha

    line = refresh_after_annotation(
        fixture.root,
        fixture.selected_annotation,
        fixture.selected_image,
    )
    captured = capsys.readouterr()
    assert "incoming frame synchronization skipped" in captured.out
    assert "image collision for the same frame stem" in captured.out
    assert sha256_file(destination_image) == collision_sha
    assert sha256_file(destination_annotation) == old_annotation_sha
    assert "combined target 1/300 positive" in line


def test_synchronize_session_promotes_all_complete_annotations_with_batch_tags(
    tmp_path: Path,
) -> None:
    fixture = _incoming_fixture(
        tmp_path,
        material="wood",
        lighting="day",
    )
    update_frame_tags_csv_many(
        fixture.annotation_dir,
        {
            fixture.selected_image.name: {
                "distance_bin": "far",
                "elevation_bin": "mid",
                "occlusion": "heavy",
                "truncation": "medium",
            },
            fixture.unselected_image.name: {
                "distance_bin": "far",
                "elevation_bin": "mid",
                "occlusion": "heavy",
                "truncation": "medium",
            },
        },
    )

    line = refresh_after_annotation(
        fixture.root,
        fixture.selected_annotation,
        fixture.selected_image,
        synchronize_session=True,
    )

    for image, annotation in (
        (fixture.selected_image, fixture.selected_annotation),
        (fixture.unselected_image, fixture.unselected_annotation),
    ):
        assert (
            fixture.destination_dir / "rgb" / image.name
        ).read_bytes() == image.read_bytes()
        assert (
            fixture.destination_annotations / annotation.name
        ).read_bytes() == annotation.read_bytes()

    destination_tags = load_frame_tag_overrides(fixture.destination_dir)
    assert set(destination_tags) == {
        fixture.selected_stem,
        fixture.unselected_stem,
    }
    for stem in (fixture.selected_stem, fixture.unselected_stem):
        assert destination_tags[stem]["distance_bin"] == "far"
        assert destination_tags[stem]["elevation_bin"] == "mid"
        assert destination_tags[stem]["occlusion"] == "heavy"
        assert destination_tags[stem]["truncation"] == "medium"
    assert destination_tags[fixture.selected_stem]["view_bin"] == "front"
    assert destination_tags[fixture.unselected_stem]["view_bin"] == "rear"

    rows = [
        row
        for row in load_frames(fixture.root)
        if row["session_id"] == fixture.destination_session
    ]
    assert len(rows) == 2
    assert {row["distance_bin"] for row in rows} == {"far"}
    assert {row["elevation_bin"] for row in rows} == {"mid"}
    assert {row["occlusion"] for row in rows} == {"heavy"}
    assert {row["truncation"] for row in rows} == {"medium"}
    assert {row["view_bin"] for row in rows} == {"front", "rear"}
    progress = compute_progress(load_frames(fixture.root))
    assert progress.positive_total == 2
    assert progress.wood == 2
    assert progress.day == 2
    assert progress.far == 2
    assert progress.mid == 2
    assert progress.occlusion == 2
    assert progress.truncation == 2
    assert "combined target 2/300 positive" in line
    assert "incoming: 2 new, 2 tag rows synced" in line


def test_evaluation_refresh_modules_are_package_importable() -> None:
    promotion = importlib.import_module("scripts.evaluation.incoming_promotion")
    status = importlib.import_module("scripts.evaluation.eval_dataset_status")

    assert promotion.__package__ == "scripts.evaluation"
    assert callable(promotion.promote_incoming_annotation)
    assert callable(promotion.promote_annotated_incoming)
    assert status.__package__ == "scripts.evaluation"
    assert callable(status.refresh_after_annotation)
