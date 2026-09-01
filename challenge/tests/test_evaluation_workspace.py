from __future__ import annotations

import json
import re
import hashlib
from collections import Counter
from pathlib import Path

import pytest

from scripts.evaluation.eval_dataset_status import refresh_after_annotation
from scripts.evaluation.eval_workspace import (
    FRAME_COLUMNS,
    KNOWN_DEV_NEG_DUPLICATE_PATHS,
    KNOWN_DEV_NEG_DUPLICATE_SHA256,
    KNOWN_DEV_NEG_DUPLICATE_WORKSPACE_PATHS,
    REUSED_DEV_EVAL_ALIAS_NOTE,
    WorkspaceError,
    _workspace_readme,
    atomic_write_csv,
    condition_membership,
    compute_progress,
    copy2_verified,
    evaluation_population_views,
    infer_annotation_tags,
    load_frames,
    load_targets,
    metadata_unknown,
    read_csv,
    render_overlay_audit,
    render_progress_report,
    render_priority_report,
    resolve_frame_image_path,
    scaffold_workspace,
    sha256_file,
    validate_frozen_dev_evaluation_population,
    validate_active_image_sha_uniqueness,
    write_manifest_views,
)
from scripts.evaluation.import_existing_evaluation_data import (
    ExpectedCounts,
    _read_manifest,
    manifest_membership_sha256,
    resolve_legacy_image,
    validate_membership_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _valid_frame_row(
    frame_id: str,
    image_sha256: str,
    *,
    role: str = "DEV",
    positive: bool = True,
    paper_subset: str | None = None,
    image_path: str | None = None,
) -> dict[str, str]:
    row = {column: "" for column in FRAME_COLUMNS}
    positive_text = "true" if positive else "false"
    subset = paper_subset or (
        "FINAL_POSITIVE" if role == "FINAL" and positive
        else "FINAL_NEGATIVE" if role == "FINAL"
        else "DEV_PLASTIC_EXTRA" if positive
        else "DEV_NEG2689"
    )
    image_value = image_path or f"dev_existing/sessions/fixture/rgb/{frame_id}.png"
    row.update(
        {
            "frame_id": frame_id,
            "population_role": role,
            "paper_subset": subset,
            "controlled_eval_eligible": "false",
            "cross_shape_eval_eligible": "false",
            "session_id": frame_id.split("__", 1)[0],
            "object_type": "plastic" if positive else "none",
            "lighting": "unknown",
            "occlusion": "unknown",
            "truncation": "unknown",
            "distance_bin": "unknown",
            "size_bin": "unknown",
            "elevation_bin": "unknown",
            "view_bin": "unknown",
            "is_positive": positive_text,
            "is_annotated": "false",
            "image_path": image_value,
            "annotation_path": (
                f"dev_existing/annotations/fixture/{frame_id}.json" if positive else ""
            ),
            "overlay_path": (
                f"dev_existing/annotations/fixture/_overlays/{frame_id}.png"
                if positive else ""
            ),
            "source_dataset": "test_fixture",
            "source_image_path": image_value if not positive else "",
            "image_sha256": image_sha256,
            "storage_mode": "independent_copy" if positive else "source_reference_read_only",
        }
    )
    return row


def _small_memberships():
    plastic = [{"image": f"plastic/{index}.png"} for index in range(3)]
    controlled = plastic[:2]
    wood = [
        {"image_path": "wood/a.png", "session_id": "wood_a"},
        {"image_path": "wood/b.png", "session_id": "wood_b"},
    ]
    negative = [{"image": f"negative/{index}.png"} for index in range(2)]
    multishape = controlled + wood
    expected = ExpectedCounts(
        plastic=3,
        controlled_plastic=2,
        plastic_excluded=1,
        wood=2,
        wood_sessions=(("wood_a", 1), ("wood_b", 1)),
        multishape=4,
        negative=2,
    )
    return plastic, controlled, wood, negative, multishape, expected


def test_membership_guard_accepts_exact_union_and_rejects_drift() -> None:
    plastic, controlled, wood, negative, multishape, expected = _small_memberships()
    validate_membership_contract(
        plastic_items=plastic,
        controlled_items=controlled,
        wood_items=wood,
        negative_items=negative,
        multishape_items=multishape,
        expected=expected,
    )

    with pytest.raises(RuntimeError, match="count mismatch"):
        validate_membership_contract(
            plastic_items=plastic[:-1],
            controlled_items=controlled,
            wood_items=wood,
            negative_items=negative,
            multishape_items=multishape,
            expected=expected,
        )

    wrong_controlled = [controlled[0], {"image": "not_in_dev.png"}]
    with pytest.raises(RuntimeError, match="not a subset"):
        validate_membership_contract(
            plastic_items=plastic,
            controlled_items=wrong_controlled,
            wood_items=wood,
            negative_items=negative,
            multishape_items=multishape,
            expected=expected,
        )


def test_repository_audited_memberships_match_frozen_contract() -> None:
    manifest_root = REPO_ROOT / "challenge/real_gt_v2/manifests"

    def items(name: str):
        return json.loads((manifest_root / f"{name}.json").read_text())["items"]

    plastic = items("DEV_POS140")
    controlled = items("COMMON_DEV_POS128")
    wood = items("DEV_WOOD_POS45")
    negative = items("DEV_NEG2689")
    multishape_path = manifest_root / "COMMON_DEV_MULTISHAPE_POS.json"
    multishape = json.loads(multishape_path.read_text())["items"] if multishape_path.exists() else controlled + wood
    validate_membership_contract(
        plastic_items=plastic,
        controlled_items=controlled,
        wood_items=wood,
        negative_items=negative,
        multishape_items=multishape,
    )

    controlled_images = [REPO_ROOT / (item.get("image_path") or item["image"]) for item in controlled]
    wood_images = [REPO_ROOT / item["image_path"] for item in wood]
    controlled_sha = [sha256_file(path) for path in controlled_images + wood_images]
    assert len(controlled_sha) == 173
    assert len(set(controlled_sha)) == 173

    # The frozen negative population intentionally preserves this known exact
    # duplicate pair instead of silently changing membership.
    negative_by_id = {str(item["frame_id"]): REPO_ROOT / item["image"] for item in negative}
    assert sha256_file(negative_by_id["000238"]) == sha256_file(negative_by_id["000239"])


def test_all_noncolocated_legacy_images_resolve_by_explicit_session_mapping() -> None:
    hash_cache: dict[Path, str] = {}
    noncolocated = []
    for relative in (
        "challenge/data/01_real/eval_canonical",
        "challenge/data/01_real/manual_gt",
    ):
        for annotation in (REPO_ROOT / relative).rglob("*.json"):
            sibling_exists = any(
                annotation.with_suffix(suffix).is_file()
                for suffix in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
            )
            if not sibling_exists:
                noncolocated.append(annotation)
    assert len(noncolocated) == 62
    resolved = [resolve_legacy_image(REPO_ROOT, path, hash_cache)[0] for path in noncolocated]
    assert all(path is not None and path.is_file() for path in resolved)

def test_copy_is_independent_and_source_metadata_is_unchanged(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_image = source_dir / "frame.png"
    source_annotation = source_dir / "frame.json"
    source_image.write_bytes(b"not-a-decoded-image-but-stable-source-bytes")
    source_annotation.write_text('{"objects": []}\n', encoding="utf-8")
    before = {
        path: (sha256_file(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in (source_image, source_annotation)
    }
    file_count_before = len(list(source_dir.iterdir()))

    working_image = tmp_path / "workspace/rgb/frame.png"
    working_annotation = tmp_path / "workspace/annotations/frame.json"
    copy2_verified(source_image, working_image)
    copy2_verified(source_annotation, working_annotation)
    assert source_image.stat().st_ino != working_image.stat().st_ino
    assert source_annotation.stat().st_ino != working_annotation.stat().st_ino

    working_image.write_bytes(b"workspace edit")
    working_annotation.write_text('{"objects": [{"edited": true}]}\n', encoding="utf-8")
    after = {
        path: (sha256_file(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in (source_image, source_annotation)
    }
    assert after == before
    assert len(list(source_dir.iterdir())) == file_count_before


def test_scaffold_materializes_named_final_session_templates(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    positive = {
        path.name
        for path in (root / "final/positive/sessions").iterdir()
        if path.is_dir()
    }
    negative = {
        path.name
        for path in (root / "final/negative/sessions").iterdir()
        if path.is_dir()
    }
    assert positive == {
        "plastic_day_01",
        "plastic_night_01",
        "plastic_occ_01",
        "plastic_trunc_01",
        "wood_day_01",
        "wood_night_01",
        "wood_occ_01",
        "wood_trunc_01",
    }
    assert negative == {"negative_day_01", "negative_night_01"}
    for name in positive:
        session = root / "final/positive/sessions" / name
        assert (session / "rgb").is_dir()
        assert (session / "session.json").is_file()
        assert (session / "frame_tags.csv").is_file()
    for name in negative:
        session = root / "final/negative/sessions" / name
        assert (session / "rgb").is_dir()
        assert (session / "session.json").is_file()
    sessions = read_csv(root / "manifests/sessions.csv")
    assert len(sessions) == 10
    assert {row["notes"] for row in sessions} == {"EMPTY_SESSION_TEMPLATE"}


def test_read_only_source_reference_has_unambiguous_resolver(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    root = repo_root / "data/evaluation/pallet_eval_v1"
    root.mkdir(parents=True)
    source = repo_root / "data/pallet/raw_data/negative/rgb/000001.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"negative")
    row = {
        "image_path": "data/pallet/raw_data/negative/rgb/000001.png",
        "source_image_path": "data/pallet/raw_data/negative/rgb/000001.png",
        "storage_mode": "source_reference_read_only",
    }
    assert resolve_frame_image_path(root, row, repo_root=repo_root) == source


def test_dev_negative_session_manifest_points_to_source_without_fake_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    image_path = "data/pallet/raw_data/negative/rgb/000001.png"
    row = _valid_frame_row(
        "dev_negative__000001",
        hashlib.sha256(b"negative").hexdigest(),
        positive=False,
        image_path=image_path,
    )
    row["controlled_eval_eligible"] = "true"
    write_manifest_views(root, [row])
    negative = next(
        entry
        for entry in read_csv(root / "manifests/sessions.csv")
        if entry["session_id"] == "dev_negative"
    )
    assert negative["session_path"] == "data/pallet/raw_data/negative/rgb"
    assert negative["session_metadata_path"] == ""


def _write_final_fixture(root: Path) -> tuple[Path, Path]:
    session = root / "final/positive/sessions/wood_night_01"
    rgb = session / "rgb"
    rgb.mkdir(parents=True, exist_ok=True)
    image = rgb / "000001.png"
    image.write_bytes(b"fixture-image")
    (session / "session.json").write_text(
        json.dumps(
            {
                "session_id": "wood_night_01",
                "population_role": "FINAL",
                "object_type": "wood",
                "lighting": "night",
                "capture_protocol": "partial_occlusion",
                "default_tags": {"occlusion": "medium", "truncation": "mild"},
            }
        ),
        encoding="utf-8",
    )
    (session / "frame_tags.csv").write_text(
        "frame,distance_bin,size_bin,elevation_bin,view_bin\n"
        "000001.png,far,medium,high,unknown\n",
        encoding="utf-8",
    )
    annotation = root / "final/positive/annotations/wood_night_01/000001.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_type": "wood_small_80x59x14",
                        "occlusion_level": "unknown",
                        "truncation": {"is_truncated": True},
                        "keypoint_annotations": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return image, annotation


def test_final_progress_overlap_unknown_and_deletion_refresh(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    image, annotation = _write_final_fixture(root)

    line = refresh_after_annotation(root, annotation, image)
    assert "combined target 1/300 positive" in line
    frames = load_frames(root)
    progress = compute_progress(frames)
    assert progress.positive_total == 1
    assert progress.wood == 1
    assert progress.night == 1
    assert progress.occlusion == 1
    assert progress.truncation == 1
    assert progress.far == 1
    assert progress.high == 1
    assert progress.low == 0 and progress.mid == 0
    assert progress.unknown_metadata == 1
    progress_report = (root / "reports/ANNOTATION_PROGRESS.md").read_text()
    assert "UNKNOWN_METADATA" in progress_report
    assert (
        "Status               EMPTY — NO REGISTERED FINAL_EVAL ALIAS"
        in progress_report
    )
    priority_report = (root / "reports/NEXT_ANNOTATION_PRIORITY.md").read_text()
    assert "New annotation required   NO" in priority_report
    assert "NEEDS_METADATA" not in priority_report

    # A physical frame remains indexed after its workspace annotation is
    # deleted, but it no longer contributes to completed FINAL progress.
    annotation.rename(annotation.with_suffix(".json.deleted"))
    line = refresh_after_annotation(root, annotation, image, deleted=True)
    assert "combined target 0/300 positive" in line
    progress = compute_progress(load_frames(root))
    assert progress.positive_total == 0
    final_rows = read_csv(root / "manifests/FINAL_POSITIVE.csv")
    assert len(final_rows) == 1
    assert final_rows[0]["is_annotated"] == "false"


@pytest.mark.parametrize(
    ("distance", "legacy_size", "expected"),
    [
        ("far", "unknown", True),
        ("far", "large", True),
        ("near", "small", False),
        ("unknown", "small", False),
    ],
)
def test_far_membership_ignores_legacy_size(
    distance: str,
    legacy_size: str,
    expected: bool,
) -> None:
    matched = condition_membership({
        "distance_bin": distance,
        "size_bin": legacy_size,
    })
    assert ("far" in matched) is expected
    assert "far_small" not in matched


def test_legacy_size_unknown_does_not_make_active_metadata_unknown() -> None:
    row = {
        "object_type": "plastic",
        "lighting": "day",
        "occlusion": "none",
        "truncation": "none",
        "distance_bin": "near",
        "size_bin": "unknown",
        "elevation_bin": "mid",
        "view_bin": "front",
    }
    assert metadata_unknown(row) is False


def test_noncontrolled_dev_and_unverified_are_not_all_available(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    rows = []
    for frame_id, role in (("dev__1", "DEV"), ("legacy__1", "DEV_UNVERIFIED")):
        digest = hashlib.sha256(frame_id.encode()).hexdigest()
        row = _valid_frame_row(
            frame_id,
            digest,
            role=role,
            paper_subset="DEV_PLASTIC_EXTRA" if role == "DEV" else "NONE",
        )
        rows.append(row)
    atomic_write_csv(root / "manifests/frames.csv", rows, FRAME_COLUMNS)

    # Any path inside the workspace is sufficient to trigger a recount.
    marker = root / "dev_existing/annotations/dev/1.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    line = refresh_after_annotation(root, marker)
    assert "combined target 0/300 positive" in line
    # The audited-but-FT-overlap DEV row is retained for review but does not
    # enter a controlled evaluation union.
    all_available = read_csv(root / "manifests/ALL_AVAILABLE.csv")
    assert all_available == []
    audited = read_csv(root / "manifests/DEV_PLASTIC_AUDITED140.csv")
    assert [row["population_role"] for row in audited] == ["DEV"]
    assert len(load_frames(root)) == 2


def test_nonfinal_annotation_refreshes_explicit_gt_tags(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    annotation = root / "dev_existing/annotations/dev/000001.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_type": "wood_small_80x59x14",
                        "occlusion_level": "heavy",
                        "truncation": {"is_truncated": True},
                        "keypoint_annotations": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    image = root / "dev_existing/sessions/dev/rgb/000001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"dev-image")
    row = _valid_frame_row(
        "dev__000001",
        sha256_file(image),
        paper_subset="DEV_WOOD_POS45",
        image_path="dev_existing/sessions/dev/rgb/000001.png",
    )
    row.update(
        {
            "session_id": "dev",
            "object_type": "wood",
            "is_annotated": "true",
            "annotation_path": "dev_existing/annotations/dev/000001.json",
            "overlay_path": "dev_existing/annotations/dev/_overlays/000001.png",
            "annotation_sha256": sha256_file(annotation),
            "controlled_eval_eligible": "true",
            "cross_shape_eval_eligible": "true",
        }
    )
    atomic_write_csv(root / "manifests/frames.csv", [row], FRAME_COLUMNS)
    refresh_after_annotation(root, annotation)
    refreshed = load_frames(root)[0]
    assert refreshed["object_type"] == "wood"
    assert refreshed["occlusion"] == "heavy"
    assert refreshed["truncation"] == "mild"
    assert refreshed["annotation_sha256"] == sha256_file(annotation)


def test_global_sha_guard_allows_only_exact_frozen_negative_pair() -> None:
    rows = []
    for frame_id, image_path in KNOWN_DEV_NEG_DUPLICATE_PATHS.items():
        row = _valid_frame_row(
            frame_id,
            KNOWN_DEV_NEG_DUPLICATE_SHA256,
            positive=False,
            paper_subset="DEV_NEG2689",
            image_path=image_path,
        )
        row["controlled_eval_eligible"] = "true"
        row["source_dataset"] = "real_gt_v2_negative_audited"
        row["source_image_path"] = image_path
        row["source_image_sha256"] = KNOWN_DEV_NEG_DUPLICATE_SHA256
        rows.append(row)
    validate_active_image_sha_uniqueness(rows)

    materialized = [dict(row) for row in rows]
    for row in materialized:
        frame_id = row["frame_id"]
        row["storage_mode"] = "independent_copy"
        row["image_path"] = KNOWN_DEV_NEG_DUPLICATE_WORKSPACE_PATHS[frame_id]
        row["source_image_path"] = KNOWN_DEV_NEG_DUPLICATE_PATHS[frame_id]
    validate_active_image_sha_uniqueness(materialized)

    impostor = [dict(row) for row in rows]
    impostor[1]["paper_subset"] = "FINAL_NEGATIVE"
    impostor[1]["population_role"] = "FINAL"
    with pytest.raises(WorkspaceError, match="duplicate active image SHA"):
        validate_active_image_sha_uniqueness(impostor)

    third = _valid_frame_row(
        "plastic_final__copied",
        KNOWN_DEV_NEG_DUPLICATE_SHA256,
        role="FINAL",
        positive=True,
    )
    with pytest.raises(WorkspaceError, match="duplicate active image SHA"):
        validate_active_image_sha_uniqueness(rows + [third])


def test_refresh_rejects_dev_image_copied_into_final_before_manifest_write(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    payload = b"same-active-image"
    dev_image = root / "dev_existing/sessions/dev/rgb/000001.png"
    dev_image.parent.mkdir(parents=True)
    dev_image.write_bytes(payload)
    dev_row = _valid_frame_row(
        "dev__000001",
        sha256_file(dev_image),
        image_path="dev_existing/sessions/dev/rgb/000001.png",
    )
    atomic_write_csv(root / "manifests/frames.csv", [dev_row], FRAME_COLUMNS)

    final_image = root / "final/positive/sessions/plastic_day_01/rgb/000001.png"
    final_image.write_bytes(payload)
    planned_annotation = root / "final/positive/annotations/plastic_day_01/000001.json"
    with pytest.raises(WorkspaceError, match="DEV/FINAL copies or promotions"):
        refresh_after_annotation(root, planned_annotation, final_image)
    # The failed refresh must not publish the derived FINAL row.
    assert [row["frame_id"] for row in load_frames(root)] == ["dev__000001"]


def test_routine_refresh_rehashes_changed_final_rgb(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    image, annotation = _write_final_fixture(root)
    refresh_after_annotation(root, annotation, image)
    before = load_frames(root)[0]["image_sha256"]
    image.write_bytes(b"changed-final-image")
    refresh_after_annotation(root, annotation, image)
    after = load_frames(root)[0]["image_sha256"]
    assert after == sha256_file(image)
    assert after != before


def test_explicit_eval_views_freeze_dev_alias_and_deduplicate_all_available(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)

    dev_positive = _valid_frame_row(
        "dev__controlled",
        hashlib.sha256(b"dev-positive").hexdigest(),
        paper_subset="COMMON_DEV_PLASTIC_POS128",
    )
    dev_positive.update(
        {
            "controlled_eval_eligible": "true",
            "cross_shape_eval_eligible": "true",
            "is_annotated": "true",
            "annotation_sha256": hashlib.sha256(b"annotation").hexdigest(),
        }
    )
    dev_extra = _valid_frame_row(
        "dev__ft_overlap",
        hashlib.sha256(b"dev-extra").hexdigest(),
        paper_subset="DEV_PLASTIC_EXTRA",
    )

    duplicate_negative_sha = hashlib.sha256(b"duplicate-negative").hexdigest()
    dev_negatives = []
    for index in range(2):
        row = _valid_frame_row(
            f"dev_negative__{index}",
            duplicate_negative_sha,
            positive=False,
            paper_subset="DEV_NEG2689",
        )
        row["controlled_eval_eligible"] = "true"
        dev_negatives.append(row)

    final_eligible = _valid_frame_row(
        "final__eligible",
        hashlib.sha256(b"final-eligible").hexdigest(),
        role="FINAL",
    )
    final_eligible.update(
        {
            "controlled_eval_eligible": "true",
            "cross_shape_eval_eligible": "true",
            "is_annotated": "true",
            "annotation_sha256": hashlib.sha256(b"final-annotation").hexdigest(),
        }
    )
    final_qa_rejected = _valid_frame_row(
        "final__qa_rejected",
        hashlib.sha256(b"final-qa-rejected").hexdigest(),
        role="FINAL",
    )
    final_qa_rejected.update(
        {
            "is_annotated": "true",
            "annotation_sha256": hashlib.sha256(b"rejected-annotation").hexdigest(),
        }
    )
    final_unannotated = _valid_frame_row(
        "final__unannotated",
        hashlib.sha256(b"final-unannotated").hexdigest(),
        role="FINAL",
    )
    final_unannotated["controlled_eval_eligible"] = "true"
    final_negative = _valid_frame_row(
        "final_negative__1",
        hashlib.sha256(b"final-negative").hexdigest(),
        role="FINAL",
        positive=False,
    )
    final_negative["controlled_eval_eligible"] = "true"

    rows = [
        dev_positive,
        dev_extra,
        *dev_negatives,
        final_eligible,
        final_qa_rejected,
        final_unannotated,
        final_negative,
    ]
    write_manifest_views(root, rows)

    assert len(read_csv(root / "manifests/DEV_PLASTIC_AUDITED140.csv")) == 2
    assert len(read_csv(root / "manifests/DEV_EVAL_POSITIVE.csv")) == 1
    assert len(read_csv(root / "manifests/DEV_EVAL_NEGATIVE.csv")) == 2
    assert len(read_csv(root / "manifests/FINAL_POSITIVE.csv")) == 3
    final_eval_positive = read_csv(root / "manifests/FINAL_EVAL_POSITIVE.csv")
    assert [row["frame_id"] for row in final_eval_positive] == ["dev__controlled"]
    dev_alias = final_eval_positive[0]
    assert dev_alias["population_role"] == "DEV"
    assert dev_alias["notes"] == REUSED_DEV_EVAL_ALIAS_NOTE

    final_eval_negative = read_csv(root / "manifests/FINAL_EVAL_NEGATIVE.csv")
    assert len(final_eval_negative) == 2
    negative_aliases = [
        row for row in final_eval_negative if row["population_role"] == "DEV"
    ]
    assert len(negative_aliases) == 2
    assert all(
        row["notes"] == REUSED_DEV_EVAL_ALIAS_NOTE for row in negative_aliases
    )
    assert len(read_csv(root / "manifests/ALL_AVAILABLE_POSITIVE.csv")) == 2
    assert len(read_csv(root / "manifests/ALL_AVAILABLE_NEGATIVE.csv")) == 2
    assert len(read_csv(root / "manifests/ALL_AVAILABLE.csv")) == 4
    progress = compute_progress(rows)
    assert progress.positive_total == 2
    assert progress.negative == 2


def test_eval_population_fails_closed_on_duplicate_dev_positive_sha() -> None:
    digest = hashlib.sha256(b"duplicate-positive").hexdigest()
    rows = []
    for index in range(2):
        row = _valid_frame_row(
            f"dev__{index}",
            digest,
            paper_subset="COMMON_DEV_PLASTIC_POS128",
        )
        row["controlled_eval_eligible"] = "true"
        rows.append(row)
    with pytest.raises(WorkspaceError, match="duplicate image SHA256"):
        evaluation_population_views(rows)


def test_eval_population_fails_closed_on_positive_negative_sha_overlap() -> None:
    digest = hashlib.sha256(b"polarity-overlap").hexdigest()
    positive = _valid_frame_row(
        "dev__positive",
        digest,
        paper_subset="COMMON_DEV_PLASTIC_POS128",
    )
    positive["controlled_eval_eligible"] = "true"
    negative = _valid_frame_row(
        "dev_negative__1",
        digest,
        positive=False,
        paper_subset="DEV_NEG2689",
    )
    negative["controlled_eval_eligible"] = "true"
    with pytest.raises(WorkspaceError, match="positive/negative populations overlap"):
        evaluation_population_views([positive, negative])


def test_progress_report_orders_dev_alias_physical_final_and_all_available(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    dev = _valid_frame_row(
        "dev__controlled",
        hashlib.sha256(b"dev").hexdigest(),
        paper_subset="COMMON_DEV_PLASTIC_POS128",
    )
    dev.update(
        {
            "controlled_eval_eligible": "true",
            "cross_shape_eval_eligible": "true",
            "is_annotated": "true",
            "annotation_sha256": hashlib.sha256(b"annotation").hexdigest(),
            "lighting": "day",
            "truncation": "none",
            "elevation_bin": "low",
        }
    )
    overlay = root / dev["overlay_path"]
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"overlay")

    report = render_progress_report(root, [dev], load_targets(root))
    assert report.index("# DEV evaluation population") < report.index(
        "# FINAL_EVAL alias status"
    ) < report.index(
        "# Combined evaluation target progress"
    ) < report.index("# All available evaluation")
    assert "Combined positive       1 / 173" in report
    assert "Annotated positive      1 / 173" in report
    assert "Review overlays         1 / 173" in report
    assert "Lighting tagged         1 / 173" in report
    assert "Occlusion tagged        0 / 173" in report
    assert "Elevation tagged        1 / 173" in report
    assert "Status               READY — REUSED DEV_EVAL, NOT HELD OUT" in report
    assert "FINAL_EVAL positive     1" in report
    # 2026-09-01 부터 minimum/preferred 를 함께 보인다 (한 줄 300 만 보고
    # domain experiment 진척으로 오해하지 않도록).
    assert re.search(r"Current positive\s+1 / 400 preferred", report)
    assert re.search(r"\s+1 / 300 minimum", report)
    assert "ALL positive            1" in report


def test_overlay_audit_names_reused_alias_and_physical_final_separately(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    dev = _valid_frame_row(
        "dev__controlled",
        hashlib.sha256(b"dev-overlay").hexdigest(),
        paper_subset="COMMON_DEV_PLASTIC_POS128",
    )
    dev.update(
        {
            "controlled_eval_eligible": "true",
            "is_annotated": "true",
            "annotation_sha256": hashlib.sha256(b"annotation").hexdigest(),
        }
    )
    overlay = root / dev["overlay_path"]
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"overlay")

    physical_final = _valid_frame_row(
        "final__controlled",
        hashlib.sha256(b"physical-final-overlay").hexdigest(),
        role="FINAL",
    )
    physical_final.update(
        {
            "controlled_eval_eligible": "true",
            "is_annotated": "true",
            "annotation_sha256": hashlib.sha256(b"final-annotation").hexdigest(),
        }
    )
    final_overlay = root / physical_final["overlay_path"]
    final_overlay.parent.mkdir(parents=True, exist_ok=True)
    final_overlay.write_bytes(b"final-overlay")

    report = render_overlay_audit(root, [dev, physical_final])
    assert "FINAL_EVAL_ALIAS                  1          1         0" in report
    assert "PHYSICAL_FINAL                    1          1         0" in report
    assert "\nFINAL " not in report


def test_imported_workspace_frozen_dev_views_match_population_contract() -> None:
    root = REPO_ROOT / "data/evaluation/pallet_eval_v1"
    frames = load_frames(root)
    views = evaluation_population_views(frames)
    validate_frozen_dev_evaluation_population(views)
    assert len(views["DEV_PLASTIC_AUDITED140"]) == 140
    assert len(views["DEV_EVAL_POSITIVE"]) == 173
    assert len({row["image_sha256"] for row in views["DEV_EVAL_POSITIVE"]}) == 173
    assert len(views["DEV_EVAL_NEGATIVE"]) == 2689
    assert len({row["image_sha256"] for row in views["DEV_EVAL_NEGATIVE"]}) == 2688
    assert len(views["FINAL_EVAL_POSITIVE"]) == 173
    assert len(views["FINAL_EVAL_NEGATIVE"]) == 2689
    assert len({row["image_sha256"] for row in views["FINAL_EVAL_NEGATIVE"]}) == 2688
    for population in ("FINAL_EVAL_POSITIVE", "FINAL_EVAL_NEGATIVE"):
        assert all(row["population_role"] == "DEV" for row in views[population])
        assert all(row["notes"] == REUSED_DEV_EVAL_ALIAS_NOTE for row in views[population])
    assert len(views["ALL_AVAILABLE_NEGATIVE"]) == 2688


def test_repository_combined_target_progress_matches_all_available_views() -> None:
    root = REPO_ROOT / "data/evaluation/pallet_eval_v1"
    frames = load_frames(root)
    views = evaluation_population_views(frames)
    progress = compute_progress(frames)
    assert progress.positive_total == len(views["ALL_AVAILABLE_POSITIVE"])
    assert progress.negative == len(views["ALL_AVAILABLE_NEGATIVE"])
    assert progress.plastic + progress.wood == progress.positive_total
    # Frame condition tags are intentionally edited in this repository, so do
    # not freeze their live counters to a historical snapshot.
    for count in (
        progress.clean,
        progress.occlusion,
        progress.truncation,
        progress.far,
        progress.low,
        progress.mid,
        progress.high,
        progress.unknown_metadata,
    ):
        assert 0 <= count <= progress.positive_total


def test_workspace_execution_alias_matches_registered_evaluator_pair() -> None:
    root = REPO_ROOT / "data/evaluation/pallet_eval_v1"
    manifest_root = REPO_ROOT / "challenge/real_gt_v2/manifests"
    positive_rows = read_csv(root / "manifests/FINAL_EVAL_POSITIVE.csv")
    negative_rows = read_csv(root / "manifests/FINAL_EVAL_NEGATIVE.csv")
    positive_items = json.loads(
        (manifest_root / "COMMON_DEV_MULTISHAPE_POS.json").read_text("utf-8")
    )["items"]
    negative_items = json.loads(
        (manifest_root / "DEV_NEG2689.json").read_text("utf-8")
    )["items"]

    assert len(positive_rows) == len(positive_items) == 173
    assert len(negative_rows) == len(negative_items) == 2689
    assert Counter(row["source_annotation_path"] for row in positive_rows) == Counter(
        item.get("gt_v2_path") or item["label"] for item in positive_items
    )
    # The workspace annotation is intentionally editable.  Its current hash
    # may diverge from the immutable imported source after a reviewed manual
    # correction; both hashes must instead verify their own declared paths.
    for row in positive_rows:
        assert sha256_file(root / row["annotation_path"]) == row["annotation_sha256"]
        assert (
            sha256_file(REPO_ROOT / row["source_annotation_path"])
            == row["source_annotation_sha256"]
        )
    assert Counter(row["image_sha256"] for row in positive_rows) == Counter(
        sha256_file(REPO_ROOT / (item.get("image_path") or item["image"]))
        for item in positive_items
    )
    assert Counter(row["source_image_path"] for row in negative_rows) == Counter(
        item.get("image_path") or item["image"] for item in negative_items
    )
    assert all(
        row["source_image_sha256"] == row["image_sha256"]
        for row in negative_rows
    )
    assert len({row["image_sha256"] for row in negative_rows}) == 2688


def test_repository_workspace_contract_declares_explicit_eval_populations() -> None:
    root = REPO_ROOT / "data/evaluation/pallet_eval_v1"
    contract = json.loads((root / "DATASET_CONTRACT.json").read_text("utf-8"))
    targets = load_targets(root)

    assert targets["progress_population"] == "ALL_AVAILABLE"
    # 2026-09-01 domain 재설계로 condition target 상향 (DATASET_TARGETS.json)
    assert targets["minimum_condition_coverage"]["far"] == 80
    assert "far_small" not in targets["minimum_condition_coverage"]
    assert contract["condition_queries"]["far"] == "distance_bin == far"
    assert "far_small" not in contract["condition_queries"]
    assert contract["invariants"]["size_bin_is_legacy_compatibility_only"] is True
    assert "small" in contract["allowed_values"]["size_bin"]
    assert contract["invariants"]["evaluation_target_progress_uses_all_available"] is True
    assert (
        contract["invariants"]["evaluation_target_progress_sha256_deduplicated"]
        is True
    )
    assert "dev_is_never_counted_toward_final_targets" not in contract["invariants"]

    populations = contract["evaluation_populations"]
    assert set(populations) == {"DEV_EVAL", "FINAL_EVAL", "ALL_AVAILABLE"}
    assert populations["DEV_EVAL"]["held_out_final"] is False
    assert populations["FINAL_EVAL"]["held_out_final"] is False
    assert populations["FINAL_EVAL"]["reuses_dev_eval"] is True
    assert populations["FINAL_EVAL"]["includes_physical_final"] is False
    assert populations["FINAL_EVAL"]["alias_row_note"] == REUSED_DEV_EVAL_ALIAS_NOTE
    assert populations["ALL_AVAILABLE"]["held_out_final"] is False
    alias_policy = contract["final_eval_alias_policy"]
    assert alias_policy["physical_copy"] is False
    assert alias_policy["changes_active_frame_role"] is False
    assert alias_policy["mandatory_new_annotation"] is False
    assert alias_policy["row_provenance_note"] == REUSED_DEV_EVAL_ALIAS_NOTE
    binding = contract["paper_evaluator_binding"]
    assert binding == {
        "population_role": "DEV",
        "positive_manifest": (
            "challenge/real_gt_v2/manifests/COMMON_DEV_MULTISHAPE_POS.json"
        ),
        "negative_manifest": "challenge/real_gt_v2/manifests/DEV_NEG2689.json",
        "positive_rows": 173,
        "negative_rows": 2689,
        "negative_unique_images": 2688,
        "pair_sha256": (
            "2cfa7011d8ba3677b11019c103e2ccbaeeac53521c9291ed632f94c8d2c5c887"
        ),
    }
    from challenge.evaluation_v2.paper_real_eval import validate_evaluation_request

    pair = validate_evaluation_request(
        positive_manifest=REPO_ROOT / binding["positive_manifest"],
        negative_manifest=REPO_ROOT / binding["negative_manifest"],
        population_role=binding["population_role"],
        allow_unavailable_final=False,
    )
    assert pair.ready is True
    assert pair.pair_sha256 == binding["pair_sha256"]

    for name in (
        "DEV_EVAL_POSITIVE.csv",
        "DEV_EVAL_NEGATIVE.csv",
        "FINAL_EVAL_POSITIVE.csv",
        "FINAL_EVAL_NEGATIVE.csv",
        "ALL_AVAILABLE_POSITIVE.csv",
        "ALL_AVAILABLE_NEGATIVE.csv",
        "DEV_PLASTIC_AUDITED140.csv",
    ):
        assert (root / "manifests" / name).is_file()


def test_load_targets_migrates_legacy_far_small_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    path = root / "DATASET_TARGETS.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    coverage = payload["minimum_condition_coverage"]
    coverage["far_small"] = coverage.pop("far")
    path.write_text(json.dumps(payload), encoding="utf-8")

    migrated = load_targets(root)
    # scaffold 기본값 (2026-09-01 domain 재설계로 60 -> 80)
    assert migrated["minimum_condition_coverage"]["far"] == 80
    assert "far_small" not in migrated["minimum_condition_coverage"]

    payload["minimum_condition_coverage"]["far"] = 81
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkspaceError, match="conflicting far"):
        load_targets(root)


def test_repository_workspace_readme_matches_generator() -> None:
    root = REPO_ROOT / "data/evaluation/pallet_eval_v1"
    assert (root / "README.md").read_text("utf-8") == _workspace_readme()


def test_bbox_outside_fraction_marks_explicit_truncation(tmp_path: Path) -> None:
    annotation = tmp_path / "frame.json"

    def write(fraction: float) -> None:
        annotation.write_text(
            json.dumps(
                {
                    "objects": [
                        {
                            "object_type": "plastic",
                            "occlusion_level": "unknown",
                            "truncation": {
                                "is_truncated": False,
                                "bbox_outside_fraction": fraction,
                            },
                            "keypoint_annotations": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    write(0.125)
    assert infer_annotation_tags(annotation)["truncation"] == "mild"
    write(0.0)
    assert infer_annotation_tags(annotation)["truncation"] == "none"


def test_load_frames_rejects_missing_header_and_contract_typos(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    manifest = root / "manifests/frames.csv"
    manifest.write_text("frame_id,image_path\nframe,image.png\n", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="missing required columns"):
        load_frames(root)

    valid = _valid_frame_row("dev__valid", hashlib.sha256(b"valid").hexdigest())
    mutations = (
        ("lighting", "dya", "invalid lighting"),
        ("is_positive", "True", "lowercase true or false"),
        ("image_path", "../escape.png", "safe relative path"),
        ("storage_mode", "hardlink", "invalid storage_mode"),
    )
    for field, value, message in mutations:
        row = dict(valid)
        row[field] = value
        atomic_write_csv(manifest, [row], FRAME_COLUMNS)
        with pytest.raises(WorkspaceError, match=message):
            load_frames(root)


def test_duplicate_frame_tag_aliases_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    tags = root / "final/positive/sessions/plastic_day_01/frame_tags.csv"
    tags.write_text(
        "frame,distance_bin,size_bin,elevation_bin,view_bin\n"
        "000001.png,far,small,high,front\n"
        "plastic_day_01__000001,near,large,low,back\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkspaceError, match="duplicate/conflicting frame tag alias"):
        refresh_after_annotation(
            root,
            root / "final/positive/annotations/plastic_day_01/000001.json",
        )


def _write_small_negative_manifest(repo_root: Path, items: list[dict[str, str]]) -> Path:
    path = repo_root / "challenge/real_gt_v2/manifests/DEV_NEG2689.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "pallet_pose_population_manifest_v1",
        "population_id": "DEV_NEG2689",
        "kind": "NEGATIVE",
        "role": "DEV",
        "membership_status": "AVAILABLE",
        "frozen": True,
        "expected_count": len(items),
        "membership_sha256": manifest_membership_sha256(items),
        "items": items,
        "provenance": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_hash_and_unique_membership_are_fail_closed(tmp_path: Path) -> None:
    items = [
        {"frame_id": "000001", "image": "data/negative/000001.png", "domain": "NEG"},
        {"frame_id": "000002", "image": "data/negative/000002.png", "domain": "NEG"},
    ]
    path = _write_small_negative_manifest(tmp_path, items)
    assert len(_read_manifest(tmp_path, "DEV_NEG2689", 2)["items"]) == 2

    tampered = json.loads(path.read_text())
    tampered["items"][1]["image"] = "data/negative/tampered.png"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(WorkspaceError, match="membership SHA mismatch"):
        _read_manifest(tmp_path, "DEV_NEG2689", 2)

    duplicate_id = [dict(items[0]), dict(items[1], frame_id="000001")]
    _write_small_negative_manifest(tmp_path, duplicate_id)
    with pytest.raises(WorkspaceError, match="duplicate frame_id"):
        _read_manifest(tmp_path, "DEV_NEG2689", 2)

    duplicate_path = [dict(items[0]), dict(items[1], image=items[0]["image"])]
    _write_small_negative_manifest(tmp_path, duplicate_path)
    with pytest.raises(WorkspaceError, match="duplicate image path"):
        _read_manifest(tmp_path, "DEV_NEG2689", 2)


def test_empty_workspace_priority_report_does_not_require_new_annotation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pallet_eval_v1"
    scaffold_workspace(root)
    report = render_priority_report([], load_targets(root))
    assert "Status                    EMPTY" in report
    assert "New annotation required   NO" in report
    assert "Positive                     0 / 300" in report
    assert "Negative                     0 / 1500" in report
    assert "Counting population       ALL_AVAILABLE" in report
    assert "NEXT PRIORITY" not in report
    assert "score=" not in report
