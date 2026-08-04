"""The canonical evaluation set is the frames the user marked eval, nothing else.

Written after 2026-08-04, when every PAPER_S2 screen turned out to have been
judged on data/_eval_sets/* -- a 05-27 combination that predates the eval/train
toggle -- instead of the frames actually marked eval in the annotation tool.
"""
from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC = ROOT / "_docs/EVAL_SET_CANONICAL.md"
EVAL_FOLDERS = {
    "challenge/data/_outside_eval_manual_gt": 22,
    "challenge/data/capture0403noapril_manual_gt": 12,
    "challenge/data/capturepalletcad_manual_gt": 22,
}
EXPECTED_TOTAL = 56
FORBIDDEN_EVAL_SOURCES = ("_eval_sets/outside_combined", "_eval_sets/night_combined")


def _split_of(path: pathlib.Path) -> str:
    payload = json.loads(path.read_text("utf-8"))
    objects = payload.get("objects") or [{}]
    return objects[0].get("split", "(none)")


def collect_eval_frames() -> dict[str, list[pathlib.Path]]:
    found: dict[str, list[pathlib.Path]] = {}
    for folder in EVAL_FOLDERS:
        directory = ROOT / folder
        if not directory.is_dir():
            continue
        found[folder] = sorted(
            path for path in directory.rglob("*.json") if _split_of(path) == "eval")
    return found


def test_split_lives_under_objects_not_at_the_top_level() -> None:
    """The mistake that started this: a top-level read reports 'no split'."""
    found = collect_eval_frames()
    if not found:
        pytest.skip("annotation folders not present")
    sample = next(iter(next(iter(found.values()))), None)
    if sample is None:
        pytest.skip("no eval frame")
    payload = json.loads(sample.read_text("utf-8"))
    assert "split" not in payload, "top-level split would be a different convention"
    assert payload["objects"][0]["split"] == "eval"


def test_eval_frame_count_matches_the_canonical_document() -> None:
    found = collect_eval_frames()
    if len(found) != len(EVAL_FOLDERS):
        pytest.skip("annotation folders not all present")
    for folder, expected in EVAL_FOLDERS.items():
        assert len(found[folder]) == expected, (folder, len(found[folder]), expected)
    assert sum(len(v) for v in found.values()) == EXPECTED_TOTAL


def test_every_eval_frame_has_an_image_and_a_cuboid() -> None:
    found = collect_eval_frames()
    if not found:
        pytest.skip("annotation folders not present")
    for paths in found.values():
        for path in paths:
            assert path.with_suffix(".png").is_file(), path
            objects = json.loads(path.read_text("utf-8"))["objects"][0]
            assert objects.get("projected_cuboid"), path
            assert objects.get("gt_source") == "manual", path


def test_the_canonical_document_exists_and_names_the_folders() -> None:
    assert DOC.is_file(), "the canonical eval-set document must exist"
    text = DOC.read_text("utf-8")
    for folder in EVAL_FOLDERS:
        assert folder in text
    assert "objects[0].split" in text
    for forbidden in FORBIDDEN_EVAL_SOURCES:
        assert forbidden in text, "the superseded source must be named as forbidden"


def test_new_eval_manifests_must_not_be_built_from_the_superseded_combination() -> None:
    """A manifest may only cite _eval_sets if it also records that it is stale."""
    results = ROOT / "data/pallet/results"
    if not results.is_dir():
        pytest.skip("no results tree")
    offenders = []
    for manifest in results.rglob("*manifest*.json"):
        try:
            text = manifest.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(source in text for source in FORBIDDEN_EVAL_SOURCES):
            # the historical PAPER_S2 manifest is grandfathered but must be
            # flagged in the canonical document, which the test above checks
            if manifest.name != "mechanism_val_manifest.json":
                offenders.append(str(manifest.relative_to(ROOT)))
    assert not offenders, (
        "new manifests must use objects[0].split == 'eval', see "
        f"_docs/EVAL_SET_CANONICAL.md: {offenders}")


def test_train_marked_frames_are_never_counted_as_eval() -> None:
    found = collect_eval_frames()
    if not found:
        pytest.skip("annotation folders not present")
    for folder in EVAL_FOLDERS:
        directory = ROOT / folder
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.json"):
            if _split_of(path) == "train":
                assert path not in found[folder]
