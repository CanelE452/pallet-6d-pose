"""Challenge-only geometry registry: adds the field pallet, never edits the paper lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.annotate.object_geometry_registry import (
    PLASTIC_OBJECT_TYPE,
    PLASTIC_SQUARE_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    DEFAULT_REGISTRY_PATH,
    load_object_geometry_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHALLENGE_REGISTRY_PATH = (
    REPO_ROOT / "challenge" / "config" / "CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json"
)


def _objects_by_type(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {entry["object_type"]: entry for entry in payload["objects"]}


def test_paper_registry_bytes_are_untouched() -> None:
    """The paper registry SHA is pinned in ~50 result artifacts; it must not move."""
    digest = hashlib.sha256(DEFAULT_REGISTRY_PATH.read_bytes()).hexdigest()
    assert digest == "0c7a10729b6db18cbe47fa4adb158e2f26ec7a7c9458f59ee60d023c282f0627"


def test_challenge_registry_copies_paper_objects_verbatim() -> None:
    """Locked objects are copied, not re-typed: a drift here would split the two files."""
    paper = _objects_by_type(DEFAULT_REGISTRY_PATH)
    challenge = _objects_by_type(CHALLENGE_REGISTRY_PATH)
    for object_type in (PLASTIC_OBJECT_TYPE, WOOD_OBJECT_TYPE):
        assert challenge[object_type] == paper[object_type]


def test_challenge_registry_adds_only_the_field_pallet() -> None:
    challenge = _objects_by_type(CHALLENGE_REGISTRY_PATH)
    assert set(challenge) == {
        PLASTIC_OBJECT_TYPE,
        WOOD_OBJECT_TYPE,
        PLASTIC_SQUARE_OBJECT_TYPE,
    }


def test_field_pallet_dimensions_match_the_measured_110x110x15() -> None:
    registry = load_object_geometry_registry(CHALLENGE_REGISTRY_PATH)
    spec = registry.resolve(PLASTIC_SQUARE_OBJECT_TYPE)
    assert spec.physical_dimensions_m == {"x": 1.10, "y": 0.15, "z": 1.10}
    assert spec.symmetry_status == "UNREVIEWED"
    assert spec.symmetry_contract is None


def test_field_pallet_is_the_default_only_in_the_challenge_registry() -> None:
    assert load_object_geometry_registry(
        CHALLENGE_REGISTRY_PATH).default_object_type == PLASTIC_SQUARE_OBJECT_TYPE
    assert load_object_geometry_registry(
        DEFAULT_REGISTRY_PATH).default_object_type == PLASTIC_OBJECT_TYPE


def test_paper_registry_still_rejects_a_missing_locked_object(tmp_path) -> None:
    """Relaxing 'exactly' to 'at least' must not let a locked object disappear."""
    payload = json.loads(CHALLENGE_REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["objects"] = [
        entry for entry in payload["objects"]
        if entry["object_type"] != WOOD_OBJECT_TYPE
    ]
    broken = tmp_path / "registry.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="locked plastic and wood"):
        load_object_geometry_registry(broken)
