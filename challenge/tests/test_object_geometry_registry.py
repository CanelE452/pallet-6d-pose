"""Frozen two-object registry contract."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.annotate.object_geometry_registry import (
    ObjectGeometryRegistryError,
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    load_object_geometry_registry,
)


def test_registry_has_exact_plastic_and_wood_named_dimensions() -> None:
    registry = load_object_geometry_registry()
    assert registry.sha256 == hashlib.sha256(registry.source_path.read_bytes()).hexdigest()
    assert registry.sha256 == "0c7a10729b6db18cbe47fa4adb158e2f26ec7a7c9458f59ee60d023c282f0627"
    assert set(registry.objects) == {PLASTIC_OBJECT_TYPE, WOOD_OBJECT_TYPE}
    assert registry.resolve("PLASTIC_STANDARD").physical_dimensions_m == {
        "x": 1.10,
        "y": 0.11,
        "z": 1.30,
    }
    assert registry.resolve("WOOD_SMALL").physical_dimensions_m == {
        "x": 0.80,
        "y": 0.14,
        "z": 0.59,
    }
    assert registry.resolve(WOOD_OBJECT_TYPE).symmetry_status == "UNREVIEWED"
    assert registry.resolve(WOOD_OBJECT_TYPE).symmetry_contract is None


def test_registry_rejects_unknown_and_dimension_tampering(tmp_path) -> None:
    registry = load_object_geometry_registry()
    with pytest.raises(ObjectGeometryRegistryError, match="UNKNOWN_OBJECT_TYPE"):
        registry.resolve("filename_inferred_wood")
    payload = json.loads(registry.source_path.read_text("utf-8"))
    payload["objects"][1]["physical_dimensions_m"]["x"] = 1.10
    tampered = tmp_path / "registry.json"
    tampered.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(ObjectGeometryRegistryError, match="source_measurement_cm"):
        load_object_geometry_registry(tampered)
