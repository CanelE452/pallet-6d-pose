"""Import/call ordering cannot leak wood dimensions into plastic annotation."""

from __future__ import annotations

import importlib

from scripts.annotate import annotate_pnp, annotate_wood
from scripts.annotate.object_geometry_registry import (
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    load_object_geometry_registry,
)


def test_import_order_does_not_mutate_legacy_plastic_default() -> None:
    before = annotate_pnp.PALLET_DIMS
    importlib.reload(annotate_wood)
    registry = load_object_geometry_registry()
    assert annotate_pnp.PALLET_DIMS == before == (1.1, 1.3, 0.11)
    assert registry.resolve(PLASTIC_OBJECT_TYPE).legacy_wdh_tuple == before
    assert registry.resolve(WOOD_OBJECT_TYPE).legacy_wdh_tuple == (0.8, 0.59, 0.14)
    assert annotate_pnp.PALLET_DIMS == before
