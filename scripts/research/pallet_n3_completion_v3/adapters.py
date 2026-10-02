"""Strict factory for the two frozen N3 base estimators."""
from __future__ import annotations

import sys

from . import common as C


def _dope_class():
    directory = C.ROOT / "scripts/research/pallet_dope_refiner_20261001_v1"
    previous = sys.modules.get("common")
    try:
        common = C.local_module("bound_dope_common", directory / "common.py")
        sys.modules["common"] = common
        module = C.local_module("bound_dope_adapter", directory / "dope_adapter.py")
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    return module.FrozenDopeAdapter


def build_adapter(backbone: str, device="cuda"):
    if backbone == "dope":
        adapter = _dope_class()(device=device)
        adapter.checkpoint_sha256 = C.sha256(C.DOPE_WEIGHTS)
        adapter.trained_baseline_loaded = True
        if adapter.checkpoint_sha256 != "0de80490cb3b4f9b11565db7a4aea6338f64edb8f9614910bfb52bf03ce0dc3f":
            raise RuntimeError("DOPE checkpoint identity drift")
        return adapter
    if backbone == "resnet18":
        from .resnet_constant_adapter import FrozenConstantResnetAdapter
        adapter = FrozenConstantResnetAdapter(device=device)
        adapter.checkpoint_sha256 = adapter.checkpoint["sha256"]
        if adapter.checkpoint_sha256 != C.sha256(C.RESNET_CONSTANT):
            raise RuntimeError("ResNet constant checkpoint identity drift")
        return adapter
    raise ValueError(backbone)


__all__ = ["build_adapter"]
