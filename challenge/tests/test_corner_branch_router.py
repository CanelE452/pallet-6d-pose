"""Raw-Q decoder and coordinate-router unit tests (Phase B4 / G)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE = ROOT / "Deep_Object_Pose/common/corner_branch_router.py"
SCALE = (640.0 / 50.0, 480.0 / 50.0)


def _cbr():
    if str(ROOT / "Deep_Object_Pose/common") not in sys.path:
        sys.path.insert(0, str(ROOT / "Deep_Object_Pose/common"))
    spec = importlib.util.spec_from_file_location("CBR", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _peak_logits(x: float, y: float, sigma: float = 1.0) -> np.ndarray:
    ys, xs = np.mgrid[0:50, 0:50]
    bump = -((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2)
    return np.repeat(bump[None], 8, axis=0).astype(np.float32)


def _executable_source() -> str:
    """Module source with docstrings and # comments removed.

    The prose explains what these decoders deliberately avoid, so a plain text
    search would match the explanation instead of the code.
    """
    import ast

    text = MODULE.read_text("utf-8")
    lines = text.split("\n")
    tree = ast.parse(text)
    drop: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            continue
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and \
                isinstance(body[0].value, ast.Constant) and \
                isinstance(body[0].value.value, str):
            drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    kept = [line.split("#")[0] for number, line in enumerate(lines, start=1)
            if number not in drop]
    return "\n".join(kept)


def test_no_sigmoid_and_no_absolute_threshold_in_executable_code() -> None:
    code = _executable_source().lower()
    assert "sigmoid" not in code
    assert "0.3" not in code, "an absolute belief threshold must not appear"
    assert "threshold" not in code


def test_constant_logit_offset_leaves_every_coordinate_unchanged() -> None:
    cbr = _cbr()
    logits = _peak_logits(31.4, 12.6)
    for name, decode in cbr.DECODERS.items():
        before = np.asarray(decode(logits, SCALE))
        after = np.asarray(decode(logits + 7.25, SCALE))
        assert np.allclose(before, after, atol=1e-6), name


def test_argmax_recovers_a_single_sharp_peak() -> None:
    cbr = _cbr()
    logits = np.full((8, 50, 50), -50.0, np.float32)
    logits[:, 37, 21] = 10.0
    points = cbr.decode_argmax(logits, SCALE)
    assert len(points) == 8
    for point in points:
        assert point == [21 * SCALE[0], 37 * SCALE[1]]


def test_local_and_dsnt_recover_a_subpixel_peak() -> None:
    cbr = _cbr()
    logits = _peak_logits(24.3, 30.7, sigma=1.2)
    for decode in (cbr.decode_local, cbr.decode_dsnt):
        point = decode(logits, SCALE)[0]
        assert abs(point[0] / SCALE[0] - 24.3) < 0.35
        assert abs(point[1] / SCALE[1] - 30.7) < 0.35


def test_local_uses_a_seven_by_seven_window_at_temperature_one_tenth() -> None:
    cbr = _cbr()
    assert cbr.WINDOW == 7 and cbr.TEMPERATURE == 0.1
    # a far-away second bump must not move the local read-out
    logits = _peak_logits(10.0, 10.0, sigma=1.0)
    logits[0] = np.maximum(logits[0], _peak_logits(45.0, 45.0, sigma=3.0)[0] - 0.2)
    point = cbr.decode_local(logits, SCALE)[0]
    assert abs(point[0] / SCALE[0] - 10.0) < 1.0


def test_dsnt_is_the_full_map_expectation() -> None:
    cbr = _cbr()
    logits = np.zeros((8, 50, 50), np.float32)  # uniform -> centre of the grid
    point = cbr.decode_dsnt(logits, SCALE)[0]
    assert abs(point[0] / SCALE[0] - 24.5) < 1e-6
    assert abs(point[1] / SCALE[1] - 24.5) < 1e-6


def test_decoders_return_eight_corners_and_never_a_centroid() -> None:
    cbr = _cbr()
    assert cbr.N_CORNERS == 8
    logits = _peak_logits(20.0, 20.0)
    for decode in cbr.DECODERS.values():
        assert len(decode(logits, SCALE)) == 8


def test_decoders_are_finite_on_extreme_logits() -> None:
    cbr = _cbr()
    for logits in (np.full((8, 50, 50), -1e4, np.float32),
                   np.full((8, 50, 50), 1e4, np.float32),
                   _peak_logits(0.0, 49.0)):
        for decode in cbr.DECODERS.values():
            points = np.asarray(decode(logits, SCALE))
            assert np.isfinite(points).all()


def test_anisotropic_scaling_is_applied() -> None:
    cbr = _cbr()
    logits = np.full((8, 50, 50), -50.0, np.float32)
    logits[:, 10, 40] = 5.0
    point = cbr.decode_argmax(logits, SCALE)[0]
    assert point == [40 * 640 / 50, 10 * 480 / 50]


def test_exact_oracle_takes_the_smaller_error() -> None:
    cbr = _cbr()
    base = np.array([10.0, 5.0, 7.0])
    proposal = np.array([4.0, 9.0, 7.0])
    taken = cbr.route_oracle(base, proposal)
    assert list(taken) == [True, False, False]  # equal keeps the base


def test_margin_oracle_keeps_the_base_on_a_near_tie() -> None:
    cbr = _cbr()
    base = np.array([10.0, 10.0, 10.0])
    proposal = np.array([6.0, 8.0, 14.0])
    taken = cbr.route_oracle(base, proposal, margin=3.0)
    assert list(taken) == [True, False, False]


def test_router_labels_drop_ties_and_never_see_gt_positions() -> None:
    cbr = _cbr()
    base = np.array([10.0, 10.0, 10.0])
    proposal = np.array([6.0, 9.0, 20.0])
    label, usable = cbr.router_labels(base, proposal, margin=3.0)
    assert list(usable) == [True, False, True]
    assert list(label) == [1.0, 0.0, 0.0]
    source = MODULE.read_text("utf-8")
    assert "gt_points" not in source and "gt_xy" not in source


def test_routing_is_hard_with_no_soft_mixture() -> None:
    code = _executable_source().lower()
    for banned in ("lerp", "blend", "mixture", "torch.where(", "convex"):
        assert banned not in code, banned
    # the oracle and the label helper both return booleans, never weights
    cbr = _cbr()
    taken = cbr.route_oracle(np.array([10.0]), np.array([4.0]))
    assert taken.dtype == bool


def test_router_is_a_small_mlp_producing_one_logit() -> None:
    cbr = _cbr()
    torch.manual_seed(0)
    router = cbr.CoordinateRouter(17)
    out = router(torch.randn(5, 17))
    assert out.shape == (5,)
    assert sum(p.numel() for p in router.parameters()) < 20000


def test_no_forbidden_branch_in_the_router_module() -> None:
    text = MODULE.read_text("utf-8").lower()
    for banned in ("graph", "diffpnp", "semantic_line", "mask_head", "voting"):
        assert banned not in text
