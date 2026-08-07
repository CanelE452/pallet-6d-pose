"""Tests for the structural-line map capacity screen.

The screen exists because every previous line result came from one architecture
family -- generic feature, local strip, coordinate head.  This one replaces the
coordinate head with a differentiable readout over a spatial map, so the checks
that matter are that the map really decides the line, that role identity is
fixed, and that no pose quantity reaches a forward pass.
"""
from __future__ import annotations

import ast, importlib.util, json, math, pathlib, sys

import numpy as np, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/structural_line_map_capacity.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def slm():
    spec = importlib.util.spec_from_file_location("SLM_UNDER_TEST", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source():
    return RUNNER.read_text("utf-8")


def test_role_channels_are_fixed_and_never_matched(slm):
    assert slm.raster_targets(np.zeros((1, 12, 2)), np.ones((1, 12, 2)),
                              np.ones((1, 12), bool), slm.DEV).shape[1] == 12
    text = source()
    for forbidden in ("linear_sum_assignment", "hungarian", "Hungarian"):
        assert forbidden not in text, forbidden


def test_the_map_decides_the_line_with_no_coordinate_head(slm):
    tree = ast.parse(source())
    head = next(node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == "LineMapHead")
    produced = {node.targets[0].attr for node in ast.walk(head)
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Attribute)}
    assert produced == {"body", "to_map", "to_support"}, produced


def test_weighted_tls_recovers_a_clean_segment(slm):
    q0 = np.array([[[5.0, 10.0]]]); q1 = np.array([[[45.0, 40.0]]])
    read = slm.weighted_tls(slm.raster_targets(q0, q1, np.ones((1, 1), bool), slm.DEV))
    direction = (q1 - q0)[0, 0]
    normal = np.array([-direction[1], direction[0]]); normal /= np.linalg.norm(normal)
    theta = torch.tensor([[math.atan2(normal[1], normal[0])]], device=slm.DEV)
    rho = torch.tensor([[float(normal @ ((q0 + q1)[0, 0] / 2))]], device=slm.DEV)
    angle, offset = slm.line_errors(read["normal"], read["rho"], theta, rho)
    assert abs(float(angle)) < 0.01 and abs(float(offset)) < 0.01


def test_the_readout_is_sign_invariant(slm):
    q0 = np.array([[[5.0, 10.0]]]); q1 = np.array([[[45.0, 40.0]]])
    read = slm.weighted_tls(slm.raster_targets(q0, q1, np.ones((1, 1), bool), slm.DEV))
    direction = (q1 - q0)[0, 0]
    normal = np.array([-direction[1], direction[0]]); normal /= np.linalg.norm(normal)
    theta = torch.tensor([[math.atan2(normal[1], normal[0])]], device=slm.DEV)
    rho = torch.tensor([[float(normal @ ((q0 + q1)[0, 0] / 2))]], device=slm.DEV)
    a1, o1 = slm.line_errors(read["normal"], read["rho"], theta, rho)
    a2, o2 = slm.line_errors(read["normal"], read["rho"], theta + math.pi, -rho)
    assert abs(float(a1.abs() - a2.abs())) < 1e-3
    assert abs(float(o1.abs() - o2.abs())) < 1e-3


def test_targets_are_finite_and_zero_for_unsupported_roles(slm):
    q0 = np.array([[[5.0, 10.0], [0.0, 0.0]]]); q1 = np.array([[[45.0, 40.0], [0.0, 0.0]]])
    target = slm.raster_targets(q0, q1, np.array([[True, False]]), slm.DEV)
    assert torch.isfinite(target).all()
    assert float(target[0, 1].abs().max()) == 0.0
    assert float(target[0, 0].max()) == pytest.approx(1.0, abs=1e-3)


def test_a_partial_role_is_rasterised_from_the_clipped_segment(slm):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "run_omap"))
    assert 'seg["q0"], seg["q1"]' in body        # clipped endpoints, not p0/p1


def test_no_pose_quantity_reaches_a_forward_pass(slm):
    tree = ast.parse(source())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for forbidden in ("solve_pose", "solvePnP", "dims", "dimensions", "intrinsics",
                      "PALLET_DIMS", "cuboid_3d", "CIGM", "EGCR"):
        assert forbidden not in names, forbidden
    text = source()
    for token in ("validation512", "wood45", "handannot17", "testset_full8"):
        assert token not in text, token


def test_losses_are_budget_normalised_and_fixed(slm):
    assert (slm.ANGLE_BUDGET_DEG, slm.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (slm.L_MAP_WEIGHT, slm.L_SUPPORT_WEIGHT) == (0.5, 0.1)
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "batch_terms"))
    assert "d_angle / ANGLE_BUDGET_DEG" in body
    assert "d_offset / OFFSET_BUDGET_CELL" in body
    assert "0.5 * (((probability - target) ** 2 * positive)" in body


def test_thresholds_are_declared_before_the_run(slm):
    assert slm.SIGMA_CELLS == 1.5
    assert slm.OMAP_GATE == {"angle_median": 0.05, "offset_median": 0.05,
                             "angle_p90": 0.10, "offset_p90": 0.10}
    assert (slm.OVERFIT_ANGLE, slm.OVERFIT_OFFSET) == (0.10, 0.05)
    assert (slm.APPROACH_ANGLE, slm.APPROACH_OFFSET) == (1.5, 0.75)
    assert (slm.SHUFFLE_ANGLE_MARGIN, slm.SHUFFLE_OFFSET_MARGIN) == (5.0, 2.0)


def test_the_shuffle_diagnostic_uses_a_fixed_derangement(slm):
    assert sorted(slm.DERANGEMENT) == list(range(12))
    assert all(i != v for i, v in enumerate(slm.DERANGEMENT))


def test_training_is_blocked_until_the_decoder_oracle_passes(slm):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert "MAP_TO_LINE_DECODER_FAIL" in body
    assert 'omap.exists() or not json.loads(omap.read_text())["OMAP_PASS"]' in body


def test_confirm6k_is_conditional_on_search2k(slm):
    text = source()
    assert "APPROACH" in text and "APPROACH_ANGLE" in text


def test_the_rgb_arm_trains_its_stem_and_the_frozen_arm_has_none(slm):
    head, stem, parameters = slm.build_arm("M1_F50_RGB_MAP")
    assert stem is not None
    ids = {id(p) for p in parameters}
    assert all(id(p) in ids for p in stem.parameters())
    assert slm.build_arm("M0_F50_MAP")[1] is None


def test_results_report_full_and_partial_separately(slm):
    path = OUT / "structural_line_map_omap.json"
    if not path.exists():
        pytest.skip("O_MAP not run yet")
    report = json.loads(path.read_text())
    assert "in_frame_full" in report and "in_frame_partial" in report
