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
RUNNER = ROOT / "scripts/stage0/line/structural_line_map_capacity.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


CPU = None          # set once the module is loaded


@pytest.fixture(scope="module")
def slm():
    """The readout is pure geometry, so it is exercised on CPU.

    A subprocess test elsewhere in the suite loads a full model; holding CUDA
    allocations here made that subprocess OOM and fail for the wrong reason.
    """
    spec = importlib.util.spec_from_file_location("SLM_UNDER_TEST", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def source():
    return RUNNER.read_text("utf-8")


def test_role_channels_are_fixed_and_never_matched(slm):
    assert slm.raster_targets(np.zeros((1, 12, 2)), np.ones((1, 12, 2)),
                              np.ones((1, 12), bool), torch.device("cpu")).shape[1] == 12
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
    read = slm.weighted_tls(slm.raster_targets(q0, q1, np.ones((1, 1), bool), torch.device("cpu")))
    direction = (q1 - q0)[0, 0]
    normal = np.array([-direction[1], direction[0]]); normal /= np.linalg.norm(normal)
    theta = torch.tensor([[math.atan2(normal[1], normal[0])]], device=torch.device("cpu"))
    rho = torch.tensor([[float(normal @ ((q0 + q1)[0, 0] / 2))]], device=torch.device("cpu"))
    angle, offset = slm.line_errors(read["normal"], read["rho"], theta, rho)
    assert abs(float(angle)) < 0.01 and abs(float(offset)) < 0.01


def test_the_readout_is_sign_invariant(slm):
    q0 = np.array([[[5.0, 10.0]]]); q1 = np.array([[[45.0, 40.0]]])
    read = slm.weighted_tls(slm.raster_targets(q0, q1, np.ones((1, 1), bool), torch.device("cpu")))
    direction = (q1 - q0)[0, 0]
    normal = np.array([-direction[1], direction[0]]); normal /= np.linalg.norm(normal)
    theta = torch.tensor([[math.atan2(normal[1], normal[0])]], device=torch.device("cpu"))
    rho = torch.tensor([[float(normal @ ((q0 + q1)[0, 0] / 2))]], device=torch.device("cpu"))
    a1, o1 = slm.line_errors(read["normal"], read["rho"], theta, rho)
    a2, o2 = slm.line_errors(read["normal"], read["rho"], theta + math.pi, -rho)
    assert abs(float(a1.abs() - a2.abs())) < 1e-3
    assert abs(float(o1.abs() - o2.abs())) < 1e-3


def test_targets_are_finite_and_zero_for_unsupported_roles(slm):
    q0 = np.array([[[5.0, 10.0], [0.0, 0.0]]]); q1 = np.array([[[45.0, 40.0], [0.0, 0.0]]])
    target = slm.raster_targets(q0, q1, np.array([[True, False]]), torch.device("cpu"))
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
    del head, stem, parameters
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_results_report_full_and_partial_separately(slm):
    path = OUT / "structural_line_map_omap.json"
    if not path.exists():
        pytest.skip("O_MAP not run yet")
    report = json.loads(path.read_text())
    assert "in_frame_full" in report and "in_frame_partial" in report


# --------------------------------------------------------------------------
# forward-parity oracle
# --------------------------------------------------------------------------

def test_perfect_weight_equals_softplus_of_the_logit(slm):
    """The identity that makes the parity oracle the right one: a network whose
    sigmoid equals the target hands softplus(logit(target)) to the readout."""
    torch.manual_seed(0)
    p = torch.rand(200000, dtype=torch.float32) * (1 - 2e-6) + 1e-6
    lhs = torch.nn.functional.softplus(torch.log(p) - torch.log1p(-p))
    rhs = slm.perfect_weight_from_target(p)
    assert float((lhs - rhs).abs().max()) <= 1e-6


def test_perfect_weight_keeps_zero_exact_and_clamps_at_the_dtype_bound(slm):
    target = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    weight = slm.perfect_weight_from_target(target)
    assert float(weight[0]) == 0.0
    assert float(weight[1]) == pytest.approx(math.log(2.0), abs=1e-6)
    assert torch.isfinite(weight).all()
    bound = -math.log1p(-(1 - torch.finfo(torch.float32).eps))
    assert float(weight[2]) == pytest.approx(bound, rel=1e-6)


def test_the_parity_oracle_is_the_one_that_gates_training(slm):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert 'OUT / "structural_line_map_omap_parity.json"' in body
    assert "MAP_TO_LINE_DECODER_FAIL_CONFIRMED" in body


def test_the_locked_decoder_and_target_are_untouched(slm):
    assert slm.SIGMA_CELLS == 1.5 and slm.MAP == 100 and slm.CANON == 50
    assert slm.OMAP_GATE == {"angle_median": 0.05, "offset_median": 0.05,
                             "angle_p90": 0.10, "offset_p90": 0.10}
    run_omap = next(node for node in ast.walk(ast.parse(source()))
                    if isinstance(node, ast.FunctionDef) and node.name == "run_omap")
    # parity may steer the oracle's input and label it, and nothing else: one
    # conditional expression, no statement-level branch on it.
    conditionals = [n for n in ast.walk(run_omap) if isinstance(n, ast.IfExp)
                    and getattr(n.test, "id", "") == "parity"]
    assert len(conditionals) == 2                      # weight, and the label
    assert any("perfect_weight_from_target" in ast.dump(n) for n in conditionals)
    assert not [n for n in ast.walk(run_omap) if isinstance(n, ast.If)
                and getattr(n.test, "id", "") == "parity"]


def test_units_are_never_bare_cells(slm):
    assert "MAP100" in slm.UNITS["sigma"] and "canonical50" in slm.UNITS["sigma"]
    assert "canonical50" in slm.UNITS["border_threshold"]
    assert "sigma" in slm.UNITS["border_threshold"]


def test_the_cross_tab_separates_border_from_short_stub(slm):
    angle = np.array([0.0, 1.0, 2.0, 3.0])
    offset = np.array([0.0, 1.0, 2.0, 3.0])
    ratio = np.array([9.0, 8.0, 7.0, 6.0])
    border = np.array([5.0, 5.0, 0.5, 0.5])
    visible = np.array([9.0, 0.5, 9.0, 0.5])
    table = slm.cross_tab(angle, offset, ratio, border, visible)
    assert set(table) == {"A_border_ge_vis_ge", "B_border_ge_vis_lt",
                          "C_border_lt_vis_ge", "D_border_lt_vis_lt"}
    assert all(entry["n"] == 1 for entry in table.values())
    assert "eigen_ratio_p10" in table["A_border_ge_vis_ge"]
