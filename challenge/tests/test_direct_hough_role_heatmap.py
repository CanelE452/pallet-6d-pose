"""Tests for the line-native fixed-role Hough screen.

This is a representation and readout family switch, so what must be pinned is
that the line equivalence holds everywhere, that nothing regresses a coordinate,
and that each oracle gates the next stage.
"""
from __future__ import annotations

import ast, importlib.util, json, math, pathlib, sys

import numpy as np, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/direct_hough_role_heatmap.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def dh():
    spec = importlib.util.spec_from_file_location("DH_UNDER_TEST", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def source():
    return RUNNER.read_text("utf-8")


def load_json(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text("utf-8"))


def test_the_hypothesis_embedding_respects_line_equivalence(dh):
    worst = 0.0
    for angle in (-0.1, 0.0, 0.1, 89.9, 179.9, 180.1, 359.9):
        theta = torch.tensor([angle], device=dh.DEV)
        rho = torch.tensor([17.5], device=dh.DEV)
        a = dh.hypothesis_features(theta, rho)
        b = dh.hypothesis_features(theta + 180.0, -rho)
        worst = max(worst, float((a - b).abs().max()))
    assert worst <= 1e-6


def test_line_distance_is_wrap_aware_with_the_rho_sign_flip(dh):
    theta_h = torch.tensor([0.5], device=dh.DEV)
    rho_h = torch.tensor([10.0], device=dh.DEV)
    # the same line written the other way round must be near, not 180 away
    theta_gt = torch.tensor([179.5], device=dh.DEV)
    rho_gt = torch.tensor([-10.0], device=dh.DEV)
    angle, offset, _ = dh.line_distance(theta_h, rho_h, theta_gt, rho_gt)
    assert float(angle) == pytest.approx(1.0, abs=1e-4)
    assert float(offset) == pytest.approx(0.0, abs=1e-4)
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "line_distance"))
    assert "for k in (-1, 0, 1)" in body and "sign" in body


def test_nothing_regresses_a_coordinate(dh):
    tree = ast.parse(source())
    head = next(node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == "DirectHoughHead")
    produced = {n.targets[0].attr for n in ast.walk(head)
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)}
    assert produced == {"hypothesis", "project", "dim"}
    body = ast.get_source_segment(source(), head)
    for forbidden in ("to_theta", "to_rho", "theta_head", "rho_head"):
        assert forbidden not in body, forbidden


def test_the_lattice_ceiling_is_half_the_task_budget(dh):
    assert dh.THETA_STEP == 1.0 and dh.RHO_STEP == 1.0
    assert dh.GRID_CEILING == {"angle": 0.5, "offset": 0.25}
    assert dh.GRID_CEILING["angle"] == dh.CAP.ANGLE_BUDGET_DEG / 2
    assert dh.GRID_CEILING["offset"] == dh.CAP.OFFSET_BUDGET_CELL / 2


def test_the_nearest_hypothesis_uses_one_declared_distance(dh):
    assert (dh.DISTANCE_ANGLE_UNIT, dh.DISTANCE_OFFSET_UNIT) == (1.0, 0.5)
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "run_domain"))
    assert "squared.argmin(0)" in body      # joint distance, not independent rounding


def test_the_encoder_is_q1s_unchanged(dh):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.ClassDef) and node.name == "DirectHoughModel"))
    assert "RQ.RoleQueryGlobal()" in body and "ARCH.AbsoluteXY()" in body
    assert dh.QUERY_DIM == dh.RQ.QUERY_DIM and dh.ROLES == dh.RQ.ROLES
    text = source()
    for forbidden in ("SupportingLineHead", "map_loss", "MAP200"):
        assert forbidden not in text, forbidden


def test_unsupported_roles_get_no_gradient(dh):
    grid_theta, grid_rho, valid = dh.lattice()
    scores = torch.zeros(1, 2, grid_theta.numel(), device=dh.DEV,
                         requires_grad=True)
    target = torch.zeros_like(scores)
    target[..., 0] = 1.0
    support = torch.tensor([[True, False]], device=dh.DEV)
    dh.cross_entropy(scores, target, support, valid).backward()
    assert float(scores.grad[0, 1].abs().max()) == 0.0
    assert float(scores.grad[0, 0].abs().max()) > 0.0


def test_each_stage_gates_the_next():
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    for label in ("HOUGH_DOMAIN_COVERAGE_FAIL", "O_GRID_CEILING_FAIL",
                  "ROLE_SPECIFIC_HOUGH_DOMAIN_BIAS",
                  "DIRECT_HOUGH_TARGET_WIRING_FAIL",
                  "HOUGH_SCORER_FORMULATION_FAIL",
                  "DIRECT_HOUGH_NETWORK_FIT_FAIL"):
        assert label in body, label


def test_the_baseline_is_read_at_full_precision(dh):
    assert dh.BASELINE_SOURCE == ("role_query_results.json", "Q1_ROLE_QUERY_GLOBAL")
    base = dh.baseline_reference()
    assert any(abs(base[k] - round(base[k], 4)) > 0
               for k in ("angle_median", "offset_median"))
    # numeric literals in code, never prose: the module docstring cites 4.1793
    # as the number this screen is measured against
    numbers = {n.value for n in ast.walk(ast.parse(source()))
               if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    for literal in (4.1793, 1.8788, 2.5076, 1.1273):
        assert literal not in numbers, literal


def test_tail_diagnostics_are_recorded(dh):
    report = dh.summarise(np.array([1.0, 6.0, 11.0, 3.0]),
                          np.array([0.5, 1.0, 3.0, 2.5]))
    for key in ("frac_angle_gt5", "frac_angle_gt10", "frac_offset_gt2"):
        assert key in report
    assert report["frac_angle_gt5"] == pytest.approx(0.5)
    assert report["frac_angle_gt10"] == pytest.approx(0.25)


def test_overfit_uses_the_real_task_gate_not_an_ultra_tight_one(dh):
    assert dh.OVERFIT_STEPS == (1500, 3000)
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert 'final["PASS"] and final["SAFETY"]' in body
    assert "0.10" not in body and "0.05" not in body


def test_the_oracles_passed_when_run(dh):
    domain = load_json("direct_hough_domain.json")
    assert domain["HOUGH_DOMAIN_COVERAGE"] is True
    assert domain["O_GRID_PASS"] is True
    assert domain["ROLE_SPECIFIC_HOUGH_DOMAIN_BIAS"] == []
    assert domain["overall"]["outside_domain"] == 0
    target = load_json("direct_hough_target_oracle.json")
    assert target["O_TARGET_PASS"] is True
    scorer = load_json("direct_hough_scorer_oracle.json")
    assert scorer["O_SCORER_PASS"] is True
