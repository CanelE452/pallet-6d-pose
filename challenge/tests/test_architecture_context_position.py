"""Tests for the global-context x absolute-position architecture screen.

The screen is a factorial, so the tests pin that exactly two factors move, that
neither can stand in for the other, and that all four arms start as the same
function.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import numpy as np, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/architecture_context_position_screen.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def arch():
    spec = importlib.util.spec_from_file_location("ARCH_UNDER_TEST", RUNNER)
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


def test_both_added_branches_are_zero_initialised(arch):
    position = arch.AbsoluteXY()
    assert float(position.project.weight.abs().max()) == 0.0
    assert float(position.project.bias.abs().max()) == 0.0
    context = arch.GlobalContentContext()
    assert float(context.body[-1].weight.abs().max()) == 0.0
    assert float(context.body[-1].bias.abs().max()) == 0.0
    feature = torch.randn(2, arch.F50_CHANNELS, arch.MAP, arch.MAP)
    assert torch.allclose(position(feature), feature, atol=1e-7)
    assert torch.allclose(context(feature), feature, atol=1e-7)


def test_global_context_cannot_carry_position(arch):
    """A spatial average is translation-invariant, so G cannot stand in for P."""
    context = arch.GlobalContentContext()
    with torch.no_grad():
        context.body[-1].weight.normal_()
        context.body[-1].bias.normal_()
    feature = torch.randn(1, arch.F50_CHANNELS, arch.MAP, arch.MAP)
    rolled = torch.roll(feature, shifts=(7, 11), dims=(-2, -1))
    a = context(feature)
    b = context(rolled)
    assert torch.allclose(torch.roll(a, shifts=(7, 11), dims=(-2, -1)), b, atol=1e-5)
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.ClassDef) and node.name == "GlobalContentContext"))
    assert "mean((-2, -1))" in body
    for forbidden in ("pyramid", "unfold", "grid", "linspace", "meshgrid"):
        assert forbidden not in body, forbidden


def test_the_position_factor_is_raw_normalised_xy_only(arch):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.ClassDef) and node.name == "AbsoluteXY"))
    assert "linspace(-1.0, 1.0" in body
    for forbidden in ("sin", "cos", "Embedding", "** 2", "intrinsic", "fx", "cx"):
        assert forbidden not in body, forbidden
    position = arch.AbsoluteXY()
    assert position.grid.shape[1] == 2
    assert float(position.grid.min()) == -1.0 and float(position.grid.max()) == 1.0


def test_the_factorial_is_exactly_two_factors(arch):
    assert arch.ARMS == {"A_G0P0": (False, False), "B_G1P0": (True, False),
                         "C_G0P1": (False, True), "D_G1P1": (True, True)}
    for name, (g, p) in arch.ARMS.items():
        model, _ = arch.build_arm(name)
        assert (model.context is not None) == g
        assert (model.position is not None) == p
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_the_locked_head_and_recipe_are_untouched(arch):
    assert arch.MAP == 100 and arch.CAP.SIGMA_CELLS == 1.5
    assert arch.MARKS == (1250, 2500, 5000, 8515)
    assert (arch.CAP.ANGLE_BUDGET_DEG, arch.CAP.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (arch.CAP.SAFETY_ANGLE, arch.CAP.SAFETY_OFFSET) == (2.0, 1.0)
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.ClassDef) and node.name == "ArmModel"))
    assert "CAP.SupportingLineHead" in body
    # identifiers and string literals, never prose: the module docstring says
    # "no MAP200, no CIGM, no dimensions", which a substring search reads as use.
    tree = ast.parse(source())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("MAP200", "solve_pose", "solvePnP", "CIGM", "dims",
                      "dimensions", "intrinsics"):
        assert forbidden not in names, forbidden
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for forbidden in ("validation512", "wood45", "untouched", "eval56_manual"):
        assert not any(forbidden in v for v in literals), forbidden


def test_thresholds_are_derived_from_the_locked_baseline(arch):
    assert arch.BASELINE == {"angle_median": 5.5966, "offset_median": 2.2597}
    assert arch.REDUCTION == 0.40
    assert arch.THRESHOLD_40["angle_median"] == pytest.approx(3.35796, abs=1e-5)
    assert arch.THRESHOLD_40["offset_median"] == pytest.approx(1.35582, abs=1e-5)


def test_qualification_needs_both_metrics(arch):
    both = {"angle_median": 3.0, "offset_median": 1.0, "PASS": False, "SAFETY": False}
    one = {"angle_median": 3.0, "offset_median": 2.0, "PASS": False, "SAFETY": False}
    assert arch.qualify(both)["QUALIFIES"] and arch.qualify(both)["REDUCTION_40"]
    assert not arch.qualify(one)["QUALIFIES"]
    assert arch.qualify(one)["PARTIAL"]
    passing = {"angle_median": 0.9, "offset_median": 0.4, "PASS": True, "SAFETY": True}
    assert arch.qualify(passing)["ABSOLUTE_PASS"]
    nosafety = dict(passing, SAFETY=False)
    assert not arch.qualify(nosafety)["ABSOLUTE_PASS"]


def test_the_winner_rule_is_pareto_not_a_new_score(arch):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "interpret"))
    assert "pareto" in body
    assert "winner" in body and "len(pareto) == 1" in body
    for forbidden in ("score =", "weighted", "0.5 *"):
        assert forbidden not in body, forbidden


def test_the_primary_population_is_the_holdout(arch):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "interpret"))
    assert '"D2_LINE_DEV512"' in body and "D0_SEEN512" not in body


def test_training_is_blocked_until_init_equivalence(arch):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert "INIT_NOT_EQUIVALENT" in body
    assert "role shuffle is not run" in source() or "NO_QUALIFYING_ARM" in body


def test_role_shuffle_uses_the_locked_margins(arch):
    assert (arch.CAP.SHUFFLE_ANGLE_MARGIN, arch.CAP.SHUFFLE_OFFSET_MARGIN) == (5.0, 2.0)
    assert arch.CAP.DERANGEMENT == arch.CAP.SLM.DERANGEMENT


def test_init_equivalence_when_run(arch):
    report = load_json("init_equivalence.json")
    assert report["max_pair_delta"] <= arch.INIT_TOLERANCE
    assert report["INIT_EQUIVALENT"] is True
