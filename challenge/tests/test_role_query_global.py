"""Tests for the role-conditioned nonlocal decoding screen.

One factor moves, so the tests pin that the output representation is untouched,
that the global branch starts equal to the baseline without being dead, and that
nothing regresses a coordinate.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import numpy as np, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/line/role_query_global_screen.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def rq():
    spec = importlib.util.spec_from_file_location("RQ_UNDER_TEST", RUNNER)
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


def test_the_output_representation_is_untouched(rq):
    assert rq.MAP == 100 and rq.CAP.SIGMA_CELLS == 1.5
    assert rq.MARKS == (1250, 2500, 5000, 8515)
    assert (rq.CAP.ANGLE_BUDGET_DEG, rq.CAP.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.ClassDef) and node.name == "RoleQueryModel"))
    assert "CAP.SupportingLineHead" in body and "ARCH.AbsoluteXY" in body
    tree = ast.parse(source())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("MAP200", "RgbLineStem", "GlobalContentContext",
                      "solve_pose", "CIGM", "dims", "intrinsics"):
        assert forbidden not in names, forbidden


def test_the_global_branch_emits_a_map_residual_not_a_coordinate(rq):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.ClassDef) and node.name == "RoleQueryGlobal"))
    assert "einsum" in body and "brhw" in body           # a spatial residual
    for forbidden in ("theta", "rho", "to_centre", "to_direction", "Regress"):
        assert forbidden not in body, forbidden
    model, _ = rq.build_arm("Q1_ROLE_QUERY_GLOBAL")
    f50 = torch.randn(2, rq.F50_CHANNELS, rq.F50_GRID, rq.F50_GRID, device=rq.DEV)
    feature = torch.randn(2, rq.F50_CHANNELS, rq.MAP, rq.MAP, device=rq.DEV)
    residual, extra = model.global_role(f50, feature)
    assert residual.shape == (2, rq.ROLES, rq.MAP, rq.MAP)
    assert extra["descriptor"].shape == (2, rq.ROLES, rq.QUERY_DIM)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_global_reasoning_happens_on_f50_not_map100(rq):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.ClassDef) and node.name == "RoleQueryGlobal"))
    assert "grid=F50_GRID" in body
    model, _ = rq.build_arm("Q1_ROLE_QUERY_GLOBAL")
    assert model.global_role.coordinates.shape == (rq.F50_GRID ** 2, 2)
    del model


def test_role_channels_are_fixed_with_no_matching(rq):
    text = source()
    for forbidden in ("linear_sum_assignment", "hungarian", "Hungarian"):
        assert forbidden not in text, forbidden
    model, _ = rq.build_arm("Q1_ROLE_QUERY_GLOBAL")
    assert model.global_role.queries.num_embeddings == 12
    del model


def test_the_gate_starts_at_zero_and_is_not_dead(rq):
    model, _ = rq.build_arm("Q1_ROLE_QUERY_GLOBAL")
    assert float(model.global_role.alpha) == 0.0
    del model
    report = load_json("role_query_wiring.json")
    assert report["init_max_abs_diff"] <= rq.INIT_TOLERANCE
    assert report["INIT_EQUIVALENT"] is True
    assert report["alpha_grad_at_step0"] > 0
    assert report["alpha_after_one_step"] != 0.0
    assert report["attention_grad_norm_at_step2"] > 0
    assert report["GRADIENT_ALIVE"] is True


def test_the_baseline_is_read_at_full_precision(rq):
    assert rq.BASELINE_SOURCE == ("architecture_screen_results.json", "C_G0P1")
    reference = rq.baseline_reference()
    assert set(reference) == set(rq.MARKS)
    final = reference[max(rq.MARKS)]
    assert any(abs(v - round(v, 4)) > 0 for v in final.values())
    text = source()
    for literal in ("4.4705", "1.9697", "2.6823", "1.1818"):
        assert literal not in text, literal


def test_thresholds_derive_from_that_baseline(rq):
    limits = rq.thresholds()
    base = limits["baseline_full_precision"]
    assert limits["reduction_40"]["angle_median"] == pytest.approx(
        base["angle_median"] * 0.6, rel=1e-12)
    assert limits["reduction_40"]["offset_median"] == pytest.approx(
        base["offset_median"] * 0.6, rel=1e-12)
    assert limits["absolute"] == {"angle_median": 1.0, "offset_median": 0.5}
    assert limits["safety"] == {"angle_p90": 2.0, "offset_p90": 1.0}
    assert rq.REDUCTION == 0.40


def test_training_is_blocked_until_wiring_passes(rq):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert "GLOBAL_BRANCH_GRADIENT_WIRING_FAIL" in body
    assert "Q0_BASELINE_NOT_REPRODUCED" in body
    assert body.index("Q0_BASELINE_NOT_REPRODUCED") < body.index("decide(report)")


def test_promotion_requires_qualification_and_causality(rq):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert "NOT_QUALIFIED" in body
    assert "ROLE_QUERY_SEMANTICS_NOT_CAUSAL" in body
    assert (rq.CAP.SHUFFLE_ANGLE_MARGIN, rq.CAP.SHUFFLE_OFFSET_MARGIN) == (5.0, 2.0)


def test_the_decision_reads_the_last_mark_only(rq):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "decide"))
    assert "max(MARKS)" in body
    assert "D2_LINE_DEV512" in body and "D0_SEEN512" not in body
