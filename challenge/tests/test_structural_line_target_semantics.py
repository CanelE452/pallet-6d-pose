"""Tests for the target-semantics oracle.

The screen is a single-factor comparison, so most of what matters is that only
the target differs: the decoder, lattice, gates and population must come from
the locked Hough screen unchanged.
"""
from __future__ import annotations

import ast, importlib.util, json, math, pathlib, sys

import numpy as np, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/line/structural_line_target_semantics.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def sem():
    spec = importlib.util.spec_from_file_location("SEM_UNDER_TEST", RUNNER)
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


def test_the_supporting_line_target_matches_its_closed_form(sem):
    theta = np.array([[math.radians(30.0)]]); rho = np.array([[20.0]])
    tube = sem.raster_supporting_line(theta, rho, np.ones((1, 1), bool))[0, 0]
    normal = np.array([math.cos(math.radians(30.0)), math.sin(math.radians(30.0))])
    yy, xx = np.meshgrid(np.arange(sem.MAP), np.arange(sem.MAP), indexing="ij")
    distance = np.abs(normal[0] * xx + normal[1] * yy
                      - 20.0 * (sem.MAP / sem.CANON))
    reference = np.exp(-distance ** 2 / (2 * sem.SIGMA_CELLS ** 2))
    assert np.abs(tube.cpu().numpy() - reference).max() < 1e-5
    assert float(tube.max()) == pytest.approx(1.0, abs=1e-5)


def test_segment_length_never_enters_the_supporting_line_target(sem):
    """Identifiers, not prose: the docstring says length never enters, which a
    substring search reads as using it."""
    function = next(node for node in ast.walk(ast.parse(source()))
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "raster_supporting_line")
    assert [a.arg for a in function.args.args][:3] == ["theta", "rho", "hit"]
    used = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    used |= {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}
    for forbidden in ("q0", "q1", "length", "half_length", "extent", "clip",
                      "visible_segments", "raster_targets"):
        assert forbidden not in used, forbidden


def test_an_unsupported_role_stays_exactly_zero(sem):
    theta = np.array([[0.5]]); rho = np.array([[20.0]])
    tube = sem.raster_supporting_line(theta, rho, np.zeros((1, 1), bool))
    assert float(tube.abs().max()) == 0.0


def test_only_the_target_differs(sem):
    assert sem.PRIMARY == sem.H.PRIMARY == "H2_ZERO_MEAN_NCC"
    assert sem.ONUM_GATE == sem.H.ONUM_GATE
    assert sem.OMAP_GATE == sem.H.OMAP_GATE
    assert (sem.SIGMA_CELLS, sem.MAP, sem.CANON) == (sem.H.SIGMA_CELLS,
                                                     sem.H.MAP, sem.H.CANON)
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "decode_both"))
    assert body.count("H.decode(maps, coarse, xx, yy)[PRIMARY]") == 1
    assert "SLM.raster_targets" in body and "raster_supporting_line" in body


def test_no_segment_aware_template_or_extent_predictor():
    text = source()
    for forbidden in ("half_length", "extent_predictor", "segment_template",
                      "solve_pose", "solvePnP", "CIGM", "validation512",
                      "wood45", "eval56_manual"):
        assert forbidden not in text, forbidden
    tree = ast.parse(source())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for forbidden in ("build_arm", "AdamW", "backward", "dims", "intrinsics"):
        assert forbidden not in names, forbidden


def test_the_population_is_not_touched(sem):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "run_linedev"))
    assert 'seg["hit"][frame]' in body            # the same supported roles
    onum = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "run_onum_target"))
    assert "H.synthetic_segments()" in onum        # the identical 10,000 lines
    assert "short_chord" not in onum or "keep" not in onum.split("short_chord")[0][-80:]


def test_the_declared_questions_are_fixed_before_the_run(sem):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "run_onum_target"))
    assert "Q1_interior_long_p99_le_gate" in body
    assert "Q2_short_chord_p99_le_gate" in body
    assert "FINITE_SEGMENT_EXTENT_MISMATCH_CONFIRMED" in body


def test_line_dev_is_blocked_until_the_supporting_line_onum_passes():
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert "SUPPORTING_LINE_TARGET_DECODER_FAIL" in body
    assert 'json.loads(onum.read_text())["ONUM_PASS"]' in body


def test_the_synthetic_set_is_the_same_ten_thousand(sem):
    a = sem.H.synthetic_segments()
    b = sem.H.synthetic_segments()
    assert a[0].shape[1] == 10000
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])


def test_line_dev_population_when_run(sem):
    report = load_json("target_semantics_linedev.json")
    for name in sem.TARGETS:
        assert report["targets"][name]["n"] == 27684
    assert "population_sha" in report
