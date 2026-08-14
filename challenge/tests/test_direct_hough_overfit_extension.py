"""Tests for the one-shot direct-Hough overfit budget extension.

Only the step budget may move, the historical record must survive, and the
extension must not be resumable from a checkpoint whose optimizer state was
never saved.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/line/direct_hough_overfit_extension.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def ext():
    spec = importlib.util.spec_from_file_location("EXT_UNDER_TEST", RUNNER)
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


def test_only_the_step_budget_changed(ext):
    assert ext.EXTENDED_MARKS == (1500, 3000, 4500, 6000)
    assert ext.DECISION_STEP == 6000
    # everything else is imported from the locked runner rather than redefined
    tree = ast.parse(source())
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for forbidden in ("DirectHoughHead", "DirectHoughModel", "lattice",
                      "target_distribution", "cross_entropy", "train_network",
                      "hypothesis_features"):
        assert forbidden not in defined, forbidden
    body = ast.get_source_segment(source(), next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"))
    assert "DH.train_network(pool, EXTENDED_MARKS" in body


def test_the_gate_is_the_locked_one(ext):
    entry = {"angle_median": 1.0, "offset_median": 0.5,
             "angle_p90": 2.0, "offset_p90": 1.0}
    assert all(ext.gates(entry).values())
    for key, bump in (("angle_median", 1.001), ("offset_median", 0.501),
                      ("angle_p90", 2.001), ("offset_p90", 1.001)):
        assert not ext.gates({**entry, key: bump})[key]
    assert (ext.CAP.ANGLE_BUDGET_DEG, ext.CAP.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (ext.CAP.SAFETY_ANGLE, ext.CAP.SAFETY_OFFSET) == (2.0, 1.0)
    numbers = {n.value for n in ast.walk(ast.parse(source()))
               if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    for moved in (0.55, 0.5237, 2.1, 2.0957):
        assert moved not in numbers, moved


def test_the_trajectory_is_fresh_and_never_resumed(ext):
    tree = ast.parse(source())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("load_state_dict", "resume", "checkpoint_path"):
        assert forbidden not in names, forbidden
    plan = load_json("direct_hough_extension_plan.json")
    assert plan["resume"] is False
    assert plan["extension_allowance"] == "once"


def test_the_historical_record_is_read_not_overwritten(ext):
    assert ext.HISTORICAL == "direct_hough_overfit.json"
    literals = {n.value for n in ast.walk(ast.parse(source()))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    written = {n.value for n in ast.walk(ast.parse(source()))
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and n.value.endswith(".json")}
    assert "direct_hough_overfit.json" in literals
    body = ast.get_source_segment(source(), next(
        n for n in ast.walk(ast.parse(source()))
        if isinstance(n, ast.FunctionDef) and n.name == "main"))
    # the historical file appears only as a read, never in a write_text target
    assert 'OUT / "direct_hough_overfit.json").write_text' not in source()
    assert "direct_hough_extension.json" in written


def test_the_reference_is_full_precision(ext):
    reference = ext.historical_reference()
    assert set(reference) == {1500, 3000}
    assert any(abs(v - round(v, 4)) > 0 for v in reference[3000].values())
    numbers = {n.value for n in ast.walk(ast.parse(source()))
               if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    for transcribed in (0.5978, 2.0957, 1.7284, 2.7528):
        assert transcribed not in numbers, transcribed


def test_there_is_no_second_extension(ext):
    numbers = {n.value for n in ast.walk(ast.parse(source()))
               if isinstance(n, ast.Constant) and isinstance(n.value, int)}
    for further in (9000, 12000):
        assert further not in numbers, further
    assert "DIRECT_HOUGH_NETWORK_FIT_FAIL_CONFIRMED" in source()


def test_full_is_gated_on_the_extension_not_the_historical_file(ext):
    body = ast.get_source_segment(source(), next(
        n for n in ast.walk(ast.parse(source()))
        if isinstance(n, ast.FunctionDef) and n.name == "main"))
    assert 'OUT / "direct_hough_extension.json"' in body
    assert '["EXTENDED_PASS"]' in body
    assert "FULL blocked" in body


def test_full_keeps_the_original_thresholds(ext):
    body = ast.get_source_segment(source(), next(
        n for n in ast.walk(ast.parse(source()))
        if isinstance(n, ast.FunctionDef) and n.name == "main"))
    assert "DH.thresholds()" in body and "DH.MARKS" in body
    assert "DIRECT_HOUGH_ROLE_HEATMAP_VALID" in body
    assert "DIRECT_HOUGH_LINE_NATIVE_SIGNAL" in body
    assert "DIRECT_HOUGH_ROLE_HEATMAP_FAIL" in body
