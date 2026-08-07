"""Tests for the M0 data-versus-step scaling screen.

The screen's whole value is that one factor moves at a time, so the tests pin
the step arithmetic, the exposure accounting that makes B not a data condition,
and the reproduction guard on A.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/supporting_line_data_vs_step.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def scale():
    spec = importlib.util.spec_from_file_location("SCALE_UNDER_TEST", RUNNER)
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


def test_step_counts_are_checked_against_declared_values(scale):
    assert scale.EXPECTED == {"K2_steps_per_pass": 250, "FULL_steps_per_pass": 1703,
                              "S_SHORT": 1250, "S_LONG": 8515}
    got = scale.check_steps(scale.pools())
    assert got["S_SHORT"] == 1250 and got["S_LONG"] == 8515
    assert "HARD_BLOCK" in source()


def test_b_is_not_named_or_treated_as_a_data_scale_condition(scale):
    assert "B_FULL_PREFIX_SHORT" in scale.CONDITIONS
    assert "B_FULL_SHORT" not in source()
    plan = load_json("data_vs_step_plan.json")
    b = plan["exposure"]["B_FULL_PREFIX_SHORT"]
    assert b["unique_frames_seen"] < b["pool_frames"]      # pass not finished
    assert b["unseen_frames"] > 0 and b["unseen_groups"] > 0
    verdict = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "decide"))
    assert "B_FULL_PREFIX_SHORT" not in verdict            # never a causal input


def test_the_primary_data_contrast_is_c_against_d(scale):
    verdict = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "decide"))
    assert "C_to_D_angle_reduction" in verdict
    assert scale.CONDITIONS["C_K2_LONG"][1] == scale.CONDITIONS["D_FULL_LONG"][1]


def test_condition_a_must_reproduce_the_existing_checkpoint(scale):
    """The reference is the recorded run at full precision, not a transcription.

    A first version hardcoded four-decimal literals read off a report and
    compared them at 1e-6, so the guard measured the rounding and fired on an
    exact reproduction.
    """
    assert scale.REPRODUCTION_SOURCE == "seen_unseen_diagnostic.json"
    assert scale.REPRODUCTION_TOLERANCE == 1e-6
    reference = scale.reproduction_reference()
    assert set(reference) == {"D0_SEEN512", "D2_LINE_DEV512"}
    for label, entry in reference.items():
        assert set(entry) == set(scale.REPRODUCTION_KEYS)
        # full stored precision, not a four-decimal literal
        assert any(abs(v - round(v, 4)) > 0 for v in entry.values()), label
    text = source()
    for literal in ("6.6040", "2.7023", "6.8450", "2.7717"):
        assert f"({literal}" not in text and f" {literal})" not in text, literal
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert "CONDITION_A_NOT_REPRODUCED" in body
    assert body.index("CONDITION_A_NOT_REPRODUCED") < body.index("decide(report)")


def test_only_m0_and_no_architecture_change(scale):
    assert scale.ARM == "M0_F50_SLINE"
    text = source()
    for forbidden in ("M1_F50_RGB_SLINE", "MAP200", "global_context",
                      "coordinate_channel", "RgbLineStem", "CIGM", "solve_pose",
                      "validation512", "wood45"):
        assert forbidden not in text, forbidden


def test_no_filtering_or_deletion(scale):
    tree = ast.parse(source())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("unlink", "remove", "filter_frames", "exclude", "alias"):
        assert forbidden not in names, forbidden


def test_gates_and_thresholds_are_the_locked_ones(scale):
    assert scale.REDUCTION_THRESHOLD == 0.40
    assert (scale.CAP.ANGLE_BUDGET_DEG, scale.CAP.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (scale.CAP.APPROACH_ANGLE, scale.CAP.APPROACH_OFFSET) == (1.5, 0.75)
    assert scale.MARKS == (1250, 2500, 5000, 8515)


def test_exposure_is_recorded_for_every_condition(scale):
    plan = load_json("data_vs_step_plan.json")
    for name in scale.CONDITIONS:
        entry = plan["exposure"][name]
        for key in ("steps", "example_exposures", "unique_frames_seen",
                    "unique_groups_seen", "frame_visit", "unseen_frames"):
            assert key in entry, (name, key)
    assert plan["exposure"]["A_K2_SHORT"]["example_exposures"] == 10000
    assert plan["exposure"]["D_FULL_LONG"]["unique_frames_seen"] == 13618
