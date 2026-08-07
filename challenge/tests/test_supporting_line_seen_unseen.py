"""Tests for the seen-versus-unseen checkpoint diagnostic.

Nothing trains here, so what must be pinned is that the three populations are
built before any forward pass, are disjoint, and differ only in whether the
model saw the frame.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/supporting_line_seen_unseen.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def diag():
    spec = importlib.util.spec_from_file_location("DIAG_UNDER_TEST", RUNNER)
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


def test_nothing_is_trained(diag):
    tree = ast.parse(source())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("backward", "AdamW", "step_schedule", "train_steps",
                      "map_loss", "zero_grad"):
        assert forbidden not in names, forbidden
    for forbidden in ("MAP200", "solve_pose", "CIGM", "validation512",
                      "wood45", "handannot17"):
        assert forbidden not in source(), forbidden


def test_the_allocation_is_deterministic(diag):
    counts = {"b": 30, "a": 10, "c": 60}
    first = diag.proportional_quota(counts, 100, 10)
    second = diag.proportional_quota(dict(reversed(list(counts.items()))), 100, 10)
    assert first == second
    assert sum(first.values()) == 10
    tree = ast.parse(source())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "default_rng" not in names and "shuffle" not in names


def test_d0_and_d1_share_a_group_histogram_and_no_frames(diag):
    record = load_json("seen_unseen_manifests.json")
    assert record["D0_SEEN512"]["frames"] == 512
    assert record["overlap"] == {"D0_D1": 0, "D0_DEV": 0, "D1_DEV": 0}
    assert record["shortfall"] == {}
    assert (record["D0_SEEN512"]["group_histogram"]
            == record["D1_TRAIN_UNSEEN512"]["group_histogram"])


def test_d0_really_is_frames_the_model_trained_on(diag):
    seen = set(diag.V2.manifest("line_search2k"))
    d0 = set(diag.read_manifest("D0_SEEN512"))
    d1 = set(diag.read_manifest("D1_TRAIN_UNSEEN512"))
    train, _ = diag.V2.split_indices()
    assert d0 <= seen
    assert d1 <= set(train) and not (d1 & seen)


def test_the_dev_population_is_reused_not_rebuilt():
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_checkpoints"))
    assert 'V2.manifest("line_dev512")' in body


def test_the_cause_taxonomy_reads_d0_first(diag):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "diagnose"))
    order = [body.index(k) for k in ("SEARCH2K_MODEL_UNDERFIT_CONFIRMED",
                                     "WITHIN_LINE_TRAIN_GENERALIZATION_GAP",
                                     "APPEARANCE_COMBINATION_GENERALIZATION_GAP",
                                     "HARD_BLOCKED_DIAGNOSTIC_INCONSISTENCY")]
    assert order == sorted(order)
    assert diag.PRIMARY_ARM == "M0_F50_SLINE"


def test_the_existing_checkpoints_are_the_only_model_source(diag):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_checkpoints"))
    assert 'CAP.checkpoint_path(name, f"search2k_epoch{epoch}")' in body
    assert "MISSING_CHECKPOINT" in body
    for epoch in (1, 3, 5):
        for arm in ("M0_F50_SLINE", "M1_F50_RGB_SLINE"):
            assert diag.CAP.checkpoint_path(arm, f"search2k_epoch{epoch}").exists()


def test_gates_are_unchanged(diag):
    assert (diag.CAP.ANGLE_BUDGET_DEG, diag.CAP.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (diag.CAP.SAFETY_ANGLE, diag.CAP.SAFETY_OFFSET) == (2.0, 1.0)
    assert diag.CAP.EPOCH_LADDER == (1, 3, 5)
