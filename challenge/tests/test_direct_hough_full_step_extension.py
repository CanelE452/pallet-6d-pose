"""Tests for the direct-Hough long-schedule exposure screen.

Only optimizer exposure may move.  The architecture must come from the locked
runner rather than be redefined here, the recorded FULL must survive untouched,
the decision must sit at 25,545 on the dev population alone, and the verdict
thresholds must be the ones pre-registered before the run.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/line/direct_hough_full_step_extension.py"
LOCKED = ROOT / "scripts/stage0/line/direct_hough_role_heatmap.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def long_screen():
    spec = importlib.util.spec_from_file_location("LONG_UNDER_TEST", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def source():
    return RUNNER.read_text("utf-8")


def tree():
    return ast.parse(source())


def load_json(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text("utf-8"))


def function(name):
    return next(n for n in ast.walk(tree())
                if isinstance(n, ast.FunctionDef) and n.name == name)


def calls_in(name):
    """Every call target inside one function, as identifiers."""
    found = set()
    for node in ast.walk(function(name)):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Attribute):
                found.add(target.attr)
            elif isinstance(target, ast.Name):
                found.add(target.id)
    return found


def test_the_exposure_is_three_times_the_recorded_one(long_screen):
    assert long_screen.PASSES == 15
    assert long_screen.RECORDED_PASSES == 5
    assert long_screen.DECISION_STEP == 25545
    assert long_screen.LONG_MARKS == (1703, 5000, 8515, 17030, 25545)
    assert 1703 * long_screen.RECORDED_PASSES == 8515
    assert 1703 * long_screen.PASSES == long_screen.DECISION_STEP


def test_the_architecture_is_not_redefined_here():
    defined = {n.name for n in ast.walk(tree())
               if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for forbidden in ("DirectHoughHead", "DirectHoughModel", "HypothesisEncoder",
                      "lattice", "target_distribution", "cross_entropy",
                      "hypothesis_features", "evaluate_network", "batch_rows",
                      "encoder_features", "decode", "measure", "summarise",
                      "thresholds"):
        assert forbidden not in defined, forbidden


def test_the_training_loop_uses_only_locked_pieces():
    used = calls_in("train_long")
    for piece in ("lattice", "hypothesis_features", "DirectHoughModel",
                  "step_schedule", "load_pack", "batch_rows",
                  "target_distribution", "encoder_features", "cross_entropy",
                  "evaluate_network", "AdamW"):
        assert piece in used, piece


def test_the_dead_position_branch_is_neither_deleted_nor_called():
    """It consumes RNG before the encoder; touching it changes every init."""
    names = {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    assert "position" not in names
    assert "AbsoluteXY" not in names
    locked = ast.parse(LOCKED.read_text("utf-8"))
    model = next(n for n in ast.walk(locked)
                 if isinstance(n, ast.ClassDef) and n.name == "DirectHoughModel")
    assigned = {t.attr for n in ast.walk(model) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Attribute)}
    assert "position" in assigned


def test_the_optimizer_settings_are_the_locked_ones(long_screen):
    body = ast.get_source_segment(source(), function("train_long"))
    assert "lr=CAP.LR" in body and "weight_decay=CAP.WD" in body
    assert long_screen.CAP.BATCH == 8
    assert long_screen.CAP.SEED == 1


def test_the_gate_is_untouched(long_screen):
    cap = long_screen.CAP
    assert (cap.ANGLE_BUDGET_DEG, cap.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (cap.SAFETY_ANGLE, cap.SAFETY_OFFSET) == (2.0, 1.0)
    constants = {n.value for n in ast.walk(tree())
                 if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    for hardcoded in (4.179304122924805, 1.8787918090820312, 2.507582, 1.127275):
        assert hardcoded not in constants


def test_the_verdict_thresholds_are_preregistered(long_screen):
    assert long_screen.WEAK_IMPROVEMENT == 0.20
    assert long_screen.GEOMETRY_PLATEAU == 0.05
    assert long_screen.CE_PLATEAU_DROP == 0.02
    assert long_screen.CE_PLATEAU_SLOPE == -1e-5
    assert long_screen.CE_STRONG_DROP == 0.10
    scope = (ROOT / "_docs/audits/eval56_summary/canonical_corner_audit"
             / "edge_mandatory_fast_search"
             / "DIRECT_HOUGH_FULL_OPTIMIZATION_SCOPE.md").read_text("utf-8")
    for token in ("WEAK_IMPROVEMENT", "GEOMETRY_PLATEAU", "CE_PLATEAU",
                  "CE_STRONG_DROP"):
        assert token in scope, token


def test_resume_is_impossible_and_declared(long_screen):
    body = ast.get_source_segment(source(), function("build_plan"))
    assert '"resume": False' in body and '"fresh_init": True' in body
    names = {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    assert "load_state_dict" not in names


def test_the_decision_uses_the_dev_population_only(long_screen):
    body = ast.get_source_segment(source(), function("verdict"))
    assert "D2_LINE_DEV512" in body
    assert "D0_SEEN512" not in body


def test_per_role_is_recorded_late_and_never_selected(long_screen):
    assert long_screen.PER_ROLE_MARKS == (8515, 17030, 25545)
    assert "per_role" not in ast.get_source_segment(source(), function("verdict"))


def test_the_recorded_full_is_read_at_full_precision(long_screen):
    body = ast.get_source_segment(source(), function("recorded_full"))
    assert "json.loads" in body
    assert long_screen.RECORDED_FULL == "direct_hough_full.json"
    assert long_screen.RECORDED_MARK == "8515"


def test_reproduction_carries_no_hard_block():
    node = function("reproduction")
    assert not [n for n in ast.walk(node) if isinstance(n, ast.Raise)]
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    names |= {n.value for n in ast.walk(node)
              if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for invented in ("tolerance", "PARITY_TOLERANCE", "RuntimeError"):
        assert invented not in names, invented


def test_the_runner_never_writes_the_recorded_results():
    written = set()
    for node in ast.walk(tree()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_text"):
            target = node.func.value
            if isinstance(target, ast.BinOp) and isinstance(target.right, ast.Constant):
                written.add(target.right.value)
    assert written == {"direct_hough_long_plan.json",
                       "direct_hough_long_parity.json", "direct_hough_long.json"}


def test_the_checkpoint_tag_does_not_collide_with_the_recorded_run(long_screen):
    assert long_screen.TAG == "long"
    recorded = long_screen.CAP.checkpoint_path("DH_full", "step_08515")
    fresh = long_screen.CAP.checkpoint_path(f"DH_{long_screen.TAG}", "step_08515")
    assert recorded != fresh


def test_forbidden_stages_are_absent():
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    for forbidden in ("solve_pose", "solvePnP", "cigm", "CIGM", "dimensions"):
        assert forbidden not in names, forbidden
    literals = {n.value for n in ast.walk(tree())
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for sealed in ("validation512", "untouched", "eval56", "wood45",
                   "final_test", "capturenight", "handannot17", "MAP200"):
        assert not any(sealed in v for v in literals), sealed


def test_every_subcommand_is_reachable():
    """A choice argparse accepts but no branch handles runs and does nothing."""
    main = function("main")
    choices = next(kw.value for node in ast.walk(main)
                   if isinstance(node, ast.Call)
                   and getattr(node.func, "attr", "") == "add_argument"
                   for kw in node.keywords if kw.arg == "choices")
    declared = {c.value for c in choices.elts}
    body = ast.get_source_segment(source(), main)
    guarded = {c for c in declared if f'== "{c}"' in body}
    assert declared - guarded <= {"run"}, declared - guarded
    assert "run" in declared


def test_plan_matches_the_locked_arithmetic():
    plan = load_json("direct_hough_long_plan.json")
    assert plan["exposure"]["steps_per_pass"] == 1703
    assert plan["exposure"]["recorded_decision"] == 8515
    assert plan["exposure"]["long_decision"] == 25545
    assert plan["exposure"]["exposure_ratio"] == 3.0
    assert plan["frames"] == 13618
    assert plan["resume"] is False and plan["fresh_init"] is True
    assert plan["recorded_checkpoint"]["has_optimizer_state"] is False
    assert plan["recorded_full_verdict"] == "DIRECT_HOUGH_ROLE_HEATMAP_FAIL"


def test_parity_is_judged_with_the_nondeterminism_removed(long_screen):
    """Bit equality under the default kernels is not available -- the locked
    runner fails it against itself -- so parity is asked in deterministic mode
    and the default-mode spread is recorded rather than asserted away."""
    body = ast.get_source_segment(source(), function("run_parity"))
    assert "use_deterministic_algorithms" in body
    assert long_screen.DETERMINISTIC_WORKSPACE == ":4096:8"
    run_body = ast.get_source_segment(source(), function("train_long"))
    assert "use_deterministic_algorithms" not in run_body


def test_parity_holds_if_it_was_measured():
    parity = load_json("direct_hough_long_parity.json")
    assert parity["DETERMINISTIC_MODE_VERIFIED"] is True
    assert parity["deterministic_control"]["max_abs_delta"] == 0.0
    assert parity["structural_parity"]["max_abs_delta"] == 0.0
    assert parity["TRAINING_PATH_PARITY"] is True


def test_the_recorded_full_result_is_unchanged():
    recorded = load_json("direct_hough_full.json")
    assert recorded["verdict"]["DECISION"] == "DIRECT_HOUGH_ROLE_HEATMAP_FAIL"
    assert sorted(recorded["history"]) == ["1250", "2500", "5000", "8515"]
    assert "shuffle" not in recorded


def test_long_result_is_internally_consistent():
    report = load_json("direct_hough_long.json")
    history, verdict = report["history"], report["verdict"]
    assert sorted(map(int, history)) == [1703, 5000, 8515, 17030, 25545]
    final = history["25545"]["D2_LINE_DEV512"]
    assert verdict["ABSOLUTE_PASS"] == (final["PASS"] and final["SAFETY"])
    base = report["thresholds"]["baseline_full_precision"]
    assert verdict["angle_reduction_vs_baseline"] == pytest.approx(
        1.0 - final["angle_median"] / base["angle_median"], rel=1e-12)
    for mark in ("1703", "5000"):
        assert "per_role" not in history[mark]["D2_LINE_DEV512"]
    for mark in ("8515", "17030", "25545"):
        assert len(history[mark]["D2_LINE_DEV512"]["per_role"]) == 12
