"""Tests for the L2-SP coefficient lock and the S1 run behind it.

The coefficient must come from gradients at a fresh one-pass state that is never
called historical, the calibration path must not touch any held-out population,
the lock must be reproducible and sha-bound, and S1 must stay blocked until it
is locked.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/adaptation/regularized_late_a1_l2sp_run.py"
BLOCKED = ROOT / "scripts/stage0/adaptation/regularized_late_a1_full_adaptation.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("L2SP_RUN_UNDER_TEST", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def source():
    return RUNNER.read_text("utf-8")


def tree():
    return ast.parse(source())


def function(name):
    return next(n for n in ast.walk(tree())
                if isinstance(n, ast.FunctionDef) and n.name == name)


def non_docstring_literals(node=None):
    root = node or tree()
    docstrings = set()
    for item in ast.walk(root):
        if isinstance(item, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(item, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    return {n.value for n in ast.walk(root)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings}


def load_json(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text("utf-8"))


def test_the_blocked_runner_is_not_modified():
    """d543529's HARD_BLOCK must survive; this file adds, it does not rewrite."""
    blocked = ast.parse(BLOCKED.read_text("utf-8"))
    literals = {n.value for n in ast.walk(blocked)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert any("SP_CALIBRATION_REFERENCE_MISSING" in v for v in literals)


def test_the_calibration_state_is_never_called_historical(screen):
    assert screen.CALIBRATION_STEPS == 1703
    body = ast.get_source_segment(source(), function("run_calibration"))
    assert '"SP_CALIBRATION_STATE_1PASS"' in body
    assert '"is_historical_f1": False' in body
    literals = non_docstring_literals()
    for forbidden in ("HISTORICAL_F1_1703", "historical_f1_1703"):
        assert forbidden not in literals, forbidden


def test_the_rule_is_gradient_balanced_not_loss_balanced(screen):
    body = ast.get_source_segment(source(), function("run_calibration"))
    assert "task_norm / sp_norm" in body
    assert "ONE_PASS_GRADIENT_BALANCED_L2SP" in body
    assert "ce_ref / r_sp_ref" not in body


def test_lambda_claims_are_disclaimed():
    body = ast.get_source_segment(source(), function("run_calibration"))
    for flag in ("LAMBDA_OPTIMALITY_NOT_ESTABLISHED",
                 "LAMBDA_SELECTED_WITH_DEV", "LAMBDA_SWEEP"):
        assert flag in body, flag


def test_no_lambda_literal_is_hardcoded(screen):
    constants = {n.value for n in ast.walk(tree())
                 if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    assert 150.3369063067523 not in constants
    for arbitrary in (1e-3, 1e-2, 0.01, 150.0):
        assert arbitrary not in constants, arbitrary


def test_the_calibration_path_touches_no_held_out_population(screen):
    guard = screen.leakage_guard()
    assert guard["CALIBRATION_LEAKAGE_GUARD_CLEAN"] is True
    assert guard["violations"] == {}
    assert set(guard["functions_checked"]) == {
        "run_calibration", "calibration_state", "accumulate_task_gradient",
        "sp_gradient"}
    for name in guard["functions_checked"]:
        used = non_docstring_literals(function(name))
        for forbidden in screen.FORBIDDEN_IN_CALIBRATION:
            assert not any(forbidden in v for v in used), (name, forbidden)


def test_calibration_uses_only_line_train(screen):
    body = ast.get_source_segment(source(), function("main"))
    assert "pool = V2.split_indices()[0]" in body
    calibration = ast.get_source_segment(source(), function("calibration_state"))
    assert "populations" not in calibration
    accumulate = ast.get_source_segment(source(),
                                        function("accumulate_task_gradient"))
    assert "optimiser" not in accumulate and "step()" not in accumulate


def test_the_two_numerical_regimes_are_separated(screen):
    calibration = ast.get_source_segment(source(), function("run_calibration"))
    assert "use_deterministic_algorithms(True)" in calibration
    assert screen.DETERMINISTIC_WORKSPACE == ":4096:8"
    training = ast.get_source_segment(source(), function("train_s1"))
    assert "use_deterministic_algorithms" not in training


def test_the_penalty_shares_a_denominator_per_module(screen):
    a1 = screen.LATE.AdaptableA1(screen.SP.FIRST_TRAINABLE_INDEX)
    reference = screen.ModuleSPReference(a1)
    assert len(reference.modules) == 4
    assert float(reference.penalty(a1)) == pytest.approx(0.0, abs=1e-12)
    body = ast.get_source_segment(source(), function("penalty"))
    assert "denominator + SP_EPS" in body
    assert "conv.bias - entry" in body
    del a1


def test_the_reference_is_immutable_and_verified(screen):
    a1 = screen.LATE.AdaptableA1(screen.SP.FIRST_TRAINABLE_INDEX)
    reference = screen.ModuleSPReference(a1)
    assert reference.verify() is True
    with torch.no_grad():
        reference.modules[0]["weight"].add_(1.0)
    with pytest.raises(RuntimeError, match="SP reference mutated"):
        reference.verify()
    del a1


def test_the_calibration_checkpoint_is_never_an_initialisation():
    body = ast.get_source_segment(source(), function("run_calibration"))
    assert "coefficient audit only, never an S1 initialisation" in body
    train = ast.get_source_segment(source(), function("train_s1"))
    for forbidden in ("load_state_dict", "checkpoint_path()", "torch.load"):
        assert forbidden not in train, forbidden


def test_the_run_is_sha_bound_to_the_locked_checkpoint():
    body = ast.get_source_segment(source(), function("locked_coefficient"))
    assert "checkpoint_sha256" in body
    assert "sha mismatch" in body
    assert "SP_COEFFICIENT_NOT_LOCKED" in body


def test_repeatability_is_required_and_not_averaged(screen):
    assert screen.REPEAT_TOLERANCE == 1e-8
    body = ast.get_source_segment(source(), function("main"))
    assert "SP_COEFFICIENT_NOT_REPRODUCIBLE" in body
    assert "mean(" not in body.split("repeat")[1][:400]


def test_gradient_sanity_expects_zero_at_w0():
    body = ast.get_source_segment(source(), function("run_gradient_sanity"))
    assert "at_zero <= STEP0_TOLERANCE" in body
    assert "optimiser.step()" in body


def test_g_definitions_are_inherited_unchanged(screen):
    assert screen.SP.G3_MAX_DEGRADATION == 0.10
    assert screen.SP.G3_MIN_GAP_CLOSURE == 0.20
    assert screen.SP.G5_MEDIAN_BAND == 0.05
    assert screen.SP.G5_GAP_BAND == 0.10
    defined = {n.name for n in ast.walk(tree())
               if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    assert "judge" not in defined


def test_the_schedule_and_gate_are_the_locked_ones(screen):
    assert screen.MARKS == (1703, 5000, 8515, 17030, 25545)
    assert screen.DECISION_STEP == 25545
    assert screen.CAP.BATCH == 8
    assert (screen.CAP.ANGLE_BUDGET_DEG, screen.CAP.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (screen.CAP.SAFETY_ANGLE, screen.CAP.SAFETY_OFFSET) == (2.0, 1.0)


def test_no_scheduler_clipping_or_early_stopping():
    names = {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    for forbidden in ("clip_grad_norm_", "lr_scheduler", "StepLR",
                      "early_stop", "patience"):
        assert forbidden not in names, forbidden


def guard_list_literals():
    """The names inside FORBIDDEN_IN_CALIBRATION itself.

    That tuple exists precisely to name the sets the calibration must not touch,
    so its own entries are declarations of a prohibition, not uses of it.  A
    scan that cannot tell those apart fails on the guard that protects it.
    """
    for node in ast.walk(tree()):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id == "FORBIDDEN_IN_CALIBRATION"):
                    return {n.value for n in ast.walk(node.value)
                            if isinstance(n, ast.Constant)
                            and isinstance(n.value, str)}
    return set()


def test_forbidden_stages_are_absent():
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    for forbidden in ("solve_pose", "solvePnP", "cigm", "CIGM", "dimensions",
                      "position", "AbsoluteXY", "F50LineAdapter",
                      "RoleRefinementBlock", "LowRankDelta"):
        assert forbidden not in names, forbidden
    literals = non_docstring_literals() - guard_list_literals()
    for sealed in ("validation512", "wood45", "handannot17", "MAP200"):
        assert not any(sealed in v for v in literals), sealed


def test_every_subcommand_is_reachable():
    main = function("main")
    choices = next(kw.value for node in ast.walk(main)
                   if isinstance(node, ast.Call)
                   and getattr(node.func, "attr", "") == "add_argument"
                   for kw in node.keywords if kw.arg == "choices")
    declared = {c.value for c in choices.elts}
    body = ast.get_source_segment(source(), main)
    guarded = {c for c in declared if f'== "{c}"' in body}
    assert declared - guarded <= {"run"}, declared - guarded


def test_the_runner_never_writes_earlier_results():
    """Some paths are written through a local, so the locals are resolved too."""
    root = tree()
    aliases = {}
    for node in ast.walk(root):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.right, ast.Constant)
                and isinstance(node.value.right.value, str)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = node.value.right.value
    written = set()
    for node in ast.walk(root):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_text"):
            target = node.func.value
            if isinstance(target, ast.BinOp) and isinstance(target.right, ast.Constant):
                written.add(target.right.value)
            elif isinstance(target, ast.Name) and target.id in aliases:
                written.add(aliases[target.id])
    assert written == {"l2sp_coefficient_calibration.json",
                       "l2sp_coefficient_repeat.json",
                       "l2sp_coefficient_lock.json", "l2sp_plan.json",
                       "l2sp_step0.json", "l2sp_gradient.json",
                       "l2sp_memory.json", "l2sp_result.json"}
    assert "l2sp_calibration.json" not in written


def test_the_historical_block_record_is_intact():
    report = load_json("l2sp_calibration.json")
    assert report["BLOCK"] == "SP_CALIBRATION_REFERENCE_MISSING"
    assert report["LAMBDA_LOCKED"] is False
    assert report["checkpoint"]["late_weight_tensors_found"] == 0


def test_the_lock_if_measured():
    locked = load_json("l2sp_coefficient_lock.json")
    assert locked["SP_COEFFICIENT_LOCKED"] is True
    assert locked["is_historical_f1"] is False
    assert locked["rule"] == "ONE_PASS_GRADIENT_BALANCED_L2SP"
    assert locked["state_name"] == "SP_CALIBRATION_STATE_1PASS"
    assert locked["lambda_sp"] == pytest.approx(
        locked["task_grad_norm"] / locked["sp_unit_grad_norm"], rel=1e-12)
    assert locked["task_grad_norm"] > 0 and locked["sp_unit_grad_norm"] > 0
    assert locked["repeat"]["SP_COEFFICIENT_REPRODUCIBLE"] is True
    assert locked["repeat"]["relative_difference"] <= 1e-8
    assert locked["leakage_guard"]["CALIBRATION_LEAKAGE_GUARD_CLEAN"] is True
    assert locked["CALIBRATION_NUMERICAL_REGIME"] == "deterministic"
    assert locked["ACTUAL_TRAINING_NUMERICAL_REGIME"] == "default"
    assert locked["LAMBDA_OPTIMALITY_NOT_ESTABLISHED"] is True
    assert locked["supersedes"] == "SP_CALIBRATION_REFERENCE_MISSING"


def test_calibration_used_the_whole_train_split():
    report = load_json("l2sp_coefficient_calibration.json")
    assert report["frames_accumulated"] == 13618
    assert report["calibration_steps"] == 1703
    assert report["deterministic"] is True


def test_result_is_internally_consistent():
    report = load_json("l2sp_result.json")
    v, h = report["verdict"], report["history"]
    assert sorted(map(int, h)) == [1703, 5000, 8515, 17030, 25545]
    s1 = h["25545"]["D2_LINE_DEV512"]
    assert v["ABSOLUTE_PASS"] == (s1["PASS"] and s1["SAFETY"])
    assert v["CIGM"] == "BLOCKED"
    assert report["coefficient"]["SP_COEFFICIENT_LOCKED"] is True
    for mark in h:
        assert h[mark]["lambda_sp"] == report["coefficient"]["lambda_sp"]
        assert "weight_drift" in h[mark] and "scaled_sp_mean_last250" in h[mark]
