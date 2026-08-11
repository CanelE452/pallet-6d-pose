"""Tests for the L2-SP anchoring screen.

The penalty must pull toward the pretrained weights rather than toward zero and
must not replace the existing weight decay; trainability and optimizer groups
must match F1 exactly; lambda must come from a train-side calibration and never
from held-out geometry; and the run must stay blocked while that calibration is
unavailable.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/regularized_late_a1_full_adaptation.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("L2SP_UNDER_TEST", RUNNER)
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


def non_docstring_literals():
    root = tree()
    docstrings = set()
    for node in ast.walk(root):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    return {n.value for n in ast.walk(root)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings}


def strings_in(node):
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def load_json(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text("utf-8"))


def test_two_arms_and_one_factor(screen):
    assert screen.ARMS == ("S0_F1_HISTORICAL", "S1_F1_PLUS_L2SP")
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    for forbidden in ("F50LineAdapter", "RoleRefinementBlock", "LowRankDelta",
                      "LowRankLateA1", "position", "AbsoluteXY"):
        assert forbidden not in names, forbidden


def test_the_penalty_pulls_toward_w0_not_zero(screen):
    """A weight moved off W0 must be penalised; a weight at W0 must not be,
    however large it is -- that is what distinguishes L2-SP from weight decay."""
    a1 = screen.LATE.AdaptableA1(screen.FIRST_TRAINABLE_INDEX)
    reference = screen.SPReference(a1)
    assert float(reference.penalty(a1)) == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():
        name, parameter = screen.late_parameters(a1)[0]
        parameter.add_(0.01)
    assert float(reference.penalty(a1)) > 0.0
    del a1


def test_the_reference_is_immutable_and_verified(screen):
    a1 = screen.LATE.AdaptableA1(screen.FIRST_TRAINABLE_INDEX)
    reference = screen.SPReference(a1)
    assert reference.verify() is True
    assert all(not t.requires_grad for _, t in reference.entries)
    assert len(reference.audit) == len(reference.entries)
    for row in reference.audit:
        assert set(row) >= {"name", "shape", "numel", "norm", "sha256"}
    with torch.no_grad():
        reference.entries[0][1].add_(1.0)
    with pytest.raises(RuntimeError, match="SP reference mutated"):
        reference.verify()
    del a1


def test_the_penalty_is_normalised_and_averaged():
    body = ast.get_source_segment(source(), function("penalty"))
    assert "reference).pow(2).sum()" in body
    assert "reference.pow(2).sum() + SP_EPS" in body
    assert ".mean()" in body


def test_weight_decay_is_not_replaced(screen):
    body = ast.get_source_segment(source(), function("optimiser_for"))
    assert "weight_decay=CAP.WD" in body
    assert screen.CAP.WD == 1e-4


def test_optimizer_groups_match_f1(screen):
    body = ast.get_source_segment(source(), function("optimiser_for"))
    assert "CAP.LR * A1_LR_SCALE" in body
    assert screen.A1_LR_SCALE == 0.1
    assert screen.LATE.A1_LR_SCALE == screen.A1_LR_SCALE
    assert screen.FIRST_TRAINABLE_INDEX == screen.LATE.FIRST_TRAINABLE_INDEX


def test_trainability_matches_f1(screen):
    assert screen.EXPECTED_TRAINABLE == 5014912
    body = ast.get_source_segment(source(), function("build_arm"))
    assert "L2SP_TRAINABILITY_MISMATCH" in body


def test_lambda_is_calibrated_from_train_side_only():
    body = ast.get_source_segment(source(), function("run_calibration"))
    assert "train_loss_mean_last250" in body
    for forbidden in ("D2_LINE_DEV512", "D0_SEEN512", "angle_median",
                      "offset_median"):
        assert forbidden not in body, forbidden
    assert "ce_ref / r_sp_ref" in body


def test_lambda_optimality_is_disclaimed(screen):
    body = ast.get_source_segment(source(), function("run_calibration"))
    assert "LAMBDA_OPTIMALITY_NOT_ESTABLISHED" in body
    plan = ast.get_source_segment(source(), function("build_plan"))
    assert "LAMBDA_OPTIMALITY_NOT_ESTABLISHED" in plan
    assert '"lambda_sweep": False' in plan


def test_no_lambda_literal_is_hardcoded():
    constants = {n.value for n in ast.walk(tree())
                 if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    for arbitrary in (1e-3, 1e-2, 0.001, 0.01, 0.1, 1.0):
        if arbitrary in (0.1, 1.0):
            continue          # A1_LR_SCALE and neutral factors are not lambda
        assert arbitrary not in constants, arbitrary


def test_missing_calibration_blocks_everything(screen):
    body = ast.get_source_segment(source(), function("locked_lambda"))
    assert "SP_CALIBRATION_REFERENCE_MISSING" in body
    main = ast.get_source_segment(source(), function("main"))
    assert "lambda_sp = locked_lambda()" in main
    calibrate = main.index('== "calibrate"')
    locked = main.index("lambda_sp = locked_lambda()")
    assert calibrate < locked, "calibrate must be reachable without a lambda"


def test_the_reference_weights_are_read_not_reconstructed():
    body = ast.get_source_segment(source(), function("adapted_late_weights"))
    assert "torch.load" in body
    names = {n.attr for n in ast.walk(function("run_calibration"))
             if isinstance(n, ast.Attribute)}
    for forbidden in ("train_network", "train_arm", "step_schedule"):
        assert forbidden not in names, forbidden


def test_gradient_sanity_does_not_demand_a_zero_gradient(screen):
    """At W == W0 the SP gradient is zero by construction."""
    body = ast.get_source_segment(source(), function("run_gradient_sanity"))
    assert "optimiser.step()" in body
    assert "at_zero <= STEP0_TOLERANCE" in body
    assert "sp_grad_positive" in body


def test_step0_requires_function_equivalence(screen):
    assert screen.STEP0_TOLERANCE == 1e-6
    body = ast.get_source_segment(source(), function("run_step0"))
    for key in ("f50", "descriptor", "logits", "task_loss"):
        assert f'gaps["{key}"]' in body
    assert "sp_value <= STEP0_TOLERANCE" in body


def test_the_schedule_and_gate_are_the_locked_ones(screen):
    assert screen.MARKS == (1703, 5000, 8515, 17030, 25545)
    assert screen.DECISION_STEP == 25545
    assert screen.CAP.BATCH == 8
    assert (screen.CAP.ANGLE_BUDGET_DEG, screen.CAP.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (screen.CAP.SAFETY_ANGLE, screen.CAP.SAFETY_OFFSET) == (2.0, 1.0)


def test_g3_and_g5_are_preregistered(screen):
    assert screen.G3_MAX_DEGRADATION == 0.10
    assert screen.G3_MIN_GAP_CLOSURE == 0.20
    assert screen.G5_MEDIAN_BAND == 0.05
    assert screen.G5_GAP_BAND == 0.10


def test_baselines_are_loaded_not_transcribed(screen):
    assert screen.F1_RESULT == "late_a1_adaptation.json"
    body = ast.get_source_segment(source(), function("f1_reference"))
    assert "json.loads" in body
    constants = {n.value for n in ast.walk(tree())
                 if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    for hardcoded in (2.070244, 1.077348, 9.702057, 4.125696, 3.735687):
        assert hardcoded not in constants


def test_no_composite_score_is_built():
    body = ast.get_source_segment(source(), function("judge"))
    names = {n.id for n in ast.walk(function("judge")) if isinstance(n, ast.Name)}
    for forbidden in ("score", "composite", "combined"):
        assert forbidden not in names, forbidden
    assert "conditions" in body


def test_diagnostics_cannot_decide():
    for node in ast.walk(function("judge")):
        if not isinstance(node, ast.If):
            continue
        assigns_decision = any(
            isinstance(inner, ast.Assign)
            and any(isinstance(t, ast.Subscript)
                    and isinstance(t.slice, ast.Constant)
                    and t.slice.value == "DECISION" for t in inner.targets)
            for inner in ast.walk(node))
        if not assigns_decision:
            continue
        guard = strings_in(node.test)
        for forbidden in ("context_only", "weight_drift", "feature_drift",
                          "D0_SEEN512"):
            assert forbidden not in guard, forbidden


def test_the_causal_limit_is_recorded():
    body = ast.get_source_segment(source(), function("judge"))
    assert "CAUSAL_LIMIT" in body
    assert "drift is not shown to" in body


def test_instability_forbids_a_retry():
    body = ast.get_source_segment(source(), function("judge"))
    assert "REGULARIZED_LATE_A1_UNSTABLE" in body
    assert '"RETRY_WITH_NEW_LAMBDA"] = "FORBIDDEN"' in body


def test_no_scheduler_or_clipping_or_early_stopping():
    names = {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    for forbidden in ("clip_grad_norm_", "clip_grad_value_", "lr_scheduler",
                      "StepLR", "CosineAnnealingLR", "early_stop", "patience"):
        assert forbidden not in names, forbidden


def test_forbidden_stages_are_absent():
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    for forbidden in ("solve_pose", "solvePnP", "cigm", "CIGM", "dimensions"):
        assert forbidden not in names, forbidden
    literals = non_docstring_literals()
    for sealed in ("validation512", "untouched", "eval56", "wood45",
                   "final_test", "capturenight", "handannot17", "MAP200"):
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
    written = set()
    for node in ast.walk(tree()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_text"):
            target = node.func.value
            if isinstance(target, ast.BinOp) and isinstance(target.right, ast.Constant):
                written.add(target.right.value)
    assert written == {"l2sp_calibration.json", "l2sp_plan.json",
                       "l2sp_step0.json", "l2sp_gradient.json",
                       "l2sp_memory.json", "l2sp_result.json"}


def test_the_recorded_calibration_state():
    """Either lambda is locked, or the block is recorded with its reason."""
    report = load_json("l2sp_calibration.json")
    assert report["LAMBDA_OPTIMALITY_NOT_ESTABLISHED"] is True
    assert report["calibration_step"] == 1703
    assert report["CE_ref"] > 0
    if report["LAMBDA_LOCKED"]:
        assert report["R_SP_ref"] > 0
        assert report["lambda_sp"] == pytest.approx(
            report["CE_ref"] / report["R_SP_ref"], rel=1e-12)
    else:
        assert report["BLOCK"] == "SP_CALIBRATION_REFERENCE_MISSING"
        assert report["R_SP_ref"] is None and report["lambda_sp"] is None
        assert report["checkpoint"]["late_weight_tensors_found"] == 0


def test_result_is_internally_consistent():
    report = load_json("l2sp_result.json")
    v, h = report["verdict"], report["history"]
    assert sorted(map(int, h)) == [1703, 5000, 8515, 17030, 25545]
    s1 = h["25545"]["D2_LINE_DEV512"]
    assert v["ABSOLUTE_PASS"] == (s1["PASS"] and s1["SAFETY"])
    assert v["CIGM"] == "BLOCKED"
    for mark in h:
        assert "weight_drift" in h[mark] and "sp_mean_last250" in h[mark]
