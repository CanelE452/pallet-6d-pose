"""Tests for the late-A1 adaptation screen.

One factor moves: whether net.vgg[19:27] receives gradient. The block boundary
must be read from the model rather than named, the learning rate must be the one
pre-registered, F0 may be reused only on proven code-path parity, and the
decision must sit at 25,545 on the dev population alone.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/late_a1_adaptation_screen.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("LATE_A1_UNDER_TEST", RUNNER)
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


def load_json(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text("utf-8"))


def test_two_arms_and_one_factor(screen):
    assert screen.ARMS == ("F0_FROZEN_A1", "F1_LATE_A1_TRAINABLE")
    assert screen.FIRST_TRAINABLE_INDEX == 19


def test_the_block_boundary_is_read_from_the_model(screen):
    """net.vgg[18] must be the last pooling layer, or 19 is an invented cut."""
    a1 = screen.AdaptableA1(None)
    children = dict(a1.vgg.named_children())
    assert isinstance(children["18"], torch.nn.MaxPool2d)
    later = [int(i) for i, m in children.items()
             if int(i) > 18 and isinstance(m, torch.nn.MaxPool2d)]
    assert later == []
    del a1


def test_unfreeze_touches_only_the_last_block(screen):
    frozen = screen.AdaptableA1(None)
    assert frozen.parameters_to_train() == []
    adapted = screen.AdaptableA1(screen.FIRST_TRAINABLE_INDEX)
    for index, child in adapted.vgg.named_children():
        for parameter in child.parameters():
            assert parameter.requires_grad == (int(index) >= 19), index
    outside = [p for name, p in adapted.inner.model.named_parameters()
               if not name.startswith("net.vgg.") and p.requires_grad]
    assert outside == []
    report = adapted.report()
    assert report["trainable_params"] == 5014912
    assert report["trainable_fraction_of_vgg"] == pytest.approx(0.683, abs=0.005)
    del frozen, adapted


def test_no_normalisation_so_eval_mode_is_inert(screen):
    a1 = screen.AdaptableA1(screen.FIRST_TRAINABLE_INDEX)
    report = a1.report()
    assert report["normalisation_layers"] == 0
    assert report["eval_mode"] is True
    del a1


def test_the_frozen_arm_still_detaches(screen):
    a1 = screen.AdaptableA1(None)
    images = torch.zeros(2, 3, 400, 400, device=screen.DEV)
    f50, _, _ = a1(images)
    assert f50.requires_grad is False
    del a1


def test_the_adapted_arm_carries_gradient(screen):
    a1 = screen.AdaptableA1(screen.FIRST_TRAINABLE_INDEX)
    images = torch.zeros(2, 3, 400, 400, device=screen.DEV)
    f50, _, _ = a1(images)
    assert f50.requires_grad is True
    f50.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() >= 0
               for p in a1.parameters_to_train())
    del a1


def test_the_a1_learning_rate_is_preregistered_and_unswept(screen):
    assert screen.A1_LR_SCALE == 0.1
    assert screen.CAP.LR == 1e-3
    body = ast.get_source_segment(source(), function("train_arm"))
    assert "CAP.LR * A1_LR_SCALE" in body
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    for forbidden in ("sweep", "lr_grid", "search_lr"):
        assert forbidden not in names


def test_everything_else_comes_from_the_locked_runner():
    used = set()
    for node in ast.walk(function("train_arm")):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Attribute):
                used.add(target.attr)
            elif isinstance(target, ast.Name):
                used.add(target.id)
    for piece in ("lattice", "hypothesis_features", "DirectHoughModel",
                  "step_schedule", "load_pack", "batch_rows",
                  "target_distribution", "cross_entropy", "AdamW"):
        assert piece in used, piece
    defined = {n.name for n in ast.walk(tree())
               if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for forbidden in ("DirectHoughHead", "DirectHoughModel", "HypothesisEncoder",
                      "lattice", "target_distribution", "cross_entropy",
                      "hypothesis_features", "decode", "measure", "summarise"):
        assert forbidden not in defined, forbidden


def test_the_dead_position_branch_is_untouched():
    names = {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    assert "position" not in names
    assert "AbsoluteXY" not in names


def test_f0_reuse_requires_proven_parity(screen):
    body = ast.get_source_segment(source(), function("main"))
    assert 'parity["F0_CODE_PATH_PARITY"]' in body
    assert "phase_a_history()" in body
    parity_body = ast.get_source_segment(source(), function("run_parity"))
    assert "use_deterministic_algorithms" in parity_body
    train_body = ast.get_source_segment(source(), function("train_arm"))
    assert "use_deterministic_algorithms" not in train_body


def test_the_decision_uses_the_dev_population_only():
    body = ast.get_source_segment(source(), function("judge"))
    assert "D2_LINE_DEV512" in body
    assert "D0_SEEN512" not in body


def test_the_gate_is_untouched(screen):
    cap = screen.CAP
    assert (cap.ANGLE_BUDGET_DEG, cap.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (cap.SAFETY_ANGLE, cap.SAFETY_OFFSET) == (2.0, 1.0)
    assert screen.DECISION_STEP == 25545
    assert screen.MARKS == (1703, 5000, 8515, 17030, 25545)
    constants = {n.value for n in ast.walk(tree())
                 if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    for hardcoded in (4.179304122924805, 1.8787918090820312, 3.735687, 1.972937):
        assert hardcoded not in constants


def test_similar_to_f0_is_preregistered(screen):
    assert screen.SIMILAR_TO_F0 == 0.05
    scope = (ROOT / "_docs/audits/eval56_summary/canonical_corner_audit"
             / "edge_mandatory_fast_search/LATE_A1_ADAPTATION_SCOPE.md").read_text("utf-8")
    for token in ("SIMILAR_TO_F0", "LATE_A1_ADAPTATION_OVERFITS",
                  "FROZEN_A1_NOT_PRIMARY_LIMIT", "LATE_A1_ADAPTATION_INCONCLUSIVE"):
        assert token in scope, token


def test_phase_b_is_blocked_if_phase_a_had_passed():
    body = ast.get_source_segment(source(), function("main"))
    assert "DIRECT_HOUGH_LONG_SCHEDULE_VALID_CANDIDATE" in body
    assert "PHASE_B_BLOCKED" in body


def test_the_runner_never_writes_earlier_results():
    written = set()
    for node in ast.walk(tree()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_text"):
            target = node.func.value
            if isinstance(target, ast.BinOp) and isinstance(target.right, ast.Constant):
                written.add(target.right.value)
    assert written == {"late_a1_plan.json", "late_a1_parity.json",
                       "late_a1_adaptation.json"}


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
    main = function("main")
    choices = next(kw.value for node in ast.walk(main)
                   if isinstance(node, ast.Call)
                   and getattr(node.func, "attr", "") == "add_argument"
                   for kw in node.keywords if kw.arg == "choices")
    declared = {c.value for c in choices.elts}
    body = ast.get_source_segment(source(), main)
    guarded = {c for c in declared if f'== "{c}"' in body}
    assert declared - guarded <= {"run"}, declared - guarded


def test_plan_records_the_measured_block():
    plan = load_json("late_a1_plan.json")
    assert plan["block"] == "net.vgg[19:27]"
    assert plan["a1"]["trainable_params"] == 5014912
    assert plan["a1"]["normalisation_layers"] == 0
    assert plan["a1_lr"] == pytest.approx(1e-4)
    assert plan["lr_sweep"] is False
    assert plan["phase_a_decision"] == "LONG_SCHEDULE_STILL_OPTIMIZING_BUT_TASK_FAIL"


def test_parity_control_holds_if_measured():
    parity = load_json("late_a1_parity.json")
    assert parity["DETERMINISTIC_MODE_VERIFIED"] is True
    assert parity["deterministic_control"]["max_abs_delta"] == 0.0
    assert parity["F0_SOURCE"] in ("phase_a_reuse", "fresh_rerun")


def test_result_is_internally_consistent():
    report = load_json("late_a1_adaptation.json")
    v, h = report["verdict"], report["histories"]
    f1 = h["F1_LATE_A1_TRAINABLE"]["25545"]["D2_LINE_DEV512"]
    assert v["ABSOLUTE_PASS"] == (f1["PASS"] and f1["SAFETY"])
    assert v["F1"]["angle_median"] == f1["angle_median"]
    assert report["a1_trained"]["trainable_params"] == 5014912
    assert v["CIGM"] == "BLOCKED"
