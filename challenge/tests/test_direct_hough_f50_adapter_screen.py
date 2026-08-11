"""Tests for the constrained F50 line-adapter screen.

A1 must stay fully frozen, the adapter must be identity at initialisation and
actually receive gradient afterwards, the shape must be the fixed one, and the
decision must sit at 25,545 on the dev population against Phase A's F0 -- with
Phase B's F1 present only as context.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/direct_hough_f50_adapter_screen.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("F50_ADAPTER_UNDER_TEST", RUNNER)
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


def test_the_adapter_shape_is_the_fixed_one(screen):
    assert screen.BOTTLENECK == 32
    assert screen.F50_CHANNELS == 128
    adapter = screen.F50LineAdapter()
    shapes = [(m.in_channels, m.out_channels, m.kernel_size)
              for m in adapter.body if isinstance(m, torch.nn.Conv2d)]
    assert shapes == [(128, 32, (1, 1)), (32, 32, (3, 3)), (32, 128, (1, 1))]
    assert adapter.body[2].padding == (1, 1)
    assert sum(1 for m in adapter.body if isinstance(m, torch.nn.ReLU)) == 2


def test_the_parameter_count_is_computed_not_asserted(screen):
    """The expected count is derived here from the shapes, not transcribed."""
    adapter = screen.F50LineAdapter()
    expected = ((128 * 32 + 32) + (32 * 32 * 9 + 32) + (32 * 128 + 128) + 1)
    assert adapter.report()["params"] == expected
    assert adapter.report()["alpha_params"] == 1
    constants = {n.value for n in ast.walk(tree())
                 if isinstance(n, ast.Constant) and isinstance(n.value, int)}
    assert expected not in constants


def test_alpha_is_zero_at_init_and_learnable(screen):
    adapter = screen.F50LineAdapter()
    assert float(adapter.alpha) == 0.0
    assert adapter.alpha.requires_grad is True
    assert adapter.alpha.numel() == 1


def test_the_adapter_is_identity_at_init(screen):
    adapter = screen.F50LineAdapter()
    f50 = torch.randn(2, 128, 50, 50)
    assert torch.equal(adapter(f50), f50)


def test_the_residual_is_gated_by_alpha():
    body = ast.get_source_segment(source(), function("forward"))
    assert "f50 + self.alpha * self.body(f50)" in body


def test_a1_is_fully_frozen(screen):
    a1 = screen.frozen_a1()
    assert screen.trainable_a1_params(a1) == 0
    assert all(not p.requires_grad for p in a1.parameters())
    del a1


def test_the_optimizer_never_receives_a1_parameters():
    """`a1.parameters()` also appears where A1 grads are *checked*, so the
    question is what AdamW is handed, not what the function mentions."""
    for name in ("run_wiring", "run_memory", "train_adapter"):
        calls = [n for n in ast.walk(function(name))
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "AdamW"]
        assert calls, name
        for call in calls:
            sources = set()
            for node in ast.walk(call):
                if isinstance(node, ast.Call) and getattr(
                        node.func, "attr", "") == "parameters":
                    owner = node.func.value
                    sources.add(getattr(owner, "id", None)
                                or getattr(owner, "attr", None))
            assert "a1" not in sources, (name, sources)
            assert sources <= {"model", "adapter"}, (name, sources)


def test_the_pair_shares_base_weights_without_rng_interference(screen):
    """The adapter is built after both base models, so it cannot shift them."""
    baseline, candidate, adapter = screen.build_pair()
    left, right = baseline.state_dict(), candidate.state_dict()
    assert sorted(left) == sorted(right)
    assert all(torch.equal(left[k], right[k]) for k in left)
    body = ast.get_source_segment(source(), function("build_pair"))
    assert body.index("DirectHoughModel()") < body.index("F50LineAdapter()")
    del baseline, candidate, adapter


def test_step0_tolerance_and_probe_are_locked(screen):
    assert screen.STEP0_TOLERANCE == 1e-6
    assert screen.PROBE_FRAMES == 32


def test_no_scheduler_or_clipping_was_added():
    names = {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    for forbidden in ("clip_grad_norm_", "clip_grad_value_", "lr_scheduler",
                      "StepLR", "CosineAnnealingLR", "OneCycleLR"):
        assert forbidden not in names, forbidden


def test_everything_downstream_comes_from_the_locked_runner():
    used = set()
    for node in ast.walk(function("train_adapter")):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Attribute):
                used.add(target.attr)
            elif isinstance(target, ast.Name):
                used.add(target.id)
    for piece in ("lattice", "hypothesis_features", "step_schedule", "load_pack",
                  "batch_rows", "target_distribution", "cross_entropy", "AdamW"):
        assert piece in used, piece
    defined = {n.name for n in ast.walk(tree())
               if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for forbidden in ("DirectHoughHead", "DirectHoughModel", "HypothesisEncoder",
                      "lattice", "target_distribution", "cross_entropy",
                      "hypothesis_features", "decode", "measure", "summarise",
                      "RoleQueryGlobal"):
        assert forbidden not in defined, forbidden


def test_the_dead_position_branch_is_untouched():
    names = {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    assert "position" not in names
    assert "AbsoluteXY" not in names


def test_the_schedule_and_gate_are_the_locked_ones(screen):
    assert screen.MARKS == (1703, 5000, 8515, 17030, 25545)
    assert screen.DECISION_STEP == 25545
    assert screen.CAP.BATCH == 8
    assert (screen.CAP.ANGLE_BUDGET_DEG, screen.CAP.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (screen.CAP.SAFETY_ANGLE, screen.CAP.SAFETY_OFFSET) == (2.0, 1.0)
    assert screen.REDUCTION == 0.40


def test_baselines_are_loaded_not_transcribed(screen):
    body = ast.get_source_segment(source(), function("baselines"))
    assert "json.loads" in body
    assert screen.PHASE_A_RESULT == "direct_hough_long.json"
    assert screen.PHASE_B_RESULT == "late_a1_adaptation.json"
    constants = {n.value for n in ast.walk(tree())
                 if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    for hardcoded in (3.735687, 1.972937, 27.843582, 8.279793,
                      2.070244, 1.077348, 9.702057, 4.125696):
        assert hardcoded not in constants


def test_f1_is_context_only_and_selects_nothing():
    body = ast.get_source_segment(source(), function("judge"))
    for decisive in ("REDUCTION_40", "ABSOLUTE_PASS", "SPECIALIZES"):
        assert decisive in body
    # every F1 use is labelled as context, and the reduction test reads F0
    assert 'out["vs_F0"]["angle_median"] >= REDUCTION' in body
    assert "F1_context_only" in body
    assert 'out["vs_F1_context_only"]["angle_median"] >= REDUCTION' not in body


def strings_in(node):
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def assigned_value(name):
    """The expression bound to `name` in judge(), however it is written.

    `ABSOLUTE_PASS` lives inside the `out = {...}` literal while `REDUCTION_40`
    is a later subscript assignment, so both forms have to be found.
    """
    judge = function("judge")
    for node in ast.walk(judge):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == name:
                    return value
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == name):
                    return node.value
    raise AssertionError(f"{name!r} is never bound in judge()")


def test_the_decision_uses_the_dev_population_only():
    """D0 appears throughout judge() as a diagnostic; what matters is that the
    two decisive quantities are computed without it."""
    for decisive in ("ABSOLUTE_PASS", "REDUCTION_40"):
        assert "D0_SEEN512" not in strings_in(assigned_value(decisive)), decisive
    assert "D2_LINE_DEV512" in strings_in(function("judge"))


def test_generalization_ratios_are_diagnostic_not_gates():
    """No DECISION branch may be conditioned on the ratios or on SPECIALIZES."""
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
        for forbidden in ("SPECIALIZES", "generalization", "D0_SEEN512",
                          "F50_ADAPTER_PARETO_BETTER_THAN_LATE_UNFREEZE"):
            assert forbidden not in guard, forbidden
    assert "generalization" in strings_in(function("train_adapter"))


def test_pareto_is_recorded_but_cannot_pass_the_task(screen):
    body = ast.get_source_segment(source(), function("judge"))
    assert "F50_ADAPTER_PARETO_BETTER_THAN_LATE_UNFREEZE" in body
    pareto = body.index("F50_ADAPTER_PARETO_BETTER_THAN_LATE_UNFREEZE")
    decision = body.index('out["DECISION"] = "F50_ADAPTER_VALID_CANDIDATE"')
    assert pareto < decision
    assert 'PARETO_BETTER_THAN_LATE_UNFREEZE"]:' not in body


def test_instability_forbids_a_retry():
    body = ast.get_source_segment(source(), function("judge"))
    assert "F50_ADAPTER_TRAINING_UNSTABLE" in body
    assert '"RETRY_WITH_NEW_LR"] = "FORBIDDEN"' in body


def test_role_encoder_depth_is_not_implemented_here():
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    defined = {n.name for n in ast.walk(tree())
               if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for forbidden in ("extra_attention", "ResidualCrossAttention", "depth_screen"):
        assert forbidden not in names and forbidden not in defined
    # The companion assertion that scripts/stage0/role_encoder_depth_screen.py
    # did not exist was scoped to the adapter commit, where the point was that
    # feature adaptation and encoder depth must not move in the same run. That
    # screen is now pre-registered separately, so the check that survives is the
    # permanent one: this runner does not implement it.


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


def test_the_runner_never_writes_earlier_results():
    written = set()
    for node in ast.walk(tree()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_text"):
            target = node.func.value
            if isinstance(target, ast.BinOp) and isinstance(target.right, ast.Constant):
                written.add(target.right.value)
    assert written == {"f50_adapter_plan.json", "f50_adapter_step0.json",
                       "f50_adapter_wiring.json", "f50_adapter_memory.json",
                       "f50_adapter.json"}


def test_plan_records_a_frozen_a1():
    plan = load_json("f50_adapter_plan.json")
    assert plan["audit"]["a1_trainable_params"] == 0
    assert plan["audit"]["adapter"]["bottleneck"] == 32
    assert plan["lr_sweep"] is False and plan["bottleneck_sweep"] is False
    assert plan["scheduler"] is None and plan["gradient_clipping"] is None


def test_step0_equivalence_if_measured():
    report = load_json("f50_adapter_step0.json")
    assert report["F50_ADAPTER_STEP0_EQUIVALENT"] is True
    assert report["logit_max_abs"] <= report["tolerance"]
    assert report["descriptor_max_abs"] <= report["tolerance"]
    assert report["alpha_at_init"] == 0.0


def test_gradient_wiring_if_measured():
    report = load_json("f50_adapter_wiring.json")
    assert report["F50_ADAPTER_GRADIENT_WIRING"] is True
    assert report["alpha_grad_at_step0"] > 0
    assert report["first_conv_grad_norm"] > 0
    assert report["last_conv_grad_norm"] > 0
    assert report["a1_params_with_grad"] == 0


def test_result_is_internally_consistent():
    report = load_json("f50_adapter.json")
    v, h = report["verdict"], report["history"]
    assert sorted(map(int, h)) == [1703, 5000, 8515, 17030, 25545]
    f2 = h["25545"]["D2_LINE_DEV512"]
    assert v["ABSOLUTE_PASS"] == (f2["PASS"] and f2["SAFETY"])
    assert v["F2"]["angle_median"] == f2["angle_median"]
    assert v["CIGM"] == "BLOCKED"
    for mark in ("8515", "17030", "25545"):
        assert len(h[mark]["D2_LINE_DEV512"]["per_role"]) == 12
    for mark in h:
        assert "adapter_use" in h[mark] and "generalization" in h[mark]
