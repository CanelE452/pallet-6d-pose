"""Tests for the low-rank late-A1 adaptation screen.

Every original A1 parameter stays frozen, the branch is exactly zero at
initialisation and linear so it can be folded back into the kernel, the rank is
the fixed one, and the decision sits at 25,545 on the dev population against
Phase A's F0 with F1, F2 and R1 as context only.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/late_a1_low_rank_adaptation_screen.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("LOW_RANK_UNDER_TEST", RUNNER)
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


def test_the_rank_is_fixed_and_below_every_channel_dimension(screen):
    assert screen.RANK == 8
    backbone = screen.LowRankLateA1(screen.RANK)
    rows = backbone.audit()
    assert len(rows) == screen.EXPECTED_LATE_CONVS == 4
    assert min(min(r["in_channels"], r["out_channels"]) for r in rows) > screen.RANK
    del backbone


def test_the_late_convs_are_read_from_the_model(screen):
    backbone = screen.LowRankLateA1(None)
    assert backbone.late_indices == [19, 21, 23, 25]
    for i in backbone.late_indices:
        assert isinstance(backbone.vgg[i], torch.nn.Conv2d)
    assert isinstance(backbone.vgg[18], torch.nn.MaxPool2d)
    total = sum(sum(p.numel() for p in backbone.vgg[i].parameters())
                for i in backbone.late_indices)
    assert total == screen.EXPECTED_LATE_BASE_PARAMS == 5014912
    del backbone


def test_the_branch_is_linear_with_no_nonlinearity(screen):
    delta = screen.LowRankDelta(torch.nn.Conv2d(256, 128, 3, padding=1))
    modules = [m for m in delta.modules() if not isinstance(
        m, (screen.LowRankDelta,))]
    assert all(isinstance(m, torch.nn.Conv2d) for m in modules)
    assert not [m for m in delta.modules()
                if isinstance(m, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU))]
    assert delta.down.bias is None and delta.up.bias is None
    assert delta.down.kernel_size == (1, 1)
    assert delta.up.kernel_size == (3, 3)


def test_the_up_projection_is_exactly_zero_at_init(screen):
    delta = screen.LowRankDelta(torch.nn.Conv2d(256, 128, 3, padding=1))
    assert float(delta.up.weight.abs().max()) == 0.0
    assert float(delta.down.weight.abs().max()) > 0.0
    x = torch.randn(2, 256, 12, 12)
    assert float(delta(x).abs().max()) == 0.0


def test_no_extra_scalar_gate_was_added(screen):
    delta = screen.LowRankDelta(torch.nn.Conv2d(256, 128, 3, padding=1))
    scalars = [n for n, p in delta.named_parameters() if p.numel() == 1]
    assert scalars == []
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    for forbidden in ("alpha", "beta", "gate"):
        assert forbidden not in names, forbidden


def test_the_branch_folds_into_one_kernel(screen):
    """A 1x1 then a k x k with nothing between is a k x k."""
    base = torch.nn.Conv2d(64, 32, 3, padding=1)
    delta = screen.LowRankDelta(base, rank=4)
    with torch.no_grad():
        delta.up.weight.normal_(0.0, 0.1)
    x = torch.randn(2, 64, 11, 11)
    with torch.no_grad():
        unmerged = delta(x)
        merged = torch.nn.functional.conv2d(x, delta.effective_weight(),
                                            padding=delta.padding)
    assert float((unmerged - merged).abs().max()) <= screen.MERGE_TOLERANCE
    assert delta.effective_weight().shape == base.weight.shape


def test_every_original_a1_parameter_is_frozen(screen):
    backbone = screen.LowRankLateA1(screen.RANK)
    assert screen.trainable_a1_origin(backbone) == 0
    assert all(not p.requires_grad
               for p in backbone.inner.model.parameters())
    assert all(p.requires_grad for p in backbone.low_rank_parameters())
    del backbone


def test_low_rank_params_are_counted_separately(screen):
    backbone = screen.LowRankLateA1(screen.RANK)
    decoder = screen.DH.DirectHoughModel()
    audit = screen.parameter_audit(decoder, backbone)
    assert audit["a1_origin_trainable_params"] == 0
    assert audit["LOW_RANK_FEATURE_PARAMS"] > 0
    expected = sum(r["in_channels"] * screen.RANK
                   + screen.RANK * r["out_channels"] * r["kernel"][0] * r["kernel"][1]
                   for r in backbone.audit())
    assert audit["LOW_RANK_FEATURE_PARAMS"] == expected
    assert audit["f1_full_late_trainable"] == 5014912
    del backbone, decoder


def test_the_frozen_arm_carries_no_gradient(screen):
    backbone = screen.LowRankLateA1(None).to(screen.DEV)
    images = torch.zeros(2, 3, 400, 400, device=screen.DEV)
    assert backbone(images).requires_grad is False
    del backbone


def test_the_adapted_arm_carries_gradient(screen):
    backbone = screen.LowRankLateA1(screen.RANK).to(screen.DEV)
    images = torch.zeros(2, 3, 400, 400, device=screen.DEV)
    out = backbone(images)
    assert out.requires_grad is True
    del backbone


def test_the_optimizer_never_receives_base_parameters():
    for name in ("run_wiring", "run_memory", "train_arm"):
        calls = [n for n in ast.walk(function(name))
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "AdamW"]
        assert calls, name
    body = ast.get_source_segment(source(), function("trainable_groups"))
    assert "low_rank_parameters()" in body
    assert "inner" not in body and "vgg" not in body


def test_the_pair_shares_decoder_weights(screen):
    d0, frozen, d1, adapted = screen.build_pair()
    left, right = d0.state_dict(), d1.state_dict()
    assert all(torch.equal(left[k], right[k]) for k in left)
    body = ast.get_source_segment(source(), function("build_pair"))
    assert body.index("DirectHoughModel()") < body.index("LowRankLateA1")
    assert frozen.deltas is None and adapted.deltas is not None
    del d0, frozen, d1, adapted


def test_the_baseline_arm_uses_no_adapter_and_no_extra_block(screen):
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    for forbidden in ("F50LineAdapter", "RoleRefinementBlock",
                      "DeeperRoleModel", "adapter"):
        assert forbidden not in names, forbidden
    assert screen.ARMS == ("L0_FROZEN_A1", "L1_LOW_RANK_LATE_A1")


def test_no_scheduler_or_clipping_was_added():
    names = {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    for forbidden in ("clip_grad_norm_", "clip_grad_value_", "lr_scheduler",
                      "StepLR", "CosineAnnealingLR", "OneCycleLR"):
        assert forbidden not in names, forbidden


def test_the_schedule_and_gate_are_the_locked_ones(screen):
    assert screen.MARKS == (1703, 5000, 8515, 17030, 25545)
    assert screen.DECISION_STEP == 25545
    assert screen.CAP.BATCH == 8
    assert (screen.CAP.ANGLE_BUDGET_DEG, screen.CAP.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (screen.CAP.SAFETY_ANGLE, screen.CAP.SAFETY_OFFSET) == (2.0, 1.0)
    assert screen.REDUCTION == 0.40
    assert screen.STEP0_TOLERANCE == 1e-6
    assert screen.MERGE_TOLERANCE == 1e-5


def test_baselines_are_loaded_not_transcribed(screen):
    assert screen.PHASE_A_RESULT == "direct_hough_long.json"
    body = ast.get_source_segment(source(), function("baseline_f0"))
    assert "json.loads" in body
    constants = {n.value for n in ast.walk(tree())
                 if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    for hardcoded in (3.735687, 1.972937, 2.070244, 1.077348,
                      3.084991, 1.554134, 2.909729, 1.230865):
        assert hardcoded not in constants


def assigned_value(name):
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


def test_the_decision_reads_f0_and_never_the_context_arms():
    for decisive in ("ABSOLUTE_PASS", "REDUCTION_40"):
        used = strings_in(assigned_value(decisive))
        for forbidden in ("context_only", "F1_LATE_A1", "F2_ADAPTER",
                          "R1_ROLE_DEPTH", "D0_SEEN512"):
            assert forbidden not in used, (decisive, forbidden)
    body = ast.get_source_segment(source(), function("judge"))
    assert 'out["vs_F0"]["angle_median"] >= REDUCTION' in body
    assert 'out["vs_F1_context_only"]["angle_median"] >= REDUCTION' not in body


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
        for forbidden in ("SPECIALIZES", "generalization", "D0_SEEN512",
                          "PARETO_BETTER_THAN_FULL_UNFREEZE", "low_rank_use",
                          "LOW_RANK_CONV_MERGEABLE"):
            assert forbidden not in guard, forbidden


def test_insufficient_refutes_nothing():
    body = ast.get_source_segment(source(), function("judge"))
    assert "LATE_A1_LOW_RANK_INSUFFICIENT" in body
    assert '"LOW_RANK_ADAPTATION_REFUTED"] = False' in body
    assert '"FULL_UNFREEZE_PROVEN_REQUIRED"] = False' in body
    assert '"F1_SIGNAL"] = "unchanged"' in body
    assert '"SCOPE"' in body


def test_instability_forbids_a_retry():
    body = ast.get_source_segment(source(), function("judge"))
    assert "LOW_RANK_A1_UNSTABLE" in body
    assert '"RETRY_WITH_NEW_LR_OR_RANK"] = "FORBIDDEN"' in body


def test_forbidden_stages_are_absent():
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    for forbidden in ("solve_pose", "solvePnP", "cigm", "CIGM", "dimensions",
                      "position", "AbsoluteXY"):
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
    assert written == {"low_rank_a1_plan.json", "low_rank_a1_merge.json",
                       "low_rank_a1_step0.json", "low_rank_a1_wiring.json",
                       "low_rank_a1_memory.json", "low_rank_a1_parity.json",
                       "low_rank_a1.json"}


def test_plan_records_the_audit():
    plan = load_json("low_rank_a1_plan.json")
    assert plan["rank"] == 8 and plan["rank_sweep"] is False
    assert len(plan["late_conv_audit"]) == 4
    assert plan["late_base_params"] == 5014912
    assert plan["audit"]["a1_origin_trainable_params"] == 0
    assert plan["post_f50_adapter"] is False
    assert plan["extra_role_block"] is False
    assert plan["extra_scalar_gate"] is False
    assert plan["scheduler"] is None and plan["gradient_clipping"] is None


def test_merge_holds_if_measured():
    report = load_json("low_rank_a1_merge.json")
    assert report["LOW_RANK_CONV_MERGEABLE"] is True
    assert report["max_abs_delta"] <= report["tolerance"]
    assert len(report["per_conv"]) == 4


def test_step0_equivalence_if_measured():
    report = load_json("low_rank_a1_step0.json")
    assert report["LOW_RANK_A1_STEP0_EQUIVALENT"] is True
    for key in ("late_f50", "descriptor", "logits", "loss",
                "frozen_path_against_a1"):
        assert report["gaps"][key] <= report["tolerance"], key
    assert report["up_weight_zero"] is True


def test_gradient_wiring_if_measured():
    report = load_json("low_rank_a1_wiring.json")
    assert report["LOW_RANK_A1_GRADIENT_WIRING"] is True
    assert report["base_params_with_grad"] == 0
    assert report["a1_origin_trainable_params"] == 0
    for index, entry in report["step0"].items():
        assert entry["up_grad_norm"] > 0, index
    for index, entry in report["step2"].items():
        assert entry["down_grad_norm"] > 0, index
    for index, moved in report["up_moved_after_one_step"].items():
        assert moved > 0, index


def test_result_is_internally_consistent():
    report = load_json("low_rank_a1.json")
    v, h = report["verdict"], report["history"]
    assert sorted(map(int, h)) == [1703, 5000, 8515, 17030, 25545]
    l1 = h["25545"]["D2_LINE_DEV512"]
    assert v["ABSOLUTE_PASS"] == (l1["PASS"] and l1["SAFETY"])
    assert v["L1"]["angle_median"] == l1["angle_median"]
    assert v["CIGM"] == "BLOCKED"
    for mark in ("8515", "17030", "25545"):
        assert len(h[mark]["D2_LINE_DEV512"]["per_role"]) == 12
    for mark in h:
        assert "low_rank_use" in h[mark] and "generalization" in h[mark]
    assert set(v["context_only"]) == {"F1_LATE_A1", "F2_ADAPTER", "R1_ROLE_DEPTH"}
