"""Tests for the role-encoder depth screen.

Exactly one extra block, gated to zero so step 0 is the F2 function; no new role
embeddings and therefore no assignment; A1 frozen and the adapter untouched; the
decision at 25,545 on the dev population against F2, with F1 as context only.
"""
from __future__ import annotations

import ast, importlib.util, json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/line/role_encoder_depth_screen.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("ROLE_DEPTH_UNDER_TEST", RUNNER)
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


def strings_in(node):
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def load_json(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text("utf-8"))


def test_exactly_one_extra_block(screen):
    assert screen.ARMS == ("R0_F2_ADAPTER", "R1_EXTRA_ROLE_BLOCK")
    blocks = [n for n in ast.walk(tree())
              if isinstance(n, ast.Call)
              and getattr(n.func, "id", "") == "RoleRefinementBlock"]
    for node in ast.walk(function("build_pair")):
        pass
    built = [n for n in ast.walk(function("build_pair"))
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "RoleRefinementBlock"]
    assert len(built) == 1
    assert blocks, "the block is never constructed"


def test_the_block_shape_is_the_locked_one(screen):
    assert (screen.QUERY_DIM, screen.QUERY_HEADS, screen.ROLES) == (64, 4, 12)
    assert screen.FFN_HIDDEN == 128
    block = screen.RoleRefinementBlock()
    assert block.attention.embed_dim == 64
    assert block.attention.num_heads == 4
    linears = [m for m in block.ffn if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(64, 128), (128, 64)]
    assert isinstance(block.norm_query, torch.nn.LayerNorm)
    assert isinstance(block.norm_ffn, torch.nn.LayerNorm)


def test_beta_is_zero_at_init_and_learnable(screen):
    block = screen.RoleRefinementBlock()
    assert float(block.beta) == 0.0
    assert block.beta.requires_grad is True
    assert block.beta.numel() == 1


def test_the_block_is_identity_at_init(screen):
    block = screen.RoleRefinementBlock()
    descriptor = torch.randn(2, 12, 64)
    tokens = torch.randn(2, 2500, 64)
    assert torch.equal(block(descriptor, tokens), descriptor)


def test_the_residual_is_gated_by_beta():
    body = ast.get_source_segment(source(), function("forward"))
    assert "descriptor + self.beta * change" in body


def test_no_new_role_embeddings_and_no_assignment(screen):
    block = screen.RoleRefinementBlock()
    assert not [m for m in block.modules() if isinstance(m, torch.nn.Embedding)]
    names = {n.id for n in ast.walk(tree()) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree()) if isinstance(n, ast.Attribute)}
    for forbidden in ("Embedding", "linear_sum_assignment", "hungarian",
                      "assignment", "queries"):
        assert forbidden not in names, forbidden


def test_role_channels_are_preserved(screen):
    """Channel k in must be channel k out, so a role cannot be relabelled."""
    block = screen.RoleRefinementBlock()
    with torch.no_grad():
        block.beta.fill_(1.0)
    descriptor = torch.randn(1, 12, 64)
    tokens = torch.randn(1, 2500, 64)
    out = block(descriptor, tokens)
    assert out.shape == descriptor.shape
    permuted = block(descriptor[:, [1, 0] + list(range(2, 12))], tokens)
    assert torch.allclose(permuted[:, 0], out[:, 1], atol=1e-5)


def test_a1_is_frozen_and_the_adapter_comes_from_the_locked_screen(screen):
    a1 = screen.ADAPTER.frozen_a1()
    assert screen.ADAPTER.trainable_a1_params(a1) == 0
    del a1
    defined = {n.name for n in ast.walk(tree())
               if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for forbidden in ("F50LineAdapter", "DirectHoughModel", "DirectHoughHead",
                      "RoleQueryGlobal", "lattice", "cross_entropy",
                      "target_distribution", "frozen_a1"):
        assert forbidden not in defined, forbidden


def test_the_optimizer_never_receives_a1_parameters():
    for name in ("run_wiring", "run_memory", "train_arm"):
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


def test_the_pair_shares_base_weights(screen):
    r0, adapter_r0, r1, adapter_r1, block = screen.build_pair()
    left, right = r0.base.state_dict(), r1.base.state_dict()
    assert all(torch.equal(left[k], right[k]) for k in left)
    la, ra = adapter_r0.state_dict(), adapter_r1.state_dict()
    assert all(torch.equal(la[k], ra[k]) for k in la)
    body = ast.get_source_segment(source(), function("build_pair"))
    assert body.index("F50LineAdapter()") < body.index("RoleRefinementBlock()")
    assert r0.block is None and r1.block is not None
    del r0, r1, adapter_r0, adapter_r1, block


def test_step0_demands_exactness_where_it_is_available(screen):
    assert screen.STEP0_EXACT == 0.0
    assert screen.STEP0_TOLERANCE == 1e-6
    body = ast.get_source_segment(source(), function("run_step0"))
    assert 'gaps["adapted_f50"] == STEP0_EXACT' in body
    assert 'gaps["first_descriptor"] == STEP0_EXACT' in body
    for approximate in ("final_descriptor", "logits", "loss"):
        assert f'gaps["{approximate}"] <= STEP0_TOLERANCE' in body


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


def test_baselines_are_loaded_not_transcribed(screen):
    body = ast.get_source_segment(source(), function("baselines"))
    assert "json.loads" in body
    assert screen.ADAPTER_RESULT == "f50_adapter.json"
    assert screen.LATE_A1_RESULT == "late_a1_adaptation.json"
    constants = {n.value for n in ast.walk(tree())
                 if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    for hardcoded in (3.084991, 1.554134, 17.584229, 6.242823,
                      2.070244, 1.077348, 3.735687, 1.972937):
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


def test_the_decision_reads_r0_and_never_f1():
    for decisive in ("ABSOLUTE_PASS", "REDUCTION_40"):
        used = strings_in(assigned_value(decisive))
        assert "F1_context_only" not in used, decisive
        assert "D0_SEEN512" not in used, decisive
    reduction = ast.get_source_segment(source(), function("judge"))
    assert 'out["vs_R0"]["angle_median"] >= REDUCTION' in reduction
    assert 'out["vs_F1_context_only"]["angle_median"] >= REDUCTION' not in reduction


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
                          "PARETO_BETTER_THAN_LATE_UNFREEZE", "block_use"):
            assert forbidden not in guard, forbidden


def test_insufficient_does_not_exonerate_capacity():
    body = ast.get_source_segment(source(), function("judge"))
    assert "ROLE_ENCODER_DEPTH_INSUFFICIENT" in body
    assert '"ROLE_ENCODER_CAPACITY_EXONERATED"] = False' in body
    assert '"SCOPE"' in body
    addendum = (ROOT / "_docs/audits/eval56_summary/canonical_corner_audit"
                / "edge_mandatory_fast_search/F50_ADAPTER_SCOPE_ADDENDUM.md"
                ).read_text("utf-8")
    assert "role-encoder capacity is the bottleneck" in addendum


def test_instability_forbids_a_retry():
    body = ast.get_source_segment(source(), function("judge"))
    assert "ROLE_ENCODER_DEPTH_UNSTABLE" in body
    assert '"RETRY_WITH_NEW_LR"] = "FORBIDDEN"' in body


def non_docstring_literals():
    """Every string constant except the docstrings.

    Docstrings are prose: a sentence saying a module is left untouched is not a
    reference to the sealed `untouched` set, and a scan that cannot tell them
    apart fails on well-written documentation.
    """
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
    assert written == {"role_depth_plan.json", "role_depth_step0.json",
                       "role_depth_wiring.json", "role_depth_memory.json",
                       "role_depth.json"}


def test_plan_records_the_growth():
    plan = load_json("role_depth_plan.json")
    audit = plan["audit"]
    assert audit["a1_trainable_params"] == 0
    assert audit["new_refinement_block_params"] > 0
    assert (audit["trainable_total_R1"]
            == audit["trainable_total_R0"] + audit["new_refinement_block_params"])
    assert plan["new_role_embeddings"] == 0
    assert plan["role_assignment"] is None
    assert plan["lr_sweep"] is False and plan["block_count_sweep"] is False
    assert plan["scheduler"] is None and plan["gradient_clipping"] is None


def test_step0_equivalence_if_measured():
    report = load_json("role_depth_step0.json")
    assert report["ROLE_DEPTH_STEP0_EQUIVALENT"] is True
    assert report["gaps"]["adapted_f50"] == 0.0
    assert report["gaps"]["first_descriptor"] == 0.0
    for key in ("final_descriptor", "logits", "loss"):
        assert report["gaps"][key] <= report["tolerance"]
    assert report["beta_at_init"] == 0.0


def test_gradient_wiring_if_measured():
    report = load_json("role_depth_wiring.json")
    assert report["ROLE_DEPTH_GRADIENT_WIRING"] is True
    assert report["beta_grad_at_step0"] > 0
    for key in ("new_attention_grad_norm", "new_ffn_grad_norm",
                "adapter_grad_norm", "first_role_block_grad_norm",
                "head_grad_norm"):
        assert report[key] > 0, key
    assert report["a1_params_with_grad"] == 0


def test_result_is_internally_consistent():
    report = load_json("role_depth.json")
    v, h = report["verdict"], report["history"]
    assert sorted(map(int, h)) == [1703, 5000, 8515, 17030, 25545]
    r1 = h["25545"]["D2_LINE_DEV512"]
    assert v["ABSOLUTE_PASS"] == (r1["PASS"] and r1["SAFETY"])
    assert v["R1"]["angle_median"] == r1["angle_median"]
    assert v["CIGM"] == "BLOCKED"
    for mark in ("8515", "17030", "25545"):
        assert len(h[mark]["D2_LINE_DEV512"]["per_role"]) == 12
    for mark in h:
        assert "block_use" in h[mark] and "generalization" in h[mark]
