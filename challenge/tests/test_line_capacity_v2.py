"""Tests for the V2 line-refinement capacity gate.

The V2 runner exists because V1's conclusion was produced by its own confounds.
Four of V2's own defects were then found after it had been committed: the
coverage audit sampled 256 frames of a 16,011-frame split, it scored ground
truth that never enters the image, the O1B oracle mask was true by construction,
and the loss averaged over roles the metric excluded.  Each of those is pinned
here, because a screen whose population moves is not a screen.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

RUNNER = ROOT / "scripts/stage0/line_feature_capacity_v2.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def v2():
    spec = importlib.util.spec_from_file_location("V2_UNDER_TEST", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def edges():
    import instance_edge_topology as IET
    return [tuple(e) for e in IET.build_topology()["edges"]]


def source():
    return RUNNER.read_text("utf-8")


def load_json(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text("utf-8"))


# --------------------------------------------------------------------------
# coverage population
# --------------------------------------------------------------------------

def test_coverage_runs_over_the_whole_split(v2):
    """256 frames of 16,011 is a sample, and it was reported as the split."""
    train, dev = v2.split_indices()
    assert (len(train), len(dev)) == (13618, 2393)
    report = load_json("coverage_fullsplit.json")
    assert report["dev"]["counts"]["frames"] == len(dev)
    assert report["train"]["counts"]["frames"] == len(train)


def test_no_truncated_index_list_reaches_the_audit():
    tree = ast.parse(source())
    main = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main)
             if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "coverage_full"]
    assert calls, "the audit no longer calls coverage_full"
    for call in calls:
        assert isinstance(call.args[0], ast.Name), ast.dump(call.args[0])
        assert call.args[0].id in {"full_train", "full_dev"}


def test_coverage_is_geometry_only_and_never_decodes_an_image(v2, monkeypatch, edges):
    import cv2
    monkeypatch.setattr(cv2, "imread", lambda *a, **k:
                        pytest.fail("coverage decoded a PNG"))
    _, dev = v2.split_indices()
    report = v2.coverage_full(dev[:8], "dev", (0,), edges)
    assert report["counts"]["frames"] == 8


def test_every_role_of_every_frame_is_accounted_for(v2, edges):
    _, dev = v2.split_indices()
    report = v2.coverage_full(dev[:16], "dev", (0,), edges)
    counts = report["counts"]
    assert counts["roles"] == 16 * len(edges)
    assert (counts["degenerate"] + counts["off_frame_full"]
            + counts["in_frame_partial"] + counts["in_frame_full"]) == counts["roles"]


# --------------------------------------------------------------------------
# segment clipping
# --------------------------------------------------------------------------

def test_clip_segment_keeps_a_fully_interior_segment(v2):
    p0 = np.array([[10.0, 10.0]]); p1 = np.array([[30.0, 40.0]])
    t_lo, t_hi, hit = v2.clip_segment(p0, p1)
    assert hit[0] and t_lo[0] == pytest.approx(0.0) and t_hi[0] == pytest.approx(1.0)


def test_clip_segment_trims_a_partial_segment(v2):
    p0 = np.array([[-20.0, 25.0]]); p1 = np.array([[20.0, 25.0]])
    t_lo, t_hi, hit = v2.clip_segment(p0, p1)
    assert hit[0]
    assert t_lo[0] == pytest.approx(0.5)          # enters at x = 0
    assert t_hi[0] == pytest.approx(1.0)
    q0 = p0 + (p1 - p0) * t_lo[:, None]
    assert q0[0] == pytest.approx([0.0, 25.0])


def test_clip_segment_rejects_a_fully_outside_segment(v2):
    p0 = np.array([[-40.0, 25.0]]); p1 = np.array([[-10.0, 25.0]])
    assert not v2.clip_segment(p0, p1)[2][0]


def test_clip_segment_rejects_a_segment_parallel_and_beyond_a_border(v2):
    p0 = np.array([[10.0, 80.0]]); p1 = np.array([[40.0, 80.0]])
    assert not v2.clip_segment(p0, p1)[2][0]


def test_visible_segments_classifies_the_three_cases(v2):
    p0 = np.array([[10.0, 10.0], [-20.0, 25.0], [-40.0, 25.0], [5.0, 5.0]])
    p1 = np.array([[30.0, 40.0], [20.0, 25.0], [-10.0, 25.0], [5.0, 5.0]])
    length = np.array([36.0, 40.0, 30.0, 0.0])
    seg = v2.visible_segments(p0, p1, length)
    assert list(seg["in_frame_full"]) == [True, False, False, False]
    assert list(seg["in_frame_partial"]) == [False, True, False, False]
    assert list(seg["off_frame_full"]) == [False, False, True, False]
    assert list(seg["degenerate"]) == [False, False, False, True]
    assert list(seg["hit"]) == [True, True, False, False]


def test_off_frame_roles_are_excluded_from_the_scored_population(v2, edges):
    """An edge that never enters the image has no local evidence to read; it is
    counted, not scored, and the split really does contain such roles."""
    _, dev = v2.split_indices()
    report = v2.coverage_full(dev[:64], "dev", (0,), edges)
    counts = report["counts"]
    assert counts["off_frame_full"] > 0
    assert counts["pairs"] == counts["in_frame_full"] + counts["in_frame_partial"]


# --------------------------------------------------------------------------
# the O1B oracle
# --------------------------------------------------------------------------

def test_segment_support_mask_is_not_identically_one(v2):
    """The first implementation built the sample positions from the GT interval
    and then tested membership in that interval, so O1B was O1A."""
    feature = torch.zeros(1, 1, 50, 50)
    normal = torch.tensor([[[1.0, 0.0]]])
    rho = torch.tensor([[25.0]])
    sample = v2.sample_strip(feature, normal, rho, 1.0)
    q0 = np.array([[[25.0, 20.0]]]); q1 = np.array([[[25.0, 30.0]]])
    support = v2.segment_support_mask(q0, q1, sample, 1.0)
    fraction = float(support.mean())
    assert 0.0 < fraction < 1.0
    assert fraction == pytest.approx(11 / 50, abs=0.05)


def test_segment_support_mask_marks_the_right_portion(v2):
    feature = torch.zeros(1, 1, 50, 50)
    normal = torch.tensor([[[1.0, 0.0]]])
    rho = torch.tensor([[25.0]])
    sample = v2.sample_strip(feature, normal, rho, 1.0)
    q0 = np.array([[[25.0, 0.0]]]); q1 = np.array([[[25.0, 49.0]]])
    assert float(v2.segment_support_mask(q0, q1, sample, 1.0).mean()) == pytest.approx(1.0)
    q0 = np.array([[[25.0, 100.0]]]); q1 = np.array([[[25.0, 120.0]]])
    assert float(v2.segment_support_mask(q0, q1, sample, 1.0).mean()) == 0.0


def test_o1b_feature_is_the_same_image_evidence_as_o1a(v2):
    """GT restricts which longitudinal portion counts.  It never supplies a
    feature value, so the two arms read byte-identical evidence."""
    assert v2.ARMS["O1A"][0] == "gradient"
    assert v2.ARMS["O1B"][0] == "gradient_segment"
    assert v2.ARMS["O1A"][1:] == v2.ARMS["O1B"][1:]
    tree = ast.parse(source())
    build = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "build_feature")
    branch = next(node for node in ast.walk(build) if isinstance(node, ast.If))
    assert {c.value for c in ast.walk(branch.test) if isinstance(c, ast.Constant)} == {
        "gradient", "gradient_segment"}
    body = ast.get_source_segment(source(), branch.body[0])
    assert "scharr_evidence" in body and "gt" not in body.lower()


def test_o1b_support_is_partial_on_real_frames(v2, edges):
    """A synthetic case can be arranged; the split has to show it too."""
    result = load_json("line_capacity_v2_arms.json")
    if "O1B" not in result:
        pytest.skip("O1B not run yet")
    fraction = result["O1B"][str(v2.EPOCH_LADDER[-1])].get("support_fraction")
    assert fraction is not None and 0.0 < fraction < 1.0


# --------------------------------------------------------------------------
# loss and population
# --------------------------------------------------------------------------

def test_masked_mean_ignores_unsupported_roles(v2):
    values = torch.tensor([[1.0, 1000.0]])
    mask = torch.tensor([[True, False]])
    assert float(v2.masked_mean(values, mask)) == pytest.approx(1.0)


def test_budget_losses_can_return_per_role_terms(v2):
    theta = torch.zeros(2, 3); rho = torch.zeros(2, 3)
    losses = v2.budget_losses(theta + 0.01, rho + 0.5, theta, rho, reduce=False)
    assert losses["theta_per_role"].shape == (2, 3)
    assert "L_theta" not in losses


def test_training_and_evaluation_share_one_masked_path(v2):
    """The reduced loss used to be built before the mask, so train averaged over
    roles the metric had thrown away.  One code path is the fix."""
    tree = ast.parse(source())
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "step_batch")
    assert [a.arg for a in step.args.args].count("train") == 0
    body = ast.get_source_segment(source(), step)
    assert "masked_mean(losses[\"theta_per_role\"], mask)" in body
    assert "L_theta" not in body


def test_every_arm_scores_the_identical_population(v2):
    results = load_json("line_capacity_v2_arms.json")
    last = str(v2.EPOCH_LADDER[-1])
    shas = {name: entry[last]["population_sha"]
            for name, entry in results.items() if last in entry}
    assert len(shas) >= 2 and len(set(shas.values())) == 1, shas


def test_coarse_validity_is_resolution_invariant(v2):
    """F100 reads a 100x100 map and F50 a 50x50 one; the population must not
    depend on which."""
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "step_batch"))
    assert "line_rect_intersection(normal_c, rho_c, GRID, GRID)[2]" in body


def test_the_strip_footprint_is_the_same_physical_size_for_every_arm(v2):
    feature50 = torch.zeros(1, 1, 50, 50)
    feature100 = torch.zeros(1, 1, 100, 100)
    normal = torch.tensor([[[1.0, 0.0]]]); rho = torch.tensor([[25.0]])
    a = v2.sample_strip(feature50, normal, rho, 1.0)
    b = v2.sample_strip(feature100, normal, rho, 2.0)
    # transverse extent in canonical cells
    span_a = (a["t"].max() - a["t"].min()) / 1.0
    span_b = (b["t"].max() - b["t"].min()) / 2.0
    assert float(span_a) == pytest.approx(float(span_b), rel=0.02)


def test_validity_mask_is_produced_and_fed_to_the_refiner(v2):
    feature = torch.zeros(1, 1, 50, 50)
    normal = torch.tensor([[[1.0, 0.0]]]); rho = torch.tensor([[25.0]])
    sample = v2.sample_strip(feature, normal, rho, 1.0)
    assert sample["inside"].shape == (1, 1, v2.TRANSVERSE_SAMPLES, v2.LONGITUDINAL)
    assert 0.0 < float(sample["inside"].mean()) <= 1.0


# --------------------------------------------------------------------------
# checkpoints, seed, guards
# --------------------------------------------------------------------------

def test_checkpoints_exist_for_every_reported_epoch(v2):
    results = load_json("line_capacity_v2_arms.json")
    for name in results:
        for epoch in v2.EPOCH_LADDER:
            assert v2.checkpoint_path(name, epoch).exists(), (name, epoch)


def test_checkpoints_carry_their_provenance(v2):
    results = load_json("line_capacity_v2_arms.json")
    name = next(iter(results))
    state = torch.load(v2.checkpoint_path(name, v2.EPOCH_LADDER[-1]),
                       map_location="cpu", weights_only=False)
    for key in ("arm", "epoch", "model", "optimizer", "seed", "runner_sha",
                "split_sha", "radius_cell", "jitter", "gate"):
        assert key in state, key


def test_the_epoch5_decision_survives_a_reload(v2):
    results = load_json("line_capacity_v2_arms.json")
    for name, entry in results.items():
        assert entry["reload_parity"]["match"], name
        assert entry["reload_parity"]["max_delta"] == 0.0, name


def test_the_decision_reads_epoch_five_only(v2):
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "decide"))
    assert "EPOCH_LADDER[-1]" in body
    assert "max(" not in body and "min(" not in body and "sorted(" not in body


def test_there_is_no_seed_option_that_does_nothing():
    tree = ast.parse(source())
    added = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "add_argument"]
    names = {c.value for call in added for c in call.args if isinstance(c, ast.Constant)}
    assert "--seed" not in names
    assert "SEED, LR, WD, BATCH = 1," in source()


def test_sealed_tokens_are_blocked(v2):
    for token in ("capturenight08", "capturepallet09", "testset_full8_manifest",
                  "handannot17", "wood_pallet_20260618"):
        with pytest.raises(RuntimeError, match="BLOCKED"):
            v2.guard(pathlib.Path(f"/tmp/{token}/x.json"))


def test_the_gate_touches_no_pose_no_dimensions_and_no_held_out_set():
    """Named in code, not in prose: the docstring says 'no validation512', which
    a substring search would read as a use of it."""
    tree = ast.parse(source())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for forbidden in ("solve_pose", "solvePnP", "solve_pnp", "dims", "dimensions"):
        assert forbidden not in names, forbidden
    loaded = {c.value for call in ast.walk(tree) if isinstance(call, ast.Call)
              and getattr(call.func, "id", "") in {"manifest", "load_frame",
                                                   "load_geometry"}
              for c in call.args if isinstance(c, ast.Constant)}
    assert "validation512" not in loaded
    assert loaded <= {"line_search2k", "line_dev512", "line_smoke512"}, loaded


def test_the_rgb_stem_parameters_are_optimised(v2):
    refiner, stem, parameters = v2.build_arm("C3_RGB_STEM")
    assert stem is not None
    ids = {id(p) for p in parameters}
    assert all(id(p) in ids for p in stem.parameters())
    plain = v2.build_arm("C0_F50")
    assert plain[1] is None


def test_the_configured_radius_is_the_smallest_that_clears_the_gate(v2):
    report = load_json("coverage_fullsplit.json")
    assert report["chosen_radius"] == v2.TRANSVERSE_RADIUS_CELL
    for radius in v2.RADIUS_CANDIDATES:
        if radius >= report["chosen_radius"]:
            continue
        assert min(report[s]["radii"][f"{radius:g}"]["pair_coverage"]
                   for s in ("dev", "train")) < v2.COVERAGE_GATE
