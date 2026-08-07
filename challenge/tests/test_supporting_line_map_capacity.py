"""Tests for the supporting-line map capacity screen -- the first training run.

The decoder is settled, so what has to be pinned here is that the training
signal is honest: map-only loss, no gradient through a non-differentiable
argmax, unsupported roles masked rather than supervised as empty, and no
segment extent anywhere in the target.
"""
from __future__ import annotations

import ast, importlib.util, json, math, pathlib, sys

import numpy as np, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
RUNNER = ROOT / "scripts/stage0/supporting_line_map_capacity.py"
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def cap():
    spec = importlib.util.spec_from_file_location("CAP_UNDER_TEST", RUNNER)
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


def test_the_loss_never_touches_the_hough_argmax():
    """decode_maps is the only path to a line and it is under no_grad, so the
    non-differentiable argmax cannot be in any graph."""
    tree = ast.parse(source())
    decode = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "decode_maps")
    decorators = {ast.dump(d) for d in decode.decorator_list}
    assert any("no_grad" in d for d in decorators)
    loss = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "map_loss")
    used = {n.id for n in ast.walk(loss) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(loss) if isinstance(n, ast.Attribute)}
    for forbidden in ("decode", "decode_maps", "argmax", "theta", "rho", "measure"):
        assert forbidden not in used, forbidden


def test_only_the_map_loss_is_optimised():
    tree = ast.parse(source())
    called = {getattr(n.func, "id", "") or getattr(n.func, "attr", "")
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "map_loss" in called
    for forbidden in ("budget_losses", "smooth_l1_loss",
                      "binary_cross_entropy_with_logits", "orientation_loss"):
        assert forbidden not in called, forbidden


def test_an_unsupported_role_is_masked_not_taught_to_be_empty(cap):
    """The test is about gradient, not loss value: a balanced mean is unchanged
    by extra negatives that already have the same error, so comparing two loss
    numbers proves nothing.  What matters is that an unsupervised channel is
    pushed nowhere."""
    target = torch.zeros(1, 2, cap.MAP, cap.MAP, device=cap.DEV)
    target[0, 0, 50, :] = 1.0
    logit = torch.full_like(target, 4.0).requires_grad_(True)
    cap.map_loss(logit, target,
                 torch.tensor([[True, False]], device=cap.DEV)).backward()
    assert float(logit.grad[0, 1].abs().max()) == 0.0     # unsupported: untouched
    assert float(logit.grad[0, 0].abs().max()) > 0.0      # supported: supervised
    none = cap.map_loss(logit.detach(), target,
                        torch.tensor([[False, False]], device=cap.DEV))
    assert float(none) == 0.0


def test_there_is_no_support_head(cap):
    head = cap.SupportingLineHead(128)
    produced = {name for name, _ in head.named_children()}
    assert produced == {"body", "to_map"}
    assert "to_support" not in source()
    assert head.to_map.out_channels == 12


def test_segment_extent_never_reaches_the_target():
    tree = ast.parse(source())
    geometry = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "geometry")
    body = ast.get_source_segment(source(), geometry)
    assert "SEM.raster_supporting_line(theta, rho, seg[\"hit\"])" in body
    assert "raster_targets" not in body           # the finite-segment tube


def test_the_population_is_the_locked_one(cap):
    assert cap.LINE_DEV_POPULATION_SHA == "00c605b9116e214b"
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert "POPULATION_CHANGED" in body and "27684" in body


def test_the_decoder_is_the_locked_hough_path(cap):
    assert cap.PRIMARY == cap.H.PRIMARY == "H2_ZERO_MEAN_NCC"
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "decode_maps"))
    assert "H.decode(maps, coarse, xx, yy)[PRIMARY]" in body
    assert "torch.sigmoid" in source() and "softplus" not in source()
    for forbidden in ("weighted_tls", "centroid", "covariance"):
        assert forbidden not in source(), forbidden


def test_all_thresholds_are_declared_before_the_run(cap):
    assert cap.OLOSS_GATE == {"angle_median": 0.05, "offset_median": 0.05,
                              "angle_p90": 0.10, "offset_p90": 0.10}
    assert cap.OVERFIT_GATE == {"angle_median": 0.10, "offset_median": 0.05,
                                "angle_p90": 0.25, "offset_p90": 0.15}
    assert (cap.ANGLE_BUDGET_DEG, cap.OFFSET_BUDGET_CELL) == (1.0, 0.5)
    assert (cap.SAFETY_ANGLE, cap.SAFETY_OFFSET) == (2.0, 1.0)
    assert (cap.APPROACH_ANGLE, cap.APPROACH_OFFSET) == (1.5, 0.75)
    assert (cap.SHUFFLE_ANGLE_MARGIN, cap.SHUFFLE_OFFSET_MARGIN) == (5.0, 2.0)
    assert cap.SIGMA_CELLS == 1.5


def test_role_channels_are_fixed(cap):
    assert sorted(cap.DERANGEMENT) == list(range(12))
    assert all(i != v for i, v in enumerate(cap.DERANGEMENT))
    for forbidden in ("linear_sum_assignment", "hungarian", "Hungarian"):
        assert forbidden not in source(), forbidden


def test_training_is_blocked_until_the_loss_oracle_passes():
    body = ast.get_source_segment(source(), next(
        node for node in ast.walk(ast.parse(source()))
        if isinstance(node, ast.FunctionDef) and node.name == "main"))
    assert "SUPPORTING_LINE_MAP_LOSS_NOT_IDENTIFIABLE" in body
    assert 'json.loads(oloss.read_text())["OLOSS_PASS"]' in body
    assert "confirm6k" in body and "APPROACH" in body


def test_no_pose_quantity_and_no_sealed_set():
    tree = ast.parse(source())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("solve_pose", "solvePnP", "dims", "intrinsics", "CIGM",
                      "EGCR", "projected_cuboid_3d"):
        assert forbidden not in names, forbidden
    for token in ("validation512", "wood45", "handannot17", "testset_full8"):
        assert token not in source(), token


def test_the_rgb_arm_trains_its_stem(cap):
    head, stem, parameters = cap.build_arm("M1_F50_RGB_SLINE")
    assert stem is not None
    ids = {id(p) for p in parameters}
    assert all(id(p) in ids for p in stem.parameters())
    assert cap.build_arm("M0_F50_SLINE")[1] is None
    del head, stem, parameters
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_provenance_records_every_locked_component(cap):
    keys = set(cap.provenance())
    assert {"runner_sha", "target_semantics_sha", "hough_decoder_sha",
            "split_sha", "population_sha", "sigma_map100_pixel", "seed"} <= keys


def test_reload_parity_when_run(cap):
    results = load_json("supporting_line_map_arms.json")
    for name, entry in results.items():
        for stage in ("search2k", "confirm6k"):
            parity = entry.get(f"{stage}_reload_parity")
            if parity is not None:
                assert parity["max_delta"] == 0.0, (name, stage)
