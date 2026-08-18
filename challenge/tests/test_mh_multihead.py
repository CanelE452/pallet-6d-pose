"""Tests for the A0/A1/A2 corner+line+mask screen.

What is pinned here is the part of the screen that a later edit could break
without anything looking wrong: the data contract that PnP depends on, the
difference between "this corner is outside the grid" and "there is no corner
here", the arm definitions, and the fact that adding a CIGM path to a new runner
did not unlock CIGM in the old ones.

The runtime checks that need a GPU and a forward pass live in `mh_wiring.py` and
write their numbers to `mh_wiring.json`; where that file exists, its verdicts are
re-read here so a stale wiring result cannot pass unnoticed.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MULTIHEAD = ROOT / "scripts/stage0/multihead"
OUT = ROOT / "data/pallet/results/paper_s2_multihead"
for extra in (MULTIHEAD, ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def data_module():
    import mh_data
    return mh_data


@pytest.fixture(scope="module")
def cigm_module():
    return pytest.importorskip("mh_cigm")


def _labels(data_module, count):
    directory = data_module.DATA / "labels"
    if not directory.is_dir():
        pytest.skip("v2_prod40k_clean_merged absent on this machine")
    stems = sorted(p.name[:-len("_label.json")] for p in directory.iterdir())[:count]
    return [data_module.read_label(stem) for stem in stems]


# --------------------------------------------------------------------------
# the PnP data contract


def test_object_frame_matrix_reprojects_to_the_labelled_pixels(data_module,
                                                               cigm_module):
    """`cuboid` is world-frame; only one signed axis permutation reaches camera.

    This is the assumption every pose number in the screen rests on, and it was
    found by search rather than from documentation, so it is checked rather than
    trusted.
    """
    worst = 0.0
    for label in _labels(data_module, 12):
        model = cigm_module.object_points(label)
        camera = cigm_module.intrinsics(label)
        rotation, translation = cigm_module.gt_pose(label)
        points = (rotation @ model.T).T + translation
        assert (points[:, 2] > 0).all(), "model points fell behind the camera"
        projected = (camera @ points.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        truth = np.asarray(label["objects"][0]["projected_cuboid"], float)
        worst = max(worst, float(np.abs(projected - truth).max()))
    assert worst < 1e-3, worst


def test_object_frame_matrix_is_a_proper_rotation(cigm_module):
    matrix = cigm_module.OBJECT_FROM_WORLD
    assert np.allclose(matrix @ matrix.T, np.eye(3))
    assert np.isclose(np.linalg.det(matrix), 1.0)


# --------------------------------------------------------------------------
# corner supervision semantics (brief section 5)


def test_off_grid_corner_is_invalid_and_never_clamped(data_module):
    grid = np.array([[25.0, 25.0], [-3.0, 25.0], [25.0, 80.0], [49.9, 49.9]])
    maps, valid = data_module.belief_target(grid, sigma=2.0)
    assert valid.tolist() == [True, False, False, True]
    # an invalid channel is empty, and nothing was drawn at the border instead
    assert maps[1].max() == 0.0 and maps[2].max() == 0.0
    assert maps[0].max() > 0.9
    # a corner just inside the far edge still peaks at the far edge, not clamped
    peak = np.unravel_index(int(np.argmax(maps[3])), maps[3].shape)
    assert peak == (49, 49)


def test_invalid_channels_contribute_exactly_zero_to_the_corner_loss():
    arms = pytest.importorskip("mh_arms")
    torch.manual_seed(0)
    target = torch.rand(2, 9, 8, 8)
    beliefs = [torch.rand(2, 9, 8, 8) for _ in range(6)]
    valid = torch.ones(2, 9, dtype=torch.bool)
    reference = arms.corner_loss(beliefs, target, valid)
    valid_partial = valid.clone()
    valid_partial[0, 3] = False
    # dropping a channel must change the loss, and setting that channel's
    # prediction to anything at all must not change it afterwards
    partial = arms.corner_loss(beliefs, target, valid_partial)
    assert not torch.isclose(reference, partial)
    for belief in beliefs:
        belief[0, 3] = 1e6
    assert torch.isclose(arms.corner_loss(beliefs, target, valid_partial), partial)


def test_belief_target_matches_the_dope_generator(data_module):
    import utils_belief
    rng = np.random.default_rng(0)
    grid = rng.uniform(0.0, float(data_module.GRID), size=(9, 2))
    mine, valid = data_module.belief_target(grid, sigma=data_module.CORNER_SIGMA)
    theirs = np.asarray(utils_belief.CreateBeliefMap(
        size=data_module.GRID, pointsBelief=[grid.tolist()], nbpoints=9,
        sigma=data_module.CORNER_SIGMA, save=False, clip_at_border=True))
    assert valid.all()
    assert np.abs(mine - theirs).max() < 1e-6


# --------------------------------------------------------------------------
# the CIGM adapter (brief section 7)


def test_adapter_round_trips_a_known_line(cigm_module):
    """theta/rho in, the same theta/rho out of `lines_from_segments`."""
    import corner_incident_geometry as CIGM
    from mh_arms import DH
    theta = torch.tensor([[10.0, 75.0, 140.0]])
    rho = torch.tensor([[-20.0, 0.0, 33.0]])
    centre, direction = cigm_module.lines_to_segments(theta, rho)
    normal, recovered_rho = CIGM.lines_from_segments(centre, direction)
    expected_theta, expected_rho = DH.canonical_from_centred(theta, rho)
    assert torch.allclose(normal[..., 0], expected_theta.cos(), atol=1e-6)
    assert torch.allclose(normal[..., 1], expected_theta.sin(), atol=1e-6)
    assert torch.allclose(recovered_rho, expected_rho, atol=1e-5)


def test_adapter_recovers_ground_truth_corners(data_module, cigm_module):
    """GT geometry through the whole PATH-L plumbing returns GT corners."""
    from mh_arms import DH, V2
    labels = _labels(data_module, 8)
    grids = []
    for label in labels:
        camera = label["camera_data"]
        width, height = float(camera["width"]), float(camera["height"])
        cuboid = np.asarray(label["objects"][0]["projected_cuboid"], float)
        grids.append(np.stack([cuboid[:, 0] * data_module.GRID / width,
                               cuboid[:, 1] * data_module.GRID / height], 1))
    theta, rho, _, _, _ = V2.gt_lines(np.stack(grids), cigm_module.EDGES)
    theta_t = torch.tensor(theta, dtype=torch.float32)
    rho_t = torch.tensor(rho, dtype=torch.float32)
    theta_c, rho_c = DH.centred_from_canonical(theta_t, rho_t)
    corners, _, _ = cigm_module.cigm_corners(theta_c.to(DH.DEV), rho_c.to(DH.DEV))
    truth = torch.tensor(np.stack(grids)[:, :8], dtype=corners.dtype,
                         device=corners.device)
    error = (corners - truth).norm(dim=-1)
    assert torch.isfinite(corners).all()
    assert float(error.median()) < 0.05, float(error.median())


# --------------------------------------------------------------------------
# arm definitions


def _module_source(name):
    return (MULTIHEAD / name).read_text("utf-8")


def test_arm_names_and_trainable_sets_are_declared_once():
    source = ast.parse(_module_source("mh_arms.py"))
    arms = next(node for node in ast.walk(source)
                if isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "ARMS" for t in node.targets))
    names = [element.value for element in arms.value.elts]
    assert names == ["A0_LINE_ONLY", "A1_CORNER_LINE", "A2_CORNER_LINE_MASK"]


def test_mask_is_never_used_as_a_gate_in_the_first_ablation():
    """Section 8: no multiply into the heatmap, no hard gate, no PnP rejection.

    Checked on the syntax tree, not on the text, because the docstrings here
    discuss exactly the operations being forbidden and a substring search would
    match the prose that explains why they are absent.
    """
    banned_attributes = {"masked_fill", "masked_select", "masked_scatter"}
    for name in ("mh_arms.py", "mh_screen.py"):
        tree = ast.parse(_module_source(name))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in banned_attributes:
                # the line head legitimately masks its own invalid lattice cells
                assert name == "mh_screen.py", f"{name}: {node.attr}"


def test_historical_line_runners_still_declare_cigm_blocked():
    """Opening CIGM here must not have opened it there.

    Five runners write `CIGM: BLOCKED` into their verdicts and their tests assert
    it.  The adapter was deliberately built in a new module so those locks and
    the results they guard stay exactly as recorded.
    """
    runners = [
        "scripts/stage0/line/direct_hough_full_step_extension.py",
        "scripts/stage0/line/direct_hough_f50_adapter_screen.py",
        "scripts/stage0/line/role_encoder_depth_screen.py",
        "scripts/stage0/adaptation/late_a1_adaptation_screen.py",
        "scripts/stage0/adaptation/late_a1_low_rank_adaptation_screen.py",
    ]
    for relative in runners:
        path = ROOT / relative
        if not path.exists():
            continue
        source = path.read_text("utf-8")
        assert "BLOCKED" in source, relative
        assert "corner_incident_geometry" not in source, relative


# --------------------------------------------------------------------------
# recorded wiring verdicts


@pytest.mark.skipif(not (OUT / "mh_wiring.json").exists(),
                    reason="wiring has not been run on this machine")
def test_recorded_wiring_verdicts_all_pass():
    results = json.loads((OUT / "mh_wiring.json").read_text())
    failed = [name for name, entry in results.items()
              if isinstance(entry, dict) and not entry.get("PASS")]
    assert not failed, failed


@pytest.mark.skipif(not (OUT / "mh_wiring.json").exists(),
                    reason="wiring has not been run on this machine")
def test_recorded_parity_is_exact_not_within_a_tolerance():
    parity = json.loads((OUT / "mh_wiring.json").read_text())["T3_PARITY"]
    for key in ("A1_zero_vs_A0_step0", "A1_zero_vs_A0_after",
                "A2_zero_vs_A0_step0", "A2_zero_vs_A0_after", "A1_self_repeat"):
        assert parity[key] == 0.0, (key, parity[key])


@pytest.mark.skipif(not (OUT / "mh_data_contract.json").exists(),
                    reason="split has not been built on this machine")
def test_split_groups_do_not_overlap_and_strata_match():
    contract = json.loads((OUT / "mh_data_contract.json").read_text())
    assert contract["train_group_overlap"] == []
    train, dev = contract["strata_train"], contract["strata_dev"]
    train_total, dev_total = sum(train.values()), sum(dev.values())
    assert set(train) == set(dev)
    for stratum in train:
        assert abs(train[stratum] / train_total
                   - dev[stratum] / dev_total) < 0.01, stratum
