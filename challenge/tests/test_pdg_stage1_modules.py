"""Stage-1 module tests: TACA, target policy, PRH/KVH/VAPA.

These pin the parts of Phase O that apply to the implemented modules. Training
and evaluation are not covered here because they have not been run.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDG = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/pdg_unified_program")
for path in (ROOT / "Deep_Object_Pose/common", ROOT / "Deep_Object_Pose/train"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pdg_heads as HEADS            # noqa: E402
import pdg_taca as TACA              # noqa: E402
import pdg_targets as TARGETS        # noqa: E402


def code_only(path: pathlib.Path) -> str:
    """Source with comments and string literals removed.

    A module that explains in prose why it does not call `_trunc_pad_back`, or
    that its target is a hull rather than an RLE mask, would otherwise fail a
    raw substring check for exactly the thing it is refusing to do.
    """
    import io
    import tokenize
    return " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(path.read_text("utf-8")).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING))


def test_head_and_checkpoint_unchanged():
    log = subprocess.run(["git", "log", "--format=%H"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    assert "e2214a724b047226d79c0a05722815e6c33b0dc3" in log
    digest = hashlib.sha256(
        (ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth").read_bytes()).hexdigest()
    assert digest == "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"


def test_membership_reconciliation_is_case_1():
    payload = json.loads((PDG / "pdg_membership_reconciliation.json").read_text("utf-8"))
    assert payload["verdict"] == "CASE_1_SERIALIZATION_ONLY"
    assert payload["membership_drift"] is False
    assert payload["frame_ids_identical"] is True
    assert payload["symmetric_difference"] == []


def test_holdout_still_sealed():
    lock = json.loads((PDG / "pdg_membership_lock.json").read_text("utf-8"))
    assert lock["confirmatory_SEALED"]["E44"]["n"] == 44
    assert lock["confirmatory_SEALED"]["W45"]["n"] == 45
    assert "SEALED" in lock["holdout_state"]


# ---- TACA
def test_taca_sampling_is_fixed_50_25_25():
    assert TACA.SAMPLING == (("legacy", 0.50), ("frame_edge_truncation", 0.25),
                             ("constant_margin_scale", 0.25))
    counts = {"legacy": 0, "frame_edge_truncation": 0, "constant_margin_scale": 0}
    rng = random.Random(1)
    for _ in range(20000):
        counts[TACA.pick_class(rng)] += 1
    assert abs(counts["legacy"] / 20000 - 0.50) < 0.02
    assert abs(counts["frame_edge_truncation"] / 20000 - 0.25) < 0.02
    assert abs(counts["constant_margin_scale"] / 20000 - 0.25) < 0.02


def test_taca_uses_constant_grey_and_never_reflect():
    source = code_only(ROOT / "Deep_Object_Pose/common/pdg_taca.py")
    assert TACA.PAD_VALUE == (127, 127, 127)
    assert TACA.PAD_PIXELS == 100
    assert "BORDER_CONSTANT" in source
    assert "BORDER_REFLECT" not in source
    assert "BORDER_REPLICATE" not in source


def test_taca_truncation_keeps_corners_off_screen():
    """The point of the module: no pad-back, so an off-screen corner stays out."""
    source = code_only(ROOT / "Deep_Object_Pose/common/pdg_taca.py")
    assert "_trunc_pad_back" not in source
    rng = random.Random(3)
    image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    points = np.array([[150, 150], [490, 150], [490, 330], [150, 330],
                       [170, 130], [510, 130], [510, 310], [170, 310],
                       [320, 240]], dtype=np.float64)
    produced = 0
    for _ in range(40):
        result = TACA.frame_edge_truncation(image, points, rng)
        if result is None:
            continue
        produced += 1
        _, moved, stats = result
        assert stats["off_screen_corners"] >= 1
        assert TACA.IN_FRAME_MIN <= stats["in_frame_corners"] <= TACA.IN_FRAME_MAX
        assert TACA.BORDER_MIN <= stats["border_proximity_px"] <= TACA.BORDER_MAX
        assert TACA.WIDTH_RATIO_MIN <= stats["bbox_width_ratio"] <= TACA.WIDTH_RATIO_MAX
    assert produced > 0, "the truncation branch never produced a sample"


def test_existing_augmentation_pads_truncation_back_in():
    """Why TACA exists, pinned: the legacy path undoes its own truncation."""
    legacy = (ROOT / "Deep_Object_Pose/common/utils_dataset.py").read_text("utf-8")
    body = legacy[legacy.index("def apply_truncation_aug"):]
    assert "_trunc_pad_back(crop_img, kps_c)" in body
    pad_back = legacy[legacy.index("def _trunc_pad_back"):legacy.index("def apply_truncation_aug")]
    assert "BORDER_REFLECT_101" in pad_back


def test_taca_scale_branch_geometry_is_exact():
    image = np.zeros((480, 640, 3), np.uint8)
    points = np.array([[0, 0], [640, 480], [320, 240]] + [[100, 100]] * 6,
                      dtype=np.float64)
    _, moved, _ = TACA.constant_margin_scale(image, points)
    canvas_w, canvas_h = 640 + 200, 480 + 200
    expected_x = (points[:, 0] + 100) * (640 / canvas_w)
    expected_y = (points[:, 1] + 100) * (480 / canvas_h)
    assert np.abs(moved[:, 0] - expected_x).max() < 1e-9
    assert np.abs(moved[:, 1] - expected_y).max() < 1e-9


# ---- target policy
def test_role_specific_sigma():
    assert TARGETS.CORNER_SIGMA == 2.0 and TARGETS.CENTROID_SIGMA == 2.5
    for channel in range(8):
        assert TARGETS.channel_sigma(channel) == 2.0
    assert TARGETS.channel_sigma(8) == 2.5
    corner = TARGETS.gaussian_channel((25.0, 25.0), 2.0)
    centroid = TARGETS.gaussian_channel((25.0, 25.0), 2.5)
    assert (centroid >= 0.5).sum() > (corner >= 0.5).sum()


def test_off_screen_corner_has_no_target_and_zero_mask():
    points = np.array([[100, 100], [500, 100], [500, 300], [100, 300],
                       [-50, 90], [700, 90], [700, 290], [-50, 290],
                       [300, 200]], dtype=np.float64)
    result = TARGETS.build_targets(points, 640, 480)
    for channel in (4, 5, 6, 7):
        assert result["belief"][channel].sum() == 0.0
        assert result["belief_mask"][channel] == 0.0
        assert result["affinity_mask"][2 * channel] == 0.0
        assert result["affinity_mask"][2 * channel + 1] == 0.0
        assert result["visibility"][channel] == TARGETS.VIS_OFF_SCREEN
    for channel in (0, 1, 2, 3, 8):
        assert result["belief_mask"][channel] == 1.0
        assert result["belief"][channel].max() > 0.9
    assert result["truncated"] is True


def test_no_border_clamp_and_no_sentinel():
    source = code_only(ROOT / "Deep_Object_Pose/train/pdg_targets.py")
    assert "clip" not in source
    assert "sentinel" not in source.lower()
    points = np.array([[-500, -500]] * 9, dtype=np.float64)
    result = TARGETS.build_targets(points, 640, 480)
    assert result["belief"].sum() == 0.0
    assert result["belief_mask"].sum() == 0.0


def test_in_frame_occluded_keeps_its_target():
    points = np.array([[100, 100]] * 9, dtype=np.float64)
    occluded = np.zeros(9, dtype=bool)
    occluded[2] = True
    result = TARGETS.build_targets(points, 640, 480, occluded=occluded)
    assert result["belief_mask"][2] == 1.0
    assert result["belief"][2].max() > 0.9
    assert result["visibility"][2] == TARGETS.VIS_OCCLUDED
    assert result["visibility"][0] == TARGETS.VIS_VISIBLE


def test_validity_comes_from_coordinates_not_from_an_empty_map():
    points = np.array([[100, 100]] * 9, dtype=np.float64)
    source_valid = np.ones(9, dtype=bool)
    source_valid[5] = False
    result = TARGETS.build_targets(points, 640, 480, source_valid=source_valid)
    assert result["belief_mask"][5] == 0.0
    assert result["visibility_mask"][5] == 0.0     # never rendered, so no label
    assert result["visibility_mask"][0] == 1.0


def test_palletness_target_is_object_hull_not_mask():
    points = np.array([[100, 100], [500, 100], [500, 300], [100, 300],
                       [120, 90], [520, 90], [520, 290], [120, 290],
                       [300, 200]], dtype=np.float64)
    target = TARGETS.palletness_target(points, 640, 480)
    assert target.shape == (50, 50)
    assert 0.0 < target.mean() < 1.0
    source = code_only(ROOT / "Deep_Object_Pose/train/pdg_targets.py")
    assert "convexHull" in source
    assert "rle" not in source.lower()


# ---- heads
def test_head_shapes():
    feature = torch.randn(2, 128, 50, 50)
    belief = torch.randn(2, 9, 50, 50)
    assert HEADS.PalletnessResponseHead()(feature).shape == (2, 1, 50, 50)
    assert HEADS.KeypointVisibilityHead()(feature, belief).shape == (2, 9, 3)
    assert HEADS.TruncationHead()(feature).shape == (2, 1)


def test_visibility_head_detaches_the_belief_input():
    feature = torch.randn(2, 128, 50, 50)
    belief = torch.randn(2, 9, 50, 50, requires_grad=True)
    head = HEADS.KeypointVisibilityHead()
    head(feature, belief).sum().backward()
    assert belief.grad is None, "visibility loss must not reshape the belief map"
    assert all(torch.isfinite(p.grad).all() for p in head.trunk.parameters())


def test_vapa_threshold_is_fixed_and_drops_only_off_screen():
    assert HEADS.VAPA_OFF_SCREEN_THRESHOLD == 0.5
    points = [[1.0, 1.0]] * 9
    probabilities = [0.1, 0.9, 0.2, 0.6, 0.05, 0.99, 0.3, 0.49, 0.5]
    kept, dropped = HEADS.visibility_aware_assembly(points, probabilities)
    assert dropped == [1, 3, 5, 8]
    assert kept[0] is not None and kept[7] is not None
    assert kept[1] is None and kept[8] is None


def test_vapa_uses_no_ground_truth():
    import io
    import tokenize
    source = (ROOT / "Deep_Object_Pose/common/pdg_heads.py").read_text("utf-8")
    body = source[source.index("def visibility_aware_assembly"):]
    code = " ".join(t.string for t in
                    tokenize.generate_tokens(io.StringIO(body).readline)
                    if t.type not in (tokenize.COMMENT, tokenize.STRING))
    for name in ("gt", "GT", "oracle", "target"):
        assert name not in code, name


def test_no_learned_pnp_or_seeded_gn_added():
    for name in ("pdg_heads.py", "pdg_taca.py"):
        source = code_only(ROOT / "Deep_Object_Pose/common" / name)
        for forbidden in ("solvePnP", "gauss_newton", "rodrigues"):
            assert forbidden not in source, (name, forbidden)


def test_no_weights_tracked_or_staged():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.splitlines()
    assert not [p for p in tracked if p.endswith((".pth", ".pt"))]
