"""Phase 13 tests for the PAPER_S2 micro architecture screen.

These protect the invariants that decide whether a gate verdict means anything:
zero-init identity, frozen base, bounded residual, target-semantics semantics,
edge-loss symmetry, and mechanism-val / final-test isolation.

Tests needing trained artifacts skip when the screen has not been run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage0" / "paper_s2" / "paper_s2_micro_arch_screen.py"
for _p in (
    ROOT / "Deep_Object_Pose" / "common",
    ROOT / "Deep_Object_Pose" / "train",
    ROOT / "scripts" / "stage0",
    ROOT / "scripts" / "data_prep" / "eval",
    ROOT / "challenge" / "scripts",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SPEC = importlib.util.spec_from_file_location("paper_s2_micro_arch_screen", SCRIPT)
MS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MS)

OUT = MS.OUT_DIR
WEIGHTS = MS.WEIGHT_DIR


def _require(*paths: Path) -> None:
    for path in paths:
        if not path.exists():
            pytest.skip(f"screen artifact not built: {path.name}")


@pytest.fixture(scope="module")
def residual_head():
    from belief_residual import BoundedBeliefResidual

    return BoundedBeliefResidual(
        feature_channels=128, belief_channels=9, hidden_channels=16, amplitude=0.25
    )


# --- 1/3/4/5. residual head contract ---------------------------------------
def test_zero_init_gives_exact_identity(residual_head) -> None:
    feature = torch.randn(2, 128, 50, 50)
    belief = torch.randn(2, 9, 50, 50)
    final, delta = residual_head(feature, belief)
    assert float((final - belief).abs().max()) == 0.0
    assert float(delta.abs().max()) == 0.0


def test_residual_is_bounded(residual_head) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        residual_head.conv2.weight.normal_(0.0, 5.0)
        residual_head.conv2.bias.normal_(0.0, 5.0)
    feature = torch.randn(4, 128, 50, 50) * 10.0
    belief = torch.randn(4, 9, 50, 50) * 10.0
    _, delta = residual_head(feature, belief)
    assert float(delta.abs().max()) <= residual_head.amplitude + 1e-6
    residual_head.reset_output_to_zero()


def test_residual_receives_gradient_and_base_stays_frozen() -> None:
    model = MS.ScreenModel(with_residual=True)
    trainable = model.trainable_parameters("residual")
    assert trainable, "residual parameters must be trainable"
    images = torch.randn(1, 3, 400, 400)
    out = model(images)
    out["belief_final"].sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    for name, parameter in model.base.named_parameters():
        assert parameter.grad is None, f"frozen base parameter got a gradient: {name}"


def test_trainable_parameter_counts_are_comparable() -> None:
    residual_model = MS.ScreenModel(with_residual=True)
    tail_model = MS.ScreenModel(with_residual=False)
    residual_count = sum(
        p.numel() for p in residual_model.trainable_parameters("residual")
    )
    tail_count = sum(p.numel() for p in tail_model.trainable_parameters("m6_2_tail"))
    ratio = residual_count / tail_count
    assert 0.8 <= ratio <= 1.2, (
        f"capacity advantage: residual {residual_count} vs tail {tail_count} "
        f"(ratio {ratio:.2f})"
    )


def test_m6_2_tail_scope_selects_only_the_last_two_convolutions() -> None:
    model = MS.ScreenModel(with_residual=False)
    model.trainable_parameters("m6_2_tail")
    trainable = {
        name for name, p in model.base.named_parameters() if p.requires_grad
    }
    assert trainable == {
        "m6_2.10.weight", "m6_2.10.bias", "m6_2.12.weight", "m6_2.12.bias",
    }


# --- 6/7. sampler determinism ----------------------------------------------
def test_sampler_order_identical_across_arms() -> None:
    _require(OUT / "run_configs")
    for control, candidate in (("M0_B", "B1"), ("M0_A", "A1")):
        control_path = OUT / "run_configs" / f"{control}.json"
        candidate_path = OUT / "run_configs" / f"{candidate}.json"
        if not (control_path.is_file() and candidate_path.is_file()):
            pytest.skip(f"{control}/{candidate} not both run")
        a = json.loads(control_path.read_text("utf-8"))
        b = json.loads(candidate_path.read_text("utf-8"))
        assert a["seed"] == b["seed"] == MS.SEED
        assert a["batch_size"] == b["batch_size"]
        assert a["manifest"] == b["manifest"]
        common = min(a["epochs_run"], b["epochs_run"])
        assert common >= 1
        if a["epochs_run"] == b["epochs_run"]:
            assert a["sampler_order_hash"] == b["sampler_order_hash"], (
                f"{control} and {candidate} saw different sample orders"
            )


# --- 8/9/10/11/12. target semantics ----------------------------------------
def test_border_centre_inside_target_is_zero_in_legacy_and_nonzero_in_a1() -> None:
    from utils_belief import CreateBeliefMap

    points = [[[1.0, 1.0]] + [[25.0, 25.0]] * 8]
    legacy = np.asarray(
        CreateBeliefMap(50, points, 9, sigma=MS.SIGMA, clip_at_border=False)
    )
    corrected = np.asarray(
        CreateBeliefMap(50, points, 9, sigma=MS.SIGMA, clip_at_border=True)
    )
    assert legacy[0].sum() == 0.0, "legacy must leave the border positive empty"
    assert corrected[0].sum() > 0.0, "A1 must draw the clipped Gaussian"


def test_legitimate_off_image_point_is_masked_not_negative() -> None:
    from utils_belief import CreateBeliefMap, spatial_keypoint_validity

    points = np.full((9, 2), 25.0)
    points[3] = [-14.0, 20.0]  # outside, but a real coordinate
    validity = spatial_keypoint_validity(points, 50)
    assert validity[3] == 0.0
    assert validity[0] == 1.0
    drawn = np.asarray(
        CreateBeliefMap(50, [points.tolist()], 9, sigma=MS.SIGMA, clip_at_border=True)
    )
    assert drawn[3].sum() == 0.0, "an off-map point must not be clamped to the border"


def test_exact_sentinel_is_masked() -> None:
    from utils_belief import spatial_keypoint_validity

    points = np.full((9, 2), 25.0)
    points[5] = [-100.0, -100.0]
    validity = spatial_keypoint_validity(points, 50)
    assert validity[5] == 0.0


def test_corrupt_coordinates_fail_closed() -> None:
    from utils_belief import spatial_keypoint_validity

    points = np.full((9, 2), 25.0)
    points[2] = [np.nan, 3.0]
    with pytest.raises(ValueError):
        spatial_keypoint_validity(points, 50)


def test_affinity_valid_iff_corner_and_centroid_valid() -> None:
    from heatmap_refinement import pseudo_label_channel_masks

    validity = torch.ones(9)
    validity[8] = 0.0
    belief_mask, affinity_mask = pseudo_label_channel_masks(validity)
    assert float(affinity_mask.sum()) == 0.0, "no corner affinity without the centroid"
    validity = torch.ones(9)
    validity[2] = 0.0
    _, affinity_mask = pseudo_label_channel_masks(validity)
    assert float(affinity_mask[4]) == 0.0 and float(affinity_mask[5]) == 0.0
    assert float(affinity_mask[0]) == 1.0


def test_default_flags_keep_legacy_targets_identical() -> None:
    """Both opt-in flags off must reproduce the historical target exactly."""
    from utils_belief import CreateBeliefMap

    points = [[[1.0, 1.0], [25.0, 25.0], [48.9, 3.0]] + [[30.0, 30.0]] * 6]
    baseline = np.asarray(CreateBeliefMap(50, points, 9, sigma=MS.SIGMA))
    with_default = np.asarray(
        CreateBeliefMap(50, points, 9, sigma=MS.SIGMA, clip_at_border=False)
    )
    assert np.array_equal(baseline, with_default)


# --- 13/14/15. B2 edge loss -------------------------------------------------
def _decoder():
    from diffpnp3d_loss import LocalSoftArgmax2D

    return LocalSoftArgmax2D(
        window=7, temperature=0.1, orig_size=(50, 50), belief_size=(50, 50)
    )


def _gaussian_maps(points: np.ndarray, size: int = 50, sigma: float = 2.0):
    ys = torch.arange(size, dtype=torch.float32)
    xs = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    stack = []
    for x, y in points:
        stack.append(
            torch.exp(-(((xx - float(x)) ** 2 + (yy - float(y)) ** 2) / (2 * sigma**2)))
        )
    return torch.stack(stack).unsqueeze(0)


def test_edge_loss_is_near_zero_for_perfect_geometry() -> None:
    points = np.array(
        [[12, 12], [34, 12], [34, 30], [12, 30],
         [16, 16], [30, 16], [30, 27], [16, 27], [23, 21]], dtype=np.float32
    )
    maps = _gaussian_maps(points)
    mask = torch.ones(1, 9)
    loss = MS.edge_loss(maps, maps, mask, _decoder())
    assert float(loss) < 1e-6


def test_edge_set_is_left_right_symmetric() -> None:
    depth = set(MS.DEPTH_EDGES)
    assert depth == {(0, 4), (1, 5), (2, 6), (3, 7)}
    left = {e for e in depth if e[0] in MS.DEPTH_LEFT_KP}
    right = {e for e in depth if e[0] in MS.DEPTH_RIGHT_KP}
    assert len(left) == len(right) == 2, "depth edges must cover both sides"
    perimeter = set(MS.FAR_PERIMETER_EDGES)
    assert perimeter == {(4, 5), (5, 6), (6, 7), (7, 4)}
    assert all(i in MS.FAR_KP and j in MS.FAR_KP for i, j in perimeter)


def test_edge_loss_excludes_invalid_endpoints() -> None:
    points = np.array(
        [[12, 12], [34, 12], [34, 30], [12, 30],
         [16, 16], [30, 16], [30, 27], [16, 27], [23, 21]], dtype=np.float32
    )
    target = _gaussian_maps(points)
    moved = points.copy()
    moved[4] = [40, 40]
    prediction = _gaussian_maps(moved)
    full_mask = torch.ones(1, 9)
    masked = torch.ones(1, 9)
    masked[0, 4] = 0.0
    assert float(MS.edge_loss(prediction, target, full_mask, _decoder())) > float(
        MS.edge_loss(prediction, target, masked, _decoder())
    )


# --- 16/17/18/19. integrity -------------------------------------------------
def test_mechanism_val_absent_from_training_manifests() -> None:
    _require(OUT / "micro_train_B_manifest.json", OUT / "micro_train_A_manifest.json")
    forbidden = MS.mechanism_val_paths()
    assert forbidden, "mechanism-val membership must be non-empty"
    for name in ("micro_train_B_manifest.json", "micro_train_A_manifest.json"):
        manifest = json.loads((OUT / name).read_text("utf-8"))
        for frame in manifest["frames"]:
            assert str(Path(frame["image_path"]).resolve()) not in forbidden
            assert str(Path(frame["json_path"]).resolve()) not in forbidden


def test_final_test_prohibited_paths_fail_closed() -> None:
    audit = MS.FZ.InputAudit()
    with pytest.raises(RuntimeError):
        audit.guard(ROOT / "data/pallet/raw_data/night/capturenight09/rgb/x.png")
    _require(OUT / "micro_train_B_manifest.json")
    for name in ("micro_train_B_manifest.json", "micro_train_A_manifest.json"):
        manifest = json.loads((OUT / name).read_text("utf-8"))
        for frame in manifest["frames"]:
            blob = f"{frame['image_path']}{frame['json_path']}".lower()
            for token in MS.FZ.PROHIBITED_INPUT_TOKENS:
                assert token not in blob


def test_checkpoint_sha_unchanged() -> None:
    assert MS.FZ.sha256_file(MS.FZ.WEIGHTS) == MS.FZ.WEIGHTS_SHA256


def test_screen_writes_only_under_its_own_result_and_weight_directories() -> None:
    assert MS.OUT_DIR.name == "paper_s2_micro_arch_screen"
    assert MS.WEIGHT_DIR.name == "paper_s2_micro_arch_screen"
    assert MS.FZ.WEIGHTS.parent != MS.WEIGHT_DIR


def test_ablation_table_changes_only_the_intended_element() -> None:
    """One *element* per comparison, per the Phase 12 table.

    M0_B->B1 necessarily also moves the trainable scope: adding a residual head
    is what makes the head the trainable scope.  That coupling is controlled by
    the parameter-count check, not by the table, so it is listed explicitly
    here instead of being silently tolerated.
    """
    table = MS.ablation_table().set_index("arm")
    for control, candidate, expected in (
        ("M0_B", "B1", {"residual", "trainable"}),
        ("M0_A", "A1", {"target_semantics"}),
        ("B1", "B2", {"edge_loss"}),
    ):
        differences = {
            column
            for column in table.columns
            if table.loc[control, column] != table.loc[candidate, column]
        }
        assert differences == expected, (
            f"{control}->{candidate} changes {sorted(differences)}, "
            f"expected {sorted(expected)}"
        )
    # Manifest and target semantics must be shared inside each family.
    assert table.loc["M0_B", "training_manifest"] == table.loc["B1", "training_manifest"]
    assert table.loc["M0_A", "trainable"] == table.loc["A1", "trainable"]
    assert table.loc["M0_B", "target_semantics"] == table.loc["B1", "target_semantics"]


def test_manifest_b_records_unmet_hard_criterion() -> None:
    """The B manifest must not silently pad an unsatisfiable hard criterion."""
    _require(OUT / "micro_train_B_manifest.json")
    manifest = json.loads((OUT / "micro_train_B_manifest.json").read_text("utf-8"))
    criterion = manifest["hard_criterion"]
    assert "criterion_met" in criterion
    if not criterion["criterion_met"]:
        assert criterion["n_above_specified"] < MS.MANIFEST_B_HARD
        assert criterion["threshold_actually_used_px"] is not None


def test_gate_verdicts_are_wellformed() -> None:
    for name in ("B1", "A1", "B2"):
        path = OUT / f"gate_{name}.json"
        if not path.is_file():
            continue
        gate = json.loads(path.read_text("utf-8"))
        assert gate["verdict"] in ("PASS", "FAIL")
        assert gate["verdict"] == (
            "PASS" if (gate["primary_pass"] and gate["guard_pass"]) else "FAIL"
        )
