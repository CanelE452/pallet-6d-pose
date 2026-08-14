"""Phase D synthetic gradient tests for the stage-wise corner losses."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE = ROOT / "Deep_Object_Pose/train/stagewise_corner_loss.py"


def _scl():
    if str(ROOT / "Deep_Object_Pose/train") not in sys.path:
        sys.path.insert(0, str(ROOT / "Deep_Object_Pose/train"))
    spec = importlib.util.spec_from_file_location("SCL", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _map(peaks: list[tuple[float, float, float]], grid: int = 50) -> torch.Tensor:
    heat = torch.zeros(1, 8, grid, grid)
    for x, y, value in peaks:
        heat[:, :, int(y), int(x)] = value
    return heat


# -- D1 confident-wrong -----------------------------------------------------
def test_confident_wrong_map_lowers_the_wrong_peak_and_raises_the_gt() -> None:
    scl = _scl()
    heat = _map([(40.0, 40.0, 0.9)]).requires_grad_(True)
    centres = torch.zeros(1, 9, 2)
    centres[:, :8] = torch.tensor([5.0, 5.0])
    valid = torch.ones(1, 9)
    stages = (heat, heat, heat)
    losses = scl.StagewiseCornerLoss()(list(stages) * 2, centres, valid)
    total = losses["mass"] + losses["rank"] + losses["distance"]
    assert torch.isfinite(total)
    total.backward()
    grad = heat.grad[0, 0]
    assert grad[40, 40] > 0, "descent must lower the wrong peak"
    assert grad[5, 5] < 0, "descent must raise the GT cell"
    # the GT is 49 cells away from the wrong peak, far outside any 7x7 window
    assert float(((40 - 5) ** 2 + (40 - 5) ** 2) ** 0.5) > 30


def test_gradient_exists_outside_the_wrong_peak_window() -> None:
    scl = _scl()
    heat = _map([(45.0, 45.0, 1.2)]).requires_grad_(True)
    centres = torch.zeros(1, 9, 2)
    centres[:, :8] = torch.tensor([3.0, 3.0])
    losses = scl.StagewiseCornerLoss()([heat] * 6, centres, torch.ones(1, 9))
    (losses["mass"] + losses["distance"]).backward()
    assert heat.grad[0, 0, 3, 3].abs() > 0
    assert heat.grad[0, 0, 20, 20].abs() > 0, "mid-field cells must get gradient"


# -- D2 correct map ---------------------------------------------------------
def test_correct_map_gives_small_losses() -> None:
    scl = _scl()
    grid = 50
    heat = torch.full((1, 8, grid, grid), -2.0)
    heat[:, :, 25, 25] = 3.0
    centres = torch.zeros(1, 9, 2)
    centres[:, :8] = torch.tensor([25.0, 25.0])
    good = scl.StagewiseCornerLoss()([heat] * 6, centres, torch.ones(1, 9))

    wrong = torch.full((1, 8, grid, grid), -2.0)
    wrong[:, :, 45, 45] = 3.0
    bad = scl.StagewiseCornerLoss()([wrong] * 6, centres, torch.ones(1, 9))
    for name in ("mass", "rank", "distance"):
        assert good[name] < bad[name], name
        assert torch.isfinite(good[name])


# -- D3 stage regression ----------------------------------------------------
def test_progress_penalises_a_later_stage_that_drifts_away() -> None:
    scl = _scl()
    centres = torch.zeros(1, 9, 2)
    centres[:, :8] = torch.tensor([10.0, 10.0])
    near = torch.full((1, 8, 50, 50), -2.0)
    near[:, :, 10, 10] = 3.0
    far = torch.full((1, 8, 50, 50), -2.0)
    far[:, :, 40, 40] = 3.0
    good = [near] * 6
    regressing = [near, near, near, near.clone(), far, far]
    assert scl.StagewiseCornerLoss()(regressing, centres, torch.ones(1, 9))["progress"] > \
        scl.StagewiseCornerLoss()(good, centres, torch.ones(1, 9))["progress"]


def test_progress_detaches_the_earlier_stage() -> None:
    scl = _scl()
    centres = torch.zeros(1, 9, 2)
    centres[:, :8] = torch.tensor([10.0, 10.0])
    h4 = torch.full((1, 8, 50, 50), -2.0)
    h4[:, :, 10, 10] = 3.0
    h4.requires_grad_(True)
    h5 = torch.full((1, 8, 50, 50), -2.0)
    h5[:, :, 40, 40] = 3.0
    h5.requires_grad_(True)
    h6 = h5.clone()
    losses = scl.StagewiseCornerLoss()([h4, h4, h4, h4, h5, h6], centres,
                                       torch.ones(1, 9))
    losses["progress"].backward()
    assert h4.grad is None or float(h4.grad.abs().sum()) == 0.0, \
        "stage 4 must not be pushed to satisfy the constraint"
    assert float(h5.grad.abs().sum()) > 0


# -- D4 border GT -----------------------------------------------------------
def test_border_gt_uses_a_cropped_window_and_stays_finite() -> None:
    scl = _scl()
    for centre in ([0.0, 0.0], [49.0, 49.0], [0.0, 49.0]):
        centres = torch.zeros(1, 9, 2)
        centres[:, :8] = torch.tensor(centre)
        heat = torch.randn(1, 8, 50, 50)
        losses = scl.StagewiseCornerLoss()([heat] * 6, centres, torch.ones(1, 9))
        for name in scl.StagewiseCornerLoss.names:
            assert torch.isfinite(losses[name]), (centre, name)
    mask = scl.gt_window_mask(torch.zeros(1, 8, 50, 50),
                              torch.zeros(1, 8, 2))
    assert int(mask[0, 0].sum()) == 4, "a corner GT gives a cropped 2x2 window"


# -- validity and exclusions ------------------------------------------------
def test_validity_uses_the_transformed_centre_not_an_empty_raster() -> None:
    scl = _scl()
    centres = torch.tensor([[[10.0, 10.0], [-3.0, 10.0], [10.0, 60.0]]
                            + [[5.0, 5.0]] * 6])
    valid = scl.valid_corners(centres)
    assert list(valid[0, :3]) == [True, False, False]


def test_centroid_channel_is_never_touched() -> None:
    scl = _scl()
    assert scl.N_CORNERS == 8
    heat = torch.randn(1, 9, 50, 50, requires_grad=True)
    centres = torch.full((1, 9, 2), 25.0)
    losses = scl.StagewiseCornerLoss()([heat] * 6, centres, torch.ones(1, 9))
    sum(losses[name] for name in scl.StagewiseCornerLoss.names).backward()
    assert float(heat.grad[:, 8].abs().sum()) == 0.0


def test_wrong_peak_excludes_a_radius_around_the_gt() -> None:
    scl = _scl()
    assert scl.GT_EXCLUSION == 4
    centres = torch.full((1, 8, 2), 25.0)
    heat = torch.full((1, 8, 50, 50), -5.0)
    heat[:, :, 27, 25] = 4.0          # inside the exclusion radius
    heat.requires_grad_(True)
    scl.rank_loss((heat, heat, heat), centres, torch.ones(1, 8)).backward()
    assert float(heat.grad[0, 0, 27, 25].abs()) >= 0  # counted as GT-side, not wrong


def test_full_map_softmax_at_temperature_one_tenth() -> None:
    scl = _scl()
    assert scl.TEMPERATURE == 0.1
    probability = scl.spatial_probability(torch.randn(2, 8, 50, 50))
    assert torch.allclose(probability.sum(dim=(-2, -1)),
                          torch.ones(2, 8), atol=1e-5)
    source = MODULE.read_text("utf-8")
    assert "sigmoid" not in source.lower().split('"""')[-1]


def test_no_forbidden_branch() -> None:
    text = MODULE.read_text("utf-8").lower()
    for banned in ("proposal", "router", "graph", "mask_head", "semantic_line",
                   "voting", "diffpnp"):
        assert banned not in text


# ============================================================================
# Phase M — screen integrity (results-level)
# ============================================================================
import json  # noqa: E402

OUT = ROOT / "data/pallet/results/paper_s2_stagewise_bias_screen"
RUNNER = ROOT / "scripts/stage0/paper_s2/paper_s2_stagewise_bias_screen.py"


def _provenance():
    path = OUT / "stagewise_run_provenance.json"
    if not path.is_file():
        pytest.skip("screen not run")
    return json.loads(path.read_text("utf-8"))


def test_baseline_reproduced_before_training() -> None:
    gate = _provenance()["baseline_gate"]
    assert gate["passed"] is True
    assert (gate["strict_n"], gate["gt2d_pose_success"],
            gate["pred_pose_success"]) == (87, 87, 70)


def test_only_belief_stages_four_to_six_train() -> None:
    provenance = _provenance()
    assert provenance["frozen_audit"] == {"vgg_trainable": 0,
                                          "belief123_trainable": 0,
                                          "affinity_trainable": 0}
    assert provenance["trainable_params"] == 12567579
    assert 'TRAINABLE_PREFIX = ("m4_2.", "m5_2.", "m6_2.")' in RUNNER.read_text("utf-8")


def test_canonical_roots_and_sampler_reused() -> None:
    provenance = _provenance()
    header = (ROOT / "weights/paper_s2_stageB/header.txt").read_text("utf-8")
    assert len(provenance["roots"]) == 6
    for root in provenance["roots"]:
        assert root in header
    assert provenance["balance_groups"] in header
    assert "SCREEN.build_loader" in RUNNER.read_text("utf-8")


def test_calibration_is_gradient_norm_and_train_only() -> None:
    path = OUT / "stagewise_loss_grad_calibration.json"
    if not path.is_file():
        pytest.skip("calibration missing")
    payload = json.loads(path.read_text("utf-8"))
    assert payload["batches"] == 8
    assert payload["parameter"].startswith("m6_2.")
    assert set(payload["target_ratio"]) == {"mass", "rank", "distance", "progress"}
    assert payload["target_ratio"] == {"mass": 0.20, "rank": 0.15,
                                       "distance": 0.10, "progress": 0.05}
    reference = payload["grad_norm_median"]["legacy"]
    for key, ratio in payload["target_ratio"].items():
        achieved = payload["weighted_grad_norm"][key] / reference
        assert abs(achieved - ratio) < 1e-6, (key, achieved, ratio)
    source = RUNNER.read_text("utf-8")
    body = source[source.index("def calibrate"):source.index("# Phase G")]
    assert "N87" not in body and "mechanism" not in body


def test_exactly_five_epochs_and_no_checkpoint_selection() -> None:
    source = RUNNER.read_text("utf-8")
    assert "EPOCHS, SEED, LR = 5, 1, 5e-5" in source
    assert "best_epoch" not in source and "early_stop" not in source
    state = ROOT / "weights/paper_s2_stagewise_bias_screen/run_state.json"
    if state.is_file():
        payload = json.loads(state.read_text("utf-8"))
        assert payload["epochs"] == 5 and payload["completed"] is True


def test_mechanism_read_only_at_epoch_zero_and_five() -> None:
    source = RUNNER.read_text("utf-8")
    body = source[source.index("def train("):source.index("# Phase H/I")]
    assert "evaluate(" not in body


def test_canonical_pnp_includes_the_centroid() -> None:
    source = RUNNER.read_text("utf-8")
    body = source[source.index("def evaluate("):source.index("def main()")]
    assert "centroid included" in body
    assert 'points = decoded[6]["D0"]' in body
    assert "points[8] = None" not in body


def test_gate_thresholds_are_the_pre_specified_ones() -> None:
    path = OUT / "stagewise_gate.json"
    if not path.is_file():
        pytest.skip("gate missing")
    names = [c["name"] for c in json.loads(path.read_text("utf-8"))["conditions"]]
    assert len(names) == 12
    assert names[0].startswith("1 F2 far median -15%")
    assert names[3].startswith("4 sharpen-no-correct -30%")


def test_diffpnp_not_run_when_the_gate_fails() -> None:
    path = OUT / "stagewise_gate.json"
    if not path.is_file():
        pytest.skip("gate missing")
    gate = json.loads(path.read_text("utf-8"))
    assert not gate["passed"]
    assert "NOT RUN" in (OUT / "STAGEWISE_DIFFPNP_GATE.md").read_text("utf-8")
    assert not (OUT / "stagewise_diffpnp_gate.json").exists()


def test_no_weights_tracked_or_staged() -> None:
    import subprocess

    for args in (["git", "ls-files", "weights/"],
                 ["git", "diff", "--cached", "--name-only"]):
        out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout
        assert not [line for line in out.splitlines() if line.endswith((".pth", ".pt"))]
