"""Integrity tests for the corner proposal replacement screen."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data/pallet/results/paper_s2_corner_replacement_screen"
WEIGHTS = ROOT / "weights/paper_s2/paper_s2_corner_replacement_screen"
SCRIPT = ROOT / "scripts/stage0/paper_s2/paper_s2_corner_replacement_screen.py"
MODULE = ROOT / "Deep_Object_Pose/common/corner_proposal_replacement.py"
EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"


def _cpr():
    for extra in (ROOT / "Deep_Object_Pose/common",):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location("CPR", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provenance():
    path = OUT / "corner_replacement_provenance.json"
    if not path.is_file():
        pytest.skip("screen not run")
    return json.loads(path.read_text("utf-8"))


# -- identity ---------------------------------------------------------------
def test_initial_checkpoint_is_untouched() -> None:
    if not EP57.is_file():
        pytest.skip("checkpoint absent")
    assert hashlib.sha256(EP57.read_bytes()).hexdigest() == EP57_SHA


def test_no_weights_are_tracked_or_staged() -> None:
    for args in (["git", "ls-files", "weights/"],
                 ["git", "diff", "--cached", "--name-only"]):
        out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout
        assert not [line for line in out.splitlines() if line.endswith((".pth", ".pt"))]


def test_baseline_reproduced_before_training() -> None:
    gate = _provenance()["baseline_gate"]
    assert gate["passed"] is True
    assert (gate["strict_n"], gate["gt2d_pose_success"], gate["pred_pose_success"]) \
        == (87, 87, 70)


# -- dataset ----------------------------------------------------------------
def test_canonical_roots_and_sampler_are_reused_verbatim() -> None:
    provenance = _provenance()
    header = (ROOT / "weights/paper_s2_stageB/header.txt").read_text("utf-8")
    for root in provenance["roots"]:
        assert root in header, f"{root} is not a canonical Stage-B root"
    assert provenance["balance_groups"] in header
    assert len(provenance["roots"]) == 6


def test_loader_construction_is_shared_with_the_trainer() -> None:
    """The screen must call the trainer's own builder, not a copy."""
    source = SCRIPT.read_text("utf-8")
    assert "TRAIN.build_training_loader(" in source
    trainer = (ROOT / "Deep_Object_Pose/train/train.py").read_text("utf-8")
    assert "def build_training_loader(" in trainer
    assert trainer.count("training_dataset = CleanVisiiDopeLoader(") == 1


def test_mechanism_set_is_not_in_training() -> None:
    provenance = _provenance()
    for root in provenance["roots"]:
        assert "filterval" not in root and "mechanism" not in root


# -- architecture -----------------------------------------------------------
def test_feature_layers_are_discovered_not_hardcoded() -> None:
    source = MODULE.read_text("utf-8")
    assert "def find_feature_layer" in source
    provenance = _provenance()
    assert provenance["features"]["index_high"] != provenance["features"]["index_low"]


def test_exactly_eight_proposal_maps_and_no_centroid() -> None:
    cpr = _cpr()
    assert cpr.N_CORNERS == 8
    torch.manual_seed(0)
    branch = cpr.CornerProposalReplacement(256, 128)
    result = branch(torch.randn(2, 256, 100, 100), torch.randn(2, 128, 50, 50),
                    torch.rand(2, 8, 50, 50), torch.rand(2, 8, 50, 50),
                    torch.rand(2, 9, 50, 50), torch.randn(2, 8, 3), torch.rand(2, 3))
    assert result["proposal"].shape == (2, 8, 50, 50)
    assert result["refined"].shape == (2, 8, 50, 50)
    assert result["gate"].shape == (2, 8)


def test_gate_is_in_unit_interval_and_starts_near_one_percent() -> None:
    cpr = _cpr()
    torch.manual_seed(0)
    branch = cpr.CornerProposalReplacement(256, 128)
    result = branch(torch.randn(2, 256, 100, 100), torch.randn(2, 128, 50, 50),
                    torch.rand(2, 8, 50, 50), torch.rand(2, 8, 50, 50),
                    torch.rand(2, 9, 50, 50), torch.randn(2, 8, 3), torch.rand(2, 3))
    gate = result["gate"]
    assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0
    assert abs(float(gate.mean()) - 0.01) < 0.005
    assert float((result["refined"] - result["base"]).abs().max()) <= 0.02


def test_proposal_gradient_reaches_the_whole_map() -> None:
    """A GT far from the current peak must still receive gradient."""
    cpr = _cpr()
    torch.manual_seed(0)
    maps = torch.zeros(1, 8, 50, 50, requires_grad=True)
    centres = torch.full((1, 8, 2), 45.0)
    valid = torch.ones(1, 8)
    diagonal = torch.tensor([40.0])
    cpr.proposal_objective(maps, centres, valid, diagonal).backward()
    corner = maps.grad[0, 0]
    assert float(corner.abs().sum()) > 0
    assert float(corner[44:47, 44:47].abs().sum()) > 0, "no gradient at the GT cell"
    assert float(corner[0:5, 0:5].abs().sum()) > 0, "gradient is confined to a window"


def test_belief_range_transform_matches_the_audited_operating_point() -> None:
    source = MODULE.read_text("utf-8")
    assert "sigmoid" in source
    assert "min -0.030, max 1.004" in source, "the range audit must be recorded"


def test_no_forbidden_branch() -> None:
    text = (MODULE.read_text("utf-8") + SCRIPT.read_text("utf-8")).lower()
    for banned in ("message_passing", "graph_block", "semantic_line",
                   "diffpnp", "voting", "vector_field"):
        if banned == "diffpnp":
            # the canonical Stage-B options carry a diffpnp index path; the
            # screen must not build a DiffPnP loss from it
            assert "diffpnp3d_loss" not in text and "diffpnploss" not in text
            continue
        assert banned not in text


# -- training discipline ----------------------------------------------------
def test_exactly_five_epochs_no_selection_no_extension() -> None:
    module_source = SCRIPT.read_text("utf-8")
    assert "EPOCHS = 5" in module_source
    assert "early_stop" not in module_source and "best_epoch" not in module_source
    state = WEIGHTS / "run_state.json"
    if state.is_file():
        payload = json.loads(state.read_text("utf-8"))
        assert payload["epochs"] == 5
        assert payload["epoch"] <= 5


def test_loss_calibration_used_train_data_only_and_is_frozen() -> None:
    path = WEIGHTS / "loss_calibration.json"
    if not path.is_file():
        pytest.skip("calibration not written")
    payload = json.loads(path.read_text("utf-8"))
    assert payload["batches"] == 20
    assert payload["target_share"] == 0.20
    assert payload["lambda_proposal"] > 0 and payload["lambda_refined"] > 0
    source = SCRIPT.read_text("utf-8")
    calibrate = source[source.index("def calibrate"):source.index("def train(")]
    assert "N87" not in calibrate and "mechanism" not in calibrate


def test_mechanism_is_evaluated_only_at_epoch_zero_and_five() -> None:
    source = SCRIPT.read_text("utf-8")
    train_body = source[source.index("def train("):source.index("# Phase E")]
    assert "evaluate_mechanism" not in train_body, "N87 read during training"
    # offline re-evaluation must refuse any checkpoint other than 0 or 5
    assert 'BLOCKED: N87 may only be evaluated at epoch 0 or 5' in source
    assert '"epoch_005.pth", "last.pth"' in source


def test_reevaluate_refuses_an_intermediate_epoch() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--reevaluate",
         "weights/paper_s2/paper_s2_corner_replacement_screen/epoch_003.pth"],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert result.returncode != 0
    assert "BLOCKED" in (result.stdout + result.stderr)


def test_all_three_paths_are_evaluated() -> None:
    for name in ("mechanism_epoch0.parquet", "mechanism_epoch5.parquet"):
        path = OUT / name
        if not path.is_file():
            pytest.skip("evaluation not finished")
        import pandas as pd

        table = pd.read_parquet(path)
        for arm in ("base", "proposal", "refined"):
            assert f"{arm}_pose_success" in table.columns
            assert f"{arm}_reproj" in table.columns


def test_validity_uses_the_transformed_gt_centre() -> None:
    source = SCRIPT.read_text("utf-8")
    body = source[source.index("def screen_losses"):source.index("# Phase D")]
    assert "refine_keypoints_valid" in body
    assert "inside" in body and "BELIEF_GRID - 1" in body


def test_dataset_change_is_additive_only() -> None:
    loader = (ROOT / "Deep_Object_Pose/common/utils_dataset.py").read_text("utf-8")
    assert 'out["dims_m"]' in loader and 'out["dims_valid"]' in loader
    # the added block must not touch the image/belief/affinity pipeline
    block = loader[loader.index('out["dims_m"]') - 1200:loader.index('out["dims_valid"]')]
    for forbidden in ("beliefs =", "affinities =", "img_tensor ="):
        assert forbidden not in block
