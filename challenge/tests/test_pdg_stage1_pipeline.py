"""Tests for the Stage-1 autonomous runner."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/pdg_unified_program")
RUN = OUT / "stage1_runner"
PIPELINE = ROOT / "scripts/stage0/pdg_stage1_pipeline.py"


@pytest.fixture(scope="module")
def pipeline():
    import importlib.util
    for path in (ROOT / "scripts/stage0", ROOT / "Deep_Object_Pose/common",
                 ROOT / "Deep_Object_Pose/train"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("pdg_pipeline", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ep57_sha(pipeline):
    digest = hashlib.sha256(
        (ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth").read_bytes()).hexdigest()
    assert digest == pipeline.EP57_SHA


def test_phase_list_and_fixed_settings(pipeline):
    assert pipeline.PHASES == ["PREPARE", "PARITY", "CALIBRATE", "SMOKE_A1",
                               "SMOKE_A2", "TRAIN_A1", "TRAIN_A2", "EVALUATE",
                               "DECIDE", "REPORT"]
    assert pipeline.EPOCHS == 3
    assert pipeline.SEED == 1
    assert pipeline.SMOKE_STEPS == 100
    assert pipeline.LAMBDA_CLAMP == (1e-5, 10.0)
    assert pipeline.TARGET_RATIO == {"palletness": 0.10, "visibility": 0.15,
                                     "truncation": 0.05}


def test_wrapper_parity_recorded_and_within_tolerance():
    payload = json.loads((OUT / "wrapper_parity.json").read_text("utf-8"))
    assert payload["wrapper"]["img"] <= 1e-6
    assert payload["wrapper"]["kp"] <= 1e-6
    assert payload["wrapper"]["belief"] <= 1e-6
    assert payload["wrapper"]["affinity"] <= 1e-6
    assert payload["a1_corner_delta"] <= 1e-6
    for arm in ("A1", "A2"):
        assert payload["step0"][arm]["h6"] <= 1e-8
        assert payload["step0"][arm]["a6"] <= 1e-8


def test_calibration_lambdas_are_inside_the_clamp():
    payload = json.loads((OUT / "grad_calibration.json").read_text("utf-8"))
    for key, value in payload["lambda"].items():
        assert 1e-5 <= value <= 10.0, (key, value)
    assert set(payload["lambda"]) == {"palletness", "visibility", "truncation"}
    assert payload["warmup_steps"] == 50


def test_holdout_guard_blocks_a_sealed_image(pipeline):
    sealed = pipeline.sealed_image_paths()
    assert len(sealed) == 101, len(sealed)          # eval56 56 + wood 45
    pipeline.install_holdout_guard(sealed)
    import cv2
    target = sorted(sealed)[0]
    with pytest.raises(SystemExit) as caught:
        cv2.imread(target)
    assert "HARD_BLOCKED" in str(caught.value)
    assert pipeline.HOLDOUT_HITS["e44"] + pipeline.HOLDOUT_HITS["w45"] >= 1


def test_state_records_zero_holdout_opens():
    state = json.loads((RUN / "state.json").read_text("utf-8"))
    assert state["holdout"]["e44_open"] == 0
    assert state["holdout"]["w45_open"] == 0
    assert state["holdout"]["final_test_open"] == 0


def test_dataset_arms_and_taca_gating(pipeline):
    import pdg_stage1_dataset as DS
    source = (ROOT / "Deep_Object_Pose/train/pdg_stage1_dataset.py").read_text("utf-8")
    assert "self.truncation_aug_prob = 1.0 if arm == \"A2\" else 0.0" in source
    assert "_taca_seam" in source
    assert "BORDER_REFLECT" not in source


def test_no_weights_staged():
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                            capture_output=True, text=True).stdout.splitlines()
    assert not [p for p in staged if p.endswith((".pth", ".pt"))]
