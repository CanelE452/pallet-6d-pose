from pathlib import Path

import pytest

from . import report as R


def protocol_fixture():
    return {
        "schema": "resnet18_full_dim_local_refiner_protocol_v1",
        "status": "SEALED_BEFORE_SOURCE_CACHE_OR_HEAD_TRAINING",
        "baseline_arm": "FULL",
        "baseline_trainable": False,
        "arms": ["D0", "P0", "P5", "P5_CONSTANT"],
        "seeds": [1, 2, 3],
        "steps": 6000,
        "batch": 16,
        "fits": 12,
        "training": {
            "partitions": {"train": 55980, "calibration": 1004,
                           "selection": 1031, "heldout": 1985},
            "real_training_images": 0,
        },
        "calibration": {"real_selection": False},
        "selection": {"real_selection": False},
        "evaluation": {"independent_TEST": False, "physical_6D_ground_truth": False},
        "interpretation": {
            "all_arms_share_dimension_conditioned_FULL_baseline": True,
            "P5_minus_P5_CONSTANT": "incremental input contrast",
        },
    }


def stat(delta, low, high, *, seeds=True):
    row = {
        "status": "COMPLETE", "delta": delta, "low": low, "high": high,
        "frames": 319, "units": 13, "resamples": 10000,
        "random_seed": 20260914,
    }
    if seeds:
        row["seeds"] = 3
    return row


def claim_fixture():
    causal = {
        "conditional_keypoint_median": {"session": stat(-1.0, -1.5, -.5)},
        "ALL_GT_PCK": {"10": stat(.02, .01, .03, seeds=False)},
        "pose": {
            "translation_error_cm": stat(-.8, -1.1, -.2),
            "rotation_error_deg": stat(-.4, -.7, -.1),
            "coverage": stat(.0, -.01, .01, seeds=False),
        },
    }
    package = {
        "conditional_keypoint_median": {"session": stat(-.5, -1.0, .1)},
        "pose": {
            "translation_error_cm": stat(-.2, -.8, .5),
            "rotation_error_deg": stat(-.1, -.4, .2),
        },
    }
    return {
        "paired": {"results": {
            "P5_minus_P5_CONSTANT": causal,
            "P5_minus_FULL": package,
        }},
        "protocol": protocol_fixture(),
        "dimension_support": {"objects": {"wood": {
            "outside_train_axiswise_box": True,
        }}},
    }


def test_missing_inventory_does_not_create_output(tmp_path: Path):
    absent = tmp_path / "missing.json"
    destination = tmp_path / "report-output"
    with pytest.raises(R.ReportContractError, match="inputs are incomplete"):
        R.require_complete_inventory([absent])
    assert not destination.exists()


def test_protocol_rejects_method_or_role_drift():
    good = protocol_fixture()
    R.validate_protocol_shape(good)
    bad = protocol_fixture()
    bad["arms"] = ["D0", "P0", "P5"]
    with pytest.raises(R.ReportContractError, match="arms drift"):
        R.validate_protocol_shape(bad)
    bad = protocol_fixture()
    bad["evaluation"]["independent_TEST"] = True
    with pytest.raises(R.ReportContractError, match="independent TEST"):
        R.validate_protocol_shape(bad)


def test_claim_audit_separates_exploratory_support_from_generalization():
    audit = R.derive_claim_audit(claim_fixture())
    assert audit["claims"]["incremental_head_dimension_input_2D_on_reused_DEV"]["status"] == "SUPPORTED_EXPLORATORY"
    assert audit["claims"]["incremental_head_dimension_input_TR_on_reused_DEV"]["status"] == "SUPPORTED_EXPLORATORY"
    assert audit["broad_claim_gates"]["stable_joint_TR_generalization"] is False
    assert audit["broad_claim_gates"]["square_refiner_transfer_evaluated"] is False


def test_claim_audit_requires_both_T_and_R_intervals():
    fixture = claim_fixture()
    fixture["paired"]["results"]["P5_minus_P5_CONSTANT"]["pose"]["rotation_error_deg"] = stat(
        -.1, -.5, .4)
    audit = R.derive_claim_audit(fixture)
    result = audit["claims"]["incremental_head_dimension_input_TR_on_reused_DEV"]
    assert result["point_estimates_both_improve"] is True
    assert result["both_session_intervals_support_benefit"] is False
    assert result["status"] == "DIRECTION_ONLY"

