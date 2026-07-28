from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.stage0 import paper_s2_rgb1_eval as evaluator


def _frame(split: str, domain: str, fid: str) -> dict[str, str]:
    return {"split": split, "domain": domain, "fid": fid}


def test_training_exclusion_records_counts_and_does_not_mutate() -> None:
    canonical = {
        "filterval": [
            _frame("filterval", "outside", "100"),
            _frame("filterval", "manual", "200"),
        ],
        "handannot17": [_frame("handannot17", "cad", "300")],
    }
    before = deepcopy(canonical)

    evaluated, excluded = evaluator._exclude_training_fids(
        canonical, ["200", "300"])

    assert canonical == before
    assert evaluator._real_membership_counts(canonical) == {
        "total": 3,
        "by_split": {"filterval": 2, "handannot17": 1},
        "by_split_domain": {
            "filterval": {"manual": 1, "outside": 1},
            "handannot17": {"cad": 1},
        },
    }
    assert evaluator._real_membership_counts(evaluated) == {
        "total": 1,
        "by_split": {"filterval": 1, "handannot17": 0},
        "by_split_domain": {
            "filterval": {"outside": 1},
            "handannot17": {},
        },
    }
    assert excluded == [
        {"split": "filterval", "domain": "manual", "fid": "200"},
        {"split": "handannot17", "domain": "cad", "fid": "300"},
    ]
    assert (evaluator._real_membership_digest(canonical) !=
            evaluator._real_membership_digest(evaluated))


@pytest.mark.parametrize(
    ("configured", "error"),
    [
        (["missing"], AssertionError),
        (["100", "100"], ValueError),
        ("100", TypeError),
        ([100], TypeError),
    ],
)
def test_training_exclusion_rejects_invalid_config(configured, error) -> None:
    canonical = {"filterval": [_frame("filterval", "outside", "100")]}
    with pytest.raises(error):
        evaluator._exclude_training_fids(canonical, configured)


def test_training_exclusion_rejects_ambiguous_canonical_fid() -> None:
    canonical = {
        "filterval": [_frame("filterval", "outside", "100")],
        "handannot17": [_frame("handannot17", "cad", "100")],
    }
    with pytest.raises(AssertionError, match="occur exactly once"):
        evaluator._exclude_training_fids(canonical, ["100"])


def _header_config(*, diffpnp: bool, aspect_resize: bool) -> dict:
    flags = ["--mask_aux"]
    if diffpnp:
        flags.append("--diffpnp")
    if aspect_resize:
        flags.append("--aspect_resize")
    return {
        "common_train_flags": flags,
        "locked": {
            "input_size": 400,
            "sigma": 2.0,
            "seed": 42,
            "quick_epoch_size": 6000,
            "quick_delta_epochs": 3,
            "base_epoch": 57,
            "base_checkpoint": "weights/base.pth",
            "data": ["data/pallet/training_data/example"],
        },
        "arms": {
            "filtered_st": {
                "features": [],
                "trainable_scope": "all",
            },
        },
    }


def _header_text(*, diffpnp: bool, aspect_resize: bool | None) -> str:
    tokens = [
        "imagesize=400",
        "sigma=2.0",
        "manualseed=42",
        "epoch_size=6000",
        "epochs=60",
        "mask_aux=True",
        f"diffpnp={diffpnp}",
        "heatmap_pnp_enhance=False",
        "clip_belief_border=False",
        "mask_belief_fusion=False",
        "extent_loss=False",
        "corner_quality=False",
        "projected_span_loss=False",
        "trainable_scope='all'",
        str((evaluator.ROOT / "weights/base.pth").resolve()),
        str((evaluator.ROOT /
             "data/pallet/training_data/example").resolve()),
    ]
    if aspect_resize is not None:
        tokens.append(f"aspect_resize={aspect_resize}")
    return "Namespace(" + ", ".join(tokens) + ")"


def test_header_validation_supports_no_diffpnp_aspect_resize(tmp_path) -> None:
    directory = tmp_path / "filtered_st"
    directory.mkdir()
    (directory / "header.txt").write_text(
        _header_text(diffpnp=False, aspect_resize=True))
    config = _header_config(diffpnp=False, aspect_resize=True)

    evaluator._validate_training_header(
        config, "quick", "filtered_st", directory)

    (directory / "header.txt").write_text(
        _header_text(diffpnp=True, aspect_resize=True))
    with pytest.raises(RuntimeError, match="diffpnp=False"):
        evaluator._validate_training_header(
            config, "quick", "filtered_st", directory)


def test_header_validation_keeps_legacy_no_aspect_field(tmp_path) -> None:
    directory = tmp_path / "filtered_st"
    directory.mkdir()
    (directory / "header.txt").write_text(
        _header_text(diffpnp=True, aspect_resize=None))
    config = _header_config(diffpnp=True, aspect_resize=False)

    evaluator._validate_training_header(
        config, "quick", "filtered_st", directory)
