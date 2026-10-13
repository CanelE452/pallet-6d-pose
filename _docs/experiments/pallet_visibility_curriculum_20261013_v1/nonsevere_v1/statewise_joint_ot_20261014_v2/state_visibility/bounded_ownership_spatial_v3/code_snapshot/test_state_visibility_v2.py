"""State contract: zero S queries, real O target, and separate T provenance."""
from pathlib import Path

import numpy as np
import pytest

from .state_visibility_v2 import (StateVisibilityModel, geometry_state_gates,
                                  extract_geometry_ownership, extract_nonS_rgb26,
                                  infer_state_visibility, patch_radius)


def inputs():
    image = np.zeros((48, 64, 3), np.uint8)
    q = np.array([[8, 8], [56, 8], [56, 40], [8, 40],
                  [16, 16], [48, 16], [48, 32], [16, 32]], float)
    return image, q, np.zeros(8, bool), ["B"] * 4 + ["I"] * 4, q.copy()


def test_self_excluded_before_patch_feature_or_probability_calls():
    image, q, hidden, roles, projected = inputs()
    hidden[[0, 6]] = True
    q[0] = [-100000, -100000]
    q[6] = [100000, 100000]
    roles[0] = roles[6] = "S"
    queried = []
    model = StateVisibilityModel({"visible_threshold": .5, "abstention_margin": .05}, {}, object())

    def extract(image, points, ids, radius, backbone, *, pallet_mask):
        queried.extend(ids.tolist())
        assert radius == patch_radius(points, ids)
        assert not hidden[ids].any()
        return np.zeros((len(ids), 40)), np.zeros((len(ids), 2560))

    def predict(local, spatial, model):
        return np.array([.9, .1, .5, .9, .1, .9])

    result = infer_state_visibility(image, q, hidden, roles, projected, model,
        pallet_mask=np.zeros(image.shape[:2]), feature_extractor=extract, probability_predictor=predict)
    assert queried == [1, 2, 3, 4, 5, 7]
    assert result["states"] == ["S", "B", "O", "U", "I", "O", "S", "I"]
    assert result["p_visible"][0] is result["p_visible"][6] is None
    assert result["records"][0]["reason"] == "fixed_initial_geometric_self_occlusion"


def test_all_self_never_loads_model_or_segmenter():
    image, q, hidden, roles, projected = inputs()
    hidden[:] = True
    result = infer_state_visibility(image, q, hidden, roles, projected, Path("nonexistent-model.json"))
    assert result["states"] == ["S"] * 8
    assert result["rgb_queried_ids"] == []


def test_raw_OOB_is_not_physical_T_and_initial_T_remains_conditional():
    image, q, hidden, roles, projected = inputs()
    q[0] = [100, 20]  # Detector error alone cannot establish T.
    projected[1] = [-2, 8]
    records, ids = geometry_state_gates(image, q, hidden, roles, projected)
    assert records[0]["state"] == "U"
    assert records[0]["raw_query_out_of_frame"] is True
    assert records[0]["conditional_truncation"] is False
    assert records[1]["state"] == "T"
    assert records[1]["raw_query_out_of_frame"] is False
    assert records[1]["outside_canvas_side"] == ["left"]
    assert records[1]["physical_truncation_certified"] is False
    assert 0 not in ids and 1 not in ids


def test_independent_T_requires_explicit_source_and_preserves_S_priority():
    image, q, hidden, roles, projected = inputs()
    with pytest.raises(ValueError, match="provenance"):
        geometry_state_gates(image, q, hidden, roles, projected,
            independent_truncation={0: {"outside_canvas_side": "right"}})
    proof = dict(independent_of_initial_pose=True, source="known_crop_transform", outside_canvas_side="right")
    records, ids = geometry_state_gates(image, q, hidden, roles, projected,
        independent_truncation={0: proof})
    assert records[0]["physical_truncation_certified"] is True
    assert records[0]["state"] == "T" and 0 not in ids
    hidden[0] = True
    records, _ = geometry_state_gates(image, q, hidden, roles, projected,
        independent_truncation={0: proof})
    assert records[0]["state"] == "S"
    assert records[0]["physical_truncation_certified"] is False


def test_rgb26_erases_S_before_global_scene_and_is_invariant_to_S_coordinates(monkeypatch):
    from . import appearance as A
    image, q, hidden, roles, projected = inputs()
    image[:] = np.random.default_rng(4).integers(0, 256, image.shape, dtype=np.uint8)
    ids = np.array([1, 2, 3, 4, 5, 7])
    original = A._scene
    calls = []

    def guarded_scene(image, masked_q, hidden, valid, cfg):
        assert np.isnan(masked_q[[0, 6]]).all()
        assert not valid[[0, 6]].any()
        calls.append(masked_q.copy())
        return original(image, masked_q, hidden, valid, cfg)

    monkeypatch.setattr(A, "_scene", guarded_scene)
    x1, d1 = extract_nonS_rgb26(image, q, ids)
    q[[0, 6]] = [[-1e9, -1e9], [1e9, 1e9]]
    x2, d2 = extract_nonS_rgb26(image, q, ids)
    assert len(calls) == 2
    assert np.array_equal(x1, x2)
    assert d1["patch_radius_px"] == d2["patch_radius_px"]
    assert d1["palette"] == d2["palette"]


def test_failed_semantic_O_validation_abstains_instead_of_recasting_palette_absence():
    image, q, hidden, roles, projected = inputs()
    model = StateVisibilityModel(dict(feature_family="NON_S_RGB26", visible_threshold=.5,
        abstention_margin=.025, external_O_enabled=False), {}, None)
    result = infer_state_visibility(image, q, hidden, roles, projected, model,
        probability_predictor=lambda local, spatial, model: np.full(len(local), .01))
    assert result["states"] == ["U"] * 8
    assert all(row["rgb_evaluated"] for row in result["records"])


def test_geometry_guided_RGB_features_never_use_S_raw_or_projected_coordinates():
    image, q, hidden, roles, projected = inputs()
    image[:] = np.random.default_rng(7).integers(0, 256, image.shape, dtype=np.uint8)
    ids = np.array([1, 2, 3, 4, 5, 7])
    x1 = extract_geometry_ownership(image, q, ids, projected)
    q[[0, 6]] = [[-1e8, -1e8], [1e8, 1e8]]
    projected[[0, 6]] = [[-1e8, 1e8], [1e8, -1e8]]
    x2 = extract_geometry_ownership(image, q, ids, projected)
    assert x1.shape == (6, 24)
    assert np.array_equal(x1, x2)
