"""Contract canaries for actual-O CAL fitting and real-RGB crop domains."""
from copy import deepcopy

import numpy as np
import pytest

from . import ot_domains_v2 as D


def fixture():
    image = np.arange(80*100*3, dtype=np.uint8).reshape(80, 100, 3)
    q = np.array([[10., 10.], [30., 10.], [60., 10.], [90., 10.],
                  [10., 60.], [30., 60.], [60., 60.], [90., 60.], [50., 40.]])
    xyz = np.array([[x, y, z] for z in (-.1, .1) for y in (-.2, .2) for x in (-.3, .3)])
    K = np.array([[120., 0., 50.], [0., 120., 40.], [0., 0., 1.]])
    return image, q, K, xyz


def cal_row(error=5., **kw):
    row = dict(split="CAL", state="O", external_label_confirmed=True,
               label_source="HUMAN", raw_xy=[10., 10.], target_xy=[10.+error, 10.],
               image_hw=[80, 100], scale_px=100., native_corner=0,
               record_id="cal1", session="recording1")
    row.update(kw)
    return row


class PoisonCoordinates(dict):
    def __getitem__(self, key):
        if key in ("target_xy", "raw_xy", "target"):
            raise AssertionError("Excluded coordinate target was accessed")
        return super().__getitem__(key)


def test_fit_only_real_cal_external_o_and_early_self_exclusion():
    model = D.fit_external_o_error_model([
        cal_row(error=5.), cal_row(error=10., record_id="cal2", session="recording2"),
        PoisonCoordinates(split="CAL", state="S"),
        PoisonCoordinates(split="CAL", state="O", self_hidden=True),
        PoisonCoordinates(split="CAL", state="T"),
    ])
    assert model["sample_count"] == 2
    assert model["normalized_radius"] == pytest.approx(.1)
    assert model["excluded"]["not_actual_external_O_or_fixed_S"] == 3
    assert model["self_hidden_samples_used"] is False
    assert model["calibrated_joint_coverage"] is False
    assert model["confidence_guarantee"] is None


@pytest.mark.parametrize("change,match", [
    ({"split": "TEST"}, "Only CAL"), ({"split": "FINAL"}, "Only CAL"),
    ({"severity": "SEVERE"}, "nonsevere"),
    ({"external_label_confirmed": False}, "Actual HUMAN"),
    ({"label_source": "POSE_PSEUDOLABEL"}, "Actual HUMAN"),
    ({"target_xy": [-1., -1.]}, "real in-frame"),
])
def test_bad_calibration_provenance_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        D.fit_external_o_error_model([cal_row(**change)])


def test_scale_and_native_geometry_never_use_s():
    image, q, K, xyz = fixture()
    states = ["O", "V", "S", "S", "V", "V", "S", "S"]
    q[[2, 3, 6, 7]] = 1e12
    xyz[[2, 3, 6, 7]] = np.nan
    assert D.nonself_prediction_scale(q, states) == pytest.approx(np.hypot(20, 50))
    model = D.fit_external_o_error_model([cal_row()])
    output = D.build_ot_domains(image, q, states, K, xyz, model)
    assert [r["native_corner"] for r in output["domains"]] == [0]
    assert output["excluded_S_ids"] == [2, 3, 6, 7]
    assert output["S_native_geometry_queried"] is False
    assert output["S_image_features_queried"] is False


def test_known_s_mask_overrides_human_o_before_geometry():
    image, q, K, xyz = fixture()
    xyz[0] = np.nan
    output = D.build_ot_domains(image, q, ["O"]+["V"]*7, K, xyz,
                                self_hidden=[True]+[False]*7)
    assert output["domains"] == []
    assert output["effective_states"][0] == "S"


def test_final_boundary_interior_states_are_direct_visible_aliases():
    assert D.canonical_states(["B", "I", "O", "T", "S", "U", "B", "I"]) == ["V", "V", "O", "T", "S", "U", "V", "V"]


def test_o_center_is_raw_image_query_not_pose_projection():
    image, q, K, xyz = fixture()
    model = D.fit_external_o_error_model([cal_row(error=10.)])
    output = D.build_ot_domains(image, q, ["O"]+["V"]*7, K, xyz, model)
    region = output["domains"][0]
    assert region["center_xy"] == q[0].tolist()
    assert region["half_axes_px"][0] == pytest.approx(max(8., np.hypot(80, 50)*.1))
    assert region["independent_precise_support"] is False
    assert output["precise_image_anchor_count"] == 0


def test_missing_o_and_t_stay_broad_without_invented_location():
    image, q, K, xyz = fixture()
    q[0] = [-1., -1.]
    q[1] = [np.nan, np.nan]
    model = D.fit_external_o_error_model([cal_row()])
    output = D.build_ot_domains(image, q, ["O", "T"]+["V"]*6, K, xyz, model)
    o, t = output["domains"]
    assert o["bounds_xyxy"] == [0., 0., 100., 80.]
    assert o["localization_available"] is False
    assert t["outside_canvas_side"] == "unknown"
    assert t["halfplanes"] == []
    assert "center_xy" not in t
    assert t["coordinate_predictor_available"] is False
    assert t["uses_O_model"] is False


@pytest.mark.parametrize("changed", ["image", "query", "K", "xyz", "state", "record", "domain"])
def test_bindings_reject_changed_current_inputs_and_domain_geometry(changed):
    image, q, K, xyz = fixture()
    states = ["O"]+["V"]*7
    output = D.build_ot_domains(image, q, states, K, xyz, D.fit_external_o_error_model([cal_row()]))
    record_id = "current"
    if changed == "image": image[0, 0] ^= 1
    if changed == "query": q[0, 0] += 1
    if changed == "K": K[0, 0] += 1
    if changed == "xyz": xyz[0, 0] += .1
    if changed == "state": states[0] = "S"
    if changed == "record": record_id = "another"
    if changed == "domain": output["domains"][0]["center_xy"][0] += 1
    with pytest.raises(ValueError, match="binding changed"):
        D.validate_domain_bindings(output, image, q, states, K, xyz, record_id=record_id)


def test_model_binding_rejects_tampering():
    image, q, K, xyz = fixture()
    model = D.fit_external_o_error_model([cal_row()])
    model["normalized_radius"] = .0001
    with pytest.raises(ValueError, match="model binding changed"):
        D.build_ot_domains(image, q, ["O"]+["V"]*7, K, xyz, model)


def test_actual_crop_pixels_queries_camera_and_validity_polygon():
    image, q, K, xyz = fixture()
    cropped = D.apply_real_crop(image, K, q, [20, 5, 80, 70], source_inframe_native_ids=[0, 3])
    assert np.array_equal(cropped["image_bgr"], image[5:70, 20:80])
    assert np.allclose(cropped["raw_points"], q-[20, 5])
    assert np.allclose(cropped["K"], [[120., 0., 30.], [0., 120., 35.], [0., 0., 1.]])
    # Project then translate equals projection with transformed K.
    camera_point = np.array([.1, .2, 2.])
    before = K @ camera_point; after = cropped["K"] @ camera_point
    assert np.allclose(after[:2]/after[2], before[:2]/before[2]-[20, 5])
    states = ["T", "V", "V", "T"]+["S"]*4
    output = D.build_ot_domains(cropped["image_bgr"], cropped["raw_points"], states,
                                cropped["K"], xyz, truncation_sides={0: "left", 3: "right"},
                                crop_transform=cropped["transform"])
    left, right = output["domains"]
    assert left["halfplanes"][0] == [1., 0., 0.]
    assert right["halfplanes"][0] == [-1., 0., -60.]
    assert left["source_crop_validity_intersection"] is True
    assert left["localization_available"] is False
    assert left["outside_margin_px"] == 0.
    assert len(left["halfplanes"]) == 5
    assert cropped["transform"]["dimensions_changed"] is False


def test_crop_binding_rejects_wrong_camera_even_when_pixels_match():
    image, q, K, xyz = fixture()
    cropped = D.apply_real_crop(image, K, q, [20, 0, 80, 80], source_inframe_native_ids=[0])
    with pytest.raises(ValueError, match="current K"):
        D.build_ot_domains(cropped["image_bgr"], cropped["raw_points"], ["T"]+["V"]*7,
                           K, xyz, truncation_sides={0: "left"}, crop_transform=cropped["transform"])


def test_crop_binding_rejects_changed_rectangle_or_source_intrinsics():
    image, q, K, xyz = fixture()
    cropped = D.apply_real_crop(image, K, q, [20, 0, 80, 80], source_inframe_native_ids=[0])
    for mutation in ("rectangle", "sourceK"):
        changed = deepcopy(cropped["transform"])
        if mutation == "rectangle": changed["crop_xyxy"] = [21, 0, 81, 80]
        else: changed["source_K_sha256"] = "changed"
        with pytest.raises(ValueError, match="transform|Source K"):
            D.build_ot_domains(cropped["image_bgr"], cropped["raw_points"], ["T"]+["V"]*7,
                               cropped["K"], xyz, truncation_sides={0: "left"}, crop_transform=changed)


def test_crop_camera_binding_preserves_fractional_principal_points():
    image, q, K, xyz = fixture()
    K[0, 2], K[1, 2] = .1234567890123, .7654321098765
    cropped = D.apply_real_crop(image, K, q, [20, 5, 80, 70], source_inframe_native_ids=[0])
    output = D.build_ot_domains(cropped["image_bgr"], cropped["raw_points"], ["T"]+["V"]*7,
                                cropped["K"], xyz, truncation_sides={0: "left"}, crop_transform=cropped["transform"])
    assert output["domains"][0]["source_crop_validity_intersection"] is True


def test_inventory_reads_only_human_state_metadata_and_keeps_s_distinct():
    records = {
        "cal": PoisonCoordinates(metadata=dict(partition="CAL", session="night",
                          human_states_native=["EXTERNAL_OCCLUDED"]*2+["SELF_OCCLUDED"]*2+["DIRECT_VISIBLE"]*4)),
        "test": PoisonCoordinates(metadata=dict(partition="TEST", session="day",
                          human_states_native=["EXTERNAL_OCCLUDED"]+["DIRECT_VISIBLE"]*7)),
    }
    result = D.inventory_human_state_metadata(records)
    assert result["joint_external_O_records"] == {"CAL": ["cal"], "TEST": []}
    assert result["state_counts"]["O"] == 3
    assert result["state_counts"]["S"] == 2
    assert result["coordinate_targets_accessed"] is False
