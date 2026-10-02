"""CPU contract tests for production stages; no checkpoint forward or writes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import common as C
from .evaluation import (METHODS, dimension_support, object_subgroups, pose_summary,
                         validate_canonical_wdh, validate_rows, write_frame_csv)
from .full_adapter import decode_logits, input_module, spatial_coordinates
from . import inference
from .inference import canonical_dimensions
from .protocol import (assert_preseal_output_absence, missing_production_files,
                       validate_full_completion)
from .selection import (displacement_bank, refined_delta, validate_selection_payload,
                        write_or_verify_npz)
from .source_cache import (_cache_manifest_payload, _initialize_cache,
                           _validate_cache_file_set, _validate_cache_manifest,
                           box_iou, source_target)
from .data import CACHE_SPECS
from .dimensions import load_normalization, normalized_context
from .train import build_heads, learning_rate


torch.set_num_threads(1)


def test_full_completion_and_all_production_stages_present():
    assert missing_production_files() == []
    completion = json.loads(C.FULL_COMPLETE.read_text())
    checkpoint = Path(completion["final_checkpoint"]["path"])
    if not checkpoint.is_absolute():
        checkpoint = C.ROOT / checkpoint
    if not checkpoint.is_file():
        pytest.skip("private FULL checkpoint is intentionally absent from the public bundle")
    baseline = validate_full_completion()
    assert baseline["checkpoint"]["sha256"] == (
        "0a6b664f126ef72449ed917917177b3c1c03306fdd1a4bd5aa827228b98633ac")


def test_softargmax_decoder_uses_bound_grid_and_affine():
    zero = torch.zeros(1, 9, 96, 128)
    coordinates, log_probability, probability = spatial_coordinates(zero)
    torch.testing.assert_close(coordinates, torch.zeros_like(coordinates), atol=2e-7, rtol=0)
    torch.testing.assert_close(probability.sum(-1), torch.ones(1, 9), atol=1e-6, rtol=0)
    torch.testing.assert_close(log_probability.exp(), probability, atol=0, rtol=0)
    image = np.zeros((101, 157, 3), np.uint8)
    prepared = input_module().prepare(image, source_pre_padded=False)
    logits = torch.full((1, 9, 96, 128), -20.)
    cells = [(10, 10), (10, 110), (80, 110), (80, 10),
             (25, 25), (25, 95), (65, 95), (65, 25), (45, 60)]
    for channel, (y, x) in enumerate(cells):
        logits[0, channel, y, x] = 20.
    result = decode_logits(logits, [prepared])[0]
    assert result["points_net"].shape == (9, 2)
    assert result["points_net"].dtype == np.float32
    assert result["points_original"].dtype == np.float64
    assert result["valid"].all() and result["bbox_net"].shape == (4,)
    roundtrip = input_module().transform_points(
        result["points_original"], prepared["affine_input_to_net"])
    np.testing.assert_allclose(roundtrip, result["points_net"], atol=1e-5, rtol=0)


def test_source_target_and_box_iou_contract():
    image = np.zeros((300, 400, 3), np.uint8)
    prepared = input_module().prepare(image, source_pre_padded=True)
    keypoints = [[.2 + .05 * (index % 4), .3 + .06 * (index // 4), 2]
                 for index in range(9)]
    record = dict(
        id="fixture",
        prepared_shape_hw=[300, 400],
        targets=[dict(keypoints_normalized=keypoints,
                      box_xywh_normalized=[.5, .5, .4, .3])],
    )
    points, valid, box = source_target(record, prepared)
    assert points.shape == (9, 2) and valid.all()
    assert box.shape == (4,) and (box[2:] > box[:2]).all()
    assert box_iou(box, box) == 1.
    shifted = box + np.asarray([1000, 0, 1000, 0])
    assert box_iou(box, shifted) == 0.


def test_paired_head_initialization_and_fixed_schedule_cpu():
    optimizer = dict(lr=.001, betas=[.9, .999], weight_decay=.0001,
                     warmup_steps=100, cosine_final_lr_fraction=.1,
                     gradient_clip_norm=5.)
    models, optimizers = build_heads(1, "cpu", optimizer)
    assert set(models) == set(optimizers) == set(C.HEAD_ARMS)
    p0 = models["P0"].state_dict()
    for arm in ("P5", "P5_CONSTANT"):
        assert all(torch.equal(value, models[arm].state_dict()[key])
                   for key, value in p0.items())
    assert all(torch.equal(value, models["P5_CONSTANT"].state_dict()[key])
               for key, value in models["P5"].state_dict().items())
    assert learning_rate(1, optimizer) == .00001
    assert learning_rate(100, optimizer) == .001
    assert np.isclose(learning_rate(6000, optimizer), .0001)


def test_selection_math_has_explicit_arm_semantics_and_xy_inverse():
    assert displacement_bank().shape == (222, 2)
    rows = 2
    arrays = dict(
        boxes=np.asarray([[0, 0, 100, 80], [0, 0, 60, 50]], np.float32),
        scale_xy=np.asarray([[2., 4.], [1., 1.]], np.float64),
        raw_diagonal=np.asarray([200., 100.], np.float64),
    )
    support = np.ones((rows, 8), bool)
    logits = np.zeros((rows, 8, 222), np.float32)
    visual = refined_delta(dict(value=logits, support=support), "P0", 1., arrays, 1., None)
    for arm in ("P5", "P5_CONSTANT"):
        actual = refined_delta(dict(value=logits, support=support), arm, 1., arrays, 1., None)
        np.testing.assert_array_equal(actual, visual)
    direct_value = np.ones((rows, 8, 2), np.float32) * .1
    direct = refined_delta(dict(value=direct_value, support=support), "D0", 1., arrays, 1., None)
    diagonal = np.linalg.norm(arrays["boxes"][:, 2:] - arrays["boxes"][:, :2], axis=-1)
    np.testing.assert_allclose(direct[:, 0], .1 * diagonal[:, None] / arrays["scale_xy"],
                               atol=1e-6, rtol=1e-7)
    capped = refined_delta(dict(value=direct_value, support=support), "D0", 1., arrays, 1., .01)
    assert np.all(np.linalg.norm(capped, axis=-1) <= arrays["raw_diagonal"][:, None] * .01 + 1e-9)


def test_registry_dimensions_and_evaluation_preservation_contract():
    np.testing.assert_array_equal(
        canonical_dimensions("plastic_standard_110x130x11"), [1.1, 1.3, .11])
    points = np.asarray([[10, 10], [90, 10], [90, 70], [10, 70],
                         [20, 20], [80, 20], [80, 60], [20, 60], [50, 40]], np.float64)
    refined = {method: points.copy().tolist() for method in METHODS[1:]}
    row = dict(
        image_key="data/fixture.png", frame_id="fixture", session_id="session",
        original_hw=[80, 100], base_points=points.tolist(),
        point_valid=np.ones(9, bool).tolist(), box_original=[10., 10., 90., 70.],
        score=1., refined=refined,
    )
    validate_rows([row])
    broken = copy.deepcopy(row)
    broken["refined"][METHODS[1]][8][0] += 1
    try:
        validate_rows([broken])
    except ValueError as error:
        assert "Center" in str(error)
    else:
        raise AssertionError("Evaluation accepted a changed center point")


def test_validation_npz_and_frame_csv_are_exactly_reusable_after_interruption(tmp_path):
    arrays = dict(rows=np.arange(3), support=np.ones((3, 8), bool),
                  value=np.arange(48, dtype=np.float32).reshape(3, 8, 2))
    destination = tmp_path / "validation_D0_S1.npz"
    write_or_verify_npz(destination, **arrays)
    first = destination.read_bytes()
    write_or_verify_npz(destination, **arrays)
    assert destination.read_bytes() == first
    pending_destination = tmp_path / "validation_D0_S2.npz"
    with Path(str(pending_destination) + ".pending").open("wb") as handle:
        np.savez(handle, **arrays)
    write_or_verify_npz(pending_destination, **arrays)
    assert pending_destination.is_file() and not Path(str(pending_destination) + ".pending").exists()
    changed = dict(arrays, value=arrays["value"] + 1)
    try:
        write_or_verify_npz(destination, **changed)
    except ValueError:
        pass
    else:
        raise AssertionError("Validation reuse accepted different array content")

    record = dict(frame_id="f", session_id="s", object_type="plastic_standard_110x130x11",
                  image_key="data/f.png", canonical_matched=True, box_iou=1., gt_count=9,
                  finite_predicted_keypoints=9, missing_supervised_keypoints=0,
                  canonical_hits={"5": 9, "10": 9, "20": 9}, point_errors=[0.] * 9)
    stores = {method: [copy.deepcopy(record)] for method in METHODS}
    poses = {method: {"rows": []} for method in METHODS}
    csv_path = tmp_path / "frames.csv"
    write_frame_csv([], stores, poses, csv_path)
    csv_bytes = csv_path.read_bytes()
    write_frame_csv([], stores, poses, csv_path)
    assert csv_path.read_bytes() == csv_bytes
    stores[METHODS[0]][0]["box_iou"] = .5
    try:
        write_frame_csv([], stores, poses, csv_path)
    except ValueError:
        pass
    else:
        raise AssertionError("Frame CSV reuse accepted changed content")


def test_existing_dev_prediction_is_reused_without_predictor(monkeypatch, tmp_path):
    doc, raw = tmp_path / "doc", tmp_path / "raw"
    raw.mkdir()
    payload = {"selection_inference_replay_max_abs_px": 0.0, "sentinel": "orphaned-complete"}
    destination = raw / "DEV_PREDICTIONS.json"
    destination.write_text(json.dumps(payload))
    protocol = {"sentinel": "sealed"}
    monkeypatch.setattr(inference.C, "DOC", doc)
    monkeypatch.setattr(inference.C, "RAW", raw)
    monkeypatch.setattr(inference.C, "verify_protocol", lambda: protocol)
    monkeypatch.setattr(inference, "validate_prediction_payload",
                        lambda value, protocol=None: value)
    class ForbiddenPredictor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Existing complete predictions must not rerun inference")
    monkeypatch.setattr(inference, "Predictor", ForbiddenPredictor)
    assert inference.dev("cpu")["sentinel"] == "orphaned-complete"
    receipt = json.loads((doc / "DEV_INFERENCE_COMPLETE.json").read_text())
    assert receipt["complete"] is True and receipt["frames"] == 319


def test_cache_rejects_missing_extra_directory_and_dual_pending(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "CACHE_MANIFEST.json").touch()
    for name in CACHE_SPECS:
        (cache / f"{name}.npy").touch()
    _validate_cache_file_set(cache)
    (cache / "points.npy").unlink()
    try:
        _validate_cache_file_set(cache)
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("Cache accepted a missing array")
    (cache / "points.npy").touch()
    (cache / "rogue").mkdir()
    try:
        _validate_cache_file_set(cache)
    except ValueError as error:
        assert "extra" in str(error)
    else:
        raise AssertionError("Cache accepted an unexpected directory")
    (cache / "rogue").rmdir()
    (tmp_path / "cache.pending").mkdir()
    monkeypatch.setattr(C, "RAW", tmp_path)
    try:
        _initialize_cache(1, {})
    except ValueError as error:
        assert "Both final and pending" in str(error)
    else:
        raise AssertionError("Cache accepted simultaneous final and pending trees")


def test_cache_manifest_rejects_schema_and_provenance_mutations(monkeypatch, tmp_path):
    paths = {name: tmp_path / name for name in ("protocol.json", "source.json",
                                                "dimensions.npz", "baseline.pt")}
    for path in paths.values():
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(C, "PROTOCOL", paths["protocol.json"])
    monkeypatch.setattr(C, "SOURCE_MANIFEST", paths["source.json"])
    monkeypatch.setattr(C, "DIMENSION_SIDECAR", paths["dimensions.npz"])
    protocol = {"baseline": C.binding(paths["baseline.pt"])}
    expected = _cache_manifest_payload(7, protocol)
    assert _validate_cache_manifest(expected, 7, protocol) == expected
    mutations = []
    changed = copy.deepcopy(expected); changed["rows"] = 8; mutations.append(changed)
    changed = copy.deepcopy(expected); changed["dimension_sidecar"]["sha256"] = "0" * 64; mutations.append(changed)
    changed = copy.deepcopy(expected); changed["arrays"]["points"]["dtype"] = "float64"; mutations.append(changed)
    changed = copy.deepcopy(expected); changed["full_spatial_feature_cache"] = True; mutations.append(changed)
    for changed in mutations:
        try:
            _validate_cache_manifest(changed, 7, protocol)
        except ValueError:
            pass
        else:
            raise AssertionError("Cache manifest accepted a schema/provenance mutation")


def test_fixed_axis_wdh_rejects_width_depth_swap():
    spec = {"long_m": 1.3, "short_m": 1.1, "height_m": .11}
    np.testing.assert_array_equal(
        validate_canonical_wdh("plastic_standard_110x130x11", [1.1, 1.3, .11], spec),
        [1.1, 1.3, .11])
    try:
        validate_canonical_wdh("plastic_standard_110x130x11", [1.3, 1.1, .11], spec)
    except ValueError as error:
        assert "fixed-axis" in str(error)
    else:
        raise AssertionError("Canonical W,D,H validation accepted a W/D swap")


def test_add_auc_is_per_frame_normalized_and_failures_reduce_full_population_auc():
    values = [
        dict(add_sym_m=.05, diameter_m=1., rotation_error_deg=1.,
             translation_error_cm=1., yaw_error_deg=1., iou3d=.8, axis_correct=True),
        dict(add_sym_m=.10, diameter_m=2., rotation_error_deg=2.,
             translation_error_cm=2., yaw_error_deg=2., iou3d=.6, axis_correct=False),
    ]
    summary = pose_summary(values, 4)
    assert summary["n"] == 2 and summary["failure_count"] == 2 and summary["coverage"] == .5
    assert np.isclose(summary["add_sym_auc_full_population"],
                      .5 * summary["add_sym_auc_conditional"], atol=1e-12)
    rescaled = copy.deepcopy(values)
    rescaled[0].update(add_sym_m=.5, diameter_m=10.)
    rescaled[1].update(add_sym_m=.025, diameter_m=.5)
    assert np.isclose(pose_summary(rescaled, 4)["add_sym_auc_conditional"],
                      summary["add_sym_auc_conditional"], atol=1e-12)


def test_object_subgroups_and_dimension_ood_disclosure(tmp_path, monkeypatch):
    plastic = dict(object_type="plastic_standard_110x130x11",
                   canonical_WDH_m=[1.1, 1.3, .11])
    wood = dict(object_type="wood_small_80x59x14", canonical_WDH_m=[.8, .59, .14])
    groups = object_subgroups([plastic] * 194 + [wood] * 125, enforce_dev_counts=True)
    assert {name: len(rows) for name, rows in groups.items()} == {"plastic": 194, "wood": 125}

    # Use a compact synthetic support envelope so this unit contract remains
    # runnable when private 60k-row manifests and sidecars are not published.
    basis = np.asarray([
        [.590, .818, .064],
        [1.363, 1.300, .110],
        [1.200, 1.720, .244],
        [.590, .818, .244],
    ], np.float64)
    dimensions = np.tile(basis, (13995, 1))
    normalization = load_normalization()
    source = SimpleNamespace(
        partitions=np.full(len(dimensions), "train"),
        dimensions=dimensions,
        normalization=normalization,
        contexts=normalized_context(dimensions, normalization),
    )
    source_manifest = tmp_path / "SOURCE_MANIFEST.json"
    dimension_sidecar = tmp_path / "DIMENSION_SIDECAR.npz"
    source_manifest.write_text("{}\n")
    dimension_sidecar.write_bytes(b"fixture")
    monkeypatch.setattr(C, "SOURCE_MANIFEST", source_manifest)
    monkeypatch.setattr(C, "DIMENSION_SIDECAR", dimension_sidecar)

    support = dimension_support([plastic, wood], source=source)
    assert support["objects"]["plastic"]["outside_train_axiswise_box"] is False
    assert support["objects"]["plastic"]["outside_train_normalized_axiswise_box"] is False
    assert support["objects"]["wood"]["outside_train_axiswise_box"] is True
    assert support["objects"]["wood"]["outside_raw_axes"] == ["D"]
    assert set(support["objects"]["wood"]["outside_normalized_features"]) == {
        "logD", "log(W/D)"}


def test_preseal_guard_rejects_any_existing_output(tmp_path):
    doc, raw, protocol = tmp_path / "doc", tmp_path / "raw", tmp_path / "PROTOCOL.json"
    assert_preseal_output_absence(doc, raw, protocol)
    output = raw / "nested" / "partial.pending"
    output.parent.mkdir(parents=True)
    output.write_text("partial")
    try:
        assert_preseal_output_absence(doc, raw, protocol)
    except RuntimeError as error:
        assert "before protocol seal" in str(error)
    else:
        raise AssertionError("Pre-seal guard accepted an existing pending output")
    output.unlink()
    doc.mkdir()
    (doc / "orphan.json").write_text("{}")
    try:
        assert_preseal_output_absence(doc, raw, protocol)
    except RuntimeError as error:
        assert "before protocol seal" in str(error)
    else:
        raise AssertionError("Pre-seal guard accepted an existing DOC output")
    protocol.write_text("{}")
    assert_preseal_output_absence(doc, raw, protocol)


def test_selection_receipt_requires_frozen_grids_and_argmins(monkeypatch):
    fake_protocol_binding = {"path": "PROTOCOL.json", "sha256": "a" * 64, "bytes": 1}
    validation_binding = {"path": "VALIDATION_OUTPUTS.json", "sha256": "b" * 64, "bytes": 1}
    monkeypatch.setattr(C, "binding", lambda path: fake_protocol_binding)
    protocol = dict(
        calibration={"temperature_grid": [.5, 1., 2., 4.]},
        selection={"lambda_grid": [0., .0625, .25, 1., 4.],
                   "max_move_image_diagonal_fractions": [None, .01]},
    )
    temperatures, calibration = {}, {}
    for arm in C.HEAD_ARMS:
        for seed in C.SEEDS:
            name = C.arm_key(arm, seed)
            if arm == "D0":
                temperatures[name] = 1.
                calibration[name] = dict(
                    selected=dict(temperature=1., score=None), candidates=[],
                    supported_frames=None)
            else:
                candidates = [dict(temperature=value, score=float(index + 1))
                              for index, value in enumerate([.5, 1., 2., 4.])]
                temperatures[name] = .5
                calibration[name] = dict(selected=candidates[0], candidates=candidates,
                                         supported_frames=100)
    grids, rules = {}, {}
    for arm in C.HEAD_ARMS:
        candidates = []
        index = 0
        for lam in protocol["selection"]["lambda_grid"]:
            for cap in protocol["selection"]["max_move_image_diagonal_fractions"]:
                score = float(index + 1); index += 1
                candidates.append(dict(lam=lam, max_move_image_diagonal_fraction=cap,
                                       score=score,
                                       seeds={str(seed): {"score": score} for seed in C.SEEDS}))
        grids[arm] = candidates
        rules[arm] = candidates[0]
    selection = dict(complete=True, no_real_selection=True, heldout_accuracy_opened=False,
                     protocol=fake_protocol_binding, validation=validation_binding,
                     temperatures=temperatures, calibration=calibration,
                     rules=rules, candidates=grids)
    assert validate_selection_payload(selection, protocol=protocol,
                                      validation_binding=validation_binding) is selection
    broken = copy.deepcopy(selection)
    broken["rules"]["P5"] = broken["candidates"]["P5"][-1]
    try:
        validate_selection_payload(broken, protocol=protocol,
                                   validation_binding=validation_binding)
    except ValueError as error:
        assert "argmin" in str(error)
    else:
        raise AssertionError("Selection validation accepted a non-argmin rule")
