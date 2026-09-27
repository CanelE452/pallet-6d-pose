"""CPU-only paired export invariants; no evaluation references or training."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from . import train_pair as W
from scripts.research.pallet_type_selftrain_v1.test_recovery_pose import (
    test_only_pose_parameters_allowed, test_matched_target_masks_and_boxes,
)
from scripts.research.pallet_type_selftrain_v1.test_contract import (
    test_pseudo_label_padding_and_no_mutation, test_uncertain_keypoint_true_ignore,
    test_no_threshold_rescue, test_degenerate_box_does_not_make_training_example,
)


def example():
    c = dict(box_xyxy=[20, 10, 150, 95], keypoints_xy=[[60, 50]]*9, keypoints_conf=[.9]*9)
    d = copy.deepcopy(c); d["keypoints_xy"] = [[65, 48]]*8 + [[60, 50]]
    return dict(raw_hw=[100, 200], raw=dict(selected_index=0, candidates=[c]), refined=dict(selected_index=0, candidates=[d]))


def test_existing_args_exact_lr5_only_override():
    args = dict(W.T.ARGS, lr0=1e-5)
    method = dict(args=args, updates=320, epochs=5, batch=16, nbs=16, seed=42, optimizer="AdamW", lr=1e-5)
    assert W.locked_args(method) == args
    method["args"] = dict(args, lr0=1e-4)
    with pytest.raises(AssertionError): W.locked_args(method)


def test_replacement_is_deterministic_same_membership_order():
    names = [f"WOOD__{i:03d}.png" for i in range(200)]
    first = W.paired_real_slots(names)
    assert len(first) == 512 and first == W.paired_real_slots(names[::-1])
    assert first == np.random.default_rng(9021).choice(sorted(names), 512, replace=True).tolist()


def test_safe_ids_prevent_namespace_and_directory_escape():
    assert W.safe_id("WOOD__abcd") == "WOOD__abcd"
    for value in ["", "..", "../x", "/tmp/x", "a/b", "a\\b", "syn__image"]:
        with pytest.raises(AssertionError): W.safe_id(value)


def test_support_intersection_and_preserved_center():
    row = example(); row["refined"]["candidates"][0]["keypoints_xy"][0] = [-3, 50]
    labels, count = W.paired_labels(row)
    assert count == 8
    raw, ref = [np.asarray(labels[a].split(), float) for a in ("RAW", "REF")]
    np.testing.assert_array_equal(raw[:5], ref[:5])
    a, b = raw[5:].reshape(9, 3), ref[5:].reshape(9, 3)
    np.testing.assert_array_equal(a[:, 2], b[:, 2])
    np.testing.assert_array_equal(a[0], [.5, .5, 1])
    np.testing.assert_array_equal(a[8], b[8])
    assert not np.array_equal(a[1, :2], b[1, :2])


def test_changed_bbox_rejected():
    row = example(); row["refined"]["candidates"][0]["box_xyxy"][0] += 1
    with pytest.raises(AssertionError): W.paired_labels(row)


def test_empty_and_multiinstance_source_parsing():
    assert W.label_parts("") == ([], [], [])
    labels, _ = W.paired_labels(example())
    boxes, support, xy = W.label_parts(labels["RAW"]*2)
    assert len(boxes) == len(support) == len(xy) == 2
    assert len(support[0]) == 9


def test_pair_hash_gate_rejects_noncoordinate_difference():
    left = dict(rgb_order="a", boxes="b", support="s", source_replay="z", slots=1024, synthetic_slots=512, coordinates="raw")
    right = dict(left, coordinates="ref")
    W.assert_pair(left, right)
    for key in ["rgb_order", "boxes", "support", "source_replay", "slots", "synthetic_slots"]:
        bad = dict(right); bad[key] = "different"
        with pytest.raises(AssertionError): W.assert_pair(left, bad)
    with pytest.raises(AssertionError): W.assert_pair(left, left)


def test_actual_cpu_true_ignore_gradient_no_eval_fixture(monkeypatch):
    """Reuse old measurement function but prohibit its evaluation-image loader."""
    from scripts.self_training_yolo.v3 import verify_true_ignore_contract as V
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(981)
    image = torch.rand((1, 3, V.IMGSZ, V.IMGSZ), generator=generator)
    points = torch.tensor([[.35,.35],[.65,.35],[.65,.55],[.35,.55],[.4,.4],[.6,.4],[.6,.6],[.4,.6],[.5,.45]])
    monkeypatch.setattr(V, "_SAMPLE", (image, points, torch.tensor([[.5,.5,.6,.6]])))
    monkeypatch.setattr(V, "real_sample", lambda: (_ for _ in ()).throw(AssertionError("Evaluation fixture forbidden")))
    ignored = V.measure([1.]*9)
    assert ignored["keypoint_branch_grad"] == 0
    for name in ("kpt_location", "kpt_visibility", "rle"): assert ignored["items"][name] == 0
    base = V.measure([2.]*4+[1.]*5)
    shifted = V.measure([2.]*4+[1.]*5, shift=(8, 60.))
    assert V.close(base["total_loss"], shifted["total_loss"])
    assert V.close(base["keypoint_branch_grad"], shifted["keypoint_branch_grad"])
    stock = V.measure([2.]*4+[0.]*5, true_ignore=False)
    custom = V.measure([2.]*4+[0.]*5)
    assert V.close(stock["total_loss"], custom["total_loss"])
    assert V.close(stock["keypoint_branch_grad"], custom["keypoint_branch_grad"])
