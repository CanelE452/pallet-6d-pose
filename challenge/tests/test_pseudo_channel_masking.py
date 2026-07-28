import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Deep_Object_Pose" / "common"))

from heatmap_refinement import (  # noqa: E402
    channel_masked_mse,
    pseudo_label_channel_masks,
)
from utils_dataset import CleanVisiiDopeLoader  # noqa: E402


def test_corner_and_centroid_validity_drive_channel_masks():
    valid = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1, 1])
    belief, affinity = pseudo_label_channel_masks(valid)

    torch.testing.assert_close(belief, valid.float())
    expected_affinity = torch.ones(16)
    expected_affinity[2:4] = 0.0
    torch.testing.assert_close(affinity, expected_affinity)

    no_centroid = valid.clone()
    no_centroid[8] = 0
    belief, affinity = pseudo_label_channel_masks(no_centroid)
    assert belief[:8].sum().item() == 7.0
    assert belief[8].item() == 0.0
    assert affinity.sum().item() == 0.0


def test_all_ones_mask_is_identical_to_legacy_mse():
    generator = torch.Generator().manual_seed(7)
    for channels in (9, 16):
        prediction = torch.randn(2, channels, 5, 4, generator=generator,
                                 dtype=torch.float64, requires_grad=True)
        target = torch.randn(2, channels, 5, 4, generator=generator,
                             dtype=torch.float64)
        legacy = (prediction - target).square().mean()
        masked = channel_masked_mse(
            prediction, target, torch.ones(2, channels))

        torch.testing.assert_close(masked, legacy, rtol=0.0, atol=0.0)
        legacy_grad, = torch.autograd.grad(
            legacy, prediction, retain_graph=True)
        masked_grad, = torch.autograd.grad(masked, prediction)
        torch.testing.assert_close(
            masked_grad, legacy_grad, rtol=0.0, atol=0.0)


def test_invalid_channel_contributes_no_loss_or_gradient():
    prediction = torch.zeros(1, 3, 2, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, 0] = 1.0
    target[:, 1] = 1000.0
    mask = torch.tensor([[1.0, 0.0, 0.0]])

    loss = channel_masked_mse(prediction, target, mask)
    assert loss.item() == 1.0
    loss.backward()
    assert prediction.grad[:, 0].abs().sum().item() > 0.0
    assert prediction.grad[:, 1:].abs().sum().item() == 0.0


def _write_dataset_pair(root, validity=None):
    image = np.full((400, 400, 3), 127, dtype=np.uint8)
    Image.fromarray(image).save(root / "sample.png")
    corners = [
        [100, 100], [300, 100], [300, 220], [100, 220],
        [130, 180], [270, 180], [270, 300], [130, 300],
    ]
    annotation = {
        "objects": [{
            "class": "pallet",
            "visibility": 1,
            "projected_cuboid": corners,
            "projected_cuboid_centroid": [200, 200],
        }],
    }
    if validity is not None:
        annotation["pseudo_keypoint_valid"] = validity
    (root / "sample.json").write_text(json.dumps(annotation))


def test_dataset_always_returns_belief_and_affinity_masks(tmp_path):
    _write_dataset_pair(tmp_path)
    loader = CleanVisiiDopeLoader(
        [str(tmp_path)], objects=["pallet"], sigma=2, output_size=50,
        aspect_resize=True)
    item = loader[0]
    torch.testing.assert_close(item["belief_channel_mask"], torch.ones(9))
    torch.testing.assert_close(item["affinity_channel_mask"], torch.ones(16))

    _write_dataset_pair(tmp_path, [1, 0, 1, 1, 1, 1, 1, 1, 1])
    item = loader[0]
    assert item["belief_channel_mask"][1].item() == 0.0
    assert item["affinity_channel_mask"][2:4].sum().item() == 0.0
