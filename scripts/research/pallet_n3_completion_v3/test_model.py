from __future__ import annotations

import inspect
import unittest

import torch

from scripts.research.pallet_dope_refiner_20261001_v1.refiner import GenericPointRefiner
from scripts.research.pallet_n3_completion_v3.model import (
    BACKBONE_CONFIGS,
    LOCKED_STENCIL_FRACTION,
    DimensionSymmetryN3,
    build_n3,
    dimension_features,
    n3_loss,
    normalize_dimensions,
    select_symmetric_target,
    trainable_parameter_count,
)


def fixture(backbone="dope", batch=2):
    config = BACKBONE_CONFIGS[backbone]
    height, width = 64, 80
    p3 = torch.randn(batch, config["c3"], height // config["stride3"],
                     width // config["stride3"])
    p4 = torch.randn(batch, config["c4"], height // config["stride4"],
                     width // config["stride4"])
    points = torch.tensor([
        [[20., 20.], [50., 20.], [50., 40.], [20., 40.],
         [23., 18.], [47., 18.], [47., 42.], [23., 42.], [35., 30.]],
    ]).repeat(batch, 1, 1)
    boxes = torch.tensor([[15., 15., 55., 45.]]).repeat(batch, 1)
    valid = torch.ones(batch, 9, dtype=torch.bool)
    shape = torch.tensor([[height, width]]).repeat(batch, 1)
    context = torch.tensor([
        [-0.2, 0.3, -0.1, -0.5, 0.1],
        [0.4, -0.2, 0.2, 0.6, -0.3],
    ])[:batch]
    return dict(p3=p3, p4=p4, points=points, boxes=boxes,
                point_valid=valid, input_shape=shape,
                dimension_context=context)


class DimensionContractTests(unittest.TestCase):
    def test_features_and_normalization(self):
        dimensions = torch.tensor([[2., 1., .5]])
        features = dimension_features(dimensions)
        expected = torch.tensor([[
            torch.log(torch.tensor(2.)), 0., torch.log(torch.tensor(.5)),
            torch.log(torch.tensor(2.)), torch.log(torch.tensor(.5 / 2**.5)),
        ]])
        torch.testing.assert_close(features, expected)
        normalization = {"mean": features[0].tolist(), "scale": [2.] * 5}
        torch.testing.assert_close(
            normalize_dimensions(dimensions, normalization), torch.zeros_like(features))
        with self.assertRaises(ValueError):
            dimension_features(torch.tensor([[1., 0., 1.]]))

    def test_n3_rejects_n4_context_and_protocol_drift(self):
        model = build_n3("dope")
        values = fixture()
        values["dimension_context"] = torch.zeros(2, 8)
        with self.assertRaises(ValueError):
            model(**values, lam=0.)
        with self.assertRaises(ValueError):
            DimensionSymmetryN3(c3=256, c4=128, stride3=4, stride4=8,
                                stencil_fraction=.25)
        self.assertEqual(model.stencil_fraction, LOCKED_STENCIL_FRACTION)


class ArchitectureParityTests(unittest.TestCase):
    def test_parameter_counts_are_backbone_specific_not_yolo_count(self):
        self.assertEqual(trainable_parameter_count("dope"), 23331)
        self.assertEqual(trainable_parameter_count("resnet18"), 23331)
        self.assertNotEqual(trainable_parameter_count("dope"), 20259)

    def test_zero_metadata_final_layer_preserves_visual_logits(self):
        for backbone, config in BACKBONE_CONFIGS.items():
            with self.subTest(backbone=backbone):
                torch.manual_seed(7)
                visual = GenericPointRefiner(
                    **config, hidden=16, encoded=24,
                    stencil_fraction=LOCKED_STENCIL_FRACTION)
                torch.manual_seed(7)
                n3 = build_n3(backbone)
                visual_state = visual.state_dict()
                n3_state = n3.state_dict()
                for key, value in visual_state.items():
                    torch.testing.assert_close(value, n3_state[key], rtol=0, atol=0)
                values = fixture(backbone)
                base = visual(*(values[key] for key in (
                    "p3", "p4", "points", "boxes", "point_valid", "input_shape")),
                    lam=0.)
                output = n3(**values, lam=0.)
                torch.testing.assert_close(output["base_logits"], base["logits"], rtol=0, atol=0)
                torch.testing.assert_close(output["logits"], base["logits"], rtol=0, atol=0)
                self.assertEqual(float(output["metadata_residual"].abs().max()), 0.)

    def test_metadata_gradient_connects_after_zero_initialized_step(self):
        torch.manual_seed(11)
        model = build_n3("resnet18")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        values = fixture("resnet18")
        output = model(**values, lam=0.)
        first = output["logits"].square().mean()
        first.backward()
        self.assertGreater(float(model.metadata_scorer[-1].weight.grad.norm()), 0.)
        self.assertEqual(float(model.metadata_encoder[0].weight.grad.norm()), 0.)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        output = model(**values, lam=0.)
        output["logits"].square().mean().backward()
        self.assertGreater(float(model.metadata_encoder[0].weight.grad.norm()), 0.)
        self.assertGreater(float(model.adapt3[0].weight.grad.norm()), 0.)
        self.assertGreater(float(model.adapt4[0].weight.grad.norm()), 0.)

    def test_lambda_zero_preserves_all_nine_initial_points(self):
        model = build_n3("dope")
        values = fixture()
        output = model(**values, lam=0.)
        torch.testing.assert_close(output["points"], values["points"], rtol=0, atol=0)


class SymmetryTargetTests(unittest.TestCase):
    @staticmethod
    def values():
        gt = torch.tensor([[
            [0., 0.], [10., 0.], [10., 10.], [0., 10.],
            [2., 2.], [8., 2.], [8., 8.], [2., 8.], [5., 5.],
        ]])
        permutation = torch.tensor([[
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 0, 3, 2, 5, 4, 7, 6, 8],
        ]])
        initial = gt[:, permutation[0, 1]].clone()
        valid = torch.ones(1, 9, dtype=torch.bool)
        return initial, valid, gt, valid.clone(), permutation, torch.ones(1, 2, dtype=torch.bool)

    def test_selects_one_whole_object_branch_and_permutes_mask(self):
        initial, predicted_valid, gt, gt_valid, permutations, group_valid = self.values()
        gt_valid[0, 0] = False
        selected, valid, branch, cost = select_symmetric_target(
            initial, predicted_valid, gt, gt_valid, permutations, group_valid,
            torch.tensor([20.]))
        self.assertEqual(branch.tolist(), [1])
        torch.testing.assert_close(selected[0], gt[0, permutations[0, 1]])
        self.assertFalse(bool(valid[0, 1]))
        self.assertTrue(bool(valid[0, 0]))
        torch.testing.assert_close(selected[0, 8], gt[0, 8])
        self.assertTrue(torch.isfinite(cost[0, 0]))

    def test_missing_prediction_penalty_and_identity_first_tie(self):
        initial, predicted_valid, gt, gt_valid, permutations, group_valid = self.values()
        initial = gt.clone()
        predicted_valid[:] = False
        _, _, branch, cost = select_symmetric_target(
            initial, predicted_valid, gt, gt_valid, permutations, group_valid,
            torch.tensor([20.]))
        self.assertEqual(branch.tolist(), [0])
        torch.testing.assert_close(cost, torch.ones_like(cost))

    def test_invalid_center_permutation_is_rejected(self):
        values = list(self.values())
        values[4][0, 1, 8] = 7
        with self.assertRaises(ValueError):
            select_symmetric_target(*values, torch.tensor([20.]))

    def test_loss_uses_raw_initial_points_and_forward_has_no_gt_inputs(self):
        model = build_n3("resnet18")
        values = fixture("resnet18")
        output = model(**values, lam=0.)
        permutations = torch.arange(9)[None, None].repeat(2, 2, 1)
        permutations[:, 1] = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 8])
        batch = dict(
            gt_points=values["points"].clone(),
            gt_valid=values["point_valid"].clone(),
            permutations=permutations,
            group_valid=torch.tensor([[True, False], [True, False]]),
        )
        value = n3_loss(output, batch)
        self.assertTrue(torch.isfinite(value))
        signature = inspect.signature(model.forward)
        for forbidden in ("gt_points", "gt_valid", "permutations", "group_valid"):
            self.assertNotIn(forbidden, signature.parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)
