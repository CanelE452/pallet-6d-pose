"""CPU generated-fixture tests; no source images, model training or GT selection."""
from __future__ import annotations

import copy

import pytest
import torch

from scripts.research.pallet_dht_structured_v2.model import ARMS, EDGES, INPUT_KEYS, LayoutVerifier, raw_to_input, sample_p4

torch.set_num_threads(1)


def fixture(batch=2, layouts=3):
    generator = torch.Generator().manual_seed(123)
    p = torch.tensor([[100., 220.], [420., 220.], [420., 270.], [100., 270.],
                      [180., 100.], [450., 100.], [450., 135.], [180., 135.], [285., 184.]])
    baseline = p[None].repeat(batch, 1, 1)
    q = baseline[:, None].repeat(1, layouts, 1, 1)
    for k in range(layouts):
        q[:, k, :8] += torch.tensor([k*.8, k*(-.3)])
    h = []
    for u, v in EDGES:
        a, b = p[u], p[v]
        line = torch.stack((a[1]-b[1], b[0]-a[0], a[0]*b[1]-a[1]*b[0]))
        line = line/torch.linalg.vector_norm(line[:2])
        modes = line[None].repeat(4, 1)
        modes[:, 2] += torch.tensor([-8., 0., 8., 16.])
        h.append(modes)
    return dict(p4=torch.randn(batch, 128, 40, 40, generator=generator),
        baseline_points=baseline, layouts=q, point_conf=torch.full((batch, 9), .8),
        diagonal=torch.full((batch,), 640.*2**.5),
        raw_to_input_affine=torch.tensor([[1., 0., 0.], [0., 1., 0.]])[None].repeat(batch, 1, 1),
        input_shape_hw=torch.tensor([[640, 640]]).repeat(batch, 1),
        line_h=torch.stack(h)[None].repeat(batch, 1, 1, 1),
        peak_logits=torch.tensor([-2., 0., 1., 3.])[None, None].repeat(batch, 12, 1),
        peak_valid=torch.ones(batch, 12, 4, dtype=torch.bool))


def make_model(arm='point_segment_hough'):
    torch.manual_seed(7)
    return LayoutVerifier(arm).eval()


def test_all_arms_have_identical_parameters_and_initial_state():
    models = [make_model(a) for a in ARMS]
    sizes = [sum(p.numel() for p in m.parameters()) for m in models]
    assert len(set(sizes)) == 1
    for model in models[1:]:
        assert model.state_dict().keys() == models[0].state_dict().keys()
        for key, value in model.state_dict().items():
            assert torch.equal(value, models[0].state_dict()[key]), key


@pytest.mark.parametrize('arm', ARMS)
def test_actual_forward_shape_and_finite_diagnostics(arm):
    batch = fixture()
    cost, d = make_model(arm)(batch, return_diagnostics=True)
    assert cost.shape == (2, 3) and torch.isfinite(cost).all()
    assert d['percorner_error'].shape == (2, 3, 8)
    assert torch.isfinite(d['percorner_error']).all()
    assert d['corner_input_coverage'].shape == (2, 3, 8)
    assert d['edge_input_coverage'].shape == (2, 3, 12)
    assert d['hough_evidence_used'] == (arm == 'point_segment_hough')
    assert d['edge_interior_image_evidence_used'] == (arm != 'point_only')


def test_affine_sampling_has_correct_halfcell_and_padded_normalizer():
    feature = torch.zeros(1, 2, 40, 40)
    feature[0, 0] = torch.arange(40)[None]
    feature[0, 1] = torch.arange(40)[:, None]
    affine = torch.tensor([[[.5, 0., 70.], [0., .5, 90.]]])
    raw = torch.tensor([[[13.2, 44.4], [186.7, 201.3]]])
    image_xy = raw_to_input(raw, affine)
    actual, valid = sample_p4(feature, image_xy, torch.tensor([[384, 640]]))
    assert valid.all()
    assert torch.allclose(actual, image_xy/16.-.5, atol=2e-6, rtol=0)
    outside, valid = sample_p4(feature, torch.tensor([[[100., 400.], [-.1, 10.]]]), torch.tensor([[384, 640]]))
    assert not valid.any() and torch.equal(outside, torch.zeros_like(outside))


def test_bottom_padding_cannot_leak_through_projection_bias():
    model, batch = make_model(), fixture(batch=1)
    batch['input_shape_hw'][:] = torch.tensor([384, 640])
    changed = copy.deepcopy(batch)
    changed['p4'][:, :, 24:] = 10000.
    assert torch.equal(model(batch), model(changed))


def test_candidate_order_equivariance_and_single_candidate_independence():
    model, batch = make_model(), fixture()
    permutation = torch.tensor([2, 0, 1])
    original = model(batch)
    changed = {**batch, 'layouts': batch['layouts'][:, permutation]}
    assert torch.allclose(model(changed), original[:, permutation], atol=1e-6, rtol=0)
    alone = {**batch, 'layouts': batch['layouts'][:, 1:2]}
    assert torch.allclose(model(alone)[:, 0], original[:, 1], atol=1e-6, rtol=0)


def test_token_serialization_permutation_preserves_attached_semantic_ids():
    model, batch = make_model(), fixture()
    permutation = torch.randperm(20, generator=torch.Generator().manual_seed(4))
    cost, d = model(batch, return_diagnostics=True)
    perm_cost, perm_d = model(batch, return_diagnostics=True, token_order=permutation)
    assert torch.allclose(cost, perm_cost, atol=2e-6, rtol=0)
    assert torch.allclose(d['percorner_error'], perm_d['percorner_error'], atol=2e-6, rtol=0)


def test_semantic_relabelling_is_not_forced_to_have_equal_score():
    model, batch = make_model(), fixture(batch=1, layouts=1)
    permutation = torch.tensor([1, 5, 6, 2, 0, 4, 7, 3, 8])
    changed = {**batch, 'layouts': batch['layouts'][:, :, permutation]}
    assert not torch.allclose(model(batch), model(changed), atol=1e-7, rtol=0)


@pytest.mark.parametrize('arm', ARMS)
def test_edge_interior_feature_perturbation_isolated_from_point_control(arm):
    model, batch = make_model(arm), fixture(batch=1, layouts=1)
    changed = copy.deepcopy(batch)
    # Edge0's t=2.5/8 query is(200,220), interpolating cell centre(200,216).
    # That queried interior cell is far from every corner patch.
    changed['p4'][:, :, 13, 12] += torch.arange(128)[None]*.5
    before, after = model(batch), model(changed)
    if arm == 'point_only':
        assert torch.equal(before, after)
    else:
        assert (before-after).abs().max() > 1e-7


@pytest.mark.parametrize('arm', ARMS)
def test_hough_perturbation_isolated_from_both_controls(arm):
    model, batch = make_model(arm), fixture(batch=1)
    changed = copy.deepcopy(batch)
    changed['line_h'][:, :, :, 2] += 37.
    changed['peak_logits'] -= 2.
    before, after = model(batch), model(changed)
    if arm == 'point_segment_hough':
        assert (before-after).abs().max() > 1e-7
    else:
        assert torch.equal(before, after)


def test_line_cues_keep_absolute_sigmoid_mass_and_signed_endpoint_distances():
    model, batch = make_model(), fixture(batch=1, layouts=1)
    cues = model._line_cues(batch, batch['layouts']).reshape(1, 1, 12, 4, 4)
    assert torch.allclose(cues[0, 0, 0, :, 2], batch['peak_logits'][0, 0].sigmoid(), atol=0, rtol=0)
    assert not torch.allclose(cues[0, 0, 0, :, 2].sum(), torch.tensor(1.))
    sigma = .01*batch['diagonal'][0]
    expected = torch.tensor([-8., 0., 8., 16.])/sigma
    assert torch.allclose(cues[0, 0, 0, :, 0], expected, atol=1e-6, rtol=0)
    assert torch.allclose(cues[0, 0, 0, :, 1], expected, atol=1e-6, rtol=0)


def test_invalid_peaks_are_zero_and_cannot_inject_nonfinite_cues():
    model, batch = make_model(), fixture(batch=1, layouts=1)
    batch['peak_valid'][:, 2, 1] = False
    batch['peak_logits'][:, 2, 1] = float('nan')
    batch['line_h'][:, 2, 1] = float('nan')
    cost, d = model(batch, return_diagnostics=True)
    assert torch.isfinite(cost).all()
    cues = d['line_cues'].reshape(1, 1, 12, 4, 4)
    assert torch.equal(cues[:, :, 2, 1], torch.zeros_like(cues[:, :, 2, 1]))


def test_full_model_cost_has_finite_nonzero_visual_and_hough_gradients():
    model, batch = make_model().train(), fixture(batch=1, layouts=2)
    cost = model(batch)
    (cost.square().mean()+cost.mean()).backward()
    for name, module in [('visual', model.visual_projection), ('line', model.line_encoder),
                         ('corner', model.corner_encoder), ('edge', model.edge_encoder), ('interaction', model.interaction)]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), name
        assert sum(float(g.abs().sum()) for g in gradients) > 0, name


def test_no_batchnorm_dropout_rng_update_and_no_batch_mutation():
    model, batch = make_model().train(), fixture(batch=1, layouts=2)
    before = {k: v.clone() for k, v in batch.items()}
    rng = torch.get_rng_state().clone()
    model(batch)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(batch[k], v) for k, v in before.items())
    assert not any(isinstance(m, nn) for m in model.modules() for nn in (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))


def test_float64_cached_affines_and_line_equations_are_supported():
    model, batch = make_model(), fixture(batch=1)
    expected = model(batch)
    batch['raw_to_input_affine'] = batch['raw_to_input_affine'].double()
    batch['line_h'] = batch['line_h'].double()
    assert torch.equal(model(batch), expected)


def test_GT_target_fields_rejected_and_contract_has_no_targets():
    assert not any('gt' in k.lower() or 'target' in k.lower() for k in INPUT_KEYS)
    model, batch = make_model(), fixture(batch=1)
    with pytest.raises(ValueError, match='Target/GT'):
        model({**batch, 'gt_valid': torch.ones(1, 9)})
