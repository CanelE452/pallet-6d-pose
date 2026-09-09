"""CPU geometry, information-path and gradient checks; no dataset/GT files."""
import math
from unittest.mock import patch

import pytest
import torch

from scripts.research.pallet_dht_local_v3 import model as M


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batch():
    points = torch.tensor([[[200., 240.], [400., 240.], [400., 400.], [200., 400.],
                            [220., 260.], [420., 260.], [420., 420.], [220., 420.], [310., 330.]]])
    y, x = torch.meshgrid(torch.arange(640)+.5, torch.arange(640)+.5, indexing='ij')
    image = .25+.25*torch.sigmoid((x-204.)/1.)+.25*torch.sigmoid((y-244.)/1.)
    return dict(image_gray=image[None, None], input_content_mask=torch.ones((1, 1, 640, 640), dtype=torch.bool),
        baseline_points=points, raw_to_input_affine=torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]),
        point_valid=torch.ones((1, 9), dtype=torch.bool))


def analytic_evidence(rho=3., degrees=0.):
    inputs = batch()
    geometry = M.edge_geometry(inputs['baseline_points'], inputs['point_valid'])
    geometry['contrast_present'] = torch.ones((1, 12), dtype=torch.bool)
    u = torch.linspace(-16., 16., 33)[None, None, None]
    angle = math.radians(degrees)
    centre = (rho+geometry['along_coordinates']*math.sin(angle))/math.cos(angle)
    weights = torch.exp(-.5*((u-centre[..., None])/.5).square())
    return weights, torch.ones_like(weights, dtype=torch.bool), geometry


def test_same_parameters_and_initial_state():
    torch.manual_seed(1); first = M.LocalHoughPointRefiner('no_hough')
    torch.manual_seed(1); second = M.LocalHoughPointRefiner('hough')
    assert sum(p.numel() for p in first.parameters()) == sum(p.numel() for p in second.parameters()) == 3243
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())


@pytest.mark.parametrize('arm', M.ARMS)
def test_initial_identity_and_centroid_are_bit_exact(arm):
    data = batch()
    data['raw_to_input_affine'] = torch.tensor([[[.7619048, 0., 76.19048], [0., .7619048, 89.19048]]])
    data['baseline_points'] = (data['baseline_points']-data['raw_to_input_affine'][..., 2][:, None])/.7619048
    output, diag = M.LocalHoughPointRefiner(arm)(data)
    assert torch.equal(output, data['baseline_points'])
    assert torch.equal(output[:, 8], data['baseline_points'][:, 8])
    assert torch.count_nonzero(diag['point_delta_input']) == 0


@pytest.mark.parametrize('arm', M.ARMS)
@pytest.mark.parametrize('contrast', [0., .002])
def test_flat_and_low_contrast_remain_identity_with_open_gate(arm, contrast):
    data = batch()
    data['image_gray'].fill_(.4)
    data['image_gray'][:, :, 244:] += contrast
    model = M.LocalHoughPointRefiner(arm)
    with torch.no_grad(): model.gate_logits.fill_(1.5)
    result, diag = model(data)
    assert torch.equal(result, data['baseline_points'])
    assert torch.count_nonzero(diag['edge_confidence']) == 0


def test_affine_and_stride1_pixel_centre_sampling():
    data = batch()
    # Double precision isolates coordinate alignment from FP32 interpolation
    # roundoff; actual model FP32 behaviour is covered by identity/gradient tests.
    image = (torch.arange(640, dtype=torch.float64)+.5)/640.
    data['image_gray'] = image[None, None, None].expand(1, 1, 640, 640).clone()
    affine = torch.tensor([[[.5, 0., 50.], [0., .5, 100.]]], dtype=torch.float64)
    raw = (data['baseline_points'].double()-affine[..., 2][:, None])/.5
    corridor, valid, _, geometry = M.extract_corridors(data['image_gray'], data['input_content_mask'], raw, affine, data['point_valid'])
    assert valid.all()
    assert torch.allclose(corridor[:, :, 0], geometry['sample_xy'][..., 0]/640., atol=1e-12, rtol=0)
    delta = torch.ones((1, 9, 2), dtype=torch.float64)
    assert torch.equal(M.input_delta_to_raw(delta, affine), delta*2.)


def test_reflection_or_padding_edge_does_not_supply_evidence():
    data = batch()
    data['image_gray'].fill_(.25)
    data['image_gray'][:, :, :, 200:] = .9
    data['input_content_mask'][:, :, :, 199:] = False
    corridor, valid, strength, geometry = M.extract_corridors(**{k: data[k] for k in ('image_gray','input_content_mask')},
        points_raw=data['baseline_points'], raw_to_input_affine=data['raw_to_input_affine'], point_valid=data['point_valid'])
    assert torch.count_nonzero(strength) == 0
    assert torch.count_nonzero(geometry['contrast_max']) == 0
    assert torch.isfinite(corridor).all()


def test_horizontal_and_vertical_hough_line_offset():
    weights, mask, geometry = analytic_evidence()
    result = M.local_hough(weights, mask, geometry)
    # Roles0 and1 are horizontal and vertical with the same local normal offset.
    assert torch.allclose(result['rho_input_px'][0, :2], torch.tensor([3., 3.]), atol=.05, rtol=0)
    assert result['angle_offset_radians'][0, :2].abs().max() < math.radians(.05)
    lines = result['line_h_input'][0, :2]
    shifted = geometry['midpoint'][0, :2]+3.*geometry['normal'][0, :2]
    assert ((lines[:, :2]*shifted).sum(-1)+lines[:, 2]).abs().max() < .05


def test_slanted_local_line_recovered_in_rho_angle_coordinates():
    weights, mask, geometry = analytic_evidence(rho=2., degrees=3.)
    result = M.local_hough(weights, mask, geometry)
    assert abs(float(result['rho_input_px'][0, 0])-2.) < .15
    assert abs(float(result['angle_offset_radians'][0, 0])-math.radians(3.)) < math.radians(.2)


def test_hough_support_and_short_edges_drop_safely():
    weights, mask, geometry = analytic_evidence()
    mask.zero_()
    result = M.local_hough(weights*mask, mask, geometry)
    assert not result['edge_valid'].any()
    assert torch.count_nonzero(result['confidence']) == 0
    assert torch.isfinite(result['line_h_input']).all()
    p = torch.zeros((1, 9, 2)); v = torch.ones((1, 9), dtype=torch.bool)
    assert not M.edge_geometry(p, v)['edge_valid'].any()


def test_real_contrast_uniform_posterior_keeps_confidence_floor():
    weights, mask, geometry = analytic_evidence()
    result = M.local_hough(weights*0., mask, geometry)
    assert result['edge_valid'].all()
    assert (result['confidence'] >= .05).all()


def test_missing_corner_drops_incident_lines_and_retains_point():
    data = batch(); data['point_valid'][0, 0] = False
    model = M.LocalHoughPointRefiner('hough')
    with torch.no_grad(): model.gate_logits.fill_(1.)
    output, diag = model(data)
    assert torch.equal(output[0, 0], data['baseline_points'][0, 0])
    assert torch.count_nonzero(diag['edge_confidence'][0, [0, 3, 8]]) == 0
    assert torch.isfinite(output).all()


def test_wls_known_horizontal_vertical_solution():
    data = batch(); p = data['baseline_points']
    lines = torch.zeros((1, 12, 3)); confidence = torch.zeros((1, 12))
    lines[0, 0] = torch.tensor([0., 1., -243.]); lines[0, 3] = torch.tensor([1., 0., -202.])
    confidence[0, 0] = confidence[0, 3] = 1.
    gate = torch.full((8,), math.atanh(.5))
    output, diag = M.regularized_refine(p, lines, confidence, gate, data['point_valid'])
    assert torch.allclose(diag['wls_delta_input'][0, 0], torch.tensor([1., 1.5]), atol=1e-7, rtol=0)
    assert torch.equal(output[0, 0], torch.tensor([200.5, 240.75]))
    assert torch.equal(output[:, 8], p[:, 8])


def test_near_parallel_wls_is_finite_and_bounded():
    data = batch(); p = data['baseline_points']
    lines = torch.zeros((1, 12, 3)); confidence = torch.zeros((1, 12))
    lines[0, 0] = torch.tensor([0., 1., -10000.]); lines[0, 3] = torch.tensor([1e-8, 1., -10000.])
    confidence[0, 0] = confidence[0, 3] = 1.
    output, diag = M.regularized_refine(p, lines, confidence, torch.ones(8)*10., data['point_valid'])
    assert torch.isfinite(output).all()
    assert torch.linalg.eigvalsh(diag['wls_matrix']).min() >= 1.-1e-6
    assert torch.linalg.vector_norm(diag['point_delta_input'], dim=-1).max() <= 4.+1e-6


def test_nonfinite_line_is_ignored_without_nan_spread():
    data = batch(); lines = torch.zeros((1, 12, 3)); weights = torch.zeros((1, 12))
    lines[0, 0] = float('nan'); weights[0, 0] = float('nan')
    out, _ = M.regularized_refine(data['baseline_points'], lines, weights, torch.ones(8), data['point_valid'])
    assert torch.equal(out, data['baseline_points'])


@pytest.mark.parametrize('arm', M.ARMS)
def test_open_gate_point_loss_reaches_active_cnn_branch(arm):
    torch.manual_seed(2)
    data = batch(); model = M.LocalHoughPointRefiner(arm)
    with torch.no_grad(): model.gate_logits.fill_(.8)
    output, _ = model(data)
    target = data['baseline_points']+torch.tensor([1., 2.])
    (output[:, :8]-target[:, :8]).square().mean().backward()
    assert model.cnn[0].weight.grad is not None
    assert model.cnn[0].weight.grad.abs().sum() > 0
    active = model.weight_head.weight if arm == 'hough' else model.direct_head[-1].weight
    inactive = model.direct_head[-1].weight if arm == 'hough' else model.weight_head.weight
    assert active.grad is not None and active.grad.abs().sum() > 0
    assert inactive.grad is None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_first_identity_step_updates_gate_before_cnn():
    torch.manual_seed(2)
    data = batch(); model = M.LocalHoughPointRefiner('hough')
    out, _ = model(data)
    (out[:, :8]-data['baseline_points'][:, :8]-2.).square().mean().backward()
    assert model.gate_logits.grad.abs().sum() > 0
    assert model.cnn[0].weight.grad.abs().sum() == 0


def test_direct_control_does_not_call_hough_operator():
    with patch.object(M, 'local_hough', side_effect=AssertionError('Hough path used by control')):
        M.LocalHoughPointRefiner('no_hough')(batch())


def test_ground_truth_model_input_rejected():
    data = batch(); data['gt_points'] = data['baseline_points'].clone()
    with pytest.raises(ValueError, match='forbidden'):
        M.LocalHoughPointRefiner()(data)
