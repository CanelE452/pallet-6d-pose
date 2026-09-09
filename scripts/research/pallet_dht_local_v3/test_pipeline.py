"""CPU analytic pipeline checks; no dataset, detector or trained-model forward."""
import json

import numpy as np
import pytest
import torch

from scripts.research.pallet_dht_local_v3 import evaluation as E
from scripts.research.pallet_dht_local_v3 import train as T


def affine(gains):
    result = torch.zeros((len(gains), 2, 3), dtype=torch.float64)
    result[:, 0, 0] = torch.tensor(gains)
    result[:, 1, 1] = torch.tensor(gains)
    result[:, :, 2] = torch.tensor([123., -79.])
    return result


def protocol():
    return dict(calibration=dict(output_scale_grid=[0., .125, .25, .5, 1.],
                                 tie_tolerance=1e-12),
                synthetic_advancement=dict(native_mean_reduction_min_fraction=.01))


def observations(before=10., after=8., n=1):
    base = np.zeros((n, 9, 2), dtype=np.float64)
    predicted = base.copy()
    base[:, :8, 0] = before
    predicted[:, :8, 0] = after
    return dict(baseline=base, predicted=predicted, target=np.zeros_like(base),
                loss_valid=np.ones((n, 9), bool), diagonal=np.full(n, 100.))


def test_point_loss_analytic_input_units_mask_and_gradient():
    predicted = torch.zeros((2, 9, 2), dtype=torch.float64)
    predicted[0, 0] = torch.tensor([.5, -2.])
    predicted[1, 1] = torch.tensor([2., .25])
    predicted[:, 8] = 10000.  # Centroid is excluded even if its mask is true.
    predicted[:, 5] = -10000.  # Unknown corner cannot affect loss or gradient.
    predicted.requires_grad_()
    mask = torch.zeros((2, 9), dtype=torch.bool)
    mask[0, 0] = mask[1, 1] = mask[:, 8] = True
    targets = dict(points=torch.zeros_like(predicted), loss_valid=mask,
                   valid=torch.ones_like(mask))
    batch = dict(raw_to_input_affine=affine([1., 2.]))
    loss, count = T.point_loss(predicted, targets, batch)
    # smooth-L1(.5,-2) -> (.125+1.5)/2; (4,.5) -> (3.5+.125)/2.
    assert count == 2
    assert loss.item() == 1.3125
    loss.backward()
    expected = torch.zeros_like(predicted)
    expected[0, 0] = torch.tensor([.125, -.25])
    expected[1, 1] = torch.tensor([.5, .25])
    assert torch.equal(predicted.grad, expected)
    moved = dict(raw_to_input_affine=batch['raw_to_input_affine'].clone())
    moved['raw_to_input_affine'][:, :, 2] += 9000.
    assert T.point_loss(predicted.detach(), targets, moved)[0] == loss.detach()


def test_centroid_only_mask_is_not_a_supervised_corner_batch():
    points = torch.zeros((1, 9, 2), dtype=torch.float64)
    mask = torch.zeros((1, 9), dtype=torch.bool)
    mask[:, 8] = True
    with pytest.raises(ValueError, match='Empty supervised'):
        T.point_loss(points, dict(points=points, loss_valid=mask),
                     dict(raw_to_input_affine=affine([1.])))


def test_jitter_is_input_pixel_displacement_without_mutating_source():
    points = torch.arange(36, dtype=torch.float64).reshape(2, 9, 2)
    valid = torch.ones((2, 9), dtype=torch.bool)
    valid[0, 3] = False
    inputs = dict(baseline_points=points, point_valid=valid,
                  raw_to_input_affine=affine([.5, 2.]), image_gray=torch.tensor([.3]))
    snapshot = points.clone()
    jitter = np.full((2, 8, 2), [6., -2.], dtype=np.float64)
    augmented = T.augment(inputs, jitter)
    actual_input_delta = ((augmented['baseline_points'][:, :8]-points[:, :8])
                          * torch.tensor([.5, 2.])[:, None, None])
    expected = torch.from_numpy(jitter)*valid[:, :8, None]
    assert torch.equal(actual_input_delta, expected)
    assert torch.equal(points, snapshot)
    assert torch.equal(augmented['baseline_points'][:, 8], points[:, 8])
    assert torch.equal(augmented['baseline_points'][0, 3], points[0, 3])
    assert augmented['image_gray'] is inputs['image_gray']


def test_noise_and_sample_plan_are_reproducible_and_population_bounded():
    cfg = dict(steps=64, batch=4, clean_probability=.5,
               jitter_input_sigma_px=2., jitter_clip_input_px=6.)
    indices = np.arange(11)+100
    first = T.plans(indices, cfg, 1)
    np.random.seed(999)
    np.random.normal(size=100)
    second = T.plans(indices, cfg, 1)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert set(first[0].ravel()) == set(indices)
    for start in range(0, first[0].size-len(indices)+1, len(indices)):
        assert np.array_equal(np.sort(first[0].ravel()[start:start+len(indices)]), indices)
    noise = first[1].reshape(-1, 8, 2)
    clean = (noise == 0).all(axis=(1, 2))
    assert clean.any() and (~clean).any()
    assert np.max(np.abs(noise)) <= 6.
    other = T.plans(indices, cfg, 2)
    assert not np.array_equal(other[0], first[0])
    assert not np.array_equal(other[1], first[1])


def test_zero_scale_and_smallest_scale_tie_preserve_identity():
    values = observations(before=4., after=4.)
    values['predicted'][:, 8] = 1e9  # Ninth point never enters calibration.
    p = protocol()
    p['calibration']['output_scale_grid'] = [1., .5, 0., .125]
    selection = E.choose_scale(values, p)
    assert selection['scale'] == 0.
    assert selection['baseline']['mean_px'] == 4.
    assert all(x['metrics']['mean_px'] == 4. for x in selection['grid'])
    values['predicted'][:, :8] = 1e8
    assert E.scaled_metrics(values, 0.)['mean_px'] == 4.


def test_calibration_objective_is_pooled_normalized_points_not_frame_mean():
    values = observations(n=2)
    values['baseline'][1, :8, 0] = 20.
    values['diagonal'] = np.array([100., 400.])
    values['loss_valid'][1, 1:8] = False
    metric = E.scaled_metrics(values, 0.)
    assert metric['n_supervised_corners'] == 9
    assert metric['mean_diagonal_normalized'] == pytest.approx((8*.1+.05)/9.)
    assert metric['mean_diagonal_normalized'] != pytest.approx((.1+.05)/2.)


def test_good_crossing_denominator_and_exact_one_percent_boundary():
    values = observations(before=9., after=8., n=13)
    values['loss_valid'][:, :8].reshape(-1)[100:] = False
    # Preserve explicit contiguous array assignment; numpy's first8 view can copy.
    mask = np.arange(104).reshape(13, 8) < 100
    values['loss_valid'][:, :8] = mask
    values['baseline'][12, 4:8, 0] = 5000.
    values['predicted'][0, 0, 0] = 10.1
    base = E.scaled_metrics(values, 0.)
    metric = E.scaled_metrics(values, 1.)
    assert metric['baseline_good_le10_count'] == 100
    assert metric['good_to_bad_gt10_count'] == 1
    assert metric['good_to_bad_fraction'] == .01
    assert all(E.constraints(metric, base).values())
    values['predicted'][0, 1, 0] = 10.1
    metric = E.scaled_metrics(values, 1.)
    checks = E.constraints(metric, base)
    assert metric['good_to_bad_fraction'] == .02
    assert not checks['good_crossing_le_one_percent']
    assert checks['mean_px_no_worse'] and checks['p90_px_no_worse']


@pytest.mark.parametrize('key', ['mean_px', 'median_px', 'p90_px'])
def test_each_clean_error_constraint_is_required(key):
    baseline = dict(mean_px=5., median_px=4., p90_px=9., good_to_bad_fraction=0.)
    changed = dict(baseline)
    changed[key] += .01
    checks = E.constraints(changed, baseline)
    assert not checks[f'{key}_no_worse']
    assert not all(checks.values())


def test_scale_selection_rejects_overshoot_and_uses_registered_grid():
    # Raw displacement -6: scales .125,.25,.5,1 produce errors1.25,.5,1,4.
    chosen = E.choose_scale(observations(before=2., after=-4.), protocol())
    assert chosen['scale'] == .25
    assert chosen['selected']['metrics']['mean_px'] == .5
    assert not chosen['grid'][-1]['eligible']


def test_validation_is_opened_after_selection_and_cannot_refit_scale(tmp_path, monkeypatch):
    p = protocol()
    monkeypatch.setattr(E, 'verify', lambda run: (p, {'fixture': True}))
    checkpoint = tmp_path/'runs/hough_seed1/checkpoint_final.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'generated fixture, not a model')
    opened = []
    def predict(run, arm, seed, population, device):
        opened.append(population)
        path = tmp_path/f'evaluation/{arm}_seed{seed}/{population}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'generated fixture')
        if population == 'calibration':
            assert not (tmp_path/'SELECTION_hough_seed1.json').exists()
            return observations(before=10., after=8.)
        saved = json.loads((tmp_path/'SELECTION_hough_seed1.json').read_text())
        assert saved['scale'] == 1. and not saved['validation_used_for_selection']
        # This held-out fixture would favor0; it must not change calibration1.
        return observations(before=10., after=11.)
    monkeypatch.setattr(E, 'predict_population', predict)
    result = E.evaluate_synthetic(tmp_path, 'hough', 1, 'cpu')
    assert opened == ['calibration', 'synth_val']
    assert result['scale'] == 1.
    assert not result['advancement']['advance']


def test_population_rejects_real_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'verify', lambda run: (protocol(), {}))
    def forbidden(*args, **kwargs):
        pytest.fail('Real evaluation must fail before data/model access')
    monkeypatch.setattr(E, 'LocalData', forbidden)
    monkeypatch.setattr(E, 'load_trained', forbidden)
    with pytest.raises(ValueError, match='Synthetic evaluation only'):
        E.predict_population(tmp_path, 'hough', 1, 'real_dev', 'cpu')


def test_completed_archive_reuses_bound_values_without_forward_and_rejects_tamper(tmp_path, monkeypatch):
    bindings = {'generated_fixture': True}
    monkeypatch.setattr(E, 'verify', lambda run: (protocol(), bindings))
    checkpoint = tmp_path/'runs/hough_seed1/checkpoint_final.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'generated checkpoint fixture')
    path = tmp_path/'evaluation/hough_seed1/calibration.npz'
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, values=np.array([1., 2., 3.]))
    receipt = dict(complete=True, PASS=True, arm='hough', seed=1,
        population='calibration', bindings=bindings,
        checkpoint_sha256=E.sha(checkpoint), archive_sha256=E.sha(path))
    (path.parent/'calibration_COMPLETE.json').write_text(json.dumps(receipt))
    def forbidden(*args, **kwargs):
        pytest.fail('Completed archive must not reopen data or run a model')
    monkeypatch.setattr(E, 'LocalData', forbidden)
    monkeypatch.setattr(E, 'load_trained', forbidden)
    result = E.predict_population(tmp_path, 'hough', 1, 'calibration', 'cpu')
    assert np.array_equal(result['values'], [1., 2., 3.])
    path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(ValueError, match='Completed evaluation differs'):
        E.predict_population(tmp_path, 'hough', 1, 'calibration', 'cpu')


def test_uncompleted_archive_is_preserved_without_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'verify', lambda run: (protocol(), {}))
    checkpoint = tmp_path/'runs/hough_seed1/checkpoint_final.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixture')
    path = tmp_path/'evaluation/hough_seed1/calibration.npz'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'prior interrupted archive')
    with pytest.raises(ValueError, match='Interrupted population archive'):
        E.predict_population(tmp_path, 'hough', 1, 'calibration', 'cpu')
    assert path.read_bytes() == b'prior interrupted archive'
