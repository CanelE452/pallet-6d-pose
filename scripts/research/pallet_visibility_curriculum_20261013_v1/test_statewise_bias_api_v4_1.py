"""Meaningful full-API bias contracts; these use synthetic injected features."""
from copy import deepcopy

import numpy as np
import pytest

from .statewise_bias_api_v4_1 import StatewiseBiasV4_1
from .test_statewise_joint_api_v4 import fixture


def bias_fixture(strength):
    base, image, q, K, xyz, coarse, model = fixture()
    model = deepcopy(model)
    model['prior_fit'] = {'median_bias6': [.01, -.02, .003, .002, -.001, .014]}
    api = StatewiseBiasV4_1(
        uncertainty_model=model, state_producer=base.state_producer,
        config=deepcopy(base.config))
    api.config['solver']['bias_strength'] = strength
    return base, api, image, q, K, xyz, coarse


def test_zero_strength_is_exact_V4_full_API_pose_and_observation_parity():
    base, api, image, q, K, xyz, coarse = bias_fixture(0.)
    kwargs = {'pallet_mask': np.ones(image.shape[:2])}
    before = base.refine(image, q, K, xyz, coarse, **kwargs)
    after = api.refine(image, q, K, xyz, coarse, **kwargs)
    for key in ('states', 'mask_audit', 'frozen_visible_measurements',
                'calibrated_visible_measurements', 'actual_pose', 'keypoint_readout'):
        assert before[key] == after[key], key
    assert not after['joint_pose']['prior_audit']['bias_active_in_loss']


def test_CAL_mean_propagates_without_shifting_observations_pose_or_Q():
    _, api, image, q, K, xyz, coarse = bias_fixture(.5)
    raw_before, pose_before = q.copy(), deepcopy(coarse)
    kwargs = {'pallet_mask': np.ones(image.shape[:2])}
    result = api.refine(image, q, K, xyz, coarse, **kwargs)
    audit = result['joint_pose']['prior_audit']
    mean = api.uncertainty_model['prior_fit']['median_bias6']
    assert audit['prior_bias6'] == mean and audit['bias_strength'] == .5
    assert np.allclose(audit['effective_prior_bias6'], np.asarray(mean) * .5)
    assert audit['bias_active_in_loss']
    assert np.array_equal(q, raw_before) and coarse == pose_before
    assert result['original_coarse_pose'] == coarse
    assert np.allclose(result['actual_pose']['R_physical'], result['joint_pose']['R_physical'])
    for observed, fit in zip(result['frozen_visible_measurements'], result['calibrated_visible_measurements']):
        assert observed['xy'] == fit['xy']
        assert observed['covariance_px2'] == fit['covariance_px2']
    changed = q.copy()
    for k, state in enumerate(result['states']):
        if state == 'S':
            changed[k] = [1e8, -1e9]
    assert not np.array_equal(changed, q)
    perturbed = api.refine(image, changed, K, xyz, coarse, **kwargs)
    for key in ('states', 'mask_audit', 'frozen_visible_measurements',
                'calibrated_visible_measurements', 'actual_pose', 'keypoint_readout'):
        assert result[key] == perturbed[key], key


def test_nonzero_bias_rejects_missing_CAL_mean_and_runtime_replacement():
    _, api, image, q, K, xyz, coarse = bias_fixture(1.)
    observation = api.observe(image, q, K, xyz, coarse,
                              pallet_mask=np.ones(image.shape[:2]))
    saved = api.uncertainty_model.pop('prior_fit')
    with pytest.raises(ValueError, match='fitted CAL prior median'):
        api.solve_observations(observation)
    api.uncertainty_model['prior_fit'] = saved
    api.config['solver']['prior_bias6'] = [0.] * 6
    with pytest.raises(ValueError, match='supplied CAL model median'):
        api.solve_observations(observation)


def test_zero_prior_weight_makes_nonzero_bias_inactive_in_loss():
    _, api, image, q, K, xyz, coarse = bias_fixture(1.)
    api.config['solver']['prior_weight'] = 0.
    result = api.refine(image, q, K, xyz, coarse,
                        pallet_mask=np.ones(image.shape[:2]))
    assert not result['joint_pose']['prior_audit']['bias_active_in_loss']
    assert not result['provenance']['initial_pose_is_soft_regularizer']


@pytest.mark.parametrize('field,value', [('fit_partition', 'TEST'), ('schema', 'UNKNOWN')])
def test_mapping_models_cannot_bypass_CAL_schema_and_partition_guards(field, value):
    _, api, image, q, K, xyz, coarse = bias_fixture(.5)
    observation = api.observe(image, q, K, xyz, coarse,
                              pallet_mask=np.ones(image.shape[:2]))
    api.uncertainty_model[field] = value
    with pytest.raises(ValueError, match='valid CAL-only uncertainty model'):
        api.solve_observations(observation)
