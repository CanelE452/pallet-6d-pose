"""New-population schema/export and immutable NLL recipe; synthetic only."""
from copy import deepcopy

import numpy as np
import pytest

from . import calibrate_source_preserving_bi_edge_mixture_v4_24 as F
from .test_calibrate_complete_bi_edge_mixture_v4_20d import component, low, row


def test_exact_old_support_balanced_NLL_and_gradient():
    group = [low(component(3., .1, prior=.25)),
        low(component(-5., -.05, variant='1', prior=.25)),
        low(component(8., .2, support='b', prior=.5))]
    variance = np.array([2., .003])
    value, gradient = F.mixture_noise_nll_and_gradient(variance, [group], [.7])
    oldvalue, oldgradient = F.PARENT.mixture_noise_nll_and_gradient(variance, [group], [.7])
    assert value == oldvalue and np.array_equal(gradient, oldgradient)
    numeric = []
    for j, step in enumerate([1e-5, 1e-7]):
        delta = np.zeros(2)
        delta[j] = step
        numeric.append((F.mixture_noise_nll_and_gradient(variance + delta, [group], [.7])[0]
            - F.mixture_noise_nll_and_gradient(variance - delta, [group], [.7])[0]) / (2 * step))
    assert np.allclose(gradient, numeric, atol=1e-5, rtol=1e-6)


def test_synthetic_fit_has_new_schema_same_variances_and_no_gamma_refit():
    records = [row('synthetic_a', 'A', [component(3., .05)]),
        row('synthetic_b', 'B', [component(-4., -.04)])]
    before = deepcopy(records)
    model = F.fit_native_edge_mixture_model(records)
    old = F.PARENT.fit_native_edge_mixture_model(records)
    assert records == before and model['schema'] != old['schema']
    assert model['normal_variance_component_px2'] == old['normal_variance_component_px2']
    assert model['angular_variance_component_sine2'] == old['angular_variance_component_sine2']
    assert model['optimizer_searches'] == old['optimizer_searches']
    parameters = F.mixture_noise_parameters(model)
    assert parameters['noise_provenance']['observation_family'] == F.AUTH.NOISE_OBSERVATION_FAMILY
    assert parameters['noise_provenance']['source_policy'] == F.AUTH.SOURCE_POLICY
    assert parameters['noise_provenance']['tangent_gamma_fitted_by_this_model'] is False
    assert model['recipe']['optimizer_scaled_variance_starts'] == F.PARENT.RECIPE['optimizer_scaled_variance_starts']
    with pytest.raises(ValueError):
        F.mixture_noise_parameters(old)


def test_CAL_session_and_incompatible_exclusion_before_error_fields():
    class Poison(dict):
        def __getitem__(self, key):
            raise AssertionError('Excluded reference must not be read')
    records = [row('synthetic_a', 'A', [component(2., .05)]),
        dict(id='held', session='B', partition='CAL', compatible=True, groups=Poison()),
        dict(id='incompatible', session='C', partition='CAL', compatible=False, groups=Poison())]
    model = F.fit_native_edge_mixture_model(records, heldout_session='B')
    assert model['fit_ids'] == ['synthetic_a']
    assert model['excluded_heldout_ids'] == ['held']
    assert model['incompatible_inventory_ids'] == ['incompatible']
    assert model['inventory_frame_count'] == 3
    with pytest.raises(ValueError, match='CAL only'):
        F.fit_native_edge_mixture_model([dict(partition='TEST', groups=Poison())])
