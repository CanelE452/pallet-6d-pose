from copy import deepcopy

import numpy as np
import pytest

from . import mixture_native_edge_factor_v4_18 as M
from . import projected_native_edge_factor_v4_17 as E
from .test_projected_native_edge_factor_v4_17 import XYZ, K, SCALE, _edge


def test_mixture_full_analytic_gradient_and_input_immutability():
    p = np.r_[[.2, -.3, .1], [.15, -.2, 4.]]
    a = _edge(); b = _edge(offset_px=-250.)
    b['support_anchor_xy'] = (np.asarray(a['support_anchor_xy']) + 5.*np.asarray(a['normal_unit_xy'])).tolist()
    components = [a,b]; saved = deepcopy(components)
    value, grad, audit = M.mixture_edge_value_and_gradient(p, XYZ, K, SCALE, components,
        component_weights=[.3,.7], reliability_weight=.6)
    numeric = np.zeros(6)
    for j in range(6):
        step = np.zeros(6); step[j] = 1e-6
        numeric[j] = (M.mixture_edge_value_and_gradient(p+step, XYZ, K, SCALE, components,
            component_weights=[.3,.7], reliability_weight=.6)[0]
            - M.mixture_edge_value_and_gradient(p-step, XYZ, K, SCALE, components,
            component_weights=[.3,.7], reliability_weight=.6)[0])/2e-6
    assert np.allclose(grad, numeric, atol=3e-5, rtol=3e-5)
    assert value >= 0 and sum(audit['posterior_component_responsibilities']) == pytest.approx(1.)
    assert components == saved


def test_single_component_exact_parent_and_duplicate_list_no_likelihood_bonus():
    p = np.r_[[.2, -.3, .1], [.15, -.2, 4.]]; edge = _edge()
    value, gradient, _ = E.projected_edge_value_and_gradient(p, XYZ, K, SCALE, edge, weight=.6)
    for components in ([edge], [edge,edge], [edge]*17):
        other, derivative, _ = M.mixture_edge_value_and_gradient(p, XYZ, K, SCALE, components,
            reliability_weight=.6)
        assert other == pytest.approx(value, rel=1e-12, abs=1e-12)
        assert np.allclose(derivative, gradient, rtol=1e-12, atol=1e-12)


def test_group_reliability_once_and_validation_and_off_before_poison_fields():
    p = np.r_[[.2, -.3, .1], [.15, -.2, 4.]]; edge = _edge()
    full, grad, _ = M.mixture_edge_value_and_gradient(p, XYZ, K, SCALE, [edge])
    half, derivative, _ = M.mixture_edge_value_and_gradient(p, XYZ, K, SCALE, [edge], reliability_weight=.5)
    assert half == full*.5 and np.array_equal(derivative, grad*.5)
    value, derivative, _ = M.mixture_edge_value_and_gradient(None, None, None, None, object(), reliability_weight=0.)
    assert value == 0. and np.array_equal(derivative, np.zeros(6))
    with pytest.raises(ValueError):
        M.mixture_edge_value_and_gradient(p, XYZ, K, SCALE, [edge,{**edge,'native_neighbor':2}])
    with pytest.raises(ValueError):
        M.mixture_edge_value_and_gradient(p, XYZ, K, SCALE, [edge], component_weights=[0.])
