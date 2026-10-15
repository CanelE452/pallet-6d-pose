"""Unchanged numerical recipe with a strict, separate observer generation."""
from copy import deepcopy

import pytest

from . import calibrate_grazing_edge_mixture_v4_18b as F
from .test_calibrate_native_edge_mixture_v4_18 import component,row


def records():
    return [row('a','A',[component(3.,.1),component(7.,.2)]),
        row('b','B',[component(-3.,-.04)]),row('c','B',[component(2.,.05)])]


def test_numeric_fitter_and_three_starts_exact_parent_without_module_mutation():
    data=records();before=deepcopy(data);recipe=deepcopy(F.OLD.RECIPE);schema=F.OLD.SCHEMA
    old=F.OLD.fit_native_edge_mixture_model(data);new=F.fit_native_edge_mixture_model(data)
    assert data==before and F.OLD.RECIPE==recipe and F.OLD.SCHEMA==schema
    for key in ['normal_variance_component_px2','angular_variance_component_sine2',
            'training_mixture_nll','optimizer_searches','optimizer_conditioning_variance_scales','fit_ids']:
        assert new[key]==old[key]
    assert new['schema']!=old['schema'] and new['recipe']['old_boundary_all_used']is False
    assert new['recipe']['observation_generation']==F.OBS.OBSERVATION_GENERATION


def test_old_model_rejected_and_new_components_exported_without_double_floor():
    old=F.OLD.fit_native_edge_mixture_model(records());new=F.fit_native_edge_mixture_model(records())
    with pytest.raises(ValueError):F.mixture_noise_parameters(old)
    with pytest.raises(ValueError):F.OLD.mixture_noise_parameters(new)
    params=F.mixture_noise_parameters(new)
    assert params['normal_variance_component_px2']==new['normal_variance_component_px2']
    assert params['angular_variance_component_sine2']==new['angular_variance_component_sine2']
    assert params['noise_provenance']['observation_generation']==F.OBS.OBSERVATION_GENERATION
    assert not params['noise_provenance']['old_boundary_all_or_old_observer_model_used']


def test_heldout_incompatible_and_other_partitions_gated_before_errors():
    class Poison(dict):
        def __getitem__(self,key):raise AssertionError('Excluded reference error read')
    data=records();data[1]['groups']=[Poison()];data[2]['compatible']=False;data[2]['groups']=[Poison()]
    model=F.fit_native_edge_mixture_model(data,heldout_session='B')
    assert model['fit_ids']==['a'] and set(model['excluded_heldout_ids'])=={'b','c'}
    with pytest.raises(ValueError,match='CAL only'):
        F.fit_native_edge_mixture_model([dict(partition='REMAINDER',groups=[Poison()])])


def test_runtime_floor_optimizer_and_weighting_recipe_unchanged():
    for key in ['normal_floor','angular_floor','weighting','estimator','huber_scale',
            'optimizer_scaled_variance_starts','optimizer_maximum_iterations','optimizer_ftol','optimizer_gtol']:
        assert F.RECIPE[key]==F.OLD.RECIPE[key]
    assert F.RECIPE['normal_angle_correlation']==0. and F.RECIPE['bias_correction']==0.
