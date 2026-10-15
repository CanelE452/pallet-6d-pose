"""Separate generation with exactly frozen SOURCE24 math; synthetic only."""
from copy import deepcopy

import numpy as np
import pytest

from . import calibrate_prepartial_RGB_edge_mixture_v4_26 as F
from .test_calibrate_complete_bi_edge_mixture_v4_20d import component,low,row


def test_same_support_balanced_uncentered_cov_volume_NLL_and_gradient():
    group=[low(component(3.,.1,prior=.25)),low(component(-5.,-.05,variant='1',prior=.25)),
        low(component(8.,.2,support='b',prior=.5))]
    variance=np.array([2.,.003])
    value,gradient=F.mixture_noise_nll_and_gradient(variance,[group],[.7])
    old,oldgradient=F.SOURCE.mixture_noise_nll_and_gradient(variance,[group],[.7])
    assert value==old and np.array_equal(gradient,oldgradient)
    numeric=[]
    for j,step in enumerate([1e-5,1e-7]):
        delta=np.zeros(2);delta[j]=step
        numeric.append((F.mixture_noise_nll_and_gradient(variance+delta,[group],[.7])[0]
            -F.mixture_noise_nll_and_gradient(variance-delta,[group],[.7])[0])/(2*step))
    assert np.allclose(gradient,numeric,atol=1e-5,rtol=1e-6)


def test_synthetic_fit_new_generation_old_variance_and_three_starts_exact():
    records=[row('synthetic_a','A',[component(3.,.05)]),row('synthetic_b','B',[component(-4.,-.04)])]
    before=deepcopy(records)
    model=F.fit_native_edge_mixture_model(records)
    old=F.SOURCE.fit_native_edge_mixture_model(records)
    assert records==before and model['schema']==F.SCHEMA and model['schema']!=old['schema']
    for key in ('normal_variance_component_px2','angular_variance_component_sine2',
                'optimizer_searches','optimizer_conditioning_variance_scales','training_mixture_nll'):
        assert model[key]==old[key]
    assert len(model['optimizer_searches'])==3
    exported=F.mixture_noise_parameters(model)
    assert exported['noise_provenance']['observation_family']==F.AUTH.NOISE_OBSERVATION_FAMILY
    assert exported['noise_provenance']['source_policy']==F.AUTH.SOURCE_POLICY
    assert exported['noise_provenance']['tangent_gamma_fitted_by_this_model'] is False
    assert exported['normal_variance_component_px2']==model['sigma_normal_px']**2
    assert exported['angular_variance_component_sine2']==model['sigma_sine']**2
    with pytest.raises(ValueError):F.mixture_noise_parameters(old)


def test_excluded_rows_gate_before_errors_and_inventory_is_never_trimmed():
    class Poison(dict):
        def __getitem__(self,key):raise AssertionError('Excluded CAL group/error accessed')
    records=[row('synthetic_a','A',[component(2.,.05)]),
        dict(id='held',session='B',partition='CAL',compatible=True,groups=Poison()),
        dict(id='incompatible',session='C',partition='CAL',compatible=False,groups=Poison())]
    model=F.fit_native_edge_mixture_model(records,heldout_session='B')
    assert model['fit_ids']==['synthetic_a'] and model['excluded_heldout_ids']==['held']
    assert model['incompatible_inventory_ids']==['incompatible'] and model['inventory_frame_count']==3
    with pytest.raises(ValueError,match='CAL only'):
        F.fit_native_edge_mixture_model([dict(partition='REMAINDER',groups=Poison())])


def test_private_clone_changes_only_recipe_generation_and_paths():
    assert F._PREPARE.__globals__['PREPARED']==F.PREPARED
    assert F._PREPARE.__globals__['AUTH'] is F.AUTH
    assert F._PREPARE.__globals__['__name__']==F.__name__
    assert F._FIT.__globals__['_verify_preparation'] is F._verify_preparation
    assert F._FIT.__globals__['fit_native_edge_mixture_model'] is F.fit_native_edge_mixture_model
    assert F.SOURCE.prepare.__globals__['AUTH'] is F.SOURCE.AUTH
    assert F.SOURCE.fit_native_edge_mixture_model.__globals__['SCHEMA']==F.SOURCE.SCHEMA
    for key in ('component_prior','optimizer_scaled_variance_starts','optimizer_maximum_iterations',
                'optimizer_ftol','optimizer_gtol','huber_scale','normal_angle_correlation'):
        assert F.RECIPE[key]==F.SOURCE.RECIPE[key]


def test_runtime_floor_components_not_doubled_or_relabelled():
    records=[row('synthetic_a','A',[component(3.,.05)])]
    model=F.fit_native_edge_mixture_model(records)
    params=F.mixture_noise_parameters(model)
    for span,rms in ((6.,0.),(15.,1.),(70.,1.25)):
        floors=F.conditional_covariance_floors(span,max(1.,rms))
        assert np.array_equal(floors,F.SOURCE.conditional_covariance_floors(span,max(1.,rms)))
        assert params['normal_variance_component_px2']+floors[0]==pytest.approx(
            model['normal_variance_component_px2']+max(1.,rms)**2)
        assert params['angular_variance_component_sine2']+floors[1]==pytest.approx(
            model['angular_variance_component_sine2']+2*max(1.,rms)**2/(span**2+2*max(1.,rms)**2))
