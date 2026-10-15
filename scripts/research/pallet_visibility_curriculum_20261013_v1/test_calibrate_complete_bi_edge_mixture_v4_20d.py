"""Support-prior noise math only; no real CAL labels, images or pose fits."""
from copy import deepcopy
import numpy as np
import pytest
from . import calibrate_complete_bi_edge_mixture_v4_20d as F
from .test_calibrate_native_edge_mixture_v4_18 import component as base_component


def component(normal,sine,support='a',variant='0',prior=1.):
    return {**base_component(normal,sine), 'support_id':support,'fit_variant_id':variant,
        'component_id':support+':'+variant,'mixture_component_prior':prior}


def row(fid,session,components,compatible=True):
    return dict(id=fid,partition='CAL',session=session,compatible=compatible,groups=[dict(components=components)])


def low(c):
    return dict(error=np.array([c['normal_error_px'],c['angular_sine_error']]),
        floor=np.asarray(c['conditional_variance_floor']),mixture_component_prior=c['mixture_component_prior'])


def test_weighted_uncentered_volume_NLL_and_analytic_gradient():
    group=[low(component(3.,.1,prior=.25)),low(component(-5.,-.05,variant='1',prior=.25)),
        low(component(8.,.2,support='b',prior=.5))]
    variance=np.array([2.,.003]);value,gradient=F.mixture_noise_nll_and_gradient(variance,[group],[.7])
    numeric=[]
    for j,step in enumerate([1e-5,1e-7]):
        delta=np.zeros(2);delta[j]=step
        numeric.append((F.mixture_noise_nll_and_gradient(variance+delta,[group],[.7])[0]
            -F.mixture_noise_nll_and_gradient(variance-delta,[group],[.7])[0])/(2*step))
    assert np.allclose(gradient,numeric,atol=1e-5,rtol=1e-6)
    zero=[[dict(error=np.zeros(2),floor=np.array([1.,.01]),mixture_component_prior=1.)]]
    assert F.mixture_noise_nll_and_gradient([0.,0.],zero,[1.])[0] < F.mixture_noise_nll_and_gradient([1e4,1e4],zero,[1.])[0]-20.
    # Splitting an identical density's prior does not increase evidence.
    duplicated=[{**deepcopy(c),'mixture_component_prior':c['mixture_component_prior']/5}for c in group for _ in range(5)]
    vv,gg=F.mixture_noise_nll_and_gradient(variance,[duplicated],[.7])
    assert vv==pytest.approx(value,abs=1e-12)and np.allclose(gg,gradient,atol=1e-12)


def test_conditioning_scales_use_same_hierarchical_prior_not_uniform_variants():
    components=[component(i,i*.01,variant=str(i),prior=.125)for i in range(1,5)]
    components.append(component(20.,.2,support='b',prior=.5))
    for c in components:
        c['actual_inlier_span_px']=200.
        c['conditional_variance_floor']=F.conditional_covariance_floors(200.,c['normal_floor_px']).tolist()
    model=F.fit_native_edge_mixture_model([row('a','A',components)])
    assert model['optimizer_conditioning_variance_scales']==pytest.approx([(4./.67449)**2,(.04/.67449)**2])
    assert model['fit_alternative_count']==5 and model['fit_distinct_support_count']==2
    assert len(model['optimizer_searches'])==3
    assert model['recipe']['optimizer_scaled_variance_starts']==F.OLD.RECIPE['optimizer_scaled_variance_starts']


def test_excluded_or_nonCAL_rows_gate_before_error_reads_and_full_inventory_kept():
    class Poison(dict):
        def __getitem__(self,key):raise AssertionError('Excluded error/group read')
    records=[row('a','A',[component(2.,.05)]),dict(id='held',session='B',partition='CAL',compatible=True,groups=Poison()),
        dict(id='incomp',session='C',partition='CAL',compatible=False,groups=Poison())]
    model=F.fit_native_edge_mixture_model(records,heldout_session='B')
    assert model['fit_ids']==['a']and model['inventory_frame_count']==3 and model['excluded_heldout_ids']==['held']
    with pytest.raises(ValueError,match='CAL only'):F.fit_native_edge_mixture_model([dict(partition='TEST',groups=Poison())])


def test_runtime_export_only_components_new_schema_rejects_old_model():
    records=[row('a','A',[component(3.,.05)]),row('b','B',[component(-3.,-.05)])]
    model=F.fit_native_edge_mixture_model(records);params=F.mixture_noise_parameters(model)
    assert params['noise_provenance']['observation_family']==F.AUTH.NOISE_OBSERVATION_FAMILY
    assert params['normal_variance_component_px2']==pytest.approx(model['sigma_normal_px']**2)
    assert params['angular_variance_component_sine2']==pytest.approx(model['sigma_sine']**2)
    assert not params['noise_provenance']['archived_complete_tangent_calibrated']
    tampered=deepcopy(model);tampered['normal_variance_component_px2']+=1.
    with pytest.raises(ValueError,match='digest'):F.mixture_noise_parameters(tampered)
    old=F.OLD.fit_native_edge_mixture_model([{**r,'groups':[dict(components=[base_component(3.,.05)])]}for r in records])
    with pytest.raises(ValueError):F.mixture_noise_parameters(old)


def test_unequal_support_mass_duplicate_variant_or_old_uniform_format_rejected():
    with pytest.raises(ValueError):F.fit_native_edge_mixture_model([row('a','A',[base_component(1.,.1)])])
    unequal=[component(1.,.1,prior=.5),component(2.,.1,variant='1',prior=.25),component(3.,.1,support='b',prior=.25)]
    with pytest.raises(ValueError,match='Equal support'):F.fit_native_edge_mixture_model([row('a','A',unequal)])
    duplicated=[component(1.,.1,prior=.5),component(1.,.1,prior=.5)]
    with pytest.raises(ValueError,match='duplicate'):F.fit_native_edge_mixture_model([row('a','A',duplicated)])


def test_equal_session_frame_edge_weights_preserve_all_components_without_mutation():
    records=[row('a','A',[component(2.,.1)]),row('b','B',[component(3.,.1)]),row('c','B',[component(4.,.15)])]
    records.append(dict(id='unsupported',partition='CAL',session='B',compatible=True,groups=[]))
    before=deepcopy(records);model=F.fit_native_edge_mixture_model(records)
    assert records==before and model['session_weights']=={'A':.5,'B':.5}
    assert model['fit_weight_sum']==pytest.approx(1.)and model['unsupported_training_ids']==['unsupported']
    assert model['recipe']['normal_angle_correlation']==0. and model['archived_complete_tangent_total_error_not_calibrated']
