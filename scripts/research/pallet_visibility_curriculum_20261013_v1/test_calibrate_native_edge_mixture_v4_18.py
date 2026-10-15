from copy import deepcopy

import numpy as np
import pytest

from . import calibrate_native_edge_mixture_v4_18 as F


def component(normal, sine, span=20., rms=.2):
    return dict(normal_error_px=normal, angular_sine_error=sine, actual_inlier_span_px=span,
        normal_floor_px=max(1.,rms), conditional_variance_floor=F.conditional_covariance_floors(span,rms).tolist())


def row(fid,session,components,compatible=True):
    return dict(id=fid,partition='CAL',session=session,compatible=compatible,
        groups=[dict(components=components)])


def test_uncentered_NLL_covariance_volume_prevents_unbounded_noise():
    groups=[[dict(error=np.zeros(2),floor=np.array([1.,.01]))]]
    narrow,_=F.mixture_noise_nll_and_gradient([0.,0.],groups,[1.])
    broad,_=F.mixture_noise_nll_and_gradient([10000.,10000.],groups,[1.])
    assert narrow < broad-20.
    model=F.fit_native_edge_mixture_model([row('a','A',[component(0.,0.)])])
    assert model['normal_variance_component_px2']==pytest.approx(0.,abs=1e-8)
    assert model['angular_variance_component_sine2']==pytest.approx(0.,abs=1e-8)
    assert not model['recipe']['runtime_peak_centering_used_for_scale_fit']


def test_full_mixture_variance_gradient_and_no_duplicate_list_bonus():
    groups=[[dict(error=np.array([3.,.1]),floor=np.array([1.,.002])),
        dict(error=np.array([-5.,-.05]),floor=np.array([1.44,.008]))]]
    variance=np.array([2.,.003]);value,gradient=F.mixture_noise_nll_and_gradient(variance,groups,[.7])
    numeric=[]
    for j,step in enumerate([1e-5,1e-7]):
        delta=np.zeros(2);delta[j]=step
        numeric.append((F.mixture_noise_nll_and_gradient(variance+delta,groups,[.7])[0]
            -F.mixture_noise_nll_and_gradient(variance-delta,groups,[.7])[0])/(2*step))
    assert np.allclose(gradient,numeric,rtol=1e-6,atol=1e-5)
    duplicate=F.mixture_noise_nll_and_gradient(variance,[groups[0]*5],[.7])
    assert duplicate[0]==pytest.approx(value,abs=1e-12)
    assert np.allclose(duplicate[1],gradient,atol=1e-12)


def test_same_runtime_RMS_span_floors_and_component_only_export():
    floor=F.conditional_covariance_floors(20.,1.2)
    assert floor[0]==pytest.approx(1.44)
    assert floor[1]==pytest.approx(np.sin(np.arctan2(np.sqrt(2.)*1.2,20.))**2)
    assert F.conditional_covariance_floors(20.,0.).tolist()==F.conditional_covariance_floors(20.,1.).tolist()
    model=F.fit_native_edge_mixture_model([row('a','A',[component(3.,.05)]),row('b','B',[component(-3.,-.05)])])
    params=F.mixture_noise_parameters(model)
    assert params['normal_variance_component_px2']==pytest.approx(model['sigma_normal_px']**2)
    assert params['angular_variance_component_sine2']==pytest.approx(model['sigma_sine']**2)
    assert params['normal_angle_correlation']==0.
    bad=deepcopy(model);bad['sigma_sine']+=.1
    with pytest.raises(ValueError,match='digest'):F.mixture_noise_parameters(bad)


def test_heldout_incompatible_and_other_partition_gate_before_error_fields():
    class Poison(dict):
        def __getitem__(self,key):raise AssertionError('Excluded CAL error read')
    records=[row('a','A',[component(2.,.05)]),row('held','B',[component(0.,0.)]),
        row('incompatible','C',[component(0.,0.)],False)]
    records[1]['groups']=[Poison()];records[2]['groups']=[Poison()]
    model=F.fit_native_edge_mixture_model(records,heldout_session='B')
    assert model['fit_ids']==['a'] and model['excluded_heldout_ids']==['held']
    with pytest.raises(ValueError,match='CAL only'):
        F.fit_native_edge_mixture_model([dict(partition='TEST',groups=[Poison()])])


def test_session_frame_edge_weighting_retains_all_alternatives_and_inventory():
    records=[row('a','A',[component(2.,.1),component(20.,.8)]),row('b','B',[component(3.,.1)]),
        row('c','B',[component(4.,.15)])]
    records.append(dict(id='unsupported',partition='CAL',session='B',compatible=True,groups=[]))
    before=deepcopy(records)
    model=F.fit_native_edge_mixture_model(records)
    assert records==before and model['fit_frame_count']==3 and model['fit_alternative_count']==4
    assert model['session_weights']=={'A':.5,'B':.5}
    assert model['unsupported_training_ids']==['unsupported'] and model['inventory_frame_count']==4
    assert model['fit_weight_sum']==pytest.approx(1.)
    assert model['normal_variance_component_px2']>=0 and model['angular_variance_component_sine2']>=0


def test_invalid_sine_floor_inventory_fail_closed():
    with pytest.raises(ValueError):F.fit_native_edge_mixture_model([row('a','A',[component(1.,1.1)])])
    bad=component(1.,.1);bad['conditional_variance_floor']=[1.,0.]
    with pytest.raises(ValueError,match='floors differ'):F.fit_native_edge_mixture_model([row('a','A',[bad])])
    with pytest.raises(ValueError):F.conditional_covariance_floors(0.,.1)
    with pytest.raises(ValueError):F.fit_native_edge_mixture_model([row('a','A',[component(1.,.1)])],inventory_ids=['other'])


def test_structural_S_skip_inventory_and_physical_edge_deduplication():
    class Skip(dict):
        def get(self,key,*args):
            if key=='chains':raise AssertionError('S skip chain fields read')
            return super().get(key,*args)
    observation=dict(edge_records=[Skip(native_corner=6,supported=False),
        dict(native_corner=0,native_neighbor=1,supported=True,chains=[{},{}]),
        dict(native_corner=1,native_neighbor=0,supported=True,chains=[{}])])
    assert F._observation_inventory(observation)==(1,3)
