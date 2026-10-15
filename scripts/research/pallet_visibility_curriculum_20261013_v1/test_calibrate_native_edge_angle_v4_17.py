from copy import deepcopy
import numpy as np
import pytest

from . import calibrate_native_edge_angle_v4_17 as F


def row(fid,session,error,*,compatible=True,span=20.):
    return dict(id=fid,partition='CAL',session=session,compatible=compatible,
        edges=[dict(signed_angular_sine_error=error,actual_inlier_span_px=span)])


def test_session_then_frame_then_edge_equal_weight_without_error_trimming():
    records=[row('a','A',.2),row('b1','B',.01),row('b2','B',.01),row('b3','B',.01),row('c','C',.3)]
    before=deepcopy(records);model=F.fit_native_edge_angle_model(records)
    assert np.isclose(model['sigma_sine'],.2/.67449)
    assert model['fit_edge_count']==5 and model['fit_ids']==[r['id']for r in records]
    assert np.isclose(model['fit_weight_sum'],1.) and records==before


def test_heldout_and_incompatible_errors_are_not_accessed_and_TEST_rejected():
    class Poison(dict):
        def __getitem__(self,key):
            raise AssertionError('Excluded edge error was read')
    records=[row('a','A',.02),row('b','B',.1),row('bad','C',0.,compatible=False)]
    records[1]['edges']=[Poison()];records[2]['edges']=[Poison()]
    model=F.fit_native_edge_angle_model(records,heldout_session='B')
    assert model['fit_ids']==['a'] and model['excluded_heldout_ids']==['b']
    with pytest.raises(ValueError,match='CAL only'):
        F.fit_native_edge_angle_model([dict(partition='TEST',edges=[Poison()])])


def test_exact_SUM_pixel_span_floor_zero_scatter_and_digest_guard():
    model=F.fit_native_edge_angle_model([row('a','A',0.),row('b','B',0.)])
    assert model['sigma_sine']==0.
    assert F.angular_variance_for_edge(model,10.)['angular_variance_rad2']==pytest.approx(.01)
    model=F.fit_native_edge_angle_model([row('a','A',.05),row('b','B',.1)])
    v=F.angular_variance_for_edge(model,10.)
    assert v['angular_variance_rad2']==pytest.approx(model['sigma_sine']**2+.01)
    assert v['normal_angle_correlation']==0. and not v['probability_guarantee']
    altered=deepcopy(model);altered['sigma_sine']*=2
    with pytest.raises(ValueError,match='digest'):
        F.angular_variance_for_edge(altered,10.)


def test_invalid_span_sine_duplicate_and_inventory_fail_closed():
    with pytest.raises(ValueError):F.fit_native_edge_angle_model([row('a','A',1.1)])
    with pytest.raises(ValueError):F.fit_native_edge_angle_model([row('a','A',.1,span=0.)])
    with pytest.raises(ValueError):F.fit_native_edge_angle_model([row('a','A',.1),row('a','B',.2)])
    with pytest.raises(ValueError):F.fit_native_edge_angle_model([row('a','A',.1)],inventory_ids=['a','missing'])


def test_LOGO_excludes_all_held_session_and_keeps_unsupported_inventory():
    records=[row('a','A',.01),row('b','A',.2),row('c','B',.1),row('d','C',.3)]
    records.append(dict(id='none',partition='CAL',session='B',compatible=True,edges=[]))
    model=F.fit_native_edge_angle_model(records,heldout_session='A')
    assert set(model['fit_ids'])=={'c','d'} and model['unsupported_training_ids']==['none']
    assert model['compatible_inventory_count']==5 and model['inventory_frame_count']==5
    assert not(set(model['fit_ids'])&set(model['excluded_heldout_ids']))


def test_AUTH_export_supplies_only_CAL_component_so_floor_is_not_duplicated():
    model=F.fit_native_edge_angle_model([row('a','A',.05),row('b','B',.1)])
    params=F.authentication_angular_parameters(model)
    total=F.angular_variance_for_edge(model,20.)
    assert params['angular_variance_rad2']==model['sigma_sine']**2
    assert params['angular_variance_rad2']+1/20.**2==pytest.approx(total['angular_variance_rad2'])
    assert params['angular_variance_provenance']['fit_partition']=='CAL'
