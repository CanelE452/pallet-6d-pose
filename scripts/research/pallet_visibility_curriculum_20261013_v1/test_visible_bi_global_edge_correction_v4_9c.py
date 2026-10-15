"""Focused contracts for all-visible reevaluation, with no dataset labels."""
from copy import deepcopy
import numpy as np
import pytest
from . import visible_bi_global_edge_correction_v4_9c as G
from .test_visible_bi_global_edge_correction_v4_9b import fixture,Poison


def test_existing_corrected_point_can_be_remeasured_from_two_RGB_edges():
    args=list(fixture());args[-1][0]['source']='I_TARGET_LINES_ALREADY_CORRECTED'
    before=deepcopy(args[-1]);out=G.correct_visible_bi_global_edges(*args)
    assert out['changed_native_ids']==[0]
    record=out['records'][0]
    assert record['original_corrected'] and record['original_source']=='I_TARGET_LINES_ALREADY_CORRECTED'
    assert np.linalg.norm(np.asarray(record['measured_xy'])-[59.5,59.5])<1.3
    assert all(ray['independent_both_endpoint_support']['passed'] for ray in record['selected_rays'])
    assert args[-1]==before
    m=out['visible_measurements'][0]
    assert m['fit_covariance_px2']==before[0]['fit_covariance_px2'] and m['fit_reliability_weight']==before[0]['fit_reliability_weight']


def test_unsupported_already_corrected_rows_exactly_preserved():
    args=list(fixture());args[0][:]=70;before=deepcopy(args[-1])
    out=G.correct_visible_bi_global_edges(*args)
    assert out['changed_native_ids']==[] and out['visible_measurements']==before
    assert out['queried_native_ids']==[0,1,2,3,4]


def test_S_U_O_T_gates_before_measurement_and_projection_reads():
    args=list(fixture());args[3][5]='T'
    for k in (5,6,7):
        args[-1].append(dict(native_corner=k,state=args[3][k],xy=Poison(),covariance_px2=Poison()))
        args[2]['initial_projection_xy'][k]=Poison()
    out=G.correct_visible_bi_global_edges(*args)
    assert all(k<5 for k in out['queried_native_ids'])
    assert out['S_coordinate_reads']==out['U_coordinate_reads']==out['O_T_coordinate_reads']==0


def test_two_endpoints_must_have_separate_actual_validation_support():
    f=np.array([.05,.1,.5,.9,.95]);central=np.array([0,1,1,1,0],bool)
    assert not G._endpoint_support(f,central,np.arange(5),[1,1,1,1,0])['passed']
    assert G._endpoint_support(f,central,np.arange(5),[1,1,1,1,1])['passed']
    assert not G._endpoint_support(f,np.ones(5,bool),np.arange(5),np.ones(5,bool))['passed']


def test_refined_full_lines_use_central15_and_actual_finite_support():
    out=G.correct_visible_bi_global_edges(*fixture())
    for ray in out['records'][0]['selected_rays']:
        assert ray['central_sample_count']==15
        assert ray['independent_both_endpoint_support']['first_endpoint_inliers']>=1
        assert ray['independent_both_endpoint_support']['second_endpoint_inliers']>=1
        assert all(not np.isclose(t,ray['central_fraction_grid'],atol=1e-12,rtol=0).any() for t in ray['endpoint_validation_fraction_grid'])
    assert out['records'][0]['finite_support_guard']['passed']


def test_ownership_context_required_and_no_mask_gradient_fake_point():
    args=list(fixture());args[1][:]=0
    out=G.correct_visible_bi_global_edges(*args)
    assert out['changed_native_ids']==[] and out['visible_measurements']==args[-1]


def test_unknown_GT_or_ID_option_rejected():
    with pytest.raises(ValueError,match='Unknown'):
        G.correct_visible_bi_global_edges(*fixture(),configuration={'GT_xy':[0,0]})
