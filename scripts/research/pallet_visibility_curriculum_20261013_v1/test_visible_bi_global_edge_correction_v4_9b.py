"""Small actual-gradient/state/frozen-input contracts for the new producer."""
from copy import deepcopy
import numpy as np
import pytest
from . import visible_bi_global_edge_correction_v4_9b as G


class Poison:
    def __array__(self,*args,**kwargs):raise AssertionError('Excluded coordinate converted')
    def __deepcopy__(self,memo):raise AssertionError('Excluded coordinate copied')


def fixture():
    image=np.zeros((160,160,3),np.uint8)+40
    image[:60,60:]=110;image[60:,:60]=180;image[60:,60:]=230
    mask=np.ones((160,160),np.float32)
    xy=[[53.,55.],[120.,60.],[120.,120.],[60.,120.],[60.,12.],[120.,12.],[100.,25.],[50.,130.]]
    states=['I','B','B','B','B','U','S','O']
    geometry=dict(image_hw=[160,160],initial_self_hidden=[False]*6+[True,False],initial_projection_xy=deepcopy(xy))
    fit=[dict(native_corner=k,state=states[k],xy=xy[k],covariance_px2=[[4.,0.],[0.,4.]],
        fit_covariance_px2=[[25.,0.],[0.,25.]],fit_reliability_weight=.8,
        source='VISIBLE_BACKBONE_FALLBACK' if k==0 else 'B_TARGET_BACKGROUND_LINES',
        image_measurement=True,measurement_is_pose_projection=False) for k in range(5)]
    return image,mask,geometry,states,fit


def test_synthetic_internal_RGB_junction_and_frozen_fit():
    args=fixture();before=deepcopy(args[-1]);r=G.correct_visible_bi_global_edges(*args)
    assert r['changed_native_ids']==[0]
    m=next(m for m in r['visible_measurements'] if m['native_corner']==0)
    assert np.linalg.norm(np.asarray(m['xy'])-[59.5,59.5])<1.3
    assert m['source']=='I_GLOBAL_ACTUAL_RGB_EDGE_INTERSECTION'
    assert m['fit_covariance_px2']==before[0]['fit_covariance_px2'] and m['fit_reliability_weight']==.8
    assert args[-1]==before
    assert r['records'][0]['finite_support_guard']['passed']
    assert all(m==before[m['native_corner']] for m in r['visible_measurements'] if m['native_corner']!=0)


def test_excluded_S_U_O_coordinates_never_copied_or_queried():
    args=list(fixture());fit=args[-1]
    for k in (5,6,7):fit.append(dict(native_corner=k,state=args[3][k],xy=Poison(),covariance_px2=Poison()))
    args[2]['initial_projection_xy'][6]=Poison()
    r=G.correct_visible_bi_global_edges(*args)
    assert not any(k in r['queried_native_ids'] for k in (5,6,7))
    assert r['S_coordinate_reads']==r['U_coordinate_reads']==r['O_T_coordinate_reads']==0
    assert all(m['native_corner']<5 for m in r['visible_measurements'])


def test_blank_RGB_abstains_without_fabricated_full_point():
    args=list(fixture());args[0][:]=70
    r=G.correct_visible_bi_global_edges(*args)
    assert r['changed_native_ids']==[] and r['visible_measurements']==args[-1]
    assert not r['records'][0]['corrected']


def test_internal_ridge_outside_target_mask_is_rejected():
    args=list(fixture());args[1][:]=0
    r=G.correct_visible_bi_global_edges(*args)
    assert r['changed_native_ids']==[]


def test_endpoint_probes_are_actual_inliers_and_unknown_config_rejected():
    r=G.correct_visible_bi_global_edges(*fixture())
    record=r['records'][0]
    assert record['corrected']
    for line in record['selected_rays']:
        assert line['endpoint_support_measured_separately']
        assert len(line['observed_points'])>=len(line['central_observed_points'])
        assert line['corrected_endpoint_normal_residual_px']
    with pytest.raises(ValueError,match='Unknown'):G.correct_visible_bi_global_edges(*fixture(),configuration={'GT_center':True})


def test_long_edge_contradiction_requires_corrected_endpoint_agreement():
    line=dict(normal=[0.,1.],offset=-20.,actual_inlier_tangent_interval_px=[-100.,-30.],edge=1)
    assert not G._support(np.array([120.,20.]),[line],11.)['passed']
    assert G._support(np.array([105.,20.]),[line],11.)['passed']


def test_short_edge_endpoint_validation_does_not_enter_central_fit():
    result=G.correct_visible_bi_global_edges(*fixture())
    for line in result["edge_records"]:
        if line["supported"]:
            assert line["central_sample_count"]==G.DEFAULT_CONFIGURATION["along_samples"]
            assert len(line["central_fraction_grid"])==15
            assert all(not np.isclose(t,line["central_fraction_grid"],atol=1e-12,rtol=0).any() for t in line["endpoint_validation_fraction_grid"])
