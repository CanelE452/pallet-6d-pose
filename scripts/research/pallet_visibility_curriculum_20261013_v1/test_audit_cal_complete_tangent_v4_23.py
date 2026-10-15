from copy import deepcopy
import numpy as np
import pytest
from . import audit_cal_complete_tangent_v4_23 as A


def fixture():
    return dict(native_corner=2,state='B',point_cost_revoked=True,active_physical_edges=[[2,3]],
        tangent_kind='ARCHIVED_COMPLETE_BI_TANGENT',tangent_unit_xy=[1.,0.],tangent_variance_px2=16.,
        tangent_axis_source='IMMUTABLE_ORIGINAL_OBSERVER_MODEL_GUIDE_SAME_ACROSS_WD',
        original_measurement_archive=dict(original_observed_xy=[4.,6.],original_measurement_source='B_TARGET_BACKGROUND_LINES',
            original_effective_fit_covariance_px2=[[4.,0.],[0.,9.]],original_observed_covariance_px2=[[1.,0.],[0.,1.]],
            original_fit_reliability_weight=.8))


def test_exact_error_and_covariance_chart_and_sigma_squared_four():
    p=fixture();saved=deepcopy(p);r=A.point_record(p,[1.,2.])
    assert p==saved and r['radial_error_px']==5.
    assert r['effective_point_mahalanobis_radius']==pytest.approx(np.sqrt(9/4+16/9))
    assert r['tangent']['absolute_error_px']==3. and r['tangent']['current_sigma_px']==4.
    assert r['tangent']['absolute_normalized_error']==.75 and r['source_group']=='RGB_COMPLETE'
    p['tangent_variance_px2']=64.
    with pytest.raises(ValueError,match='variance x4'):A.point_record(p,[1.,2.])


def test_S_gate_precedes_archived_measurement_access():
    class Poison(dict):
        def __getitem__(self,key):raise AssertionError('S measurement read')
    with pytest.raises(ValueError,match='Only nonS'):
        A.point_record(dict(state='S',original_measurement_archive=Poison()),Poison())


def test_frame_balance_differs_from_point_pool_and_empirical_tangent_coverage():
    rows=[]
    for fid,session,values in [('a','A',[0.,0.]),('b','B',[9.])]:
        for value in values:
            p=fixture();p['original_measurement_archive']['original_observed_xy']=[value,0.]
            r=A.point_record(p,[0.,0.]);r.update(id=fid,session=session);rows.append(r)
    s=A.summarize(rows)
    assert s['points']==3 and s['frames']==2
    assert s['weightings']['POINT_POOLED']['radial_error_px']['mean']==3.
    assert s['weightings']['FRAME_BALANCED']['radial_error_px']['mean']==4.5
    assert s['weightings']['FRAME_BALANCED']['current_tangent_empirical_within_sigma']['1']==.5
    assert s['weightings']['POINT_POOLED']['current_tangent_empirical_within_sigma']['1']==pytest.approx(2/3)
