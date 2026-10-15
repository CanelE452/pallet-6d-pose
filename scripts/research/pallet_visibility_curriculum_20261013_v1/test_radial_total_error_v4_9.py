from copy import deepcopy
import numpy as np
import pytest

from .statewise_radial_total_error_v4_9 import (
    _basis, _effective_covariance, fit_radial_total_error_model,
    apply_radial_total_error_model)


def _training():
    points=[];cases=[]
    for session in ('a','b','c','d'):
        for frame in range(12):
            rid=f'{session}{frame}'
            cases.append(dict(id=rid,partition='CAL',session=session,compatible=True,nonself_scale_px=100.))
            for k in range(2):
                points.append(dict(id=rid,partition='CAL',session=session,compatible=True,partial_B=False,
                    state='B',group='B:corrected',reference_error_rt_normalized=[.02,.003],native_corner=k))
    return points,cases


def _geometry():
    from .statewise_uncertainty_calibration_v4 import _cuboid
    return dict(native_xyz=_cuboid([1.,.2,1.]).tolist(),R_cf=np.eye(3).tolist(),centroid=[0.,0.,5.],
        K=[[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]],initial_self_hidden=[False]*7+[True])


def test_known_radial_bias_removed_without_moving_archive_or_noise_inputs():
    points,cases=_training();model=fit_radial_total_error_model(points,cases)
    g=_geometry();basis,reference,_=_basis(g,0);biased=reference+basis.T@np.array([2.,.3])
    source=dict(native_corner=0,state='B',source='B_TARGET_BACKGROUND_LINES',xy=biased.tolist(),
        covariance_px2=(np.eye(2)*.5).tolist(),fit_covariance_px2=(np.eye(2)*9.).tolist(),fit_reliability_weight=.8)
    original=deepcopy(source)
    result=apply_radial_total_error_model([source],g,['B']*7+['S'],model,nonself_scale_px=100.)['visible_measurements'][0]
    assert np.allclose(result['xy'],reference)
    assert source==original and result['original_xy_before_CAL_bias']==original['xy']
    assert result['fit_reliability_weight']==.8


def test_S_coordinate_poison_rejected_and_partial_reference_exact():
    points,cases=_training();model=fit_radial_total_error_model(points,cases);g=_geometry()
    class Poison(dict):
        def __getitem__(self,key):
            if key in ('xy','covariance_px2'):raise AssertionError('Prohibited S coordinate read')
            return super().__getitem__(key)
    with pytest.raises(ValueError,match='Initial S'):
        apply_radial_total_error_model([Poison(native_corner=7,state='S')],g,['B']*7+['S'],model,nonself_scale_px=100.)
    partial=dict(native_corner=0,state='B',source='VISIBLE_BACKBONE_FALLBACK',xy=[123.,456.],covariance_px2=[[4.,0.],[0.,4.]])
    result=apply_radial_total_error_model([partial],g,['B']*7+['S'],model,nonself_scale_px=100.,partial_B_ids=[0])
    assert result['visible_measurements']==[partial]
    assert not result['records'][0]['applied']


def test_heldout_session_and_TEST_rejection_cover_all_fit_statistics():
    points,cases=_training()
    for point in points:
        if point['session']=='d':point['reference_error_rt_normalized']=[20.,-30.]
    model=fit_radial_total_error_model(points,cases,heldout_session='d')
    assert not any(i.startswith('d') for i in model['fit_ids'])
    assert np.allclose(model['pooled']['bias_rt_normalized'],[.02,.003])
    assert np.linalg.eigvalsh(model['pooled']['total_scatter_rt_normalized']).max()<.001
    points[0]['partition']='TEST'
    with pytest.raises(ValueError,match='CAL records only'):fit_radial_total_error_model(points,cases)


def test_effective_total_covariance_keeps_conditional_PSD_lower_bound():
    conditional=np.array([[4.,1.2],[1.2,2.]])
    model_total=np.array([[.1,0.],[0.,30.]])
    total=_effective_covariance(model_total,conditional)
    assert np.linalg.eigvalsh(total-conditional).min()>-1e-10
    assert np.linalg.eigvalsh(total-model_total).min()>-1e-10
    assert np.linalg.eigvalsh(total).min()>0
