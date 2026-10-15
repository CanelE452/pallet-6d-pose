from copy import deepcopy
import numpy as np
import pytest
from . import calibrate_archived_complete_tangent_v4_24 as F


def point(error=3.,axis=9.,state='B',source='B_TARGET_BACKGROUND_LINES'):
    return dict(native_corner=2,state=state,source=source,source_group='RGB_COMPLETE',
        signed_tangent_error_px=error,original_variance_axis_px2=axis,exact_saved_old_tangent_variance_px2=4.*axis)


def row(fid,session,points,compatible=True):
    return dict(id=fid,partition='CAL',session=session,compatible=compatible,tangents=points)


def test_scalar_analytic_gradient_and_logvariance_and_error_sign():
    errors=np.array([2.,-7.,.2]);axes=np.array([3.,40.,.5]);weights=np.array([.2,.3,.5]);gamma=.7
    value,gradient=F.scalar_tangent_nll_and_gradient(gamma,errors,axes,weights)
    eps=1e-6
    numeric=(F.scalar_tangent_nll_and_gradient(gamma+eps,errors,axes,weights)[0]
        -F.scalar_tangent_nll_and_gradient(gamma-eps,errors,axes,weights)[0])/(2*eps)
    assert gradient==pytest.approx(numeric,rel=1e-7,abs=1e-8)
    assert F.scalar_tangent_nll_and_gradient(gamma,-errors,axes,weights)==pytest.approx((value,gradient))
    assert F.scalar_tangent_nll_and_gradient(0.,[0.],[9.],[1.])[0] < F.scalar_tangent_nll_and_gradient(1e8,[0.],[9.],[1.])[0]
    with pytest.raises(ValueError):F.scalar_tangent_nll_and_gradient(-1.,errors,axes,weights)


def test_zero_errors_select_boundary_gamma_zero_and_pure_runtime_variance():
    records=[row('a','A',[point(0.)]),row('b','B',[point(0.,4.)])];saved=deepcopy(records)
    model=F.fit_archived_complete_tangent_model(records)
    assert records==saved and model['gamma']==pytest.approx(0.,abs=1e-12)
    assert len(model['optimizer_searches'])==3 and model['bias_correction_px']==0.
    t=np.array([.6,.8]);cov=np.array([[4.,1.],[1.,9.]]);tc=t.copy();cc=cov.copy()
    output=F.calibrated_archived_tangent_variance(model,t,cov)
    flipped=F.calibrated_archived_tangent_variance(model,-t,cov)
    assert output['tangent_variance_px2']==1. and output==flipped
    assert output['exact_old_x4_variance_px2']==pytest.approx(4*(t@cov@t))
    assert np.array_equal(t,tc)and np.array_equal(cov,cc)
    bad=deepcopy(model);bad['gamma']+=1
    with pytest.raises(ValueError,match='digest'):F.tangent_variance_parameters(bad)
    with pytest.raises(ValueError):F.calibrated_archived_tangent_variance(model,[2.,0.],cov)


def test_session_frame_point_equal_and_no_error_or_source_trimming():
    records=[row('a','A',[point(2.),point(20.)]),row('b','B',[point(-4.)]),row('c','B',[point(6.,state='I',source='VISIBLE_BACKBONE_FALLBACK')]),
        row('empty','B',[])]
    _,_,_,_,_,pp,weights=F._training(records)
    assert weights.tolist()==[.25,.25,.25,.25]and len(pp)==4
    model=F.fit_archived_complete_tangent_model(records)
    assert model['fit_tangent_count']==4 and model['unsupported_training_ids']==['empty']
    assert model['source_counts']['VISIBLE_BACKBONE_FALLBACK']==1 and model['state_counts']=={'B':3,'I':1}
    assert model['session_weights']=={'A':.5,'B':.5}


def test_excluded_CAL_and_partition_gates_before_numeric_errors():
    class Poison(dict):
        def __getitem__(self,key):raise AssertionError('Excluded numeric field read')
    records=[row('a','A',[point()]),row('held','B',Poison()),row('incomp','C',Poison(),False)]
    model=F.fit_archived_complete_tangent_model(records,heldout_session='B')
    assert model['fit_ids']==['a']and model['excluded_heldout_ids']==['held']and model['inventory_frame_count']==3
    with pytest.raises(ValueError,match='CAL only'):F.fit_archived_complete_tangent_model([dict(partition='TEST',tangents=Poison())])
    assert F._prepared_tangent(dict(native_corner=2,state='S',original_measurement_archive=Poison()),[False,False,True,False,False,False,False,False])is None


def test_matched_old_control_exactly_four_axis_not_floor_plus_four_axis():
    errors=np.array([3.,-4.]);axes=np.array([2.,5.]);weights=np.array([.5,.5]);saved=weights.copy()
    old=F._descriptive(errors,axes,weights,4.*axes)
    gamma4=F.scalar_tangent_nll_and_gradient(4.,errors,axes,weights)[0]
    assert not np.isclose(old['uncentered_NLL'],gamma4)
    assert np.array_equal(weights,saved)
    assert F.PIXEL_FLOOR_VARIANCE_PX2==1.
