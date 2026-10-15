"""Outside-region semantics, finite derivatives, and forbidden-read gates."""
from copy import deepcopy
import numpy as np
import pytest

from . import bounded_o_localization_prior_v4_21 as P


def fixture(offset=(6.,4.)):
    xyz=np.array([[-.5,-.1,-.7],[.5,-.1,-.7],[.5,.1,-.7],[-.5,.1,-.7],
                  [-.5,-.1,.7],[.5,-.1,.7],[.5,.1,.7],[-.5,.1,.7]])
    K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    p=np.array([.15,-.1,.05,.1,.1,3.])
    q=P.J.projection_and_jacobian(p,xyz,K,1.)[0]
    record=dict(native_corner=0,state='O',shape='ellipse',localization_available=True,
        independent_precise_support=False,center_xy=(q[0]-offset).tolist(),
        half_axes_px=[4.,7.],angle_deg=23.,source='archived_predictor_localization')
    return xyz,K,p,record


def test_zero_anywhere_inside_is_not_point_correspondence():
    xyz,K,p,record=fixture((.2,.3));saved=deepcopy(record)
    regions=P.prepare_o_localization_regions([record],['O']+['U']*7,[False]*8)
    for dx in (0.,.001):
        x=p.copy();x[3]+=dx
        value,gradient,audit=P.bounded_o_localization_value_and_gradient(x,xyz,K,1.,regions,epsilon=.1)
        assert value==0. and np.array_equal(gradient,np.zeros(6))
        assert audit['O_point_terms']==0 and audit['inside_ellipse_penalty_exactly_zero']
    assert record==saved


def test_rotated_anisotropic_outside_region_gradient():
    xyz,K,p,record=fixture()
    regions=P.prepare_o_localization_regions([record],['O']+['U']*7,[False]*8)
    value,gradient,audit=P.bounded_o_localization_value_and_gradient(p,xyz,K,1.,regions,epsilon=.05)
    assert value>0 and audit['regions'][0]['outside_residual']>0
    finite=[]
    for j in range(6):
        step=np.zeros(6);step[j]=1e-6
        high=P.bounded_o_localization_value_and_gradient(p+step,xyz,K,1.,regions,epsilon=.05)[0]
        low=P.bounded_o_localization_value_and_gradient(p-step,xyz,K,1.,regions,epsilon=.05)[0]
        finite.append((high-low)/2e-6)
    np.testing.assert_allclose(gradient,finite,rtol=2e-5,atol=2e-5)


def test_single_frame_cap_not_multiplied_by_region_count():
    xyz,K,p,record=fixture((1000.,1000.))
    second=deepcopy(record);second['native_corner']=1
    regions=P.prepare_o_localization_regions([record,second],['O','O']+['U']*6,[False]*8)
    value,gradient,audit=P.bounded_o_localization_value_and_gradient(p,xyz,K,1.,regions,epsilon=.2)
    assert 0<=value<=-2*np.log(.2)+1e-12
    assert np.isclose(value,-2*np.log(.2)) and np.isfinite(gradient).all()
    assert audit['region_count']==2 and not audit['independence_or_pose_probability_claim']


def test_forbidden_states_gate_before_coordinates_and_empty_is_zero():
    class Poison:
        def __array__(self,*args):raise AssertionError('Forbidden coordinate read')
        def __deepcopy__(self,memo):raise AssertionError('Forbidden source copy')
    for state in ('S','B','I','U','T'):
        with pytest.raises(ValueError,match='Only non-S external O'):
            P.prepare_o_localization_regions([dict(native_corner=0,state=state,center_xy=Poison())],
                [state]+['U']*7,[state=='S']+[False]*7)
    value,gradient,_=P.bounded_o_localization_value_and_gradient(Poison(),Poison(),Poison(),1.,[],epsilon=.1)
    assert value==0 and np.array_equal(gradient,np.zeros(6))
