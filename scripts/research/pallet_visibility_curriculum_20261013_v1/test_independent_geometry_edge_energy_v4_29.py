"""Analytical scalar checks; no dataset, consumer or new calibration."""
from copy import deepcopy
import numpy as np
import pytest

from . import independent_geometry_edge_energy_v4_29 as A
from . import projected_native_edge_factor_v4_17 as OLD
from . import joint_ot_pnp as V


def edge():
    return dict(native_corner=0, native_neighbor=1, normal_unit_xy=[0., 1.],
                offset_px=-3., support_anchor_xy=[2., 3.], normal_variance_px2=4.,
                angular_variance_rad2=.04, normal_angle_correlation=.2,
                fit_reliability_weight=.7)


def test_closed_form_radial_point_and_rank_two_normal_energy():
    q=np.zeros((8,2));q[0]=[3.,4.]
    m=dict(native_corner=0,xy=[0.,0.],covariance_px2=np.eye(2),fit_reliability_weight=.5)
    expected=.5*18.*(np.sqrt(1.+25./9.)-1.)
    assert A.point_energy(q,m,3.) == pytest.approx(expected)
    b=dict(native_corner=0,fit_reliability_weight=.5,lines=[
        dict(normal_unit_xy=[1.,0.],offset_px=0.,normal_variance_px2=1.),
        dict(normal_unit_xy=[0.,1.],offset_px=0.,normal_variance_px2=1.)])
    assert A.legacy_normal_energy(q,b,3.) == pytest.approx(expected)


def test_scalar_line_matches_frozen_primitive_under_endpoint_and_coefficient_signs():
    xyz=np.asarray([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
                    [0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.]])
    K=np.asarray([[100.,0.,40.],[0.,100.,30.],[0.,0.,1.]])
    R=np.eye(3);t=np.asarray([.1,.2,3.]);scale=2.
    p=V._pose_vector(R,t,scale)
    q=V.project_native(xyz,R,t,K)[0]
    for reverse in (False,True):
        for coefficient in (-1.,1.):
            e=edge()
            if reverse:e.update(native_corner=1,native_neighbor=0)
            e['normal_unit_xy']=(np.asarray(e['normal_unit_xy'])*coefficient).tolist()
            e['offset_px']*=coefficient
            expected=OLD.projected_edge_value_and_gradient(p,xyz,K,scale,e,weight=.7,huber_scale=3.)[0]
            assert A.actual_line_energy(q,e,3.)['energy']==pytest.approx(expected,rel=1e-12,abs=1e-12)


def test_directed_support_cannot_rescue_a_native_endpoint_flip():
    q=np.zeros((8,2));q[0]=[0.,0.];q[1]=[10.,0.]
    g=dict(native_corner=0,native_neighbor=1,normal_unit_xy=[0.,1.],
           actual_inlier_xy=[[x,0.]for x in range(1,10)],
           activation_owners=[dict(native_corner=0,fixed_tangent_sign=-1.),
                              dict(native_corner=1,fixed_tangent_sign=1.)])
    good=A.original_support_slacks(q,g)
    assert all(min(x['slacks'])>=0 for x in good['directed'])
    flipped=q.copy();flipped[[0,1]]=flipped[[1,0]]
    bad=A.original_support_slacks(flipped,g)
    assert all(min(x['slacks'])<0 for x in bad['directed'])
    assert not bad['sign_from_pose']


def test_model_order_ambiguity_does_not_create_fixed_support_evidence():
    q=np.zeros((8,2));q[1]=[10.,0.]
    g=dict(native_corner=0,native_neighbor=1,normal_unit_xy=[0.,1.],
           actual_inlier_xy=[[1.,0.],[9.,0.]],
           activation_owners=[dict(native_corner=0,fixed_tangent_sign=None)])
    assert A.original_support_slacks(q,g)['directed'][0]['slacks']==[]
