"""Derivative equivalence with perspective, clipping and arc-weight changes."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .common import cuboid,project
from .joint_ot_pnp import _pose_vector
from . import statewise_global_outline_solver_v4_10 as OLD
from .global_outline_analytic_factor_v4_11 import outline_value_and_gradient,_clip_with_jacobian


def scene(t,rotvec):
    xyz=cuboid(1.3,.11,1.1)
    K=np.array([[590.,3.,320.],[0.,615.,240.],[.0001,0.,1.]])
    R=Rotation.from_rotvec(rotvec).as_matrix();t=np.asarray(t,float)
    mask=np.zeros((480,640),np.uint8)
    xy=project(xyz,R,t,K)
    cv2.fillConvexPoly(mask,np.rint(cv2.convexHull(xy.astype(np.float32))).astype(np.int32),1)
    observation=OLD.observe_global_outline(mask,mask.shape)
    scale=np.linalg.norm(np.ptp(xyz,axis=0));p=_pose_vector(R,t,scale)
    return xyz,K,scale,p,observation


@pytest.mark.parametrize('translation,rotation',[
    ([0.,.3,2.2],[.12,.65,.07]),
    ([-.9,.3,2.2],[.12,.65,.07]),
    ([.9,.3,2.2],[.12,.65,.07]),
    ([0.,-.55,2.2],[.1,.4,-.06]),
])
def test_analytic_gradient_matches_same_objective_central_derivative(translation,rotation):
    xyz,K,scale,p,observation=scene(translation,rotation)
    # Move away from exact pixel/hull contacts to a differentiable local pose.
    p=p+np.array([.013,-.017,.011,.009,-.007,.019])
    value,gradient,audit=outline_value_and_gradient(p,xyz,K,scale,observation)
    expected,_,old_audit=OLD.outline_value_and_gradient(p,xyz,K,scale,observation)
    assert value==pytest.approx(expected,rel=1e-12,abs=1e-10)
    assert audit['model_to_observed']['count']==old_audit['model_to_observed']['count']
    step=1e-7
    finite=np.array([(OLD._outline_value(p+np.eye(6)[j]*step,xyz,K,scale,observation,
        OLD.DEFAULT_OUTLINE_CONFIGURATION)[0]-OLD._outline_value(p-np.eye(6)[j]*step,xyz,K,scale,observation,
        OLD.DEFAULT_OUTLINE_CONFIGURATION)[0])/(2*step) for j in range(6)])
    np.testing.assert_allclose(gradient,finite,rtol=1e-5,atol=2e-4)


def test_clipped_endpoints_jacobians_include_motion_of_clip_parameter():
    a=np.array([-20.,40.]);b=np.array([40.,80.]);ja=np.zeros((2,6));jb=np.zeros((2,6))
    ja[:,0]=[3.,2.];jb[:,0]=[-2.,5.];bounds=((7.,100.),(7.,100.))
    clipped=_clip_with_jacobian(a,b,ja,jb,bounds)
    step=1e-6
    plus=OLD._clip(a+step*ja[:,0],b+step*jb[:,0],bounds)
    minus=OLD._clip(a-step*ja[:,0],b-step*jb[:,0],bounds)
    np.testing.assert_allclose(clipped[2][:,0],(plus[0]-minus[0])/(2*step),atol=1e-7)
    np.testing.assert_allclose(clipped[3][:,0],(plus[1]-minus[1])/(2*step),atol=1e-7)
