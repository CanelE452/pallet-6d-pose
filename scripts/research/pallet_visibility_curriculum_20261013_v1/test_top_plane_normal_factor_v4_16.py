import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from . import top_plane_normal_factor_v4_16 as N


def test_normal_factor_analytic_gradient_general_small_and_near_pi():
    coarse=Rotation.from_rotvec([.1,-.2,.05]).as_matrix()
    for phi in ([.3,.1,-.15],[.100000001,-.2,.05],[3.0,.2,.1],[3.6,.1,.2],[-6.2,.1,.1]):
        p=np.r_[phi,[1.,2.,3.]];_,gradient,_=N.normal_value_and_gradient(p,coarse,.08)
        numerical=[]
        for k in range(6):
            step=np.zeros(6);step[k]=1e-6
            numerical.append((N.normal_value_and_gradient(p+step,coarse,.08)[0]-N.normal_value_and_gradient(p-step,coarse,.08)[0])/2e-6)
        assert np.allclose(gradient,numerical,rtol=2e-5,atol=2e-5)


def test_native_yaw_and_translation_have_zero_direct_cost_and_rank_two():
    coarse=Rotation.from_rotvec([.2,.1,-.3]).as_matrix()
    for angle in (.3,np.pi/2,np.pi):
        candidate=coarse@Rotation.from_rotvec([0,angle,0]).as_matrix()
        p=np.r_[Rotation.from_matrix(candidate).as_rotvec(),[80.,-23.,.005]]
        value,gradient,_=N.normal_value_and_gradient(p,coarse,.08)
        assert abs(value)<1e-12 and np.linalg.norm(gradient)<1e-9
    p=np.r_[Rotation.from_matrix(coarse).as_rotvec(),[0.,0.,1.]]
    _,jacobian,_=N.normal_residual_and_jacobian(p,coarse,.08)
    assert np.linalg.matrix_rank(jacobian,tol=1e-9)==2
    assert np.array_equal(jacobian[:,3:],np.zeros((2,3)))


def test_WD_family_normal_factor_is_invariant_and_inputs_immutable():
    coarse=Rotation.from_rotvec([.2,-.3,.1]).as_matrix()
    candidate=Rotation.from_rotvec([.4,.1,-.2]).as_matrix();swap=Rotation.from_rotvec([0,np.pi/2,0]).as_matrix()
    before=coarse.copy();p=np.r_[Rotation.from_matrix(candidate).as_rotvec(),[0.,0.,1.]]
    p2=np.r_[Rotation.from_matrix(candidate@swap.T).as_rotvec(),[40.,20.,12.]]
    value=N.normal_value_and_gradient(p,coarse,.1)[0]
    assert np.isclose(value,N.normal_value_and_gradient(p2,coarse@swap.T,.1)[0],atol=1e-10)
    assert np.array_equal(coarse,before)


def test_oriented_antipode_is_explicitly_nonsmooth_and_zero_weight_is_exact_off():
    p=np.r_[[np.pi,0.,0.],[0.,0.,1.]]
    with pytest.raises(N.NormalCutLocusError):N.normal_value_and_gradient(p,np.eye(3),.1)
    value,gradient,audit=N.normal_value_and_gradient(p,np.eye(3),.1,weight=0)
    assert value==0 and np.array_equal(gradient,np.zeros(6)) and not audit['enabled']
