import numpy as np
import pytest
from .facing_rank_features import describe, camera_matrix
from .identity_rank_features import QUARTER, HALF


def fixture():
    q = np.array([[-1,-.1,-.7],[1,-.1,-.7],[1,.1,-.7],[-1,.1,-.7],
                  [-1,-.1,.7],[1,-.1,.7],[1,.1,.7],[-1,.1,.7]], float)
    a=.3; r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    xyz=np.vstack([q,[0,0,0]])@r.T+[.4,.2,5]
    k=camera_matrix(dict(fx=600,fy=610,cx=320,cy=240))
    p=xyz@k.T
    return p[:,:2]/p[:,2:], k


def test_exact_projection_front_and_group_behavior():
    q,k=fixture(); old=q.copy()
    x,d=describe(q,k)
    assert d['available'] and d['facing_margin']>0 and d['axis_nonorthogonality']<1e-10
    np.testing.assert_allclose(describe(q[HALF],k)[0],x,atol=1e-7)
    np.testing.assert_allclose(describe(q[QUARTER],k)[0],-x,atol=1e-7)
    np.testing.assert_array_equal(q,old)


def test_noisy_group_invariance_and_camera_rescaling():
    q,k=fixture();q+=np.random.default_rng(19).normal(0,2,q.shape)
    x,_=describe(q,k)
    np.testing.assert_allclose(describe(q[HALF],k)[0],x,atol=1e-6)
    np.testing.assert_allclose(describe(q[QUARTER],k)[0],-x,atol=1e-6)
    a=np.array([[2,0,37],[0,2,13],[0,0,1.]])
    np.testing.assert_allclose(describe(q*2+[37,13],a@k)[0],x,atol=1e-6)


def test_degenerate_or_invalid_inputs_abstain_not_reject():
    q,k=fixture();q[:8]=12
    x,d=describe(q,k);assert not d['available'];assert not x.any()
    q,k=fixture();q[0]=-1
    assert not describe(q,k)[1]['available']
    with pytest.raises(ValueError): describe(q,np.zeros((3,3)))


def test_center_fallback_preserves_group_contract():
    q,k=fixture();q[8]=-1;x,_=describe(q,k)
    np.testing.assert_allclose(describe(q[QUARTER],k)[0],-x,atol=1e-6)
