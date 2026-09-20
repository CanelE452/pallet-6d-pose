import numpy as np
from .identity_rank_linear import difference,normalize,probability,train_linear


def test_exact_c2_invariance_and_candidate_exchange():
    x=np.random.default_rng(1).normal(size=(5,2,2,4648)).astype(np.float32)
    np.testing.assert_array_equal(difference(x),difference(x[:,:,::-1]))
    np.testing.assert_array_equal(difference(x[:,::-1]),-difference(x))
    s=np.maximum(np.sqrt(np.mean(difference(x)**2,axis=0)),.01)
    a=normalize(difference(x),s);b=normalize(difference(x[:,::-1]),s)
    np.testing.assert_array_equal(a,-b)
    w=np.random.default_rng(2).normal(0,.01,4648).astype(np.float32)
    np.testing.assert_allclose(probability(a,w)[:,::-1],probability(b,w),atol=1e-7)


def test_convex_regularized_fit_learns_known_direction_without_bias():
    x=np.array([[-2,0],[-1,0],[1,0],[2,0]],np.float32);y=np.array([0,0,1,1])
    w,s=train_linear(x,y,None,None,.1,device='cpu')
    assert w[0]>0 and w[1]==0 and s['final_objective']<s['initial_objective']
    np.testing.assert_array_equal(probability(x,w).argmax(-1),y)
    np.testing.assert_array_equal(probability(np.zeros((1,2),np.float32),w),[[.5,.5]])


def test_stronger_regularization_reduces_weight_norm():
    x=np.array([[-2,0],[-1,0],[1,0],[2,0]],np.float32);y=np.array([0,0,1,1])
    a,_=train_linear(x,y,None,None,.01,device='cpu');b,_=train_linear(x,y,None,None,1.,device='cpu')
    assert np.linalg.norm(b)<np.linalg.norm(a)
