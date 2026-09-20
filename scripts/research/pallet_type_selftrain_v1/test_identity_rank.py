import numpy as np
import torch
from .identity_rank_features import describe,Ranker,choose,QUARTER,HALF


def fixture():
    rng=np.random.default_rng(1)
    return rng.normal(size=(64,80,80)).astype(np.float16),rng.normal(size=(128,40,40)).astype(np.float16),rng.uniform(100,400,(9,2)),np.array([80,90,420,440]),[640,640]


def test_quarter_is_not_equivalent_but_half_is():
    np.testing.assert_array_equal(HALF,[5,4,7,6,1,0,3,2,8])
    np.testing.assert_array_equal(QUARTER[QUARTER[QUARTER[QUARTER]]],np.arange(9))
    assert not np.array_equal(QUARTER,np.arange(9)) and not np.array_equal(QUARTER,HALF)


def test_descriptor_equivariance_and_no_input_mutation():
    p3,p4,q,b,shape=fixture();original=q.copy()
    x=describe(p3,p4,q,b,shape);h=describe(p3,p4,q[HALF],b,shape);r=describe(p3,p4,q[QUARTER],b,shape)
    np.testing.assert_array_equal(x[:,::-1],h)
    np.testing.assert_array_equal(r[0],x[1]);np.testing.assert_array_equal(r[1],x[0,::-1])
    np.testing.assert_array_equal(q,original)
    torch.manual_seed(2);model=Ranker()
    with torch.no_grad():
        np.testing.assert_allclose(model(torch.tensor(x[None])),model(torch.tensor(h[None])),atol=1e-6,rtol=0)


def test_source_calibrated_abstention():
    assert choose([.1,.9],.9)==1
    assert choose([.1,.9],.95)==0
    assert choose([.1,.9],None)==0
    assert choose([.5,.5],.5)==0


def test_padded_spatial_features_do_not_leak():
    p3,p4,q,b,shape=fixture();q[:]=[600,600]
    x=describe(p3,p4,q,b,[320,640])
    assert not x[...,:4608].any()
