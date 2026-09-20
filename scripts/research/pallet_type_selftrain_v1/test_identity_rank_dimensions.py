import numpy as np
import pytest
from .identity_rank_dimensions import conditioned_difference,plastic_dimensions
from .identity_rank_linear import difference


def test_fixed_dimensions_preserve_c2_and_exchange_antisymmetry():
    x=np.random.default_rng(3).normal(size=(5,2,2,4648)).astype(np.float32)
    dims=np.array([[1.1,1.3,.11],[.8,1.2,.1],[1,1,.2],[.9,1.1,.12],[1.2,1.3,.15]])
    original=x.copy();before=dims.copy();d=conditioned_difference(x,dims)
    assert d.shape==(5,13944)
    np.testing.assert_array_equal(d,conditioned_difference(x[:,:,::-1],dims))
    np.testing.assert_array_equal(-d,conditioned_difference(x[:,::-1],dims))
    np.testing.assert_array_equal(d[:,:4648],difference(x))
    np.testing.assert_array_equal(x,original);np.testing.assert_array_equal(dims,before)


def test_dimension_ratios_unit_invariant_and_broadcast():
    x=np.random.default_rng(4).normal(size=(3,2,2,4648)).astype(np.float32)
    dims=plastic_dimensions()
    np.testing.assert_allclose(dims,[1.1,1.3,.11])
    a=conditioned_difference(x,dims)
    np.testing.assert_allclose(a,conditioned_difference(x,dims*100),atol=3e-6)
    np.testing.assert_array_equal(a,conditioned_difference(x,np.tile(dims,(3,1))))
    np.testing.assert_array_equal(a[0],conditioned_difference(x[0],dims))


def test_dimensions_act_as_interaction_not_candidate_independent_intercept():
    x=np.random.default_rng(5).normal(size=(2,2,4648)).astype(np.float32)
    a=conditioned_difference(x,[1,2,.1]);b=conditioned_difference(x,[2,1,.1])
    np.testing.assert_array_equal(a[:4648],b[:4648])
    np.testing.assert_allclose(a[4648:9296],-b[4648:9296],atol=1e-6)
    np.testing.assert_array_equal(a[9296:],b[9296:])
    np.testing.assert_array_equal(conditioned_difference(np.zeros_like(x),[1,2,.1]),np.zeros(13944))


@pytest.mark.parametrize('dims',[[1,0,.1],[1,-1,.1],[1,2,float('nan')],[1,2],[1,2,float('inf')]])
def test_invalid_dimensions_rejected(dims):
    with pytest.raises(ValueError):conditioned_difference(np.zeros((2,2,4648)),dims)
