import numpy as np
from . import dino_large_gate as L


def item():
    return dict(row=17,points=np.full((9,2),50.,np.float32),target=np.full((9,2),200.,np.float32),
        valid=np.ones(9,bool),target_valid=np.array([1,1,0,1,1,1,1,1,0],bool),bbox_diagonal=400.,matrix=np.diag([2.,2.,1.]))


def test_small_views_byte_identical():
    x=item()
    assert L.perturb(x,0) is x
    for view in [1,2]:
        a=L.perturb(x,view);rng=np.random.default_rng(np.random.SeedSequence([20261001,17,view]));b=L.P.perturb(x,rng,True)
        np.testing.assert_array_equal(a['points'],b['points']);np.testing.assert_array_equal(a['valid'],b['valid'])


def test_large_single_and_all_preserve_targets_center_unknown():
    x=item();before={k:v.copy() for k,v in x.items() if isinstance(v,np.ndarray)}
    for view,count in [(3,1),(4,7)]:
        y=L.perturb(x,view);changed=(y['points']!=x['points']).any(-1)
        assert changed.sum()==count and not changed[2] and not changed[8]
        d=np.linalg.norm(y['points'][changed]-x['target'][changed],axis=-1)/800
        assert (d>=.25-1e-6).all() and (d<=.60+1e-6).all()
        np.testing.assert_array_equal(y['target'],x['target']);np.testing.assert_array_equal(y['target_valid'],x['target_valid'])
    for k,v in before.items():np.testing.assert_array_equal(x[k],v)


def test_deterministic_and_no_targets_no_corruption():
    x=item();np.testing.assert_array_equal(L.perturb(x,4)['points'],L.perturb(x,4)['points'])
    x['target_valid'][:]=False
    for view in [3,4]:np.testing.assert_array_equal(L.perturb(x,view)['points'],x['points'])
