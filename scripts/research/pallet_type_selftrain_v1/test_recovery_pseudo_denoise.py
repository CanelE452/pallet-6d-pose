import numpy as np
from .recovery_pseudo_denoise import perturb


def fixture():
    points=np.arange(18,dtype=np.float32).reshape(9,2)+100
    target=points+3
    valid=np.ones(9,bool);valid[[2,8]]=False
    return dict(points=points,valid=np.ones(9,bool),target=target,target_valid=valid,
        matrix=np.diag([.5,.5,1.]),bbox_diagonal=400.)


def test_corruption_leaves_targets_and_ignored_points_unchanged():
    x=fixture();points=x['points'].copy();target=x['target'].copy();rng=np.random.default_rng(1)
    y=perturb(x,rng,force=True)
    np.testing.assert_array_equal(x['points'],points)
    np.testing.assert_array_equal(x['target'],target)
    np.testing.assert_array_equal(y['target'],target)
    np.testing.assert_array_equal(y['points'][[2,8]],points[[2,8]])
    changed=np.linalg.norm(y['points']-points,axis=-1)>0
    assert changed.any() and not (changed&~x['target_valid']).any()
    displacement=np.linalg.norm(y['points'][changed]-target[changed],axis=-1)/.5
    assert (displacement>=20-1e-4).all() and (displacement<=60+1e-4).all()


def test_deterministic_balanced_modes_and_no_alias_mutation():
    x=fixture();rng=np.random.default_rng(4);counts=[]
    for _ in range(1000):
        y=perturb(x,rng);counts.append(int((np.linalg.norm(y['points']-x['points'],axis=-1)>0).sum()))
        y['points'][8]=0
        assert (x['points'][8]!=0).all()
    assert set(counts)=={0,1,7}
    assert 430<counts.count(0)<570 and 180<counts.count(1)<320
    np.testing.assert_array_equal(perturb(x,np.random.default_rng(10))['points'],perturb(x,np.random.default_rng(10))['points'])


def test_empty_target_is_noop():
    x=fixture();x['target_valid'][:]=False
    np.testing.assert_array_equal(perturb(x,np.random.default_rng(1),True)['points'],x['points'])
