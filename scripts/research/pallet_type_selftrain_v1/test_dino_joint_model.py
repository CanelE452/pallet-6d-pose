import numpy as np
import torch
from . import dino_joint_model as J


def test_rotation_is_whole_graph_permutation():
    edges={(min(a,b),max(a,b)) for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]}
    for p in J.PERMS:
        assert sorted(p)==list(range(9)) and p[8]==8
        assert {(min(p[a],p[b]),max(p[a],p[b])) for a,b in edges}==edges


def test_all_candidates_whole_and_features_finite():
    torch.set_num_threads(2);old=torch.arange(18).reshape(1,9,2).float()*10+100;z=torch.randn(1,9,192,144)
    x,c=J.describe(z,z,old,torch.tensor([300.]));assert x.shape==(1,12,98) and c.shape==(1,12,9,2)
    for i,p in enumerate(J.PERMS):torch.testing.assert_close(c[:,i],old[:,p],atol=0,rtol=0)


def test_choose_one_entire_candidate_and_fallback():
    score=np.zeros((3,12));score[0,9]=.9;score[1,3]=.8;score[2,0]=1
    np.testing.assert_array_equal(J.choose(score,.85),[9,0,0])


def test_error_scoring_respects_declared_group_only():
    q=np.arange(18).reshape(9,2)*20.;v=np.ones(9,bool);v[8]=False
    e=J.canonical_errors([q[J.Q]],q,v,[np.arange(9)],1.);assert np.nanmean(e)>0
    e=J.canonical_errors([q[J.Q]],q,v,J.PERMS,1.);assert np.nanmax(e)==0


def test_benefit_rejects_new_damage_even_if_mean_improves():
    e=np.full((1,12,8),100.);e[:,0,0]=1;e[:,1,:]=1;e[:,2,:]=1;e[:,2,0]=20
    y=J.beneficial(e);assert y[0,1] and not y[0,2] and not y[0,0]
