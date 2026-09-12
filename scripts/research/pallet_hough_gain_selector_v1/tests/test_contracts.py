import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from scripts.research.pallet_hough_gain_selector_v1.core import action_features, assigned_target, choose, GainSelector
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.runner import seed_all, sequence
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.model import LocalLineFusionV2


def layout(x):
    a=np.zeros((9,2));a[:8,0]=x
    return a


def target(p,q,y,perms=None):
    return assigned_target(p,q,np.ones(9,bool),y,np.ones(9,bool),perms or [list(range(9))],100.)


def evidence():
    obs=dict(features=torch.arange(192*24*24).reshape(1,192,24,24).float()/1000,
             content=torch.ones(1,1,24,24),box=torch.tensor([[0.,0.,100.,100.]]),
             image_hw=torch.tensor([[100.,100.]]),dims=torch.ones(1,3),
             base_points=torch.tensor(layout(20.)).float()[None],point_valid=torch.ones(1,9,dtype=torch.bool),
             point_sigma=torch.ones(1,9))
    q=obs['base_points'].clone();q[:,:8]+=1
    lines=torch.zeros(1,12,3);lines[...,0]=1
    pred=dict(points=q,raw_line=lines,utility=torch.full((1,12),.2),mode_mass=torch.full((1,12),.4),ambiguity=torch.full((1,12),.5))
    return obs,pred


def test_equal_actions_zero_gain():
    assert target(layout(2),layout(2),layout(0))['gain_px']==0


def test_better_Q_positive():
    assert target(layout(2),layout(1),layout(0))['gain_px']==1


def test_worse_Q_negative():
    assert target(layout(1),layout(2),layout(0))['gain_px']==-1


def test_same_actions_different_GT_reverse_sign():
    assert target(layout(2),layout(1),layout(0))['gain_px']>0
    assert target(layout(2),layout(1),layout(4))['gain_px']<0


def test_inference_allowlist_rejects_GT_and_no_GT_read(monkeypatch):
    obs,pred=evidence()
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('inference attempted file/GT access'))
    x,_=action_features(obs,pred);assert torch.isfinite(x).all()
    with pytest.raises(ValueError):action_features(dict(obs,target_points=torch.zeros(1,9,2)),pred)


def test_features_directly_include_both_actions_and_local_image():
    obs,pred=evidence();x,schema=action_features(obs,pred)
    for name,expected in [('P_roi',obs['base_points'][:,:8]/100),('Q_roi',pred['points'][:,:8]/100)]:
        start,end=schema[name];torch.testing.assert_close(x[:,start:end],expected.flatten(1))
    changed=copy.deepcopy(pred);changed['points'][:,:8,0]+=3
    other,_=action_features(obs,changed);assert not torch.equal(x,other)
    assert schema['image_local_P_Q_edge'][1]-schema['image_local_P_Q_edge'][0]==336


def test_selection_is_exact_P_or_Q_not_blend():
    obs,pred=evidence();p=obs['base_points'].repeat(2,1,1);q=pred['points'].repeat(2,1,1)
    out,pick=choose(p,q,torch.tensor([.1,-.1]),torch.tensor([100.,100.]),.05)
    assert torch.equal(out[0],q[0]) and torch.equal(out[1],p[1]) and pick.tolist()==[True,False]


def test_below_equal_threshold_and_nonfinite_exact_P():
    obs,pred=evidence();p=obs['base_points'].repeat(3,1,1);q=pred['points'].repeat(3,1,1)
    out,pick=choose(p,q,torch.tensor([0.,.01,float('nan')]),torch.full((3,),100.),1.)
    assert torch.equal(out,p) and not pick.any()


def test_center8_preserved_and_bad_upstream_rejected():
    obs,pred=evidence();p=obs['base_points'];q=pred['points']
    out,_=choose(p,q,torch.tensor([.1]),torch.tensor([100.]),0)
    assert torch.equal(out[:,8],p[:,8])
    q=q.clone();q[:,8]+=1
    with pytest.raises(ValueError):choose(p,q,torch.tensor([.1]),torch.tensor([100.]),0)


def test_one_baseline_symmetry_for_both_actions():
    y=layout(0);y[:8,0]=np.arange(8);p=y.copy();q=y.copy();q[:8]=q[:8][::-1]
    perms=[list(range(9)),list(range(7,-1,-1))+[8]]
    r=target(p,q,y,perms)
    assert r['choice']==0 and r['p_mean']==0 and r['q_mean']>0 and r['gain_px']<0
    assert np.array_equal(q,y[perms[1]])  # independent Q assignment would hide harm


def test_same_seed_init_and_order_reproduce():
    seed_all(1);a=GainSelector(10,torch.zeros(10),torch.ones(10))
    seed_all(1);b=GainSelector(10,torch.zeros(10),torch.ones(10))
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert sequence(1792,128000,1)==sequence(1792,128000,1)
    assert sequence(1792,100,1)!=sequence(1792,100,2)


def test_only_selector_updates_upstream_tensor_detached():
    x=torch.randn(4,10,requires_grad=True);model=GainSelector(10,torch.zeros(10),torch.ones(10))
    before=copy.deepcopy(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=.001)
    F.smooth_l1_loss(model(x),torch.ones(4)).backward();opt.step()
    assert x.grad is None and any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_actual_frozen_Hough_WLS_parameter_diff_zero():
    torch.set_num_threads(2)
    upstream=LocalLineFusionV2(arm='hough',common_seed=1).eval().requires_grad_(False)
    before=copy.deepcopy(upstream.state_dict());obs,_=evidence()
    with torch.no_grad():
        result=upstream(obs)
        pred={k:result[k] for k in ('points','raw_line','utility','mode_mass','ambiguity')}
        x,_=action_features(obs,pred)
    selector=GainSelector(x.shape[1],torch.zeros(x.shape[1]),torch.ones(x.shape[1]))
    opt=torch.optim.AdamW(selector.parameters(),lr=.001)
    F.smooth_l1_loss(selector(x),torch.ones(1)).backward();opt.step()
    assert all(torch.equal(v,upstream.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None and not p.requires_grad for p in upstream.parameters())
