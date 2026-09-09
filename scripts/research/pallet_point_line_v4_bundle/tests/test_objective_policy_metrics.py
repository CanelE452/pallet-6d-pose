import numpy as np,pytest,torch
from pointline_v4.objective import candidate_targets,quality_loss
from pointline_v4.policy import select_layout
from pointline_v4.metrics import error_summary,safety_counts,paired_session_bootstrap
from conftest import make_observation


def test_masked_nan_gt_never_enters_loss():
    o=make_observation();gt=o.baseline.clone();gt[:,3]=float('nan');mask=o.point_valid.clone();mask[:,3]=False
    t=candidate_targets(o.layouts,gt,mask,o.diagonal);assert torch.isfinite(t['cost']).all()
    mask[:,3]=True
    with pytest.raises(ValueError):candidate_targets(o.layouts,gt,mask,o.diagonal)


def test_candidate_quality_gradient_not_gated_by_identity_selection():
    o=make_observation();t=candidate_targets(o.layouts,o.baseline,o.point_valid,o.diagonal)
    cost=torch.zeros(1,4,requires_grad=True);corner=torch.zeros(1,4,8,requires_grad=True)
    loss,_=quality_loss(cost,corner,t,o.candidate_valid);loss.backward()
    assert (cost.grad.abs()>0).all()
    out,ix=select_layout(cost.detach(),o.layouts,o.candidate_valid,None)
    assert torch.equal(out,o.baseline) and ix.item()==0


def test_empty_supervision_fails_instead_of_fake_update():
    o=make_observation();t=candidate_targets(o.layouts,o.baseline,torch.zeros_like(o.point_valid),o.diagonal)
    with pytest.raises(ValueError):quality_loss(torch.zeros(1,4),torch.zeros(1,4,8),t,o.candidate_valid)


def test_invalid_candidate_has_no_loss_gradient():
    o=make_observation();o.candidate_valid[:,2]=False
    t=candidate_targets(o.layouts,o.baseline,o.point_valid,o.diagonal)
    cost=torch.randn(1,4,requires_grad=True);corner=torch.randn(1,4,8,requires_grad=True)
    loss,_=quality_loss(cost,corner,t,o.candidate_valid);loss.backward()
    assert cost.grad[0,2]==0 and not corner.grad[0,2].any()


def test_strict_margin_and_ties_preserve_identity():
    o=make_observation();cost=torch.tensor([[1.,.5,1.,2.]])
    out,ix=select_layout(cost,o.layouts,o.candidate_valid,.5);assert ix.item()==0
    out,ix=select_layout(cost,o.layouts,o.candidate_valid,.49);assert ix.item()==1
    assert torch.equal(out[:,8],o.baseline[:,8])
    _,ix=select_layout(torch.zeros_like(cost),o.layouts,o.candidate_valid,0.);assert ix.item()==0


def test_invalid_low_score_never_selected():
    o=make_observation();o.candidate_valid[:,2]=False
    _,ix=select_layout(torch.tensor([[1.,2.,-100.,3.]]),o.layouts,o.candidate_valid,0.);assert ix.item()==0


def test_supervision_detached_from_coordinate_graph():
    o=make_observation();q=o.layouts.clone().requires_grad_();gt=o.baseline.clone().requires_grad_()
    t=candidate_targets(q,gt,o.point_valid,o.diagonal);assert not t['cost'].requires_grad


def test_metrics_missing_points_stay_in_pck_denominator():
    o=make_observation();pred=o.baseline.numpy();valid=o.point_valid.numpy().copy();valid[0,0]=False
    s=error_summary(pred,pred,np.ones((1,9),bool),valid,np.ones(1,bool),np.array([50.]))
    assert s['observed_points']==7 and s['gt_points']==8 and s['missing_points']==1
    assert s['pck10_all_gt']==7/8


def test_median_p90_same_population_and_8_vs_9():
    gt=np.zeros((1,9,2));pred=gt.copy();pred[0,:,0]=np.arange(9)
    args=(pred,gt,np.ones((1,9),bool),np.ones((1,9),bool),np.ones(1,bool),np.array([50.]))
    a=error_summary(*args);b=error_summary(*args,corners_only=False)
    assert a['median_px']==3.5 and b['median_px']==4.
    assert a['p90_px']==np.quantile(np.arange(8),.9)


def test_safety_counts_do_not_select_frames_by_new_model():
    x=safety_counts(np.array([[1,9,25.]]),np.array([[11,8,20.]]),np.ones((1,3),bool))
    assert x['good_count']==2 and x['good_damaged']==1 and x['hard_count']==1


def test_paired_cluster_bootstrap_reproducible_negative_difference():
    args=(np.array([-1.,-1.,-1.,-1.,-1.]),['a','a','b','b','c'])
    a=paired_session_bootstrap(*args,draws=1000);b=paired_session_bootstrap(*args,draws=1000)
    assert a==b and a['lower']==-1 and a['upper']==-1


def test_bootstrap_rejects_single_session_and_nan():
    with pytest.raises(ValueError):paired_session_bootstrap([1.,2.],['a','a'])
    with pytest.raises(ValueError):paired_session_bootstrap([1.,np.nan],['a','b'])
