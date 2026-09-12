import math
import pytest
import torch
from dataclasses import replace
from plpose_v5.hough import *
from plpose_v5.model import *
from plpose_v5.contracts import *
from plpose_v5.geometry import *
from plpose_v5.objective import *
from plpose_v5.fixtures import geometry_fixture

torch.set_num_threads(2)

@pytest.mark.parametrize('hw',[(12,12),(8,12)])
def test_dht_constant_preserved(hw):
    h,w=hw;lattice=Lattice(h,w,12,25)
    x=torch.ones(2,3,h,w);mask=torch.ones(2,1,h,w);mask[1,:,:,0:3]=0
    v,m=lattice(x,mask)
    torch.testing.assert_close(v[m.expand_as(v)>0],torch.ones_like(v[m.expand_as(v)>0]))

def test_dht_gradient_and_locality():
    L=Lattice(12,12,12,25);x=torch.randn(1,1,12,12,requires_grad=True)
    v,m=L(x,torch.ones_like(x));v[0,0,0,12].backward()
    assert x.grad.abs().sum()>0 and (x.grad!=0).sum()<12*12

def test_seam_reverses_rho():
    x=torch.arange(12.).reshape(1,1,3,4);y=hough_pad(x)
    torch.testing.assert_close(y[0,0,0,1:-1],x[0,0,-1].flip(-1))
    torch.testing.assert_close(y[0,0,-1,1:-1],x[0,0,0].flip(-1))

def make_observation(batch=1,grid=12,channels=8):
    d,K,R,t,p,m=geometry_fixture(batch,dtype=torch.float32)
    box=torch.cat((p[:,:8].amin(1)-20,p[:,:8].amax(1)+20),-1)
    obs=Observation(torch.randn(batch,channels,grid,grid),torch.ones(batch,1,grid,grid,dtype=torch.bool),box,K,d,torch.tensor([480.,640.]).repeat(batch,1))
    gt=Supervision(p,torch.ones(batch,9,dtype=torch.bool),R,t,[SymmetrySpec(2,True,'fixture')]*batch)
    return obs,gt

@pytest.mark.parametrize('arm',['point','direct','hough'])
def test_end_to_end_finite_and_all_heads_receive_gradients(arm):
    torch.manual_seed(3);obs,gt=make_observation()
    cfg=ModelConfig(arm=arm,in_channels=8,channels=16,grid=12,theta_bins=12,rho_bins=25,starts=2,solver_iterations=2)
    model=PointLinePoseModel(cfg);obs.features.requires_grad_();out=model(obs);loss,detail=compute_loss(out,obs,gt,model.lattice.lines)
    assert torch.isfinite(loss);loss.backward();assert torch.isfinite(obs.features.grad).all() and obs.features.grad.abs().sum()>0
    assert model.point_map.weight.grad.abs().sum()>0
    if model.line_head is not None:
        grads=[p.grad for p in model.line_head.parameters() if p.grad is not None]
        assert grads and sum(float(g.abs().sum()) for g in grads)>0
    for name,p in model.named_parameters():
        if p.grad is not None:assert torch.isfinite(p.grad).all(),name
    validate_rotation(out['R'])

def test_shared_initialization():
    models={a:PointLinePoseModel(ModelConfig(arm=a,in_channels=8,channels=16,grid=12,theta_bins=12,rho_bins=25,starts=2)) for a in ('point','direct','hough')}
    names=initialize_matched(models)
    for n in names:
        for m in models.values():torch.testing.assert_close(m.state_dict()[n],models['point'].state_dict()[n],atol=0,rtol=0)

def test_valid_symmetry_permutation_changes_no_shared_loss():
    torch.manual_seed(4);obs,gt=make_observation();model=PointLinePoseModel(ModelConfig(arm='hough',in_channels=8,channels=16,grid=12,theta_bins=12,rho_bins=25,starts=2,solver_iterations=1))
    out=model(obs);first,diag=compute_loss(out,obs,gt,model.lattice.lines);G,p=symmetry_permutations(obs.dims[0],gt.specs[0]);p=p[1]
    shifted=Supervision(gt.points[:,p],gt.valid[:,p],gt.R@G[1],gt.t,gt.specs);second,detail=compute_loss(out,obs,shifted,model.lattice.lines)
    torch.testing.assert_close(first,second,rtol=1e-5,atol=1e-5);assert diag['shared_symmetry_for_all_terms']

def test_missing_gt_nan_does_not_contaminate_loss():
    obs,gt=make_observation();gt.valid[:,3]=False;gt.points[:,3]=float('nan')
    model=PointLinePoseModel(ModelConfig(arm='hough',in_channels=8,channels=16,grid=12,theta_bins=12,rho_bins=25,starts=2,solver_iterations=1))
    loss,detail=compute_loss(model(obs),obs,gt,model.lattice.lines);loss.backward();assert torch.isfinite(loss)

def test_same_image_with_changed_dims_is_accepted_but_wrong_c4_rejected():
    obs,gt=make_observation();obs.dims[:,0]=.9
    model=PointLinePoseModel(ModelConfig(arm='point',in_channels=8,channels=16,grid=12,theta_bins=12,rho_bins=25,starts=2,solver_iterations=1));out=model(obs)
    gt.specs=[SymmetrySpec(4,True,'deliberately invalid')]
    with pytest.raises(ValueError):compute_loss(out,obs,gt,model.lattice.lines)

def test_forward_no_target_parameter():
    import inspect
    assert list(inspect.signature(PointLinePoseModel.forward).parameters)==['self','obs']
    obs,gt=make_observation();assert 'points' not in vars(obs) and 'R' not in vars(obs) and 'valid' not in vars(obs)

def test_C4_shared_loss_keeps_center_and_all_terms_consistent():
    torch.manual_seed(41);obs,gt=make_observation();obs.dims[:,1]=obs.dims[:,0];gt.points=project(cuboid(obs.dims),gt.R,gt.t,obs.K)[0];gt.specs=[SymmetrySpec(4,True,'square generated fixture')]
    model=PointLinePoseModel(ModelConfig(arm='hough',in_channels=8,channels=16,grid=12,theta_bins=12,rho_bins=25,starts=4,solver_iterations=1));out=model(obs);loss,detail=compute_loss(out,obs,gt,model.lattice.lines)
    G,p=symmetry_permutations(obs.dims[0],gt.specs[0])
    for idx in (1,2,3):
        shifted=Supervision(gt.points[:,p[idx]],gt.valid[:,p[idx]],gt.R@G[idx],gt.t,gt.specs);same,_=compute_loss(out,obs,shifted,model.lattice.lines);torch.testing.assert_close(loss,same,rtol=2e-5,atol=2e-5)
    assert len(detail['symmetry_total_costs'][0])==4

def test_mode_decoder_keeps_far_peaks_separate_and_gradient():
    L=Lattice(12,12,12,25);logits=torch.full((1,12,300),-12.);logits[:,:,4*25+7]=8.;logits[:,:,4*25+17]=7.;logits.requires_grad_();A=torch.eye(3)[None]
    lines,lp,valid=decode_modes(logits,torch.ones_like(logits,dtype=torch.bool),L,A,2);assert valid.all();assert (lines[:,:,0,2]-lines[:,:,1,2]).abs().min()>.7
    (lines.square().sum()+lp.square().sum()).backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0

def test_cuboid_outputs_have_one_rigid_pose():
    obs,gt=make_observation();model=PointLinePoseModel(ModelConfig(arm='hough',in_channels=8,channels=16,grid=12,theta_bins=12,rho_bins=25,starts=2,solver_iterations=1));out=model(obs)
    expected,depth=project(cuboid(obs.dims),out['R'],out['t'],obs.K);torch.testing.assert_close(out['points'],expected,atol=0,rtol=0)

def test_final_pose_loss_alone_reaches_hough_without_auxiliary_loss():
    torch.manual_seed(3);obs,gt=make_observation();obs.features.requires_grad_()
    model=PointLinePoseModel(ModelConfig(arm='hough',in_channels=8,channels=16,grid=12,theta_bins=12,rho_bins=25,starts=2,solver_iterations=3));out=model(obs)
    loss=(out['t']-gt.t).square().sum();parameters=list(model.line_head.parameters());grad=torch.autograd.grad(loss,parameters+[obs.features],allow_unused=True)
    assert sum(float(g.abs().sum()) for g in grad[:-1] if g is not None)>0
    assert grad[-1] is not None and torch.isfinite(grad[-1]).all() and grad[-1].abs().sum()>0
