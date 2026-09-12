import torch
import pytest
from dataclasses import replace
from plpose_v5.fixtures import geometry_fixture
from plpose_v5.geometry import so3_exp,project,cuboid
from plpose_v5.solver import *

torch.set_num_threads(2)

@pytest.mark.parametrize('point_use,line_use',[(True,False),(True,True),(False,True)])
def test_exact_observations_recover_nearby_pose(point_use,line_use):
    d,K,R,t,p,m=geometry_fixture()
    if not point_use:m=replace(m,point_valid=torch.zeros_like(m.point_valid),points=torch.full_like(m.points,float('nan')))
    if not line_use:m=replace(m,line_valid=torch.zeros_like(m.line_valid),lines=torch.full_like(m.lines,float('nan')))
    R0=so3_exp(torch.tensor([[.035,-.025,.02]],dtype=R.dtype))@R
    t0=t+torch.tensor([[.03,-.02,.05]],dtype=t.dtype)
    config=SolverConfig(iterations=15,line_weight=.5 if line_use else 0.)
    out=PointLineRefiner(config)(m,d,K,R0[:,None],t0[:,None])
    assert bool(out['pose_valid'].all())
    assert torch.linalg.vector_norm(out['t']-t)<1e-4
    assert torch.linalg.vector_norm(out['points']-p)<1e-3
    trace=out['objective_trace']
    assert (trace[...,1:]<=trace[...,:-1]+1e-9).all()

def test_two_starts_selected_by_observations_not_gt():
    d,K,R,t,p,m=geometry_fixture(noise=.1)
    R0=torch.stack((R,so3_exp(torch.tensor([[0.,1.,0.]],dtype=R.dtype))@R),1)
    t0=t[:,None].repeat(1,2,1)
    out=PointLineRefiner(SolverConfig(iterations=3))(m,d,K,R0,t0)
    assert int(out['selected'])==0
    assert out['energy'].shape==(1,2)

def test_no_observations_is_invalid_not_a_successful_pose():
    d,K,R,t,p,m=geometry_fixture()
    m=replace(m,point_valid=torch.zeros_like(m.point_valid),line_valid=torch.zeros_like(m.line_valid))
    out=PointLineRefiner(SolverConfig(iterations=2))(m,d,K,R[:,None],t[:,None])
    assert not out['pose_valid'].any()
    torch.testing.assert_close(out['R'],R);torch.testing.assert_close(out['t'],t)
    assert (out['information_rank']==0).all()

def test_zero_line_weight_exact_point_control():
    d,K,R,t,p,m=geometry_fixture(noise=1.)
    corrupted=replace(m,lines=m.lines+torch.tensor([0.,0.,70.],dtype=R.dtype))
    solver=PointLineRefiner(SolverConfig(iterations=4,line_weight=0))
    out=solver(m,d,K,R[:,None],t[:,None]);other=solver(corrupted,d,K,R[:,None],t[:,None])
    torch.testing.assert_close(out['t'],other['t'],atol=0,rtol=0)
    torch.testing.assert_close(out['R'],other['R'],atol=0,rtol=0)

def test_final_pose_gradient_reaches_point_observations():
    d,K,R,t,p,m=geometry_fixture(noise=.8)
    m.points=m.points.clone().requires_grad_()
    out=PointLineRefiner(SolverConfig(iterations=3))(m,d,K,R[:,None],t[:,None])
    loss=(out['t']-t).square().sum();grad=torch.autograd.grad(loss,m.points)[0]
    assert torch.isfinite(grad).all() and grad[:,:8].abs().sum()>0
    assert not grad[:,8].any()

def test_final_pose_gradient_reaches_line_offset():
    d,K,R,t,p,m=geometry_fixture(noise=1.)
    offset=torch.zeros(1,12,1,dtype=R.dtype,requires_grad=True)
    m.lines=torch.cat((m.lines[...,:2],m.lines[...,2:]+offset[...,None]),-1)
    out=PointLineRefiner(SolverConfig(iterations=3,line_weight=1))(m,d,K,R[:,None],t[:,None])
    loss=(out['t']-t).square().sum();grad=torch.autograd.grad(loss,offset)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0

def test_joint_mode_not_independent_endpoint_cherry_pick():
    d,K,R,t,p,m=geometry_fixture();a,b=p[0,0],p[0,1]
    n=torch.tensor([1.,0.],dtype=R.dtype)
    modes=torch.stack((torch.cat((n,-a[0:1])),torch.cat((n,-b[0:1]))))
    m.lines=modes[None,None].repeat(1,12,1,1)
    m.line_valid=torch.zeros(1,12,2,dtype=torch.bool);m.line_valid[:,0]=True
    m.line_logprob=torch.zeros(1,12,2,dtype=R.dtype)-torch.log(torch.tensor(2.,dtype=R.dtype))
    m.line_sigma=torch.ones(1,12,2,dtype=R.dtype)
    result=linearize(R,t,cuboid(d),K,m,SolverConfig(line_weight=1))
    assert float(result[3])>1.

@pytest.mark.parametrize('kind',['negative_sigma','nonunit_line','bad_rotation'])
def test_reject_invalid_contract(kind):
    d,K,R,t,p,m=geometry_fixture()
    if kind=='negative_sigma':m.point_sigma=-m.point_sigma
    elif kind=='nonunit_line':m.lines=m.lines*2
    else:R=R*2
    with pytest.raises(ValueError):PointLineRefiner()(m,d,K,R[:,None],t[:,None])

def test_unrolled_point_derivative_matches_finite_difference():
    d,K,R,t,p,m=geometry_fixture(noise=.7,seed=19)
    solver=PointLineRefiner(SolverConfig(iterations=3,line_weight=.2));m.points=m.points.clone().requires_grad_()
    z=solver(m,d,K,R[:,None],t[:,None])['t'][0,2];analytic=torch.autograd.grad(z,m.points)[0][0,0,0]
    eps=1e-4;base=m.points.detach();plus=base.clone();minus=base.clone();plus[0,0,0]+=eps;minus[0,0,0]-=eps
    zp=solver(replace(m,points=plus),d,K,R[:,None],t[:,None])['t'][0,2]
    zm=solver(replace(m,points=minus),d,K,R[:,None],t[:,None])['t'][0,2]
    numeric=(zp-zm)/(2*eps)
    torch.testing.assert_close(analytic,numeric,atol=2e-5,rtol=1e-3)

def test_line_homogeneous_sign_does_not_change_pose():
    d,K,R,t,p,m=geometry_fixture(noise=1.);solver=PointLineRefiner(SolverConfig(iterations=3))
    a=solver(m,d,K,R[:,None],t[:,None]);b=solver(replace(m,lines=-m.lines),d,K,R[:,None],t[:,None])
    torch.testing.assert_close(a['t'],b['t'],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(a['R'],b['R'],atol=1e-12,rtol=1e-12)

def test_line_mode_order_does_not_change_pose():
    d,K,R,t,p,m=geometry_fixture(noise=.4)
    modes=torch.cat((m.lines,m.lines+torch.tensor([0.,0.,5.],dtype=R.dtype)),2)
    m=replace(m,lines=modes,line_valid=m.line_valid.expand(-1,-1,2),line_sigma=m.line_sigma.expand(-1,-1,2),
              line_logprob=torch.tensor([.7,.3],dtype=R.dtype).log()[None,None].expand(1,12,2))
    reverse=replace(m,lines=m.lines.flip(2),line_valid=m.line_valid.flip(2),line_sigma=m.line_sigma.flip(2),line_logprob=m.line_logprob.flip(2))
    solver=PointLineRefiner(SolverConfig(iterations=3));a=solver(m,d,K,R[:,None],t[:,None]);b=solver(reverse,d,K,R[:,None],t[:,None])
    torch.testing.assert_close(a['t'],b['t'],atol=1e-10,rtol=1e-10)
