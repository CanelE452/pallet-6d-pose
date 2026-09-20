import numpy as np
import torch
from . import dino_joint_translate_model as M


def sample():
    return torch.tensor([[120.,120.],[240.,120.],[240.,200.],[120.,200.],[140.,140.],[260.,140.],[260.,220.],[140.,220.],[190.,170.]])


def maps(target):
    yy,xx=torch.meshgrid(torch.arange(192),torch.arange(144),indexing='ij');xy=torch.stack([xx,yy],-1).float()
    return -((xy[None]-target[:8,None,None]/4)**2).sum(-1)/8


def test_shared_large_translation_recovers_pattern():
    target=sample();q=target+torch.tensor([64.,-48.]);p=M.propose(maps(target),q,torch.ones(9,dtype=torch.bool))
    assert p['permutation']==0 and p['delta_crop']==[-64.,48.] and p['gain']>0


def test_identity_tie_keeps_exact_input():
    q=sample();p=M.propose(torch.zeros(8,192,144),q,torch.ones(9,dtype=torch.bool))
    assert p['permutation']==0 and p['delta_crop']==[0.,0.] and p['gain']==0


def test_whole_role_permutation_not_point_splicing():
    target=sample();q=target[np.asarray(M.PERMS[1])];p=M.propose(maps(target),q,torch.ones(9,dtype=torch.bool))
    out=M.restore(q.numpy(),p,np.eye(3));np.testing.assert_allclose(out[:8],target[:8].numpy(),atol=1e-5)


def test_restore_preserves_center_and_relative_shape():
    q=sample().numpy().astype(float);p=dict(available=True,permutation=2,delta_crop=[40.,-60.])
    matrix=np.array([[2.,0.,100.],[0.,2.,200.],[0.,0.,1.]])
    out=M.restore(q,p,matrix);perm=np.asarray(M.PERMS[2])
    np.testing.assert_array_equal(out[:8]-q[perm[:8]],np.tile([20.,-30.],(8,1)))
    np.testing.assert_array_equal(out[8],q[8]);np.testing.assert_array_equal(M.restore(q,p,matrix,False),q)


def test_missing_input_is_fallback_not_frame_rejection():
    q=sample();v=torch.ones(9,dtype=torch.bool);v[0]=False
    p=M.propose(maps(q),q,v);assert not p['available']
    np.testing.assert_array_equal(M.restore(q.numpy(),p,np.eye(3)),q.numpy())


def test_logmass_two_identical_heads_and_finite_bounds():
    z=torch.randn(1,9,192,144);m=M.logmass(z,z)
    assert torch.isfinite(m).all() and (m<=0).all()
