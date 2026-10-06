import unittest
import numpy as np,torch
from .a_common import POSE
from .geometry import build_bank,permute_bank,project,rotations
from .scorer import JointActionScorer,decode_bank,joint_loss

class Contracts(unittest.TestCase):
    def fixture(self):
        K=np.array([[600,0,320],[0,600,240],[0,0,1.]])
        R=rotations(np.array([[.25,.4,-.1]]))[0];t=np.array([.1,-.1,3.]);uv,_=project(R[None],t[None],POSE.cuboid(1.3,.11,1.1),K)
        q=np.r_[uv[0],[[340.,220.]]];return q,K,np.array([1.3,.11,1.1])
    def test_noop_cap_permutation(self):
        q,K,xyz=self.fixture();q[0]+=[2.,-3.];bank=build_bank(q,K,xyz,(480,640));a=bank['points']
        self.assertGreater(len(a),1);self.assertLessEqual(len(a),201);self.assertTrue(np.array_equal(a[0],q));self.assertTrue(np.array_equal(a[:,8],np.broadcast_to(q[8],a[:,8].shape)))
        self.assertLessEqual(np.linalg.norm(a[:,:8]-q[:8],axis=-1).max(),8.000001)
        perm=permute_bank(bank,'fixed')['points']
        for i in range(8):self.assertEqual(sorted(map(tuple,a[:,i])),sorted(map(tuple,perm[:,i])))
        self.assertTrue(np.array_equal(perm[0],q))
    def test_rotation_centre_and_missing(self):
        q,K,xyz=self.fixture();q[1]=[-1,-1];valid=np.ones(9,bool);valid[1]=False
        bank=build_bank(q,K,xyz,(480,640),valid);self.assertTrue((bank['points'][:,1]==[-1,-1]).all())
        valid[:4]=False;self.assertEqual(len(build_bank(q,K,xyz,(480,640),valid)['points']),1)
        # Camera-axis rotation leaves the object centre t unchanged.
        t=np.array([.1,.2,3]);centre,good=project(rotations(np.array([[.3,0,0]])),t[None],np.zeros((1,3)),K);self.assertTrue(good[0])
        homogeneous=K@t;self.assertTrue(np.allclose(centre[0,0],homogeneous[:2]/homogeneous[2],atol=1e-12))
    def test_dynamic_radial_parity_and_ties(self):
        torch.manual_seed(1);head=JointActionScorer(5,c3=2,c4=3,hidden=2,encoded=3)
        q=torch.ones((1,9,2))*20;box=torch.tensor([[2.,2.,40.,30.]]);valid=torch.ones((1,9),dtype=torch.bool);shape=torch.tensor([[64,64]])
        args=(torch.randn(1,2,8,8),torch.randn(1,3,4,4),q,box,valid,shape);ctx=torch.zeros(1,5)
        old=head(*args,context=ctx,lam=0);delta=old['candidate_displacements'];cand=q[:,None].repeat(1,222,1,1);cand[:,1:,:8]+=delta[:,:-1,None,:]
        out=head.forward_bank(*args,context=ctx,candidate_points=cand,action_valid=torch.ones((1,222),dtype=torch.bool))
        reindex=torch.cat([old['logits'][:,:,-1:],old['logits'][:,:,:-1]],-1)
        self.assertTrue(torch.allclose(out['logits'],reindex,atol=2e-6,rtol=1e-6))
        out['logits']=torch.zeros_like(out['logits']);pred,index=decode_bank(out,'J');self.assertEqual(index.item(),0);self.assertTrue(torch.equal(pred,q))
    def test_missing_backward_and_joint_bank_membership(self):
        torch.manual_seed(1);head=JointActionScorer(5,c3=2,c4=3,hidden=2,encoded=3)
        points=torch.full((1,9,2),20.);points[:,2]=float('nan');valid=torch.ones(1,9,dtype=torch.bool);valid[:,2]=False
        cand=points[:,None].repeat(1,3,1,1);cand[:,1,:8]+=1; cand[:,2,:8]-=1
        out=head.forward_bank(torch.randn(1,2,8,8),torch.randn(1,3,4,4),points,torch.tensor([[2.,2.,40.,30.]]),valid,torch.tensor([[64,64]]),context=torch.zeros(1,5),candidate_points=cand,action_valid=torch.ones(1,3,dtype=torch.bool))
        batch=dict(gt_points=points.nan_to_num(),gt_valid=valid,permutations=torch.arange(9)[None,None],group_valid=torch.ones(1,1,dtype=torch.bool))
        value,_=joint_loss(out,batch);self.assertTrue(torch.isfinite(value));value.backward()
        for name,param in head.named_parameters():
            if param.grad is not None:self.assertTrue(torch.isfinite(param.grad).all(),name)
        points=torch.ones(1,9,2);cand=points[:,None].repeat(1,2,1,1);cand[:,1,:8]+=1
        o=dict(candidate_points=cand,points_raw=points,point_support=torch.tensor([[True,False,False,False,False,False,False,False]]),action_valid=torch.ones(1,2,dtype=torch.bool),logits=torch.tensor([[[0.,1.]]]).expand(1,8,2))
        chosen,index=decode_bank(o,'J');self.assertEqual(index.item(),1);self.assertTrue(torch.equal(chosen,cand[:,1]))
if __name__=='__main__':unittest.main()
