import itertools,unittest
import numpy as np
from a_joint import joint_minimum,objective

class JointTests(unittest.TestCase):
    def test_exact_against_exhaustive(self):
        rng=np.random.default_rng(915);methods=set()
        for _ in range(60):
            costs=rng.normal(size=(4,4,3));valid=np.arange(4)[None,:]<np.array([1,2,4,4])[:,None]
            branch,audit=joint_minimum(costs,valid);methods.add(audit['method'])
            expected=min(objective(costs,np.array(b)) for b in itertools.product(range(1),range(2),range(4),range(4)))
            self.assertAlmostEqual(objective(costs,branch),expected,places=7)
        self.assertIn('certified_MILP',methods);self.assertIn('certified_linear_bound',methods)
    def test_global_not_individual_clamp(self):
        c=np.array([[[0,-10,0],[1,0,0]],[[0,12,0],[1,0,0]]],float)
        b,_=joint_minimum(c,np.ones((2,2),bool))
        self.assertEqual(b.tolist(),[0,1]);self.assertEqual(objective(c,b),1)
    def test_opposing_heads_share_branch(self):
        c=np.array([[[0,0,9],[0,8,0]]],float)
        b,_=joint_minimum(c,np.ones((1,2),bool));self.assertEqual(b.tolist(),[1])
    def test_identity_ties_and_no_annotation(self):
        b,_=joint_minimum(np.zeros((3,4,3)),np.ones((3,4),bool));self.assertEqual(b.tolist(),[0,0,0])
    def test_reject_nonfinite(self):
        with self.assertRaises(AssertionError):joint_minimum(np.full((1,1,3),np.nan),np.ones((1,1),bool))
if __name__=='__main__':unittest.main()
