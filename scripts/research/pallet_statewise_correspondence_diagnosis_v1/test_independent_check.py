import unittest
import numpy as np
from .independent_check import independent_error


class IndependentMetricContract(unittest.TestCase):
    def test_C2_does_not_admit_wrong_WD_90deg(self):
        I=np.eye(3)
        ref=dict(t=[0.,0.,2.],R=I.tolist(),order=2)
        pose=dict(available=True,centroid=[.03,.04,2.],R_physical=I.tolist())
        t,r,success=independent_error(pose,ref)
        self.assertAlmostEqual(t,5.)
        self.assertEqual(r,0.)
        self.assertTrue(success)
        pose['R_physical']=np.diag([-1.,1.,-1.]).tolist()
        self.assertAlmostEqual(independent_error(pose,ref)[1],0.)
        pose['R_physical']=[[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
        self.assertAlmostEqual(independent_error(pose,ref)[1],90.)
        self.assertFalse(independent_error(pose,ref)[2])

    def test_unavailable_retains_failure_without_zero_error(self):
        self.assertEqual(independent_error(dict(available=False),dict(R=np.eye(3).tolist())),
                         (None,None,False))


if __name__=='__main__':unittest.main()
