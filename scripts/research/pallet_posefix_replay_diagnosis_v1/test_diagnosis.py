"""CPU contract regressions, no images / GT selection / training."""
import unittest
import numpy as np
import run as R
from analysis import cosines, types

class Contracts(unittest.TestCase):
    def test_cap_independent_corners_and_center(self):
        p=np.zeros((9,2)); q=p.copy(); q[0]=[30,40]; q[1]=[1,0]; q[8]=[999,999]
        out=R.correct(p,q,np.ones(9,bool),dict(lam=1,max_move_image_diagonal_fraction=.01),(600,800))
        np.testing.assert_allclose(out[0],[6,8]); np.testing.assert_array_equal(out[1],[1,0]); np.testing.assert_array_equal(out[8],p[8])
    def test_each_pass_cap_is_from_input(self):
        p=np.zeros((9,2)); p[0]=[10,0]; q=p.copy(); q[0]=[100,0]
        out=R.correct(p,q,np.ones(9,bool),dict(lam=1,max_move_image_diagonal_fraction=.01),(600,800))
        np.testing.assert_array_equal(out[0],[20,0])
    def test_displacement_reversal_and_zero(self):
        d=np.zeros((1,3,8,2));d[0,:,0,0]=[2,-1,1]
        c,n=cosines(d);np.testing.assert_array_equal(c[0,:,0],[-1,-1]);assert np.isnan(c[0,:,1:]).all()
    def test_symmetry_whole_branch(self):
        gt=np.arange(18).reshape(9,2).astype(float);perm=[5,4,7,6,1,0,3,2,8]
        m=R.measure(gt[perm],gt,np.ones(9,bool),[list(range(9)),perm],(600,800))
        self.assertEqual(m['branch'],1); self.assertEqual(m['E_sym'],0); self.assertGreater(m['E_fixed'],0)
    def test_crop_roundtrip(self):
        p=np.arange(18).reshape(9,2).astype(float); m=R.axis_aligned_crop_matrix([10,20,200,150])
        np.testing.assert_allclose(R.transform_points(R.transform_points(p,m),np.linalg.inv(m)),p,rtol=0,atol=1e-12)
    def test_types(self):
        errors=np.array([[[5],[3],[4],[2]]],float);norm=np.ones((1,3,1));cos=np.array([[[-1],[-1]]]);hit=np.ones((1,3,1),bool)
        t=types(errors,norm,cos,hit)
        assert t['oscillation'][0,0] and t['one_pass_best_then_regress'][0,0] and t['repeated_cap'][0,0]
        assert not t['monotonic_improve'][0,0] and not t['no_response'][0,0]

if __name__=='__main__': unittest.main()
