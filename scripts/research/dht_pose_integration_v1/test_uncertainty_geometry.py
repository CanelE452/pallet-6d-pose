"""CPU analytic tests of posterior sign geometry and weighted corner fusion."""
import unittest

import numpy as np

import fusion as F
import uncertainty_geometry as U


class UncertaintyGeometryTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([[110,150],[420,150],[420,230],[110,230],
                               [180,100],[490,100],[490,180],[180,180]], float)
        self.lines, _ = F.corners_to_lines(self.truth, np.ones(8, bool))
        self.points = np.vstack([self.truth + [19,-11], [321,210]])
        self.valid = np.ones(9, bool)

    def test_unit_variances_recover_original_fusion(self):
        for lam in [0,.0625,.25,1,4]:
            a, va = F.fuse_corners(self.points,self.valid,self.lines,lam,640,480)
            b, vb = U.fuse(self.points,self.valid,self.lines,np.ones((9,2)),np.ones((8,2)),lam,640,480)
            np.testing.assert_allclose(a,b,atol=1e-11,rtol=0)
            np.testing.assert_equal(va,vb)

    def test_uncertain_lines_are_ignored_in_limit(self):
        q,_ = U.fuse(self.points,self.valid,self.lines,np.ones((9,2)),np.full((8,2),1e18),4,640,480)
        np.testing.assert_allclose(q,self.points,atol=1e-12,rtol=0)

    def test_anisotropic_point_precision(self):
        # Every corner gets x=0,y=0 constraints; uncertain x changes more than y.
        lines=np.tile(np.array([[0.,0.],[0.,10.]]),(8,1,1))
        lines[4:]=np.array([[0.,0.],[10.,0.]])
        p=np.tile([10.,10.],(9,1))
        q,_=U.fuse(p,self.valid,lines,np.tile([100.,1.],(9,1)),np.ones((8,2)),1,640,480)
        np.testing.assert_allclose(q[:8],np.tile([10/101,5],(8,1)),atol=1e-12)
        np.testing.assert_equal(q[8],p[8])

    def test_missing_center_identity_and_cap(self):
        p=self.points.copy();p[0]=np.nan
        mask=self.valid.copy();mask[1]=False
        q,v=U.fuse(p,mask,self.lines,np.ones((9,2)),np.ones((8,2)),1,640,480,max_move=2)
        np.testing.assert_equal(q[[0,1,8]],p[[0,1,8]])
        self.assertFalse(v[0]);self.assertFalse(v[1])
        self.assertLessEqual(np.nanmax(np.linalg.norm(q-p,axis=-1)),2+1e-12)
        identity,_=U.fuse(p,mask,self.lines,np.ones((9,2)),np.ones((8,2)),0,640,480)
        np.testing.assert_equal(identity,p)

    def test_uniform_scale_equivariance_above_floor(self):
        vp=np.full((9,2),30.);vl=np.full((8,2),20.)
        a,_=U.fuse(self.points,self.valid,self.lines,vp,vl,1,640,480)
        b,_=U.fuse(self.points*3,self.valid,self.lines*3,vp*9,vl*9,1,1920,1440)
        np.testing.assert_allclose(b,a*3,atol=1e-10)

    def test_opposite_hough_normals_same_physical_line_zero_spread(self):
        # (theta0,rho7) and (theta180,rho-7) are the identical line.
        m=U.posterior_moments([[.6,.4]],[0,180],[7,-7],[0],640,480)
        np.testing.assert_allclose(m['second_moment_about_mode'],0,atol=1e-24)
        np.testing.assert_allclose(m['local_mean_delta_theta_rho'],0,atol=1e-12)

    def test_seam_spread_is_small_not_179_degrees(self):
        m=U.posterior_moments([[.6,.4]],[0,179],[7,-7],[0],640,480)
        self.assertLess(m['global_angular_rms_rad'][0],np.deg2rad(1))
        self.assertEqual(m['local_mass'][0],1)

    def test_quadratic_form_equals_explicit_residual_difference(self):
        p=np.array([[.5,.3,.2]])
        theta,rho=np.array([0,45,179]),np.array([0,3,-1])
        m=U.posterior_moments(p,theta,rho,[0],640,480)
        h=U.candidate_homogeneous(theta,rho,640,480)
        h*=np.where(h[:,:2]@h[0,:2]>=0,1.,-1.)[:,None]
        z=np.array([231.,117.,1.])
        explicit=float(np.sum(p[0]*((h-h[0])@z)**2))
        compact=float(z@m['second_moment_about_mode'][0]@z)
        self.assertAlmostEqual(explicit,compact,places=9)

    def test_single_mode_zero_spread_and_psd(self):
        m=U.posterior_moments([[1,0,0]],[0,45,90],[0,0,0],[0],640,480)
        np.testing.assert_equal(m['second_moment_about_mode'],np.zeros((1,3,3)))
        np.testing.assert_equal(m['entropy_normalized'],[0])
        m=U.posterior_moments([[.5,.3,.2]],[0,45,90],[0,1,2],[0],640,480)
        self.assertGreaterEqual(np.linalg.eigvalsh(m['second_moment_about_mode']).min(),-1e-10)


if __name__=='__main__':
    unittest.main(verbosity=2)
