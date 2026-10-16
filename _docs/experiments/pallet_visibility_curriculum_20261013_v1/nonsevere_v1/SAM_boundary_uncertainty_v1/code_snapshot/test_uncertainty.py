import unittest
import numpy as np

from .uncertainty import intersection_uncertainty,fuse_observations


def edge(index,points,normal,offset):
    return dict(edge=index,supported=True,observed_points=points.tolist(),observation_confidence=[1.]*len(points),
        inlier_mask=[True]*len(points),normal=normal,offset=offset)


class OutlineUncertaintyTests(unittest.TestCase):
    def correction(self,noise=0.,angle=np.pi/2,seed=9):
        rng=np.random.default_rng(seed);s=np.linspace(-30,30,25)
        normal1=np.array([0.,1.]);normal2=np.array([np.sin(angle),np.cos(angle)])
        tangent2=np.array([-normal2[1],normal2[0]])
        p1=np.column_stack([s,rng.normal(0,noise,len(s))]);p2=s[:,None]*tangent2+rng.normal(0,noise,(len(s),1))*normal2
        return dict(points=[[0.,0.]],edge_records=[edge(0,p1,normal1.tolist(),0.),edge(1,p2,normal2.tolist(),0.)],
            corner_records=[dict(corner=0,refined=True,incident_shared_edges=[0,1],candidate=[0.,0.])])

    def test_exact_raster_edges_have_positive_covariance_and_no_claim_of_zero_noise(self):
        c=self.correction();unc,lines=intersection_uncertainty(c)
        np.testing.assert_allclose(unc[0]['bootstrap_covariance_px2'],np.zeros((2,2)),atol=1e-12)
        np.testing.assert_allclose(unc[0]['covariance_px2'],np.eye(2),atol=1e-12)
        self.assertEqual(unc[0]['successful_pair_resamples'],64)

    def test_noisier_and_nearly_parallel_measured_lines_increase_uncertainty(self):
        perfect,_=intersection_uncertainty(self.correction())
        noisy,_=intersection_uncertainty(self.correction(noise=1.))
        shallow,_=intersection_uncertainty(self.correction(angle=.30))
        self.assertGreater(noisy[0]['eigenvalues_px2'][-1],perfect[0]['eigenvalues_px2'][-1])
        self.assertGreater(shallow[0]['eigenvalues_px2'][-1],perfect[0]['eigenvalues_px2'][-1]*10)

    def test_conservative_fusion_is_bounded_and_leaves_unsupported_observations_unchanged(self):
        original=np.array([[8.,5.],[20.,30.]])
        c=self.correction();c['points']=[[0.,0.],[100.,90.]];unc,_=intersection_uncertainty(c)
        output,records=fuse_observations(original,c,unc,2.)
        np.testing.assert_allclose(output[0],[4.,2.5])
        np.testing.assert_array_equal(output[1],original[1]);np.testing.assert_array_equal(original[0],[8,5])
        self.assertEqual(records[0]['alpha'],.5)
        unc[0]['eigenvalues_px2'][-1]=100.
        conservative,_=fuse_observations(original,c,unc,1.)
        self.assertLess(np.linalg.norm(conservative[0]-original[0]),.1)


if __name__=='__main__':unittest.main()
