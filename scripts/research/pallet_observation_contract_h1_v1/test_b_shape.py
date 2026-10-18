import unittest
import numpy as np
from .b_shape_control import CONFIG,boxes,camera,conservative_owner,project,query


class PrimitiveControlTests(unittest.TestCase):
    def test_primitive_mesh_stays_inside_registered_cuboid(self):
        W,D,H=CONFIG['target_WDH_m'];lo=np.array([-W/2,-D/2,0.]);hi=np.array([W/2,D/2,H])
        for a,b in boxes('PROCEDURAL_PALLET_WITH_OPENINGS',CONFIG):
            self.assertTrue((a>=lo-1e-15).all());self.assertTrue((b<=hi+1e-15).all())

    def test_mixed_supersamples_never_get_single_owner(self):
        high=np.array([[1,1,0,2],[1,1,0,0]])
        self.assertEqual(conservative_owner(high,[1,2],2).tolist(),[[1,-1]])

    def test_camera_is_proper_and_fixed_query_is_finite(self):
        R,eye=camera(CONFIG);np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-15);self.assertAlmostEqual(np.linalg.det(R),1.)
        q=query(CONFIG);self.assertTrue(np.isfinite(q['sample_xy']).all());self.assertEqual(len(q['sample_xy']),193)


if __name__=='__main__':unittest.main()
