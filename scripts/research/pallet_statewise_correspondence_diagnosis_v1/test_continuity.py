"""Original infinite-line normal factor invariants; no data or optimization."""
from copy import deepcopy
import unittest

import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_pnp as V
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_boundary_normal_solver_v4_4 as N


class OriginalNormalFactor(unittest.TestCase):
    def setUp(self):
        self.xyz=C.cuboid(1.2,.14,1.)
        self.K=np.array([[600.,0.,320.],[0.,580.,240.],[0.,0.,1.]])
        self.scale=float(np.linalg.norm(np.ptp(self.xyz,axis=0)))
        theta=.23
        R=np.array([[np.cos(theta),0.,np.sin(theta)],[0.,1.,0.],[-np.sin(theta),0.,np.cos(theta)]])
        self.p=V._pose_vector(R,np.array([.07,-.03,4.]),self.scale)

    @staticmethod
    def line(a,b):
        direction=np.asarray(b,float)-a
        n=np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
        return dict(normal_unit_xy=n,offset_px=-float(n@a),normal_variance_px2=7.5)

    def evaluate(self,line,k=0):
        boundary=[dict(native_corner=k,state='B',lines=[line])]
        return N._normal_objective_pair(self.p,self.xyz,self.K,boundary,self.scale,.8,1.25)

    def test_endpoint_reversal_same_infinite_line_cost_and_gradient(self):
        a=np.array([260.,215.]);b=np.array([410.,251.])
        first=self.evaluate(self.line(a,b));second=self.evaluate(self.line(b,a))
        self.assertAlmostEqual(first[0],second[0],places=12)
        np.testing.assert_allclose(first[1],second[1],rtol=0.,atol=1e-12)

    def test_native_owner_change_is_not_certified_by_line_invariance(self):
        line=self.line(np.array([260.,215.]),np.array([410.,251.]))
        # The line is identical but another physical model corner is a
        # different hypothesis. Reversal alone cannot verify its owner.
        self.assertNotAlmostEqual(self.evaluate(line,k=0)[0],self.evaluate(deepcopy(line),k=4)[0],places=6)


if __name__=='__main__':unittest.main()
