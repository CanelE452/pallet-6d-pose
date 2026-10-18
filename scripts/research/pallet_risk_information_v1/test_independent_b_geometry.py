from __future__ import annotations
import math,unittest
import numpy as np
from . import independent_b_geometry as I


class IndependentBGeometryTests(unittest.TestCase):
    def query(self,center=(.5,-2.),normal=(0.,1.)):return dict(native_corner=0,native_neighbor=1,center_xy=list(center),outward_normal_xy=list(normal))
    def evaluate(self,query=None,xy=None,depth=None):
        if xy is None:xy=np.asarray([[0.,0.],[1.,0.]])
        if depth is None:depth=np.ones(len(xy))
        return I.finite_intersection(query or self.query(),xy,depth,2.,(100,100))
    def test_linear_system_interior_offset(self):
        a=self.evaluate();self.assertEqual(a['geometry_status'],'SUPPORTED_FINITE_SEGMENT');self.assertEqual(a['lambda_segment'],.5);self.assertEqual(a['offset_base_px'],1.);self.assertTrue(np.array_equal(a['intersection_xy'],[.5,0.]))
    def test_before_and_after_endpoints(self):
        self.assertEqual(self.evaluate(self.query((-.5,-2.)))['geometry_status'],'BEFORE_FINITE_ENDPOINT');self.assertEqual(self.evaluate(self.query((1.5,-2.)))['geometry_status'],'AFTER_FINITE_ENDPOINT')
    def test_parallel_degenerate_nonpositive(self):
        self.assertEqual(self.evaluate(self.query(normal=(1.,0.)))['geometry_status'],'PARALLEL_NORMAL_AND_EDGE');self.assertEqual(self.evaluate(xy=np.zeros((2,2)))['geometry_status'],'DEGENERATE_PROJECTED_EDGE');self.assertEqual(self.evaluate(depth=np.zeros(2))['geometry_status'],'NONPOSITIVE_DEPTH')
    def test_nonfinite_query_has_no_implicit_support(self):self.assertEqual(self.evaluate(self.query((math.nan,0.)))['geometry_status'],'NONFINITE_PROJECTION')
    def test_origin_Q_projection_is_same_physical_pose(self):
        xyz=np.array([[-1.,-2.,-3.],[1.,2.,3.]])+[.4,.5,.6];Q=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        R=I.rotation_vector_matrix([.1,.2,.3]);center=np.array([.2,.3,20.]);K=np.diag([500.,500.,1.]);Rn=R@Q.T;tn=center-Rn@xyz.mean(0)
        a,_=I.project(xyz,Rn,tn,K);b,_=I.project((xyz-xyz.mean(0))@Q,R,center,K);self.assertTrue(np.allclose(a,b,atol=1e-12,rtol=0))
    def test_dynamic_native_order_only_permutation(self):
        xyz=np.array([[i,j,k]for i in(-1.,1.)for j in(-.1,.1)for k in(-2.,2.)]);shuffled=xyz[[3,7,0,1,6,4,5,2]]
        p,gap,unique=I.permutation(xyz,shuffled);self.assertTrue(unique);self.assertEqual(gap,0.);self.assertTrue(np.array_equal(xyz,shuffled[p]))
    def test_quarter_turn_rectangle_is_not_vertex_equivalence(self):
        xyz=np.array([[i,j,k]for i in(-1.,1.)for j in(-.1,.1)for k in(-2.,2.)]);rot=I.rotation_vector_matrix([0.,math.pi/2.,0.]);_,gap,_=I.permutation(xyz@rot.T,xyz);self.assertGreater(gap,.5)
    def test_zero_and_pi_exponential(self):
        self.assertTrue(np.array_equal(I.rotation_vector_matrix([0.,0.,0.]),np.eye(3)));self.assertTrue(np.allclose(I.rotation_vector_matrix([0.,math.pi,0.]),np.diag([-1.,1.,-1.]),atol=1e-15,rtol=0))


if __name__=='__main__':unittest.main()
