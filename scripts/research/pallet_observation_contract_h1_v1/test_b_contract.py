"""Predeclared analytic counterexamples; no campaign images/GT/models."""
import unittest
import numpy as np
from . import b_query as B
from . import typed_observation as T

BOX=np.array([[0.,0.],[10,0],[10,10],[0,10]])
EVIDENCE=dict(association_confirmed=True,basis='analytic_counterexample',edge_id='edge0',target_instance_id='target')


class TypedCounterexamples(unittest.TestCase):
    def edge(self,a,b):
        return dict(id='p',type='KNOWN_PHYSICAL_EDGE',semantic_owner='target',edge_evidence=EVIDENCE,
            edge_a=a,edge_b=b,center=[.5,2.],normal=[0.,1.])

    def candidate(self,a,b):return {'edge0':dict(xy=[a,b],target_instance_id='target',fixed_pose_binding_sha256='0'*64)}

    def test_case1_same_infinite_line_different_finite_segments(self):
        observation=self.edge([999.,0.],[9999.,0.])
        a=T.evaluate(observation,BOX,image_hw=(100,100),depth=np.ones(8),candidate_edges=self.candidate([0.,0.],[1.,0.]))
        b=T.evaluate(observation,BOX,image_hw=(100,100),depth=np.ones(8),candidate_edges=self.candidate([2.,0.],[3.,0.]))
        self.assertTrue(a['relation']['finite_support']);self.assertFalse(b['relation']['finite_support'])
        self.assertEqual(b['relation']['geometry_status'],'BEFORE_FINITE_ENDPOINT')

    def test_case2_virtual_corner_without_material_edge_cannot_be_precise(self):
        q=self.edge([0.,0.],[1.,0.]);q['edge_evidence']={'B_state':True,'reprojection_error_px':0.}
        result=T.evaluate(q,BOX,image_hw=(100,100),depth=np.ones(8))
        self.assertFalse(result['constraint_applied']);self.assertEqual(result['status'],'UNVERIFIED_EDGE_ASSOCIATION')

    def test_case3_internal_hole_or_occlusion_boundary_not_forced_onto_cuboid(self):
        q=dict(id='hole',type='TARGET_MEMBERSHIP_ONLY',target_points=[[3.,3.],[7.,3.],[7.,7.],[3.,7.]],
            membership_evidence=dict(basis='analytic_counterexample',target_instance_id='target'))
        result=T.evaluate(q,BOX,image_hw=(100,100),depth=np.ones(8))
        self.assertTrue(result['relation']['all_inside']);self.assertFalse(result['precise_edge_constraint'])

    def test_case4_cone_contamination_does_not_certify_target_pixels(self):
        q=dict(id='cone',type='TARGET_MEMBERSHIP_ONLY',semantic_owner='occluder',target_points=[[5.,5.]],
            membership_evidence=dict(basis='renderer_target_object_mask',target_instance_id='target'))
        self.assertFalse(T.evaluate(q,BOX,image_hw=(100,100),depth=np.ones(8))['contract_eligible_input'])
        q['semantic_owner']='target';q['membership_evidence']['basis']='SAM_prediction'
        self.assertFalse(T.evaluate(q,BOX,image_hw=(100,100),depth=np.ones(8))['contract_eligible_input'])

    def test_case5_deleted_observations_and_huge_containment_not_new_accuracy(self):
        missing=T.evaluate_fixed_universe([],['p'],BOX,image_hw=(100,100),depth=np.ones(8))
        self.assertEqual(missing['status'],'INCOMPLETE_OBSERVATION_UNIVERSE');self.assertIsNone(missing['total_cost'])
        q=dict(id='p',type='TARGET_MEMBERSHIP_ONLY',target_points=[[5.,5.]],
            membership_evidence=dict(basis='analytic_counterexample',target_instance_id='target'))
        huge=BOX*1000-5000;r=T.evaluate(q,huge,image_hw=(100,100),depth=np.ones(8))
        self.assertTrue(r['relation']['all_inside']);self.assertTrue(r['relation']['large_projection_degeneracy_possible'])
        self.assertFalse(r['candidate_ranking_allowed']);self.assertFalse(r['relation']['identifies_pose'])

    def test_case6_no_certified_input_applies_no_new_correction(self):
        r=T.evaluate_fixed_universe([dict(id='p',type='UNRESOLVED')],['p'],BOX,image_hw=(100,100))
        self.assertEqual(r['new_correction_applied'],0);self.assertEqual(r['status'],'NO_CERTIFIED_OBSERVATION_NO_NEW_CORRECTION')
        self.assertTrue(r['identity_output_is_not_safety_evidence'])

    def test_unknown_camera_depth_never_certifies_membership(self):
        q=dict(id='p',type='TARGET_MEMBERSHIP_ONLY',target_points=[[5.,5.]],membership_evidence=dict(basis='analytic_counterexample',target_instance_id='target'))
        self.assertFalse(T.evaluate(q,BOX,image_hw=(100,100))['constraint_applied'])

    def test_candidate_binding_required_and_stale_observation_edge_ignored(self):
        q=self.edge([0.,0.],[1.,0.]);kw=dict(image_hw=(100,100),depth=np.ones(8))
        self.assertEqual(T.evaluate(q,BOX,**kw)['status'],'UNBOUND_CANDIDATE_EDGE')
        candidates=self.candidate([0.,0.],[1.,0.]);first=T.evaluate(q,BOX,candidate_edges=candidates,**kw)
        q['edge_a']=[999.,999.];q['edge_b']=[-999.,-999.]
        self.assertEqual(first,T.evaluate(q,BOX,candidate_edges=candidates,**kw))
        self.assertFalse(first['independent_physical_ownership_verified'])


class QueryGeometryTests(unittest.TestCase):
    def test_intersection_matches_independent_linear_system(self):
        a,b,c,n=np.array([2.,3.]),np.array([8.,5.]),np.array([4.,10.]),np.array([.6,.8])
        lam,offset=np.linalg.solve(np.column_stack((b-a,-n)),c-a)
        r=B.finite_intersection(a,b,c,n,image_hw=(100,100),depth=np.ones(8))
        self.assertAlmostEqual(r['lambda_segment'],lam);self.assertAlmostEqual(r['normal_offset_px'],offset)

    def test_q00_exact_and_factorial_gt_both_overrun_retained(self):
        initial=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
        reference=initial*.1;query=dict(support_id='p',native_corner=0,native_neighbor=1,
            along_base_px=2.,center_xy=[2.,0.],outward_normal_xy=[0.,-1.],sample_offsets_base_px=[-1.,0.,1.],supported=True)
        r=B.factorial_query(query,initial,reference,[0.,0.],pixel_scale=1.,image_hw=(100,100),depth=np.ones(8))
        self.assertEqual(r[0]['center_xy'],query['center_xy']);self.assertEqual(r[0]['outward_normal_xy'],query['outward_normal_xy'])
        self.assertTrue(r[-1]['query_domain_overrun']);self.assertFalse(r[-1]['finite_support'])
        self.assertFalse(r[-1]['optical_distribution_bound_to_this_query']);self.assertEqual(r[-1]['support_id'],'p')

    def test_gt_direction_recomputes_normal_and_outward_sign(self):
        initial=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
        R=np.array([[0.,-1.],[1.,0.]]);ref=initial@R.T+20
        q=dict(support_id='p',native_corner=0,native_neighbor=1,along_base_px=2.,center_xy=[2.,0.],outward_normal_xy=[0.,-1.],sample_offsets_base_px=[0.],supported=True)
        r=B.factorial_query(q,initial,ref,[0.,0.],pixel_scale=1.,image_hw=(100,100),depth=np.ones(8))
        np.testing.assert_allclose(r[2]['tangent_xy'],[0.,1.]);np.testing.assert_allclose(r[2]['outward_normal_xy'],[1.,0.])

    def test_parallel_depth_and_degenerate_reasons(self):
        self.assertEqual(B.finite_intersection([0,0],[1,0],[0,1],[1,0],image_hw=(10,10))['geometry_status'],'PARALLEL_NORMAL_AND_EDGE')
        self.assertEqual(B.finite_intersection([0,0],[0,0],[0,1],[0,1],image_hw=(10,10))['geometry_status'],'DEGENERATE_PROJECTED_EDGE')
        self.assertEqual(B.finite_intersection([0,0],[1,0],[0,1],[0,1],image_hw=(10,10),depth=np.array([-1.]))['geometry_status'],'NONPOSITIVE_DEPTH')


if __name__=='__main__':unittest.main()
