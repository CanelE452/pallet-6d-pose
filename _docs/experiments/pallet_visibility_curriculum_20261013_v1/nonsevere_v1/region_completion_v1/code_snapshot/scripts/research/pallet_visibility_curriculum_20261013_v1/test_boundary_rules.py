"""Count branches, observable DOF and independent measurement invariants."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import boundary_rules as B
from scripts.research.pallet_visibility_curriculum_20261013_v1 import curriculum as M


def scene(height=.15,front=False):
    a,b,c=.55,height/2,.65
    xyz=np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                  [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])
    R=np.eye(3) if front else cv2.Rodrigues(np.array([.7,-.5,.15]))[0]
    t=np.array([.02,.01,3.5])
    K=np.array([[700.,0.,320.],[0.,700.,240.],[0.,0.,1.]])
    return xyz,R,t,K


def observation(q,xyz,K,R0,t0,count,hidden=None):
    raw=np.vstack([q,[[317.25,241.75]]])
    H=np.zeros(8,bool) if hidden is None else np.asarray(hidden,bool)
    p=M.project(xyz,R0,t0,K)
    initial_hull=cv2.convexHull(p.astype(np.float32),returnPoints=False).ravel().tolist()
    hull=cv2.convexHull(q.astype(np.float32),returnPoints=False).ravel().tolist()
    assert set(hull)==set(initial_hull)
    outline=np.isin(np.arange(8),initial_hull)
    selected=hull[:count]
    edges=[]
    for e,(a,b) in enumerate(zip(hull,hull[1:]+hull[:1])):
        delta=q[b]-q[a];normal=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
        samples=q[a]+np.linspace(.15,.85,9)[:,None]*delta
        edges.append(dict(edge=e,endpoints=[a,b],supported=True,samples=9,transition_sample_count=9,
            normal=normal.tolist(),offset=-float(q[a]@normal),inlier_count=9,inlier_fraction=1.,
            inlier_span_fraction=1.,inlier_rms_px=0.,angle_change_deg=0.,observed_points=samples.tolist(),
            observation_confidence=[1.]*9,inlier_mask=[True]*9,length_px=float(np.linalg.norm(delta))))
    corners=[];covariance={}
    for k in range(8):
        incident=[edge['edge'] for edge in edges if k in edge['endpoints']]
        refined=k in selected
        row=dict(corner=k,eligible=bool(outline[k] and not H[k]),refined=refined)
        if len(incident)==2:
            normal=np.stack([edges[e]['normal'] for e in incident])
            row.update(incident_shared_edges=incident,intersection_sine=abs(float(np.linalg.det(normal))),
                candidate=q[k].tolist(),shift_px=0.)
        corners.append(row)
        if refined:covariance[str(k)]=dict(covariance_px2=np.eye(2).tolist(),successful_pair_resamples=64,
            incident_shared_edges=incident)
    return dict(image_hw=[480,640],raw_points=raw.tolist(),fused_points=raw.tolist(),self_hidden=H.tolist(),fixed_H=H.tolist(),
        boundary_visibility=dict(expected_outline=outline.tolist(),conditional_truncation=[False]*8,
            geometric_boundary_candidates=(outline&~H).tolist()),
        visibility_appearance=dict(states=['SELF_OCCLUDED' if H[k] else 'VISIBLE_INTERIOR' for k in range(8)],
            visibility_score=[.9]*8,boundary_score=[.1]*8,local_accuracy_score=[.8]*8,
            diagnostics=dict(corners=[])),
        correction=dict(independent_image_evidence=True,reprojected_points_used=False,edge_records=edges,
            corner_records=corners,observed_refined_ids=selected,points=raw.tolist()),intersection_covariance=covariance)


class BoundaryRuleTests(unittest.TestCase):
    def test_five_actual_boundary_points_refine_only_their_independent_measurements(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t,K)
        R0=cv2.Rodrigues(np.array([.01,-.015,.006]))[0]@R;t0=t+np.array([.03,-.02,.04])
        obs=observation(q,xyz,K,R0,t0,5)
        original=M._fit;used=[]
        def capture(*args,**kwargs):
            used.append(list(args[7]));return original(*args,**kwargs)
        with patch.object(M,'_fit',capture):
            result=B.solve_boundary_rules(obs,xyz,K,R0,t0,dict(coordinate_source='RAW'))
        self.assertEqual(result['reliable_boundary_count'],5);self.assertTrue(result['accepted'])
        self.assertEqual(result['branch'],'FIVE_OR_MORE_FULL_6DOF');self.assertEqual(result['estimated_dof'],6)
        self.assertEqual(used,[result['reliable_boundary_ids']])
        self.assertLess(np.linalg.norm(np.asarray(result['t'])-t),np.linalg.norm(t0-t)/10)
        self.assertGreater(result['aggregate_before']['objective'],result['aggregate_after']['objective'])
        self.assertEqual(result['raw_points'][8],obs['raw_points'][8])

    def test_four_points_require_full_pose_geometry_and_initial_branch_prior(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t,K)
        R0=cv2.Rodrigues(np.array([.01,-.015,.006]))[0]@R;t0=t+np.array([.03,-.02,.04])
        result=B.solve_boundary_rules(observation(q,xyz,K,R0,t0,4),xyz,K,R0,t0)
        self.assertEqual(result['reliable_boundary_count'],4);self.assertTrue(result['accepted'])
        self.assertEqual(result['branch'],'FOUR_FULL_6DOF_INITIAL_PRIOR')
        self.assertEqual(len(result['stages'])-1,7);self.assertEqual(result['ambiguity_probe_conflicts'],[])
        xyz,R,t,K=scene(height=.05,front=True);q=M.project(xyz,R,t,K)
        result=B.solve_boundary_rules(observation(q,xyz,K,R,t,4),xyz,K,R,t)
        self.assertEqual(result['reliable_boundary_count'],4);self.assertFalse(result['accepted'])
        self.assertEqual(result['fallback_reason'],'unstable_pose_geometry')
        np.testing.assert_array_equal(result['R'],R);np.testing.assert_array_equal(result['t'],t)

    def test_three_points_estimate_translation_and_keep_rotation_exact(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t,K);t0=t+np.array([.03,-.02,.04])
        obs=observation(q,xyz,K,R,t0,3)
        result=B.solve_boundary_rules(obs,xyz,K,R,t0)
        self.assertEqual(result['reliable_boundary_count'],3);self.assertTrue(result['accepted'])
        self.assertEqual(result['branch'],'THREE_TRANSLATION_ONLY');self.assertEqual(result['estimated_dof'],3)
        np.testing.assert_array_equal(result['R'],R)
        self.assertLess(np.linalg.norm(np.asarray(result['t'])-t),np.linalg.norm(t0-t)/10)
        self.assertEqual(result['initial_geometry']['jacobian_rank'],3)
        self.assertEqual(result['unknown_added_ids'],[]);self.assertEqual(result['interior_supplement_ids'],[])

    def test_geometry_only_visibility_does_not_make_missing_two_line_points(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t,K);t0=t+np.array([.03,-.02,.04])
        for count in (0,1,2):
            with self.subTest(count=count):
                result=B.solve_boundary_rules(observation(q,xyz,K,R,t0,count),xyz,K,R,t0)
                self.assertEqual(result['reliable_boundary_count'],count);self.assertFalse(result['accepted'])
                self.assertGreater(result['geometric_boundary_count'],count)
                self.assertEqual(result['used_ids'],[])
                np.testing.assert_array_equal(result['R'],R);np.testing.assert_array_equal(result['t'],t0)

    def test_near_equal_distinct_four_point_candidates_are_ambiguous(self):
        xyz,R,t,K=scene();cfg=M.CurriculumConfig()
        second=cv2.Rodrigues(np.array([np.deg2rad(6),0,0]))[0]@R
        candidates=[dict(probe_index=0,R=R,t=t,objective=dict(objective=1.)),
                    dict(probe_index=1,R=second,t=t,objective=dict(objective=1.05))]
        self.assertEqual(len(B._ambiguity_records(candidates,xyz,t,cfg)),1)
        candidates[1]['objective']['objective']=1.2
        self.assertEqual(B._ambiguity_records(candidates,xyz,t,cfg),[])

    def test_original_hidden_and_center_slots_cannot_become_fit_measurements(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t,K)
        raw=observation(q,xyz,K,R,t,5);H=np.zeros(8,bool);H[raw['correction']['observed_refined_ids'][0]]=True
        obs=observation(q,xyz,K,R,t,5,H)
        result=B.solve_boundary_rules(obs,xyz,K,R,t)
        self.assertEqual(result['reliable_boundary_count'],4)
        self.assertTrue(all(not H[k] for stage in result['stages'] for k in stage['used_ids']))
        obs['fused_points'][8][0]+=1
        with self.assertRaisesRegex(ValueError,'hidden points or center'):
            B.solve_boundary_rules(obs,xyz,K,R,t)

    def test_three_point_coherent_wrong_translation_cannot_escape_initial_trust(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t+np.array([.4,0,0]),K)
        result=B.solve_boundary_rules(observation(q,xyz,K,R,t,3),xyz,K,R,t)
        self.assertEqual(result['reliable_boundary_count'],3);self.assertFalse(result['accepted'])
        self.assertEqual(result['fallback_reason'],'exceeds_initial_pose_motion_limit')
        np.testing.assert_array_equal(result['R'],R);np.testing.assert_array_equal(result['t'],t)

    def test_unavailable_raw_pixel_cannot_be_moved_even_if_not_in_fit_pool(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t,K);obs=observation(q,xyz,K,R,t,2)
        outside=next(k for k in range(8) if not obs['boundary_visibility']['expected_outline'][k])
        obs['raw_points'][outside]=[-20.,240.];obs['fused_points'][outside]=[-15.,240.]
        with self.assertRaisesRegex(ValueError,'missing/outside'):
            B.solve_boundary_rules(obs,xyz,K,R,t)

    def test_bound_other_instance_region_removes_strong_point_despite_high_RGB_support(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t,K);obs=observation(q,xyz,K,R,t,3)
        k=obs['correction']['observed_refined_ids'][0]
        obs['occluding_evidence']={k:dict(schema='independent_occluding_region_v1',native_corner=k,
            query_xy=obs['raw_points'][k],independent_of_target_palette=True,different_instance_from_target=True,
            region_contains_query=True,region_observed_rgb=True,instance_id='other:unit_fixture',
            source_reference='analytic_other_instance_region',source_type='instance_segmentation')}
        result=B.solve_boundary_rules(obs,xyz,K,R,t)
        self.assertEqual(result['reliable_boundary_count'],2);self.assertFalse(result['accepted'])
        self.assertEqual(result['states'][k],'OTHER_OBJECT_SUSPECTED')
        self.assertFalse(result['evidence']['records'][k]['external_occlusion_confirmed'])

    def test_self_consistent_truncation_flags_cannot_change_initial_projection_population(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t,K);obs=observation(q,xyz,K,R,t,5)
        for k in obs['correction']['observed_refined_ids'][:2]:
            obs['boundary_visibility']['conditional_truncation'][k]=True
            obs['boundary_visibility']['geometric_boundary_candidates'][k]=False
        with self.assertRaisesRegex(ValueError,'Conditional truncation'):
            B.solve_boundary_rules(obs,xyz,K,R,t)

    def test_changed_fused_pixel_requires_actual_intersection_and_recorded_sigma2_formula(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t,K);obs=observation(q,xyz,K,R,t,5)
        k=obs['correction']['observed_refined_ids'][0]
        obs['fused_points'][k][0]+=.25
        with self.assertRaisesRegex(ValueError,'binding'):
            B.solve_boundary_rules(obs,xyz,K,R,t)
        obs=observation(q,xyz,K,R,t,5);shift=np.array([.4,-.2]);obs['fusion']=[]
        for edge in obs['correction']['edge_records']:
            edge['observed_points']=(np.asarray(edge['observed_points'])+shift).tolist()
            edge['offset']-=float(np.asarray(edge['normal'])@shift)
        for corner in obs['correction']['corner_records']:
            if not corner['refined']:continue
            k=corner['corner'];candidate=q[k]+shift;corner['candidate']=candidate.tolist();corner['shift_px']=float(np.linalg.norm(shift))
            obs['correction']['points'][k]=candidate.tolist()
            obs['intersection_covariance'][str(k)]['candidate']=candidate.tolist()
            obs['fused_points'][k]=(q[k]+.5*shift).tolist()
            obs['fusion'].append(dict(corner=k,alpha=.5,detector_sigma_assumed_px=2.,largest_intersection_variance_px2=1.,
                applied_shift_px=float(np.linalg.norm(.5*shift))))
        result=B.solve_boundary_rules(obs,xyz,K,R,t)
        self.assertEqual(result['reliable_boundary_count'],5)
        obs['fusion'][0]['alpha']=.4
        with self.assertRaisesRegex(ValueError,'declared sigma2'):
            B.solve_boundary_rules(obs,xyz,K,R,t)

    def test_fractional_canvas_dimensions_are_not_silently_rounded(self):
        xyz,R,t,K=scene();q=M.project(xyz,R,t,K);obs=observation(q,xyz,K,R,t,5)
        with self.assertRaises(ValueError):
            B.solve_boundary_rules(obs,xyz,K,R,t,dict(image_hw=[480.5,640]))


if __name__=='__main__':unittest.main()
