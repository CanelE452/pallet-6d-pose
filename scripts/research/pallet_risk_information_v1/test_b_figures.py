"""Constructed representative rules only; no actual diagnostic rows/render."""
from copy import deepcopy
import unittest
from . import b_figures as B


def row(fid,phases,gtcosts=None,e0cost=0.,feasible=True):
    result=[];gtcosts=gtcosts or [1.]*len(phases)
    for i,(n,supported) in enumerate(phases):
        finite=dict(fixed_profile_count=n,supported_profiles=supported,finite_support=n==supported)
        c=dict(objective='G13_FIXED_POSE_fixed',reference_original_constraints_feasible=feasible,
            cost_comparison_valid=True,selected_h_min_slack=[1.],reference_cost=gtcosts[i],selected_E0_cost=e0cost)
        f=dict(family=B.FAMILY,raw_physical_finite_support=finite,same_objective_comparisons=[c])
        result.append(dict(phase_index=i,families=[f]))
    return dict(id=fid,registered_phases=result,selected_E0_available=True,
        selected_E0_metric_diagnostic_only=dict(success_5cm5deg=False))


class CaseRules(unittest.TestCase):
    def test_largest_unavoidable_reject(self):
        a=row('a',[(10,0),(10,9)]);b=row('b',[(10,4),(10,5)])
        c=B.select_cases([a,b],1e-7)[0]
        self.assertEqual(c['row']['id'],'b');self.assertEqual(c['selection_value'],5)

    def test_absent_not_a_pass(self):
        c=B.select_cases([row('a',[(0,0)])],1e-7)
        self.assertEqual(c[0]['status'],'NOT_APPLICABLE');self.assertEqual(c[1]['status'],'NOT_APPLICABLE')

    def test_representation_first_valid_allowed_phase(self):
        c=B.select_cases([row('a',[(5,4),(5,5)])],1e-7)[1]
        self.assertEqual(c['phase_indices'],[0,1])

    def test_score_lower_than_best_feasible_reference(self):
        a=row('a',[(3,3),(3,3)],gtcosts=[10.,2.],e0cost=3.)
        self.assertEqual(B.select_cases([a],1e-7)[2]['status'],'NOT_APPLICABLE')

    def test_selected_must_also_be_original_feasible(self):
        a=row('a',[(3,3)],gtcosts=[10.],e0cost=1.)
        a['registered_phases'][0]['families'][0]['same_objective_comparisons'][0]['selected_h_min_slack']=[-1.]
        self.assertEqual(B.select_cases([a],1e-7)[2]['status'],'NOT_APPLICABLE')

    def test_cost_gap_maximum_and_same_objective(self):
        a=row('a',[(3,3)],gtcosts=[10.],e0cost=1.)
        b=row('b',[(3,3)],gtcosts=[12.],e0cost=2.)
        c=B.select_cases([a,b],1e-7)[2]
        self.assertEqual(c['row']['id'],'b');self.assertEqual(c['selection_value'],10.)

    def test_hash_tie_independent_of_input_order(self):
        a=row('a',[(10,0)]);b=row('b',[(10,0)])
        x=B.select_cases([a,b],1e-7);y=B.select_cases([b,a],1e-7)
        self.assertEqual(x[0]['row']['id'],y[0]['row']['id'])

    def test_actual_success_never_selected_as_failed_score_case(self):
        a=row('a',[(3,3)]);a['selected_E0_metric_diagnostic_only']['success_5cm5deg']=True
        self.assertEqual(B.select_cases([a],1e-7)[2]['status'],'NOT_APPLICABLE')

    def test_real_cuboid_adjacency_not_guessed_indices(self):
        xyz=[[x,y,z] for x in [0,2] for y in [0,1] for z in [0,3]]
        self.assertEqual(len(B._edges(xyz)),12)

    def test_no_feasible_gt_not_score_conflict(self):
        a=row('a',[(3,3)],feasible=False)
        self.assertEqual(B.select_cases([a],1e-7)[2]['status'],'NOT_APPLICABLE')

    def test_selected_projection_matches_exact_original_expression(self):
        from types import SimpleNamespace
        import numpy as np
        from scipy.spatial.transform import Rotation
        from .feature_support import OLD
        xyz=np.asarray([[x,y,z] for x in [0.,1.4] for y in [0.,.18] for z in [0.,.8]])
        Q=Rotation.from_rotvec([0.,np.pi/2,0.]).as_matrix();K=np.asarray([[700.,0.,320.],[0.,700.,240.],[0.,0.,1.]])
        pose=dict(R_physical=(Rotation.from_rotvec([.1,.2,.3]).as_matrix().astype(np.float32)).tolist(),centroid=[.1,.2,3.])
        f=dict(native_xyz=xyz.tolist(),physical_registration_Q=Q.tolist())
        q,depth=B._selected_projection(dict(selected_E0_pose=pose),f,K)
        c=SimpleNamespace(xyz=xyz,Q=Q,scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3))
        expected=OLD.J.projection_and_jacobian(OLD._native_pose(pose,c),xyz,K,c.scale)
        np.testing.assert_array_equal(q,expected[0]);np.testing.assert_array_equal(depth,expected[1])


if __name__=='__main__':unittest.main()
