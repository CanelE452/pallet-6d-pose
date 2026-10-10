"""Meaningful synthetic contracts; root owns every CPU execution.

Natural Generic/SciPy recovery is separate from explicitly injected numerical
roots used only to test branch preservation and ambiguity dispatch.
"""
import copy
from itertools import combinations
from math import comb
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from . import solver as S
from ..pallet_finite_edge_consensus_repair_20261011_v13.test_solver import fixture


EMPTY=dict(corners=[],lines=[],queries=[])


def sparse(data,ids):
    points=data['points'].copy();points[:8]=np.nan
    points[list(ids)]=data['points'][list(ids)]
    return points


def selected_lines(data,edges):
    return dict(corners=[],lines=[copy.deepcopy(l) for l in data['observation']['lines'] if l['edge'] in edges],queries=[])


def true_accepted(result,data):
    return [c for c in result['all_candidate_solutions'] if c['accepted'] and
        np.max(np.abs(np.asarray(c['centroid'])-data['z'][3:]))<1e-5]


class FactorSubsetContract(unittest.TestCase):
    def test_natural_direct_roots_do_not_require_selected_point_or_initial_pose(self):
        data=fixture();points=sparse(data,(0,1,2,3,4,5))
        solved=S.FactorSubsetConsensusBank(points,data['K'],data['xyz']).solve(EMPTY,
            initial_pose=dict(available=False),point_result=dict(available=False,state='AMBIGUOUS_PNP'))
        self.assertEqual(solved['operation_counts']['proposal_optimizer_calls'],0)
        self.assertEqual(solved['operation_counts']['direct_four_ID_requests'],comb(6,4))
        self.assertTrue(solved['direct_point_roots_do_not_require_point_solver_available'])
        self.assertTrue(true_accepted(solved,data))
        self.assertTrue(all(c['proposal_kind']=='DIRECT_FINITE_FOUR_POINT_ROOT' for c in solved['proposal_solutions']))

    def test_natural_wrong_one_and_two_points_leave_correct_pose_candidate(self):
        for bad in ((2,),(2,6)):
            with self.subTest(bad=bad):
                data=fixture();points=data['points'].copy()
                for k in bad:points[k]+=[40.,-40.]
                solved=S.FactorSubsetConsensusBank(points,data['K'],data['xyz']).solve(EMPTY,initial_pose=dict(available=False))
                candidates=true_accepted(solved,data);self.assertTrue(candidates)
                for c in candidates:
                    self.assertTrue(set(c['fit_input_ids']).isdisjoint(bad));self.assertGreaterEqual(len(c['fit_input_ids']),4)
                    self.assertTrue(c['stable_incidence_guard']['accepted'])
                self.assertIn(solved['state'],('NEW_POSE','AMBIGUOUS_FINITE_CONSENSUS'))
                self.assertLessEqual(solved['operation_counts']['local_optimizer_calls'],solved['call_bounds']['LOCAL_optimizer_calls_at_most'])

    def test_natural_line_outlier_can_be_excluded_without_rejecting_remaining_points(self):
        data=fixture();points=sparse(data,(0,1,2,3,4,5));observation=selected_lines(data,(2,))
        line=observation['lines'][0];normal=np.asarray(line['normal']);line['offset']+=40.
        line['support_points']=(np.asarray(line['support_points'])+40.*normal).tolist()
        solved=S.FactorSubsetConsensusBank(points,data['K'],data['xyz']).solve(observation,initial_pose=dict(available=False))
        candidates=true_accepted(solved,data);self.assertTrue(candidates)
        self.assertEqual(solved['line_edges'],[2])
        self.assertTrue(all(2 not in c['fit_line_edges'] for c in candidates))
        self.assertTrue(any(2 in a['refit']['removed_line_ids'] for a in solved['optimizer_attempts']))

    def test_natural_zero_point_three_line_has_actual_normal_incidence(self):
        data=fixture();points=sparse(data,());observation=selected_lines(data,(0,2,9))
        solved=S.FactorSubsetConsensusBank(points,data['K'],data['xyz']).solve(observation,initial_pose=data['pose'])
        self.assertEqual(solved['operation_counts']['mixed_subset_requests'],1)
        self.assertEqual(solved['operation_counts']['direct_point_roots'],0)
        self.assertTrue(solved['proposal_solutions']);self.assertTrue(true_accepted(solved,data))
        for c in true_accepted(solved,data):
            guard=c['stable_incidence_guard'];self.assertEqual(guard['actual_fit_input_ids'],[])
            self.assertEqual(guard['incidence_scalar_rows'],6);self.assertTrue(guard['accepted'])
        self.assertFalse(solved['mixed_root_completeness_proven'])

    def test_exhaustive_mixed_factory_and_missing_start_typed(self):
        data=fixture(square=False);points=sparse(data,(0,1));observation=selected_lines(data,(0,2,9))
        solved=S.FactorSubsetConsensusBank(points,data['K'],data['xyz']).solve(observation,initial_pose=dict(available=False))
        expected=list(combinations([('POINT',0),('POINT',1),('EDGE',0),('EDGE',2),('EDGE',9)],3))
        actual=[tuple(tuple(v) for v in s['factor_blocks']) for s in solved['mixed_subset_enumeration']]
        self.assertEqual(actual,expected);self.assertEqual(solved['call_bounds']['mixed_subsets'],10)
        self.assertEqual(solved['operation_counts']['mixed_dimension_requests'],20)
        self.assertEqual(solved['operation_counts']['local_optimizer_calls'],0)
        self.assertEqual(solved['state'],'NO_LOCAL_INITIALIZATION')
        self.assertEqual(solved['failure_category_counts']['initialization'],20)

    def test_three_points_only_and_two_lines_are_insufficient_typed(self):
        data=fixture();points=sparse(data,(0,1,2))
        solved=S.FactorSubsetConsensusBank(points,data['K'],data['xyz']).solve(EMPTY,initial_pose=data['pose'])
        self.assertEqual(solved['state'],'INSUFFICIENT_POINT_OBSERVATIONS');self.assertEqual(solved['operation_counts']['local_optimizer_calls'],0)
        solved=S.FactorSubsetConsensusBank(sparse(data,()),data['K'],data['xyz']).solve(selected_lines(data,(0,2)),initial_pose=data['pose'])
        self.assertEqual(solved['state'],'INSUFFICIENT_ACTUAL_INCIDENCE');self.assertFalse(solved['available'])
        # Explicit consensus injection isolates the four-point standard
        # contract; it does not claim natural three-inlier convergence.
        bank=S.FactorSubsetConsensusBank(sparse(data,(0,1,2,3)),data['K'],data['xyz']);real_score=bank._score
        def three_point_consensus(*args,**kwargs):
            score,projected=real_score(*args,**kwargs)
            score.update(final_inliers=[0,1,2],final_line_inliers=[],inlier_count=3,
                unique_line_inlier_count=0,total_inlier_factor_count=3)
            return score,projected
        with patch.object(bank,'_score',side_effect=three_point_consensus):
            solved=bank.solve(EMPTY,initial_pose=dict(available=False))
        self.assertTrue(solved['proposal_solutions']);self.assertFalse(solved['available'])
        self.assertEqual(solved['operation_counts']['refit_optimizer_calls'],0)
        self.assertTrue(all(a['refit']['rejected']=='INSUFFICIENT_POINT_INLIER_OBSERVATIONS'
            for a in solved['optimizer_attempts']))

    def test_degenerate_four_point_layout_is_distinct_from_numeric_generation(self):
        data=fixture();points=sparse(data,(0,1,2,3));points[:4]=np.array([[100.+20*k,100.] for k in range(4)])
        degenerate_bank=S.FactorSubsetConsensusBank(points,data['K'],data['xyz'])
        solved=degenerate_bank.solve(EMPTY,initial_pose=data['pose'])
        self.assertEqual(solved['state'],'DEGENERATE_POINT_OBSERVATIONS')
        self.assertTrue(all(g['generation_state']=='DEGENERATE_POINT_SUBSET' for g in solved['direct_point_generation']))
        replay=degenerate_bank.solve(EMPTY,initial_pose=data['pose']);self.assertEqual(replay['state'],'DEGENERATE_POINT_OBSERVATIONS')
        points=sparse(data,(0,1,2,3));bank=S.FactorSubsetConsensusBank(points,data['K'],data['xyz'])
        with patch.object(bank.bank,'_generic',return_value=[]):solved=bank.solve(EMPTY,initial_pose=data['pose'])
        self.assertEqual(solved['state'],'NUMERICAL_GENERATION_FAILURE');self.assertFalse(solved['available'])

    def test_shared_point_bank_reuses_same_four_ID_problem(self):
        data=fixture();points=sparse(data,(0,1,2,3));shared=S.P.PoseBank(points,data['K'],data['xyz'])
        shared.solve(hidden=(),robust=True);before=shared.ledger.copy()
        bank=S.FactorSubsetConsensusBank(points,data['K'],data['xyz'],point_bank=shared)
        solved=bank.solve(EMPTY,initial_pose=dict(available=False))
        self.assertTrue(solved['direct_point_coordinate_bank_reused'])
        self.assertEqual(shared.ledger['generic_calls'],before['generic_calls'])
        self.assertGreater(solved['hypothesis_bank_operation_counts']['generic_cache_hits'],0)
        changed=points.copy();changed[0,0]+=1.
        with self.assertRaises(ValueError):S.FactorSubsetConsensusBank(changed,data['K'],data['xyz'],point_bank=shared)

    def test_raw_branches_same_ID_are_preserved_and_ambiguous(self):
        data=fixture();points=sparse(data,(0,1,2,3));bank=S.FactorSubsetConsensusBank(points,data['K'],data['xyz'])
        z1=data['z'].copy();z2=z1.copy();z2[3]+=.002
        candidates=[]
        for index,z in enumerate((z1,z2)):
            projected,_=S.S13.V5.projection_jacobian(data['model'],z,data['K'])
            candidates.append(S.Candidate(0,(0,1,2,3),index,'EXPLICIT_CPU_ROOT',z[:3].reshape(3,1),z[3:].reshape(3,1),projected))
        def explicit_optimizer(fun,z0,**kwargs):
            rr=fun(z0);kwargs['jac'](z0)
            return SimpleNamespace(success=True,x=z0.copy(),status=1,nfev=1,njev=1,cost=S.V9.soft_l1_cost(rr))
        with patch.object(bank.bank,'_generic',return_value=candidates),patch.object(S.S13,'least_squares',side_effect=explicit_optimizer):
            solved=bank.solve(EMPTY,initial_pose=dict(available=False),point_result=dict(available=False,state='AMBIGUOUS_PNP'))
        self.assertEqual(len(solved['proposal_solutions']),2);self.assertEqual(len(solved['all_candidate_solutions']),2)
        self.assertEqual(solved['state'],'AMBIGUOUS_FINITE_CONSENSUS');self.assertFalse(solved['available'])
        self.assertTrue(solved['alternatives'][0]['same_actual8px_consensus_as_selected'])
        self.assertFalse(solved['alternatives'][0]['numerically_equivalent_to_selected'])

    def test_H_coordinates_and_center_do_not_change_proposals_or_scores(self):
        data=fixture();results=[]
        for xy in ([10.,10.],[600.,450.]):
            points=data['points'].copy();points[6:8]=xy;points[8]=[123.,234.]
            results.append(S.FactorSubsetConsensusBank(points,data['K'],data['xyz']).solve(EMPTY,hidden=(6,7),initial_pose=dict(available=False)))
        for result in results:
            self.assertTrue(set(result['used']).isdisjoint((6,7)))
            self.assertTrue(all(set(p['fit_input_ids']).isdisjoint((6,7,8)) for p in result['proposal_solutions']))
            self.assertEqual(result['operation_counts']['hidden_reprojection_assignments'],2 if result['available'] else 0)
            if result['available']:np.testing.assert_allclose(np.asarray(result['points_final'])[6:8],np.asarray(result['projected'])[6:8],atol=1e-12)
            np.testing.assert_array_equal(result['points_final'][8],[123.,234.]);self.assertFalse(result['reprojected_points_reused_as_observations'])
        for a,b in zip(results[0]['proposal_solutions'],results[1]['proposal_solutions']):
            np.testing.assert_allclose(a['projected'],b['projected'],atol=1e-12)
            self.assertEqual(a['whole_pool_score'],b['whole_pool_score'])

    def test_consumed_edges_stay_absent_and_consistent_duplicates_collapse(self):
        data=fixture();points=sparse(data,(0,1));observation=copy.deepcopy(data['observation'])
        observation['corners']=[dict(id=0,xy=points[0].tolist(),edges=[0,8])]
        observation['lines'].append(copy.deepcopy(observation['lines'][1]))
        solved=S.FactorSubsetConsensusBank(points,data['K'],data['xyz']).solve(observation,adopted_boundary_corner_ids=(0,),initial_pose=dict(available=False))
        self.assertEqual(solved['consumed_edges'],[0,8]);self.assertEqual(solved['line_edges'].count(2),1)
        self.assertTrue(all(set(s['line_edges']).isdisjoint((0,8)) for s in solved['mixed_subset_enumeration']))
        self.assertEqual(solved['operation_counts']['local_optimizer_calls'],0)

    def test_mixed_numerical_failures_do_not_drop_direct_candidates(self):
        data=fixture();points=sparse(data,(0,1,2,3));observation=selected_lines(data,(2,))
        real=S.S13.least_squares
        def controlled_proposal_failure(fun,z0,**kw):
            rr=fun(z0)
            if len(rr)==9:raise ValueError('explicit mixed subset proposal failure')
            return real(fun,z0,**kw)
        with patch.object(S.S13,'least_squares',side_effect=controlled_proposal_failure):
            solved=S.FactorSubsetConsensusBank(points,data['K'],data['xyz']).solve(observation,initial_pose=data['pose'])
        self.assertGreater(solved['operation_counts']['proposal_optimizer_exceptions'],0)
        self.assertTrue(any(p['proposal_kind']=='DIRECT_FINITE_FOUR_POINT_ROOT' for p in solved['proposal_solutions']))
        self.assertTrue(solved['all_candidate_solutions'])


if __name__=='__main__':unittest.main()
