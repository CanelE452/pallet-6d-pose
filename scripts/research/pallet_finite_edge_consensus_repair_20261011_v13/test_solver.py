"""Synthetic geometry contract draft; execution requires root own CPU freeze.

Natural SciPy fixtures and explicitly injected optimizer branches are separate.
No real frame, checkpoint, image, GT dataset or model inference is loaded.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from . import solver as S


def fixture(*, square=True, fractions=(.2,.5,.8)):
    xyz=np.array([1.,.2,1. if square else .8]);K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    bank=S.FiniteEdgeConsensusBank(np.full((9,2),np.nan),K,xyz);model=bank.bank.models[0]
    z=np.array([.18,-.12,.07,.05,.04,4.]);q,_=S.V5.projection_jacobian(model,z,K)
    points=np.vstack((q,[320.,240.]));R=cv2.Rodrigues(z[:3])[0]
    pose=dict(available=True,R_cf=R.tolist(),R_physical=R.tolist(),centroid=z[3:].tolist(),cf_extents=xyz.tolist())
    raw=[]
    for edge in (0,2,4,8,9,10,11):
        a,b=S.EDGES[edge];direction=q[b]-q[a];normal=np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
        support=[(q[a]+t*direction).tolist() for t in fractions];queries=[edge*7+i for i in range(len(fractions))]
        raw.append(dict(edge=edge,endpoints=[a,b],normal=normal.tolist(),offset=float(normal@q[a]),
            queries=queries,support_points=support,query_radii_px=[1.]*len(fractions)))
    return dict(xyz=xyz,K=K,model=model,z=z,points=points,pose=pose,
        observation=dict(corners=[],lines=raw,queries=[]))


def numeric_lines(data):
    bank=S.FiniteEdgeConsensusBank(data['points'],data['K'],data['xyz'])
    return bank.factor_validator._lines(data['observation'],(),set(range(8)))[0]


class FiniteEdgeContract(unittest.TestCase):
    def test_exact_forward_J_inside_before_after(self):
        data=fixture(fractions=(-.2,.4,1.3));lines=numeric_lines(data)
        residual,J,_,blocks=S.finite_residual_jacobian(data['model'],data['z'],data['K'],data['points'],(0,4),lines)
        self.assertIn('BEFORE_FIRST_ENDPOINT',blocks[0]['hinge_branches']);self.assertIn('AFTER_SECOND_ENDPOINT',blocks[0]['hinge_branches'])
        delta=1e-6;numeric=np.empty_like(J)
        for k in range(6):
            plus=data['z'].copy();minus=data['z'].copy();plus[k]+=delta;minus[k]-=delta
            numeric[:,k]=(S.finite_residual_jacobian(data['model'],plus,data['K'],data['points'],(0,4),lines)[0]-S.finite_residual_jacobian(data['model'],minus,data['K'],data['points'],(0,4),lines)[0])/(2*delta)
        np.testing.assert_allclose(J,numeric,rtol=2e-5,atol=2e-5);self.assertTrue(np.isfinite(residual).all())

    def test_finite_support_detects_infinite_line_false_match(self):
        data=fixture(fractions=(-.2,.4,1.3));line=numeric_lines(data)[:1]
        residual,_,_,blocks=S.finite_residual_jacobian(data['model'],data['z'],data['K'],data['points'],(),line)
        np.testing.assert_allclose(residual[:2],0.,atol=1e-10)
        self.assertGreater(float(np.linalg.norm(residual[2:])),8.)
        self.assertEqual(blocks[0]['block_factor_count'],1)

    def test_partial_support_does_not_pull_endpoints_to_visible_interval(self):
        data=fixture(fractions=(.3,.4,.5));line=numeric_lines(data)[:1]
        residual,J,_,blocks=S.finite_residual_jacobian(data['model'],data['z'],data['K'],data['points'],(),line)
        np.testing.assert_allclose(residual,0.,atol=1e-10);np.testing.assert_array_equal(J[2:],0.)
        self.assertEqual(blocks[0]['hinge_branches'],['INSIDE_CLOSED_SEGMENT']*3)

    def test_query_order_and_replication_keep_one_edge_norm(self):
        data=fixture(fractions=(-.2,.4,1.3));line=numeric_lines(data)[0]
        original=S.finite_residual_jacobian(data['model'],data['z'],data['K'],data['points'],(),[line])[0]
        expanded=copy.deepcopy(line);expanded['source_line']['support_points']*=2
        expanded['source_line']['query_radii_px']*=2;expanded['support_query_ids']=[line['edge']*7+i for i in range(6)]
        expanded['source_line']['queries']=list(expanded['support_query_ids'])
        new=S.finite_residual_jacobian(data['model'],data['z'],data['K'],data['points'],(),[expanded])[0]
        self.assertAlmostEqual(float(original@original),float(new@new),places=8)
        bank=S.FiniteEdgeConsensusBank(data['points'],data['K'],data['xyz']);score,_=bank._score(data['model'],data['z'],(),[line])
        self.assertEqual(score['total_inlier_factor_count'],0);self.assertEqual(len(score['edge_block_norm_px']),1)

    def test_natural_zero_to_three_points_plus_lines(self):
        for count in range(4):
            with self.subTest(actual_points=count):
                data=fixture();points=data['points'].copy();points[:8]=np.nan;points[:count]=data['points'][:count]
                solved=S.FiniteEdgeConsensusBank(points,data['K'],data['xyz']).solve(data['observation'],initial_pose=data['pose'])
                self.assertTrue(solved['available']);self.assertEqual(solved['fit_input_ids'],list(range(count)))
                self.assertTrue(solved['fit_line_edges']);self.assertEqual(solved['operation_counts']['proposal_optimizer_calls'],1)
                self.assertEqual(solved['operation_counts']['refit_optimizer_calls'],1)
                for key in ('observed_normal_joint','modeled_normal_joint'):self.assertEqual(solved['point_line_geometry'][key]['numerical_rank'],6)

    def test_natural_wrong_unused_line_removed_remaining_pose(self):
        data=fixture();observation=copy.deepcopy(data['observation']);observation['lines']=[line for line in observation['lines'] if line['edge']==2]
        bad=observation['lines'][0];normal=np.asarray(bad['normal']);bad['offset']+=40.
        bad['support_points']=(np.asarray(bad['support_points'])+40.*normal).tolist()
        solved=S.FiniteEdgeConsensusBank(data['points'],data['K'],data['xyz']).solve(observation,initial_pose=data['pose'])
        self.assertTrue(solved['available']);self.assertEqual(solved['line_edges'],[2]);self.assertEqual(solved['fit_line_edges'],[])
        self.assertEqual(solved['fit_input_ids'],list(range(8)));np.testing.assert_allclose(solved['centroid'],data['pose']['centroid'],atol=1e-5)
        self.assertIn(2,solved['optimizer_attempts'][0]['refit']['removed_line_ids'])

    def test_natural_wrong_point_removed_remaining_pose(self):
        data=fixture();points=data['points'].copy();points[2]+=[40.,-40.]
        solved=S.FiniteEdgeConsensusBank(points,data['K'],data['xyz']).solve(dict(corners=[],lines=[],queries=[]),initial_pose=data['pose'])
        self.assertTrue(solved['available']);self.assertNotIn(2,solved['fit_input_ids']);self.assertEqual(len(solved['fit_input_ids']),7)
        np.testing.assert_allclose(solved['centroid'],data['pose']['centroid'],atol=1e-5)

    def test_consumed_edges_and_consistent_duplicates_are_not_votes(self):
        data=fixture();observation=copy.deepcopy(data['observation']);observation['corners']=[dict(id=0,xy=data['points'][0].tolist(),edges=[0,8])]
        observation['lines'].append(copy.deepcopy(observation['lines'][1]))
        solved=S.FiniteEdgeConsensusBank(data['points'],data['K'],data['xyz']).solve(observation,adopted_boundary_corner_ids=(0,),initial_pose=data['pose'])
        self.assertTrue(solved['available']);self.assertEqual(solved['consumed_edges'],[0,8]);self.assertNotIn(0,solved['fit_line_edges']);self.assertNotIn(8,solved['fit_line_edges'])
        self.assertEqual(solved['line_edges'].count(2),1)

    def test_hidden_initial_coordinates_never_enter_fit_or_score(self):
        data=fixture();results=[]
        for xy in ([10.,10.],[600.,450.]):
            points=data['points'].copy();points[6:8]=xy
            results.append(S.FiniteEdgeConsensusBank(points,data['K'],data['xyz']).solve(data['observation'],hidden=(6,7),initial_pose=data['pose']))
        self.assertTrue(all(r['available'] for r in results));np.testing.assert_allclose(results[0]['centroid'],results[1]['centroid'],atol=1e-10)
        for result in results:
            self.assertTrue(set(result['fit_input_ids']).isdisjoint({6,7}));np.testing.assert_allclose(np.asarray(result['points_final'])[6:8],np.asarray(result['projected'])[6:8],atol=1e-10)
            np.testing.assert_array_equal(result['points_final'][8],data['points'][8]);self.assertFalse(result['reprojected_points_reused_as_observations'])

    def test_incomplete_actual_support_and_inconsistent_duplicate_fail_typed(self):
        data=fixture();observation=copy.deepcopy(data['observation']);observation['lines'][0]['support_points']=observation['lines'][0]['support_points'][:2]
        solved=S.FiniteEdgeConsensusBank(data['points'],data['K'],data['xyz']).solve(observation,initial_pose=data['pose'])
        self.assertEqual(solved['state'],'INVALID_OBSERVATION_CONTRACT');self.assertEqual(solved['operation_counts']['local_optimizer_calls'],0)
        observation=copy.deepcopy(data['observation']);duplicate=copy.deepcopy(observation['lines'][0]);duplicate['offset']+=1.;observation['lines'].append(duplicate)
        solved=S.FiniteEdgeConsensusBank(data['points'],data['K'],data['xyz']).solve(observation,initial_pose=data['pose']);self.assertEqual(solved['state'],'INVALID_OBSERVATION_CONTRACT')

    def test_optimizer_exceptions_are_counted_and_preserved(self):
        data=fixture(square=False)
        with patch.object(S,'least_squares',side_effect=ValueError('explicit controlled CPU optimizer failure')):
            solved=S.FiniteEdgeConsensusBank(data['points'],data['K'],data['xyz']).solve(data['observation'],initial_pose=data['pose'])
        self.assertFalse(solved['available']);counts=solved['operation_counts'];self.assertEqual(counts['proposal_optimizer_calls'],2)
        self.assertEqual(counts['proposal_optimizer_returns'],0);self.assertEqual(counts['proposal_optimizer_exceptions'],2)
        self.assertEqual(counts['local_optimizer_calls'],counts['local_optimizer_completions']+counts['local_optimizer_exceptions'])

    def test_rank_six_can_still_be_weak_under_inherited_normalization(self):
        J=np.eye(6);J[:,5]=J[:,4]+1e-6*J[:,5];info=S.V5.rank_info(J)
        self.assertEqual(info['numerical_rank'],6);self.assertTrue(S.weak(info));self.assertEqual(info['normalization'],'unit L2 Jacobian columns')

    def test_scoring_only_collapsed_edge_is_outlier_not_frame_exception(self):
        data=fixture();line=[l for l in numeric_lines(data) if l['edge']==8]
        z=np.r_[np.zeros(3),data['xyz'][0]/2,data['xyz'][1]/2,4.]
        bank=S.FiniteEdgeConsensusBank(data['points'],data['K'],data['xyz']);score,_=bank._score(data['model'],z,(1,2,3,5,6,7),line)
        self.assertEqual(score['edge_block_norm_px'],[None]);self.assertEqual(score['final_line_inliers'],[])
        self.assertTrue(np.isfinite(score['factor_truncated_sse_px2']));self.assertIsNone(score['factor_sse_px2'])

    def test_controlled_distinct_same_consensus_non_tie_is_ambiguous(self):
        data=fixture();A=data['z'].copy();B=A.copy();A[3]-=.005;B[3]+=.005
        qa,_=S.V5.projection_jacobian(data['model'],A,data['K']);qb,_=S.V5.projection_jacobian(data['model'],B,data['K'])
        points=np.vstack((.4*qa+.6*qb,data['points'][8]));first=copy.deepcopy(data['pose']);second=copy.deepcopy(first);first['centroid']=A[3:].tolist();second['centroid']=B[3:].tolist()
        def injected(fun,z0,**kwargs):
            chosen=A if z0[3]<data['z'][3] else B;rr=fun(chosen)
            return SimpleNamespace(x=chosen.copy(),success=True,status=1,nfev=1,njev=1,cost=S.V9.soft_l1_cost(rr))
        with patch.object(S,'least_squares',side_effect=injected):
            solved=S.FiniteEdgeConsensusBank(points,data['K'],data['xyz']).solve(dict(corners=[],lines=[],queries=[]),initial_pose=first,point_result=second)
        self.assertFalse(solved['available']);self.assertEqual(solved['state'],'OBSERVATION_CONSENSUS_AMBIGUITY')
        self.assertTrue(all(c['accepted'] for c in solved['all_candidate_solutions']));self.assertTrue(any(not a['numerical_objective_tie'] for a in solved['alternatives']))
        self.assertEqual(solved['operation_counts']['local_optimizer_calls'],4);self.assertTrue(solved['unresolved_ambiguity'])

    def test_exact_duplicate_start_cache_and_both_registry_dimensions(self):
        data=fixture(square=False);bank=S.FiniteEdgeConsensusBank(data['points'],data['K'],data['xyz'])
        solved=bank.solve(data['observation'],initial_pose=data['pose'],point_result=copy.deepcopy(data['pose']))
        self.assertEqual({a['dimension_index'] for a in solved['optimizer_attempts']},{0,1})
        self.assertEqual(len(solved['optimizer_attempts']),4);self.assertEqual(solved['operation_counts']['proposal_cache_hits'],2)
        self.assertLessEqual(solved['operation_counts']['proposal_optimizer_calls'],4);self.assertLessEqual(solved['operation_counts']['refit_optimizer_calls'],4)



if __name__=='__main__':
    raise SystemExit('Use cpu_checks freeze, then cpu_checks run; direct numerical execution is not the authority.')
