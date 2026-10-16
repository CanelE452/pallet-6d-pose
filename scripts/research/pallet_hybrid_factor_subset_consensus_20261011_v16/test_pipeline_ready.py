"""Small synthetic hybrid composition checks; root owns CPU execution.

The fixture declares a controlled accepted V4 selection; it does not claim
to test a fresh V4 selector. Actual S15 Generic/SciPy solves remain natural.
No model, RGB, human label, dataset GT or production raw row is read.
"""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from . import pipeline_ready as P
from ..pallet_finite_edge_consensus_repair_20261011_v13.test_solver import fixture


def packet():
    data=fixture();native=data['points'].copy();native[[4,7]]=np.nan
    native[0]+=[.7,.4];q=native.copy();q[0]=data['points'][0]
    H=[6];obs=copy.deepcopy(data['observation'])
    obs['lines']=[line for line in obs['lines'] if line['edge'] in (0,2,8)]
    queries=[dict(query=i,calibrated_model_coverage=False,selected_xy=None) for i in range(84)]
    for line in obs['lines']:
        for index,xy in zip(line['queries'],line['support_points']):
            queries[index].update(calibrated_model_coverage=True,selected_xy=copy.deepcopy(xy))
    obs.update(id='synthetic:hybrid',session='synthetic',head_arm='IMAGE_ROLE',GT_input=False,
        initial_N3_pose=copy.deepcopy(data['pose']),predicted_N3_hidden=H,
        native_N3_points=native.tolist(),raw_logits_sha256='a'*64,queries=queries,
        corners=[dict(id=0,xy=data['points'][0].tolist(),edges=[0,8],radius_px=1.)])
    record=dict(id=0,accepted=True,candidate_xy=data['points'][0].tolist(),
        native_N3_xy=native[0].tolist(),heldout_fit_requested_excluded_ids=[0,6],
        fit_exclusion_verified=True,projected_coordinate_used_as_observation=False)
    contract=dict(schema='actual_observation_only_cornerwise_contract_v4',
        cornerwise_policy=copy.deepcopy(P.S4.POLICY),hybrid_boundary_corner_ids=[0],
        validated_boundary_corner_ids=[0],native_N3_RGB_corner_ids=list(range(1,8)),
        actual_self_hidden_ids=H,same_N3_bank_reused_across_heldouts=True,
        heldout_projection_is_observation=False,LOO_initial_prior_includes_heldout_influence=False,
        cornerwise_records=[record],cornerwise_selection=dict(controlled_fixture=True))
    row=dict(id=obs['id'],session=obs['session'],method=P.PARENT_METHOD,head_arm='IMAGE_ROLE',
        head_mode='IMAGE_ROLE',observation_supply='CORNERWISE_HYBRID',
        head_checkpoint_sha256=P.V8.ROLE_CHECKPOINT_SHA256,oracle=False,
        partial_lines_not_pose_inputs=True,final_numeric_pose_has_initial_prior=False,
        fixed_metadata=dict(preserved=True),calibration_binding=dict(sha256='b'*64),
        K=data['K'].tolist(),xyz=data['xyz'].tolist(),raw_hw=[480,640],selected_index=0,
        input_points=q.tolist(),hidden_initial=H,predicted_initial_N3_hidden=H,
        observation_contract=contract,selected_corner_ids=[0],initial_pose=copy.deepcopy(data['pose']),
        solver=dict(available=False,prior_used=False,known_dimension_constraint_used=False,hidden=H),
        observation_raw_logits_sha256=obs['raw_logits_sha256'])
    done=dict(complete=True,frames=245,geometry_rows=1960,observation_rows=735,fixed_rows=490,
        cleanup_error=None,protocol_sha256='c'*64,geometry_sha256='d'*64,inference_receipt_sha256='e'*64)
    return dict(parent_row=row,native_N3_points=native.tolist(),parent_completion=done,role_observation=obs),data


class HybridComposition(unittest.TestCase):
    def test_natural_native_and_adopted_boundary_remain_actual_fit_observations(self):
        p,data=packet();before=P.semantic(p);rows,ledger=P.solve_hybrid_packet(p)
        self.assertEqual([r['method'] for r in rows],list(P.METHODS))
        for row in rows:
            self.assertTrue(row['new_pose_estimated'])
            self.assertIn(0,row['solver']['fit_input_ids'])
            self.assertTrue(set(row['solver']['fit_input_ids'])&{1,2,3,5})
            self.assertNotIn(6,row['solver']['fit_input_ids'])
            np.testing.assert_allclose(row['actual_pose']['centroid'],data['z'][3:],atol=1e-5,rtol=0)
            self.assertTrue(row['native_N3_is_numeric_final_pose_observation'])
            self.assertFalse(row['missing_sparse_filled_for_numeric_fit'])
        self.assertEqual(ledger['actual_filtered_solver_calls'],2)
        self.assertEqual(P.semantic(p),before)

    def test_consumed_adopted_edges_native_edges_and_point_only_are_separate(self):
        p,_=packet();rows,ledger=P.solve_hybrid_packet(p)
        line=rows[0]['solver'];point=rows[1]['solver']
        self.assertEqual(line['consumed_edges'],[0,8]);self.assertEqual(line['line_edges'],[2])
        self.assertTrue(set(line['consumed_edges']).isdisjoint(line['fit_line_edges']))
        self.assertEqual(point['line_edges'],[]);self.assertEqual(point['fit_line_edges'],[])
        self.assertEqual(point['consumed_edges'],[])
        self.assertEqual(point['operation_counts']['mixed_subset_requests'],0)
        self.assertEqual(ledger['point_line_ledger']['endpoint_decision_records'],1)
        self.assertEqual(ledger['point_line_ledger']['observation_filter']['unknown_retained_edge_ids'],[2])

    def test_same_raw_four_point_hypotheses_are_reused_across_two_real_solves(self):
        p,_=packet();rows,ledger=P.solve_hybrid_packet(p)
        self.assertEqual(rows[0]['solver']['operation_counts']['local_solve_calls'],1)
        self.assertEqual(rows[1]['solver']['operation_counts']['local_solve_calls'],1)
        self.assertGreater(ledger['shared_point_bank_after_line']['generic_calls'],0)
        self.assertEqual(ledger['point_only_hypothesis_bank_operation_counts']['generic_calls'],0)
        self.assertEqual(ledger['point_only_hypothesis_bank_operation_counts']['generic_cache_misses'],0)
        self.assertGreater(ledger['point_only_hypothesis_bank_operation_counts']['generic_cache_hits'],0)

    def test_hidden_xy_and_center_do_not_change_fit_but_NEW_hidden_display_is_projection(self):
        p,_=packet();left,_=P.solve_hybrid_packet(p);changed=copy.deepcopy(p)
        for target in (changed['native_N3_points'],changed['role_observation']['native_N3_points'],
                       changed['parent_row']['input_points']):
            target[6]=[100.,100.];target[8]=[123.,234.]
        right,_=P.solve_hybrid_packet(changed)
        for a,b in zip(left,right):
            self.assertEqual(a['output_status'],b['output_status'])
            np.testing.assert_allclose(a['actual_pose']['centroid'],b['actual_pose']['centroid'],atol=1e-7,rtol=0)
            np.testing.assert_allclose(a['actual_pose']['R_cf'],b['actual_pose']['R_cf'],atol=1e-7,rtol=0)
            np.testing.assert_array_equal(np.asarray(b['native_points'])[6],np.asarray(b['solver']['projected'])[6])
            np.testing.assert_array_equal(np.asarray(b['native_points'])[8],[123.,234.])
            self.assertEqual(b['reprojected_ids'],[6]);self.assertFalse(b['reprojections_reused_as_observations'])

    def test_wrong_source_incomplete_parent_and_boundary_only_packet_reject_before_numeric_dispatch(self):
        base,_=packet()
        for kind in ('native_changed','incomplete','boundary_only','truth'):
            p=copy.deepcopy(base)
            if kind=='native_changed':p['parent_row']['input_points'][1][0]+=1.
            elif kind=='incomplete':p['parent_completion']['complete']=False
            elif kind=='boundary_only':p['parent_row']['method']='IMAGE_ROLE_BOUNDARY_ONLY'
            else:p['role_observation']['ground_truth']={}
            with self.subTest(kind=kind),patch.object(P,'PoseBank',side_effect=AssertionError('numeric dispatch')):
                with self.assertRaises(ValueError):P.solve_hybrid_packet(p)
