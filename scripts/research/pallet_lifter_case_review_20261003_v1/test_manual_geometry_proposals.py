"""Synthetic-only tests: no model, production labels or GUI are accessed."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts.annotate import annotate_pnp as pnp
from scripts.research.pallet_lifter_case_review_20261003_v1 import manual_geometry_proposals as module


class ManualGeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = np.asarray([[610., 0., 320.], [0., 610., 240.], [0., 0., 1.]])
        self.xyz = pnp.make_pallet_keypoints_3d(1.1, 1.1, .15)[:8]
        self.contract = dict(version='synthetic_source_cuboid_fixture',
            corners=[dict(id=i, xyz_m=point.tolist()) for i, point in enumerate(self.xyz)])

    def record(self, ids=(0, 1, 2, 3), rotation=None, translation=(0., 0., 5.)):
        rotation = np.eye(3) if rotation is None else rotation
        uv = pnp.project_3d(self.xyz, rotation, np.asarray(translation), self.K)
        return dict(corners=[dict(id=i, visibility='direct_visible' if i in ids else None,
            definition_confirmed=i in ids, x=uv[i][0] if i in ids else None,
            y=uv[i][1] if i in ids else None, external_occlusion=False,
            self_occlusion=False, out_of_frame=False, definition_uncertain=False)
            for i in range(8)])

    def call(self, record=None, K=None, contract=None):
        return module.proposal_for_manual_record(
            self.record() if record is None else record,
            self.contract if contract is None else contract,
            self.K if K is None else K, (640, 480))

    def test_exact_front_four_proposes_far_states_only_and_preserves_inputs(self):
        record = self.record()
        before = copy.deepcopy(record)
        contract_before = copy.deepcopy(self.contract)
        matrix_before = self.K.copy()
        result = self.call(record)
        self.assertEqual(result['reason'], 'ok')
        self.assertEqual(set(result['proposals']), {4, 5, 6, 7})
        self.assertTrue(all(p['axis'] == 'self_occlusion' for p in result['proposals'].values()))
        self.assertEqual(result['evidence']['manual_clicked_ids'], [0, 1, 2, 3])
        np.testing.assert_allclose(result['projected'][0],
            [record['corners'][0]['x'], record['corners'][0]['y']], atol=1e-6)
        self.assertEqual(record, before)
        self.assertEqual(self.contract, contract_before)
        np.testing.assert_array_equal(self.K, matrix_before)
        self.assertTrue(all(record['corners'][i]['x'] is None for i in range(4, 8)))
        self.assertFalse(result['input_modified'])
        self.assertFalse(result['model_predictions_read'])
        self.assertEqual(len(result['helper_sha256']), 64)

    def test_insufficient_invalid_camera_and_degenerate_clicks_fail_closed(self):
        self.assertEqual(self.call(self.record((0, 1, 2)))['reason'], 'need_four_confirmed_manual_clicks')
        for bad in (np.zeros((3, 3)), np.eye(3) * -1., [[1, 2]], np.ones((3, 3)) * np.nan):
            result = self.call(K=bad)
            self.assertEqual(result['reason'], 'invalid_intrinsics')
            self.assertFalse(result['proposals'])
        record = self.record()
        for i in range(4):
            record['corners'][i]['x'], record['corners'][i]['y'] = 100. + 10. * i, 100.
        self.assertEqual(self.call(record)['reason'], 'degenerate_manual_clicks')

    def test_planar_second_physically_valid_branch_is_retained(self):
        result = self.call(self.record(translation=(0., .2, 5.)))
        self.assertGreaterEqual(result['candidate_count'], 2)
        self.assertTrue(any(c['source'].startswith('existing_planar_IPPE_branch')
                            for c in result['candidates']))
        # The mirror branch disagrees about hidden slots; do not select the
        # zero-reprojection-error pose merely to manufacture C proposals.
        self.assertFalse(result['proposals'])
        self.assertIsNone(result['evidence']['reprojection_cutoff_px'])

    def test_disagreeing_hidden_candidates_do_not_propose_either_corner(self):
        first = dict(R=pnp.euler_to_R(17., 0., 0.), t=np.asarray([0., .2, 5.]), source='synthetic_plus')
        second = dict(R=pnp.euler_to_R(-17., 0., 0.), t=np.asarray([0., .2, 5.]), source='synthetic_minus')
        with patch.object(module, '_collect_candidates', return_value=[first, second]):
            result = self.call(self.record(translation=(0., .2, 5.)))
        self.assertEqual(result['candidate_count'], 2)
        self.assertNotIn(6, result['proposals'])
        self.assertNotIn(7, result['proposals'])

    def test_outside_uses_original_bounds_and_requires_all_candidate_consensus(self):
        record = self.record((0, 3, 4, 7), translation=(2., .2, 4.))
        before = copy.deepcopy(record)
        result = self.call(record)
        self.assertEqual({i: p['axis'] for i, p in result['proposals'].items()},
            {1: 'out_of_frame', 2: 'out_of_frame', 5: 'out_of_frame', 6: 'out_of_frame'})
        self.assertEqual(record, before)
        self.assertTrue(all(record['corners'][i]['x'] is None for i in (1, 2, 5, 6)))
        first = dict(R=np.eye(3), t=np.asarray([2., .2, 4.]), source='synthetic_outside')
        second = dict(R=np.eye(3), t=np.asarray([1.8, .2, 4.]), source='synthetic_inside')
        with patch.object(module, '_collect_candidates', return_value=[first, second]):
            mixed = self.call(self.record((0, 3, 4, 7), translation=(1.8, .2, 4.)))
        self.assertEqual(mixed['candidate_count'], 2)
        self.assertEqual(mixed['proposals'][1]['axis'], 'out_of_frame')
        self.assertNotIn(5, mixed['proposals'])

    def test_inconsistent_manual_corner_order_has_no_geometric_proposals(self):
        record = self.record()
        record['corners'][0]['x'], record['corners'][1]['x'] = (
            record['corners'][1]['x'], record['corners'][0]['x'])
        result = self.call(record)
        self.assertEqual(result['reason'], 'manual_corner_order_conflict')
        self.assertFalse(result['proposals'])

    def test_existing_manual_unknown_or_occlusion_states_are_never_overwritten(self):
        record = self.record()
        record['corners'][4].update(visibility='uncertain', definition_uncertain=True)
        record['corners'][5].update(visibility='not_direct_visible', external_occlusion=True)
        before = copy.deepcopy(record)
        result = self.call(record)
        self.assertEqual(set(result['proposals']), {6, 7})
        self.assertEqual(record, before)
        self.assertTrue(all(record['corners'][i]['x'] is None for i in range(4, 8)))

    def test_bad_or_changed_canonical_contract_and_false_direct_click_rejected(self):
        changed = copy.deepcopy(self.contract)
        changed['corners'][0]['xyz_m'][0] = -.65
        self.assertEqual(self.call(contract=changed)['reason'], 'invalid_frozen_corner_contract')
        record = self.record()
        record['corners'][0]['definition_confirmed'] = False
        self.assertEqual(self.call(record)['reason'], 'invalid_direct_click')
        record = self.record()
        record['corners'][0]['self_occlusion'] = True
        self.assertEqual(self.call(record)['reason'], 'invalid_direct_click')

    def test_reflection_behind_camera_and_no_pose_are_not_status_proposals(self):
        cases = [dict(R=np.diag([-1., 1., 1.]), t=np.asarray([0., 0., 5.])),
                 dict(R=np.eye(3), t=np.asarray([0., 0., -.5]))]
        with patch.object(module, '_collect_candidates', return_value=cases):
            result = self.call()
        self.assertEqual(result['reason'], 'no_geometrically_valid_manual_pose')
        self.assertFalse(result['proposals'])


if __name__ == '__main__':
    unittest.main()
