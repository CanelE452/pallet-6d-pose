"""Generated geometry only; no real reference/annotation/model data is read."""
from copy import deepcopy
import unittest
from unittest import mock

import numpy as np

from . import analysis as A
from . import diagnose as D


def fixture():
    dims = [1., .15, 1.2]
    K = [[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]]
    ref = dict(R=np.eye(3).tolist(), t=[0., 0., 4.], xyz=dims, order=2)
    projections = D.reference_projections(ref, K)
    pts = projections[0].tolist()
    geo = dict(native_xyz=D.cuboid(dims).tolist(), physical_registration_Q=np.eye(3).tolist())
    p = dict(available=True, R_cf=np.eye(3).tolist(), R_physical=np.eye(3).tolist(),
             cf_extents=dims, centroid=[0., 0., 4.])
    states = ['B', 'B', 'S', 'O', 'T', 'U', 'I', 'B']
    visible = [dict(native_corner=k, xy=pts[k], source='B_TARGET_BACKGROUND_LINES') for k in (0, 1)]
    post = dict(initial_geometry=geo, states=states, frozen_visible_measurements=visible[:1])
    packet = dict(id='fixture', registered_family=dict(native_xyz=geo['native_xyz'], physical_registration_Q=geo['physical_registration_Q']),
                  activation=dict(complete_ids=[0]), solver_input=dict(K=K, states=states, visible=visible,
                  boundary_observations=[dict(native_corner=1, state='B', lines=[dict(normal_unit_xy=[1., 0.],
                      offset_px=-pts[1][0], normal_variance_px2=1.)])]))
    line = dict(id='fixture', A_original_N3=dict(points=pts, geometry=geo),
                B_original_pre_e=dict(states=states), D_post_e=dict(observation=post, native_points=pts),
                audit=[dict(original_state=s, post_e_same_owner_state=s,
                            e_owner_typed_to_post_full=False, e_owner_legacy_to_post_full=False) for s in states])
    frame = dict(id='fixture', xyz=dims, K=K, candidate_metadata=dict(box_xyxy=[0, 0, 100, 100]),
                 arms=dict(RAW_YOLO=dict(native_points=pts, coarse_pose=p)))
    return line, packet, ref, frame


class ResidualDiagnosticContracts(unittest.TestCase):
    def test_known_vertex_bijection_is_exact_model_arithmetic(self):
        dims = [1., .15, 1.2]
        own = D.owners(D.cuboid(dims), np.eye(3), dims)
        self.assertEqual(own, list(range(8)))
        reflected = D.owners(D.cuboid(dims), np.diag([-1., 1., -1.]), dims)
        self.assertEqual(sorted(reflected), list(range(8)))
        self.assertNotEqual(reflected, own)
        bad = D.cuboid(dims); bad[0] += 1e-5
        with self.assertRaises(ValueError):
            D.owners(bad, np.eye(3), dims)

    def test_one_whole_C2_phase_never_pointwise_best(self):
        _, packet, ref, _ = fixture()
        projected = D.reference_projections(ref, packet['solver_input']['K'])
        points = [projected[0][0], projected[1][1]]+[None]*6
        phase, costs = D.whole_C2_phase(points, list(range(8)), projected)
        errors = [D.point_record('x', 'fixture', k, points[k], k, 'B', 'test', 'test', False,
                  projected, phase, anchor='one-whole-object') for k in (0, 1)]
        self.assertEqual(len(costs), 2)
        self.assertTrue(any(r['error_px'] > 0 for r in errors))
        self.assertEqual(min(errors[0]['phase0_error_px'], errors[0]['phase180_error_px']), 0)
        self.assertEqual(min(errors[1]['phase0_error_px'], errors[1]['phase180_error_px']), 0)

    def test_partial_is_not_promoted_to_full_point(self):
        line, packet, ref, frame = fixture()
        out = D.analyze_CAL(line, packet, ref, frame)
        complete = [r for r in out['points'] if r['stage'] == 'ORIGINAL_E_COMPLETE']
        self.assertEqual([r['native_corner'] for r in complete], [0])
        self.assertEqual(len(out['partials']), 1)
        self.assertFalse(out['partials'][0]['is_complete_2D_point'])
        self.assertAlmostEqual(out['partials'][0]['signed_normal_residual_px'], 0.)
        self.assertTrue(all(r['state'] in ('B', 'I') for r in out['points'] if r['active_point_factor']))
        self.assertTrue(all(r['semantic_owner_status'] == D.OWNER_STATUS for r in out['points']))
        self.assertEqual(out['audit']['native_channel_model_owner_changed_n'], 0)

    def test_missing_rotation_retains_known_image_movement(self):
        r = D.movement_record('x', 'fixture', 0, [1, 2], [4, 6], 0, 'B', [None, None], None,
                              state_source='fixture')
        self.assertEqual(r['movement_px'], 5.)
        self.assertIsNone(r['input_error_px'])
        self.assertIsNone(r['good_input_lt5'])
        summary = D.movement_summary([r])
        self.assertEqual(summary['available_n'], 0)
        self.assertEqual(summary['all_movement_px']['n'], 1)

    def test_PF_good_input_damage_is_fixed_input_owner(self):
        line, _, ref, frame = fixture()
        pf = dict(native_points=deepcopy(frame['arms']['RAW_YOLO']['native_points']))
        pf['native_points'][0][0] += 20
        records = D.posefix_movements(line, frame, pf, ref)
        summary = D.movement_summary(records)
        self.assertEqual(summary['good_input_lt5_n'], 8)
        self.assertEqual(summary['damaged_good_to_gt10_n'], 1)
        self.assertEqual(records[0]['model_physical_vertex'], 0)
        self.assertIn('NOT_PF_OWN_VISIBILITY_GT', records[0]['state_source'])

    def test_no_postseal_GO_means_no_IO(self):
        with mock.patch.object(A, 'sha', side_effect=AssertionError('IO before GO')):
            with self.assertRaises(PermissionError):
                D.run(folder='missing', source_root='missing', single_folder='missing', lineage_folder='missing',
                      score_folder='missing', reference_manifest='missing', candidate_file='missing',
                      expected_prediction_seal_sha256='x', expected_candidate_seal_sha256='x',
                      expected_reference_manifest_sha256='x', output='missing')


if __name__ == '__main__':
    unittest.main()
