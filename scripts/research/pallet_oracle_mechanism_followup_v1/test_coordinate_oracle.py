"""Diagnostics must not silently become attainability claims or identity remaps."""
import unittest
import numpy as np
from . import coordinate_oracle as O


def points():
    return [dict(frame_id='f', recording='r', severity='s', corner_id=i,
                 errors={'R0': a, 'TEACHER': b}, missing={'R0': False, 'TEACHER': False})
            for i, (a, b) in enumerate([(1., 30.), (30., 1.)])]


class CoordinateOracleContracts(unittest.TestCase):
    def test_whole_output_cannot_mix_points(self):
        result, _ = O.coordinate_summary(points(), ('R0', 'TEACHER'))
        self.assertEqual(result['whole_output_by_objective']['10']['PCK']['10']['correct'], 1)
        self.assertEqual(result['per_point_fixed_identity']['PCK']['10']['correct'], 2)

    def test_order_invariant_tie(self):
        self.assertEqual(O.choose_whole(points(), ('R0', 'TEACHER'), 10), 'R0')
        self.assertEqual(O.choose_whole(points(), ('TEACHER', 'R0'), 10), 'R0')

    def test_fixed_corner_is_not_nearest_corner(self):
        pred = dict(selected_index=0, candidates=[dict(keypoints_xy=[[100, 0], [0, 0]])])
        error, missing = O.fixed_error(pred, 0, np.array([0, 0]), 800)
        self.assertEqual(error, 100)
        self.assertFalse(missing)

    def test_missing_and_mismatch_penalty_not_locally_repaired(self):
        p = points(); p[0]['errors']['R0'] = 800.; p[0]['missing']['R0'] = True
        self.assertEqual(O.local_errors(p, 'R0', 12), [800., 18.])
        pred = dict(selected_index=0, candidates=[dict(keypoints_xy=[[0, 0]])])
        self.assertEqual(O.fixed_error(pred, 0, np.array([0, 0]), 800, matched=False), (800., True))
        self.assertEqual(O.stats([0., 800.])['PCK']['10']['fraction'], .5)

    def test_cross_tab_threshold_and_missing(self):
        p = points(); p[0]['missing']['R0'] = True
        r = O.cross_tab(p, 'TEACHER', 'R0', 10)
        self.assertEqual([r[k] for k in ('both_correct', 'teacher_only', 'student_only', 'both_wrong')], [0, 1, 0, 1])

    def test_frozen_numeric_contract(self):
        path = O.C.DOC / 'ORACLE_COORDINATE_RESULTS.json'
        if not path.exists():
            self.skipTest('Run frozen scoring first')
        result = O.C.read(path)
        self.assertTrue(result['GT_DEPENDENT'])
        self.assertFalse(result['production_training_or_inference_input'])
        protocol = O.C.read(O.C.DOC / 'ORACLE_COORDINATE_PROTOCOL.json')
        for b in protocol['inputs']:
            O.C.verify(b)
        for material, n in [('PLASTIC', 985), ('WOOD', 346)]:
            groups = result['materials'][material]['strict_fixed_ID']
            self.assertEqual(groups['ALL']['points'], n)
            self.assertEqual(sum(g['points'] for name, g in groups.items() if name.startswith('recording:')), n)
            for group in groups.values():
                for threshold in ('5', '10', '20'):
                    whole = group['whole_output_by_objective'][threshold]['PCK'][threshold]['correct']
                    per = group['per_point_fixed_identity']['PCK'][threshold]['correct']
                    self.assertGreaterEqual(per, whole)
                    for arm in O.ARMS[material]:
                        self.assertGreaterEqual(whole, group['baselines'][arm]['PCK'][threshold]['correct'])
            pose = result['materials'][material]['whole_production_pose_expert_oracle']
            self.assertEqual(pose['frames'], 128 if material == 'PLASTIC' else 45)
            self.assertTrue(all(g >= -1e-12 for g in pose['gaps'].values()))
        visible = result['materials']['PLASTIC_VERIFIED66']['groups']['ALL']
        self.assertEqual([visible['baselines'][a]['PCK']['10']['correct'] for a in O.ARMS['PLASTIC']], [44, 50, 43, 43])


if __name__ == '__main__':
    unittest.main()
