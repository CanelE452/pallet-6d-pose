"""Visibility cancellation on synthetic temporary input; no desktop or real labels."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_native_review as fixtures


class VisibilityResetPreservesPointsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.native, self.state = self.fixture.native, self.fixture.state
        self.native.reviewer = None
        self.native.load(self.state, self.fixture.path)
        self.context = self.fixture.fixture.ctx
        self.originals = {path: path.read_bytes() for path in (
            self.fixture.fixture.manifest, self.fixture.fixture.plan,
            self.fixture.fixture.contract, self.context.images['test:0'])}
        self.points = [[4., 3.], [12., 3.], [12., 7.],
                       [4., 7.], [6., 2.], [10., 2.]]

    def complete(self):
        for index, point in enumerate(self.points):
            self.state.active = index
            self.fixture.click(*point)
        for index in (6, 7):
            self.native.mark(self.state, index, 'self_occlusion')

    def load_manual_review(self):
        self.complete()
        self.native.persist_recovery(self.state)
        self.native.load(self.state, self.fixture.path)
        return copy.deepcopy(self.native.record['corners'])

    def assert_manual_points(self, points=None):
        points = points or self.points
        self.assertEqual(self.state.kps_2d[:6], points)
        for index, point in enumerate(points):
            corner = self.native.record['corners'][index]
            self.assertEqual([corner['x'], corner['y']], point)
            self.assertEqual(corner['visibility'], 'direct_visible')
            self.assertTrue(corner['definition_confirmed'])
            self.assertFalse(self.state.extrap_mask[index])
            self.assertEqual(self.state.keypoint_annotations[index]['source'], 'manual_click')

    def assert_only_local_draft(self):
        self.assertIsNone(self.native.reviewer)
        self.assertIsNone(self.native.record['reviewer'])
        self.assertFalse(self.native.profile_path.exists())
        self.assertFalse(self.fixture.fixture.store.exists())
        self.assertFalse((self.native.ctx.store_path.parent / 'LIFTER_REFERENCE_REVIEWED.json').exists())
        self.assertEqual(self.context.export()['records'], [])
        for path, original in self.originals.items():
            self.assertEqual(path.read_bytes(), original, str(path))

    def test_r_restores_loaded_visibility_and_preserves_all_actual_clicks(self):
        baseline = self.load_manual_review()
        self.fixture.key('0')
        self.fixture.key('i')
        self.fixture.key('6')
        self.fixture.key('e')
        with (patch.object(self.native, 'save', side_effect=AssertionError('No submit from R')),
                patch('open_existing_annotation.choose_profile', side_effect=AssertionError('No profile'))):
            self.assertIsNone(self.fixture.key('r'))
        self.assertEqual(self.native.record['corners'], baseline)
        self.assert_manual_points()
        self.assertTrue(self.native.record['object']['target_identity_confirmed'])
        self.assertTrue(all(c['self_occlusion'] for c in self.native.record['corners'][6:]))
        self.assertTrue(all(c['x'] is None and c['y'] is None
                            for c in self.native.record['corners'][6:]))
        self.assert_only_local_draft()

    def test_d_cancels_only_selected_visibility_change_and_keeps_click(self):
        baseline = self.load_manual_review()
        self.fixture.key('0')
        self.fixture.key('i')
        self.fixture.key('6')
        self.fixture.key('e')
        self.fixture.key('0')
        self.fixture.key('d')
        self.assertEqual(self.native.record['corners'][0], baseline[0])
        self.assertTrue(self.native.record['corners'][6]['external_occlusion'])
        self.assert_manual_points()
        self.fixture.key('6')
        self.fixture.key('d')
        self.assertEqual(self.native.record['corners'][6], baseline[6])
        self.fixture.key('0')
        self.fixture.key('d')
        self.assert_manual_points()
        self.assert_only_local_draft()

    def test_r_keeps_new_actual_click_in_initially_blank_frame(self):
        self.fixture.click(4, 3)
        self.fixture.key('0')
        self.fixture.key('i')
        self.fixture.key('r')
        corner = self.native.record['corners'][0]
        self.assertEqual(self.state.kps_2d[0], [4., 3.])
        self.assertEqual([corner['x'], corner['y']], [4., 3.])
        self.assertEqual(corner['visibility'], 'direct_visible')
        self.assertEqual(self.state.keypoint_annotations[0]['source'], 'manual_click')
        self.assertTrue(all(c['visibility'] is None for c in self.native.record['corners'][1:]))
        self.assertTrue(self.native.record['object']['target_identity_confirmed'])
        self.assert_only_local_draft()

    def test_r_keeps_latest_actual_coordinate_after_overwrite_and_visibility_edit(self):
        self.load_manual_review()
        self.fixture.key('0')
        self.fixture.click(5, 4)
        self.fixture.key('0')
        self.fixture.key('i')
        self.fixture.key('r')
        expected = copy.deepcopy(self.points)
        expected[0] = [5., 4.]
        self.assert_manual_points(expected)
        self.assert_only_local_draft()

    def test_r_retains_new_actual_click_on_previously_hidden_corner(self):
        baseline = self.load_manual_review()
        self.assertTrue(baseline[6]['self_occlusion'])
        self.fixture.key('6')
        self.fixture.click(8, 5)
        self.fixture.key('r')
        corner = self.native.record['corners'][6]
        self.assertEqual(corner['visibility'], 'direct_visible')
        self.assertEqual([corner['x'], corner['y']], [8., 5.])
        self.assertEqual(self.state.kps_2d[6], [8., 5.])
        self.assertEqual(self.state.keypoint_annotations[6]['source'], 'manual_click')
        self.assertFalse(self.state.extrap_mask[6])
        self.assertFalse(corner['self_occlusion'])
        self.assertEqual(self.native.record['corners'][7], baseline[7])
        self.assert_manual_points()
        self.assert_only_local_draft()

    def test_z_undoes_r_and_restores_unsaved_visibility_changes(self):
        baseline = self.load_manual_review()
        self.fixture.key('0')
        self.fixture.key('i')
        self.fixture.key('6')
        self.fixture.key('e')
        previous = self.native.snapshot(self.state)
        self.fixture.key('r')
        self.assertEqual(self.native.record['corners'], baseline)
        self.fixture.key('z')
        self.assertEqual(self.native.record['corners'], previous['record']['corners'])
        self.assertEqual(self.native.record['object'], previous['record']['object'])
        self.assertEqual(self.state.kps_2d, previous['kps_2d'])
        self.assertEqual(self.state.keypoint_annotations, previous['keypoint_annotations'])
        self.assertTrue(self.native.record['corners'][0]['self_occlusion'])
        self.assertTrue(self.native.record['corners'][6]['external_occlusion'])
        self.assert_only_local_draft()

    def test_successful_s_establishes_new_visibility_checkpoint_without_profile(self):
        self.load_manual_review()
        self.fixture.key('6')
        self.fixture.key('e')
        with patch('open_existing_annotation.choose_profile', side_effect=AssertionError('No profile')):
            self.assertEqual(self.fixture.key('s'), 'save-next')
        saved = copy.deepcopy(self.native.record['corners'])
        self.fixture.key('6')
        self.fixture.key('i')
        self.fixture.key('r')
        self.assertEqual(self.native.record['corners'], saved)
        self.assertTrue(self.native.record['corners'][6]['external_occlusion'])
        self.assert_manual_points()
        self.assert_only_local_draft()

    def test_r_keeps_saved_hidden_classification_until_an_actual_click_after_save(self):
        self.load_manual_review()
        self.fixture.key('0')
        self.fixture.key('i')
        self.assertEqual(self.fixture.key('s'), 'save-next')
        saved_hidden = copy.deepcopy(self.native.record['corners'][0])
        self.assertTrue(saved_hidden['self_occlusion'])
        self.assertIsNone(saved_hidden['x'])
        self.fixture.key('r')
        self.assertEqual(self.native.record['corners'][0], saved_hidden)
        self.assertIsNone(self.state.kps_2d[0])

        # A raw click after S is new geometry that R must preserve, even though
        # the saved visibility checkpoint for this corner is hidden.
        self.fixture.key('0')
        self.fixture.click(5, 4)
        self.fixture.key('0')
        self.fixture.key('i')
        self.fixture.key('r')
        corner = self.native.record['corners'][0]
        self.assertEqual(corner['visibility'], 'direct_visible')
        self.assertEqual([corner['x'], corner['y']], [5., 4.])
        self.assertEqual(self.state.kps_2d[0], [5., 4.])
        self.assertEqual(self.state.keypoint_annotations[0]['source'], 'manual_click')
        self.assertFalse(self.state.extrap_mask[0])
        self.assert_only_local_draft()

    def test_r_does_not_promote_generated_pnp_point_from_current_state_or_history(self):
        self.state.kps_2d[0] = [7., 5.]
        self.state.extrap_mask[0] = True
        self.native.editor._set_keypoint_state(self.state, 0, [7., 5.],
            source='pnp_projected', visibility=1, reason='unknown')
        self.native.remember(self.state)
        self.native.mark(self.state, 0, 'self_occlusion')
        self.fixture.key('r')
        corner = self.native.record['corners'][0]
        self.assertIsNone(corner['visibility'])
        self.assertIsNone(corner['x'])
        self.assertIsNone(corner['y'])
        self.assertNotEqual(self.state.keypoint_annotations[0].get('source'), 'manual_click')
        self.assert_only_local_draft()

    def test_r_does_not_import_actual_points_from_other_frame_or_review_pass(self):
        foreign = self.native.snapshot(self.state)
        foreign['record']['frame_id'] = 'another:0'
        foreign['record']['corners'][0].update(
            visibility='direct_visible', definition_confirmed=True, x=4., y=3.)
        foreign['kps_2d'][0] = [4., 3.]
        self.native.history.append(foreign)
        other_pass = copy.deepcopy(foreign)
        other_pass['record'].update(frame_id='test:0', review_pass='repeat')
        self.native.history.append(other_pass)
        self.fixture.key('0')
        self.fixture.key('i')
        self.fixture.key('r')
        self.assertTrue(all(c['visibility'] is None for c in self.native.record['corners']))
        self.assertEqual(self.state.kps_2d, [None] * 9)
        self.assert_only_local_draft()


if __name__ == '__main__':
    unittest.main()
