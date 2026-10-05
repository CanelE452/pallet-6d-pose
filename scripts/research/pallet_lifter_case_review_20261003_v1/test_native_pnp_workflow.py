"""Real native PnP actions on synthetic temporary frames; no production UI/labels."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_native_review as nativefixtures
import open_existing_annotation as adapter
import annotate
import annotate_pnp as pnp
from object_geometry_registry import load_object_geometry_registry


class NativePnpWorkflowTests(unittest.TestCase):
    def setUp(self):
        # Use the native test fixture without inheriting/re-running its nineteen
        # unrelated tests.  All image/store/recovery writes stay in its tempdir.
        self.fixture = nativefixtures.NativeTests()
        original_class = adapter.NativeReview
        original_save = annotate._save_state_annotation
        original_manip = annotate._handle_manip_key
        self.addCleanup(setattr, annotate, '_save_state_annotation', original_save)
        self.addCleanup(setattr, annotate, '_handle_manip_key', original_manip)
        self.addCleanup(self.fixture.doCleanups)
        self.K = np.asarray([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        self.camera = dict(K=self.K.tolist(), image_hw=[480, 640],
            dimensions_wdh_m=[1.1, 1.1, .15], pnp_xyz_m=[1.1, .15, 1.1],
            distortion_coefficients=[0.] * 5,
            intrinsics_source='SYNTHETIC TEST CAMERA ONLY', metadata_sha256='0' * 64)
        self.camera_patch = patch.object(adapter, 'recorded_camera', return_value=self.camera)
        self.camera_patch.start()
        self.addCleanup(self.camera_patch.stop)
        with patch.object(nativefixtures, 'NativeReview',
                side_effect=lambda *a, **kw: original_class(*a, **kw, native_pnp=True)):
            self.fixture.setUp()
        self.native = self.fixture.native
        self.state = self.fixture.state
        self.path = self.fixture.path
        self.context = self.fixture.fixture.ctx
        self.root = self.fixture.fixture.root
        image_path = self.context.images['test:0']
        image_path.write_bytes(nativefixtures.fixtures.png(640, 480))
        frame = self.context.frames['test:0']
        frame.update(width=640, height=480,
            image_sha256=nativefixtures.fixtures.sha256(image_path))
        self.state.img = np.zeros((480, 640, 3), np.uint8)
        self.state.img_shape = self.state.img.shape
        registry_path = adapter.ROOT / 'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
        self.state.geometry_spec = load_object_geometry_registry(registry_path).resolve('plastic_square')
        self.state.geometry_registry_path = str(registry_path)
        self.state.population_role = 'DEV'
        self.state.split = 'eval'
        self.native.load(self.state, self.path)
        rotation = pnp.euler_to_R(12., -18., 2.)
        self.projected = pnp.project_3d(pnp.make_pallet_keypoints_3d(1.1, 1.1, .15),
            rotation, np.asarray([0., 0., 4.8]), self.K)

    def key(self, char):
        return annotate._handle_click_key(ord(char), self.state,
            str(self.path), '', str(self.context.images['test:0']), np.eye(3))

    def click(self, index, xy=None):
        self.key(str(index))
        self.fixture.click(*(self.projected[index] if xy is None else xy))

    def four_points(self):
        for index in range(4):
            self.click(index)
        annotate.update_pose(self.state, np.eye(3), force=True)
        self.assertIsNotNone(self.state.pose)
        self.assertLess(self.state.pose['reproj_error_px'], 1e-3)

    def assisted_documents(self):
        paths = sorted(self.root.rglob('*.PNP_ASSISTED.json'))
        self.assertTrue(paths, 'Native PnP action must save its separate assisted artifact')
        return [json.loads(path.read_text()) for path in paths]

    def manual_corners(self):
        return [copy.deepcopy(c) for c in self.native.record['corners']]

    def test_real_four_manual_points_use_verified_camera_before_any_c(self):
        self.four_points()
        self.assertEqual(sum(c['visibility'] == 'direct_visible'
            for c in self.native.record['corners']), 4)
        np.testing.assert_allclose(self.state.kps_2d[:4], self.projected[:4], atol=1e-9)
        for index in range(4):
            self.assertEqual(self.state.keypoint_annotations[index]['source'], 'manual_click')
            self.assertFalse(self.state.extrap_mask[index])
        # K=identity would not reproduce this front face at subpixel error
        # with the correct calibrated projection and physical dimensions.
        reprojected = pnp.project_3d(pnp.make_pallet_keypoints_3d(*self.state.pose['dims']),
            self.state.pose['R'], self.state.pose['t'], self.K)
        np.testing.assert_allclose(reprojected[:4], self.projected[:4], atol=1e-3)
        self.assertFalse(self.fixture.fixture.store.exists())

    def test_g_autofill_preserves_manual_reference_and_marks_projection(self):
        self.four_points()
        before = self.manual_corners()
        manual = copy.deepcopy(self.state.kps_2d[:4])
        self.key('g')  # No C or eight-state visibility gate before native PnP.
        self.assertEqual(self.state.kps_2d[:4], manual)
        for index in range(4, 8):
            self.assertIsNotNone(self.state.kps_2d[index])
            self.assertEqual(self.state.keypoint_annotations[index]['source'], 'pnp_projected')
            self.assertTrue(self.state.extrap_mask[index])
            self.assertIsNone(self.native.record['corners'][index]['x'])
            self.assertIsNone(self.native.record['corners'][index]['y'])
            self.assertNotEqual(self.native.record['corners'][index]['visibility'], 'direct_visible')
        self.assertEqual(self.manual_corners()[:4], before[:4])
        docs = self.assisted_documents()
        self.assertTrue(all(doc.get('evaluation_use') is False for doc in docs))
        self.assertEqual(self.context.export()['records'], [])

    def test_f_near_only_save_preserves_direct_clicks_and_pending_visibility(self):
        self.four_points()
        before = self.manual_corners()
        result = self.key('f')
        self.assertEqual(self.manual_corners()[:4], before[:4])
        self.assertTrue(all(c['x'] is None and c['y'] is None
            for c in self.native.record['corners'][4:]))
        self.assertIn(result, (None, 'save-next'))
        self.assertEqual(self.context.export()['records'], [])
        self.assisted_documents()

    def test_s_saves_keypoints_and_pnp_without_faking_full_visibility_review(self):
        self.four_points()
        before = self.manual_corners()
        self.assertIn(self.key('s'), (None, 'save-next'))
        self.assertEqual(self.manual_corners()[:4], before[:4])
        self.assertEqual(self.context.export()['records'], [])
        self.assertEqual(self.context.counts()['primary_reviewed'], 0)
        self.assisted_documents()

    def test_explicit_click_can_correct_one_generated_point_without_promoting_others(self):
        self.four_points()
        self.key('g')
        corrected = np.asarray(self.projected[4]) + np.asarray([.75, -.25])
        self.click(4, corrected)
        corner = self.native.record['corners'][4]
        self.assertEqual(corner['visibility'], 'direct_visible')
        np.testing.assert_allclose([corner['x'], corner['y']], corrected, atol=1e-9)
        self.assertEqual(self.state.keypoint_annotations[4]['source'], 'manual_click')
        self.assertFalse(self.state.extrap_mask[4])
        self.assertTrue(all(c['x'] is None and c['visibility'] != 'direct_visible'
            for c in self.native.record['corners'][5:]))

    def test_manipulation_save_does_not_replace_manual_ground_truth(self):
        self.four_points()
        before = self.manual_corners()
        self.state.mode = 'manip'
        self.state.locked_pose = dict(R=self.state.pose['R'].copy(),
            t=self.state.pose['t'].copy(), dims=tuple(self.state.pose['dims']))
        for name in ('_axis_assignment_candidates', '_camera_facing_hypothesis',
                '_physical_dimensions_m', '_axis_assignment'):
            if name in self.state.pose:
                self.state.locked_pose[name] = copy.deepcopy(self.state.pose[name])
        annotate._handle_manip_key(ord('d'), self.state, str(self.path), '',
            str(self.context.images['test:0']), np.eye(3))
        annotate.update_pose(self.state, np.eye(3), force=True)
        annotate._handle_manip_key(ord('s'), self.state, str(self.path), '',
            str(self.context.images['test:0']), np.eye(3))
        self.assertEqual(self.manual_corners()[:4], before[:4])
        self.assertTrue(all(c['x'] is None for c in self.native.record['corners'][4:]))
        self.assertEqual(self.context.export()['records'], [])
        self.assisted_documents()

    def test_two_line_and_extrapolation_are_not_promoted_to_manual_clicks(self):
        for index in (0, 1, 2):
            self.click(index)
        self.key('3')
        self.key('x')
        self.assertEqual(self.state.keypoint_annotations[3]['source'], 'extrapolated')
        self.assertTrue(self.state.extrap_mask[3])
        self.assertNotEqual(self.native.record['corners'][3]['visibility'], 'direct_visible')
        self.assertIsNone(self.native.record['corners'][3]['x'])
        self.key('4')
        self.key('t')
        self.assertTrue(self.state.line_mode)
        # Four line endpoint clicks generate an intersection, not a direct
        # point observation.  The helper uses original native mouse routing.
        x, y = self.projected[4]
        for xy in ((x - 6, y), (x + 6, y), (x, y - 6), (x, y + 6)):
            self.fixture.click(*xy)
        self.assertFalse(self.state.line_mode)
        self.assertEqual(self.state.keypoint_annotations[4]['source'], 'extrapolated')
        self.assertTrue(self.state.extrap_mask[4])
        self.assertNotEqual(self.native.record['corners'][4]['visibility'], 'direct_visible')
        self.assertIsNone(self.native.record['corners'][4]['x'])

    def test_g_recovery_reopen_preserves_assisted_state_and_manual_record(self):
        self.four_points()
        self.key('g')
        points = copy.deepcopy(self.state.kps_2d)
        sources = copy.deepcopy(self.state.keypoint_annotations)
        masks = copy.deepcopy(self.state.extrap_mask)
        manual = self.manual_corners()
        self.assertEqual(self.key('T'), 'quit')
        self.assertTrue(self.native.restart_requested)
        self.assertTrue(self.native.recovery_path.is_file())
        fresh = adapter.NativeReview(self.context, self.native.reviewer,
            self.native.workspace, native_pnp=True)
        fresh.path_map[str(self.path)] = ('test:0', 'primary')
        fresh.editor = annotate
        fresh.load(self.state, self.path)
        self.assertEqual(self.state.kps_2d, points)
        self.assertEqual(self.state.keypoint_annotations, sources)
        self.assertEqual(self.state.extrap_mask, masks)
        self.assertEqual(fresh.record['corners'], manual)
        self.assertEqual(self.context.export()['records'], [])


if __name__ == '__main__':
    unittest.main()
