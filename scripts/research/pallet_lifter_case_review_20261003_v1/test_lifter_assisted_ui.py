"""Synthetic assisted-editor and recovery checks; never production input or GUI."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import test_native_review as native_fixtures
from test_native_review import annotate, ROOT
from open_existing_annotation import NativeReview


class AssistedUiTests(unittest.TestCase):
    click = native_fixtures.NativeTests.click
    key = native_fixtures.NativeTests.key

    def setUp(self):
        # Reuse the same validated temporary raw-image/contract fixture, without
        # inheriting or repeating the unrelated native-editor test methods.
        with patch('open_existing_annotation.recorded_camera', return_value=None):
            native_fixtures.NativeTests.setUp(self)

    def _fake_geometry(self, proposals):
        self.native.camera = dict(K=np.eye(3).tolist(), metadata_sha256='synthetic_camera_only')
        self.native.geometry = dict(proposals=proposals, projected=[],
            helper_sha256='synthetic_helper_only', reason='synthetic_test_only')

    def _four_manual_clicks(self):
        for index, xy in enumerate(((2., 2.), (8., 2.), (8., 7.), (2., 7.))):
            self.key(str(index))
            self.click(*xy)

    def test_c_confirms_only_pending_self_out_without_reference_coordinates(self):
        with patch.object(self.native, 'refresh_geometry'):
            self._four_manual_clicks()
            self.native.mark(self.state, 6, 'external_occlusion')
            before = copy.deepcopy(self.native.record['corners'])
            self._fake_geometry({
                0: dict(axis='out_of_frame', projected_xy=[-5., 3.]),
                4: dict(axis='self_occlusion', projected_xy=[3., 4.]),
                5: dict(axis='out_of_frame', projected_xy=[-1., 4.]),
                6: dict(axis='self_occlusion', projected_xy=[4., 4.]),
                7: dict(axis='external_occlusion', projected_xy=[5., 4.]),
            })
            with patch('tkinter.messagebox.askyesno', side_effect=AssertionError('C is already the explicit action')):
                self.native.confirm_geometry(self.state)
        after = self.native.record['corners']
        self.assertEqual(after[:4], before[:4])
        self.assertEqual(after[6], before[6])
        self.assertIsNone(after[7]['visibility'])
        self.assertTrue(after[4]['self_occlusion'])
        self.assertTrue(after[5]['out_of_frame'])
        for index in (4, 5):
            self.assertIsNone(after[index]['x'])
            self.assertIsNone(after[index]['y'])
            self.assertIsNone(self.state.kps_2d[index])
            confirmation = after[index]['geometry_confirmation']
            self.assertFalse(confirmation['guessed_coordinate_saved_as_reference'])
        assistance = self.native.record['actual_geometry_assistance'][-1]
        self.assertEqual(assistance['corner_ids'], [4, 5])
        self.assertFalse(assistance['evaluated_model_predictions_used'])
        self.assertFalse(assistance['reference_coordinates_generated'])
        self.assertFalse(self.fixture.store.exists())

    def test_assisted_save_preserves_manual_clicks_and_records_true_assistance(self):
        with patch.object(self.native, 'refresh_geometry'):
            self._four_manual_clicks()
            self._fake_geometry({i: dict(axis='self_occlusion', projected_xy=[3., 4.]) for i in range(4, 8)})
            self.native.confirm_geometry(self.state)
        self.assertTrue(self.native.save(self.state, self.path, 'reviewed'))
        record = self.fixture.ctx.export()['records'][0]
        self.assertTrue(record['reviewer']['machine_assistance'])
        self.assertFalse(record['reviewer']['previous_prediction_exposure'])
        self.assertEqual([c['x'] for c in record['corners'][:4]], [2., 8., 8., 2.])
        self.assertTrue(all(c['x'] is None and c['y'] is None for c in record['corners'][4:]))
        self.assertFalse(record['actual_geometry_assistance'][0]['evaluated_model_predictions_used'])
        self.assertEqual(self.fixture.ctx.counts()['primary_reviewed'], 1)

    def test_c_without_four_clicks_does_not_create_human_states_or_coordinates(self):
        before = copy.deepcopy(self.native.record)
        with patch('tkinter.messagebox.askyesno', side_effect=AssertionError('No missing-point popup')):
            self.key('c')
        self.assertEqual(self.native.record, before)
        self.assertEqual(self.state.kps_2d, [None] * 9)
        self.assertFalse(self.fixture.store.exists())
        self.assertNotIn('actual_geometry_assistance', self.native.record)

    def test_unconfirmed_local_recovery_never_saves_exports_or_invents_reviewer(self):
        self.native.reviewer = None
        self.native.load(self.state, self.path)
        self.click()
        before = copy.deepcopy(self.native.record)
        with patch.object(self.fixture.ctx, 'save', side_effect=AssertionError('Not an approved reference')):
            with patch.object(self.fixture.ctx, 'export', side_effect=AssertionError('Not an evaluation export')):
                with patch('open_existing_annotation.choose_profile', side_effect=AssertionError('No implicit profile answer')):
                    self.native.persist_recovery(self.state)
        recovery = json.loads(self.native.recovery_path.read_text())
        self.assertFalse(recovery['evaluation_use'])
        self.assertEqual(recovery['source_kind'], 'automatic_local_draft_recovery')
        recovered_record = recovery['drafts']['test:0|primary']['record']
        self.assertIsNone(recovered_record['reviewer'])
        self.assertEqual(recovered_record, before)
        self.assertEqual(self.fixture.ctx.store['records'], [])
        self.assertFalse(self.native.profile_path.exists())
        self.assertFalse(self.fixture.store.exists())
        self.assertFalse((self.fixture.root / 'LIFTER_REFERENCE_REVIEWED.json').exists())

    def test_resume_restores_exact_frame_view_clicks_but_never_prefills_repeat(self):
        self.native.reviewer = None
        self.native.load(self.state, self.path)
        self.click(5., 4.)
        self.state.active = 5
        self.state.zoom = 2.5
        self.state.pan = [13, 21]
        self.native.persist_recovery(self.state)
        fresh = NativeReview(self.fixture.ctx, None, self.fixture.root / 'native')
        fresh.editor = annotate
        fresh.path_map[str(self.path)] = ('test:0', 'primary')
        restored = annotate.State()
        restored.img = np.zeros((10, 20, 3), np.uint8)
        with patch('open_existing_annotation.recorded_camera', return_value=None):
            fresh.load(restored, self.path)
        self.assertEqual(restored.kps_2d[0], [5., 4.])
        self.assertEqual(restored.active, 5)
        self.assertEqual(restored.zoom, 2.5)
        self.assertEqual(restored.pan, [13, 21])
        self.assertIsNone(fresh.record['reviewer'])
        repeat = self.fixture.root / 'native/repeat/raw.json'
        fresh.path_map[str(repeat)] = ('test:0', 'repeat')
        fresh.load(restored, repeat)
        self.assertEqual(restored.kps_2d, [None] * 9)
        self.assertTrue(all(c['visibility'] is None for c in fresh.record['corners']))
        self.assertEqual(self.fixture.ctx.store['records'], [])

    def test_resume_queue_starts_at_saved_pending_frame(self):
        frame = copy.deepcopy(self.fixture.ctx.frames['test:0'])
        frame.update(frame_id='test:1', saved_frame_index=1)
        self.fixture.ctx.frames['test:1'] = frame
        self.fixture.ctx.images['test:1'] = self.fixture.root / 'other.png'
        self.native.record['frame_id'] = 'test:1'
        self.native.persist_recovery(self.state)
        fresh = NativeReview(self.fixture.ctx, None, self.fixture.root / 'native')
        from object_geometry_registry import load_object_geometry_registry
        registry = load_object_geometry_registry(ROOT / 'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')
        sessions, contexts = fresh.contexts('', argparse.Namespace(), registry, ROOT)
        self.assertEqual(contexts['review:primary']['frame_paths'][0], str(self.fixture.root / 'other.png'))
        self.assertEqual(len(sessions), 2)
        self.assertEqual(self.fixture.ctx.store['records'], [])

    def test_wheel_zooms_without_annotation_or_profile_answers(self):
        before = copy.deepcopy(self.native.record)
        self.state.zoom = 1.
        with patch('open_existing_annotation.choose_profile', side_effect=AssertionError('View-only action')):
            annotate.on_mouse(annotate.cv2.EVENT_MOUSEWHEEL, 150, 150, 120 << 16, self.state)
        self.assertGreater(self.state.zoom, 1.)
        self.assertEqual(self.native.record, before)
        self.assertEqual(self.state.kps_2d, [None] * 9)
        self.assertEqual(self.fixture.ctx.store['records'], [])

    def test_q_t_preserve_unconfirmed_actual_clicks_without_completing_review(self):
        self.native.reviewer = None
        self.native.load(self.state, self.path)
        self.click(5., 4.)
        before = copy.deepcopy(self.native.record)
        with patch('open_existing_annotation.choose_profile', side_effect=AssertionError('No invented confirmation')):
            self.assertEqual(self.key('q'), 'quit')
            self.assertFalse(self.native.restart_requested)
            self.assertEqual(self.key('t'), 'quit')
            self.assertTrue(self.native.restart_requested)
        recovered = json.loads(self.native.recovery_path.read_text())['drafts']['test:0|primary']['record']
        self.assertEqual(recovered, before)
        self.assertIsNone(recovered['reviewer'])
        self.assertEqual(self.fixture.ctx.store['records'], [])

    def test_s_partial_keeps_frame_and_never_promotes_recovery_to_reviewed(self):
        self.native.reviewer = None
        self.native.load(self.state, self.path)
        self.click()
        before = copy.deepcopy(self.native.record)
        with patch('open_existing_annotation.choose_profile', side_effect=AssertionError('Incomplete local draft')):
            self.assertIsNone(self.key('s'))
        self.assertEqual(self.native.record, before)
        self.assertEqual(self.fixture.ctx.store['records'], [])
        self.assertFalse(json.loads(self.native.recovery_path.read_text())['evaluation_use'])

    def test_frame_navigation_preserves_both_primary_and_repeat_recovery(self):
        self.native.reviewer = None
        self.native.load(self.state, self.path)
        self.click(5., 4.)
        self.assertEqual(self.key('n'), 'next')
        repeat = self.fixture.root / 'native/repeat/raw.json'
        self.native.path_map[str(repeat)] = ('test:0', 'repeat')
        self.native.load(self.state, repeat)
        self.assertEqual(self.state.kps_2d, [None] * 9)
        self.click(6., 5.)
        self.native.persist_recovery(self.state)
        drafts = json.loads(self.native.recovery_path.read_text())['drafts']
        self.assertEqual(drafts['test:0|primary']['record']['corners'][0]['x'], 5.)
        self.assertEqual(drafts['test:0|repeat']['record']['corners'][0]['x'], 6.)
        self.native.load(self.state, self.path)
        self.assertEqual(self.state.kps_2d[0], [5., 4.])
        self.assertEqual(self.fixture.ctx.store['records'], [])

    def test_change_reviewer_never_restores_previous_persons_recovery_coordinates(self):
        self.click(5., 4.)
        self.native.persist_recovery(self.state)
        previous_recovery = self.native.recovery_path.read_bytes()
        fresh = NativeReview(self.fixture.ctx, None, self.fixture.root / 'native')
        fresh.change_reviewer = True
        fresh.editor = annotate
        fresh.path_map[str(self.path)] = ('test:0', 'primary')
        restored = annotate.State()
        restored.img = np.zeros((10, 20, 3), np.uint8)
        with patch('open_existing_annotation.recorded_camera', return_value=None):
            fresh.load(restored, self.path)
        self.assertEqual(restored.kps_2d, [None] * 9)
        self.assertTrue(all(c['visibility'] is None for c in fresh.record['corners']))
        self.assertIsNone(fresh.record['reviewer'])
        self.assertFalse(fresh.record['object']['target_identity_confirmed'])
        self.assertEqual(self.native.recovery_path.read_bytes(), previous_recovery)
        self.assertEqual(self.fixture.ctx.store['records'], [])

    def test_confirmed_reviewer_mismatch_never_restores_other_persons_coordinates(self):
        self.click(5., 4.)
        self.native.persist_recovery(self.state)
        other = copy.deepcopy(self.fixture.reviewer)
        other['id'] = 'OTHER_SYNTHETIC_PERSON'
        fresh = NativeReview(self.fixture.ctx, other, self.fixture.root / 'native')
        fresh.editor = annotate
        fresh.path_map[str(self.path)] = ('test:0', 'primary')
        restored = annotate.State()
        restored.img = np.zeros((10, 20, 3), np.uint8)
        with patch('open_existing_annotation.recorded_camera', return_value=None):
            fresh.load(restored, self.path)
        self.assertEqual(restored.kps_2d, [None] * 9)
        self.assertTrue(all(c['visibility'] is None for c in fresh.record['corners']))
        self.assertEqual(fresh.record['reviewer']['id'], other['id'])
        self.assertFalse(fresh.record['reviewer']['previous_annotation_exposure'])
        self.assertFalse(fresh.record['object']['target_identity_confirmed'])
        self.assertEqual(self.fixture.ctx.store['records'], [])


if __name__ == '__main__':
    unittest.main()
