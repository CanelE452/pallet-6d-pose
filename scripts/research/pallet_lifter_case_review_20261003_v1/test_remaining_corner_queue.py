"""Focused unfinished-corner queues on temporary synthetic data only."""
from __future__ import annotations

import argparse
import copy
import io
import json
from contextlib import redirect_stdout
from unittest.mock import Mock, patch
import unittest

import numpy as np
from PIL import ImageDraw

import open_existing_annotation as adapter
from test_native_review import annotate, annotate_draw
from test_native_review_batch import NativeReviewBatchTests
from annotate_pnp import make_pallet_keypoints_3d


class RemainingCornerQueueTests(unittest.TestCase):
    def setUp(self):
        # The fixture owns a TemporaryDirectory and hash-bound synthetic images.
        NativeReviewBatchTests.setUp(self)
        self.parent_batch_bytes = self.batch_path.read_bytes()
        self.parent_ids = list(self.batch_ids)
        self.remaining_ids = self.parent_ids[2:7]
        self.remaining_path = self.root / 'REMAINING_FIVE_SYNTHETIC.json'
        self.remaining_batch = copy.deepcopy(self.batch)
        self.remaining_batch.update(
            task_scope='remaining_missing_corners_of_existing_batch',
            frame_ids=self.remaining_ids,
            focus_corner_ids={fid: [3 if i == 0 else 5]
                              for i, fid in enumerate(self.remaining_ids)},
            parent_frame_ids=self.parent_ids,
            purpose='Remaining work queue only; the original evaluation batch stays intact.')
        self.remaining_path.write_text(json.dumps(self.remaining_batch))
        self.context.contract['corners'] = [dict(id=i, xyz_m=xyz.tolist())
            for i, xyz in enumerate(make_pallet_keypoints_3d(1.1, 1.1, .15)[:8])]
        original_editor = {name: getattr(annotate, name) for name in (
            'on_mouse', '_handle_click_key', 'render', 'load_existing_annotation',
            'update_pose', '_handle_manip_key')}
        original_panel = annotate_draw.build_panel
        original_wait = annotate.cv2.waitKey

        def restore_editor():
            for name, value in original_editor.items():
                setattr(annotate, name, value)
            annotate_draw.build_panel = original_panel
            annotate.cv2.waitKey = original_wait

        self.addCleanup(restore_editor)
        self.native = adapter.NativeReview(self.context, None, self.root / 'native',
            native_pnp=False, batch_plan=self.remaining_path)
        # Exercise the process-local waitKey wrapper without blocking on any
        # desktop GUI or injecting keys into a real annotation process.
        self.raw_wait = Mock(return_value=-1)
        annotate.cv2.waitKey = self.raw_wait
        self.native.install(annotate)
        self.state = annotate.State()
        self.state.img = np.zeros((10, 20, 3), np.uint8)
        self.state.img_shape = self.state.img.shape
        self.state.mode = 'click'
        self.state.locked_pose = None
        self.state.zoom = 1.
        self.state.pan = [0, 0]
        self.state.line_mode = False

    def load_recovered(self, fid, missing):
        # Evidence is entirely synthetic. No native key/click handlers are used
        # to invent operator interactions in production annotations.
        record = self.fixture.sample()
        record.update(frame_id=fid, review_pass='primary', status='draft', reviewer=None)
        for c in record['corners']:
            c.update(visibility='direct_visible' if c['id'] < 6 else 'not_direct_visible',
                x=4. + c['id'] if c['id'] < 6 else None,
                y=3. if c['id'] < 6 else None,
                external_occlusion=False, self_occlusion=c['id'] >= 6,
                out_of_frame=False, definition_uncertain=False, definition_confirmed=c['id'] < 6,
                reason='SYNTHETIC UNIT TEST ONLY')
        record['corners'][missing].update(visibility=None, x=None, y=None,
            self_occlusion=False, definition_confirmed=False)
        path = self.root / 'native' / 'primary' / (fid.replace(':', '_') + '.json')
        self.native.path_map[str(path.resolve())] = (fid, 'primary')
        recovery = dict(frame_id=fid, review_pass='primary', base_revision=None,
            record=copy.deepcopy(record), view=dict(active=0, zoom=1., pan=[0, 0]))
        self.native.recovery = dict(frame_id=fid, review_pass='primary',
            drafts={fid + '|primary': recovery})
        self.assertTrue(self.native.load(self.state, path))
        return record, path

    def assert_originals_preserved(self):
        NativeReviewBatchTests.assert_originals_unchanged(self)
        self.assertEqual(self.batch_path.read_bytes(), self.parent_batch_bytes)
        self.assertFalse(self.fixture.store.exists())

    def test_current_missing_corner_overrides_recovered_active_zero(self):
        for fid, missing in ((self.remaining_ids[0], 3), (self.remaining_ids[1], 5)):
            with self.subTest(fid=fid, missing=missing):
                before, _ = self.load_recovered(fid, missing)
                self.assertEqual(self.state.active, missing)
                self.assertEqual(self.native.record['corners'], before['corners'])
                self.assertEqual(self.native.record['object'], before['object'])
                self.assertIsNone(self.state.kps_2d[missing])
                for c in before['corners']:
                    if c['visibility'] == 'direct_visible':
                        self.assertEqual(self.state.kps_2d[c['id']], [c['x'], c['y']])
        self.assert_originals_preserved()

    def test_actual_missing_state_takes_priority_over_stale_task_hint(self):
        # This queue originally points to corner3; after edits the unfinished
        # point can be5. Its current state must determine focus.
        before, _ = self.load_recovered(self.remaining_ids[0], 5)
        self.assertEqual(self.state.active, 5)
        self.assertEqual(self.native.record['corners'], before['corners'])
        self.assert_originals_preserved()

    def test_only_five_tasks_open_without_replacing_original_twelve(self):
        sessions, contexts = self.native.contexts('', argparse.Namespace(), self.registry, adapter.ROOT)
        self.assertEqual(len(sessions), 1)
        reverse = {str(path): fid for fid, path in self.context.images.items()}
        actual = [reverse[path] for path in contexts['review:primary']['frame_paths']]
        self.assertEqual(actual, self.remaining_ids)
        completed_seven = set(self.parent_ids) - set(self.remaining_ids)
        self.assertEqual(len(completed_seven), 7)
        self.assertTrue(completed_seven.isdisjoint(actual))
        self.assertEqual(self.native.requested_counts()['primary_required'], 115)
        self.assertEqual(self.native.batch_counts()['primary_required'], 5)
        self.assertFalse(self.native.batch_counts()['full_population_completed'])
        self.assert_originals_preserved()

    def test_five_task_panel_has_no_bulk_or_twelve_frame_claim(self):
        self.load_recovered(self.remaining_ids[0], 3)
        text = []
        original_text = ImageDraw.ImageDraw.text

        def capture(draw, xy, value, *args, **kwargs):
            text.append(value)
            return original_text(draw, xy, value, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, 'text', new=capture):
            panel = annotate_draw.build_panel(880, self.state.active,
                self.state.kps_2d, self.state.pose, 0, 5, self.state.zoom, False)
        self.assertEqual(panel.shape, (880, annotate_draw.PANEL_W, 3))
        self.assertNotIn(ord('B'), [key for _, key in self.native.panel_hits])
        self.assertNotIn(ord('b'), [key for _, key in self.native.panel_hits])
        self.assertTrue(any('5장' in line for line in text))
        self.assertFalse(any('12장' in line for line in text))
        self.assert_originals_preserved()

    def test_b_does_not_submit_or_open_profile_in_five_task_queue(self):
        before, path = self.load_recovered(self.remaining_ids[0], 3)
        with patch('approve_existing_clicks.approve_existing_clicks',
                side_effect=AssertionError('No bulk submission from a five-task queue')) as bulk, \
                patch.object(self.native, 'choose_history_profile',
                side_effect=AssertionError('No provenance popup from B in this queue')) as profile:
            for char in ('B', 'b'):
                self.assertIsNone(annotate._handle_click_key(ord(char), self.state,
                    str(path), '', '', np.eye(3)))
        bulk.assert_not_called()
        profile.assert_not_called()
        self.assertEqual(self.native.record['corners'], before['corners'])
        self.assertEqual(self.context.export()['records'], [])
        self.assert_originals_preserved()

    def test_raw_and_queued_lowercase_v_reach_visible_selection_before_native_guard(self):
        self.load_recovered(self.remaining_ids[0], 3)
        self.raw_wait.return_value = ord('v')
        # This is exactly the native main loop's sequence: waitKey & 0xFF,
        # then its lowercase-v split-toggle guard, then adapter dispatch.
        raw = annotate.cv2.waitKey(20) & 0xFF
        self.assertEqual(raw, ord('V'))
        self.assertNotEqual(raw, ord('v'))
        self.native.queued_key = ord('v')
        self.raw_wait.reset_mock()
        queued = annotate.cv2.waitKey(20) & 0xFF
        self.assertEqual(queued, ord('V'))
        self.raw_wait.assert_not_called()
        self.assertIsNone(self.native.queued_key)
        self.assert_originals_preserved()

    def test_lowercase_v_alias_is_limited_to_focused_visibility_queue(self):
        self.raw_wait.return_value = ord('v')
        self.native.batch['task_scope'] = 'ordinary_partial_review'
        self.assertEqual(annotate.cv2.waitKey(20) & 0xFF, ord('v'))
        self.native.batch['task_scope'] = 'remaining_missing_corners_of_existing_batch'
        self.native.native_pnp = True
        self.assertEqual(annotate.cv2.waitKey(20) & 0xFF, ord('v'))
        self.assert_originals_preserved()

    def test_focused_fit_view_uses_a_key_that_native_main_does_not_swallow(self):
        before, path = self.load_recovered(self.remaining_ids[0], 3)
        annotate_draw.build_panel(880, self.state.active,
            self.state.kps_2d, self.state.pose, 0, 5, self.state.zoom, False)
        keys = [key for _, key in self.native.panel_hits]
        self.assertIn(ord('F'), keys)
        self.assertNotIn(ord('v'), keys)
        self.state.zoom, self.state.pan = 2., [13, 17]
        self.native.queued_key = ord('F')
        key = annotate.cv2.waitKey(20) & 0xFF
        self.assertEqual(key, ord('F'))
        annotate._handle_click_key(key, self.state, str(path), '', '', np.eye(3))
        self.assertEqual(self.state.zoom, 1.)
        self.assertEqual(self.state.pan, [0, 0])
        self.assertEqual(self.native.record['corners'], before['corners'])
        self.assert_originals_preserved()

    def test_shift_s_is_save_alias_in_focused_queue_and_draft_elsewhere(self):
        self.raw_wait.return_value = ord('S')
        self.assertEqual(annotate.cv2.waitKey(20) & 0xFF, ord('s'))
        self.native.batch['task_scope'] = 'ordinary_partial_review'
        self.assertEqual(annotate.cv2.waitKey(20) & 0xFF, ord('S'))
        self.assert_originals_preserved()

    def test_save_with_a_missing_corner_reports_incomplete_without_fabricating_completion(self):
        before, path = self.load_recovered(self.remaining_ids[0], 3)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertIsNone(annotate._handle_click_key(ord('s'), self.state,
                str(path), '', '', np.eye(3)))
        self.assertIn('S_NOT_COMPLETE', output.getvalue())
        self.assertIn('[3]', output.getvalue())
        self.assertIn('미입력', self.native.notice)
        self.assertNotIn('저장 완료', self.native.notice)
        self.assertEqual(self.native.record['corners'], before['corners'])
        self.assertFalse(self.native.visibility_progress_path.exists())
        self.assertFalse(path.with_name(path.stem + '.VISIBILITY_SAVED.json').exists())
        self.assertTrue(self.native.recovery_path.exists())
        self.assertEqual(self.context.export()['records'], [])
        self.assert_originals_preserved()

    def test_raw_manual_click_and_save_preserve_coordinates_without_pnp_autofill(self):
        for fid, missing in ((self.remaining_ids[0], 3), (self.remaining_ids[1], 5)):
            with self.subTest(fid=fid, missing=missing):
                before, path = self.load_recovered(fid, missing)
                x, y = 8., 4.
                screen_x = (annotate.MARGIN_L + x - self.state.pan[0]) * self.state.zoom
                screen_y = (annotate.MARGIN_T + y - self.state.pan[1]) * self.state.zoom
                with patch('open_existing_annotation.choose_profile',
                        side_effect=AssertionError('An actual point and S must not open provenance UI')):
                    annotate.on_mouse(annotate.cv2.EVENT_LBUTTONDOWN,
                        screen_x, screen_y, 0, self.state)
                    self.assertEqual(self.state.kps_2d[missing], [x, y])
                    self.assertEqual(self.native.record['corners'][missing]['visibility'], 'direct_visible')
                    self.assertEqual(self.state.keypoint_annotations[missing]['source'], 'manual_click')
                    self.assertIsNone(self.state.pose)
                    self.assertIsNone(self.state.kps_2d[8])
                    for old in before['corners']:
                        if old['id'] != missing:
                            self.assertEqual(self.native.record['corners'][old['id']], old)
                    self.assertEqual(annotate._handle_click_key(ord('s'), self.state,
                        str(path), '', '', np.eye(3)), 'save-next')
                from save_visibility_locally import load_visibility_record
                saved = load_visibility_record(self.native, fid, 'primary')
                self.assertEqual(saved['corners'][missing]['x'], x)
                self.assertEqual(saved['corners'][missing]['y'], y)
                self.assertIsNone(saved['reviewer'])
                self.assertEqual(self.context.export()['records'], [])
                self.native.load(self.state, path)
                self.assertEqual(self.state.kps_2d[missing], [x, y])
                self.assertEqual(self.native.record['corners'][missing]['visibility'], 'direct_visible')
                self.assertIsNone(self.state.kps_2d[8])
                self.assertIsNone(self.state.pose)
        self.assert_originals_preserved()


if __name__ == '__main__':
    unittest.main()
