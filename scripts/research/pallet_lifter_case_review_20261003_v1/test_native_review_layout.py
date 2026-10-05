"""Raster and input regression checks on temporary synthetic review fixtures."""
from __future__ import annotations

import copy
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from PIL import ImageDraw

import test_native_review as native_fixtures
from test_native_review import annotate, annotate_draw
from open_existing_annotation import initial_window_size


class NativeReviewLayoutTests(unittest.TestCase):
    def setUp(self):
        # Reuse fixture setup without inheriting its unrelated action tests.
        with patch('open_existing_annotation.recorded_camera', return_value=None):
            native_fixtures.NativeTests.setUp(self)
        self.native.batch = dict(frame_ids=[f'synthetic:{i}' for i in range(12)])
        self.native.editor_state = self.state
        self.state.active = 7
        self.state.dirty = True

    def test_initial_window_fits_small_desktop_workarea(self):
        for width,height in ((1280,693),(1366,741),(1920,1053)):
            output=f'_NET_WORKAREA(CARDINAL) = 0, 27, {width}, {height}\n_NET_CURRENT_DESKTOP(CARDINAL) = 0\n'
            with self.subTest(width=width,height=height), patch('open_existing_annotation.subprocess.run',
                    return_value=subprocess.CompletedProcess([],0,stdout=output)):
                window_width,window_height=initial_window_size()
                self.assertLessEqual(window_width,width-100)
                self.assertLessEqual(window_height,height-140)

    def panel(self, height):
        return annotate_draw.build_panel(height, self.state.active,
            self.state.kps_2d, self.state.pose, 0, 12, self.state.zoom, True)

    def assert_hits_inside_panel(self, height):
        for rect, key in self.native.panel_hits:
            with self.subTest(rect=rect, key=chr(key)):
                x0, y0, x1, y1 = rect
                self.assertGreaterEqual(x0, 0)
                self.assertGreaterEqual(y0, 0)
                self.assertLessEqual(x1, annotate_draw.PANEL_W)
                self.assertLessEqual(y1, height)
                self.assertGreater(x1, x0)
                self.assertGreater(y1, y0)

    def test_footer_does_not_cover_sidebar_raster_or_last_status(self):
        self.state.img = np.zeros((480, 640, 3), np.uint8)
        self.state.img_shape = self.state.img.shape
        self.state.zoom = 1.
        self.state.pan = [0, 0]
        expected = self.panel(880)
        with patch.object(self.native, 'refresh_geometry'):
            image = annotate.render(self.state, 0, 12, 'synthetic.png')
        canvas_width = 640 + annotate.MARGIN_L + annotate.MARGIN_R
        self.assertEqual(image.shape, (880, canvas_width + annotate_draw.PANEL_W, 3))
        np.testing.assert_array_equal(image[:, canvas_width:], expected)
        self.assertEqual({key for _, key in self.native.panel_hits if ord('0') <= key <= ord('7')},
            {ord(str(i)) for i in range(8)})
        self.assert_hits_inside_panel(880)
        self.assertFalse(self.fixture.store.exists())

    def test_all_korean_text_bounds_fit_normal_panels_with_proposals(self):
        # Capture the real font metrics used to draw, including conditional
        # proposal text and dirty-state notes; checking image shape misses this.
        self.native.geometry['proposals'] = {
            i: dict(axis='self_occlusion' if i % 2 else 'out_of_frame',
                    projected_xy=[4., 3.]) for i in range(8)}
        original_text = ImageDraw.ImageDraw.text
        modes = ((False, 'click', 7), (True, 'click', 7),
                 (True, 'manip', 7), (True, 'click', 8))
        for height, (native_pnp, mode, active) in (
                (height, config) for height in (720, 880) for config in modes):
            with self.subTest(height=height, native_pnp=native_pnp, mode=mode, active=active):
                self.native.native_pnp = native_pnp
                self.state.mode = mode
                self.state.active = active
                drawn = []

                def capture(draw, xy, text, *args, **kwargs):
                    font = kwargs.get('font')
                    drawn.append((text, draw.textbbox(xy, text, font=font,
                        anchor=kwargs.get('anchor'))))
                    return original_text(draw, xy, text, *args, **kwargs)

                with patch.object(ImageDraw.ImageDraw, 'text', new=capture):
                    self.panel(height)
                self.assertTrue(drawn)
                self.assertEqual(self.native.panel_scroll, 0)
                self.assertLessEqual(self.native.panel_content_height, height)
                for text, (left, top, right, bottom) in drawn:
                    with self.subTest(text=text, bounds=(left, top, right, bottom)):
                        self.assertGreaterEqual(left, 0)
                        self.assertGreaterEqual(top, 0)
                        self.assertLessEqual(right, annotate_draw.PANEL_W)
                        self.assertLessEqual(bottom, height)
                self.assert_hits_inside_panel(height)

    def test_short_sidebar_scroll_reaches_last_status_without_zooming_image(self):
        self.state.zoom = 1.6
        self.state.pan = [8, 12]
        height = 410  # The temporary fixture's 10px raw image plus margins.
        self.panel(height)
        self.assertGreater(self.native.panel_content_height, height)
        self.assertEqual(self.native.panel_scroll, 0)
        before_record = copy.deepcopy(self.native.record)
        before_points = copy.deepcopy(self.state.kps_2d)
        before_pan = list(self.state.pan)
        canvas_width = self.state.img.shape[1] + annotate.MARGIN_L + annotate.MARGIN_R
        for _ in range(30):
            annotate.on_mouse(annotate.cv2.EVENT_MOUSEWHEEL,
                canvas_width + 100, 150, -120 << 16, self.state)
            self.panel(height)
        self.assertGreater(self.native.panel_scroll, 0)
        self.assertEqual(self.state.zoom, 1.6)
        self.assertEqual(self.state.pan, before_pan)
        self.assertEqual(self.native.record, before_record)
        self.assertEqual(self.state.kps_2d, before_points)
        self.assert_hits_inside_panel(height)
        rect, _ = next((rect, key) for rect, key in self.native.panel_hits
            if key == ord('7'))
        self.state.active = 0
        x = canvas_width + (rect[0] + rect[2]) // 2
        y = (rect[1] + rect[3]) // 2
        annotate.on_mouse(annotate.cv2.EVENT_LBUTTONDOWN, x, y, 0, self.state)
        self.assertEqual(self.state.active, 7)
        self.assertEqual(self.state.kps_2d, before_points)
        self.assertFalse(self.fixture.store.exists())

    def test_guidance_bars_do_not_place_points_on_obscured_raw_image(self):
        self.state.img = np.zeros((480, 640, 3), np.uint8)
        self.state.img_shape = self.state.img.shape
        self.state.zoom = 4.
        self.state.pan = [200, 200]
        before_points = copy.deepcopy(self.state.kps_2d)
        before_record = copy.deepcopy(self.native.record)
        for y in (20, 810):
            with self.subTest(y=y):
                # At this zoom/pan both positions lie inside the raw photo in
                # the inverse transform, beneath the painted guidance bars.
                annotate.on_mouse(annotate.cv2.EVENT_LBUTTONDOWN, 100, y, 0, self.state)
                self.assertEqual(self.state.kps_2d, before_points)
                self.assertEqual(self.native.record, before_record)
        self.assertFalse(self.fixture.store.exists())


if __name__ == '__main__':
    unittest.main()
