"""Temporary synthetic visible-button actions; no production GUI or annotations."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_native_review as fixtures
from select_visible_corner import select_visible_corner
from serve import ValidationError


class SelectVisibleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.native, self.state = self.fixture.native, self.fixture.state
        self.native.visible_pending = None

    def test_history_restores_actual_click_without_save_or_profile(self):
        self.fixture.click(4, 3)
        self.state.active = 0
        self.native.mark(self.state, 0, 'self_occlusion')
        with (patch.object(self.native, 'save', side_effect=AssertionError('No submission')),
                patch('open_existing_annotation.choose_profile', side_effect=AssertionError('No profile'))):
            self.assertTrue(select_visible_corner(self.native, self.state))
        corner = self.native.record['corners'][0]
        self.assertEqual([corner['x'], corner['y']], [4., 3.])
        self.assertEqual(corner['visibility'], 'direct_visible')
        self.assertFalse(corner['self_occlusion'])
        self.assertFalse(self.state.extrap_mask[0])
        self.assertEqual(self.state.keypoint_annotations[0]['source'], 'manual_click')
        self.assertFalse(self.fixture.fixture.store.exists())

    def test_generated_point_never_becomes_direct_until_actual_raw_click(self):
        self.native.mark(self.state, 0, 'self_occlusion')
        self.native.history.clear()
        self.state.active = 0
        self.state.kps_2d[0] = [7., 5.]
        self.state.extrap_mask[0] = True
        self.native.editor._set_keypoint_state(self.state, 0, [7., 5.],
            source='pnp_projected', visibility=1, reason='unknown')
        before = copy.deepcopy(self.native.record['corners'][0])
        with patch('open_existing_annotation.choose_profile', side_effect=AssertionError('No profile')):
            self.assertFalse(select_visible_corner(self.native, self.state))
            self.assertEqual(self.native.visible_pending, 0)
            self.assertEqual(self.native.record['corners'][0], before)
            self.fixture.click(6, 4)
        corner = self.native.record['corners'][0]
        self.assertEqual(corner['visibility'], 'direct_visible')
        self.assertEqual([corner['x'], corner['y']], [6., 4.])
        self.assertFalse(self.state.extrap_mask[0])
        self.assertFalse(self.fixture.fixture.store.exists())

    def proof(self, *, valid=True):
        path = self.fixture.path.with_name(self.fixture.path.stem + '.PNP_ASSISTED.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        corners = copy.deepcopy(self.native.record['corners'])
        corners[0].update(visibility='direct_visible', definition_confirmed=True, x=4., y=3.)
        doc = dict(schema='lifter_native_pnp_assistance_v1', bindings=self.native.ctx.bindings,
            frame_id='test:0', review_pass='primary', source_kind='human_assisted_annotation',
            evaluation_use=False,
            reference_scope='actual_manual_clicks_only',
            image_sha256=self.native.ctx.frames['test:0']['image_sha256'] if valid else 'wrong',
            manual_reference_corners=corners, editor_kps_2d=[[9., 9.]] * 9,
            annotation={'projected_cuboid': [[9., 9.]] * 8})
        path.write_text(json.dumps(doc))
        return path

    def test_bound_proof_uses_manual_reference_instead_of_generated_coordinates(self):
        path = self.proof()
        raw = path.read_bytes()
        self.assertTrue(select_visible_corner(self.native, self.state))
        self.assertEqual(self.state.kps_2d[0], [4., 3.])
        self.assertEqual(path.read_bytes(), raw)
        self.assertFalse(self.fixture.fixture.store.exists())

    def test_invalid_binding_and_invalid_manual_coordinates_are_rejected(self):
        path = self.proof(valid=False)
        with self.assertRaises(ValidationError):
            select_visible_corner(self.native, self.state)
        path = self.proof()
        doc = json.loads(path.read_text()); doc['manual_reference_corners'][0]['x'] = float('nan')
        path.write_text(json.dumps(doc))
        with self.assertRaises(ValidationError):
            select_visible_corner(self.native, self.state)
        self.assertFalse(self.fixture.fixture.store.exists())


if __name__ == '__main__':
    unittest.main()
