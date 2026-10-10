"""Synthetic-only tests of visibility and the actual frozen F call boundary."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from .adapter import correspondence_mask, MaskContractError, SIGNS
from .visibility import visibility


def frozen_pose():
    root = Path(__file__).resolve().parents[3]
    directory = root / 'scripts/research/pallet_dim_conditioned_p_v1'
    before = sys.path[:]
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location('vis_unit_frozen_pose', directory / 'pose.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = before


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No metadata(), checkpoint, image, annotation or scorer is loaded.
        cls.pose = frozen_pose()

    def setUp(self):
        self.xyz = np.asarray([1.1, .11, 1.3])
        self.model = SIGNS * (self.xyz / 2)
        self.K = np.asarray([[700., 0., 320.], [0., 710., 240.], [0., 0., 1.]])
        self.q = cv2.projectPoints(np.vstack([self.model, np.zeros((1, 3))]),
            np.asarray([-.25, .4, .1]), np.asarray([.03, -.02, 3.]), self.K, None)[0].reshape(9, 2)

    def test_known_signed_faces_and_hidden_corner(self):
        q = np.asarray([[0., 0.], [2., 0.], [2., 2.], [0., 2.],
                        [1., -1.], [3., -1.], [3., 1.], [1., 1.], [1.5, .5]])
        result = visibility(q, np.ones(8, bool))
        self.assertEqual(result['facing'], dict(front=True, back=False, top=True,
                                              bottom=False, left=False, right=True))
        self.assertEqual(result['hidden_mask'], [False] * 7 + [True])
        self.assertEqual(result['hidden_count'], 1)
        self.assertFalse(result['fallback'])

    def test_unknown_is_conservative_and_input_immutable(self):
        q = np.asarray([[0., 0.], [2., 0.], [2., 2.], [0., 2.],
                        [1., -1.], [3., -1.], [3., 1.], [1., 1.], [1.5, .5]])
        support = np.ones(8, bool); support[4] = False
        saved = q.copy(), support.copy()
        result = visibility(q, support)
        self.assertIsNone(result['facing']['back'])
        self.assertTrue(result['visible_mask'][7])
        self.assertTrue(result['effective_mask'][7])
        self.assertFalse(result['effective_mask'][4])
        np.testing.assert_array_equal(q, saved[0])
        np.testing.assert_array_equal(support, saved[1])
        q[4] = np.nan
        self.assertTrue(visibility(q, np.ones(8, bool))['visible_mask'][7])

    def test_less_than_four_falls_back_to_original_support(self):
        # Zero signed areas face nowhere; no arbitrary epsilon/visibility tie.
        result = visibility(np.zeros((9, 2)), np.ones(8, bool))
        self.assertEqual(result['visible_supported_count'], 0)
        self.assertTrue(result['fallback'])
        self.assertEqual(result['effective_mask'], [True] * 8)
        support = np.asarray([True] * 3 + [False] * 5)
        result = visibility(self.q, support)
        self.assertTrue(result['fallback'])
        self.assertEqual(result['effective_mask'], support.tolist())

    def test_all_is_exact_noop_through_actual_frozen_F(self):
        expected = self.pose.infer(self.q, self.K, self.xyz, False)
        self.assertTrue(expected['available'])
        with correspondence_mask(self.q, self.K, np.ones(8, bool)) as audit:
            actual = self.pose.infer(self.q, self.K, self.xyz, False)
        self.assertEqual(actual, expected)
        self.assertEqual(audit['solvePnP'], 3)
        self.assertEqual(audit['solvePnPRefineLM'], 3)
        self.assertEqual(audit['filtered_solvePnP'], 0)
        self.assertEqual(audit['filtered_solvePnPRefineLM'], 0)
        self.assertTrue(audit['original_points_preserved'])

    def test_actual_F_vis_spy_keeps_correct_canonical_args(self):
        mask = np.asarray([True, True, True, True, True, True, True, False])
        indices = np.flatnonzero(mask)
        calls = []
        original_solve, original_lm = cv2.solvePnP, cv2.solvePnPRefineLM
        projection = cv2.projectPoints
        def spy(function, name):
            def invoked(model, image, K, *args, **kwargs):
                np.testing.assert_array_equal(image, self.q[indices])
                np.testing.assert_array_equal(np.sign(model), SIGNS[indices])
                np.testing.assert_array_equal(K, self.K)
                self.assertEqual(len(model), 7)
                if name == 'solvePnP':
                    self.assertEqual(kwargs['flags'], cv2.SOLVEPNP_SQPNP)
                calls.append(name)
                return function(model, image, K, *args, **kwargs)
            return invoked
        with patch.object(cv2, 'solvePnP', spy(original_solve, 'solvePnP')), \
             patch.object(cv2, 'solvePnPRefineLM', spy(original_lm, 'solvePnPRefineLM')):
            with correspondence_mask(self.q, self.K, mask) as audit:
                actual = self.pose.infer(self.q, self.K, self.xyz, False)
                self.assertIs(cv2.projectPoints, projection)
        self.assertTrue(actual['available'])
        self.assertEqual(calls.count('solvePnP'), 3)
        self.assertEqual(calls.count('solvePnPRefineLM'), 3)
        self.assertEqual(audit['filtered_solvePnP'], 3)
        self.assertEqual(audit['filtered_solvePnPRefineLM'], 3)
        self.assertTrue(audit['projectPoints_unpatched'])

    def test_invalid_order_fails_closed_and_restores_functions(self):
        original_solve, original_lm = cv2.solvePnP, cv2.solvePnPRefineLM
        with self.assertRaises(MaskContractError):
            with correspondence_mask(self.q, self.K, np.ones(8, bool)):
                cv2.solvePnP(self.model[::-1], self.q[:8], self.K, None,
                             flags=cv2.SOLVEPNP_SQPNP)
        self.assertIs(cv2.solvePnP, original_solve)
        self.assertIs(cv2.solvePnPRefineLM, original_lm)

    def test_other_frame_camera_and_missing_fallback_fail_closed(self):
        with correspondence_mask(self.q, self.K, np.ones(8, bool)):
            with self.assertRaises(MaskContractError):
                cv2.solvePnP(self.model, self.q[:8] + 1, self.K, None)
            with self.assertRaises(MaskContractError):
                cv2.solvePnP(self.model, self.q[:8], self.K * 2, None)
        with self.assertRaises(MaskContractError):
            with correspondence_mask(self.q, self.K, [True] * 3 + [False] * 5):
                pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
