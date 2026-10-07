"""Small gate/transfer fixtures; no neural model or optimizer is constructed.

The single geometric parity fixture makes exactly two production F calls on a
perfect synthetic cuboid.  It counts the actual OpenCV solver calls separately.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from . import common as C
from . import pilot as P
from . import teacher as T

COUNTS = dict(actual_F_calls=0, solvePnP=0, solvePnPGeneric=0,
              solvePnPRefineLM=0, neural_forward_examples=0, optimizer_updates=0)


def summary(z=5., translation=8., mean_z=5., failed=0):
    return dict(failed=failed, z_abs_cm=dict(median=z, mean=mean_z),
                translation_cm=dict(median=translation))


class PilotContracts(unittest.TestCase):
    def decision(self, corrected, depth=16, recordings=('A', 'B'), accepted=2):
        return P.gate(summary(), corrected, depth, set(recordings), accepted)

    def test_both_medians_and_mean_improve_opens_pilot(self):
        self.assertEqual(self.decision(summary(4., 7., 4.))['status'], 'PILOT_GO')

    def test_both_worse_is_not_helpful(self):
        self.assertEqual(self.decision(summary(6., 9., 6.))['status'], 'TEACHER_NOT_HELPFUL')

    def test_mixed_pose_directions_remain_unresolved(self):
        for z, t in [(4., 9.), (6., 7.)]:
            self.assertEqual(self.decision(summary(z, t, 4.))['status'], 'TEACHER_UNRESOLVED')

    def test_equal_median_does_not_satisfy_strict_gate(self):
        self.assertEqual(self.decision(summary(5., 7., 4.))['status'], 'TEACHER_UNRESOLVED')

    def test_worsening_mean_blocks_improved_medians(self):
        self.assertEqual(self.decision(summary(4., 7., 6.))['status'], 'TEACHER_UNRESOLVED')

    def test_insufficient_depth_reference_blocks(self):
        x = self.decision(summary(4., 7., 4.), depth=15)
        self.assertEqual(x['reason'], 'fewer_than_16_depth_reference_frames')

    def test_single_recording_blocks_even_many_corrections(self):
        x = self.decision(summary(4., 7., 4.), recordings=('A',), accepted=32)
        self.assertEqual(x['reason'], 'extremely_few_corrections_or_one_recording')

    def test_one_correction_blocks_even_two_recording_names(self):
        x = self.decision(summary(4., 7., 4.), accepted=1)
        self.assertEqual(x['reason'], 'extremely_few_corrections_or_one_recording')

    def test_new_numeric_failure_blocks(self):
        x = self.decision(summary(4., 7., 4., failed=1))
        self.assertEqual(x['reason'], 'new_numeric_pose_failures')

    def test_fallbacks_stay_in_full128_gate(self):
        def row(z, t):
            return dict(available=True, z_abs_cm=z, translation_cm=t,
                        rotation_deg=1., ADDsym_m=.01)
        # Four accepted proposals improve; 124 abstentions keep the raw output.
        raw = P.summarize([row(10., 10.) for _ in range(128)])
        corrected = P.summarize([row(1., 1.) for _ in range(4)] +
                                [row(10., 10.) for _ in range(124)])
        self.assertEqual(corrected['frames'], 128)
        self.assertEqual(corrected['z_abs_cm']['n'], 128)
        self.assertEqual(corrected['z_abs_cm']['median'], 10.)
        self.assertEqual(P.gate(raw, corrected, 128, {'A', 'B'}, 4)['status'],
                         'TEACHER_UNRESOLVED')

    def test_transfer_preserves_center_unsupported_and_raw_residual(self):
        K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        R = cv2.Rodrigues(np.array([.15, -.3, .05]))[0]
        t = np.array([.1, .06, 3.]); td = t * (.9)
        dims = np.array([1.1, .11, 1.3])
        p0 = T.project(K, R, t, dims); pd = T.project(K, R, td, dims)
        residual = np.arange(16, dtype=float).reshape(8, 2) * .1
        q = np.vstack([p0 + residual, [[301.25, 239.75]]])
        q[2] = [-1., -1.]; q[4] = [np.nan, np.nan]
        support = np.array([True, False, True, True, True, True, True, True, True])
        corrected = T.transfer(q, support, K, R, t, td, dims)
        valid = np.array([True, False, False, True, False, True, True, True])
        np.testing.assert_allclose(corrected[:8][valid] - pd[valid], residual[valid],
                                   rtol=0, atol=1e-12)
        np.testing.assert_array_equal(corrected[np.r_[~valid, True]],
                                      q[np.r_[~valid, True]])
        np.testing.assert_array_equal(T.transfer(q, support, K, R, t, t, dims), q)

    def test_actual_F_recovers_known_center_ray_after_transfer(self):
        # Imports the production solver only; no checkpoint/forward/trainer call.
        from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
        K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        R = cv2.Rodrigues(np.array([.25, -.35, .1]))[0]
        t0 = np.array([.1, -.07, 3.]); td = t0 * (2.8 / 3.)
        dims = np.array([1.1, .11, 1.3])
        q0 = np.vstack([T.project(K, R, t0, dims), [[320., 240.]]])
        qd = T.transfer(q0, np.ones(9, bool), K, R, t0, td, dims)
        np.testing.assert_allclose(qd[:8], T.project(K, R, td, dims), rtol=0, atol=1e-12)
        originals = {key: getattr(cv2, key) for key in
                     ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM')}
        def counted(name):
            def call(*args, **kwargs):
                COUNTS[name] += 1
                return originals[name](*args, **kwargs)
            return call
        with patch.object(cv2, 'solvePnP', counted('solvePnP')), \
             patch.object(cv2, 'solvePnPGeneric', counted('solvePnPGeneric')), \
             patch.object(cv2, 'solvePnPRefineLM', counted('solvePnPRefineLM')):
            poses = []
            for q in (q0, qd):
                COUNTS['actual_F_calls'] += 1
                poses.append(D.Pose.infer(q, K, dims, False))
        for pose, expected in zip(poses, (t0, td)):
            self.assertTrue(pose['available'])
            np.testing.assert_allclose(pose['centroid'], expected, rtol=0, atol=1e-7)
            np.testing.assert_allclose(pose['R_cf'], R, rtol=0, atol=1e-7)
            np.testing.assert_array_equal(pose['cf_extents'], dims)
        self.assertEqual(poses[0]['selected_hypothesis'], poses[1]['selected_hypothesis'])
        self.assertEqual(COUNTS['actual_F_calls'], 2)


def main():
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PilotContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        test_names=unittest.defaultTestLoader.getTestCaseNames(PilotContracts),
        seconds=time.monotonic()-start, execution=COUNTS,
        scope='CPU toy gate/geometry tests; production F on two synthetic perfect-cuboid point sets only; no dataset inference or bootstrap of real rows',
        sources=[C.bind(Path(__file__)), C.bind(Path(P.__file__)), C.bind(Path(T.__file__))])
    C.save(C.DOC/'PILOT_TEST_RESULTS.json', receipt, freeze=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
