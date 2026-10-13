"""CAL-only live/cache parity, immutable inputs and inference access checks.

No scoring annotation is read. The two fixtures are already sealed, prediction-
only CAL artifacts. Skip the replay when these optional research fixtures are
not installed; no TEST population is a fallback.
"""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import unittest

import cv2
import numpy as np

from . import common as C
from .api import VisibilityCurriculum


_durable = C.DOC
_default = _durable if all((_durable / path).exists() for path in (
    'CAL_INPUTS.json', 'cal_segmentation_v2/PREDICTIONS.jsonl.gz',
    'cal_boundary_hybrid_v1/PREDICTIONS.jsonl.gz')) else C.SCRATCH
SCRATCH = Path(os.environ.get('PALLET_CAL_BOUNDARY_FIXTURES', str(_default)))
INPUTS = SCRATCH / 'CAL_INPUTS.json'
BASELINE = SCRATCH / 'cal_segmentation_v2' / 'PREDICTIONS.jsonl.gz'
HYBRID = SCRATCH / 'cal_boundary_hybrid_v1' / 'PREDICTIONS.jsonl.gz'


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def production_access_canary():
    import builtins
    import io
    originals = builtins.open, io.open, Path.open
    forbidden = ('SCORING_METADATA', 'ORACLE_CAL_INPUTS', 'TARGETS.json',
                 'CAL_INPUTS.json', 'PREDICTIONS.jsonl.gz', 'SCORED_ROWS.jsonl.gz',
                 'STATIC_VISIBILITY_MERGE_AUDIT', 'GEOMETRY_RESOLVED_POSE_GT',
                 '/annotations/', 'AXIS_REVIEW_MANIFEST')

    def wrapped(function):
        def guarded(path, *args, **kwargs):
            if any(name in str(path) for name in forbidden):
                raise AssertionError('Production inference read a research cache or annotation: ' + str(path))
            return function(path, *args, **kwargs)
        return guarded

    builtins.open, io.open, Path.open = map(wrapped, originals)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = originals


class LiveBoundaryParity(unittest.TestCase):
    @unittest.skipUnless(INPUTS.exists() and BASELINE.exists() and HYBRID.exists(),
                         'Optional sealed CAL-only replay fixtures are unavailable')
    def test_all_76_frames_match_both_frozen_v1_and_strict_hybrid(self):
        inputs = json.loads(INPUTS.read_text())
        frames = inputs['frames']
        self.assertEqual(len(frames), 76)
        self.assertTrue(all(f['partition'] == 'CAL' for f in frames))
        source_root = C.ROOT if all((C.ROOT / f['image']).exists() for f in frames) else Path(inputs['source_root'])
        if not all((source_root / f['image']).exists() for f in frames):
            self.skipTest('Optional historical CAL image files are unavailable')
        baseline = {(r['id'], r['method']): r for r in rows(BASELINE)}
        hybrid = {r['id']: r for r in rows(HYBRID) if r['arm'] == 'SEGMENTED_PLUS_RGB_STRICT'}
        self.assertEqual(set(hybrid), {f['id'] for f in frames})
        self.assertEqual(set(r[0] for r in baseline), set(hybrid))
        for directory in (BASELINE.parent, HYBRID.parent):
            seal = json.loads((directory / 'PREDICTION_SEAL.json').read_text())
            self.assertTrue(seal['before_scoring'])
            self.assertEqual(seal['sha256'], digest(directory / 'PREDICTIONS.jsonl.gz'))
        portable_v1 = VisibilityCurriculum.from_config(C.DOC / 'LIVE_CONFIG_V1.json')
        portable_v2 = VisibilityCurriculum.from_config(C.DOC / 'LIVE_CONFIG_V2.json')
        self.assertFalse(portable_v1.refine_boundary)
        self.assertTrue(portable_v2.refine_boundary)
        cv2.setNumThreads(1)
        corrections, corrected_frames = 0, 0
        for index, frame in enumerate(frames):
            image = cv2.imread(str(source_root / frame['image']))
            self.assertIsNotNone(image)
            q = np.asarray(frame['points']['N3_SUBPIX'], float)
            camera = np.asarray(frame['K'], float)
            initial = deepcopy(frame['initial_poses']['N3_SUBPIX'])
            preserved = (image.copy(), q.copy(), camera.copy(), deepcopy(initial))
            with production_access_canary():
                old = portable_v1.refine(image, q, camera, initial)
                new = portable_v2.refine(image, q, camera, initial)
            np.testing.assert_array_equal(image, preserved[0])
            np.testing.assert_array_equal(q, preserved[1])
            np.testing.assert_array_equal(camera, preserved[2])
            self.assertEqual(initial, preserved[3])
            expected_old = baseline[frame['id'], 'SEGMENTED_RGB_CURRICULUM']
            expected_new = hybrid[frame['id']]
            self.assertEqual(old['schema'], 'pallet_visibility_curriculum_live_v1')
            self.assertNotIn('correction', old)
            self.assertEqual(new['schema'], 'pallet_visibility_curriculum_live_v2')
            np.testing.assert_array_equal(old['observed_points'], q)
            np.testing.assert_array_equal(new['observed_points'], q)
            np.testing.assert_allclose(old['actual_pose']['R_cf'], expected_old['solution']['R'], atol=1e-7, rtol=1e-7)
            np.testing.assert_allclose(old['actual_pose']['centroid'], expected_old['solution']['t'], atol=1e-7, rtol=1e-7)
            np.testing.assert_allclose(new['corrected_measurements'][:8], expected_new['corrected_points'], atol=1e-9, rtol=0)
            np.testing.assert_allclose(new['actual_pose']['R_cf'], expected_new['solution']['R'], atol=1e-7, rtol=1e-7)
            np.testing.assert_allclose(new['actual_pose']['centroid'], expected_new['solution']['t'], atol=1e-7, rtol=1e-7)
            self.assertEqual(new['solution']['accepted'], expected_new['solution']['accepted'])
            self.assertEqual(new['solution']['used_ids'], expected_new['solution']['used_ids'])
            self.assertEqual(new['correction']['observed_refined_ids'], expected_new['refined_ids'])
            self.assertEqual(new['raw_rgb_appearance']['states'], baseline[frame['id'], 'INITIAL']['appearance']['states'])
            hidden = np.asarray(new['self_hidden'], bool)
            self.assertEqual(new['self_hidden'], expected_new['initial_self_hidden'])
            np.testing.assert_array_equal(np.asarray(new['corrected_measurements'])[:8][hidden], q[:8][hidden])
            self.assertFalse(set(new['solution']['used_ids']) & set(np.flatnonzero(hidden)))
            self.assertEqual(new['actual_pose']['cf_extents'], initial['cf_extents'])
            self.assertEqual(new['actual_pose']['selected_hypothesis'], initial.get('selected_hypothesis'))
            np.testing.assert_array_equal(new['corrected_measurements'][8], q[8])
            np.testing.assert_array_equal(new['completed_points'][8], q[8])
            self.assertFalse(new['provenance']['ground_truth_inputs'])
            self.assertFalse(new['provenance']['projected_points_used_for_fit'])
            self.assertTrue(new['provenance']['observed_points_preserved'])
            count = len(new['correction']['observed_refined_ids'])
            corrections += count
            corrected_frames += bool(count)
            if (index + 1) % 10 == 0:
                print('CAL_LIVE_BOUNDARY_PARITY', index + 1, '/76', flush=True)
        self.assertEqual(corrections, 3)
        self.assertEqual(corrected_frames, 2)


if __name__ == '__main__':
    unittest.main()
