"""Test public input isolation and reproduce the sealed batch pipeline."""
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from . import common as C
from .api import VisibilityCurriculum
from .run import no_annotations


class LiveApiTests(unittest.TestCase):
    def test_rejects_invalid_pose_before_image_inference(self):
        image = np.zeros((20, 20, 3), np.uint8)
        pose = dict(available=True, R_cf=np.eye(3).tolist(), centroid=[0, 0, 3], cf_extents=[1, .1, 1])
        q = np.ones((8, 2))
        K = np.diag([100., 100., 1.])
        for changes in ({'available': False}, {'R_cf': (np.eye(3) * 2).tolist()},
                        {'centroid': [0, 0, -3]}, {'cf_extents': [1, 0, 1]}):
            with self.subTest(changes=changes), patch('scripts.research.pallet_visibility_curriculum_20261013_v1.api.infer_appearance', side_effect=AssertionError('must not run')):
                with self.assertRaises(ValueError):
                    VisibilityCurriculum().refine(image, q, K, dict(pose, **changes))

    def test_real_images_match_sealed_first_pipeline_and_preserve_inputs(self):
        records_path = C.SCRATCH / 'cal_segmentation_v2/PREDICTIONS.jsonl.gz'
        if not records_path.exists():
            self.skipTest('Frozen experiment records are unavailable')
        records = {row['id']: row for row in C.rows(records_path) if row['method'] == 'SEGMENTED_RGB_CURRICULUM'}
        frames = C.read(C.SCRATCH / 'CAL_INPUTS.json')['frames']
        selected = []
        for session in sorted({f['session'] for f in frames}):
            selected.append(next(f for f in frames if f['session'] == session))
        engine = VisibilityCurriculum.from_config(C.DOC / 'LIVE_CONFIG_V1.json')
        for frame in selected:
            image = cv2.imread(str(C.ROOT / frame['image']))
            image_before = image.copy()
            q = np.asarray(frame['points']['N3_SUBPIX'], float)
            before = q.copy()
            pose = frame['initial_poses']['N3_SUBPIX']
            pose_before = json.dumps(pose)
            with self.subTest(frame=frame['id']), no_annotations():
                result = engine.refine(image, q, frame['K'], pose)
            frozen = records[frame['id']]
            np.testing.assert_allclose(result['actual_pose']['R_cf'], frozen['actual_pose']['R_cf'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(result['actual_pose']['centroid'], frozen['actual_pose']['centroid'], rtol=0, atol=1e-12)
            self.assertEqual(result['solution']['used_ids'], frozen['solution']['used_ids'])
            self.assertEqual(result['self_hidden'], frozen['self_hidden'])
            self.assertEqual(result['appearance']['states'], frozen['appearance']['states'])
            self.assertFalse(result['provenance']['projected_points_used_for_fit'])
            self.assertTrue(result['provenance']['center_preserved'])
            self.assertTrue(all(not result['self_hidden'][i] for i in result['solution']['used_ids']))
            np.testing.assert_array_equal(image, image_before)
            np.testing.assert_array_equal(q, before)
            self.assertEqual(json.dumps(pose), pose_before)


if __name__ == '__main__':
    unittest.main()
