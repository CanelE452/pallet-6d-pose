"""Exported FULL-CAL model parity and independent current-image input canary.

Only CAL prediction inputs, sealed RGB features and label-free full-fit model
predictions are read. Training labels and pose scoring artifacts are blocked.
"""
from copy import deepcopy
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import unittest

import cv2
import numpy as np

from . import common as C
from .joint_observation import (legacy_rgb26, load_joint_model,
    predict_feature_rows, predict_joint_observation, sha256)

DOC = C.ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1'
DEFAULT = DOC/'cal_joint_spatial_obs_v1'
ARTIFACT = Path(os.environ.get('JOINT_OBSERVATION_ARTIFACTS', DEFAULT))
MODEL_DIR = ARTIFACT/'deployment_export_v2'


def read(path):
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def no_training_or_scoring():
    originals = builtins.open, io.open, Path.open
    forbidden = ('ORACLE_', 'LABEL_AUDIT', 'TARGETS.json', 'SCORING_METADATA', 'OOF_OBSERVATIONS',
        'SCORED_ROWS', 'RESULTS.json', '/annotations/', 'GEOMETRY_RESOLVED_POSE_GT', 'TEST_INPUTS', 'FINAL_INPUTS')
    def wrap(function):
        def guarded(path, *args, **kwargs):
            if any(name in str(path) for name in forbidden):
                raise AssertionError('Observation inference attempted label/scoring read: ' + str(path))
            return function(path, *args, **kwargs)
        return guarded
    builtins.open, io.open, Path.open = map(wrap, originals)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = originals


@unittest.skipUnless((MODEL_DIR/'MODEL.json').exists(), 'Frozen CAL joint model fixture unavailable')
class JointObservationExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with no_training_or_scoring():
            cls.model = load_joint_model(MODEL_DIR/'MODEL.json', device=cls.device)
            cls.keys = read(ARTIFACT/'FEATURE_KEYS.json')
            with np.load(ARTIFACT/'FEATURES.npz') as archive:
                cls.rgb = archive['RGB26'].copy()
                cls.spatial = np.column_stack([archive['F4'], archive['F8']])
                cls.radius = archive['RADIUS'].copy()
            cls.expected = read(MODEL_DIR/'FULL_PREDICTION_CACHE.json')
            cls.inputs = read(ARTIFACT/'CAL_INPUTS.json') if (ARTIFACT/'CAL_INPUTS.json').exists() else read(ARTIFACT.parent/'CAL_INPUTS.json')
        cls.lookup = {(row['id'],row['corner']):i for i,row in enumerate(cls.keys)}
        cls.report = dict(model_sha256=sha256(MODEL_DIR/'MODEL.json'), parameters_sha256=sha256(MODEL_DIR/'PARAMETERS.npz'),
            device=cls.device, TEST_reads=0, FINAL_reads=0, GT_reads=0)

    def test_cached_export_matches_independent_sklearn_predictions(self):
        with no_training_or_scoring():
            probability, delta = predict_feature_rows(self.rgb,self.spatial,self.radius,self.model)
        expected_p = np.asarray([r['P_usable'] for r in self.expected])
        expected_delta = np.asarray([r['RGB_delta_px'] for r in self.expected])
        np.testing.assert_allclose(probability,expected_p,rtol=0,atol=1e-6)
        np.testing.assert_allclose(delta,expected_delta,rtol=0,atol=1e-5)
        np.testing.assert_array_equal(probability>=.45,expected_p>=.45)
        self.assertLessEqual(np.linalg.norm(delta,axis=1).max(),16+1e-8)
        self.report['export_parameter_parity'] = dict(rows=len(probability),
            probability_max_abs=float(np.abs(probability-expected_p).max()),
            delta_max_abs_px=float(np.abs(delta-expected_delta).max()),threshold_decisions_equal=True)

    def test_current_images_match_full_fit_cache_and_preserve_native_inputs(self):
        max_p, max_delta, max_rgb, admitted, frames = 0., 0., 0., 0, 0
        with no_training_or_scoring():
            for name in ('ORACLE_CAL_INPUTS.json','LABEL_AUDIT.json','OOF_OBSERVATIONS.json'):
                with self.assertRaises(AssertionError):
                    read(ARTIFACT/name)
            for frame in self.inputs['frames']:
                self.assertEqual(frame['partition'],'CAL')
                q = np.asarray(frame['points']['N3_SUBPIX'],float)
                initial = deepcopy(frame['initial_poses']['N3_SUBPIX'])
                K = np.asarray(frame['K'],float)
                xyz = C.cuboid(*initial['cf_extents'])
                H = C.self_hidden(xyz,initial['R_cf'],initial['centroid'])
                q_before, H_before, K_before, initial_before = q.copy(), H.copy(), K.copy(), deepcopy(initial)
                image_path = Path(self.inputs['source_root'])/frame['image']
                self.assertEqual(sha256(image_path),frame['image_sha256'])
                image = cv2.imread(str(image_path))
                result = predict_joint_observation(image,q,H,initial,K,self.model)
                expected_ids = [k for k in range(8) if (frame['id'],k) in self.lookup]
                self.assertEqual(np.flatnonzero(result['eligible']).tolist(),expected_ids)
                observed, RGBdelta = np.asarray(result['points']),np.asarray(result['RGB_delta_px'])
                raw_rgb = legacy_rgb26(result['raw_appearance'])
                expected_refined = []
                for k in expected_ids:
                    i = self.lookup[frame['id'],k]
                    ep, ed = self.expected[i]['P_usable'],np.asarray(self.expected[i]['RGB_delta_px'])
                    max_p = max(max_p,abs(result['P_usable'][k]-ep))
                    max_delta = max(max_delta,float(np.abs(RGBdelta[k]-ed).max()))
                    max_rgb = max(max_rgb,float(np.abs(raw_rgb[k]-self.rgb[i]).max()))
                    self.assertEqual(result['diagnostics']['patch_radius_px'],self.radius[i])
                    candidate = q[k]+ed
                    height,width = image.shape[:2]
                    if ep>=.45 and 0<=candidate[0]<width and 0<=candidate[1]<height:
                        expected_refined.append(k)
                        np.testing.assert_allclose(observed[k],candidate,rtol=0,atol=1e-5)
                    else:
                        np.testing.assert_array_equal(observed[k],q[k])
                self.assertEqual(result['observed_refined_ids'],expected_refined)
                self.assertTrue(all(not H[k] for k in result['observed_refined_ids']))
                np.testing.assert_array_equal(q,q_before)
                np.testing.assert_array_equal(H,H_before)
                np.testing.assert_array_equal(K,K_before)
                self.assertEqual(initial,initial_before)
                np.testing.assert_array_equal(observed[8:],q[8:])
                for key in ('geometry_features_used','projected_points_used','GT_input'):
                    self.assertFalse(result['diagnostics'][key])
                json.dumps(result,allow_nan=False)
                admitted += len(expected_refined)
                frames += 1
        self.assertEqual(frames,76)
        self.assertLess(max_p,1e-6)
        self.assertLess(max_delta,1e-5)
        self.assertEqual(max_rgb,0)
        self.report['current_image_parity'] = dict(frames=frames,eligible_points=len(self.keys),
            corrected_points=admitted,probability_max_abs=max_p,delta_max_abs_px=max_delta,RGB26_max_abs=max_rgb,
            native_order_H_center_initial_pose_immutable=True,annotation_read_canary='PASS')

    def test_geometry_is_not_read_and_missing_H_OOB_priority(self):
        frame = self.inputs['frames'][0]
        image = cv2.imread(str(Path(self.inputs['source_root'])/frame['image']))
        q = np.asarray(frame['points']['N3_SUBPIX'],float)
        H = np.ones(8,bool)
        before = q.copy()
        with no_training_or_scoring():
            # Objects would fail if either geometry argument were consumed.
            hidden = predict_joint_observation(image,q,H,object(),object(),self.model)
            np.testing.assert_array_equal(hidden['points'],q)
            self.assertFalse(any(hidden['eligible']))
            self.assertEqual(hidden['observed_refined_ids'],[])
            modified = q.copy()
            modified[0] = [-1,-1]
            modified[1] = [np.nan,5]
            modified[2] = [-7,5]
            modified_before = modified.copy()
            result = predict_joint_observation(image,modified,np.zeros(8,bool),object(),object(),self.model)
            self.assertFalse(any(result['eligible'][:3]))
            self.assertTrue(all(k>=3 for k in result['observed_refined_ids']))
            np.testing.assert_array_equal(modified,modified_before)
            np.testing.assert_array_equal(result['points'][8:],q[8:])
            json.dumps(result,allow_nan=False)
        np.testing.assert_array_equal(q,before)
        self.report['input_canary'] = dict(geometry_arguments_unused=True,H_OOB_missing_priority=True,
            nonfinite_output_JSON_safe=True)

    @classmethod
    def tearDownClass(cls):
        # Validation evidence contains no labels/metrics and never selects a head.
        if len(cls.report) >= 9:
            path = Path(os.environ.get('JOINT_OBSERVATION_QA_OUTPUT',MODEL_DIR/'WRAPPER_QA.json'))
            path.write_text(json.dumps(cls.report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    unittest.main()
