"""Actual RGB soft-support fields are opt-in; old model predictions are exact."""
from copy import deepcopy
import unittest

import cv2
import numpy as np

from . import calibration as C
from .appearance import infer_appearance
from .segmented_appearance import infer_segmented_appearance
from .test_appearance import scene


class CalibrationFeatureTests(unittest.TestCase):
    def appearances(self):
        image, points, hidden = scene()
        raw = infer_appearance(image, points, hidden)
        probability = np.zeros(image.shape[:2], np.float32)
        cv2.fillConvexPoly(probability, cv2.convexHull(points.astype(np.int32)), 1.)
        segmented = infer_segmented_appearance(image, points, hidden, probability)
        return raw, segmented

    def test_real_raw_and_segmented_evidence_reads_actual_soft_fields(self):
        raw, segmented = self.appearances()
        self.assertEqual(C.SCALARS[4:6], ('soft_inward_color_support', 'soft_nearby_color_support'))
        for appearance in (raw, segmented):
            old = C.features(appearance)
            new = C.features(appearance, 'soft_support_v2')
            expected = np.asarray([[r['soft_inward_material_support'], r['soft_nearby_inward_material_support']]
                                   for r in appearance['diagnostics']['corners']])
            np.testing.assert_array_equal(old[:, 4:6], 0.)
            np.testing.assert_array_equal(new[:, 4:6], expected)
            self.assertGreater(float(new[:, 4:6].mean()), .5)
            np.testing.assert_array_equal(new[:, :4], old[:, :4])
            np.testing.assert_array_equal(new[:, 6:], old[:, 6:])
        np.testing.assert_array_equal(C.features(raw, 'soft_support_v2')[:, 4:6],
                                      C.features(segmented, 'soft_support_v2')[:, 4:6])

    def test_old_default_model_is_numerically_identical_to_explicit_legacy(self):
        appearance, _ = self.appearances()
        rng = np.random.default_rng(26013)
        model = dict(schema='legacy-test', mean=rng.normal(size=26).tolist(),
                     scale=rng.uniform(.5, 2., 26).tolist(), coefficients=rng.normal(size=26).tolist(),
                     intercept=.25, visible_threshold=.5, boundary_threshold=.35)
        original = deepcopy(appearance)
        old = C.predict(appearance, model)
        explicit = C.predict(appearance, dict(model, feature_version='legacy_v1'))
        self.assertEqual(old, explicit)
        x = C.features(appearance)
        logits=((x-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['coefficients'])+model['intercept']
        expected=1/(1+np.exp(-np.clip(logits,-60,60)))
        np.testing.assert_allclose(old['diagnostics']['calibration']['visibility_probabilities'], expected, atol=1e-12, rtol=0.)
        self.assertEqual(appearance, original)

    def test_explicit_v2_model_uses_soft_support_and_preserves_geometry_states(self):
        appearance, _ = self.appearances()
        appearance['states'][6:] = ['SELF_OCCLUDED', 'OUT_OF_FRAME']
        coeff=np.zeros(26);coeff[4:6]=[2.,3.]
        model=dict(schema='v2-test',feature_version='soft_support_v2',mean=[0.]*26,scale=[1.]*26,
                   coefficients=coeff.tolist(),intercept=0.,visible_threshold=.6,boundary_threshold=.35)
        actual=C.predict(appearance, model)
        x=C.features(appearance, 'soft_support_v2')
        expected=1/(1+np.exp(-(2*x[:,4]+3*x[:,5])))
        np.testing.assert_allclose(actual['diagnostics']['calibration']['visibility_probabilities'], expected, atol=1e-12, rtol=0.)
        self.assertGreater(actual['visibility_score'][4], .9)
        self.assertEqual(actual['states'][6:], appearance['states'][6:])
        self.assertEqual(actual['visibility_score'][6:], appearance['visibility_score'][6:])
        with self.assertRaises(ValueError):
            C.features(appearance, 'unregistered-feature-version')
        with self.assertRaises(ValueError):
            C.predict(appearance, dict(model, feature_version='unregistered-feature-version'))


if __name__=='__main__':
    unittest.main()
