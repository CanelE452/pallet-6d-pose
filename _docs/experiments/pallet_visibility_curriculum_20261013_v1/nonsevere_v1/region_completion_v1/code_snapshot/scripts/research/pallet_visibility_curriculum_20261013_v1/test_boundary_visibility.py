"""Restricted first-stage assessment cannot change pose or point measurements."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import common as C
from . import boundary_visibility as B
from .boundary_visibility import BoundaryVisibility


class BoundaryVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.weights = self.root / "fresh.pt"
        self.weights.write_bytes(b"unit mock fresh checkpoint")
        self.image = np.zeros((100, 120, 3), np.uint8)
        self.K = np.array([[60., 0., 60.], [0., 60., 50.], [0., 0., 1.]])
        axis = np.deg2rad(25.)
        R = np.array([[np.cos(axis), 0., np.sin(axis)], [0., 1., 0.], [-np.sin(axis), 0., np.cos(axis)]])
        self.initial = dict(available=True, R_cf=R.tolist(), centroid=[0., .2, 4.], cf_extents=[1., .2, .8])
        corners = C.project(C.cuboid(*self.initial["cf_extents"]), R, self.initial["centroid"], self.K)
        self.q = np.vstack([corners, [61., 52.]])
        self.model = dict(schema="unit", mean=[0.] * 26, scale=[1.] * 26, coefficients=[0.] * 26,
                          intercept=np.log(.8 / .2), visible_threshold=.2, boundary_threshold=.35)

    def evidence(self, image, points, hidden, weights, appearance_config, device):
        return dict(states=["SELF_OCCLUDED" if x else "VISIBLE_INTERIOR" for x in hidden],
            visibility_score=[0. if x else .8 for x in hidden], boundary_score=[.1] * 8,
            local_accuracy_score=[.3] * 8, diagnostics=dict(patch_radius_px=4,
                corners=[dict(reason="frozen_source_CNN_mask_and_raw_RGB_features") for _ in range(8)]))

    def assess(self, model=None):
        api = BoundaryVisibility(model or self.model, self.weights)
        with patch.object(B, "_infer_evidence", side_effect=self.evidence):
            return api.assess(self.image, self.q, self.K, self.initial)

    def test_geometric_outline_does_not_require_RGB_boundary_state(self):
        result = self.assess()
        self.assertTrue(result["selected_corner_ids"])
        for k in result["selected_corner_ids"]:
            self.assertEqual(result["states"][k], "VISIBLE")
            self.assertEqual(result["records"][k]["RGB_state"], "VISIBLE_INTERIOR")
            self.assertEqual(result["records"][k]["RGB_boundary_role"], "INTERIOR")
        for k in set(range(8)) - set(result["selected_corner_ids"]):
            self.assertEqual(result["states"][k], "NOT_ASSESSED")
            self.assertIsNone(result["p_visible"][k])

    def test_negative_RGB_is_suspected_not_confirmed_external_occlusion(self):
        model = deepcopy(self.model)
        model["intercept"] = np.log(.1 / .9)
        result = self.assess(model)
        for k in result["selected_corner_ids"]:
            self.assertEqual(result["states"][k], "EXTERNAL_OCCLUSION_SUSPECTED")
            self.assertFalse(result["records"][k]["external_occlusion_confirmed"])

    def test_actual_canvas_and_bad_raw_outside_are_separate_from_truncation(self):
        result = self.assess()
        k = result["selected_corner_ids"][0]
        self.q[k] = [130., 30.]
        changed = self.assess()
        self.assertFalse(changed["geometric_boundary_candidates"][k])
        self.assertTrue(changed["records"][k]["prediction_out_of_frame"])
        self.assertFalse(changed["records"][k]["conditional_truncation"])
        self.K[0, 2] = 400.
        outside = self.assess()
        self.assertTrue(all(outside["conditional_truncation"]))
        self.assertFalse(any(outside["geometric_boundary_candidates"]))

    def test_preserves_readonly_image_points_camera_center_and_initial(self):
        self.q[0] = [-1., -1.]
        arrays = [self.image, self.q, self.K]
        snapshots = [x.copy() for x in arrays]
        for x in arrays:
            x.flags.writeable = False
        before = deepcopy(self.initial)
        with patch("cv2.solvePnP", side_effect=AssertionError("No pose fitting")), patch("cv2.solvePnPRefineLM", side_effect=AssertionError("No pose fitting")):
            result = self.assess()
        for x, snapshot in zip(arrays, snapshots):
            np.testing.assert_array_equal(x, snapshot)
            self.assertFalse(x.flags.writeable)
        self.assertEqual(self.initial, before)
        np.testing.assert_array_equal(result["observed_points"], self.q)
        self.assertEqual(result["observed_points"][8], [61., 52.])
        self.assertTrue(result["records"][0]["prediction_missing"])
        self.assertEqual(result["provenance"]["pose_fits"], 0)
        self.assertEqual(result["provenance"]["joint_spatial_backbone_forwards"], 0)
        self.assertFalse(result["provenance"]["projected_coordinates_sampled"])

    def test_portable_config_hash_binding_and_tamper_rejection(self):
        modelpath = self.root / "visible.json"
        modelpath.write_text(json.dumps(self.model))
        configpath = self.root / "config.json"
        configpath.write_text(json.dumps(dict(visibility_model="visible.json", segmentation_weights="fresh.pt",
            model_sha256=dict(visibility_model=C.sha(modelpath), segmentation_weights=C.sha(self.weights)))))
        api = BoundaryVisibility.from_config(configpath)
        self.assertEqual(api.config_binding["sha256"], C.sha(configpath))
        self.weights.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "binding differs"):
            BoundaryVisibility.from_config(configpath)


if __name__ == "__main__":
    unittest.main()
