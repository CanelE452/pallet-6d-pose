from copy import deepcopy
import json
import unittest

import numpy as np

from .boundary_evidence import summarize_boundary_evidence


def fixture():
    q = np.array([[20., 20.], [80., 20.], [90., 60.], [50., 90.], [15., 60.],
                  [50., 50.], [90., 90.], [50., 60.], [50., 55.]])
    edges = []
    for k in range(5):
        a, b = q[k], q[(k + 1) % 5]
        d = b - a
        normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        points = a + np.linspace(.15, .85, 9)[:, None] * d
        edges.append(dict(edge=k, endpoints=[k, (k + 1) % 5], supported=True,
            observed_points=points.tolist(), observation_confidence=[1.] * 9,
            inlier_mask=[True] * 9, normal=normal.tolist(), offset=-float(normal @ a),
            samples=9, transition_sample_count=9, inlier_count=9, inlier_fraction=1.,
            inlier_span_fraction=1., inlier_rms_px=0., angle_change_deg=0.))
    corners = [dict(corner=k, refined=k < 5,
        incident_shared_edges=[(k - 1) % 5, k] if k < 5 else [],
        candidate=q[k].tolist() if k < 5 else None, shift_px=0., intersection_sine=1.) for k in range(8)]
    return dict(raw_points=q.tolist(), self_hidden=[False] * 7 + [True],
        boundary_visibility=dict(expected_outline=[True] * 5 + [False, True, False],
            conditional_truncation=[False] * 6 + [True, False],
            geometric_boundary_candidates=[True] * 5 + [False] * 3),
        visibility_appearance=dict(states=["VISIBLE_BOUNDARY"] * 5 + ["VISIBLE_INTERIOR", "UNKNOWN", "SELF_OCCLUDED"],
            visibility_score=[.9] * 6 + [.1, .0], boundary_score=[.8] * 8,
            local_accuracy_score=[0.] * 8, diagnostics={}),
        correction=dict(independent_image_evidence=True, reprojected_points_used=False,
            edge_records=edges, corner_records=corners))


def external_record(q, *, depth=False):
    record = dict(schema="independent_occluding_region_v1", native_corner=0, query_xy=q,
        independent_of_target_palette=True, different_instance_from_target=True,
        region_contains_query=True, region_observed_rgb=True, instance_id="forklift-fork-17",
        source_reference="independent_sensor_record:test", source_type="instance_segmentation")
    if depth:
        record.update(source_type="depth_instance", verified_depth_order=dict(
            foreground_depth_m=1., requested_point_depth_m=2., depth_margin_uncertainty_m=.1,
            requested_point_binding="independently_verified_physical_corner",
            target_depth_origin="independently_verified_requested_point_depth", same_camera_ray=True,
            verification_reference="point_bound_depth_record:test"))
    return record


class BoundaryEvidenceTests(unittest.TestCase):
    def test_five_four_three_counts_keep_same_geometric_population(self):
        observation = fixture()
        for expected, weakened in ((5, []), (4, [0]), (3, [0, 1])):
            for k in weakened:
                observation["visibility_appearance"]["visibility_score"][k] = .1
            result = summarize_boundary_evidence(observation, image_hw=(100, 100))
            self.assertEqual(result["strong_count"], expected)
            self.assertEqual(result["geometric_boundary_count"], 5)
            self.assertFalse(result["count_is_pose_rank_proof"])

    def test_five_verified_lines_count_is_not_pose_uniqueness_or_precision(self):
        observation = fixture()
        before = json.dumps(observation, sort_keys=True)
        result = summarize_boundary_evidence(observation, image_hw=(100, 100))
        self.assertEqual(result["strong_ids"], list(range(5)))
        self.assertEqual(result["geometric_boundary_count"], 5)
        self.assertEqual(result["geometry_visible_expected_outline_count"], 5)
        self.assertFalse(result["count_is_pose_rank_proof"])
        self.assertEqual(result["records"][5]["state"], "VISIBLE_INTERIOR")
        self.assertEqual(result["records"][5]["coordinate_accuracy_score"], 0.)
        self.assertFalse(result["records"][5]["external_occlusion_confirmed"])
        self.assertEqual(json.dumps(observation, sort_keys=True), before)
        self.assertEqual(len(result["records"]), 8)
        self.assertEqual(observation["raw_points"][8], [50., 55.])

    def test_palette_negative_or_line_gap_is_not_other_object(self):
        observation = fixture()
        observation["visibility_appearance"]["visibility_score"][0] = .1
        observation["correction"]["edge_records"][0] = dict(edge=0, endpoints=[0, 1],
            supported=False, reason="insufficient_material_background_transitions", samples=9)
        result = summarize_boundary_evidence(observation, image_hw=(100, 100))
        self.assertEqual(result["records"][0]["state"], "UNKNOWN_OR_OCCLUSION")
        self.assertFalse(result["records"][0]["external_occlusion_confirmed"])
        self.assertNotIn(0, result["strong_ids"])
        self.assertFalse(result["line_records"][0]["directionality_proves_non_target_is_other_object"])

    def test_external_instance_without_order_only_suspected(self):
        observation = fixture()
        observation["visibility_appearance"]["visibility_score"][0] = .1
        result = summarize_boundary_evidence(observation, image_hw=(100, 100),
            occluding_evidence={0: external_record(observation["raw_points"][0])})
        self.assertEqual(result["records"][0]["state"], "OTHER_OBJECT_SUSPECTED")
        self.assertFalse(result["records"][0]["external_occlusion_confirmed"])

    def test_same_color_high_RGB_support_cannot_override_other_instance(self):
        observation = fixture()
        evidence = external_record(observation["raw_points"][0])
        result = summarize_boundary_evidence(observation, image_hw=(100, 100), occluding_evidence={0: evidence})
        record = result["records"][0]
        self.assertEqual(record["p_visible"], .9)
        self.assertTrue(record["two_actual_image_lines"])
        self.assertTrue(record["RGB_material_supported"])
        self.assertEqual(record["state"], "OTHER_OBJECT_SUSPECTED")
        self.assertNotIn(0, result["strong_ids"])
        self.assertEqual(result["strong_count"], 4)
        self.assertFalse(record["external_occlusion_confirmed"])
        self.assertFalse(record["occluder_evidence"]["source_facts_verified_by_this_helper"])
        self.assertEqual(record["boundary_evidence_level"], "RGB_SUPPORT_CONFLICTS_WITH_DIFFERENT_INSTANCE")

    def test_independent_depth_bound_to_requested_point_required(self):
        observation = fixture()
        evidence = external_record(observation["raw_points"][0], depth=True)
        result = summarize_boundary_evidence(observation, image_hw=(100, 100), occluding_evidence={0: evidence})
        self.assertEqual(result["records"][0]["state"], "OTHER_OBJECT_OCCLUSION")
        self.assertTrue(result["records"][0]["external_occlusion_confirmed"])
        self.assertNotIn(0, result["strong_ids"])
        # Initial projection depth cannot certify a mislocalized physical point.
        observation["visibility_appearance"]["visibility_score"][0] = .1
        evidence["verified_depth_order"]["target_depth_origin"] = "initial_pose_projection"
        result = summarize_boundary_evidence(observation, image_hw=(100, 100), occluding_evidence={0: evidence})
        self.assertEqual(result["records"][0]["state"], "OTHER_OBJECT_SUSPECTED")
        self.assertFalse(result["records"][0]["external_occlusion_confirmed"])
        evidence["query_xy"] = [30., 30.]
        result = summarize_boundary_evidence(observation, image_hw=(100, 100), occluding_evidence={0: evidence})
        self.assertEqual(result["records"][0]["state"], "UNKNOWN_OR_OCCLUSION")

    def test_conditional_geometry_and_bad_raw_outside_are_separate(self):
        observation = fixture()
        observation["boundary_visibility"]["conditional_truncation"][7] = True
        observation["raw_points"][5] = [-2., 50.]
        result = summarize_boundary_evidence(observation, image_hw=(100, 100))
        self.assertEqual(result["records"][7]["state"], "SELF_OCCLUDED")
        self.assertTrue(result["records"][7]["truncated_conditional"])
        self.assertEqual(result["records"][6]["state"], "TRUNCATED")
        self.assertEqual(result["records"][5]["state"], "UNKNOWN")
        self.assertTrue(result["records"][5]["raw_prediction_out_of_frame"])
        self.assertFalse(result["records"][5]["truncated_conditional"])

    def test_missing_native_slot_is_not_projection_completed_or_counted(self):
        observation = fixture()
        observation["raw_points"][2] = [-1., -1.]
        observation["boundary_visibility"]["geometric_boundary_candidates"][2] = False
        for k in (1, 2):
            observation["correction"]["edge_records"][k] = dict(edge=k,
                endpoints=[k, (k + 1) % 5], supported=False, reason="ineligible_endpoint")
        result = summarize_boundary_evidence(observation, image_hw=(100, 100))
        self.assertEqual(result["records"][2]["state"], "UNKNOWN")
        self.assertTrue(result["records"][2]["raw_prediction_missing"])
        self.assertIsNone(result["records"][2]["raw_query_xy"])
        self.assertNotIn(2, result["strong_ids"])
        self.assertEqual(observation["raw_points"][2], [-1., -1.])
        self.assertEqual(observation["raw_points"][8], [50., 55.])

    def test_fabricated_projection_or_inconsistent_intersection_cannot_count(self):
        observation = fixture()
        observation["correction"]["corner_records"][0]["candidate"] = [24., 24.]
        result = summarize_boundary_evidence(observation, image_hw=(100, 100))
        self.assertNotIn(0, result["strong_ids"])
        observation["correction"]["reprojected_points_used"] = True
        with self.assertRaises(ValueError):
            summarize_boundary_evidence(observation, image_hw=(100, 100))
        observation = fixture()
        observation["visibility_appearance"]["diagnostics"]["joint_observation"] = {"P_usable": True}
        with self.assertRaises(ValueError):
            summarize_boundary_evidence(observation, image_hw=(100, 100))

    def test_covariance_is_optional_validation_not_semantic_confidence(self):
        observation = fixture()
        observation["intersection_covariance"] = {"0": dict(covariance_px2=[[1., 0.], [0., 1.]],
            successful_pair_resamples=15, incident_shared_edges=[4, 0])}
        result = summarize_boundary_evidence(observation, image_hw=(100, 100))
        self.assertNotIn(0, result["strong_ids"])
        self.assertTrue(result["records"][0]["two_actual_image_lines"])
        self.assertFalse(result["records"][0]["covariance_is_calibrated_semantic_confidence"])
        self.assertEqual(result["records"][0]["state"], "VISIBLE_BOUNDARY")
        with self.assertRaises(ValueError):
            summarize_boundary_evidence(observation)


if __name__ == "__main__":
    unittest.main()
