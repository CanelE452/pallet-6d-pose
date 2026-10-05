"""Guard linkage semantics that could otherwise promote predictions into truth."""
import unittest

from time_physical_connection import duplicated_adjacent, full_pose_valid, reference_gate


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class ReferenceLinkageGuardTests(unittest.TestCase):
    def test_runtime_pose_without_independent_chain_is_rejected(self):
        result = reference_gate({"reference_R": IDENTITY, "reference_t_m": [0.0, 0.0, 3.0], "units": "m"})
        self.assertFalse(result["accepted"])
        self.assertIn("independent_metrology_provenance_missing", result["failures"])

    def test_clock_linkage_is_required_even_with_matching_object(self):
        record = self.complete_reference()
        record["exposure_time_mapping_verified"] = False
        self.assertEqual(reference_gate(record)["failures"], ["exposure_clock_binding_missing"])

    def test_nominal_extrinsics_do_not_pass_survey_gate(self):
        record = self.complete_reference()
        record["surveyed_transform_chain_verified"] = False
        self.assertFalse(reference_gate(record)["accepted"])

    def test_missing_height_and_roll_pitch_do_not_make_full_pose(self):
        self.assertFalse(full_pose_valid({"available": True, "x_m": 0.0, "z_m": 3.0, "yaw_deg": 0.0}))

    def test_reflection_and_behind_camera_are_invalid(self):
        reflection = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertFalse(full_pose_valid({"available": True, "R_physical": reflection, "centroid": [0, 0, 3]}))
        self.assertFalse(full_pose_valid({"available": True, "R_physical": IDENTITY, "centroid": [0, 0, -3]}))

    def test_full_valid_pose_does_not_imply_independence(self):
        self.assertTrue(full_pose_valid({"available": True, "R_physical": IDENTITY, "centroid": [0, 0, 3]}))
        self.assertFalse(reference_gate({})["accepted"])

    def test_valid_complete_reference_requires_all_linkage_records(self):
        self.assertTrue(reference_gate(self.complete_reference())["accepted"])

    def test_timestamp_duplicates_are_counted_without_removing_frames(self):
        frames = [{"timestamp": x} for x in [10, 10, 12, 12, 15]]
        self.assertEqual(duplicated_adjacent(frames, "timestamp"), 2)
        self.assertEqual(len(frames), 5)

    @staticmethod
    def complete_reference():
        return {"independent_source_verified": True, "same_target_object_verified": True, "exposure_time_mapping_verified": True, "surveyed_transform_chain_verified": True, "units": "m", "reference_R": IDENTITY, "reference_t_m": [0, 0, 3]}


if __name__ == "__main__":
    unittest.main()
