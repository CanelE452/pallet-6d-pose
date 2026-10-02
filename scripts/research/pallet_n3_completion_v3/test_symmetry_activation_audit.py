import unittest

import numpy as np

from .symmetry_activation_audit import select_branches, summarize


class SymmetryActivationAuditTests(unittest.TestCase):
    def test_exact_selector_preserves_identity_ties_and_finds_swap(self):
        identity = np.arange(9, dtype=np.int64)
        swap = np.array([1, 0, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        permutations = np.tile(identity, (2, 4, 1))
        permutations[:, 1] = swap
        group_valid = np.zeros((2, 4), dtype=bool)
        group_valid[:, :2] = True
        points = np.zeros((2, 9, 2), dtype=np.float32)
        points[:, 0, 0] = 1
        points[:, 1, 0] = 2
        gt_points = points.copy()
        # Row zero is an exact identity tie. Row one is closer after swapping.
        gt_points[1, 0, 0] = 2
        gt_points[1, 1, 0] = 1
        arrays = {
            "points": points,
            "point_valid": np.ones((2, 9), dtype=bool),
            "boxes": np.tile(np.array([0, 0, 10, 10], np.float32), (2, 1)),
            "gt_points": gt_points,
            "gt_valid": np.ones((2, 9), dtype=bool),
        }
        branches = select_branches(
            arrays, permutations, group_valid, np.array([0, 1]))
        np.testing.assert_array_equal(branches, [0, 1])

    def test_summary_reports_non_identity_frequency(self):
        result = summarize(np.array([0, 0, 1, 3]), branch_width=4)
        self.assertEqual(result["branch_counts"],
                         {"0": 2, "1": 1, "2": 0, "3": 1})
        self.assertEqual(result["non_identity_count"], 2)
        self.assertEqual(result["non_identity_fraction"], .5)


if __name__ == "__main__":
    unittest.main()
