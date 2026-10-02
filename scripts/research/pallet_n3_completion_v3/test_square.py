from __future__ import annotations

import unittest

import numpy as np

from scripts.research.pallet_n3_completion_v3 import square


class SquareTruthContractTests(unittest.TestCase):
    def test_manual_denominators_and_c4_contract(self):
        declared = square.truth_rows(in_frame_only=False)
        in_frame = square.truth_rows(in_frame_only=True)
        self.assertEqual(len(declared), 119)
        self.assertEqual(sum(np.asarray(row["valid"])[:8].sum() for row in declared), 602)
        self.assertEqual(sum(np.asarray(row["valid"])[:8].sum() for row in in_frame), 600)
        self.assertTrue(all(len(row["permutations"]) == 4 for row in declared))
        self.assertTrue(all(row["material"] == "plastic" for row in declared))
        self.assertTrue(all(row["occlusion"] == "unclassified" for row in declared))

    def test_matching_box_uses_known_in_frame_points(self):
        for row in square.truth_rows():
            box = np.asarray(row["box"])
            self.assertEqual(box.shape, (4,))
            self.assertTrue(np.isfinite(box).all())
            self.assertTrue((box[2:] > box[:2]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
