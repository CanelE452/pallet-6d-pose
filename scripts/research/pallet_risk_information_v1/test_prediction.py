"""Returned-action integrity, minimax semantics and visual chunk coverage."""
import unittest
import numpy as np
import torch

from .predict import output_for


def entry(index):
    return dict(candidate_id=str(index), candidate_aliases=[str(index)],
        source_baseline_aliases=["N0"] if index == 0 else [],
        action_role="N0" if index == 0 else "OTHER",
        actual_pose=dict(available=True, R_physical=np.eye(3).tolist(),
                         centroid=[index, 0, 3], original_provenance="retained"))


class FixedOutputs:
    def __init__(self, values, arm="R_CONT_F0", visual=False):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.arm, self.extra_dim, self.visual = arm, 288 if visual else 0, visual

    def __call__(self, f0, *_):
        return self.values[f0[:, 0].long()]


class PredictionContract(unittest.TestCase):
    def test_minimax_returns_one_whole_original_pose(self):
        entries = [entry(i) for i in range(3)]
        f0 = np.zeros((3, 188), np.float32)
        f0[:, 0] = np.arange(3)
        model = FixedOutputs([[.5, .8], [.3, 1.2], [.9, .1]])
        result = output_for({}, model, f0, entries, None, None, "cpu")
        self.assertEqual(result["actual_pose"], entries[0]["actual_pose"])
        self.assertIsNot(result["actual_pose"], entries[0]["actual_pose"])
        self.assertEqual(result["action_role"], "N0")
        self.assertFalse(result["calibrated_probability"])

    def test_missing_action_has_explicit_no_pose(self):
        result = output_for({}, FixedOutputs([]), np.empty((0, 188)), [], None, None, "cpu")
        self.assertEqual(result["prediction_status"], "NO_POSE")
        self.assertFalse(result["actual_pose"]["available"])

    def test_visual_chunks_cover_every_actual_action_once(self):
        entries = [entry(i) for i in range(19)]
        f0 = np.zeros((19, 188), np.float32)
        f0[:, 0] = np.arange(19)
        requested = []

        def provider(frame, actions, indices):
            self.assertIs(actions, entries)
            requested.extend(indices)
            self.assertLessEqual(len(indices), 8)
            return np.zeros((len(indices), 20, 4, 16, 16), np.float32), np.zeros((len(indices), 20), np.float32)

        result = output_for({}, FixedOutputs(range(19), arm="V_ORDER_RGB", visual=True),
            f0, entries, np.zeros((19, 288), np.float32), provider, "cpu")
        self.assertEqual(requested, list(range(19)))
        self.assertEqual(result["actual_pose"], entries[18]["actual_pose"])
        self.assertEqual(result["selected_candidate_id"], "18")


if __name__ == "__main__":
    unittest.main()
