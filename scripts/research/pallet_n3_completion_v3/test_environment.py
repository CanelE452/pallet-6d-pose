"""Tests for the dual-interpreter environment receipt."""
from __future__ import annotations

import unittest

from scripts.research.pallet_n3_completion_v3 import environment as E


class EnvironmentAuditTests(unittest.TestCase):
    def test_roles_are_explicit_and_not_presented_as_one_environment(self):
        def fake(path):
            yolo = "yolo" in str(path)
            return {
                "executable": str(path), "torch": "2.1.1+cu118",
                "ultralytics": "8.4.60" if yolo else "8.0.120",
                "has_C3k2": yolo,
            }

        payload = E.build("pallet-pose", "pallet-yolo26", probe_fn=fake)
        self.assertTrue(payload["complete"])
        self.assertFalse(payload["boundary"]["single_process_environment_mixing"])
        self.assertFalse(payload["boundary"]["yolo_runtime_compared_in_new_fixed26_benchmark"])
        roles = payload["roles"]
        self.assertFalse(roles["dope_resnet_training_inference_runtime"]["has_C3k2"])
        self.assertTrue(roles["yolo26_square_and_offline_lifter"]["has_C3k2"])

    def test_wrong_capability_boundary_is_rejected(self):
        def both_new(_):
            return {"has_C3k2": True}

        with self.assertRaisesRegex(RuntimeError, "primary"):
            E.build("a", "b", probe_fn=both_new)


if __name__ == "__main__":
    unittest.main()
