"""CPU contract tests using random heads, synthetic arrays and a mock detector.

No real image, pretrained detector, trained candidate or GPU is used. These
tests establish wiring/preservation only, never accuracy of an untrained head.
"""
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import inference as I
from source_data import letterbox_transform

torch.set_num_threads(1)


class FakeExtractor:
    def __init__(self, weights, device="cpu"):
        self.calls = 0
        self.closed = False
        self.mode = "normal"
        self.last_capture = None

    def predict(self, bgr, *, already_padded=False):
        self.calls += 1
        pad = 0 if already_padded else 100
        canvas = (bgr.shape[0] + 2 * pad, bgr.shape[1] + 2 * pad)
        shape = letterbox_transform(canvas).input_shape_hw
        points = np.array([[40., 50.], [160., 50.], [160., 120.], [40., 120.],
                           [65., 25.], [185., 25.], [185., 95.], [65., 95.],
                           [111.123456789, 77.987654321]], dtype=np.float64)
        if already_padded:
            points += 100
        if self.mode == "missing":
            points[:8] = np.nan
        if self.mode == "degenerate":
            points[:8] = points[0]
        candidates = []
        for index, score in enumerate((.21, .89, .52)):
            candidates.append({"candidate_index": index, "score": score,
                "box_xyxy": np.array([30., 15., 195., 130.]) + (100 if already_padded else 0),
                "keypoints_xy": points.copy() + index * .123456789,
                "keypoints_conf": np.linspace(.31, .999, 9, dtype=np.float64)})
        if self.mode == "empty":
            candidates = []
        generator = torch.Generator().manual_seed(1702)
        captured = {"p3": torch.randn(1, 64, 80, 80, generator=generator).half(),
                    "p4": torch.randn(1, 128, 40, 40, generator=generator).half(),
                    "input_shape": shape, "canvas_shape": canvas, "added_border": pad,
                    "candidates": candidates, "selected_index": 1 if candidates else None}
        self.last_capture = captured
        return captured

    def close(self):
        self.closed = True


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pallet-line-inference-contract-")
        self.root = Path(self.tmp.name)
        self.baseline = self.root / "mock_baseline.pt"
        self.baseline.write_bytes(b"mock detector placeholder; never loaded by torch")
        self.manifest = self.root / "SOURCE_MANIFEST.json"
        self.protocol = self.root / "TRAIN_PROTOCOL.json"
        self.manifest.write_text('{"synthetic_test_fixture":true}\n')
        self.protocol.write_text(json.dumps({
            "schema": "pallet_line_pose_training_protocol_v1", "complete": True,
            "steps": 6000, "test_fixture": True,
            "source_sha256": {str(I.HERE / name): I.sha(I.HERE / name)
                              for name in ("model.py", "features.py", "source_data.py")},
            "calibration": {"temperature_grid": [.5, 1., 2., 4.]},
            "selection": {"lambda_grid": [0., .0625, .25, 1., 4.],
                          "max_move_image_diagonal_fractions": [None, .01]}}))
        self.checkpoint = self.root / "random_head_fixture.pt"
        self.selection = self.root / "synthetic_selection_fixture.json"
        torch.manual_seed(127)
        self.blob = {"schema": "pallet_line_pose_checkpoint_v1", "complete": True,
                     "step": 6000, "arm": "image_joint", "seed": 1,
                     "model_config": dict(I.MODEL_CONFIG),
                     "model_state_dict": I.PalletLinePoseHead(**I.MODEL_CONFIG).state_dict(),
                     "baseline_checkpoint": str(self.baseline),
                     "baseline_checkpoint_sha256": I.sha(self.baseline),
                     "source_manifest_sha256": I.sha(self.manifest),
                     "train_protocol_sha256": I.sha(self.protocol)}
        self.write_checkpoint()
        self.rule = self.selection_data()
        self.write_selection()
        self.patches = [patch.object(I, "BASELINE_SHA", I.sha(self.baseline)),
                        patch.object(I, "FrozenYoloFeatures", FakeExtractor)]
        for item in self.patches:
            item.start()
        self.bgr = np.zeros((240, 320, 3), dtype=np.uint8)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def write_checkpoint(self):
        torch.save(self.blob, self.checkpoint)

    def selection_data(self, arm="image_joint", lam=1., cap=None):
        return {"schema": "pallet_line_pose_synthetic_selection_v1", "complete": True,
                "no_real_selection": True,
                "source_manifest": {"path": str(self.manifest), "sha256": I.sha(self.manifest)},
                "training_protocol": {"path": str(self.protocol), "sha256": I.sha(self.protocol)},
                "model_sha256": I.sha(I.HERE / "model.py"),
                "readout_sha256": I.sha(I.HERE / "readout.py"),
                "temperatures": {arm: {"1": {"temperature": 2.}}},
                "selected_rules": {arm: {"lam": lam, "max_move_image_diagonal_fraction": cap}},
                "runs": [{"arm": arm, "seed": 1, "checkpoint": str(self.checkpoint),
                          "checkpoint_sha256": I.sha(self.checkpoint)}]}

    def write_selection(self):
        self.selection.write_text(json.dumps(self.rule))

    def predictor(self):
        return I.PalletLinePoseInference(self.checkpoint, self.selection, "cpu")

    def assert_preserved(self, output, captured):
        self.assertEqual(output["selected_index"], captured["selected_index"])
        self.assertEqual(len(output["candidates"]), len(captured["candidates"]))
        for index, (actual, original) in enumerate(zip(output["candidates"], captured["candidates"])):
            self.assertEqual(actual["candidate_index"], original["candidate_index"])
            self.assertEqual(actual["score"], original["score"])
            for key in ("box_xyxy", "keypoints_conf"):
                self.assertEqual(actual[key].tobytes(), original[key].tobytes())
            if index != captured["selected_index"]:
                self.assertEqual(actual["keypoints_xy"].tobytes(), original["keypoints_xy"].tobytes())
            self.assertEqual(actual["keypoints_xy"][8].tobytes(), original["keypoints_xy"][8].tobytes())

    def test_one_forward_actual_random_head_and_preservation(self):
        with self.predictor() as predictor:
            self.assertFalse(any(p.requires_grad for p in predictor.head.parameters()))
            self.assertFalse(any(m.training for m in predictor.head.modules()))
            output = predictor.predict(self.bgr, measure_time=True, include_logits=True)
            self.assertEqual(predictor.extractor.calls, 1)
            self.assertTrue(output["head_used"])
            self.assert_preserved(output, predictor.extractor.last_capture)
            self.assertEqual(output["diagnostics"]["logits"].shape, (8, 222))
            self.assertEqual(output["diagnostics"]["line_h_supplied_image"].shape, (8, 3))
            self.assertEqual(output["diagnostics"]["peak_line_h_supplied_image"].shape, (8, 3))
            for key in ("peak_probability_conditional", "expected_sample_coverage", "peak_sample_coverage"):
                values = output["diagnostics"][key]
                self.assertEqual(values.shape, (8,))
                self.assertTrue(((values >= 0) & (values <= 1 + 1e-6)).all())
            self.assertGreater(output["inference_ms"], 0)
            self.assertTrue(np.isfinite(output["candidates"][1]["keypoints_xy"]).all())
            json.dumps(I.jsonable(output), allow_nan=False)
        self.assertTrue(predictor.extractor.closed)

    def test_lambda_zero_skips_head_and_readout_and_preserves_exact_bits(self):
        self.rule["selected_rules"]["image_joint"]["lam"] = 0.
        self.write_selection()
        with self.predictor() as predictor, patch.object(predictor.head, "forward", side_effect=AssertionError("head called")), patch.object(I, "readout", side_effect=AssertionError("readout called")):
            output = predictor.predict(self.bgr)
            self.assertFalse(output["head_used"])
            self.assertFalse(output["refinement_applied"])
            for actual, original in zip(output["candidates"], predictor.extractor.last_capture["candidates"]):
                self.assertEqual(actual["keypoints_xy"].tobytes(), original["keypoints_xy"].tobytes())
            self.assert_preserved(output, predictor.extractor.last_capture)

    def test_missing_detection_and_missing_points_fall_back(self):
        for mode, status in (("empty", "no_detection_baseline_preserved"),
                             ("missing", "no_valid_corner_baseline_preserved")):
            with self.subTest(mode=mode), self.predictor() as predictor, patch.object(predictor.head, "forward", side_effect=AssertionError("head called")):
                predictor.extractor.mode = mode
                output = predictor.predict(self.bgr)
                self.assertEqual(output["status"], status)
                self.assertFalse(output["head_used"])
                self.assert_preserved(output, predictor.extractor.last_capture)
                json.dumps(I.jsonable(output), allow_nan=False)

    def test_degenerate_lines_preserve_the_whole_baseline(self):
        with self.predictor() as predictor:
            predictor.extractor.mode = "degenerate"
            output = predictor.predict(self.bgr)
            self.assertEqual(output["status"], "no_valid_line_baseline_preserved")
            self.assertFalse(output["refinement_applied"])
            for actual, original in zip(output["candidates"], predictor.extractor.last_capture["candidates"]):
                self.assertEqual(actual["keypoints_xy"].tobytes(), original["keypoints_xy"].tobytes())

    def test_cap_uses_unpadded_raw_diagonal_and_validates_selection(self):
        self.rule["selected_rules"]["image_joint"]["max_move_image_diagonal_fraction"] = .01
        self.write_selection()
        with self.predictor() as predictor:
            padded = np.zeros((440, 520, 3), dtype=np.uint8)
            output = predictor.predict(padded, already_padded=True)
            self.assertEqual(output["raw_shape_hw"], [240, 320])
            self.assertAlmostEqual(output["diagnostics"]["cap_input_px"],
                                   4 * output["diagnostics"]["gain"])
            self.assertLessEqual(float(output["diagnostics"]["move_px"].max()), 4.0001)
            self.assert_preserved(output, predictor.extractor.last_capture)

    def test_geometry_arm_is_forwarded_and_other_arm_supported(self):
        for arm in ("geometry_joint", "image_line_only"):
            with self.subTest(arm=arm):
                self.blob["arm"] = arm
                self.write_checkpoint()
                self.rule = self.selection_data(arm)
                self.write_selection()
                with self.predictor() as predictor, patch.object(predictor.head, "forward", wraps=predictor.head.forward) as forward:
                    output = predictor.predict(self.bgr)
                    self.assertEqual(forward.call_args.kwargs["geometry_only"], arm == "geometry_joint")
                    self.assertEqual(forward.call_args.kwargs["lam"], 0.)
                    self.assert_preserved(output, predictor.extractor.last_capture)

    def test_incomplete_checkpoint_wrong_hash_and_real_selection_rejected(self):
        good_blob, good_rule = copy.deepcopy(self.blob), copy.deepcopy(self.rule)
        for change in ("step", "complete", "baseline", "selection_hash", "real_selection", "source_hash", "temperature"):
            with self.subTest(change=change):
                self.blob, self.rule = copy.deepcopy(good_blob), copy.deepcopy(good_rule)
                if change == "step":
                    self.blob["step"] = 100
                elif change == "complete":
                    self.blob["complete"] = False
                elif change == "baseline":
                    self.blob["baseline_checkpoint_sha256"] = "invalid"
                self.write_checkpoint()
                self.rule["runs"][0]["checkpoint_sha256"] = I.sha(self.checkpoint)
                if change == "selection_hash":
                    self.rule["runs"][0]["checkpoint_sha256"] = "invalid"
                elif change == "real_selection":
                    self.rule["no_real_selection"] = False
                elif change == "source_hash":
                    self.rule["source_manifest"]["sha256"] = "invalid"
                elif change == "temperature":
                    self.rule["temperatures"]["image_joint"]["1"]["temperature"] = 0
                self.write_selection()
                with self.assertRaises(ValueError):
                    self.predictor()

    def test_no_gt_argument_and_training_mode_rejected(self):
        for fn in (I.PalletLinePoseInference.predict, I.branch_inputs, I.readout):
            names = inspect.signature(fn).parameters
            self.assertFalse(any("gt" in key or "target" in key for key in names))
        with self.predictor() as predictor:
            with self.assertRaises(TypeError):
                predictor.predict(self.bgr, gt_points=np.zeros((9, 2)))
            predictor.head.train()
            with self.assertRaises(RuntimeError):
                predictor.predict(self.bgr)
            predictor.head.eval()
            predictor.close()
            with self.assertRaises(RuntimeError):
                predictor.predict(self.bgr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
