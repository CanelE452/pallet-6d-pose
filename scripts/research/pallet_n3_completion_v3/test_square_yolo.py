from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.research.pallet_n3_completion_v3 import common as C
from scripts.research.pallet_n3_completion_v3 import square
from scripts.research.pallet_n3_completion_v3 import square_yolo as S


PERMUTATIONS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [4, 0, 3, 7, 5, 1, 2, 6, 8],
    [5, 4, 7, 6, 1, 0, 3, 2, 8],
    [1, 5, 6, 2, 0, 4, 7, 3, 8],
]


def candidate(points, *, score=.9, box=(0., 0., 10., 10.)):
    return {
        "candidate_index": 0, "score": score, "box_xyxy": list(box),
        "keypoints_xy": np.asarray(points, dtype=float).tolist(),
        "keypoints_conf": [.8] * 9,
    }


def prediction(frame_id, points, *, hw=(100, 100), detected=True):
    return {
        "id": frame_id, "raw_hw": list(hw),
        "candidates": [candidate(points)] if detected else [],
        "selected_index": 0 if detected else None,
    }


def binding(path: Path) -> dict:
    return {"path": str(path), "sha256": C.sha256(path), "bytes": path.stat().st_size}


class SelectionAndReuseTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([
            [1, 1], [9, 1], [9, 9], [1, 9],
            [2, 2], [8, 2], [8, 8], [2, 8], [5, 5],
        ], dtype=float)
        self.record = {
            "id": "frame", "original_hw": [100, 100],
            "canonical_WDH_m": S.DIMENSIONS_WDH_M,
            "image": {"path": "unused.png", "sha256": "unused", "bytes": 0},
        }

    def _legacy(self):
        rows = [prediction("frame", self.points)]
        return {
            "complete": True, "GT_input": False, "camera_input": False,
            "dimensions_input": S.DIMENSIONS_WDH_M,
            "predictions": {name: copy.deepcopy(rows) for name in S.LEGACY_METHODS},
        }

    def test_selection_is_frozen_synthetic_only_for_six_missing_heads(self):
        contract = S.selection_contract(verify_checkpoints=False)
        self.assertEqual(set(contract["methods"]), set(S.NEW_METHODS))
        self.assertFalse(contract["square_reselection_or_tuning"])
        self.assertTrue(all(row["temperature"] == 1. for row in contract["methods"].values()))
        self.assertTrue(all(row["rule"] == {
            "lam": 1., "max_move_image_diagonal_fraction": .01}
                            for row in contract["methods"].values()))
        self.assertEqual(
            contract["methods"]["OLD_P_seed1"]["checkpoint_sha256"],
            "bfbc2c9e8d2c8dfae4937d5ad7b4a7e10dcaab7180db106ae135895a5b08b033")
        self.assertEqual(
            contract["methods"]["N3_DIM_SYM_seed3"]["checkpoint_sha256"],
            "4b04cae793c9a74fd346b84501eed1bac2e689fac257727ff1489e1235c208e3")

    def test_legacy_reuse_requires_metrics_and_dataset_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_path = root / "snapshot.json"
            prediction_path = root / "predictions.json"
            metrics_path = root / "metrics.json"
            snapshot_path.write_text(json.dumps({"records": [self.record]}) + "\n")
            payload = self._legacy()
            payload["dataset"] = binding(snapshot_path)
            prediction_path.write_text(json.dumps(payload) + "\n")
            metrics_path.write_text(json.dumps({"predictions": binding(prediction_path)}) + "\n")
            loaded, _, records = S.load_legacy(
                prediction_path, metrics_path, snapshot_path, expected_count=1)
            self.assertEqual(records[0]["id"], "frame")
            self.assertEqual(loaded["predictions"]["R0"], payload["predictions"]["R0"])

            prediction_path.write_text(prediction_path.read_text() + " ")
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                S.load_legacy(prediction_path, metrics_path, snapshot_path, expected_count=1)

    def test_merge_drops_irrelevant_n0_and_reuses_r0_n2_exactly(self):
        legacy = self._legacy()
        generated = {}
        for method in S.NEW_METHODS:
            displacement = np.zeros((9, 2), dtype=float)
            displacement[:8, 0] = .1
            row = prediction("frame", self.points + displacement)
            # Center is immutable even though the eight corners move.
            row["candidates"][0]["keypoints_xy"][8] = self.points[8].tolist()
            generated[method] = [row]
        merged = S.merge_predictions(legacy, generated, [self.record])
        self.assertEqual(tuple(merged), S.METHODS)
        self.assertNotIn("N0_BASE_REPLAY_seed1", merged)
        for method in S.REUSED_METHODS:
            self.assertEqual(merged[method], legacy["predictions"][method])

    def test_preservation_rejects_detection_or_center_changes(self):
        base = prediction("frame", self.points)
        changed = copy.deepcopy(base)
        changed["candidates"][0]["score"] = .8
        with self.assertRaisesRegex(RuntimeError, "score"):
            S.assert_detection_preserved(base, changed)
        changed = copy.deepcopy(base)
        changed["candidates"][0]["keypoints_xy"][8][0] += 1
        with self.assertRaisesRegex(RuntimeError, "Center"):
            S.assert_detection_preserved(base, changed)


class MinimalInferenceTests(unittest.TestCase):
    def test_one_extractor_forward_and_only_six_missing_heads(self):
        points = np.array([
            [1, 1], [9, 1], [9, 9], [1, 9],
            [2, 2], [8, 2], [8, 8], [2, 8], [5, 5],
        ], dtype=float)
        base = prediction("frame", points, hw=(20, 30))
        legacy = {"predictions": {"R0": [base]}}
        records = [{
            "id": "frame", "original_hw": [20, 30],
            "image": {"path": "fake.png"},
        }]

        class Extractor:
            calls = 0

            def predict(self, image):
                self.calls += 1
                return {"candidates": copy.deepcopy(base["candidates"]), "selected_index": 0}

        extractor = Extractor()
        called = []

        def predict_head(head, arm, captured, dimensions, order, temperature, rule, raw_hw, norm):
            called.append((head, arm, tuple(dimensions), order, temperature, raw_hw))
            rows = copy.deepcopy(captured["candidates"])
            rows[0]["keypoints_xy"][0][0] += .1
            return {"candidates": rows, "selected_index": 0, "head_used": True}, None

        specs = {}
        heads = {}
        for method in S.NEW_METHODS:
            arm, seed = method.rsplit("_seed", 1)
            specs[method] = {
                "arm": arm, "seed": int(seed), "temperature": 1.,
                "rule": {"lam": 1., "max_move_image_diagonal_fraction": .01},
            }
            heads[method] = method
        output = S.infer_missing_rows(
            records, legacy, extractor=extractor, heads=heads, method_specs=specs,
            predict_fn=predict_head, serial_fn=lambda value: copy.deepcopy(value),
            image_loader=lambda path: np.zeros((20, 30, 3), np.uint8),
            normalization={}, dimensions=S.DIMENSIONS_WDH_M, order=4)
        self.assertEqual(extractor.calls, 1)
        self.assertEqual(len(called), 6)
        self.assertEqual(set(output), set(S.NEW_METHODS))
        self.assertEqual({row[1] for row in called}, {"OLD_P", "N3_DIM_SYM"})
        self.assertFalse(any("N2" in row[1] or row[1] == "R0" for row in called))


class ExactEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([
            [1, 1], [9, 1], [9, 9], [1, 9],
            [2, 2], [8, 2], [8, 8], [2, 8], [5, 5],
        ], dtype=float)
        self.truth = [
            {"id": "a", "session": "same", "hw": [100, 100],
             "gt": self.gt.tolist(), "valid": [True] * 8 + [False],
             "box": [0, 0, 10, 10], "permutations": PERMUTATIONS},
            {"id": "b", "session": "same", "hw": [100, 100],
             "gt": self.gt.tolist(), "valid": [True] * 8 + [False],
             "box": [0, 0, 10, 10], "permutations": PERMUTATIONS},
        ]

    def test_c4_scoring_full_penalty_adverse_counts_and_seed_mean(self):
        # Frame a uses another approved whole-object phase and must still be exact.
        rows = [prediction("a", self.gt[PERMUTATIONS[1]]),
                prediction("b", self.gt)]
        predictions = {method: copy.deepcopy(rows) for method in S.METHODS}
        predictions["N3_DIM_SYM_seed1"][1] = prediction("b", self.gt, detected=False)
        result = S.evaluate_predictions(
            predictions, {"manual_declared": self.truth, "manual_in_frame": self.truth})
        mode = result["modes"]["manual_declared"]
        self.assertEqual(mode["manual_corner_denominator"], 16)
        self.assertEqual(mode["summary"]["R0"]["full_penalty_P90_px"], 0.)
        bad = mode["summary"]["N3_DIM_SYM_seed1"]
        self.assertEqual(bad["matched"], 1)
        self.assertEqual(bad["missing"], 1)
        self.assertGreater(bad["full_penalty_P90_px"], 100.)
        adverse = mode["comparisons"]["R0_to_N3_DIM_SYM"]["seeds"]["1"]
        self.assertEqual(adverse["paired_frames"]["harmed_frames"], 1)
        self.assertEqual(adverse["paired_frames"]["improved_frames"], 0)
        expected = np.mean([
            mode["summary"][f"N3_DIM_SYM_seed{seed}"]["full_penalty_P90_px"]
            for seed in S.SEEDS])
        actual = mode["families"]["N3_DIM_SYM"]["mean_of_seed_summaries"]["full_penalty_P90_px"]
        self.assertAlmostEqual(actual, expected)
        self.assertIsNone(result["pose_3d"]["translation"]["value"])
        self.assertEqual(result["pose_3d"]["rotation"]["display"], "x")
        self.assertFalse(result["dimension_effect_identifiable"])

    def test_method_inventory_cannot_silently_omit_old_p(self):
        predictions = {method: [prediction("a", self.gt), prediction("b", self.gt)]
                       for method in S.METHODS if not method.startswith("OLD_P")}
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            S.evaluate_predictions(
                predictions, {"manual_declared": self.truth, "manual_in_frame": self.truth})


class RealSquareContractTests(unittest.TestCase):
    def test_fixed_population_and_both_manual_denominators(self):
        declared = square.truth_rows(in_frame_only=False)
        in_frame = square.truth_rows(in_frame_only=True)
        self.assertEqual(len(declared), 119)
        self.assertEqual(sum(np.asarray(row["valid"])[:8].sum() for row in declared), 602)
        self.assertEqual(sum(np.asarray(row["valid"])[:8].sum() for row in in_frame), 600)
        self.assertTrue(all(len(row["permutations"]) == 4 for row in declared))


if __name__ == "__main__":
    unittest.main(verbosity=2)
