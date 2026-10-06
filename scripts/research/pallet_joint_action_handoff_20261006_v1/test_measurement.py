"""Non-scientific format fixtures: no real capture, training or pose accuracy."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).parent))
import measurement as m

IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def png(pixel):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"\0" + bytes([pixel, 0, 0]))) + chunk(b"IEND", b"")


def fixture(root, count=6):
    """Create explicitly fake acquisitions and model outputs inside a temp directory."""
    root = Path(root)
    evidence = root / "NON_SCIENTIFIC_EVIDENCE.txt"
    evidence.write_text("NON SCIENTIFIC FORMAT FIXTURE; every measurement, time and model hash is synthetic.\n")
    ref_art = {"path": evidence.name, "sha256": m.sha256(evidence)}
    models = [{"method": name, "weights_sha256": c * 64, "code_sha256": "c" * 64,
               "pose_contract_sha256": "d" * 64, "selection_rule_sha256": "e" * 64,
               "locked_at_ns": 1, "training_recording_ids": [], "training_pose_ids": [],
               "selection_recording_ids": [], "selection_pose_ids": []} for name, c in (("Base", "a"), ("N3", "b"))]
    data = {"schema": m.SCHEMA, "data_kind": "format_fixture", "coordinate_contract": m.COORDINATE_CONTRACT,
            "baseline_method": "Base", "models": models, "pairs": []}
    predictions = []
    covariance = [[1e-6 if i == j else 0 for j in range(6)] for i in range(6)]
    for i in range(count):
        trace_path = root / f"stability{i}.json"
        m.write_json(trace_path, {"schema": "pallet_relative_pose_stability_20261006_v1", "coordinate_contract": m.COORDINATE_CONTRACT,
                                 "clock_id": "fixture_clock", "records": [{"timestamp_ns": t, "R_physical": IDENTITY,
                                 "centroid_m": [0, 0, 2], "source_record_id": f"non_scientific_trace_{i}_{t}"} for t in (100, 101)]})
        pair = {"pair_id": f"pair{i}", "split": "independent_test", "recording_id": f"record{i}",
                "physical_pose_id": f"pose{i}", "adjacent_interval_id": f"interval{i}", "pallet_id": "pallet_fixture", "camera_id": "camera_fixture",
                "dimensions_WDH_m": [1.2, 0.8, 0.15], "dimension_survey": ref_art,
                "symmetry_order": 1, "symmetry_evidence": ref_art, "occlusion_kind": "physical_external",
                "occluder_description": "synthetic fixture placeholder, no real occluder", "used_for_selection": False,
                "first_model_exposure_ns": 10, "camera": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "image_wh": [1, 1],
                "distortion_model": "none", "distortion_coefficients": [], "intrinsic_calibration": ref_art,
                "time_sync_calibration": ref_art, "reference_extrinsic_calibration": ref_art, "clock_id": "fixture_clock"},
                "same_pose_tolerance": {"translation_m": .002, "rotation_deg": .1, "max_sync_error_ns": 1, "max_trace_gap_ns": 1, "basis": ref_art},
                "stability_record": {"path": trace_path.name, "sha256": m.sha256(trace_path)}, "uncertainty": {"tangent_order": ["tx_m", "ty_m", "tz_m", "rx_rad", "ry_rad", "rz_rad"],
                "clean_covariance": covariance, "occluded_covariance": covariance, "clean_occluded_cross_covariance": covariance,
                "budget": ref_art}}
        for condition, ci in (("clean", 0), ("occluded", 1)):
            path = root / f"pair{i}_{condition}.png"
            path.write_bytes(png(i * 2 + ci + 1))
            pair[condition] = {"frame_id": path.stem, "image": {"path": path.name, "sha256": m.sha256(path)},
                               "timestamp_ns": 100 + ci, "clock_id": "fixture_clock", "corner_sources": ["unknown"] * 8,
                               "label_review": ref_art, "reference": {"method": "optical_mocap", "independent_of_model_and_2d_labels": True,
                               "derived_from": [], "coordinate_contract": m.COORDINATE_CONTRACT, "clock_id": "fixture_clock", "timestamp_ns": 100 + ci,
                               "R_physical": IDENTITY, "centroid_m": [0, 0, 2], "source": ref_art, "source_record_id": f"non_scientific_reference_{i}_{ci}"}}
            for model in models:
                state = "success"
                if model["method"] == "Base" and ((i == 1 and ci == 0) or (i == 2 and ci == 1) or i == 3):
                    state = "pnp_failure"
                if model["method"] == "N3" and i == 4 and ci == 1:
                    state = "no_detection"
                row = {"schema": m.PREDICTION_SCHEMA, "coordinate_contract": m.COORDINATE_CONTRACT,
                       "method": model["method"], "frame_id": path.stem, "image_sha256": m.sha256(path), "state": state,
                       "reference_access": False, **{field: model[field] for field in ("weights_sha256", "code_sha256", "pose_contract_sha256", "selection_rule_sha256")}}
                if state == "success":
                    row.update(R_physical=IDENTITY, centroid_m=[.01 if ci == 0 else .02 + i * .001, 0, 2])
                predictions.append(row)
        data["pairs"].append(pair)
    return data, predictions


class MeasurementContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pallet_non_scientific_fixture_")
        self.root = Path(self.temp.name)
        self.data, self.predictions = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        return m.validate_manifest(self.data, self.root, allow_fixture=True)

    def test_fixture_is_explicit_and_schema_valid(self):
        self.assertEqual(self.validate(), [])
        self.assertTrue(m.validate_manifest(self.data, self.root))

    def test_all_failure_quadrants_and_common_intersection(self):
        result, rows, deltas = m.evaluate(self.data, self.predictions, resamples=100)
        self.assertFalse(result["scientific_evidence"])
        self.assertEqual(len(rows), 24)
        self.assertEqual(result["methods"]["Base"]["fixed_pair_failure_quadrants"],
                         {"both_success": 3, "clean_only_failure": 1, "occluded_only_failure": 1, "both_failure": 1})
        self.assertEqual(result["common_success_pair_ids"], ["pair0", "pair5"])
        self.assertEqual(len(deltas), 4)
        self.assertAlmostEqual(result["methods"]["Base"]["common_pair_metrics"]["translation_cm"]["paired_occluded_minus_clean"]["mean"], 1.25)
        self.assertEqual(result["bootstrap"]["status"], "DONE")

    def test_missing_prediction_stays_in_denominator(self):
        result, rows, _ = m.evaluate(self.data, self.predictions[1:], resamples=5)
        self.assertEqual(result["total_pairs"], 6)
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0]["state"], "missing_prediction")
        self.assertIsNone(rows[0]["translation_cm"])

    def test_split_leakage_recording_and_pose(self):
        self.data["pairs"][1]["recording_id"] = "record0"
        self.data["pairs"][1]["split"] = "train"
        self.assertTrue(any("split leakage" in e for e in self.validate()))

    def test_training_or_selection_exposure_invalidates_independence(self):
        self.data["models"][1]["selection_pose_ids"] = ["pose0"]
        self.assertTrue(any("exposure leakage" in e for e in self.validate()))

    def test_late_model_lock_invalidates_independence(self):
        self.data["models"][1]["locked_at_ns"] = 11
        self.assertTrue(any("locked after" in e for e in self.validate()))

    def test_pose_drift_and_sync_detected(self):
        self.data["pairs"][0]["occluded"]["reference"]["centroid_m"] = [.01, 0, 2]
        self.data["pairs"][0]["occluded"]["reference"]["timestamp_ns"] = 1000
        errors = self.validate()
        self.assertTrue(any("not preserved" in e for e in errors))
        self.assertTrue(any("synchronization" in e for e in errors))

    def test_nonindependent_pnp_reference_rejected(self):
        self.data["pairs"][0]["clean"]["reference"]["method"] = "pnp_from_2d_labels"
        self.assertTrue(any("provenance" in e for e in self.validate()))

    def test_stability_trace_requires_interval_coverage(self):
        pair = self.data["pairs"][0]
        trace_path = self.root / pair["stability_record"]["path"]
        trace = m.read_json(trace_path)
        trace["records"][0]["timestamp_ns"] = 101
        trace["records"][1]["timestamp_ns"] = 102
        m.write_json(trace_path, trace)
        pair["stability_record"]["sha256"] = m.sha256(trace_path)
        self.assertTrue(any("does not cover" in e for e in self.validate()))

    def test_duplicate_bytes_missing_file_and_hash_change(self):
        self.data["pairs"][0]["occluded"]["image"] = self.data["pairs"][0]["clean"]["image"]
        self.assertTrue(any("duplicate image" in e for e in self.validate()))
        (self.root / "pair1_clean.png").unlink()
        self.assertTrue(any("missing file" in e for e in self.validate()))

    def test_impossible_correlated_covariance_rejected(self):
        cov = self.data["pairs"][0]["uncertainty"]["clean_covariance"]
        self.data["pairs"][0]["uncertainty"]["clean_occluded_cross_covariance"] = [[-2 * x for x in row] for row in cov]
        self.assertTrue(any("joint clean/occluded covariance" in e for e in self.validate()))

    def test_malformed_pose_and_lock_mismatch_rejected(self):
        rows = copy.deepcopy(self.predictions)
        rows[0]["R_physical"] = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        with self.assertRaisesRegex(ValueError, "malformed successful"):
            m.evaluate(self.data, rows, resamples=5)
        rows = copy.deepcopy(self.predictions)
        rows[0]["weights_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "lock mismatch"):
            m.evaluate(self.data, rows, resamples=5)

    def test_c2_rotation_is_y_axis_and_wdh_mapping(self):
        pred = {"R_physical": m.rotations(2)[1], "centroid_m": [0, 0, 2]}
        reference = {"R_physical": IDENTITY, "centroid_m": [0, 0, 2]}
        value = m.pose_errors(pred, reference, [1.2, .15, .8], 2)
        self.assertAlmostEqual(value["rotation_deg"], 0)
        self.assertAlmostEqual(value["ADDsym_m"], 0)

    def test_empty_intersection_and_one_common_recording_no_fake_ci(self):
        data, predictions = fixture(self.root, count=1)
        result, _, _ = m.evaluate(data, predictions)
        self.assertEqual(result["bootstrap"]["status"], "NOT_ESTIMATED")
        predictions[0]["state"] = "pnp_failure"
        result, _, _ = m.evaluate(data, predictions)
        self.assertEqual(result["common_success_pairs"], 0)
        self.assertIsNone(result["methods"]["Base"]["common_pair_metrics"]["translation_cm"]["paired_occluded_minus_clean"]["median"])

    def test_import_validate_evaluate_real_commands(self):
        source = self.root / "fixture.json"
        imported = self.root / "imported.json"
        outputs = self.root / "predictions.jsonl"
        m.write_json(source, self.data)
        outputs.write_text("\n".join(json.dumps(r) for r in self.predictions) + "\n")
        base = [sys.executable, str(Path(m.__file__))]
        for command, manifest, output in (("import", source, imported), ("validate", imported, self.root / "validation.json"), ("evaluate", imported, self.root / "scored")):
            args = base + [command, "--manifest", str(manifest), "--data-root", str(self.root), "--allow-fixture", "--output", str(output)]
            if command == "evaluate":
                args += ["--predictions", str(outputs)]
            result = subprocess.run(args, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        result = m.read_json(self.root / "scored" / "SUMMARY.json")
        self.assertEqual(result["evidence_role"], "format_fixture_not_scientific")
        self.assertEqual(result["bootstrap"]["resamples"], 10000)

    def test_precision_source_rejects_unmatched_existing_receipt(self):
        scores = self.root / "YOLO_SCORES.json"
        receipt = self.root / "receipt.json"
        m.write_json(scores, {"format_fixture": True})
        m.write_json(receipt, {"inputs": [{"path": "historical/YOLO_SCORES.json", "sha256": "0" * 64}]})
        with self.assertRaisesRegex(ValueError, "does not match"):
            m.precision_source(scores, receipt, self.root / "precision_output")


if __name__ == "__main__":
    unittest.main()
