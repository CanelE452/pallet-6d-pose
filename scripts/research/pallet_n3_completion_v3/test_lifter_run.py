from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.research.pallet_n3_completion_v3 import lifter
from scripts.research.pallet_n3_completion_v3 import lifter_run as run


def fixture_meta():
    return {
        "pallet_size_m": {"width": 1.1, "length": 1.1, "height": .15},
        "intrinsics": {
            "fx": 605.906494140625,
            "fy": 605.9697875976562,
            "ppx": 317.59619140625,
            "ppy": 256.29229736328125,
            "width": 640,
            "height": 480,
            "model": "distortion.inverse_brown_conrady",
            "coeffs": [0., 0., 0., 0., 0.],
        },
    }


class CameraAndSamplingContractTests(unittest.TestCase):
    def test_camera_uses_recorded_ppx_ppy_and_exact_square_dimensions(self):
        camera = run.camera_contract(fixture_meta())
        self.assertEqual(camera["dimensions_wdh_m"], [1.1, 1.1, .15])
        self.assertEqual(camera["pnp_xyz_m"], [1.1, .15, 1.1])
        self.assertEqual(camera["K"], [
            [605.906494140625, 0., 317.59619140625],
            [0., 605.9697875976562, 256.29229736328125],
            [0., 0., 1.],
        ])
        self.assertIn("ppx/ppy", camera["intrinsics_source"])

    def test_camera_rejects_dimension_or_distortion_substitution(self):
        for mutate in (
                lambda value: value["pallet_size_m"].__setitem__("length", 1.3),
                lambda value: value["intrinsics"].__setitem__("coeffs", [0, 0, .1, 0, 0]),
                lambda value: value["intrinsics"].__setitem__("width", 1280)):
            meta = fixture_meta()
            mutate(meta)
            with self.assertRaises(ValueError):
                run.camera_contract(meta)

    def test_time_grid_is_first_eligible_anchored_and_not_frame_stride(self):
        timestamps = [100.0, 190.0, 1099.0, 1101.0, 2080.0, 2110.0]
        rows = [{
            lifter.SENSOR_TIME_FIELD: timestamp,
            "frame_i": index + 36,
            "camera_frame_number": index + 500,
            "camera_timestamp_domain": "hardware_clock",
        } for index, timestamp in enumerate(timestamps)]
        selected = run.fixed_sample_rows(rows, session_id="fixture")
        self.assertEqual([row["video_index"] for row in selected], [0, 3, 5])
        self.assertEqual(
            [row[lifter.SENSOR_TIME_FIELD] for row in selected],
            [100.0, 1101.0, 2110.0])

    def test_real_fixed_plan_has_all_four_full_span_counts(self):
        plan = run.build_plan(verify=False)
        self.assertEqual(plan["total_raw_video_frames"], 8910)
        self.assertEqual(plan["total_sampled_frames"], 846)
        self.assertEqual(
            {key: value["sampled_frames"] for key, value in plan["sessions"].items()},
            run.EXPECTED_SAMPLE_COUNTS)
        self.assertTrue(plan["selection_locked_before_inference"])
        self.assertFalse(plan["performance_or_accuracy_values_read_for_selection"])
        for session in plan["sessions"].values():
            self.assertEqual(session["frames"][0]["video_index"], 0)
            self.assertEqual(session["frames"][0]["frame_i"], 36)
        run.validate_plan(plan, verify_files=False)


class HistoricalBindingTests(unittest.TestCase):
    def test_all_strict_historical_bindings_resolve_by_hash_and_size(self):
        paths = run.verify_legacy_bindings()
        self.assertEqual(set(paths), {row[0] for row in run.LEGACY_BINDINGS})
        for name, relative, digest, size in run.LEGACY_BINDINGS:
            with self.subTest(name=name):
                path = paths[name]
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, size)
                self.assertEqual(run.C.sha256(path), digest)
                self.assertEqual(path.name, relative.name)

    def test_selection_is_fixed_synthetic_n3_seed1(self):
        selected = run.fixed_selection_contract()
        self.assertEqual((selected["arm"], selected["seed"]), ("N3_DIM_SYM", 1))
        self.assertEqual(selected["temperature"], 1.0)
        self.assertEqual(selected["rule"], {
            "lam": 1.0, "max_move_image_diagonal_fraction": .01})
        self.assertIn("synthetic", selected["selection_source"])

    def test_old_functions_load_from_the_strictly_bound_source_without_cuda(self):
        modules = run.load_legacy_modules()
        bound = run.verify_legacy_bindings()
        self.assertEqual(Path(modules["inference"].__file__).resolve(),
                         bound["inference"].resolve())
        self.assertEqual(Path(modules["pose"].__file__).resolve(),
                         bound["pose"].resolve())
        for name in ("load_head", "predict_captured", "registry_input"):
            self.assertTrue(callable(getattr(modules["inference"], name)))
        self.assertTrue(callable(modules["pose"].infer))

    def test_n3_checkpoint_strict_loads_on_cpu_before_cuda_execution(self):
        import torch

        modules = run.load_legacy_modules()
        path = run.verify_legacy_bindings()["n3_checkpoint"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.assertIs(checkpoint["complete"], True)
        self.assertEqual(checkpoint["step"], 6000)
        self.assertEqual(
            checkpoint["baseline_checkpoint_sha256"],
            "970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7")
        head = modules["refiner"].model(run.ARM, checkpoint["config"])
        incompatible = head.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])


class SharedForwardFixtureTests(unittest.TestCase):
    def test_one_base_forward_is_shared_and_pnp_receives_same_recorded_contract(self):
        points = np.stack([
            np.linspace(100., 180., 9), np.linspace(120., 200., 9)], axis=-1)
        candidate = {
            "candidate_index": 0,
            "score": .9,
            "box_xyxy": np.asarray([90., 110., 190., 210.]),
            "keypoints_xy": points,
            "keypoints_conf": np.ones(9),
        }

        class Extractor:
            calls = 0

            def predict(self, image):
                self.calls += 1
                return {
                    "candidates": [copy.deepcopy(candidate)],
                    "selected_index": 0,
                    "p3": object(), "p4": object(),
                    "canvas_shape": (680, 840), "input_shape": (512, 640),
                    "added_border": 100,
                }

            def close(self):
                pass

        class Inference:
            captured_identity = None

            @staticmethod
            def serial(candidates):
                return [{key: value.tolist() if hasattr(value, "tolist") else value
                         for key, value in row.items()} for row in candidates]

            @classmethod
            def predict_captured(cls, head, arm, captured, dimensions, order,
                                 temperature, rule, raw_hw, normalization):
                cls.captured_identity = id(captured)
                refined = copy.deepcopy(captured["candidates"])
                refined[0]["keypoints_xy"][0] += np.asarray([1., -2.])
                return ({"candidates": cls.serial(refined), "selected_index": 0,
                         "head_used": True}, {"fixture": True})

        class Pose:
            calls = []

            @classmethod
            def infer(cls, selected, K, xyz, source=False):
                cls.calls.append((np.asarray(selected), np.asarray(K),
                                  np.asarray(xyz), source))
                angle = math.radians(179.)
                rotation = np.asarray([
                    [math.cos(angle), 0., math.sin(angle)],
                    [0., 1., 0.],
                    [-math.sin(angle), 0., math.cos(angle)],
                ])
                return {
                    "available": True,
                    "R_cf": rotation.tolist(),
                    "R_physical": rotation.tolist(),
                    "centroid": [.2, .1, 2.0],
                    "cf_extents": [1.1, .15, 1.1],
                    "selected_hypothesis": "SQUARE_IDENTICAL_WD",
                    "reprojection_px": 1.25,
                }

        extractor = Extractor()
        stack = run.LegacyStack(
            extractor=extractor, head=object(), inference=Inference, pose=Pose,
            dimensions_wdh=np.asarray([1.1, 1.1, .15]), symmetry_order=4,
            temperature=1., rule={"lam": 1.,
                                  "max_move_image_diagonal_fraction": .01},
            normalization={}, checkpoint=Path("fixture.pt"))
        metadata = {
            "session_id": "fixture", "video_index": 0, "frame_i": 36,
            "camera_frame_number": 100,
            lifter.SENSOR_TIME_FIELD: 1000.,
            "camera_timestamp_domain": "hardware_clock",
        }
        output = run.infer_shared_frame(
            np.zeros((480, 640, 3), np.uint8), metadata,
            run.camera_contract(fixture_meta()), stack)
        self.assertEqual(extractor.calls, 1)
        self.assertTrue(output["base_forward_shared"])
        self.assertEqual(set(output["methods"]), {"R0", "N3_seed1"})
        self.assertEqual(len(Pose.calls), 2)
        for _, K, xyz, source in Pose.calls:
            self.assertEqual(K[0, 2], 317.59619140625)
            np.testing.assert_array_equal(xyz, [1.1, .15, 1.1])
            self.assertFalse(source)
        self.assertTrue(output["n3"]["movement"]["center_preserved"])
        self.assertTrue(output["n3"]["movement"]["finite_mask_preserved"])
        self.assertAlmostEqual(output["methods"]["R0"]["yaw_deg"], 179.)
        self.assertTrue(output["methods"]["N3_seed1"]["fresh"])


def prediction(available, yaw=None, *, fresh=None):
    if fresh is None:
        fresh = available
    return {
        "available": available,
        "fresh": fresh if available else False,
        "pos_x_m": .1 if available else None,
        "pos_y_m": .2 if available else None,
        "pos_z_m": 2. if available else None,
        "yaw_deg": yaw if available else None,
    }


def row(session_id, timestamp, r0, n3):
    return {
        "session_id": session_id,
        lifter.SENSOR_TIME_FIELD: float(timestamp),
        "in_view": None,
        "methods": {"R0": r0, "N3_seed1": n3},
    }


class StatisticsAndReferenceTests(unittest.TestCase):
    def setUp(self):
        self.sessions = {
            "173507": [
                row("173507", 0, prediction(True, 179.), prediction(False)),
                row("173507", 1000, prediction(True, -179.), prediction(False)),
                row("173507", 2000, prediction(False), prediction(True, 10.)),
                row("173507", 3000, prediction(False), prediction(True, 11.)),
                row("173507", 4000, prediction(True, -178.), prediction(True, 12.)),
            ],
            "174126": [row("174126", 100, prediction(True, 50.),
                           prediction(True, -179.))],
            "174342": [row("174342", 200, prediction(True, -50.),
                           prediction(True, 179.))],
            "174925": [row("174925", 300, prediction(True, 90.),
                           prediction(True, 90.))],
        }

    def test_coverage_fresh_longest_missing_and_wrapped_yaw_are_separate(self):
        summary = run.summarize(self.sessions)
        r0 = summary["R0"]["overall"]
        self.assertEqual(r0["coverage"]["frames"], 8)
        self.assertEqual(r0["coverage"]["available_outputs"], 6)
        self.assertEqual(r0["coverage"]["fresh_predictions"], 6)
        self.assertEqual(r0["coverage"]["visibility_unknown_frames"], 8)
        longest = r0["missing"]["longest_observed_sensor_time_run"]
        self.assertEqual(longest["session_id"], "173507")
        self.assertEqual(longest["frames"], 2)
        self.assertEqual(longest["observed_span_s"], 1.)
        jitter = r0["yaw_wrap_jitter"]["fresh_predictions_only"]
        # The only adjacent, uninterrupted R0 pair is 179 -> -179: two degrees.
        self.assertEqual(jitter["adjacent_positive_dt_pairs"], 1)
        self.assertAlmostEqual(jitter["median_abs_step"], 2.)
        self.assertAlmostEqual(jitter["median_abs_rate_per_s"], 2.)
        self.assertTrue(jitter["angular_wrap_applied"])
        self.assertTrue(jitter["session_boundaries_never_joined"])
        self.assertTrue(r0["missing"]["sampled_timeline_not_full_frame_timeline"])

    def test_no_independent_reference_stays_null_plus_x(self):
        value = run.accuracy_x()
        run._validate_accuracy_x(value)
        self.assertFalse(value["independent_ground_truth"])
        for name in ("independent_accuracy", "position_error_m", "yaw_error_deg"):
            self.assertIsNone(value[name]["value"])
            self.assertEqual(value[name]["status"], "x")
        invalid = copy.deepcopy(value)
        invalid["position_error_m"] = {"value": .1, "status": "measured"}
        with self.assertRaises(ValueError):
            run._validate_accuracy_x(invalid)

    def test_module_exposes_no_vehicle_action_surface(self):
        source = inspect.getsource(run)
        forbidden = (
            "import canlib", "write" + "Sync(", "send_" + "command(",
            "set_" + "joystick(", "move_" + "fork(")
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)
        self.assertIn("actual_control_invoked", source)


class RawSchemaTests(unittest.TestCase):
    def test_complete_raw_recomputes_statistics_and_keeps_accuracy_x(self):
        plan = run.build_plan(verify=False)
        payload = run._progress_identity(plan)
        sessions = {}
        for session_id in lifter.USABLE_SESSION_IDS:
            frames = []
            for identity in plan["sessions"][session_id]["frames"]:
                unavailable = {
                    "available": False, "fresh": False, "detected": False,
                    "selected_index": None, "head_used": False,
                    "points_xy": None, "pos_x_m": None, "pos_y_m": None,
                    "pos_z_m": None, "yaw_deg": None,
                    "pose": {"available": False},
                }
                frames.append({
                    **identity, "in_view": None,
                    "visible_state": "UNKNOWN_NOT_ANNOTATED",
                    "base_forward_shared": True,
                    "methods": {
                        "R0": dict(unavailable), "N3_seed1": dict(unavailable)},
                    "n3": {
                        "arm": run.ARM, "seed": run.SEED, "temperature": 1.,
                        "rule": {"lam": 1.,
                                 "max_move_image_diagonal_fraction": .01},
                        "movement": {
                            "comparable_corners": 0, "mean_move_px": None,
                            "max_move_px": None, "center_preserved": True,
                            "finite_mask_preserved": True,
                        },
                        "diagnostic_present": False,
                    },
                })
            sessions[session_id] = {
                "camera": plan["sessions"][session_id]["camera"],
                "frames": frames,
            }
        total = plan["total_sampled_frames"]
        payload.update({
            "complete": True, "sessions": sessions,
            "base_forward_count": total, "shared_base_forward_count": total,
            "n3_call_count": total,
            "statistics": run.summarize({sid: sessions[sid]["frames"]
                                         for sid in lifter.USABLE_SESSION_IDS}),
            "accuracy": run.accuracy_x(),
        })
        self.assertEqual(run.validate_raw(payload, plan, verify_files=False), payload)
        broken = copy.deepcopy(payload)
        broken["accuracy"]["yaw_error_deg"] = {"value": 1., "status": "measured"}
        with self.assertRaisesRegex(ValueError, "accuracy field populated"):
            run.validate_raw(broken, plan, verify_files=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
