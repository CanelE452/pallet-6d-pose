"""CPU-only fixture tests for the N3 hash-bound orchestrator."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.research.pallet_n3_completion_v3 import run as R


def _write(path: Path, value: str = "fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


class PlanTests(unittest.TestCase):
    def test_every_declared_subprocess_is_a_real_current_cli(self):
        root = Path(tempfile.mkdtemp())
        python = "/fixture/python"
        expected_modules = {
            "preflight": ("environment", "preflight"),
            "smoke": ("train",), "train": ("train",),
            "calibrate": ("selection", "selection"),
            "infer": ("inference", "inference"),
            "evaluate": ("evaluation",), "square": ("square",),
            "square-yolo": ("square_yolo",), "runtime": ("runtime",),
            "lifter": ("lifter_run", "lifter_run", "lifter_run"),
            "reuse": ("reuse",),
            "audit": ("symmetry_activation_audit", "dimension_sensitivity_audit"),
            "verify": ("integrity",),
        }
        for stage, modules in expected_modules.items():
            backbone = "dope" if stage in R.PER_BACKBONE else None
            plan = R.build_plan(stage, backbone, python=python, root=root)
            self.assertTrue(plan.commands, stage)
            self.assertEqual(tuple(command[2].rsplit(".", 1)[-1]
                                   for command in plan.commands), modules)
            for command in plan.commands:
                self.assertEqual(command[:2], (python, "-m"))
                self.assertTrue(command[2].startswith(f"{R.PACKAGE}."))
                self.assertNotIn("shell=True", command)
        evaluation = R.build_plan("evaluate", "dope", python=python, root=root)
        self.assertIn("--predictions", evaluation.commands[0])
        calibration = R.build_plan("calibrate", "dope", python=python, root=root)
        self.assertEqual([command[3] for command in calibration.commands],
                         ["validation", "calibrate"])
        tests = R.build_plan("tests", python=python, root=root)
        self.assertEqual(tests.commands[0][:4], (python, "-m", "pytest", "-q"))
        # No report.py exists in this fixture.  The stage must be an honest
        # artifact gate rather than a command that cannot run.
        report = R.build_plan("report", python=python, root=root)
        self.assertEqual(report.commands, ())
        self.assertEqual(report.internal, "report_gate")

    def test_current_report_cli_and_full_output_inventory_are_bound(self):
        root = R.repository_root()
        plan = R.build_plan("report", python="/fixture/python", root=root)
        self.assertEqual(plan.commands, ((
            "/fixture/python", "-m", f"{R.PACKAGE}.report", "generate"),))
        self.assertIn("scripts/research/pallet_n3_completion_v3/report.py", plan.code)
        expected = {
            *(str(path) for path in R.REPORT_OUTPUTS),
            *(str(R.DOC_REL / "figures" / name) for name in R.REPORT_FIGURES),
            str(R.DOC_REL / "figures/SELECTION_MANIFEST.json"),
            *(str(R.DOC_REL / "table_fragments" / name) for name in R.REPORT_TEX),
            *(str(R.RAW_REL / "report" / name) for name in R.REPORT_RAW),
        }
        self.assertEqual(set(plan.outputs), expected)
        self.assertNotIn(str(R.DOC_REL / "VERIFY_RESULTS.json"), plan.inputs)

    def test_evaluate_binds_the_exact_imported_pose_source(self):
        root = R.repository_root()
        plan = R.build_plan(
            "evaluate", "dope", python="/fixture/python", root=root)
        pose = "scripts/research/pallet_dim_conditioned_p_v1/pose.py"
        nonexistent_shim = "scripts/research/pallet_n3_completion_v3/pose.py"
        self.assertIn(pose, plan.code)
        self.assertNotIn(nonexistent_shim, plan.code)
        self.assertTrue(all(R._resolve(root, path) is not None
                            for path in plan.code))
        binding = R.file_binding(root, pose)
        self.assertEqual(binding["path"], pose)
        self.assertRegex(binding["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(binding["bytes"], 0)

    def test_all_expands_frozen_order_and_selected_backbone(self):
        root = Path(tempfile.mkdtemp())
        plans = R.expand_plans("all", "dope", root=root, python="/fixture/python")
        self.assertEqual([plan.name for plan in plans], list(R.ALL_ORDER))
        self.assertTrue(all(plan.backbone == "dope" for plan in plans
                            if plan.name in R.PER_BACKBONE))
        both = R.expand_plans("train", "all", root=root, python="/fixture/python")
        self.assertEqual([plan.backbone for plan in both], ["dope", "resnet18"])
        self.assertEqual(R.ALL_EXECUTION_ORDER[-5:],
                         ("report", "verify", "report", "verify", "manifest"))

    def test_parser_exposes_every_required_stage(self):
        parser = R.build_parser()
        for stage in R.STAGES:
            args = parser.parse_args([stage])
            self.assertEqual(args.stage, stage)

    def test_yolo26_stages_use_the_explicit_compatible_interpreter(self):
        root = Path(tempfile.mkdtemp())
        ordinary = "/fixture/pallet-pose/python"
        yolo = "/fixture/pallet-yolo26/python"
        square = R.build_plan(
            "square-yolo", python=ordinary, yolo_python=yolo, root=root)
        lifter = R.build_plan(
            "lifter", python=ordinary, yolo_python=yolo, root=root)
        runtime = R.build_plan(
            "runtime", "dope", python=ordinary, yolo_python=yolo, root=root)
        self.assertTrue(all(command[0] == yolo for command in square.commands))
        self.assertTrue(all(command[0] == yolo for command in lifter.commands))
        self.assertTrue(all(command[0] == ordinary for command in runtime.commands))
        self.assertEqual(square.config["python_executable"], yolo)
        self.assertEqual(runtime.config["python_executable"], ordinary)

    def test_global_audit_is_between_reuse_and_report_and_hash_binds_inputs(self):
        root = R.repository_root()
        plan = R.build_plan("audit", python="/fixture/python", root=root)
        self.assertIsNone(plan.backbone)
        self.assertFalse(plan.gpu)
        self.assertEqual(tuple(command[2].rsplit(".", 1)[-1]
                               for command in plan.commands),
                         ("symmetry_activation_audit",
                          "dimension_sensitivity_audit"))
        self.assertTrue(all(command[0] == "/fixture/python"
                            for command in plan.commands))
        self.assertEqual(R.ALL_ORDER[R.ALL_ORDER.index("reuse") + 1], "audit")
        self.assertEqual(R.ALL_ORDER[R.ALL_ORDER.index("audit") + 1], "report")
        self.assertEqual(set(plan.outputs), {
            str(R.DOC_REL / "SYMMETRY_ACTIVATION_AUDIT.json"),
            str(R.DOC_REL / "DIMENSION_SENSITIVITY_AUDIT.json"),
        })
        self.assertIn(str(R.DOC_REL / "ENVIRONMENT_AUDIT.json"), plan.inputs)
        self.assertIn(str(R.RAW_REL / "orders/dope_seed1.npy"), plan.inputs)
        self.assertIn(str(R.RAW_REL / "runs/resnet18/seed3/last.pt"), plan.inputs)
        self.assertIn(
            "scripts/research/pallet_n3_completion_v3/symmetry_activation_audit.py",
            plan.code)
        self.assertIn(
            "scripts/research/pallet_dope_refiner_20261001_v1/refiner.py",
            plan.code)

    def test_audit_artifacts_are_manifest_candidates_and_verify_inputs(self):
        candidates = {str(path) for path in R._manifest_candidates()}
        expected = {
            str(R.DOC_REL / "ENVIRONMENT_AUDIT.json"),
            str(R.DOC_REL / "SYMMETRY_ACTIVATION_AUDIT.json"),
            str(R.DOC_REL / "DIMENSION_SENSITIVITY_AUDIT.json"),
        }
        self.assertTrue(expected <= candidates)
        verify = R.build_plan("verify", python="/fixture/python",
                              root=R.repository_root())
        self.assertTrue(expected <= set(verify.inputs))


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        _write(self.root / "input.bin", "input-v1")
        _write(self.root / "code.py", "code-v1")
        self.plan = R.StagePlan(
            name="fixture", backbone=None,
            commands=(("/fixture/python", "fixture-command"),),
            inputs=("input.bin",), code=("code.py",), outputs=("output.bin",),
            config={"alpha": 1}, gpu=False)

    def tearDown(self):
        self.directory.cleanup()

    def _runner(self, calls: list[list[str]], content: str = "result"):
        def run(command, *, cwd, check):
            self.assertFalse(check)
            self.assertEqual(Path(cwd), self.root)
            calls.append(command)
            _write(self.root / "output.bin", content)
            return SimpleNamespace(returncode=0)
        return run

    def test_matching_receipt_skips_and_binds_every_hash_class(self):
        calls: list[list[str]] = []
        first = R.execute_plan(self.plan, root=self.root,
                               run_command=self._runner(calls))
        second = R.execute_plan(self.plan, root=self.root,
                                run_command=self._runner(calls))
        self.assertEqual(first["status"], "COMPLETE")
        self.assertEqual(second["status"], "SKIPPED")
        self.assertEqual(len(calls), 1)
        receipt = json.loads(R.receipt_path(self.root, self.plan).read_text())
        for name in ("inputs", "code", "outputs"):
            self.assertEqual(len(receipt[name]), 1)
            self.assertRegex(receipt[name][0]["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(receipt[f"{name}_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(receipt["config_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(receipt["restart_policy"],
                         "skip only when all input/code/config/output hashes match")

    def test_changed_output_forces_rerun(self):
        calls: list[list[str]] = []
        R.execute_plan(self.plan, root=self.root, run_command=self._runner(calls))
        _write(self.root / "output.bin", "tampered")
        result = R.execute_plan(self.plan, root=self.root,
                                run_command=self._runner(calls, "restored"))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(calls), 2)
        self.assertEqual((self.root / "output.bin").read_text(), "restored")

    def test_changed_input_forces_rerun(self):
        calls: list[list[str]] = []
        R.execute_plan(self.plan, root=self.root, run_command=self._runner(calls))
        _write(self.root / "input.bin", "input-v2")
        result = R.execute_plan(self.plan, root=self.root,
                                run_command=self._runner(calls))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(calls), 2)

    def test_success_exit_without_output_is_failure_receipt(self):
        def empty_runner(command, *, cwd, check):
            return SimpleNamespace(returncode=0)
        with self.assertRaises(R.StageError):
            R.execute_plan(self.plan, root=self.root, run_command=empty_runner)
        receipt = json.loads(R.receipt_path(self.root, self.plan).read_text())
        self.assertFalse(receipt["complete"])
        self.assertEqual(receipt["status"], "FAILED")
        self.assertIn("missing required files", receipt["error"])

    def test_nonzero_subprocess_stops_without_claiming_completion(self):
        def failed(command, *, cwd, check):
            return SimpleNamespace(returncode=7)
        with self.assertRaises(R.StageError):
            R.execute_plan(self.plan, root=self.root, run_command=failed)
        receipt = json.loads(R.receipt_path(self.root, self.plan).read_text())
        self.assertEqual(receipt["returncodes"], [7])
        self.assertEqual(receipt["outputs"], [])

    def test_changed_embedded_source_binding_cannot_be_omitted_from_receipt(self):
        plan = R.StagePlan(
            name="fixture_json", backbone=None,
            commands=(("/fixture/python", "fixture-command"),),
            inputs=("input.bin",), code=("code.py",), outputs=("output.json",),
            config={}, gpu=False)
        def runner(command, *, cwd, check):
            _write(self.root / "output.json", json.dumps({
                "complete": True,
                "source": {"path": "missing-source.bin", "sha256": "0" * 64,
                           "bytes": 1}}))
            return SimpleNamespace(returncode=0)
        with self.assertRaisesRegex(R.StageError, "missing or changed binding"):
            R.execute_plan(plan, root=self.root, run_command=runner)
        receipt = json.loads(R.receipt_path(self.root, plan).read_text())
        self.assertFalse(receipt["complete"])

    def test_verify_partial_exit_two_is_a_completed_audit(self):
        plan = R.StagePlan(
            name="verify", backbone=None,
            commands=(("/fixture/python", "verify"),),
            inputs=("input.bin",), code=("code.py",), outputs=("verify.json",),
            config={}, accepted_returncodes=(0, 2))
        def runner(command, *, cwd, check):
            _write(self.root / "verify.json", json.dumps({
                "schema": "pallet_n3_completion_v3_verify_v1",
                "status": "PARTIAL", "integrity_status": "PASS",
                "complete": False, "remaining_x": []}))
            return SimpleNamespace(returncode=2)
        result = R.execute_plan(plan, root=self.root, run_command=runner)
        self.assertEqual(result["status"], "COMPLETE")
        receipt = json.loads(R.receipt_path(self.root, plan).read_text())
        self.assertEqual(receipt["returncodes"], [2])
        self.assertEqual(receipt["accepted_returncodes"], [0, 2])

    def test_verify_blocked_integrity_is_not_accepted_as_completion(self):
        plan = R.StagePlan(
            name="verify", backbone=None,
            commands=(("/fixture/python", "verify"),),
            inputs=("input.bin",), code=("code.py",), outputs=("verify.json",),
            config={}, accepted_returncodes=(0, 2))
        def runner(command, *, cwd, check):
            _write(self.root / "verify.json", json.dumps({
                "schema": "pallet_n3_completion_v3_verify_v1",
                "status": "BLOCKED_INTEGRITY", "complete": False}))
            return SimpleNamespace(returncode=2)
        with self.assertRaisesRegex(R.StageError, "BLOCKED_INTEGRITY"):
            R.execute_plan(plan, root=self.root, run_command=runner)
        receipt = json.loads(R.receipt_path(self.root, plan).read_text())
        self.assertFalse(receipt["complete"])


class StatusAndGateTests(unittest.TestCase):
    def test_manifest_is_replayable_partial_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("run.py", "common.py"):
                _write(root / R.HERE_REL / name, f"# {name}\n")
            _write(root / R.DOC_REL / "PROTOCOL.json", "{}")
            plan = R.build_plan("manifest", python="/fixture/python", root=root)
            result = R.execute_plan(plan, root=root)
            self.assertEqual(result["status"], "COMPLETE")
            manifest = json.loads((root / R.DOC_REL / "RUN_MANIFEST.json").read_text())
            self.assertEqual(manifest["git"]["starting_head"], "d1524b3")
            self.assertEqual(manifest["git"]["final_head"]["display"], "x")
            self.assertEqual(manifest["training"]["expected_fits"], 6)
            self.assertEqual(manifest["training"]["expected_updates"], 36_000)
            self.assertEqual(manifest["training"]["expected_sample_exposures"], 576_000)
            self.assertEqual(manifest["overall_status"], "PARTIAL")
            self.assertFalse(manifest["overall_complete"])
            self.assertTrue({"environment_audit", "symmetry_activation_audit",
                             "dimension_sensitivity_audit"}
                            <= set(manifest["bindings"]))
            self.assertTrue(all(
                manifest["bindings"][name]["status"] == "MISSING"
                for name in ("environment_audit", "symmetry_activation_audit",
                             "dimension_sensitivity_audit")))
            self.assertEqual({row["item"] for row in manifest["scope"]["exclusions"]},
                             {"PDF generation", "actual lifter control", "new self-training"})

    def test_fresh_progress_is_running_across_pid_namespaces_then_becomes_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # train.py writes TRAIN_PROGRESS immediately before a 500-step
            # checkpoint, so the checkpoint may legitimately have the later
            # mtime at the exact same step.
            _write(root / R.RAW_REL / "runs/dope/seed3/last.pt", "checkpoint")
            progress = {
                "complete": False, "backbone": "dope", "seed": 3,
                "step": 500, "elapsed_seconds": 90.,
                "gpu": {"time": "fixture", "processes": "3537436, python, 9000"},
            }
            fresh = R._training_status(
                root, "dope", 3, progress, [],
                progress_mtime=1000., observed_epoch=1100.)
            self.assertEqual(fresh["status"], "RUNNING")
            self.assertFalse(fresh["process_visible_in_this_pid_namespace"])
            self.assertIn("fresh advancing", fresh["running_evidence"])
            stale = R._training_status(
                root, "dope", 3, progress, [],
                progress_mtime=1000., observed_epoch=1200.)
            self.assertEqual(stale["status"], "INTERRUPTED_RESUMABLE")

    def test_status_is_gpu_free_and_reports_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = {
                "complete": False, "backbone": "dope", "seed": 2,
                "step": 2800, "pid": 1234, "updated_at": "fixture-time",
            }
            path = root / R.RAW_REL / "TRAIN_PROGRESS.json"
            _write(path, json.dumps(progress))
            snapshot = R.status_snapshot(
                root=root, backbone="dope", python="/fixture/python",
                process_reader=lambda: "1234 77 /fixture/python -m scripts.research.pallet_n3_completion_v3.train train dope\n")
            seed2 = next(row for row in snapshot["training"] if row["seed"] == 2)
            self.assertEqual(seed2["status"], "RUNNING")
            self.assertEqual(seed2["step"], 2800)
            self.assertTrue(seed2["process_visible_in_this_pid_namespace"])
            self.assertFalse(snapshot["GPU_queried"])
            self.assertEqual(snapshot["matching_processes"][0]["pid"], "1234")
            self.assertNotIn("nvidia-smi", inspect.getsource(R.status_snapshot))

    def test_report_without_cli_is_explicit_blocked_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = R.build_plan("report", python="/fixture/python", root=root)
            for path in (*plan.inputs, *plan.code):
                _write(root / path, "{}")
            with self.assertRaisesRegex(R.StageBlocked, "report CLI is not implemented"):
                R.execute_plan(plan, root=root)
            receipt = json.loads(R.receipt_path(root, plan).read_text())
            self.assertFalse(receipt["complete"])
            self.assertEqual(receipt["returncodes"], [])


if __name__ == "__main__":
    unittest.main()
