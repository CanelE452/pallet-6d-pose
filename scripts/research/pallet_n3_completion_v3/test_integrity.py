from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.research.pallet_n3_completion_v3 import common as C
from scripts.research.pallet_n3_completion_v3 import integrity as I
from scripts.research.pallet_n3_completion_v3.model import build_n3


def protocol_fixture() -> dict:
    optimizer = {"name": "AdamW", **C.OPTIMIZER}
    optimizer["betas"] = list(optimizer["betas"])
    return {
        "schema": "pallet_n3_completion_v3_protocol_v1",
        "methods": {"base_inputs": ["RGB"]},
        "backbones": {
            "dope": {
                "checkpoint": {"sha256": "dope-base"},
                "frozen": True, "retrained": False, "base_dimension_input": False,
            },
            "resnet18": {
                "source_checkpoint": {"sha256": "resnet-base"},
                "frozen": True, "retrained": False, "base_dimension_input": False,
            },
        },
        "training": {
            "backbones": ["dope", "resnet18"], "seeds": [1, 2, 3],
            "fits": 6, "steps_per_fit": 6000, "batch": 16,
            "exposures_per_fit": 96_000, "total_updates": 36_000,
            "total_exposures": 576_000,
            "optimizer": optimizer,
            "precision": {"model": "FP32", "AMP": False, "TF32": False},
            "selection": "fixed final step6000 only",
            "outcome_driven_extension": False,
        },
    }


def fit_fixture(backbone="dope", seed=1):
    protocol = protocol_fixture()
    order = np.arange(6000 * 16, dtype=np.int64).reshape(6000, 16)
    order_digest = I.order_hash(order)
    model = {"weight": torch.tensor([1., 2.], dtype=torch.float32)}
    final_hash = I.state_hash(model)
    base_sha = "dope-base" if backbone == "dope" else "resnet-base"
    optimizer = copy.deepcopy(C.OPTIMIZER)
    optimizer["betas"] = list(optimizer["betas"])
    receipt = {
        "schema": "pallet_n3_completion_v3_fit_v1", "complete": True,
        "backbone": backbone, "seed": seed, "smoke": False,
        "steps": 6000, "exposures": 96_000,
        "order_prefix_sha256": order_digest,
        "initial_state_sha256": "initial", "final_state_sha256": final_hash,
        "base_checkpoint_sha256": base_sha,
        "optimizer": optimizer, "FP32": True, "AMP": False,
        "TF32_matmul": False, "TF32_cudnn": False,
        "real_training_images": 0, "base_retrained": False,
        "symmetry_supervision": True, "dimension_input_to_N3": True,
        "base_receives_dimensions": False, "final_step_only": True,
        "history": [{"step": 1}, {"step": 6000}],
        "first_step": {"zero_effect_logits": True},
        "second_step": {"metadata_encoder_connected": True},
    }
    checkpoint = {
        "complete": True, "backbone": backbone, "seed": seed,
        "smoke": False, "step": 6000, "steps": 6000,
        "protocol_sha256": "protocol", "order_sha256": order_digest,
        "initial_state_sha256": "initial", "base_checkpoint_sha256": base_sha,
        "model_state_dict": model,
        "optimizer_state_dict": {
            "state": {0: {
                "step": torch.tensor(6000., dtype=torch.float32),
                "exp_avg": torch.zeros(2, dtype=torch.float32),
                "exp_avg_sq": torch.ones(2, dtype=torch.float32),
            }},
            "param_groups": [{
                "lr": C.learning_rate(6000), "betas": C.OPTIMIZER["betas"],
                "weight_decay": C.OPTIMIZER["weight_decay"], "params": [0],
            }],
        },
        "torch_rng_state": torch.zeros(32, dtype=torch.uint8),
        "cuda_rng_state": [torch.ones(32, dtype=torch.uint8)],
    }
    return protocol, receipt, checkpoint, order


def protocol_audit_fixtures() -> dict[str, dict]:
    environment = {
        "schema": "pallet_n3_completion_v3_environment_audit_v1",
        "complete": True,
        "roles": {
            "dope_resnet_training_inference_runtime": {"has_C3k2": False},
            "yolo26_square_and_offline_lifter": {"has_C3k2": True},
        },
        "boundary": {"single_process_environment_mixing": False},
    }

    def counts(total, non_identity):
        return {
            "rows_or_exposures": total,
            "branch_counts": {"0": total - non_identity, "1": non_identity,
                              "2": 0, "3": 0},
            "identity_count": total - non_identity,
            "non_identity_count": non_identity,
            "non_identity_fraction": non_identity / total,
        }

    symmetry = {
        "schema": "pallet_n3_completion_v3_symmetry_activation_audit_v1",
        "complete": True,
        "algorithm": {"whole_object_permutation": True,
                      "identity_first_exact_tie": True, "inference_use": False},
        "backbones": {
            backbone: {
                "unique_usable_rows": counts(10, 1 if backbone == "dope" else 0),
                "seeds": {str(seed): counts(96_000, seed if backbone == "dope" else 0)
                          for seed in I.SEEDS},
            }
            for backbone in I.BACKBONES
        },
        "interpretation": {"objective_applied": True},
    }
    fits = {}
    for backbone, seed in I.EXPECTED_FITS:
        fits[f"{backbone}_seed{seed}"] = {
            "status": "PASS", "visual_inputs_bitwise_identical": True,
            "base_logits_bitwise_identical": True, "finite": True,
            "changed_logit_entries": 1776,
        }
    sensitivity = {
        "schema": "pallet_n3_completion_v3_dimension_sensitivity_audit_v1",
        "complete": True, "fits": fits,
        "summary": {"status": "PASS", "fits_verified": 6,
                    "all_six_changed": True},
    }
    return {"environment_audit": environment,
            "symmetry_activation_audit": symmetry,
            "dimension_sensitivity_audit": sensitivity}


class ProtocolAndFitTests(unittest.TestCase):
    def test_exact_six_fit_budget_and_optimizer_contract(self):
        protocol, receipt, checkpoint, order = fit_fixture()
        protocol_audit = I.validate_protocol_payload(protocol)
        self.assertEqual(protocol_audit["total_exposures"], 576_000)
        result = I.validate_fit_payload(
            receipt, checkpoint, order, protocol, "protocol", "dope", 1)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["steps"], 6000)
        self.assertEqual(result["exposures"], 96_000)
        self.assertEqual(result["precision"], {"FP32": True, "AMP": False, "TF32": False})
        self.assertEqual(result["rng_state"], "CPU and CUDA states present")

    def test_wrong_step_tf32_order_and_rng_are_hard_failures(self):
        protocol, receipt, checkpoint, order = fit_fixture()
        cases = []
        wrong = copy.deepcopy(checkpoint); wrong["step"] = 5999
        cases.append((receipt, wrong, order, "step6000"))
        wrong_receipt = copy.deepcopy(receipt); wrong_receipt["TF32_cudnn"] = True
        cases.append((wrong_receipt, checkpoint, order, "TF32"))
        wrong_order = order.copy(); wrong_order[0, 0] += 1
        cases.append((receipt, checkpoint, wrong_order, "order hash"))
        wrong_rng = copy.deepcopy(checkpoint); wrong_rng["cuda_rng_state"] = []
        cases.append((receipt, wrong_rng, order, "CUDA RNG"))
        for candidate_receipt, candidate_checkpoint, candidate_order, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(I.IntegrityError, message):
                    I.validate_fit_payload(
                        candidate_receipt, candidate_checkpoint, candidate_order,
                        protocol, "protocol", "dope", 1)

    def test_protocol_cannot_shrink_576k_budget(self):
        protocol = protocol_fixture()
        protocol["training"]["total_exposures"] = 575_999
        with self.assertRaisesRegex(I.IntegrityError, "total exposure"):
            I.validate_protocol_payload(protocol)


class DimensionAndBaseInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normalization = C.read(C.NORMALIZATION)

    def test_actual_n3_logits_change_with_only_wdh_changed(self):
        head = build_n3("dope")
        with torch.no_grad():
            # The trained checkpoint should have learned this layer. This
            # deterministic fixture makes the same structural fact observable.
            head.metadata_scorer[-1].weight.fill_(.01)
            head.metadata_scorer[-1].bias.zero_()
        result = I.measure_dimension_sensitivity(
            "dope", head.state_dict(), self.normalization, "fixture")
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["visual_inputs_bitwise_identical"])
        self.assertTrue(result["base_logits_bitwise_identical"])
        self.assertGreater(result["changed_logit_entries"], 0)
        self.assertGreater(result["max_abs_logit_delta"], 0.)

    def test_zero_initialized_untrained_head_does_not_fake_sensitivity(self):
        head = build_n3("dope")
        result = I.measure_dimension_sensitivity(
            "dope", head.state_dict(), self.normalization, "fixture")
        self.assertEqual(result["status"], "BLOCKED_INTEGRITY")
        self.assertIsNone(result["value"])
        self.assertEqual(result["display"], "x")

    def test_bound_base_apis_are_rgb_only_and_dimensions_enter_n3_batch(self):
        training = {
            "fits": {f"{backbone}_seed{seed}": {
                "status": "PASS", "precision": {"FP32": True}}
                for backbone in I.BACKBONES for seed in I.SEEDS}}
        result = I.prove_base_rgb_only(C.ROOT, protocol_fixture(), training)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["adapter_infer_receives_dimensions"])
        self.assertTrue(result["n3_batch_receives_dimensions"])
        for arguments in result["adapter_infer_arguments"].values():
            self.assertFalse(any("dimension" in name for name in arguments))


class ArtifactAndMissingPolicyTests(unittest.TestCase):
    def test_all_three_protocol_audits_are_required_integrity_artifacts(self):
        rows = {name: (relative, category, json_file)
                for name, relative, category, json_file in I.ARTIFACT_SPECS}
        self.assertTrue(set(protocol_audit_fixtures()) <= set(rows))
        for name in protocol_audit_fixtures():
            self.assertEqual(rows[name][1:], ("protocol", True))

    def test_protocol_audit_semantics_pass_and_drift_is_blocked(self):
        fixtures = protocol_audit_fixtures()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = []
            for name, payload in fixtures.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload))
                specs.append((name, path.name, "protocol", True))
            passed = I.inspect_artifacts(root, specs=tuple(specs))
            self.assertEqual(passed["protocol"]["status"], "PASS")

            mutations = {
                "environment_audit": lambda value: value["roles"][
                    "yolo26_square_and_offline_lifter"].update(has_C3k2=False),
                "symmetry_activation_audit": lambda value: value["backbones"][
                    "dope"]["seeds"]["1"]["branch_counts"].update({"0": 1}),
                "dimension_sensitivity_audit": lambda value: value["fits"][
                    "dope_seed1"].update(changed_logit_entries=0),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    candidate = copy.deepcopy(fixtures[name])
                    mutate(candidate)
                    (root / f"{name}.json").write_text(json.dumps(candidate))
                    result = I.inspect_artifacts(
                        root, specs=((name, f"{name}.json", "protocol", True),))
                    self.assertEqual(result["protocol"]["status"],
                                     "BLOCKED_INTEGRITY")
                    (root / f"{name}.json").write_text(json.dumps(fixtures[name]))

    def test_runtime_raw_pose_null_is_allowed_only_in_locked_row_schema(self):
        payload = {
            "schema": "pallet_n3_completion_v3_runtime_raw_v1",
            "complete": True,
            "warmup": [{"path": "n3_seed1_only", "output": {"pose": None}}],
            "measurements": [{"path": "base_e2e", "output": {
                "pose": {"available": False, "status": "UNAVAILABLE"}}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runtime.json"
            path.write_text(json.dumps(payload))
            spec = (("runtime_dope_raw", "runtime.json", "runtime", True),)
            passed = I.inspect_artifacts(root, specs=spec)
            row = passed["runtime"]["artifacts"]["runtime_dope_raw"]
            self.assertEqual(passed["runtime"]["status"], "PASS")
            self.assertEqual(row["schema_valid_nullable_runtime_pose_count"], 1)

            invalid = copy.deepcopy(payload)
            invalid["unrelated"] = {"pose": None}
            path.write_text(json.dumps(invalid))
            blocked = I.inspect_artifacts(root, specs=spec)
            self.assertEqual(blocked["runtime"]["status"], "BLOCKED_INTEGRITY")

            invalid = copy.deepcopy(payload)
            invalid["measurements"][0]["output"]["pose"] = {
                "available": False, "status": "OK"}
            path.write_text(json.dumps(invalid))
            blocked = I.inspect_artifacts(root, specs=spec)
            self.assertEqual(blocked["runtime"]["status"], "BLOCKED_INTEGRITY")

    def test_missing_artifact_is_explicit_x_not_numeric_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            result = I.inspect_artifacts(
                directory, specs=(("missing", "missing.json", "evaluation", True),))
        entry = result["evaluation"]["artifacts"]["missing"]
        self.assertEqual(result["evaluation"]["status"], "INCOMPLETE")
        self.assertEqual(entry["status"], "MISSING")
        self.assertIsNone(entry["value"])
        self.assertEqual(entry["display"], "x")

    def test_present_artifact_and_embedded_hash_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"bound")
            payload = root / "result.json"
            payload.write_text(json.dumps({
                "complete": True,
                "input": {"path": "source.bin", "sha256": C.sha256(source),
                          "bytes": source.stat().st_size},
            }) + "\n")
            specs = (("result", "result.json", "evaluation", True),)
            passed = I.inspect_artifacts(root, specs=specs)
            self.assertEqual(passed["evaluation"]["status"], "PASS")
            self.assertEqual(
                passed["evaluation"]["artifacts"]["result"]["embedded_bindings_verified"], 1)
            source.write_bytes(b"changed")
            blocked = I.inspect_artifacts(root, specs=specs)
            self.assertEqual(blocked["evaluation"]["status"], "BLOCKED_INTEGRITY")

    def test_unexplained_null_is_rejected_but_declared_x_is_valid(self):
        self.assertEqual(I.explicit_x_issues(
            {"status": "MISSING", "result": None, "display": "x"}), [])
        self.assertEqual(I.explicit_x_issues(
            {"physical_pose_reference": False, "pose_metrics": None}), [])
        self.assertEqual(I.explicit_x_issues(
            {"status": "COMPLETE", "result": None}), ["$.result"])

    def test_na_marker_explains_protocol_defined_null(self):
        self.assertEqual(I.explicit_x_issues(
            {"value": None, "status": "NA", "display": "NA",
             "reason": "zero denominator"}), [])
        self.assertEqual(I.explicit_x_issues(
            {"result": None, "status": "NOT_APPLICABLE",
             "reason": "metric is undefined for this population"}), [])

    def test_lifter_accuracy_must_remain_null_x_without_reference(self):
        valid = {
            "independent_ground_truth": False,
            "independent_accuracy": {"value": None, "status": "x"},
            "position_error_m": {"value": None, "status": "x"},
            "yaw_error_deg": {"value": None, "status": "x"},
        }
        I._validate_lifter_x({"accuracy": valid})
        invalid = copy.deepcopy(valid)
        invalid["yaw_error_deg"] = {"value": 0., "status": "COMPLETE"}
        with self.assertRaisesRegex(I.IntegrityError, "yaw_error"):
            I._validate_lifter_x({"accuracy": invalid})

    def test_contractual_x_prevents_overall_complete_even_when_integrity_passes(self):
        passed = {"status": "PASS"}
        artifacts = {"evaluation": {"status": "PASS", "artifacts": {}}}
        gap = [{"id": "semantic_gap", "display": "x", "value": None}]
        self.assertEqual(
            I._overall_status(passed, passed, passed, passed, artifacts),
            "OVERALL_COMPLETE")
        self.assertEqual(
            I._overall_status(passed, passed, passed, passed, artifacts, gap),
            "PARTIAL")

    def test_reuse_partial_labels_and_safe_cohort_are_remaining_x(self):
        payload = {
            "contracts": {
                "D": {
                    "status": "PARTIAL_LABELS",
                    "frame_counts": {"unclassified": 191},
                    "corner_visibility": {
                        "reason": "corner_visibility_collected=false"},
                },
                "I": {
                    "status": "PARTIAL_SAFE_COHORT",
                    "DEV319_table": {
                        "status": "BLOCKED_CONTRACT",
                        "reason": "No common non-exposed DEV319 panel exists"},
                },
            }}
        rows = I.contract_remaining_x("reuse_summary", "reuse", payload)
        self.assertEqual({row["id"] for row in rows}, {
            "reuse_D_partial_labels", "reuse_I_partial_safe_cohort"})
        self.assertTrue(all(row["value"] is None and row["display"] == "x"
                            for row in rows))
        self.assertIn("191", rows[0]["reason"])

    def test_lifter_and_square_unsupported_accuracy_stay_remaining_x(self):
        lifter = I.contract_remaining_x("lifter_raw", "lifter", {
            "accuracy": {"independent_ground_truth": False}})
        self.assertEqual(lifter[0]["id"],
                         "lifter_accuracy_position_yaw_no_independent_gt")
        self.assertIn("position error", lifter[0]["reason"])
        square = I.contract_remaining_x("square_dope", "square", {
            "physical_pose_reference": False, "pose_metrics": None})
        self.assertEqual(square[0]["status"],
                         "BLOCKED_NO_INDEPENDENT_CANONICAL_POSE_REFERENCE")
        self.assertIn("6D", square[0]["reason"])

    def test_partial_report_can_pass_integrity_but_figure_hash_drift_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            doc = root / "_docs/experiments/pallet_n3_completion_v3"
            figures = doc / "figures"
            figures.mkdir(parents=True)
            tables = doc / "TABLES.json"
            tables.write_text(json.dumps({"schema": "pallet_n3_completion_v3_report_v1"}))
            names = (
                "training_curves.png", "backbone_comparison.png", "runtime.png",
                "dev_overlays.png", "lifter.png", "subgroups_or_thresholds.png")
            rows = []
            for name in names:
                path = figures / name
                path.write_bytes(name.encode())
                rows.append({"file": name, "status": "MISSING",
                             "sha256": C.sha256(path), "bytes": path.stat().st_size})
            manifest = figures / "SELECTION_MANIFEST.json"
            manifest.write_text(json.dumps({
                "schema": "pallet_n3_completion_v3_figure_selection_v1",
                "complete": False, "overall_status": "PARTIAL", "figures": rows,
                "TABLES_json": {"path": str(tables.relative_to(root)),
                                "sha256": C.sha256(tables),
                                "bytes": tables.stat().st_size}}))
            tex = doc / "table_fragments/runtime.tex"
            tex.parent.mkdir(parents=True)
            tex.write_text(f"% Generated from TABLES.json sha256: {C.sha256(tables)}\n")
            specs = (
                ("report_tables", str(tables.relative_to(root)), "report", True),
                ("report_figure_manifest", str(manifest.relative_to(root)), "report", True),
                ("report_tex_runtime", str(tex.relative_to(root)), "report", False),
            )
            passed = I.inspect_artifacts(root, specs=specs)
            self.assertEqual(passed["report"]["status"], "PASS")
            (figures / names[0]).write_bytes(b"tampered")
            blocked = I.inspect_artifacts(root, specs=specs)
            self.assertEqual(blocked["report"]["status"], "BLOCKED_INTEGRITY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
