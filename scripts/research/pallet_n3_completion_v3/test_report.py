from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from scripts.research.pallet_n3_completion_v3 import integrity as I
from scripts.research.pallet_n3_completion_v3 import report as R


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _headline(offset: float = 0.) -> dict:
    return {
        "matched_pooled_corner8_median_px": 10. + offset,
        "matched_pooled_corner8_P90_px": 20. + offset,
        "full_PCK": {"5": .2, "10": .5 + offset / 100., "20": .8},
        "full_PCK10_fraction": .5 + offset / 100.,
        "E_sym": .1 + offset / 1000.,
        "full_penalty_P90_px": 30. + offset,
        "pose_translation_cm_median": 5. + offset,
        "pose_translation_cm_P90": 9. + offset,
        "pose_rotation_deg_median": 3. + offset,
        "pose_rotation_deg_P90": 8. + offset,
        "pose_yaw_deg_median": 2. + offset,
        "pose_yaw_deg_P90": 7. + offset,
        "pose_IoU3D_median": .6 - offset / 100.,
        "pose_ADDsym_AUC_full": .4 - offset / 100.,
        "pose_coverage": 1.,
        "frames": 319,
        "matched_frames": 311,
        "full_supervised_corners": 2499,
        "observed_corners": 2445,
        "pose_available_frames": 319,
    }


def _corner_row(frame_id="f0", error=1.) -> dict:
    return {
        "id": frame_id, "session": "s0", "material": "plastic",
        "occlusion": "clean", "evaluable": True, "detected": True,
        "matched": True, "E_sym": .01, "E_fixed": .01,
        "frame_mean_px": error, "canonical_errors": [error] * 8,
        "canonical_valid": [True] * 8,
    }


def _method(offset: float, frame_id="f0") -> dict:
    headline = _headline(offset)
    return {
        "status": "COMPLETE", "headline": headline,
        "result": {
            "corner_rows": [_corner_row(frame_id, 1. + offset)],
            "pose": {
                "translation_cm": {"median": 5. + offset, "P90": 9. + offset},
                "rotation_deg": {"median": 3. + offset, "P90": 8. + offset},
                "yaw_deg": {"median": 2. + offset, "P90": 7. + offset},
                "denominators": {"coverage": 1.},
            },
            "pose_rows": [{"id": frame_id, "available": True,
                           "translation_cm": 5. + offset,
                           "rotation_deg": 3. + offset,
                           "yaw_deg": 2. + offset}],
        },
    }


def _evaluation(backbone: str) -> dict:
    def bootstrap(seed: int) -> dict:
        metrics = {}
        for index, key in enumerate((
                "matched_pooled_corner8_median_px",
                "matched_pooled_corner8_P90_px",
                "full_PCK10_fraction", "E_sym")):
            value = 0. if seed == 1 and index == 0 else (-seed - index / 10.)
            metrics[key] = {
                "observed_delta": value,
                "invalid_draws": 0,
                "status": "COMPLETE",
                "CI95": [value - .25, value + .25] if value else [0., 0.],
            }
        return {
            "direction": "candidate_minus_base", "unit": "session",
            "sessions": 13, "resamples": 10000, "seed": 20260917,
            "paired_sampling": True, "recalculates_each_statistic": True,
            "multiplicity_adjusted": False, "metrics": metrics,
        }
    return {
        "schema": "pallet_n3_completion_v3_evaluation_v1",
        "complete": True, "backbone": backbone,
        "methods": {
            "base": _method(0.),
            "n3_seed1": _method(-1.),
            "n3_seed2": _method(-2.),
            "n3_seed3": _method(-3.),
        },
        "comparisons": {
            f"base_to_n3_seed{seed}": {"result": {"bootstrap": bootstrap(seed)}}
            for seed in (1, 2, 3)
        },
        "contract": {"pose_reference": (
            "DEV pose reconstructed from image annotation and registered geometry; "
            "not independent physical 6D measurement")},
    }


def _reuse() -> dict:
    def sub(offset):
        return _headline(offset)

    def seeded(offset):
        return {
            "status": "COMPLETE",
            "per_seed": {
                str(seed): {"status": "COMPLETE", "headline": sub(offset - seed / 10.)}
                for seed in (1, 2, 3)
            },
            "mean": {"headline": sub(offset)},
        }

    material = {name: sub(index) for index, name in enumerate(("plastic", "wood"))}
    occlusion = {name: sub(index) for index, name in enumerate(
        ("clean", "moderate", "severe", "unclassified"))}
    subgroup_methods = {
        key: {"material": material, "occlusion": occlusion}
        for key in ("R0", "OLD_P", "N2_DIM_ONLY", "N3_DIM_SYM")
    }

    def damage(mode: str) -> dict:
        result = {
            "corner_change": {"improved": 10., "no_change": 0., "worsened": 2.},
            "frame_change": {"improved": 5., "no_change": 0., "worsened": 1.},
            "good5_to_bad10_corners": 0., "bad20_to_good10_corners": 3.,
            "movement": {"median_px": .5, "P90_px": 1.,
                         "cap_violations": 0.},
            "triangle_lower_bound_violations": 0.,
            "matched_comparable_frames": 6.,
            "observed_comparable_corners": 12.,
        }
        if mode != "no_output_cap":
            result.update(initial_error_inside_cap_corners=8.,
                          initial_error_outside_cap_corners=4.,
                          cap_hit_corners=0.)
        return result

    def paired(seed: int) -> dict:
        return {
            "paired_with_fixed_branch_2d_frames": 6,
            "translation_cm": {"median_delta": -.1 * seed, "improved": 3,
                               "no_change": 0, "worsened": 3},
            "rotation_deg": {"median_delta": .1 * seed, "improved": 2,
                             "no_change": 0, "worsened": 4},
            "yaw_deg": {"median_delta": 0., "improved": 2,
                        "no_change": 2, "worsened": 2},
            "joint_direction_counts": {
                "2d_improved__translation_improved": 2,
                "2d_improved__translation_worsened": 3,
                "2d_worsened__translation_improved": 1,
                "2d_worsened__translation_worsened": 0,
            },
            "new_pose_failures": 0, "pose_recoveries": 0,
        }

    cap_methods = {}
    for method_index, method in enumerate(("N2_DIM_ONLY", "N3_DIM_SYM")):
        modes = {}
        for mode_index, mode in enumerate(("cap1pct", "cap2pct", "no_output_cap")):
            modes[mode] = {
                "status": "COMPLETE",
                "mean_headline": sub(-1. - method_index - mode_index / 10.),
                "mean_damage_counts": damage(mode),
                "per_seed": {
                    str(seed): {
                        "fixed_base_branch_damage": damage(mode),
                        "paired_2d_pose": paired(seed),
                    } for seed in (1, 2, 3)
                },
            }
        cap_methods[method] = modes

    safe_methods = {
        "R0": {"status": "COMPLETE", "mean": {"headline": sub(0.)}},
        "source_only_update": {"status": "COMPLETE", "mean": {"headline": sub(-.1)}},
        "raw_pseudo_student": {"status": "COMPLETE", "mean": {"headline": sub(-.2)}},
        "corrected_pseudo_student": {"status": "COMPLETE", "mean": {"headline": sub(-.3)}},
        "R0_plus_N3": seeded(-1.),
    }
    return {
        "schema": "pallet_n3_completion_v3_reuse_v1",
        "status": "COMPLETE_WITH_DECLARED_BLOCKS",
        "contracts": {
            "A": {"status": "COMPLETE", "methods": {
                "R0": {"status": "COMPLETE", "mean": {"headline": _headline(0.)}},
                "OLD_P": seeded(-.5), "N2_DIM_ONLY": seeded(-.8),
                "N3_DIM_SYM": seeded(-1.),
            }},
            "B": {"status": "COMPLETE", "methods": {
                "N0_BASE_REPLAY": seeded(-.2), "N1_SYM_ONLY": seeded(-.4),
                "N2_DIM_ONLY": seeded(-.8), "N3_DIM_SYM": seeded(-1.),
            }, "mean_metric_deltas": {
                "N2_minus_N0": {
                    "matched_pooled_corner8_median_px": -.6,
                    "matched_pooled_corner8_P90_px": -.6,
                    "full_PCK10_fraction": .006, "E_sym": -.0006},
                "N1_minus_N0": {
                    "matched_pooled_corner8_median_px": -.2,
                    "matched_pooled_corner8_P90_px": -.2,
                    "full_PCK10_fraction": .002, "E_sym": -.0002},
                "N3_minus_N2": {
                    "matched_pooled_corner8_median_px": .0001,
                    "matched_pooled_corner8_P90_px": -.2,
                    "full_PCK10_fraction": 0., "E_sym": -.0002},
                "N3_minus_N1": {
                    "matched_pooled_corner8_median_px": -.6,
                    "matched_pooled_corner8_P90_px": -.6,
                    "full_PCK10_fraction": .006, "E_sym": -.0006},
            }},
            "C": {"status": "COMPLETE", "frame_counts": {"plastic": 2, "wood": 1},
                  "methods": subgroup_methods},
            "D": {"status": "PARTIAL_LABELS",
                  "frame_counts": {"clean": 1, "moderate": 1, "severe": 1,
                                   "unclassified": 0},
                  "methods": subgroup_methods},
            "E": {"status": "COMPLETE", "methods": cap_methods},
            "H": {"status": "COMPLETE", "methods": {
                "D": seeded(.5), "L": seeded(.2), "PoseFix": seeded(-.2),
            }},
            "I": {
                "status": "PARTIAL_SAFE_COHORT",
                "DEV319_table": {"status": "BLOCKED_CONTRACT",
                                 "reason": "fixture exposure contract absent"},
                "safe_common_cohort": {
                    "status": "COMPLETE_REUSED_DEV", "population": "HELDOUT128",
                    "exposure": {"student_train_RGB_overlap": 0,
                                 "teacher_session_overlap": 0,
                                 "independent_test": False,
                                 "LR5_selection_history": "historical reused DEV"},
                    "methods": safe_methods,
                },
            },
        },
    }


def _square(backbone: str) -> dict:
    mode = _evaluation(backbone)
    mode["population"] = {"sessions": 1}
    return {"schema": "pallet_n3_completion_v3_square_evaluation_v1",
            "complete": True, "backbone": backbone,
            "manual_declared_corners": 602, "manual_in_frame_corners": 600,
            "modes": {"manual_declared": mode}}


def _square_yolo() -> dict:
    return {
        "schema": "pallet_n3_completion_v3_square_yolo_evaluation_v1",
        "complete": True,
        "dimensions_wdh_m": [1.1, 1.1, .15],
        "dimension_effect_identifiable": False,
        "confidence_interval": "x: one correlated capture session; not estimated",
        "pose_3d": {"status": "BLOCKED_REFERENCE"},
        "modes": {"manual_declared": {"families": {
            "R0": {"single": dict(_headline(), total_frames=119, matched=110)},
            "OLD_P": {"mean_of_seed_summaries": _headline(-.5)},
            "N2_DIM_ONLY": {"mean_of_seed_summaries": _headline(-.8)},
            "N3_DIM_SYM": {"mean_of_seed_summaries": _headline(-1.)},
        }, "manual_corner_denominator": 602}},
    }


def _runtime(backbone: str) -> dict:
    def entry(value):
        return {"cuda_event_ms": {"median_ms": value, "p90_ms": value + 1,
                                  "fps_from_median": 1000. / value},
                "peak_allocated_bytes": 1000}
    return {"schema": "pallet_n3_completion_v3_runtime_receipt_v1",
            "complete": True, "backbone": backbone,
            "summary": {"base_e2e": entry(10.), "n3_seed1_e2e": entry(12.),
                        "n3_seed1_only": entry(2.)},
            "parameters": {"base_total": 100, "base_plus_n3_total": 123,
                           "n3_total": 23}}


def _lifter() -> dict:
    def method(available):
        return {"overall": {
            "coverage": {"frames": 3, "available_fraction": available,
                         "fresh_fraction": available},
            "missing": {"longest_until_next_sample_sensor_time_run": {
                "until_next_sample_s": 1.}},
            "yaw_wrap_jitter": {"available_outputs_including_held": {
                "median_abs_step": 1., "p90_abs_step": 2.}},
        }}
    return {"schema": "pallet_n3_completion_v3_lifter_receipt_v1",
            "complete": True, "statistics": {"R0": method(.8), "N3_seed1": method(.9)},
            "actual_control_invoked": False}


def _fit(backbone: str, seed: int) -> dict:
    return {"schema": "pallet_n3_completion_v3_fit_v1", "complete": True,
            "backbone": backbone.lower(), "seed": seed, "smoke": False,
            "steps": 6000, "exposures": 96000, "elapsed_seconds": 10.,
            "trainable_parameters": 23, "dimension_input_to_N3": True,
            "symmetry_supervision": True, "base_receives_dimensions": False,
            "checkpoint": {"sha256": f"{seed:064x}"},
            "history": [{"step": 1, "loss": 2.}, {"step": 6000, "loss": 1.}]}


def _points(value: float) -> list[list[float]]:
    return [[value, 0.] for _ in range(9)]


def _predictions(root: Path) -> tuple[dict, list[dict]]:
    specs = (("a_improve", 10., 1.), ("b_near", 5., 4.9),
             ("c_adverse", 1., 5.), ("d_missing", 2., None))
    frames, truth = [], []
    for index, (frame_id, base, n3) in enumerate(specs):
        image_path = root / f"{frame_id}.png"
        plt.imsave(image_path, np.full((12, 16, 3), index / 5. + .1))
        n3_block = {"points": _points(n3)} if n3 is not None else {"points": None}
        frames.append({"id": frame_id, "session_id": "s0",
                       "image_key": image_path.name,
                       "predictions": {"base": {"points": _points(base)},
                                       "n3_seed1": n3_block}})
        truth.append({"id": frame_id, "session": "s0", "gt": _points(0.),
                      "valid": [True] * 9, "permutations": [list(range(9))]})
    return {"schema": "pallet_n3_completion_v3_prediction_v1",
            "complete": True, "frames": frames}, truth


def _populate(doc: Path, raw: Path, root: Path) -> list[dict]:
    _write(doc / "PROTOCOL.json", {
        "schema": "pallet_n3_completion_v3_protocol_v1",
        "purpose": "fixture estimator-specific N3 application",
        "methods": {
            "base_inputs": ["RGB"],
            "n3_inputs": ["frozen features", "registered physical dimensions [W,D,H]"],
            "n3_inference_forbidden_inputs": ["ground truth", "symmetry branch label"],
            "preserved_outputs": ["score", "point ordering"],
        },
        "n3": {"dimension_features": ["log(W)", "log(D)", "log(H)"]},
        "symmetry_supervision": {
            "source": "approved whole-object permutations",
            "selection_reference": "raw frozen base predictions",
            "inference_uses_symmetry_target": False,
        },
        "data": {"train_supervision": "synthetic only", "new_real_training_images": 0},
        "evaluation": {"seed_aggregation": "per seed; mean statistics; no ensemble"},
        "backbones": {"resnet18": {
            "source_checkpoint": {"path": "fixture_epoch10.pt", "sha256": "a" * 64},
            "adapter": {
            "source_training": "10-epoch synthetic CONSTANT architecture-control arm; this is not the 60-epoch RGB baseline",
            "input": {"checkpoint_selection": "fixed final epoch60; stale source field"},
        }, "deviation": "10-epoch CONSTANT-fold RGB-only base; not 60-epoch pure-RGB"}},
    })
    _write(doc / "ENVIRONMENT_AUDIT.json", {
        "schema": "pallet_n3_completion_v3_environment_audit_v1", "complete": True,
        "roles": {
            "dope_resnet_training_inference_runtime": {
                "executable": "/fixture/pallet-pose/python", "torch": "2.1.1+cu118",
                "torch_cuda": "11.8", "ultralytics": "8.0.120", "has_C3k2": False,
            },
            "yolo26_square_and_offline_lifter": {
                "executable": "/fixture/pallet-yolo26/python", "torch": "2.1.1+cu118",
                "torch_cuda": "11.8", "ultralytics": "8.4.60", "has_C3k2": True,
            },
        },
        "boundary": {"reason": "YOLO26 checkpoint requires C3k2/Pose26 support",
                     "same_GPU_for_new_measurements": "fixture GPU",
                     "yolo_runtime_compared_in_new_fixed26_benchmark": False},
    })
    _write(doc / "SYMMETRY_ACTIVATION_AUDIT.json", {
        "schema": "pallet_n3_completion_v3_symmetry_activation_audit_v1",
        "complete": True,
        "source_sidecar": {
            "valid_permutations_per_row_counts": {
                "0": 0, "1": 20, "2": 40, "3": 0, "4": 0},
            "four_way_rows": 0,
        },
        "backbones": {
            "dope": {
                "unique_usable_rows": {"rows_or_exposures": 44,
                                       "non_identity_count": 1,
                                       "non_identity_fraction": 1 / 44},
                "seeds": {str(seed): {"rows_or_exposures": 96,
                                      "non_identity_count": seed}
                          for seed in (1, 2, 3)},
            },
            "resnet18": {
                "unique_usable_rows": {"rows_or_exposures": 55,
                                       "non_identity_count": 0,
                                       "non_identity_fraction": 0.},
                "seeds": {str(seed): {"rows_or_exposures": 96,
                                      "non_identity_count": 0}
                          for seed in (1, 2, 3)},
            },
        },
        "interpretation": {"claim_boundary": "same objective; activation differs"},
    })
    dimension_fits = {}
    for backbone in ("dope", "resnet18"):
        for seed in (1, 2, 3):
            dimension_fits[f"{backbone}_seed{seed}"] = {
                "status": "PASS", "backbone": backbone, "seed": seed,
                "visual_inputs_bitwise_identical": True,
                "base_logits_bitwise_identical": True,
                "changed_logit_entries": 1776,
                "max_abs_logit_delta": float(seed),
                "checkpoint_sha256": f"{seed:064x}",
            }
    _write(doc / "DIMENSION_SENSITIVITY_AUDIT.json", {
        "schema": "pallet_n3_completion_v3_dimension_sensitivity_audit_v1",
        "complete": True, "dimension_a_wdh_m": [1.1, 1.3, .11],
        "dimension_b_wdh_m": [.8, .59, .14], "fits": dimension_fits,
        "summary": {"status": "PASS", "fits_verified": 6,
                    "total_changed_logit_entries": 10656},
        "claim_boundary": "active path only; no causal accuracy attribution",
    })
    _write(doc / "REUSE_RESULTS.json", _reuse())
    _write(doc / "VERIFY_RESULTS.json", {
        "schema": "pallet_n3_completion_v3_verify_v1",
        "status": "PARTIAL_WITH_DECLARED_X", "integrity_status": "PASS",
        "complete": False,
    })
    reuse_frame = {"core": {}, "cap": {
        "E_N3_DIM_SYM_seed1_cap1pct": {
            "fixed_branch": [{"id": "f0", "delta_px": -1.}],
            "pose_scores": [{"id": "f0", "available": True,
                             "translation_cm": .5, "rotation_deg": 1., "yaw_deg": 1.}],
        },
    }}
    for method in ("R0", "N3_DIM_SYM_seed1", "N3_DIM_SYM_seed2", "N3_DIM_SYM_seed3"):
        reuse_frame["core"][method] = {
            "corner_scores": [_corner_row()],
            "pose_scores": [{"id": "f0", "available": True,
                             "translation_cm": 1., "rotation_deg": 2., "yaw_deg": 3.}],
        }
    _write(raw / "reuse" / "PER_FRAME_SCORES.json", reuse_frame)
    _write(raw / "evaluation" / "dope.json", _evaluation("dope"))
    _write(raw / "evaluation" / "resnet18.json", _evaluation("resnet18"))
    _write(doc / "SQUARE_YOLO_RESULTS.json", _square_yolo())
    _write(doc / "SQUARE_DOPE_RESULTS.json", _square("dope"))
    _write(doc / "SQUARE_RESNET18_RESULTS.json", _square("resnet18"))
    for backbone in ("DOPE", "RESNET18"):
        for seed in (1, 2, 3):
            _write(doc / f"TRAIN_{backbone}_SEED{seed}.json", _fit(backbone, seed))
        _write(doc / f"TRAINING_{backbone}_COMPLETE.json", {"complete": True})
        _write(doc / f"VALIDATION_{backbone}_COMPLETE.json", {"complete": True})
        _write(doc / f"SELECTION_{backbone}.json", {
            "schema": "pallet_n3_completion_v3_selection_v1", "complete": True,
            "temperatures": {"1": .5, "2": 1., "3": 2.},
            "fixed_rule": {"lam": 1., "max_move_image_diagonal_fraction": .01},
            "real_selection": False,
        })
    _write(doc / "RUNTIME_DOPE_SEED1.json", _runtime("dope"))
    _write(doc / "RUNTIME_RESNET18_SEED1.json", _runtime("resnet18"))
    _write(raw / "runtime" / "dope_seed1.json", {"complete": True})
    _write(raw / "runtime" / "resnet18_seed1.json", {"complete": True})
    lifter = _lifter()
    _write(doc / "LIFTER_YOLO_R0_N3_SEED1_COMPLETE.json", lifter)
    _write(raw / "lifter" / "YOLO_R0_N3_SEED1_RAW.json", {
        "complete": True, "sessions": {"s0": {"frames": [{
            "session_id": "s0", "camera_sensor_timestamp_ms": 0.,
            "methods": {"R0": {"available": True, "pos_x_m": 0.,
                                  "pos_z_m": 1., "yaw_deg": 1.},
                        "N3_seed1": {"available": True, "pos_x_m": .1,
                                     "pos_z_m": .9, "yaw_deg": 2.}}}]}}})
    predictions, truth = _predictions(root)
    _write(raw / "predictions" / "dope_DEV319.json", predictions)
    _write(raw / "predictions" / "resnet18_DEV319.json", predictions)
    return truth


def test_cell_keeps_zero_distinct_from_x_and_na():
    assert R.cell(0) == {"value": 0, "status": "COMPLETE", "display": "0", "reason": None}
    assert R.cell(status="MISSING")["display"] == "x"
    assert R.cell(status="NA")["display"] == "NA"


def test_seed_mean_reads_pose_yaw_from_result():
    status, values = R._seed_mean(_evaluation("dope"))
    assert status == "COMPLETE"
    assert values["matched_pooled_corner8_median_px"] == 8.
    assert values["pose_yaw_deg_median"] == 0.


def test_overlay_selection_is_deterministic_and_declared_extreme(tmp_path):
    payload, truth = _predictions(tmp_path)
    rows = R.overlay_records(payload, truth)
    selected = R.select_overlay_cases(rows)
    assert selected["improvement"]["id"] == "a_improve"
    assert selected["near_no_change"]["id"] == "b_near"
    assert selected["adverse"]["id"] == "c_adverse"
    assert selected["missing"]["id"] == "d_missing"


def test_normalized_metrics_are_actual_rows_only():
    inputs = {
        "reuse_per_frame": {"payload": {"core": {
            "R0": {"corner_scores": [_corner_row()], "pose_scores": []}}}},
        "eval_dope": {"payload": _evaluation("dope")},
    }
    frames, corners = R.normalized_metric_rows(inputs)
    assert len(frames) == 5  # one YOLO plus base/three DOPE methods
    assert len(corners) == 5 * 8
    assert {row["backbone"] for row in frames} == {"YOLO", "DOPE"}


def test_generate_with_missing_inputs_keeps_partial_and_emits_every_artifact(tmp_path):
    doc, raw, output = tmp_path / "doc", tmp_path / "raw", tmp_path / "out"
    result = R.generate(doc=doc, raw=raw, output_doc=output,
                        output_raw=raw, root=tmp_path, truth_rows=[])
    assert result["tables"]["overall_status"] == "PARTIAL"
    assert result["tables"]["overall_complete"] is False
    assert result["tables"]["execution_complete"] is False
    for filename in ("TABLES.json", "TABLES.md", "FINAL_REPORT_KO.md", "REMAINING_X.md"):
        assert (output / filename).is_file()
    for filename in (*R.FIGURE_FILES, "SELECTION_MANIFEST.json"):
        assert (output / "figures" / filename).is_file()
    for filename in ("backbone_dev_headline.tex", "square_2d.tex", "runtime.tex"):
        assert (output / "table_fragments" / filename).is_file()
    assert (raw / "report" / "FRAME_METRICS.csv").is_file()
    assert "independent_TEST" in (output / "REMAINING_X.md").read_text()


def test_generate_complete_executable_fixture_still_reports_protocol_partial(tmp_path):
    doc, raw, output = tmp_path / "doc", tmp_path / "raw", tmp_path / "out"
    truth = _populate(doc, raw, tmp_path)
    result = R.generate(doc=doc, raw=raw, output_doc=output,
                        output_raw=raw, root=tmp_path, truth_rows=truth)
    assert result["tables"]["execution_complete"] is True
    assert "verify" not in result["tables"]["inventory"]
    assert result["tables"]["post_generation_integrity"] == {
        "status": "SEPARATE_POST_GENERATION_AUDIT",
        "artifact": "_docs/experiments/pallet_n3_completion_v3/VERIFY_RESULTS.json",
        "hash_bound_in_report": False,
        "reason": (
            "VERIFY_RESULTS binds the generated report, so the report cannot in turn "
            "bind VERIFY_RESULTS without a circular content-hash dependency."),
    }
    assert result["tables"]["overall_status"] == "PARTIAL"
    assert result["manifest"]["inference_GT_input"] is False
    overlay = next(item for item in result["manifest"]["figures"]
                   if item["file"] == "dev_overlays.png")
    assert overlay["actual_overlays"] == 8
    assert {(case["backbone"], case["category"]) for case in overlay["cases"]} == {
        (backbone, category)
        for backbone in ("DOPE", "ResNet-18")
        for category in ("improvement", "near_no_change", "adverse", "missing")
    }
    assert "not representative" in overlay["extreme_selection_disclosure"]
    lifter = next(item for item in result["manifest"]["figures"]
                  if item["file"] == "lifter.png")
    assert lifter["actual_panels"] == 6
    subgroup = next(item for item in result["manifest"]["figures"]
                    if item["file"] == "subgroups_or_thresholds.png")
    assert subgroup["actual_panels"] == 4
    assert subgroup["paired_scatter_points"] == 1
    report = (output / "FINAL_REPORT_KO.md").read_text()
    assert "GT를 입력하지 않았다" in report
    assert "OVERALL_COMPLETE" in report
    assert "10-epoch CONSTANT-fold RGB-only" in report
    assert "60-epoch pure-RGB" in report
    assert "같은 N3 가중치를 옮긴 실험" in report
    assert "2D frame 오차 감소는 T·R·yaw 동시 감소를 뜻하지 않는다" in report
    assert "whole-package 비교" in report
    assert "HELDOUT128" in report and "새 self-training을 수행하지 않았" in report
    assert "Ultralytics `8.4.60`" in report and "C3k2" in report
    assert "non-identity target이 활성화되었다" in report
    tables_hash = R._sha256(output / "TABLES.json")
    assert tables_hash in (output / "table_fragments" / "runtime.tex").read_text()
    with (raw / "report" / "FRAME_METRICS.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12  # YOLO 4 methods + 2 backbones x 4 methods
    assert {row["backbone"] for row in rows} == {"YOLO", "DOPE", "ResNet-18"}


def test_extended_tables_keep_seeds_ci_denominators_and_declared_x(tmp_path):
    doc, raw, output = tmp_path / "doc", tmp_path / "raw", tmp_path / "out"
    truth = _populate(doc, raw, tmp_path)
    result = R.generate(doc=doc, raw=raw, output_doc=output,
                        output_raw=raw, root=tmp_path, truth_rows=truth)
    tables = result["tables"]["tables"]
    assert len(tables["yolo_cap_damage"]["rows"]) == 6
    assert len(tables["yolo_paired_2d_pose"]["rows"]) == 6
    assert len(tables["backbone_dev_per_seed"]["rows"]) == 8
    assert len(tables["backbone_paired_ci"]["rows"]) == 6
    assert [row["params"]["value"] for row in tables["runtime"]["rows"][:3]] == [
        100, 123, 23]
    assert result["tables"]["square_contract"]["manual_declared_corners"] == 602
    assert result["tables"]["square_contract"]["manual_in_frame_corners"] == 600
    assert result["tables"]["square_contract"]["sessions"] == 1
    assert {row["contrast"]["value"]
            for row in tables["yolo_ablation_deltas"]["rows"]} == {
        "N2 - N0", "N1 - N0", "N3 - N2", "N3 - N1",
    }
    yolo_square = [row for row in tables["square_green0918"]["rows"]
                   if row["backbone"]["value"] == "YOLO"]
    assert [(row["method"]["value"], row["seed_count"]["value"])
            for row in yolo_square] == [
        ("Base", 1), ("OLD_P seed mean", 3),
        ("N2 seed mean", 3), ("N3 seed mean", 3),
    ]
    for row in tables["square_green0918"]["rows"]:
        for key in ("t_median_cm", "r_median_deg", "yaw_median_deg",
                    "t_p90_cm", "r_p90_deg", "yaw_p90_deg",
                    "iou3d_median", "addsym_auc", "pose_available_frames"):
            assert row[key]["status"] == "BLOCKED_REFERENCE"
    no_cap = [row for row in tables["yolo_cap_damage"]["rows"]
              if row["cap"]["value"] == "none"]
    assert all(row["cap_hit_corners"]["display"] == "NA" for row in no_cap)
    assert all(row["no_change_corners"]["value"] == 0. for row in no_cap)
    first_ci = tables["backbone_paired_ci"]["rows"][0]
    assert first_ci["median_delta"]["value"] == 0.
    assert first_ci["median_ci95"]["value"] == [0., 0.]
    assert first_ci["pck10_pp_delta"]["value"] == -120.
    safe = [row for row in tables["update_alternatives"]["rows"]
            if row["population"]["value"] == "HELDOUT128"]
    assert all(row["student_rgb_overlap"]["value"] == 0 for row in safe)
    blocked = [row for row in tables["update_alternatives"]["rows"]
               if row["population"]["value"] == "DEV319"]
    assert len(blocked) == 3
    assert all(row["median_px"]["display"] == "x" for row in blocked)
    symmetry = tables["symmetry_activation"]["rows"]
    assert symmetry[0]["unique_non_identity"]["value"] == 1
    assert symmetry[1]["unique_non_identity"]["value"] == 0
    sensitivity = tables["dimension_sensitivity"]["rows"]
    assert len(sensitivity) == 6
    assert all(row["changed_logits"]["value"] == 1776 for row in sensitivity)
    assert all(row["base_logits_identical"]["value"] is True for row in sensitivity)


def test_report_verify_repetition_has_no_content_hash_cycle(tmp_path):
    doc = tmp_path / "_docs/experiments/pallet_n3_completion_v3"
    raw = tmp_path / "data/pallet/results/pallet_n3_completion_v3"
    truth = _populate(doc, raw, tmp_path)
    _write(doc / "PROTOCOL.json", {"bindings": []})
    _write(
        tmp_path / "_docs/experiments/pallet_dim_conditioned_p_v1/"
                   "DIM_NORMALIZATION_LOCK.json",
        {"mean": [0.] * 5, "scale": [1.] * 5},
    )

    report_specs = tuple(
        row for row in I.ARTIFACT_SPECS if row[2] == "report")
    inspect_artifacts = I.inspect_artifacts

    def inspect_only_report(root):
        return inspect_artifacts(root, specs=report_specs)

    def local_write(path, value, *, freeze=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(value, ensure_ascii=False, indent=2,
                          allow_nan=False) + "\n"
        if freeze and path.exists():
            assert path.read_text() == text
            return
        path.write_text(text)

    passed = {"status": "PASS"}
    training = {"status": "PASS"}
    patches = (
        patch.object(I, "validate_protocol_payload", return_value=dict(passed)),
        patch.object(I, "inspect_training", return_value=(training, {}, {})),
        patch.object(I, "inspect_sensitivity", return_value=dict(passed)),
        patch.object(I, "prove_base_rgb_only", return_value=dict(passed)),
        patch.object(I, "inspect_artifacts", side_effect=inspect_only_report),
        patch.object(I.C, "write", side_effect=local_write),
    )
    for context in patches:
        context.start()
    try:
        # Exercise the formerly cyclic order explicitly.  A stale fixture
        # VERIFY_RESULTS already exists from _populate(), but neither report
        # generation is allowed to read or bind it.
        R.generate(doc=doc, raw=raw, output_doc=doc, output_raw=raw,
                   root=tmp_path, truth_rows=truth)
        first = I.verify(tmp_path, doc / "VERIFY_RESULTS.json")
        R.generate(doc=doc, raw=raw, output_doc=doc, output_raw=raw,
                   root=tmp_path, truth_rows=truth)
        tables = json.loads((doc / "TABLES.json").read_text())
        assert "verify" not in tables["inventory"]
        second = I.verify(tmp_path, doc / "VERIFY_RESULTS.json")
        stable = (doc / "VERIFY_RESULTS.json").read_bytes()
        fresh = I.verify(tmp_path, doc / "VERIFY_RESULTS.json")
    finally:
        for context in reversed(patches):
            context.stop()

    assert first["integrity_status"] == "PASS"
    assert second["integrity_status"] == "PASS"
    assert fresh["integrity_status"] == "PASS"
    assert (doc / "VERIFY_RESULTS.json").read_bytes() == stable


def test_markdown_zero_does_not_become_na_or_x():
    spec = {"title": "zero", "columns": [{"key": "v", "label": "Value"}],
            "rows": [{"v": R.cell(0.)}]}
    rendered = R._markdown_table(spec)
    assert "| 0 |" in rendered
    assert "NA" not in rendered and "| x |" not in rendered
