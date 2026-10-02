"""Build and seal the experiment protocol only after FULL finishes and code freezes."""
from __future__ import annotations

from pathlib import Path

from . import common as C
from .dimensions import FEATURE_NAMES, load_normalization


PRODUCTION_FILES = (
    "common.py", "dimensions.py", "refiner.py", "data.py", "resume.py",
    # These execution stages are deliberately required before sealing. They do
    # not exist in the foundation-only phase, so training cannot start early.
    "full_adapter.py", "source_cache.py", "train.py", "selection.py",
    "inference.py", "evaluation.py",
)

EVALUATION_DEPENDENCIES = (
    C.ROOT / "scripts/research/pallet_resnet18_refiner_20261001_v1/evaluation.py",
    C.ROOT / "scripts/research/pallet_sensors_refinement_closeout_v1/paired_stats_helper.py",
    C.POSE_CODE / "run_pose_evaluation.py",
    C.POSE_CODE / "evaluate_pose_by_session.py",
    C.POSE_CODE / "pose_evaluation_paths.py",
    C.POSE_CODE / "symmetry_aware_pose_metrics.py",
    C.ROOT / "challenge/evaluation_v2/paper_real_eval.py",
    C.ROOT / "challenge/evaluation_v2/pnp_selector.py",
    C.ROOT / "challenge/evaluation_v2/oriented_iou3d.py",
    C.POSE_RAW / "AXIS_REVIEW_MANIFEST.json",
    C.POSE_RAW / "POSE_EVAL_OBJECT_CONTRACT.json",
    C.POSE_RAW / "GEOMETRY_RESOLVED_POSE_GT.json",
    C.DEV_NEG,
    C.ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json",
)


def missing_production_files() -> list[str]:
    return [name for name in PRODUCTION_FILES if not (C.HERE / name).is_file()]


def validate_full_completion(path: Path | str = C.FULL_COMPLETE) -> dict:
    completion_path = Path(path)
    if not completion_path.is_file():
        raise RuntimeError("Frozen FULL training has not completed; protocol cannot be sealed")
    completion = C.read_json(completion_path)
    expected = dict(complete=True, arm="FULL", epochs=10, updates=34990,
                    images_seen=559800, real_training_images=0, final_epoch_fixed=True)
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(f"FULL completion contract mismatch for {key}")
    protocol_path = C.verify_binding(completion["protocol"])
    if protocol_path != C.FULL_PROTOCOL.resolve():
        raise ValueError("FULL completion points to an unexpected protocol")
    protocol = C.read_json(protocol_path)
    if protocol.get("schema") != "resnet18_dimension_dsnt_full_source_train_v1":
        raise ValueError("Unexpected FULL protocol schema")
    if protocol.get("epochs") != 10 or protocol.get("total_updates_per_arm") != 34990:
        raise ValueError("FULL source budget drift")
    if protocol.get("real_training_images") != 0 or protocol.get("validation", "").find("fixed epoch10") < 0:
        raise ValueError("FULL selection/training contract drift")
    checkpoint_path = C.verify_binding(completion["final_checkpoint"])
    return dict(completion=C.binding(completion_path), protocol=C.binding(protocol_path),
                checkpoint=C.binding(checkpoint_path))


def build_protocol(baseline: dict, code_bindings: list[dict]) -> dict:
    normalization = load_normalization()
    return dict(
        schema="resnet18_full_dim_local_refiner_protocol_v1",
        purpose=("Frozen FULL ResNet correction transfer with visual D0/P0, dimension P5, "
                 "and same-architecture zero-context P5_CONSTANT."),
        status="SEALED_BEFORE_SOURCE_CACHE_OR_HEAD_TRAINING",
        baseline=baseline["checkpoint"],
        baseline_protocol=baseline["protocol"],
        baseline_training_complete=baseline["completion"],
        baseline_trainable=False,
        baseline_arm="FULL",
        baseline_decoder="Nine-point DSNT spatial-softmax expectation; no argmax confidence threshold.",
        feature_taps=dict(names=["layer2", "layer3"], channels=[128, 256], strides=[8, 16],
                          dimension_context_enters_taps=False),
        arms=list(C.HEAD_ARMS), seeds=list(C.SEEDS), steps=C.STEPS, batch=C.BATCH,
        fits=len(C.HEAD_ARMS) * len(C.SEEDS),
        head_config=C.HEAD_CONFIG,
        inputs=dict(
            images="Frozen FULL RGB preprocessing; no detector or GT crop.",
            dimension_features=list(FEATURE_NAMES),
            dimension_normalization=C.binding(C.DIMENSION_NORMALIZATION),
            dimension_source=C.binding(C.DIMENSION_SIDECAR),
            P5="Canonical fixed-axis normalized 5D supplied to the head.",
            P5_CONSTANT="Identical P5 architecture with context forced to zero inside forward.",
            P0_D0="No direct dimension vector at the correction head.",
            camera_intrinsics_input=False, GT_pose_input=False, GT_keypoints_inference_input=False,
        ),
        training=dict(
            source_manifest=C.binding(C.SOURCE_MANIFEST),
            partitions=dict(train=55980, calibration=1004, selection=1031, heldout=1985),
            real_training_images=0,
            usable_filter=("TRAIN only: frozen base hull IoU>=0.5 with sole source GT and at least "
                           "one jointly valid corner; all source rows remain audited."),
            order="One deterministic seed order shared by D0/P0/P5/P5_CONSTANT.",
            exposures_per_fit=C.STEPS * C.BATCH,
            optimizer=dict(name="AdamW", lr=0.001, weight_decay=0.0001,
                           betas=[0.9, 0.999], warmup_steps=100,
                           cosine_final_lr_fraction=0.1, gradient_clip_norm=5.0),
            final_checkpoint_only=True,
            frozen_prefix_online=True,
            no_full_spatial_feature_cache=True,
            exact_resume=("All four model/optimizer states, Python/NumPy/Torch/CUDA RNG, step, "
                          "history and protocol/baseline/order hashes are atomically replaced "
                          "after every completed update; only the rolling final file is retained."),
        ),
        calibration=dict(population="synthetic calibration only", temperature_grid=[0.5, 1.0, 2.0, 4.0],
                         D_temperature=1.0, real_selection=False),
        selection=dict(population="synthetic selection only",
                       lambda_grid=[0.0, 0.0625, 0.25, 1.0, 4.0],
                       max_move_image_diagonal_fractions=[None, 0.01],
                       objective="Mean frame original-diagonal-normalized 8-corner error, mean over seeds.",
                       heldout_not_opened_until_rules_frozen=True, real_selection=False),
        evaluation=dict(
            DEV319="Reused DEV: 319 frames, 13 sessions, 2818 supervised landmarks.",
            primary="Within-estimator before/after; full-GT PCK and conditional all9-finite median/P90.",
            subgroups="Plastic194 and wood125 are reported separately as well as together.",
            dimension_support="Canonical real W,D,H and normalized 5D are checked against source TRAIN support.",
            pose=("Existing geometry-reconstructed reference; coverage and failures retained; "
                  "ADD AUC uses per-frame normalized ADD and reports conditional and full-population values."),
            bootstrap=dict(draws=10000, seed=20260914, unit="session", exploratory=True),
            square_sets="2D transfer only; constant dimensions cannot identify dimension causality.",
            independent_TEST=False, physical_6D_ground_truth=False,
        ),
        interpretation=dict(
            all_arms_share_dimension_conditioned_FULL_baseline=True,
            P5_minus_P5_CONSTANT=("Causal contrast for the incremental five-value head input, "
                                  "conditional on the already dimension-conditioned FULL baseline."),
            P5_minus_P0=("Package contrast that also changes head capacity/path; not a pure "
                         "dimension-input causal estimate."),
            FULL_alone_cannot_establish_dimension_benefit=True,
        ),
        invariants=dict(center8_preserved=True, base_box_preserved=True, base_score_preserved=True,
                        instance_preserved=True, missing_mask_preserved=True,
                        corners_modified=list(range(8))),
        normalization_summary=dict(rows=normalization["rows"], input=normalization["input"]),
        bindings=code_bindings,
    )


def assert_preseal_output_absence(doc: Path | str | None = None,
                                  raw: Path | str | None = None,
                                  protocol: Path | str | None = None) -> None:
    """Reject any production artifact before the first protocol seal.

    Once a protocol already exists, ``seal`` is an idempotent verification
    operation and production outputs may legitimately be present.
    """
    doc = C.DOC if doc is None else Path(doc)
    raw = C.RAW if raw is None else Path(raw)
    protocol_path = C.PROTOCOL if protocol is None else Path(protocol)
    if protocol_path.is_file():
        return
    if protocol_path.exists():
        raise RuntimeError(f"Protocol path exists but is not a file: {protocol_path}")
    found = []
    for root in (doc, raw):
        if root.exists():
            found.extend(path for path in root.rglob("*") if path.is_file())
    if found:
        labels = ", ".join(str(path) for path in sorted(found)[:10])
        raise RuntimeError("Production artifacts exist before protocol seal: " + labels)


def seal() -> dict:
    missing = missing_production_files()
    if missing:
        raise RuntimeError("Foundation phase only; missing production stages: " + ", ".join(missing))
    if C.PROTOCOL.exists():
        return C.verify_protocol()
    assert_preseal_output_absence()
    baseline = validate_full_completion()
    files = [C.HERE / name for name in PRODUCTION_FILES]
    files += [C.FULL_PROTOCOL, C.SOURCE_MANIFEST, C.DIMENSION_SIDECAR,
              C.DIMENSION_NORMALIZATION, C.DEV_POS, C.GEOMETRY_REGISTRY]
    files += list(EVALUATION_DEPENDENCIES)
    payload = build_protocol(baseline, [C.binding(path) for path in files])
    C.write_frozen_json(C.DOC / "PROTOCOL.json", payload)
    return payload


if __name__ == "__main__":
    seal()
