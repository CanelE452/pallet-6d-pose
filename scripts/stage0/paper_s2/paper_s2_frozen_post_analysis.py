#!/usr/bin/env python3
"""Post-analysis for a completed PAPER_S2 frozen-diagnostic run.

The script is intentionally separate from ``paper_s2_frozen_diagnostic.py``.
It never runs inference, PnP, training, checkpoint selection, or bootstrap
resampling. It consumes the frozen CSV/JSON outputs and creates only these new
artifacts inside the supplied run directory:

* analysis_summary.json
* frozen_tables.csv
* frozen_tables.md
* covariance_ellipse_examples.png
* confidence_vs_error.png
* metric_ci_barplot.png
* flip_reliability.png

Validity rules:

* strict_filterval (outside44 + night43, full N=87) is primary.
* exploratory_pl_pool_manual (N=36) is descriptive/exploratory only.
* synthetic_fixed_val (N=500) is order-free/channel-agnostic only.
* The 10,000-replicate confidence intervals are copied from summary.json.
  They are never recomputed here.
* Training/covariance-weighting ablations remain explicitly BLOCKED.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_NAMES = (
    "analysis_summary.json",
    "frozen_tables.csv",
    "frozen_tables.md",
    "covariance_ellipse_examples.png",
    "confidence_vs_error.png",
    "metric_ci_barplot.png",
    "flip_reliability.png",
)
REQUIRED_INPUTS = (
    "manifest.json",
    "summary.json",
    "frames.csv",
    "keypoints.csv",
    "yaw_ladder.csv",
    "keypoint_influence.csv",
    "kp5_geometry.csv",
    "solver_comparison.csv",
    "flip_consistency.csv",
    "flip_keypoints.csv",
)
STRICT = "strict_filterval"
EXPLORATORY = "exploratory_pl_pool_manual"
SYNTHETIC = "synthetic_fixed_val"
DIRECT_SOLVERS = (
    "EPnP",
    "EPnP+RANSAC",
    "SQPNP",
    "SQPNP+RefineLM",
    "ITERATIVE",
)
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
FROZEN_DIAGNOSTIC_SHA256 = (
    "58ba873be26eb9b66af817ec9bce277d864ffe11472ca3230f20bd0e965b704b"
)
LOCKED_DATASET_HASHES = {
    "q1_list_sha256_locked": (
        "5a88384f045faf22dda48465b440e69dba78bc94420f10a3db5217390befb56d"
    ),
    "strict_filterval_identity_sha256_locked": (
        "2795991dbf7f2c3dcc45132ea18a048a1373893aa0023b9e4bbc81266c1123dd"
    ),
    "exploratory_manual36_identity_sha256_locked": (
        "1d8c8998623258c8ca90a3dd5c47eb4c49c6136d17b6dadfa49df9357dcb3f4b"
    ),
    "legacy_filterval123_identity_sha256_locked": (
        "ee5f766347bd1bf33ceec899c7d167a33bc5e4f0cc4680e860cb78a9efc68766"
    ),
}
PROHIBITED_PATH_PARTS = {
    "final-test",
    "final_test",
    "handannot17",
    "capturenight08",
    "capturenight09",
    "capturepallet07",
    "capturepallet09",
}
TABLE_COLUMNS = (
    "section",
    "validity_scope",
    "split_role",
    "domain",
    "source_session",
    "slice_type",
    "slice_value",
    "keypoint",
    "variant",
    "input_kind",
    "solver",
    "comparison",
    "metric",
    "statistic",
    "value",
    "ci95_low",
    "ci95_high",
    "n_total",
    "n_conditioned",
    "n_success",
    "n_failure",
    "conditioning",
    "ci_method",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create frozen paper tables/figures from a completed diagnostic run"
    )
    parser.add_argument("run_dir", type=Path, help="completed frozen diagnostic run")
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="allow a smoke_complete input for schema/plot validation",
    )
    return parser.parse_args()


def finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def boolean(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(jsonable(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if jsonable(row.get(key)) is None
                        else jsonable(row.get(key))
                    )
                    for key in TABLE_COLUMNS
                }
            )


def write_text_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def validate_run(
    run_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    data: dict[str, list[dict[str, str]]],
    allow_smoke: bool,
) -> dict[str, Any]:
    path_parts = {part.lower() for part in run_dir.parts}
    prohibited_parts = sorted(path_parts & PROHIBITED_PATH_PARTS)
    if prohibited_parts:
        raise RuntimeError(
            "post-analysis refuses a sealed/final-test run directory: "
            + ", ".join(prohibited_parts)
        )
    for name in REQUIRED_INPUTS:
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"required frozen input missing: {run_dir / name}")
    existing = [name for name in OUTPUT_NAMES if (run_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "post-analysis refuses to overwrite existing artifacts: "
            + ", ".join(existing)
        )

    manifest_status = str(manifest.get("run_status"))
    summary_status = str(summary.get("run_status"))
    if manifest_status != summary_status:
        raise RuntimeError(
            f"manifest/summary status mismatch: {manifest_status} != {summary_status}"
        )
    status = summary_status
    source = manifest.get("source", {})
    if source.get("script_sha256") != FROZEN_DIAGNOSTIC_SHA256:
        raise RuntimeError("unexpected frozen diagnostic source SHA256")
    if manifest.get("frozen_inference") is not True:
        raise RuntimeError("manifest does not assert frozen_inference=true")
    if manifest.get("training_or_checkpoint_selection_performed") is not False:
        raise RuntimeError("training/checkpoint selection occurred in source run")
    manifest_checkpoint = manifest.get("checkpoint", {}).get("sha256")
    if not manifest_checkpoint or summary.get("checkpoint_sha256") != manifest_checkpoint:
        raise RuntimeError("manifest/summary checkpoint SHA256 mismatch")

    access = manifest.get("input_access_audit", {})
    access_counters = {
        name: access.get(name)
        for name in (
            "prohibited_path_open_attempt_count",
            "final_test_input_open_count",
            "handannot17_input_open_count",
        )
    }
    if any(value != 0 for value in access_counters.values()):
        raise RuntimeError(f"sealed-input access audit is not all zero: {access_counters}")
    contract = manifest.get("dataset_contract", {})
    hash_mismatches = {
        key: contract.get(key)
        for key, expected in LOCKED_DATASET_HASHES.items()
        if contract.get(key) != expected
    }
    if hash_mismatches:
        raise RuntimeError(f"locked dataset identity mismatch: {hash_mismatches}")

    bootstrap = summary.get("bootstrap_95ci")
    if not isinstance(bootstrap, dict):
        raise RuntimeError("summary.bootstrap_95ci is missing")
    if (
        bootstrap.get("replicates") != 10000
        or finite(bootstrap.get("confidence_level")) != 0.95
        or "strict_filterval" not in str(bootstrap.get("scope"))
    ):
        raise RuntimeError("frozen bootstrap contract is not strict/95%/10,000")
    for family in ("ladder", "solver"):
        comparisons = bootstrap.get(family)
        if not isinstance(comparisons, dict) or not comparisons:
            raise RuntimeError(f"bootstrap family missing or empty: {family}")
        for comparison, metrics in comparisons.items():
            if not isinstance(metrics, dict):
                raise RuntimeError(f"invalid bootstrap comparison: {family}:{comparison}")
            for metric, result in metrics.items():
                if metric == "success_outcomes":
                    continue
                if not isinstance(result, dict) or result.get("replicates") != 10000:
                    raise RuntimeError(
                        "bootstrap result is not a frozen 10,000-replicate result: "
                        f"{family}:{comparison}:{metric}"
                    )

    frames = data["frames.csv"]
    errors = [row for row in frames if row.get("status") != "ok"]
    role_counts = {
        role: sum(row.get("split_role") == role for row in frames)
        for role in (STRICT, EXPLORATORY, SYNTHETIC)
    }
    frame_ids = [row.get("frame_uid") for row in frames]
    if any(not frame_id for frame_id in frame_ids) or len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError("frames.csv has blank or duplicate frame_uid values")
    role_by_frame = {row["frame_uid"]: row.get("split_role") for row in frames}
    for name, rows in data.items():
        for row in rows:
            frame_uid = row.get("frame_uid")
            if (
                frame_uid not in role_by_frame
                or row.get("split_role") != role_by_frame[frame_uid]
            ):
                raise RuntimeError(
                    f"{name} contains an unknown or role-mismatched frame: {frame_uid}"
                )
    if errors:
        raise RuntimeError(f"frozen run has {len(errors)} frame errors")

    is_full = status == "complete" and bool(summary.get("full_requested"))
    if is_full:
        expected = {STRICT: 87, EXPLORATORY: 36, SYNTHETIC: 500}
        if role_counts != expected or len(frames) != 623:
            raise RuntimeError(
                f"full-run completeness mismatch: counts={role_counts}, n={len(frames)}"
            )
        domain_counts = Counter(
            (row.get("split_role"), row.get("domain")) for row in frames
        )
        expected_domain_counts = {
            (STRICT, "outside"): 44,
            (STRICT, "night"): 43,
            (EXPLORATORY, "manual"): 36,
            (SYNTHETIC, "synthetic"): 500,
        }
        if dict(domain_counts) != expected_domain_counts:
            raise RuntimeError(
                f"full-run domain composition mismatch: {dict(domain_counts)}"
            )
        manual_sessions = {
            row.get("source_session")
            for row in frames
            if row.get("split_role") == EXPLORATORY
        }
        if manual_sessions != {"capturepallet11"}:
            raise RuntimeError(
                f"manual36 provenance mismatch: sessions={manual_sessions}"
            )
        manifest_run_dir = Path(str(manifest.get("run_dir", ""))).resolve()
        if manifest_run_dir != run_dir:
            raise RuntimeError(
                f"manifest run_dir mismatch: {manifest_run_dir} != {run_dir}"
            )
    elif not allow_smoke:
        raise RuntimeError(
            "post-analysis requires run_status=complete full N=623; "
            "use --allow-smoke only for schema validation"
        )
    elif status != "smoke_complete":
        raise RuntimeError(f"unsupported smoke status: {status}")
    return {
        "run_status": status,
        "is_full": is_full,
        "role_counts": role_counts,
        "frame_count": len(frames),
        "frame_error_count": len(errors),
        "sealed_access_audit": access_counters,
        "frozen_diagnostic_sha256": FROZEN_DIAGNOSTIC_SHA256,
        "locked_dataset_hashes_verified": True,
        "bootstrap_contract_verified": True,
    }


def validity_for(role: str) -> str:
    if role == STRICT:
        return "PRIMARY_STRICT_N87"
    if role == EXPLORATORY:
        return "EXPLORATORY_PL_POOL_MANUAL_N36"
    if role == SYNTHETIC:
        return "SYNTHETIC_ORDER_FREE_CHANNEL_AGNOSTIC_ONLY"
    return "UNKNOWN_NOT_FOR_CONCLUSION"


def table_row(**kwargs: Any) -> dict[str, Any]:
    row = {column: None for column in TABLE_COLUMNS}
    row.update(kwargs)
    return row


def group_rows(
    rows: Iterable[dict[str, str]],
    key: Callable[[dict[str, str]], Any],
) -> list[tuple[Any, list[dict[str, str]]]]:
    grouped: dict[Any, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return sorted(grouped.items(), key=lambda item: str(item[0]))


def add_numeric_descriptives(
    output: list[dict[str, Any]],
    rows: Sequence[dict[str, str]],
    column: str,
    *,
    section: str,
    role: str,
    conditioning: str,
    domain: Optional[str] = None,
    session: Optional[str] = None,
    slice_type: Optional[str] = None,
    slice_value: Optional[str] = None,
    keypoint: Optional[Any] = None,
    variant: Optional[str] = None,
    input_kind: Optional[str] = None,
    solver: Optional[str] = None,
    notes: Optional[str] = None,
    success_column: Optional[str] = None,
) -> None:
    values = [finite(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    successes = (
        [boolean(row.get(success_column)) for row in rows]
        if success_column is not None
        else []
    )
    n_success = (
        sum(value is True for value in successes) if success_column is not None else None
    )
    n_failure = (
        sum(value is False for value in successes) if success_column is not None else None
    )
    statistics: list[tuple[str, Optional[float]]] = [
        ("mean", float(np.mean(values)) if values else None),
        ("median", float(np.median(values)) if values else None),
        ("p25", float(np.percentile(values, 25)) if values else None),
        ("p75", float(np.percentile(values, 75)) if values else None),
    ]
    for statistic, value in statistics:
        output.append(
            table_row(
                section=section,
                validity_scope=validity_for(role),
                split_role=role,
                domain=domain,
                source_session=session,
                slice_type=slice_type,
                slice_value=slice_value,
                keypoint=keypoint,
                variant=variant,
                input_kind=input_kind,
                solver=solver,
                metric=column,
                statistic=statistic,
                value=value,
                n_total=len(rows),
                n_conditioned=len(values),
                n_success=n_success,
                n_failure=n_failure,
                conditioning=conditioning,
                notes=notes,
            )
        )


def add_success_rate(
    output: list[dict[str, Any]],
    rows: Sequence[dict[str, str]],
    column: str,
    *,
    section: str,
    role: str,
    conditioning: str,
    domain: Optional[str] = None,
    session: Optional[str] = None,
    slice_type: Optional[str] = None,
    slice_value: Optional[str] = None,
    keypoint: Optional[Any] = None,
    variant: Optional[str] = None,
    input_kind: Optional[str] = None,
    solver: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    values = [boolean(row.get(column)) for row in rows]
    known = [value for value in values if value is not None]
    n_success = sum(value is True for value in known)
    n_failure = sum(value is False for value in known)
    output.append(
        table_row(
            section=section,
            validity_scope=validity_for(role),
            split_role=role,
            domain=domain,
            source_session=session,
            slice_type=slice_type,
            slice_value=slice_value,
            keypoint=keypoint,
            variant=variant,
            input_kind=input_kind,
            solver=solver,
            metric=column,
            statistic="success_rate",
            value=(n_success / len(known)) if known else None,
            n_total=len(rows),
            n_conditioned=len(known),
            n_success=n_success,
            n_failure=n_failure,
            conditioning=conditioning,
            notes=notes,
        )
    )


def add_pose_summary(
    output: list[dict[str, Any]],
    rows: Sequence[dict[str, str]],
    *,
    section: str,
    role: str,
    variant: Optional[str] = None,
    input_kind: Optional[str] = None,
    solver: Optional[str] = None,
    domain: Optional[str] = None,
    session: Optional[str] = None,
    slice_type: Optional[str] = None,
    slice_value: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    add_success_rate(
        output,
        rows,
        "pose_success",
        section=section,
        role=role,
        variant=variant,
        input_kind=input_kind,
        solver=solver,
        domain=domain,
        session=session,
        slice_type=slice_type,
        slice_value=slice_value,
        conditioning="all pose attempts; failures retained",
        notes=notes,
    )
    add_numeric_descriptives(
        output,
        rows,
        "solver_runtime_ms",
        section=section,
        role=role,
        variant=variant,
        input_kind=input_kind,
        solver=solver,
        domain=domain,
        session=session,
        slice_type=slice_type,
        slice_value=slice_value,
        conditioning="all pose attempts with finite runtime, including failures",
        success_column="pose_success",
        notes=notes,
    )
    for metric in (
        "yaw_error_vs_oracle_sym180_deg",
        "gt_fixed_reproj_error_px",
        "adds180_vs_oracle_m",
    ):
        add_numeric_descriptives(
            output,
            rows,
            metric,
            section=section,
            role=role,
            variant=variant,
            input_kind=input_kind,
            solver=solver,
            domain=domain,
            session=session,
            slice_type=slice_type,
            slice_value=slice_value,
            conditioning=(
                "successful rows with finite common-reference metric; "
                "success/failure rate reported separately"
            ),
            success_column="pose_success",
            notes=notes,
        )


def add_accuracy_thresholds(
    output: list[dict[str, Any]],
    rows: Sequence[dict[str, str]],
    column: str,
    *,
    section: str,
    role: str,
    variant: str,
    domain: Optional[str] = None,
    session: Optional[str] = None,
    slice_type: Optional[str] = None,
    slice_value: Optional[str] = None,
) -> None:
    errors = [finite(row.get(column)) for row in rows]
    errors = [value for value in errors if value is not None]
    for threshold in (5.0, 10.0, 20.0):
        successes = sum(value <= threshold for value in errors)
        output.append(
            table_row(
                section=section,
                validity_scope=validity_for(role),
                split_role=role,
                domain=domain,
                source_session=session,
                slice_type=slice_type,
                slice_value=slice_value,
                variant=variant,
                metric=f"keypoint_accuracy_at_{int(threshold)}px",
                statistic="accuracy",
                value=successes / len(errors) if errors else None,
                n_total=len(rows),
                n_conditioned=len(errors),
                n_success=successes,
                n_failure=len(errors) - successes,
                conditioning=(
                    "finite correspondence-valid GT error; no detection-threshold "
                    f"conditioning; accurate iff error <= {threshold:g}px"
                ),
                notes="descriptive frozen accuracy; no operating threshold selected",
            )
        )


def elevation_bin(row: dict[str, str]) -> str:
    value = finite(row.get("gt_elevation_deg"))
    if value is None:
        return "missing"
    if value < 3.0:
        return "lt3"
    if value < 8.0:
        return "3_to_8"
    if value < 15.0:
        return "8_to_15"
    return "ge15"


def bbox_bin(row: dict[str, str]) -> str:
    width = finite(row.get("gt_bbox_w"))
    height = finite(row.get("gt_bbox_h"))
    image_width = finite(row.get("image_width"))
    image_height = finite(row.get("image_height"))
    if (
        width is None
        or height is None
        or image_width is None
        or image_height is None
        or image_width * image_height <= 0.0
    ):
        return "missing"
    fraction = width * height / (image_width * image_height)
    if fraction < 0.02:
        return "lt0.02"
    if fraction < 0.05:
        return "0.02_to_0.05"
    if fraction < 0.10:
        return "0.05_to_0.10"
    return "ge0.10"


def visible_bin(row: dict[str, str]) -> str:
    value = finite(row.get("gt_v_geom"))
    if value is None:
        return "missing"
    count = int(value)
    return "8" if count >= 8 else ("7" if count == 7 else "le6")


def covariance_area(row: dict[str, str]) -> Optional[float]:
    xx = finite(row.get("cov_px_xx"))
    xy = finite(row.get("cov_px_xy"))
    yy = finite(row.get("cov_px_yy"))
    if xx is None or xy is None or yy is None:
        return None
    determinant = xx * yy - xy * xy
    return math.pi * math.sqrt(max(determinant, 0.0))


def spearman(
    x: Iterable[Optional[float]], y: Iterable[Optional[float]]
) -> tuple[Optional[float], Optional[float], int]:
    pairs = [
        (first, second)
        for first, second in zip(x, y)
        if first is not None and second is not None
    ]
    if len(pairs) < 3:
        return None, None, len(pairs)
    from scipy.stats import spearmanr

    first = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    second = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return None, None, len(pairs)
    result = spearmanr(first, second)
    return finite(result.statistic), finite(result.pvalue), len(pairs)


def add_correlation(
    output: list[dict[str, Any]],
    rows: Sequence[dict[str, str]],
    x_name: str,
    x_values: Sequence[Optional[float]],
    y_name: str,
    y_values: Sequence[Optional[float]],
    *,
    role: str,
    domain: Optional[str] = None,
    session: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    rho, pvalue, count = spearman(x_values, y_values)
    common = dict(
        section="confidence_error_spearman",
        validity_scope=validity_for(role),
        split_role=role,
        domain=domain,
        source_session=session,
        comparison=f"{x_name}_vs_{y_name}",
        metric="spearman_rho",
        n_total=len(rows),
        n_conditioned=count,
        conditioning=(
            "all finite correspondence-valid heatmaps, regardless of detection; "
            "no confidence threshold conditioning"
        ),
        notes=notes,
    )
    output.append(table_row(**common, statistic="rho", value=rho))
    output.append(table_row(**common, statistic="two_sided_pvalue", value=pvalue))


def add_covariance_coverage(
    output: list[dict[str, Any]],
    rows: Sequence[dict[str, str]],
    *,
    role: str,
    domain: Optional[str] = None,
    session: Optional[str] = None,
    slice_type: Optional[str] = None,
    slice_value: Optional[str] = None,
) -> None:
    distances = [finite(row.get("mahalanobis_gt")) for row in rows]
    distances = [value for value in distances if value is not None]
    for nominal in COVERAGE_LEVELS:
        threshold = math.sqrt(-2.0 * math.log(1.0 - nominal))
        empirical = (
            float(np.mean(np.asarray(distances) <= threshold)) if distances else None
        )
        output.append(
            table_row(
                section="covariance_coverage",
                validity_scope=validity_for(role),
                split_role=role,
                domain=domain,
                source_session=session,
                slice_type=slice_type,
                slice_value=slice_value,
                metric=f"local_covariance_coverage_{int(100 * nominal)}",
                statistic="empirical_coverage",
                value=empirical,
                n_total=len(rows),
                n_conditioned=len(distances),
                conditioning=(
                    "finite GT correspondence and finite local-7x7 Mahalanobis distance; "
                    "includes below-threshold heatmaps"
                ),
                notes=(
                    f"nominal={nominal:.2f}; 2D Gaussian radial threshold="
                    f"{threshold:.6f}; calibration-only, not pose weighting"
                ),
            )
        )


def add_bootstrap_passthrough(
    output: list[dict[str, Any]], bootstrap: dict[str, Any]
) -> None:
    for family in ("ladder", "solver"):
        comparisons = bootstrap.get(family, {})
        if not isinstance(comparisons, dict):
            continue
        for comparison, metrics in sorted(comparisons.items()):
            if not isinstance(metrics, dict):
                continue
            for metric, result in sorted(metrics.items()):
                if metric == "success_outcomes" or not isinstance(result, dict):
                    continue
                if "mean_delta" not in result:
                    continue
                output.append(
                    table_row(
                        section="bootstrap_ci_passthrough",
                        validity_scope="PRIMARY_STRICT_N87",
                        split_role=STRICT,
                        comparison=f"{family}:{comparison}",
                        metric=metric,
                        statistic="mean_paired_delta_95ci",
                        value=result.get("mean_delta"),
                        ci95_low=result.get("ci95_low"),
                        ci95_high=result.get("ci95_high"),
                        n_total=result.get("n_total_paired_frames"),
                        n_conditioned=result.get("n_pairs"),
                        conditioning=result.get("conditioning"),
                        ci_method=result.get("method"),
                        notes=(
                            "EXACT PASSTHROUGH from frozen summary.json; "
                            f"replicates={result.get('replicates')}, "
                            f"seed={result.get('seed')}; {result.get('limitation')}"
                        ),
                    )
                )


def build_tables(
    data: dict[str, list[dict[str, str]]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    frames = data["frames.csv"]
    keypoints = data["keypoints.csv"]

    # Dataset membership and frame failures are never silently conditioned out.
    for role, role_rows in group_rows(frames, lambda row: row.get("split_role")):
        ok = sum(row.get("status") == "ok" for row in role_rows)
        output.append(
            table_row(
                section="dataset_membership",
                validity_scope=validity_for(str(role)),
                split_role=role,
                metric="frame_count",
                statistic="count",
                value=len(role_rows),
                n_total=len(role_rows),
                n_conditioned=len(role_rows),
                n_success=ok,
                n_failure=len(role_rows) - ok,
                conditioning="all frozen frames in this split role",
                notes="strict/manual/synthetic roles are never pooled for conclusions",
            )
        )
        for domain, domain_rows in group_rows(role_rows, lambda row: row.get("domain")):
            output.append(
                table_row(
                    section="dataset_membership",
                    validity_scope=validity_for(str(role)),
                    split_role=role,
                    domain=domain,
                    metric="frame_count",
                    statistic="count",
                    value=len(domain_rows),
                    n_total=len(domain_rows),
                    n_conditioned=len(domain_rows),
                    n_success=sum(row.get("status") == "ok" for row in domain_rows),
                    n_failure=sum(row.get("status") != "ok" for row in domain_rows),
                    conditioning="all frozen frames in domain",
                )
            )

    # Missing metadata is reported, never imputed.
    metadata_columns = (
        "K_fx",
        "dim_W_m",
        "gt_elevation_deg",
        "gt_distance_m",
        "gt_view_azimuth_deg",
        "lighting_metadata",
        "background_metadata",
        "occlusion_metadata",
        "filter_pass",
    )
    for role, role_rows in group_rows(frames, lambda row: row.get("split_role")):
        for column in metadata_columns:
            missing = sum(row.get(column, "") == "" for row in role_rows)
            output.append(
                table_row(
                    section="metadata_missingness",
                    validity_scope=validity_for(str(role)),
                    split_role=role,
                    metric=column,
                    statistic="missing_rate",
                    value=missing / len(role_rows) if role_rows else None,
                    n_total=len(role_rows),
                    n_conditioned=len(role_rows),
                    n_failure=missing,
                    conditioning="all frames; blank means null, never inferred",
                )
            )

    # Synthetic results are deliberately restricted to order-free frame metrics
    # and channel-agnostic heatmap distributions.
    synth_frames = [row for row in frames if row.get("split_role") == SYNTHETIC]
    for metric in (
        "argmax_order_free_mean_px",
        "softargmax_order_free_mean_px",
        "softargmax_order_free_median_px",
        "n_softargmax_detected",
    ):
        add_numeric_descriptives(
            output,
            synth_frames,
            metric,
            section="synthetic_order_free",
            role=SYNTHETIC,
            conditioning="all synthetic frames with finite order-free frame metric",
            notes=(
                "No kp-id, kp5, fixed-correspondence PnP, or yaw conclusion is valid"
            ),
        )
    synth_kp = [
        row for row in keypoints if row.get("split_role") == SYNTHETIC
    ]
    for metric in (
        "peak",
        "peak_second_ratio",
        "entropy_normalized",
        "cov_eig_major_px2",
        "cov_eig_minor_px2",
    ):
        add_numeric_descriptives(
            output,
            synth_kp,
            metric,
            section="synthetic_channel_agnostic_heatmap",
            role=SYNTHETIC,
            conditioning="all finite heatmap channels pooled without keypoint identity",
            notes="channel distribution only; GT corner correspondence is invalid",
        )

    # Real keypoint accuracy/covariance: strict primary and manual exploratory
    # remain separate at every aggregation level.
    for role in (STRICT, EXPLORATORY):
        role_kp = [
            row
            for row in keypoints
            if row.get("split_role") == role
            and boolean(row.get("kp_error_valid_for_conclusion")) is True
        ]
        for metric in (
            "argmax_error_gt_px",
            "softargmax_error_gt_px",
            "peak",
            "entropy_normalized",
        ):
            add_numeric_descriptives(
                output,
                role_kp,
                metric,
                section="keypoint_localization",
                role=role,
                conditioning=(
                    "GT-correspondence-valid keypoints with finite metric; "
                    "not conditioned on detection"
                ),
            )
        for domain, rows in group_rows(role_kp, lambda row: row.get("domain")):
            add_numeric_descriptives(
                output,
                rows,
                "softargmax_error_gt_px",
                section="keypoint_localization_by_domain",
                role=role,
                domain=domain,
                conditioning="GT-correspondence-valid keypoints; finite error",
            )
            add_covariance_coverage(output, rows, role=role, domain=domain)
        for session, rows in group_rows(
            role_kp, lambda row: row.get("source_session") or "missing"
        ):
            add_numeric_descriptives(
                output,
                rows,
                "softargmax_error_gt_px",
                section="keypoint_localization_by_session",
                role=role,
                session=session,
                conditioning="GT-correspondence-valid keypoints; finite error",
            )
            add_covariance_coverage(output, rows, role=role, session=session)
        if role == STRICT:
            for slice_name, slicer in (
                ("elevation_bin", elevation_bin),
                ("bbox_area_bin", bbox_bin),
                ("in_frame_corner_bin", visible_bin),
            ):
                for value, rows in group_rows(role_kp, slicer):
                    add_numeric_descriptives(
                        output,
                        rows,
                        "softargmax_error_gt_px",
                        section="keypoint_localization_by_slice",
                        role=role,
                        slice_type=slice_name,
                        slice_value=value,
                        conditioning="GT-correspondence-valid keypoints; finite error",
                    )
                    add_covariance_coverage(
                        output,
                        rows,
                        role=role,
                        slice_type=slice_name,
                        slice_value=value,
                    )
        add_covariance_coverage(output, role_kp, role=role)

        errors = [finite(row.get("softargmax_error_gt_px")) for row in role_kp]
        confidence_signals = {
            "peak": [finite(row.get("peak")) for row in role_kp],
            "peak_second_ratio": [
                finite(row.get("peak_second_ratio")) for row in role_kp
            ],
            "entropy_normalized": [
                finite(row.get("entropy_normalized")) for row in role_kp
            ],
            "local_covariance_area_px2": [
                covariance_area(row) for row in role_kp
            ],
        }
        for name, values in confidence_signals.items():
            add_correlation(
                output,
                role_kp,
                name,
                values,
                "softargmax_error_gt_px",
                errors,
                role=role,
                notes=(
                    "Spearman is descriptive; no threshold was selected or tuned"
                ),
            )
        if role == STRICT:
            for domain, rows in group_rows(role_kp, lambda row: row.get("domain")):
                add_correlation(
                    output,
                    rows,
                    "peak",
                    [finite(row.get("peak")) for row in rows],
                    "softargmax_error_gt_px",
                    [finite(row.get("softargmax_error_gt_px")) for row in rows],
                    role=role,
                    domain=domain,
                )
                add_correlation(
                    output,
                    rows,
                    "local_covariance_area_px2",
                    [covariance_area(row) for row in rows],
                    "softargmax_error_gt_px",
                    [finite(row.get("softargmax_error_gt_px")) for row in rows],
                    role=role,
                    domain=domain,
                )

    # Signed centroid residuals.
    for role in (STRICT, EXPLORATORY):
        centroids = [
            dict(
                row,
                _centroid_dx=str(
                    (finite(row.get("softargmax_x")) or 0.0)
                    - (finite(row.get("gt_x")) or 0.0)
                )
                if finite(row.get("softargmax_x")) is not None
                and finite(row.get("gt_x")) is not None
                else "",
                _centroid_dy=str(
                    (finite(row.get("softargmax_y")) or 0.0)
                    - (finite(row.get("gt_y")) or 0.0)
                )
                if finite(row.get("softargmax_y")) is not None
                and finite(row.get("gt_y")) is not None
                else "",
            )
            for row in keypoints
            if row.get("split_role") == role
            and row.get("keypoint") == "8"
            and boolean(row.get("gt_correspondence_valid")) is True
        ]
        for metric in ("_centroid_dx", "_centroid_dy", "softargmax_error_gt_px"):
            add_numeric_descriptives(
                output,
                centroids,
                metric,
                section="centroid_signed_residual",
                role=role,
                keypoint=8,
                conditioning="centroid GT and local-softargmax both finite",
            )
        if role == STRICT:
            for domain, rows in group_rows(centroids, lambda row: row.get("domain")):
                for metric in ("_centroid_dx", "_centroid_dy"):
                    add_numeric_descriptives(
                        output,
                        rows,
                        metric,
                        section="centroid_signed_residual_by_domain",
                        role=role,
                        domain=domain,
                        keypoint=8,
                        conditioning="centroid GT and local-softargmax both finite",
                    )

    # Y0-Y9 frozen cause ladder, including failures and common-reference pose
    # metrics. Strict results receive domain/session/geometric-slice breakdowns.
    ladder = data["yaw_ladder.csv"]
    for role in (STRICT, EXPLORATORY):
        role_rows = [row for row in ladder if row.get("split_role") == role]
        for stage, rows in group_rows(role_rows, lambda row: row.get("stage")):
            notes = (
                f"description={rows[0].get('description') if rows else None}; "
                f"mask_basis={rows[0].get('mask_basis') if rows else None}"
            )
            add_pose_summary(
                output,
                rows,
                section="yaw_cause_ladder",
                role=role,
                variant=stage,
                solver=rows[0].get("solver") if rows else None,
                notes=notes,
            )
        if role == STRICT:
            for (stage, domain), rows in group_rows(
                role_rows, lambda row: (row.get("stage"), row.get("domain"))
            ):
                add_pose_summary(
                    output,
                    rows,
                    section="yaw_cause_ladder_by_domain",
                    role=role,
                    variant=stage,
                    solver=rows[0].get("solver") if rows else None,
                    domain=domain,
                )
            for (stage, session), rows in group_rows(
                role_rows,
                lambda row: (
                    row.get("stage"),
                    row.get("source_session") or "missing",
                ),
            ):
                add_pose_summary(
                    output,
                    rows,
                    section="yaw_cause_ladder_by_session",
                    role=role,
                    variant=stage,
                    solver=rows[0].get("solver") if rows else None,
                    session=session,
                )
            for slice_name, slicer in (
                ("elevation_bin", elevation_bin),
                ("bbox_area_bin", bbox_bin),
                ("in_frame_corner_bin", visible_bin),
            ):
                for (stage, value), rows in group_rows(
                    role_rows, lambda row, fn=slicer: (row.get("stage"), fn(row))
                ):
                    add_pose_summary(
                        output,
                        rows,
                        section="yaw_cause_ladder_by_slice",
                        role=role,
                        variant=stage,
                        solver=rows[0].get("solver") if rows else None,
                        slice_type=slice_name,
                        slice_value=value,
                    )

    # Keypoint influence; signed error change is candidate minus Y2.
    influence = data["keypoint_influence.csv"]
    for role in (STRICT, EXPLORATORY):
        role_rows = [row for row in influence if row.get("split_role") == role]
        for (variant, keypoint), rows in group_rows(
            role_rows, lambda row: (row.get("variant"), row.get("keypoint"))
        ):
            add_success_rate(
                output,
                rows,
                "candidate_pose_success",
                section="keypoint_influence",
                role=role,
                keypoint=keypoint,
                variant=variant,
                conditioning="all attempted frames; failures retained",
            )
            for metric in (
                "candidate_error_minus_baseline_yaw_sym180_deg",
                "candidate_error_minus_baseline_gt_fixed_reproj_px",
                "candidate_error_minus_baseline_adds180_vs_oracle_m",
            ):
                add_numeric_descriptives(
                    output,
                    rows,
                    metric,
                    section="keypoint_influence",
                    role=role,
                    keypoint=keypoint,
                    variant=variant,
                    conditioning=(
                        "candidate and Y2 baseline both successful with finite common-"
                        "reference metric; success rate reported separately"
                    ),
                    success_column="candidate_pose_success",
                    notes="signed candidate error minus Y2 error; negative improves",
                )

    # kp5 compression geometry and explicit slices.
    kp5 = data["kp5_geometry.csv"]
    for role in (STRICT, EXPLORATORY):
        role_rows = [row for row in kp5 if row.get("split_role") == role]
        for metric in (
            "kp5_vector_dx_px",
            "kp5_vector_dy_px",
            "kp5_error_px",
            "far_top_4_5_ratio",
            "vertical_5_6_ratio",
            "depth_1_5_ratio",
            "far_face_perimeter_ratio",
        ):
            add_numeric_descriptives(
                output,
                role_rows,
                metric,
                section="kp5_geometry",
                role=role,
                keypoint=5,
                conditioning="finite predicted and GT geometry for the named quantity",
            )
        if role == STRICT:
            for slice_column in ("elevation_bin", "bbox_area_bin", "visible_bin"):
                for value, rows in group_rows(
                    role_rows, lambda row, key=slice_column: row.get(key) or "missing"
                ):
                    add_numeric_descriptives(
                        output,
                        rows,
                        "kp5_error_px",
                        section="kp5_geometry_by_slice",
                        role=role,
                        keypoint=5,
                        slice_type=slice_column,
                        slice_value=value,
                        conditioning="finite kp5 predicted/GT point",
                    )

    # Solver success, common-reference accuracy, and runtime.
    solvers = data["solver_comparison.csv"]
    for role in (STRICT, EXPLORATORY):
        role_rows = [row for row in solvers if row.get("split_role") == role]
        for (input_kind, solver), rows in group_rows(
            role_rows, lambda row: (row.get("input_kind"), row.get("solver"))
        ):
            add_pose_summary(
                output,
                rows,
                section="solver_comparison",
                role=role,
                input_kind=input_kind,
                solver=solver,
                notes=(
                    f"oracle_reference="
                    f"{rows[0].get('oracle_reference') if rows else None}; "
                    "direct solvers use locked-Y0 reference"
                ),
            )
        if role == STRICT:
            for (domain, input_kind, solver), rows in group_rows(
                role_rows,
                lambda row: (
                    row.get("domain"),
                    row.get("input_kind"),
                    row.get("solver"),
                ),
            ):
                add_pose_summary(
                    output,
                    rows,
                    section="solver_comparison_by_domain",
                    role=role,
                    domain=domain,
                    input_kind=input_kind,
                    solver=solver,
                )
            for (session, input_kind, solver), rows in group_rows(
                role_rows,
                lambda row: (
                    row.get("source_session") or "missing",
                    row.get("input_kind"),
                    row.get("solver"),
                ),
            ):
                add_pose_summary(
                    output,
                    rows,
                    section="solver_comparison_by_session",
                    role=role,
                    session=session,
                    input_kind=input_kind,
                    solver=solver,
                )
            for slice_name, slicer in (
                ("elevation_bin", elevation_bin),
                ("bbox_area_bin", bbox_bin),
                ("in_frame_corner_bin", visible_bin),
            ):
                for (value, input_kind, solver), rows in group_rows(
                    role_rows,
                    lambda row, fn=slicer: (
                        fn(row),
                        row.get("input_kind"),
                        row.get("solver"),
                    ),
                ):
                    add_pose_summary(
                        output,
                        rows,
                        section="solver_comparison_by_slice",
                        role=role,
                        input_kind=input_kind,
                        solver=solver,
                        slice_type=slice_name,
                        slice_value=value,
                    )

    # Flip accuracy/reliability, matched-detected coordinate conditioning explicit.
    flip_frames = data["flip_consistency.csv"]
    flip_kp = data["flip_keypoints.csv"]
    for role in (STRICT, EXPLORATORY):
        role_frames = [
            row for row in flip_frames if row.get("split_role") == role
        ]
        add_success_rate(
            output,
            role_frames,
            "pose_success",
            section="flip_pose",
            role=role,
            conditioning="all flip pose attempts",
        )
        for metric in (
            "mean_matched_detected_softargmax_consistency_px",
            "mean_belief_equivariance_rmse",
            "flip_vs_original_yaw_sym180_deg",
            "yaw_error_vs_oracle_sym180_deg",
            "adds180_vs_oracle_m",
            "flip_solver_runtime_ms",
        ):
            add_numeric_descriptives(
                output,
                role_frames,
                metric,
                section="flip_pose",
                role=role,
                conditioning=(
                    "matched-detected keypoints for coordinate aggregate; finite "
                    "successful poses for pose metric"
                ),
                success_column="pose_success",
            )
        role_kp = [
            row
            for row in flip_kp
            if row.get("split_role") == role
            and boolean(row.get("keypoint_channel_correspondence_valid")) is True
        ]
        add_success_rate(
            output,
            role_kp,
            "matched_detected",
            section="flip_keypoint",
            role=role,
            conditioning="all keypoint channels; matched means detected in both views",
        )
        for variant, column in (
            ("original", "original_softargmax_error_gt_px"),
            ("flip_unwarped", "flip_unwarped_softargmax_error_gt_px"),
        ):
            add_accuracy_thresholds(
                output,
                role_kp,
                column,
                section="flip_accuracy",
                role=role,
                variant=variant,
            )
            add_numeric_descriptives(
                output,
                role_kp,
                column,
                section="flip_accuracy",
                role=role,
                variant=variant,
                conditioning=(
                    "finite correspondence-valid GT error; not conditioned on "
                    "detection"
                ),
            )
        delta_rows = []
        for row in role_kp:
            original_error = finite(row.get("original_softargmax_error_gt_px"))
            flipped_error = finite(row.get("flip_unwarped_softargmax_error_gt_px"))
            delta_rows.append(
                dict(
                    row,
                    _flip_error_delta=(
                        str(flipped_error - original_error)
                        if original_error is not None and flipped_error is not None
                        else ""
                    ),
                )
            )
        add_numeric_descriptives(
            output,
            delta_rows,
            "_flip_error_delta",
            section="flip_accuracy",
            role=role,
            variant="flip_unwarped_minus_original",
            conditioning=(
                "finite paired original and unwarped-flip GT errors; negative improves"
            ),
        )
        if role == STRICT:
            for grouping, group_name in (
                (lambda row: row.get("domain"), "domain"),
                (lambda row: row.get("source_session") or "missing", "session"),
            ):
                for value, rows in group_rows(role_kp, grouping):
                    for variant, column in (
                        ("original", "original_softargmax_error_gt_px"),
                        ("flip_unwarped", "flip_unwarped_softargmax_error_gt_px"),
                    ):
                        add_accuracy_thresholds(
                            output,
                            rows,
                            column,
                            section=f"flip_accuracy_by_{group_name}",
                            role=role,
                            variant=variant,
                            domain=value if group_name == "domain" else None,
                            session=value if group_name == "session" else None,
                        )
            for slice_name, slicer in (
                ("elevation_bin", elevation_bin),
                ("bbox_area_bin", bbox_bin),
                ("in_frame_corner_bin", visible_bin),
            ):
                for value, rows in group_rows(role_kp, slicer):
                    for variant, column in (
                        ("original", "original_softargmax_error_gt_px"),
                        ("flip_unwarped", "flip_unwarped_softargmax_error_gt_px"),
                    ):
                        add_accuracy_thresholds(
                            output,
                            rows,
                            column,
                            section="flip_accuracy_by_slice",
                            role=role,
                            variant=variant,
                            slice_type=slice_name,
                            slice_value=value,
                        )
        matched = [
            row for row in role_kp if boolean(row.get("matched_detected")) is True
        ]
        add_numeric_descriptives(
            output,
            matched,
            "matched_detected_softargmax_consistency_px",
            section="flip_keypoint",
            role=role,
            conditioning="detected in both original and flipped inference",
        )
        add_correlation(
            output,
            matched,
            "flip_consistency_px",
            [
                finite(row.get("matched_detected_softargmax_consistency_px"))
                for row in matched
            ],
            "original_softargmax_error_gt_px",
            [
                finite(row.get("original_softargmax_error_gt_px")) for row in matched
            ],
            role=role,
            notes="flip reliability is descriptive; no acceptance threshold selected",
        )

    add_bootstrap_passthrough(output, summary.get("bootstrap_95ci", {}))

    # Training claims are explicitly outside the frozen-inference evidence.
    for ablation in (
        "C0-C4 covariance-weighted pose training/inference",
        "decoder retraining or loss ablation",
        "kp5/centroid supervision ablation",
        "self-training causal ablation",
    ):
        output.append(
            table_row(
                section="training_ablations",
                validity_scope="BLOCKED",
                metric=ablation,
                statistic="status",
                value=None,
                n_total=0,
                n_conditioned=0,
                conditioning="not run in frozen diagnostic",
                notes=(
                    "BLOCKED: requires separately trained matched checkpoints/runs; "
                    "no causal training conclusion may be drawn from this post-analysis"
                ),
            )
        )
    return output


def markdown_escape(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def short_number(value: Any, digits: int = 3) -> str:
    number = finite(value)
    return "NA" if number is None else f"{number:.{digits}g}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(markdown_escape(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def build_markdown(
    validation: dict[str, Any],
    table_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    def select(section: str, predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
        return [
            row
            for row in table_rows
            if row.get("section") == section and predicate(row)
        ]

    membership = select("dataset_membership", lambda row: row.get("domain") is None)
    coverage = select(
        "covariance_coverage",
        lambda row: row.get("split_role") == STRICT
        and row.get("domain") is None
        and row.get("source_session") is None
        and row.get("slice_type") is None,
    )
    correlations = select(
        "confidence_error_spearman",
        lambda row: row.get("split_role") == STRICT
        and row.get("domain") is None
        and row.get("statistic") == "rho",
    )
    centroid = select(
        "centroid_signed_residual",
        lambda row: row.get("split_role") == STRICT
        and row.get("statistic") in {"mean", "median"},
    )
    solver = select(
        "solver_comparison",
        lambda row: row.get("split_role") == STRICT
        and row.get("input_kind") == "predicted_softargmax"
        and row.get("solver") in DIRECT_SOLVERS
        and (
            row.get("statistic") == "success_rate"
            or (
                row.get("statistic") == "median"
                and row.get("metric")
                in {
                    "yaw_error_vs_oracle_sym180_deg",
                    "gt_fixed_reproj_error_px",
                    "adds180_vs_oracle_m",
                    "solver_runtime_ms",
                }
            )
        ),
    )
    ci_rows = select(
        "bootstrap_ci_passthrough",
        lambda row: row.get("metric")
        in {"yaw_error_vs_oracle_sym180_deg", "pose_success_rate_delta"},
    )
    blocked = select("training_ablations", lambda row: True)

    lines = [
        "# Frozen PAPER_S2 post-analysis",
        "",
        f"- Source run status: `{validation['run_status']}`",
        f"- Full run: `{validation['is_full']}`",
        f"- Frozen checkpoint SHA: `{summary.get('checkpoint_sha256')}`",
        f"- Frozen script SHA: `{manifest.get('source', {}).get('script_sha256')}`",
        "- CI policy: exact passthrough of the frozen 10,000-replicate summary; no resampling here.",
        "- Missing metadata remains null and is never imputed.",
        "",
        "## Validity",
        "",
        "- **Primary:** strict filter-val outside/night only (full N=87).",
        "- **Exploratory:** manual36 from the capturepallet11 PL pool.",
        "- **Synthetic:** order-free frame aggregates and channel-agnostic heatmap distributions only.",
        "- **Training ablations:** BLOCKED without separately trained matched checkpoints.",
        "",
        "## Dataset membership and failures",
        "",
        markdown_table(
            ("role", "validity", "frames", "success", "failure"),
            [
                (
                    row.get("split_role"),
                    row.get("validity_scope"),
                    row.get("value"),
                    row.get("n_success"),
                    row.get("n_failure"),
                )
                for row in membership
            ],
        ),
        "",
        "## Local covariance coverage",
        "",
        markdown_table(
            ("nominal metric", "empirical", "n", "conditioning"),
            [
                (
                    row.get("metric"),
                    row.get("value"),
                    row.get("n_conditioned"),
                    row.get("conditioning"),
                )
                for row in coverage
            ],
        ),
        "",
        "## Confidence / covariance association with keypoint error",
        "",
        markdown_table(
            ("comparison", "Spearman rho", "n", "notes"),
            [
                (
                    row.get("comparison"),
                    row.get("value"),
                    row.get("n_conditioned"),
                    row.get("notes"),
                )
                for row in correlations
            ],
        ),
        "",
        "## Signed centroid residual",
        "",
        markdown_table(
            ("metric", "statistic", "value px", "n"),
            [
                (
                    row.get("metric"),
                    row.get("statistic"),
                    row.get("value"),
                    row.get("n_conditioned"),
                )
                for row in centroid
            ],
        ),
        "",
        "## Direct solver comparison (predicted softargmax, locked-Y0 reference)",
        "",
        markdown_table(
            ("solver", "metric", "statistic", "value", "success", "failure", "conditioning"),
            [
                (
                    row.get("solver"),
                    row.get("metric"),
                    row.get("statistic"),
                    row.get("value"),
                    row.get("n_success"),
                    row.get("n_failure"),
                    row.get("conditioning"),
                )
                for row in solver
            ],
        ),
        "",
        "## Frozen paired 95% confidence intervals",
        "",
        markdown_table(
            ("comparison", "metric", "delta", "95% low", "95% high", "n", "method"),
            [
                (
                    row.get("comparison"),
                    row.get("metric"),
                    row.get("value"),
                    row.get("ci95_low"),
                    row.get("ci95_high"),
                    row.get("n_conditioned"),
                    row.get("ci_method"),
                )
                for row in ci_rows
            ],
        ),
        "",
        "## Training ablations",
        "",
        markdown_table(
            ("ablation", "status", "reason"),
            [
                (row.get("metric"), "BLOCKED", row.get("notes"))
                for row in blocked
            ],
        ),
        "",
        "The machine-readable long-form source for every table above, including domain/session/slice rows, is `frozen_tables.csv`.",
        "",
    ]
    return "\n".join(lines)


def covariance_ellipse_figure(
    path: Path, keypoints: list[dict[str, str]]
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    rows = [
        row
        for row in keypoints
        if row.get("split_role") == STRICT
        and boolean(row.get("kp_error_valid_for_conclusion")) is True
        and all(
            finite(row.get(column)) is not None
            for column in (
                "softargmax_x",
                "softargmax_y",
                "gt_x",
                "gt_y",
                "cov_px_xx",
                "cov_px_xy",
                "cov_px_yy",
                "softargmax_error_gt_px",
            )
        )
    ]
    rows.sort(key=lambda row: finite(row["softargmax_error_gt_px"]) or 0.0)
    if rows:
        indices = np.unique(
            np.rint(np.linspace(0, len(rows) - 1, min(9, len(rows)))).astype(int)
        )
        chosen = [rows[int(index)] for index in indices]
    else:
        chosen = []
    figure, axes = plt.subplots(
        3, 3, figsize=(12, 12), constrained_layout=True
    )
    axes_flat = axes.ravel()
    chi2_90 = -2.0 * math.log(1.0 - 0.90)
    for axis, row in zip(axes_flat, chosen):
        pred = np.array(
            [finite(row["softargmax_x"]), finite(row["softargmax_y"])],
            dtype=np.float64,
        )
        gt = np.array([finite(row["gt_x"]), finite(row["gt_y"])], dtype=np.float64)
        delta = gt - pred
        covariance = np.array(
            [
                [finite(row["cov_px_xx"]), finite(row["cov_px_xy"])],
                [finite(row["cov_px_xy"]), finite(row["cov_px_yy"])],
            ],
            dtype=np.float64,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        major = eigenvectors[:, order[0]]
        angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
        ellipse = Ellipse(
            (0.0, 0.0),
            width=2.0 * math.sqrt(chi2_90 * eigenvalues[0]),
            height=2.0 * math.sqrt(chi2_90 * eigenvalues[1]),
            angle=angle,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.20,
        )
        axis.add_patch(ellipse)
        axis.scatter([0.0], [0.0], marker="x", color="tab:blue", label="prediction")
        axis.scatter([delta[0]], [delta[1]], marker="o", color="tab:red", label="GT")
        radius = max(
            5.0,
            float(np.linalg.norm(delta)) * 1.25,
            math.sqrt(chi2_90 * eigenvalues[0]) * 1.25,
        )
        axis.set_xlim(-radius, radius)
        axis.set_ylim(radius, -radius)
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
        axis.set_title(
            f"{row.get('domain')}/{row.get('source_session')} kp{row.get('keypoint')}\n"
            f"err={finite(row.get('softargmax_error_gt_px')):.2f}px "
            f"M={short_number(row.get('mahalanobis_gt'))}"
        )
    for axis in axes_flat[len(chosen) :]:
        axis.text(0.5, 0.5, "No example", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    figure.suptitle(
        "Strict filter-val local-7x7 covariance: 90% ellipses (prediction-centered)"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "n_candidates": len(rows),
        "n_examples": len(chosen),
        "conditioning": "strict correspondence-valid rows with finite 2x2 covariance",
    }


def confidence_error_figure(
    path: Path, keypoints: list[dict[str, str]]
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in keypoints
        if row.get("split_role") == STRICT
        and boolean(row.get("kp_error_valid_for_conclusion")) is True
        and finite(row.get("softargmax_error_gt_px")) is not None
    ]
    signals: list[tuple[str, Callable[[dict[str, str]], Optional[float]]]] = [
        ("peak", lambda row: finite(row.get("peak"))),
        ("peak / second", lambda row: finite(row.get("peak_second_ratio"))),
        ("entropy normalized", lambda row: finite(row.get("entropy_normalized"))),
        ("local covariance area (px²)", covariance_area),
    ]
    figure, axes = plt.subplots(
        2, 2, figsize=(12, 10), constrained_layout=True
    )
    correlation_results = {}
    for axis, (name, getter) in zip(axes.ravel(), signals):
        pairs = [
            (
                getter(row),
                finite(row.get("softargmax_error_gt_px")),
                boolean(row.get("detected")) is True,
            )
            for row in rows
        ]
        pairs = [pair for pair in pairs if pair[0] is not None and pair[1] is not None]
        detected = [pair for pair in pairs if pair[2]]
        below = [pair for pair in pairs if not pair[2]]
        if detected:
            axis.scatter(
                [pair[0] for pair in detected],
                [pair[1] for pair in detected],
                s=14,
                alpha=0.55,
                label="detected",
            )
        if below:
            axis.scatter(
                [pair[0] for pair in below],
                [pair[1] for pair in below],
                s=20,
                alpha=0.65,
                marker="x",
                label="below threshold",
            )
        rho, pvalue, count = spearman(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        )
        correlation_results[name] = {"rho": rho, "pvalue": pvalue, "n": count}
        axis.set_title(f"{name}: Spearman ρ={short_number(rho)}, n={count}")
        axis.set_xlabel(name)
        axis.set_ylabel("softargmax GT error (px)")
        axis.grid(alpha=0.25)
        if detected or below:
            axis.legend()
    figure.suptitle(
        "Strict filter-val confidence/covariance vs error (no detection conditioning)"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {"n_rows": len(rows), "correlations": correlation_results}


def metric_ci_figure(path: Path, bootstrap: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        (
            "Ladder Δ yaw err_180 vs Y2",
            bootstrap.get("ladder", {}),
            "yaw_error_vs_oracle_sym180_deg",
            lambda name: name,
        ),
        (
            "Direct solver Δ yaw err_180 vs EPnP\n(predicted softargmax)",
            {
                name: value
                for name, value in bootstrap.get("solver", {}).items()
                if name.startswith("predicted_softargmax:")
            },
            "yaw_error_vs_oracle_sym180_deg",
            lambda name: name.split(":", 1)[-1].replace("_minus_EPnP", ""),
        ),
        (
            "Pose success-rate Δ",
            bootstrap.get("ladder", {}),
            "pose_success_rate_delta",
            lambda name: name,
        ),
    ]
    figure, axes = plt.subplots(
        1, 3, figsize=(18, 6), constrained_layout=True
    )
    plotted_counts = {}
    for axis, (title, comparisons, metric, labeler) in zip(axes, panels):
        points = []
        for name, metrics in sorted(comparisons.items()):
            result = metrics.get(metric) if isinstance(metrics, dict) else None
            if not isinstance(result, dict):
                continue
            mean = finite(result.get("mean_delta"))
            low = finite(result.get("ci95_low"))
            high = finite(result.get("ci95_high"))
            if mean is None or low is None or high is None:
                continue
            points.append((labeler(name), mean, low, high))
        plotted_counts[title] = len(points)
        if points:
            positions = np.arange(len(points))
            means = np.asarray([point[1] for point in points])
            lower = means - np.asarray([point[2] for point in points])
            upper = np.asarray([point[3] for point in points]) - means
            axis.errorbar(
                positions,
                means,
                yerr=np.vstack([lower, upper]),
                fmt="o",
                capsize=4,
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xticks(
                positions,
                [point[0] for point in points],
                rotation=55,
                ha="right",
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No frozen CI rows",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_title(title)
        axis.set_ylabel("candidate − baseline (negative improves error)")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Frozen 10,000-replicate paired 95% CIs (exact summary.json passthrough)"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {"plotted_counts": plotted_counts, "ci_recomputed": False}


def flip_reliability_figure(
    path: Path, flip_keypoints: list[dict[str, str]]
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in flip_keypoints
        if row.get("split_role") == STRICT
        and boolean(row.get("matched_detected")) is True
        and finite(row.get("matched_detected_softargmax_consistency_px")) is not None
        and finite(row.get("original_softargmax_error_gt_px")) is not None
    ]
    figure, axes = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )
    if rows:
        x = np.asarray(
            [
                finite(row["matched_detected_softargmax_consistency_px"])
                for row in rows
            ]
        )
        y = np.asarray(
            [finite(row["original_softargmax_error_gt_px"]) for row in rows]
        )
        keypoint = np.asarray([int(row["keypoint"]) for row in rows])
        scatter = axes[0].scatter(x, y, c=keypoint, cmap="tab10", s=18, alpha=0.65)
        figure.colorbar(scatter, ax=axes[0], label="keypoint")
        rho, pvalue, count = spearman(x.tolist(), y.tolist())
        axes[0].set_title(
            "matched-detected: "
            f"ρ={short_number(rho)}, p={short_number(pvalue)}, n={count}"
        )
        axes[0].set_xlabel("flip softargmax consistency (px)")
        axes[0].set_ylabel("original GT error (px)")
        order = np.argsort(x)
        sorted_error = y[order]
        requested_coverages = np.linspace(0.1, 1.0, 10)
        retained_counts = np.maximum(
            1, np.ceil(requested_coverages * len(rows)).astype(int)
        )
        actual_coverages = retained_counts / len(rows)
        for threshold in (5.0, 10.0, 20.0):
            accuracies = [
                float(np.mean(sorted_error[:count] <= threshold))
                for count in retained_counts
            ]
            axes[1].plot(
                actual_coverages,
                accuracies,
                marker="o",
                label=f"error ≤ {threshold:g}px",
            )
        axes[1].set_xlim(0.0, 1.02)
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_xlabel("retained coverage (lowest inconsistency first)")
        axes[1].set_ylabel("keypoint accuracy")
        axes[1].set_title("Descriptive accuracy–coverage reliability")
        axes[1].legend()
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No matched-detected strict rows",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Strict filter-val flip reliability")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "n_matched_detected": len(rows),
        "conditioning": "strict keypoints detected in both original and flipped views",
        "reliability_policy": (
            "retain lowest flip-consistency rows; 5/10/20px accuracy; "
            "descriptive only, no operating point selected"
        ),
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    source_summary = read_json(run_dir / "summary.json")
    data = {
        name: read_csv(run_dir / name)
        for name in REQUIRED_INPUTS
        if name.endswith(".csv")
    }
    validation = validate_run(
        run_dir,
        manifest,
        source_summary,
        data,
        args.allow_smoke,
    )
    table_rows = build_tables(data, source_summary)
    markdown = build_markdown(
        validation, table_rows, manifest, source_summary
    )

    figure_info = {
        "covariance_ellipse_examples.png": covariance_ellipse_figure(
            run_dir / "covariance_ellipse_examples.png", data["keypoints.csv"]
        ),
        "confidence_vs_error.png": confidence_error_figure(
            run_dir / "confidence_vs_error.png", data["keypoints.csv"]
        ),
        "metric_ci_barplot.png": metric_ci_figure(
            run_dir / "metric_ci_barplot.png",
            source_summary.get("bootstrap_95ci", {}),
        ),
        "flip_reliability.png": flip_reliability_figure(
            run_dir / "flip_reliability.png", data["flip_keypoints.csv"]
        ),
    }
    write_csv_new(run_dir / "frozen_tables.csv", table_rows)
    write_text_new(run_dir / "frozen_tables.md", markdown)

    input_hashes = {
        name: sha256_file(run_dir / name) for name in REQUIRED_INPUTS
    }
    analysis_summary = {
        "format_version": 1,
        "status": "complete" if validation["is_full"] else "smoke_validation_only",
        "run_dir": str(run_dir),
        "source_validation": validation,
        "source_hashes": input_hashes,
        "post_analysis_source": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "git_head": git_head(),
        },
        "validity": {
            "primary": "strict_filterval only; full expected N=87",
            "exploratory": "manual36 capturepallet11 PL-pool; descriptive only",
            "synthetic": (
                "order-free frame aggregates and channel-agnostic heatmap "
                "distributions only; no kp-id/PnP/yaw claims"
            ),
        },
        "conditioning_policy": {
            "continuous_pose_metrics": (
                "finite successful pose pairs only, with success/failure rates shown "
                "separately"
            ),
            "keypoint_error": (
                "finite GT-correspondence-valid rows; not conditioned on detection"
            ),
            "flip_coordinate": "matched-detected keypoints only",
            "missing_metadata": "null and never imputed",
        },
        "confidence_intervals": {
            "source": "summary.json bootstrap_95ci",
            "recomputed": False,
            "exact_passthrough": source_summary.get("bootstrap_95ci"),
        },
        "training_ablations": {
            "status": "BLOCKED",
            "reason": (
                "requires separately trained matched checkpoints/runs; frozen "
                "inference cannot establish causal training effects"
            ),
            "blocked_items": [
                "C0-C4 covariance-weighted pose training/inference",
                "decoder/loss retraining ablation",
                "kp5/centroid supervision ablation",
                "self-training causal ablation",
            ],
        },
        "table_row_count": len(table_rows),
        "table_sections": sorted(
            {str(row.get("section")) for row in table_rows}
        ),
        "figures": figure_info,
        "artifacts": list(OUTPUT_NAMES),
    }
    write_json_new(run_dir / "analysis_summary.json", analysis_summary)
    print(
        json.dumps(
            {
                "status": analysis_summary["status"],
                "run_dir": str(run_dir),
                "table_rows": len(table_rows),
                "artifacts": list(OUTPUT_NAMES),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
