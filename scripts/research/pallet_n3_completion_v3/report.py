"""Build the N3 completion tables, Korean report, and audit figures.

This module is deliberately CPU-only.  It consumes completion receipts and
frozen evaluation JSON, never checkpoints, and remains runnable while a fit or
an evaluation is absent.  Missing evidence is rendered as ``x``; ``NA`` is
reserved for a known zero denominator or a metric that the protocol declares
undefined.  Numeric zero is therefore never conflated with either state.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pallet-pose-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import common as C


SCHEMA = "pallet_n3_completion_v3_report_v1"
BACKBONES = ("YOLO", "DOPE", "ResNet-18")
EVAL_FILES = {"DOPE": "dope.json", "ResNet-18": "resnet18.json"}
TRAIN_KEYS = tuple(
    (backbone, seed)
    for backbone in ("DOPE", "RESNET18") for seed in C.SEEDS)
METRICS = (
    ("median_px", "2D median (px)", "matched_pooled_corner8_median_px"),
    ("p90_px", "2D P90 (px)", "matched_pooled_corner8_P90_px"),
    ("pck10", "PCK10 (fraction)", "full_PCK10_fraction"),
    ("e_sym", "E_sym", "E_sym"),
    ("t_median_cm", "T median (cm)", "pose_translation_cm_median"),
    ("r_median_deg", "R median (deg)", "pose_rotation_deg_median"),
    ("yaw_median_deg", "Yaw median (deg)", "pose_yaw_deg_median"),
    ("pose_coverage", "Pose coverage", "pose_coverage"),
)
EXTENDED_METRICS = (
    ("pck5", "PCK5 (fraction)", "full_PCK5_fraction"),
    ("pck20", "PCK20 (fraction)", "full_PCK20_fraction"),
    ("full_p90_px", "Penalty P90 (px)", "full_penalty_P90_px"),
    ("t_p90_cm", "T P90 (cm)", "pose_translation_cm_P90"),
    ("r_p90_deg", "R P90 (deg)", "pose_rotation_deg_P90"),
    ("yaw_p90_deg", "Yaw P90 (deg)", "pose_yaw_deg_P90"),
    ("iou3d_median", "IoU3D median", "pose_IoU3D_median"),
    ("addsym_auc", "ADDsym AUC", "pose_ADDsym_AUC_full"),
)
DENOMINATORS = (
    ("frames", "Frames"),
    ("matched_frames", "Matched frames"),
    ("full_supervised_corners", "GT corners"),
    ("observed_corners", "Observed corners"),
    ("pose_available_frames", "Pose frames"),
)
FIGURE_FILES = (
    "training_curves.png", "backbone_comparison.png", "runtime.png",
    "dev_overlays.png", "lifter.png", "subgroups_or_thresholds.png",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def cell(value: Any = None, *, status: str | None = None,
         reason: str | None = None) -> dict:
    """Create one lossless report cell.

    COMPLETE accepts numeric zero.  MISSING/BLOCKED are displayed as x, and
    NA is used only for a metric known to be undefined by its denominator or
    protocol.  Non-finite numbers are never serialized.
    """
    normalized = str(status or ("COMPLETE" if value is not None else "MISSING")).upper()
    if normalized in {"X", "MISSING", "INCOMPLETE"} or normalized.startswith("BLOCKED"):
        return {"value": None, "status": normalized, "display": "x", "reason": reason}
    if normalized in {"NA", "N/A", "UNDEFINED", "NOT_APPLICABLE"}:
        return {"value": None, "status": "NA", "display": "NA", "reason": reason}
    if value is None:
        return {"value": None, "status": "MISSING", "display": "x", "reason": reason}
    if isinstance(value, float) and not math.isfinite(value):
        return {"value": None, "status": "NA", "display": "NA",
                "reason": reason or "non-finite/undefined"}
    if isinstance(value, bool):
        display = "true" if value else "false"
    elif isinstance(value, int) or (isinstance(value, float) and value == 0):
        display = str(value)
    elif isinstance(value, float):
        display = f"{value:.6g}"
    else:
        display = str(value)
    return {"value": value, "status": normalized, "display": display, "reason": reason}


def _missing(reason: str) -> dict:
    return cell(status="MISSING", reason=reason)


def _na(reason: str) -> dict:
    return cell(status="NA", reason=reason)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(
        _json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    pending = path.with_name(path.name + ".pending")
    pending.write_text(text)
    pending.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def read_optional(path: Path, *, root: Path | None = None) -> dict:
    """Read JSON with an explicit inventory record instead of raising."""
    root = C.ROOT if root is None else Path(root)
    record = {"path": _relative(path, root), "present": path.is_file(),
              "status": "MISSING", "payload": None, "error": None}
    if not path.is_file():
        return record
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        record.update(status="INVALID", error=f"{type(exc).__name__}: {exc}")
        return record
    record.update(status="LOADED", payload=payload, sha256=_sha256(path),
                  bytes=path.stat().st_size)
    return record


def _inventory_record(record: Mapping[str, Any]) -> dict:
    return {key: record.get(key) for key in
            ("path", "present", "status", "sha256", "bytes", "error")
            if record.get(key) is not None or key in {"present", "status"}}


def load_inputs(doc: Path = C.DOC, raw: Path = C.RAW,
                *, root: Path = C.ROOT) -> dict:
    """Load only documented report inputs; absent files remain inventory rows."""
    doc, raw, root = Path(doc), Path(raw), Path(root)
    paths: dict[str, Path] = {
        "protocol": doc / "PROTOCOL.json",
        "environment_audit": doc / "ENVIRONMENT_AUDIT.json",
        "symmetry_activation_audit": doc / "SYMMETRY_ACTIVATION_AUDIT.json",
        "dimension_sensitivity_audit": doc / "DIMENSION_SENSITIVITY_AUDIT.json",
        "reuse": doc / "REUSE_RESULTS.json",
        "reuse_per_frame": raw / "reuse" / "PER_FRAME_SCORES.json",
        "eval_dope": raw / "evaluation" / "dope.json",
        "eval_resnet18": raw / "evaluation" / "resnet18.json",
        "square_yolo": doc / "SQUARE_YOLO_RESULTS.json",
        "square_dope": doc / "SQUARE_DOPE_RESULTS.json",
        "square_resnet18": doc / "SQUARE_RESNET18_RESULTS.json",
        "runtime_dope": doc / "RUNTIME_DOPE_SEED1.json",
        "runtime_resnet18": doc / "RUNTIME_RESNET18_SEED1.json",
        "runtime_raw_dope": raw / "runtime" / "dope_seed1.json",
        "runtime_raw_resnet18": raw / "runtime" / "resnet18_seed1.json",
        "lifter_receipt": doc / "LIFTER_YOLO_R0_N3_SEED1_COMPLETE.json",
        "lifter_raw": raw / "lifter" / "YOLO_R0_N3_SEED1_RAW.json",
        "pred_dope": raw / "predictions" / "dope_DEV319.json",
        "pred_resnet18": raw / "predictions" / "resnet18_DEV319.json",
    }
    for backbone, seed in TRAIN_KEYS:
        paths[f"train_{backbone.lower()}_{seed}"] = doc / f"TRAIN_{backbone}_SEED{seed}.json"
    for backbone in ("DOPE", "RESNET18"):
        paths[f"training_complete_{backbone.lower()}"] = doc / f"TRAINING_{backbone}_COMPLETE.json"
        paths[f"validation_{backbone.lower()}"] = doc / f"VALIDATION_{backbone}_COMPLETE.json"
        paths[f"selection_{backbone.lower()}"] = doc / f"SELECTION_{backbone}.json"
    return {key: read_optional(path, root=root) for key, path in paths.items()}


def _get(mapping: Any, *path: str) -> Any:
    current = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _headline_value(headline: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(headline, Mapping):
        return None
    if key in {"full_PCK5_fraction", "full_PCK10_fraction", "full_PCK20_fraction"}:
        value = headline.get(key)
        if value is None:
            threshold = key.removeprefix("full_PCK").removesuffix("_fraction")
            value = _get(headline, "full_PCK", threshold)
        if value is None:
            value = _get(headline, "PCK", threshold)
        return value
    return headline.get(key)


def _pose_value(method: Mapping[str, Any], key: str) -> Any:
    pose = _get(method, "result", "pose")
    if not isinstance(pose, Mapping):
        return None
    lookup = {
        "pose_translation_cm_median": ("translation_cm", "median"),
        "pose_translation_cm_P90": ("translation_cm", "P90"),
        "pose_rotation_deg_median": ("rotation_deg", "median"),
        "pose_rotation_deg_P90": ("rotation_deg", "P90"),
        "pose_yaw_deg_median": ("yaw_deg", "median"),
        "pose_yaw_deg_P90": ("yaw_deg", "P90"),
        "pose_IoU3D_median": ("IoU3D", "median"),
        "pose_ADDsym_AUC_full": ("ADDsym_AUC_full",),
        "pose_available_frames": ("denominators", "available_frames"),
        "pose_coverage": ("denominators", "coverage"),
    }
    return _get(pose, *lookup[key]) if key in lookup else None


def _entry_headline(entry: Any) -> Mapping[str, Any] | None:
    """Return a headline from any frozen result-family schema used here."""
    if not isinstance(entry, Mapping):
        return None
    for path in (("mean", "headline"), ("result", "headline"),
                 ("mean_headline",), ("headline",),
                 ("mean_of_seed_summaries",)):
        value = _get(entry, *path)
        if isinstance(value, Mapping):
            return value
    return None


def _evidence_cell(value: Any, *, available: bool, reason: str) -> dict:
    return cell(value) if available and value is not None else _missing(reason)


def _add_denominators(row: dict, headline: Mapping[str, Any] | None, *,
                      available: bool, reason: str) -> dict:
    for key, _ in DENOMINATORS:
        row[key] = _evidence_cell(
            headline.get(key) if isinstance(headline, Mapping) else None,
            available=available, reason=reason)
    return row


def _ci_cell(metric: Mapping[str, Any] | None, *, scale: float = 1.) -> dict:
    if not isinstance(metric, Mapping):
        return _missing("paired session-bootstrap metric is absent")
    status = str(metric.get("status", "MISSING"))
    interval = metric.get("CI95")
    if status != "COMPLETE" or not isinstance(interval, Sequence) or len(interval) != 2:
        return {"value": None, "status": status, "display": "x", "reason": (
            "paired session bootstrap did not yield a complete CI; "
            f"invalid_draws={metric.get('invalid_draws')}")}
    if not all(_finite(value) for value in interval):
        return _na("paired session-bootstrap CI is non-finite")
    values = [float(interval[0]) * scale, float(interval[1]) * scale]
    return cell(values)


def _delta_cell(metric: Mapping[str, Any] | None, *, scale: float = 1.) -> dict:
    value = metric.get("observed_delta") if isinstance(metric, Mapping) else None
    return cell(float(value) * scale) if _finite(value) else _missing(
        "paired observed delta is absent")


def _row_from_headline(backbone: str, method: str,
                       headline: Mapping[str, Any] | None, *,
                       status: str = "COMPLETE", source: str = "",
                       method_entry: Mapping[str, Any] | None = None,
                       seeds: int = 1) -> dict:
    row = {"backbone": cell(backbone), "method": cell(method),
           "seed_count": cell(seeds), "source": cell(source)}
    for name, _, key in (*METRICS, *EXTENDED_METRICS):
        value = _headline_value(headline, key)
        if value is None and method_entry is not None and key.startswith("pose_"):
            value = _pose_value(method_entry, key)
        if status != "COMPLETE":
            row[name] = cell(status=status, reason=f"{backbone} {method} unavailable")
        elif value is None:
            row[name] = _missing(f"{key} absent from source")
        else:
            row[name] = cell(value)
    _add_denominators(
        row, headline, available=status == "COMPLETE",
        reason=f"{backbone} {method} denominator unavailable")
    if (status == "COMPLETE" and method_entry is not None
            and row["pose_available_frames"]["status"] != "COMPLETE"):
        available_frames = _pose_value(method_entry, "pose_available_frames")
        if available_frames is not None:
            row["pose_available_frames"] = cell(available_frames)
    return row


def _seed_mean(eval_payload: Mapping[str, Any]) -> tuple[str, dict | None]:
    methods = eval_payload.get("methods")
    if not isinstance(methods, Mapping):
        return "MISSING", None
    entries = [methods.get(f"n3_seed{seed}") for seed in C.SEEDS]
    if any(not isinstance(entry, Mapping) or entry.get("status") != "COMPLETE"
           for entry in entries):
        return "INCOMPLETE", None
    values: dict[str, Any] = {}
    for _, _, key in (*METRICS, *EXTENDED_METRICS):
        observed = []
        for entry in entries:
            value = _headline_value(entry.get("headline"), key)
            if value is None and key.startswith("pose_"):
                value = _pose_value(entry, key)
            if not _finite(value):
                break
            observed.append(float(value))
        if len(observed) == len(entries):
            values[key] = statistics.fmean(observed)
    # Counts describe one preserved evaluation population, so they must agree
    # across fits.  Averaging inconsistent denominators would conceal drift.
    for key, _ in DENOMINATORS:
        observed = []
        for entry in entries:
            value = _headline_value(entry.get("headline"), key)
            if value is None and key == "pose_available_frames":
                value = _pose_value(entry, key)
            observed.append(value)
        if (all(_finite(value) for value in observed)
                and len({float(value) for value in observed}) == 1):
            values[key] = observed[0]
    return "COMPLETE", values


def backbone_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows = []
    reuse_record = inputs.get("reuse", {})
    candidate_reuse = reuse_record.get("payload") if reuse_record.get("status") == "LOADED" else None
    reuse = candidate_reuse if _get(candidate_reuse, "schema") == "pallet_n3_completion_v3_reuse_v1" else None
    source = reuse_record.get("path", "REUSE_RESULTS.json")
    for method, source_key, seeds in (("Base", "R0", 1), ("N3 seed mean", "N3_DIM_SYM", 3)):
        entry = _get(reuse, "contracts", "A", "methods", source_key)
        headline = _get(entry, "mean", "headline")
        status = "COMPLETE" if isinstance(headline, Mapping) else "MISSING"
        rows.append(_row_from_headline("YOLO", method, headline, status=status,
                                       source=source, seeds=seeds))
    for backbone, key in EVAL_FILES.items():
        record = inputs.get("eval_" + ("dope" if backbone == "DOPE" else "resnet18"), {})
        candidate = record.get("payload") if record.get("status") == "LOADED" else None
        payload = (candidate if isinstance(candidate, Mapping) and
                   candidate.get("schema") == "pallet_n3_completion_v3_evaluation_v1"
                   else None)
        base = _get(payload, "methods", "base")
        base_status = "COMPLETE" if isinstance(base, Mapping) and base.get("status") == "COMPLETE" else "MISSING"
        rows.append(_row_from_headline(
            backbone, "Base", base.get("headline") if isinstance(base, Mapping) else None,
            status=base_status, source=record.get("path", key), method_entry=base, seeds=1))
        mean_status, mean = _seed_mean(payload or {})
        rows.append(_row_from_headline(backbone, "N3 seed mean", mean,
                                       status=mean_status, source=record.get("path", key),
                                       seeds=3))
    return rows


def _square_yolo_rows(payload: Mapping[str, Any], source: str) -> list[dict]:
    mode = _get(payload, "modes", "manual_declared")
    families = mode.get("families") if isinstance(mode, Mapping) else None
    requested = (("R0", "Base", 1, "single"),
                 ("OLD_P", "OLD_P seed mean", 3, "mean_of_seed_summaries"),
                 ("N2_DIM_ONLY", "N2 seed mean", 3, "mean_of_seed_summaries"),
                 ("N3_DIM_SYM", "N3 seed mean", 3, "mean_of_seed_summaries"))
    if not isinstance(families, Mapping):
        return [_row_from_headline("YOLO", method, None, status="MISSING",
                                   source=source, seeds=seeds)
                for _, method, seeds, _ in requested]
    rows = []
    for family, method, seeds, summary_key in requested:
        raw_headline = _get(families, family, summary_key)
        headline = dict(raw_headline) if isinstance(raw_headline, Mapping) else None
        if isinstance(headline, dict):
            aliases = {
                "frames": headline.get("total_frames"),
                "matched_frames": headline.get("matched"),
                "full_supervised_corners": mode.get("manual_corner_denominator"),
            }
            for key, value in aliases.items():
                if headline.get(key) is None and value is not None:
                    headline[key] = value
        rows.append(_row_from_headline(
            "YOLO", method, headline,
            status="COMPLETE" if isinstance(headline, Mapping) else "MISSING",
            source=source, seeds=seeds))
    return rows


def square_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    output = []
    for backbone, key in (("YOLO", "square_yolo"), ("DOPE", "square_dope"),
                          ("ResNet-18", "square_resnet18")):
        record = inputs.get(key, {})
        candidate = record.get("payload") if record.get("status") == "LOADED" else None
        payload = candidate if isinstance(candidate, Mapping) and candidate.get("complete") is True else None
        if backbone == "YOLO" and isinstance(payload, Mapping):
            rows = _square_yolo_rows(payload, record.get("path", ""))
        elif isinstance(payload, Mapping):
            mode = _get(payload, "modes", "manual_declared") or {}
            base = _get(mode, "methods", "base")
            mean_status, mean = _seed_mean(mode)
            rows = [
                _row_from_headline(backbone, "Base",
                    base.get("headline") if isinstance(base, Mapping) else None,
                    status=("COMPLETE" if isinstance(base, Mapping) and
                            base.get("status") == "COMPLETE" else "MISSING"),
                    source=record.get("path", ""), method_entry=base),
                _row_from_headline(backbone, "N3 seed mean", mean,
                    status=mean_status, source=record.get("path", ""), seeds=3),
            ]
        else:
            rows = [_row_from_headline(backbone, method, None, status="MISSING",
                                       source=record.get("path", key), seeds=seeds)
                    for method, seeds in (("Base", 1), ("N3 seed mean", 3))]
        for row in rows:
            blocked = "GREEN0918_119 has no independent canonical pose reference"
            for key in ("t_median_cm", "r_median_deg", "yaw_median_deg",
                        "pose_coverage", "t_p90_cm", "r_p90_deg",
                        "yaw_p90_deg", "iou3d_median", "addsym_auc",
                        "pose_available_frames"):
                row[key] = cell(status="BLOCKED_REFERENCE", reason=blocked)
        output.extend(rows)
    return output


def training_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows = []
    for backbone, seed in TRAIN_KEYS:
        record = inputs.get(f"train_{backbone.lower()}_{seed}", {})
        payload = record.get("payload") if record.get("status") == "LOADED" else None
        valid = (isinstance(payload, Mapping) and payload.get("complete") is True
                 and payload.get("smoke") is False and payload.get("seed") == seed
                 and str(payload.get("backbone", "")).upper() == backbone
                 and payload.get("steps") == C.STEPS)
        if valid:
            rows.append({
                "backbone": cell("ResNet-18" if backbone == "RESNET18" else backbone),
                "seed": cell(seed), "status": cell("COMPLETE"),
                "steps": cell(payload.get("steps")),
                "exposures": cell(payload.get("exposures")),
                "elapsed_seconds": cell(payload.get("elapsed_seconds")),
                "trainable_parameters": cell(payload.get("trainable_parameters")),
                "dimension_input_to_N3": cell(payload.get("dimension_input_to_N3")),
                "symmetry_supervision": cell(payload.get("symmetry_supervision")),
                "base_receives_dimensions": cell(payload.get("base_receives_dimensions")),
                "checkpoint_sha256": cell(_get(payload, "checkpoint", "sha256")),
                "source": cell(record.get("path", "")),
            })
        else:
            reason = "fit receipt missing or not a complete non-smoke 6000-step fit"
            rows.append({"backbone": cell("ResNet-18" if backbone == "RESNET18" else backbone),
                         "seed": cell(seed), "status": _missing(reason),
                         "steps": _missing(reason), "elapsed_seconds": _missing(reason),
                         "exposures": _missing(reason),
                         "trainable_parameters": _missing(reason),
                         "dimension_input_to_N3": _missing(reason),
                         "symmetry_supervision": _missing(reason),
                         "base_receives_dimensions": _missing(reason),
                         "checkpoint_sha256": _missing(reason),
                         "source": cell(record.get("path", ""))})
    return rows


def runtime_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows = []
    for backbone, key in (("DOPE", "runtime_dope"),
                          ("ResNet-18", "runtime_resnet18")):
        record = inputs.get(key, {})
        candidate = record.get("payload") if record.get("status") == "LOADED" else None
        payload = candidate if isinstance(candidate, Mapping) and candidate.get("complete") is True else None
        summary = payload.get("summary") if isinstance(payload, Mapping) else None
        for path_name, label in (("base_e2e", "Base E2E"),
                                 ("n3_seed1_e2e", "Base + N3 E2E"),
                                 ("n3_seed1_only", "N3 only")):
            timing = _get(summary, path_name, "cuda_event_ms")
            status = "COMPLETE" if isinstance(timing, Mapping) else "MISSING"
            def timing_cell(metric: str) -> dict:
                value = timing.get(metric) if isinstance(timing, Mapping) else None
                return cell(value, status=status, reason=f"{key}/{path_name} unavailable")
            row = {
                "backbone": cell(backbone), "path": cell(label),
                "median_ms": timing_cell("median_ms"),
                "p90_ms": timing_cell("p90_ms"),
                "fps": timing_cell("fps_from_median"),
                "params": cell(_get(payload, "parameters", {
                    "base_e2e": "base_total",
                    "n3_seed1_e2e": "base_plus_n3_total",
                    "n3_seed1_only": "n3_total",
                }[path_name]), status=status,
                               reason="parameter count unavailable"),
                "peak_allocated_bytes": cell(_get(summary, path_name, "peak_allocated_bytes"),
                                              status=status, reason="peak memory unavailable"),
                "source": cell(record.get("path", "")),
            }
            rows.append(row)
    return rows


def selection_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows = []
    for backbone, key in (("DOPE", "selection_dope"),
                          ("ResNet-18", "selection_resnet18")):
        record = inputs.get(key, {})
        payload = record.get("payload") if record.get("status") == "LOADED" else None
        valid = (isinstance(payload, Mapping)
                 and payload.get("schema") == "pallet_n3_completion_v3_selection_v1"
                 and payload.get("complete") is True
                 and payload.get("real_selection") is False)
        for seed in C.SEEDS:
            temperature = _get(payload, "temperatures", str(seed)) if valid else None
            status = "COMPLETE" if valid and _finite(temperature) else "MISSING"
            rows.append({
                "backbone": cell(backbone), "seed": cell(seed),
                "temperature": cell(temperature, status=status,
                                    reason="synthetic calibration receipt unavailable"),
                "lambda": cell(_get(payload, "fixed_rule", "lam"), status=status,
                               reason="fixed cap rule unavailable"),
                "cap_fraction": cell(_get(payload, "fixed_rule",
                                           "max_move_image_diagonal_fraction"), status=status,
                                     reason="fixed cap rule unavailable"),
                "real_selection": cell(payload.get("real_selection") if valid else None,
                                       status=status, reason="selection receipt unavailable"),
                "source": cell(record.get("path", "")),
            })
    return rows


def lifter_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    record = inputs.get("lifter_receipt", {})
    candidate = record.get("payload") if record.get("status") == "LOADED" else None
    payload = candidate if isinstance(candidate, Mapping) and candidate.get("complete") is True else None
    stats = payload.get("statistics") if isinstance(payload, Mapping) else None
    rows = []
    for method in ("R0", "N3_seed1"):
        value = stats.get(method) if isinstance(stats, Mapping) else None
        coverage = _get(value, "overall", "coverage")
        jitter = _get(value, "overall", "yaw_wrap_jitter",
                      "available_outputs_including_held")
        missing = _get(value, "overall", "missing")
        status = "COMPLETE" if isinstance(coverage, Mapping) else "MISSING"
        def got(source: Any, key: str, *, na_if_zero: bool = False) -> dict:
            number = source.get(key) if isinstance(source, Mapping) else None
            if number is None and na_if_zero and isinstance(source, Mapping):
                return _na(f"{key} undefined because there are no adjacent valid pairs")
            return cell(number, status=status, reason="lifter receipt unavailable")
        longest = None
        if isinstance(missing, Mapping):
            longest = missing.get("longest_until_next_sample_sensor_time_run")
            if not isinstance(longest, Mapping):
                longest = missing.get("longest_observed_sensor_time_run")
            if isinstance(longest, Mapping):
                longest = longest.get("until_next_sample_s",
                                      longest.get("observed_span_s"))
        rows.append({
            "method": cell(method),
            "frames": got(coverage, "frames"),
            "available_fraction": got(coverage, "available_fraction"),
            "fresh_fraction": got(coverage, "fresh_fraction"),
            "longest_missing_s": cell(longest, status=status,
                                      reason="missing-run statistic unavailable"),
            "yaw_step_median_deg": got(jitter, "median_abs_step", na_if_zero=True),
            "yaw_step_p90_deg": got(jitter, "p90_abs_step", na_if_zero=True),
            "independent_accuracy": _missing(
                "visible state and independent position/yaw reference are unavailable"),
            "source": cell(record.get("path", "")),
        })
    return rows


def subgroup_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    record = inputs.get("reuse", {})
    candidate = record.get("payload") if record.get("status") == "LOADED" else None
    reuse = candidate if _get(candidate, "schema") == "pallet_n3_completion_v3_reuse_v1" else None
    rows = []
    for contract, group_key, groups in (
        ("C", "material", ("plastic", "wood")),
        ("D", "occlusion", ("clean", "moderate", "severe", "unclassified")),
    ):
        counts = _get(reuse, "contracts", contract, "frame_counts") or {}
        for group in groups:
            for method, source_method, seeds in (
                    ("Base", "R0", 1), ("OLD_P seed mean", "OLD_P", 3),
                    ("N2 seed mean", "N2_DIM_ONLY", 3),
                    ("N3 seed mean", "N3_DIM_SYM", 3)):
                headline = _get(reuse, "contracts", contract, "methods", source_method,
                                group_key, group)
                status = "COMPLETE" if isinstance(headline, Mapping) else "MISSING"
                row = {"group_type": cell(group_key), "group": cell(group),
                       "method": cell(method), "seed_count": cell(seeds),
                       "frames": cell(counts.get(group), status=status,
                                      reason="subgroup absent")}
                for name, _, key in METRICS:
                    value = _headline_value(headline, key)
                    row[name] = cell(value, status=status,
                                     reason=f"{group_key}/{group} unavailable")
                _add_denominators(row, headline, available=status == "COMPLETE",
                                  reason=f"{group_key}/{group} denominator unavailable")
                rows.append(row)
    return rows


def _reuse_payload(inputs: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any] | None:
    record = inputs.get("reuse", {})
    payload = record.get("payload") if record.get("status") == "LOADED" else None
    return (payload if isinstance(payload, Mapping) and
            payload.get("schema") == "pallet_n3_completion_v3_reuse_v1" else None)


def ablation_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    reuse = _reuse_payload(inputs)
    source = inputs.get("reuse", {}).get("path", "REUSE_RESULTS.json")
    rows = []
    specs = (
        ("N0_BASE_REPLAY", "N0: replay local refiner", False, False),
        ("N1_SYM_ONLY", "N1: symmetry only", False, True),
        ("N2_DIM_ONLY", "N2: dimensions only", True, False),
        ("N3_DIM_SYM", "N3: dimensions + symmetry", True, True),
    )
    for key, label, dimensions, symmetry in specs:
        entry = _get(reuse, "contracts", "B", "methods", key)
        headline = _entry_headline(entry)
        available = isinstance(headline, Mapping)
        row = _row_from_headline(
            "YOLO", label, headline, status="COMPLETE" if available else "MISSING",
            source=source, method_entry=entry, seeds=3)
        row.update(dimension_input=cell(dimensions), symmetry_supervision=cell(symmetry))
        _add_denominators(row, headline, available=available,
                          reason=f"contract B {key} is unavailable")
        rows.append(row)
    return rows


def ablation_delta_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    reuse = _reuse_payload(inputs)
    deltas = _get(reuse, "contracts", "B", "mean_metric_deltas")
    specs = (
        ("N2_minus_N0", "N2 - N0", "dimension contribution under fixed supervision"),
        ("N1_minus_N0", "N1 - N0", "symmetry-only contribution"),
        ("N3_minus_N2", "N3 - N2", "increment from symmetry with dimensions"),
        ("N3_minus_N1", "N3 - N1", "increment from dimensions with symmetry"),
    )
    rows = []
    for key, label, role in specs:
        entry = deltas.get(key) if isinstance(deltas, Mapping) else None
        available = isinstance(entry, Mapping)
        reason = f"contract B delta {key} is unavailable"
        def got(metric: str, scale: float = 1.) -> dict:
            value = entry.get(metric) if isinstance(entry, Mapping) else None
            return (cell(float(value) * scale) if available and _finite(value)
                    else _missing(reason))
        rows.append({
            "contrast": cell(label), "interpretation_role": cell(role),
            "median_delta_px": got("matched_pooled_corner8_median_px"),
            "p90_delta_px": got("matched_pooled_corner8_P90_px"),
            "pck10_delta_pp": got("full_PCK10_fraction", 100.),
            "e_sym_delta": got("E_sym"),
        })
    return rows


def cap_damage_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    reuse = _reuse_payload(inputs)
    rows = []
    for method, label in (("N2_DIM_ONLY", "N2"), ("N3_DIM_SYM", "N3")):
        for mode, cap_label in (("cap1pct", "1%"), ("cap2pct", "2%"),
                                ("no_output_cap", "none")):
            entry = _get(reuse, "contracts", "E", "methods", method, mode)
            headline = _get(entry, "mean_headline")
            damage = _get(entry, "mean_damage_counts")
            available = (isinstance(headline, Mapping) and isinstance(damage, Mapping)
                         and entry.get("status") == "COMPLETE") if isinstance(entry, Mapping) else False
            reason = f"contract E {method}/{mode} is unavailable"
            def got(path: Sequence[str], *, na_without_cap: bool = False) -> dict:
                value = _get(damage, *path)
                if na_without_cap and mode == "no_output_cap" and value is None:
                    return _na("undefined when no output cap is applied")
                return _evidence_cell(value, available=available, reason=reason)
            row = _row_from_headline(
                "YOLO", label, headline,
                status="COMPLETE" if available else "MISSING",
                source=inputs.get("reuse", {}).get("path", ""), seeds=3)
            row.update({
                "cap": cell(cap_label),
                "inside_cap_corners": got(("initial_error_inside_cap_corners",),
                                           na_without_cap=True),
                "outside_cap_corners": got(("initial_error_outside_cap_corners",),
                                            na_without_cap=True),
                "cap_hit_corners": got(("cap_hit_corners",), na_without_cap=True),
                "improved_corners": got(("corner_change", "improved")),
                "no_change_corners": got(("corner_change", "no_change")),
                "worsened_corners": got(("corner_change", "worsened")),
                "improved_frames": got(("frame_change", "improved")),
                "no_change_frames": got(("frame_change", "no_change")),
                "worsened_frames": got(("frame_change", "worsened")),
                "good5_to_bad10": got(("good5_to_bad10_corners",)),
                "bad20_to_good10": got(("bad20_to_good10_corners",)),
                "movement_median_px": got(("movement", "median_px")),
                "movement_p90_px": got(("movement", "P90_px")),
                "cap_violations": got(("movement", "cap_violations")),
                "triangle_violations": got(("triangle_lower_bound_violations",)),
                "comparable_frames": got(("matched_comparable_frames",)),
                "comparable_corners": got(("observed_comparable_corners",)),
            })
            rows.append(row)
    return rows


def paired_2d_pose_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    reuse = _reuse_payload(inputs)
    rows = []
    for method, label in (("N2_DIM_ONLY", "N2"), ("N3_DIM_SYM", "N3")):
        seed_rows = _get(reuse, "contracts", "E", "methods", method,
                         "cap1pct", "per_seed")
        for seed in C.SEEDS:
            entry = (seed_rows.get(str(seed), seed_rows.get(seed))
                     if isinstance(seed_rows, Mapping) else None)
            paired = _get(entry, "paired_2d_pose")
            damage = _get(entry, "fixed_base_branch_damage")
            available = isinstance(paired, Mapping) and isinstance(damage, Mapping)
            reason = f"contract E {method} seed {seed} paired result is unavailable"
            def got(source: Mapping[str, Any] | None, *path: str) -> dict:
                return _evidence_cell(_get(source, *path), available=available, reason=reason)
            joint = paired.get("joint_direction_counts", {}) if isinstance(paired, Mapping) else {}
            rows.append({
                "method": cell(label), "seed": cell(seed), "cap": cell("1%"),
                "paired_frames": got(paired, "paired_with_fixed_branch_2d_frames"),
                "frame_2d_improved": got(damage, "frame_change", "improved"),
                "frame_2d_worsened": got(damage, "frame_change", "worsened"),
                "translation_delta_median_cm": got(paired, "translation_cm", "median_delta"),
                "translation_improved": got(paired, "translation_cm", "improved"),
                "translation_no_change": got(paired, "translation_cm", "no_change"),
                "translation_worsened": got(paired, "translation_cm", "worsened"),
                "rotation_delta_median_deg": got(paired, "rotation_deg", "median_delta"),
                "rotation_improved": got(paired, "rotation_deg", "improved"),
                "rotation_no_change": got(paired, "rotation_deg", "no_change"),
                "rotation_worsened": got(paired, "rotation_deg", "worsened"),
                "yaw_delta_median_deg": got(paired, "yaw_deg", "median_delta"),
                "yaw_improved": got(paired, "yaw_deg", "improved"),
                "yaw_no_change": got(paired, "yaw_deg", "no_change"),
                "yaw_worsened": got(paired, "yaw_deg", "worsened"),
                "2d_better_t_better": _evidence_cell(
                    joint.get("2d_improved__translation_improved"),
                    available=available, reason=reason),
                "2d_better_t_worse": _evidence_cell(
                    joint.get("2d_improved__translation_worsened"),
                    available=available, reason=reason),
                "2d_worse_t_better": _evidence_cell(
                    joint.get("2d_worsened__translation_improved"),
                    available=available, reason=reason),
                "2d_worse_t_worse": _evidence_cell(
                    joint.get("2d_worsened__translation_worsened"),
                    available=available, reason=reason),
                "new_pose_failures": got(paired, "new_pose_failures"),
                "pose_recoveries": got(paired, "pose_recoveries"),
            })
    return rows


def comparator_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    reuse = _reuse_payload(inputs)
    source = inputs.get("reuse", {}).get("path", "REUSE_RESULTS.json")
    specs = (
        ("A", "R0", "R0: no refiner", "initial estimate", 1),
        ("A", "OLD_P", "P: local distribution", "dimension-free refiner", 3),
        ("H", "D", "D: direct regression", "output/loss package", 3),
        ("H", "L", "L: line structure", "structure package", 3),
        ("H", "PoseFix", "PoseFix-style", "image-pose refiner package", 3),
        ("A", "N3_DIM_SYM", "N3: dimensions + symmetry", "proposed package", 3),
    )
    rows = []
    for contract, key, label, role, seeds in specs:
        entry = _get(reuse, "contracts", contract, "methods", key)
        headline = _entry_headline(entry)
        available = isinstance(headline, Mapping)
        row = _row_from_headline(
            "YOLO", label, headline, status="COMPLETE" if available else "MISSING",
            source=source, method_entry=entry, seeds=seeds)
        row["comparison_role"] = cell(role)
        _add_denominators(row, headline, available=available,
                          reason=f"contract {contract} {key} is unavailable")
        rows.append(row)
    return rows


def update_alternative_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    reuse = _reuse_payload(inputs)
    source = inputs.get("reuse", {}).get("path", "REUSE_RESULTS.json")
    contract = _get(reuse, "contracts", "I")
    safe = _get(contract, "safe_common_cohort")
    population = safe.get("population") if isinstance(safe, Mapping) else None
    exposure = safe.get("exposure") if isinstance(safe, Mapping) else None
    rows = []
    specs = (
        ("R0", "R0", 1), ("source_only_update", "Synthetic-only update", 1),
        ("raw_pseudo_student", "Raw pseudo-label student", 1),
        ("corrected_pseudo_student", "Corrected pseudo-label student", 1),
        ("R0_plus_N3", "R0 + N3 seed mean", 3),
    )
    for key, label, seeds in specs:
        entry = _get(safe, "methods", key)
        headline = _entry_headline(entry)
        available = isinstance(headline, Mapping)
        row = _row_from_headline(
            "YOLO", label, headline, status="COMPLETE" if available else "MISSING",
            source=source, method_entry=entry, seeds=seeds)
        row.update(population=cell(population or "HELDOUT128 safe common cohort"),
                   evidence_role=cell("reused development safe cohort; not independent TEST"),
                   student_rgb_overlap=cell(_get(exposure, "student_train_RGB_overlap")),
                   teacher_session_overlap=cell(_get(exposure, "teacher_session_overlap")),
                   independent_test=cell(_get(exposure, "independent_test")),
                   selection_history=cell(_get(exposure, "LR5_selection_history")))
        _add_denominators(row, headline, available=available,
                          reason=f"contract I safe cohort {key} is unavailable")
        rows.append(row)
    blocked = _get(contract, "DEV319_table")
    blocked_reason = (_get(blocked, "reason") or
                      "No common exposure-safe DEV319 panel for update alternatives")
    for key, label, _ in specs[1:4]:
        row = _row_from_headline(
            "YOLO", label, None, status="BLOCKED_CONTRACT", source=source,
            seeds=1)
        row.update(population=cell("DEV319"),
                   evidence_role=cell(status="BLOCKED_CONTRACT", reason=blocked_reason),
                   student_rgb_overlap=cell(status="BLOCKED_CONTRACT", reason=blocked_reason),
                   teacher_session_overlap=cell(status="BLOCKED_CONTRACT", reason=blocked_reason),
                   independent_test=cell(status="BLOCKED_CONTRACT", reason=blocked_reason),
                   selection_history=cell(status="BLOCKED_CONTRACT", reason=blocked_reason))
        for name, _, _ in METRICS:
            row[name] = cell(status="BLOCKED_CONTRACT", reason=blocked_reason)
        _add_denominators(row, None, available=False, reason=blocked_reason)
        rows.append(row)
    return rows


def backbone_seed_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows = []
    for backbone, key in EVAL_FILES.items():
        record_key = "eval_dope" if backbone == "DOPE" else "eval_resnet18"
        record = inputs.get(record_key, {})
        payload = record.get("payload") if record.get("status") == "LOADED" else None
        methods = payload.get("methods") if isinstance(payload, Mapping) else None
        for method_key, label, seed in (("base", "Base", 0),
                                         ("n3_seed1", "N3", 1),
                                         ("n3_seed2", "N3", 2),
                                         ("n3_seed3", "N3", 3)):
            entry = methods.get(method_key) if isinstance(methods, Mapping) else None
            headline = entry.get("headline") if isinstance(entry, Mapping) else None
            available = (isinstance(headline, Mapping) and
                         entry.get("status") == "COMPLETE") if isinstance(entry, Mapping) else False
            row = _row_from_headline(
                backbone, label, headline,
                status="COMPLETE" if available else "MISSING",
                source=record.get("path", key), method_entry=entry,
                seeds=1)
            row["seed"] = cell("base" if seed == 0 else seed)
            rows.append(row)
    return rows


def backbone_ci_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows = []
    metric_specs = (
        ("median", "matched_pooled_corner8_median_px", 1.),
        ("p90", "matched_pooled_corner8_P90_px", 1.),
        ("pck10_pp", "full_PCK10_fraction", 100.),
        ("e_sym", "E_sym", 1.),
    )
    for backbone, key in EVAL_FILES.items():
        record_key = "eval_dope" if backbone == "DOPE" else "eval_resnet18"
        payload = inputs.get(record_key, {}).get("payload")
        for seed in C.SEEDS:
            comparison = _get(payload, "comparisons", f"base_to_n3_seed{seed}", "result")
            bootstrap = _get(comparison, "bootstrap")
            locked = (isinstance(bootstrap, Mapping)
                      and bootstrap.get("direction") == "candidate_minus_base"
                      and bootstrap.get("unit") == "session"
                      and bootstrap.get("resamples") == 10000
                      and bootstrap.get("seed") == 20260917
                      and bootstrap.get("paired_sampling") is True
                      and bootstrap.get("recalculates_each_statistic") is True)
            reason = ("paired bootstrap unavailable" if not isinstance(bootstrap, Mapping)
                      else "paired bootstrap does not match the locked session/10000/20260917 contract")
            row = {
                "backbone": cell(backbone), "seed": cell(seed),
                "unit": _evidence_cell(_get(bootstrap, "unit"),
                                        available=locked, reason=reason),
                "sessions": _evidence_cell(_get(bootstrap, "sessions"),
                                            available=locked, reason=reason),
                "resamples": _evidence_cell(_get(bootstrap, "resamples"),
                                             available=locked, reason=reason),
                "bootstrap_seed": _evidence_cell(_get(bootstrap, "seed"),
                                                  available=locked, reason=reason),
            }
            for output_key, metric_key, scale in metric_specs:
                metric = _get(bootstrap, "metrics", metric_key) if locked else None
                if locked:
                    row[f"{output_key}_delta"] = _delta_cell(metric, scale=scale)
                    row[f"{output_key}_ci95"] = _ci_cell(metric, scale=scale)
                else:
                    row[f"{output_key}_delta"] = cell(
                        status="BLOCKED_CONTRACT", reason=reason)
                    row[f"{output_key}_ci95"] = cell(
                        status="BLOCKED_CONTRACT", reason=reason)
            rows.append(row)
    return rows


def symmetry_activation_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    payload = inputs.get("symmetry_activation_audit", {}).get("payload")
    valid = (isinstance(payload, Mapping)
             and payload.get("schema") ==
             "pallet_n3_completion_v3_symmetry_activation_audit_v1"
             and payload.get("complete") is True)
    rows = []
    for backbone, label in (("dope", "DOPE"), ("resnet18", "ResNet-18")):
        block = _get(payload, "backbones", backbone) if valid else None
        unique = _get(block, "unique_usable_rows")
        row = {
            "backbone": cell(label),
            "usable_rows": _evidence_cell(
                _get(unique, "rows_or_exposures"), available=valid,
                reason="symmetry activation audit unavailable"),
            "unique_non_identity": _evidence_cell(
                _get(unique, "non_identity_count"), available=valid,
                reason="symmetry activation audit unavailable"),
            "unique_non_identity_fraction": _evidence_cell(
                _get(unique, "non_identity_fraction"), available=valid,
                reason="symmetry activation audit unavailable"),
        }
        for seed in C.SEEDS:
            seed_row = _get(block, "seeds", str(seed))
            row[f"seed{seed}_exposures"] = _evidence_cell(
                _get(seed_row, "rows_or_exposures"), available=valid,
                reason="symmetry activation audit unavailable")
            row[f"seed{seed}_non_identity"] = _evidence_cell(
                _get(seed_row, "non_identity_count"), available=valid,
                reason="symmetry activation audit unavailable")
        rows.append(row)
    return rows


def dimension_sensitivity_rows(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    payload = inputs.get("dimension_sensitivity_audit", {}).get("payload")
    valid = (isinstance(payload, Mapping)
             and payload.get("schema") ==
             "pallet_n3_completion_v3_dimension_sensitivity_audit_v1"
             and payload.get("complete") is True)
    rows = []
    for backbone, label in (("dope", "DOPE"), ("resnet18", "ResNet-18")):
        for seed in C.SEEDS:
            entry = _get(payload, "fits", f"{backbone}_seed{seed}") if valid else None
            passed = isinstance(entry, Mapping) and entry.get("status") == "PASS"
            reason = f"dimension sensitivity audit unavailable for {backbone} seed {seed}"
            rows.append({
                "backbone": cell(label), "seed": cell(seed),
                "status": _evidence_cell(
                    entry.get("status") if isinstance(entry, Mapping) else None,
                    available=passed, reason=reason),
                "visual_inputs_identical": _evidence_cell(
                    _get(entry, "visual_inputs_bitwise_identical"),
                    available=passed, reason=reason),
                "base_logits_identical": _evidence_cell(
                    _get(entry, "base_logits_bitwise_identical"),
                    available=passed, reason=reason),
                "changed_logits": _evidence_cell(
                    _get(entry, "changed_logit_entries"),
                    available=passed, reason=reason),
                "max_abs_logit_delta": _evidence_cell(
                    _get(entry, "max_abs_logit_delta"),
                    available=passed, reason=reason),
                "checkpoint_sha256": _evidence_cell(
                    _get(entry, "checkpoint_sha256"),
                    available=passed, reason=reason),
            })
    return rows


def _columns(*items: tuple[str, str]) -> list[dict]:
    return [{"key": key, "label": label} for key, label in items]


def build_tables(inputs: Mapping[str, Mapping[str, Any]]) -> dict:
    metric_columns = [(name, label) for name, label, _ in METRICS]
    extended_metric_columns = [(name, label) for name, label, _ in EXTENDED_METRICS]
    denominator_columns = list(DENOMINATORS)
    tables = {
        "backbone_dev": {
            "title": "Reused DEV319: base to N3",
            "columns": _columns(("backbone", "Backbone"), ("method", "Method"),
                                ("seed_count", "Seeds"), *metric_columns,
                                *extended_metric_columns,
                                *denominator_columns),
            "rows": backbone_rows(inputs),
            "note": ("N3 is the arithmetic mean of seed-level statistics, not an ensemble. "
                     "Absolute backbone ranking is not valid because conditional match sets differ."),
        },
        "backbone_dev_per_seed": {
            "title": "DOPE / ResNet-18 DEV319 per-seed results",
            "columns": _columns(("backbone", "Backbone"), ("method", "Method"),
                                ("seed", "Seed"), *metric_columns,
                                *extended_metric_columns,
                                *denominator_columns),
            "rows": backbone_seed_rows(inputs),
            "note": "Base is listed once per backbone; N3 seeds are independent fits, never an ensemble.",
        },
        "backbone_paired_ci": {
            "title": "DOPE / ResNet-18 paired session-bootstrap deltas (N3 - Base)",
            "columns": _columns(
                ("backbone", "Backbone"), ("seed", "Seed"),
                ("sessions", "Sessions"), ("resamples", "Resamples"),
                ("bootstrap_seed", "Bootstrap seed"),
                ("median_delta", "Median delta (px)"),
                ("median_ci95", "Median delta CI95"),
                ("p90_delta", "P90 delta (px)"),
                ("p90_ci95", "P90 delta CI95"),
                ("pck10_pp_delta", "PCK10 delta (pp)"),
                ("pck10_pp_ci95", "PCK10 delta CI95"),
                ("e_sym_delta", "E_sym delta"),
                ("e_sym_ci95", "E_sym delta CI95")),
            "rows": backbone_ci_rows(inputs),
            "note": ("10,000 whole-session paired resamples, seed 20260917. "
                     "Negative is favorable except PCK10, where positive is favorable; no multiplicity adjustment."),
        },
        "symmetry_activation": {
            "title": "Actual whole-object symmetry target activation",
            "columns": _columns(
                ("backbone", "Backbone"), ("usable_rows", "Usable source rows"),
                ("unique_non_identity", "Non-identity rows"),
                ("unique_non_identity_fraction", "Non-identity fraction"),
                ("seed1_exposures", "Seed1 exposures"),
                ("seed1_non_identity", "Seed1 non-identity"),
                ("seed2_exposures", "Seed2 exposures"),
                ("seed2_non_identity", "Seed2 non-identity"),
                ("seed3_exposures", "Seed3 exposures"),
                ("seed3_non_identity", "Seed3 non-identity")),
            "rows": symmetry_activation_rows(inputs),
            "note": ("The objective is wired for both backbones, but target-branch activation is empirical. "
                     "Zero means measured zero, not x."),
        },
        "dimension_sensitivity": {
            "title": "Trained N3 dimension-path sensitivity with fixed visual evidence",
            "columns": _columns(
                ("backbone", "Backbone"), ("seed", "Seed"),
                ("status", "Audit"),
                ("visual_inputs_identical", "Visual inputs identical"),
                ("base_logits_identical", "Base logits identical"),
                ("changed_logits", "Changed logits"),
                ("max_abs_logit_delta", "Max absolute delta"),
                ("checkpoint_sha256", "Checkpoint SHA-256")),
            "rows": dimension_sensitivity_rows(inputs),
            "note": ("Only registered W,D,H changes. This proves the trained metadata path is active, "
                     "not the causal accuracy gain of dimensions."),
        },
        "yolo_ablation": {
            "title": "YOLO controlled N0/N1/N2/N3 ablation",
            "columns": _columns(
                ("method", "Method"), ("dimension_input", "Dimensions"),
                ("symmetry_supervision", "Symmetry supervision"),
                *metric_columns, *extended_metric_columns, *denominator_columns),
            "rows": ablation_rows(inputs),
            "note": ("N0/N1/N2/N3 use the locked eight-corner evaluator. "
                     "Differences must be read metric by metric; tiny decimal changes are not a universal gain."),
        },
        "yolo_ablation_deltas": {
            "title": "YOLO controlled ablation contrasts",
            "columns": _columns(
                ("contrast", "Contrast"), ("interpretation_role", "Role"),
                ("median_delta_px", "Median delta (px)"),
                ("p90_delta_px", "P90 delta (px)"),
                ("pck10_delta_pp", "PCK10 delta (pp)"),
                ("e_sym_delta", "E_sym delta")),
            "rows": ablation_delta_rows(inputs),
            "note": ("Each value is the first named method minus the second. "
                     "Negative is favorable for error metrics and positive is favorable for PCK10."),
        },
        "yolo_cap_damage": {
            "title": "YOLO cap, damage, and recovery audit (seed-statistic mean)",
            "columns": _columns(
                ("method", "Method"), ("cap", "Output cap"),
                ("median_px", "2D median (px)"), ("p90_px", "2D P90 (px)"),
                ("pck10", "PCK10"), ("inside_cap_corners", "Initial inside cap"),
                ("outside_cap_corners", "Initial outside cap"),
                ("cap_hit_corners", "Cap hits"),
                ("improved_corners", "Corners improved"),
                ("no_change_corners", "Corners unchanged"),
                ("worsened_corners", "Corners worsened"),
                ("improved_frames", "Frames improved"),
                ("no_change_frames", "Frames unchanged"),
                ("worsened_frames", "Frames worsened"),
                ("good5_to_bad10", "<5 to >10 px"),
                ("bad20_to_good10", ">20 to <10 px"),
                ("movement_median_px", "Move median (px)"),
                ("movement_p90_px", "Move P90 (px)"),
                ("cap_violations", "Cap violations"),
                ("triangle_violations", "Bound violations"),
                ("comparable_frames", "Comparable frames"),
                ("comparable_corners", "Comparable corners")),
            "rows": cap_damage_rows(inputs),
            "note": ("Counts are arithmetic means of three seed-level counts and may therefore be fractional. "
                     "The fixed base symmetry branch is retained for damage/recovery diagnosis."),
        },
        "yolo_paired_2d_pose": {
            "title": "YOLO paired 2D-to-pose direction audit at the locked 1% cap",
            "columns": _columns(
                ("method", "Method"), ("seed", "Seed"),
                ("paired_frames", "Paired frames"),
                ("frame_2d_improved", "2D improved"),
                ("frame_2d_worsened", "2D worsened"),
                ("translation_delta_median_cm", "T median delta (cm)"),
                ("translation_improved", "T improved"),
                ("translation_no_change", "T unchanged"),
                ("translation_worsened", "T worsened"),
                ("rotation_delta_median_deg", "R median delta (deg)"),
                ("rotation_improved", "R improved"),
                ("rotation_no_change", "R unchanged"),
                ("rotation_worsened", "R worsened"),
                ("yaw_delta_median_deg", "Yaw median delta (deg)"),
                ("yaw_improved", "Yaw improved"),
                ("yaw_no_change", "Yaw unchanged"),
                ("yaw_worsened", "Yaw worsened"),
                ("2d_better_t_better", "2D+ / T+"),
                ("2d_better_t_worse", "2D+ / T-"),
                ("2d_worse_t_better", "2D- / T+"),
                ("2d_worse_t_worse", "2D- / T-"),
                ("new_pose_failures", "New pose failures"),
                ("pose_recoveries", "Pose recoveries")),
            "rows": paired_2d_pose_rows(inputs),
            "note": ("Deltas are candidate minus base; negative T/R/yaw deltas are favorable. "
                     "These are paired descriptive counts, separate from session-bootstrap corner CIs."),
        },
        "comparators": {
            "title": "YOLO whole-package comparator audit",
            "columns": _columns(("method", "Method"),
                                ("comparison_role", "Role"),
                                ("seed_count", "Seeds"), *metric_columns,
                                *extended_metric_columns,
                                *denominator_columns),
            "rows": comparator_rows(inputs),
            "note": ("D/L/PoseFix and N3 differ in inputs, output parameterization, loss, or training budget. "
                     "This is a whole-package comparison, not a single-factor causal ablation."),
        },
        "update_alternatives": {
            "title": "Update alternatives: safe HELDOUT128 and blocked DEV319 rows",
            "columns": _columns(("population", "Population"),
                                ("method", "Method"),
                                ("evidence_role", "Evidence role"),
                                ("student_rgb_overlap", "Student RGB overlap"),
                                ("teacher_session_overlap", "Teacher session overlap"),
                                ("independent_test", "Independent TEST"),
                                ("selection_history", "Selection history"),
                                *metric_columns, *extended_metric_columns,
                                *denominator_columns),
            "rows": update_alternative_rows(inputs),
            "note": ("HELDOUT128 is a predeclared exposure-safe reused-development cohort. "
                     "DEV319 update cells stay x because a common exposure contract is unavailable."),
        },
        "square_green0918": {
            "title": "GREEN0918_119 square audit (manual declared)",
            "columns": _columns(("backbone", "Backbone"), ("method", "Method"),
                                ("seed_count", "Seeds"), *metric_columns,
                                *extended_metric_columns,
                                *denominator_columns),
            "rows": square_rows(inputs),
            "note": "One fixed dimension vector; independent 3D pose is x and dimension effect is not identifiable.",
        },
        "training": {
            "title": "Six N3 fits",
            "columns": _columns(("backbone", "Backbone"), ("seed", "Seed"),
                                ("status", "Status"), ("steps", "Steps"),
                                ("exposures", "Exposures"),
                                ("elapsed_seconds", "Elapsed (s)"),
                                ("trainable_parameters", "Trainable params"),
                                ("dimension_input_to_N3", "Dimensions to N3"),
                                ("symmetry_supervision", "Symmetry supervision"),
                                ("base_receives_dimensions", "Dimensions to base"),
                                ("checkpoint_sha256", "Checkpoint SHA-256")),
            "rows": training_rows(inputs),
        },
        "selection": {
            "title": "Synthetic-only fixed selection",
            "columns": _columns(("backbone", "Backbone"), ("seed", "Seed"),
                                ("temperature", "Temperature"), ("lambda", "Lambda"),
                                ("cap_fraction", "Image-diagonal cap"),
                                ("real_selection", "Real outcome used")),
            "rows": selection_rows(inputs),
            "note": "Temperature is fixed on synthetic calibration; real outcome used must be false.",
        },
        "runtime": {
            "title": "Locked RTX runtime",
            "columns": _columns(("backbone", "Backbone"), ("path", "Path"),
                                ("median_ms", "Median (ms)"), ("p90_ms", "P90 (ms)"),
                                ("fps", "FPS"), ("params", "Params"),
                                ("peak_allocated_bytes", "Peak allocated (B)")),
            "rows": runtime_rows(inputs),
        },
        "lifter": {
            "title": "Offline lifter case study",
            "columns": _columns(("method", "Method"), ("frames", "Frames"),
                                ("available_fraction", "Available"),
                                ("fresh_fraction", "Fresh"),
                                ("longest_missing_s", "Longest missing (s)"),
                                ("yaw_step_median_deg", "Yaw step median (deg)"),
                                ("yaw_step_p90_deg", "Yaw step P90 (deg)"),
                                ("independent_accuracy", "Independent accuracy")),
            "rows": lifter_rows(inputs),
            "note": "Visible state is unknown; position/yaw accuracy is x.",
        },
        "subgroups": {
            "title": "YOLO material and occlusion subgroups",
            "columns": _columns(("group_type", "Group type"), ("group", "Group"),
                                ("method", "Method"), ("seed_count", "Seeds"),
                                *metric_columns, *denominator_columns),
            "rows": subgroup_rows(inputs),
            "note": "Occlusion labels cover only the declared subset; unclassified is retained.",
        },
    }
    complete_fits = sum(row["status"]["status"] == "COMPLETE"
                        for row in tables["training"]["rows"])
    complete_backbones = sum(
        row["median_px"]["status"] == "COMPLETE"
        for row in tables["backbone_dev"]["rows"] if row["method"]["value"] == "N3 seed mean")
    required_complete = (
        "training_complete_dope", "training_complete_resnet18",
        "validation_dope", "validation_resnet18",
        "selection_dope", "selection_resnet18", "eval_dope", "eval_resnet18",
        "square_yolo", "square_dope", "square_resnet18",
        "runtime_dope", "runtime_resnet18", "runtime_raw_dope",
        "runtime_raw_resnet18", "lifter_receipt", "lifter_raw",
        "pred_dope", "pred_resnet18", "environment_audit",
        "symmetry_activation_audit", "dimension_sensitivity_audit",
    )
    receipts_complete = all(
        isinstance(inputs.get(key, {}).get("payload"), Mapping)
        and inputs[key]["payload"].get("complete") is True
        for key in required_complete)
    reuse = inputs.get("reuse", {}).get("payload")
    expected_reuse_statuses = {
        "A": "COMPLETE", "B": "COMPLETE", "C": "COMPLETE",
        "D": "PARTIAL_LABELS", "E": "COMPLETE", "H": "COMPLETE",
        "I": "PARTIAL_SAFE_COHORT",
    }
    reuse_ok = (isinstance(reuse, Mapping)
                and reuse.get("schema") == "pallet_n3_completion_v3_reuse_v1"
                and all(_get(reuse, "contracts", key, "status") == status
                        for key, status in expected_reuse_statuses.items())
                and _get(reuse, "contracts", "I", "DEV319_table", "status") == "BLOCKED_CONTRACT"
                and _get(reuse, "contracts", "I", "safe_common_cohort", "status") == "COMPLETE_REUSED_DEV")
    square_yolo = inputs.get("square_yolo", {}).get("payload")
    square_families = _get(square_yolo, "modes", "manual_declared", "families")
    square_yolo_ok = (isinstance(square_yolo, Mapping)
                      and square_yolo.get("complete") is True
                      and isinstance(square_families, Mapping)
                      and all(family in square_families for family in
                              ("R0", "OLD_P", "N2_DIM_ONLY", "N3_DIM_SYM")))
    protocol = inputs.get("protocol", {}).get("payload")
    methods_contract = protocol.get("methods") if isinstance(protocol, Mapping) else None
    resnet_contract = _get(protocol, "backbones", "resnet18")
    square_payload = inputs.get("square_yolo", {}).get("payload")
    square_pose_payload = inputs.get("square_dope", {}).get("payload")
    eval_dope_payload = inputs.get("eval_dope", {}).get("payload")
    runtime_dope = inputs.get("runtime_dope", {}).get("payload")
    runtime_resnet = inputs.get("runtime_resnet18", {}).get("payload")
    environment_audit = inputs.get("environment_audit", {}).get("payload")
    environment_roles = (_get(environment_audit, "roles")
                         if _get(environment_audit, "schema") ==
                         "pallet_n3_completion_v3_environment_audit_v1" else None)
    symmetry_audit = inputs.get("symmetry_activation_audit", {}).get("payload")
    dimension_audit = inputs.get("dimension_sensitivity_audit", {}).get("payload")
    return {
        "schema": SCHEMA, "generated_utc": _utc_now(),
        "complete": (complete_fits == 6 and complete_backbones == 3
                     and receipts_complete and reuse_ok and square_yolo_ok),
        "post_generation_integrity": {
            "status": "SEPARATE_POST_GENERATION_AUDIT",
            "artifact": "_docs/experiments/pallet_n3_completion_v3/VERIFY_RESULTS.json",
            "hash_bound_in_report": False,
            "reason": (
                "VERIFY_RESULTS binds the generated report, so the report cannot in turn "
                "bind VERIFY_RESULTS without a circular content-hash dependency."),
        },
        "legend": {
            "x": "required result is absent, blocked, or not measured",
            "NA": "metric is defined as not applicable/undefined for this population",
            "0": "measured numeric zero",
        },
        "evidence_role": "REUSED_DEV; not independent TEST",
        "inference_GT_input": False,
        "method_contract": {
            "purpose": _get(protocol, "purpose"),
            "base_inputs": (_get(methods_contract, "base_inputs")
                            if isinstance(methods_contract, Mapping) else None),
            "n3_inputs": (_get(methods_contract, "n3_inputs")
                          if isinstance(methods_contract, Mapping) else None),
            "n3_inference_forbidden_inputs": _get(
                methods_contract, "n3_inference_forbidden_inputs"),
            "preserved_outputs": (_get(methods_contract, "preserved_outputs")
                                  if isinstance(methods_contract, Mapping) else None),
            "dimension_features": _get(protocol, "n3", "dimension_features"),
            "symmetry_source": _get(protocol, "symmetry_supervision", "source"),
            "symmetry_selection_reference": _get(
                protocol, "symmetry_supervision", "selection_reference"),
            "symmetry_inference_uses_target": _get(
                protocol, "symmetry_supervision", "inference_uses_symmetry_target"),
            "train_supervision": _get(protocol, "data", "train_supervision"),
            "new_real_training_images": _get(
                protocol, "data", "new_real_training_images"),
            "seed_aggregation": _get(
                protocol, "evaluation", "seed_aggregation"),
            "resnet18_source_training": _get(
                resnet_contract, "adapter", "source_training"),
            "resnet18_deviation": _get(resnet_contract, "deviation"),
            "resnet18_source_checkpoint": _get(
                resnet_contract, "source_checkpoint"),
            "resnet18_nested_checkpoint_selection": _get(
                resnet_contract, "adapter", "input", "checkpoint_selection"),
            "dev_pose_reference": _get(
                eval_dope_payload, "contract", "pose_reference"),
            "valid_permutations_per_row_counts": _get(
                symmetry_audit, "source_sidecar", "valid_permutations_per_row_counts"),
            "four_way_symmetry_rows": _get(
                symmetry_audit, "source_sidecar", "four_way_rows"),
            "symmetry_claim_boundary": _get(
                symmetry_audit, "interpretation", "claim_boundary"),
            "dimension_a_wdh_m": _get(dimension_audit, "dimension_a_wdh_m"),
            "dimension_b_wdh_m": _get(dimension_audit, "dimension_b_wdh_m"),
            "dimension_sensitivity_summary": _get(dimension_audit, "summary"),
            "dimension_claim_boundary": _get(dimension_audit, "claim_boundary"),
        },
        "square_contract": {
            "dimensions_wdh_m": _get(square_payload, "dimensions_wdh_m"),
            "manual_declared_corners": (_get(
                square_pose_payload, "manual_declared_corners") or
                _get(square_payload, "modes", "manual_declared",
                     "manual_corner_denominator")),
            "manual_in_frame_corners": _get(
                square_pose_payload, "manual_in_frame_corners"),
            "sessions": _get(
                square_pose_payload, "modes", "manual_declared",
                "population", "sessions"),
            "confidence_interval": _get(square_payload, "confidence_interval"),
            "dimension_effect_identifiable": _get(
                square_payload, "dimension_effect_identifiable"),
            "dimension_effect_note": _get(square_payload, "dimension_effect_note"),
            "pose_3d": (_get(square_payload, "pose_3d") or
                        _get(square_payload, "physical_pose_reference")),
        },
        "software_environment": {
            "dope_resnet": (_get(
                environment_roles, "dope_resnet_training_inference_runtime") or {
                    "torch": (_get(runtime_dope, "hardware", "torch") or
                              _get(runtime_resnet, "hardware", "torch")),
                    "torch_cuda": (_get(runtime_dope, "hardware", "torch_cuda") or
                                   _get(runtime_resnet, "hardware", "torch_cuda")),
                }),
            "yolo_square_lifter": (_get(
                environment_roles, "yolo26_square_and_offline_lifter") or {}),
            "boundary": (_get(environment_audit, "boundary") or {}),
            "source": inputs.get("environment_audit", {}).get("path"),
        },
        "inventory": {key: _inventory_record(value) for key, value in inputs.items()},
        "tables": tables,
    }


def _format_cell(value: Any) -> str:
    if not isinstance(value, Mapping) or "display" not in value:
        return str(value)
    if value.get("display") in {"x", "NA"}:
        return value["display"]
    raw = value.get("value")
    if isinstance(raw, float):
        return "0" if raw == 0 else f"{raw:.4g}"
    text = str(value.get("display", raw))
    return text[:16] + "…" if len(text) > 17 and len(text) == 64 else text


def _markdown_table(spec: Mapping[str, Any]) -> str:
    columns = spec["columns"]
    lines = [f"### {spec['title']}", "",
             "| " + " | ".join(column["label"] for column in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in spec["rows"]:
        lines.append("| " + " | ".join(
            _format_cell(row.get(column["key"], _missing("cell absent")))
            for column in columns) + " |")
    if spec.get("note"):
        lines.extend(("", f"_Note: {spec['note']}_"))
    return "\n".join(lines)


def render_tables_md(tables: Mapping[str, Any]) -> str:
    body = ["# N3 completion tables", "",
            "`x` = missing/blocked/unmeasured; `NA` = protocol-defined not applicable; numeric `0` = measured zero.", ""]
    for spec in tables["tables"].values():
        body.extend((_markdown_table(spec), ""))
    return "\n".join(body).rstrip() + "\n"


def _tex_escape(value: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


def render_tex_fragment(table: Mapping[str, Any], *, tables_sha256: str) -> str:
    columns = table["columns"]
    lines = [f"% Generated from TABLES.json sha256: {tables_sha256}",
             "% x = missing/blocked; NA = protocol-defined not applicable; 0 = measured zero",
             r"\begin{tabular}{" + "l" * len(columns) + "}",
             r"\hline",
             " & ".join(_tex_escape(column["label"]) for column in columns) + r" \\",
             r"\hline"]
    for row in table["rows"]:
        lines.append(" & ".join(_tex_escape(_format_cell(
            row.get(column["key"], _missing("cell absent")))) for column in columns) + r" \\")
    lines.extend((r"\hline", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def write_tex_fragments(output_doc: Path, tables: Mapping[str, Any],
                        tables_sha256: str) -> list[dict]:
    folder = output_doc / "table_fragments"
    mapping = {
        "backbone_dev_headline.tex": "backbone_dev",
        "square_2d.tex": "square_green0918",
        "runtime.tex": "runtime",
    }
    output = []
    for filename, key in mapping.items():
        path = folder / filename
        _write(path, render_tex_fragment(tables["tables"][key],
                                         tables_sha256=tables_sha256))
        output.append({"path": str(path), "sha256": _sha256(path),
                       "bytes": path.stat().st_size, "table": key,
                       "TABLES_json_sha256": tables_sha256})
    return output


def _evaluation_metric_sets(inputs: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, str, list, list]]:
    for backbone, key in (("DOPE", "eval_dope"), ("ResNet-18", "eval_resnet18")):
        payload = inputs.get(key, {}).get("payload")
        if not isinstance(payload, Mapping) or payload.get("schema") != "pallet_n3_completion_v3_evaluation_v1":
            continue
        methods = payload.get("methods")
        if not isinstance(methods, Mapping):
            continue
        for method in ("base", "n3_seed1", "n3_seed2", "n3_seed3"):
            entry = methods.get(method)
            result = entry.get("result") if isinstance(entry, Mapping) and entry.get("status") == "COMPLETE" else None
            corners = result.get("corner_rows") if isinstance(result, Mapping) else None
            poses = result.get("pose_rows") if isinstance(result, Mapping) else None
            if isinstance(corners, list):
                yield backbone, method, corners, poses if isinstance(poses, list) else []


def _reuse_metric_sets(inputs: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, str, list, list]]:
    payload = inputs.get("reuse_per_frame", {}).get("payload")
    core = payload.get("core") if isinstance(payload, Mapping) else None
    if not isinstance(core, Mapping):
        return
    for method in ("R0", "N3_DIM_SYM_seed1", "N3_DIM_SYM_seed2", "N3_DIM_SYM_seed3"):
        entry = core.get(method)
        corners = entry.get("corner_scores") if isinstance(entry, Mapping) else None
        poses = entry.get("pose_scores") if isinstance(entry, Mapping) else None
        if isinstance(corners, list):
            yield "YOLO", method, corners, poses if isinstance(poses, list) else []


def normalized_metric_rows(inputs: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Flatten actual per-frame/per-corner evidence without synthesizing rows."""
    frame_rows, corner_rows = [], []
    for backbone, method, corners, poses in (*list(_reuse_metric_sets(inputs)),
                                               *list(_evaluation_metric_sets(inputs))):
        pose_by_id = {str(row.get("id")): row for row in poses if isinstance(row, Mapping)}
        for row in corners:
            if not isinstance(row, Mapping) or row.get("id") is None:
                continue
            frame_id = str(row["id"])
            pose = pose_by_id.get(frame_id, {})
            frame_rows.append({
                "backbone": backbone, "method": method, "frame_id": frame_id,
                "session": row.get("session"), "material": row.get("material"),
                "occlusion": row.get("occlusion"), "evaluable": row.get("evaluable"),
                "detected": row.get("detected"), "matched": row.get("matched"),
                "E_sym": row.get("E_sym"), "E_fixed": row.get("E_fixed"),
                "frame_mean_px": row.get("frame_mean_px"),
                "pose_available": pose.get("available"),
                "translation_cm": pose.get("translation_cm"),
                "rotation_deg": pose.get("rotation_deg"),
                "yaw_deg": pose.get("yaw_deg"),
            })
            errors = row.get("canonical_errors")
            valid = row.get("canonical_valid")
            if not isinstance(errors, list):
                errors = row.get("errors") if isinstance(row.get("errors"), list) else []
            if not isinstance(valid, list) or len(valid) != 8:
                valid = [index < len(errors) and errors[index] is not None for index in range(8)]
            for corner_index in range(8):
                error = errors[corner_index] if corner_index < len(errors) else None
                corner_rows.append({
                    "backbone": backbone, "method": method, "frame_id": frame_id,
                    "session": row.get("session"), "corner_index": corner_index,
                    "supervised": bool(valid[corner_index]),
                    "error_px": error,
                    "matched": row.get("matched"), "detected": row.get("detected"),
                    "material": row.get("material"), "occlusion": row.get("occlusion"),
                })
    return frame_rows, corner_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
    pending.replace(path)
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size,
            "rows": len(rows)}


def write_raw_metrics(inputs: Mapping[str, Mapping[str, Any]], output_raw: Path) -> dict:
    frames, corners = normalized_metric_rows(inputs)
    folder = output_raw / "report"
    frame_fields = ("backbone", "method", "frame_id", "session", "material", "occlusion",
                    "evaluable", "detected", "matched", "E_sym", "E_fixed", "frame_mean_px",
                    "pose_available", "translation_cm", "rotation_deg", "yaw_deg")
    corner_fields = ("backbone", "method", "frame_id", "session", "corner_index",
                     "supervised", "error_px", "matched", "detected", "material", "occlusion")
    return {
        "FRAME_METRICS.csv": _write_csv(folder / "FRAME_METRICS.csv", frames, frame_fields),
        "CORNER_METRICS.csv": _write_csv(folder / "CORNER_METRICS.csv", corners, corner_fields),
    }


def _placeholder(ax, message: str = "x: input unavailable") -> None:
    ax.set_axis_off()
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes,
            color="#666666", fontsize=11, wrap=True)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_training(inputs: Mapping[str, Mapping[str, Any]], path: Path) -> dict:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    complete = 0
    sources = []
    for ax, (backbone, seed) in zip(axes.flat, TRAIN_KEYS):
        key = f"train_{backbone.lower()}_{seed}"
        record = inputs.get(key, {})
        payload = record.get("payload") if record.get("status") == "LOADED" else None
        history = payload.get("history") if isinstance(payload, Mapping) else None
        valid_rows = [row for row in history or []
                      if _finite(row.get("step")) and _finite(row.get("loss"))]
        if (isinstance(payload, Mapping) and payload.get("complete") is True
                and payload.get("smoke") is False and payload.get("steps") == C.STEPS
                and valid_rows):
            steps = [row["step"] for row in valid_rows]
            losses = [row["loss"] for row in valid_rows]
            ax.plot(steps, losses, color="#276FBF", linewidth=1.7)
            ax.set_title(f"{'ResNet-18' if backbone == 'RESNET18' else backbone} seed {seed}")
            ax.grid(alpha=.25)
            complete += 1
            sources.append(record.get("path"))
        else:
            _placeholder(ax, f"{backbone} seed {seed}\nx: fit unavailable")
    fig.supxlabel("Training step")
    fig.supylabel("Training loss")
    fig.suptitle("Six fixed N3 fits")
    fig.tight_layout()
    _save(fig, path)
    return {"file": path.name, "status": "COMPLETE" if complete == 6 else "PARTIAL",
            "actual_series": complete, "expected_series": 6,
            "sources": [item for item in sources if item]}


def _numeric(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, Mapping) and value.get("status") == "COMPLETE" and _finite(value.get("value")):
        return float(value["value"])
    return None


def figure_backbones(tables: Mapping[str, Any], path: Path) -> dict:
    rows = tables["tables"]["backbone_dev"]["rows"]
    seed_rows = tables["tables"].get("backbone_dev_per_seed", {}).get("rows", [])
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    x = np.arange(len(BACKBONES), dtype=float)
    width = .36
    actual = 0
    seed_points = 0
    for ax, (key, label, _) in zip(axes.flat, METRICS):
        base, n3 = [], []
        for backbone in BACKBONES:
            selected = [row for row in rows if row["backbone"]["value"] == backbone]
            baseline = next((row for row in selected if row["method"]["value"] == "Base"), {})
            corrected = next((row for row in selected if row["method"]["value"] == "N3 seed mean"), {})
            base.append(_numeric(baseline, key))
            n3.append(_numeric(corrected, key))
        if not any(value is not None for value in base + n3):
            _placeholder(ax)
            ax.set_title(label)
            continue
        base_plot = [np.nan if value is None else value for value in base]
        n3_plot = [np.nan if value is None else value for value in n3]
        ax.bar(x - width / 2, base_plot, width, label="Base", color="#8A9BA8")
        ax.bar(x + width / 2, n3_plot, width, label="N3 seed mean", color="#E07A5F")
        for backbone_index, backbone in enumerate(BACKBONES):
            values = [
                _numeric(row, key) for row in seed_rows
                if row.get("backbone", {}).get("value") == backbone
                and row.get("method", {}).get("value") == "N3"
                and row.get("seed", {}).get("value") in C.SEEDS
            ]
            values = [value for value in values if value is not None]
            if values:
                offsets = np.linspace(-.06, .06, len(values))
                ax.scatter(np.full(len(values), x[backbone_index] + width / 2) + offsets,
                           values, s=18, facecolor="white", edgecolor="#7F2F25",
                           linewidth=.8, zorder=4,
                           label="N3 seeds" if seed_points == 0 else None)
                seed_points += len(values)
        ax.set_xticks(x, BACKBONES, rotation=15)
        ax.set_title(label)
        ax.grid(axis="y", alpha=.25)
        actual += sum(value is not None for value in base + n3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(fontsize=8)
    fig.suptitle("Base to N3 on reused DEV319")
    fig.tight_layout()
    _save(fig, path)
    return {"file": path.name, "status": "COMPLETE" if actual == 48 else "PARTIAL",
            "actual_metric_cells": actual, "expected_metric_cells": 48,
            "seed_points": seed_points,
            "aggregation": "bars use arithmetic mean of seed-level statistics; dots are individual fits"}


def figure_runtime(tables: Mapping[str, Any], path: Path) -> dict:
    rows = tables["tables"]["runtime"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [f"{row['backbone']['value']}\n{row['path']['value']}" for row in rows]
    colors = ["#8A9BA8" if row["path"]["value"] == "Base E2E" else
              "#E07A5F" if row["path"]["value"] == "Base + N3 E2E" else "#81B29A"
              for row in rows]
    actual = 0
    for ax, metric, title in ((axes[0], "median_ms", "Median latency"),
                              (axes[1], "p90_ms", "P90 latency")):
        values = [_numeric(row, metric) for row in rows]
        if not any(value is not None for value in values):
            _placeholder(ax)
            ax.set_title(title)
            continue
        ax.bar(np.arange(len(rows)), [np.nan if value is None else value for value in values],
               color=colors)
        ax.set_xticks(np.arange(len(rows)), labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("CUDA event time (ms)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
        actual += sum(value is not None for value in values)
    fig.suptitle("Locked RTX runtime: seed 1, 26 frames, 5 repeats")
    fig.tight_layout()
    _save(fig, path)
    return {"file": path.name, "status": "COMPLETE" if actual == 12 else "PARTIAL",
            "actual_metric_cells": actual, "expected_metric_cells": 12,
            "selection_role": "descriptive only; never used for model selection"}


def _points(block: Any) -> np.ndarray | None:
    if not isinstance(block, Mapping):
        return None
    value = block.get("points", block.get("points_original"))
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return result if result.shape == (9, 2) else None


def _frame_methods(frame: Mapping[str, Any]) -> tuple[Any, Any]:
    predictions = frame.get("predictions")
    if isinstance(predictions, Mapping):
        base = predictions.get("base", predictions.get("R0"))
        n3 = predictions.get("n3_seed1", predictions.get("N3_seed1"))
    else:
        base = frame.get("base")
        n3 = frame.get("n3_seed1", frame.get("N3_seed1"))
    return base, n3


def _symmetry_error(prediction: np.ndarray | None, truth: np.ndarray,
                    valid: np.ndarray, permutations: Any) -> float | None:
    if prediction is None or prediction.shape != (9, 2):
        return None
    mask = np.asarray(valid, bool)[:8]
    if not mask.any():
        return None
    candidates = permutations if isinstance(permutations, Sequence) else [list(range(9))]
    errors = []
    for permutation in candidates:
        try:
            indices = np.asarray(permutation, dtype=int)
            if indices.shape[0] >= 8 and np.all((indices[:8] >= 0) & (indices[:8] < 9)):
                aligned = prediction[indices[:8]]
            else:
                continue
        except (TypeError, ValueError, IndexError):
            continue
        usable = mask & np.isfinite(aligned).all(-1) & np.isfinite(truth[:8]).all(-1)
        if usable.any():
            errors.append(float(np.mean(np.linalg.norm(aligned[usable] - truth[:8][usable], axis=1))))
    if not errors:
        aligned = prediction[:8]
        usable = mask & np.isfinite(aligned).all(-1) & np.isfinite(truth[:8]).all(-1)
        if not usable.any():
            return None
        return float(np.mean(np.linalg.norm(aligned[usable] - truth[:8][usable], axis=1)))
    return min(errors)


def overlay_records(prediction_payload: Mapping[str, Any],
                    truth_rows: Sequence[Mapping[str, Any]], *,
                    scored_errors: Mapping[str, tuple[Any, Any]] | None = None) -> list[dict]:
    """Join prediction rows to frozen truth for post-hoc figure selection only."""
    frames = prediction_payload.get("frames", prediction_payload.get("records"))
    if not isinstance(frames, Sequence):
        return []
    truth = {str(row.get("id", row.get("frame_id"))): row for row in truth_rows}
    output = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        frame_id = str(frame.get("id", frame.get("frame_id", "")))
        target = truth.get(frame_id)
        if target is None:
            continue
        base_block, n3_block = _frame_methods(frame)
        base_points, n3_points = _points(base_block), _points(n3_block)
        gt = np.asarray(target.get("gt"), dtype=float)
        valid = np.asarray(target.get("valid"), dtype=bool)
        if gt.shape != (9, 2) or valid.shape != (9,):
            continue
        base_error = _symmetry_error(base_points, gt, valid, target.get("permutations"))
        n3_error = _symmetry_error(n3_points, gt, valid, target.get("permutations"))
        if scored_errors is not None and frame_id in scored_errors:
            locked_base, locked_n3 = scored_errors[frame_id]
            base_error = float(locked_base) if _finite(locked_base) else None
            n3_error = float(locked_n3) if _finite(locked_n3) else None
        output.append({
            "id": frame_id, "session_id": frame.get("session_id", target.get("session")),
            "image_key": frame.get("image_key", frame.get("image_path")),
            "image_sha256": frame.get("image_sha256"),
            "base_points": None if base_points is None else base_points.tolist(),
            "n3_points": None if n3_points is None else n3_points.tolist(),
            "gt": gt.tolist(), "valid": valid.tolist(),
            "base_error": base_error, "n3_error": n3_error,
            "delta_base_minus_n3": (None if base_error is None or n3_error is None
                                     else base_error - n3_error),
        })
    return output


def _locked_overlay_errors(payload: Any) -> dict[str, tuple[Any, Any]]:
    methods = payload.get("methods") if isinstance(payload, Mapping) else None
    if not isinstance(methods, Mapping):
        return {}
    result = {}
    by_method = []
    for method in ("base", "n3_seed1"):
        rows = _get(methods, method, "result", "corner_rows")
        if not isinstance(rows, list):
            return {}
        by_method.append({str(row.get("id")): row.get("frame_mean_px")
                          for row in rows if isinstance(row, Mapping) and row.get("id") is not None})
    for frame_id in sorted(set(by_method[0]) & set(by_method[1])):
        result[frame_id] = (by_method[0][frame_id], by_method[1][frame_id])
    return result


def select_overlay_cases(records: Sequence[Mapping[str, Any]]) -> dict[str, dict | None]:
    """Select deterministic post-hoc extremes; this is never inference selection."""
    complete = [dict(row) for row in records if _finite(row.get("delta_base_minus_n3"))]
    missing = [dict(row) for row in records
               if row.get("base_error") is None or row.get("n3_error") is None]
    by_id = lambda row: str(row.get("id", ""))
    return {
        "improvement": (sorted(complete, key=lambda row: (-float(row["delta_base_minus_n3"]), by_id(row)))[0]
                        if complete else None),
        "near_no_change": (sorted(complete, key=lambda row: (abs(float(row["delta_base_minus_n3"])), by_id(row)))[0]
                           if complete else None),
        "adverse": (sorted(complete, key=lambda row: (float(row["delta_base_minus_n3"]), by_id(row)))[0]
                    if complete else None),
        "missing": sorted(missing, key=by_id)[0] if missing else None,
    }


def _checkout_roots(root: Path) -> list[Path]:
    roots = [Path(root).resolve(), C.ROOT.resolve()]
    dot_git = Path(root) / ".git"
    if dot_git.is_file():
        marker = dot_git.read_text().strip()
        if marker.startswith("gitdir:"):
            git_dir = Path(marker.split(":", 1)[1].strip())
            if not git_dir.is_absolute():
                git_dir = (Path(root) / git_dir).resolve()
            common = git_dir / "commondir"
            if common.is_file():
                common_dir = Path(common.read_text().strip())
                if not common_dir.is_absolute():
                    common_dir = (git_dir / common_dir).resolve()
                roots.append(common_dir.parent.resolve())
    return list(dict.fromkeys(roots))


def _resolve_image(row: Mapping[str, Any], roots: Sequence[Path]) -> Path | None:
    key = row.get("image_key")
    if not key:
        return None
    path = Path(str(key))
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    expected = row.get("image_sha256")
    for candidate in candidates:
        if candidate.is_file() and (not expected or _sha256(candidate) == expected):
            return candidate
    return None


def _binding_if_present(relative: Path, roots: Sequence[Path], root: Path) -> dict | None:
    for candidate_root in roots:
        path = candidate_root / relative
        if path.is_file():
            return {"path": _relative(path, root), "sha256": _sha256(path),
                    "bytes": path.stat().st_size}
    return None


def figure_overlays(inputs: Mapping[str, Mapping[str, Any]], path: Path,
                    *, root: Path = C.ROOT,
                    truth_rows: Sequence[Mapping[str, Any]] | None = None) -> dict:
    truth_error = None
    if truth_rows is None:
        try:
            from .evaluation import load_dev319_truth
            truth_rows = load_dev319_truth(include_pose=False)
        except Exception as exc:  # report remains executable with x
            truth_error = f"{type(exc).__name__}: {exc}"
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    categories = (("improvement", "Improvement extreme"),
                  ("near_no_change", "Near no change"),
                  ("adverse", "Adverse extreme"), ("missing", "Missing output"))
    roots = _checkout_roots(root)
    actual = 0
    cases = []
    prediction_sources = {}
    for row_index, (key, backbone, eval_key) in enumerate((
            ("pred_dope", "DOPE", "eval_dope"),
            ("pred_resnet18", "ResNet-18", "eval_resnet18"))):
        record = inputs.get(key, {})
        payload = record.get("payload") if record.get("status") == "LOADED" else None
        prediction_sources[backbone] = record.get("path")
        joined = []
        if isinstance(payload, Mapping) and truth_rows is not None:
            scored = _locked_overlay_errors(inputs.get(eval_key, {}).get("payload"))
            joined = overlay_records(payload, truth_rows, scored_errors=scored or None)
        chosen = select_overlay_cases(joined)
        for column, (category, title) in enumerate(categories):
            ax = axes[row_index, column]
            row = chosen[category]
            if row is None:
                status = "NA" if joined else "MISSING"
                marker = "NA: no qualifying frame" if joined else "x: prediction/truth unavailable"
                _placeholder(ax, f"{backbone} — {title}\n{marker}")
                cases.append({"backbone": backbone, "category": category,
                              "status": status, "frame_id": None})
                continue
            image_path = _resolve_image(row, roots)
            if image_path is None:
                _placeholder(ax, f"{backbone} — {title}\nx: bound image unavailable")
                cases.append({"backbone": backbone, "category": category,
                              "status": "MISSING_IMAGE", "frame_id": row["id"],
                              "delta_base_minus_n3": row.get("delta_base_minus_n3")})
                continue
            ax.imshow(plt.imread(image_path))
            for points, color, marker, label in (
                (row.get("gt"), "#2CA02C", "o", "GT"),
                (row.get("base_points"), "#1F77B4", "x", "Base"),
                (row.get("n3_points"), "#D62728", "+", "N3 seed 1"),
            ):
                if points is None:
                    continue
                array = np.asarray(points, dtype=float)[:8]
                finite = np.isfinite(array).all(-1)
                if finite.any():
                    ax.scatter(array[finite, 0], array[finite, 1], c=color,
                               marker=marker, s=26, linewidths=1.3, label=label)
            delta = row.get("delta_base_minus_n3")
            suffix = "" if delta is None else f"; Base-N3={delta:+.2f}px"
            ax.set_title(f"{backbone} — {title}: {row['id']}{suffix}", fontsize=8)
            ax.set_axis_off()
            ax.legend(loc="best", fontsize=6)
            actual += 1
            cases.append({"backbone": backbone, "category": category,
                          "status": "COMPLETE", "frame_id": row["id"],
                          "session_id": row.get("session_id"),
                          "delta_base_minus_n3": delta,
                          "base_error": row.get("base_error"),
                          "n3_error": row.get("n3_error"),
                          "GT_corner8_xy": np.asarray(row.get("gt"), dtype=float)[:8].tolist(),
                          "GT_corner8_valid": np.asarray(row.get("valid"), dtype=bool)[:8].tolist(),
                          "base_corner8_xy": (None if row.get("base_points") is None else
                                              np.asarray(row["base_points"], dtype=float)[:8].tolist()),
                          "n3_seed1_corner8_xy": (None if row.get("n3_points") is None else
                                                  np.asarray(row["n3_points"], dtype=float)[:8].tolist()),
                          "image_path": _relative(image_path, root),
                          "image_sha256": _sha256(image_path)})
    fig.suptitle("Post-hoc DEV visualization by estimator")
    fig.tight_layout()
    _save(fig, path)
    truth_relative = C.DEV.relative_to(C.ROOT)
    resolved = all(case["status"] in {"COMPLETE", "NA"} for case in cases)
    return {
        "file": path.name, "status": "COMPLETE" if resolved else "PARTIAL",
        "backbones": ["DOPE", "ResNet-18"], "actual_overlays": actual,
        "expected_panels": 8, "cases": cases,
        "selection_rule": {
            "improvement": "maximum Base error minus N3 seed-1 error",
            "near_no_change": "minimum absolute Base-minus-N3 error delta",
            "adverse": "minimum Base error minus N3 seed-1 error",
            "missing": "lexicographically first frame lacking Base or N3 evaluable points",
        },
        "extreme_selection_disclosure": (
            "GT is used only after inference to select illustrative extremes; cases are not representative."),
        "inference_GT_input": False,
        "GT_used_for_model_or_temperature_selection": False,
        "ranking_metric": "locked matched frame_mean_px when evaluation rows are available",
        "truth_error": truth_error,
        "truth_manifest": _binding_if_present(truth_relative, roots, root),
        "prediction_sources": prediction_sources,
    }


def _raw_lifter_frames(payload: Any) -> list[dict]:
    sessions = payload.get("sessions") if isinstance(payload, Mapping) else None
    if not isinstance(sessions, Mapping):
        return []
    output = []
    for session_id, session in sessions.items():
        frames = session.get("frames") if isinstance(session, Mapping) else None
        for frame in frames or []:
            if isinstance(frame, Mapping):
                row = dict(frame)
                row.setdefault("session_id", session_id)
                output.append(row)
    return output


def figure_lifter(inputs: Mapping[str, Mapping[str, Any]], path: Path) -> dict:
    receipt_record = inputs.get("lifter_receipt", {})
    receipt = receipt_record.get("payload") if receipt_record.get("status") == "LOADED" else None
    raw_record = inputs.get("lifter_raw", {})
    raw = raw_record.get("payload") if raw_record.get("status") == "LOADED" else None
    stats = receipt.get("statistics") if isinstance(receipt, Mapping) else None
    methods = ("R0", "N3_seed1")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    axes = axes.flat
    actual = 0

    available, fresh = [], []
    for method in methods:
        available.append(_get(stats, method, "overall", "coverage", "available_fraction"))
        fresh.append(_get(stats, method, "overall", "coverage", "fresh_fraction"))
    if all(_finite(value) for value in available + fresh):
        x = np.arange(2)
        axes[0].bar(x - .18, available, .36, label="Available", color="#4C78A8")
        axes[0].bar(x + .18, fresh, .36, label="Fresh", color="#F58518")
        axes[0].set_xticks(x, methods)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Fraction")
        axes[0].set_title("Coverage summary")
        axes[0].legend(fontsize=8)
        actual += 1
    else:
        _placeholder(axes[0])
        axes[0].set_title("Coverage summary")

    medians, p90s = [], []
    for method in methods:
        jitter = _get(stats, method, "overall", "yaw_wrap_jitter",
                      "available_outputs_including_held")
        medians.append(jitter.get("median_abs_step") if isinstance(jitter, Mapping) else None)
        p90s.append(jitter.get("p90_abs_step") if isinstance(jitter, Mapping) else None)
    if all(_finite(value) for value in medians + p90s):
        x = np.arange(2)
        axes[1].bar(x - .18, medians, .36, label="Median", color="#59A14F")
        axes[1].bar(x + .18, p90s, .36, label="P90", color="#E15759")
        axes[1].set_xticks(x, methods)
        axes[1].set_ylabel("Absolute yaw step (deg)")
        axes[1].set_title("Wrap-aware yaw jitter")
        axes[1].legend(fontsize=8)
        actual += 1
    else:
        _placeholder(axes[1])
        axes[1].set_title("Wrap-aware yaw jitter")

    longest = []
    for method in methods:
        missing = _get(stats, method, "overall", "missing")
        run = _get(missing, "longest_until_next_sample_sensor_time_run")
        if not isinstance(run, Mapping):
            run = _get(missing, "longest_observed_sensor_time_run")
        duration = _get(run, "until_next_sample_s") if isinstance(run, Mapping) else None
        if duration is None and isinstance(run, Mapping):
            duration = _get(run, "observed_span_s")
        longest.append(duration)
    if all(_finite(value) for value in longest):
        axes[2].bar(np.arange(2), longest, color=("#4C78A8", "#E15759"))
        axes[2].set_xticks(np.arange(2), methods)
        axes[2].set_ylabel("Seconds")
        axes[2].set_title("Longest missing run")
        axes[2].grid(axis="y", alpha=.25)
        actual += 1
    else:
        _placeholder(axes[2])
        axes[2].set_title("Longest missing run")

    frames = _raw_lifter_frames(raw)
    plotted_panels = 0
    timeline_session = None
    if frames:
        first_session = str(frames[0].get("session_id"))
        timeline_session = first_session
        selected = [row for row in frames if str(row.get("session_id")) == first_session]
        if selected:
            origin = float(selected[0].get("camera_sensor_timestamp_ms", 0.))
            for ax, field, ylabel, title in (
                    (axes[3], "pos_x_m", "Camera x (m)", "Lateral estimate"),
                    (axes[4], "pos_z_m", "Camera z (m)", "Forward estimate"),
                    (axes[5], "yaw_deg", "Yaw representative (deg)", "Yaw estimate")):
                method_series = 0
                for method, color in zip(methods, ("#4C78A8", "#E15759")):
                    xs, ys = [], []
                    for row in selected:
                        block = _get(row, "methods", method)
                        value = block.get(field) if isinstance(block, Mapping) else None
                        timestamp = row.get("camera_sensor_timestamp_ms")
                        if (_finite(timestamp) and _finite(value)
                                and block.get("available", False)):
                            xs.append((float(timestamp) - origin) / 1000.)
                            ys.append(float(value))
                    if xs:
                        ax.plot(xs, ys, marker=".", markersize=2, linewidth=.8,
                                label=method, color=color)
                        method_series += 1
                if method_series:
                    ax.set_xlabel("Sensor time from first sample (s)")
                    ax.set_ylabel(ylabel)
                    ax.set_title(f"{title}: {first_session}")
                    ax.legend(fontsize=7)
                    ax.grid(alpha=.2)
                    actual += 1
                    plotted_panels += 1
                else:
                    _placeholder(ax, f"x: {field} timeline unavailable")
                    ax.set_title(title)
    if not frames:
        for ax, title in zip(axes[3:], ("Lateral estimate", "Forward estimate", "Yaw estimate")):
            _placeholder(ax, "x: raw lifter timeline unavailable")
            ax.set_title(title)
    for ax in axes[:3]:
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Offline lifter case study (accuracy reference unavailable)")
    fig.tight_layout()
    _save(fig, path)
    return {"file": path.name, "status": "COMPLETE" if actual == 6 else "PARTIAL",
            "actual_panels": actual, "expected_panels": 6,
            "timeline_session": timeline_session,
            "timeline_fields": ["pos_x_m", "pos_z_m", "yaw_deg"],
            "receipt_source": receipt_record.get("path"),
            "raw_source": raw_record.get("path"),
            "visible_state": "UNKNOWN_NOT_ANNOTATED",
            "independent_accuracy": {"value": None, "status": "MISSING", "display": "x"}}


def figure_subgroups(inputs: Mapping[str, Mapping[str, Any]], path: Path) -> dict:
    record = inputs.get("reuse", {})
    reuse = record.get("payload") if record.get("status") == "LOADED" else None
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flat
    actual = 0
    thresholds = (5, 10, 20)
    for method, color, marker in (("R0", "#4C78A8", "o"),
                                  ("N3_DIM_SYM", "#E15759", "s")):
        headline = _get(reuse, "contracts", "A", "methods", method, "mean", "headline")
        values = [_get(headline, "full_PCK", str(threshold)) for threshold in thresholds]
        if all(_finite(value) for value in values):
            axes[0].plot(thresholds, values, color=color, marker=marker,
                         label="Base" if method == "R0" else "N3 seed mean")
    if axes[0].lines:
        axes[0].set_xlabel("Error threshold (px)")
        axes[0].set_ylabel("PCK fraction")
        axes[0].set_xticks(thresholds)
        axes[0].set_ylim(0, 1)
        axes[0].set_title("DEV319 error-threshold curve")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=.25)
        actual += 1
    else:
        _placeholder(axes[0])

    groups = ("plastic", "wood", "clean", "moderate", "severe")
    base_values, n3_values = [], []
    for group in groups:
        contract, group_type = ("C", "material") if group in {"plastic", "wood"} else ("D", "occlusion")
        base_values.append(_get(reuse, "contracts", contract, "methods", "R0", group_type,
                                group, "matched_pooled_corner8_median_px"))
        n3_values.append(_get(reuse, "contracts", contract, "methods", "N3_DIM_SYM", group_type,
                              group, "matched_pooled_corner8_median_px"))
    if any(_finite(value) for value in base_values + n3_values):
        x = np.arange(len(groups))
        axes[1].bar(x - .18, [np.nan if not _finite(v) else v for v in base_values],
                    .36, label="Base", color="#4C78A8")
        axes[1].bar(x + .18, [np.nan if not _finite(v) else v for v in n3_values],
                    .36, label="N3 seed mean", color="#E15759")
        axes[1].set_xticks(x, groups, rotation=25, ha="right")
        axes[1].set_ylabel("2D median (px)")
        axes[1].set_title("Material / occlusion subgroups")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=.25)
        actual += 1
    else:
        _placeholder(axes[1])

    damage = _get(reuse, "contracts", "E", "methods", "N3_DIM_SYM",
                  "cap1pct", "mean_damage_counts")
    comparable_corners = _get(damage, "observed_comparable_corners")
    comparable_frames = _get(damage, "matched_comparable_frames")
    changes = (_get(damage, "corner_change", "improved"),
               _get(damage, "corner_change", "worsened"),
               _get(damage, "frame_change", "improved"),
               _get(damage, "frame_change", "worsened"))
    if (all(_finite(value) for value in changes)
            and _finite(comparable_corners) and _finite(comparable_frames)
            and comparable_corners > 0 and comparable_frames > 0):
        fractions = [changes[0] / comparable_corners, changes[1] / comparable_corners,
                     changes[2] / comparable_frames, changes[3] / comparable_frames]
        axes[2].bar(np.arange(4), fractions,
                    color=("#59A14F", "#E15759", "#59A14F", "#E15759"))
        axes[2].set_xticks(np.arange(4),
                           ("Corner +", "Corner -", "Frame +", "Frame -"),
                           rotation=20)
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel("Fraction of paired denominator")
        axes[2].set_title("N3 1% cap: paired improvement / harm")
        axes[2].grid(axis="y", alpha=.25)
        actual += 1
    else:
        _placeholder(axes[2], "x: cap damage audit unavailable")
        axes[2].set_title("N3 1% cap: paired improvement / harm")

    raw_record = inputs.get("reuse_per_frame", {})
    raw = raw_record.get("payload") if raw_record.get("status") == "LOADED" else None
    fixed = _get(raw, "cap", "E_N3_DIM_SYM_seed1_cap1pct", "fixed_branch")
    candidate_pose = _get(raw, "cap", "E_N3_DIM_SYM_seed1_cap1pct", "pose_scores")
    base_pose = _get(raw, "core", "R0", "pose_scores")
    paired_x, paired_y = [], []
    if isinstance(fixed, Sequence) and isinstance(candidate_pose, Sequence) and isinstance(base_pose, Sequence):
        base_by_id = {str(row.get("id")): row for row in base_pose if isinstance(row, Mapping)}
        candidate_by_id = {str(row.get("id")): row for row in candidate_pose if isinstance(row, Mapping)}
        for row in fixed:
            if not isinstance(row, Mapping):
                continue
            frame_id = str(row.get("id"))
            base = base_by_id.get(frame_id)
            candidate = candidate_by_id.get(frame_id)
            dx = row.get("delta_px")
            before = base.get("translation_cm") if isinstance(base, Mapping) else None
            after = candidate.get("translation_cm") if isinstance(candidate, Mapping) else None
            if (_finite(dx) and _finite(before) and _finite(after)
                    and base.get("available") and candidate.get("available")):
                paired_x.append(float(dx))
                paired_y.append(float(after) - float(before))
    if paired_x:
        colors = ["#59A14F" if x < 0 and y < 0 else
                  "#F28E2B" if x < 0 <= y else
                  "#76B7B2" if x >= 0 > y else "#E15759"
                  for x, y in zip(paired_x, paired_y)]
        axes[3].scatter(paired_x, paired_y, s=12, alpha=.65, c=colors,
                        linewidths=0)
        axes[3].axvline(0, color="black", linewidth=.7)
        axes[3].axhline(0, color="black", linewidth=.7)
        axes[3].set_xlabel("Delta 2D frame mean (px); lower is better")
        axes[3].set_ylabel("Delta translation error (cm); lower is better")
        axes[3].set_title("N3 seed 1: paired 2D / translation")
        axes[3].grid(alpha=.2)
        actual += 1
    else:
        _placeholder(axes[3], "x: paired 2D / pose rows unavailable")
        axes[3].set_title("N3 seed 1: paired 2D / translation")

    fig.suptitle("YOLO N3 subgroup, threshold, cap, and paired-pose evidence")
    fig.tight_layout()
    _save(fig, path)
    return {"file": path.name, "status": "COMPLETE" if actual == 4 else "PARTIAL",
            "actual_panels": actual, "expected_panels": 4,
            "paired_scatter_points": len(paired_x),
            "occlusion_label_status": _get(reuse, "contracts", "D", "status") or "MISSING",
            "source": record.get("path"),
            "paired_source": raw_record.get("path")}


def remaining_items(tables: Mapping[str, Any]) -> list[dict]:
    items: list[dict] = []
    seen = set()
    for table_name, table in tables["tables"].items():
        for row_index, row in enumerate(table["rows"]):
            for key, value in row.items():
                if not isinstance(value, Mapping) or value.get("display") not in {"x", "NA"}:
                    continue
                identity = (table_name, row_index, key, value.get("status"), value.get("reason"))
                if identity in seen:
                    continue
                seen.add(identity)
                items.append({"table": table_name, "row": row_index, "field": key,
                              "display": value.get("display"), "status": value.get("status"),
                              "reason": value.get("reason")})
    # Protocol-level limitations remain even after all executable work finishes.
    protocol = (
        ("occlusion_corner_visibility", "x", "BLOCKED_LABEL",
         "191 DEV frames are unclassified and corner-level visibility is not fixed."),
        ("fair_common_DEV319_backbone_table", "x", "BLOCKED_CONTRACT",
         "A common fair conditional DEV319 comparison population is not available."),
        ("GREEN0918_independent_6D", "x", "BLOCKED_REFERENCE",
         "GREEN0918_119 has no independent canonical 6D reference."),
        ("lifter_independent_accuracy", "x", "BLOCKED_REFERENCE",
         "Lifter visible state and independent position/yaw reference are unavailable."),
        ("independent_TEST", "x", "MISSING",
         "All accuracy results are reused development evidence, not an independent TEST."),
    )
    for field, display, status, reason in protocol:
        items.append({"table": "protocol", "row": None, "field": field,
                      "display": display, "status": status, "reason": reason})
    return items


def render_remaining(items: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Remaining x / NA", "",
             "`x`는 아직 없거나 계약상 막힌 근거이고, `NA`는 해당 모집단에서 정의되지 않는 값이다. 실제 숫자 0은 이 목록에 들어오지 않는다.", "",
             "| 종류 | 위치 | 필드 | 상태 | 이유 |", "|---|---|---|---|---|"]
    for item in items:
        location = item["table"] if item.get("row") is None else f"{item['table']}[{item['row']}]"
        reason = str(item.get("reason") or "source value unavailable").replace("|", "\\|")
        lines.append(f"| {item['display']} | {location} | {item['field']} | {item['status']} | {reason} |")
    return "\n".join(lines) + "\n"


def _pair_summary(tables: Mapping[str, Any], backbone: str) -> str:
    rows = tables["tables"]["backbone_dev"]["rows"]
    selected = [row for row in rows if row["backbone"]["value"] == backbone]
    base = next((row for row in selected if row["method"]["value"] == "Base"), None)
    n3 = next((row for row in selected if row["method"]["value"] == "N3 seed mean"), None)
    if base is None or n3 is None:
        return "x"
    def pair(key: str, suffix: str = "", scale: float = 1.) -> str:
        left, right = _numeric(base, key), _numeric(n3, key)
        return ("x" if left is None or right is None else
                f"{left * scale:.3f}→{right * scale:.3f}{suffix}")
    return (f"conditional 2D median {pair('median_px', ' px')}, "
            f"conditional P90 {pair('p90_px', ' px')}, "
            f"PCK10 {pair('pck10', '%', 100.)}, E_sym {pair('e_sym')}, "
            f"full-penalty P90 {pair('full_p90_px', ' px')}, "
            f"T {pair('t_median_cm', ' cm')}, R {pair('r_median_deg', '°')}, "
            f"yaw {pair('yaw_median_deg', '°')}")


def _method_pair(tables: Mapping[str, Any], backbone: str) -> tuple[dict | None, dict | None]:
    rows = tables["tables"]["backbone_dev"]["rows"]
    selected = [row for row in rows if row["backbone"]["value"] == backbone]
    return (next((row for row in selected if row["method"]["value"] == "Base"), None),
            next((row for row in selected if row["method"]["value"] == "N3 seed mean"), None))


def _direction_summary(tables: Mapping[str, Any], backbone: str) -> str:
    base, n3 = _method_pair(tables, backbone)
    if base is None or n3 is None:
        return "비교 결과 x"
    specs = (("median_px", "2D median", False), ("p90_px", "2D P90", False),
             ("pck10", "PCK10", True), ("t_median_cm", "T median", False),
             ("r_median_deg", "R median", False),
             ("yaw_median_deg", "yaw median", False),
             ("full_p90_px", "penalty P90", False))
    groups = {"improved": [], "same": [], "worsened": [], "missing": []}
    for key, label, higher in specs:
        left, right = _numeric(base, key), _numeric(n3, key)
        if left is None or right is None:
            groups["missing"].append(label)
            continue
        delta = right - left
        if abs(delta) <= 1e-12:
            groups["same"].append(label)
        elif (delta > 0) == higher:
            groups["improved"].append(label)
        else:
            groups["worsened"].append(label)
    phrases = []
    for key, korean in (("improved", "개선 방향"), ("same", "동일"),
                        ("worsened", "악화 방향"), ("missing", "x")):
        if groups[key]:
            phrases.append(f"{korean}: {', '.join(groups[key])}")
    return "; ".join(phrases) if phrases else "비교 가능한 지표 x"


def _ablation_claim(tables: Mapping[str, Any]) -> str:
    rows = tables["tables"]["yolo_ablation_deltas"]["rows"]
    row = next((item for item in rows if item["contrast"]["value"] == "N3 - N2"), None)
    if row is None:
        return "N3−N2 결과는 x이다."
    values = {key: _numeric(row, key) for key in
              ("median_delta_px", "p90_delta_px", "pck10_delta_pp", "e_sym_delta")}
    if any(value is None for value in values.values()):
        return "N3−N2 결과는 일부 또는 전부 x이다."
    return ("N3−N2는 median {median_delta_px:+.6f} px, P90 {p90_delta_px:+.6f} px, "
            "PCK10 {pck10_delta_pp:+.6f} pp, E_sym {e_sym_delta:+.9f}였다. "
            "따라서 대칭 감독의 추가 효과는 지표별로 작고 혼합되어 있으며, "
            "필수적이거나 보편적인 향상이라고 해석하지 않는다.").format(**values)


def _subgroup_claim(tables: Mapping[str, Any]) -> str:
    rows = tables["tables"]["subgroups"]["rows"]
    clauses = []
    for group in ("clean", "moderate", "severe"):
        selected = [row for row in rows if row["group"]["value"] == group]
        base = next((row for row in selected if row["method"]["value"] == "Base"), None)
        n3 = next((row for row in selected if row["method"]["value"] == "N3 seed mean"), None)
        if base is None or n3 is None:
            clauses.append(f"{group}=x")
            continue
        def pair(key: str, unit: str) -> str:
            left, right = _numeric(base, key), _numeric(n3, key)
            return "x" if left is None or right is None else f"{left:.3f}→{right:.3f}{unit}"
        frames = _numeric(base, "frames")
        sample = "n=x" if frames is None else f"n={int(frames)}"
        clauses.append(
            f"{group}({sample}): 2D {pair('median_px', ' px')}, T {pair('t_median_cm', ' cm')}, "
            f"R {pair('r_median_deg', '°')}")
    unclassified = next((row for row in rows
                         if row["group"]["value"] == "unclassified"
                         and row["method"]["value"] == "Base"), None)
    count = _numeric(unclassified or {}, "frames")
    suffix = "x" if count is None else f"{int(count)}개"
    return "; ".join(clauses) + f". 나머지 미분류는 {suffix}로 유지했다."


def _table_block(tables: Mapping[str, Any], key: str) -> str:
    return _markdown_table(tables["tables"][key])


def render_final_report(tables: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    fits = tables["tables"]["training"]["rows"]
    fit_count = sum(row["status"]["status"] == "COMPLETE" for row in fits)
    method = tables.get("method_contract", {})
    square = tables.get("square_contract", {})
    environments = tables.get("software_environment", {})

    def listed(value: Any) -> str:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return "x" if value is None else str(value)
        return ", ".join(str(item) for item in value) if value else "x"

    dimensions = square.get("dimensions_wdh_m")
    dimension_text = ("x" if not isinstance(dimensions, Sequence) else
                      "[" + ", ".join(str(value) for value in dimensions) + "] m")
    resnet_source = method.get("resnet18_source_training") or "x"
    resnet_deviation = method.get("resnet18_deviation") or "x"
    resnet_checkpoint = method.get("resnet18_source_checkpoint") or {}
    resnet_nested_epoch = method.get("resnet18_nested_checkpoint_selection") or "x"
    execution = "완료" if tables.get("execution_complete") else "미완료"
    overlay = next((entry for entry in manifest.get("figures", [])
                    if entry.get("file") == "dev_overlays.png"), {})
    overlay_count = overlay.get("actual_overlays", "x")
    overlay_expected = overlay.get("expected_panels", "x")
    overlay_na = sum(case.get("status") == "NA" for case in overlay.get("cases", []))
    primary_env = environments.get("dope_resnet", {})
    yolo_env = environments.get("yolo_square_lifter", {})
    environment_boundary = environments.get("boundary", {})
    permutation_counts = method.get("valid_permutations_per_row_counts") or {}
    dimension_summary = method.get("dimension_sensitivity_summary") or {}
    return f"""# Pallet N3 v3 실행 및 결과 보고서

## 판정

최종 과학적 판정은 **PARTIAL**이다. 계약상 실행 가능한 파이프라인의 상태는 **{execution}**이고, DOPE·ResNet-18 N3 학습 receipt는 6개 중 **{fit_count}개**가 완료 상태다. 실행 완료와 주장 완결성은 다른 판정이다. DEV319의 일부 가림 라벨, estimator 사이의 공정한 공통 조건 집단, GREEN0918의 독립 6D 기준, 리프터의 독립 위치·yaw 기준, 독립 TEST가 없으므로 `OVERALL_COMPLETE`로 올리지 않는다. 없는 수치는 복사하거나 보간하지 않고 `x`, 정의되지 않는 값은 `NA`, 측정된 영은 숫자 `0`으로 보존했다.

이 보고서에서 허용되는 중심 주장은 **동일한 N3 설계와 학습 계약을 세 frozen estimator에 각각 적용했을 때 재사용 개발 집단에서 관측된 변화**다. 이는 estimator 사이에 같은 N3 가중치를 옮긴 실험도, 모든 backbone에 대한 보편적 robustness 증명도 아니다.

## 정확한 입력·감독 경계

- frozen base 입력: **{listed(method.get('base_inputs'))}**
- backbone별 N3 입력: **{listed(method.get('n3_inputs'))}**
- N3 치수 특징: **{listed(method.get('dimension_features'))}**
- 대칭 감독 출처: **{method.get('symmetry_source') or 'x'}**
- 대칭 branch 선택 기준: **{method.get('symmetry_selection_reference') or 'x'}**
- N3 추론 금지 입력: **{listed(method.get('n3_inference_forbidden_inputs'))}**
- 보존 출력: **{listed(method.get('preserved_outputs'))}**
- 학습 감독과 새 real 이미지 수: **{method.get('train_supervision') or 'x'}**, **{method.get('new_real_training_images') if method.get('new_real_training_images') is not None else 'x'}**

따라서 base는 RGB-only 함수로 고정되고, 물리 치수 `[W,D,H]`는 각 backbone 전용 N3 head의 학습과 추론에 들어간다. 대칭 감독은 raw frozen base 출력에 대해 승인된 whole-object permutation 하나를 고르는 학습 target이며, 추론에서는 GT·visibility·symmetry branch label을 입력하지 않는다. Base와 N3의 6D 평가는 같은 prediction-only PnP 계약으로 계산한다. 이 경계 때문에 결과를 “base 자체가 치수를 학습했다”거나 “추론에서 정답 대칭을 골랐다”고 설명하면 안 된다.

ResNet-18 기준 모델은 **{resnet_source}**이다. 구체적 제한은 “{resnet_deviation}”이다. 즉 이 결과는 **10-epoch CONSTANT-fold RGB-only runtime base**에 대한 것이며, 60-epoch pure-RGB baseline 결과로 바꾸어 부를 수 없다.

Protocol 내부 adapter input에 남은 `checkpoint_selection="{resnet_nested_epoch}"` 문구는 이전 60-epoch 입력 명세의 잔재다. 이번 실행의 authoritative source는 `{resnet_checkpoint.get('path', 'x')}`와 SHA `{resnet_checkpoint.get('sha256', 'x')}`로 고정된 **epoch10 CONSTANT checkpoint**다. 보고서와 표는 nested epoch60 문구를 실제 학습 근거로 사용하지 않는다.

### 실제 대칭 target 활성화

{_table_block(tables, 'symmetry_activation')}

동일한 symmetry-aware objective는 두 backbone에 연결되었지만 실제 target branch 활성은 달랐다. sidecar의 valid permutation 수는 1개인 row **{permutation_counts.get('1', 'x')}개**, 2개인 row **{permutation_counts.get('2', 'x')}개**, 4개인 row **{permutation_counts.get('4', 'x')}개**다. 따라서 ResNet-18에서 non-identity target이 활성화되었다거나, 네 방향 C4 target으로 학습해 GREEN 정사각형에 전이했다고 주장할 수 없다. 허용되는 표현은 “같은 whole-object 대칭 선택 목적함수를 적용했다”이며 실제 non-identity 사용 빈도는 위 표와 함께 보고한다.

### 실제 치수 경로 민감도

{_table_block(tables, 'dimension_sensitivity')}

visual tensors·initial points·box·validity·base logits를 고정하고 `[W,D,H]`만 바꾼 감사에서 PASS한 head는 **{dimension_summary.get('fits_verified', 'x')}개**, 변한 logit은 총 **{dimension_summary.get('total_changed_logit_entries', 'x')}개**다. 이는 학습된 여섯 head의 치수 경로가 실제로 활성이라는 근거다. 정확도 향상 중 얼마가 치수만의 인과 효과인지는 증명하지 않으며, 그 분리는 YOLO N0/N1/N2/N3 통제 표에 한정한다.

## 재사용 DEV319 결과

YOLO: **{_pair_summary(tables, 'YOLO')}**  
DOPE: **{_pair_summary(tables, 'DOPE')}**  
ResNet-18: **{_pair_summary(tables, 'ResNet-18')}**

![Base to N3](figures/backbone_comparison.png)

YOLO 관측 방향은 **{_direction_summary(tables, 'YOLO')}**다. DOPE는 **{_direction_summary(tables, 'DOPE')}**, ResNet-18은 **{_direction_summary(tables, 'ResNet-18')}**다. 이 문장은 지표별 방향을 그대로 열거하며, 일부 지표 개선을 T·R·yaw의 안정적 동시 개선으로 확대하지 않는다.

위 N3 값은 seed 1·2·3의 **통계 산술평균**이며 예측 앙상블이 아니다. DOPE와 YOLO는 decoding, score, box와 조건부 matched 집단이 동등하지 않으므로 절대 backbone 정확도 순위를 만들 수 없다. 세 estimator에서 각각 학습한 N3의 적용 결과를 비교할 수 있을 뿐이다. 모든 정확도 수치는 **REUSED_DEV** 근거이고 독립 TEST 일반화를 뜻하지 않는다.

DEV pose reference는 “{method.get('dev_pose_reference') or 'x'}”다. 따라서 DEV의 T/R/yaw는 고정 evaluator 안의 비교 근거이지만 독립 물리 계측 6D 정확도로 부를 수 없다.

{_table_block(tables, 'backbone_dev')}

### DOPE·ResNet-18 seed별 결과

{_table_block(tables, 'backbone_dev_per_seed')}

### Paired session-bootstrap

{_table_block(tables, 'backbone_paired_ci')}

CI는 각 seed를 base와 같은 frame/session으로 묶어 session 단위 10,000회 재표집한 기술 통계다. CI가 0을 가로지르는 지표는 방향이 고정되었다고 주장하지 않는다. 다중비교 보정은 하지 않았고, 표의 분모와 conditional/full-population 정의를 함께 읽어야 한다.

## 학습 및 선택

![Six fits](figures/training_curves.png)

학습 곡선은 `complete=true`, `smoke=false`, step6000 완료 receipt가 있는 fit만 그렸다. 각 fit은 별도 seed와 backbone 전용 head이고, base에는 치수를 넣지 않는다. 온도와 1% 이동 cap은 synthetic calibration에서 고정했으며 실제 DEV 성능, runtime, 그림 사례, lifter 결과를 선택에 쓰지 않았다.

{_table_block(tables, 'training')}

{_table_block(tables, 'selection')}

## B: YOLO N0/N1/N2/N3 통제 실험

{_table_block(tables, 'yolo_ablation')}

{_table_block(tables, 'yolo_ablation_deltas')}

{_ablation_claim(tables)} 이 인과 분해는 YOLO의 고정 계약 안에서만 성립한다. DOPE와 ResNet-18에는 base 대 N3만 있으므로 그 두 estimator의 변화에서 치수 입력 효과와 대칭 감독 효과를 따로 식별할 수 없다.

## E: cap, 손상·회복, 2D–pose 대응

{_table_block(tables, 'yolo_cap_damage')}

`none` 행에서 cap inside/outside/hit은 정의되지 않아 `NA`이고, 손상·회복 count의 숫자 0과 구분한다. 1%와 2%는 frozen base가 고른 whole-object branch에 고정해 움직임으로 생긴 개선과 악화를 함께 센 결과다.

{_table_block(tables, 'yolo_paired_2d_pose')}

2D frame 오차 감소는 T·R·yaw 동시 감소를 뜻하지 않는다. 위 표는 각 seed의 개선·무차이·악화, 새 pose 실패, pose 회복, 2D/T 방향 조합을 모두 남긴다. 따라서 이 자료로 허용되는 주장은 “작은 보정이 관측된 2D 오차를 줄인 사례와 속도·손상 trade-off가 있다”이며, 모든 frame의 6D가 함께 좋아졌다는 주장은 허용되지 않는다.

## H: 기존 보정기와의 비교

{_table_block(tables, 'comparators')}

D, L, PoseFix-style, P, N3는 입력, 출력 parameterization, loss 또는 학습 budget이 다른 **whole-package 비교**다. 이 표는 상대 결과를 보여 주지만 어느 한 구성요소의 단일 요인 인과 효과를 증명하지 않는다.

## I: update 대안과 self-training 경계

{_table_block(tables, 'update_alternatives')}

HELDOUT128 행은 노출 계약이 명시된 **재사용 개발 안전 cohort**다. 독립 TEST가 아니다. 공통 노출 계약이 없는 DEV319 update 비교는 `x`로 유지했다. 이번 실행에서는 새 self-training을 수행하지 않았고, 이 표는 이미 완료되어 동결된 동일 계약 결과만 재사용한다.

## GREEN0918_119 정사각형 감사

정사각형 119장의 선언 치수는 **{dimension_text}**다. 수동 선언 corner는 **{square.get('manual_declared_corners', 'x')}개**, 그중 image 안 corner는 **{square.get('manual_in_frame_corners', 'x')}개**다. 표의 `GT corners`는 full-population 평가 분모인 선언 corner 수를 뜻하며 in-frame subset과 혼동하지 않는다. 이 자료는 **{square.get('sessions', 'x')}개 capture session**뿐이므로 bootstrap CI는 `{square.get('confidence_interval') or 'x'}`로 유지한다. 한 치수 벡터만 있는 재사용 개발 2D 감사이므로 full trained package의 전이는 볼 수 있지만 프레임별 치수 변화의 인과 효과는 식별할 수 없다. 독립 canonical 6D 기준이 없어 T/R/yaw·IoU3D·ADDsym은 `x`다. YOLO는 R0/OLD_P/N2/N3를, DOPE와 ResNet-18은 Base/N3를 같은 표에 남긴다.

{_table_block(tables, 'square_green0918')}

## 실행시간

![Runtime](figures/runtime.png)

runtime은 고정 DEV319 26프레임, seed 1, warmup 20, repeat 5, batch 1의 동일 RTX 측정이다. CUDA event의 median/P90, parameter 수, peak memory를 receipt에 남긴다. runtime은 모델이나 결과 선택에 사용하지 않는다.

소프트웨어 환경은 하나가 아니었다. DOPE·ResNet-18 학습/평가/runtime은 `{primary_env.get('executable') or 'x'}`의 PyTorch `{primary_env.get('torch') or 'x'}`·CUDA `{primary_env.get('torch_cuda') or 'x'}`·Ultralytics `{primary_env.get('ultralytics') or 'x'}`를 사용했다. YOLO26 정사각형·리프터 단계는 `{yolo_env.get('executable') or 'x'}`의 Ultralytics `{yolo_env.get('ultralytics') or 'x'}`를 사용했다. 전자는 C3k2가 `{primary_env.get('has_C3k2') if primary_env.get('has_C3k2') is not None else 'x'}`, 후자는 `{yolo_env.get('has_C3k2') if yolo_env.get('has_C3k2') is not None else 'x'}`이며, 경계 이유는 “{environment_boundary.get('reason') or 'x'}”다. `run.py --yolo-python`이 interpreter 경계를 stage receipt config에 묶고, `ENVIRONMENT_AUDIT.json`이 executable과 Ultralytics module SHA를 보존한다. 두 결과가 동일 software environment에서 나왔다고 해석하지 않는다. 새 측정 GPU 기록은 `{environment_boundary.get('same_GPU_for_new_measurements') or 'x'}`지만 YOLO runtime은 이번 fixed26 benchmark에서 비교하지 않았다.

{_table_block(tables, 'runtime')}

## DEV 사례 그림

![Post-hoc overlays](figures/dev_overlays.png)

추론에는 GT를 입력하지 않았다(`inference_GT_input=false`). 그림은 DOPE·ResNet-18의 improvement/near-no-change/adverse/missing **{overlay_expected}개 panel**을 고정했고, 실제 overlay는 **{overlay_count}개**, 조건에 맞는 missing 사례가 없어 `NA`인 panel은 **{overlay_na}개**다. 사례는 추론을 모두 끝낸 뒤 GT 8-corner 오차로 고른 극단 예시이며 대표 표본이 아니다. 정확한 frame ID, 좌표, 오차, 원본 이미지 SHA와 선정 규칙은 `figures/SELECTION_MANIFEST.json`에 있다. GT는 inference, 학습, checkpoint, temperature, cap 선택에 사용되지 않았다.

## 리프터 offline case study

![Lifter](figures/lifter.png)

고정 세션의 sensor-time 표본에서 coverage, fresh output, missing run, wrap-aware yaw jitter와 출력 위치 x/z·yaw 시계열을 기술한다. 시계열은 예측 출력의 거동이며 정확도 곡선이 아니다. visible state가 `UNKNOWN_NOT_ANNOTATED`이고 독립 위치·yaw 기준이 없으므로 정확도, 위치 오차, yaw 오차는 `x`다. 실제 리프터 제어는 호출하지 않았다.

{_table_block(tables, 'lifter')}

## 재료·가림 및 threshold

![Subgroups](figures/subgroups_or_thresholds.png)

{_subgroup_claim(tables)} 이 비교는 synthetic-only N3를 라벨된 real subgroup에서 평가한 것이며, clean 영상으로 self-training한 모델의 moderate 전이 실험이 아니다. clean/moderate/severe로 검토된 일부 frame만으로 전체 DEV319의 가림 성능을 대표하지 않는다. corner별 visibility 감독과 나머지 frame의 가림 분류는 `x`다.

{_table_block(tables, 'subgroups')}

threshold 그림은 PCK@5/10/20, material/occlusion 2D median, N3 1% cap 손상·회복 비율, seed1의 paired 2D–translation 산점을 함께 표시한다. 산점의 서로 다른 사분면은 2D와 T 방향이 항상 일치하지 않음을 보여 주는 기술 자료다.

## 논문에서 쓸 수 있는 표현과 한계

- 쓸 수 있음: “RGB-only frozen estimator의 출력·feature와 물리 치수를 받는 작은 N3 head를 estimator별로 학습했고, 승인된 whole-object 대칭 감독을 사용했다.”
- 쓸 수 있음: “재사용 DEV에서 YOLO·DOPE·ResNet-18의 결과를 seed별/분모별로 보고했고, 개선·무차이·악화와 paired CI를 모두 보존했다.”
- 쓸 수 있음: “YOLO 통제 ablation에서 치수와 대칭의 증분 효과는 지표별로 달랐으며 N3−N2는 혼합 결과였다.”
- 제한 필요: “여러 estimator에 같은 절차를 적용했다.” 동일 weight transfer나 backbone 불변 robustness로 확대하지 않는다.
- 제한 필요: “2D 오차가 감소했다.” 이를 T·R·yaw가 안정적으로 동시에 개선되었다는 문장으로 바꾸지 않는다.
- 쓸 수 없음: 독립 TEST 일반화, GREEN0918 독립 6D 정확도, 리프터 실물 정확도, 모든 가림 단계 개선, 치수 변화의 단독 인과 효과.

## 파일 안내

- `TABLES.json`: 표의 값·상태·이유·분모·CI를 보존하는 기계 판독 source of truth
- `TABLES.md`: 모든 표를 한 번에 읽는 문서 (`x`, `NA`, 숫자 0 구분)
- `table_fragments/*.tex`: backbone DEV, square 2D, runtime LaTeX 조각; 각 파일 주석에 `TABLES.json` SHA 기록
- `{_get(tables, 'raw_evidence', 'FRAME_METRICS.csv', 'path') or 'data/.../report/FRAME_METRICS.csv'}`: 실제 frame-level 정규화 수치와 SHA `{_get(tables, 'raw_evidence', 'FRAME_METRICS.csv', 'sha256') or 'x'}`
- `{_get(tables, 'raw_evidence', 'CORNER_METRICS.csv', 'path') or 'data/.../report/CORNER_METRICS.csv'}`: 실제 corner-level 정규화 수치와 SHA `{_get(tables, 'raw_evidence', 'CORNER_METRICS.csv', 'sha256') or 'x'}`
- `figures/SELECTION_MANIFEST.json`: 여섯 그림의 source, SHA, panel 상태와 post-hoc 선정 규칙
- `REMAINING_X.md`: 실행 누락과 계약상 남는 근거를 분리한 목록
- `VERIFY_RESULTS.json`: 보고서 생성 뒤 수행한 독립 무결성 감사. 순환 해시를 피하기 위해 보고서 입력 inventory에는 이 파일을 넣지 않는다.

보고서 상태: `{tables['overall_status']}`. 생성된 그림 {len(manifest.get('figures', []))}개. 계약에 따라 PDF 생성, 새 self-training, 실제 리프터 제어는 수행하지 않았다.
"""


def _hash_figures(figures_dir: Path, entries: list[dict]) -> None:
    for entry in entries:
        path = figures_dir / entry["file"]
        if path.is_file():
            entry["sha256"] = _sha256(path)
            entry["bytes"] = path.stat().st_size


def generate(*, doc: Path = C.DOC, raw: Path = C.RAW,
             output_doc: Path | None = None, output_raw: Path | None = None,
             root: Path = C.ROOT,
             truth_rows: Sequence[Mapping[str, Any]] | None = None) -> dict:
    """Generate every report artifact, even when some inputs are unavailable."""
    doc, raw, root = Path(doc), Path(raw), Path(root)
    output_doc = doc if output_doc is None else Path(output_doc)
    output_raw = raw if output_raw is None else Path(output_raw)
    figures_dir = output_doc / "figures"
    inputs = load_inputs(doc, raw, root=root)
    tables = build_tables(inputs)
    # Overall scientific status is intentionally never inferred from fit count.
    tables["execution_complete"] = tables.pop("complete")
    tables["overall_status"] = "PARTIAL"
    tables["overall_complete"] = False
    tables["overall_reason"] = (
        "Protocol-level labels/references remain x: occlusion visibility, fair common DEV319, "
        "GREEN0918 independent 6D, lifter independent accuracy, and independent TEST.")
    raw_evidence = write_raw_metrics(inputs, output_raw)
    for entry in raw_evidence.values():
        entry["path"] = _relative(Path(entry["path"]), root)
    tables["raw_evidence"] = raw_evidence

    # TABLES.json is the fixed numeric source for both Markdown and TeX.
    _write(output_doc / "TABLES.json", tables)
    tables_sha256 = _sha256(output_doc / "TABLES.json")
    tex_fragments = write_tex_fragments(output_doc, tables, tables_sha256)
    for entry in tex_fragments:
        entry["path"] = _relative(Path(entry["path"]), root)

    figures = [
        figure_training(inputs, figures_dir / "training_curves.png"),
        figure_backbones(tables, figures_dir / "backbone_comparison.png"),
        figure_runtime(tables, figures_dir / "runtime.png"),
        figure_overlays(inputs, figures_dir / "dev_overlays.png", root=root,
                        truth_rows=truth_rows),
        figure_lifter(inputs, figures_dir / "lifter.png"),
        figure_subgroups(inputs, figures_dir / "subgroups_or_thresholds.png"),
    ]
    _hash_figures(figures_dir, figures)
    manifest = {
        "schema": "pallet_n3_completion_v3_figure_selection_v1",
        "generated_utc": _utc_now(), "complete": all(
            entry["status"] == "COMPLETE" for entry in figures),
        "overall_status": "PARTIAL",
        "inference_GT_input": False,
        "GT_role": (
            "Evaluation and post-hoc overlay ranking only; never inference, training, checkpoint, "
            "temperature, cap, runtime, or lifter selection."),
        "evidence_role": "REUSED_DEV; not independent TEST",
        "input_inventory": tables["inventory"],
        "TABLES_json": {"path": _relative(output_doc / "TABLES.json", root),
                        "sha256": tables_sha256,
                        "bytes": (output_doc / "TABLES.json").stat().st_size},
        "tex_fragments": tex_fragments,
        "raw_evidence": raw_evidence,
        "figures": figures,
    }
    remaining = remaining_items(tables)
    _write(output_doc / "TABLES.md", render_tables_md(tables))
    _write(output_doc / "REMAINING_X.md", render_remaining(remaining))
    _write(figures_dir / "SELECTION_MANIFEST.json", manifest)
    _write(output_doc / "FINAL_REPORT_KO.md", render_final_report(tables, manifest))
    return {"tables": tables, "manifest": manifest, "remaining": remaining,
            "outputs": [str(output_doc / name) for name in
                        ("TABLES.json", "TABLES.md", "FINAL_REPORT_KO.md", "REMAINING_X.md")]
                       + [str(figures_dir / name) for name in (*FIGURE_FILES, "SELECTION_MANIFEST.json")]
                       + [item["path"] for item in tex_fragments]
                       + [str(output_raw / "report" / name) for name in raw_evidence]}


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("generate",), default="generate")
    parser.add_argument("--doc", type=Path, default=C.DOC)
    parser.add_argument("--raw", type=Path, default=C.RAW)
    parser.add_argument("--output-doc", type=Path)
    parser.add_argument("--output-raw", type=Path)
    parser.add_argument("--root", type=Path, default=C.ROOT)
    arguments = parser.parse_args(argv)
    result = generate(doc=arguments.doc, raw=arguments.raw,
                      output_doc=arguments.output_doc, output_raw=arguments.output_raw,
                      root=arguments.root)
    print(json.dumps({
        "overall_status": result["tables"]["overall_status"],
        "execution_complete": result["tables"]["execution_complete"],
        "figures_complete": result["manifest"]["complete"],
        "remaining_items": len(result["remaining"]),
        "outputs": result["outputs"],
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
