#!/usr/bin/env python3
"""Build the fixed Paper-S2 FULL7 19-frame pseudo-label dataset.

The membership is the filename set already cached under
``filterval_passfail_withf3/<domain>/pass/*.jpg``.  Those JPEGs are annotated
evaluation overlays: this script uses *only their domain and filename*, never
their pixels.  Every training image is resolved independently to one unique
``data/pallet/raw_data/**/rgb/<frame>`` source.

Pseudo labels are the nine raw heatmap peaks produced by
``paper_s2_stageB/net_epoch_0057.pth`` with the Stage-B squash-parity evaluator.
They are not GT points and are not PnP-reprojected.  Missing channels are
encoded as ``[-100, -100]`` and accompanied by ``pseudo_keypoint_valid``.

Safety defaults:

* no flag (or ``--dry-run``) performs a CPU-only inventory check; it does not
  import torch, load the model, create a directory, or copy an image;
* ``--execute`` is required for inference and writing;
* a non-empty target is rejected unless ``--force`` is explicit;
* selection membership, domain counts, checkpoint hash, original-RGB paths,
  and cached baseline predictions are all checked fail-closed.

The 19 frames are evaluation-derived.  A model trained with this dataset must
not subsequently report independent metrics on these same frames.
"""

from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data/pallet/raw_data"
TRAINING_ROOT = ROOT / "data/pallet/training_data"
OUTPUT_DIR = TRAINING_ROOT / "paper_s2_full7_pl19_r1"

SELECTION_ROOT = (
    ROOT
    / "data/pallet/eval_results/paper_s2_scratch_diffpnp"
    / "filterval_passfail_withf3"
)
CHECKPOINT = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
CHECKPOINT_SHA256 = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"

DOMAIN_ORDER = ("outside", "manual", "noapril", "night", "cad")
EXPECTED_COUNTS = {
    "outside": 10,
    "manual": 3,
    "noapril": 1,
    "night": 5,
    "cad": 0,
}
EXPECTED_TOTAL = 19
EXPECTED_MEMBERSHIP_SHA256 = (
    "75c95e46fb700e36d3b463bb0a14adf4f67aa197c76ecd548155c254a3162926"
)

FULL7_FILTERS = (
    "f1_peak",
    "f2_peak_ratio",
    "f3_flip",
    "f4_tta_stab",
    "f5_rear_conf",
    "f6_frsep",
    "f7_posdepth",
)
FILTER_THRESHOLDS = {
    "f1_peak": 0.5,
    "f2_peak_ratio": 1.5,
    "f3_flip_px_max": 10.0,
    "f4_tta_stab_px_max": 5.0,
    "f5_rear_conf": 0.5,
    "f6_frsep": 0.06,
    "f7_posdepth": True,
}
INFERENCE_THRESHOLD = 0.3
MISSING_KEYPOINT = [-100.0, -100.0]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# These caches contain GT-derived metrics too, but this script deliberately
# consumes only fid/n_det/pred8.  In particular, gt8 is never read or emitted.
BASELINE_CACHES = (
    ROOT
    / "data/pallet/eval_results/paper_s2_rgb1_projected_span_v1/quick/cache"
    / "baseline_ep57__filterval__9f81b37ec7a518bd.json",
    ROOT
    / "data/pallet/eval_results/paper_s2_rgb1_projected_span_v1/quick/cache"
    / "baseline_ep57__handannot17__68af2c3c34f61393.json",
)
BASELINE_TOLERANCE_PX = 0.05


class BuildError(RuntimeError):
    """Fail-closed validation error."""


def rel(path: Path) -> str:
    """Repository-relative POSIX path for manifests and logs."""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def membership_sha(rows: list[dict[str, Any]]) -> str:
    text = "".join(
        f"{row['domain']}\t{row['frame']}\n"
        for row in sorted(rows, key=lambda x: (x["domain"], x["frame"]))
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_selection() -> list[dict[str, Any]]:
    """Read only domain/frame membership from the cached overlay filenames."""
    if not SELECTION_ROOT.is_dir():
        raise BuildError(f"selection root missing: {SELECTION_ROOT}")

    rows: list[dict[str, Any]] = []
    actual_counts: dict[str, int] = {}
    seen: set[str] = set()
    for domain in DOMAIN_ORDER:
        pass_dir = SELECTION_ROOT / domain / "pass"
        if not pass_dir.is_dir():
            raise BuildError(f"selection pass directory missing: {pass_dir}")
        files = sorted(pass_dir.glob("*.jpg"))
        actual_counts[domain] = len(files)
        for overlay in files:
            fid = overlay.stem
            if not fid.isdigit():
                raise BuildError(f"non-frame overlay name: {overlay}")
            if fid in seen:
                raise BuildError(f"duplicate frame id across domains: {fid}")
            seen.add(fid)
            rows.append(
                {
                    "domain": domain,
                    "frame": fid,
                    "selection_overlay": rel(overlay),
                }
            )

    if actual_counts != EXPECTED_COUNTS:
        raise BuildError(
            f"FULL7 count drift: expected={EXPECTED_COUNTS}, actual={actual_counts}"
        )
    if len(rows) != EXPECTED_TOTAL:
        raise BuildError(f"FULL7 total drift: expected=19, actual={len(rows)}")
    digest = membership_sha(rows)
    if digest != EXPECTED_MEMBERSHIP_SHA256:
        raise BuildError(
            "FULL7 membership drift: "
            f"expected={EXPECTED_MEMBERSHIP_SHA256}, actual={digest}"
        )
    return sorted(rows, key=lambda x: (DOMAIN_ORDER.index(x["domain"]), x["frame"]))


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_original_rgb(path: Path) -> Path:
    """Reject overlays, symlinks, and anything outside raw_data/**/rgb."""
    if path.is_symlink():
        raise BuildError(f"original RGB must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file() or not _under(resolved, RAW_ROOT):
        raise BuildError(f"RGB source is not a raw_data file: {path}")
    if path.parent.name != "rgb":
        raise BuildError(f"RGB source parent is not named rgb: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise BuildError(f"unsupported RGB suffix: {path}")
    if any("overlay" in part.lower() for part in path.parts):
        raise BuildError(f"overlay path rejected as training RGB: {path}")
    if "eval_results" in path.parts:
        raise BuildError(f"evaluation output rejected as training RGB: {path}")
    return resolved


def resolve_original_rgbs(rows: list[dict[str, Any]]) -> None:
    """Resolve every selected stem to exactly one raw_data/**/rgb image."""
    wanted = {row["frame"] for row in rows}
    hits: dict[str, list[Path]] = {fid: [] for fid in wanted}
    for rgb_dir in RAW_ROOT.rglob("rgb"):
        if not rgb_dir.is_dir():
            continue
        for child in rgb_dir.iterdir():
            if (
                child.is_file()
                and not child.is_symlink()
                and child.suffix.lower() in IMAGE_SUFFIXES
                and child.stem in wanted
            ):
                hits[child.stem].append(child)

    for row in rows:
        candidates = hits[row["frame"]]
        if len(candidates) != 1:
            raise BuildError(
                f"frame {row['frame']} must resolve to one original RGB; "
                f"found {len(candidates)}: {candidates}"
            )
        source = validate_original_rgb(candidates[0])
        row["source_rgb_path"] = source
        row["source_rgb"] = rel(source)

        camera_path = source.parent.parent / "cam_K.txt"
        if not camera_path.is_file() or not _under(camera_path, RAW_ROOT):
            raise BuildError(f"camera calibration missing for {source}: {camera_path}")
        row["camera_path"] = camera_path.resolve()
        row["camera_calibration"] = rel(camera_path)


def read_camera_matrix(path: Path) -> list[list[float]]:
    values: list[float] = []
    for token in path.read_text(encoding="utf-8").split():
        try:
            values.append(float(token))
        except ValueError as exc:
            raise BuildError(f"invalid camera calibration token in {path}: {token}") from exc
    if len(values) != 9 or not all(math.isfinite(v) for v in values):
        raise BuildError(f"camera calibration must contain 9 finite values: {path}")
    matrix = [values[0:3], values[3:6], values[6:9]]
    if abs(matrix[2][2] - 1.0) > 1e-6:
        raise BuildError(f"unexpected camera calibration bottom-right value: {path}")
    return matrix


def load_baseline_predictions() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Load only cached prediction fields used for inference-parity checks."""
    by_frame: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, str]] = []
    for cache in BASELINE_CACHES:
        if not cache.is_file():
            raise BuildError(f"baseline prediction cache missing: {cache}")
        payload = json.loads(cache.read_text(encoding="utf-8"))
        checkpoint = Path(payload.get("checkpoint", ""))
        if checkpoint.name != CHECKPOINT.name:
            raise BuildError(f"baseline cache checkpoint mismatch: {cache}")
        provenance.append({"path": rel(cache), "sha256": sha256_file(cache)})
        for cached in payload.get("rows", []):
            fid = str(cached.get("fid", ""))
            if not fid:
                continue
            # Deliberately whitelist inference-only fields.  Do not retain gt8
            # or any error/pose metric from the evaluation cache.
            entry = {
                "n_det": int(cached["n_det"]),
                "pred8": cached["pred8"],
                "cache": rel(cache),
            }
            if fid in by_frame:
                raise BuildError(f"duplicate frame across baseline caches: {fid}")
            by_frame[fid] = entry
    return by_frame, provenance


def validate_checkpoint() -> str:
    if not CHECKPOINT.is_file():
        raise BuildError(f"checkpoint missing: {CHECKPOINT}")
    digest = sha256_file(CHECKPOINT)
    if digest != CHECKPOINT_SHA256:
        raise BuildError(
            f"checkpoint hash mismatch: expected={CHECKPOINT_SHA256}, actual={digest}"
        )
    return digest


def target_nonempty() -> bool:
    return OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir())


def validate_target(force: bool) -> None:
    if OUTPUT_DIR.exists() and not OUTPUT_DIR.is_dir():
        raise BuildError(f"target exists but is not a directory: {OUTPUT_DIR}")
    if target_nonempty() and not force:
        raise BuildError(
            f"target is non-empty: {OUTPUT_DIR}; pass --force explicitly to replace it"
        )


def prepare_inventory(force: bool) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, str]], str
]:
    rows = scan_selection()
    resolve_original_rgbs(rows)
    for row in rows:
        # Parse calibration during dry-run too, without importing numpy/torch.
        row["camera_matrix"] = read_camera_matrix(row["camera_path"])
    checkpoint_hash = validate_checkpoint()
    baselines, baseline_provenance = load_baseline_predictions()
    missing_cache = [row["frame"] for row in rows if row["frame"] not in baselines]
    if missing_cache:
        raise BuildError(f"selected frames missing baseline predictions: {missing_cache}")
    validate_target(force)
    return rows, baselines, baseline_provenance, checkpoint_hash


def print_inventory(
    rows: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    checkpoint_hash: str,
    force: bool,
) -> None:
    print("[selection] FULL7 = " + " & ".join(FULL7_FILTERS))
    print(f"[selection] total={len(rows)} counts={EXPECTED_COUNTS}")
    print(f"[selection] membership_sha256={membership_sha(rows)}")
    print(f"[checkpoint] {rel(CHECKPOINT)} sha256={checkpoint_hash}")
    print(f"[baseline] cached prediction rows={sum(r['frame'] in baselines for r in rows)}/19")
    print(f"[target] {rel(OUTPUT_DIR)} nonempty={target_nonempty()} force={force}")
    for row in rows:
        print(
            f"  {row['domain']:<8} {row['frame']}  "
            f"raw={row['source_rgb']}  K={row['camera_calibration']}  "
            f"cached_n_det={baselines[row['frame']]['n_det']}"
        )


def convert_predictions(pred8: Any, pred_c: Any) -> tuple[list[list[float]], list[bool]]:
    """Convert evaluator output to 9 NDDS-like points plus validity mask."""
    points: list[list[float]] = []
    valid: list[bool] = []
    for i in range(8):
        x, y = float(pred8[i][0]), float(pred8[i][1])
        ok = math.isfinite(x) and math.isfinite(y)
        valid.append(ok)
        points.append([x, y] if ok else list(MISSING_KEYPOINT))
    if pred_c is None:
        valid.append(False)
        points.append(list(MISSING_KEYPOINT))
    else:
        x, y = float(pred_c[0]), float(pred_c[1])
        ok = math.isfinite(x) and math.isfinite(y)
        valid.append(ok)
        points.append([x, y] if ok else list(MISSING_KEYPOINT))
    return points, valid


def compare_to_baseline(
    points: list[list[float]], valid: list[bool], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Compare regenerated eight corner peaks with the cached ep57 predictions."""
    cached = baseline["pred8"]
    cached_valid: list[bool] = []
    distances: list[float] = []
    for i in range(8):
        cx, cy = float(cached[i][0]), float(cached[i][1])
        cok = math.isfinite(cx) and math.isfinite(cy)
        cached_valid.append(cok)
        if valid[i] and cok:
            distances.append(math.hypot(points[i][0] - cx, points[i][1] - cy))

    regenerated_n_det = sum(valid[:8])
    mask_match = valid[:8] == cached_valid
    n_det_match = regenerated_n_det == int(baseline["n_det"])
    max_px = max(distances) if distances else 0.0
    mean_px = sum(distances) / len(distances) if distances else 0.0
    match = mask_match and n_det_match and max_px <= BASELINE_TOLERANCE_PX
    return {
        "cache": baseline["cache"],
        "cached_n_detected_corners": int(baseline["n_det"]),
        "regenerated_n_detected_corners": regenerated_n_det,
        "valid_mask_match": mask_match,
        "n_detected_match": n_det_match,
        "compared_corner_count": len(distances),
        "mean_corner_delta_px": mean_px,
        "max_corner_delta_px": max_px,
        "tolerance_px": BASELINE_TOLERANCE_PX,
        "centroid_compared": False,
        "match": match,
    }


def load_evaluator_helpers(device: str) -> tuple[Any, Any, Any]:
    """Import the current Stage-B evaluator only in explicit execute mode."""
    stage0 = ROOT / "scripts/stage0"
    sys.path.insert(0, str(stage0))
    import cv2  # type: ignore
    import torch  # type: ignore
    import paper_s2_rgb1_eval as evaluator  # type: ignore

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise BuildError("CUDA execution requested but torch.cuda.is_available() is false")
    if abs(float(evaluator.THRESH) - INFERENCE_THRESHOLD) > 1e-12:
        raise BuildError(
            f"evaluator threshold drift: expected={INFERENCE_THRESHOLD}, "
            f"actual={evaluator.THRESH}"
        )
    if int(evaluator.INPUT_SIZE) != 400 or int(evaluator.GRID_SIZE) != 50:
        raise BuildError(
            "current evaluator violated the locked 400 input / 50 heatmap grid"
        )
    spec = evaluator.CheckpointSpec(
        name="paper_s2_stageB_ep57",
        arm="baseline",
        path=CHECKPOINT,
        epoch=57,
        features=(),
        baseline=True,
    )
    model = evaluator.load_model(spec, device)
    return cv2, evaluator, model


def infer_current_evaluator(
    evaluator: Any, model: Any, image: Any, device: str
) -> tuple[list[list[float]], list[bool], list[float]]:
    """Decode current evaluator grid peaks into original-image coordinates."""
    decoded = evaluator.infer(model, image, device)
    kps_grid = decoded["kps_grid"]
    if len(kps_grid) != 9:
        raise BuildError(f"evaluator returned {len(kps_grid)} keypoints, expected 9")
    height, width = image.shape[:2]
    sx = width / float(evaluator.GRID_SIZE)
    sy = height / float(evaluator.GRID_SIZE)
    points: list[list[float]] = []
    valid: list[bool] = []
    confidences: list[float] = []
    for index, point in enumerate(kps_grid):
        if len(point) < 3:
            raise BuildError(
                f"evaluator keypoint {index} lacks peak confidence: {point}")
        x_grid, y_grid, confidence = map(float, point[:3])
        ok = x_grid >= 0.0 and y_grid >= 0.0
        valid.append(ok)
        points.append(
            [x_grid * sx, y_grid * sy] if ok else list(MISSING_KEYPOINT)
        )
        confidences.append(confidence)
    return points, valid, confidences


def ndds_annotation(
    row: dict[str, Any],
    width: int,
    height: int,
    points: list[list[float]],
    valid: list[bool],
    confidences: list[float],
    checkpoint_hash: str,
) -> dict[str, Any]:
    K = row["camera_matrix"]
    pseudo_meta = {
        "source_model": "paper_s2_stageB_ep57",
        "source_checkpoint": rel(CHECKPOINT),
        "source_checkpoint_sha256": checkpoint_hash,
        "inference_preprocess": "squash_640x480_to_400x400",
        "heatmap_threshold": INFERENCE_THRESHOLD,
        "label_geometry": "raw_heatmap_peaks_no_pnp_reprojection",
        "selection": "FULL7_cached_membership",
        "selection_membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "gt_annotations_used": False,
        "selection_overlay_pixels_used": False,
    }
    obj = {
        "class": "pallet",
        "name": "real_pallet",
        "visibility": 1,
        "gt_source": "pseudo",
        "source_model": "paper_s2_stageB_ep57",
        "projected_cuboid": points[:8],
        "projected_cuboid_centroid": points[8],
        "pseudo_keypoint_valid": valid,
        "pseudo_keypoint_confidence": confidences,
        "pseudo_provenance": pseudo_meta,
    }
    return {
        "camera_data": {
            "width": width,
            "height": height,
            "intrinsics": {
                "fx": K[0][0],
                "fy": K[1][1],
                "cx": K[0][2],
                "cy": K[1][2],
            },
            "calibration_source": row["camera_calibration"],
        },
        "source_model": "paper_s2_stageB_ep57",
        "source_rgb": row["source_rgb"],
        "pseudo_keypoint_valid": valid,
        "objects": [obj],
    }


def publish_staging(staging: Path, force: bool) -> None:
    """Atomically publish a complete staging directory on the same filesystem."""
    validate_target(force)
    if OUTPUT_DIR.exists():
        if target_nonempty():
            if not force:
                raise BuildError(f"target became non-empty during build: {OUTPUT_DIR}")
            shutil.rmtree(OUTPUT_DIR)
        else:
            OUTPUT_DIR.rmdir()
    os.replace(staging, OUTPUT_DIR)


def execute_build(
    rows: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    baseline_provenance: list[dict[str, str]],
    checkpoint_hash: str,
    device: str,
    force: bool,
) -> None:
    cv2, evaluator, model = load_evaluator_helpers(device)

    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_DIR.name}.staging-", dir=TRAINING_ROOT))
    records: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, start=1):
            source = row["source_rgb_path"]
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise BuildError(f"cv2 could not read original RGB: {source}")
            height, width = image.shape[:2]

            points, valid, confidences = infer_current_evaluator(
                evaluator, model, image, device)
            comparison = compare_to_baseline(points, valid, baselines[row["frame"]])
            if not comparison["match"]:
                raise BuildError(
                    f"ep57 prediction drift for {row['frame']}: {comparison}"
                )

            destination_image = staging / source.name
            destination_json = staging / f"{row['frame']}.json"
            if destination_image.exists() or destination_json.exists():
                raise BuildError(f"duplicate staging destination for {row['frame']}")
            shutil.copy2(source, destination_image)
            annotation = ndds_annotation(
                row,
                width,
                height,
                points,
                valid,
                confidences,
                checkpoint_hash,
            )
            destination_json.write_text(
                json.dumps(annotation, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            records.append(
                {
                    "frame": row["frame"],
                    "domain": row["domain"],
                    "source_rgb": row["source_rgb"],
                    "camera_calibration": row["camera_calibration"],
                    "selection_overlay_membership": row["selection_overlay"],
                    "output_image": destination_image.name,
                    "output_json": destination_json.name,
                    "n_detected_corners": sum(valid[:8]),
                    "n_valid_keypoints": sum(valid),
                    "pseudo_keypoint_valid": valid,
                    "pseudo_keypoint_confidence": confidences,
                    "baseline_prediction_compare": comparison,
                }
            )
            print(
                f"[{index:02d}/{len(rows)}] {row['domain']:<8} {row['frame']} "
                f"corners={sum(valid[:8])}/8 valid9={sum(valid)}/9 "
                f"baseline_max_delta={comparison['max_corner_delta_px']:.6f}px"
            )

        if len(records) != EXPECTED_TOTAL:
            raise BuildError(f"write count mismatch: {len(records)} != {EXPECTED_TOTAL}")
        comparison_mismatches = [
            record["frame"]
            for record in records
            if not record["baseline_prediction_compare"]["match"]
        ]
        if comparison_mismatches:
            raise BuildError(f"baseline comparison failures: {comparison_mismatches}")

        records_path = staging / "_records.json"
        records_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        valid_hist: dict[str, int] = {}
        for record in records:
            key = str(record["n_valid_keypoints"])
            valid_hist[key] = valid_hist.get(key, 0) + 1
        manifest = {
            "schema": "paper_s2_full7_pseudo_labels_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_model": "paper_s2_stageB_ep57",
            "source_checkpoint": rel(CHECKPOINT),
            "source_checkpoint_sha256": checkpoint_hash,
            "inference": {
                "helper": "scripts/stage0/paper_s2/paper_s2_rgb1_eval.py::infer",
                "preprocess": "squash_640x480_to_400x400",
                "heatmap_threshold": INFERENCE_THRESHOLD,
                "label_geometry": "raw_heatmap_peaks_no_pnp_reprojection",
                "device": device,
            },
            "selection": {
                "name": "FULL7",
                "expression": " & ".join(FULL7_FILTERS),
                "thresholds": FILTER_THRESHOLDS,
                "root": rel(SELECTION_ROOT),
                "membership_source": "pass JPG filenames only",
                "membership_sha256": membership_sha(rows),
                "counts_by_domain": EXPECTED_COUNTS,
                "n_selected": len(rows),
                "overlay_pixels_used": False,
            },
            "data_safety": {
                "original_rgb_root": rel(RAW_ROOT),
                "all_sources_are_unique_raw_data_rgb": True,
                "gt_annotations_used": False,
                "pnp_reprojection_used_for_labels": False,
                "evaluation_contamination_warning": (
                    "These 19 frames came from filterval/manual/handannot evaluation "
                    "membership and cannot remain independent evaluation frames after training."
                ),
            },
            "output_dir": rel(OUTPUT_DIR),
            "n_written": len(records),
            "valid_keypoint_count_histogram": valid_hist,
            "missing_keypoint_encoding": MISSING_KEYPOINT,
            "records": "_records.json",
            "baseline_prediction_comparison": {
                "fields_consumed": ["fid", "n_det", "pred8"],
                "gt_fields_consumed": [],
                "caches": baseline_provenance,
                "n_compared": len(records),
                "n_match": len(records) - len(comparison_mismatches),
                "n_mismatch": len(comparison_mismatches),
                "corner_tolerance_px": BASELINE_TOLERANCE_PX,
                "centroid_available_in_cache": False,
            },
        }
        (staging / "_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Final staging integrity: exactly 19 image/JSON pairs plus two metadata files.
        image_count = sum(
            1 for path in staging.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        label_count = sum(
            1
            for path in staging.glob("*.json")
            if path.name not in {"_manifest.json", "_records.json"}
        )
        if image_count != EXPECTED_TOTAL or label_count != EXPECTED_TOTAL:
            raise BuildError(
                f"staging pair count mismatch: images={image_count}, labels={label_count}"
            )
        publish_staging(staging, force)
        print(f"[done] wrote {len(records)} original-RGB/JSON pairs -> {OUTPUT_DIR}")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only inventory check; this is the default",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="load ep57, regenerate pseudo labels, and publish the dataset",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device used only with --execute (default: cuda)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --execute, replace the exact non-empty output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows, baselines, baseline_provenance, checkpoint_hash = prepare_inventory(
            args.force
        )
        print_inventory(rows, baselines, checkpoint_hash, args.force)
        if not args.execute:
            print(
                "[dry-run PASS] CPU-only path/hash/cache validation complete; "
                "no model imported, no inference run, no files written."
            )
            return 0
        execute_build(
            rows,
            baselines,
            baseline_provenance,
            checkpoint_hash,
            args.device,
            args.force,
        )
        return 0
    except BuildError as exc:
        print(f"[FAIL CLOSED] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
