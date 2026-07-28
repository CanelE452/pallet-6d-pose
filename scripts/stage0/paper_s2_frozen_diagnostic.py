#!/usr/bin/env python3
"""Frozen, read-only diagnostic audit for the canonical PAPER_S2 ep57 model.

This script deliberately evaluates only:

* the fixed Q1 synthetic validation list (500 frames), and
* stage25's filter-val membership (outside 44 + night 43 + archived manual 36).

The strict real-data primary is the outside/night subset (N=87).  The archived
manual subset (N=36) came from the capturepallet11 PL pool and is therefore
labelled exploratory rather than silently folded into the primary result.

No training, checkpoint selection, pseudo-label generation, final-test input,
or hand-annotated final-test input is performed.  Every run creates a new
directory and every output file is opened in exclusive-create mode.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
WEIGHTS_SHA256 = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"
Q1_LIST_SHA256 = "5a88384f045faf22dda48465b440e69dba78bc94420f10a3db5217390befb56d"
STRICT_FILTERVAL_ID_SHA256 = "2795991dbf7f2c3dcc45132ea18a048a1373893aa0023b9e4bbc81266c1123dd"
EXPLORATORY_MANUAL36_ID_SHA256 = "1d8c8998623258c8ca90a3dd5c47eb4c49c6136d17b6dadfa49df9357dcb3f4b"
LEGACY_FILTERVAL123_ID_SHA256 = "ee5f766347bd1bf33ceec899c7d167a33bc5e4f0cc4680e860cb78a9efc68766"
SYNTH_LIST = (
    ROOT
    / "data/pallet/results/paper_s2_scratch_diffpnp/q1_split/val_list.json"
)
MANUAL36 = (
    ROOT
    / "data/pallet/eval_results/achieve/paper_base_v2_s2/"
    "stage0_gt_candidates/manual_gt"
)
OUT_ROOT = (
    ROOT / "data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit"
)

for _p in (
    ROOT / "Deep_Object_Pose/common",
    ROOT / "scripts/stage0",
    ROOT / "challenge/scripts",
):
    sys.path.insert(0, str(_p))

from models import DopeNetwork  # noqa: E402
import annotate_pnp as APNP  # noqa: E402
import stage25_paperbase_eval as STAGE25  # noqa: E402
from stage18_elevation_threshold import elev_from_pose  # noqa: E402


INPUT_SIZE = 400
N_KEYPOINTS = 9
BELIEF_THRESHOLD = 0.3
LOCAL_RADIUS = 3
LOCAL_TEMPERATURE = 0.1
MOMENT_TEMPERATURE = 1.0
COV_REGULARIZER_PX2 = 1e-6
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FLIP_PARTNER = (1, 0, 3, 2, 5, 4, 7, 6, 8)
SEALED_SESSIONS = frozenset(
    ("capturepallet09", "capturepallet07", "capturenight09", "capturenight08")
)
PROHIBITED_INPUT_TOKENS = tuple(SEALED_SESSIONS) + (
    "testset_full8_manifest",
    "handannot17",
)

CSV_NAMES = (
    "frames",
    "keypoints",
    "yaw_ladder",
    "keypoint_influence",
    "keypoint_perturbation",
    "kp5_perturbation",
    "kp5_geometry",
    "solver_comparison",
    "flip_consistency",
    "flip_keypoints",
)
FIGURE_NAMES = (
    "yaw_cause_ladder.png",
    "keypoint_influence_delta_yaw.png",
    "kp5_perturbation_sensitivity.png",
    "centroid_residual_vs_elevation.png",
    "covariance_coverage_calibration.png",
    "solver_yaw_reproj_add.png",
    "flip_equivariance_vs_keypoint_error.png",
)
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_REPLICATES = 10_000


def finite_float(value: Any) -> Optional[float]:
    """Return a finite Python float, otherwise None."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean_error(exc: BaseException) -> str:
    """Compact exception text suitable for a machine-readable row."""
    return f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:400]}"


def jsonable(value: Any) -> Any:
    """Recursively convert numpy and non-finite values to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def csv_value(value: Any) -> Any:
    value = jsonable(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def membership_identity_sha256(rows: Iterable[dict[str, Any]]) -> str:
    """Stable identity lock independent of repository absolute paths."""
    lines = sorted(
        "\t".join(
            (
                str(row.get("domain", "")),
                str(row.get("fid", "")),
                str(row.get("source_session") or ""),
                str(row.get("split_role", "")),
            )
        )
        for row in rows
    )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_head() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_path_status(path: Path) -> Optional[str]:
    try:
        relative = path.resolve().relative_to(ROOT)
        result = subprocess.run(
            ["git", "status", "--short", "--", str(relative)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "clean"
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class InputAudit:
    """Guard and count files whose contents are opened as evaluation inputs."""

    def __init__(self) -> None:
        self.json_paths: set[str] = set()
        self.image_paths: set[str] = set()
        self.prohibited_attempts: list[str] = []

    def guard(self, path: os.PathLike[str] | str) -> Path:
        result = Path(path).resolve()
        lower = str(result).lower()
        hit = next((token for token in PROHIBITED_INPUT_TOKENS if token in lower), None)
        if hit is not None:
            self.prohibited_attempts.append(str(result))
            raise RuntimeError(f"prohibited sealed input path ({hit}): {result}")
        return result

    def read_json(self, path: os.PathLike[str] | str) -> Any:
        safe = self.guard(path)
        self.json_paths.add(str(safe))
        with safe.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def read_image(self, path: os.PathLike[str] | str) -> Optional[np.ndarray]:
        safe = self.guard(path)
        self.image_paths.add(str(safe))
        return cv2.imread(str(safe), cv2.IMREAD_COLOR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical PAPER_S2 ep57 frozen synthetic/filter-val diagnostic"
    )
    parser.add_argument(
        "--max-synth",
        type=int,
        default=0,
        help="smoke limit for Q1 synthetic frames; 0 means all 500",
    )
    parser.add_argument(
        "--max-filterval",
        type=int,
        default=0,
        help="smoke limit for legacy filter-val frames; 0 means all 123",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device (default: CUDA when available)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="new output subdirectory name; default is a UTC timestamp",
    )
    args = parser.parse_args()
    if args.max_synth < 0 or args.max_filterval < 0:
        parser.error("--max-synth and --max-filterval must be >= 0")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def make_run_dir(run_name: Optional[str]) -> tuple[Path, str]:
    if run_name is None:
        run_name = dt.datetime.now(dt.timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_name)
        or run_name in (".", "..")
    ):
        raise ValueError(
            "--run-name must be 1-128 safe characters and begin with alphanumeric"
        )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_ROOT / run_name
    run_dir.mkdir(exist_ok=False)
    return run_dir, run_name


def safe_stage25_filterval() -> list[dict[str, Any]]:
    """Call stage25 membership with a map that never scans sealed sessions."""
    if not MANUAL36.is_dir():
        raise FileNotFoundError(f"archived manual36 directory not found: {MANUAL36}")

    fid_to_session: dict[str, str] = {}

    def allowed_session_map() -> dict[str, str]:
        mapping: dict[str, str] = {}
        raw_root = ROOT / "data/pallet/raw_data"
        for domain, sessions in STAGE25.FILTER_VAL_SESSIONS.items():
            for session in sorted(sessions):
                if session in SEALED_SESSIONS:
                    raise RuntimeError(f"sealed session in filter-val lock: {session}")
                rgb_dir = raw_root / domain / session / "rgb"
                if not rgb_dir.is_dir():
                    continue
                for png in rgb_dir.glob("*.png"):
                    mapping[png.stem] = session
        fid_to_session.update(mapping)
        return mapping

    original_builder = STAGE25.build_session_map
    original_manual = STAGE25.MANUAL_GT_DIR
    try:
        STAGE25.build_session_map = allowed_session_map
        STAGE25.MANUAL_GT_DIR = str(MANUAL36)
        raw_rows = STAGE25.frames_filterval()
    finally:
        STAGE25.build_session_map = original_builder
        STAGE25.MANUAL_GT_DIR = original_manual

    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for domain, fid, json_path, image_path in raw_rows:
        counts[domain] = counts.get(domain, 0) + 1
        if domain == "manual":
            split_role = "exploratory_pl_pool_manual"
            source_session = "capturepallet11"
            primary = False
        else:
            split_role = "strict_filterval"
            source_session = fid_to_session.get(str(fid))
            primary = True
            if source_session is None:
                raise RuntimeError(f"stage25 frame has no allowed-session provenance: {fid}")
            if source_session in SEALED_SESSIONS:
                raise RuntimeError(f"sealed filter-val provenance: {source_session}/{fid}")
        rows.append(
            {
                "dataset": "filterval",
                "domain": domain,
                "fid": str(fid),
                "json": str(Path(json_path).resolve()),
                "png": str(Path(image_path).resolve()),
                "source_session": source_session,
                "split_role": split_role,
                "is_primary": primary,
                "legacy_aggregate": "legacy_filterval_123",
                "entry": None,
            }
        )

    expected = {"outside": 44, "night": 43, "manual": 36}
    if counts != expected or len(rows) != 123:
        raise RuntimeError(
            f"stage25 filter-val contract changed: counts={counts}, total={len(rows)}"
        )
    strict_hash = membership_identity_sha256(
        row for row in rows if row["split_role"] == "strict_filterval"
    )
    manual_hash = membership_identity_sha256(
        row for row in rows if row["split_role"] == "exploratory_pl_pool_manual"
    )
    aggregate_hash = membership_identity_sha256(rows)
    for label, actual, expected_hash in (
        ("strict_filterval", strict_hash, STRICT_FILTERVAL_ID_SHA256),
        ("exploratory_manual36", manual_hash, EXPLORATORY_MANUAL36_ID_SHA256),
        ("legacy_filterval123", aggregate_hash, LEGACY_FILTERVAL123_ID_SHA256),
    ):
        if expected_hash and actual != expected_hash:
            raise RuntimeError(
                f"{label} membership identity lock changed: {actual} != {expected_hash}"
            )
    return rows


def synth_frames(audit: InputAudit) -> list[dict[str, Any]]:
    actual_sha = sha256_file(SYNTH_LIST)
    if actual_sha != Q1_LIST_SHA256:
        raise RuntimeError(
            f"Q1 val-list SHA changed: {actual_sha} != {Q1_LIST_SHA256}"
        )
    raw = audit.read_json(SYNTH_LIST)
    if not isinstance(raw, list) or len(raw) != 500:
        raise RuntimeError(f"Q1 val-list contract changed: expected 500, got {len(raw)}")
    rows = []
    for item in raw:
        rows.append(
            {
                "dataset": "synthetic_q1_val",
                "domain": "synthetic",
                "fid": str(item["fid"]),
                "json": str(Path(item["json"]).resolve()),
                "png": str(Path(item["png"]).resolve()),
                "source_session": None,
                "split_role": "synthetic_fixed_val",
                "is_primary": True,
                "legacy_aggregate": "q1_fixed_val_500",
                "entry": item.get("entry"),
            }
        )
    return rows


def load_model(device: torch.device) -> tuple[DopeNetwork, int]:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"canonical checkpoint not found: {WEIGHTS}")
    actual_sha = sha256_file(WEIGHTS)
    if actual_sha != WEIGHTS_SHA256:
        raise RuntimeError(
            f"canonical checkpoint SHA mismatch: {actual_sha} != {WEIGHTS_SHA256}"
        )
    try:
        state = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(WEIGHTS), map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError("canonical checkpoint is not a state dictionary")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model = DopeNetwork(numVec=0, numSeg=1)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    return model, len(state)


def preprocess_squash(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)


def infer_belief(
    model: DopeNetwork, image_bgr: np.ndarray, device: torch.device
) -> tuple[np.ndarray, float]:
    tensor = preprocess_squash(image_bgr).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(tensor)
        belief_tensor = outputs[0][-1][0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    belief = belief_tensor.detach().float().cpu().numpy()
    if belief.shape[0] != N_KEYPOINTS:
        raise RuntimeError(f"expected 9 belief channels, got {belief.shape}")
    return belief, elapsed_ms


def point_valid(point: Any) -> bool:
    if point is None:
        return False
    try:
        value = np.asarray(point, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return (
        len(value) >= 2
        and bool(np.isfinite(value[:2]).all())
        and not (value[0] == -1.0 and value[1] == -1.0)
    )


def point_xy(point: Any) -> Optional[list[float]]:
    if not point_valid(point):
        return None
    value = np.asarray(point, dtype=np.float64).reshape(-1)
    return [float(value[0]), float(value[1])]


def point_inside(point: Any, width: int, height: int) -> Optional[bool]:
    xy = point_xy(point)
    if xy is None:
        return None
    return bool(0.0 <= xy[0] < width and 0.0 <= xy[1] < height)


def euclidean(a: Any, b: Any) -> Optional[float]:
    aa, bb = point_xy(a), point_xy(b)
    if aa is None or bb is None:
        return None
    return float(np.linalg.norm(np.asarray(aa) - np.asarray(bb)))


def order_free_corner_metrics(
    predicted: list[Optional[list[float]]],
    ground_truth: list[Optional[list[float]]],
) -> dict[str, Any]:
    """Hungarian corner distance without assuming the synthetic channel order."""
    pred = [point_xy(point) for point in predicted[:8] if point_valid(point)]
    gt = [point_xy(point) for point in ground_truth[:8] if point_valid(point)]
    if not pred or not gt:
        return {
            "matched_count": 0,
            "mean_px": None,
            "median_px": None,
            "max_px": None,
        }
    from scipy.optimize import linear_sum_assignment

    pred_array = np.asarray(pred, dtype=np.float64)
    gt_array = np.asarray(gt, dtype=np.float64)
    cost = np.linalg.norm(
        pred_array[:, np.newaxis, :] - gt_array[np.newaxis, :, :], axis=2
    )
    row_indices, column_indices = linear_sum_assignment(cost)
    distances = cost[row_indices, column_indices]
    return {
        "matched_count": int(len(distances)),
        "mean_px": float(np.mean(distances)),
        "median_px": float(np.median(distances)),
        "max_px": float(np.max(distances)),
    }


def heatmap_stats(
    heatmap: np.ndarray,
    scale_x: float,
    scale_y: float,
    gt_point: Any,
) -> dict[str, Any]:
    """Raw argmax, 7x7 local softargmax, and full-softmax moments."""
    values = np.asarray(heatmap, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).any():
        return {"valid": False}
    work = values.copy()
    finite = np.isfinite(work)
    floor = float(np.min(work[finite]) - 100.0)
    work[~finite] = floor
    height, width = work.shape
    flat_index = int(np.argmax(work))
    arg_y, arg_x = np.unravel_index(flat_index, work.shape)
    peak = float(work[arg_y, arg_x])

    y_indices = np.clip(
        np.arange(arg_y - LOCAL_RADIUS, arg_y + LOCAL_RADIUS + 1),
        0,
        height - 1,
    )
    x_indices = np.clip(
        np.arange(arg_x - LOCAL_RADIUS, arg_x + LOCAL_RADIUS + 1),
        0,
        width - 1,
    )
    patch = work[np.ix_(y_indices, x_indices)]
    local_logits = (patch - float(np.max(patch))) / LOCAL_TEMPERATURE
    local_prob = np.exp(local_logits)
    local_prob /= float(local_prob.sum())
    local_xs, local_ys = np.meshgrid(x_indices, y_indices)
    soft_x = float(np.sum(local_prob * local_xs))
    soft_y = float(np.sum(local_prob * local_ys))
    local_dx, local_dy = local_xs - soft_x, local_ys - soft_y
    local_cov_xx_grid = float(np.sum(local_prob * local_dx * local_dx))
    local_cov_xy_grid = float(np.sum(local_prob * local_dx * local_dy))
    local_cov_yy_grid = float(np.sum(local_prob * local_dy * local_dy))
    local_entropy = float(
        -np.sum(local_prob * np.log(np.maximum(local_prob, 1e-300)))
    )
    local_entropy_norm = local_entropy / math.log(float(local_prob.size))

    outside = np.ones_like(work, dtype=bool)
    outside[
        max(0, arg_y - LOCAL_RADIUS) : min(height, arg_y + LOCAL_RADIUS + 1),
        max(0, arg_x - LOCAL_RADIUS) : min(width, arg_x + LOCAL_RADIUS + 1),
    ] = False
    second = float(np.max(work[outside])) if bool(outside.any()) else None
    ratio = peak / second if second is not None and second > 0.0 else None
    margin = peak - second if second is not None else None

    logits = (work - float(np.max(work))) / MOMENT_TEMPERATURE
    probability = np.exp(logits)
    probability /= float(probability.sum())
    ys, xs = np.mgrid[0:height, 0:width]
    global_mean_x = float(np.sum(probability * xs))
    global_mean_y = float(np.sum(probability * ys))
    global_dx, global_dy = xs - global_mean_x, ys - global_mean_y
    global_cov_xx_grid = float(np.sum(probability * global_dx * global_dx))
    global_cov_xy_grid = float(np.sum(probability * global_dx * global_dy))
    global_cov_yy_grid = float(np.sum(probability * global_dy * global_dy))
    global_entropy = float(
        -np.sum(probability * np.log(np.maximum(probability, 1e-300)))
    )
    global_entropy_norm = global_entropy / math.log(float(width * height))

    # A second full-map representation keeps raw-value normalization separate
    # from interpreting the heatmap as logits. Negative raw beliefs cannot be
    # probabilities, so this representation is explicitly clipped at zero.
    raw_nonnegative = np.maximum(work, 0.0)
    raw_sum = float(raw_nonnegative.sum())
    raw_probability = raw_nonnegative / raw_sum if raw_sum > 0.0 else None
    if raw_probability is None:
        raw_mean_x = raw_mean_y = None
        raw_cov_xx_grid = raw_cov_xy_grid = raw_cov_yy_grid = None
        raw_entropy = raw_entropy_norm = None
    else:
        raw_mean_x = float(np.sum(raw_probability * xs))
        raw_mean_y = float(np.sum(raw_probability * ys))
        raw_dx, raw_dy = xs - raw_mean_x, ys - raw_mean_y
        raw_cov_xx_grid = float(np.sum(raw_probability * raw_dx * raw_dx))
        raw_cov_xy_grid = float(np.sum(raw_probability * raw_dx * raw_dy))
        raw_cov_yy_grid = float(np.sum(raw_probability * raw_dy * raw_dy))
        raw_entropy = float(
            -np.sum(raw_probability * np.log(np.maximum(raw_probability, 1e-300)))
        )
        raw_entropy_norm = raw_entropy / math.log(float(width * height))

    arg_px = [arg_x * scale_x, arg_y * scale_y]
    soft_px = [soft_x * scale_x, soft_y * scale_y]
    mean_px = list(soft_px)
    covariance = np.array(
        [
            [
                local_cov_xx_grid * scale_x * scale_x,
                local_cov_xy_grid * scale_x * scale_y,
            ],
            [
                local_cov_xy_grid * scale_x * scale_y,
                local_cov_yy_grid * scale_y * scale_y,
            ],
        ],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    major = eigenvectors[:, order[0]]
    major_angle = math.degrees(math.atan2(float(major[1]), float(major[0])))

    mahalanobis = None
    gt = point_xy(gt_point)
    if gt is not None:
        delta = np.asarray(gt, dtype=np.float64) - np.asarray(mean_px)
        try:
            inv = np.linalg.inv(
                covariance + COV_REGULARIZER_PX2 * np.eye(2, dtype=np.float64)
            )
            square = float(delta @ inv @ delta)
            mahalanobis = math.sqrt(max(square, 0.0)) if math.isfinite(square) else None
        except np.linalg.LinAlgError:
            mahalanobis = None

    return {
        "valid": True,
        "peak": peak,
        "second_peak": second,
        "peak_second_ratio": ratio,
        "peak_second_margin": margin,
        "detected": peak >= BELIEF_THRESHOLD,
        "argmax_grid_x": float(arg_x),
        "argmax_grid_y": float(arg_y),
        "argmax_x": float(arg_px[0]),
        "argmax_y": float(arg_px[1]),
        "softargmax_grid_x": soft_x,
        "softargmax_grid_y": soft_y,
        "softargmax_x": float(soft_px[0]),
        "softargmax_y": float(soft_px[1]),
        "moment_representation": "local_7x7_logits_softmax_T0.1",
        "moment_grid_x": soft_x,
        "moment_grid_y": soft_y,
        "moment_x": float(mean_px[0]),
        "moment_y": float(mean_px[1]),
        "entropy_nats": local_entropy,
        "entropy_normalized": local_entropy_norm,
        "cov_grid_xx": local_cov_xx_grid,
        "cov_grid_xy": local_cov_xy_grid,
        "cov_grid_yy": local_cov_yy_grid,
        "cov_px_xx": float(covariance[0, 0]),
        "cov_px_xy": float(covariance[0, 1]),
        "cov_px_yy": float(covariance[1, 1]),
        "cov_eig_major_px2": float(eigenvalues[0]),
        "cov_eig_minor_px2": float(eigenvalues[1]),
        "cov_major_angle_deg": major_angle,
        "mahalanobis_gt": mahalanobis,
        "argmax_error_gt_px": euclidean(arg_px, gt),
        "softargmax_error_gt_px": euclidean(soft_px, gt),
        "moment_error_gt_px": euclidean(mean_px, gt),
        "global_logits_representation": "full_map_logits_softmax_T1.0",
        "global_logits_moment_grid_x": global_mean_x,
        "global_logits_moment_grid_y": global_mean_y,
        "global_logits_moment_x": global_mean_x * scale_x,
        "global_logits_moment_y": global_mean_y * scale_y,
        "global_logits_entropy_nats": global_entropy,
        "global_logits_entropy_normalized": global_entropy_norm,
        "global_logits_cov_grid_xx": global_cov_xx_grid,
        "global_logits_cov_grid_xy": global_cov_xy_grid,
        "global_logits_cov_grid_yy": global_cov_yy_grid,
        "global_logits_cov_px_xx": global_cov_xx_grid * scale_x * scale_x,
        "global_logits_cov_px_xy": global_cov_xy_grid * scale_x * scale_y,
        "global_logits_cov_px_yy": global_cov_yy_grid * scale_y * scale_y,
        "rawnorm_representation": "full_map_clip(raw_belief,0,inf)_then_sum_normalize",
        "rawnorm_available": raw_probability is not None,
        "rawnorm_moment_grid_x": raw_mean_x,
        "rawnorm_moment_grid_y": raw_mean_y,
        "rawnorm_moment_x": None if raw_mean_x is None else raw_mean_x * scale_x,
        "rawnorm_moment_y": None if raw_mean_y is None else raw_mean_y * scale_y,
        "rawnorm_entropy_nats": raw_entropy,
        "rawnorm_entropy_normalized": raw_entropy_norm,
        "rawnorm_cov_grid_xx": raw_cov_xx_grid,
        "rawnorm_cov_grid_xy": raw_cov_xy_grid,
        "rawnorm_cov_grid_yy": raw_cov_yy_grid,
        "rawnorm_cov_px_xx": (
            None if raw_cov_xx_grid is None else raw_cov_xx_grid * scale_x * scale_x
        ),
        "rawnorm_cov_px_xy": (
            None if raw_cov_xy_grid is None else raw_cov_xy_grid * scale_x * scale_y
        ),
        "rawnorm_cov_px_yy": (
            None if raw_cov_yy_grid is None else raw_cov_yy_grid * scale_y * scale_y
        ),
        "_arg_px": arg_px,
        "_soft_px": soft_px,
        "_moment_px": mean_px,
        "_cov_px": covariance,
    }


def intrinsics_from_json(data: dict[str, Any]) -> Optional[np.ndarray]:
    try:
        intrinsics = data["camera_data"]["intrinsics"]
        values = (
            finite_float(intrinsics.get("fx")),
            finite_float(intrinsics.get("fy")),
            finite_float(intrinsics.get("cx")),
            finite_float(intrinsics.get("cy")),
        )
        if any(value is None for value in values):
            return None
        fx, fy, cx, cy = values
        if fx <= 0.0 or fy <= 0.0:
            return None
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    except (KeyError, TypeError):
        return None


def dims_from_frame(spec: dict[str, Any], obj: dict[str, Any]) -> Optional[tuple[float, float, float]]:
    entry = spec.get("entry")
    if isinstance(entry, dict) and isinstance(entry.get("dims"), dict):
        raw = entry["dims"]
        candidate = (raw.get("W"), raw.get("D"), raw.get("H"))
    else:
        raw = obj.get("dimensions_m")
        if not isinstance(raw, dict):
            return None
        candidate = (raw.get("width"), raw.get("depth"), raw.get("height"))
    values = tuple(finite_float(value) for value in candidate)
    if any(value is None or value <= 0.0 for value in values):
        return None
    return values  # type: ignore[return-value]


def gt_points_from_object(obj: dict[str, Any]) -> list[Optional[list[float]]]:
    corners = obj.get("projected_cuboid")
    result: list[Optional[list[float]]] = []
    if isinstance(corners, list):
        result.extend(point_xy(value) for value in corners[:8])
    result.extend([None] * (8 - len(result)))
    centroid = point_xy(obj.get("projected_cuboid_centroid"))
    if centroid is None and isinstance(corners, list) and len(corners) > 8:
        centroid = point_xy(corners[8])
    result.append(centroid)
    return result


def stored_pose_from_object(obj: dict[str, Any]) -> Optional[dict[str, np.ndarray]]:
    transform = obj.get("pose_transform")
    if transform is None:
        return None
    try:
        matrix = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return {"R": matrix[:3, :3].copy(), "t": matrix[:3, 3].copy()}


def yaw_deg(rotation: np.ndarray) -> float:
    """Camera yaw of the pallet local forward/Z axis."""
    return float(math.degrees(math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))))


def orientation_yaw_pitch_roll(rotation: np.ndarray) -> tuple[float, float, float]:
    """Y-X-Z decomposition: R = Ry(yaw) Rx(pitch) Rz(roll).

    This decomposition keeps yaw identical to the pallet local-forward/Z
    definition used by the primary metric.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    pitch = math.asin(float(np.clip(-rotation[1, 2], -1.0, 1.0)))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1e-8:
        yaw = math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))
        roll = math.atan2(float(rotation[1, 0]), float(rotation[1, 1]))
    else:
        yaw = math.atan2(float(-rotation[2, 0]), float(rotation[0, 0]))
        roll = 0.0
    return tuple(math.degrees(value) for value in (yaw, pitch, roll))


def wrap180(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def yaw_difference(a: float, b: float) -> dict[str, float]:
    signed = wrap180(a - b)
    raw = abs(signed)
    sym = min(raw, abs(180.0 - raw))
    return {"signed": signed, "raw": raw, "sym180": sym}


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a @ b.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def add_error(
    pose: Optional[dict[str, Any]],
    reference: Optional[dict[str, Any]],
    dims: Optional[tuple[float, float, float]],
) -> Optional[float]:
    if pose is None or reference is None or dims is None:
        return None
    pose_dims = tuple(float(value) for value in pose.get("dims", dims))
    reference_dims = tuple(float(value) for value in reference.get("dims", dims))
    if not np.allclose(pose_dims, reference_dims, rtol=0.0, atol=1e-12):
        return None
    points = APNP.make_pallet_keypoints_3d(*pose_dims)
    first = (np.asarray(pose["R"]) @ points.T).T + np.asarray(pose["t"])
    second = (np.asarray(reference["R"]) @ points.T).T + np.asarray(reference["t"])
    return float(np.mean(np.linalg.norm(first - second, axis=1)))


def adds_error(
    pose: Optional[dict[str, Any]],
    reference: Optional[dict[str, Any]],
    dims: Optional[tuple[float, float, float]],
) -> Optional[float]:
    """180-degree pallet-symmetry-aware ADD, null across W/D hypotheses."""
    if pose is None or reference is None or dims is None:
        return None
    pose_dims = tuple(float(value) for value in pose.get("dims", dims))
    reference_dims = tuple(float(value) for value in reference.get("dims", dims))
    if not np.allclose(pose_dims, reference_dims, rtol=0.0, atol=1e-12):
        return None
    points = APNP.make_pallet_keypoints_3d(*pose_dims)
    first = (np.asarray(pose["R"]) @ points.T).T + np.asarray(pose["t"])
    second = (np.asarray(reference["R"]) @ points.T).T + np.asarray(reference["t"])
    direct = float(np.mean(np.linalg.norm(first - second, axis=1)))
    permutation_180 = np.asarray((5, 4, 7, 6, 1, 0, 3, 2, 8), dtype=np.int64)
    symmetric = float(
        np.mean(np.linalg.norm(first - second[permutation_180], axis=1))
    )
    return min(direct, symmetric)


def add_validity(
    pose: Optional[dict[str, Any]],
    reference: Optional[dict[str, Any]],
    dims: Optional[tuple[float, float, float]],
) -> tuple[Optional[bool], Optional[str]]:
    if pose is None or reference is None or dims is None:
        return None, "missing_pose_reference_or_dimensions"
    pose_dims = tuple(float(value) for value in pose.get("dims", dims))
    reference_dims = tuple(float(value) for value in reference.get("dims", dims))
    if not np.allclose(pose_dims, reference_dims, rtol=0.0, atol=1e-12):
        return False, "selected_WD_hypothesis_mismatch"
    return True, None


def fixed_observation_reprojection(
    pose: Optional[dict[str, Any]],
    observations: list[Optional[list[float]]],
    intrinsics: Optional[np.ndarray],
    fallback_dims: Optional[tuple[float, float, float]],
) -> tuple[Optional[float], int]:
    """Evaluate any pose on the same fixed GT 2D correspondences.

    This is comparable across LOO/replacement/perturbation variants, unlike
    each solver's own-input reprojection score.
    """
    if pose is None or intrinsics is None or fallback_dims is None:
        return None, 0
    try:
        selected_dims = tuple(pose.get("dims", fallback_dims))
        object_points = APNP.make_pallet_keypoints_3d(*selected_dims)
        projected = APNP.project_3d(
            object_points,
            np.asarray(pose["R"], dtype=np.float64),
            np.asarray(pose["t"], dtype=np.float64),
            intrinsics,
        )
    except Exception:
        return None, 0
    valid_indices = [
        index
        for index, observation in enumerate(observations[:N_KEYPOINTS])
        if point_valid(observation)
    ]
    errors = []
    for index in valid_indices:
        observation = observations[index]
        if not point_valid(projected[index]):
            return None, len(valid_indices)
        errors.append(euclidean(projected[index], observation))
    errors = [value for value in errors if value is not None]
    return (float(np.mean(errors)), len(valid_indices)) if errors else (None, 0)


def comparative_pose_changes(
    pose: Optional[dict[str, Any]],
    baseline: Optional[dict[str, Any]],
    oracle: Optional[dict[str, Any]],
    observations: list[Optional[list[float]]],
    intrinsics: Optional[np.ndarray],
    dims: Optional[tuple[float, float, float]],
) -> dict[str, Any]:
    """Separate pose displacement from signed change in absolute error."""
    candidate_fixed, candidate_fixed_n = fixed_observation_reprojection(
        pose, observations, intrinsics, dims
    )
    baseline_fixed, baseline_fixed_n = fixed_observation_reprojection(
        baseline, observations, intrinsics, dims
    )
    result: dict[str, Any] = {
        "candidate_pose_success": pose is not None,
        "baseline_pose_success": baseline is not None,
        "pose_success_delta": int(pose is not None) - int(baseline is not None),
        "pose_success_transition": (
            f"{'success' if baseline is not None else 'failure'}"
            f"->{'success' if pose is not None else 'failure'}"
        ),
        "gt_fixed_reproj_error_px": candidate_fixed,
        "gt_fixed_reproj_n": candidate_fixed_n,
        "baseline_gt_fixed_reproj_error_px": baseline_fixed,
        "baseline_gt_fixed_reproj_n": baseline_fixed_n,
        "candidate_error_minus_baseline_gt_fixed_reproj_px": (
            None
            if candidate_fixed is None or baseline_fixed is None
            else candidate_fixed - baseline_fixed
        ),
        "solver_own_input_reproj_difference_px_diagnostic_only": None,
        "pose_displacement_yaw_signed_deg": None,
        "pose_displacement_yaw_raw_deg": None,
        "pose_displacement_yaw_sym180_deg": None,
        "pose_displacement_translation_m": None,
        "pose_displacement_add_m": None,
        "candidate_error_minus_baseline_yaw_sym180_deg": None,
        "candidate_error_minus_baseline_add_vs_oracle_m": None,
        "candidate_error_minus_baseline_adds180_vs_oracle_m": None,
        "baseline_wd_hypothesis": (
            None if baseline is None else baseline.get("_wd_hypothesis")
        ),
        "candidate_wd_hypothesis": (
            None if pose is None else pose.get("_wd_hypothesis")
        ),
        "wd_hypothesis_changed": None,
        "wd_hypothesis_transition": None,
    }
    if pose is None or baseline is None:
        return result
    own_candidate = finite_float(pose.get("reproj_error_px"))
    own_baseline = finite_float(baseline.get("reproj_error_px"))
    if own_candidate is not None and own_baseline is not None:
        result["solver_own_input_reproj_difference_px_diagnostic_only"] = (
            own_candidate - own_baseline
        )
    displacement = yaw_difference(yaw_deg(pose["R"]), yaw_deg(baseline["R"]))
    baseline_hypothesis = baseline.get("_wd_hypothesis")
    candidate_hypothesis = pose.get("_wd_hypothesis")
    result.update(
        {
            "pose_displacement_yaw_signed_deg": displacement["signed"],
            "pose_displacement_yaw_raw_deg": displacement["raw"],
            "pose_displacement_yaw_sym180_deg": displacement["sym180"],
            "pose_displacement_translation_m": float(
                np.linalg.norm(np.asarray(pose["t"]) - np.asarray(baseline["t"]))
            ),
            "pose_displacement_add_m": add_error(pose, baseline, dims),
            "wd_hypothesis_changed": candidate_hypothesis != baseline_hypothesis,
            "wd_hypothesis_transition": (
                f"{baseline_hypothesis}->{candidate_hypothesis}"
            ),
        }
    )
    if oracle is not None:
        candidate_yaw_error = yaw_difference(
            yaw_deg(pose["R"]), yaw_deg(oracle["R"])
        )["sym180"]
        baseline_yaw_error = yaw_difference(
            yaw_deg(baseline["R"]), yaw_deg(oracle["R"])
        )["sym180"]
        candidate_add = add_error(pose, oracle, dims)
        baseline_add = add_error(baseline, oracle, dims)
        candidate_adds = adds_error(pose, oracle, dims)
        baseline_adds = adds_error(baseline, oracle, dims)
        result["candidate_error_minus_baseline_yaw_sym180_deg"] = (
            candidate_yaw_error - baseline_yaw_error
        )
        result["candidate_error_minus_baseline_add_vs_oracle_m"] = (
            None
            if candidate_add is None or baseline_add is None
            else candidate_add - baseline_add
        )
        result["candidate_error_minus_baseline_adds180_vs_oracle_m"] = (
            None
            if candidate_adds is None or baseline_adds is None
            else candidate_adds - baseline_adds
        )
    return result


def pose_fields(
    pose: Optional[dict[str, Any]],
    oracle: Optional[dict[str, Any]],
    stored: Optional[dict[str, Any]],
    dims: Optional[tuple[float, float, float]],
) -> dict[str, Any]:
    if pose is None:
        return {
            "pose_success": False,
            "yaw_primary_metric": "yaw_error_sym180_deg",
            "yaw_deg": None,
            "pitch_deg": None,
            "roll_deg": None,
            "reproj_error_px": None,
            "auto_swap_dims": None,
            "selected_dim_W_m": None,
            "selected_dim_D_m": None,
            "selected_dim_H_m": None,
            "wd_hypothesis": None,
            "t_x_m": None,
            "t_y_m": None,
            "t_z_m": None,
            "yaw_error_vs_oracle_raw_deg": None,
            "yaw_error_vs_oracle_sym180_deg": None,
            "pitch_error_vs_oracle_deg": None,
            "roll_error_vs_oracle_deg": None,
            "rotation_error_vs_oracle_deg": None,
            "translation_error_vs_oracle_m": None,
            "add_vs_oracle_m": None,
            "adds180_vs_oracle_m": None,
            "add_vs_oracle_valid": None,
            "add_vs_oracle_invalid_reason": "missing_pose",
            "yaw_error_vs_stored_raw_deg": None,
            "yaw_error_vs_stored_sym180_deg": None,
            "pitch_error_vs_stored_deg": None,
            "roll_error_vs_stored_deg": None,
            "rotation_error_vs_stored_deg": None,
            "translation_error_vs_stored_m": None,
            "add_vs_stored_m": None,
            "adds180_vs_stored_m": None,
            "add_vs_stored_valid": None,
            "add_vs_stored_invalid_reason": "missing_pose",
        }
    rotation = np.asarray(pose["R"], dtype=np.float64)
    translation = np.asarray(pose["t"], dtype=np.float64).reshape(3)
    pose_yaw, pose_pitch, pose_roll = orientation_yaw_pitch_roll(rotation)
    result: dict[str, Any] = {
        "pose_success": True,
        "yaw_primary_metric": "yaw_error_sym180_deg",
        "yaw_deg": pose_yaw,
        "pitch_deg": pose_pitch,
        "roll_deg": pose_roll,
        "reproj_error_px": finite_float(pose.get("reproj_error_px")),
        "auto_swap_dims": pose.get("_auto_swap_dims"),
        "selected_dim_W_m": (
            None if dims is None else finite_float(pose.get("dims", dims)[0])
        ),
        "selected_dim_D_m": (
            None if dims is None else finite_float(pose.get("dims", dims)[1])
        ),
        "selected_dim_H_m": (
            None if dims is None else finite_float(pose.get("dims", dims)[2])
        ),
        "wd_hypothesis": pose.get("_wd_hypothesis", "as_given_locked"),
        "t_x_m": float(translation[0]),
        "t_y_m": float(translation[1]),
        "t_z_m": float(translation[2]),
    }
    for label, reference in (("oracle", oracle), ("stored", stored)):
        if reference is None:
            result.update(
                {
                    f"yaw_error_vs_{label}_raw_deg": None,
                    f"yaw_error_vs_{label}_sym180_deg": None,
                    f"pitch_error_vs_{label}_deg": None,
                    f"roll_error_vs_{label}_deg": None,
                    f"rotation_error_vs_{label}_deg": None,
                    f"translation_error_vs_{label}_m": None,
                    f"add_vs_{label}_m": None,
                    f"adds180_vs_{label}_m": None,
                    f"add_vs_{label}_valid": None,
                    f"add_vs_{label}_invalid_reason": "missing_reference",
                }
            )
            continue
        reference_rotation = np.asarray(reference["R"], dtype=np.float64)
        reference_translation = np.asarray(reference["t"], dtype=np.float64).reshape(3)
        ref_yaw, ref_pitch, ref_roll = orientation_yaw_pitch_roll(reference_rotation)
        difference = yaw_difference(pose_yaw, ref_yaw)
        add_is_valid, add_invalid_reason = add_validity(pose, reference, dims)
        result.update(
            {
                f"yaw_error_vs_{label}_raw_deg": difference["raw"],
                f"yaw_error_vs_{label}_sym180_deg": difference["sym180"],
                f"pitch_error_vs_{label}_deg": abs(
                    wrap180(pose_pitch - ref_pitch)
                ),
                f"roll_error_vs_{label}_deg": abs(wrap180(pose_roll - ref_roll)),
                f"rotation_error_vs_{label}_deg": rotation_error_deg(
                    rotation, reference_rotation
                ),
                f"translation_error_vs_{label}_m": float(
                    np.linalg.norm(translation - reference_translation)
                ),
                f"add_vs_{label}_m": add_error(pose, reference, dims),
                f"adds180_vs_{label}_m": adds_error(pose, reference, dims),
                f"add_vs_{label}_valid": add_is_valid,
                f"add_vs_{label}_invalid_reason": add_invalid_reason,
            }
        )
    return result


def current_solve(
    points: list[Optional[list[float]]],
    intrinsics: Optional[np.ndarray],
    dims: Optional[tuple[float, float, float]],
    image_shape: tuple[int, ...],
    auto_swap_dims: bool,
) -> tuple[Optional[dict[str, Any]], float, Optional[str]]:
    if intrinsics is None:
        return None, 0.0, "missing_intrinsics"
    if dims is None:
        return None, 0.0, "missing_dimensions"
    valid = sum(point_valid(point) for point in points)
    if valid < 4:
        return None, 0.0, "fewer_than_4_correspondences"
    started = time.perf_counter()
    try:
        pose = APNP.solve_pose(
            points,
            intrinsics,
            dims=dims,
            img_shape=image_shape,
            auto_swap_dims=auto_swap_dims,
        )
        if pose is not None:
            pose = dict(pose)
            pose["_auto_swap_dims"] = bool(auto_swap_dims)
            pose.setdefault(
                "_wd_hypothesis",
                "canonical_selected" if auto_swap_dims else "as_given_locked",
            )
        elapsed = 1000.0 * (time.perf_counter() - started)
        return pose, elapsed, None if pose is not None else "solver_returned_none"
    except Exception as exc:  # diagnostic output must preserve the failure
        elapsed = 1000.0 * (time.perf_counter() - started)
        return None, elapsed, clean_error(exc)


class CurrentSolveCache:
    def __init__(
        self,
        intrinsics: Optional[np.ndarray],
        dims: Optional[tuple[float, float, float]],
        image_shape: tuple[int, ...],
        auto_swap_dims: bool = True,
    ) -> None:
        self.intrinsics = intrinsics
        self.dims = dims
        self.image_shape = image_shape
        self.auto_swap_dims = bool(auto_swap_dims)
        self.cache: dict[tuple[Any, ...], tuple[Optional[dict[str, Any]], float, Optional[str]]] = {}

    @staticmethod
    def key(points: Iterable[Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for point in points:
            xy = point_xy(point)
            values.append(None if xy is None else (round(xy[0], 9), round(xy[1], 9)))
        return tuple(values)

    def solve(
        self, points: list[Optional[list[float]]]
    ) -> tuple[Optional[dict[str, Any]], float, Optional[str], bool]:
        key = self.key(points)
        if key in self.cache:
            pose, elapsed, error = self.cache[key]
            return pose, elapsed, error, True
        value = current_solve(
            points,
            self.intrinsics,
            self.dims,
            self.image_shape,
            self.auto_swap_dims,
        )
        self.cache[key] = value
        return value[0], value[1], value[2], False


def direct_solve(
    points: list[Optional[list[float]]],
    intrinsics: Optional[np.ndarray],
    dims: Optional[tuple[float, float, float]],
    solver: str,
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {
        "solver_available": True,
        "solver_error": None,
        "solver_runtime_ms": 0.0,
        "ransac_inlier_count": None,
        "positive_depth_fraction": None,
        "cheirality_pass": None,
    }
    if intrinsics is None:
        info["solver_error"] = "missing_intrinsics"
        return None, info
    if dims is None:
        info["solver_error"] = "missing_dimensions"
        return None, info
    object_all = APNP.make_pallet_keypoints_3d(*dims)
    valid_indices = [i for i, point in enumerate(points) if point_valid(point)]
    if len(valid_indices) < 4:
        info["solver_error"] = "fewer_than_4_correspondences"
        return None, info
    obj = np.ascontiguousarray(object_all[valid_indices], dtype=np.float64)
    img = np.ascontiguousarray(
        np.asarray([point_xy(points[i]) for i in valid_indices]), dtype=np.float64
    )
    started = time.perf_counter()
    try:
        inliers = None
        if solver == "EPnP":
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, intrinsics, None, flags=cv2.SOLVEPNP_EPNP
            )
        elif solver == "EPnP+RANSAC":
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                img,
                intrinsics,
                None,
                flags=cv2.SOLVEPNP_EPNP,
                iterationsCount=100,
                reprojectionError=8.0,
                confidence=0.99,
            )
        elif solver in ("SQPNP", "SQPNP+RefineLM"):
            if not hasattr(cv2, "SOLVEPNP_SQPNP"):
                info["solver_available"] = False
                info["solver_error"] = "opencv_has_no_sqpnp"
                return None, info
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, intrinsics, None, flags=cv2.SOLVEPNP_SQPNP
            )
            if ok and solver.endswith("+RefineLM"):
                if not hasattr(cv2, "solvePnPRefineLM"):
                    info["solver_available"] = False
                    info["solver_error"] = "opencv_has_no_solvepnprefinelm"
                    return None, info
                refined = cv2.solvePnPRefineLM(
                    obj,
                    img,
                    intrinsics,
                    None,
                    np.asarray(rvec, dtype=np.float64),
                    np.asarray(tvec, dtype=np.float64),
                )
                if refined is not None:
                    rvec, tvec = refined
        elif solver == "ITERATIVE":
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, intrinsics, None, flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            raise ValueError(f"unknown solver: {solver}")
        if not ok:
            info["solver_error"] = "solver_returned_false"
            return None, info
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        projected, _ = cv2.projectPoints(
            obj, rvec, translation.reshape(3, 1), intrinsics, None
        )
        reprojection = float(
            np.mean(np.linalg.norm(projected.reshape(-1, 2) - img, axis=1))
        )
        camera_points = (rotation @ obj.T).T + translation
        positive_depth_fraction = float(np.mean(camera_points[:, 2] > 0.0))
        info["positive_depth_fraction"] = positive_depth_fraction
        info["cheirality_pass"] = bool(
            translation[2] > 0.0 and positive_depth_fraction == 1.0
        )
        if not info["cheirality_pass"]:
            info["solver_error"] = "negative_depth_or_cheirality_failure"
            return None, info
        pose = {
            "R": rotation,
            "t": translation,
            "rvec": np.asarray(rvec).reshape(3),
            "reproj_error_px": reprojection,
            "positive_depth_fraction": positive_depth_fraction,
            "dims": dims,
            "_wd_hypothesis": "as_given_direct",
            "_auto_swap_dims": False,
        }
        info["ransac_inlier_count"] = None if inliers is None else int(len(inliers))
        return pose, info
    except Exception as exc:
        info["solver_error"] = clean_error(exc)
        return None, info
    finally:
        info["solver_runtime_ms"] = 1000.0 * (time.perf_counter() - started)


def base_metadata(
    spec: dict[str, Any],
    data: dict[str, Any],
    obj: dict[str, Any],
    image: np.ndarray,
    intrinsics: Optional[np.ndarray],
    dims: Optional[tuple[float, float, float]],
    gt_points: list[Optional[list[float]]],
    stored: Optional[dict[str, Any]],
) -> dict[str, Any]:
    height, width = image.shape[:2]
    finite_corners = [point for point in gt_points[:8] if point_valid(point)]
    in_frame = [
        bool(point_inside(point, width, height)) for point in finite_corners
    ]
    if finite_corners:
        array = np.asarray(finite_corners)
        bbox = (
            float(np.min(array[:, 0])),
            float(np.min(array[:, 1])),
            float(np.max(array[:, 0]) - np.min(array[:, 0])),
            float(np.max(array[:, 1]) - np.min(array[:, 1])),
        )
    else:
        bbox = (None, None, None, None)
    elevation = None
    if obj.get("pose_transform") is not None:
        try:
            elevation = finite_float(elev_from_pose(obj["pose_transform"]))
        except Exception:
            elevation = None
    stored_yaw = yaw_deg(stored["R"]) if stored is not None else None
    distance = (
        float(np.linalg.norm(np.asarray(stored["t"]))) if stored is not None else None
    )
    view_azimuth = (
        None
        if stored is None
        else float(
            math.degrees(
                math.atan2(
                    float(np.asarray(stored["t"])[0]),
                    float(np.asarray(stored["t"])[2]),
                )
            )
        )
    )
    camera = data.get("camera_data")
    camera_width = camera.get("width") if isinstance(camera, dict) else None
    camera_height = camera.get("height") if isinstance(camera, dict) else None
    entry = spec.get("entry")
    correspondence_valid = spec["dataset"] != "synthetic_q1_val"
    return {
        "frame_uid": f"{spec['dataset']}:{spec['domain']}:{spec['fid']}",
        "dataset": spec["dataset"],
        "domain": spec["domain"],
        "fid": spec["fid"],
        "split_role": spec["split_role"],
        "is_primary": spec["is_primary"],
        "legacy_aggregate": spec["legacy_aggregate"],
        "source_session": spec.get("source_session"),
        "json_path": spec["json"],
        "image_path": spec["png"],
        "image_width": int(width),
        "image_height": int(height),
        "metadata_camera_width": finite_float(camera_width),
        "metadata_camera_height": finite_float(camera_height),
        "K_fx": None if intrinsics is None else float(intrinsics[0, 0]),
        "K_fy": None if intrinsics is None else float(intrinsics[1, 1]),
        "K_cx": None if intrinsics is None else float(intrinsics[0, 2]),
        "K_cy": None if intrinsics is None else float(intrinsics[1, 2]),
        "dim_W_m": None if dims is None else dims[0],
        "dim_D_m": None if dims is None else dims[1],
        "dim_H_m": None if dims is None else dims[2],
        "gt_visibility": finite_float(obj.get("visibility")),
        "gt_source": obj.get("gt_source"),
        "lighting_metadata": None,
        "background_metadata": None,
        "occlusion_metadata": None,
        "gt_annotation_reproj_error_px": finite_float(obj.get("reproj_error_px")),
        "gt_v_geom": int(sum(in_frame)),
        "gt_truncated": bool(sum(in_frame) < 8),
        "gt_bbox_x": bbox[0],
        "gt_bbox_y": bbox[1],
        "gt_bbox_w": bbox[2],
        "gt_bbox_h": bbox[3],
        "gt_pose_available": stored is not None,
        "gt_yaw_deg": stored_yaw,
        "gt_view_azimuth_deg": view_azimuth,
        "gt_elevation_deg": elevation,
        "gt_distance_m": distance,
        "q1_V": entry.get("V") if isinstance(entry, dict) else None,
        "q1_V8": entry.get("V8") if isinstance(entry, dict) else None,
        "q1_best_sym": entry.get("best_sym") if isinstance(entry, dict) else None,
        "keypoint_channel_correspondence_valid": correspondence_valid,
        "kp_identity_pose_conclusion_valid": correspondence_valid,
        "analysis_validity": (
            "fixed_correspondence_geometry_valid"
            if correspondence_valid
            else "order_free_corner_aggregate_and_heatmap_distribution_only"
        ),
        "filter_pass": None,
    }


def compact_metadata(base: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "frame_uid",
        "dataset",
        "domain",
        "fid",
        "split_role",
        "is_primary",
        "legacy_aggregate",
        "source_session",
        "image_width",
        "image_height",
        "K_fx",
        "K_fy",
        "K_cx",
        "K_cy",
        "dim_W_m",
        "dim_D_m",
        "dim_H_m",
        "gt_visibility",
        "gt_source",
        "gt_annotation_reproj_error_px",
        "gt_v_geom",
        "gt_truncated",
        "gt_bbox_x",
        "gt_bbox_y",
        "gt_bbox_w",
        "gt_bbox_h",
        "gt_elevation_deg",
        "gt_distance_m",
        "gt_view_azimuth_deg",
        "lighting_metadata",
        "background_metadata",
        "occlusion_metadata",
        "keypoint_channel_correspondence_valid",
        "kp_identity_pose_conclusion_valid",
        "analysis_validity",
        "filter_pass",
    )
    return {key: base.get(key) for key in keys}


def pnp_points(
    stats: list[dict[str, Any]], kind: str
) -> list[Optional[list[float]]]:
    key = "_arg_px" if kind == "argmax" else "_soft_px"
    return [
        point_xy(item.get(key)) if item.get("detected") else None for item in stats
    ]


def edge_length(points: list[Optional[list[float]]], first: int, second: int) -> Optional[float]:
    return euclidean(points[first], points[second])


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def elevation_bin(value: Any) -> Optional[str]:
    number = finite_float(value)
    if number is None:
        return None
    if number < 3.0:
        return "lt3"
    if number < 8.0:
        return "3_to_8"
    if number < 15.0:
        return "8_to_15"
    return "ge15"


def bbox_area_bin(base: dict[str, Any]) -> Optional[str]:
    width = finite_float(base.get("gt_bbox_w"))
    height = finite_float(base.get("gt_bbox_h"))
    image_width = finite_float(base.get("image_width"))
    image_height = finite_float(base.get("image_height"))
    if None in (width, height, image_width, image_height) or image_width * image_height <= 0:
        return None
    fraction = width * height / (image_width * image_height)
    if fraction < 0.02:
        return "lt0.02"
    if fraction < 0.05:
        return "0.02_to_0.05"
    if fraction < 0.10:
        return "0.05_to_0.10"
    return "ge0.10"


def visible_bin(value: Any) -> Optional[str]:
    number = finite_float(value)
    if number is None:
        return None
    count = int(number)
    return "8" if count >= 8 else ("7" if count == 7 else "le6")


def process_frame(
    model: DopeNetwork,
    device: torch.device,
    spec: dict[str, Any],
    audit: InputAudit,
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    data = audit.read_json(spec["json"])
    image = audit.read_image(spec["png"])
    if image is None:
        raise FileNotFoundError(f"cv2.imread failed: {spec['png']}")
    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("frame JSON has no object")
    obj = objects[0]
    intrinsics = intrinsics_from_json(data)
    dims = dims_from_frame(spec, obj)
    gt_points = gt_points_from_object(obj)
    stored = stored_pose_from_object(obj)
    base = base_metadata(
        spec, data, obj, image, intrinsics, dims, gt_points, stored
    )
    compact = compact_metadata(base)
    height, width = image.shape[:2]

    belief, inference_ms = infer_belief(model, image, device)
    flipped_image = cv2.flip(image, 1)
    flipped_belief, flip_inference_ms = infer_belief(model, flipped_image, device)
    scale_x = width / float(belief.shape[2])
    scale_y = height / float(belief.shape[1])
    correspondence_valid = spec["dataset"] != "synthetic_q1_val"
    stats = [
        heatmap_stats(
            belief[index],
            scale_x,
            scale_y,
            gt_points[index] if correspondence_valid or index == 8 else None,
        )
        for index in range(N_KEYPOINTS)
    ]

    for index, item in enumerate(stats):
        row = dict(compact)
        gt = point_xy(gt_points[index])
        row.update(
            {
                "keypoint": index,
                "is_kp5": index == 5,
                "is_centroid": index == 8,
                "gt_x": None if gt is None else gt[0],
                "gt_y": None if gt is None else gt[1],
                "gt_finite": gt is not None,
                "gt_in_frame": point_inside(gt, width, height),
                "gt_occluded": None,
                "gt_correspondence_valid": correspondence_valid or index == 8,
                "kp_error_valid_for_conclusion": correspondence_valid or index == 8,
                "belief_threshold": BELIEF_THRESHOLD,
            }
        )
        row.update({key: value for key, value in item.items() if not key.startswith("_")})
        tables["keypoints"].append(row)

    arg_points = pnp_points(stats, "argmax")
    soft_points = pnp_points(stats, "softargmax")
    # The canonical historical evaluator lets APNP search the supplied W/D and
    # swapped D/W hypotheses. A second cache keeps dimensions locked as a
    # diagnostic control; it never replaces the canonical ladder result.
    cache = CurrentSolveCache(intrinsics, dims, image.shape, auto_swap_dims=True)
    locked_cache = CurrentSolveCache(
        intrinsics, dims, image.shape, auto_swap_dims=False
    )
    run_pose_diagnostics = correspondence_valid
    if run_pose_diagnostics:
        oracle, oracle_ms, oracle_error, oracle_cached = cache.solve(gt_points)
        locked_oracle, locked_oracle_ms, locked_oracle_error, _ = (
            locked_cache.solve(gt_points)
        )
    else:
        oracle, oracle_ms, oracle_error, oracle_cached = (
            None,
            0.0,
            "not_applicable_synthetic_channel_order_ambiguous",
            False,
        )
        locked_oracle, locked_oracle_ms, locked_oracle_error = (
            None,
            0.0,
            "not_applicable_synthetic_channel_order_ambiguous",
        )

    mean8_centroid = None
    if all(point_valid(point) for point in soft_points[:8]):
        mean8_centroid = np.mean(np.asarray(soft_points[:8]), axis=0).tolist()
    corners_only = list(soft_points)
    corners_only[8] = None
    geometry_pose = None
    geometry_pose_error = "not_applicable"
    if run_pose_diagnostics:
        geometry_pose, _, geometry_pose_error, _ = cache.solve(corners_only)
    geometry_centroid = None
    if geometry_pose is not None and intrinsics is not None:
        selected_dims = tuple(geometry_pose.get("dims", dims))
        projected = APNP.project_3d(
            APNP.make_pallet_keypoints_3d(*selected_dims),
            np.asarray(geometry_pose["R"]),
            np.asarray(geometry_pose["t"]),
            intrinsics,
        )
        geometry_centroid = point_xy(projected[8])
    y3 = list(soft_points)
    y3[8] = point_xy(gt_points[8])
    y4 = list(soft_points)
    y4[8] = point_xy(geometry_centroid)
    y5 = list(soft_points)
    y5[5] = point_xy(gt_points[5])
    y6 = list(soft_points)
    y6[5] = None
    y7 = list(corners_only)
    y8 = [
        point if point_inside(gt_points[index], width, height) else None
        for index, point in enumerate(soft_points)
    ]
    ladder = (
        ("Y0", "GT 2D keypoints -> canonical current PnP", gt_points, "current"),
        (
            "Y1",
            "raw argmax detections -> canonical current PnP",
            arg_points,
            "current",
        ),
        (
            "Y2",
            "local softargmax detections -> canonical current PnP",
            soft_points,
            "current",
        ),
        ("Y3", "softargmax + GT centroid -> canonical current PnP", y3, "current"),
        (
            "Y4",
            "softargmax + centroid projected from corners-only PnP -> canonical current PnP",
            y4,
            "current",
        ),
        ("Y5", "softargmax + GT kp5 -> canonical current PnP", y5, "current"),
        ("Y6", "softargmax excluding kp5 -> canonical current PnP", y6, "current"),
        (
            "Y7",
            "softargmax excluding centroid -> canonical current PnP",
            y7,
            "current",
        ),
        (
            "Y8",
            "softargmax masked to GT in-frame keypoints -> canonical current PnP",
            y8,
            "current",
        ),
        (
            "Y9",
            "softargmax detections -> SQPNP+RefineLM",
            soft_points,
            "SQPNP+RefineLM",
        ),
    )
    if not run_pose_diagnostics:
        ladder = ()
    ladder_pose: dict[str, Optional[dict[str, Any]]] = {}
    for stage, description, points, solver in ladder:
        if solver == "current":
            pose, runtime_ms, error, cache_hit = cache.solve(list(points))
            solver_info = {
                "solver_runtime_ms": runtime_ms,
                "solver_error": error,
                "solver_available": True,
                "solver_cache_hit": cache_hit,
                "ransac_inlier_count": None,
            }
        else:
            pose, solver_info = direct_solve(
                list(points), intrinsics, dims, solver
            )
            solver_info["solver_cache_hit"] = False
        ladder_pose[stage] = pose
        row = dict(compact)
        row.update(
            {
                "stage": stage,
                "description": description,
                "solver": (
                    "current_canonical_autoswap" if solver == "current" else solver
                ),
                "n_input_points": int(sum(point_valid(point) for point in points)),
                "geometry_centroid_method": (
                    "solve thresholded softargmax corners0-7 without centroid, then "
                    "project 3D keypoint8/origin from that pose"
                    if stage == "Y4"
                    else None
                ),
                "geometry_seed_pose_success": (
                    geometry_pose is not None if stage == "Y4" else None
                ),
                "geometry_seed_pose_error": (
                    geometry_pose_error if stage == "Y4" else None
                ),
                "mask_basis": (
                    "GT coordinate in image bounds only; not visibility or occlusion"
                    if stage == "Y8"
                    else None
                ),
            }
        )
        row.update(solver_info)
        row.update(pose_fields(pose, oracle, stored, dims))
        fixed_reproj, fixed_reproj_n = fixed_observation_reprojection(
            pose, gt_points, intrinsics, dims
        )
        row["gt_fixed_reproj_error_px"] = fixed_reproj
        row["gt_fixed_reproj_n"] = fixed_reproj_n
        tables["yaw_ladder"].append(row)

    baseline = ladder_pose.get("Y2")
    for index in (range(N_KEYPOINTS) if run_pose_diagnostics else ()):
        for variant in ("leave_one_out", "replace_one_with_gt"):
            points = list(soft_points)
            if variant == "leave_one_out":
                points[index] = None
                gt_replacement_available = None
            else:
                replacement = point_xy(gt_points[index])
                points[index] = replacement
                gt_replacement_available = replacement is not None
            pose, runtime_ms, error, cache_hit = cache.solve(points)
            row = dict(compact)
            row.update(
                {
                    "keypoint": index,
                    "is_kp5": index == 5,
                    "is_centroid": index == 8,
                    "variant": variant,
                    "gt_replacement_available": gt_replacement_available,
                    "n_input_points": int(sum(point_valid(point) for point in points)),
                    "solver": "current_canonical_autoswap",
                    "solver_runtime_ms": runtime_ms,
                    "solver_error": error,
                    "solver_cache_hit": cache_hit,
                }
            )
            row.update(pose_fields(pose, oracle, stored, dims))
            row.update(
                comparative_pose_changes(
                    pose, baseline, oracle, gt_points, intrinsics, dims
                )
            )
            tables["keypoint_influence"].append(row)

    # A local, output-grid perturbation test isolates kp5 sensitivity without
    # retraining or interpreting the model heatmap as a covariance-weighted PnP.
    if spec["split_role"] == "strict_filterval":
        perturbations = tuple(
            (axis, delta if axis == "x" else 0.0, delta if axis == "y" else 0.0)
            for axis in ("x", "y")
            for magnitude in (1.0, 2.0, 4.0, 8.0)
            for delta in (-magnitude, magnitude)
        )
        for axis, dx_grid, dy_grid in perturbations:
            points = list(soft_points)
            source_kp5 = point_xy(points[5])
            if source_kp5 is not None:
                points[5] = [
                    source_kp5[0] + dx_grid * scale_x,
                    source_kp5[1] + dy_grid * scale_y,
                ]
            pose, runtime_ms, error, cache_hit = cache.solve(points)
            row = dict(compact)
            row.update(
                {
                    "keypoint": 5,
                    "axis": axis,
                    "delta_grid_x": dx_grid,
                    "delta_grid_y": dy_grid,
                    "delta_pixel_x": dx_grid * scale_x,
                    "delta_pixel_y": dy_grid * scale_y,
                    "kp5_prediction_available": source_kp5 is not None,
                    "n_input_points": int(sum(point_valid(point) for point in points)),
                    "solver": "current_canonical_autoswap",
                    "solver_runtime_ms": runtime_ms,
                    "solver_error": error,
                    "solver_cache_hit": cache_hit,
                }
            )
            row.update(pose_fields(pose, oracle, stored, dims))
            row.update(
                comparative_pose_changes(
                    pose, baseline, oracle, gt_points, intrinsics, dims
                )
            )
            tables["kp5_perturbation"].append(row)

        # D3 dense sensitivity grid. A direct fixed-dimension solver keeps the
        # 9 keypoints x 16 perturbations computationally tractable. It has its
        # own same-solver baseline and is never mixed with the canonical kp5
        # curve above.
        perturb_baseline, perturb_baseline_info = direct_solve(
            soft_points, intrinsics, dims, "SQPNP+RefineLM"
        )
        for keypoint in range(N_KEYPOINTS):
            for axis, dx_grid, dy_grid in perturbations:
                points = list(soft_points)
                source = point_xy(points[keypoint])
                if source is not None:
                    points[keypoint] = [
                        source[0] + dx_grid * scale_x,
                        source[1] + dy_grid * scale_y,
                    ]
                pose, solver_info = direct_solve(
                    points, intrinsics, dims, "SQPNP+RefineLM"
                )
                row = dict(compact)
                row.update(
                    {
                        "keypoint": keypoint,
                        "is_kp5": keypoint == 5,
                        "is_centroid": keypoint == 8,
                        "axis": axis,
                        "delta_grid_x": dx_grid,
                        "delta_grid_y": dy_grid,
                        "delta_pixel_x": dx_grid * scale_x,
                        "delta_pixel_y": dy_grid * scale_y,
                        "prediction_available": source is not None,
                        "n_input_points": int(
                            sum(point_valid(point) for point in points)
                        ),
                        "solver": "SQPNP+RefineLM_FIXED_DIMS",
                        "baseline_solver": "SQPNP+RefineLM_FIXED_DIMS",
                        "baseline_solver_error": perturb_baseline_info.get(
                            "solver_error"
                        ),
                    }
                )
                row.update(solver_info)
                row.update(pose_fields(pose, locked_oracle, stored, dims))
                row.update(
                    comparative_pose_changes(
                        pose,
                        perturb_baseline,
                        locked_oracle,
                        gt_points,
                        intrinsics,
                        dims,
                    )
                )
                tables["keypoint_perturbation"].append(row)

        pred_edges = {
            "far_top_4_5": edge_length(soft_points, 4, 5),
            "vertical_5_6": edge_length(soft_points, 5, 6),
            "depth_1_5": edge_length(soft_points, 1, 5),
        }
        gt_edges = {
            "far_top_4_5": edge_length(gt_points, 4, 5),
            "vertical_5_6": edge_length(gt_points, 5, 6),
            "depth_1_5": edge_length(gt_points, 1, 5),
        }
        pred_far_perimeter = [
            edge_length(soft_points, first, second)
            for first, second in ((4, 5), (5, 6), (6, 7), (7, 4))
        ]
        gt_far_perimeter = [
            edge_length(gt_points, first, second)
            for first, second in ((4, 5), (5, 6), (6, 7), (7, 4))
        ]
        pred_perimeter = (
            float(sum(pred_far_perimeter))
            if all(value is not None for value in pred_far_perimeter)
            else None
        )
        gt_perimeter = (
            float(sum(gt_far_perimeter))
            if all(value is not None for value in gt_far_perimeter)
            else None
        )
        far_errors = [
            euclidean(soft_points[index], gt_points[index]) for index in (4, 5, 6, 7)
        ]
        kp5_pred, kp5_gt = point_xy(soft_points[5]), point_xy(gt_points[5])
        kp5_dx = (
            None if kp5_pred is None or kp5_gt is None else kp5_pred[0] - kp5_gt[0]
        )
        kp5_dy = (
            None if kp5_pred is None or kp5_gt is None else kp5_pred[1] - kp5_gt[1]
        )
        image_area = float(width * height)
        bbox_area_fraction = (
            None
            if base.get("gt_bbox_w") is None or base.get("gt_bbox_h") is None
            else float(base["gt_bbox_w"] * base["gt_bbox_h"] / image_area)
        )
        geometry_row = dict(compact)
        geometry_row.update(
            {
                "elevation_bin": elevation_bin(base.get("gt_elevation_deg")),
                "bbox_area_fraction": bbox_area_fraction,
                "bbox_area_bin": bbox_area_bin(base),
                "visible_bin": visible_bin(base.get("gt_v_geom")),
                "kp5_pred_x": None if kp5_pred is None else kp5_pred[0],
                "kp5_pred_y": None if kp5_pred is None else kp5_pred[1],
                "kp5_gt_x": None if kp5_gt is None else kp5_gt[0],
                "kp5_gt_y": None if kp5_gt is None else kp5_gt[1],
                "kp5_vector_dx_px": kp5_dx,
                "kp5_vector_dy_px": kp5_dy,
                "kp5_error_px": euclidean(kp5_pred, kp5_gt),
                "far_top_4_5_pred_px": pred_edges["far_top_4_5"],
                "far_top_4_5_gt_px": gt_edges["far_top_4_5"],
                "far_top_4_5_ratio": safe_ratio(
                    pred_edges["far_top_4_5"], gt_edges["far_top_4_5"]
                ),
                "vertical_5_6_pred_px": pred_edges["vertical_5_6"],
                "vertical_5_6_gt_px": gt_edges["vertical_5_6"],
                "vertical_5_6_ratio": safe_ratio(
                    pred_edges["vertical_5_6"], gt_edges["vertical_5_6"]
                ),
                "depth_1_5_pred_px": pred_edges["depth_1_5"],
                "depth_1_5_gt_px": gt_edges["depth_1_5"],
                "depth_1_5_ratio": safe_ratio(
                    pred_edges["depth_1_5"], gt_edges["depth_1_5"]
                ),
                "far_face_mean_corner_error_px": (
                    float(np.mean(far_errors))
                    if all(value is not None for value in far_errors)
                    else None
                ),
                "far_face_pred_perimeter_px": pred_perimeter,
                "far_face_gt_perimeter_px": gt_perimeter,
                "far_face_perimeter_ratio": safe_ratio(pred_perimeter, gt_perimeter),
            }
        )
        tables["kp5_geometry"].append(geometry_row)

    direct_inputs = (
        ("gt", gt_points),
        ("predicted_argmax", arg_points),
        ("predicted_softargmax", soft_points),
    )
    direct_solvers = (
        "EPnP",
        "EPnP+RANSAC",
        "SQPNP",
        "SQPNP+RefineLM",
        "ITERATIVE",
    )
    for input_kind, points in (direct_inputs if run_pose_diagnostics else ()):
        for solver in direct_solvers:
            pose, solver_info = direct_solve(points, intrinsics, dims, solver)
            row = dict(compact)
            row.update(
                {
                    "input_kind": input_kind,
                    "solver": solver,
                    "n_input_points": int(sum(point_valid(point) for point in points)),
                    "positive_depth_fraction": (
                        None if pose is None else pose.get("positive_depth_fraction")
                    ),
                    "oracle_reference": "Y0_locked_dims",
                    "oracle_wd_hypothesis": (
                        None
                        if locked_oracle is None
                        else locked_oracle.get("_wd_hypothesis")
                    ),
                }
            )
            row.update(solver_info)
            row.update(pose_fields(pose, locked_oracle, stored, dims))
            row["hypothesis_transition_vs_oracle"] = (
                None
                if pose is None or locked_oracle is None
                else (
                    f"{locked_oracle.get('_wd_hypothesis')}"
                    f"->{pose.get('_wd_hypothesis')}"
                )
            )
            fixed_reproj, fixed_reproj_n = fixed_observation_reprojection(
                pose, gt_points, intrinsics, dims
            )
            row["gt_fixed_reproj_error_px"] = fixed_reproj
            row["gt_fixed_reproj_n"] = fixed_reproj_n
            tables["solver_comparison"].append(row)
        for solver, solver_cache in (
            ("CURRENT_CANONICAL_AUTOSWAP", cache),
            ("CURRENT_LOCKED_DIMS", locked_cache),
        ):
            pose, runtime_ms, error, cache_hit = solver_cache.solve(list(points))
            comparison_oracle = (
                oracle if solver == "CURRENT_CANONICAL_AUTOSWAP" else locked_oracle
            )
            oracle_reference = (
                "Y0_canonical_autoswap"
                if solver == "CURRENT_CANONICAL_AUTOSWAP"
                else "Y0_locked_dims"
            )
            positive_depth_fraction = None
            cheirality_pass = None
            if pose is not None:
                selected_dims = tuple(pose.get("dims", dims))
                object_points = APNP.make_pallet_keypoints_3d(*selected_dims)
                camera_points = (
                    np.asarray(pose["R"]) @ object_points.T
                ).T + np.asarray(pose["t"])
                positive_depth_fraction = float(
                    np.mean(camera_points[:, 2] > 0.0)
                )
                cheirality_pass = bool(
                    np.asarray(pose["t"]).reshape(3)[2] > 0.0
                    and positive_depth_fraction == 1.0
                )
                if not cheirality_pass:
                    error = "negative_depth_or_cheirality_failure"
                    pose = None
            row = dict(compact)
            row.update(
                {
                    "input_kind": input_kind,
                    "solver": solver,
                    "n_input_points": int(sum(point_valid(point) for point in points)),
                    "solver_available": True,
                    "solver_error": error,
                    "solver_runtime_ms": runtime_ms,
                    "solver_cache_hit": cache_hit,
                    "ransac_inlier_count": None,
                    "positive_depth_fraction": positive_depth_fraction,
                    "cheirality_pass": cheirality_pass,
                    "oracle_reference": oracle_reference,
                    "oracle_wd_hypothesis": (
                        None
                        if comparison_oracle is None
                        else comparison_oracle.get("_wd_hypothesis")
                    ),
                }
            )
            row.update(pose_fields(pose, comparison_oracle, stored, dims))
            row["hypothesis_transition_vs_oracle"] = (
                None
                if pose is None or comparison_oracle is None
                else (
                    f"{comparison_oracle.get('_wd_hypothesis')}"
                    f"->{pose.get('_wd_hypothesis')}"
                )
            )
            fixed_reproj, fixed_reproj_n = fixed_observation_reprojection(
                pose, gt_points, intrinsics, dims
            )
            row["gt_fixed_reproj_error_px"] = fixed_reproj
            row["gt_fixed_reproj_n"] = fixed_reproj_n
            tables["solver_comparison"].append(row)

    flip_points: list[Optional[list[float]]] = []
    matched_coordinate_errors: list[float] = []
    all_argmax_coordinate_errors: list[float] = []
    matched_detected_count = 0
    original_only_detected_count = 0
    flip_only_detected_count = 0
    neither_detected_count = 0
    belief_rmses: list[float] = []
    belief_maes: list[float] = []
    for index in range(N_KEYPOINTS):
        partner = FLIP_PARTNER[index]
        flip_stats = heatmap_stats(
            flipped_belief[partner],
            scale_x,
            scale_y,
            gt_points[index] if correspondence_valid or index == 8 else None,
        )
        flip_soft = point_xy(flip_stats.get("_soft_px"))
        flip_arg = point_xy(flip_stats.get("_arg_px"))
        flip_moment = point_xy(flip_stats.get("_moment_px"))
        for point in (flip_soft, flip_arg, flip_moment):
            if point is not None:
                point[0] = (width - scale_x) - point[0]
        flip_points.append(
            flip_soft if flip_stats.get("detected") and flip_soft is not None else None
        )
        original_map = np.asarray(belief[index], dtype=np.float64)
        equivariant_map = np.asarray(flipped_belief[partner, :, ::-1], dtype=np.float64)
        difference = original_map - equivariant_map
        belief_rmse = float(np.sqrt(np.mean(difference * difference)))
        belief_mae = float(np.mean(np.abs(difference)))
        belief_rmses.append(belief_rmse)
        belief_maes.append(belief_mae)

        original_cov = stats[index].get("_cov_px")
        flipped_cov = flip_stats.get("_cov_px")
        transformed_cov = None
        covariance_frobenius = None
        if original_cov is not None and flipped_cov is not None:
            jacobian = np.diag([-1.0, 1.0])
            transformed_cov = jacobian @ np.asarray(flipped_cov) @ jacobian
            covariance_frobenius = float(
                np.linalg.norm(np.asarray(original_cov) - transformed_cov, ord="fro")
            )
        coordinate_error = euclidean(stats[index].get("_soft_px"), flip_soft)
        argmax_coordinate_error = euclidean(stats[index].get("_arg_px"), flip_arg)
        original_detected = bool(stats[index].get("detected"))
        flipped_detected = bool(flip_stats.get("detected"))
        matched_detected = original_detected and flipped_detected
        if argmax_coordinate_error is not None:
            all_argmax_coordinate_errors.append(argmax_coordinate_error)
        if matched_detected:
            matched_detected_count += 1
            if coordinate_error is not None:
                matched_coordinate_errors.append(coordinate_error)
        elif original_detected and not flipped_detected:
            original_only_detected_count += 1
        elif flipped_detected and not original_detected:
            flip_only_detected_count += 1
        else:
            neither_detected_count += 1
        row = dict(compact)
        row.update(
            {
                "keypoint": index,
                "flip_partner_channel": partner,
                "is_kp5": index == 5,
                "is_centroid": index == 8,
                "original_detected": original_detected,
                "flipped_detected": flipped_detected,
                "matched_detected": matched_detected,
                "original_argmax_x": (
                    None
                    if point_xy(stats[index].get("_arg_px")) is None
                    else point_xy(stats[index].get("_arg_px"))[0]
                ),
                "original_argmax_y": (
                    None
                    if point_xy(stats[index].get("_arg_px")) is None
                    else point_xy(stats[index].get("_arg_px"))[1]
                ),
                "flip_unwarped_argmax_x": None if flip_arg is None else flip_arg[0],
                "flip_unwarped_argmax_y": None if flip_arg is None else flip_arg[1],
                "argmax_consistency_px_all_channels_diagnostic": argmax_coordinate_error,
                "original_softargmax_x": (
                    None
                    if point_xy(stats[index].get("_soft_px")) is None
                    else point_xy(stats[index].get("_soft_px"))[0]
                ),
                "original_softargmax_y": (
                    None
                    if point_xy(stats[index].get("_soft_px")) is None
                    else point_xy(stats[index].get("_soft_px"))[1]
                ),
                "flip_unwarped_softargmax_x": None if flip_soft is None else flip_soft[0],
                "flip_unwarped_softargmax_y": None if flip_soft is None else flip_soft[1],
                "softargmax_consistency_px": coordinate_error,
                "matched_detected_softargmax_consistency_px": (
                    coordinate_error if matched_detected else None
                ),
                "original_softargmax_error_gt_px": stats[index].get(
                    "softargmax_error_gt_px"
                ),
                "moment_consistency_px": euclidean(
                    stats[index].get("_moment_px"), flip_moment
                ),
                "flip_unwarped_softargmax_error_gt_px": euclidean(
                    flip_soft,
                    gt_points[index] if correspondence_valid or index == 8 else None,
                ),
                "covariance_frobenius_px2": covariance_frobenius,
                "flip_unwarped_cov_xx": (
                    None if transformed_cov is None else float(transformed_cov[0, 0])
                ),
                "flip_unwarped_cov_xy": (
                    None if transformed_cov is None else float(transformed_cov[0, 1])
                ),
                "flip_unwarped_cov_yy": (
                    None if transformed_cov is None else float(transformed_cov[1, 1])
                ),
                "belief_equivariance_rmse": belief_rmse,
                "belief_equivariance_mae": belief_mae,
            }
        )
        tables["flip_keypoints"].append(row)

    if run_pose_diagnostics:
        flip_pose, flip_solver_ms, flip_solver_error, flip_cache_hit = cache.solve(
            flip_points
        )
    else:
        flip_pose, flip_solver_ms, flip_solver_error, flip_cache_hit = (
            None,
            0.0,
            "not_applicable_synthetic_channel_order_ambiguous",
            False,
        )
    flip_row = dict(compact)
    flip_row.update(
        {
            "flip_mapping": "0<->1,2<->3,4<->5,6<->7,8->8",
            "coordinate_unwarp": (
                "x_grid_original=(belief_width-1)-x_grid_flipped; "
                "x_pixel=(image_width-belief_scale_x)-x_pixel_flipped"
            ),
            "n_original_soft_points": int(sum(point_valid(point) for point in soft_points)),
            "n_flip_soft_points": int(sum(point_valid(point) for point in flip_points)),
            "n_matched_detected_keypoints": matched_detected_count,
            "n_original_only_detected_keypoints": original_only_detected_count,
            "n_flip_only_detected_keypoints": flip_only_detected_count,
            "n_neither_detected_keypoints": neither_detected_count,
            "mean_matched_detected_softargmax_consistency_px": (
                float(np.mean(matched_coordinate_errors))
                if matched_coordinate_errors
                else None
            ),
            "max_matched_detected_softargmax_consistency_px": (
                float(np.max(matched_coordinate_errors))
                if matched_coordinate_errors
                else None
            ),
            "mean_all_channel_argmax_consistency_px_diagnostic": (
                float(np.mean(all_argmax_coordinate_errors))
                if all_argmax_coordinate_errors
                else None
            ),
            "mean_belief_equivariance_rmse": float(np.mean(belief_rmses)),
            "max_belief_equivariance_rmse": float(np.max(belief_rmses)),
            "mean_belief_equivariance_mae": float(np.mean(belief_maes)),
            "original_pose_success": baseline is not None,
            "flip_solver_runtime_ms": flip_solver_ms,
            "flip_solver_error": flip_solver_error,
            "flip_solver_cache_hit": flip_cache_hit,
        }
    )
    flip_row.update(pose_fields(flip_pose, oracle, stored, dims))
    if flip_pose is not None and baseline is not None:
        flip_delta = yaw_difference(
            yaw_deg(flip_pose["R"]), yaw_deg(baseline["R"])
        )
        flip_row.update(
            {
                "flip_vs_original_yaw_signed_deg": flip_delta["signed"],
                "flip_vs_original_yaw_raw_deg": flip_delta["raw"],
                "flip_vs_original_yaw_sym180_deg": flip_delta["sym180"],
                "flip_vs_original_rotation_error_deg": rotation_error_deg(
                    np.asarray(flip_pose["R"]), np.asarray(baseline["R"])
                ),
                "flip_vs_original_translation_error_m": float(
                    np.linalg.norm(
                        np.asarray(flip_pose["t"]) - np.asarray(baseline["t"])
                    )
                ),
                "flip_vs_original_add_m": add_error(flip_pose, baseline, dims),
            }
        )
    else:
        flip_row.update(
            {
                "flip_vs_original_yaw_signed_deg": None,
                "flip_vs_original_yaw_raw_deg": None,
                "flip_vs_original_yaw_sym180_deg": None,
                "flip_vs_original_rotation_error_deg": None,
                "flip_vs_original_translation_error_m": None,
                "flip_vs_original_add_m": None,
            }
        )
    tables["flip_consistency"].append(flip_row)

    arg_order_free = order_free_corner_metrics(arg_points, gt_points)
    soft_order_free = order_free_corner_metrics(soft_points, gt_points)
    if run_pose_diagnostics:
        locked_baseline, _, locked_baseline_error, _ = locked_cache.solve(soft_points)
    else:
        locked_baseline = None
        locked_baseline_error = (
            "not_applicable_synthetic_channel_order_ambiguous"
        )
    frame_row = dict(base)
    frame_row.update(
        {
            "status": "ok",
            "error": None,
            "inference_ms": inference_ms,
            "flip_inference_ms": flip_inference_ms,
            "belief_height": int(belief.shape[1]),
            "belief_width": int(belief.shape[2]),
            "belief_scale_x": scale_x,
            "belief_scale_y": scale_y,
            "n_argmax_detected": int(sum(point_valid(point) for point in arg_points)),
            "n_softargmax_detected": int(sum(point_valid(point) for point in soft_points)),
            "argmax_order_free_matched_corners": arg_order_free["matched_count"],
            "argmax_order_free_mean_px": arg_order_free["mean_px"],
            "argmax_order_free_median_px": arg_order_free["median_px"],
            "argmax_order_free_max_px": arg_order_free["max_px"],
            "softargmax_order_free_matched_corners": soft_order_free["matched_count"],
            "softargmax_order_free_mean_px": soft_order_free["mean_px"],
            "softargmax_order_free_median_px": soft_order_free["median_px"],
            "softargmax_order_free_max_px": soft_order_free["max_px"],
            "mean8_2d_centroid_x": (
                None if mean8_centroid is None else mean8_centroid[0]
            ),
            "mean8_2d_centroid_y": (
                None if mean8_centroid is None else mean8_centroid[1]
            ),
            "mean8_2d_centroid_error_gt_px": euclidean(
                mean8_centroid, gt_points[8]
            ),
            "geometry_projected_centroid_x": (
                None if geometry_centroid is None else geometry_centroid[0]
            ),
            "geometry_projected_centroid_y": (
                None if geometry_centroid is None else geometry_centroid[1]
            ),
            "geometry_projected_centroid_error_gt_px": euclidean(
                geometry_centroid, gt_points[8]
            ),
            "y0_pose_success": oracle is not None,
            "y0_solver_runtime_ms": oracle_ms,
            "y0_solver_error": oracle_error,
            "y0_solver_cache_hit": oracle_cached,
            "y0_yaw_deg": None if oracle is None else yaw_deg(oracle["R"]),
            "y0_reproj_error_px": (
                None if oracle is None else finite_float(oracle.get("reproj_error_px"))
            ),
            "y2_pose_success": baseline is not None,
            "y2_yaw_deg": None if baseline is None else yaw_deg(baseline["R"]),
            "y2_reproj_error_px": (
                None if baseline is None else finite_float(baseline.get("reproj_error_px"))
            ),
            "locked_y0_pose_success": locked_oracle is not None,
            "locked_y0_solver_error": locked_oracle_error,
            "locked_y0_solver_runtime_ms": locked_oracle_ms,
            "locked_y0_yaw_deg": (
                None if locked_oracle is None else yaw_deg(locked_oracle["R"])
            ),
            "locked_y2_pose_success": locked_baseline is not None,
            "locked_y2_solver_error": locked_baseline_error,
            "locked_y2_yaw_deg": (
                None if locked_baseline is None else yaw_deg(locked_baseline["R"])
            ),
            "flip_pose_success": flip_pose is not None,
            "mean_flip_softargmax_consistency_px": (
                float(np.mean(matched_coordinate_errors))
                if matched_coordinate_errors
                else None
            ),
            "n_flip_matched_detected_keypoints": matched_detected_count,
        }
    )
    tables["frames"].append(frame_row)


def failed_frame_row(spec: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "frame_uid": f"{spec['dataset']}:{spec['domain']}:{spec['fid']}",
        "dataset": spec["dataset"],
        "domain": spec["domain"],
        "fid": spec["fid"],
        "split_role": spec["split_role"],
        "is_primary": spec["is_primary"],
        "legacy_aggregate": spec["legacy_aggregate"],
        "source_session": spec.get("source_session"),
        "json_path": spec["json"],
        "image_path": spec["png"],
        "status": "error",
        "error": clean_error(exc),
        "filter_pass": None,
    }


def field_order(rows: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    fields = field_order(rows)
    if not fields:
        fields = ["frame_uid"]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})
    return fields


def group_summary(frames: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in frames:
        groups.setdefault(str(row.get("split_role")), []).append(row)
    result: dict[str, Any] = {}
    for name, rows in groups.items():
        ok = [row for row in rows if row.get("status") == "ok"]
        result[name] = {
            "n": len(rows),
            "n_ok": len(ok),
            "n_error": len(rows) - len(ok),
            "y0_pose_success": sum(bool(row.get("y0_pose_success")) for row in ok),
            "y0_pose_success_rate": (
                float(np.mean([bool(row.get("y0_pose_success")) for row in ok]))
                if ok
                else None
            ),
            "y2_pose_success": sum(bool(row.get("y2_pose_success")) for row in ok),
            "y2_pose_success_rate": (
                float(np.mean([bool(row.get("y2_pose_success")) for row in ok]))
                if ok
                else None
            ),
            "flip_pose_success": sum(bool(row.get("flip_pose_success")) for row in ok),
            "flip_pose_success_rate": (
                float(np.mean([bool(row.get("flip_pose_success")) for row in ok]))
                if ok
                else None
            ),
        }
    strict = groups.get("strict_filterval", [])
    exploratory = groups.get("exploratory_pl_pool_manual", [])
    result["legacy_filterval_aggregate_123"] = {
        "n": len(strict) + len(exploratory),
        "strict_primary_n": len(strict),
        "exploratory_manual_n": len(exploratory),
    }
    return result


def paired_bootstrap_ci(
    records: list[tuple[str, str, float]],
    seed: int,
    replicates: int,
    total_pair_count: Optional[int] = None,
    conditioning: Optional[str] = None,
) -> dict[str, Any]:
    """Mean paired-delta CI, clustered by capture session when support allows."""
    clean = [
        (str(frame), str(session), float(value))
        for frame, session, value in records
        if finite_float(value) is not None
    ]
    if not clean:
        return {
            "n_pairs": 0,
            "n_total_paired_frames": total_pair_count,
            "n_excluded": total_pair_count,
            "n_clusters": 0,
            "mean_delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "method": None,
            "replicates": replicates,
            "seed": seed,
            "conditioning": conditioning,
            "limitation": "no complete paired observations",
        }
    clusters: dict[str, list[float]] = {}
    for _, session, value in clean:
        clusters.setdefault(session, []).append(value)
    cluster_sizes = {name: len(values) for name, values in sorted(clusters.items())}
    rng = np.random.default_rng(seed)
    n_clusters = len(clusters)
    singleton_structure = all(len(values) == 1 for values in clusters.values())
    if n_clusters >= 4 and not singleton_structure:
        sums = np.asarray([sum(values) for values in clusters.values()], dtype=np.float64)
        counts = np.asarray([len(values) for values in clusters.values()], dtype=np.float64)
        draws = rng.integers(0, n_clusters, size=(replicates, n_clusters))
        bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        method = "session_cluster_bootstrap"
        limitation = (
            f"only {n_clusters} capture sessions; CI reflects between-session "
            "resampling and does not separately identify within-session temporal dependence"
        )
        block_length = None
    else:
        # Circular temporal blocks are a fail-safe for smoke runs or a future
        # split whose session labels collapse to too few/singleton clusters.
        ordered = np.asarray(
            [value for _, _, value in sorted(clean, key=lambda item: item[0])],
            dtype=np.float64,
        )
        n = len(ordered)
        block_length = max(1, min(n, int(round(math.sqrt(n)))))
        n_blocks = int(math.ceil(n / block_length))
        starts = rng.integers(0, n, size=(replicates, n_blocks))
        offsets = np.arange(block_length, dtype=np.int64)
        indices = (starts[:, :, np.newaxis] + offsets) % n
        sampled = ordered[indices.reshape(replicates, -1)[:, :n]]
        bootstrap = sampled.mean(axis=1)
        method = "circular_temporal_block_bootstrap"
        limitation = (
            f"session-cluster bootstrap unavailable ({n_clusters} clusters, "
            f"singleton_structure={singleton_structure}); temporal order uses fid "
            f"lexical order with circular blocks of length {block_length}"
        )
    return {
        "n_pairs": len(clean),
        "n_total_paired_frames": (
            len(clean) if total_pair_count is None else total_pair_count
        ),
        "n_excluded": (
            0 if total_pair_count is None else total_pair_count - len(clean)
        ),
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "mean_delta": float(np.mean([value for _, _, value in clean])),
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
        "method": method,
        "block_length": block_length,
        "replicates": replicates,
        "seed": seed,
        "conditioning": conditioning,
        "limitation": limitation,
    }


def bootstrap_diagnostics(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Paired strict-N87 ladder and solver deltas with fixed-seed 95% CIs."""
    metrics = (
        "yaw_error_vs_oracle_sym180_deg",
        "gt_fixed_reproj_error_px",
        "add_vs_oracle_m",
    )
    ladder_rows = [
        row
        for row in tables["yaw_ladder"]
        if row.get("split_role") == "strict_filterval"
    ]
    ladder_by_frame: dict[str, dict[str, dict[str, Any]]] = {}
    for row in ladder_rows:
        ladder_by_frame.setdefault(str(row["frame_uid"]), {})[str(row["stage"])] = row
    ladder_result: dict[str, Any] = {}
    stages = sorted({str(row["stage"]) for row in ladder_rows})
    for stage in stages:
        if stage == "Y2":
            continue
        stage_result: dict[str, Any] = {}
        paired_frames = []
        for frame_uid, by_stage in ladder_by_frame.items():
            candidate = by_stage.get(stage)
            baseline = by_stage.get("Y2")
            if (
                candidate is not None
                and baseline is not None
                and candidate.get("source_session") is not None
            ):
                paired_frames.append(
                    (
                        frame_uid,
                        str(candidate["source_session"]),
                        candidate,
                        baseline,
                    )
                )
        success_records = [
            (
                frame_uid,
                session,
                float(bool(candidate.get("pose_success")))
                - float(bool(baseline.get("pose_success"))),
            )
            for frame_uid, session, candidate, baseline in paired_frames
        ]
        stage_result["pose_success_rate_delta"] = paired_bootstrap_ci(
            success_records,
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
            total_pair_count=len(paired_frames),
            conditioning="all paired frames; success=1 and failure=0",
        )
        stage_result["success_outcomes"] = {
            "n_total": len(paired_frames),
            "candidate_success": sum(
                bool(candidate.get("pose_success"))
                for _, _, candidate, _ in paired_frames
            ),
            "baseline_success": sum(
                bool(baseline.get("pose_success"))
                for _, _, _, baseline in paired_frames
            ),
            "candidate_only_success": sum(
                bool(candidate.get("pose_success"))
                and not bool(baseline.get("pose_success"))
                for _, _, candidate, baseline in paired_frames
            ),
            "baseline_only_success": sum(
                not bool(candidate.get("pose_success"))
                and bool(baseline.get("pose_success"))
                for _, _, candidate, baseline in paired_frames
            ),
            "both_failed": sum(
                not bool(candidate.get("pose_success"))
                and not bool(baseline.get("pose_success"))
                for _, _, candidate, baseline in paired_frames
            ),
        }
        for metric in metrics:
            pairs: list[tuple[str, str, float]] = []
            for frame_uid, session, candidate, baseline in paired_frames:
                value = finite_float(candidate.get(metric))
                base_value = finite_float(baseline.get(metric))
                if value is None or base_value is None:
                    continue
                pairs.append((frame_uid, session, value - base_value))
            stage_result[metric] = paired_bootstrap_ci(
                pairs,
                seed=BOOTSTRAP_SEED,
                replicates=BOOTSTRAP_REPLICATES,
                total_pair_count=len(paired_frames),
                conditioning=(
                    "complete finite metric pairs only; consult success_outcomes "
                    "and pose_success_rate_delta for failures"
                ),
            )
        ladder_result[f"{stage}_minus_Y2"] = stage_result

    solver_rows = [
        row
        for row in tables["solver_comparison"]
        if row.get("split_role") == "strict_filterval"
        and row.get("solver")
        in {"EPnP", "EPnP+RANSAC", "SQPNP", "SQPNP+RefineLM", "ITERATIVE"}
    ]
    solver_by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in solver_rows:
        key = (str(row["frame_uid"]), str(row["input_kind"]))
        solver_by_key.setdefault(key, {})[str(row["solver"])] = row
    solver_result: dict[str, Any] = {}
    input_kinds = sorted({str(row["input_kind"]) for row in solver_rows})
    solvers = sorted({str(row["solver"]) for row in solver_rows})
    for input_kind in input_kinds:
        for solver in solvers:
            if solver == "EPnP":
                continue
            comparison: dict[str, Any] = {}
            paired_frames = []
            for (frame_uid, kind), by_solver in solver_by_key.items():
                if kind != input_kind:
                    continue
                candidate = by_solver.get(solver)
                baseline = by_solver.get("EPnP")
                if (
                    candidate is not None
                    and baseline is not None
                    and candidate.get("source_session") is not None
                ):
                    paired_frames.append(
                        (
                            frame_uid,
                            str(candidate["source_session"]),
                            candidate,
                            baseline,
                        )
                    )
            success_records = [
                (
                    frame_uid,
                    session,
                    float(bool(candidate.get("pose_success")))
                    - float(bool(baseline.get("pose_success"))),
                )
                for frame_uid, session, candidate, baseline in paired_frames
            ]
            comparison["pose_success_rate_delta"] = paired_bootstrap_ci(
                success_records,
                seed=BOOTSTRAP_SEED,
                replicates=BOOTSTRAP_REPLICATES,
                total_pair_count=len(paired_frames),
                conditioning="all paired frames; success=1 and failure=0",
            )
            comparison["success_outcomes"] = {
                "n_total": len(paired_frames),
                "candidate_success": sum(
                    bool(candidate.get("pose_success"))
                    for _, _, candidate, _ in paired_frames
                ),
                "baseline_success": sum(
                    bool(baseline.get("pose_success"))
                    for _, _, _, baseline in paired_frames
                ),
                "candidate_only_success": sum(
                    bool(candidate.get("pose_success"))
                    and not bool(baseline.get("pose_success"))
                    for _, _, candidate, baseline in paired_frames
                ),
                "baseline_only_success": sum(
                    not bool(candidate.get("pose_success"))
                    and bool(baseline.get("pose_success"))
                    for _, _, candidate, baseline in paired_frames
                ),
                "both_failed": sum(
                    not bool(candidate.get("pose_success"))
                    and not bool(baseline.get("pose_success"))
                    for _, _, candidate, baseline in paired_frames
                ),
            }
            for metric in metrics:
                pairs = []
                for frame_uid, session, candidate, baseline in paired_frames:
                    value = finite_float(candidate.get(metric))
                    base_value = finite_float(baseline.get(metric))
                    if value is None or base_value is None:
                        continue
                    pairs.append((frame_uid, session, value - base_value))
                comparison[metric] = paired_bootstrap_ci(
                    pairs,
                    seed=BOOTSTRAP_SEED,
                    replicates=BOOTSTRAP_REPLICATES,
                    total_pair_count=len(paired_frames),
                    conditioning=(
                        "complete finite metric pairs only; consult success_outcomes "
                        "and pose_success_rate_delta for failures"
                    ),
                )
            solver_result[f"{input_kind}:{solver}_minus_EPnP"] = comparison
    return {
        "scope": "strict_filterval only (outside44 + night43; full run N=87)",
        "primary_yaw_metric": "yaw_error_vs_oracle_sym180_deg",
        "paired_delta_sign": "candidate minus named baseline; negative is improvement",
        "confidence_level": 0.95,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "ladder_baseline": (
            "Y2 predicted local-softargmax -> canonical current PnP with W/D auto-swap"
        ),
        "solver_baseline": (
            "EPnP within the same fixed input_kind; direct fixed-dimension solvers only, "
            "all referenced to locked-dims Y0"
        ),
        "ladder": ladder_result,
        "solver": solver_result,
    }


def numeric_rows(
    rows: Iterable[dict[str, Any]], key: str
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected, values = [], []
    for row in rows:
        value = finite_float(row.get(key))
        if value is not None:
            selected.append(row)
            values.append(value)
    return selected, np.asarray(values, dtype=np.float64)


def save_figure_new(figure: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite figure: {path}")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def no_data(axis: Any, text: str = "No complete strict-filterval observations") -> None:
    axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def generate_figures(
    run_dir: Path,
    tables: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Create compact audit PNGs from strict-filterval rows only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strict_ladder = [
        row
        for row in tables["yaw_ladder"]
        if row.get("split_role") == "strict_filterval"
    ]
    stages = [f"Y{index}" for index in range(10)]
    ladder_data = [
        [
            finite_float(row.get("yaw_error_vs_oracle_sym180_deg"))
            for row in strict_ladder
            if row.get("stage") == stage
            and finite_float(row.get("yaw_error_vs_oracle_sym180_deg")) is not None
        ]
        for stage in stages
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    valid = [(stage, data) for stage, data in zip(stages, ladder_data) if data]
    if valid:
        labels = []
        for name, _ in valid:
            rows = [row for row in strict_ladder if row.get("stage") == name]
            labels.append(
                f"{name}\n{sum(bool(row.get('pose_success')) for row in rows)}/{len(rows)} ok"
            )
        ax.boxplot([data for _, data in valid], tick_labels=labels)
        ax.set_ylabel("yaw err_180 (deg; primary)")
        ax.set_xlabel("cause ladder stage")
        ax.grid(axis="y", alpha=0.25)
    else:
        no_data(ax)
    ax.set_title("Strict filter-val yaw cause ladder")
    save_figure_new(fig, run_dir / "yaw_cause_ladder.png")

    strict_influence = [
        row
        for row in tables["keypoint_influence"]
        if row.get("split_role") == "strict_filterval"
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(N_KEYPOINTS)
    width_bar = 0.38
    plotted = False
    for offset, variant, label in (
        (-width_bar / 2, "leave_one_out", "leave one out"),
        (width_bar / 2, "replace_one_with_gt", "replace with GT"),
    ):
        means = []
        for keypoint in range(N_KEYPOINTS):
            values = [
                finite_float(
                    row.get("candidate_error_minus_baseline_yaw_sym180_deg")
                )
                for row in strict_influence
                if row.get("variant") == variant and row.get("keypoint") == keypoint
            ]
            values = [value for value in values if value is not None]
            means.append(float(np.mean(values)) if values else np.nan)
        if np.isfinite(means).any():
            ax.bar(x + offset, means, width_bar, label=label)
            plotted = True
    if plotted:
        ax.set_xticks(x, [str(index) for index in range(N_KEYPOINTS)])
        ax.set_xlabel("keypoint (5=kp5, 8=centroid)")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("mean Δ yaw err_180 (deg; negative=better)")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    else:
        no_data(ax)
    ax.set_title("Strict filter-val keypoint influence")
    save_figure_new(fig, run_dir / "keypoint_influence_delta_yaw.png")

    strict_perturb = [
        row
        for row in tables["kp5_perturbation"]
        if row.get("split_role") == "strict_filterval"
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plotted = False
    for axis_name, marker in (("x", "o"), ("y", "s")):
        points = []
        for delta in (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0):
            values = [
                finite_float(
                    row.get("candidate_error_minus_baseline_yaw_sym180_deg")
                )
                for row in strict_perturb
                if row.get("axis") == axis_name
                and row.get("kp5_prediction_available")
                and finite_float(
                    row.get("delta_grid_x" if axis_name == "x" else "delta_grid_y")
                )
                == delta
            ]
            values = [value for value in values if value is not None]
            if values:
                points.append((delta, float(np.mean(values))))
        if points:
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker=marker,
                label=f"kp5 {axis_name}-shift",
            )
            plotted = True
    if plotted:
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("kp5 perturbation (belief-grid cells)")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("mean Δ yaw err_180 (deg; negative=better)")
        ax.legend()
        ax.grid(alpha=0.25)
    else:
        no_data(ax)
    ax.set_title("Strict filter-val kp5 local perturbation sensitivity")
    save_figure_new(fig, run_dir / "kp5_perturbation_sensitivity.png")

    centroid_rows = [
        row
        for row in tables["keypoints"]
        if row.get("split_role") == "strict_filterval" and row.get("keypoint") == 8
    ]
    pairs = [
        (
            finite_float(row.get("gt_elevation_deg")),
            finite_float(row.get("softargmax_error_gt_px")),
            str(row.get("domain")),
        )
        for row in centroid_rows
    ]
    pairs = [pair for pair in pairs if pair[0] is not None and pair[1] is not None]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if pairs:
        for domain in sorted({pair[2] for pair in pairs}):
            subset = [pair for pair in pairs if pair[2] == domain]
            ax.scatter(
                [pair[0] for pair in subset],
                [pair[1] for pair in subset],
                s=22,
                alpha=0.75,
                label=domain,
            )
        if len(pairs) >= 3:
            coefficients = np.polyfit(
                np.asarray([pair[0] for pair in pairs]),
                np.asarray([pair[1] for pair in pairs]),
                1,
            )
            xx = np.linspace(min(pair[0] for pair in pairs), max(pair[0] for pair in pairs), 100)
            ax.plot(xx, np.polyval(coefficients, xx), color="black", linewidth=1.2)
        ax.set_xlabel("GT elevation (deg)")
        ax.set_ylabel("centroid softargmax residual (px)")
        ax.legend()
        ax.grid(alpha=0.25)
    else:
        no_data(ax)
    ax.set_title("Strict filter-val centroid residual vs elevation")
    save_figure_new(fig, run_dir / "centroid_residual_vs_elevation.png")

    covariance_rows = [
        row
        for row in tables["keypoints"]
        if row.get("split_role") == "strict_filterval"
        and row.get("kp_error_valid_for_conclusion")
    ]
    covariance_rows = [
        row
        for row in covariance_rows
        if finite_float(row.get("mahalanobis_gt")) is not None
        and finite_float(row.get("softargmax_error_gt_px")) is not None
        and finite_float(row.get("cov_px_xx")) is not None
        and finite_float(row.get("cov_px_yy")) is not None
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if covariance_rows:
        mahalanobis = np.asarray(
            [finite_float(row["mahalanobis_gt"]) for row in covariance_rows]
        )
        nominal = np.linspace(0.05, 0.95, 19)
        thresholds = np.sqrt(-2.0 * np.log(1.0 - nominal))
        empirical = np.asarray(
            [np.mean(mahalanobis <= threshold) for threshold in thresholds]
        )
        axes[0].plot(nominal, empirical, marker="o", markersize=3)
        axes[0].plot([0, 1], [0, 1], "--", color="black", linewidth=1)
        axes[0].set_xlabel("nominal 2D Gaussian coverage")
        axes[0].set_ylabel("empirical coverage")
        axes[0].grid(alpha=0.25)
        sigma = np.sqrt(
            np.asarray(
                [
                    finite_float(row["cov_px_xx"]) + finite_float(row["cov_px_yy"])
                    for row in covariance_rows
                ]
            )
        )
        error = np.asarray(
            [finite_float(row["softargmax_error_gt_px"]) for row in covariance_rows]
        )
        axes[1].scatter(sigma, error, s=12, alpha=0.5)
        axes[1].set_xlabel("sqrt(trace(local covariance)) (px)")
        axes[1].set_ylabel("softargmax error (px)")
        axes[1].grid(alpha=0.25)
    else:
        no_data(axes[0])
        no_data(axes[1])
    fig.suptitle("Strict filter-val local 7x7 covariance calibration (calibration-only)")
    save_figure_new(fig, run_dir / "covariance_coverage_calibration.png")

    solver_rows = [
        row
        for row in tables["solver_comparison"]
        if row.get("split_role") == "strict_filterval"
        and row.get("input_kind") == "predicted_softargmax"
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    metric_titles = (
        ("yaw_error_vs_oracle_sym180_deg", "yaw err_180 (deg)"),
        ("gt_fixed_reproj_error_px", "fixed-GT reprojection (px)"),
        ("add_vs_oracle_m", "ADD vs Y0 (m)"),
    )
    solver_order = (
        "EPnP",
        "EPnP+RANSAC",
        "SQPNP",
        "SQPNP+RefineLM",
        "ITERATIVE",
    )
    for axis, (metric, title) in zip(axes, metric_titles):
        data, labels = [], []
        for solver in solver_order:
            values = [
                finite_float(row.get(metric))
                for row in solver_rows
                if row.get("solver") == solver
            ]
            values = [value for value in values if value is not None]
            if values:
                data.append(values)
                labels.append(solver.replace("+", "\n+"))
        if data:
            axis.boxplot(data, tick_labels=labels)
            axis.tick_params(axis="x", labelsize=8)
            axis.set_ylabel(title)
            axis.grid(axis="y", alpha=0.25)
        else:
            no_data(axis)
    success_rates = []
    success_labels = []
    for solver in solver_order:
        rows = [row for row in solver_rows if row.get("solver") == solver]
        if rows:
            success_labels.append(solver.replace("+", "\n+"))
            success_rates.append(
                100.0 * np.mean([bool(row.get("pose_success")) for row in rows])
            )
    if success_rates:
        axes[3].bar(np.arange(len(success_rates)), success_rates)
        axes[3].set_xticks(np.arange(len(success_rates)), success_labels)
        axes[3].tick_params(axis="x", labelsize=8)
        axes[3].set_ylim(0.0, 105.0)
        axes[3].set_ylabel("pose success (%)")
        axes[3].grid(axis="y", alpha=0.25)
    else:
        no_data(axes[3])
    fig.suptitle(
        "Strict filter-val direct fixed-dims solvers: predicted softargmax vs locked Y0"
    )
    save_figure_new(fig, run_dir / "solver_yaw_reproj_add.png")

    flip_rows = [
        row
        for row in tables["flip_keypoints"]
        if row.get("split_role") == "strict_filterval"
        and row.get("matched_detected")
    ]
    flip_pairs = [
        (
            finite_float(row.get("matched_detected_softargmax_consistency_px")),
            finite_float(row.get("original_softargmax_error_gt_px")),
            int(row.get("keypoint")),
        )
        for row in flip_rows
    ]
    flip_pairs = [
        pair for pair in flip_pairs if pair[0] is not None and pair[1] is not None
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if flip_pairs:
        scatter = ax.scatter(
            [pair[0] for pair in flip_pairs],
            [pair[1] for pair in flip_pairs],
            c=[pair[2] for pair in flip_pairs],
            cmap="tab10",
            s=18,
            alpha=0.65,
        )
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("keypoint")
        ax.set_xlabel("flip softargmax consistency (px)")
        ax.set_ylabel("original keypoint GT error (px)")
        ax.grid(alpha=0.25)
    else:
        no_data(ax)
    ax.set_title("Strict filter-val flip equivariance vs localization error")
    save_figure_new(fig, run_dir / "flip_equivariance_vs_keypoint_error.png")
    return list(FIGURE_NAMES)


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    cv2.setRNGSeed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    audit = InputAudit()
    synth_all = synth_frames(audit)
    filterval_all = safe_stage25_filterval()
    synth_selected = (
        synth_all[: args.max_synth] if args.max_synth else synth_all
    )
    filterval_selected = (
        filterval_all[: args.max_filterval] if args.max_filterval else filterval_all
    )
    selected = synth_selected + filterval_selected
    if not selected:
        raise RuntimeError("no frames selected")

    model, state_key_count = load_model(device)
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in CSV_NAMES}
    started_utc = dt.datetime.now(dt.timezone.utc)
    wall_start = time.perf_counter()
    for index, spec in enumerate(selected, start=1):
        frame_tables: dict[str, list[dict[str, Any]]] = {
            name: [] for name in CSV_NAMES
        }
        try:
            process_frame(model, device, spec, audit, frame_tables)
            for name in CSV_NAMES:
                tables[name].extend(frame_tables[name])
        except Exception as exc:
            tables["frames"].append(failed_frame_row(spec, exc))
            print(
                f"[error {index}/{len(selected)}] {spec['dataset']}:{spec['fid']} "
                f"{clean_error(exc)}",
                flush=True,
            )
        if index == 1 or index % 10 == 0 or index == len(selected):
            print(
                f"[progress] {index}/{len(selected)} "
                f"{spec['dataset']}:{spec['domain']}:{spec['fid']}",
                flush=True,
            )

    if audit.prohibited_attempts:
        raise RuntimeError(
            f"prohibited input access was attempted: {audit.prohibited_attempts}"
        )
    finished_utc = dt.datetime.now(dt.timezone.utc)
    elapsed_seconds = time.perf_counter() - wall_start
    full_requested = args.max_synth == 0 and args.max_filterval == 0
    expected_selected = 623 if full_requested else len(selected)
    frame_count_complete = len(tables["frames"]) == expected_selected
    frame_errors = [
        row for row in tables["frames"] if row.get("status") != "ok"
    ]
    primary_frame_errors = [
        row for row in frame_errors if bool(row.get("is_primary"))
    ]
    if full_requested and (not frame_count_complete or frame_errors):
        run_status = "incomplete"
        exit_code = 2
    elif full_requested:
        run_status = "complete"
        exit_code = 0
    elif frame_errors:
        run_status = "smoke_incomplete"
        exit_code = 0
    else:
        run_status = "smoke_complete"
        exit_code = 0
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    repository_head = git_head()
    script_status = git_path_status(script_path)
    bootstrap = bootstrap_diagnostics(tables)
    summary = {
        "run_status": run_status,
        "full_requested": full_requested,
        "selected_frame_count": len(selected),
        "recorded_frame_count": len(tables["frames"]),
        "frame_error_count": len(frame_errors),
        "primary_frame_error_count": len(primary_frame_errors),
        "checkpoint": str(WEIGHTS),
        "checkpoint_sha256": WEIGHTS_SHA256,
        "grouped_frames": group_summary(tables["frames"]),
        "table_row_counts": {name: len(rows) for name, rows in tables.items()},
        "elapsed_seconds": elapsed_seconds,
        "primary_scope": "strict_filterval N87 in a full run",
        "exploratory_scope": "manual36 from capturepallet11 PL-pool",
        "synthetic_validity": (
            "Q1 N500 has ambiguous corner-channel convention: use only order-free "
            "corner aggregates, heatmap/covariance distributions, centroid, and stored "
            "GT pose metadata; fixed kp-id/PnP/yaw conclusions are invalid and omitted"
        ),
        "missing_metadata_policy": (
            "Missing metadata is null. No global dimensions, intrinsics, visibility, "
            "occlusion, session, or pose metadata is inferred."
        ),
        "covariance_pose_weighting_status": (
            "N/A_calibration_only: C0-C4 covariance-weighted PnP is not invoked by "
            "this frozen audit"
        ),
        "current_solver_policy": (
            "Y0-Y8 use the canonical historical APNP W/D auto-swap search. "
            "A locked per-frame W,D control is reported separately in solver_comparison."
        ),
        "influence_delta_policy": (
            "Signed candidate-error minus Y2-baseline error uses yaw err_180, ADD vs "
            "Y0, and reprojection on the same fixed GT 2D observations. Pose-to-pose "
            "displacement and solver-own-input reprojection are separately named."
        ),
        "bootstrap_95ci": bootstrap,
    }

    run_dir, run_name = make_run_dir(args.run_name)
    schemas: dict[str, list[str]] = {}
    for name in CSV_NAMES:
        schemas[f"{name}.csv"] = write_csv_new(run_dir / f"{name}.csv", tables[name])
    generated_figures = generate_figures(run_dir, tables)
    write_json_new(run_dir / "summary.json", summary)
    manifest = {
        "format_version": 1,
        "run_status": run_status,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "started_utc": started_utc.isoformat(),
        "finished_utc": finished_utc.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "command_args": vars(args),
        "frozen_inference": True,
        "training_or_checkpoint_selection_performed": False,
        "source": {
            "script_path": str(script_path),
            "script_sha256": script_sha,
            "git_head": repository_head,
            "script_git_status": script_status,
        },
        "randomness": {
            "python_side_seed": 0,
            "numpy_seed": 0,
            "torch_seed": 0,
            "opencv_rng_seed": 0,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "torch_deterministic_algorithms_warn_only": True,
        },
        "checkpoint": {
            "path": str(WEIGHTS),
            "sha256": WEIGHTS_SHA256,
            "state_key_count": state_key_count,
            "architecture": "DopeNetwork(numVec=0,numSeg=1)",
            "load_state_dict_strict": True,
        },
        "device": str(device),
        "opencv_version": cv2.__version__,
        "torch_version": torch.__version__,
        "dataset_contract": {
            "q1_list_path": str(SYNTH_LIST),
            "q1_list_sha256_locked": Q1_LIST_SHA256,
            "synthetic_q1_fixed_full": 500,
            "synthetic_selected": len(synth_selected),
            "strict_filterval_primary_full": 87,
            "exploratory_pl_pool_manual_full": 36,
            "legacy_filterval_aggregate_full": 123,
            "strict_filterval_identity_sha256_locked": STRICT_FILTERVAL_ID_SHA256,
            "exploratory_manual36_identity_sha256_locked": (
                EXPLORATORY_MANUAL36_ID_SHA256
            ),
            "legacy_filterval123_identity_sha256_locked": (
                LEGACY_FILTERVAL123_ID_SHA256
            ),
            "membership_identity_hash_format": (
                "SHA256 of sorted UTF-8 lines: domain<TAB>fid<TAB>source_session"
                "<TAB>split_role<LF>"
            ),
            "filterval_selected": len(filterval_selected),
            "filterval_full_domain_counts": {
                "outside": 44,
                "night": 43,
                "manual": 36,
            },
            "manual36_provenance": (
                "capturepallet11 PL-pool; exploratory only, excluded from strict primary"
            ),
            "synthetic_corner_channel_warning": (
                "Q1 training_data/val corner channels are convention-ambiguous; "
                "kp-id, kp5, fixed-correspondence PnP, and yaw conclusions are invalid. "
                "The script nulls per-corner GT correspondence errors and skips synthetic "
                "fixed-correspondence pose diagnostics."
            ),
        },
        "input_access_audit": {
            "opened_evaluation_json_count": len(audit.json_paths),
            "opened_evaluation_image_count": len(audit.image_paths),
            "prohibited_path_open_attempt_count": len(audit.prohibited_attempts),
            "final_test_input_open_count": 0,
            "handannot17_input_open_count": 0,
            "sealed_sessions": sorted(SEALED_SESSIONS),
            "stage25_membership_note": (
                "frames_filterval called with build_session_map replaced by an "
                "allowed-session-only scanner; sealed raw session directories were "
                "not scanned for image inputs"
            ),
        },
        "preprocessing": {
            "color": "BGR to RGB",
            "resize": "anisotropic squash to 400x400, cv2.INTER_LINEAR",
            "normalization_mean": MEAN.tolist(),
            "normalization_std": STD.tolist(),
            "belief_to_image": "x*=image_width/belief_width; y*=image_height/belief_height",
            "belief_threshold": BELIEF_THRESHOLD,
        },
        "heatmap_decode": {
            "argmax": "raw finite heatmap argmax",
            "local_softargmax": {
                "window": (
                    "exact operational 7x7 gather around raw argmax; boundary indices "
                    "are clamped and therefore duplicated at an edge"
                ),
                "radius": LOCAL_RADIUS,
                "temperature": LOCAL_TEMPERATURE,
            },
            "second_peak": "maximum outside the local 7x7 argmax window",
            "canonical_covariance": {
                "representation": "local 7x7 logits-softmax around raw argmax",
                "temperature": LOCAL_TEMPERATURE,
                "moments": "full 2x2 covariance including xy cross term",
                "entropy_unit": "nats",
                "covariance_regularizer_px2_for_mahalanobis": COV_REGULARIZER_PX2,
            },
            "global_logits_moments": {
                "representation": "stable full-map spatial softmax treating belief as logits",
                "temperature": MOMENT_TEMPERATURE,
                "entropy_unit": "nats",
            },
            "global_raw_value_moments": {
                "representation": "clip raw belief at zero, then sum-normalize full map",
                "warning": (
                    "separate descriptive representation; never conflated with logits-softmax"
                ),
            },
        },
        "geometry": {
            "dimensions": "per-frame W,D,H only; missing dimensions remain null",
            "intrinsics": "per-frame camera_data.intrinsics only",
            "current_pnp": (
                "canonical Y0-Y8: annotate_pnp.solve_pose(auto_swap_dims=True); "
                "locked W,D diagnostic control also reported"
            ),
            "yaw": (
                "degrees(atan2(R[0,2],R[2,2])) from the pallet local forward/Z axis"
            ),
            "yaw_symmetry_error": "min(abs(wrap180(delta)), abs(180-abs(wrap180(delta))))",
            "yaw_primary_metric": "180-degree-symmetry-aware err_180; raw/direct retained",
            "view_azimuth": "degrees(atan2(t_x,t_z)); distinct from object yaw",
            "add": "mean 3D distance across the frame's 9 pallet keypoints",
            "add_hypothesis_rule": (
                "ADD and ADD-S are null with invalid_reason when candidate/reference "
                "selected W/D hypotheses differ; ADD-S also minimizes the pallet's "
                "180-degree local-Y symmetry correspondence"
            ),
            "Y4_centroid": (
                "solve thresholded predicted corners0-7 without centroid, then project "
                "the 3D keypoint8/origin from that pose; arithmetic 2D mean is kept only "
                "as a separately named frame diagnostic"
            ),
            "dimension_hypothesis": (
                "per-frame dimensions are the only hypotheses supplied. Canonical APNP "
                "may select (W,D,H) or (D,W,H); locked and direct controls do not swap. "
                "Selected W,D,H, auto_swap flag, and hypothesis are written per pose row."
            ),
            "exact_missing_sentinel": (
                "only exact coordinate pair (-1,-1) is missing; other negative/off-image "
                "coordinates remain valid"
            ),
            "fixed_observation_reprojection": (
                "candidate poses are projected onto the same valid GT 2D correspondences "
                "before LOO/replacement/perturbation error deltas are computed"
            ),
            "Y8_mask": (
                "GT coordinate in image bounds only. It is not an annotation visibility "
                "or occlusion mask."
            ),
        },
        "perturbation": {
            "grid_amplitudes": [-8, -4, -2, -1, 1, 2, 4, 8],
            "axes": ["x", "y"],
            "kp5_canonical_curve": (
                "canonical current APNP with W/D auto-swap; same-solver Y2 baseline"
            ),
            "all_keypoints_curve": (
                "kp0-8, SQPNP+RefineLM fixed-dimension direct solver; same-solver "
                "unperturbed baseline, kept separate from canonical kp5 curve"
            ),
            "signed_error_delta": (
                "candidate absolute error minus baseline absolute error; negative improves"
            ),
        },
        "covariance_pose_weighting": {
            "status": "N/A_calibration_only",
            "reason": (
                "This frozen diagnostic uses the current unweighted PnP path. C0-C4 "
                "pose weighting claims are not made."
            ),
        },
        "bootstrap_95ci": {
            "scope": bootstrap["scope"],
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "default_method": "session_cluster_bootstrap",
            "fallback": "circular temporal block bootstrap for too few/singleton sessions",
            "failure_handling": (
                "success-rate deltas include every paired frame with success=1/failure=0; "
                "continuous metrics explicitly report complete-pair conditioning and "
                "excluded counts"
            ),
            "result_file": "summary.json",
        },
        "metadata_policy": (
            "Missing metadata is serialized as null/empty CSV. It is never filled from "
            "a global default or inferred from another frame."
        ),
        "solver_comparison": {
            "solvers": [
                "CURRENT_CANONICAL_AUTOSWAP",
                "CURRENT_LOCKED_DIMS",
                "EPnP",
                "EPnP+RANSAC",
                "SQPNP",
                "SQPNP+RefineLM",
                "ITERATIVE",
            ],
            "ransac_reprojection_px": 8.0,
            "ransac_iterations": 100,
            "inputs": ["gt", "predicted_argmax", "predicted_softargmax"],
            "reference_policy": (
                "Direct fixed-dimension solvers and CURRENT_LOCKED use locked-dims Y0. "
                "CURRENT_CANONICAL uses canonical auto-swap Y0 and is retained only as "
                "a CSV diagnostic; solver figure/bootstrap include direct solvers only."
            ),
            "cheirality": (
                "direct candidates require t_z>0 and every used 3D correspondence at "
                "positive camera depth; failures remain explicit rows"
            ),
        },
        "flip": {
            "channel_partner": list(FLIP_PARTNER),
            "coordinate_unwarp": (
                "x_grid=(belief_width-1)-x_grid_flipped; equivalently "
                "x_pixel=(image_width-belief_scale_x)-x_pixel_flipped; y unchanged"
            ),
            "covariance_unwarp": "diag(-1,1) @ covariance @ diag(-1,1)",
            "pose_references": (
                "pose_fields vs_oracle columns always use Y0. Dedicated "
                "flip_vs_original_* columns compare flip pose with original Y2."
            ),
            "aggregate_policy": (
                "coordinate aggregates and the error scatter use only keypoints detected "
                "in both original and flipped inference. All-channel raw argmax consistency "
                "is separately suffixed diagnostic and never mixed into the matched metric."
            ),
        },
        "transactionality": (
            "Each frame writes to local buffers and commits all tables only after the "
            "entire frame succeeds; late errors cannot leave partial analytical rows."
        ),
        "outputs": {
            "summary.json": "run and role-separated row counts",
            **{name: f"machine-readable table ({len(tables[name[:-4]])} rows)"
               for name in schemas},
            **{name: "strict-filterval diagnostic figure" for name in generated_figures},
        },
        "csv_columns": schemas,
    }
    write_json_new(run_dir / "manifest.json", manifest)
    print(f"[done] output={run_dir}", flush=True)
    print(
        "[summary] "
        + json.dumps(
            {
                "run_status": summary["run_status"],
                "grouped_frames": summary["grouped_frames"],
                "table_row_counts": summary["table_row_counts"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "bootstrap_file": str(run_dir / "summary.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
