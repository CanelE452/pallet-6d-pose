#!/usr/bin/env python3
"""Evaluate/select PAPER_S2 single-RGB improvement checkpoints.

This evaluator is deliberately separate from the historical final-test scripts.
It reads only:

* the fixed synthetic ``q1_500`` or ``val_1500`` split;
* ``filterval_123`` as a development guard;
* ``handannot17`` as a report-only secondary set.

The sealed real final-test session names are asserted absent from every resolved
frame path.  Model selection uses synthetic + filterval only.  handannot17 can
never change ``BEST_ARM``.

For every model it reports both the backward-compatible PnP result and the new
fail-closed single-image result.  Predicted corner uncertainty is passed into
the safe PnP path.  A learned W/D prior is deliberately disabled because its
available label did not match the solver hypothesis; ambiguous W/D candidates
are rejected instead.  Input and output grids remain locked to 400 and 50.
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
import inspect
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Deep_Object_Pose" / "common"))
sys.path.insert(0, str(ROOT / "scripts" / "data_prep" / "eval"))
sys.path[:0] = [str(ROOT / "scripts" / "data_prep" / _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, str(ROOT / "scripts" / "stage0"))
sys.path.insert(0, str(ROOT / "challenge" / "scripts"))

sys.path[:0] = [str(ROOT / "challenge" / "scripts" / _s)
                for _s in ("annotate", "infer", "live")]
import models as dope_models  # noqa: E402
from heatmap_refinement import (  # noqa: E402
    decode_refinement_outputs,
    unpack_dope_output,
)
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from eval_pvnet_heads import split_metrics  # noqa: E402
from paper_s2_mask_coverage_filter import gt_coverage_features  # noqa: E402
from stage18_elevation_threshold import elev_from_pose  # noqa: E402
import annotate_pnp as APNP  # noqa: E402


GRID_SIZE = 50
INPUT_SIZE = 400
THRESH = 0.3
N_DET_MIN = 6
GOOD_PX = 10.0
GROSS_PX = 20.0
REAL_DIMS = (1.1, 1.3, 0.12)
MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)

Q1_LIST = (ROOT / "data" / "pallet" / "results" /
           "paper_s2_scratch_diffpnp" / "q1_split" / "val_list.json")
VAL_INDEX = (ROOT / "data" / "pallet" / "results" /
             "paper_s2_scratch_diffpnp" / "pnp_valid_3d_index" / "val.json")
TRAINING_ROOT = ROOT / "data" / "pallet" / "training_data"
ORTHO_JSON = (ROOT / "data" / "pallet" / "eval_results" /
              "paper_s2_scratch_diffpnp" / "orthogonal_filters_exp.json")
HAND_MANIFEST = (ROOT / "data" / "pallet" / "eval_results" /
                 "stage22_myannot_eval" / "testset_full8_manifest.txt")
MANUAL_GT_CANDIDATES = (
    ROOT / "data" / "pallet" / "eval_results" /
    "stage0_gt_candidates" / "manual_gt",
    ROOT / "data" / "pallet" / "eval_results" / "achieve" /
    "paper_base_v2_s2" / "stage0_gt_candidates" / "manual_gt",
)

@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    arm: str
    path: Path
    epoch: int
    features: tuple[str, ...]
    trainable_scope: str = "all"
    baseline: bool = False


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _manual_gt_dir() -> Path:
    for candidate in MANUAL_GT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "manual GT directory not found: " +
        ", ".join(map(str, MANUAL_GT_CANDIDATES)))


def _assert_no_sealed_paths(paths: Iterable[Path], sealed: Iterable[str]) -> None:
    bad = []
    for path in paths:
        text = str(path)
        for name in sealed:
            if name in text:
                bad.append((text, name))
    if bad:
        raise AssertionError(f"sealed final-test path(s) reached: {bad[:3]}")


def _real_membership_digest(
        real: dict[str, list[dict[str, Any]]]) -> str:
    """Hash the frozen real-set IDs so flattened paths cannot weaken sealing."""
    rows = [
        {"split": split, "domain": str(frame["domain"]),
         "fid": str(frame["fid"])}
        for split, frames in real.items() for frame in frames
    ]
    rows.sort(key=lambda row: (row["split"], row["domain"], row["fid"]))
    blob = json.dumps(
        rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _real_membership_counts(
        real: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Return JSON-stable counts for an auditable real-set membership."""
    by_split = {split: len(frames) for split, frames in sorted(real.items())}
    by_split_domain = {
        split: dict(sorted(Counter(str(frame["domain"])
                                  for frame in frames).items()))
        for split, frames in sorted(real.items())
    }
    return {
        "total": sum(by_split.values()),
        "by_split": by_split,
        "by_split_domain": by_split_domain,
    }


def _exclude_training_fids(
        canonical: dict[str, list[dict[str, Any]]],
        configured_fids: Any,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    """Remove explicitly trained frames after validating canonical membership.

    ``configured_fids`` is deliberately strict: each entry must be a unique,
    non-empty string and must occur exactly once across the canonical real
    membership.  This prevents a typo, duplicate config entry, or ambiguous
    cross-split ID from silently changing the evaluation population.
    """
    if configured_fids is None:
        configured_fids = []
    if not isinstance(configured_fids, list):
        raise TypeError("evaluation.training_exclude_fids must be a JSON list")

    exclude_fids = []
    for index, fid in enumerate(configured_fids):
        if not isinstance(fid, str) or not fid or fid != fid.strip():
            raise TypeError(
                "evaluation.training_exclude_fids entries must be "
                f"non-empty trimmed strings; index={index}, value={fid!r}")
        exclude_fids.append(fid)
    duplicates = sorted(fid for fid, count in Counter(exclude_fids).items()
                        if count != 1)
    if duplicates:
        raise ValueError(
            "duplicate evaluation.training_exclude_fids: "
            f"{duplicates}")

    locations: dict[str, list[dict[str, str]]] = {}
    for split, frames in canonical.items():
        for frame in frames:
            fid = str(frame["fid"])
            locations.setdefault(fid, []).append({
                "split": str(split),
                "domain": str(frame["domain"]),
                "fid": fid,
            })
    invalid = {fid: len(locations.get(fid, [])) for fid in exclude_fids
               if len(locations.get(fid, [])) != 1}
    if invalid:
        raise AssertionError(
            "every evaluation.training_exclude_fids entry must occur exactly "
            f"once in canonical real membership; occurrences={invalid}")

    excluded = [locations[fid][0] for fid in exclude_fids]
    excluded_set = set(exclude_fids)
    evaluated = {
        split: [frame for frame in frames
                if str(frame["fid"]) not in excluded_set]
        for split, frames in canonical.items()
    }
    removed = (sum(len(frames) for frames in canonical.values()) -
               sum(len(frames) for frames in evaluated.values()))
    if removed != len(exclude_fids):
        raise AssertionError(
            f"training exclusion count drift: removed={removed}, "
            f"configured={len(exclude_fids)}")
    return evaluated, excluded


def build_synthetic(name: str) -> list[dict[str, Any]]:
    if name == "q1_500":
        frames = _read_json(Q1_LIST)
        expected = 500
    elif name == "val_1500":
        index = _read_json(VAL_INDEX)
        frames = []
        for rel, entry in sorted(index.items()):
            jp = TRAINING_ROOT / rel
            ip = jp.with_suffix(".png")
            if jp.is_file() and ip.is_file():
                frames.append({
                    "fid": jp.stem,
                    "json": str(jp),
                    "png": str(ip),
                    "entry": entry,
                })
        expected = 1500
    else:
        raise ValueError(f"unknown synthetic set: {name}")
    if len(frames) != expected:
        raise AssertionError(
            f"synthetic membership drift for {name}: {len(frames)} != {expected}")
    for frame in frames:
        jp = Path(frame["json"]).resolve()
        ip = Path(frame["png"]).resolve()
        if not jp.is_relative_to((TRAINING_ROOT / "val").resolve()):
            raise AssertionError(f"synthetic frame is outside held-out val: {jp}")
        if not ip.is_relative_to((TRAINING_ROOT / "val").resolve()):
            raise AssertionError(f"synthetic image is outside held-out val: {ip}")
    return frames


def build_real() -> dict[str, list[dict[str, Any]]]:
    """Rebuild the exact 123+17 membership without importing archived scripts."""
    prior = _read_json(ORTHO_JSON)
    manual = _manual_gt_dir()
    filterval = []
    seen: set[tuple[str, str]] = set()
    for row in prior:
        domain, fid = str(row["dom"]), str(row["fid"])
        if domain == "manual":
            base = manual
        elif domain in ("outside", "night"):
            base = ROOT / "data" / "_eval_sets" / f"{domain}_combined"
        else:
            raise AssertionError(f"unexpected filterval domain: {domain}")
        ip, jp = base / f"{fid}.png", base / f"{fid}.json"
        if not ip.is_file() or not jp.is_file():
            raise FileNotFoundError(f"missing filterval pair: {ip} / {jp}")
        key = (domain, fid)
        if key in seen:
            raise AssertionError(f"duplicate real frame: {key}")
        seen.add(key)
        filterval.append({
            "split": "filterval",
            "domain": domain,
            "fid": fid,
            "png": str(ip),
            "json": str(jp),
            "dims": REAL_DIMS,
        })

    handannot = []
    for line in HAND_MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        domain, fid, jrel, irel = line.split()
        jp, ip = ROOT / jrel, ROOT / irel
        if not ip.is_file() or not jp.is_file():
            raise FileNotFoundError(f"missing handannot pair: {ip} / {jp}")
        key = (domain, fid)
        if key in seen:
            raise AssertionError(f"duplicate real frame: {key}")
        seen.add(key)
        handannot.append({
            "split": "handannot17",
            "domain": domain,
            "fid": fid,
            "png": str(ip),
            "json": str(jp),
            "dims": REAL_DIMS,
        })

    counts = Counter((r["split"], r["domain"])
                     for r in filterval + handannot)
    expected = Counter({
        ("filterval", "outside"): 44,
        ("filterval", "night"): 43,
        ("filterval", "manual"): 36,
        ("handannot17", "cad"): 11,
        ("handannot17", "noapril"): 6,
    })
    if counts != expected or len(filterval) != 123 or len(handannot) != 17:
        raise AssertionError(
            f"real membership drift: counts={counts}, "
            f"filterval={len(filterval)}, handannot={len(handannot)}")
    return {"filterval": filterval, "handannot17": handannot}


def preprocess_squash(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    value = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(value.transpose(2, 0, 1)).float().unsqueeze(0)


def camera_from_json(data: dict[str, Any]) -> np.ndarray:
    intr = data["camera_data"]["intrinsics"]
    return np.asarray([
        [intr["fx"], 0.0, intr["cx"]],
        [0.0, intr["fy"], intr["cy"]],
        [0.0, 0.0, 1.0],
    ], np.float64)


def _state_without_module(path: Path, device: str) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {path}")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value
                 for key, value in state.items()}
    return state


def model_kwargs(features: Iterable[str]) -> dict[str, Any]:
    features = set(features)
    kwargs: dict[str, Any] = {"numVec": 0, "numSeg": 1}
    optional = {
        "mask_belief_fusion": "maskBeliefFusion",
        "corner_quality": "cornerQuality",
    }
    signature = inspect.signature(dope_models.DopeNetwork)
    for feature, parameter in optional.items():
        if feature in features:
            if parameter not in signature.parameters:
                raise RuntimeError(
                    f"DopeNetwork does not yet expose '{parameter}' required "
                    f"for feature '{feature}'")
            kwargs[parameter] = True
    return kwargs


def load_model(spec: CheckpointSpec, device: str) -> torch.nn.Module:
    state = _state_without_module(spec.path, device)
    expected = model_kwargs(spec.features)
    infer_kwargs = getattr(
        dope_models, "refinement_model_kwargs_from_state_dict", None)
    if infer_kwargs is None:
        raise RuntimeError(
            "models.refinement_model_kwargs_from_state_dict is required")
    inferred = {"numVec": 0, **infer_kwargs(state)}
    for key in ("numSeg", "maskBeliefFusion", "cornerQuality"):
        if key == "numSeg":
            matches = int(inferred.get(key, 0)) == int(expected.get(key, 0))
        else:
            matches = bool(inferred.get(key)) == bool(expected.get(key, False))
        if not matches:
            raise RuntimeError(
                f"checkpoint feature mismatch for {spec.name}: "
                f"{key} inferred={inferred.get(key)} expected="
                f"{expected.get(key, False)}")
    model = dope_models.DopeNetwork(**inferred)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch for {spec.name}: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}")
    return model.to(device).eval()


def infer(model: torch.nn.Module, img: np.ndarray, device: str) -> dict[str, Any]:
    with torch.no_grad():
        output = model(preprocess_squash(img).to(device))
    beliefs, _, _, _, aux = unpack_dope_output(output)
    belief = beliefs[-1][0]
    if tuple(belief.shape) != (9, GRID_SIZE, GRID_SIZE):
        raise RuntimeError(f"belief grid drift: {tuple(belief.shape)}")
    belief_np = belief.detach().cpu().numpy()
    # Historical PAPER_S2 parity: main coordinates must stay on the deployed
    # scipy smooth + local NMS + weighted-centroid decoder.
    kps_grid = extract_keypoints_from_belief(belief_np, THRESH)
    height, width = img.shape[:2]
    uncertainty = None
    log_sigma = aux.get("corner_log_sigma")
    if log_sigma is not None:
        if tuple(log_sigma.shape) != (1, 9, GRID_SIZE, GRID_SIZE):
            raise RuntimeError(
                f"bad corner log-sigma shape: {tuple(log_sigma.shape)}")
        # The quality head is supervised at the integer smoothed-NMS peak,
        # not at the rounded weighted-centroid coordinate.  Decode only the
        # auxiliary sigma through the same helper used by the training loss;
        # the canonical scipy decoder above remains the source of keypoints.
        refinement = decode_refinement_outputs(
            belief, aux, threshold=THRESH)
        sigma_grid = refinement["sigma9"].detach().cpu().numpy()
        pixel_scale = math.sqrt(
            ((width / GRID_SIZE) ** 2 + (height / GRID_SIZE) ** 2) / 2.0)
        uncertainty = []
        for channel, point in enumerate(kps_grid):
            if point[0] < 0:
                uncertainty.append(1.0)  # ignored for a missing correspondence
                continue
            uncertainty.append(float(sigma_grid[channel]) * pixel_scale)
    if aux.get("wd_as_given_logit") is not None:
        raise RuntimeError(
            "W/D prior head is disabled for this experiment: its training "
            "label does not match the safe-PnP as-given hypothesis")
    return {
        "kps_grid": kps_grid,
        "uncertainty": uncertainty,
    }


def _hungarian(pred: np.ndarray, gt: np.ndarray,
               min_points: int = N_DET_MIN) -> tuple[np.ndarray | None,
                                                     np.ndarray | None,
                                                     np.ndarray | None]:
    valid_idx = np.flatnonzero(np.isfinite(pred[:, 0]))
    if len(valid_idx) < min_points:
        return None, None, None
    cost = np.linalg.norm(pred[valid_idx, None, :] - gt[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols], valid_idx[rows], cols


def _pose_projection(pose: dict[str, Any], camera: np.ndarray) -> np.ndarray:
    projected = pose.get("projected_all")
    if projected is None:
        projected = APNP.project_with_pose(
            pose["R"], pose["t"], camera, pose["dims"])
    return np.asarray(projected, np.float64)[:8]


def _honest8(pose: dict[str, Any] | None, camera: np.ndarray,
             gt8: np.ndarray) -> float:
    if pose is None:
        return math.inf
    projected = _pose_projection(pose, camera)
    bad = ((projected[:, 0] == -1.0) & (projected[:, 1] == -1.0))
    projected[bad] = np.nan
    distances, _, _ = _hungarian(projected, gt8, min_points=6)
    if distances is None:
        return math.inf
    return float(np.mean(distances))


def evaluate_frame(model: torch.nn.Module, frame: dict[str, Any],
                   device: str) -> dict[str, Any]:
    img = cv2.imread(frame["png"])
    if img is None:
        raise FileNotFoundError(f"failed to read evaluation image: {frame['png']}")
    annotation = _read_json(Path(frame["json"]))
    objects = annotation.get("objects") or []
    if not objects:
        raise ValueError(f"evaluation annotation has no object: {frame['json']}")
    obj = objects[0]
    gt8 = np.asarray(obj["projected_cuboid"], np.float64)[:8]
    camera = camera_from_json(annotation)
    height, width = img.shape[:2]
    in_frame = ((gt8[:, 0] >= 0) & (gt8[:, 0] < width) &
                (gt8[:, 1] >= 0) & (gt8[:, 1] < height))

    prediction = infer(model, img, device)
    kps_grid = prediction["kps_grid"]
    scale_x, scale_y = width / GRID_SIZE, height / GRID_SIZE
    pred8 = np.full((8, 2), np.nan, np.float64)
    for index, point in enumerate(kps_grid[:8]):
        if point[0] >= 0:
            pred8[index] = (float(point[0]) * scale_x,
                            float(point[1]) * scale_y)
    pred_centroid = None
    if kps_grid[8][0] >= 0:
        pred_centroid = [float(kps_grid[8][0]) * scale_x,
                         float(kps_grid[8][1]) * scale_y]
    kps9 = [None if not np.isfinite(pred8[i, 0]) else pred8[i].tolist()
            for i in range(8)] + [pred_centroid]

    raw_metrics = split_metrics(pred8, gt8)
    distances, _, _ = _hungarian(pred8, gt8)
    worst2 = (float(np.mean(np.sort(distances)[-2:]))
              if distances is not None and len(distances) >= 2 else math.inf)

    dims_value = frame.get("dims")
    if dims_value is None:
        dims_dict = frame["entry"]["dims"]
        dims = (float(dims_dict["W"]), float(dims_dict["D"]),
                float(dims_dict["H"]))
    else:
        dims = tuple(map(float, dims_value))

    uncertainty = prediction["uncertainty"]
    legacy_pose = None
    safe_result = {
        "accepted": False,
        "reason": "insufficient_corners",
        "reasons": ["insufficient_corners"],
        "pose": None,
    }
    n_det = int(raw_metrics["n_det"])
    if n_det >= N_DET_MIN:
        legacy_pose = APNP.solve_pose(
            kps9, camera, dims=dims, img_shape=img.shape)
    if n_det >= 7:
        safe_result = APNP.solve_pose_safe(
            kps9, camera, dims=dims, img_shape=img.shape,
            keypoint_uncertainties=uncertainty,
            wd_as_given_prob=None,
            min_corners=7, reject_wd_ambiguity=True)

    legacy_honest = _honest8(legacy_pose, camera, gt8)
    safe_pose = safe_result.get("pose") if safe_result.get("accepted") else None
    safe_honest = _honest8(safe_pose, camera, gt8)
    if legacy_pose is None:
        legacy_projection = np.full((8, 2), np.nan, np.float64)
    else:
        legacy_projection = _pose_projection(legacy_pose, camera)
    raw_coverage = gt_coverage_features(
        pred8, gt8, width, height, source_n_det=n_det)
    coverage = gt_coverage_features(
        legacy_projection, gt8, width, height, source_n_det=n_det)
    pnp_to_raw_area = None
    if (raw_coverage.get("pred_area") is not None
            and coverage.get("pred_area") is not None
            and float(raw_coverage["pred_area"]) > 1.0e-8):
        pnp_to_raw_area = float(
            coverage["pred_area"] / raw_coverage["pred_area"])

    pnp_tz_ratio = None
    transform = obj.get("pose_transform")
    if legacy_pose is not None and transform is not None:
        try:
            gt_tz = float(np.asarray(transform, np.float64).reshape(4, 4)[2, 3])
            pred_tz = float(np.asarray(legacy_pose["t"], np.float64).reshape(-1)[2])
            if np.isfinite(gt_tz) and np.isfinite(pred_tz) and gt_tz > 1.0e-8:
                pnp_tz_ratio = pred_tz / gt_tz
        except (KeyError, TypeError, ValueError, IndexError):
            pass

    uncertainty_error_pairs = []
    if uncertainty is not None:
        # The uncertainty target is channel-ordered and safe PnP also consumes
        # channel-ordered correspondences.  Hungarian/order-free distance is a
        # useful pose metric, but would mis-score calibrated symmetric swaps.
        for index in range(8):
            if np.isfinite(pred8[index]).all():
                ordered_error = np.linalg.norm(pred8[index] - gt8[index])
                uncertainty_error_pairs.append([
                    float(uncertainty[index]), float(ordered_error)])

    elevation = None
    if transform is not None:
        try:
            elevation = elev_from_pose(transform)
        except Exception:
            pass

    pose_diag = safe_result.get("pose") or {}
    return {
        "fid": frame["fid"],
        "split": frame.get("split", "synthetic"),
        "domain": frame.get("domain", "synthetic"),
        "v_geom": int(in_frame.sum()),
        "n_det": n_det,
        "det": int(n_det >= N_DET_MIN),
        "corner": raw_metrics["overall"],
        "front": raw_metrics["front"],
        "back": raw_metrics["back"],
        "worst2": worst2,
        "pnp_ok": int(legacy_pose is not None),
        "pnp_honest8": legacy_honest,
        "safe_accepted": int(bool(safe_result.get("accepted"))),
        "safe_reason": safe_result.get("reason", "unknown"),
        "safe_honest8": safe_honest,
        "uc_eligible": bool(coverage["uc_eligible"]),
        "uc_main": bool(coverage["uc_main"]),
        "uc_gt_cover": coverage["gt_cover"],
        "uc_r_min": coverage["r_min"],
        "raw_uc_main": bool(raw_coverage["uc_main"]),
        "raw_area_ratio": raw_coverage["area_ratio"],
        "raw_r_min": raw_coverage["r_min"],
        "pnp_area_ratio": coverage["area_ratio"],
        "pnp_to_raw_area_ratio": pnp_to_raw_area,
        "pnp_tz_ratio": pnp_tz_ratio,
        "safe_wd_ambiguous": bool(pose_diag.get("_wd_ambiguous", False)),
        "safe_wd_score_gap_px": pose_diag.get("_wd_score_gap_px"),
        "safe_wd_threshold_px": pose_diag.get("_wd_ambiguity_threshold_px"),
        "uncertainty_error_pairs": uncertainty_error_pairs,
        "elev": elevation,
        "pred8": pred8.tolist(),
        "gt8": gt8.tolist(),
    }


def _finite(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values
            if value is not None and np.isfinite(value)]


def _median(values: Iterable[Any]) -> float | None:
    valid = _finite(values)
    return round(float(np.median(valid)), 2) if valid else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    good = gross = ncorner = 0
    for row in rows:
        distances, _, _ = _hungarian(
            np.asarray(row["pred8"], float), np.asarray(row["gt8"], float))
        if distances is None:
            continue
        for distance in distances:
            ncorner += 1
            good += int(distance < GOOD_PX)
            gross += int(distance > GROSS_PX)

    legacy_good = [row for row in rows
                   if row["pnp_ok"] and row["pnp_honest8"] < GOOD_PX]
    legacy_bad = [row for row in rows
                  if row["pnp_ok"] and row["pnp_honest8"] > GROSS_PX]
    safe_good_kept = [row for row in legacy_good
                      if row["safe_accepted"] and
                      row["safe_honest8"] < GOOD_PX]
    safe_bad_rejected = [row for row in legacy_bad if not row["safe_accepted"]]
    accepted = [row for row in rows if row["safe_accepted"]]
    safe_good = [row for row in accepted if row["safe_honest8"] < GOOD_PX]
    safe_gross = [row for row in accepted if row["safe_honest8"] > GROSS_PX]
    uc_eligible = [row for row in rows if row["uc_eligible"]]
    uc_main = [row for row in uc_eligible if row["uc_main"]]
    uncertainty_pairs = [pair for row in rows
                         for pair in row["uncertainty_error_pairs"]]
    uncertainty_corr = None
    if len(uncertainty_pairs) >= 8:
        uncertainty_array = np.asarray(uncertainty_pairs, float)
        result = spearmanr(uncertainty_array[:, 0], uncertainty_array[:, 1])
        if np.isfinite(result.statistic):
            uncertainty_corr = round(float(result.statistic), 3)

    reasons = Counter(row["safe_reason"] for row in rows
                      if not row["safe_accepted"])
    wd_ambiguous = [row for row in rows if row["safe_wd_ambiguous"]]
    return {
        "n": n,
        "det_pct": round(100.0 * np.mean([row["det"] for row in rows]), 1),
        "front_med": _median(row["front"] for row in rows),
        "rear_med": _median(row["back"] for row in rows),
        "corner_med": _median(row["corner"] for row in rows),
        "worst2_med": _median(row["worst2"] for row in rows),
        "good_pct": round(100.0 * good / ncorner, 1) if ncorner else None,
        "gross_pct": round(100.0 * gross / ncorner, 1) if ncorner else None,
        "ncorner": ncorner,
        "legacy_pnp_pct": round(100.0 * np.mean(
            [row["pnp_ok"] for row in rows]), 1),
        "legacy_honest8_med": _median(
            row["pnp_honest8"] for row in rows if row["pnp_ok"]),
        "safe_accept_pct": round(100.0 * len(accepted) / n, 1),
        "safe_honest8_med": _median(row["safe_honest8"] for row in accepted),
        "safe_good_yield_pct": round(100.0 * len(safe_good) / n, 1),
        "safe_gross_accepted_pct": round(100.0 * len(safe_gross) / n, 1),
        "safe_good_retention_pct": (
            round(100.0 * len(safe_good_kept) / len(legacy_good), 1)
            if legacy_good else None),
        "safe_bad_reject_pct": (
            round(100.0 * len(safe_bad_rejected) / len(legacy_bad), 1)
            if legacy_bad else None),
        "legacy_good_n": len(legacy_good),
        "legacy_bad_n": len(legacy_bad),
        "safe_good_n": len(safe_good),
        "safe_gross_n": len(safe_gross),
        "uc_eligible_n": len(uc_eligible),
        "uc_main_n": len(uc_main),
        "uc_main_pct": (
            round(100.0 * len(uc_main) / len(uc_eligible), 1)
            if uc_eligible else None),
        "uc_gt_cover_med": _median(
            row["uc_gt_cover"] for row in uc_eligible),
        "uc_r_min_med": _median(row["uc_r_min"] for row in uc_eligible),
        "raw_uc_main_n": sum(
            int(bool(row.get("raw_uc_main"))) for row in rows),
        "raw_area_ratio_med": _median(
            row.get("raw_area_ratio") for row in rows if row["det"]),
        "raw_r_min_med": _median(
            row.get("raw_r_min") for row in rows if row["det"]),
        "pnp_area_ratio_med": _median(
            row.get("pnp_area_ratio") for row in rows if row["pnp_ok"]),
        "pnp_to_raw_area_ratio_med": _median(
            row.get("pnp_to_raw_area_ratio")
            for row in rows if row["pnp_ok"]),
        "uc_pnp_tz_ratio_med": _median(
            row.get("pnp_tz_ratio") for row in uc_main),
        "non_uc_pnp_tz_ratio_med": _median(
            row.get("pnp_tz_ratio") for row in uc_eligible
            if not row["uc_main"]),
        "safe_reject_reasons": dict(sorted(reasons.items())),
        "wd_ambiguous_n": len(wd_ambiguous),
        "wd_ambiguous_reject_n": reasons.get("wd_ambiguous", 0),
        "wd_score_gap_med": _median(
            row["safe_wd_score_gap_px"] for row in wd_ambiguous),
        "uncertainty_error_spearman": uncertainty_corr,
    }


def _cache_key(spec: CheckpointSpec, set_name: str,
               frames: list[dict[str, Any]]) -> str:
    stat = spec.path.stat()
    dependency_paths = (
        Path(__file__),
        ROOT / "challenge" / "scripts" / "annotate_pnp.py",
        ROOT / "Deep_Object_Pose" / "common" / "models.py",
        ROOT / "Deep_Object_Pose" / "common" / "heatmap_refinement.py",
        ROOT / "scripts" / "data_prep" / "eval" /
        "filter_pr_camfacing.py",
        ROOT / "scripts" / "stage0" / "eval_harness" / "eval_pvnet_heads.py",
        ROOT / "scripts" / "stage0" / "paper_s2" / "paper_s2_mask_coverage_filter.py",
        ROOT / "scripts" / "stage0" / "stage_screens" / "stage18_elevation_threshold.py",
    )
    source_hash = hashlib.sha256()
    for path in dependency_paths:
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    membership = []
    for frame in frames:
        record = {
            "fid": str(frame["fid"]),
            "split": str(frame.get("split", "synthetic")),
            "domain": str(frame.get("domain", "synthetic")),
            "entry": frame.get("entry"),
            "dims": frame.get("dims"),
            "files": [],
        }
        for key in ("json", "png"):
            path = Path(frame[key]).resolve()
            file_stat = path.stat()
            record["files"].append({
                "kind": key,
                "path": str(path),
                "size": file_stat.st_size,
                "mtime_ns": file_stat.st_mtime_ns,
            })
        membership.append(record)
    payload = {
        "path": str(spec.path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "features": spec.features,
        "trainable_scope": spec.trainable_scope,
        "set": set_name,
        "membership": membership,
        "evaluator": 4,
        "source_sha256": source_hash.hexdigest(),
        "safe_pnp_policy": {
            "min_corners": 7,
            "reject_wd_ambiguity": True,
            "wd_as_given_prob": None,
            "uncertainty_grid_sigma_clip": [0.25, 20.0],
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def evaluate_set(spec: CheckpointSpec, set_name: str,
                 frames: list[dict[str, Any]], device: str,
                 cache_dir: Path, use_cache: bool = True) -> dict[str, Any]:
    key = _cache_key(spec, set_name, frames)
    cache = cache_dir / f"{spec.name}__{set_name}__{key}.json"
    if use_cache and cache.is_file():
        print(f"[cache] {spec.name}/{set_name}")
        return _read_json(cache)
    print(f"[eval] {spec.name}/{set_name}: n={len(frames)}")
    model = load_model(spec, device)
    rows = []
    for index, frame in enumerate(frames, 1):
        rows.append(evaluate_frame(model, frame, device))
        if index % 100 == 0:
            print(f"  {index}/{len(frames)}", flush=True)
    if len(rows) != len(frames):
        raise AssertionError(
            f"evaluation row-count drift for {set_name}: "
            f"{len(rows)} != {len(frames)}")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    result = {"checkpoint": str(spec.path), "name": spec.name,
              "arm": spec.arm, "features": list(spec.features),
              "trainable_scope": spec.trainable_scope,
              "set": set_name, "summary": summarize(rows), "rows": rows}
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, indent=2, default=_json_default))
    return result


def _epoch_from_name(path: Path) -> int | None:
    match = re.search(r"_(\d{4})\.pth$", path.name)
    return int(match.group(1)) if match else None


def _validate_training_header(config: dict[str, Any], phase: str,
                              arm: str, directory: Path) -> None:
    """Fail closed when a loss-only arm is stored under the wrong label."""
    path = directory / "header.txt"
    if not path.is_file():
        raise FileNotFoundError(f"training provenance header missing: {path}")
    header = path.read_text()
    locked = config["locked"]
    arm_config = config["arms"][arm]
    features = set(arm_config["features"])
    common_train_flags = config.get("common_train_flags", [])
    if not isinstance(common_train_flags, list):
        raise TypeError("common_train_flags must be a JSON list")
    diffpnp_enabled = "--diffpnp" in common_train_flags
    aspect_resize_enabled = "--aspect_resize" in common_train_flags
    bool_flags = (
        "clip_belief_border",
        "mask_belief_fusion",
        "extent_loss",
        "corner_quality",
        "projected_span_loss",
    )
    expected_tokens = [
        f"imagesize={int(locked['input_size'])}",
        f"sigma={float(locked['sigma'])}",
        f"manualseed={int(locked['seed'])}",
        f"epoch_size={int(locked[f'{phase}_epoch_size'])}",
        f"epochs={int(locked['base_epoch']) + int(locked[f'{phase}_delta_epochs'])}",
        "mask_aux=True",
        f"diffpnp={diffpnp_enabled}",
        "heatmap_pnp_enhance=False",
    ]
    # aspect_resize was added after the oldest experiment headers.  Require
    # the positive flag when configured, and validate False when a newer
    # header exposes the field, without invalidating legacy headers that do
    # not contain it at all.
    if aspect_resize_enabled or "aspect_resize=" in header:
        expected_tokens.append(f"aspect_resize={aspect_resize_enabled}")
    expected_tokens.extend(
        f"{flag}={flag in features}" for flag in bool_flags)
    # Added after the original experiment headers. Require it for new signed
    # arms, and validate it when a modern header exposes the field, without
    # invalidating immutable legacy provenance files.
    if ("signed_footprint_loss" in features
            or "signed_footprint_loss=" in header):
        expected_tokens.append(
            f"signed_footprint_loss="
            f"{'signed_footprint_loss' in features}")
    if "trainable_scope" in arm_config:
        expected_tokens.append(
            f"trainable_scope='{arm_config['trainable_scope']}'")
    expected_tokens.append(str((ROOT / locked["base_checkpoint"]).resolve()))
    expected_tokens.extend(str((ROOT / rel).resolve()) for rel in locked["data"])
    missing = [token for token in expected_tokens if token not in header]
    if missing:
        raise RuntimeError(
            f"training provenance mismatch for {phase}/{arm}: {missing}")
    if "raw_data" in header or re.search(r"(?:^|/)scan(?:/|$)", header):
        raise RuntimeError(f"forbidden data path in training header: {path}")


def _assert_frozen_scope_checkpoint(
        base_state: dict[str, torch.Tensor], checkpoint: Path,
        trainable_scope: str) -> None:
    """Prove a scoped checkpoint changed only its pre-registered branch.

    This runs during checkpoint discovery, before cache lookup, so a stale
    evaluation cache cannot hide an accidentally unfrozen core.
    """
    if trainable_scope == "all":
        return
    if trainable_scope not in ("mask_fusion", "belief_tail"):
        raise ValueError(f"unsupported trainable_scope: {trainable_scope}")
    candidate = _state_without_module(checkpoint, "cpu")
    base_keys = set(base_state)
    candidate_keys = set(candidate)
    missing = sorted(base_keys - candidate_keys)
    extra = sorted(candidate_keys - base_keys)
    expected_extra = (
        [
            "m_mask_belief_fusion.0.bias",
            "m_mask_belief_fusion.0.weight",
            "m_mask_belief_fusion.2.bias",
            "m_mask_belief_fusion.2.weight",
        ]
        if trainable_scope == "mask_fusion" else []
    )
    if missing or extra != expected_extra:
        raise RuntimeError(
            f"frozen checkpoint key drift for {checkpoint}: "
            f"missing={missing}, extra={extra}")
    allowed_changed_prefixes = (
        ("m6_2.10.", "m6_2.12.")
        if trainable_scope == "belief_tail" else ()
    )
    changed_core = [key for key in sorted(base_keys)
                    if not torch.equal(candidate[key], base_state[key])]
    changed_forbidden = [
        key for key in changed_core
        if not key.startswith(allowed_changed_prefixes)
    ]
    if changed_forbidden:
        raise RuntimeError(
            f"frozen core changed in {checkpoint}: {changed_forbidden[:20]}")
    if trainable_scope == "belief_tail" and not changed_core:
        raise RuntimeError(f"belief-tail checkpoint did not update: {checkpoint}")
    trainable_keys = expected_extra + changed_core
    bad_trainable = [
        key for key in trainable_keys
        if candidate[key].is_floating_point()
        and not torch.isfinite(candidate[key]).all()
    ]
    if bad_trainable:
        raise RuntimeError(
            f"non-finite scoped tensors in {checkpoint}: {bad_trainable}")


def discover_checkpoints(config: dict[str, Any], weights_root: Path,
                         arms: list[str], baseline_only: bool,
                         phase: str) -> list[CheckpointSpec]:
    locked = config["locked"]
    baseline_path = ROOT / locked["base_checkpoint"]
    specs = [CheckpointSpec(
        name="baseline_ep57", arm="baseline", path=baseline_path,
        epoch=int(locked["base_epoch"]), features=(), baseline=True)]
    if baseline_only:
        return specs
    frozen_base_state: dict[str, torch.Tensor] | None = None
    for arm in arms:
        features = tuple(config["arms"][arm]["features"])
        trainable_scope = config["arms"][arm].get("trainable_scope", "all")
        directory = weights_root / arm
        if not directory.is_dir():
            raise FileNotFoundError(f"arm weights directory missing: {directory}")
        _validate_training_header(config, phase, arm, directory)
        by_epoch: dict[int, Path] = {}
        # Prefer explicit per-epoch snapshots over the duplicate final snapshot.
        for path in sorted(directory.glob("net_*_*.pth")):
            if path.name.startswith("final_"):
                continue
            epoch = _epoch_from_name(path)
            if epoch is not None:
                by_epoch[epoch] = path
        for path in sorted(directory.glob("final_net_*_*.pth")):
            epoch = _epoch_from_name(path)
            if epoch is not None and epoch not in by_epoch:
                by_epoch[epoch] = path
        if not by_epoch:
            raise FileNotFoundError(f"no checkpoints found for arm '{arm}': {directory}")
        delta = int(locked[f"{phase}_delta_epochs"])
        expected_epochs = set(range(
            int(locked["base_epoch"]) + 1,
            int(locked["base_epoch"]) + delta + 1,
        ))
        found_epochs = set(by_epoch)
        if found_epochs != expected_epochs:
            raise RuntimeError(
                f"checkpoint epoch drift for {phase}/{arm}: "
                f"missing={sorted(expected_epochs - found_epochs)} "
                f"extra={sorted(found_epochs - expected_epochs)}")
        for epoch, path in sorted(by_epoch.items()):
            if trainable_scope != "all":
                if frozen_base_state is None:
                    frozen_base_state = _state_without_module(
                        baseline_path, "cpu")
                _assert_frozen_scope_checkpoint(
                    frozen_base_state, path, trainable_scope)
            specs.append(CheckpointSpec(
                name=f"{arm}_ep{epoch:04d}", arm=arm, path=path,
                epoch=epoch, features=features,
                trainable_scope=trainable_scope))
    return specs


def _metric(summary: dict[str, Any], key: str, default: float) -> float:
    value = summary.get(key)
    return float(value) if value is not None and np.isfinite(value) else default


def synthetic_guard(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Historical Stage-B-style guard, relative to the unchanged ep57 baseline."""
    return bool(
        _metric(candidate, "front_med", 1e9) <=
        _metric(baseline, "front_med", 1e9) + 1.5 and
        _metric(candidate, "det_pct", -1e9) >=
        _metric(baseline, "det_pct", -1e9) - 5.0 and
        _metric(candidate, "good_pct", -1e9) >=
        _metric(baseline, "good_pct", -1e9) - 8.0 and
        _metric(candidate, "gross_pct", 1e9) <=
        _metric(baseline, "gross_pct", 1e9) + 5.0 and
        _metric(candidate, "legacy_pnp_pct", -1e9) >=
        _metric(baseline, "legacy_pnp_pct", -1e9) - 3.0 and
        _metric(candidate, "safe_accept_pct", -1e9) >=
        _metric(baseline, "safe_accept_pct", -1e9) - 3.0 and
        _metric(candidate, "safe_good_yield_pct", -1e9) >=
        _metric(baseline, "safe_good_yield_pct", -1e9) - 3.0 and
        _metric(candidate, "safe_good_retention_pct", -1e9) >=
        _metric(baseline, "safe_good_retention_pct", -1e9) - 5.0 and
        _metric(candidate, "safe_gross_accepted_pct", 1e9) <=
        _metric(baseline, "safe_gross_accepted_pct", 1e9) + 2.0 and
        _metric(candidate, "uc_eligible_n", -1e9) >=
        _metric(baseline, "uc_eligible_n", -1e9) - 5.0 and
        _metric(candidate, "uc_main_pct", 1e9) <=
        _metric(baseline, "uc_main_pct", 1e9) + 3.0)


def filterval_guard(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        _metric(candidate, "front_med", 1e9) <=
        _metric(baseline, "front_med", 1e9) + 2.0 and
        _metric(candidate, "det_pct", -1e9) >=
        _metric(baseline, "det_pct", -1e9) - 5.0 and
        _metric(candidate, "good_pct", -1e9) >=
        _metric(baseline, "good_pct", -1e9) - 8.0 and
        _metric(candidate, "gross_pct", 1e9) <=
        _metric(baseline, "gross_pct", 1e9) + 5.0 and
        _metric(candidate, "legacy_pnp_pct", -1e9) >=
        _metric(baseline, "legacy_pnp_pct", -1e9) - 3.0 and
        _metric(candidate, "safe_accept_pct", -1e9) >=
        _metric(baseline, "safe_accept_pct", -1e9) - 3.0 and
        _metric(candidate, "safe_good_yield_pct", -1e9) >=
        _metric(baseline, "safe_good_yield_pct", -1e9) - 3.0 and
        _metric(candidate, "safe_good_retention_pct", -1e9) >=
        _metric(baseline, "safe_good_retention_pct", -1e9) - 5.0 and
        _metric(candidate, "safe_gross_accepted_pct", 1e9) <=
        _metric(baseline, "safe_gross_accepted_pct", 1e9) + 2.0 and
        _metric(candidate, "uc_eligible_n", -1e9) >=
        _metric(baseline, "uc_eligible_n", -1e9) - 5.0 and
        _metric(candidate, "uc_main_pct", 1e9) <=
        _metric(baseline, "uc_main_pct", 1e9) + 3.0)


def _rank_sum(records: list[dict[str, Any]], keys: list[str],
              maximize: Iterable[str] = ()) -> dict[str, int]:
    result = {record["name"]: 0 for record in records}
    maximize = set(maximize)
    for key in keys:
        reverse = key in maximize
        default = -1e9 if reverse else 1e9
        ordered = sorted(records, key=lambda record: _metric(
            record["summary"], key, default), reverse=reverse)
        for rank, record in enumerate(ordered):
            result[record["name"]] += rank
    return result


def select_per_arm(specs: list[CheckpointSpec],
                   synth_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = synth_results["baseline_ep57"]["summary"]
    selected = {}
    for arm in sorted({spec.arm for spec in specs if not spec.baseline}):
        records = [synth_results[spec.name] for spec in specs if spec.arm == arm]
        guarded = [record for record in records
                   if synthetic_guard(record["summary"], baseline)]
        pool = guarded or records
        ranks = _rank_sum(
            pool,
            ["rear_med", "legacy_honest8_med", "safe_honest8_med",
             "uc_main_pct", "safe_gross_accepted_pct",
             "safe_accept_pct", "safe_good_yield_pct"],
            maximize=("safe_accept_pct", "safe_good_yield_pct"),
        )
        best = min(pool, key=lambda record: (
            ranks[record["name"]],
            _metric(record["summary"], "corner_med", 1e9)))
        selected[arm] = {
            "name": best["name"],
            "checkpoint": best["checkpoint"],
            "synthetic_guard": bool(best in guarded),
            "guard_pool_empty": not bool(guarded),
        }
    return selected


def choose_arm(selected: dict[str, Any], synth_results: dict[str, dict[str, Any]],
               real_results: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    synth_base = synth_results["baseline_ep57"]["summary"]
    real_base = real_results["baseline_ep57"]["filterval"]["summary"]
    # Keep ep57 in the actual rank pool.  Passing a non-regression guard is not
    # by itself evidence that another training arm is better than the baseline.
    eligible = [{
        "name": "baseline_ep57",
        "arm": "baseline",
        "summary": synth_base,
        "real_summary": real_base,
    }]
    rejected = {}
    for arm, choice in selected.items():
        name = choice["name"]
        synth = synth_results[name]["summary"]
        real = real_results[name]["filterval"]["summary"]
        reasons = []
        if not synthetic_guard(synth, synth_base):
            reasons.append("synthetic_guard")
        if not filterval_guard(real, real_base):
            reasons.append("filterval_guard")
        if reasons:
            rejected[arm] = reasons
        else:
            eligible.append({"name": name, "arm": arm, "summary": synth,
                             "real_summary": real})
    ranks = {record["name"]: 0 for record in eligible}
    rank_specs = (
        ("summary", "rear_med", False),
        ("summary", "legacy_honest8_med", False),
        ("summary", "safe_honest8_med", False),
        ("summary", "uc_main_pct", False),
        ("summary", "safe_gross_accepted_pct", False),
        ("summary", "safe_accept_pct", True),
        ("summary", "safe_good_yield_pct", True),
        ("real_summary", "rear_med", False),
        ("real_summary", "legacy_honest8_med", False),
        ("real_summary", "safe_honest8_med", False),
        ("real_summary", "uc_main_pct", False),
        ("real_summary", "safe_gross_accepted_pct", False),
        ("real_summary", "safe_accept_pct", True),
        ("real_summary", "safe_good_yield_pct", True),
    )
    for source, key, maximize in rank_specs:
        default = -1e9 if maximize else 1e9
        ordered = sorted(
            eligible,
            key=lambda record: _metric(record[source], key, default),
            reverse=maximize,
        )
        for rank, record in enumerate(ordered):
            ranks[record["name"]] += rank
    best = min(eligible, key=lambda record: (
        ranks[record["name"]],
        _metric(record["real_summary"], "corner_med", 1e9)))
    reason = (
        "baseline retained after synthetic+filterval rank comparison"
        if best["arm"] == "baseline" else
        "lowest guarded synthetic+filterval accuracy/under-coverage/safe-yield "
        "rank-sum among baseline and training arms"
    )
    return {
        "best_arm": best["arm"],
        "best_checkpoint": synth_results[best["name"]]["checkpoint"],
        "best_name": best["name"],
        "reason": reason,
        "rank_sums": ranks,
        "rejected": rejected,
    }


def choose_frozen_fusion(
        specs: list[CheckpointSpec],
        synth_results: dict[str, dict[str, Any]],
        real_results: dict[str, dict[str, dict[str, Any]]],
        config: dict[str, Any]) -> dict[str, Any]:
    """Select a short scoped checkpoint only when coverage is preserved.

    Every ep58--60 checkpoint is evaluated on filterval.  Counts, rather than
    rounded percentages, enforce the pre-registered anti-forgetting limits.
    handannot17 is intentionally absent from this decision.
    """
    policy = config["evaluation"]["preservation"]
    baseline_spec = next(spec for spec in specs if spec.baseline)
    baseline_name = baseline_spec.name
    synth_base = synth_results[baseline_name]["summary"]

    def counts(name: str) -> dict[str, int]:
        record = real_results[name]["filterval"]
        rows = record["rows"]
        summary = record["summary"]
        return {
            "detection": sum(int(bool(row["det"])) for row in rows),
            "legacy_pnp": sum(int(bool(row["pnp_ok"])) for row in rows),
            "safe_accept": sum(
                int(bool(row["safe_accepted"])) for row in rows),
            "safe_good": int(summary["safe_good_n"]),
            "safe_gross": int(summary["safe_gross_n"]),
            "uc_eligible": int(summary["uc_eligible_n"]),
            "uc_main": int(summary["uc_main_n"]),
        }

    base_counts = counts(baseline_name)
    eligible: list[tuple[CheckpointSpec, dict[str, int]]] = []
    rejected: dict[str, list[str]] = {}
    all_counts = {baseline_name: base_counts}
    for spec in specs:
        if spec.baseline:
            continue
        candidate_counts = counts(spec.name)
        all_counts[spec.name] = candidate_counts
        synth = synth_results[spec.name]["summary"]
        real = real_results[spec.name]["filterval"]["summary"]
        reasons = []
        if not synthetic_guard(synth, synth_base):
            reasons.append("synthetic_guard")
        if not filterval_guard(real, real_results[baseline_name]["filterval"]["summary"]):
            reasons.append("filterval_guard")
        limits = (
            ("detection", "max_detection_drop_frames", "min"),
            ("legacy_pnp", "max_legacy_pnp_drop_frames", "min"),
            ("safe_accept", "max_safe_accept_drop_frames", "min"),
            ("uc_eligible", "max_uc_eligible_drop_frames", "min"),
            ("safe_gross", "max_safe_gross_increase_frames", "max"),
        )
        for metric, setting, direction in limits:
            allowance = int(policy[setting])
            if direction == "min":
                threshold = base_counts[metric] - allowance
                if candidate_counts[metric] < threshold:
                    reasons.append(
                        f"{metric}={candidate_counts[metric]}<{threshold}")
            else:
                threshold = base_counts[metric] + allowance
                if candidate_counts[metric] > threshold:
                    reasons.append(
                        f"{metric}={candidate_counts[metric]}>{threshold}")
        min_safe_good = (
            base_counts["safe_good"]
            + int(policy["min_safe_good_delta_frames"]))
        if candidate_counts["safe_good"] < min_safe_good:
            reasons.append(
                f"safe_good={candidate_counts['safe_good']}<{min_safe_good}")
        max_uc_main = (
            base_counts["uc_main"]
            - int(policy["min_uc_improvement_frames"]))
        if candidate_counts["uc_main"] > max_uc_main:
            reasons.append(
                f"uc_main={candidate_counts['uc_main']}>{max_uc_main}")
        if reasons:
            rejected[spec.name] = reasons
        else:
            eligible.append((spec, candidate_counts))

    if not eligible:
        return {
            "best_arm": "baseline",
            "best_checkpoint": str(baseline_spec.path),
            "best_name": baseline_name,
            "reason": "baseline retained: no scoped checkpoint met the "
                      "pre-registered count-preservation and UC-improvement limits",
            "rank_sums": {},
            "rejected": rejected,
            "filterval_counts": all_counts,
            "preservation_policy": policy,
        }

    best_spec, _ = min(
        eligible,
        key=lambda item: (
            item[1]["uc_main"],
            item[1]["safe_gross"],
            -item[1]["safe_good"],
            _metric(
                real_results[item[0].name]["filterval"]["summary"],
                "safe_honest8_med", 1e9),
            item[0].epoch,
        ),
    )
    return {
        "best_arm": best_spec.arm,
        "best_checkpoint": str(best_spec.path),
        "best_name": best_spec.name,
        "reason": "scoped checkpoint met all pre-registered preservation "
                  "limits and had the best UC/gross/good-yield ordering",
        "rank_sums": {},
        "rejected": rejected,
        "filterval_counts": all_counts,
        "preservation_policy": policy,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def report_markdown(result: dict[str, Any]) -> str:
    evaluated_membership = result.get("evaluated_real_membership", {})
    evaluated_counts = evaluated_membership.get("counts", {})
    by_split = evaluated_counts.get("by_split", {})
    filterval_n = by_split.get("filterval", 123)
    handannot_n = by_split.get("handannot17", 17)
    canonical_membership = result.get("canonical_real_membership", {})
    exclusions = result.get("training_exclusions", [])
    lines = [
        f"# {result['experiment']} — {result['phase']} evaluation",
        "",
        "Protocol locks: RGB single frame, input 400, belief 50, init ep57, "
        "configured training provenance verified.",
        f"Selection: synthetic + filterval{filterval_n}. "
        f"handannot17 (n={handannot_n}) is report-only. "
        "Sealed final-test sessions were not enumerated.",
        "Real membership: "
        f"canonical n={canonical_membership.get('counts', {}).get('total', '-')}, "
        f"evaluated n={evaluated_counts.get('total', '-')}, "
        f"training exclusions={len(exclusions)}.",
        "",
        "## Synthetic checkpoints",
        "",
        "| checkpoint | arm | det% | front | rear | corner | legacy h8 | "
        "safe h8 | UC% | safe yield% | safe accept% | WD ambiguous rejects |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in result["synthetic_results"].items():
        summary = record["summary"]
        lines.append(
            f"| {name} | {record['arm']} | {_fmt(summary.get('det_pct'))} | "
            f"{_fmt(summary.get('front_med'))} | {_fmt(summary.get('rear_med'))} | "
            f"{_fmt(summary.get('corner_med'))} | "
            f"{_fmt(summary.get('legacy_honest8_med'))} | "
            f"{_fmt(summary.get('safe_honest8_med'))} | "
            f"{_fmt(summary.get('uc_main_pct'))} | "
            f"{_fmt(summary.get('safe_good_yield_pct'))} | "
            f"{_fmt(summary.get('safe_accept_pct'))} | "
            f"{_fmt(summary.get('wd_ambiguous_reject_n'))} |")

    lines += ["", "## Per-arm synthetic selection", ""]
    for arm, choice in result["per_arm_selection"].items():
        lines.append(
            f"- `{arm}`: `{choice['name']}`; synthetic_guard="
            f"{choice['synthetic_guard']}")

    lines += ["", "## Real development and secondary check", ""]
    for name, sets in result["real_results"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            "| set | det% | front | rear | corner | legacy h8 | safe h8 | "
            "UC n/% | safe yield% | safe accept% | good retention% | "
            "bad reject% | sigma/error rho | "
            "WD ambiguous rejects |")
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|")
        for set_name in ("filterval", "handannot17"):
            summary = sets[set_name]["summary"]
            lines.append(
                f"| {set_name} | {_fmt(summary.get('det_pct'))} | "
                f"{_fmt(summary.get('front_med'))} | {_fmt(summary.get('rear_med'))} | "
                f"{_fmt(summary.get('corner_med'))} | "
                f"{_fmt(summary.get('legacy_honest8_med'))} | "
                f"{_fmt(summary.get('safe_honest8_med'))} | "
                f"{_fmt(summary.get('uc_main_n'))}/"
                f"{_fmt(summary.get('uc_main_pct'))} | "
                f"{_fmt(summary.get('safe_good_yield_pct'))} | "
                f"{_fmt(summary.get('safe_accept_pct'))} | "
                f"{_fmt(summary.get('safe_good_retention_pct'))} | "
                f"{_fmt(summary.get('safe_bad_reject_pct'))} | "
                f"{_fmt(summary.get('uncertainty_error_spearman'))} | "
                f"{_fmt(summary.get('wd_ambiguous_reject_n'))} |")
        lines.append("")

    decision = result["decision"]
    lines += [
        "## Decision",
        "",
        f"- BEST_ARM=`{decision['best_arm']}`",
        f"- BEST_CKPT=`{decision['best_checkpoint']}`",
        f"- reason: {decision['reason']}",
        f"- rank_sums: `{json.dumps(decision.get('rank_sums', {}), sort_keys=True)}`",
        f"- rejected: `{json.dumps(decision.get('rejected', {}), sort_keys=True)}`",
        "",
        "W/D learned prior: disabled; ambiguous candidates are rejected.",
        "",
        "handannot17 did not participate in this decision.",
    ]
    if "preservation_policy" in decision:
        lines += [
            "",
        "Scoped-training preservation policy: "
            f"`{json.dumps(decision['preservation_policy'], sort_keys=True)}`",
            "Filterval exact counts: "
            f"`{json.dumps(decision['filterval_counts'], sort_keys=True)}`",
        ]
    if result.get("smoke_limit"):
        lines += ["", "**SMOKE ONLY:** frame limit was active; this decision is invalid."]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("quick", "full"), required=True)
    parser.add_argument("--synthetic", choices=("q1_500", "val_1500"), required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", required=True, help="Comma-separated arm names")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="Smoke only: cap each set; selection is marked invalid")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Smoke the ep57 evaluator without trained arm dirs")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_json(args.config)
    locked = config["locked"]
    if (int(locked["input_size"]) != INPUT_SIZE or
            int(locked["belief_size"]) != GRID_SIZE or
            int(locked["base_epoch"]) != 57):
        raise AssertionError("config violated 400/50/ep57 locks")
    expected_synthetic = config["evaluation"][f"{args.phase}_synthetic"]
    if args.synthetic != expected_synthetic:
        raise AssertionError(
            f"phase/synthetic mismatch: {args.phase} requires "
            f"{expected_synthetic}, got {args.synthetic}")

    arms = [value for value in args.arms.split(",") if value]
    unknown = sorted(set(arms) - set(config["arms"]))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    sealed = config["evaluation"]["sealed_session_names"]
    if "final_test" not in config["evaluation"]["selection_excludes"]:
        raise AssertionError("final_test must be excluded from selection")
    if "handannot17" not in config["evaluation"]["selection_excludes"]:
        raise AssertionError("handannot17 must be report-only")

    synthetic = build_synthetic(args.synthetic)
    canonical_real = build_real()
    canonical_membership_digest = _real_membership_digest(canonical_real)
    expected_digest = config["evaluation"].get("membership_sha256")
    if canonical_membership_digest != expected_digest:
        raise AssertionError(
            "real evaluation membership drift: "
            f"{canonical_membership_digest} != {expected_digest}")
    canonical_membership_counts = _real_membership_counts(canonical_real)
    all_paths = [Path(frame[key]) for frame in synthetic
                 for key in ("json", "png")]
    all_paths += [Path(frame[key]) for values in canonical_real.values()
                  for frame in values for key in ("json", "png")]
    _assert_no_sealed_paths(all_paths, sealed)

    # Canonical count/digest and final-test sealing are verified before any
    # training-overlap exclusion can reduce the real evaluation membership.
    real, training_exclusions = _exclude_training_fids(
        canonical_real,
        config["evaluation"].get("training_exclude_fids"),
    )
    evaluated_membership_counts = _real_membership_counts(real)
    evaluated_membership_digest = _real_membership_digest(real)
    expected_evaluated_digest = config["evaluation"].get(
        "evaluated_membership_sha256")
    if (expected_evaluated_digest is not None
            and evaluated_membership_digest != expected_evaluated_digest):
        raise AssertionError(
            "post-exclusion real evaluation membership drift: "
            f"{evaluated_membership_digest} != {expected_evaluated_digest}")

    if args.limit:
        synthetic = synthetic[:args.limit]
        real = {name: frames[:args.limit] for name, frames in real.items()}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    specs = discover_checkpoints(
        config, args.weights_root, arms, args.baseline_only, args.phase)
    cache_dir = args.out_dir / "cache"
    use_cache = not args.no_cache

    synth_results = {}
    for spec in specs:
        evaluated = evaluate_set(
            spec, args.synthetic, synthetic, device, cache_dir, use_cache)
        synth_results[spec.name] = evaluated

    if args.baseline_only:
        per_arm = {}
        chosen_specs = [specs[0]]
    else:
        per_arm = select_per_arm(specs, synth_results)
        by_name = {spec.name: spec for spec in specs}
        if config["evaluation"].get("evaluate_all_real_checkpoints", False):
            chosen_specs = specs
        else:
            chosen_specs = [specs[0]] + [by_name[value["name"]]
                                         for value in per_arm.values()]

    real_results = {}
    for spec in chosen_specs:
        real_results[spec.name] = {}
        for set_name, frames in real.items():
            real_results[spec.name][set_name] = evaluate_set(
                spec, set_name, frames, device, cache_dir, use_cache)

    if args.baseline_only:
        decision = {
            "best_arm": "baseline",
            "best_checkpoint": str(specs[0].path),
            "reason": "baseline-only smoke",
        }
    elif config["evaluation"].get("selection_policy") in (
            "frozen_fusion_preserve", "scoped_span_preserve"):
        decision = choose_frozen_fusion(
            specs, synth_results, real_results, config)
    else:
        decision = choose_arm(per_arm, synth_results, real_results)
    if args.limit:
        decision["valid"] = False
        decision["invalid_reason"] = "--limit smoke run"
    else:
        decision["valid"] = True

    result = {
        "experiment": config["experiment"],
        "phase": args.phase,
        "synthetic_set": args.synthetic,
        "device": device,
        "smoke_limit": args.limit or None,
        "selection_sets": ["synthetic", "filterval"],
        "report_only_sets": ["handannot17"],
        "safe_pnp_policy": {
            "min_corners": 7,
            "corner_uncertainty": "predicted_sigma_px_when_available",
            "wd_learned_prior": "disabled_invalid_label_semantics",
            "wd_ambiguity": "reject",
        },
        "sealed_final_test_asserted_absent": True,
        # Backward-compatible alias: this remains the frozen canonical digest.
        "real_membership_sha256": canonical_membership_digest,
        "canonical_real_membership": {
            "counts": canonical_membership_counts,
            "sha256": canonical_membership_digest,
        },
        "evaluated_real_membership": {
            "counts": evaluated_membership_counts,
            "sha256": evaluated_membership_digest,
        },
        "training_exclusions": training_exclusions,
        "specs": [{"name": spec.name, "arm": spec.arm,
                   "checkpoint": str(spec.path), "epoch": spec.epoch,
                   "features": list(spec.features),
                   "trainable_scope": spec.trainable_scope}
                  for spec in specs],
        "synthetic_results": synth_results,
        "per_arm_selection": per_arm,
        "real_results": real_results,
        "decision": decision,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, default=_json_default))
    (args.out_dir / "REPORT.md").write_text(report_markdown(result))
    print(f"[save] {args.out_dir / 'results.json'}")
    print(f"[save] {args.out_dir / 'REPORT.md'}")
    print(f"BEST_ARM={decision['best_arm']}")
    print(f"BEST_CKPT={decision['best_checkpoint']}")


if __name__ == "__main__":
    main()
