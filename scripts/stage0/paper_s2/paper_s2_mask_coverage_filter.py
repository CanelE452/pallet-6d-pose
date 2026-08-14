"""Validate a mask-coverage filter for PAPER_S2 without retraining.

The target failure is *under-coverage*: raw keypoints form a plausible cuboid
that occupies only a smaller region inside the real pallet.  Corner belief
peaks alone cannot identify this failure, so this script compares the Stage B
mask-aux output with the raw-keypoint footprint.

Protocol
--------
* model: weights/paper_s2_stageB/net_epoch_0057.pth, loaded with numSeg=1
* input/decode: 400x400 squash parity, belief/seg output at 50x50
* calibration: filterval 123 (outside44 + night43 + manual36)
* holdout report: designated cad11 + noapril6 manifest
* no model training and no final-test sessions

The deployed score never uses GT.  GT is used only to build an evaluation
label that distinguishes an inner/smaller prediction from a shifted or large
wrong prediction.  Results are written below the current model result folder:

  data/pallet/eval_results/paper_s2_scratch_diffpnp/mask_coverage_filter/

Usage:
  python scripts/stage0/paper_s2/paper_s2_mask_coverage_filter.py
  python scripts/stage0/paper_s2/paper_s2_mask_coverage_filter.py --limit 4 --no-overlays
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/home/minjae/Documents/github/pallet-pose")
sys.path.insert(0, str(ROOT / "Deep_Object_Pose" / "common"))
sys.path.insert(0, str(ROOT / "scripts" / "data_prep" / "eval"))
sys.path.insert(0, str(ROOT / "challenge" / "scripts"))
sys.path[:0] = [str(ROOT / "challenge" / "scripts" / _s)
                for _s in ("annotate", "infer", "live")]
sys.path.insert(0, str(ROOT / "scripts" / "stage0"))

from models import DopeNetwork  # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
import annotate_pnp as APNP  # noqa: E402


WEIGHTS = ROOT / "weights" / "paper_s2_stageB" / "net_epoch_0057.pth"
BASE_RESULT = ROOT / "data" / "pallet" / "eval_results" / "paper_s2_scratch_diffpnp"
ORTHO_JSON = BASE_RESULT / "orthogonal_filters_exp.json"
MANIFEST = (ROOT / "data" / "pallet" / "eval_results" /
            "stage22_myannot_eval" / "testset_full8_manifest.txt")
OUT_DIR = BASE_RESULT / "mask_coverage_filter"
OVERLAY_DIR = OUT_DIR / "overlays"

MANUAL_GT_CANDIDATES = [
    ROOT / "data" / "pallet" / "eval_results" / "stage0_gt_candidates" / "manual_gt",
    ROOT / "data" / "pallet" / "eval_results" / "achieve" /
    "paper_base_v2_s2" / "stage0_gt_candidates" / "manual_gt",
]

THRESH = 0.3
GOOD_PX = 10.0
N_DET_MIN = 6
GRID_SIZE = 50
PALLET_DIMS = (1.1, 1.3, 0.12)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Relative-to-frame mask thresholds.  Low values preserve the weak mask tail
# that extended across the real pallet in the original half=whole example.
REL_THRESHOLDS = (0.02, 0.05, 0.10, 0.20)
ROI_EXPAND = 0.75
QUANT_CELL_TOL = 1
MIN_CONNECTED_SUPPORT_PIXELS = 8
MIN_PERSISTENT_EXTENSION = 0.15
MIN_PERSISTENT_LEVELS = 2
# Relative thresholds are meaningless when the head is essentially all-zero.
# A 0.1 probability range is the minimum evidence required to let the mask
# veto a frame; otherwise coverage is UNAVAILABLE rather than FAIL.
MIN_MASK_CONTRAST = 0.10
INITIAL_SCORE_TAU = 0.05
MIN_TARGET_PRECISION = 0.50

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _json_number(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    raise TypeError(type(v).__name__)


def _existing_manual_dir() -> Path:
    for p in MANUAL_GT_CANDIDATES:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "manual_gt not found; checked: " + ", ".join(map(str, MANUAL_GT_CANDIDATES)))


def _status_from_existing_overlay(domain: str, fid: str) -> str | None:
    root = BASE_RESULT / "filterval_passfail" / domain
    for status in ("pass", "fail", "underdet"):
        if (root / status / f"{fid}.jpg").is_file():
            return status
    return None


def collect_frames() -> list[dict]:
    """Use exact prior filterval membership plus the active 17-frame manifest.

    This intentionally does not import stage25_paperbase_eval: archive cleanup
    moved two paths that its module-level imports still reference.
    """
    prior = json.loads(ORTHO_JSON.read_text())
    manual_dir = _existing_manual_dir()
    frames: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for old in prior:
        dom, fid = str(old["dom"]), str(old["fid"])
        if dom == "manual":
            base = manual_dir
        elif dom in ("outside", "night"):
            base = ROOT / "data" / "_eval_sets" / f"{dom}_combined"
        else:
            raise ValueError(f"unexpected filterval domain: {dom}")
        ip, jp = base / f"{fid}.png", base / f"{fid}.json"
        if not ip.is_file() or not jp.is_file():
            raise FileNotFoundError(f"missing filterval pair: {ip} / {jp}")
        key = (dom, fid)
        if key in seen:
            raise ValueError(f"duplicate frame: {key}")
        seen.add(key)
        frames.append({
            "split": "filterval", "domain": dom, "fid": fid,
            "ip": str(ip), "jp": str(jp),
            "prior_n_det": int(old["n_det"]),
            "prior_corner_med": (float(old["corner_med"])
                                 if old.get("corner_med") is not None else None),
            "prior_good": bool(old["good"]),
            "deploy_pass": bool(old["deploy_pass"]),
            "f4_tight_pass": bool(old["deploy_pass"] and
                                  old["f_scores"]["f4_tta_stab"] <= 0.8),
            "existing_status": _status_from_existing_overlay(dom, fid),
        })

    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        dom, fid, jrel, irel = line.split()
        ip, jp = ROOT / irel, ROOT / jrel
        if not ip.is_file() or not jp.is_file():
            raise FileNotFoundError(f"missing holdout pair: {ip} / {jp}")
        key = (dom, fid)
        if key in seen:
            raise ValueError(f"duplicate frame: {key}")
        seen.add(key)
        frames.append({
            "split": "handannot17", "domain": dom, "fid": fid,
            "ip": str(ip), "jp": str(jp),
            "prior_n_det": None, "prior_corner_med": None,
            "prior_good": None, "deploy_pass": None,
            "f4_tight_pass": None,
            "existing_status": _status_from_existing_overlay(dom, fid),
        })

    counts = Counter((x["split"], x["domain"]) for x in frames)
    expected = {
        ("filterval", "outside"): 44,
        ("filterval", "night"): 43,
        ("filterval", "manual"): 36,
        ("handannot17", "cad"): 11,
        ("handannot17", "noapril"): 6,
    }
    if counts != expected or len(frames) != 140:
        raise AssertionError(f"dataset membership drift: {counts}, N={len(frames)}")
    return frames


def load_model(device: str) -> DopeNetwork:
    state = torch.load(WEIGHTS, map_location=device)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    seg_keys = [k for k in state if k.startswith("m_seg")]
    if not seg_keys:
        raise RuntimeError("Stage B checkpoint has no mask-aux head")
    model = DopeNetwork(numVec=0, numSeg=1)
    missing, unexpected = model.load_state_dict(state, strict=False)
    seg_missing = [k for k in missing if k.startswith("m_seg")]
    if seg_missing:
        raise RuntimeError(f"mask head did not load: {seg_missing[:3]}")
    if unexpected:
        print(f"[warn] unexpected checkpoint keys: {unexpected[:5]}")
    return model.to(device).eval()


def preprocess_squash(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (400, 400), interpolation=cv2.INTER_LINEAR)
    x = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)


def read_annotation(jp: str) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(Path(jp).read_text())
    objects = data.get("objects") or []
    if not objects:
        raise ValueError(f"no objects in {jp}")
    gt8 = np.asarray(objects[0]["projected_cuboid"], np.float32)[:8]
    if gt8.shape != (8, 2) or not np.all(np.isfinite(gt8)):
        raise ValueError(f"bad projected_cuboid in {jp}")
    intr = data["camera_data"]["intrinsics"]
    camera = np.array([
        [intr["fx"], 0.0, intr["cx"]],
        [0.0, intr["fy"], intr["cy"]],
        [0.0, 0.0, 1.0],
    ], np.float64)
    return gt8, camera


def infer(model: DopeNetwork, img: np.ndarray, device: str):
    with torch.no_grad():
        out = model(preprocess_squash(img).to(device))
    beliefs, segs = out[0], out[3]
    belief = beliefs[-1][0].detach().cpu().numpy()
    seg_logit = segs[-1][0, 0].detach().cpu().numpy()
    seg_prob = 1.0 / (1.0 + np.exp(-np.clip(seg_logit, -30.0, 30.0)))
    if belief.shape != (9, GRID_SIZE, GRID_SIZE):
        raise RuntimeError(f"unexpected belief shape: {belief.shape}")
    if seg_prob.shape != (GRID_SIZE, GRID_SIZE):
        raise RuntimeError(f"unexpected mask shape: {seg_prob.shape}")
    kps = extract_keypoints_from_belief(belief, THRESH)

    gh, gw = belief.shape[1:]
    grid8 = np.full((8, 2), np.nan, np.float32)
    for i, k in enumerate(kps[:8]):
        if k[0] >= 0:
            grid8[i] = (float(k[0]), float(k[1]))
    grid_center = None
    if kps[8][0] >= 0:
        grid_center = np.array([float(kps[8][0]), float(kps[8][1])], np.float32)
    return belief, seg_prob.astype(np.float32), grid8, grid_center


def grid_to_image(grid8: np.ndarray, width: int, height: int,
                  grid_width: int, grid_height: int) -> np.ndarray:
    pred = grid8.copy().astype(np.float32)
    valid = np.isfinite(pred[:, 0])
    pred[valid, 0] *= width / float(grid_width)
    pred[valid, 1] *= height / float(grid_height)
    return pred


def corner_median(pred8: np.ndarray, gt8: np.ndarray) -> float:
    valid = np.isfinite(pred8[:, 0])
    if int(valid.sum()) < N_DET_MIN:
        return float("inf")
    pred = pred8[valid]
    cost = np.linalg.norm(pred[:, None, :] - gt8[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return float(np.median(cost[ri, ci]))


def honest8_mean(pred8: np.ndarray | None, gt8: np.ndarray) -> float:
    if pred8 is None:
        return float("inf")
    pred8 = np.asarray(pred8, float)
    valid = np.isfinite(pred8[:, 0])
    if int(valid.sum()) < 8:
        return float("inf")
    cost = np.linalg.norm(pred8[:, None, :] - gt8[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return float(np.mean(cost[ri, ci]))


def solve_full_projection(pred8: np.ndarray, pred_center: np.ndarray | None,
                          camera: np.ndarray, image_shape) -> np.ndarray | None:
    keypoints = [None if not np.isfinite(pred8[i, 0]) else
                 [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    keypoints.append(None if pred_center is None else
                     [float(pred_center[0]), float(pred_center[1])])
    if sum(k is not None for k in keypoints) < N_DET_MIN:
        return None
    try:
        pose = APNP.solve_pose(keypoints, camera, dims=PALLET_DIMS,
                               img_shape=image_shape)
    except Exception:
        return None
    if pose is None or pose.get("projected_all") is None:
        return None
    projected = np.asarray(pose["projected_all"], np.float32)[:8]
    if projected.shape != (8, 2) or not np.all(np.isfinite(projected)):
        return None
    return projected


def _border_median(prob: np.ndarray, width: int = 2) -> float:
    border = np.concatenate([
        prob[:width].ravel(), prob[-width:].ravel(),
        prob[width:-width, :width].ravel(),
        prob[width:-width, -width:].ravel(),
    ])
    return float(np.median(border))


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int],
               pad: int = 0) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
    out = np.zeros(shape, np.uint8)
    if x1 >= x0 and y1 >= y0:
        out[y0:y1 + 1, x0:x1 + 1] = 1
    return out


def _keep_components_touching_seed(binary: np.ndarray, seed: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    out = np.zeros_like(binary, np.uint8)
    for label in range(1, n):
        component = labels == label
        if np.any(component & (seed > 0)):
            out[component] = 1
    return out


def _hull_mask(points: np.ndarray, shape: tuple[int, int], pad: int = 0) -> np.ndarray:
    valid = points[np.isfinite(points[:, 0])]
    out = np.zeros(shape, np.uint8)
    if len(valid) < 3:
        return out
    hull = cv2.convexHull(np.round(valid).astype(np.int32))
    cv2.fillConvexPoly(out, hull, 1)
    if pad > 0:
        kernel = np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)
        out = cv2.dilate(out, kernel)
    return out


def mask_coverage_features(prob: np.ndarray, grid8: np.ndarray) -> dict:
    """GT-free persistent low-threshold mask-extension features."""
    valid = grid8[np.isfinite(grid8[:, 0])]
    if len(valid) < N_DET_MIN:
        return {
            "mask_available": False, "mask_reliable": False,
            "mask_bg": None, "mask_max": float(prob.max()),
            "mask_contrast": None, "score_cover": None, "score_effective": None,
            "score_extension": None,
            "same_side_levels": 0, "max_kept_pixels": 0,
            "soft_connected_pixels": 0,
            "soft_outside_hull": None, "binary05_over_kpbbox": None,
            "levels": [],
        }

    h, w = prob.shape
    fx0, fy0 = valid.min(axis=0)
    fx1, fy1 = valid.max(axis=0)
    kp_w = max(float(fx1 - fx0), 1.0)
    kp_h = max(float(fy1 - fy0), 1.0)
    bbox = (
        int(np.floor(fx0)), int(np.floor(fy0)),
        int(np.ceil(fx1)), int(np.ceil(fy1)),
    )
    rx0 = max(0, int(math.floor(fx0 - ROI_EXPAND * kp_w)))
    ry0 = max(0, int(math.floor(fy0 - ROI_EXPAND * kp_h)))
    rx1 = min(w - 1, int(math.ceil(fx1 + ROI_EXPAND * kp_w)))
    ry1 = min(h - 1, int(math.ceil(fy1 + ROI_EXPAND * kp_h)))
    roi = _bbox_mask(prob.shape, (rx0, ry0, rx1, ry1))
    seed = _bbox_mask(prob.shape, bbox, pad=QUANT_CELL_TOL)

    bg = _border_median(prob)
    pmax = float(prob.max())
    contrast = max(pmax - bg, 0.0)
    levels = []
    direction_hits = Counter()
    max_kept_pixels = 0
    close_kernel = np.ones((3, 3), np.uint8)

    for rel in REL_THRESHOLDS:
        threshold = bg + rel * contrast
        binary = ((prob >= threshold) & (roi > 0)).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        kept = _keep_components_touching_seed(binary, seed)
        ys, xs = np.where(kept > 0)
        area = int(len(xs))
        max_kept_pixels = max(max_kept_pixels, area)
        if area == 0:
            levels.append({
                "rel": rel, "threshold": threshold, "pixels": 0,
                "outside_fraction": 0.0, "extension": 0.0,
                "direction": None, "product": 0.0,
            })
            continue

        exts = {
            "left": max(0.0, (fx0 - float(xs.min()) - QUANT_CELL_TOL) / kp_w),
            "right": max(0.0, (float(xs.max()) - fx1 - QUANT_CELL_TOL) / kp_w),
            "top": max(0.0, (fy0 - float(ys.min()) - QUANT_CELL_TOL) / kp_h),
            "bottom": max(0.0, (float(ys.max()) - fy1 - QUANT_CELL_TOL) / kp_h),
        }
        direction = max(exts, key=exts.get)
        extension = float(exts[direction])
        outside = float(np.count_nonzero((kept > 0) & (seed == 0)) / area)
        product = extension * outside
        if extension >= MIN_PERSISTENT_EXTENSION:
            direction_hits[direction] += 1
        levels.append({
            "rel": rel, "threshold": float(threshold), "pixels": area,
            "outside_fraction": outside, "extension": extension,
            "direction": direction, "product": float(product),
            "extensions": {k: float(v) for k, v in exts.items()},
        })

    extension_score = float(np.mean([x["product"] for x in levels]))
    same_side_levels = max(direction_hits.values(), default=0)
    extension_reliable = (contrast >= MIN_MASK_CONTRAST and
                          max_kept_pixels >= MIN_CONNECTED_SUPPORT_PIXELS and
                          same_side_levels >= MIN_PERSISTENT_LEVELS)

    # Primary soft-mass signal.  Restrict mass to the low-threshold component(s)
    # connected to the raw-keypoint footprint so a remote bright pixel/blob in
    # the expanded ROI cannot veto a frame by itself.
    soft_threshold = bg + REL_THRESHOLDS[0] * contrast
    soft_support = ((prob >= soft_threshold) & (roi > 0)).astype(np.uint8)
    soft_support = cv2.morphologyEx(soft_support, cv2.MORPH_CLOSE, close_kernel)
    soft_support = _keep_components_touching_seed(soft_support, seed)
    connected_pixels = int(soft_support.sum())
    weight = np.clip(prob - bg, 0.0, None) * soft_support
    hull = _hull_mask(grid8, prob.shape, pad=QUANT_CELL_TOL)
    total_weight = float(weight.sum())
    soft_outside = (float(weight[hull == 0].sum() / total_weight)
                    if total_weight > 1e-8 else None)
    reliable = bool(contrast >= MIN_MASK_CONTRAST and
                    connected_pixels >= MIN_CONNECTED_SUPPORT_PIXELS and
                    soft_outside is not None)
    score_effective = float(soft_outside) if reliable else None

    binary05 = ((prob >= 0.5) & (roi > 0)).astype(np.uint8)
    kp_bbox_area = max(1, int(round(kp_w * kp_h)))
    binary05_ratio = float(binary05.sum() / kp_bbox_area)

    return {
        "mask_available": True,
        "mask_reliable": bool(reliable),
        "mask_bg": bg, "mask_max": pmax, "mask_contrast": contrast,
        "kp_bbox_grid": [float(fx0), float(fy0), float(fx1), float(fy1)],
        "roi_grid": [rx0, ry0, rx1, ry1],
        "score_cover": soft_outside, "score_effective": score_effective,
        "score_extension": extension_score,
        "extension_reliable": bool(extension_reliable),
        "soft_connected_pixels": connected_pixels,
        "same_side_levels": int(same_side_levels),
        "persistent_direction": (direction_hits.most_common(1)[0][0]
                                 if direction_hits else None),
        "max_kept_pixels": int(max_kept_pixels),
        "soft_outside_hull": soft_outside,
        "binary05_over_kpbbox": binary05_ratio,
        "levels": levels,
    }


def _clip_convex(poly: np.ndarray, width: int, height: int) -> np.ndarray | None:
    poly = cv2.convexHull(np.asarray(poly, np.float32)).reshape(-1, 2)
    rect = np.array([
        [0.0, 0.0], [width - 1.0, 0.0],
        [width - 1.0, height - 1.0], [0.0, height - 1.0],
    ], np.float32)
    area, clipped = cv2.intersectConvexConvex(poly, rect)
    if area <= 1e-6 or clipped is None:
        return None
    return cv2.convexHull(clipped.astype(np.float32)).reshape(-1, 2)


def _area(poly: np.ndarray | None) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    return float(abs(cv2.contourArea(np.asarray(poly, np.float32))))


def _centroid(poly: np.ndarray) -> np.ndarray:
    moments = cv2.moments(np.asarray(poly, np.float32))
    if abs(moments["m00"]) < 1e-8:
        return np.mean(poly, axis=0)
    return np.array([moments["m10"] / moments["m00"],
                     moments["m01"] / moments["m00"]], np.float32)


def gt_coverage_features(pred8: np.ndarray, gt8: np.ndarray,
                         width: int, height: int,
                         source_n_det: int | None = None) -> dict:
    """GT-only diagnostic label for inner/smaller predictions."""
    valid = pred8[np.isfinite(pred8[:, 0])]
    n_det = int(len(valid))
    gt_orig = cv2.convexHull(gt8.astype(np.float32)).reshape(-1, 2)
    gt_clip = _clip_convex(gt8, width, height)
    pred_clip = _clip_convex(valid, width, height) if len(valid) >= 3 else None
    gt_orig_area, gt_area, pred_area = _area(gt_orig), _area(gt_clip), _area(pred_clip)
    visible = gt_area / gt_orig_area if gt_orig_area > 1e-8 else 0.0

    if pred_clip is None or gt_clip is None or pred_area <= 0 or gt_area <= 0:
        return {
            "uc_eligible": False, "visible_fraction": visible,
            "gt_area": gt_area, "pred_area": pred_area,
            "intersection_area": 0.0, "pred_contain": 0.0,
            "gt_cover": 0.0, "area_ratio": None,
            "r_major": None, "r_minor": None, "r_min": None,
            "center_in_gt": False, "center_offset": None,
            "side_gaps": None,
            "uc_strict": False, "uc_main": False, "uc_loose": False,
        }

    inter_area, _ = cv2.intersectConvexConvex(
        cv2.convexHull(gt_clip.astype(np.float32)),
        cv2.convexHull(pred_clip.astype(np.float32)))
    inter_area = float(max(inter_area, 0.0))
    pred_contain = inter_area / pred_area
    gt_cover = inter_area / gt_area
    area_ratio = pred_area / gt_area
    pred_center = _centroid(pred_clip)
    gt_center = _centroid(gt_clip)
    center_in = cv2.pointPolygonTest(
        cv2.convexHull(gt_clip.astype(np.float32)),
        (float(pred_center[0]), float(pred_center[1])), False) >= 0

    # GT PCA axes: ratios/gaps are orientation-aware and use the same projected
    # 2D footprint.  Area/containment remain primary when axes are ambiguous.
    centered = gt8 - gt8.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order].T
    ratios, side_gaps = [], []
    center_terms = []
    for axis in axes:
        g = gt8 @ axis
        p = valid @ axis
        g0, g1 = float(g.min()), float(g.max())
        p0, p1 = float(p.min()), float(p.max())
        span = max(g1 - g0, 1e-6)
        ratios.append((p1 - p0) / span)
        side_gaps.extend([(p0 - g0) / span, (g1 - p1) / span])
        center_terms.append(((pred_center - gt_center) @ axis) / span)
    r_major, r_minor = float(ratios[0]), float(ratios[1])
    r_min = min(r_major, r_minor)
    center_offset = float(np.linalg.norm(center_terms))
    min_gap = float(min(side_gaps))
    detection_count = n_det if source_n_det is None else int(source_n_det)
    eligible = detection_count >= N_DET_MIN and visible >= 0.90

    def label(contain: float, gap: float, cover: float, ratio: float) -> bool:
        return bool(eligible and center_in and pred_contain >= contain and
                    min_gap >= gap and (gt_cover <= cover or r_min <= ratio))

    return {
        "uc_eligible": bool(eligible), "visible_fraction": float(visible),
        "gt_area": gt_area, "pred_area": pred_area,
        "intersection_area": inter_area,
        "pred_contain": float(pred_contain), "gt_cover": float(gt_cover),
        "area_ratio": float(area_ratio),
        "r_major": r_major, "r_minor": r_minor, "r_min": float(r_min),
        "center_in_gt": bool(center_in), "center_offset": center_offset,
        "side_gaps": [float(x) for x in side_gaps],
        "uc_strict": label(0.95, 0.00, 0.60, 0.70),
        "uc_main": label(0.90, -0.05, 0.75, 0.80),
        "uc_loose": label(0.85, -0.10, 0.85, 0.88),
    }


def evaluate_flag(rows: list[dict], tau: float, label: str = "uc_main") -> dict:
    use = [r for r in rows if r["uc_eligible"]]
    flags = [float(r["score_effective"] or 0.0) >= tau for r in use]
    target = [bool(r[label]) for r in use]
    tp = sum(f and y for f, y in zip(flags, target))
    fp = sum(f and not y for f, y in zip(flags, target))
    fn = sum((not f) and y for f, y in zip(flags, target))
    tn = sum((not f) and (not y) for f, y in zip(flags, target))
    good = [bool(r["good"]) for r in use]
    good_total = sum(good)
    good_lost = sum(f and g for f, g in zip(flags, good))
    return {
        "n": len(use), "positives": sum(target), "flagged": sum(flags),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "good_total": good_total, "good_lost": good_lost,
        "good_loss_rate": good_lost / good_total if good_total else None,
    }


def choose_threshold(rows: list[dict]) -> tuple[float, dict, list[dict]]:
    use = [r for r in rows if r["uc_eligible"]]
    unique = sorted(set(float(r["score_effective"] or 0.0) for r in use))
    candidates = sorted(set([INITIAL_SCORE_TAU] + [x for x in unique if x > 0]))
    sweep = []
    for tau in candidates:
        result = evaluate_flag(use, tau, "uc_main")
        result["tau"] = tau
        sweep.append(result)

    feasible = [x for x in sweep
                if x["flagged"] > 0 and x["tp"] > 0 and
                (x["precision"] is not None and x["precision"] >= MIN_TARGET_PRECISION) and
                (x["good_loss_rate"] is None or x["good_loss_rate"] <= 0.10) and
                (x["specificity"] is None or x["specificity"] >= 0.90)]
    if feasible:
        best = max(feasible, key=lambda x: (
            x["recall"] or 0.0, x["precision"] or 0.0,
            -(x["good_loss_rate"] or 0.0), x["tau"]))
    else:
        # Diagnostic fallback: fixed pre-registered starting point.  The report
        # marks it non-adoptable when the conservative constraints are unmet.
        best = min(sweep, key=lambda x: abs(x["tau"] - INITIAL_SCORE_TAU))
    return float(best["tau"]), best, sweep


def ranking_metrics(rows: list[dict], label: str) -> dict:
    use = [r for r in rows if r["uc_eligible"]]
    y = np.array([bool(r[label]) for r in use], int)
    score = np.array([float(r["score_effective"] or 0.0) for r in use], float)
    if len(np.unique(y)) < 2:
        return {"auprc": None, "auroc": None}
    return {
        "auprc": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
    }


def accepted_summary(rows: list[dict], key: str, tau: float) -> dict | None:
    base = [r for r in rows if r.get(key) is not None and bool(r[key])]
    if not base:
        return None
    kept = [r for r in base if float(r["score_effective"] or 0.0) < tau]

    def one(xs):
        good = sum(bool(r["good"]) for r in xs)
        return {"n": len(xs), "good": good, "bad": len(xs) - good,
                "purity": good / len(xs) if xs else None}
    return {"before": one(base), "after": one(kept), "rejected": len(base) - len(kept)}


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "Y" if v else "n"
    if isinstance(v, int):
        return str(v)
    return f"{float(v):.{digits}f}"


def _eval_line(name: str, e: dict) -> str:
    return (f"{name:<12}{e['n']:>5}{e['positives']:>5}{e['flagged']:>6}"
            f"{e['tp']:>4}{e['fp']:>4}{_fmt(e['precision']):>8}"
            f"{_fmt(e['recall']):>8}{_fmt(e['specificity']):>8}"
            f"{e['good_lost']:>7}/{e['good_total']:<3}")


def build_report(records: list[dict], tau: float, selection: dict,
                 sweep: list[dict]) -> str:
    calibration = [r for r in records if r["split"] == "filterval"]
    holdout = [r for r in records if r["split"] == "handannot17"]
    lines = [
        "# PAPER_S2 mask-coverage filter validation",
        "",
        "- no retraining: Stage B ep57 mask-aux + belief outputs from one forward",
        "- calibration: filterval 123 (outside44/night43/manual36)",
        "- fixed-threshold secondary check: handannot17 (cad11/noapril6); this set is",
        "  development-touched, because the original mask hypothesis inspected one noapril frame",
        "- score is GT-free; GT is used only for evaluation labels",
        f"- selected threshold: `score_effective >= {tau:.6f}` flags under-coverage",
        "",
        "## Metric",
        "",
        "Primary `score_cover` is the soft mask probability mass outside the raw-keypoint",
        "convex hull dilated by one 50x50 cell, divided by positive mask mass in the",
        "low-threshold component(s) connected to that footprint inside a 0.75x-expanded",
        "keypoint ROI. The per-frame border median is removed first.",
        f"Mask probability contrast must be >={MIN_MASK_CONTRAST:.2f}; lower-contrast frames",
        "are UNAVAILABLE rather than rejected.",
        "",
        "A secondary diagnostic uses relative mask thresholds r={0.02,0.05,0.10,0.20}.",
        "It keeps components touching the keypoint box and records mean(E*O), where E is",
        "directional extension and O is outside-box support. This extension score is saved",
        "per frame but is not the selected filter score because calibration separation was",
        "weaker than soft outside mass.",
        "",
        "The main GT diagnostic label is computed from the final PnP `projected_all` 8-point",
        "footprint, not the incomplete raw 6-8 point hull. It requires prediction center",
        "inside GT, >=90% containment, side gaps >=-5%, and GT coverage<=75% or minimum",
        "PCA-axis span ratio<=80%. Frames with GT visible fraction<90%, n_det<6, or PnP",
        "failure are excluded from threshold fitting. Safety retention uses honest8<10px.",
        "",
        "## Dataset / label counts",
        "",
        "```",
        f"{'split/domain':<22}{'N':>5}{'det>=6':>8}{'reliable':>10}{'eligible':>10}"
        f"{'strict':>8}{'main':>7}{'loose':>8}{'posegood':>10}",
        "-" * 86,
    ]
    groups = [
        ("filterval/all", calibration),
        ("  outside", [r for r in calibration if r["domain"] == "outside"]),
        ("  night", [r for r in calibration if r["domain"] == "night"]),
        ("  manual", [r for r in calibration if r["domain"] == "manual"]),
        ("handannot17/all", holdout),
        ("  cad", [r for r in holdout if r["domain"] == "cad"]),
        ("  noapril", [r for r in holdout if r["domain"] == "noapril"]),
    ]
    for name, rows in groups:
        lines.append(
            f"{name:<22}{len(rows):>5}{sum(r['n_det']>=6 for r in rows):>8}"
            f"{sum(bool(r.get('mask_reliable')) for r in rows):>10}"
            f"{sum(r['uc_eligible'] for r in rows):>10}"
            f"{sum(r['uc_strict'] for r in rows):>8}"
            f"{sum(r['uc_main'] for r in rows):>7}"
            f"{sum(r['uc_loose'] for r in rows):>8}"
            f"{sum(r['good'] for r in rows):>8}")
    lines += ["```", "", "## Selected threshold results", "", "```",
              f"{'set':<12}{'N':>5}{'UC+':>5}{'flag':>6}{'TP':>4}{'FP':>4}"
              f"{'Prec':>8}{'Recall':>8}{'Spec':>8}{'pose lost':>11}",
              "-" * 79]
    for label in ("uc_strict", "uc_main", "uc_loose"):
        e = evaluate_flag(calibration, tau, label)
        lines.append(_eval_line("cal/" + label[3:9], e))
    lines.append(_eval_line("secondary", evaluate_flag(holdout, tau, "uc_main")))
    lines.append(_eval_line("all/main", evaluate_flag(records, tau, "uc_main")))
    lines += ["```", ""]

    flagged_rows = [r for r in records
                    if r["uc_eligible"] and float(r["score_effective"] or 0.0) >= tau]
    lines += ["## Flagged frames at selected threshold", "", "```",
              f"{'split':<12}{'domain':<9}{'fid':<21}{'score':>8}{'UC':>4}"
              f"{'pose':>6}{'cmed':>7}{'h8':>7}{'old':>10}{'f4tight':>9}",
              "-" * 95]
    for r in sorted(flagged_rows, key=lambda x: float(x["score_effective"]),
                    reverse=True):
        lines.append(
            f"{r['split']:<12}{r['domain']:<9}{r['fid']:<21}"
            f"{r['score_effective']:>8.3f}{('Y' if r['uc_main'] else 'n'):>4}"
            f"{('Y' if r['good'] else 'n'):>6}{_fmt(r['corner_med'],1):>7}"
            f"{_fmt(r['honest8'],1):>7}"
            f"{(r.get('existing_status') or '-'):>10}"
            f"{('-' if r.get('f4_tight_pass') is None else ('Y' if r['f4_tight_pass'] else 'n')):>9}")
    lines += ["```", ""]

    rank = {label: ranking_metrics(calibration, label)
            for label in ("uc_strict", "uc_main", "uc_loose")}
    lines += ["## Ranking metrics on calibration", "", "```",
              f"{'label':<10}{'AUPRC':>10}{'AUROC':>10}"]
    for label, result in rank.items():
        lines.append(f"{label:<10}{_fmt(result['auprc']):>10}{_fmt(result['auroc']):>10}")
    lines += ["```", ""]

    lines += ["## Existing filter impact on filterval", "",
              "Here `good` means final PnP honest8 mean <10px (not partial raw-corner median).", ""]
    for key, name in (("deploy_pass", "DEPLOY6"), ("f4_tight_pass", "DEPLOY6 + f4<=0.8")):
        s = accepted_summary(calibration, key, tau)
        if s is None:
            continue
        b, a = s["before"], s["after"]
        lines.append(
            f"- {name}: {b['n']}={b['good']} good/{b['bad']} bad "
            f"(purity {_fmt(b['purity'])}) -> {a['n']}={a['good']} good/{a['bad']} bad "
            f"(purity {_fmt(a['purity'])}); rejected {s['rejected']}")
    lines.append("")

    lines += ["## Fixed selected threshold by calibration domain", "", "```",
              f"{'domain':<12}{'N':>5}{'UC+':>5}{'flag':>6}{'TP':>4}{'FP':>4}"
              f"{'Prec':>8}{'Recall':>8}{'Spec':>8}{'pose lost':>11}",
              "-" * 79]
    fixed_domain_results = {}
    for domain in ("outside", "night", "manual"):
        e = evaluate_flag([r for r in calibration if r["domain"] == domain],
                          tau, "uc_main")
        fixed_domain_results[domain] = e
        lines.append(_eval_line(domain, e))
    lines += ["```", ""]

    # Leave-one-domain-out: threshold chosen only on the other two domains.
    lines += ["## Leave-one-domain-out stability", "", "```",
              f"{'held domain':<14}{'tau':>10}{'N':>5}{'UC+':>5}{'flag':>6}"
              f"{'TP':>4}{'FP':>4}{'Prec':>8}{'Recall':>8}",
              "-" * 72]
    for domain in ("outside", "night", "manual"):
        train = [r for r in calibration if r["domain"] != domain]
        test = [r for r in calibration if r["domain"] == domain]
        dtau, _, _ = choose_threshold(train)
        e = evaluate_flag(test, dtau, "uc_main")
        lines.append(
            f"{domain:<14}{dtau:>10.5f}{e['n']:>5}{e['positives']:>5}{e['flagged']:>6}"
            f"{e['tp']:>4}{e['fp']:>4}{_fmt(e['precision']):>8}{_fmt(e['recall']):>8}")
    lines += ["```", ""]

    raw = evaluate_flag(calibration, INITIAL_SCORE_TAU, "uc_main")
    constrained = (selection["tp"] > 0 and
                   (selection["precision"] or 0) >= MIN_TARGET_PRECISION and
                   (selection["specificity"] or 0) >= 0.90 and
                   (selection["good_loss_rate"] is None or
                    selection["good_loss_rate"] <= 0.10))
    positive_domain_results = [e for e in fixed_domain_results.values()
                               if e["positives"] > 0]
    domain_stable = bool(positive_domain_results) and all(
        e["tp"] > 0 and (e["precision"] or 0) >= MIN_TARGET_PRECISION
        for e in positive_domain_results)
    secondary_eval = evaluate_flag(holdout, tau, "uc_main")
    secondary_safe = bool(
        secondary_eval["tp"] > 0 and
        (secondary_eval["precision"] or 0) >= MIN_TARGET_PRECISION and
        (secondary_eval["good_loss_rate"] is None or
         secondary_eval["good_loss_rate"] <= 0.10))
    lines += ["## Decision", ""]
    if constrained and domain_stable and secondary_safe:
        lines.append(
            f"The calibration sweep found a threshold satisfying >={MIN_TARGET_PRECISION:.0%} "
            "target precision, <=10% pose-good loss, and >=90% non-undercoverage "
            "specificity. Treat it as a review-tier candidate; "
            "the development-touched secondary set still cannot justify a hard reject.")
    elif constrained:
        lines.append(
            "The pooled calibration threshold meets the numeric constraints, but fixed-domain "
            "or secondary-check safety fails. Keep the score only as a diagnostic/manual-review "
            "ranking feature. Do not wire it into the pseudo-label hard-reject AND.")
    else:
        lines.append(
            "No calibration threshold met the conservative adoption constraints while "
            "catching an under-coverage positive. Keep this signal diagnostic-only; do "
            "not add it as a hard pseudo-label reject.")
    lines += [
        "",
        f"Initial sweep anchor tau={INITIAL_SCORE_TAU:.2f}: "
        f"flag={raw['flagged']}, TP={raw['tp']}, FP={raw['fp']}, "
        f"precision={_fmt(raw['precision'])}, recall={_fmt(raw['recall'])}.",
        "",
        "Caveats: mask head was trained as a weak synthetic auxiliary head; 50x50",
        "quantization and small per-domain UC counts make threshold estimates unstable.",
        "The old absolute mask>=0.5 area ratio is reported per frame for comparison but",
        "is not used in the rule. The secondary set is not an untouched holdout.",
        "",
        "## Conservative-feasible calibration sweep points",
        "",
        "```",
        f"{'tau':>10}{'flag':>6}{'TP':>4}{'FP':>4}{'Prec':>8}{'Recall':>8}"
        f"{'Spec':>8}{'goodloss':>10}",
    ]
    feasible_sweep = [x for x in sweep
                      if x["tp"] > 0 and
                      (x["precision"] is not None and
                       x["precision"] >= MIN_TARGET_PRECISION) and
                      (x["good_loss_rate"] is None or x["good_loss_rate"] <= 0.10) and
                      (x["specificity"] is None or x["specificity"] >= 0.90)]
    shown = sorted(feasible_sweep or sweep, key=lambda x: (
        -(x["recall"] or 0), -(x["precision"] or 0),
        x["good_loss_rate"] or 0, x["tau"]))[:12]
    for x in shown:
        lines.append(
            f"{x['tau']:>10.5f}{x['flagged']:>6}{x['tp']:>4}{x['fp']:>4}"
            f"{_fmt(x['precision']):>8}{_fmt(x['recall']):>8}"
            f"{_fmt(x['specificity']):>8}{_fmt(x['good_loss_rate']):>10}")
    lines += ["```", ""]
    return "\n".join(lines)


def _draw_dashed(img: np.ndarray, a: np.ndarray, b: np.ndarray,
                 color: tuple[int, int, int], thickness: int = 1,
                 dash: int = 8) -> None:
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length < 1:
        return
    unit = vec / length
    for start in np.arange(0, length, 2 * dash):
        p0 = a + unit * start
        p1 = a + unit * min(start + dash, length)
        cv2.line(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                 color, thickness, cv2.LINE_AA)


def make_overlay(record: dict, visual: dict, tau: float, category: str) -> np.ndarray:
    img = cv2.imread(record["ip"])
    pred8 = np.asarray(visual["pred8"], np.float32)
    projected8 = (None if visual.get("projected8") is None else
                  np.asarray(visual["projected8"], np.float32))
    grid8 = np.asarray(visual["grid8"], np.float32)
    gt8 = np.asarray(visual["gt8"], np.float32)
    prob = np.asarray(visual["prob"], np.float32)
    h, w = img.shape[:2]

    left = img.copy()
    for a, b in EDGES:
        _draw_dashed(left, gt8[a], gt8[b], (0, 220, 0), 1)
        if projected8 is not None and np.isfinite(projected8[[a, b], 0]).all():
            cv2.line(left, tuple(np.round(projected8[a]).astype(int)),
                     tuple(np.round(projected8[b]).astype(int)), (0, 255, 255), 1,
                     cv2.LINE_AA)
        if np.isfinite(pred8[[a, b], 0]).all():
            cv2.line(left, tuple(np.round(pred8[a]).astype(int)),
                     tuple(np.round(pred8[b]).astype(int)), (0, 0, 255), 2,
                     cv2.LINE_AA)
    for i, p in enumerate(pred8):
        if not np.isfinite(p[0]):
            continue
        color = (0, 0, 255) if i < 4 else (255, 150, 0)
        q = tuple(np.round(p).astype(int))
        cv2.circle(left, q, 4, color, -1)
        cv2.putText(left, str(i), (q[0] + 4, q[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)

    prob_orig = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.clip(prob_orig * 255, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    right = cv2.addWeighted(img, 0.45, heat, 0.55, 0)

    # Visualize the actual score support: cyan ROI, green dilated raw hull, and
    # red tint for connected soft-mask pixels contributing outside that hull.
    hull_grid = _hull_mask(grid8, prob.shape, pad=QUANT_CELL_TOL)
    if record.get("roi_grid") and record.get("kp_bbox_grid"):
        rx0, ry0, rx1, ry1 = record["roi_grid"]
        roi_grid = _bbox_mask(prob.shape, (rx0, ry0, rx1, ry1))
        x0, y0, x1, y1 = record["kp_bbox_grid"]
        seed_grid = _bbox_mask(
            prob.shape, (int(np.floor(x0)), int(np.floor(y0)),
                         int(np.ceil(x1)), int(np.ceil(y1))),
            pad=QUANT_CELL_TOL)
        threshold = float(record["mask_bg"] or 0.0) + REL_THRESHOLDS[0] * float(
            record["mask_contrast"] or 0.0)
        support_grid = ((prob >= threshold) & (roi_grid > 0)).astype(np.uint8)
        support_grid = cv2.morphologyEx(
            support_grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        support_grid = _keep_components_touching_seed(support_grid, seed_grid)
        outside_grid = ((support_grid > 0) & (hull_grid == 0)).astype(np.uint8)
        outside_orig = cv2.resize(outside_grid, (w, h), interpolation=cv2.INTER_NEAREST)
        red = np.zeros_like(right)
        red[..., 2] = 255
        right = np.where(outside_orig[..., None].astype(bool),
                         (right * 0.45 + red * 0.55).astype(np.uint8), right)
        cv2.rectangle(
            right,
            (int(round(rx0 * w / prob.shape[1])), int(round(ry0 * h / prob.shape[0]))),
            (int(round(rx1 * w / prob.shape[1])), int(round(ry1 * h / prob.shape[0]))),
            (255, 255, 0), 1)

    hull_orig = cv2.resize(hull_grid, (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(hull_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(right, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    for abs_tau, color in ((0.10, (255, 255, 255)),
                           (0.25, (0, 255, 255)),
                           (0.50, (255, 0, 255))):
        contours, _ = cv2.findContours((prob_orig >= abs_tau).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(right, contours, -1, color, 1, cv2.LINE_AA)

    flagged = float(record["score_effective"] or 0.0) >= tau
    status = record.get("existing_status") or "-"
    line1 = (f"{category} {record['domain']}/{record['fid']} "
             f"det={record['n_det']}/8 old={status} flag={'Y' if flagged else 'n'}")
    line2 = (f"soft-out={_fmt(record['score_cover'],4)} "
             f"ext={_fmt(record.get('score_extension'),4)} "
             f"dir={record.get('persistent_direction') or '-'} "
             f"levels={record['same_side_levels']} contrast={_fmt(record['mask_contrast'],3)} "
             f"reliable={'Y' if record.get('mask_reliable') else 'n'}")
    line3 = (f"GT-only: cmed={_fmt(record['corner_med'],1)} h8={_fmt(record['honest8'],1)} "
             f"posegood={'Y' if record['good'] else 'n'} UC={'Y' if record['uc_main'] else 'n'} "
             f"contain={_fmt(record['pred_contain'],2)} "
             f"cover={_fmt(record['gt_cover'],2)} rmin={_fmt(record['r_min'],2)}")
    canvas = np.hstack([left, right])
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (0, 0, 0), -1)
    for y, text_line in ((17, line1), (37, line2), (57, line3)):
        cv2.putText(canvas, text_line, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def write_overlays(records: list[dict], visuals: dict, tau: float) -> None:
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    for old in OVERLAY_DIR.glob("*.jpg"):
        old.unlink()

    calibration = [r for r in records if r["split"] == "filterval" and r["n_det"] >= 6]
    holdout = [r for r in records if r["split"] == "handannot17" and r["n_det"] >= 6]
    flagged = [r for r in calibration if float(r["score_effective"] or 0.0) >= tau]
    holdout_flagged = [r for r in holdout if float(r["score_effective"] or 0.0) >= tau]
    categories = {
        "caught": [r for r in flagged if r["uc_main"]][:12],
        "falseflag_good": [r for r in flagged if r["good"] and not r["uc_main"]][:12],
        "missed": [r for r in calibration if r["uc_main"] and r not in flagged][:12],
        "secondary_caught": [r for r in holdout_flagged if r["uc_main"]][:12],
        "secondary_falseflag_posegood": [r for r in holdout_flagged
                                          if r["pose_good"] and not r["uc_main"]][:12],
        "secondary_missed": [r for r in holdout if r["uc_main"] and
                              r not in holdout_flagged][:12],
        "top": sorted(calibration, key=lambda r: float(r["score_effective"] or 0.0),
                      reverse=True)[:16],
    }
    written = set()
    for category, rows in categories.items():
        for rank, record in enumerate(rows):
            key = (category, record["domain"], record["fid"])
            if key in written:
                continue
            written.add(key)
            overlay = make_overlay(
                record, visuals[(record["domain"], record["fid"])], tau, category)
            path = OVERLAY_DIR / f"{category}_{rank:02d}_{record['domain']}_{record['fid']}.jpg"
            if not cv2.imwrite(str(path), overlay):
                raise OSError(f"failed to write overlay: {path}")


def run(args) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    frames = collect_frames()
    if args.limit:
        frames = frames[:args.limit]
    print(f"[data] {len(frames)} frames; device={device}; no retraining")
    model = load_model(device)
    records, visuals = [], {}

    for index, frame in enumerate(frames, 1):
        img = cv2.imread(frame["ip"])
        if img is None:
            raise FileNotFoundError(frame["ip"])
        gt8, camera = read_annotation(frame["jp"])
        _, prob, grid8, grid_center = infer(model, img, device)
        pred8 = grid_to_image(grid8, img.shape[1], img.shape[0],
                              prob.shape[1], prob.shape[0])
        pred_center = None
        if grid_center is not None:
            pred_center = np.array([
                grid_center[0] * img.shape[1] / prob.shape[1],
                grid_center[1] * img.shape[0] / prob.shape[0],
            ], np.float32)
        n_det = int(np.isfinite(pred8[:, 0]).sum())
        cmed = corner_median(pred8, gt8)
        projected8 = solve_full_projection(pred8, pred_center, camera, img.shape)
        honest8 = honest8_mean(projected8, gt8)
        mask_features = mask_coverage_features(prob, grid8)
        raw_gt_features = gt_coverage_features(
            pred8, gt8, img.shape[1], img.shape[0], source_n_det=n_det)
        if projected8 is not None:
            pose_gt_features = gt_coverage_features(
                projected8, gt8, img.shape[1], img.shape[0], source_n_det=n_det)
        else:
            pose_gt_features = gt_coverage_features(
                np.full((8, 2), np.nan, np.float32), gt8,
                img.shape[1], img.shape[0], source_n_det=0)
        record = {
            **frame, "n_det": n_det,
            "corner_med": None if not np.isfinite(cmed) else float(cmed),
            "corner_good": bool(np.isfinite(cmed) and cmed < GOOD_PX),
            "pnp_ok": bool(projected8 is not None),
            "honest8": None if not np.isfinite(honest8) else float(honest8),
            "pose_good": bool(np.isfinite(honest8) and honest8 < GOOD_PX),
            # Selection safety uses the final full-8 PnP projection, not the
            # incomplete raw 6-8 point hull.  `good` remains the generic key
            # consumed by threshold selection/report helpers.
            "good": bool(np.isfinite(honest8) and honest8 < GOOD_PX),
            **mask_features, **pose_gt_features,
            **{f"raw_{k}": v for k, v in raw_gt_features.items()},
        }
        records.append(record)
        visuals[(frame["domain"], frame["fid"])] = {
            "prob": prob, "grid8": grid8,
            "pred8": pred8, "projected8": projected8,
            "gt8": gt8,
        }
        if frame["split"] == "filterval" and frame["prior_n_det"] != n_det:
            print(f"[warn] n_det drift {frame['domain']}/{frame['fid']}: "
                  f"prior={frame['prior_n_det']} now={n_det}")
        print(f"[{index:03d}/{len(frames)}] {frame['domain']:<8} {frame['fid']} "
              f"det={n_det} cm={_fmt(record['corner_med'],1)} "
              f"h8={_fmt(record['honest8'],1)} score={_fmt(record['score_effective'],4)} "
              f"UC={int(record['uc_main'])}")

    if args.limit:
        print("[limit] smoke run completed; reports are not written")
        return

    calibration = [r for r in records if r["split"] == "filterval"]
    tau, selection, sweep = choose_threshold(calibration)
    report = build_report(records, tau, selection, sweep)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "records.json").write_text(json.dumps(
        {"weights": str(WEIGHTS), "threshold": tau,
         "recommended_use": "diagnostic_manual_review_only_not_hard_reject",
         "score_config": {
             "score_name": "connected_soft_mass_outside_dilated_raw_hull",
             "comparator": ">=",
             "rel_thresholds": REL_THRESHOLDS, "roi_expand": ROI_EXPAND,
             "quant_cell_tol": QUANT_CELL_TOL,
             "border_median_width": 2,
             "soft_component_relative_threshold": REL_THRESHOLDS[0],
             "hull_dilation_cells": QUANT_CELL_TOL,
             "min_connected_support_pixels": MIN_CONNECTED_SUPPORT_PIXELS,
             "min_mask_contrast": MIN_MASK_CONTRAST,
             "min_persistent_extension": MIN_PERSISTENT_EXTENSION,
             "min_persistent_levels": MIN_PERSISTENT_LEVELS,
             "initial_sweep_anchor": INITIAL_SCORE_TAU,
             "minimum_target_precision": MIN_TARGET_PRECISION,
             "gt_footprint": "PnP projected_all first 8 points",
             "retention_good": "honest8_mean < 10px",
             "pallet_dims_m": PALLET_DIMS,
         },
         "selection": selection, "records": records},
        indent=2, default=_json_number) + "\n")
    (OUT_DIR / "REPORT.md").write_text(report + "\n")
    if not args.no_overlays:
        write_overlays(records, visuals, tau)
    print(report)
    print(f"[save] {OUT_DIR / 'records.json'}")
    print(f"[save] {OUT_DIR / 'REPORT.md'}")
    if not args.no_overlays:
        print(f"[save] {OVERLAY_DIR}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke-test first N frames; writes no reports")
    parser.add_argument("--no-overlays", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
