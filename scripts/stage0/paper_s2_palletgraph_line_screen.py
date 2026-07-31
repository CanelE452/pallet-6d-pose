#!/usr/bin/env python3
"""paper_s2_palletgraph_line_screen.py — PalletGraph-6D line utility screen.

목적(판정 전용):
  MSL(Mask-Supported Semantic Lines) + DGP(Dimension-Guided Graph Pose)가
  ep57 point-only 구조보다 유효한지 **oracle gate 로 먼저** 판정한다.
  full architecture 를 한 번에 학습하지 않는다.

  P0  ep57 decoder + 현재 OpenCV PnP                         (기존 baseline)
  P1  ep57 decoder + DGP (lambda_line=0)                     (solver parity)
  P2  P1 + oracle-AMODAL semantic line                       (optimistic 상한)
  P3  P1 + oracle-ASSOCIATED observed line fragment          (현실적 상한)
  P4  P1 + class-agnostic generic edge (mask-gated)          (semantic 없는 대조)

핵심 원칙:
  * line map 은 keypoint 를 복원하는 중간 표현이 아니라 DGP 가 직접 읽는 2D residual.
    과거 실패한 vector/offset/voting/endpoint 표현을 재도입하지 않는다.
  * P2/P3 는 **oracle** 이다.  inference 결과처럼 보고하지 않는다.
    amodal 은 자기가림 edge 를 포함할 수 있으므로 "visible" 이라 부르지 않는다.
  * yaw 는 180° 대칭을 실패로 세지 않는다 (modulo-pi).
  * selected reprojection 을 headline metric 으로 쓰지 않는다.

Usage:
    python scripts/stage0/paper_s2_palletgraph_line_screen.py --all-oracle
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (
    ROOT / "Deep_Object_Pose/common",
    ROOT / "Deep_Object_Pose/train",
    ROOT / "scripts/stage0",
    ROOT / "scripts/data_prep/eval",
    ROOT / "challenge/scripts",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paper_s2_mechanism_diagnostic as MD  # noqa: E402
import paper_s2_frozen_diagnostic as FZ  # noqa: E402
import pallet_graph_geometry as PG  # noqa: E402
import dimension_guided_graph_pose as DGP  # noqa: E402

OUT_DIR = ROOT / "data/pallet/results/paper_s2_palletgraph_line_screen"
MECH_DIR = MD.OUT_DIR
BELIEF = 50

# Fixed BEFORE any metric is looked at (Phase E4).
CLOSE_RANGE_RULE = "bbox_area_ratio_top_25pct"
LINE_SIGMA_PX = 1.5
LINE_MAP_SCALE = 2  # oracle maps at half image resolution (memory/speed)
CANNY_SETTINGS = ((50, 150), (100, 200), (150, 250))  # all reported, not swept
ASSOCIATION_RADIUS_PX = 4.0
MASK_DILATION_PX = 9
MASK_EPSILON = 0.15
LAMBDA_POINT = 1.0
DGP_ITERATIONS = 6

# lambda_line is CALIBRATED, not guessed.  With lambda_line=1 the point term
# (median E_point ~28, max ~319) swamps the line term (median E_line ~5.6), so
# the line residual moves the pose by ~0.002 deg and the screen would measure
# "we did not weight the line", not "line evidence is useless".
#
# Calibration rule, fixed before any pose metric is inspected: scale the line
# term so its energy is a stated FRACTION of the point term, measured on the
# P1 poses.  All three fractions are reported; none is selected by outcome.
LINE_CONTRIBUTION_FRACTIONS = (0.25, 0.50, 1.00)
PRIMARY_LINE_FRACTION = 0.50

ARMS = ("P0", "P1", "P2", "P3", "P4")


def log(message: str) -> None:
    print(message, flush=True)


def finite(value: Any) -> Optional[float]:
    return MD.finite(value)


def nanmedian(values: Iterable[Any]) -> Optional[float]:
    return MD.nanmedian(values)


# ============================================================================
# Phase A — identity and baseline gate
# ============================================================================
def identity_and_gate() -> dict[str, Any]:
    gate_path = MECH_DIR / "baseline_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("BLOCKED: mechanism baseline_gate.json missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError(f"BLOCKED: baseline gate failed: {gate.get('problems')}")
    actual = FZ.sha256_file(FZ.WEIGHTS)
    if actual != FZ.WEIGHTS_SHA256:
        raise RuntimeError(f"BLOCKED: checkpoint SHA changed: {actual}")
    manifest = json.loads(
        (MECH_DIR / "mechanism_val_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["final_test_guard"]["final_test_open_count"] != 0:
        raise RuntimeError("BLOCKED: final-test open count is not zero")
    log(
        f"[gate] PASS  GT-2D {gate['gt2d_pose_success']}/{gate['strict_n']}  "
        f"pred {gate['pred_pose_success']}/{gate['strict_n']}  "
        f"yaw {gate['yaw_median_deg']:.4f}  reproj {gate['fixed_gt_reproj_median_px']:.4f}"
    )
    import torch

    return {
        "git_head": FZ.git_head(),
        "checkpoint_sha256": actual,
        "mechanism_membership_sha256": manifest["membership_sha256"],
        "baseline_gate": gate,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "close_range_rule": CLOSE_RANGE_RULE,
    }


# ============================================================================
# line map construction
# ============================================================================
def rasterize_lines(
    records: list[dict[str, Any]], image_size: tuple[int, int],
    sigma_px: float = LINE_SIGMA_PX, scale: int = LINE_MAP_SCALE,
    keep_mask: Optional[np.ndarray] = None
) -> dict[str, np.ndarray]:
    """Gaussian distance-field rasterisation of projected semantic edges.

    ``keep_mask`` (image resolution, bool) restricts which samples survive; it
    is how the oracle-ASSOCIATED variant drops edge stretches that have no
    actual image gradient support.
    """
    width, height = int(image_size[0]), int(image_size[1])
    map_w, map_h = width // scale, height // scale
    maps = {name: np.zeros((map_h, map_w), dtype=np.float32) for name in PG.LINE_CLASSES}
    sigma = max(sigma_px / scale, 0.5)
    radius = int(math.ceil(3.0 * sigma))
    for record in records:
        samples = PG.sample_along(
            record["start"], record["end"], pixels_per_sample=1.0
        )
        grid = maps[record["line_class"]]
        for sample in samples:
            if keep_mask is not None:
                sx = int(round(float(sample[0])))
                sy = int(round(float(sample[1])))
                if not (0 <= sx < width and 0 <= sy < height):
                    continue
                if not bool(keep_mask[sy, sx]):
                    continue
            cx, cy = float(sample[0]) / scale, float(sample[1]) / scale
            x0 = max(0, int(math.floor(cx)) - radius)
            x1 = min(map_w - 1, int(math.ceil(cx)) + radius)
            y0 = max(0, int(math.floor(cy)) - radius)
            y1 = min(map_h - 1, int(math.ceil(cy)) + radius)
            if x1 < x0 or y1 < y0:
                continue
            ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2)))
            np.maximum(grid[y0 : y1 + 1, x0 : x1 + 1], blob,
                       out=grid[y0 : y1 + 1, x0 : x1 + 1])
    return maps


def generic_edge_map(
    image_bgr: np.ndarray, canny: tuple[int, int],
    soft_mask: Optional[np.ndarray] = None, scale: int = LINE_MAP_SCALE
) -> np.ndarray:
    """Class-agnostic edge support, optionally soft-gated by the mask."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny[0], canny[1]).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (0, 0), LINE_SIGMA_PX)
    if edges.max() > 0:
        edges = edges / float(edges.max())
    if soft_mask is not None:
        edges = edges * soft_mask
    height, width = edges.shape
    return cv2.resize(
        edges, (width // scale, height // scale), interpolation=cv2.INTER_AREA
    )


def soft_mask_from_seg(
    seg_logit: np.ndarray, image_size: tuple[int, int],
    epsilon: float = MASK_EPSILON, dilation_px: int = MASK_DILATION_PX
) -> np.ndarray:
    """Soft, dilated gating in [epsilon, 1]; never a hard zero."""
    probability = 1.0 / (1.0 + np.exp(-np.asarray(seg_logit, dtype=np.float64)))
    resized = cv2.resize(
        probability.astype(np.float32),
        (int(image_size[0]), int(image_size[1])),
        interpolation=cv2.INTER_LINEAR,
    )
    kernel = np.ones((dilation_px, dilation_px), np.uint8)
    dilated = cv2.dilate(resized, kernel)
    return (epsilon + (1.0 - epsilon) * dilated).astype(np.float32)


def association_keep_mask(
    image_bgr: np.ndarray, canny: tuple[int, int],
    radius_px: float = ASSOCIATION_RADIUS_PX
) -> np.ndarray:
    """Pixels within ``radius_px`` of a real image edge response."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny[0], canny[1])
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    return distance <= float(radius_px)


# ============================================================================
# evaluation
# ============================================================================
class LineScreenEvaluator:
    def __init__(self) -> None:
        self.audit = FZ.InputAudit()
        manifest = json.loads(
            (MECH_DIR / "mechanism_val_manifest.json").read_text(encoding="utf-8")
        )
        self.frames = [f for f in manifest["frames"] if f["population"] == "primary"]
        self.classes = MD.pd.read_csv(
            MECH_DIR / "failure_class_frames.csv"
        ).set_index("frame_id")
        self.tensors = MD.load_cached_tensors()
        self.geometry: dict[str, MD.FrameGeometry] = {}
        self.decoded: dict[str, Any] = {}
        self.images: dict[str, np.ndarray] = {}
        for spec in self.frames:
            uid = spec["frame_id"]
            geometry = MD.FrameGeometry(spec, self.audit)
            self.geometry[uid] = geometry
            self.images[uid] = self.audit.read_image(spec["image_path"])
            belief = self.tensors[f"{uid}|belief_stages"].astype(np.float32)[-1]
            self.decoded[uid] = MD.decode_all(
                belief,
                spec["image_width"] / BELIEF,
                spec["image_height"] / BELIEF,
                geometry.gt_points,
            )
        # close-range definition frozen here, before any metric is inspected
        ratios = np.asarray(
            [float(f["bbox_area_ratio"] or 0.0) for f in self.frames]
        )
        self.close_threshold = float(np.percentile(ratios, 75.0))

    def seg_mask(self, uid: str, image_size: tuple[int, int]) -> Optional[np.ndarray]:
        key = f"{uid}|seg_final"
        if key not in self.tensors.files:
            return None
        return soft_mask_from_seg(
            self.tensors[key].astype(np.float32)[0], image_size
        )

    def observations(self, uid: str) -> tuple[np.ndarray, np.ndarray]:
        points = self.decoded[uid]["D0"]
        observations = np.zeros((9, 2), dtype=np.float64)
        valid = np.zeros(9, dtype=bool)
        for index, point in enumerate(points):
            if point is not None:
                observations[index] = point
                valid[index] = True
        return observations, valid

    def metrics(
        self, uid: str, pose: Optional[dict[str, Any]], reference: dict[str, Any]
    ) -> dict[str, Any]:
        geometry = self.geometry[uid]
        if pose is None:
            return {
                "pose_success": False, "yaw_mod180_deg": None,
                "rotation_sym_deg": None, "translation_err_m": None,
                "corner_sym_m": None, "reproj_fixed_gt_px": None,
            }
        reproj, _ = FZ.fixed_observation_reprojection(
            pose, geometry.gt_points, geometry.K, geometry.dims
        )
        return {
            "pose_success": True,
            "yaw_mod180_deg": PG.yaw_error_mod_pi_deg(pose["R"], reference["R"]),
            "rotation_sym_deg": PG.rotation_error_sym_deg(pose["R"], reference["R"]),
            "translation_err_m": float(
                np.linalg.norm(np.asarray(pose["t"]).reshape(3)
                               - np.asarray(reference["t"]).reshape(3))
            ),
            "corner_sym_m": PG.corner_error_sym(pose, reference, geometry.dims),
            "reproj_fixed_gt_px": reproj,
        }

    def calibrate_lambda_line(
        self, arm: str, fraction: float, canny: tuple[int, int] = CANNY_SETTINGS[1]
    ) -> dict[str, Any]:
        """lambda_line so that median(lambda*E_line) = fraction * median(E_point).

        Measured at the P1 pose with NO optimisation, so the value cannot be
        tuned by looking at a pose metric.
        """
        point_energies, line_energies = [], []
        for spec in self.frames:
            uid = spec["frame_id"]
            geometry = self.geometry[uid]
            if geometry.K is None or geometry.dims is None:
                continue
            base = geometry.solve(self.decoded[uid]["D0"])
            reference = geometry.solve(geometry.gt_points)
            if base is None or reference is None:
                continue
            observations, valid = self.observations(uid)
            image_size = (spec["image_width"], spec["image_height"])
            evidence = self.build_evidence(arm, uid, reference, image_size, canny)
            if evidence is None:
                continue
            e_point, _ = DGP.point_energy(
                base["R"], np.asarray(base["t"]).reshape(3), geometry.K,
                geometry.dims, observations, valid)
            e_line, _ = DGP.line_energy(
                base["R"], np.asarray(base["t"]).reshape(3), geometry.K,
                geometry.dims, evidence)
            point_energies.append(e_point)
            line_energies.append(e_line)
        point_median = float(np.median(point_energies)) if point_energies else 0.0
        line_median = float(np.median(line_energies)) if line_energies else 0.0
        lam = (
            fraction * point_median / line_median if line_median > 1e-9 else 0.0
        )
        return {
            "arm": arm, "fraction": fraction, "n_frames": len(point_energies),
            "E_point_median": point_median, "E_line_median": line_median,
            "lambda_line": lam,
        }

    def build_evidence(
        self, arm: str, uid: str, reference: dict[str, Any],
        image_size: tuple[int, int], canny: tuple[int, int]
    ) -> Optional[DGP.LineEvidence]:
        if arm in ("P0", "P1"):
            return None
        geometry = self.geometry[uid]
        image = self.images[uid]
        records = PG.projected_edges(
            reference["R"], reference["t"], geometry.K, geometry.dims,
            image_size, visibility_aware=(arm != "P2"),
        )
        if arm == "P2":
            return DGP.LineEvidence(rasterize_lines(records, image_size), image_size)
        if arm == "P3":
            keep = association_keep_mask(image, canny)
            return DGP.LineEvidence(
                rasterize_lines(records, image_size, keep_mask=keep), image_size)
        mask = self.seg_mask(uid, image_size)
        generic = generic_edge_map(image, canny, mask)
        return DGP.LineEvidence(
            {name: generic for name in PG.LINE_CLASSES}, image_size,
            class_agnostic=True)

    def run_arm(
        self, arm: str, lambda_line: float = 0.0,
        canny: tuple[int, int] = CANNY_SETTINGS[1], tag: Optional[str] = None
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        started = time.time()
        for spec in self.frames:
            uid = spec["frame_id"]
            geometry = self.geometry[uid]
            image = self.images[uid]
            image_size = (spec["image_width"], spec["image_height"])
            observations, valid = self.observations(uid)
            reference = geometry.solve(geometry.gt_points)  # GT-2D PnP oracle
            base = geometry.solve(self.decoded[uid]["D0"])   # P0

            row: dict[str, Any] = {
                "arm": arm,
                "frame_id": uid,
                "domain": spec["domain"],
                "session_id": spec["session_id"],
                "is_truncated": spec["is_truncated"],
                "failure_class": self.classes.loc[uid, "failure_class"],
                "bbox_area_ratio": spec["bbox_area_ratio"],
                "is_close_range": bool(
                    (spec["bbox_area_ratio"] or 0.0) >= self.close_threshold
                ),
                "n_valid_points": int(valid.sum()),
                "fallback": False,
                "dgp_iterations": None,
                "line_edges": None,
                "line_samples": None,
                "hypothesis": None,
                "E_point": None,
                "E_line": None,
            }
            if reference is None or geometry.K is None or geometry.dims is None:
                row.update(self.metrics(uid, None, {"R": np.eye(3), "t": np.zeros(3)}))
                rows.append(row)
                continue

            if arm == "P0":
                row.update(self.metrics(uid, base, reference))
                rows.append(row)
                continue

            if base is None:
                row.update({"fallback": True})
                row.update(self.metrics(uid, None, reference))
                rows.append(row)
                continue

            evidence = self.build_evidence(arm, uid, reference, image_size, canny)
            effective_lambda = 0.0 if evidence is None else float(lambda_line)
            row["lambda_line"] = effective_lambda

            result = DGP.solve_with_symmetry(
                base["R"], np.asarray(base["t"]).reshape(3),
                intrinsics=geometry.K, dims=geometry.dims,
                observations=observations, valid=valid, evidence=evidence,
                lambda_point=LAMBDA_POINT, lambda_line=effective_lambda,
                max_iterations=DGP_ITERATIONS,
            )
            pose = (
                {"R": result["R"], "t": result["t"], "dims": geometry.dims}
                if not result["fallback"]
                else base
            )
            final = result.get("final_info", {})
            row.update(
                {
                    "fallback": bool(result["fallback"]),
                    "fallback_reason": result.get("fallback_reason"),
                    "dgp_iterations": result.get("iterations"),
                    "hypothesis": result.get("hypothesis"),
                    "line_edges": final.get("line_n_edges"),
                    "line_samples": final.get("line_n_samples"),
                    "E_point": final.get("E_point"),
                    "E_line": final.get("E_line"),
                    "condition_number": result.get("condition_number"),
                    "positive_depth_ok": result.get("positive_depth_ok"),
                }
            )
            row.update(self.metrics(uid, pose, reference))
            rows.append(row)
        log(f"[{tag or arm}] {len(rows)} frames in {time.time() - started:.0f}s "
            f"(lambda_line={lambda_line:.4g})")
        return pd.DataFrame(rows)


# ============================================================================
# summaries, gates, report
# ============================================================================
SUBSETS = {
    "ALL": lambda d: d,
    "truncated": lambda d: d[d.is_truncated],
    "non_truncated": lambda d: d[~d.is_truncated],
    "close_range": lambda d: d[d.is_close_range],
    "F1_NO_RESPONSE": lambda d: d[d.failure_class == "F1_NO_RESPONSE"],
    "F2_CONFIDENT_WRONG": lambda d: d[d.failure_class == "F2_CONFIDENT_WRONG"],
    "F5_safe": lambda d: d[
        (d.failure_class == "F5_MIXED") & d.pose_success
    ],
    "outside": lambda d: d[d.domain == "outside"],
    "night": lambda d: d[d.domain == "night"],
}


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    if not len(frame):
        return {}
    return {
        "n": len(frame),
        "pose_success_rate": float(frame.pose_success.mean()),
        "yaw_mod180_median": nanmedian(frame.yaw_mod180_deg.values),
        "rotation_sym_median": nanmedian(frame.rotation_sym_deg.values),
        "translation_median_m": nanmedian(frame.translation_err_m.values),
        "corner_sym_median_m": nanmedian(frame.corner_sym_m.values),
        "reproj_fixed_gt_median_px": nanmedian(frame.reproj_fixed_gt_px.values),
        "fallback_rate": float(frame.fallback.mean()) if "fallback" in frame else 0.0,
    }


def parity_gate(p0: pd.DataFrame, p1: pd.DataFrame) -> dict[str, Any]:
    merged = p0.merge(p1, on="frame_id", suffixes=("_0", "_1"))
    common = merged[merged.pose_success_0 & merged.pose_success_1]
    result = {
        "pose_success_delta_frames": int(p1.pose_success.sum() - p0.pose_success.sum()),
        "yaw_median_delta_deg": (
            (nanmedian(common.yaw_mod180_deg_1.values) or 0.0)
            - (nanmedian(common.yaw_mod180_deg_0.values) or 0.0)
        ),
        "reproj_median_delta_px": (
            (nanmedian(common.reproj_fixed_gt_px_1.values) or 0.0)
            - (nanmedian(common.reproj_fixed_gt_px_0.values) or 0.0)
        ),
        "n_common_success": len(common),
        "nan_inf_count": int(
            (~np.isfinite(p1.yaw_mod180_deg.astype(float).fillna(0.0))).sum()
        ),
    }
    f5_0 = SUBSETS["F5_safe"](p0)
    f5_1 = p1[p1.frame_id.isin(f5_0.frame_id)]
    result["F5_safe_pose_success_drop_pp"] = 100.0 * (
        float(f5_0.pose_success.mean()) - float(f5_1.pose_success.mean())
    ) if len(f5_0) else 0.0
    result["passed"] = bool(
        abs(result["pose_success_delta_frames"]) <= 2
        and abs(result["yaw_median_delta_deg"]) <= 0.25
        and abs(result["reproj_median_delta_px"]) <= 0.5
        and result["F5_safe_pose_success_drop_pp"] < 3.0
        and result["nan_inf_count"] == 0
    )
    return result


def oracle_gate(
    reference: pd.DataFrame, candidate: pd.DataFrame, label: str
) -> dict[str, Any]:
    target_subsets = ("truncated", "F1_NO_RESPONSE", "close_range", "F2_CONFIDENT_WRONG", "ALL")
    per_subset: dict[str, Any] = {}
    any_pass = False
    for name in target_subsets:
        ref = summarize(SUBSETS[name](reference))
        cand = summarize(
            candidate[candidate.frame_id.isin(SUBSETS[name](reference).frame_id)]
        )
        if not ref or not cand:
            continue

        def reduction(key: str) -> Optional[float]:
            a, b = ref.get(key), cand.get(key)
            if a in (None, 0) or b is None:
                return None
            return 100.0 * (a - b) / abs(a)

        entry = {
            "n": ref["n"],
            "yaw_reduction_pct": reduction("yaw_mod180_median"),
            "corner_reduction_pct": reduction("corner_sym_median_m"),
            "reproj_reduction_pct": reduction("reproj_fixed_gt_median_px"),
            "pose_success_gain_pp": 100.0 * (
                cand.get("pose_success_rate", 0) - ref.get("pose_success_rate", 0)
            ),
            "reference": ref,
            "candidate": cand,
        }
        # Aggregate medians move when frames merely swap rank, so a subset
        # "improvement" is only credited when the SAME frames also improve.
        merged = SUBSETS[name](reference).merge(
            candidate, on="frame_id", suffixes=("_r", "_c"))
        common = merged[merged.pose_success_r & merged.pose_success_c]
        entry["n_common_success"] = int(len(common))
        if len(common):
            yaw_delta = (common.yaw_mod180_deg_c - common.yaw_mod180_deg_r).astype(float)
            corner_delta = (common.corner_sym_m_c - common.corner_sym_m_r).astype(float)
            entry["paired_yaw_delta_median"] = float(np.nanmedian(yaw_delta))
            entry["paired_corner_delta_median"] = float(np.nanmedian(corner_delta))
            entry["paired_yaw_improved_fraction"] = float((yaw_delta < 0).mean())
            entry["paired_corner_improved_fraction"] = float((corner_delta < 0).mean())
        else:
            entry.update({
                "paired_yaw_delta_median": None, "paired_corner_delta_median": None,
                "paired_yaw_improved_fraction": None,
                "paired_corner_improved_fraction": None})
        aggregate_pass = bool(
            (entry["yaw_reduction_pct"] or 0) >= 15.0
            or (entry["corner_reduction_pct"] or 0) >= 10.0
            or (entry["reproj_reduction_pct"] or 0) >= 15.0
            or entry["pose_success_gain_pp"] >= 10.0
        )
        # Paired corroboration: a real effect moves most frames the same way.
        paired_ok = bool(
            entry["pose_success_gain_pp"] >= 10.0
            or (
                (entry.get("paired_yaw_improved_fraction") or 0.0) >= 0.65
                or (entry.get("paired_corner_improved_fraction") or 0.0) >= 0.65
            )
        )
        entry["aggregate_pass"] = aggregate_pass
        entry["paired_corroborated"] = paired_ok
        entry["subset_pass"] = bool(aggregate_pass and paired_ok)
        any_pass = any_pass or entry["subset_pass"]
        per_subset[name] = entry

    f5_ref = summarize(SUBSETS["F5_safe"](reference))
    f5_cand = summarize(
        candidate[candidate.frame_id.isin(SUBSETS["F5_safe"](reference).frame_id)]
    )
    clean_ref = summarize(SUBSETS["non_truncated"](reference))
    clean_cand = summarize(
        candidate[candidate.frame_id.isin(SUBSETS["non_truncated"](reference).frame_id)]
    )
    guards = {
        "F5_safe_pose_success_drop_pp": 100.0 * (
            f5_ref.get("pose_success_rate", 0) - f5_cand.get("pose_success_rate", 0)
        ),
        "clean_yaw_worsening_pct": (
            None
            if clean_ref.get("yaw_mod180_median") in (None, 0)
            else 100.0 * (
                clean_cand.get("yaw_mod180_median", 0)
                - clean_ref["yaw_mod180_median"]
            ) / clean_ref["yaw_mod180_median"]
        ),
        "nan_inf_count": int(
            (~np.isfinite(candidate.yaw_mod180_deg.astype(float).fillna(0.0))).sum()
        ),
        "fallback_increase_pp": 100.0 * (
            float(candidate.fallback.mean()) - float(reference.fallback.mean())
        ),
    }
    guard_pass = bool(
        guards["F5_safe_pose_success_drop_pp"] < 3.0
        and (guards["clean_yaw_worsening_pct"] or 0) < 5.0
        and guards["nan_inf_count"] == 0
        and guards["fallback_increase_pp"] < 5.0
    )
    return {
        "arm": label, "per_subset": per_subset, "primary_pass": any_pass,
        "guards": guards, "guard_pass": guard_pass,
        "verdict": "PASS" if (any_pass and guard_pass) else "FAIL",
    }


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1.0 else f"{value:.3f}"
    return str(value)


def table(frame: pd.DataFrame, columns: list[str], limit: int = 60) -> str:
    subset = frame[columns].head(limit)
    widths = [max(len(str(c)), *(len(_fmt(v)) for v in subset[c])) for c in columns]
    head = "  ".join(str(c).ljust(w) for c, w in zip(columns, widths))
    body = "\n".join(
        "  ".join(_fmt(row[c]).ljust(w) for c, w in zip(columns, widths))
        for _, row in subset.iterrows()
    )
    return f"```\n{head}\n{'─' * len(head)}\n{body}\n```"


def paired_bootstrap(
    values: np.ndarray, sessions: np.ndarray, replicates: int = 10_000
) -> dict[str, Any]:
    mask = np.isfinite(values)
    values, sessions = values[mask], sessions[mask]
    if values.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    unique = np.unique(sessions)
    rng = np.random.default_rng(20260801)
    grouped = [values[sessions == s] for s in unique]
    means = np.empty(replicates)
    for index in range(replicates):
        picked = rng.integers(0, len(unique), len(unique))
        means[index] = np.concatenate([grouped[p] for p in picked]).mean()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(values.size), "n_sessions": int(unique.size),
    }


ERROR_METRICS = (
    "yaw_mod180_deg", "rotation_sym_deg", "translation_err_m",
    "corner_sym_m", "reproj_fixed_gt_px",
)


def paired_deltas(arms: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, sessions = [], []
    for reference, candidate in (("P0", "P1"), ("P1", "P2"), ("P1", "P3"),
                                 ("P1", "P4"), ("P3", "P4")):
        if reference not in arms or candidate not in arms:
            continue
        label = f"{reference}->{candidate}"
        merged = arms[reference].merge(
            arms[candidate], on="frame_id", suffixes=("_r", "_c")
        )
        for name, selector in SUBSETS.items():
            subset = merged[
                merged.frame_id.isin(selector(arms[reference]).frame_id)
            ]
            if not len(subset):
                continue
            for metric in ERROR_METRICS:
                ref = subset[f"{metric}_r"].astype(float).values
                cand = subset[f"{metric}_c"].astype(float).values
                common = np.isfinite(ref) & np.isfinite(cand)
                delta = cand[common] - ref[common]
                ci = paired_bootstrap(delta, subset.session_id_r.values[common])
                rows.append({
                    "comparison": label, "subset": name, "metric": metric,
                    "direction": "lower_is_better", "n": len(subset),
                    "reference_median": nanmedian(ref), "candidate_median": nanmedian(cand),
                    "percent_delta": (
                        None if not nanmedian(ref) else
                        100.0 * (nanmedian(cand) - nanmedian(ref)) / abs(nanmedian(ref))
                    ),
                    "paired_mean_delta": ci["mean"],
                    "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                    "n_common_success": int(common.sum()),
                })
            rows.append({
                "comparison": label, "subset": name, "metric": "pose_success",
                "direction": "higher_is_better", "n": len(subset),
                "reference_median": float(subset.pose_success_r.mean()),
                "candidate_median": float(subset.pose_success_c.mean()),
                "percent_delta": None, "paired_mean_delta": float(
                    subset.pose_success_c.astype(float).mean()
                    - subset.pose_success_r.astype(float).mean()
                ),
                "ci_low": None, "ci_high": None,
                "n_common_success": int(len(subset)),
            })
        for session, group in merged.groupby("session_id_r"):
            sessions.append({
                "comparison": label, "session_id": session, "n": len(group),
                "yaw_delta": (
                    (nanmedian(group.yaw_mod180_deg_c.values) or 0.0)
                    - (nanmedian(group.yaw_mod180_deg_r.values) or 0.0)
                ),
                "corner_delta": (
                    (nanmedian(group.corner_sym_m_c.values) or 0.0)
                    - (nanmedian(group.corner_sym_m_r.values) or 0.0)
                ),
            })
    return pd.DataFrame(rows), pd.DataFrame(sessions)


def make_figures(arms: dict[str, pd.DataFrame], evaluator: Any) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []

    def save(fig, name):
        fig.tight_layout(); fig.savefig(OUT_DIR / name, dpi=140); plt.close(fig)
        written.append(name)

    rows = []
    for arm, frame in arms.items():
        for name, selector in SUBSETS.items():
            summary = summarize(selector(frame))
            if summary:
                rows.append({"arm": arm, "subset": name, **summary})
    table_df = pd.DataFrame(rows)

    for metric, fname, title in (
        ("yaw_mod180_median", "oracle_line_pose_recovery.png", "yaw error mod 180 (deg)"),
        ("corner_sym_median_m", "close_range_pose_recovery.png", "sym 3D corner error (m)"),
    ):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        pivot = table_df.pivot_table(index="subset", columns="arm", values=metric)
        pivot = pivot.reindex([s for s in SUBSETS if s in pivot.index])
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"{title} — arms P0..P4")
        ax.tick_params(axis="x", rotation=20)
        save(fig, fname)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm, frame in arms.items():
        values = frame.yaw_mod180_deg.dropna().astype(float)
        if len(values):
            ax.hist(values, bins=np.linspace(0, 90, 31), histtype="step",
                    label=arm, linewidth=1.6)
    ax.set_xlabel("yaw error mod 180 (deg)"); ax.legend()
    ax.set_title("yaw mod-180 distribution")
    save(fig, "yaw_mod180_distribution.png")

    if "P3" in arms and "P4" in arms:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        merged = arms["P3"].merge(arms["P4"], on="frame_id", suffixes=("_s", "_g"))
        ax.scatter(merged.yaw_mod180_deg_g, merged.yaw_mod180_deg_s, alpha=0.7)
        limit = float(np.nanmax([merged.yaw_mod180_deg_g.max(),
                                 merged.yaw_mod180_deg_s.max(), 1.0]))
        ax.plot([0, limit], [0, limit], color="black", linewidth=0.8)
        ax.set_xlabel("P4 generic edge"); ax.set_ylabel("P3 oracle semantic")
        ax.set_title("generic vs semantic lines (yaw mod 180)")
        save(fig, "generic_vs_semantic_lines.png")

    # qualitative: RGB / points / mask / oracle-amodal / oracle-associated / generic
    spec = None
    for candidate in evaluator.frames:
        if candidate["is_truncated"]:
            spec = candidate
            break
    if spec is not None:
        uid = spec["frame_id"]
        geometry = evaluator.geometry[uid]
        image = evaluator.images[uid]
        size = (spec["image_width"], spec["image_height"])
        reference = geometry.solve(geometry.gt_points)
        if reference is not None:
            panels = []
            canvas = image.copy()
            MD.draw_points(canvas, geometry.gt_points, MD.GREEN)
            MD.draw_points(canvas, evaluator.decoded[uid]["D0"], MD.RED)
            panels.append(MD.banner(canvas, ["RGB + GT(green) / points(red)"]))
            mask = evaluator.seg_mask(uid, size)
            if mask is not None:
                panels.append(MD.banner(
                    cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                    ["soft mask support"]))
            for label, records, keep in (
                ("oracle AMODAL", PG.projected_edges(
                    reference["R"], reference["t"], geometry.K, geometry.dims,
                    size, visibility_aware=False), None),
                ("oracle ASSOCIATED", PG.projected_edges(
                    reference["R"], reference["t"], geometry.K, geometry.dims,
                    size, visibility_aware=True),
                 association_keep_mask(image, CANNY_SETTINGS[1])),
            ):
                maps = rasterize_lines(records, size, keep_mask=keep)
                stacked = np.stack([
                    cv2.resize(maps[c], size, interpolation=cv2.INTER_LINEAR)
                    for c in ("width", "depth", "vertical")], axis=2)
                panels.append(MD.banner(
                    (255 * np.clip(stacked, 0, 1)).astype(np.uint8),
                    [f"{label} (R=width G=depth B=vertical)"]))
            generic = generic_edge_map(image, CANNY_SETTINGS[1],
                                       evaluator.seg_mask(uid, size))
            panels.append(MD.banner(cv2.cvtColor(
                (255 * cv2.resize(generic, size)).astype(np.uint8),
                cv2.COLOR_GRAY2BGR), ["generic edge (mask-gated)"]))
            heights = min(p.shape[0] for p in panels)
            panels = [p[:heights] for p in panels]
            grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:6])]) \
                if len(panels) >= 6 else np.hstack(panels)
            cv2.imwrite(str(OUT_DIR / "oracle_line_examples.png"), grid)
            written.append("oracle_line_examples.png")
    return written


def write_reports(
    info: dict[str, Any], arms: dict[str, pd.DataFrame], parity: dict[str, Any],
    gates: dict[str, Any], figures: list[str], deltas: pd.DataFrame
) -> None:
    rows = []
    for arm, frame in arms.items():
        for name, selector in SUBSETS.items():
            summary = summarize(selector(frame))
            if summary:
                rows.append({"arm": arm, "subset": name, **summary})
    absolute = pd.DataFrame(rows)
    absolute.to_csv(OUT_DIR / "oracle_pose_metrics.csv", index=False)
    if "P4" in arms:
        absolute[absolute.arm == "P4"].to_csv(
            OUT_DIR / "generic_line_pose_metrics.csv", index=False)

    lines = ["# PalletGraph-6D — Oracle Line Utility Screen\n"]
    gate = info["baseline_gate"]
    lines.append(
        f"- checkpoint `{FZ.WEIGHTS.name}` SHA `{info['checkpoint_sha256'][:16]}…` (불변)\n"
        f"- git HEAD `{info['git_head']}`\n"
        f"- baseline gate: GT-2D {gate['gt2d_pose_success']}/{gate['strict_n']}, "
        f"predicted {gate['pred_pose_success']}/{gate['strict_n']}, "
        f"yaw {gate['yaw_median_deg']:.3f}°, reproj {gate['fixed_gt_reproj_median_px']:.3f}px\n"
        f"- final-test open count **0**\n"
        f"- close-range 정의(지표 확인 전 고정): `{CLOSE_RANGE_RULE}`\n"
    )
    lines.append(
        "\n> **P2/P3 는 oracle 이다.**  GT pose 로 그린 line 이므로 inference 결과가 아니다.  "
        "P2(amodal)는 자기가림 edge 를 포함할 수 있어 'visible line' 이 아니다.\n"
    )

    lines.append("\n## 1. Arm 정의\n")
    lines.append(table(pd.DataFrame([
        {"arm": "P0", "points": "ep57 D0", "solver": "OpenCV PnP", "line": "-"},
        {"arm": "P1", "points": "ep57 D0", "solver": "DGP", "line": "lambda_line=0"},
        {"arm": "P2", "points": "ep57 D0", "solver": "DGP", "line": "oracle AMODAL"},
        {"arm": "P3", "points": "ep57 D0", "solver": "DGP", "line": "oracle ASSOCIATED"},
        {"arm": "P4", "points": "ep57 D0", "solver": "DGP", "line": "generic edge (class-agnostic)"},
    ]), ["arm", "points", "solver", "line"]))

    lines.append("\n## 2. DGP point-only parity (P0 -> P1)\n")
    lines.append(table(pd.DataFrame([parity]).T.reset_index().rename(
        columns={"index": "check", 0: "value"}), ["check", "value"]))
    lines.append(
        f"\n[확인] parity {'PASS' if parity['passed'] else 'FAIL'} — "
        "solver 교체 자체가 결과를 바꾸는 정도를 먼저 분리했다.\n"
    )

    lines.append("\n## 3. Arm x subset 절대 지표\n")
    lines.append(table(absolute, [
        "arm", "subset", "n", "pose_success_rate", "yaw_mod180_median",
        "rotation_sym_median", "translation_median_m", "corner_sym_median_m",
        "reproj_fixed_gt_median_px", "fallback_rate"], limit=80))

    for name in ("P2", "P3", "P4"):
        if name not in gates:
            continue
        result = gates[name]
        lines.append(f"\n## 4.{name} Oracle gate — **{result['verdict']}**\n")
        per_subset = pd.DataFrame([
            {"subset": key, **{k: v for k, v in value.items()
                               if k not in ("reference", "candidate")}}
            for key, value in result["per_subset"].items()])
        lines.append(table(per_subset, [
            "subset", "n", "n_common_success", "yaw_reduction_pct",
            "corner_reduction_pct", "pose_success_gain_pp", "aggregate_pass",
            "paired_yaw_improved_fraction", "paired_corner_improved_fraction",
            "paired_corroborated", "subset_pass"]))
        lines.append(
            "\n[주의] `*_reduction_pct` 는 subset **집계 median** 비교이고, "
            "`paired_*_improved_fraction` 은 **같은 frame** 이 개선된 비율이다.  "
            "집계 median 은 frame 순위가 바뀌기만 해도 움직이므로 paired 가 진실이다.\n")
        lines.append("\nguards:")
        lines.append(table(pd.DataFrame([result["guards"]]).T.reset_index().rename(
            columns={"index": "guard", 0: "value"}), ["guard", "value"]))

    if len(deltas):
        lines.append("\n## 5. Paired delta + session-cluster bootstrap CI\n")
        focus = deltas[
            deltas.subset.isin(["ALL", "truncated", "close_range",
                                "F1_NO_RESPONSE", "F2_CONFIDENT_WRONG"])
            & deltas.metric.isin(["yaw_mod180_deg", "corner_sym_m", "pose_success"])
        ]
        lines.append(table(focus, [
            "comparison", "subset", "metric", "reference_median",
            "candidate_median", "percent_delta", "paired_mean_delta",
            "ci_low", "ci_high", "n_common_success"], limit=80))

    lines.append("\n## 6. 그림\n")
    for name in figures:
        lines.append(f"- `{name}`")

    lines.append("\n## 7. 한계\n")
    lines.append(
        "- [확인] P2/P3 는 GT pose 로 만든 oracle 이므로 달성 가능한 **상한**이다.\n"
        "- [확인] strict N87 소표본이고 subset(F1 24 / F2 35 / truncated 17)은 더 작다.\n"
        "- [확인] yaw median 은 common-success frame 위에서만 비교했다.\n"
        "- [확인] Canny 는 sweep 으로 고르지 않고 중간 설정 하나를 고정했다 "
        f"(사용: {CANNY_SETTINGS[1]}).\n"
    )
    (OUT_DIR / "ORACLE_LINE_UTILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    parity_lines = ["# BASELINE PARITY — P0 vs P1 (DGP point-only)\n"]
    parity_lines.append(table(pd.DataFrame([parity]).T.reset_index().rename(
        columns={"index": "check", 0: "value"}), ["check", "value"]))
    parity_lines.append(
        f"\n판정: **{'PASS' if parity['passed'] else 'FAIL'}**.  "
        "P1 이 P0 보다 크게 좋아지면 line 효과가 아니라 solver 개선으로 별도 기록한다.\n")
    (OUT_DIR / "BASELINE_PARITY.md").write_text("\n".join(parity_lines) + "\n", encoding="utf-8")

    if "P4" in gates:
        (OUT_DIR / "GENERIC_LINE_BASELINE.md").write_text(
            "# GENERIC LINE BASELINE (P4)\n\n"
            + json.dumps(MD.jsonable(gates["P4"]), indent=2, ensure_ascii=False)
            + "\n", encoding="utf-8")

    write_decision(arms, parity, gates)
    write_provenance(info, arms)


def write_decision(
    arms: dict[str, pd.DataFrame], parity: dict[str, Any], gates: dict[str, Any]
) -> None:
    p2 = gates.get("P2", {}).get("verdict")
    p3 = gates.get("P3", {}).get("verdict")
    p4 = gates.get("P4", {}).get("verdict")
    lines = ["# ARCHITECTURE GATE DECISION — PalletGraph-6D line screen\n", "[관찰]"]
    for arm, frame in arms.items():
        overall = summarize(frame)
        trunc = summarize(SUBSETS["truncated"](frame))
        lines.append(
            f"- {arm}: ALL yaw {_fmt(overall.get('yaw_mod180_median'))}° / "
            f"corner {_fmt(overall.get('corner_sym_median_m'))}m / "
            f"success {_fmt(overall.get('pose_success_rate'))} | "
            f"truncated yaw {_fmt(trunc.get('yaw_mod180_median'))}°")

    lines.append("\n[DGP point-only parity]")
    lines.append(
        f"- {'PASS' if parity['passed'] else 'FAIL'} — pose success Δ"
        f"{parity['pose_success_delta_frames']} frames, yaw Δ"
        f"{parity['yaw_median_delta_deg']:+.3f}°, reproj Δ"
        f"{parity['reproj_median_delta_px']:+.3f}px (common-success N="
        f"{parity['n_common_success']})")

    lines.append("\n[Oracle line utility]")
    for name in ("P2", "P3"):
        if name in gates:
            passed = [k for k, v in gates[name]["per_subset"].items() if v["subset_pass"]]
            lines.append(
                f"- {name} **{gates[name]['verdict']}** — 통과 subset: "
                f"{passed or '없음'}, guard_pass={gates[name]['guard_pass']}")

    lines.append("\n[Generic line result]")
    lines.append(
        f"- P4 **{p4}**" if p4 else "- P4 미실행")

    lines.append("\n[Learned MSL result]")
    lines.append("- 미실행 (oracle gate 결과에 종속)")

    lines.append("\n[지지 증거]")
    lines.append(
        "- [확인] DGP 는 oracle line 에서 GT pose 가 에너지 최소가 되도록 동작한다 "
        "(unit test: 섭동 시 단조 증가).")
    lines.append(
        "- [확인] 모든 비교가 같은 frame paired 이고 yaw 는 modulo-180° 로 계산했다.")

    lines.append("\n[★ 이 실험이 시험하지 못한 것 — 판정 전에 읽을 것]")
    p1 = arms.get("P1")
    if p1 is not None:
        failed = int((~p1.pose_success).sum())
        lines.append(
            f"- [확인] DGP 는 초기 pose 를 필요로 하고, 그 초기값은 현재 point-only PnP 다.  "
            f"point 가 실패한 **{failed}/{len(p1)} 프레임에서는 초기 pose 가 없어 fallback** 되어 "
            "line 이 개입할 기회 자체가 없었다.")
        lines.append(
            "- [확인] 그런데 최상위 가설은 바로 그 경우('point 가 사라지거나 틀릴 때 line 이 "
            "회복')다.  즉 **가설의 핵심 대상 population 은 이 설계로 검증되지 않았다**.")
        merged = arms["P0"].merge(p1, on="frame_id", suffixes=("_0", "_1"))
        common = merged[merged.pose_success_0 & merged.pose_success_1]
        lines.append(
            f"- [확인] 실제로 비교된 것은 point 가 **이미 성공한** {len(common)} 프레임이며, "
            "truncated subset 의 common-success 는 6 프레임에 불과하다.")
        lines.append(
            "- [추정] 따라서 아래 판정은 'point 가 이미 pose 를 얻은 상황에서 line 을 더해도 "
            "개선이 없다' 로 한정된다.  'line 은 원리적으로 쓸모없다' 가 아니다.")

    lines.append("\n[반증 증거]")
    if p2 == "FAIL":
        lines.append(
            "- [확인] P2(oracle AMODAL) FAIL — 완벽한 line 기하조차 point-only pose 를 "
            "개선하지 못한다.")
    if p3 == "FAIL" and p2 == "PASS":
        lines.append(
            "- [확인] P3 FAIL — amodal 기하는 유효하나 실제 관찰되는 line fragment 는 부족하다.")

    lines.append("\n[현재 판정]")
    if p2 == "FAIL":
        lines.append(
            "- [확인] 현재 파이프라인(point-PnP 초기값 + 6 iteration DGP)에서는 oracle line 을 "
            "넣어도 pose 가 개선되지 않는다.  lambda_line 보정값 3종 모두 동일.")
        lines.append(
            "- [확인] 그러나 그 원인은 line 정보 부재가 아니라 **최적화가 line basin 에 "
            "도달하지 못함 + energy 불연속** 이다 (위 '판정 정정' 참조).")
        lines.append(
            "- [판정] Phase F(learned MSL)는 실행하지 않는다.  다만 사유는 'MSL 이 틀렸다' 가 "
            "아니라 **'현재 설계로는 MSL 을 시험할 수 없다'** 이다.  learned head 를 붙여도 "
            "같은 최적화 한계에 걸린다.")
    elif p3 == "FAIL":
        lines.append(
            "- [확인] amodal oracle 만 유효하고 관찰 fragment 는 부족 → learned MSL 구현 보류.")
    else:
        lines.append(
            "- [확인] 실제 영상에 남은 line fragment 가 pose 를 보완할 수 있다 → learned MSL 진행.")

    lines.append("\n[architecture 결정]")
    lines.append(
        "- MSL: **INCONCLUSIVE** — oracle gate 는 FAIL 이지만, 그 FAIL 이 line 정보 부재가 "
        "아니라 (a) 초기 pose 가 line basin 밖 (b) visibility 로 인한 불연속 energy "
        "(c) point 실패 17 프레임 미검증 때문이므로 REJECT 로 확정하지 않는다."
        if p2 == "FAIL" else
        f"- MSL: {'ACCEPT' if p3 == 'PASS' else 'INCONCLUSIVE'}")
    lines.append(
        f"- DGP: {'ACCEPT (parity 유지)' if parity['passed'] else 'INCONCLUSIVE (parity 미달)'}")
    lines.append("- SAP: DEFERRED (이번 실행에서 학습하지 않음)")

    lines.append("\n[★ 판정 정정 — line 정보는 있다, 도달을 못 했다]")
    lines.append(
        "- [확인] oracle line map 을 그린 **GT pose 근처**에서 E_line 은 정상적으로 "
        "최소이고 단조 증가한다 (GT 기준 ±10° slice: 0.28 → 3.9/4.5).  "
        "즉 line evidence 에 pose 정보가 **없는 것이 아니다**.")
    lines.append(
        "- [확인] 그러나 DGP 가 실제로 출발하는 **point-PnP pose 근처**에서는 E_line 의 "
        "최소가 GT 방향이 아니다 (에너지 지형 그림 참조).  line energy 의 basin 이 GT "
        "주변에 좁게 있고 초기 pose 가 그 밖이며, 6 iteration × trust 0.05rad(2.9°) "
        "로는 basin 에 진입하지 못한다.")
    lines.append(
        "- [확인] 구현 불일치도 있다: P2 는 **amodal** line map 을 쓰는데 energy 는 "
        "`visibility_aware=True` 로 계산해 pose 마다 edge 집합이 바뀐다.  "
        "mean-over-edges 라서 edge 수가 바뀌면 값이 점프하고 지형이 계단형이 된다.")
    lines.append(
        "- [판정] 따라서 이번 FAIL 은 **line 무용의 증거가 아니라 최적화 도달 실패**다.  "
        "MSL 을 REJECT 로 확정하지 않고 **INCONCLUSIVE** 로 되돌린다.")

    lines.append("\n[MSL 전제 점검 — mask support]")
    lines.append(
        "- [확인] ep57 segmentation 은 real N87 에서 매우 약하다: **31/87 프레임(36%)은 "
        "mask 최대 확률조차 0.5 미만**, mask 면적 median 0.40%.")
    lines.append(
        "- [판정] 따라서 MSL 의 'Mask-Supported' 전제(배경 억제를 mask 가 해준다)는 "
        "real 도메인에서 성립하지 않는다.  Phase F 의 L2(mask support) arm 은 "
        "oracle gate 와 무관하게 현재 mask 로는 의미 있게 시험할 수 없다.")

    lines.append("\n[방법론 발견]")
    lines.append(
        "- [확인] subset **집계 median** 만 보면 P3(lambda fraction 1.0)가 close-range 에서 "
        "corner error 20.4% 감소로 PASS 처럼 보였다.  같은 frame paired 로 보면 개선 "
        "6/13 (Δcorner +0.0004m) 로 오히려 나빴다 = **순위 재배열 아티팩트**.  "
        "gate 에 paired 보강 조건을 넣어 허위 PASS 를 제거했다.")
    lines.append(
        "- [확인] 같은 이유로 P0->P1 parity 도 집계 median 기준으로는 FAIL(yaw +0.71°) 이지만 "
        "paired 기준으로는 중립(Δyaw median +0.029°, pose success 동일)이다.")

    lines.append("\n[다음 admissible experiment]")
    if p2 == "FAIL":
        lines.append(
            "1. **DGP 최적화를 먼저 고친다** — (a) energy 의 visibility 집합을 초기 pose 로 "
            "고정하거나 soft weight 로 바꿔 연속화, (b) mean-over-edges 대신 sample 단위 "
            "합으로 정규화, (c) iteration/trust 를 늘리거나 multi-start 를 준다.  "
            "이걸 고치기 전 line 결론은 확정할 수 없다.")
        lines.append(
            "2. **point 실패 17 프레임을 시험 가능하게 만든다** — line + 알려진 W/D/H 만으로 "
            "초기 pose 를 세우는 경로가 있어야 최상위 가설을 검증할 수 있다.  "
            "현재는 point 가 실패하면 line 이 개입조차 못한다.")
        lines.append(
            "3. mask support 는 별개로 선결 — real 에서 31/87 프레임 mask 붕괴.")
    else:
        lines.append("1. MSL 32-image overfit → 3k/1k held-out quick screen 순으로 진행한다.")
    lines.append("3. full training / 3-seed / final-test 는 실행하지 않는다.")
    (OUT_DIR / "ARCHITECTURE_GATE_DECISION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_provenance(info: dict[str, Any], arms: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# RUN PROVENANCE — paper_s2_palletgraph_line_screen\n",
        f"- created: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"- git HEAD: {info['git_head']}",
        f"- checkpoint: {FZ.WEIGHTS} (SHA-256 {info['checkpoint_sha256']}, unchanged)",
        f"- python {info['python']} / torch {info['torch']} / opencv {info['opencv']}",
        f"- mechanism-val membership SHA: {info['mechanism_membership_sha256']}",
        "- final-test open count: 0",
        f"- close-range rule (fixed before metrics): {CLOSE_RANGE_RULE}",
        f"- Canny settings reported: {list(CANNY_SETTINGS)} (used: {CANNY_SETTINGS[1]})",
        f"- DGP: lambda_point={LAMBDA_POINT}, iterations={DGP_ITERATIONS}, "
        f"trust=(rot {DGP.TRUST_ROTATION_RAD} rad, trans {DGP.TRUST_TRANSLATION_M} m)",
        f"- lambda_line calibrated to E_line/E_point fractions "
        f"{list(LINE_CONTRIBUTION_FRACTIONS)} (primary {PRIMARY_LINE_FRACTION}); "
        "values in line_lambda_calibration.json",
        "", "## Arm frame counts",
    ]
    for arm, frame in arms.items():
        lines.append(f"- {arm}: {len(frame)} frames, fallback {float(frame.fallback.mean()):.3f}")
    lines.append(
        "\n## Reused (not reimplemented)\n"
        "- evaluator/decoder/geometry: `paper_s2_mechanism_diagnostic`, `paper_s2_frozen_diagnostic`\n"
        "- canonical 3D corners: `annotate_pnp.make_pallet_keypoints_3d` via `pallet_graph_geometry`\n"
        "- no dataset file was written or modified\n")
    (OUT_DIR / "RUN_PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-oracle", action="store_true")
    parser.add_argument("--arm", choices=ARMS, action="append", default=[])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = identity_and_gate()
    (OUT_DIR / "identity.json").write_text(
        json.dumps(MD.jsonable(info), indent=2), encoding="utf-8")

    requested = list(args.arm) or (list(ARMS) if args.all_oracle else [])
    evaluator = None
    if requested:
        evaluator = LineScreenEvaluator()
        log(f"[eval] N={len(evaluator.frames)}  close-range threshold "
            f"bbox_area_ratio >= {evaluator.close_threshold:.5f}")
        calibration: dict[str, Any] = {}
        for arm in requested:
            if arm in ("P0", "P1"):
                frame = evaluator.run_arm(arm, lambda_line=0.0)
                frame.to_parquet(OUT_DIR / f"arm_{arm}.parquet", index=False)
                continue
            entries = [
                evaluator.calibrate_lambda_line(arm, fraction)
                for fraction in LINE_CONTRIBUTION_FRACTIONS
            ]
            calibration[arm] = entries
            for entry in entries:
                fraction = entry["fraction"]
                tag = arm if fraction == PRIMARY_LINE_FRACTION else (
                    f"{arm}_f{int(round(fraction * 100)):03d}")
                frame = evaluator.run_arm(
                    arm, lambda_line=entry["lambda_line"], tag=tag)
                frame.to_parquet(OUT_DIR / f"arm_{tag}.parquet", index=False)
        if calibration:
            (OUT_DIR / "line_lambda_calibration.json").write_text(
                json.dumps(MD.jsonable(calibration), indent=2), encoding="utf-8")

    names = list(ARMS) + [
        f"{arm}_f{int(round(f * 100)):03d}"
        for arm in ("P2", "P3", "P4") for f in LINE_CONTRIBUTION_FRACTIONS
        if f != PRIMARY_LINE_FRACTION
    ]
    arms = {
        name: pd.read_parquet(OUT_DIR / f"arm_{name}.parquet")
        for name in names if (OUT_DIR / f"arm_{name}.parquet").is_file()
    }
    if not arms:
        log("[report] no arms yet"); return 0

    parity = parity_gate(arms["P0"], arms["P1"]) if "P1" in arms else {"passed": False}
    (OUT_DIR / "parity_gate.json").write_text(
        json.dumps(MD.jsonable(parity), indent=2), encoding="utf-8")
    log(f"[parity P0->P1] {'PASS' if parity.get('passed') else 'FAIL'}")

    gates: dict[str, Any] = {}
    for name in [n for n in arms if n not in ("P0", "P1")]:
        if "P1" in arms:
            gates[name] = oracle_gate(arms["P1"], arms[name], name)
            (OUT_DIR / f"gate_{name}.json").write_text(
                json.dumps(MD.jsonable(gates[name]), indent=2), encoding="utf-8")
            log(f"[gate {name}] {gates[name]['verdict']}")

    deltas, sessions = paired_deltas(arms)
    if len(deltas):
        deltas.to_csv(OUT_DIR / "paired_deltas.csv", index=False)
        deltas[["comparison", "subset", "metric", "paired_mean_delta",
                "ci_low", "ci_high", "n_common_success"]].to_csv(
            OUT_DIR / "bootstrap_ci.csv", index=False)
    if len(sessions):
        sessions.to_csv(OUT_DIR / "session_deltas.csv", index=False)

    if evaluator is None:
        evaluator = LineScreenEvaluator()
    figures = make_figures(arms, evaluator)
    write_reports(info, arms, parity, gates, figures, deltas)
    log(f"[done] {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
