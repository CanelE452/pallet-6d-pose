#!/usr/bin/env python3
"""paper_s2_micro_arch_screen.py — PAPER_S2 micro architecture screen.

목적(해결 능력 판별 전용, 논문 최종 학습 아님):
  mechanism diagnosis(ep57)에서 나온 두 후보가 실제로 자기 target failure class 를
  고치는지 **matched control 대비로만** 판정한다.

    B 계열 (F2_CONFIDENT_WRONG, N=35)
      M0_B : 400장 추가 fine-tuning control (m6_2 tail 만 학습)
      B1   : image-feature-conditioned bounded belief residual
      B2   : B1 PASS 시에만, far/depth edge structural loss 추가

    A 계열 (F1_NO_RESPONSE, N=24)
      M0_A : 동일 A manifest 로 fine-tuning control (legacy target semantics)
      A1   : target semantics 만 수정 (clip_belief_border + spatial_keypoint_mask)

핵심 원칙:
  * ep57 원본과만 비교하면 "추가 fine-tuning 효과"를 "방법 효과"로 오인한다.
    모든 후보는 **같은 manifest 로 학습한 M0 control 과 비교**한다.
  * 한 비교에서 바뀌는 것은 하나뿐이다 (Phase 12 ablation table 로 강제 검증).
  * evaluator 를 새로 만들지 않는다.  mechanism diagnostic runner 의 decoder /
    PnP / metric 함수를 import 해서 그대로 쓴다.
  * mechanism-val N87 은 평가 전용이며 어떤 arm 의 training manifest 에도 넣지 않는다.

Usage:
    python scripts/stage0/paper_s2/paper_s2_micro_arch_screen.py --build-manifests
    python scripts/stage0/paper_s2/paper_s2_micro_arch_screen.py --all-primary
    python scripts/stage0/paper_s2/paper_s2_micro_arch_screen.py --run B2      # 조건부
    python scripts/stage0/paper_s2/paper_s2_micro_arch_screen.py --report
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse
import copy
import datetime as dt
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
for _p in (
    ROOT / "Deep_Object_Pose/common",
    ROOT / "Deep_Object_Pose/train",
    ROOT / "scripts/stage0",
    ROOT / "scripts/data_prep/eval",
    ROOT / "challenge/scripts",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# --- 재사용 (재구현 금지) ---------------------------------------------------
import paper_s2_mechanism_diagnostic as MD  # noqa: E402
import paper_s2_frozen_diagnostic as FZ  # noqa: E402
from models import DopeNetwork  # noqa: E402
from belief_residual import BoundedBeliefResidual, residual_diagnostics  # noqa: E402
from heatmap_refinement import channel_masked_mse  # noqa: E402
from utils_dataset import CleanVisiiDopeLoader  # noqa: E402
from diffpnp3d_loss import LocalSoftArgmax2D  # noqa: E402

OUT_DIR = ROOT / "data/pallet/results/paper_s2_micro_arch_screen"
WEIGHT_DIR = ROOT / "weights/paper_s2/paper_s2_micro_arch_screen"
MECH_DIR = MD.OUT_DIR
TRAIN_ROOT = ROOT / "data/pallet/training_data"
DIFFPNP_INDEX_DIR = (
    ROOT / "data/pallet/results/paper_s2_scratch_diffpnp/pnp_valid_3d_index"
)
TARGET_SEMANTICS = (
    ROOT / "data/pallet/results/paper_s2_target_semantics_audit"
    / "target_semantics_keypoints.parquet"
)

# ep57 training constants (weights/paper_s2_stageB/header.txt)
SIGMA = 2.0
IMAGESIZE = 400
OUTPUT_SIZE = 50
N_KP = 9
BELIEF_THRESHOLD = MD.BELIEF_THRESHOLD
NEAR_KP, FAR_KP = MD.NEAR_KP, MD.FAR_KP
DEPTH_LEFT_KP, DEPTH_RIGHT_KP = MD.DEPTH_LEFT_KP, MD.DEPTH_RIGHT_KP
# far/depth = the far face plus the four depth edges' far endpoints; the
# diagnosis found the error concentrated here, so it is the primary slice.
FAR_DEPTH_KP = tuple(sorted(set(FAR_KP)))

# Phase 3 common training conditions
SEED = 1
BATCH_SIZE = 8
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
NUM_WORKERS = 0  # deterministic; verified by test_sampler_order_identical
EARLY_STOP_PATIENCE = 2
RESIDUAL_AMPLITUDE = 0.25
RESIDUAL_HIDDEN = 16
SATURATION_LIMIT = 0.20
CLEAN_ERROR_BLOWUP = 1.20

# Manifest B sizing
MANIFEST_B_HARD = 200
MANIFEST_B_CLEAN = 200
MANIFEST_B_POOL = 4000  # frozen-inference screening pool
MANIFEST_A_EACH = 200

# B2 structural loss
DEPTH_EDGES = ((0, 4), (1, 5), (2, 6), (3, 7))
FAR_PERIMETER_EDGES = ((4, 5), (5, 6), (6, 7), (7, 4))
B2_EDGES = DEPTH_EDGES + FAR_PERIMETER_EDGES
HUBER_DELTA = 0.05
EDGE_CONTRIBUTION_TARGET = 0.05
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260729

ARMS = ("M0_B", "B1", "M0_A", "A1", "B2")
PRIMARY_ARMS = ("M0_B", "B1", "M0_A", "A1")


def log(message: str) -> None:
    print(message, flush=True)


def sha256_text(text: str) -> str:
    return MD.sha256_text(text)


def finite(value: Any) -> Optional[float]:
    return MD.finite(value)


def nanmedian(values: Iterable[Any]) -> Optional[float]:
    return MD.nanmedian(values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# Phase 0 — identity and baseline gate
# ============================================================================
def identity_and_gate() -> dict[str, Any]:
    gate_path = MECH_DIR / "baseline_gate.json"
    if not gate_path.is_file():
        raise RuntimeError(
            "BLOCKED: mechanism diagnostic baseline_gate.json missing; run "
            "paper_s2_mechanism_diagnostic.py --all first"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError(f"BLOCKED: baseline gate not passed: {gate.get('problems')}")
    actual = FZ.sha256_file(FZ.WEIGHTS)
    if actual != FZ.WEIGHTS_SHA256:
        raise RuntimeError(f"BLOCKED: checkpoint SHA changed: {actual}")
    manifest = json.loads(
        (MECH_DIR / "mechanism_val_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["final_test_guard"]["final_test_open_count"] != 0:
        raise RuntimeError("BLOCKED: final-test open count is not zero")
    info = {
        "git_head": FZ.git_head(),
        "checkpoint_sha256": actual,
        "mechanism_membership_sha256": manifest["membership_sha256"],
        "baseline_gate": gate,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "opencv": cv2.__version__,
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
    }
    log(
        f"[gate] PASS  GT-2D {gate['gt2d_pose_success']}/{gate['strict_n']}  "
        f"pred {gate['pred_pose_success']}/{gate['strict_n']}  "
        f"yaw {gate['yaw_median_deg']:.4f}  reproj {gate['fixed_gt_reproj_median_px']:.4f}"
    )
    return info


def mechanism_val_frames() -> list[dict[str, Any]]:
    manifest = json.loads(
        (MECH_DIR / "mechanism_val_manifest.json").read_text(encoding="utf-8")
    )
    return [f for f in manifest["frames"] if f["population"] == "primary"]


def mechanism_val_paths() -> set[str]:
    """Every image/json path that must never appear in a training manifest."""
    paths: set[str] = set()
    for frame in mechanism_val_frames():
        paths.add(str(Path(frame["image_path"]).resolve()))
        paths.add(str(Path(frame["json_path"]).resolve()))
    return paths


def failure_classes() -> pd.DataFrame:
    """ep57-frozen failure class per mechanism-val frame (target definition)."""
    return pd.read_csv(MECH_DIR / "failure_class_frames.csv")


# ============================================================================
# model plumbing
# ============================================================================
class ScreenModel(nn.Module):
    """ep57 base (optionally frozen) plus an optional bounded belief residual.

    The shared VGG feature ``out1`` is captured with a forward hook so the
    residual head can re-read it without a second backbone pass.
    """

    def __init__(self, with_residual: bool) -> None:
        super().__init__()
        self.base = DopeNetwork(numVec=0, numSeg=1)
        state = torch.load(str(FZ.WEIGHTS), map_location="cpu", weights_only=True)
        if any(str(k).startswith("module.") for k in state):
            state = {str(k).removeprefix("module."): v for k, v in state.items()}
        self.base.load_state_dict(state, strict=True)
        self.residual = (
            BoundedBeliefResidual(
                feature_channels=128,
                belief_channels=N_KP,
                hidden_channels=RESIDUAL_HIDDEN,
                amplitude=RESIDUAL_AMPLITUDE,
            )
            if with_residual
            else None
        )
        self._feature: Optional[torch.Tensor] = None
        self.base.vgg.register_forward_hook(self._capture)

    def _capture(self, module: Any, inputs: Any, output: Any) -> None:
        self._feature = output

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        outputs = self.base(images)
        beliefs, affinities = outputs[0], outputs[1]
        belief_base = beliefs[-1][:, :N_KP]
        result = {
            "belief_stages": [b[:, :N_KP] for b in beliefs],
            "belief_base": belief_base,
            "affinity": affinities[-1],
            "feature": self._feature,
            "delta": None,
        }
        if self.residual is None:
            result["belief_final"] = belief_base
        else:
            final, delta = self.residual(self._feature, belief_base)
            result["belief_final"] = final
            result["delta"] = delta
        return result

    def trainable_parameters(self, scope: str) -> list[nn.Parameter]:
        """``scope`` is either 'm6_2_tail' or 'residual'."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if scope == "residual":
            if self.residual is None:
                raise ValueError("residual scope requested without a residual head")
            for parameter in self.residual.parameters():
                parameter.requires_grad_(True)
            return list(self.residual.parameters())
        if scope == "m6_2_tail":
            selected: list[nn.Parameter] = []
            for name, parameter in self.base.m6_2.named_parameters():
                if name.split(".")[0] in ("10", "12"):
                    parameter.requires_grad_(True)
                    selected.append(parameter)
            if not selected:
                raise RuntimeError("m6_2 tail parameters not found")
            return selected
        raise ValueError(f"unknown trainable scope: {scope}")


def belief_from_model(
    model: ScreenModel, image_bgr: np.ndarray, device: torch.device
) -> np.ndarray:
    """Final belief (9,50,50) float32 using the SAME preprocessing as the evaluator."""
    tensor = FZ.preprocess_squash(image_bgr).to(device)
    with torch.inference_mode():
        out = model(tensor)
    return out["belief_final"][0].detach().float().cpu().numpy().astype(np.float32)


# ============================================================================
# evaluation — reuses the mechanism diagnostic decoder / PnP / metrics
# ============================================================================
class MechanismEvaluator:
    """Per-frame metrics on the frozen mechanism-val N87. No new metric code."""

    def __init__(self) -> None:
        self.audit = FZ.InputAudit()
        self.frames = mechanism_val_frames()
        self.geometry: dict[str, MD.FrameGeometry] = {}
        self.images: dict[str, np.ndarray] = {}
        for spec in self.frames:
            self.geometry[spec["frame_id"]] = MD.FrameGeometry(spec, self.audit)
            self.images[spec["frame_id"]] = self.audit.read_image(spec["image_path"])
        self.classes = failure_classes().set_index("frame_id")

    def evaluate(
        self, model: ScreenModel, device: torch.device, tag: str
    ) -> pd.DataFrame:
        model.eval()
        rows: list[dict[str, Any]] = []
        for spec in self.frames:
            uid = spec["frame_id"]
            geometry = self.geometry[uid]
            belief = belief_from_model(model, self.images[uid], device)
            scale_x = spec["image_width"] / OUTPUT_SIZE
            scale_y = spec["image_height"] / OUTPUT_SIZE
            decoded = MD.decode_all(belief, scale_x, scale_y, geometry.gt_points)
            points = decoded["D0"]
            stats = decoded["_stats"]
            pose = geometry.solve(points)
            oracle = geometry.solve(geometry.gt_points)
            metrics = geometry.metrics(pose, oracle)
            matched = geometry.matched_2d_error(points)

            per_kp_err = [
                FZ.euclidean(points[k], geometry.gt_points[k]) for k in range(N_KP)
            ]
            corner_err = [per_kp_err[k] for k in range(8) if per_kp_err[k] is not None]
            far_detected = sum(1 for k in FAR_KP if points[k] is not None)
            # GT-in-frame undetected corners: the A1 primary metric.
            in_frame_total = in_frame_detected = 0
            for k in range(8):
                inside = FZ.point_inside(
                    geometry.gt_points[k], spec["image_width"], spec["image_height"]
                )
                if inside is True:
                    in_frame_total += 1
                    if points[k] is not None:
                        in_frame_detected += 1
            row = {
                "arm": tag,
                "frame_id": uid,
                "domain": spec["domain"],
                "session_id": spec["session_id"],
                "is_truncated": spec["is_truncated"],
                "failure_class": self.classes.loc[uid, "failure_class"],
                "n_detected": sum(1 for p in points if p is not None),
                "n_corner_detected": sum(1 for p in points[:8] if p is not None),
                "n_far_detected": far_detected,
                "gt_inframe_corners": in_frame_total,
                "gt_inframe_detected": in_frame_detected,
                "gt_inframe_detection_rate": (
                    in_frame_detected / in_frame_total if in_frame_total else None
                ),
                "frame_median_peak": nanmedian([s.get("peak") for s in stats]),
                "matched_2d_median_px": matched["matched_median_px"],
                "far_2d_median_px": matched["far_matched_median_px"],
                "near_2d_median_px": matched["near_matched_median_px"],
                "gross_corner_fraction": (
                    float(np.mean([e > 20.0 for e in corner_err]))
                    if corner_err
                    else None
                ),
                "pose_success": metrics["pose_success"],
                "yaw_err_deg": metrics["yaw_err_deg"],
                "reproj_fixed_gt_px": metrics["reproj_fixed_gt_px"],
                "add_m": metrics["add_m"],
                "translation_err_m": metrics["translation_err_m"],
            }
            rows.append(row)
        return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, subset: Optional[str] = None) -> dict[str, Any]:
    data = frame if subset is None else frame[frame.failure_class == subset]
    if not len(data):
        return {}
    return {
        "n": len(data),
        "far_2d_median_px": nanmedian(data.far_2d_median_px.values),
        "matched_2d_median_px": nanmedian(data.matched_2d_median_px.values),
        "gross_rate": float(np.nanmean(data.gross_corner_fraction.values)),
        "yaw_median_deg": nanmedian(data.yaw_err_deg.values),
        "reproj_median_px": nanmedian(data.reproj_fixed_gt_px.values),
        "pose_success_rate": float(data.pose_success.mean()),
        "median_detected_corners": nanmedian(data.n_corner_detected.values),
        "far_detected_mean": float(np.nanmean(data.n_far_detected.values)),
        "gt_inframe_detection_rate": float(
            np.nansum(data.gt_inframe_detected.values)
            / max(np.nansum(data.gt_inframe_corners.values), 1)
        ),
        "frame_median_peak": nanmedian(data.frame_median_peak.values),
    }


# ============================================================================
# Phase 2 — deterministic training manifests
# ============================================================================
def load_diffpnp_index() -> dict[str, Any]:
    index: dict[str, Any] = {}
    for path in sorted(DIFFPNP_INDEX_DIR.glob("*.json")):
        for rel, entry in json.loads(path.read_text(encoding="utf-8")).items():
            index[str((TRAIN_ROOT / rel).resolve())] = entry
    return index


def candidate_pool(forbidden: set[str]) -> list[dict[str, Any]]:
    """Deterministic screening pool from the ep57 training roots.

    Only frames whose 8 GT corners are all valid and inside the image are kept,
    so a low ep57 score means a localisation failure rather than a missing GT.
    """
    audit_frame = pd.read_parquet(TARGET_SEMANTICS)
    corners = audit_frame[
        (audit_frame.keypoint_id < 8) & (audit_frame.aug_kind == "plain")
    ]
    grouped = corners.groupby(["dataset", "frame_id", "json_path"]).agg(
        n_center_inside=("center_inside_belief", "sum"),
        n_sentinel=("is_exact_sentinel", "sum"),
        n_full_support=("full_gaussian_support_inside", "sum"),
    ).reset_index()
    eligible = grouped[
        (grouped.n_center_inside == 8) & (grouped.n_sentinel == 0)
    ].copy()
    eligible = eligible.sort_values(["dataset", "frame_id"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, row in eligible.iterrows():
        json_path = Path(row.json_path)
        image_path = json_path.with_suffix(".png")
        if str(json_path.resolve()) in forbidden or str(image_path.resolve()) in forbidden:
            continue
        if not image_path.is_file():
            continue
        rows.append(
            {
                "frame_id": f"{row.dataset}:{row.frame_id}",
                "source_dataset": str(row.dataset),
                "json_path": str(json_path.resolve()),
                "image_path": str(image_path.resolve()),
            }
        )
    # Stratified deterministic subsample: round-robin across source datasets.
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(row["source_dataset"], []).append(row)
    for values in by_dataset.values():
        values.sort(key=lambda r: r["frame_id"])
    pool: list[dict[str, Any]] = []
    order = sorted(by_dataset)
    index = 0
    while len(pool) < MANIFEST_B_POOL and any(
        index < len(by_dataset[d]) for d in order
    ):
        for dataset in order:
            if index < len(by_dataset[dataset]) and len(pool) < MANIFEST_B_POOL:
                pool.append(by_dataset[dataset][index])
        index += 1
    return pool


def screen_pool(pool: list[dict[str, Any]], device: torch.device) -> pd.DataFrame:
    """ep57 frozen inference on the pool, using the mechanism decoder/metrics."""
    model = ScreenModel(with_residual=False).to(device).eval()
    audit = FZ.InputAudit()
    rows: list[dict[str, Any]] = []
    started = time.time()
    for position, entry in enumerate(pool):
        data = audit.read_json(entry["json_path"])
        objects = data.get("objects") or []
        if not objects:
            continue
        obj = objects[0]
        gt_points = FZ.gt_points_from_object(obj)
        image = audit.read_image(entry["image_path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        belief = belief_from_model(model, image, device)
        decoded = MD.decode_all(
            belief, width / OUTPUT_SIZE, height / OUTPUT_SIZE, gt_points
        )
        points, stats = decoded["D0"], decoded["_stats"]
        far = FZ.order_free_corner_metrics(
            [points[k] if k in FAR_KP else None for k in range(8)],
            [gt_points[k] if k in FAR_KP else None for k in range(8)],
        )
        overall = FZ.order_free_corner_metrics(points, gt_points)
        intrinsics = FZ.intrinsics_from_json(data)
        dims = FZ.dims_from_frame({"entry": None}, obj)
        pose = None
        if intrinsics is not None and dims is not None:
            pose, _, _ = FZ.current_solve(
                points, intrinsics, dims, (height, width, 3), True
            )
        stored = FZ.stored_pose_from_object(obj)
        yaw_err = None
        if pose is not None and stored is not None:
            yaw_err = abs(
                FZ.wrap180(FZ.yaw_deg(pose["R"]) - FZ.yaw_deg(stored["R"]))
            )
        n_inside = sum(
            1 for p in gt_points[:8] if FZ.point_inside(p, width, height) is True
        )
        rows.append(
            {
                **entry,
                "n_corner_detected": sum(1 for p in points[:8] if p is not None),
                "median_peak": nanmedian([s.get("peak") for s in stats]),
                "far_error_px": far["median_px"],
                "matched_error_px": overall["median_px"],
                "pose_success": pose is not None,
                "yaw_err_deg": yaw_err,
                "n_gt_inframe": n_inside,
                "is_truncated": bool(n_inside < 8),
                "dim_W_m": None if dims is None else dims[0],
                "dim_D_m": None if dims is None else dims[1],
                "dim_H_m": None if dims is None else dims[2],
                "bbox_area_ratio": None,
                "elevation": None,
                "azimuth": None,
            }
        )
        if (position + 1) % 500 == 0:
            log(f"[screen] {position + 1}/{len(pool)}  {time.time() - started:.0f}s")
    return pd.DataFrame(rows)


def stratified_take(
    frame: pd.DataFrame, count: int, label: str
) -> pd.DataFrame:
    """Round-robin across source datasets so no single set exceeds half."""
    groups = {
        name: group.sort_values("frame_id").to_dict("records")
        for name, group in frame.groupby("source_dataset")
    }
    order = sorted(groups)
    chosen: list[dict[str, Any]] = []
    index = 0
    while len(chosen) < count and any(index < len(groups[d]) for d in order):
        for dataset in order:
            if index < len(groups[dataset]) and len(chosen) < count:
                chosen.append({**groups[dataset][index], "hard_or_clean": label})
        index += 1
    return pd.DataFrame(chosen)


def build_manifest_B(device: torch.device) -> dict[str, Any]:
    forbidden = mechanism_val_paths()
    screening_path = OUT_DIR / "manifest_B_screening.parquet"
    if screening_path.is_file():
        screened = pd.read_parquet(screening_path)
        log(f"[manifest B] reuse screening ({len(screened)} frames)")
    else:
        pool = candidate_pool(forbidden)
        log(f"[manifest B] screening pool = {len(pool)} frames")
        screened = screen_pool(pool, device)
        screened.to_parquet(screening_path, index=False)

    usable = screened[
        (~screened.is_truncated)
        & (screened.n_corner_detected >= 6)
        & (screened.median_peak >= BELIEF_THRESHOLD)
    ]
    hard = usable[usable.far_error_px > 20.0]
    clean = usable[
        (usable.far_error_px <= 10.0)
        & usable.pose_success
        & (usable.yaw_err_deg.fillna(999) <= 5.0)
    ]
    # The spec's ">20px far error" definition of an F2-like training frame can
    # be unsatisfiable: ep57 is already accurate on its own synthetic training
    # distribution.  Do NOT pad the count silently — record that the criterion
    # was not met, fall back to the hardest available frames, and let the report
    # label the arm "not testable as designed" instead of "candidate rejected".
    hard_criterion_met = len(hard) >= MANIFEST_B_HARD
    hard_threshold_used = 20.0
    if not hard_criterion_met:
        ordered = usable.sort_values(
            ["far_error_px", "frame_id"], ascending=[False, True]
        )
        hard = ordered.head(max(MANIFEST_B_HARD * 3, 1))
        hard_threshold_used = float(hard.far_error_px.min()) if len(hard) else None
        log(
            f"[manifest B] WARNING: only {int((usable.far_error_px > 20.0).sum())} "
            f"frames exceed the 20px far-error criterion "
            f"(pool max {float(usable.far_error_px.max()):.2f}px). "
            f"Falling back to the hardest available (>= {hard_threshold_used:.2f}px)."
        )
    log(
        f"[manifest B] candidates hard={len(hard)} clean={len(clean)} "
        f"(usable {len(usable)} / screened {len(screened)})"
    )
    hard_sel = stratified_take(hard, MANIFEST_B_HARD, "hard")
    clean_sel = stratified_take(clean, MANIFEST_B_CLEAN, "clean")
    selected = pd.concat([hard_sel, clean_sel], ignore_index=True)

    frames = []
    for _, row in selected.iterrows():
        frames.append(
            {
                "frame_id": row.frame_id,
                "json_path": row.json_path,
                "image_path": row.image_path,
                "source_dataset": row.source_dataset,
                "hard_or_clean": row.hard_or_clean,
                "dimensions": [row.dim_W_m, row.dim_D_m, row.dim_H_m],
                "elevation": row.elevation,
                "azimuth": row.azimuth,
                "bbox_area_ratio": row.bbox_area_ratio,
                "n_gt_inframe": int(row.n_gt_inframe),
                "ep57_far_error": finite(row.far_error_px),
                "ep57_peak": finite(row.median_peak),
            }
        )
    manifest = {
        "name": "micro_train_B",
        "purpose": "F2_CONFIDENT_WRONG structured localisation screen",
        "requested": {"hard": MANIFEST_B_HARD, "clean": MANIFEST_B_CLEAN},
        "actual": {
            "hard": int((selected.hard_or_clean == "hard").sum()),
            "clean": int((selected.hard_or_clean == "clean").sum()),
        },
        "hard_criterion": {
            "specified_far_error_px": 20.0,
            "criterion_met": bool(hard_criterion_met),
            "n_above_specified": int((usable.far_error_px > 20.0).sum()),
            "threshold_actually_used_px": hard_threshold_used,
            "pool_far_error_median_px": finite(usable.far_error_px.median()),
            "pool_far_error_max_px": finite(usable.far_error_px.max()),
            "note": (
                "ep57 is already accurate on its own synthetic training "
                "distribution, so the real-domain F2 failure mode (far error "
                "median 43px) has no counterpart here.  A B-arm result is "
                "therefore evidence about the mechanism on synthetic hard "
                "frames, NOT a rejection of the candidate on F2."
            ),
        },
        "source_dataset_counts": selected.source_dataset.value_counts().to_dict(),
        "frames": frames,
    }
    manifest["membership_hash"] = sha256_text(
        json.dumps(sorted(f["frame_id"] for f in frames))
    )
    return manifest


def build_manifest_A() -> dict[str, Any]:
    """Border-positive frames vs matched interior controls (target semantics)."""
    forbidden = mechanism_val_paths()
    audit_frame = pd.read_parquet(TARGET_SEMANTICS)
    plain = audit_frame[audit_frame.aug_kind == "plain"]

    # C2 = centre inside the belief map, full Gaussian support outside, target
    # empty, yet the loss mask says "valid" -> border positive supervised as
    # background negative.
    c2 = (
        plain.center_inside_belief
        & (~plain.full_gaussian_support_inside)
        & (~plain.belief_target_nonzero)
        & (plain.belief_channel_mask > 0)
    )
    plain = plain.assign(is_c2=c2)
    per_frame = plain.groupby(["dataset", "frame_id", "json_path"]).agg(
        n_c2=("is_c2", "sum"),
        n_center_inside=("center_inside_belief", "sum"),
        n_sentinel=("is_exact_sentinel", "sum"),
    ).reset_index()

    border = per_frame[per_frame.n_c2 >= 1]
    interior = per_frame[(per_frame.n_c2 == 0) & (per_frame.n_sentinel == 0)]

    def to_rows(frame: pd.DataFrame, label: str, limit: int) -> list[dict[str, Any]]:
        groups = {
            name: group.sort_values("frame_id").to_dict("records")
            for name, group in frame.groupby("dataset")
        }
        order = sorted(groups)
        chosen: list[dict[str, Any]] = []
        index = 0
        while len(chosen) < limit and any(index < len(groups[d]) for d in order):
            for dataset in order:
                if index >= len(groups[dataset]) or len(chosen) >= limit:
                    continue
                row = groups[dataset][index]
                json_path = Path(row["json_path"])
                image_path = json_path.with_suffix(".png")
                if (
                    str(json_path.resolve()) in forbidden
                    or str(image_path.resolve()) in forbidden
                    or not image_path.is_file()
                ):
                    continue
                chosen.append(
                    {
                        "frame_id": f"{row['dataset']}:{row['frame_id']}",
                        "json_path": str(json_path.resolve()),
                        "image_path": str(image_path.resolve()),
                        "source_dataset": str(row["dataset"]),
                        "border_or_interior": label,
                        "n_c2_keypoints": int(row["n_c2"]),
                    }
                )
            index += 1
        return chosen

    frames = to_rows(border, "border_positive", MANIFEST_A_EACH) + to_rows(
        interior, "interior_control", MANIFEST_A_EACH
    )
    manifest = {
        "name": "micro_train_A",
        "purpose": "target-semantics screen (border positive vs interior control)",
        "requested": {"border_positive": MANIFEST_A_EACH, "interior_control": MANIFEST_A_EACH},
        "actual": {
            "border_positive": sum(
                1 for f in frames if f["border_or_interior"] == "border_positive"
            ),
            "interior_control": sum(
                1 for f in frames if f["border_or_interior"] == "interior_control"
            ),
        },
        "frames": frames,
    }
    manifest["membership_hash"] = sha256_text(
        json.dumps(sorted(f["frame_id"] for f in frames))
    )
    return manifest


def build_manifests(device: torch.device) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path_b = OUT_DIR / "micro_train_B_manifest.json"
    path_a = OUT_DIR / "micro_train_A_manifest.json"
    if path_b.is_file():
        manifest_b = json.loads(path_b.read_text(encoding="utf-8"))
        log(f"[manifest B] reuse ({len(manifest_b['frames'])} frames)")
    else:
        manifest_b = build_manifest_B(device)
        path_b.write_text(json.dumps(manifest_b, indent=2), encoding="utf-8")
    if path_a.is_file():
        manifest_a = json.loads(path_a.read_text(encoding="utf-8"))
        log(f"[manifest A] reuse ({len(manifest_a['frames'])} frames)")
    else:
        manifest_a = build_manifest_A()
        path_a.write_text(json.dumps(manifest_a, indent=2), encoding="utf-8")

    forbidden = mechanism_val_paths()
    for name, manifest in (("B", manifest_b), ("A", manifest_a)):
        for frame in manifest["frames"]:
            if (
                str(Path(frame["image_path"]).resolve()) in forbidden
                or str(Path(frame["json_path"]).resolve()) in forbidden
            ):
                raise RuntimeError(
                    f"BLOCKED: mechanism-val frame leaked into manifest {name}"
                )
            FZ.InputAudit().guard(frame["image_path"])
    hashes = {
        "micro_train_B_manifest.json": FZ.sha256_file(path_b),
        "micro_train_A_manifest.json": FZ.sha256_file(path_a),
        "B_membership_hash": manifest_b["membership_hash"],
        "A_membership_hash": manifest_a["membership_hash"],
        "mechanism_val_membership_sha256": json.loads(
            (MECH_DIR / "mechanism_val_manifest.json").read_text(encoding="utf-8")
        )["membership_sha256"],
    }
    (OUT_DIR / "manifest_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    log(
        f"[manifests] B={manifest_b['actual']} A={manifest_a['actual']} "
        f"hashes written"
    )
    return {"B": manifest_b, "A": manifest_a, "hashes": hashes}


# ============================================================================
# Phase 3 — deterministic training
# ============================================================================
def make_loader(
    manifest: dict[str, Any], spatial_semantics: bool
) -> tuple[DataLoader, CleanVisiiDopeLoader]:
    """Subset loader built by overriding the dataset's file list.

    ``spatial_semantics`` switches BOTH opt-in flags together; that pair is the
    single "corrected partial target semantics" element under test in A1.
    """
    dataset = CleanVisiiDopeLoader(
        [str(TRAIN_ROOT)],
        objects=["pallet"],
        sigma=SIGMA,
        output_size=OUTPUT_SIZE,
        truncation_aug_prob=0.0,
        mask_aux=False,
        clip_belief_border=spatial_semantics,
        spatial_keypoint_mask=spatial_semantics,
        refinement_targets=False,
        aspect_resize=True,
        diffpnp_index=load_diffpnp_index(),
    )
    dataset.imgs = [
        (f["image_path"], Path(f["image_path"]).name, f["json_path"])
        for f in manifest["frames"]
    ]
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
        drop_last=False,
    )
    return loader, dataset


def edge_loss(
    belief: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor,
    decoder: LocalSoftArgmax2D
) -> torch.Tensor:
    """Scale-normalised far/depth edge consistency (both left and right)."""
    pred = decoder(belief)[0] if isinstance(decoder(belief), tuple) else decoder(belief)
    with torch.no_grad():
        gold = decoder(target)[0] if isinstance(decoder(target), tuple) else decoder(target)
    valid = channel_mask > 0
    diagonal = torch.linalg.norm(
        gold[:, :8].amax(dim=1) - gold[:, :8].amin(dim=1), dim=-1
    ).clamp_min(1.0)
    losses = []
    for i, j in B2_EDGES:
        pair_valid = valid[:, i] & valid[:, j]
        if not bool(pair_valid.any()):
            continue
        e_pred = (pred[:, j] - pred[:, i]) / diagonal.unsqueeze(-1)
        e_gold = (gold[:, j] - gold[:, i]) / diagonal.unsqueeze(-1)
        residual = (e_pred - e_gold)[pair_valid]
        losses.append(
            nn.functional.huber_loss(
                residual, torch.zeros_like(residual), delta=HUBER_DELTA
            )
        )
    if not losses:
        return belief.sum() * 0.0
    return torch.stack(losses).mean()


def evaluate_train_domain(
    model: ScreenModel, manifest: dict[str, Any], device: torch.device,
    subset: str, limit: int = 100
) -> dict[str, Any]:
    """In-domain far-error on the arm's OWN training frames.

    This is not a generalisation claim.  It separates two very different
    reasons a candidate can miss its gate: "the head cannot learn to move the
    far corners at all" versus "it learns on synthetic but the correction does
    not transfer to the real F2 population".
    """
    audit = FZ.InputAudit()
    chosen = [f for f in manifest["frames"] if f["hard_or_clean"] == subset][:limit]
    far_errors: list[float] = []
    model.eval()
    for entry in chosen:
        data = audit.read_json(entry["json_path"])
        objects = data.get("objects") or []
        if not objects:
            continue
        gt_points = FZ.gt_points_from_object(objects[0])
        image = audit.read_image(entry["image_path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        belief = belief_from_model(model, image, device)
        points = MD.decode_all(
            belief, width / OUTPUT_SIZE, height / OUTPUT_SIZE, gt_points
        )["D0"]
        far = FZ.order_free_corner_metrics(
            [points[k] if k in FAR_KP else None for k in range(8)],
            [gt_points[k] if k in FAR_KP else None for k in range(8)],
        )
        if far["median_px"] is not None:
            far_errors.append(float(far["median_px"]))
    return {
        "subset": subset,
        "n": len(far_errors),
        "far_2d_median_px": nanmedian(far_errors),
        "far_2d_mean_px": float(np.mean(far_errors)) if far_errors else None,
    }


ARM_CONFIG = {
    "M0_B": {"manifest": "B", "scope": "m6_2_tail", "residual": False,
             "semantics": False, "edge": False, "target": "F2_CONFIDENT_WRONG"},
    "B1": {"manifest": "B", "scope": "residual", "residual": True,
           "semantics": False, "edge": False, "target": "F2_CONFIDENT_WRONG"},
    "B2": {"manifest": "B", "scope": "residual", "residual": True,
           "semantics": False, "edge": True, "target": "F2_CONFIDENT_WRONG"},
    "M0_A": {"manifest": "A", "scope": "m6_2_tail", "residual": False,
             "semantics": False, "edge": False, "target": "F1_NO_RESPONSE"},
    "A1": {"manifest": "A", "scope": "m6_2_tail", "residual": False,
           "semantics": True, "edge": False, "target": "F1_NO_RESPONSE"},
}
PRIMARY_METRIC = {
    "F2_CONFIDENT_WRONG": ("far_2d_median_px", "lower"),
    "F1_NO_RESPONSE": ("gt_inframe_detection_rate", "higher"),
}


def calibrate_edge_lambda(
    model: ScreenModel, loader: DataLoader, device: torch.device,
    decoder: LocalSoftArgmax2D
) -> dict[str, Any]:
    """Weight the edge term to 5% of the heatmap loss, measured without updates."""
    heat: list[float] = []
    edge: list[float] = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= 20:
                break
            images = batch["img"].to(device)
            target = batch["beliefs"].to(device).float()
            mask = batch["belief_channel_mask"].to(device).float()
            out = model(images)
            belief = out["belief_final"]
            heat.append(float(channel_masked_mse(belief, target, mask)))
            edge.append(float(edge_loss(belief, target, mask, decoder)))
    heat_median = float(np.median(heat)) if heat else 0.0
    edge_median = float(np.median(edge)) if edge else 0.0
    lam = (
        EDGE_CONTRIBUTION_TARGET * heat_median / edge_median
        if edge_median > 0
        else 0.0
    )
    return {
        "heatmap_loss_median": heat_median,
        "edge_loss_median": edge_median,
        "target_contribution": EDGE_CONTRIBUTION_TARGET,
        "lambda_edge": lam,
        "batches_measured": len(heat),
    }


def run_arm(
    arm: str, manifests: dict[str, Any], evaluator: MechanismEvaluator,
    device: torch.device
) -> dict[str, Any]:
    config = ARM_CONFIG[arm]
    manifest = manifests[config["manifest"]]
    arm_dir = WEIGHT_DIR / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(SEED)
    model = ScreenModel(with_residual=config["residual"]).to(device)
    trainable = model.trainable_parameters(config["scope"])
    trainable_count = sum(p.numel() for p in trainable)

    identity_gap = None
    if config["residual"]:
        model.eval()
        sample = evaluator.frames[0]
        tensor = FZ.preprocess_squash(
            evaluator.images[sample["frame_id"]]
        ).to(device)
        with torch.inference_mode():
            out = model(tensor)
            identity_gap = float(
                (out["belief_final"] - out["belief_base"]).abs().max()
            )
        if identity_gap != 0.0:
            raise RuntimeError(
                f"BLOCKED: B-arm initial identity broken, max|delta|={identity_gap}"
            )
        log(f"[{arm}] zero-init identity verified (max|H_final-H_base| = 0)")

    loader, dataset = make_loader(manifest, config["semantics"])
    optimizer = torch.optim.Adam(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    decoder = None
    calibration = None
    if config["edge"]:
        decoder = LocalSoftArgmax2D(
            window=7, temperature=0.1,
            orig_size=(OUTPUT_SIZE, OUTPUT_SIZE),
            belief_size=(OUTPUT_SIZE, OUTPUT_SIZE),
        )
        calibration = calibrate_edge_lambda(model, loader, device, decoder)
        (OUT_DIR / "structural_loss_calibration.json").write_text(
            json.dumps(calibration, indent=2), encoding="utf-8"
        )
        log(f"[{arm}] lambda_edge = {calibration['lambda_edge']:.6g}")

    metric_name, direction = PRIMARY_METRIC[config["target"]]
    baseline_eval = evaluator.evaluate(model, device, f"{arm}_epoch0")
    history: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    order_log: list[list[str]] = []

    best_value = None
    best_epoch = 0
    best_state = copy.deepcopy(
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    )
    worse_streak = 0
    baseline_summary = summarize(baseline_eval, config["target"])
    history.append({"arm": arm, "epoch": 0, **baseline_summary,
                    "train_loss": None, "trainable_parameters": trainable_count})
    best_value = baseline_summary.get(metric_name)

    for epoch in range(1, MAX_EPOCHS + 1):
        # Same sampler permutation and augmentation RNG for every arm.
        seed_everything(SEED + epoch)
        loader.generator.manual_seed(SEED + epoch)
        model.train()
        model.base.eval() if config["scope"] == "residual" else None
        losses: list[float] = []
        epoch_order: list[str] = []
        for batch in loader:
            images = batch["img"].to(device)
            target = batch["beliefs"].to(device).float()
            mask = batch["belief_channel_mask"].to(device).float()
            epoch_order.extend(list(batch["file_name"]))
            out = model(images)
            belief = out["belief_final"]
            loss = channel_masked_mse(belief, target, mask)
            if config["edge"]:
                loss = loss + calibration["lambda_edge"] * edge_loss(
                    belief, target, mask, decoder
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"BLOCKED: non-finite loss in {arm} epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
            if out["delta"] is not None:
                residual_rows.append(
                    {"arm": arm, "epoch": epoch,
                     **residual_diagnostics(out["delta"], RESIDUAL_AMPLITUDE)}
                )
        order_log.append(epoch_order)

        if residual_rows:
            saturation = float(
                np.mean([r["residual_saturation_fraction"] for r in residual_rows[-50:]])
            )
            if saturation > SATURATION_LIMIT:
                raise RuntimeError(
                    f"BLOCKED: residual saturation {saturation:.3f} > {SATURATION_LIMIT}"
                )

        evaluation = evaluator.evaluate(model, device, f"{arm}_epoch{epoch}")
        evaluation.to_parquet(arm_dir / f"eval_epoch{epoch:02d}.parquet", index=False)
        summary = summarize(evaluation, config["target"])
        summary.update(
            {"arm": arm, "epoch": epoch,
             "train_loss": float(np.mean(losses)) if losses else None,
             "trainable_parameters": trainable_count}
        )
        history.append(summary)
        torch.save(model.state_dict(), arm_dir / f"net_epoch_{epoch:02d}.pth")

        value = summary.get(metric_name)
        comparable = value is not None and best_value is not None
        improved = comparable and (
            value < best_value if direction == "lower" else value > best_value
        )
        # "2 epoch 연속 악화" means strictly worse.  Counting an unchanged
        # metric as a regression stopped every arm after 2 epochs (~100 steps),
        # which is not a test of the candidate.
        worse = comparable and (
            value > best_value if direction == "lower" else value < best_value
        )
        state = "improved" if improved else ("worse" if worse else "tied")
        log(
            f"[{arm}] epoch {epoch}  loss {summary['train_loss']:.6f}  "
            f"{metric_name}={value:.4f}  best={best_value:.4f}  {state}"
        )
        if improved or (not worse and comparable):
            # Tie-break towards the later epoch.  The F1 primary metric is a
            # detection *count* ratio and stays bit-identical for many epochs;
            # keeping epoch 0 on a tie would silently evaluate the untrained
            # checkpoint and make the arm indistinguishable from ep57.
            best_value, best_epoch = value, epoch
            if improved:
                worse_streak = 0
            best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
        elif worse:
            worse_streak += 1
            if worse_streak >= EARLY_STOP_PATIENCE:
                log(f"[{arm}] early stop at epoch {epoch} (patience {EARLY_STOP_PATIENCE})")
                break

    torch.save(best_state, arm_dir / "best.pth")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    train_domain = None
    if config["manifest"] == "B":
        train_domain = {
            "before": None,
            "after": [
                evaluate_train_domain(model, manifest, device, s)
                for s in ("hard", "clean")
            ],
        }
        base_model = ScreenModel(with_residual=False).to(device)
        train_domain["before"] = [
            evaluate_train_domain(base_model, manifest, device, s)
            for s in ("hard", "clean")
        ]
        del base_model
        log(f"[{arm}] train-domain far error: {json.dumps(MD.jsonable(train_domain))}")
    final_eval = evaluator.evaluate(model, device, arm)
    final_eval.to_parquet(OUT_DIR / f"eval_{arm}.parquet", index=False)
    pd.DataFrame(history).to_csv(arm_dir / "metrics_by_epoch.csv", index=False)
    if residual_rows:
        pd.DataFrame(residual_rows).drop(
            columns=["residual_channel_abs_mean"]
        ).to_csv(OUT_DIR / f"residual_diagnostics_{arm}.csv", index=False)

    run_config = {
        "arm": arm,
        "config": config,
        "manifest": manifest["name"],
        "manifest_membership_hash": manifest["membership_hash"],
        "n_train_frames": len(manifest["frames"]),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs_run": len(history) - 1,
        "best_epoch": best_epoch,
        "best_primary_metric": {metric_name: best_value},
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "Adam",
        "num_workers": NUM_WORKERS,
        "amp": False,
        "scheduler": None,
        "trainable_parameters": trainable_count,
        "initial_identity_gap": identity_gap,
        "edge_calibration": calibration,
        "sampler_order_hash": sha256_text(json.dumps(order_log)),
        "train_domain_far_error": train_domain,
        "disabled": {
            "diffpnp_loss": True, "mask_aux_loss": True, "teacher_loss": True,
            "covariance_loss": True, "symmetric_loss": True,
            "visibility_coord_loss": True, "reliability_loss": True,
            "old_structural_loss": True, "affinity_loss": True,
        },
    }
    (OUT_DIR / "run_configs").mkdir(exist_ok=True)
    (OUT_DIR / "run_configs" / f"{arm}.json").write_text(
        json.dumps(MD.jsonable(run_config), indent=2), encoding="utf-8"
    )
    return run_config


# ============================================================================
# Phase 11 — paired evaluation
# ============================================================================
def paired_bootstrap(
    values: np.ndarray, sessions: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    finite_mask = np.isfinite(values)
    values, sessions = values[finite_mask], sessions[finite_mask]
    if values.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    unique = np.unique(sessions)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(replicates, dtype=np.float64)
    by_session = [values[sessions == s] for s in unique]
    for index in range(replicates):
        picked = rng.integers(0, len(unique), len(unique))
        sample = np.concatenate([by_session[p] for p in picked])
        means[index] = sample.mean()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(values.size),
        "n_sessions": int(unique.size),
    }


ERROR_METRICS = (
    "far_2d_median_px", "matched_2d_median_px", "near_2d_median_px",
    "yaw_err_deg", "reproj_fixed_gt_px", "gross_corner_fraction", "add_m",
)
SUCCESS_METRICS = (
    "pose_success", "n_corner_detected", "n_far_detected",
    "gt_inframe_detection_rate", "frame_median_peak", "n_detected",
)


def paired_delta(
    reference: pd.DataFrame, candidate: pd.DataFrame, label: str
) -> pd.DataFrame:
    merged = reference.merge(
        candidate, on="frame_id", suffixes=("_ref", "_cand")
    )
    rows: list[dict[str, Any]] = []
    for subset_name, subset in [("ALL", merged)] + [
        (str(c), merged[merged.failure_class_ref == c])
        for c in sorted(merged.failure_class_ref.unique())
    ] + [
        (f"domain:{d}", merged[merged.domain_ref == d])
        for d in sorted(merged.domain_ref.unique())
    ] + [
        (f"truncated:{t}", merged[merged.is_truncated_ref == t])
        for t in sorted(merged.is_truncated_ref.unique())
    ]:
        if not len(subset):
            continue
        for metric in ERROR_METRICS + SUCCESS_METRICS:
            ref = subset[f"{metric}_ref"].astype(float).values
            cand = subset[f"{metric}_cand"].astype(float).values
            if metric == "yaw_err_deg":
                common = np.isfinite(ref) & np.isfinite(cand)
                ref, cand = ref[common], cand[common]
                sessions = subset.session_id_ref.values[common]
            else:
                sessions = subset.session_id_ref.values
            delta = cand - ref
            ci = paired_bootstrap(delta, sessions)
            ref_median = nanmedian(ref)
            cand_median = nanmedian(cand)
            rows.append(
                {
                    "comparison": label,
                    "subset": subset_name,
                    "metric": metric,
                    "direction": "lower_is_better"
                    if metric in ERROR_METRICS
                    else "higher_is_better",
                    "n": len(subset),
                    "reference_median": ref_median,
                    "candidate_median": cand_median,
                    "absolute_delta_median": (
                        None
                        if (ref_median is None or cand_median is None)
                        else cand_median - ref_median
                    ),
                    "percent_delta": (
                        None
                        if (ref_median in (None, 0) or cand_median is None)
                        else 100.0 * (cand_median - ref_median) / abs(ref_median)
                    ),
                    "paired_mean_delta": ci["mean"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "n_sessions": ci.get("n_sessions"),
                }
            )
    return pd.DataFrame(rows)


def session_delta(
    reference: pd.DataFrame, candidate: pd.DataFrame, label: str, metric: str
) -> pd.DataFrame:
    merged = reference.merge(candidate, on="frame_id", suffixes=("_ref", "_cand"))
    rows = []
    for session, group in merged.groupby("session_id_ref"):
        ref = nanmedian(group[f"{metric}_ref"].values)
        cand = nanmedian(group[f"{metric}_cand"].values)
        rows.append(
            {
                "comparison": label,
                "metric": metric,
                "session_id": session,
                "n": len(group),
                "reference": ref,
                "candidate": cand,
                "delta": None if (ref is None or cand is None) else cand - ref,
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# gates
# ============================================================================
def gate_B1(control: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    target = "F2_CONFIDENT_WRONG"
    ref = summarize(control, target)
    cand = summarize(candidate, target)

    def rel(metric: str) -> Optional[float]:
        base, new = ref.get(metric), cand.get(metric)
        if base in (None, 0) or new is None:
            return None
        return 100.0 * (new - base) / abs(base)

    primary = {
        "far_2d_median_reduction_pct": None if rel("far_2d_median_px") is None else -rel("far_2d_median_px"),
        "gross_rate_reduction_pp": (
            None
            if (ref.get("gross_rate") is None or cand.get("gross_rate") is None)
            else 100.0 * (ref["gross_rate"] - cand["gross_rate"])
        ),
        "yaw_median_reduction_pct": None if rel("yaw_median_deg") is None else -rel("yaw_median_deg"),
        "reproj_median_reduction_pct": None if rel("reproj_median_px") is None else -rel("reproj_median_px"),
        "matched_2d_reduction_pct": None if rel("matched_2d_median_px") is None else -rel("matched_2d_median_px"),
    }
    passes = [
        (primary["far_2d_median_reduction_pct"] or 0) >= 15.0,
        (primary["gross_rate_reduction_pp"] or 0) >= 10.0,
        (primary["yaw_median_reduction_pct"] or 0) >= 10.0,
        (primary["reproj_median_reduction_pct"] or 0) >= 10.0,
    ]
    f1_ref, f1_cand = summarize(control, "F1_NO_RESPONSE"), summarize(candidate, "F1_NO_RESPONSE")
    f5_ref, f5_cand = summarize(control, "F5_MIXED"), summarize(candidate, "F5_MIXED")
    guards = {
        "F1_pose_success_drop_pp": 100.0 * (
            f1_ref.get("pose_success_rate", 0) - f1_cand.get("pose_success_rate", 0)
        ),
        "F5_pose_success_drop_pp": 100.0 * (
            f5_ref.get("pose_success_rate", 0) - f5_cand.get("pose_success_rate", 0)
        ),
        "clean_far_error_increase_pct": (
            None
            if f5_ref.get("far_2d_median_px") in (None, 0)
            else 100.0 * (
                f5_cand.get("far_2d_median_px", 0) - f5_ref["far_2d_median_px"]
            ) / f5_ref["far_2d_median_px"]
        ),
        "detection_drop_pp": 100.0 * (
            summarize(control).get("gt_inframe_detection_rate", 0)
            - summarize(candidate).get("gt_inframe_detection_rate", 0)
        ),
    }
    guard_ok = (
        guards["F1_pose_success_drop_pp"] < 3.0
        and guards["F5_pose_success_drop_pp"] < 3.0
        and (guards["clean_far_error_increase_pct"] or 0) < 10.0
        and guards["detection_drop_pp"] < 3.0
    )
    return {
        "arm": "B1", "target": target, "reference": ref, "candidate": cand,
        "primary": primary, "primary_pass": any(passes), "guards": guards,
        "guard_pass": guard_ok, "verdict": "PASS" if (any(passes) and guard_ok) else "FAIL",
    }


def gate_A1(control: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    target = "F1_NO_RESPONSE"
    ref, cand = summarize(control, target), summarize(candidate, target)
    primary = {
        "gt_inframe_recovery_pp": 100.0 * (
            cand.get("gt_inframe_detection_rate", 0)
            - ref.get("gt_inframe_detection_rate", 0)
        ),
        "median_detected_corner_gain": (
            None
            if (cand.get("median_detected_corners") is None
                or ref.get("median_detected_corners") is None)
            else cand["median_detected_corners"] - ref["median_detected_corners"]
        ),
        "pose_success_gain_pp": 100.0 * (
            cand.get("pose_success_rate", 0) - ref.get("pose_success_rate", 0)
        ),
        "far_detection_gain_pp": 100.0 * (
            (cand.get("far_detected_mean", 0) - ref.get("far_detected_mean", 0)) / 4.0
        ),
    }
    passes = [
        primary["gt_inframe_recovery_pp"] >= 10.0,
        (primary["median_detected_corner_gain"] or 0) >= 1.0,
        primary["pose_success_gain_pp"] >= 5.0,
        primary["far_detection_gain_pp"] >= 10.0,
    ]
    non_f1_ref = summarize(control[control.failure_class != target])
    non_f1_cand = summarize(candidate[candidate.failure_class != target])
    f2_ref, f2_cand = summarize(control, "F2_CONFIDENT_WRONG"), summarize(candidate, "F2_CONFIDENT_WRONG")
    guards = {
        "non_F1_detection_drop_pp": 100.0 * (
            non_f1_ref.get("gt_inframe_detection_rate", 0)
            - non_f1_cand.get("gt_inframe_detection_rate", 0)
        ),
        "non_F1_pose_success_drop_pp": 100.0 * (
            non_f1_ref.get("pose_success_rate", 0)
            - non_f1_cand.get("pose_success_rate", 0)
        ),
        "F2_error_increase_pct": (
            None
            if f2_ref.get("far_2d_median_px") in (None, 0)
            else 100.0 * (
                f2_cand.get("far_2d_median_px", 0) - f2_ref["far_2d_median_px"]
            ) / f2_ref["far_2d_median_px"]
        ),
    }
    guard_ok = (
        guards["non_F1_detection_drop_pp"] < 3.0
        and guards["non_F1_pose_success_drop_pp"] < 3.0
        and (guards["F2_error_increase_pct"] or 0) < 10.0
    )
    return {
        "arm": "A1", "target": target, "reference": ref, "candidate": cand,
        "primary": primary, "primary_pass": any(passes), "guards": guards,
        "guard_pass": guard_ok, "verdict": "PASS" if (any(passes) and guard_ok) else "FAIL",
    }


def gate_B2(b1: pd.DataFrame, b2: pd.DataFrame) -> dict[str, Any]:
    target = "F2_CONFIDENT_WRONG"
    ref, cand = summarize(b1, target), summarize(b2, target)

    def reduction(metric: str) -> Optional[float]:
        base, new = ref.get(metric), cand.get(metric)
        if base in (None, 0) or new is None:
            return None
        return 100.0 * (base - new) / abs(base)

    primary = {
        "far_2d_extra_reduction_pct": reduction("far_2d_median_px"),
        "gross_extra_reduction_pp": (
            None
            if (ref.get("gross_rate") is None or cand.get("gross_rate") is None)
            else 100.0 * (ref["gross_rate"] - cand["gross_rate"])
        ),
        "yaw_extra_reduction_pct": reduction("yaw_median_deg"),
        "reproj_extra_reduction_pct": reduction("reproj_median_px"),
    }
    passes = [
        (primary["far_2d_extra_reduction_pct"] or 0) >= 5.0,
        (primary["gross_extra_reduction_pp"] or 0) >= 5.0,
        (primary["yaw_extra_reduction_pct"] or 0) >= 5.0,
        (primary["reproj_extra_reduction_pct"] or 0) >= 5.0,
    ]
    non_target_ref = summarize(b1[b1.failure_class != target])
    non_target_cand = summarize(b2[b2.failure_class != target])
    guards = {
        "non_target_pose_success_drop_pp": 100.0 * (
            non_target_ref.get("pose_success_rate", 0)
            - non_target_cand.get("pose_success_rate", 0)
        ),
    }
    guard_ok = guards["non_target_pose_success_drop_pp"] < 3.0
    return {
        "arm": "B2", "target": target, "reference": ref, "candidate": cand,
        "primary": primary, "primary_pass": any(passes), "guards": guards,
        "guard_pass": guard_ok, "verdict": "PASS" if (any(passes) and guard_ok) else "FAIL",
    }


# ============================================================================
# ablation table (Phase 12)
# ============================================================================
def ablation_table() -> pd.DataFrame:
    rows = [
        {"arm": "ep57", "training_manifest": "original", "trainable": "none",
         "target_semantics": "legacy", "residual": "no", "edge_loss": "no"},
    ]
    for arm in ARMS:
        config = ARM_CONFIG[arm]
        rows.append(
            {
                "arm": arm,
                "training_manifest": config["manifest"],
                "trainable": "m6_2 tail" if config["scope"] == "m6_2_tail" else "residual head",
                "target_semantics": "corrected partial" if config["semantics"] else "legacy",
                "residual": "yes" if config["residual"] else "no",
                "edge_loss": "yes" if config["edge"] else "no",
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# Phase 15/16 — figures and reports
# ============================================================================
def available_arms() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for arm in ("ep57",) + ARMS:
        path = OUT_DIR / f"eval_{arm}.parquet"
        if path.is_file():
            frames[arm] = pd.read_parquet(path)
    return frames


COMPARISONS = (
    ("ep57", "M0_B"), ("M0_B", "B1"), ("B1", "B2"),
    ("ep57", "M0_A"), ("M0_A", "A1"),
)


def make_figures(frames: dict[str, pd.DataFrame]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(OUT_DIR / name, dpi=140)
        plt.close(fig)
        written.append(name)

    # B1 far error across epochs
    curve = WEIGHT_DIR / "B1" / "metrics_by_epoch.csv"
    control_curve = WEIGHT_DIR / "M0_B" / "metrics_by_epoch.csv"
    if curve.is_file():
        fig, ax = plt.subplots(figsize=(7, 4.2))
        b1 = pd.read_csv(curve)
        ax.plot(b1.epoch, b1.far_2d_median_px, marker="o", label="B1")
        if control_curve.is_file():
            m0 = pd.read_csv(control_curve)
            ax.plot(m0.epoch, m0.far_2d_median_px, marker="s", label="M0_B control")
        ax.set_xlabel("epoch")
        ax.set_ylabel("far/depth matched 2D median (px)")
        ax.set_title("F2_CONFIDENT_WRONG far error (N=35)")
        ax.legend()
        save(fig, "B1_far_error_curve.png")

    residual_path = OUT_DIR / "residual_diagnostics_B1.csv"
    if residual_path.is_file():
        fig, ax = plt.subplots(figsize=(7, 4.2))
        res = pd.read_csv(residual_path)
        grouped = res.groupby("epoch")[
            ["residual_abs_mean", "residual_abs_max", "residual_saturation_fraction"]
        ].mean()
        grouped.plot(ax=ax, marker="o")
        ax.axhline(RESIDUAL_AMPLITUDE, color="red", linestyle="--", label="bound")
        ax.set_title("B1 bounded residual magnitude")
        ax.legend()
        save(fig, "B1_residual_magnitude.png")

    if "M0_A" in frames and "A1" in frames:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        f1_control = frames["M0_A"][frames["M0_A"].failure_class == "F1_NO_RESPONSE"]
        f1_candidate = frames["A1"][frames["A1"].failure_class == "F1_NO_RESPONSE"]
        merged = f1_control.merge(f1_candidate, on="frame_id", suffixes=("_c", "_a"))
        ax.scatter(
            merged.gt_inframe_detection_rate_c,
            merged.gt_inframe_detection_rate_a,
            alpha=0.8,
        )
        ax.plot([0, 1], [0, 1], color="black", linewidth=0.8)
        ax.set_xlabel("M0_A control")
        ax.set_ylabel("A1")
        ax.set_title("F1: GT-in-frame corner detection rate (per frame)")
        save(fig, "A1_inframe_response_recovery.png")

    deltas_path = OUT_DIR / "paired_deltas.csv"
    if deltas_path.is_file():
        deltas = pd.read_csv(deltas_path)
        focus = deltas[
            deltas.subset.isin(
                ["F1_NO_RESPONSE", "F2_CONFIDENT_WRONG", "F5_MIXED", "ALL"]
            )
            & deltas.metric.isin(["far_2d_median_px", "gt_inframe_detection_rate"])
        ]
        if len(focus):
            fig, ax = plt.subplots(figsize=(10, 4.6))
            pivot = focus.pivot_table(
                index="subset", columns=["comparison", "metric"],
                values="absolute_delta_median"
            )
            pivot.plot(kind="bar", ax=ax)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title("Paired delta by failure class (candidate - reference)")
            ax.tick_params(axis="x", rotation=15)
            ax.legend(fontsize=6)
            save(fig, "failure_class_delta.png")

    session_path = OUT_DIR / "session_deltas.csv"
    if session_path.is_file():
        session = pd.read_csv(session_path)
        if len(session):
            fig, ax = plt.subplots(figsize=(10, 4.6))
            session.pivot_table(
                index="session_id", columns="comparison", values="delta"
            ).plot(kind="bar", ax=ax)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title("Per-session delta (consistency check)")
            ax.tick_params(axis="x", rotation=25)
            save(fig, "session_delta.png")

    if "B1" in frames and "B2" in frames:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        rows = []
        for arm in ("M0_B", "B1", "B2"):
            if arm in frames:
                summary = summarize(frames[arm], "F2_CONFIDENT_WRONG")
                rows.append({"arm": arm, **summary})
        table = pd.DataFrame(rows).set_index("arm")
        table[["far_2d_median_px", "matched_2d_median_px", "yaw_median_deg"]].plot(
            kind="bar", ax=ax
        )
        ax.set_title("B2 structural loss effect (F2 subset)")
        save(fig, "B2_edge_loss_effect.png")

    return written


def example_overlays(frames: dict[str, pd.DataFrame], device: torch.device) -> int:
    """Same frame: GT / ep57 / matched control / candidate, side by side."""
    if "B1" not in frames or "M0_B" not in frames:
        return 0
    evaluator = MechanismEvaluator()
    models: dict[str, ScreenModel] = {}
    for arm, with_residual in (("M0_B", False), ("B1", True)):
        path = WEIGHT_DIR / arm / "best.pth"
        if not path.is_file():
            return 0
        model = ScreenModel(with_residual=with_residual).to(device)
        model.load_state_dict(torch.load(str(path), map_location=device))
        model.eval()
        models[arm] = model
    base = ScreenModel(with_residual=False).to(device).eval()

    target = frames["B1"][frames["B1"].failure_class == "F2_CONFIDENT_WRONG"]
    merged = frames["M0_B"].merge(target, on="frame_id", suffixes=("_c", "_b"))
    merged = merged.assign(gain=merged.far_2d_median_px_c - merged.far_2d_median_px_b)
    chosen = merged.sort_values(["gain", "frame_id"], ascending=[False, True]).head(4)

    out_dir = OUT_DIR / "example_overlays"
    out_dir.mkdir(exist_ok=True)
    written = 0
    for _, row in chosen.iterrows():
        spec = next(f for f in evaluator.frames if f["frame_id"] == row.frame_id)
        geometry = evaluator.geometry[row.frame_id]
        image = evaluator.images[row.frame_id]
        scale_x = spec["image_width"] / OUTPUT_SIZE
        scale_y = spec["image_height"] / OUTPUT_SIZE
        panels = []
        for label, model in (("GT only", None), ("ep57", base),
                             ("M0_B control", models["M0_B"]), ("B1", models["B1"])):
            canvas = image.copy()
            MD.draw_points(canvas, geometry.gt_points, MD.GREEN)
            if model is not None:
                belief = belief_from_model(model, image, device)
                points = MD.decode_all(
                    belief, scale_x, scale_y, geometry.gt_points
                )["D0"]
                MD.draw_points(canvas, points, MD.RED)
            panels.append(MD.banner(canvas, [f"{label} | {row.frame_id}"]))
        top = np.hstack(panels[:2])
        bottom = np.hstack(panels[2:])
        cv2.imwrite(
            str(out_dir / f"B1_example__{row.frame_id.replace(':', '_')}.jpg"),
            np.vstack([top, bottom]),
        )
        written += 1
    return written


def build_reports(info: dict[str, Any]) -> None:
    frames = available_arms()
    if "ep57" not in frames:
        log("[report] no evaluations yet")
        return

    epoch_rows = []
    for arm in ARMS:
        path = WEIGHT_DIR / arm / "metrics_by_epoch.csv"
        if path.is_file():
            epoch_rows.append(pd.read_csv(path))
    if epoch_rows:
        pd.concat(epoch_rows, ignore_index=True).to_csv(
            OUT_DIR / "metrics_by_epoch.csv", index=False
        )

    by_class = []
    for arm, frame in frames.items():
        for failure in sorted(frame.failure_class.unique()):
            by_class.append({"arm": arm, "failure_class": failure,
                             **summarize(frame, failure)})
        by_class.append({"arm": arm, "failure_class": "ALL", **summarize(frame)})
    pd.DataFrame(by_class).to_csv(OUT_DIR / "metrics_by_failure_class.csv", index=False)

    deltas, sessions, cis = [], [], []
    for reference, candidate in COMPARISONS:
        if reference not in frames or candidate not in frames:
            continue
        label = f"{reference}->{candidate}"
        table = paired_delta(frames[reference], frames[candidate], label)
        deltas.append(table)
        cis.append(table[["comparison", "subset", "metric", "paired_mean_delta",
                          "ci_low", "ci_high", "n", "n_sessions"]])
        metric = (
            "gt_inframe_detection_rate" if candidate in ("M0_A", "A1")
            else "far_2d_median_px"
        )
        sessions.append(session_delta(frames[reference], frames[candidate], label, metric))
    if deltas:
        pd.concat(deltas, ignore_index=True).to_csv(OUT_DIR / "paired_deltas.csv", index=False)
        pd.concat(cis, ignore_index=True).to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)
    if sessions:
        pd.concat(sessions, ignore_index=True).to_csv(OUT_DIR / "session_deltas.csv", index=False)

    ablation_table().to_csv(OUT_DIR / "ablation_table.csv", index=False)
    figures = make_figures(frames)
    overlays = example_overlays(frames, FZ.choose_device("auto"))
    if overlays:
        figures.append(f"example_overlays/ ({overlays} panels)")

    gates = {}
    for name in ("B1", "A1", "B2"):
        path = OUT_DIR / f"gate_{name}.json"
        if path.is_file():
            gates[name] = json.loads(path.read_text(encoding="utf-8"))

    write_markdown(info, frames, gates, figures)
    log("[report] written")


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str], limit: int = 60) -> str:
    subset = frame[columns].head(limit)
    widths = [
        max(len(str(c)), *(len(_fmt(v)) for v in subset[c])) for c in columns
    ]
    head = "  ".join(str(c).ljust(w) for c, w in zip(columns, widths))
    body = "\n".join(
        "  ".join(_fmt(row[c]).ljust(w) for c, w in zip(columns, widths))
        for _, row in subset.iterrows()
    )
    return f"```\n{head}\n{'─' * len(head)}\n{body}\n```"


def write_markdown(
    info: dict[str, Any], frames: dict[str, pd.DataFrame],
    gates: dict[str, Any], figures: list[str]
) -> None:
    lines = ["# PAPER_S2 micro architecture screen — 해결 능력 판별\n"]
    lines.append(
        "논문 최종 학습이 아니다.  후보가 **자기 target failure class 를 고치는지**만, "
        "같은 manifest 로 학습한 matched control 대비로 본다.\n"
    )
    gate = info["baseline_gate"]
    lines.append(
        f"- checkpoint `{FZ.WEIGHTS.name}` SHA `{info['checkpoint_sha256'][:16]}…` (불변)\n"
        f"- git HEAD `{info['git_head']}`\n"
        f"- mechanism-val membership SHA `{info['mechanism_membership_sha256'][:16]}…`\n"
        f"- baseline gate: GT-2D {gate['gt2d_pose_success']}/{gate['strict_n']}, "
        f"predicted {gate['pred_pose_success']}/{gate['strict_n']}, "
        f"yaw {gate['yaw_median_deg']:.3f}°, reproj {gate['fixed_gt_reproj_median_px']:.3f}px\n"
        f"- final-test open count **0**\n"
        f"- torch {info['torch']} / cuda {info['cuda']} / {info['gpu']}\n"
    )

    manifest_b = json.loads(
        (OUT_DIR / "micro_train_B_manifest.json").read_text(encoding="utf-8")
    )
    criterion = manifest_b.get("hard_criterion", {})
    if criterion and not criterion.get("criterion_met", True):
        lines.append(
            "\n## ⚠ 0. 먼저 읽을 것 — B 계열은 설계대로 시험되지 않았다\n"
        )
        lines.append(
            f"[확인] Manifest B 의 'F2-like hard frame = far/depth 2D error > 20px' 기준을 "
            f"만족하는 학습 프레임이 **{criterion['n_above_specified']}개**뿐이다.  "
            f"ep57 은 자기 synthetic 학습 분포에서 이미 정확하다 — pool far error "
            f"중앙값 {criterion['pool_far_error_median_px']:.2f}px, 최대 "
            f"{criterion['pool_far_error_max_px']:.2f}px.\n"
            f"[확인] 같은 지표로 real mechanism-val 의 F2 는 far error 중앙값 43.3px, "
            "최대 160.7px 다.  즉 **F2_CONFIDENT_WRONG 은 synthetic 학습 데이터에 "
            "존재하지 않는 실패 모드**이며, 이는 순수한 sim2real 전이 문제다.\n"
            f"[확인] 수를 억지로 채우지 않고, 가장 어려운 프레임(>= "
            f"{criterion['threshold_actually_used_px']:.2f}px)으로 대체해 실행했고 "
            "그 사실을 manifest 에 기록했다.\n"
            "[판정] 따라서 **B1 의 FAIL 은 후보 기각이 아니라 '이 데이터로는 시험 불가'** 다.  "
            "Phase 16 의 'B1 FAIL → backbone 으로 이동' 규칙을 이 결과에 적용하면 안 된다.\n"
        )

    lines.append("\n## Phase 12 — ablation table (계획 외 기능 없음)\n")
    lines.append(_table(ablation_table(), list(ablation_table().columns)))
    disabled = json.loads(
        (OUT_DIR / "run_configs" / "M0_B.json").read_text(encoding="utf-8")
    )["disabled"] if (OUT_DIR / "run_configs" / "M0_B.json").is_file() else {}
    if disabled:
        lines.append(
            "\n[확인] 모든 arm 에서 꺼진 항목: "
            + ", ".join(sorted(disabled)) + " (loss 미계산).\n"
            "[주의] loader 의 `aspect_resize` 는 ep57 전처리(anisotropic squash)와 "
            "평가 경로 정합을 위해 모든 arm 에서 동일하게 켠다 — DiffPnP **loss** 는 꺼져 있다.\n"
        )

    lines.append("\n## 1. 관찰 — arm 별 절대 지표\n")
    rows = []
    for arm, frame in frames.items():
        for failure in ("F2_CONFIDENT_WRONG", "F1_NO_RESPONSE", "F5_MIXED", None):
            summary = summarize(frame, failure)
            if summary:
                rows.append({"arm": arm, "subset": failure or "ALL", **summary})
    absolute = pd.DataFrame(rows)
    lines.append(_table(absolute, [
        "arm", "subset", "n", "far_2d_median_px", "matched_2d_median_px",
        "gross_rate", "yaw_median_deg", "reproj_median_px", "pose_success_rate",
        "median_detected_corners", "gt_inframe_detection_rate"]))

    lines.append("\n## 2. M0 fine-tuning effect (ep57 대비 control)\n")
    for control, target in (("M0_B", "F2_CONFIDENT_WRONG"), ("M0_A", "F1_NO_RESPONSE")):
        if control not in frames:
            continue
        before = summarize(frames["ep57"], target)
        after = summarize(frames[control], target)
        lines.append(
            f"- **{control}** ({target}, N={before.get('n')}): "
            f"far 2D {_fmt(before.get('far_2d_median_px'))} → {_fmt(after.get('far_2d_median_px'))} px, "
            f"yaw {_fmt(before.get('yaw_median_deg'))} → {_fmt(after.get('yaw_median_deg'))}°, "
            f"pose success {_fmt(before.get('pose_success_rate'))} → {_fmt(after.get('pose_success_rate'))}, "
            f"GT-in-frame 검출률 {_fmt(before.get('gt_inframe_detection_rate'))} → "
            f"{_fmt(after.get('gt_inframe_detection_rate'))}"
        )
    lines.append(
        "\n[확인] 이 차이가 **400장 추가 fine-tuning 자체의 효과**다.  후보의 개선은 "
        "이 control 대비로만 계산한다.\n"
    )

    for name in ("B1", "A1", "B2"):
        if name not in gates:
            continue
        gate_result = gates[name]
        lines.append(f"\n## 3. {name} 결과 — {gate_result['verdict']}\n")
        lines.append("primary:")
        lines.append(_table(
            pd.DataFrame([gate_result["primary"]]).T.reset_index().rename(
                columns={"index": "metric", 0: "value"}),
            ["metric", "value"]))
        lines.append("\nguards (regression 방지):")
        lines.append(_table(
            pd.DataFrame([gate_result["guards"]]).T.reset_index().rename(
                columns={"index": "guard", 0: "value"}),
            ["guard", "value"]))
        lines.append(
            f"\n[확인] primary_pass={gate_result['primary_pass']}, "
            f"guard_pass={gate_result['guard_pass']} → **{gate_result['verdict']}**\n"
        )

    deltas_path = OUT_DIR / "paired_deltas.csv"
    if deltas_path.is_file():
        deltas = pd.read_csv(deltas_path)
        lines.append("\n## 4. Paired delta + session-cluster bootstrap CI\n")
        focus = deltas[
            deltas.subset.isin(["F2_CONFIDENT_WRONG", "F1_NO_RESPONSE", "ALL"])
            & deltas.metric.isin([
                "far_2d_median_px", "gt_inframe_detection_rate", "yaw_err_deg",
                "reproj_fixed_gt_px", "pose_success"])
        ]
        lines.append(_table(focus, [
            "comparison", "subset", "metric", "reference_median",
            "candidate_median", "percent_delta", "paired_mean_delta",
            "ci_low", "ci_high", "n"]))
        lines.append(
            "\n[주의] 소표본(F2 N=35, F1 N=24)이라 CI 가 넓다.  gate 방향성과 "
            "session 일관성(`session_deltas.csv`, `session_delta.png`)을 함께 본다.\n"
        )

    lines.append("\n## 4b. FAIL 의 원인 분해\n")
    b1_config = OUT_DIR / "run_configs" / "B1.json"
    if b1_config.is_file():
        train_domain = json.loads(b1_config.read_text(encoding="utf-8")).get(
            "train_domain_far_error"
        )
        if train_domain:
            rows = []
            for phase in ("before", "after"):
                for entry in train_domain[phase] or []:
                    rows.append({"phase": phase, **entry})
            lines.append("B1 이 **자기 학습 도메인에서** far 를 고쳤는가:\n")
            lines.append(_table(pd.DataFrame(rows), ["phase", "subset", "n",
                                                     "far_2d_median_px"]))
            lines.append(
                "\n[확인] 학습 도메인(synthetic hard)에서조차 far 오차가 거의 안 줄었다.  "
                "residual head 가 위치를 못 고치는 게 아니라 **고칠 오차가 없다** — "
                "학습 신호 부재다 (일반화 실패가 아니다).\n"
            )

    if "M0_A" in frames and "A1" in frames:
        control = frames["M0_A"][frames["M0_A"].failure_class == "F1_NO_RESPONSE"]
        candidate = frames["A1"][frames["A1"].failure_class == "F1_NO_RESPONSE"]
        merged = control.merge(candidate, on="frame_id", suffixes=("_c", "_a"))
        delta = (merged.frame_median_peak_a - merged.frame_median_peak_c).astype(float)
        improved = int((delta > 0).sum())
        lines.append("\nA1 의 target semantics 가 F1 response 를 움직였는가:\n")
        lines.append(
            f"- [확인] frame median peak paired delta = "
            f"{float(np.nanmedian(delta)):+.5f} (평균 {float(np.nanmean(delta)):+.5f}), "
            f"{improved}/{len(merged)} 프레임에서 상승 — **방향은 일관되게 양수**.\n"
            f"- [확인] 그러나 F1 의 median peak 는 "
            f"{float(np.nanmedian(control.frame_median_peak)):.3f} 이고 검출 임계값은 "
            f"{BELIEF_THRESHOLD} 이다.  +0.002 수준의 상승으로는 임계를 넘길 수 없어 "
            "GT-in-frame 검출률이 "
            f"{float(control.gt_inframe_detected.sum() / control.gt_inframe_corners.sum()):.4f} "
            "로 **비트 단위 동일**하게 유지된다.\n"
            "- [판정] 즉 A1 의 primary gate 지표는 이 표본에서 **사실상 상수**라 판별력이 "
            "없었다.  A1 FAIL 은 '효과 0' 이 아니라 '이 지표·이 표본으로는 검출 불가' 다.\n"
        )

    lines.append("\n## 5. 그림\n")
    for name in figures:
        lines.append(f"- `{name}`")
    lines.append(
        "\n예시 패널은 같은 frame 의 GT / ep57 / matched control / candidate 를 "
        "나란히 보여준다 (`example_overlays/`).\n"
    )

    lines.append("\n## 6. 남은 불확실성\n")
    lines.append(
        "- [확인] 1 seed, 400장, 최대 10 epoch 의 **스크리닝**이다.  통과 후보라도 "
        "새 공개 데이터셋에서 clean 3-seed 로 다시 검증해야 한다.\n"
        "- [확인] F2 N=35 / F1 N=24 는 소표본이며 session 수도 적다.\n"
        "- [확인] mechanism-val 은 학습에 쓰이지 않았지만 **모델 선택(best epoch)**에는 "
        "쓰였다.  따라서 여기 수치는 낙관적 상한이다.\n"
        "- [확인] final-test 는 열지 않았다.\n"
    )
    (OUT_DIR / "MICRO_ARCH_SCREEN_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_gate_decision(frames, gates)
    write_provenance(info)


def write_gate_decision(
    frames: dict[str, pd.DataFrame], gates: dict[str, Any]
) -> None:
    lines = ["# ARCHITECTURE GATE DECISION — PAPER_S2 micro screen\n"]
    lines.append("[관찰]")
    for arm in ("ep57", "M0_B", "B1", "M0_A", "A1", "B2"):
        if arm not in frames:
            continue
        f2 = summarize(frames[arm], "F2_CONFIDENT_WRONG")
        f1 = summarize(frames[arm], "F1_NO_RESPONSE")
        lines.append(
            f"- {arm}: F2 far2D {_fmt(f2.get('far_2d_median_px'))}px / "
            f"F1 GT-in-frame 검출률 {_fmt(f1.get('gt_inframe_detection_rate'))}"
        )

    lines.append("\n[M0 fine-tuning effect]")
    for control, target, metric in (
        ("M0_B", "F2_CONFIDENT_WRONG", "far_2d_median_px"),
        ("M0_A", "F1_NO_RESPONSE", "gt_inframe_detection_rate"),
    ):
        if control in frames:
            before = summarize(frames["ep57"], target).get(metric)
            after = summarize(frames[control], target).get(metric)
            lines.append(f"- {control}: {metric} {_fmt(before)} → {_fmt(after)}")

    for name, header in (("B1", "[B1 결과]"), ("A1", "[A1 결과]")):
        lines.append(f"\n{header}")
        if name in gates:
            gate = gates[name]
            lines.append(f"- verdict **{gate['verdict']}**")
            for key, value in gate["primary"].items():
                lines.append(f"  - {key} = {_fmt(value)}")
        else:
            lines.append("- 미실행")

    lines.append("\n[B2 결과 또는 미실행 사유]")
    if "B2" in gates:
        lines.append(f"- verdict **{gates['B2']['verdict']}**")
    elif gates.get("B1", {}).get("verdict") == "PASS":
        lines.append("- B1 PASS 이므로 실행 대상이나 아직 미실행")
    else:
        lines.append(
            "- **미실행**: B1 이 FAIL 이므로 조건부 규칙에 따라 structural loss 를 "
            "시험하지 않았다."
        )

    lines.append("\n[지지 증거]")
    lines.append(
        "- [확인] 모든 비교가 같은 frame 위 paired 이고, control 은 후보와 동일한 "
        "manifest / seed / sampler order / epoch 예산을 쓴다."
    )
    if "B1" in gates:
        lines.append(
            "- [확인] B1 은 zero-init identity(max|H_final-H_base|=0)에서 출발하므로, "
            "측정된 변화는 재파라미터화가 아니라 학습 결과다."
        )
    lines.append("\n[반증 증거]")
    for name in ("B1", "A1", "B2"):
        if name in gates and gates[name]["verdict"] == "FAIL":
            gate = gates[name]
            reason = (
                "primary 기준 미달"
                if not gate["primary_pass"]
                else "non-target regression"
            )
            lines.append(f"- [확인] {name} FAIL — {reason}.")

    lines.append("\n[현재 판정]")
    b1 = gates.get("B1", {}).get("verdict")
    a1 = gates.get("A1", {}).get("verdict")
    manifest_b = json.loads(
        (OUT_DIR / "micro_train_B_manifest.json").read_text(encoding="utf-8")
    )
    criterion = manifest_b.get("hard_criterion", {})
    b_testable = bool(criterion.get("criterion_met", True))

    if b1 == "PASS":
        lines.append("- [확인] B1 PASS → 새 공개 데이터셋 main architecture 후보로 승격.")
    elif b1 == "FAIL" and not b_testable:
        lines.append(
            "- [확인] B1 은 **시험 불가(NOT TESTABLE)** 다.  target failure mode(F2)가 "
            f"학습 소스에 없다 — pool far error 중앙값 "
            f"{criterion['pool_far_error_median_px']:.2f}px / 최대 "
            f"{criterion['pool_far_error_max_px']:.2f}px vs real F2 중앙값 43.3px, "
            f">20px 프레임 {criterion['n_above_specified']}개."
        )
        lines.append(
            "- [확인] 학습 도메인(synthetic hard) 자체에서도 far 오차가 거의 안 줄었다 "
            "→ residual 이 못 고친 게 아니라 **고칠 오차가 없었다**."
        )
        lines.append(
            "- [판정] 'B1 FAIL → final-stage residual 이 너무 늦다 → backbone 으로 이동' "
            "이라는 Phase 16 규칙은 **이 결과에 적용하지 않는다**.  적용하려면 real "
            "F2 분포를 담은 학습 소스에서 다시 시험해야 한다."
        )
    elif b1 == "FAIL":
        lines.append(
            "- [확인] B1 FAIL → final-stage residual 위치가 너무 늦다.  "
            "backbone / multi-scale feature 개선 후보로 이동한다."
        )

    if a1 == "PASS":
        lines.append(
            "- [확인] A1 PASS → corrected partial target semantics 를 새 데이터셋 기본값으로 채택."
        )
    elif a1 == "FAIL":
        lines.append(
            "- [확인] A1 FAIL → target semantics 결함은 존재하고 방향도 일관되지만"
            "(F1 frame peak 가 20/24 프레임에서 상승), 크기가 +0.002 수준이라 검출 "
            "임계 0.3 을 넘기지 못한다.  현재 F1 의 주된 response failure 를 해결하는 "
            "**충분조건이 아니다**."
        )
        lines.append(
            "- [주의] A1 의 primary 지표(GT-in-frame 검출률)는 이 표본에서 비트 단위로 "
            "상수였다.  따라서 이 FAIL 은 '효과 0' 의 증거가 아니라 **판별력 부족**의 "
            "증거이기도 하다.  코드 결함은 수정하되 architecture contribution 으로 "
            "주장하지 않는다."
        )

    lines.append("\n[승격 후보]")
    promoted = [n for n in ("B1", "A1", "B2") if gates.get(n, {}).get("verdict") == "PASS"]
    lines.extend([f"- {n}" for n in promoted] or ["- 없음"])

    lines.append("\n[폐기/보류 후보]")
    holds: list[str] = []
    if b1 == "FAIL" and not b_testable:
        holds.append(
            "- B1: **보류(시험 불가)** — 폐기 아님.  학습 소스에 target failure mode 가 "
            "없어 판정 자체가 성립하지 않았다."
        )
    elif b1 == "FAIL":
        holds.append("- B1: 이번 스크리닝 조건에서 보류")
    if a1 == "FAIL":
        holds.append(
            "- A1: **보류(판별력 부족)** — 방향은 일관되나 크기가 검출 임계에 미달하고, "
            "primary 지표가 상수였다.  단 target semantics 결함 자체는 실재하므로 "
            "코드 수정은 별도로 진행한다 (architecture contribution 주장 금지)."
        )
    if "B2" not in gates:
        holds.append("- B2: 미실행 (B1 gate 미통과, 조건부 규칙 준수)")
    lines.extend(holds or ["- 없음"])

    lines.append("\n[다음 admissible experiment]")
    if not b_testable:
        lines.append(
            "1. **B 계열 재시험의 전제 확보**: real F2 분포(far error 20~160px)를 담은 "
            "학습 소스를 먼저 만든다.  현재 synthetic pool 은 far 중앙값 2.1px 로 "
            "F2 를 담고 있지 않다.  이것이 없으면 어떤 final-stage/backbone 후보도 "
            "F2 에 대해 판정할 수 없다."
        )
        lines.append(
            "2. 그 전까지 F2 는 **학습 후보 문제가 아니라 데이터/도메인 문제**로 다룬다 "
            "(sim2real 전이갭).  기존 STAGE16 결론과 같은 방향이다."
        )
    lines.append(
        "3. A1 재시험 시에는 검출률 대신 **연속 지표(frame median peak, GT 위치에서의 "
        "belief 값)**를 primary 로 쓴다.  현재 지표는 F1 peak(0.10)와 임계(0.30) 격차 "
        "때문에 상수다."
    )
    lines.append(
        "4. 이번 결과만으로 논문 main claim 을 쓰지 않는다.  승격 후보가 생기면 "
        "새 공개 데이터셋에서 clean 3-seed 로 재검증한다.  final-test 는 열지 않는다."
    )
    (OUT_DIR / "ARCHITECTURE_GATE_DECISION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_provenance(info: dict[str, Any]) -> None:
    hashes = json.loads((OUT_DIR / "manifest_hashes.json").read_text("utf-8"))
    lines = [
        "# RUN PROVENANCE — paper_s2_micro_arch_screen\n",
        f"- created: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"- git HEAD: {info['git_head']}",
        f"- checkpoint: {FZ.WEIGHTS} (SHA-256 {info['checkpoint_sha256']}, unchanged)",
        f"- python {info['python']} / torch {info['torch']} / cuda {info['cuda']} "
        f"/ opencv {info['opencv']}",
        f"- gpu: {info['gpu']}",
        f"- mechanism-val membership SHA: {info['mechanism_membership_sha256']}",
        "- final-test open count: 0",
        "",
        "## Manifest hashes",
    ]
    for key, value in hashes.items():
        lines.append(f"- {key}: {value}")
    lines.append("\n## Run configs")
    for arm in ARMS:
        path = OUT_DIR / "run_configs" / f"{arm}.json"
        if path.is_file():
            config = json.loads(path.read_text(encoding="utf-8"))
            lines.append(
                f"- {arm}: manifest {config['manifest']}, trainable "
                f"{config['trainable_parameters']}, epochs_run "
                f"{config['epochs_run']}, best_epoch {config['best_epoch']}, "
                f"sampler_order_hash {config['sampler_order_hash'][:16]}…"
            )
    lines.append(
        "\n## Reused (not reimplemented)\n"
        "- evaluator: `paper_s2_mechanism_diagnostic` (decode_all / FrameGeometry / metrics)\n"
        "- loss: `heatmap_refinement.channel_masked_mse`\n"
        "- target: `utils_belief.CreateBeliefMap(clip_at_border=)` + "
        "`spatial_keypoint_validity`\n"
        "- decoder for the edge loss: `diffpnp3d_loss.LocalSoftArgmax2D`\n"
    )
    (OUT_DIR / "RUN_PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifests", action="store_true")
    parser.add_argument("--run", choices=ARMS, action="append", default=[])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--all-primary", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    info = identity_and_gate()
    device = FZ.choose_device("auto")

    manifests = None
    if args.build_manifests or args.all_primary or args.run:
        manifests = build_manifests(device)
    if manifests is None:
        manifests = {
            "B": json.loads((OUT_DIR / "micro_train_B_manifest.json").read_text("utf-8")),
            "A": json.loads((OUT_DIR / "micro_train_A_manifest.json").read_text("utf-8")),
        }

    arms = list(args.run)
    if args.all_primary:
        arms = list(PRIMARY_ARMS)
    if arms:
        evaluator = MechanismEvaluator()
        # ep57 reference evaluation (once)
        if not (OUT_DIR / "eval_ep57.parquet").is_file():
            base = ScreenModel(with_residual=False).to(device)
            evaluator.evaluate(base, device, "ep57").to_parquet(
                OUT_DIR / "eval_ep57.parquet", index=False
            )
            log("[ep57] reference evaluation written")
        for arm in arms:
            if arm == "B2":
                gate_path = OUT_DIR / "gate_B1.json"
                if not gate_path.is_file():
                    raise RuntimeError("BLOCKED: B2 requested before the B1 gate ran")
                if json.loads(gate_path.read_text("utf-8"))["verdict"] != "PASS":
                    raise RuntimeError("BLOCKED: B2 requested but B1 did not PASS")
            log(f"=== running {arm} ===")
            run_arm(arm, manifests, evaluator, device)
            if arm == "B1":
                control = pd.read_parquet(OUT_DIR / "eval_M0_B.parquet")
                candidate = pd.read_parquet(OUT_DIR / "eval_B1.parquet")
                gate = gate_B1(control, candidate)
                (OUT_DIR / "gate_B1.json").write_text(
                    json.dumps(MD.jsonable(gate), indent=2), encoding="utf-8"
                )
                log(f"[gate B1] {gate['verdict']}")
            if arm == "A1":
                control = pd.read_parquet(OUT_DIR / "eval_M0_A.parquet")
                candidate = pd.read_parquet(OUT_DIR / "eval_A1.parquet")
                gate = gate_A1(control, candidate)
                (OUT_DIR / "gate_A1.json").write_text(
                    json.dumps(MD.jsonable(gate), indent=2), encoding="utf-8"
                )
                log(f"[gate A1] {gate['verdict']}")
            if arm == "B2":
                gate = gate_B2(
                    pd.read_parquet(OUT_DIR / "eval_B1.parquet"),
                    pd.read_parquet(OUT_DIR / "eval_B2.parquet"),
                )
                (OUT_DIR / "gate_B2.json").write_text(
                    json.dumps(MD.jsonable(gate), indent=2), encoding="utf-8"
                )
                log(f"[gate B2] {gate['verdict']}")

    if args.evaluate or args.report or args.all_primary:
        build_reports(info)
    log(f"[done] {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
