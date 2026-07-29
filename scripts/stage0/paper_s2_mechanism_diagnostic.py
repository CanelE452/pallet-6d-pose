#!/usr/bin/env python3
"""paper_s2_mechanism_diagnostic.py — PAPER_S2 ep57 mechanism diagnosis harness.

목적(진단 전용):
  새 공개 데이터셋에 적용할 **architecture/loss 선택 근거**를 만든다.  성능 개선도,
  새 checkpoint 도 산출물이 아니다.  산출물은 두 가지다.

    1. 각 실패 프레임에서 정보가 최초로 깨지는 단계 (first_break_stage)
    2. 그 단계에 대응하는 architecture/loss 후보의 우선순위

설계 원칙:
  * 의심 하나마다 새 감사 스크립트를 만들지 않는다.  **한 번의 model forward 결과를
    cache** 한 뒤 decoder / PnP / GT 치환 / counterfactual 을 오프라인에서 반복한다.
  * 기존 감사 스크립트의 기능을 복사해 재구현하지 않는다.  공용 함수는
    ``paper_s2_frozen_diagnostic`` / ``paper_s2_decoder_parity_audit`` /
    ``filter_pr_camfacing`` 에서 **import 해서 재사용**한다.
  * 기존 데이터·checkpoint 는 read-only.  final-test 세션은 fail-closed 로 차단한다.

Subcommands:
    --build-cache      manifest + one-pass frozen inference cache
    --counterfactuals  same-source image counterfactual (C0~C11)
    --interventions    decoder(D0~D5) + oracle keypoint(O0~O11, ablation, LOO)
    --report           failure class / first-break / decision matrix / figures
    --all              위 전부

Usage:
    python scripts/stage0/paper_s2_mechanism_diagnostic.py --all
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import torch

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

# --- 공용 함수 재사용 (재구현 금지) -----------------------------------------
import paper_s2_frozen_diagnostic as FZ  # noqa: E402
import paper_s2_decoder_parity_audit as DP  # noqa: E402
import annotate_pnp as APNP  # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402

OUT_DIR = ROOT / "data/pallet/results/paper_s2_mechanism_diagnostic"
CACHE_NPZ = OUT_DIR / "frozen_tensor_cache.npz"
CF_CACHE_NPZ = OUT_DIR / "counterfactual_tensor_cache.npz"
MANIFEST_PATH = OUT_DIR / "mechanism_val_manifest.json"
CACHE_MANIFEST_PATH = OUT_DIR / "CACHE_MANIFEST.json"

N_KP = FZ.N_KEYPOINTS
BELIEF = 50
BELIEF_THRESHOLD = FZ.BELIEF_THRESHOLD
NEAR_KP = (0, 1, 2, 3)
FAR_KP = (4, 5, 6, 7)
TOP_KP = (0, 1, 4, 5)
BOTTOM_KP = (2, 3, 6, 7)
# camera-facing 0123: near face = 0,1,2,3 / far face = 4,5,6,7.
# depth edges connect near<->far at the same top/bottom + left/right corner.
# LEFT column  = {0 near-top-L, 3 near-bottom-L, 4 far-top-L, 7 far-bottom-L}
# RIGHT column = {1 near-top-R, 2 near-bottom-R, 5 far-top-R, 6 far-bottom-R}
DEPTH_LEFT_KP = (0, 3, 4, 7)
DEPTH_RIGHT_KP = (1, 2, 5, 6)

# --- Phase E thresholds (diagnostic configuration; sensitivity swept later) --
THRESH = {
    "f1_min_detected": 6,
    "f1_min_far_detected": 2,
    "f1_min_frame_median_peak": 0.3,
    "f2_matched_error_px": 20.0,
    "f3_matched_error_px": 10.0,
    "f3_yaw_deg": 5.0,
}
SENSITIVITY_SCALES = (0.75, 1.0, 1.25)

# --- Phase D expected baseline (frozen audit full_ep57_frozen_20260728) ------
BASELINE_EXPECT = {
    "strict_n": 87,
    "gt2d_pose_success": 87,
    "pred_pose_success": 70,
    "yaw_median_deg": 6.025,
    "fixed_gt_reproj_median_px": 23.162,
}
BASELINE_TOL = {"yaw_median_deg": 0.1, "fixed_gt_reproj_median_px": 0.25}

COUNTERFACTUAL_N = 24
COUNTERFACTUAL_SEED = 20260729
GEOMETRY_TOL_PX = 0.1


# ============================================================================
# small helpers
# ============================================================================
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def jsonable(value: Any) -> Any:
    return FZ.jsonable(value)


def finite(value: Any) -> Optional[float]:
    return FZ.finite_float(value)


def nanmedian(values: Iterable[Any]) -> Optional[float]:
    arr = np.asarray([v for v in values if finite(v) is not None], dtype=np.float64)
    return float(np.median(arr)) if arr.size else None


def git_head() -> Optional[str]:
    return FZ.git_head()


def log(message: str) -> None:
    print(message, flush=True)


# ============================================================================
# Phase B — mechanism-val manifest
# ============================================================================
def image_conditions(image_bgr: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "luma_p10": float(np.percentile(gray, 10)),
        "luma_p50": float(np.percentile(gray, 50)),
        "luma_p90": float(np.percentile(gray, 90)),
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def distance_bin(value: Any) -> Optional[str]:
    number = finite(value)
    if number is None:
        return None
    if number < 2.0:
        return "d<2m"
    if number < 4.0:
        return "2-4m"
    if number < 6.0:
        return "4-6m"
    return "d>=6m"


def build_manifest(audit: FZ.InputAudit) -> dict[str, Any]:
    """strict N87 primary + manual36/synthetic exploratory, final-test fail-closed."""
    rows = FZ.safe_stage25_filterval()  # identity-locked membership (sealed-safe)
    synth = FZ.synth_frames(audit)

    frames: list[dict[str, Any]] = []
    for spec in list(rows) + list(synth):
        image_path = Path(spec["png"])
        json_path = Path(spec["json"])
        # fail-closed: prohibited session name / path prefix
        audit.guard(image_path)
        audit.guard(json_path)

        data = audit.read_json(json_path)
        objects = data.get("objects") or []
        obj = objects[0] if objects else {}
        gt_points = FZ.gt_points_from_object(obj)
        intrinsics = FZ.intrinsics_from_json(data)
        dims = FZ.dims_from_frame(spec, obj)
        pose = FZ.stored_pose_from_object(obj)

        image = audit.read_image(image_path)
        if image is None:
            raise RuntimeError(f"unreadable image: {image_path}")
        height, width = image.shape[:2]

        finite_pts = [p for p in gt_points[:8] if FZ.point_valid(p)]
        n_in = sum(
            1 for p in gt_points[:8] if FZ.point_inside(p, width, height) is True
        )
        if finite_pts:
            arr = np.asarray(finite_pts, dtype=np.float64)
            bx0, by0 = arr[:, 0].min(), arr[:, 1].min()
            bx1, by1 = arr[:, 0].max(), arr[:, 1].max()
        else:
            bx0 = by0 = bx1 = by1 = float("nan")
        bbox_w, bbox_h = bx1 - bx0, by1 - by0

        # truncation (NOT occlusion): fewer than 8 GT corners land inside the
        # image rectangle.  Identical definition to paper_s2_frozen_diagnostic
        # (`gt_truncated`), so the 17/70 split stays comparable.
        is_truncated = bool(n_in < 8)

        elevation = azimuth = None
        if pose is not None:
            try:
                from stage18_elevation_threshold import elev_from_pose

                elevation = finite(elev_from_pose(pose["R"], pose["t"]))
            except Exception:
                elevation = None
            azimuth = finite(math.degrees(
                math.atan2(float(pose["R"][0, 2]), float(pose["R"][2, 2]))
            ))

        conditions = image_conditions(image)
        role = spec["split_role"]
        record = {
            "frame_id": f"{spec['dataset']}:{spec['domain']}:{spec['fid']}",
            "dataset": spec["dataset"],
            "domain": spec["domain"],
            "session_id": spec["source_session"],
            "fid": spec["fid"],
            "json_path": str(json_path),
            "image_path": str(image_path),
            "role": role,
            "population": (
                "primary" if role == "strict_filterval" else "exploratory"
            ),
            "is_final_test": False,
            "is_truncated": is_truncated,
            "n_gt_inframe": int(n_in),
            "n_gt_valid": int(len(finite_pts)),
            "image_width": int(width),
            "image_height": int(height),
            "bbox_x": finite(bx0),
            "bbox_y": finite(by0),
            "bbox_width": finite(bbox_w),
            "bbox_height": finite(bbox_h),
            "bbox_area_ratio": finite(
                (bbox_w * bbox_h) / float(width * height)
                if np.isfinite(bbox_w) and np.isfinite(bbox_h)
                else None
            ),
            "luma_p10": conditions["luma_p10"],
            "luma_p50": conditions["luma_p50"],
            "luma_p90": conditions["luma_p90"],
            "blur_score": conditions["blur_score"],
            "distance_m": finite(
                None if pose is None else float(np.linalg.norm(pose["t"]))
            ),
            "distance_bin": distance_bin(
                None if pose is None else float(np.linalg.norm(pose["t"]))
            ),
            "elevation_deg": elevation,
            "azimuth_deg": azimuth,
            "dim_W_m": None if dims is None else dims[0],
            "dim_D_m": None if dims is None else dims[1],
            "dim_H_m": None if dims is None else dims[2],
            "camera_K_hash": (
                None
                if intrinsics is None
                else sha256_text(
                    json.dumps(np.asarray(intrinsics).round(9).tolist())
                )[:16]
            ),
            "gt_source": obj.get("gt_source"),
            "gt_pose_available": pose is not None,
            "manual_metadata_available": bool(obj.get("manual_annotation", False)),
            "extrapolated_metadata_available": bool(
                obj.get("extrapolated_mask") is not None
            ),
            "split_provenance": spec["legacy_aggregate"],
            "entry": spec.get("entry"),
        }
        frames.append(record)

    primary = [f for f in frames if f["population"] == "primary"]
    if len(primary) != BASELINE_EXPECT["strict_n"]:
        raise RuntimeError(f"primary population changed: {len(primary)}")
    trunc = sum(1 for f in primary if f["is_truncated"])
    if trunc != 17 or len(primary) - trunc != 70:
        raise RuntimeError(f"truncation split changed: {trunc}/{len(primary)}")

    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "mechanism diagnosis for architecture/loss selection",
        "checkpoint": str(FZ.WEIGHTS),
        "checkpoint_sha256": FZ.WEIGHTS_SHA256,
        "git_head": git_head(),
        "populations": {
            "primary_strict_filterval": len(primary),
            "primary_truncated": trunc,
            "primary_non_truncated": len(primary) - trunc,
            "exploratory_manual36": sum(
                1 for f in frames if f["role"] == "exploratory_pl_pool_manual"
            ),
            "exploratory_synthetic": sum(
                1 for f in frames if f["role"] == "synthetic_fixed_val"
            ),
        },
        "final_test_guard": {
            "sealed_sessions": sorted(FZ.SEALED_SESSIONS),
            "prohibited_tokens": list(FZ.PROHIBITED_INPUT_TOKENS),
            "prohibited_attempts": list(audit.prohibited_attempts),
            "final_test_open_count": 0,
        },
        "thresholds": THRESH,
        "frames": frames,
    }
    manifest["membership_sha256"] = sha256_text(
        json.dumps([f["frame_id"] for f in frames], sort_keys=True)
    )
    return manifest


# ============================================================================
# Phase C — one-pass frozen cache
# ============================================================================
def cache_key(manifest: dict[str, Any]) -> dict[str, Any]:
    parts = {
        "checkpoint_sha256": FZ.WEIGHTS_SHA256,
        "manifest_sha256": sha256_text(
            json.dumps(manifest["frames"], sort_keys=True, default=str)
        ),
        "model_source_sha256": FZ.sha256_file(
            ROOT / "Deep_Object_Pose/common/models.py"
        ),
        "frozen_helper_sha256": FZ.sha256_file(
            ROOT / "scripts/stage0/paper_s2_frozen_diagnostic.py"
        ),
        "preprocess_config": json.dumps(
            {
                "kind": "anisotropic_squash",
                "input_size": FZ.INPUT_SIZE,
                "mean": FZ.MEAN.tolist(),
                "std": FZ.STD.tolist(),
                "interpolation": "INTER_LINEAR",
            },
            sort_keys=True,
        ),
        "decoder_config": json.dumps(
            {
                "local_radius": FZ.LOCAL_RADIUS,
                "local_temperature": FZ.LOCAL_TEMPERATURE,
                "belief_threshold": BELIEF_THRESHOLD,
                "canonical_offset": 0.4395,
            },
            sort_keys=True,
        ),
        "script_sha256": FZ.sha256_file(Path(__file__).resolve()),
    }
    parts["cache_key"] = sha256_text(json.dumps(parts, sort_keys=True))
    return parts


def forward_all_stages(
    model: Any, image_bgr: np.ndarray, device: torch.device
) -> dict[str, np.ndarray]:
    """One forward; keep every belief stage, final affinity, final seg."""
    tensor = FZ.preprocess_squash(image_bgr).to(device)
    with torch.inference_mode():
        outputs = model(tensor)
    beliefs, affinities = outputs[0], outputs[1]
    seg = outputs[3] if len(outputs) > 3 and outputs[3] is not None else None
    stage_stack = np.stack(
        [b[0, :N_KP].detach().float().cpu().numpy() for b in beliefs], axis=0
    )  # (6, 9, 50, 50)
    # belief stays float32: every decoded coordinate and pose is derived from it,
    # and a float16 round-trip perturbs the temperature-0.1 local softargmax
    # enough to flip PnP candidate selection (baseline reproduction gate).
    # Affinity/seg are diagnostic-only, so float16 is fine there.
    result = {
        "belief_stages": stage_stack.astype(np.float32),
        "affinity_final": affinities[-1][0].detach().float().cpu().numpy().astype(
            np.float16
        ),
    }
    if seg is not None:
        result["seg_final"] = (
            seg[-1][0].detach().float().cpu().numpy().astype(np.float16)
        )
    return result


def decode_all(belief: np.ndarray, scale_x: float, scale_y: float,
               gt_points: list[Optional[list[float]]]) -> dict[str, Any]:
    """Every decoder variant on ONE cached heatmap (cache-only, no forward)."""
    out: dict[str, Any] = {}
    tensor = torch.from_numpy(belief.astype(np.float32)).unsqueeze(0)

    # D0 — training local softargmax (clamped window), == frozen Y2 input
    stats = [
        FZ.heatmap_stats(belief[k], scale_x, scale_y, gt_points[k])
        for k in range(N_KP)
    ]
    out["_stats"] = stats
    out["D0"] = [
        FZ.point_xy(s.get("_soft_px")) if s.get("detected") else None for s in stats
    ]
    # D4 — raw argmax
    out["D4"] = [
        FZ.point_xy(s.get("_arg_px")) if s.get("detected") else None for s in stats
    ]

    # D1 — border-safe local softargmax (valid indices only)
    d1 = []
    for k in range(N_KP):
        gx, gy, _ = DP.d1_border_safe(belief[k])
        d1.append(
            [gx * scale_x, gy * scale_y] if stats[k].get("detected") else None
        )
    out["D1"] = d1

    # D2 — canonical evaluation decoder (gaussian + NMS + 11x11 centroid + 0.4395)
    canonical = extract_keypoints_from_belief(belief, threshold=BELIEF_THRESHOLD)
    out["D2"] = [
        None if (kp[0] < 0 and kp[1] < 0) else [kp[0] * scale_x, kp[1] * scale_y]
        for kp in canonical
    ]
    out["D2_peak"] = [float(kp[2]) for kp in canonical]

    # D3 — canonical decoder without the +0.4395 offset
    out["D3"] = [
        None
        if p is None
        else [p[0] - 0.4395 * scale_x, p[1] - 0.4395 * scale_y]
        for p in out["D2"]
    ]

    # D5 — peak-local weighted centroid, canonical threshold/missing kept
    d5 = []
    for k in range(N_KP):
        if float(belief[k].max()) < BELIEF_THRESHOLD:
            d5.append(None)
            continue
        gx, gy, _ = DP.d3_eval_style_centroid(belief[k])
        d5.append([gx * scale_x, gy * scale_y])
    out["D5"] = d5
    del tensor
    return out


def build_cache(force: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = FZ.InputAudit()

    if MANIFEST_PATH.is_file() and not force:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        log(f"[manifest] reuse ({len(manifest['frames'])} frames)")
    else:
        log("[manifest] building ...")
        manifest = build_manifest(audit)
        MANIFEST_PATH.write_text(
            json.dumps(jsonable(manifest), indent=2), encoding="utf-8"
        )
        log(f"[manifest] wrote {MANIFEST_PATH} ({len(manifest['frames'])} frames)")

    key = cache_key(manifest)
    reusable = (
        CACHE_MANIFEST_PATH.is_file()
        and CACHE_NPZ.is_file()
        and (OUT_DIR / "keypoints.parquet").is_file()
        and not force
    )
    if reusable:
        stored = json.loads(CACHE_MANIFEST_PATH.read_text(encoding="utf-8"))
        if stored.get("cache_key") == key["cache_key"]:
            log(f"[cache] hit {key['cache_key'][:16]} — 0 model forwards")
            return manifest, stored
        log(
            f"[cache] MISS stored={str(stored.get('cache_key'))[:16]} "
            f"current={key['cache_key'][:16]} — rebuilding"
        )

    device = FZ.choose_device("auto")
    model, n_state = FZ.load_model(device)
    log(f"[model] ep57 loaded ({n_state} tensors) on {device}")

    frames_rows: list[dict[str, Any]] = []
    kp_rows: list[dict[str, Any]] = []
    tensors: dict[str, np.ndarray] = {}
    started = time.time()

    for index, spec in enumerate(manifest["frames"]):
        if spec["population"] != "primary":
            continue  # tensor cache is primary-only; exploratory stays separate
        image = audit.read_image(spec["image_path"])
        data = audit.read_json(spec["json_path"])
        obj = (data.get("objects") or [{}])[0]
        gt_points = FZ.gt_points_from_object(obj)
        height, width = image.shape[:2]
        scale_x, scale_y = width / BELIEF, height / BELIEF

        cached = forward_all_stages(model, image, device)
        uid = spec["frame_id"]
        tensors[f"{uid}|belief_stages"] = cached["belief_stages"]
        tensors[f"{uid}|affinity_final"] = cached["affinity_final"]
        if "seg_final" in cached:
            tensors[f"{uid}|seg_final"] = cached["seg_final"]

        stages = cached["belief_stages"].astype(np.float32)
        final = stages[-1]
        decoded = decode_all(final, scale_x, scale_y, gt_points)
        stats = decoded["_stats"]

        # ---- per-keypoint, per-stage rows ---------------------------------
        for k in range(N_KP):
            gt = gt_points[k]
            gt_xy = FZ.point_xy(gt)
            raw = obj.get("projected_cuboid")
            raw_pt = (
                raw[k] if isinstance(raw, list) and k < len(raw) else None
            )
            exact_sentinel = bool(
                raw_pt is not None
                and np.asarray(raw_pt, dtype=np.float64).reshape(-1)[0] == -1.0
                and np.asarray(raw_pt, dtype=np.float64).reshape(-1)[1] == -1.0
            )
            in_frame = FZ.point_inside(gt, width, height)
            row: dict[str, Any] = {
                "frame_id": uid,
                "domain": spec["domain"],
                "session_id": spec["session_id"],
                "is_truncated": spec["is_truncated"],
                "keypoint": k,
                "group_near_far": "near" if k in NEAR_KP else ("far" if k in FAR_KP else "centroid"),
                "group_top_bottom": "top" if k in TOP_KP else ("bottom" if k in BOTTOM_KP else "centroid"),
                "group_depth_side": (
                    "left" if k in DEPTH_LEFT_KP else ("right" if k in DEPTH_RIGHT_KP else "centroid")
                ),
                "gt_x": None if gt_xy is None else gt_xy[0],
                "gt_y": None if gt_xy is None else gt_xy[1],
                "gt_valid": gt_xy is not None,
                "exact_missing_sentinel": exact_sentinel,
                "legitimate_off_image": bool(
                    gt_xy is not None and in_frame is False
                ),
                "gt_in_frame": in_frame,
                "distance_to_border_px": (
                    None
                    if gt_xy is None
                    else float(
                        min(gt_xy[0], gt_xy[1], width - 1 - gt_xy[0], height - 1 - gt_xy[1])
                    )
                ),
                "final_peak": stats[k].get("peak"),
                "final_second_peak": stats[k].get("second_peak"),
                "final_peak_ratio": stats[k].get("peak_second_ratio"),
                "final_entropy": stats[k].get("entropy_nats"),
                "final_entropy_norm": stats[k].get("entropy_normalized"),
                "cov_px_xx": stats[k].get("cov_px_xx"),
                "cov_px_xy": stats[k].get("cov_px_xy"),
                "cov_px_yy": stats[k].get("cov_px_yy"),
                "cov_eig_major_px2": stats[k].get("cov_eig_major_px2"),
                "cov_eig_minor_px2": stats[k].get("cov_eig_minor_px2"),
                "mahalanobis_gt": stats[k].get("mahalanobis_gt"),
                "detected": bool(stats[k].get("detected")),
                "missing": not bool(stats[k].get("detected")),
            }
            for name in ("D0", "D1", "D2", "D3", "D4", "D5"):
                point = decoded[name][k]
                row[f"{name}_x"] = None if point is None else point[0]
                row[f"{name}_y"] = None if point is None else point[1]
                row[f"{name}_err_px"] = FZ.euclidean(point, gt)
                row[f"{name}_missing"] = point is None
            for stage_index in range(stages.shape[0]):
                s = FZ.heatmap_stats(
                    stages[stage_index][k], scale_x, scale_y, gt
                )
                row[f"stage{stage_index + 1}_peak"] = s.get("peak")
                row[f"stage{stage_index + 1}_argmax_x"] = s.get("argmax_x")
                row[f"stage{stage_index + 1}_argmax_y"] = s.get("argmax_y")
                row[f"stage{stage_index + 1}_soft_x"] = s.get("softargmax_x")
                row[f"stage{stage_index + 1}_soft_y"] = s.get("softargmax_y")
                row[f"stage{stage_index + 1}_err_px"] = s.get(
                    "softargmax_error_gt_px"
                )
                row[f"stage{stage_index + 1}_entropy"] = s.get("entropy_nats")
                row[f"stage{stage_index + 1}_peak_ratio"] = s.get(
                    "peak_second_ratio"
                )
                row[f"stage{stage_index + 1}_detected"] = bool(s.get("detected"))
            kp_rows.append(row)

        peaks = [stats[k].get("peak") for k in range(N_KP)]
        far_detected = sum(1 for k in FAR_KP if stats[k].get("detected"))
        frames_rows.append(
            {
                "frame_id": uid,
                "domain": spec["domain"],
                "session_id": spec["session_id"],
                "is_truncated": spec["is_truncated"],
                "image_width": width,
                "image_height": height,
                "belief_scale_x": scale_x,
                "belief_scale_y": scale_y,
                "n_detected_D0": sum(1 for p in decoded["D0"] if p is not None),
                "n_detected_D2": sum(1 for p in decoded["D2"] if p is not None),
                "n_corner_detected_D2": sum(
                    1 for p in decoded["D2"][:8] if p is not None
                ),
                "n_far_detected": far_detected,
                "frame_median_peak": nanmedian(peaks),
                "frame_min_peak": finite(np.nanmin(np.asarray(peaks, dtype=np.float64))),
                "luma_p10": spec["luma_p10"],
                "luma_p50": spec["luma_p50"],
                "blur_score": spec["blur_score"],
                "bbox_area_ratio": spec["bbox_area_ratio"],
                "elevation_deg": spec["elevation_deg"],
                "distance_m": spec["distance_m"],
                "distance_bin": spec["distance_bin"],
                "n_gt_inframe": spec["n_gt_inframe"],
            }
        )
        if (index + 1) % 10 == 0:
            log(f"[cache] {len(frames_rows)} frames forwarded")

    np.savez_compressed(CACHE_NPZ, **tensors)
    pd.DataFrame(frames_rows).to_parquet(OUT_DIR / "frames.parquet", index=False)
    pd.DataFrame(kp_rows).to_parquet(OUT_DIR / "keypoints.parquet", index=False)

    stored = dict(key)
    stored.update(
        {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "n_frames_cached": len(frames_rows),
            "n_keypoint_rows": len(kp_rows),
            "tensor_entries": len(tensors),
            "tensor_bytes": int(CACHE_NPZ.stat().st_size),
            "elapsed_seconds": time.time() - started,
            "model_forwards": len(frames_rows),
            "input_audit": {
                "json_opened": len(audit.json_paths),
                "images_opened": len(audit.image_paths),
                "prohibited_attempts": list(audit.prohibited_attempts),
            },
        }
    )
    CACHE_MANIFEST_PATH.write_text(
        json.dumps(jsonable(stored), indent=2), encoding="utf-8"
    )
    log(
        f"[cache] built {len(frames_rows)} frames in "
        f"{stored['elapsed_seconds']:.1f}s ({stored['tensor_bytes'] / 1e6:.1f} MB)"
    )
    return manifest, stored


def load_cached_tensors() -> Any:
    if not CACHE_NPZ.is_file():
        raise FileNotFoundError(f"tensor cache missing: {CACHE_NPZ}")
    return np.load(CACHE_NPZ)


# ============================================================================
# pose solving on cached correspondences
# ============================================================================
class FrameGeometry:
    """Per-frame GT/K/dims context, PnP with an in-frame memo cache."""

    def __init__(self, spec: dict[str, Any], audit: FZ.InputAudit) -> None:
        data = audit.read_json(spec["json_path"])
        obj = (data.get("objects") or [{}])[0]
        self.spec = spec
        self.obj = obj
        self.gt_points = FZ.gt_points_from_object(obj)
        self.K = FZ.intrinsics_from_json(data)
        self.dims = FZ.dims_from_frame(spec, obj)
        self.gt_pose = FZ.stored_pose_from_object(obj)
        self.shape = (int(spec["image_height"]), int(spec["image_width"]), 3)
        self.solver = FZ.CurrentSolveCache(
            self.K, self.dims, self.shape, auto_swap_dims=True
        )

    def solve(self, points: list[Optional[list[float]]]) -> Optional[dict[str, Any]]:
        pose, _, _, _ = self.solver.solve(points)
        return pose

    def metrics(
        self, pose: Optional[dict[str, Any]], reference: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        reference = self.gt_pose if reference is None else reference
        reproj, n_obs = FZ.fixed_observation_reprojection(
            pose, self.gt_points, self.K, self.dims
        )
        yaw = None if pose is None else FZ.yaw_deg(pose["R"])
        yaw_ref = None if reference is None else FZ.yaw_deg(reference["R"])
        yaw_err = (
            None
            if (yaw is None or yaw_ref is None)
            else abs(FZ.wrap180(yaw - yaw_ref))
        )
        translation_err = (
            None
            if (pose is None or reference is None)
            else float(np.linalg.norm(np.asarray(pose["t"]) - np.asarray(reference["t"])))
        )
        return {
            "pose_success": pose is not None,
            "yaw_deg": yaw,
            "yaw_err_deg": yaw_err,
            "translation_err_m": translation_err,
            "reproj_fixed_gt_px": reproj,
            "n_fixed_observations": n_obs,
            "add_m": FZ.add_error(pose, reference, self.dims),
            "adds_m": FZ.adds_error(pose, reference, self.dims),
            "rotation_err_deg": (
                None
                if (pose is None or reference is None)
                else FZ.rotation_error_deg(pose["R"], reference["R"])
            ),
        }

    def matched_2d_error(
        self, points: list[Optional[list[float]]]
    ) -> dict[str, Any]:
        overall = FZ.order_free_corner_metrics(points, self.gt_points)
        far = FZ.order_free_corner_metrics(
            [points[k] if k in FAR_KP else None for k in range(8)],
            [self.gt_points[k] if k in FAR_KP else None for k in range(8)],
        )
        near = FZ.order_free_corner_metrics(
            [points[k] if k in NEAR_KP else None for k in range(8)],
            [self.gt_points[k] if k in NEAR_KP else None for k in range(8)],
        )
        return {
            "matched_count": overall["matched_count"],
            "matched_median_px": overall["median_px"],
            "matched_mean_px": overall["mean_px"],
            "matched_max_px": overall["max_px"],
            "far_matched_median_px": far["median_px"],
            "near_matched_median_px": near["median_px"],
        }


# ============================================================================
# Phase D — baseline reproduction gate
# ============================================================================
def baseline_gate(
    manifest: dict[str, Any], geometries: dict[str, FrameGeometry],
    decoded_by_frame: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    primary = [f for f in manifest["frames"] if f["population"] == "primary"]
    gt_success = pred_success = 0
    yaws: list[float] = []
    reprojs: list[float] = []
    for spec in primary:
        geometry = geometries[spec["frame_id"]]
        oracle = geometry.solve(geometry.gt_points)
        predicted = geometry.solve(decoded_by_frame[spec["frame_id"]]["D0"])
        gt_success += int(oracle is not None)
        pred_success += int(predicted is not None)
        if predicted is not None and oracle is not None:
            yaws.append(
                abs(FZ.wrap180(FZ.yaw_deg(predicted["R"]) - FZ.yaw_deg(oracle["R"])))
            )
            value, _ = FZ.fixed_observation_reprojection(
                predicted, geometry.gt_points, geometry.K, geometry.dims
            )
            if value is not None:
                reprojs.append(value)
    result = {
        "strict_n": len(primary),
        "gt2d_pose_success": gt_success,
        "pred_pose_success": pred_success,
        "yaw_median_deg": nanmedian(yaws),
        "fixed_gt_reproj_median_px": nanmedian(reprojs),
        "expected": BASELINE_EXPECT,
        "tolerance": BASELINE_TOL,
    }
    problems = []
    if result["strict_n"] != BASELINE_EXPECT["strict_n"]:
        problems.append("frame membership mismatch")
    if result["gt2d_pose_success"] != BASELINE_EXPECT["gt2d_pose_success"]:
        problems.append("GT-2D PnP success mismatch")
    if result["pred_pose_success"] != BASELINE_EXPECT["pred_pose_success"]:
        problems.append("predicted pose success mismatch")
    for name, tol in BASELINE_TOL.items():
        actual, expected = result[name], BASELINE_EXPECT[name]
        if actual is None or abs(actual - expected) > tol:
            problems.append(f"{name} {actual} vs {expected} (tol {tol})")
    result["problems"] = problems
    result["passed"] = not problems
    return result


# ============================================================================
# Phase F/G — decoder + oracle intervention engine
# ============================================================================
def replace(
    points: list[Optional[list[float]]],
    gt_points: list[Optional[list[float]]],
    indices: Iterable[int],
) -> list[Optional[list[float]]]:
    out = [None if p is None else list(p) for p in points]
    for index in indices:
        out[index] = None if gt_points[index] is None else list(gt_points[index])
    return out


def drop(
    points: list[Optional[list[float]]], indices: Iterable[int]
) -> list[Optional[list[float]]]:
    out = [None if p is None else list(p) for p in points]
    for index in indices:
        out[index] = None
    return out


def intervention_specs(
    baseline: list[Optional[list[float]]],
    gt_points: list[Optional[list[float]]],
    peaks: list[Optional[float]],
) -> list[tuple[str, str, list[Optional[list[float]]]]]:
    specs: list[tuple[str, str, list[Optional[list[float]]]]] = [
        ("O0", "baseline", [None if p is None else list(p) for p in baseline])
    ]
    for k in range(N_KP):
        specs.append((f"O1_kp{k}", "single_gt", replace(baseline, gt_points, [k])))
    specs += [
        ("O2_near", "group_gt", replace(baseline, gt_points, NEAR_KP)),
        ("O3_far", "group_gt", replace(baseline, gt_points, FAR_KP)),
        ("O4_top", "group_gt", replace(baseline, gt_points, TOP_KP)),
        ("O5_bottom", "group_gt", replace(baseline, gt_points, BOTTOM_KP)),
        ("O6_depth_left", "group_gt", replace(baseline, gt_points, DEPTH_LEFT_KP)),
        ("O7_depth_right", "group_gt", replace(baseline, gt_points, DEPTH_RIGHT_KP)),
        ("O8_kp5_kp6", "pair_gt", replace(baseline, gt_points, (5, 6))),
        ("O9_centroid", "single_gt", replace(baseline, gt_points, (8,))),
        ("O10_all_corners", "group_gt", replace(baseline, gt_points, range(8))),
        ("O11_all", "group_gt", replace(baseline, gt_points, range(9))),
    ]
    for k in range(N_KP):
        specs.append((f"A_drop_kp{k}", "ablation_drop", drop(baseline, [k])))
    specs.append(("A_drop_far", "ablation_drop", drop(baseline, FAR_KP)))
    specs.append(("A_drop_centroid", "ablation_drop", drop(baseline, (8,))))
    detected = [
        (peaks[k], k) for k in range(N_KP) if baseline[k] is not None and peaks[k] is not None
    ]
    if detected:
        weakest = min(detected)[1]
        specs.append(
            (f"A_drop_lowest_conf(kp{weakest})", "ablation_drop", drop(baseline, [weakest]))
        )
    for k in range(N_KP):
        if baseline[k] is not None:
            specs.append((f"LOO_kp{k}", "leave_one_out", drop(baseline, [k])))
    return specs


def run_interventions(
    manifest: dict[str, Any], audit: FZ.InputAudit
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tensors = load_cached_tensors()
    primary = [f for f in manifest["frames"] if f["population"] == "primary"]

    geometries: dict[str, FrameGeometry] = {}
    decoded_by_frame: dict[str, dict[str, Any]] = {}
    for spec in primary:
        geometry = FrameGeometry(spec, audit)
        geometries[spec["frame_id"]] = geometry
        final = tensors[f"{spec['frame_id']}|belief_stages"].astype(np.float32)[-1]
        scale_x = spec["image_width"] / BELIEF
        scale_y = spec["image_height"] / BELIEF
        decoded_by_frame[spec["frame_id"]] = decode_all(
            final, scale_x, scale_y, geometry.gt_points
        )

    gate = baseline_gate(manifest, geometries, decoded_by_frame)
    log(f"[baseline gate] {json.dumps({k: gate[k] for k in ('strict_n','gt2d_pose_success','pred_pose_success','yaw_median_deg','fixed_gt_reproj_median_px','passed')}, default=str)}")
    if not gate["passed"]:
        raise RuntimeError(f"BLOCKED baseline reproduction: {gate['problems']}")

    pose_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []

    for spec in primary:
        uid = spec["frame_id"]
        geometry = geometries[uid]
        decoded = decoded_by_frame[uid]
        stats = decoded["_stats"]
        peaks = [stats[k].get("peak") for k in range(N_KP)]
        oracle = geometry.solve(geometry.gt_points)

        # ---- Phase F: decoder interventions (same cached heatmap) ----------
        for name in ("D0", "D1", "D2", "D3", "D4", "D5"):
            points = decoded[name]
            pose = geometry.solve(points)
            row = {
                "frame_id": uid,
                "domain": spec["domain"],
                "session_id": spec["session_id"],
                "is_truncated": spec["is_truncated"],
                "family": "decoder",
                "variant": name,
                "n_detected": sum(1 for p in points if p is not None),
                "n_corner_detected": sum(1 for p in points[:8] if p is not None),
            }
            row.update(geometry.matched_2d_error(points))
            row.update(geometry.metrics(pose, oracle))
            pose_rows.append(row)

        baseline_points = decoded["D0"]
        # ---- Phase G: oracle keypoint interventions ------------------------
        for name, family, points in intervention_specs(
            baseline_points, geometry.gt_points, peaks
        ):
            n_valid = sum(1 for p in points if FZ.point_valid(p))
            pose = geometry.solve(points) if n_valid >= 4 else None
            row = {
                "frame_id": uid,
                "domain": spec["domain"],
                "session_id": spec["session_id"],
                "is_truncated": spec["is_truncated"],
                "family": family,
                "variant": name,
                "n_correspondences": n_valid,
                "min_correspondence_ok": n_valid >= 4,
            }
            row.update(geometry.matched_2d_error(points))
            row.update(geometry.metrics(pose, oracle))
            row["wd_hypothesis"] = (
                None if pose is None else pose.get("_wd_hypothesis")
            )
            row["selected_dims"] = (
                None if pose is None else json.dumps(list(pose.get("dims", ())))
            )
            intervention_rows.append(row)

    poses = pd.DataFrame(pose_rows)
    interventions = pd.DataFrame(intervention_rows)
    poses.to_parquet(OUT_DIR / "poses.parquet", index=False)
    interventions.to_parquet(OUT_DIR / "interventions.parquet", index=False)
    log(
        f"[interventions] decoder rows={len(poses)} oracle rows={len(interventions)}"
    )
    return poses, interventions, gate


# ============================================================================
# Phase I — same-image counterfactuals
# ============================================================================
def affine_translate(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def affine_scale(s: float) -> np.ndarray:
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])


def apply_affine(A: np.ndarray, point: Optional[list[float]]) -> Optional[list[float]]:
    if point is None:
        return None
    vector = A @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if abs(vector[2]) < 1e-12:
        return None
    return [float(vector[0] / vector[2]), float(vector[1] / vector[2])]


def geometry_unit_test(
    A: np.ndarray, K: np.ndarray, pose: dict[str, Any],
    dims: tuple[float, float, float]
) -> float:
    """max |project(K') - A·project(K)| must be ~0 (exact algebraic identity)."""
    object_points = APNP.make_pallet_keypoints_3d(*dims)
    base = APNP.project_3d(object_points, pose["R"], pose["t"], K)
    transformed = APNP.project_3d(object_points, pose["R"], pose["t"], A @ K)
    worst = 0.0
    for original, moved in zip(base, transformed):
        if not FZ.point_valid(original) or not FZ.point_valid(moved):
            continue
        expected = apply_affine(A, list(original))
        worst = max(worst, float(np.linalg.norm(np.asarray(expected) - np.asarray(moved))))
    return worst


def make_counterfactual(
    image: np.ndarray, spec: dict[str, Any], kind: str
) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
    """Return (image, affine A original->new, note). A maps original px -> new px."""
    height, width = image.shape[:2]
    bx, by = spec["bbox_x"], spec["bbox_y"]
    bw, bh = spec["bbox_width"], spec["bbox_height"]
    if any(v is None or not np.isfinite(v) for v in (bx, by, bw, bh)):
        return None
    shallow, deep = 0.15, 0.35

    def crop_left(fraction: float) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
        x0 = int(round(max(0.0, min(width - 8.0, bx + fraction * bw))))
        if x0 <= 0:
            return None
        return image[:, x0:].copy(), affine_translate(-x0, 0.0), f"crop_left@{x0}"

    def crop_right(fraction: float) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
        x1 = int(round(min(width, max(8.0, bx + bw - fraction * bw))))
        if x1 >= width:
            return None
        return image[:, :x1].copy(), affine_translate(0.0, 0.0), f"crop_right@{x1}"

    def pad_back(
        cropped: tuple[np.ndarray, np.ndarray, str], mode: str
    ) -> tuple[np.ndarray, np.ndarray, str]:
        crop_image, A, note = cropped
        pad = width - crop_image.shape[1]
        border = cv2.BORDER_REFLECT_101 if mode == "reflect" else cv2.BORDER_CONSTANT
        padded = cv2.copyMakeBorder(
            crop_image, 0, 0, pad, 0, border, value=(0, 0, 0)
        )
        return padded, affine_translate(pad, 0.0) @ A, f"{note}+{mode}pad{pad}"

    if kind == "C0":
        return image.copy(), np.eye(3), "original"
    if kind == "C1":
        return crop_left(shallow)
    if kind == "C2":
        return crop_right(shallow)
    if kind == "C3":
        return crop_left(deep)
    if kind == "C4":
        return crop_right(deep)
    if kind in ("C5", "C7"):
        base = crop_left(shallow)
        return None if base is None else pad_back(base, "reflect" if kind == "C5" else "constant")
    if kind in ("C6", "C8"):
        base = crop_left(deep)
        return None if base is None else pad_back(base, "reflect" if kind == "C6" else "constant")
    if kind == "C9":
        s = 0.5
        resized = cv2.resize(
            image, (int(width * s), int(height * s)), interpolation=cv2.INTER_AREA
        )
        return resized, affine_scale(s), "downscale0.5"
    if kind == "C10":
        return cv2.GaussianBlur(image, (0, 0), 3.0), np.eye(3), "blur_sigma3"
    if kind == "C11":
        darker = np.clip(image.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
        return darker, np.eye(3), "luma0.45"
    raise ValueError(kind)


COUNTERFACTUAL_KINDS = tuple(f"C{i}" for i in range(12))


def select_counterfactual_sources(
    manifest: dict[str, Any], poses: pd.DataFrame, keypoints: pd.DataFrame
) -> list[dict[str, Any]]:
    """Deterministic stratified 24 from non-truncated N70."""
    primary = {
        f["frame_id"]: f
        for f in manifest["frames"]
        if f["population"] == "primary" and not f["is_truncated"]
    }
    baseline = poses[(poses.variant == "D0") & (poses.frame_id.isin(primary))].copy()
    far = (
        keypoints[keypoints.keypoint.isin(FAR_KP)]
        .groupby("frame_id")["D0_err_px"]
        .median()
        .rename("far_err_px")
    )
    baseline = baseline.merge(far, on="frame_id", how="left")
    baseline["luma_p50"] = baseline.frame_id.map(
        lambda f: primary[f]["luma_p50"]
    )
    baseline = baseline.sort_values("frame_id").reset_index(drop=True)

    used: set[str] = set()
    chosen: list[dict[str, Any]] = []

    def take(subset: pd.DataFrame, label: str, count: int) -> None:
        for _, row in subset.iterrows():
            if len(chosen) >= COUNTERFACTUAL_N:
                return
            if row.frame_id in used:
                continue
            if sum(1 for c in chosen if c["stratum"] == label) >= count:
                return
            used.add(row.frame_id)
            entry = dict(primary[row.frame_id])
            entry["stratum"] = label
            chosen.append(entry)

    clean = baseline[
        baseline.pose_success
        & (baseline.yaw_err_deg.fillna(999) < 5.0)
        & (baseline.matched_median_px.fillna(999) < 10.0)
    ].sort_values(["yaw_err_deg", "frame_id"])
    take(clean, "clean_success", 8)

    confident_wrong = baseline[
        (baseline.n_corner_detected >= 6) & (baseline.far_err_px.fillna(0) > 20.0)
    ].sort_values(["far_err_px", "frame_id"], ascending=[False, True])
    take(confident_wrong, "far_confident_wrong", 8)

    night = baseline[baseline.domain == "night"].sort_values(
        ["luma_p50", "frame_id"]
    )
    take(night, "low_light_night", 8)

    # deterministic top-up if a stratum was short
    for _, row in baseline.sort_values("frame_id").iterrows():
        if len(chosen) >= COUNTERFACTUAL_N:
            break
        if row.frame_id in used:
            continue
        used.add(row.frame_id)
        entry = dict(primary[row.frame_id])
        entry["stratum"] = "topup"
        chosen.append(entry)
    return chosen


def run_counterfactuals(
    manifest: dict[str, Any], audit: FZ.InputAudit
) -> pd.DataFrame:
    poses = pd.read_parquet(OUT_DIR / "poses.parquet")
    keypoints = pd.read_parquet(OUT_DIR / "keypoints.parquet")
    sources = select_counterfactual_sources(manifest, poses, keypoints)
    log(f"[counterfactual] {len(sources)} sources: " + json.dumps(
        pd.Series([s["stratum"] for s in sources]).value_counts().to_dict()
    ))

    device = FZ.choose_device("auto")
    model, _ = FZ.load_model(device)
    rows: list[dict[str, Any]] = []
    tensors: dict[str, np.ndarray] = {}
    worst_geometry = 0.0

    for spec in sources:
        geometry = FrameGeometry(spec, audit)
        if geometry.K is None or geometry.dims is None or geometry.gt_pose is None:
            continue
        image = audit.read_image(spec["image_path"])
        for kind in COUNTERFACTUAL_KINDS:
            built = make_counterfactual(image, spec, kind)
            if built is None:
                rows.append(
                    {
                        "frame_id": spec["frame_id"],
                        "stratum": spec["stratum"],
                        "variant": kind,
                        "status": "not_applicable",
                    }
                )
                continue
            new_image, A, note = built
            new_K = A @ geometry.K
            new_gt = [apply_affine(A, p) for p in geometry.gt_points]
            error = geometry_unit_test(A, geometry.K, geometry.gt_pose, geometry.dims)
            worst_geometry = max(worst_geometry, error)
            if error > GEOMETRY_TOL_PX:
                raise RuntimeError(
                    f"BLOCKED counterfactual geometry: {spec['frame_id']}/{kind} "
                    f"reprojection identity error {error:.4f}px > {GEOMETRY_TOL_PX}"
                )

            height, width = new_image.shape[:2]
            scale_x, scale_y = width / BELIEF, height / BELIEF
            cached = forward_all_stages(model, new_image, device)
            uid = f"{spec['frame_id']}|{kind}"
            tensors[f"{uid}|belief_final"] = cached["belief_stages"][-1]
            final = cached["belief_stages"].astype(np.float32)[-1]
            decoded = decode_all(final, scale_x, scale_y, new_gt)
            points = decoded["D0"]
            stats = decoded["_stats"]

            solver = FZ.CurrentSolveCache(
                new_K, geometry.dims, (height, width, 3), auto_swap_dims=True
            )
            pose, _, _, _ = solver.solve(points)
            oracle, _, _, _ = solver.solve(new_gt)
            reproj, n_obs = FZ.fixed_observation_reprojection(
                pose, new_gt, new_K, geometry.dims
            )
            matched = FZ.order_free_corner_metrics(points, new_gt)
            far = FZ.order_free_corner_metrics(
                [points[k] if k in FAR_KP else None for k in range(8)],
                [new_gt[k] if k in FAR_KP else None for k in range(8)],
            )
            yaw = None if pose is None else FZ.yaw_deg(pose["R"])
            yaw_ref = None if oracle is None else FZ.yaw_deg(oracle["R"])
            rows.append(
                {
                    "frame_id": spec["frame_id"],
                    "stratum": spec["stratum"],
                    "domain": spec["domain"],
                    "variant": kind,
                    "note": note,
                    "status": "ok",
                    "geometry_identity_err_px": error,
                    "image_width": width,
                    "image_height": height,
                    "n_detected": sum(1 for p in points if p is not None),
                    "n_corner_detected": sum(1 for p in points[:8] if p is not None),
                    "median_peak": nanmedian([s.get("peak") for s in stats]),
                    "matched_median_px": matched["median_px"],
                    "far_matched_median_px": far["median_px"],
                    "pose_success": pose is not None,
                    "oracle_pose_success": oracle is not None,
                    "yaw_deg": yaw,
                    "yaw_err_deg": (
                        None
                        if (yaw is None or yaw_ref is None)
                        else abs(FZ.wrap180(yaw - yaw_ref))
                    ),
                    "reproj_fixed_gt_px": reproj,
                    "n_fixed_observations": n_obs,
                }
            )
    np.savez_compressed(CF_CACHE_NPZ, **tensors)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT_DIR / "counterfactuals.parquet", index=False)
    log(
        f"[counterfactual] {len(frame)} rows, worst geometry identity error "
        f"{worst_geometry:.2e}px (tol {GEOMETRY_TOL_PX})"
    )
    return frame


# ============================================================================
# Phase E/H/J — classification
# ============================================================================
def classify_frames(
    frames: pd.DataFrame, poses: pd.DataFrame, keypoints: pd.DataFrame,
    interventions: pd.DataFrame, scale: float = 1.0
) -> pd.DataFrame:
    baseline = poses[poses.variant == "D0"].set_index("frame_id")
    solver_spread = (
        poses.groupby("frame_id")
        .agg(
            solver_success_spread=("pose_success", lambda s: int(s.max()) - int(s.min())),
            solver_yaw_spread=("yaw_err_deg", lambda s: float(np.nanmax(s) - np.nanmin(s)) if s.notna().any() else np.nan),
        )
    )
    far_err = (
        keypoints[keypoints.keypoint.isin(FAR_KP)]
        .groupby("frame_id")["D0_err_px"]
        .median()
        .rename("far_err_px")
    )
    rows = []
    for frame_id, spec in frames.set_index("frame_id").iterrows():
        base = baseline.loc[frame_id]
        detected = int(base.n_corner_detected)
        median_peak = finite(spec.frame_median_peak)
        matched = finite(base.matched_median_px)
        far = finite(far_err.get(frame_id))
        yaw_err = finite(base.yaw_err_deg)
        spread = solver_spread.loc[frame_id]

        t1_det = THRESH["f1_min_detected"]
        t1_far = THRESH["f1_min_far_detected"]
        t1_peak = THRESH["f1_min_frame_median_peak"] * scale
        t2_err = THRESH["f2_matched_error_px"] * scale
        t3_err = THRESH["f3_matched_error_px"] * scale
        t3_yaw = THRESH["f3_yaw_deg"] * scale

        far_detected = int(spec.n_far_detected)
        if (
            detected < t1_det
            or far_detected < t1_far
            or (median_peak is not None and median_peak < t1_peak)
        ):
            failure = "F1_NO_RESPONSE"
        elif (
            detected >= t1_det
            and (median_peak is None or median_peak >= t1_peak)
            and (
                (matched is not None and matched > t2_err)
                or (far is not None and far > t2_err)
            )
        ):
            failure = "F2_CONFIDENT_WRONG"
        elif (
            detected >= t1_det
            and matched is not None
            and matched <= t3_err
            and (
                not bool(base.pose_success)
                or (yaw_err is not None and yaw_err > t3_yaw)
            )
        ):
            failure = "F3_GEOMETRY_AMPLIFIED"
        elif (
            matched is not None
            and matched <= t2_err
            and (
                int(spread.solver_success_spread) > 0
                or (
                    finite(spread.solver_yaw_spread) is not None
                    and float(spread.solver_yaw_spread) > t3_yaw
                )
            )
        ):
            failure = "F4_SOLVER_SPECIFIC"
        else:
            failure = "F5_MIXED"

        rows.append(
            {
                "frame_id": frame_id,
                "domain": spec.domain,
                "session_id": spec.session_id,
                "is_truncated": bool(spec.is_truncated),
                "failure_class": failure,
                "n_corner_detected": detected,
                "n_far_detected": far_detected,
                "frame_median_peak": median_peak,
                "matched_median_px": matched,
                "far_median_px": far,
                "yaw_err_deg": yaw_err,
                "pose_success": bool(base.pose_success),
                "reproj_fixed_gt_px": finite(base.reproj_fixed_gt_px),
                "solver_success_spread": int(spread.solver_success_spread),
                "solver_yaw_spread": finite(spread.solver_yaw_spread),
            }
        )
    return pd.DataFrame(rows)


def stage_progression(keypoints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in keypoints.iterrows():
        early = [finite(row.get(f"stage{s}_err_px")) for s in (1, 2, 3)]
        late = [finite(row.get(f"stage{s}_err_px")) for s in (4, 5, 6)]
        early = [v for v in early if v is not None]
        late = [v for v in late if v is not None]
        first = finite(row.get("stage1_err_px"))
        final = finite(row.get("stage6_err_px"))
        peak1 = finite(row.get("stage1_peak"))
        peak6 = finite(row.get("stage6_peak"))
        label = "UNRESOLVED"
        if first is not None and final is not None:
            moved = abs(final - first)
            if first > 20.0 and final > 20.0:
                label = "EARLY_WRONG"
            elif first <= 10.0 and final > 20.0:
                label = "LATE_DRIFT"
            elif first > 20.0 and final <= 10.0:
                label = "RECOVERED_LATE"
            elif moved <= 2.0 and peak1 is not None and peak6 is not None and peak6 > peak1 * 1.2:
                label = "SHARPEN_ONLY"
            elif final > first + 5.0:
                label = "LATE_DRIFT"
            elif final < first - 5.0:
                label = "RECOVERED_LATE"
            else:
                label = "STABLE"
        rows.append(
            {
                "frame_id": row.frame_id,
                "keypoint": int(row.keypoint),
                "group_near_far": row.group_near_far,
                "domain": row.domain,
                "is_truncated": bool(row.is_truncated),
                "stage_label": label,
                "stage1_err_px": first,
                "stage3_err_px": finite(row.get("stage3_err_px")),
                "stage6_err_px": final,
                "early_median_px": nanmedian(early),
                "late_median_px": nanmedian(late),
                "stage1_peak": peak1,
                "stage6_peak": peak6,
                "peak_gain": (
                    None if (peak1 is None or peak6 is None or peak1 <= 0) else peak6 / peak1
                ),
            }
        )
    return pd.DataFrame(rows)


def stage_trajectory(keypoints: pd.DataFrame) -> pd.DataFrame:
    """Median localization error and peak per belief stage, per keypoint group.

    This separates "the refinement stages move the point" from "the refinement
    stages only sharpen the peak".
    """
    rows = []
    for group in ("near", "far", "centroid"):
        subset = keypoints[
            (keypoints.group_near_far == group) & keypoints.detected
        ]
        for stage in range(1, 7):
            rows.append(
                {
                    "group": group,
                    "stage": stage,
                    "n": len(subset),
                    "median_err_px": nanmedian(subset[f"stage{stage}_err_px"].values),
                    "median_peak": nanmedian(subset[f"stage{stage}_peak"].values),
                    "median_entropy": nanmedian(
                        subset[f"stage{stage}_entropy"].values
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    baseline = frame[frame.stage == 1].set_index("group")
    frame["err_delta_vs_stage1_px"] = frame.apply(
        lambda r: (
            None
            if r.median_err_px is None
            else r.median_err_px - float(baseline.loc[r.group, "median_err_px"])
        ),
        axis=1,
    )
    frame["peak_ratio_vs_stage1"] = frame.apply(
        lambda r: (
            None
            if r.median_peak is None
            else r.median_peak / float(baseline.loc[r.group, "median_peak"])
        ),
        axis=1,
    )
    return frame


def missing_response_audit(
    keypoints: pd.DataFrame, classes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split undetected corners into 'GT was in frame' vs 'GT was off image'.

    Without this split, F1_NO_RESPONSE would conflate a genuine response
    failure with the expected absence of a keypoint that is legitimately
    outside the image.  The two imply different architecture fixes.
    """
    merged = keypoints.merge(
        classes[["frame_id", "failure_class"]], on="frame_id", how="left"
    )
    corners = merged[merged.keypoint < 8]
    rows = []
    for failure, group in corners.groupby("failure_class"):
        undetected = group[~group.detected]
        rows.append(
            {
                "failure_class": failure,
                "corner_observations": len(group),
                "undetected": len(undetected),
                "undetected_gt_in_frame": int((undetected.gt_in_frame == True).sum()),
                "undetected_gt_off_image": int((undetected.gt_in_frame == False).sum()),
                "undetected_gt_sentinel": int((~undetected.gt_valid).sum()),
            }
        )
    summary = pd.DataFrame(rows)

    per_frame = (
        corners.assign(
            undetected=~corners.detected,
            undetected_in_frame=(~corners.detected) & (corners.gt_in_frame == True),
            undetected_off_image=(~corners.detected) & (corners.gt_in_frame == False),
        )
        .groupby(["frame_id", "failure_class"])[
            ["undetected", "undetected_in_frame", "undetected_off_image"]
        ]
        .sum()
        .reset_index()
    )
    per_frame["pure_response_failure"] = (
        (per_frame.undetected > 0) & (per_frame.undetected_off_image == 0)
    )
    return summary, per_frame


def counterfactual_pairs(counterfactuals: pd.DataFrame) -> pd.DataFrame:
    """Same-source paired transitions: original -> crop-only -> padded."""
    ok = counterfactuals[counterfactuals.status == "ok"]
    if not len(ok):
        return pd.DataFrame()
    indexed = {v: g.set_index("frame_id") for v, g in ok.groupby("variant")}
    rows = []
    for label, crop, reflect, constant in (
        ("shallow_left", "C1", "C5", "C7"),
        ("deep_left", "C3", "C6", "C8"),
    ):
        base = indexed["C0"]
        common = base.index
        for variant in (crop, reflect, constant):
            common = common.intersection(indexed[variant].index)
        for variant, kind in (
            (crop, "crop_only"), (reflect, "reflect_pad"), (constant, "constant_pad")
        ):
            candidate = indexed[variant].loc[common]
            reference = base.loc[common]
            rows.append(
                {
                    "depth": label,
                    "variant": variant,
                    "kind": kind,
                    "n_paired": len(common),
                    "corners_detected_median": nanmedian(
                        candidate.n_corner_detected.values
                    ),
                    "delta_corners_vs_original": nanmedian(
                        candidate.n_corner_detected.values
                        - reference.n_corner_detected.values
                    ),
                    "pose_success_rate": float(candidate.pose_success.mean()),
                    "delta_pose_success_pp": 100.0
                    * (
                        float(candidate.pose_success.mean())
                        - float(reference.pose_success.mean())
                    ),
                    "delta_peak_median": nanmedian(
                        candidate.median_peak.values - reference.median_peak.values
                    ),
                    "delta_yaw_median": nanmedian(
                        candidate.yaw_err_deg.values - reference.yaw_err_deg.values
                    ),
                    "delta_far2d_median": nanmedian(
                        candidate.far_matched_median_px.values
                        - reference.far_matched_median_px.values
                    ),
                }
            )
    return pd.DataFrame(rows)


def first_break(
    classes: pd.DataFrame, poses: pd.DataFrame, stages: pd.DataFrame
) -> pd.DataFrame:
    baseline = poses[poses.variant == "D0"].set_index("frame_id")
    rows = []
    for _, row in classes.iterrows():
        frame_id = row.frame_id
        base = baseline.loc[frame_id]
        alternatives = poses[(poses.frame_id == frame_id) & (poses.variant != "D0")]
        decoder_recovers = False
        if len(alternatives):
            gained = (~bool(base.pose_success)) & alternatives.pose_success.any()
            base_yaw = finite(base.yaw_err_deg)
            best_yaw = finite(np.nanmin(alternatives.yaw_err_deg.values)) if alternatives.yaw_err_deg.notna().any() else None
            improved = (
                base_yaw is not None
                and best_yaw is not None
                and base_yaw > 0
                and (base_yaw - best_yaw) / base_yaw >= 0.5
            )
            decoder_recovers = bool(gained or improved)

        failure = row.failure_class
        if failure == "F1_NO_RESPONSE":
            stage = "IMAGE_OR_RESPONSE"
        elif decoder_recovers:
            stage = "DECODER"
        elif failure == "F2_CONFIDENT_WRONG":
            stage = "REPRESENTATION_LOCALIZATION"
        elif failure in ("F3_GEOMETRY_AMPLIFIED", "F4_SOLVER_SPECIFIC"):
            stage = "PNP_GEOMETRY"
        elif (
            row.pose_success
            and finite(row.yaw_err_deg) is not None
            and float(row.yaw_err_deg) <= THRESH["f3_yaw_deg"]
        ):
            stage = "POSE_REFINEMENT"
        else:
            stage = "MIXED_OR_UNRESOLVED"

        subset = stages[stages.frame_id == frame_id]
        rows.append(
            {
                "frame_id": frame_id,
                "domain": row.domain,
                "is_truncated": row.is_truncated,
                "failure_class": failure,
                "first_break_stage": stage,
                "decoder_recovers": decoder_recovers,
                "n_corner_detected": row.n_corner_detected,
                "matched_median_px": row.matched_median_px,
                "yaw_err_deg": row.yaw_err_deg,
                "reproj_fixed_gt_px": row.reproj_fixed_gt_px,
                "dominant_stage_label": (
                    subset.stage_label.mode().iloc[0] if len(subset) else None
                ),
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# Phase N — figures
# ============================================================================
def make_figures(
    classes: pd.DataFrame, breaks: pd.DataFrame, oracle: pd.DataFrame,
    decoder: pd.DataFrame, stages: pd.DataFrame, counterfactuals: pd.DataFrame,
    interventions: pd.DataFrame, decision: pd.DataFrame
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []

    def save(fig: Any, name: str) -> None:
        path = OUT_DIR / name
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(name)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["F1_NO_RESPONSE", "F2_CONFIDENT_WRONG", "F3_GEOMETRY_AMPLIFIED",
             "F4_SOLVER_SPECIFIC", "F5_MIXED"]
    pivot = (
        classes.groupby(["failure_class", "domain"]).size().unstack(fill_value=0)
        .reindex(order, fill_value=0)
    )
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Failure class distribution (strict N87, ep57 frozen)")
    ax.set_ylabel("frames")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "failure_class_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    breaks.groupby(["first_break_stage", "domain"]).size().unstack(fill_value=0).plot(
        kind="bar", stacked=True, ax=ax
    )
    ax.set_title("First break stage")
    ax.set_ylabel("frames")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "first_break_stage_distribution.png")

    matrix = oracle.pivot_table(
        index="variant", columns="failure_class", values="recovered_frames",
        aggfunc="sum", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(matrix))))
    image = ax.imshow(matrix.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right", fontsize=7)
    ax.set_title("Oracle recovery matrix (frames recovered)")
    fig.colorbar(image, ax=ax)
    save(fig, "oracle_recovery_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    stages.groupby(["stage_label", "group_near_far"]).size().unstack(fill_value=0).plot(
        kind="bar", stacked=True, ax=ax
    )
    ax.set_title("Stage-wise progression label by keypoint group")
    ax.set_ylabel("keypoint observations")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "stage_progression_by_failure.png")

    far = oracle[oracle.variant.isin(
        ["O2_near", "O3_far", "O4_top", "O5_bottom", "O6_depth_left",
         "O7_depth_right", "O8_kp5_kp6", "O9_centroid", "O10_all_corners", "O11_all"]
    )]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    summary = far.groupby("variant")[["delta_yaw_median", "delta_reproj_median"]].mean()
    summary.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Group oracle recovery (mean delta vs baseline)")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "far_depth_oracle_recovery.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    decoder.set_index("variant")[
        ["pose_success_delta_pp", "delta_yaw_median", "delta_matched_2d_median"]
    ].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Decoder pose recovery vs D2 canonical")
    ax.tick_params(axis="x", rotation=0)
    save(fig, "decoder_pose_recovery.png")

    ok = counterfactuals[counterfactuals.status == "ok"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, metric, title in zip(
        axes,
        ["n_corner_detected", "matched_median_px", "yaw_err_deg"],
        ["corners detected", "matched 2D median px", "yaw err deg"],
    ):
        grouped = ok.groupby("variant")[metric].median()
        grouped = grouped.reindex(COUNTERFACTUAL_KINDS).dropna()
        axis.bar(grouped.index, grouped.values, color="steelblue")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=60, labelsize=7)
    fig.suptitle("Same-source counterfactual: original / crop-only / padded")
    save(fig, "counterfactual_original_crop_reflect.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    nodes = ["response", "2D localization", "PnP geometry", "pose"]
    values = [
        float((classes.failure_class == "F1_NO_RESPONSE").mean()),
        float((classes.failure_class == "F2_CONFIDENT_WRONG").mean()),
        float(classes.failure_class.isin(
            ["F3_GEOMETRY_AMPLIFIED", "F4_SOLVER_SPECIFIC"]).mean()),
        float((~classes.pose_success).mean()),
    ]
    ax.plot(nodes, values, marker="o", linewidth=2)
    for x, y in zip(nodes, values):
        ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 8))
    ax.set_ylabel("fraction of strict N87")
    ax.set_title("Error propagation: where information breaks")
    save(fig, "error_propagation_graph.png")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    table = decision.set_index("candidate")[["supporting_frames", "oracle_recovery_frames"]]
    table.plot(kind="barh", ax=ax)
    ax.set_title("Architecture evidence matrix")
    save(fig, "architecture_evidence_matrix.png")

    return written


# ============================================================================
# example overlays (Phase E / H / J qualitative evidence)
# ============================================================================
GREEN, RED, BLUE, YELLOW = (60, 220, 60), (60, 60, 235), (235, 160, 60), (40, 215, 235)


def draw_points(
    canvas: np.ndarray, points: Iterable[Optional[list[float]]], color: tuple,
    label: bool = True, radius: int = 5
) -> None:
    for index, point in enumerate(points):
        if point is None or not np.isfinite(point).all():
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(canvas, (x, y), radius, color, 2, cv2.LINE_AA)
        if label:
            cv2.putText(
                canvas, str(index), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA
            )


def banner(canvas: np.ndarray, lines: list[str]) -> np.ndarray:
    pad = 22 * len(lines) + 10
    out = cv2.copyMakeBorder(canvas, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    for index, text in enumerate(lines):
        cv2.putText(
            out, text, (10, 20 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (235, 235, 235), 1, cv2.LINE_AA
        )
    return out


def write_examples(
    manifest: dict[str, Any], classes: pd.DataFrame, breaks: pd.DataFrame,
    stages: pd.DataFrame, keypoints: pd.DataFrame, audit: FZ.InputAudit,
    per_group: int = 3
) -> dict[str, int]:
    """Deterministic qualitative examples: GT vs predicted, and stage1 vs stage6."""
    specs = {f["frame_id"]: f for f in manifest["frames"]}
    tensors = load_cached_tensors()
    kp_by_frame = {fid: g for fid, g in keypoints.groupby("frame_id")}
    written = {"failure_class_examples": 0, "late_drift_examples": 0,
               "error_propagation_examples": 0}

    def overlay(frame_id: str, extra: list[str], stage_mode: bool) -> Optional[np.ndarray]:
        spec = specs.get(frame_id)
        if spec is None:
            return None
        image = audit.read_image(spec["image_path"])
        if image is None:
            return None
        canvas = image.copy()
        rows = kp_by_frame[frame_id].sort_values("keypoint")
        gt = [
            None if not bool(r.gt_valid) else [r.gt_x, r.gt_y]
            for r in rows.itertuples()
        ]
        draw_points(canvas, gt, GREEN)
        if stage_mode:
            draw_points(
                canvas,
                [[r.stage1_soft_x, r.stage1_soft_y] for r in rows.itertuples()],
                BLUE, label=False, radius=8,
            )
            draw_points(
                canvas,
                [[r.stage6_soft_x, r.stage6_soft_y] for r in rows.itertuples()],
                RED,
            )
            legend = "green=GT  blue=stage1  red=stage6"
        else:
            draw_points(
                canvas,
                [
                    None if bool(r.D0_missing) else [r.D0_x, r.D0_y]
                    for r in rows.itertuples()
                ],
                RED,
            )
            missing = [int(r.keypoint) for r in rows.itertuples() if bool(r.D0_missing)]
            legend = f"green=GT  red=pred(D0)  missing={missing}"
        return banner(canvas, extra + [legend])

    root = OUT_DIR / "failure_class_examples"
    root.mkdir(exist_ok=True)
    for failure, group in classes.groupby("failure_class"):
        chosen = group.sort_values(
            ["matched_median_px", "frame_id"], ascending=[False, True]
        ).head(per_group)
        for _, row in chosen.iterrows():
            canvas = overlay(
                row.frame_id,
                [
                    f"{failure} | {row.domain} | truncated={bool(row.is_truncated)}",
                    f"corners={int(row.n_corner_detected)} peak={finite(row.frame_median_peak) or float('nan'):.2f} "
                    f"matched2D={finite(row.matched_median_px) or float('nan'):.1f}px "
                    f"yawerr={finite(row.yaw_err_deg) or float('nan'):.1f}deg "
                    f"pose_ok={bool(row.pose_success)}",
                ],
                stage_mode=False,
            )
            if canvas is None:
                continue
            name = f"{failure}__{row.frame_id.replace(':', '_')}.jpg"
            cv2.imwrite(str(root / name), canvas)
            written["failure_class_examples"] += 1

    root = OUT_DIR / "late_drift_examples"
    root.mkdir(exist_ok=True)
    drift = (
        stages[stages.stage_label == "LATE_DRIFT"]
        .groupby("frame_id").size().rename("n_drift").reset_index()
        .sort_values(["n_drift", "frame_id"], ascending=[False, True])
        .head(per_group)
    )
    for _, row in drift.iterrows():
        canvas = overlay(
            row.frame_id,
            [f"LATE_DRIFT keypoints={int(row.n_drift)} | {row.frame_id}"],
            stage_mode=True,
        )
        if canvas is None:
            continue
        cv2.imwrite(
            str(root / f"late_drift__{row.frame_id.replace(':', '_')}.jpg"), canvas
        )
        written["late_drift_examples"] += 1

    root = OUT_DIR / "error_propagation_examples"
    root.mkdir(exist_ok=True)
    for stage_name, group in breaks.groupby("first_break_stage"):
        for _, row in group.sort_values("frame_id").head(2).iterrows():
            canvas = overlay(
                row.frame_id,
                [
                    f"first_break={stage_name} | class={row.failure_class}",
                    f"corners={int(row.n_corner_detected)} "
                    f"matched2D={finite(row.matched_median_px) or float('nan'):.1f}px "
                    f"yawerr={finite(row.yaw_err_deg) or float('nan'):.1f}deg "
                    f"decoder_recovers={bool(row.decoder_recovers)}",
                ],
                stage_mode=False,
            )
            if canvas is None:
                continue
            cv2.imwrite(
                str(root / f"{stage_name}__{row.frame_id.replace(':', '_')}.jpg"),
                canvas,
            )
            written["error_propagation_examples"] += 1
    del tensors
    return written


# ============================================================================
# Phase K — architecture decision matrix
# ============================================================================
def oracle_recovery_matrix(
    interventions: pd.DataFrame, classes: pd.DataFrame
) -> pd.DataFrame:
    merged = interventions.merge(
        classes[["frame_id", "failure_class"]], on="frame_id", how="left"
    )
    baseline = merged[merged.variant == "O0"].set_index("frame_id")
    rows = []
    for (variant, failure), group in merged.groupby(["variant", "failure_class"]):
        base = baseline.reindex(group.frame_id)
        recovered = int(
            ((~base.pose_success.values) & group.pose_success.values).sum()
        )
        lost = int((base.pose_success.values & (~group.pose_success.values)).sum())
        delta_yaw = nanmedian(group.yaw_err_deg.values - base.yaw_err_deg.values)
        delta_reproj = nanmedian(
            group.reproj_fixed_gt_px.values - base.reproj_fixed_gt_px.values
        )
        delta_add = nanmedian(group.add_m.values - base.add_m.values)
        rows.append(
            {
                "variant": variant,
                "failure_class": failure,
                "n_frames": len(group),
                "recovered_frames": recovered,
                "lost_frames": lost,
                "success_rate_delta_pp": 100.0
                * (
                    float(group.pose_success.mean())
                    - float(base.pose_success.mean())
                ),
                "delta_yaw_median": delta_yaw,
                "delta_reproj_median": delta_reproj,
                "delta_add_median": delta_add,
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "failure_class"])


def decoder_recovery_table(poses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reference = poses[poses.variant == "D2"].set_index("frame_id")
    d0 = poses[poses.variant == "D0"].set_index("frame_id")
    for variant, group in poses.groupby("variant"):
        indexed = group.set_index("frame_id")
        ref = reference.reindex(indexed.index)
        gate = d0.reindex(indexed.index)
        rows.append(
            {
                "variant": variant,
                "n_frames": len(indexed),
                "pose_success": int(indexed.pose_success.sum()),
                "pose_success_delta_pp": 100.0
                * (float(indexed.pose_success.mean()) - float(ref.pose_success.mean())),
                "pose_success_delta_pp_vs_D0": 100.0
                * (float(indexed.pose_success.mean()) - float(gate.pose_success.mean())),
                "yaw_median": nanmedian(indexed.yaw_err_deg.values),
                "delta_yaw_median": nanmedian(
                    indexed.yaw_err_deg.values - ref.yaw_err_deg.values
                ),
                "matched_2d_median": nanmedian(indexed.matched_median_px.values),
                "delta_matched_2d_median": nanmedian(
                    indexed.matched_median_px.values - ref.matched_median_px.values
                ),
                "reproj_median": nanmedian(indexed.reproj_fixed_gt_px.values),
                "delta_reproj_median": nanmedian(
                    indexed.reproj_fixed_gt_px.values - ref.reproj_fixed_gt_px.values
                ),
                "n_detected_median": nanmedian(indexed.n_detected.values),
                "delta_detected_median": nanmedian(
                    indexed.n_detected.values - ref.n_detected.values
                ),
            }
        )
    table = pd.DataFrame(rows)
    yaw_ref = float(table.loc[table.variant == "D2", "yaw_median"].iloc[0])
    matched_ref = float(table.loc[table.variant == "D2", "matched_2d_median"].iloc[0])
    table["decoder_is_primary_lever"] = (
        (table.pose_success_delta_pp >= 5.0)
        | ((yaw_ref - table.yaw_median) / max(yaw_ref, 1e-9) >= 0.20)
        | ((matched_ref - table.matched_2d_median) >= 5.0)
    )
    return table


def single_keypoint_consistency(interventions: pd.DataFrame) -> pd.DataFrame:
    """Per-frame sign consistency of each single-keypoint GT replacement.

    A constant pixel correction is only plausible if a single keypoint's GT
    replacement improves the pose in most frames.
    """
    baseline = interventions[interventions.variant == "O0"].set_index("frame_id")
    rows = []
    for variant, group in interventions.groupby("variant"):
        if not str(variant).startswith("O1_kp"):
            continue
        indexed = group.set_index("frame_id")
        base = baseline.reindex(indexed.index)
        delta = indexed.yaw_err_deg.values - base.yaw_err_deg.values
        finite_delta = delta[np.isfinite(delta.astype(float))]
        rows.append(
            {
                "variant": variant,
                "n_paired": int(finite_delta.size),
                "fraction_improved": (
                    float((finite_delta < 0).mean()) if finite_delta.size else None
                ),
                "median_delta_yaw": nanmedian(finite_delta),
                "recovered_frames": int(
                    ((~base.pose_success.values) & indexed.pose_success.values).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("variant")


def architecture_decision(
    classes: pd.DataFrame, oracle: pd.DataFrame, decoder: pd.DataFrame,
    counterfactuals: pd.DataFrame, stages: pd.DataFrame,
    trajectory: pd.DataFrame, interventions: pd.DataFrame
) -> pd.DataFrame:
    counts = classes.failure_class.value_counts().to_dict()
    total = len(classes)
    baseline = interventions[interventions.variant == "O0"].set_index("frame_id")
    failed = set(baseline.index[~baseline.pose_success.astype(bool)])

    def oracle_frames(variants: Iterable[str]) -> int:
        """DISTINCT baseline-failed frames rescued by at least one variant.

        Summing per-variant counts would double-count the same frame.
        """
        subset = interventions[
            interventions.variant.isin(list(variants))
            & interventions.pose_success.astype(bool)
            & interventions.frame_id.isin(failed)
        ]
        return int(subset.frame_id.nunique())

    def oracle_yaw(variants: Iterable[str]) -> Optional[float]:
        subset = oracle[oracle.variant.isin(list(variants))]
        return nanmedian(subset.delta_yaw_median.values)

    def target_yaw(variants: Iterable[str], failure: str) -> Optional[float]:
        """Median yaw improvement of the candidate's variants INSIDE its own
        target class.  Distinct-frame recovery saturates at 17 (every baseline
        failure sits in F1), so this is the discriminating number for F2."""
        subset = oracle[
            oracle.variant.isin(list(variants)) & (oracle.failure_class == failure)
        ]
        return nanmedian(subset.delta_yaw_median.values)

    def target_reproj(variants: Iterable[str], failure: str) -> Optional[float]:
        subset = oracle[
            oracle.variant.isin(list(variants)) & (oracle.failure_class == failure)
        ]
        return nanmedian(subset.delta_reproj_median.values)

    A_VARIANTS = ["O10_all_corners", "O11_all"]
    B_VARIANTS = ["O3_far", "O8_kp5_kp6", "O6_depth_left", "O7_depth_right"]
    C_VARIANTS = [v for v in oracle.variant.unique() if str(v).startswith("A_drop")]
    LOO_VARIANTS = [v for v in oracle.variant.unique() if str(v).startswith("LOO_")]

    ok = counterfactuals[counterfactuals.status == "ok"]
    crop_collapse = None
    if len(ok):
        base = ok[ok.variant == "C0"].set_index("frame_id")
        deep = ok[ok.variant == "C3"].set_index("frame_id")
        common = base.index.intersection(deep.index)
        if len(common):
            crop_collapse = float(
                (base.loc[common].n_corner_detected.values
                 - deep.loc[common].n_corner_detected.values).mean()
            )

    sharpen = float((stages.stage_label == "SHARPEN_ONLY").mean())
    early_wrong = float((stages.stage_label == "EARLY_WRONG").mean())
    # Candidate B's stage evidence is NOT "late drift" (refuted): it is that the
    # refinement stages raise far-face confidence without moving the coordinate.
    far = trajectory[trajectory.group == "far"].set_index("stage")
    far_plateau = (
        None
        if far.empty
        else float(far.loc[6, "median_err_px"]) / float(far.loc[1, "median_err_px"])
    )
    far_peak_gain = None if far.empty else float(far.loc[6, "peak_ratio_vs_stage1"])

    rows = [
        {
            "candidate": "A_visibility_partial_keypoint",
            "target_failure": "F1_NO_RESPONSE",
            "supporting_frames": counts.get("F1_NO_RESPONSE", 0),
            "supporting_fraction": counts.get("F1_NO_RESPONSE", 0) / max(total, 1),
            "oracle_recovery_frames": oracle_frames(A_VARIANTS),
            "oracle_delta_yaw": oracle_yaw(["O11_all"]),
            "target_class_delta_yaw": target_yaw(A_VARIANTS, "F1_NO_RESPONSE"),
            "target_class_delta_reproj": target_reproj(A_VARIANTS, "F1_NO_RESPONSE"),
            "counterfactual_signal": crop_collapse,
            "stage_signal": early_wrong,
            "stage_evidence": "EARLY_WRONG fraction (response absent from stage 1)",
        },
        {
            "candidate": "B_far_depth_structured_refinement",
            "target_failure": "F2_CONFIDENT_WRONG",
            "supporting_frames": counts.get("F2_CONFIDENT_WRONG", 0),
            "supporting_fraction": counts.get("F2_CONFIDENT_WRONG", 0) / max(total, 1),
            "oracle_recovery_frames": oracle_frames(B_VARIANTS),
            "oracle_delta_yaw": oracle_yaw(["O3_far", "O8_kp5_kp6"]),
            "target_class_delta_yaw": target_yaw(B_VARIANTS, "F2_CONFIDENT_WRONG"),
            "target_class_delta_reproj": target_reproj(B_VARIANTS, "F2_CONFIDENT_WRONG"),
            "counterfactual_signal": None,
            "stage_signal": far_plateau,
            "stage_evidence": (
                f"far err stage6/stage1={far_plateau:.2f} while peak x{far_peak_gain:.1f}"
                if far_plateau is not None
                else "unavailable"
            ),
        },
        {
            "candidate": "C_calibrated_uncertainty_robust_pnp",
            "target_failure": "F3_GEOMETRY_AMPLIFIED",
            "supporting_frames": counts.get("F3_GEOMETRY_AMPLIFIED", 0),
            "supporting_fraction": counts.get("F3_GEOMETRY_AMPLIFIED", 0) / max(total, 1),
            "oracle_recovery_frames": oracle_frames(C_VARIANTS),
            "oracle_delta_yaw": oracle_yaw(LOO_VARIANTS),
            "target_class_delta_yaw": target_yaw(
                C_VARIANTS + LOO_VARIANTS, "F3_GEOMETRY_AMPLIFIED"),
            "target_class_delta_reproj": target_reproj(
                C_VARIANTS + LOO_VARIANTS, "F3_GEOMETRY_AMPLIFIED"),
            "counterfactual_signal": None,
            "stage_signal": sharpen,
            "stage_evidence": "SHARPEN_ONLY fraction (confidence without correction)",
        },
        {
            "candidate": "D_direct_pose_gated_residual",
            "target_failure": "F4_SOLVER_SPECIFIC",
            "supporting_frames": counts.get("F4_SOLVER_SPECIFIC", 0),
            "supporting_fraction": counts.get("F4_SOLVER_SPECIFIC", 0) / max(total, 1),
            "oracle_recovery_frames": 0,
            "oracle_delta_yaw": None,
            "target_class_delta_yaw": None,
            "target_class_delta_reproj": None,
            "counterfactual_signal": None,
            "stage_signal": None,
            "stage_evidence": "no stage-level signal isolated",
        },
    ]
    table = pd.DataFrame(rows)
    table["priority"] = (
        table.supporting_frames.rank(ascending=False, method="first").astype(int)
    )
    decoder_primary = bool(decoder.loc[decoder.variant != "D2", "decoder_is_primary_lever"].any())
    table["decoder_precedes"] = decoder_primary
    table["oracle_recovery_metric"] = "distinct baseline-failed frames rescued"
    return table.sort_values("priority")


# ============================================================================
# Phase O — report
# ============================================================================
def write_report(
    manifest: dict[str, Any], gate: dict[str, Any], classes: pd.DataFrame,
    breaks: pd.DataFrame, oracle: pd.DataFrame, decoder: pd.DataFrame,
    stages: pd.DataFrame, counterfactuals: pd.DataFrame, decision: pd.DataFrame,
    sensitivity: pd.DataFrame, figures: list[str], cache_info: dict[str, Any],
    trajectory: pd.DataFrame, pairs: pd.DataFrame, consistency: pd.DataFrame,
    missing_summary: pd.DataFrame, missing_frames: pd.DataFrame
) -> None:
    def table(frame: pd.DataFrame, columns: list[str], limit: int = 40) -> str:
        subset = frame[columns].head(limit)
        widths = [
            max(len(str(c)), *(len(f"{v}") for v in subset[c].map(fmt))) for c in columns
        ]
        head = "  ".join(str(c).ljust(w) for c, w in zip(columns, widths))
        line = "─" * len(head)
        body = "\n".join(
            "  ".join(fmt(row[c]).ljust(w) for c, w in zip(columns, widths))
            for _, row in subset.iterrows()
        )
        return f"```\n{head}\n{line}\n{body}\n```"

    def fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    counts = classes.failure_class.value_counts()
    break_counts = breaks.first_break_stage.value_counts()
    total = len(classes)
    ok = counterfactuals[counterfactuals.status == "ok"]

    def cf_median(variant: str, metric: str) -> Optional[float]:
        subset = ok[ok.variant == variant]
        return nanmedian(subset[metric].values) if len(subset) else None

    lines: list[str] = []
    lines.append("# PAPER_S2 ep57 — Mechanism Diagnostic Report\n")
    lines.append(
        "진단 전용.  새 모델·새 학습·checkpoint selection 없음.  "
        "산출물은 (1) 실패 프레임의 최초 붕괴 단계, (2) 그에 대응하는 architecture/loss 우선순위.\n"
    )
    lines.append(
        f"- checkpoint `{FZ.WEIGHTS.name}` SHA `{FZ.WEIGHTS_SHA256[:16]}…`\n"
        f"- git HEAD `{manifest.get('git_head')}`\n"
        f"- primary population strict filter-val **N={total}** "
        f"(outside {int((classes.domain == 'outside').sum())} / night {int((classes.domain == 'night').sum())}, "
        f"truncated {int(classes.is_truncated.sum())} / non-truncated {int((~classes.is_truncated).sum())})\n"
        f"- cache key `{cache_info.get('cache_key', '')[:16]}…`, model forwards this build: "
        f"{cache_info.get('model_forwards', 'reused (0)')}\n"
        f"- final-test open count **0** (sealed sessions fail-closed)\n"
    )

    lines.append("\n## 1. 관찰\n")
    lines.append(
        f"[확인] baseline 재현 게이트 통과: GT-2D PnP {gate['gt2d_pose_success']}/{gate['strict_n']}, "
        f"predicted local-softargmax PnP {gate['pred_pose_success']}/{gate['strict_n']}, "
        f"yaw median {gate['yaw_median_deg']:.3f}°, fixed-GT reproj median "
        f"{gate['fixed_gt_reproj_median_px']:.3f} px.\n"
    )
    lines.append(
        f"[확인] 2D correspondence 를 GT 로 전부 바꾸면 {gate['gt2d_pose_success']}/{gate['strict_n']} 가 풀린다 — "
        "solver·K·dimensions 는 병목이 아니다.\n"
    )

    lines.append("\n## 2. Failure class 분포\n")
    lines.append(table(
        counts.rename_axis("failure_class").reset_index(name="frames").assign(
            fraction=lambda d: d.frames / total
        ),
        ["failure_class", "frames", "fraction"],
    ))
    lines.append("\n클래스별 프로필 (median):\n")
    profile = classes.groupby("failure_class").agg(
        n=("frame_id", "size"),
        corners_detected=("n_corner_detected", "median"),
        far_detected=("n_far_detected", "median"),
        frame_peak=("frame_median_peak", "median"),
        matched_2d_px=("matched_median_px", "median"),
        far_2d_px=("far_median_px", "median"),
        yaw_err_deg=("yaw_err_deg", "median"),
        pose_success_rate=("pose_success", "mean"),
        truncated_rate=("is_truncated", "mean"),
    ).reset_index()
    lines.append(table(profile, list(profile.columns)))

    non_failure = classes[
        (classes.failure_class == "F5_MIXED")
        & classes.pose_success
        & (classes.yaw_err_deg.fillna(999) <= THRESH["f3_yaw_deg"])
    ]
    lines.append(
        f"\n[확인] F5_MIXED {int((classes.failure_class == 'F5_MIXED').sum())} 프레임 중 "
        f"{len(non_failure)} 개는 pose 성공 + yaw ≤ {THRESH['f3_yaw_deg']}° 로 **사실상 실패가 아니다** "
        "(catch-all 이 정상 프레임을 흡수한다).  F5 를 '혼합 실패'로 읽으면 안 된다.\n"
    )
    lines.append(
        "\n### 미검출 corner 는 화면 안이었나 밖이었나 (F1 해석의 전제)\n"
    )
    lines.append(table(missing_summary, list(missing_summary.columns)))
    f1_frames = missing_frames[missing_frames.failure_class == "F1_NO_RESPONSE"]
    f1_row = missing_summary[missing_summary.failure_class == "F1_NO_RESPONSE"]
    if len(f1_row):
        in_frame = int(f1_row.undetected_gt_in_frame.iloc[0])
        off_image = int(f1_row.undetected_gt_off_image.iloc[0])
        sentinel = int(f1_row.undetected_gt_sentinel.iloc[0])
        lines.append(
            f"\n[확인] F1 프레임에서 검출되지 않은 corner {in_frame + off_image + sentinel}개 중 "
            f"**{in_frame}개는 GT 가 화면 안**에 있었다 (화면 밖 {off_image}, sentinel {sentinel}).  "
            f"프레임 단위로도 {int(f1_frames.pure_response_failure.sum())}/{len(f1_frames)} 는 "
            "미검출 corner 가 **전부 화면 안**이다.\n"
            "[확인] 즉 F1 은 '화면 밖이라 안 잡힌 것'이 아니라 화면 안에 보이는 corner 에서도 "
            "belief 가 안 뜨는 **진짜 response 실패**가 다수다.\n"
            "[추정] 따라서 Candidate A 를 '화면 밖 supervision 문제'로만 좁히면 안 된다.  "
            f"outside-aware head 가 직접 겨냥하는 것은 화면 밖 {off_image} + sentinel {sentinel} 관측이고, "
            "나머지는 appearance/response 문제다.\n"
        )
    lines.append("\n도메인·truncation 별:\n")
    lines.append(table(
        classes.groupby(["failure_class", "domain", "is_truncated"]).size()
        .reset_index(name="frames"),
        ["failure_class", "domain", "is_truncated", "frames"],
    ))
    lines.append(
        "\n[확인] threshold sensitivity (×0.75 / ×1.0 / ×1.25):\n"
    )
    lines.append(table(sensitivity, list(sensitivity.columns)))

    lines.append("\n## 3. First-break stage\n")
    lines.append(table(
        break_counts.rename_axis("first_break_stage").reset_index(name="frames").assign(
            fraction=lambda d: d.frames / total
        ),
        ["first_break_stage", "frames", "fraction"],
    ))

    lines.append("\n## 4. Decoder recovery\n")
    lines.append(table(
        decoder,
        ["variant", "pose_success", "pose_success_delta_pp", "yaw_median",
         "delta_yaw_median", "matched_2d_median", "delta_matched_2d_median",
         "decoder_is_primary_lever"],
    ))
    primary_lever = bool(decoder.loc[decoder.variant != "D2", "decoder_is_primary_lever"].any())
    decoder_frames = int(breaks.decoder_recovers.sum())
    lines.append(
        f"\n[확인] decoder 교체만으로 판정 기준(pose success +5%p / yaw −20% / 2D −5px)을 넘는 변형이 "
        f"{'있다' if primary_lever else '없다'} → decoder 는 "
        f"{'architecture 보다 먼저 고칠 단계' if primary_lever else '보조 원인'}.\n"
    )
    lines.append(
        f"[확인] 다만 frame 단위로는 {decoder_frames} 프레임에서 어떤 decoder 하나가 baseline 을 "
        "회복한다.  population 수준 효과가 0 인데 frame 수준 회복이 존재한다는 것은 "
        "체계적 이득이 아니라 PnP candidate 선택의 불안정성에 가깝다 [추정].\n"
    )
    lines.append(
        "[확인] D2(canonical eval decoder)는 D0 대비 pose success 가 3.4%p **낮고**(67 vs 70) "
        "yaw median 은 낮다 — 두 값은 서로 다른 프레임 집합 위의 median 이므로 직접 비교하면 안 된다.\n"
    )

    lines.append("\n## 5. Keypoint / group oracle recovery\n")
    grouped = (
        oracle.groupby("variant")
        .agg(
            recovered=("recovered_frames", "sum"),
            lost=("lost_frames", "sum"),
            delta_yaw=("delta_yaw_median", "median"),
            delta_reproj=("delta_reproj_median", "median"),
        )
        .reset_index()
        .sort_values("delta_yaw")
    )
    lines.append(table(grouped, ["variant", "recovered", "lost", "delta_yaw", "delta_reproj"], limit=60))

    lines.append("\n### failure class 별 최초 회복 intervention\n")
    first_recovery = []
    for failure, group in oracle.groupby("failure_class"):
        best = group.sort_values(["recovered_frames", "delta_yaw_median"],
                                 ascending=[False, True]).head(3)
        for _, row in best.iterrows():
            first_recovery.append(
                {
                    "failure_class": failure,
                    "variant": row.variant,
                    "recovered_frames": row.recovered_frames,
                    "delta_yaw_median": row.delta_yaw_median,
                    "delta_reproj_median": row.delta_reproj_median,
                }
            )
    lines.append(table(pd.DataFrame(first_recovery), [
        "failure_class", "variant", "recovered_frames", "delta_yaw_median",
        "delta_reproj_median"], limit=40))

    lines.append("\n## 6. Stage-wise progression (belief stage 1~6)\n")
    lines.append(table(
        stages.groupby(["stage_label", "group_near_far"]).size().reset_index(name="n"),
        ["stage_label", "group_near_far", "n"],
    ))
    lines.append("\nstage 별 median 위치오차 / peak (detected keypoint):\n")
    lines.append(table(trajectory, [
        "group", "stage", "n", "median_err_px", "err_delta_vs_stage1_px",
        "median_peak", "peak_ratio_vs_stage1"], limit=30))

    far_traj = trajectory[trajectory.group == "far"].set_index("stage")
    near_traj = trajectory[trajectory.group == "near"].set_index("stage")
    early_wrong = int((stages.stage_label == "EARLY_WRONG").sum())
    late_drift = int((stages.stage_label == "LATE_DRIFT").sum())
    far_best_stage = int(far_traj.median_err_px.astype(float).idxmin())
    lines.append(
        f"\n[확인] EARLY_WRONG {early_wrong} vs LATE_DRIFT {late_drift} "
        "(EARLY_WRONG = stage1 과 stage6 모두 오차 >20px, LATE_DRIFT = stage1 ≤10px 인데 "
        "stage6 >20px).  오류의 대부분은 refinement 가 **만들어내는** 것이 아니라 "
        "stage 1 에 이미 있고 끝까지 남는다.\n"
        f"[확인] far face median 오차: stage1 {far_traj.loc[1, 'median_err_px']:.1f}px → "
        f"stage{far_best_stage} {far_traj.loc[far_best_stage, 'median_err_px']:.1f}px (최저) → "
        f"stage6 {far_traj.loc[6, 'median_err_px']:.1f}px.  "
        f"즉 stage {far_best_stage} 이후로는 더 줄지 않고 오히려 소폭 되돌아간다.  "
        f"같은 구간에서 peak 는 {far_traj.loc[far_best_stage, 'median_peak']:.2f} → "
        f"{far_traj.loc[6, 'median_peak']:.2f} 로 계속 오른다 "
        f"(stage1 대비 {far_traj.loc[6, 'peak_ratio_vs_stage1']:.1f}배).\n"
        f"[확인] 최종 far 오차 {far_traj.loc[6, 'median_err_px']:.1f}px 는 near "
        f"{near_traj.loc[6, 'median_err_px']:.1f}px 의 "
        f"{float(far_traj.loc[6, 'median_err_px']) / float(near_traj.loc[6, 'median_err_px']):.1f}배다.  "
        "near 도 stage 5 이후 정체하지만 정체 수준 자체가 다르다.\n"
        "[추정] 후반 refinement 는 far face 에서 **위치를 더 고치지 못하고 신뢰도만 올린다**.  "
        "이것이 far/rear 가 confidently-wrong 으로 나타나는 기전과 정합한다.\n"
    )

    lines.append("\n## 7. Same-image counterfactual\n")
    cf_rows = []
    for kind in COUNTERFACTUAL_KINDS:
        subset = ok[ok.variant == kind]
        if not len(subset):
            continue
        cf_rows.append(
            {
                "variant": kind,
                "n": len(subset),
                "corners_detected": nanmedian(subset.n_corner_detected.values),
                "pose_success_rate": float(subset.pose_success.mean()),
                "matched_2d_median": nanmedian(subset.matched_median_px.values),
                "far_2d_median": nanmedian(subset.far_matched_median_px.values),
                "yaw_median": nanmedian(subset.yaw_err_deg.values),
            }
        )
    lines.append(table(pd.DataFrame(cf_rows), [
        "variant", "n", "corners_detected", "pose_success_rate",
        "matched_2d_median", "far_2d_median", "yaw_median"], limit=20))
    lines.append(
        f"\n[확인] 모든 counterfactual 의 affine/K/GT 정합 오차 최대 "
        f"{float(ok.geometry_identity_err_px.max()) if len(ok) else 0.0:.2e} px "
        f"(< {GEOMETRY_TOL_PX} px 게이트).\n"
    )
    if len(pairs):
        lines.append("\n### same-source paired transition (동일 이미지, 한 요소만 변경)\n")
        lines.append(table(pairs, [
            "depth", "variant", "kind", "n_paired", "corners_detected_median",
            "delta_corners_vs_original", "pose_success_rate", "delta_pose_success_pp",
            "delta_peak_median", "delta_far2d_median"], limit=20))
        for depth in pairs.depth.unique():
            block = pairs[pairs.depth == depth].set_index("kind")
            lines.append(
                f"\n[확인] {depth}: crop-only 는 원본 대비 corner 검출 "
                f"{block.loc['crop_only', 'delta_corners_vs_original']:+.0f}개, "
                f"pose success {block.loc['crop_only', 'delta_pose_success_pp']:+.0f}%p, "
                f"peak {block.loc['crop_only', 'delta_peak_median']:+.2f}.  "
                f"같은 crop 을 원래 캔버스로 reflect-pad 하면 검출 "
                f"{block.loc['reflect_pad', 'delta_corners_vs_original']:+.0f}개 "
                f"(= 원본 수준 복귀), pose success "
                f"{block.loc['reflect_pad', 'delta_pose_success_pp']:+.0f}%p, "
                f"constant-pad 는 {block.loc['constant_pad', 'delta_pose_success_pp']:+.0f}%p."
            )
        lines.append(
            "\n[확인] 그러나 padding 이 회복하는 것은 **response** 이고, far-face 2D 오차는 "
            "모든 padding 변형에서 원본보다 나쁘다 (delta_far2d 전부 양수).\n"
            "[주의] far2d/matched 2D 중앙값은 검출된 corner 수가 변형마다 달라 짝이 달라진다 — "
            "crop-only 처럼 corner 가 2개만 남은 경우의 2D 값은 검출된 소수 점에 대한 값이므로 "
            "변형 간 직접 비교는 제한적이다.  검출 수·pose success·peak 가 더 견고한 비교축이다.\n"
            "[추정] 현재 모델은 실제 frame-edge truncation 에 취약하고 reflect-pad 가 그 실패를 "
            "가린다는 frozen-response 증거다.  matched retraining 전까지 "
            "'현재 augmentation 이 학습 실패의 인과 원인'이라고는 말할 수 없다.\n"
        )
    blur = ok[ok.variant == "C10"]
    dark = ok[ok.variant == "C11"]
    small = ok[ok.variant == "C9"]
    lines.append(
        f"\n[확인] 같은 이미지에서 blur 만 준 C10 은 corner 검출 median "
        f"{nanmedian(blur.n_corner_detected.values):.1f} 로 붕괴하지만, luma 만 낮춘 C11 "
        f"({nanmedian(dark.n_corner_detected.values):.1f}) 과 0.5배 축소한 C9 "
        f"({nanmedian(small.n_corner_detected.values):.1f}) 은 원본 수준을 유지한다 — "
        "어둠·작은 물체 자체는 response 붕괴의 원인이 아니다.\n"
    )

    lines.append("\n## 8. 지지된 원인\n")
    far_group = oracle[oracle.variant.isin(["O3_far", "O7_depth_right", "O10_all_corners"])]
    lines.append(
        f"- [확인] 2D correspondence 품질이 지배적 손실이다 — GT 치환으로 "
        f"{gate['gt2d_pose_success']}/{gate['strict_n']} 회복하고, 8 corner 만 GT 로 바꿔도 "
        f"{int(oracle[oracle.variant == 'O10_all_corners'].recovered_frames.sum())} 실패 프레임이 전부 회복된다.\n"
        f"- [확인] 실패는 두 개의 서로 다른 population 이다: response 자체가 없는 "
        f"{counts.get('F1_NO_RESPONSE', 0)} 프레임(median corner 검출 "
        f"{float(profile.loc[profile.failure_class == 'F1_NO_RESPONSE', 'corners_detected'].iloc[0]):.0f}개, "
        f"peak {float(profile.loc[profile.failure_class == 'F1_NO_RESPONSE', 'frame_peak'].iloc[0]):.2f})와, "
        f"response 는 강한데 far face 위치가 틀린 {counts.get('F2_CONFIDENT_WRONG', 0)} 프레임"
        f"(peak {float(profile.loc[profile.failure_class == 'F2_CONFIDENT_WRONG', 'frame_peak'].iloc[0]):.2f}, "
        f"far 2D {float(profile.loc[profile.failure_class == 'F2_CONFIDENT_WRONG', 'far_2d_px'].iloc[0]):.0f}px).\n"
        "- [확인] 오차는 far/depth face 에 집중된다.  far·depth-right group GT 치환의 회복 프레임 수가 "
        f"near group 보다 크다 ({int(oracle[oracle.variant == 'O3_far'].recovered_frames.sum())}/"
        f"{int(oracle[oracle.variant == 'O7_depth_right'].recovered_frames.sum())} vs "
        f"{int(oracle[oracle.variant == 'O2_near'].recovered_frames.sum())}).\n"
        "- [확인] frame-edge truncation 은 response 를 붕괴시킨다 (same-source crop-only counterfactual).\n"
    )

    lines.append("\n## 9. 반증된 원인\n")
    lines.append(
        "- [확인] solver·K·dimensions 단독 원인 — GT 2D 아래에서 87/87 이므로 기각.\n"
        f"- [확인] centroid 단독 원인 — O9_centroid 는 실패 프레임을 "
        f"{int(oracle[oracle.variant == 'O9_centroid'].recovered_frames.sum())}개 회복하고 yaw 를 오히려 "
        f"{float(oracle[oracle.variant == 'O9_centroid'].delta_yaw_median.median()):+.2f}° 바꾼다 → 기각.\n"
        "- [확인] decoder 선택 (training softargmax vs canonical eval vs argmax vs offset 제거) — "
        "population 수준에서 판정 기준 미달 → 주원인 아님.\n"
        f"- [확인] **'late-stage refinement 가 위치를 망친다(late drift)'는 가설 기각** — "
        f"EARLY_WRONG {early_wrong} vs LATE_DRIFT {late_drift}.  far face 오차는 stage 1 에서 이미 "
        f"{far_traj.loc[1, 'median_err_px']:.0f}px 이고, refinement 로 "
        f"{far_traj.loc[far_best_stage, 'median_err_px']:.0f}px 까지만 줄었다가 stage 6 "
        f"{far_traj.loc[6, 'median_err_px']:.0f}px 로 정체한다.  "
        "따라서 Candidate B 의 근거는 'late drift' 가 아니라 '초기부터 틀린 위치를 "
        "refinement 가 끝까지 못 고치고 신뢰도만 올린다' 로 수정되어야 한다.\n"
        "- [확인] 저조도·작은 물체 자체 — C11(luma)·C9(downscale) counterfactual 에서 response 유지 → 기각.\n"
        "- [추정] occlusion — true occlusion metadata 가 없어 인과 주장을 하지 않는다 (truncation 과 구분).\n"
    )

    lines.append("\n## 10-11. Architecture 후보 순위 / 보류·폐기\n")
    lines.append(table(decision, [
        "priority", "candidate", "target_failure", "supporting_frames",
        "supporting_fraction", "oracle_recovery_frames", "target_class_delta_yaw",
        "target_class_delta_reproj", "stage_evidence"]))
    lines.append(
        "\n[주의] `oracle_recovery_frames` 는 baseline 이 **실패**한 프레임의 회복 수인데, "
        "실패 17 프레임은 전부 F1 에 있으므로 A·B 모두 17 로 포화된다.  두 후보를 가르는 값은 "
        "`target_class_delta_yaw` — 각 후보의 GT 치환이 **자기 target class 안에서** 만드는 "
        "yaw 개선량이다.\n"
    )
    lines.append("\n### 단일 keypoint GT 치환의 frame 단위 일관성\n")
    lines.append(table(consistency, [
        "variant", "n_paired", "fraction_improved", "median_delta_yaw",
        "recovered_frames"], limit=12))
    kp5 = consistency[consistency.variant == "O1_kp5"]
    kp5_fraction = float(kp5.fraction_improved.iloc[0]) if len(kp5) else float("nan")
    drop_far = oracle[oracle.variant == "A_drop_far"]

    lines.append(
        "\nREJECTED / DEFERRED:\n"
        f"- REJECTED `fixed kp5 pixel correction` — kp5 단일 GT 치환은 실패 프레임을 "
        f"{int(oracle[oracle.variant == 'O1_kp5'].recovered_frames.sum())}개만 회복하고, "
        f"프레임의 {kp5_fraction:.0%} 에서만 yaw 가 개선된다.  상수 픽셀 보정으로 재현할 수 있는 "
        "일관된 부호가 아니다.\n"
        f"- REJECTED `fixed centroid upward shift` — O9_centroid 는 실패 프레임 "
        f"{int(oracle[oracle.variant == 'O9_centroid'].recovered_frames.sum())}개 회복, "
        f"yaw 중앙값 {float(oracle[oracle.variant == 'O9_centroid'].delta_yaw_median.median()):+.2f}°.\n"
        "- DEFERRED `raw covariance weighting` — local 7×7 covariance 는 calibration 없이 쓸 수 없다.  "
        "게다가 keypoint 제거 계열(A_drop_*, LOO_*) 중 실패 프레임을 회복시킨 변형은 하나도 없고 "
        f"far face 를 통째로 빼면 {int(drop_far.lost_frames.sum())} 프레임을 오히려 잃는다 → "
        "'나쁜 점을 빼거나 가중치를 낮추는' 접근만으로는 회복이 안 된다.\n"
        "- REJECTED `solver sweep only` — solver 변경은 GT 2D 아래에서 이미 87/87 이므로 상류를 못 고친다.\n"
        f"- DEFERRED `unconditional pose residual` — F4 표본이 "
        f"{counts.get('F4_SOLVER_SPECIFIC', 0)} 프레임뿐이라 direct-pose 근거가 약하다.\n"
    )

    lines.append("\n## 12. Micro-training 계획\n")
    lines.append("`MICRO_TRAIN_PLAN.md` 참조 (이번 작업에서 학습은 실행하지 않음).\n")

    lines.append("\n## 13. 남은 불확실성\n")
    lines.append(
        "- [추정] true occlusion metadata 가 없으므로 occlusion 인과 주장은 하지 않는다. "
        "본 보고서의 `truncation` 은 프레임 경계 잘림만 뜻한다.\n"
        "- [추정] counterfactual 은 frozen response 증거이며, matched retraining 전까지 "
        "'현재 augmentation 이 학습 실패의 인과 원인'이라고 말할 수 없다.\n"
        f"- [확인] primary N={total} 은 소표본이다. failure class 별 subset 은 더 작으므로 "
        "class 별 결론은 예비적이다.\n"
        "- [확인] manual36 은 exploratory PL-pool 이라 primary 결과와 합치지 않았다.\n"
    )

    lines.append("\n## 그림\n")
    for name in figures:
        lines.append(f"- `{name}`")
    lines.append(
        "\n정성 예시 (GT 초록 / 예측 빨강, stage 예시는 stage1 파랑 / stage6 빨강):\n"
        "- `failure_class_examples/` — failure class 별 대표 프레임\n"
        "- `late_drift_examples/` — LATE_DRIFT keypoint 가 많은 프레임\n"
        "- `error_propagation_examples/` — first-break stage 별 대표 프레임\n"
    )

    (OUT_DIR / "MECHANISM_DIAGNOSTIC_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_architecture_decision(
    decision: pd.DataFrame, classes: pd.DataFrame, oracle: pd.DataFrame,
    decoder: pd.DataFrame, breaks: pd.DataFrame, missing_summary: pd.DataFrame,
    missing_frames: pd.DataFrame
) -> None:
    total = len(classes)
    counts = classes.failure_class.value_counts().to_dict()
    ranked = decision.sort_values("priority")
    lines = ["# ARCHITECTURE DECISION — PAPER_S2 ep57 mechanism diagnosis\n"]
    lines.append("[관찰]")
    lines.append(
        f"strict N{total} 에서 failure class 분포는 "
        + ", ".join(f"{k} {v}" for k, v in counts.items())
        + f" 이고, first-break stage 는 "
        + ", ".join(
            f"{k} {v}" for k, v in breaks.first_break_stage.value_counts().items()
        )
        + " 이다.\n"
    )
    lines.append("[원인 후보]")
    for _, row in ranked.iterrows():
        lines.append(f"- {row.candidate} → {row.target_failure}")
    lines.append("\n[지지 증거]")
    lines.append(
        "- [확인] GT 2D 치환으로 pose 가 회복되므로 손실은 solver 하류가 아니라 2D correspondence 상류다."
    )
    for _, row in ranked.iterrows():
        lines.append(
            f"- [확인] {row.candidate}: 지지 프레임 {row.supporting_frames} "
            f"({row.supporting_fraction:.0%}), oracle 회복 {row.oracle_recovery_frames} 프레임."
        )
    def recovered(variant: str) -> int:
        return int(oracle[oracle.variant == variant].recovered_frames.sum())

    lines.append(
        f"- [확인] far group GT 치환(O3_far) {recovered('O3_far')} 프레임, "
        f"depth-right(O7, kp1/2/5/6) {recovered('O7_depth_right')} 프레임 회복 vs "
        f"near group(O2) {recovered('O2_near')}, depth-left(O6, kp0/3/4/7) "
        f"{recovered('O6_depth_left')} — far/depth 축이 near 축보다 강한 레버."
    )
    lines.append(
        "- [확인] keypoint convention 은 `annotate_pnp.make_pallet_keypoints_3d` 좌표에서 "
        "직접 확인했다: LEFT={0,3,4,7}, RIGHT={1,2,5,6}, near={0,1,2,3}, far={4,5,6,7}."
    )
    f1_row = missing_summary[missing_summary.failure_class == "F1_NO_RESPONSE"]
    f1_frames = missing_frames[missing_frames.failure_class == "F1_NO_RESPONSE"]
    lines.append("\n[반증 증거]")
    lines.append(
        "- [확인] solver / K / dimensions / centroid 단독 원인은 GT 2D 조건에서 전부 정상이므로 기각."
    )
    if len(f1_row):
        lines.append(
            f"- [확인] 'F1 은 화면 밖 keypoint 때문'이라는 해석 기각 — 미검출 corner "
            f"{int(f1_row.undetected_gt_in_frame.iloc[0])}개가 GT 화면 **안**이고, "
            f"{int(f1_frames.pure_response_failure.sum())}/{len(f1_frames)} 프레임은 "
            "미검출이 전부 화면 안이다.  Candidate A 를 outside-supervision 문제로만 "
            "좁히면 F1 의 대부분을 놓친다."
        )
    lines.append(
        f"- [확인] centroid 단독(O9_centroid)은 실패 프레임을 {recovered('O9_centroid')}개 회복한다 → "
        "centroid 보정 가설 기각."
    )
    lines.append(
        "- [확인] 'late-stage refinement 가 far face 를 망친다'는 가설 기각 "
        "(EARLY_WRONG >> LATE_DRIFT).  Candidate B 의 근거는 late drift 가 아니라 "
        "'초기 오류를 refinement 가 못 고치고 신뢰도만 올린다' 이다."
    )
    lines.append(
        "- [추정] depth-left/right 비대칭(회복 "
        f"{recovered('O6_depth_left')} vs {recovered('O7_depth_right')})은 소표본(실패 17)에서의 "
        "관찰이며, 좌우 어느 쪽이 원인인지 단정하지 않는다."
    )
    primary_lever = bool(decoder.loc[decoder.variant != "D2", "decoder_is_primary_lever"].any())
    lines.append(
        f"- [확인] decoder 교체 단독 효과는 판정 기준을 "
        f"{'넘는다' if primary_lever else '넘지 못한다'} → decoder 는 "
        f"{'선행 수정 대상' if primary_lever else '보조 원인'}."
    )
    lines.append("\n[현재 판정]")
    top = ranked.iloc[0]
    lines.append(
        f"- [확인] 최다 실패 class 는 {top.target_failure} 이며 대응 후보는 {top.candidate} 이다."
    )
    lines.append(
        "- [추정] 단일 평균 성능이 아니라 failure class 별로 architecture 를 고른다는 전제 아래, "
        "아래 우선순위는 micro-training 으로만 확정된다."
    )
    lines.append("\n[architecture 우선순위]")
    for index, (_, row) in enumerate(ranked.iterrows(), start=1):
        lines.append(f"{index}. {row.candidate} ({row.target_failure})")
    lines.append("\n[다음 admissible experiment]")
    lines.append(
        "- MICRO_TRAIN_PLAN.md 의 후보 3개를 1-seed, 200~500 frame, 5~10 epoch 으로만 실행한다. "
        "final-test 는 열지 않는다."
    )
    (OUT_DIR / "ARCHITECTURE_DECISION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_micro_train_plan(decision: pd.DataFrame, classes: pd.DataFrame) -> None:
    ranked = decision.sort_values("priority").head(3)
    blocks = {
        "A_visibility_partial_keypoint": {
            "hypothesis": "F1 미검출 corner 의 대부분은 GT 가 화면 **안**이므로(진단 §2) 원인은 "
                          "'화면 밖 supervision' 만이 아니다.  단 화면 밖/경계 keypoint 를 "
                          "background-negative 로 감독하면 경계 근처 response 가 함께 억제된다는 "
                          "가설은 남는다.  outside 를 loss 에서 빼고 crop-only 로 경계 사례를 "
                          "보여주면 in-frame response 도 함께 회복된다.",
            "change": "train.py belief target 생성에서 outside keypoint 를 loss mask 로 제외하고 "
                      "(9ch belief + 9ch validity mask), PnP 는 valid correspondence 만 사용. "
                      "증강은 reflect-pad 가 아닌 **crop-only** 로 실제 frame-edge 분포를 준다.",
            "trainable": "m6_2 final belief stage + 새 validity head",
            "frozen": "vgg backbone, stage1~5",
            "subset": "crop-only 증강 포함 400 frames (truncation 비율 40%)",
            "target": "F1_NO_RESPONSE",
        },
        "B_far_depth_structured_refinement": {
            "hypothesis": "far/depth face 는 response 는 있는데 위치가 틀린다 (confident-wrong).  "
                          "dimension-conditioned bounded residual 이 far face 를 교정한다.",
            "change": "final belief 위에 bounded residual heatmap head + FiLM(dims) 추가, "
                      "far/depth edge consistency loss 를 heatmap anchor 와 함께 사용.",
            "trainable": "residual head + FiLM",
            "frozen": "backbone, stage1~6 belief",
            "subset": "non-truncated 400 frames",
            "target": "F2_CONFIDENT_WRONG",
        },
        "C_calibrated_uncertainty_robust_pnp": {
            "hypothesis": "작은 2D error 가 PnP 에서 증폭된다.  learned localization sigma 로 "
                          "correspondence 를 가중하면 yaw 가 안정된다.",
            "change": "per-corner log-sigma head (cornerQuality) 학습 + bounded inverse-variance "
                      "weighted PnP (raw covariance 금지, calibration 후 사용).",
            "trainable": "corner quality head",
            "frozen": "backbone + belief 전부",
            "subset": "mixed 500 frames",
            "target": "F3_GEOMETRY_AMPLIFIED",
        },
        "D_direct_pose_gated_residual": {
            "hypothesis": "PnP 이전 정보는 정상인데 solver 가 catastrophic 하게 실패한다.",
            "change": "gated direct-pose branch (heatmap feature -> 6D), PnP 실패시에만 사용.",
            "trainable": "pose branch",
            "frozen": "backbone + belief",
            "subset": "F4 프레임 중심 300 frames",
            "target": "F4_SOLVER_SPECIFIC",
        },
    }
    lines = ["# MICRO TRAIN PLAN — 계획만 작성, 이번 작업에서 실행하지 않음\n"]
    lines.append(
        "공통 조건: 200~500 training frames / mechanism-val 평가 / 5~10 epoch / seed 1 / "
        "동일 initialization / 동일 sampler order / 동일 optimizer / **한 요소만 변경**.\n"
    )
    lines.append(
        "승격 gate: primary target failure class 에서 pose success +5%p **또는** "
        "keypoint gross error −10%p **또는** yaw/reprojection 10% 이상 개선이 있고, "
        "clean/non-target subset 성능 하락이 3%p 미만일 때만 1-seed full smoke 후보로 승격.\n"
    )
    for index, (_, row) in enumerate(ranked.iterrows(), start=1):
        block = blocks[row.candidate]
        target_n = int(classes.failure_class.eq(block["target"]).sum())
        lines.append(f"\n## 후보 {index} — {row.candidate}\n")
        lines.append(f"- hypothesis: {block['hypothesis']}")
        lines.append(f"- exact code change: {block['change']}")
        lines.append(f"- trainable parameters: {block['trainable']}")
        lines.append(f"- frozen parameters: {block['frozen']}")
        lines.append(f"- train subset: {block['subset']}")
        lines.append(
            f"- validation failure class: {block['target']} (mechanism-val N={target_n})"
        )
        lines.append(
            "- metrics: pose success rate, yaw median, fixed-GT reproj median, "
            "matched 2D median, gross(>20px) 비율"
        )
        lines.append(
            f"- expected result: {block['target']} 에서 pose success +5%p 이상 또는 "
            "yaw median 10% 이상 감소"
        )
        lines.append(
            "- failure result: target class 무변화이거나 non-target 하락 3%p 이상 → 후보 폐기"
        )
        lines.append(
            "- stop condition: 10 epoch 도달, 또는 val loss 2 epoch 연속 악화, 또는 gate 미달 확정"
        )
        lines.append("- runtime estimate: 400 frames × 10 epoch ≈ 25~40 분 (RTX 3080, batch 8)")
    (OUT_DIR / "MICRO_TRAIN_PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_provenance(
    manifest: dict[str, Any], cache_info: dict[str, Any], gate: dict[str, Any],
    counts: dict[str, int]
) -> None:
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception:
        gpu = "unavailable"
    lines = [
        "# RUN PROVENANCE — paper_s2_mechanism_diagnostic\n",
        f"- created: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"- git HEAD: {git_head()}",
        f"- git status: {'clean' if not subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,cwd=str(ROOT)).stdout.strip() else 'dirty'}",
        f"- checkpoint: {FZ.WEIGHTS} (SHA-256 {FZ.WEIGHTS_SHA256})",
        f"- python: {sys.version.split()[0]}",
        f"- torch: {torch.__version__} (cuda {torch.version.cuda})",
        f"- opencv: {cv2.__version__}",
        f"- pandas: {pd.__version__} / numpy: {np.__version__}",
        f"- gpu: {gpu}",
        f"- cache key: {cache_info.get('cache_key')}",
        f"- cache model forwards: {cache_info.get('model_forwards')}",
        f"- manifest membership SHA: {manifest.get('membership_sha256')}",
        f"- final-test open count: {manifest['final_test_guard']['final_test_open_count']}",
        f"- prohibited attempts: {manifest['final_test_guard']['prohibited_attempts']}",
        "",
        "## Baseline reproduction gate",
        f"- GT-2D PnP: {gate['gt2d_pose_success']}/{gate['strict_n']} (expected {BASELINE_EXPECT['gt2d_pose_success']})",
        f"- predicted PnP: {gate['pred_pose_success']}/{gate['strict_n']} (expected {BASELINE_EXPECT['pred_pose_success']})",
        f"- yaw median: {gate['yaw_median_deg']:.4f}° (expected {BASELINE_EXPECT['yaw_median_deg']} ±{BASELINE_TOL['yaw_median_deg']})",
        f"- fixed-GT reproj median: {gate['fixed_gt_reproj_median_px']:.4f} px (expected {BASELINE_EXPECT['fixed_gt_reproj_median_px']} ±{BASELINE_TOL['fixed_gt_reproj_median_px']})",
        f"- passed: {gate['passed']}",
        "",
        "## Row counts",
    ]
    for name, value in counts.items():
        lines.append(f"- {name}: {value}")
    lines.append(
        "\n## Reused prior artifacts (not recomputed)\n"
        "- `data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit/full_ep57_frozen_20260728/`"
        " — frozen frames/keypoints/yaw-ladder (baseline regression reference)\n"
        "- `data/pallet/results/paper_s2_target_semantics_audit/` — target semantics / decoder parity /"
        " truncation population / DiffPnP funnel\n"
        "- shared code: `paper_s2_frozen_diagnostic`, `paper_s2_decoder_parity_audit`,"
        " `filter_pr_camfacing.extract_keypoints_from_belief`, `annotate_pnp`\n"
    )
    (OUT_DIR / "RUN_PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# report driver
# ============================================================================
def run_report(
    manifest: dict[str, Any], cache_info: dict[str, Any],
    audit: Optional[FZ.InputAudit] = None
) -> None:
    frames = pd.read_parquet(OUT_DIR / "frames.parquet")
    keypoints = pd.read_parquet(OUT_DIR / "keypoints.parquet")
    poses = pd.read_parquet(OUT_DIR / "poses.parquet")
    interventions = pd.read_parquet(OUT_DIR / "interventions.parquet")
    counterfactuals = (
        pd.read_parquet(OUT_DIR / "counterfactuals.parquet")
        if (OUT_DIR / "counterfactuals.parquet").is_file()
        else pd.DataFrame(columns=["status", "variant", "frame_id"])
    )
    gate = json.loads((OUT_DIR / "baseline_gate.json").read_text(encoding="utf-8"))

    classes = classify_frames(frames, poses, keypoints, interventions)
    stages = stage_progression(keypoints)
    breaks = first_break(classes, poses, stages)
    oracle = oracle_recovery_matrix(interventions, classes)
    decoder = decoder_recovery_table(poses)
    trajectory = stage_trajectory(keypoints)
    pairs = counterfactual_pairs(counterfactuals)

    sensitivity_rows = []
    for scale in SENSITIVITY_SCALES:
        alt = classify_frames(frames, poses, keypoints, interventions, scale=scale)
        row = {"threshold_scale": scale}
        row.update(alt.failure_class.value_counts().to_dict())
        sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows).fillna(0)

    consistency = single_keypoint_consistency(interventions)
    missing_summary, missing_frames = missing_response_audit(keypoints, classes)
    decision = architecture_decision(
        classes, oracle, decoder, counterfactuals, stages, trajectory, interventions
    )

    classes.to_csv(OUT_DIR / "failure_class_frames.csv", index=False)
    classes.failure_class.value_counts().rename_axis("failure_class").reset_index(
        name="frames"
    ).to_csv(OUT_DIR / "failure_class_counts.csv", index=False)
    classes.groupby(["failure_class", "domain"]).size().reset_index(name="frames").to_csv(
        OUT_DIR / "failure_class_by_domain.csv", index=False
    )
    classes.groupby(["failure_class", "session_id"]).size().reset_index(
        name="frames"
    ).to_csv(OUT_DIR / "failure_class_by_session.csv", index=False)
    keypoints.merge(classes[["frame_id", "failure_class"]], on="frame_id").groupby(
        ["failure_class", "group_near_far"]
    )["D0_err_px"].median().reset_index(name="median_err_px").to_csv(
        OUT_DIR / "failure_class_by_keypoint_group.csv", index=False
    )
    oracle.to_csv(OUT_DIR / "oracle_recovery_matrix.csv", index=False)
    decoder.to_csv(OUT_DIR / "decoder_recovery.csv", index=False)
    breaks.to_csv(OUT_DIR / "first_break_stage.csv", index=False)
    breaks.first_break_stage.value_counts().rename_axis(
        "first_break_stage"
    ).reset_index(name="frames").to_csv(
        OUT_DIR / "first_break_stage_counts.csv", index=False
    )
    stages.to_csv(OUT_DIR / "stage_progression_by_kp.csv", index=False)
    stages.merge(classes[["frame_id", "failure_class"]], on="frame_id").groupby(
        ["failure_class", "stage_label"]
    ).size().reset_index(name="n").to_csv(
        OUT_DIR / "stage_progression_by_failure.csv", index=False
    )
    decision.to_csv(OUT_DIR / "architecture_decision_matrix.csv", index=False)
    sensitivity.to_csv(OUT_DIR / "failure_class_threshold_sensitivity.csv", index=False)
    trajectory.to_csv(OUT_DIR / "stage_error_trajectory.csv", index=False)
    consistency.to_csv(OUT_DIR / "single_keypoint_consistency.csv", index=False)
    missing_summary.to_csv(OUT_DIR / "missing_response_audit.csv", index=False)
    missing_frames.to_csv(OUT_DIR / "missing_response_by_frame.csv", index=False)
    if len(pairs):
        pairs.to_csv(OUT_DIR / "counterfactual_paired_transitions.csv", index=False)

    figures = make_figures(
        classes, breaks, oracle, decoder, stages, counterfactuals,
        interventions, decision
    )
    examples = write_examples(
        manifest, classes, breaks, stages, keypoints,
        audit if audit is not None else FZ.InputAudit()
    )
    log("[examples] " + json.dumps(examples))
    write_report(
        manifest, gate, classes, breaks, oracle, decoder, stages,
        counterfactuals, decision, sensitivity, figures, cache_info,
        trajectory, pairs, consistency, missing_summary, missing_frames
    )
    write_architecture_decision(
        decision, classes, oracle, decoder, breaks, missing_summary, missing_frames
    )
    write_micro_train_plan(decision, classes)
    write_provenance(
        manifest, cache_info, gate,
        {
            "frames": len(frames),
            "keypoints": len(keypoints),
            "poses": len(poses),
            "interventions": len(interventions),
            "counterfactuals": len(counterfactuals),
            "figures": len(figures),
            **{f"examples_{k}": v for k, v in examples.items()},
        },
    )
    summary = {
        "failure_class_counts": classes.failure_class.value_counts().to_dict(),
        "first_break_stage_counts": breaks.first_break_stage.value_counts().to_dict(),
        "baseline_gate": gate,
        "decoder_is_primary_lever": bool(
            decoder.loc[decoder.variant != "D2", "decoder_is_primary_lever"].any()
        ),
        "architecture_priority": decision.sort_values("priority").candidate.tolist(),
        "figures": figures,
        "examples": examples,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2), encoding="utf-8"
    )
    log("[report] " + json.dumps(summary["failure_class_counts"]))
    log("[report] " + json.dumps(summary["first_break_stage_counts"]))


# ============================================================================
# main
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--counterfactuals", action="store_true")
    parser.add_argument("--interventions", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore cache reuse")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not any(
        (args.build_cache, args.counterfactuals, args.interventions, args.report, args.all)
    ):
        args.all = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = FZ.InputAudit()

    manifest = None
    cache_info: dict[str, Any] = {}
    if args.all or args.build_cache:
        manifest, cache_info = build_cache(force=args.force)
    if manifest is None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if CACHE_MANIFEST_PATH.is_file():
            cache_info = json.loads(CACHE_MANIFEST_PATH.read_text(encoding="utf-8"))

    if args.all or args.interventions:
        _, _, gate = run_interventions(manifest, audit)
        (OUT_DIR / "baseline_gate.json").write_text(
            json.dumps(jsonable(gate), indent=2), encoding="utf-8"
        )
    if args.all or args.counterfactuals:
        run_counterfactuals(manifest, audit)
    if args.all or args.report:
        run_report(manifest, cache_info, audit)

    if audit.prohibited_attempts:
        raise RuntimeError(f"final-test access attempted: {audit.prohibited_attempts}")
    log("[done] " + str(OUT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
