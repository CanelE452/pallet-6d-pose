"""MULTI_TEACHER_CORNER_DISTILL_V1 공용 라이브러리.

이 트랙은 기존 paper artifact 를 **읽기만** 한다.  모집단·추론 recipe·6D 평가기는
`paper_pose_metric_closure_v1` 의 것을 그대로 재사용하고, 산출물만 새 namespace 에 쓴다.

정의를 여기 한 곳에만 둔다 — 게이트 스크립트마다 다시 쓰면 갈라진다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------- read-only --
CLOSURE = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
MANIFEST_PATH = CLOSURE / "AXIS_REVIEW_MANIFEST.json"
RECIPE_LOCK_PATH = CLOSURE / "INFERENCE_REPLAY_LOCK.json"
CKPT_LOCK_PATH = CLOSURE / "POSE_ARM_CHECKPOINT_LOCK.json"
POSE_CONTRACT_PATH = CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json"
CLOSURE_PREDICTIONS = CLOSURE / "predictions"

# ---------------------------------------------------------------- writable ---
TRACK = REPO_ROOT / "data/pallet/results/multiteacher_corner_distill_v1"
AUDIT = TRACK / "audit"
GATE_A = TRACK / "gate_a_teacher_headroom"
GATE_B = TRACK / "gate_b_corner_evidence"
GATE_C = TRACK / "gate_c_local_specialist"
GATE_D = TRACK / "gate_d_student_distill"
GATE_E = TRACK / "gate_e_domain_adapter"
FINAL = TRACK / "final"
LOGS = TRACK / "logs"
PREDICTIONS = TRACK / "predictions"
METHOD_LOCK_PATH = TRACK / "METHOD_LOCK.json"

# ------------------------------------------------------------- cuboid graph --
# camera-facing 0123 v4.  AXIS_REVIEW_MANIFEST.axis_definition 과 일치한다.
#   near face 0-1-2-3, far face 4-5-6-7, {0,1,4,5} 위 / {2,3,6,7} 아래, 8 centroid
AXIS_A_EDGES = ((0, 1), (2, 3), (4, 5), (6, 7))      # camera-facing WIDTH
AXIS_B_EDGES = ((0, 4), (1, 5), (2, 6), (3, 7))      # camera-facing DEPTH
VERTICAL_EDGES = ((0, 3), (1, 2), (4, 7), (5, 6))    # height
CUBOID_EDGES = AXIS_A_EDGES + AXIS_B_EDGES + VERTICAL_EDGES
SPACE_DIAGONALS = ((0, 6), (1, 7), (2, 4), (3, 5))
N_CORNERS = 8
CENTROID_INDEX = 8

INCIDENT_EDGES = {c: tuple(e for e in CUBOID_EDGES if c in e) for c in range(N_CORNERS)}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -------------------------------------------------------------- population ---
def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def dev_eval_frames() -> list[dict]:
    """PAPER_EVAL positive 319, manifest 순서 그대로."""
    return load_manifest()["frames_list"]


def load_gt(frame: dict) -> dict:
    """한 프레임의 GT keypoint 계약.

    visibility 2 = manual_click / reason 'visible'  → 눈에 보이는 코너
    visibility 1 = 주석은 있으나 가려짐/불명       → supervised 이지만 visible 아님
    visibility 0 = supervision 없음 (truncated 등)
    """
    payload = json.loads((REPO_ROOT / frame["annotation"]).read_text())
    obj = payload["objects"][0]
    entries = obj["keypoint_annotations"]
    xy = np.full((9, 2), np.nan, dtype=np.float64)
    vis = np.zeros(9, dtype=np.int64)
    in_frame = np.zeros(9, dtype=bool)
    for i, e in enumerate(entries[:9]):
        if e.get("xy") is not None:
            xy[i] = np.asarray(e["xy"], dtype=np.float64)
        vis[i] = int(e.get("visibility", 0))
        in_frame[i] = bool(e.get("in_frame", False))
    intr = payload["camera_data"]["intrinsics"]
    camera = np.array([[intr["fx"], 0.0, intr["cx"]],
                       [0.0, intr["fy"], intr["cy"]],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    return {
        "frame_id": frame["frame_id"],
        "session_id": frame["session_id"],
        "object_type": frame["object_type"],
        "paper_domain": frame["paper_domain"],
        "image": frame["image"],
        "xy": xy,
        "visibility": vis,
        "in_frame": in_frame,
        "supervised": (vis > 0) & np.isfinite(xy).all(axis=1),
        "visible": (vis == 2) & np.isfinite(xy).all(axis=1),
        "camera": camera,
        "image_size": (int(payload["camera_data"]["width"]),
                       int(payload["camera_data"]["height"])),
    }


# ----------------------------------------------------------------- metrics ---
GROSS_THRESHOLDS_PX = (20.0, 40.0)


def error_stats(errors) -> dict:
    """pooled keypoint 오차 분포.

    median/p90 은 `challenge/evaluation_v2/paper_real_eval.py::_distribution` 과
    같은 정의(pooled over keypoints, np.median / np.percentile 90)다.
    gross20/gross40 은 같은 pooled 배열 위에서 정의한 tail 비율이고,
    METHOD_LOCK 에 고정한 뒤 바꾸지 않는다.
    """
    a = np.asarray(list(errors), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "median_px": None, "p90_px": None,
                "gross20": None, "gross40": None, "mean_px": None}
    return {
        "n": int(a.size),
        "median_px": float(np.median(a)),
        "p90_px": float(np.percentile(a, 90)),
        "mean_px": float(a.mean()),
        "gross20": float(np.mean(a > GROSS_THRESHOLDS_PX[0])),
        "gross40": float(np.mean(a > GROSS_THRESHOLDS_PX[1])),
    }


def keypoint_errors(pred_xy, gt_xy, mask) -> np.ndarray:
    """마스크된 keypoint 의 유클리드 오차 (px, 원본 이미지 좌표)."""
    p = np.asarray(pred_xy, dtype=np.float64)
    g = np.asarray(gt_xy, dtype=np.float64)
    d = np.linalg.norm(p - g, axis=1)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(d)
    return d[ok]


# ------------------------------------------------------------- predictions ---
def load_prediction_file(path) -> dict:
    payload = json.loads(Path(path).read_text())
    return payload["frames"]


def prediction_keypoints(entry):
    if not entry or entry.get("status") != "OK" or not entry.get("keypoints_xy"):
        return None
    return np.asarray(entry["keypoints_xy"], dtype=np.float64)


# --------------------------------------------------------------- bootstrap ---
def bootstrap_paired(values_a, values_b, groups=None, n_boot: int = 10000,
                     seed: int = 20260905, statistic=np.median) -> dict:
    """짝지은 두 배열의 통계량 차이에 대한 CI.

    groups 를 주면 그 단위로 재표집한다(session-cluster bootstrap).
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    n = a.size
    if n == 0:
        return {"n": 0}
    observed = float(statistic(b) - statistic(a))
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        draws = np.array([float(statistic(b[i]) - statistic(a[i])) for i in idx])
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in uniq]
        draws = np.empty(n_boot, dtype=np.float64)
        for k in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            i = np.concatenate([members[j] for j in pick])
            draws[k] = float(statistic(b[i]) - statistic(a[i]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n": int(n),
        "observed_delta": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "unit": "session_cluster" if groups is not None else "frame",
    }


# --------------------------------------------------------- 2D report blocks --
def gt_corner_box(gt: dict) -> np.ndarray:
    """GT 코너 8개의 축정렬 외접상자 — evaluator 의 box 정의와 같다."""
    c = gt["xy"][:N_CORNERS]
    return np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()],
                    dtype=np.float64)


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


MATCH_IOU = 0.5


def arm_2d_report(frames_pred: dict, gts: list[dict]) -> dict:
    """한 arm 의 2D 보고 블록.

    POOLED_ALL        검출된 모든 프레임
    POOLED_MATCHED50  top-1 상자가 GT 코너 외접상자와 IoU>=0.5 인 프레임
                      (`paper_real_eval` 헤드라인과 같은 정의 — R0 로 자릿수 일치 확인)
    """
    buckets = {"POOLED_ALL": {"supervised": [], "visible": [], "corners": [],
                              "centroid": [], "by_index": {i: [] for i in range(9)}},
               "POOLED_MATCHED50": {"supervised": [], "visible": [], "corners": [],
                                    "centroid": [], "by_index": {i: [] for i in range(9)}}}
    n_pred = n_matched = 0
    for gt in gts:
        entry = frames_pred.get(gt["frame_id"])
        pred = prediction_keypoints(entry)
        if pred is None:
            continue
        n_pred += 1
        d = np.linalg.norm(pred - gt["xy"], axis=1)
        matched = box_iou(entry["box_xyxy"], gt_corner_box(gt)) >= MATCH_IOU
        n_matched += int(matched)
        keys = ["POOLED_ALL"] + (["POOLED_MATCHED50"] if matched else [])
        for key in keys:
            b = buckets[key]
            sup = gt["supervised"] & np.isfinite(d)
            b["supervised"] += list(d[sup])
            b["visible"] += list(d[gt["visible"] & np.isfinite(d)])
            b["corners"] += list(d[:N_CORNERS][sup[:N_CORNERS]])
            if sup[CENTROID_INDEX]:
                b["centroid"].append(float(d[CENTROID_INDEX]))
            for i in range(9):
                if sup[i]:
                    b["by_index"][i].append(float(d[i]))
    out = {"n_frames_total": len(gts), "n_frames_detected": n_pred,
           "n_frames_matched50": n_matched,
           "detection_coverage": n_pred / len(gts) if gts else 0.0}
    for key, b in buckets.items():
        out[key] = {name: error_stats(b[name])
                    for name in ("supervised", "visible", "corners", "centroid")}
        out[key]["by_index"] = {str(i): error_stats(b["by_index"][i]) for i in range(9)}
    return out
