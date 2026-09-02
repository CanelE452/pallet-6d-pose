"""복원이 좌표를 어디로 옮겼는지 그린다.  집계로는 안 보이는 것을 본다.

세 종류를 한 장에 담는다.

    REPAIRED              복원에 성공한 코너 (GT 로 채점된 것 포함)
    HYPOTHESIS_DISAGREE   W/D hypothesis 들이 2D 위치에 동의하지 않아 버린 코너
    AMBIGUOUS_VIEW        q >= 0.75 라 애초에 복원하지 않는 프레임

오버레이는 ASCII 만 쓴다 — OpenCV putText 는 한글을 `???` 로 찍는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v2"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from eval_workspace import load_frames, evaluation_population_views  # noqa: E402
from keypoint_scores import per_keypoint_scores  # noqa: E402
from geometry_repair import (  # noqa: E402
    N_CORNERS, REPAIR_OK, repair_keypoints,
)
from pseudo_label_filters import _project, _solve, registry_hypotheses  # noqa: E402

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
V4 = REPO_ROOT / "data/pallet/results/paper_selftrain_v4"
CACHE = V4 / "V4_PROXY_TEACHER_CACHE.json"
LOCK = V4 / "SELFTRAIN_V4_METHOD_LOCK.json"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
OUT = V4 / "repair_visual"

REGISTRY_NAME = {"plastic": "plastic_standard_110x130x11",
                 "wood": "wood_small_80x59x14"}
GREEN, BLUE, RED, YELLOW, GREY = ((0, 220, 0), (255, 170, 0), (0, 0, 255),
                                  (0, 220, 255), (160, 160, 160))
CELL = (620, 470)


def registry_dimensions(object_type: str) -> dict:
    name = REGISTRY_NAME.get(object_type, object_type)
    for entry in json.loads(REGISTRY.read_text())["objects"]:
        if entry["object_type"] == name:
            dims = entry["physical_dimensions_m"]
            return {axis: float(dims[axis]) for axis in ("x", "y", "z")}
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {object_type}")


def canonical(frame_id: str) -> str:
    return frame_id.replace("__", ":")


def hypothesis_projections(keypoints, anchors, camera, dimensions, index):
    """각 hypothesis 가 candidate 를 어디로 보내는지 — 불일치를 눈으로 보려고."""

    points = []
    for name, keypoints_3d in registry_hypotheses(dimensions):
        solved = _solve(keypoints_3d[anchors], keypoints[anchors], camera)
        if solved is None:
            continue
        projected = _project(keypoints_3d[index].reshape(1, 3), solved[0], solved[1],
                             camera)[0]
        if np.isfinite(projected).all():
            points.append((name, projected))
    return points


def banner(canvas, lines):
    y = 16
    for line in lines:
        cv2.putText(canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 15


def draw(image, gt, raw, index, repaired, hypotheses, anchors, caption):
    canvas = image.copy()
    for corner in range(N_CORNERS):
        if corner in anchors:
            cv2.circle(canvas, tuple(np.round(raw[corner]).astype(int)), 6, GREY, 1,
                       cv2.LINE_AA)
    cv2.circle(canvas, tuple(np.round(gt[index]).astype(int)), 6, GREEN, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(np.round(raw[index]).astype(int)), 5, BLUE, -1, cv2.LINE_AA)
    cv2.line(canvas, tuple(np.round(raw[index]).astype(int)),
             tuple(np.round(gt[index]).astype(int)), BLUE, 1, cv2.LINE_AA)
    if repaired is not None and np.isfinite(repaired).all():
        point = tuple(np.round(repaired).astype(int))
        cv2.circle(canvas, point, 5, RED, -1, cv2.LINE_AA)
        cv2.line(canvas, point, tuple(np.round(gt[index]).astype(int)), RED, 1,
                 cv2.LINE_AA)
    for _, point in hypotheses:
        if np.isfinite(point).all():
            cv2.drawMarker(canvas, tuple(np.round(point).astype(int)), YELLOW,
                           cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA)

    height, width = canvas.shape[:2]
    scale = min(CELL[0] / width, CELL[1] / height)
    canvas = cv2.resize(canvas, (int(width * scale), int(height * scale)))
    cell = np.zeros((CELL[1], CELL[0], 3), dtype=np.uint8)
    cell[: canvas.shape[0], : canvas.shape[1]] = canvas
    banner(cell, caption)
    cv2.rectangle(cell, (0, 0), (CELL[0] - 1, CELL[1] - 1), GREY, 1)
    return cell


def main() -> int:
    lock = json.loads(LOCK.read_text())
    reused = lock["thresholds_reused_unchanged"]
    kp_high = float(lock["new_threshold_this_track"]["KP_HIGH_CONF"])
    cache = json.loads(CACHE.read_text())
    rows = {canonical(r["frame_id"]): r for r in
            evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]}

    wanted = {REPAIR_OK: 4, "HYPOTHESIS_DISAGREE": 4, "AMBIGUOUS_VIEW": 4}
    picked: list = []

    for frame, row in rows.items():
        if all(count == 0 for count in wanted.values()):
            break
        entry = cache.get(frame)
        if not entry or not entry.get("top1"):
            continue
        top = entry["top1"]
        if float(top["box_conf"]) < float(reused["box_conf"]):
            continue
        payload = json.loads((WORKSPACE / row["annotation_path"]).read_text())
        points = payload["objects"][0]["keypoint_annotations"]
        gt = np.array([p["xy"] if p.get("xy") else [np.nan, np.nan] for p in points],
                      dtype=float)
        if not np.isfinite(gt[:8]).all():
            continue
        dimensions = registry_dimensions(row["object_type"])
        intrinsics = payload["camera_data"]["intrinsics"]
        camera = np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                           [0.0, intrinsics["fy"], intrinsics["cy"]],
                           [0.0, 0.0, 1.0]], dtype=float)
        keypoints = np.asarray(top["keypoints_xy"], dtype=float)
        confidence = np.asarray(top["keypoints_conf"], dtype=float)
        flip = entry.get("flip_top1") or {}
        scores = per_keypoint_scores(
            keypoints, confidence, camera, dimensions,
            flip_keypoints_2d=(np.asarray(flip["keypoints_xy"], dtype=float)
                               if flip else None),
            flip_conf=(np.asarray(flip["keypoints_conf"], dtype=float)
                       if flip else None),
            kp_conf_threshold=float(reused["kp_conf_floor"]),
            remove_threshold=float(reused["tau_remove"]),
            flip_threshold=float(reused["tau_flip"]),
            ambiguity_threshold=float(reused["ambiguity_q"]))
        repair = repair_keypoints(
            keypoints, confidence, camera, dimensions, scores,
            (float(entry["image_width"]), float(entry["image_height"])),
            kp_high_conf=kp_high, kp_floor=float(reused["kp_conf_floor"]),
            tau_reproj=float(reused["tau_reproj"]))

        anchors = np.flatnonzero(np.asarray(repair["anchor"]))
        for corner in range(N_CORNERS):
            status = repair["repair_status"][corner]
            if status is None or wanted.get(status, 0) == 0:
                continue
            image = cv2.imread(str(WORKSPACE / row["image_path"]))
            if image is None:
                continue
            others = anchors[anchors != corner]
            hypotheses = (hypothesis_projections(keypoints, others, camera, dimensions,
                                                 corner)
                          if len(others) >= 6 else [])
            repaired = (np.asarray(repair["repaired_xy"][corner])
                        if status == REPAIR_OK else None)
            spread = ""
            if len(hypotheses) > 1:
                stacked = np.asarray([p for _, p in hypotheses])
                value = float(np.linalg.norm(
                    stacked[:, None, :] - stacked[None, :, :], axis=-1).max())
                spread = f"  hyp spread {value:.1f}px = {value / repair['projected_diagonal_px']:.3f} diag"
            caption = [
                f"{status}   {frame}  kp{corner}",
                f"teacher conf {confidence[corner]:.3f}   anchors {len(others)}"
                f"   q {'-' if scores['q'] is None else round(scores['q'], 3)}",
                f"raw->GT {np.linalg.norm(keypoints[corner] - gt[corner]):.1f}px"
                + (f"   repaired->GT {np.linalg.norm(repaired - gt[corner]):.1f}px"
                   if repaired is not None else "")
                + spread,
                "GT green | teacher blue | repaired red | hypotheses yellow x",
            ]
            picked.append(draw(image, gt, keypoints, corner, repaired, hypotheses,
                               set(others.tolist()), caption))
            wanted[status] -= 1
            break

    if not picked:
        raise SystemExit("NOTHING_TO_DRAW")
    columns = 2
    rows_count = (len(picked) + columns - 1) // columns
    sheet = np.zeros((rows_count * CELL[1] + 28, columns * CELL[0], 3), dtype=np.uint8)
    cv2.putText(sheet, "V4 geometry repair - what it actually did", (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for index, cell in enumerate(picked):
        r, c = divmod(index, columns)
        sheet[28 + r * CELL[1]: 28 + (r + 1) * CELL[1],
              c * CELL[0]: (c + 1) * CELL[0]] = cell
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "V4_REPAIR_CASES.jpg"
    cv2.imwrite(str(path), sheet)
    print(f"wrote {path.relative_to(REPO_ROOT)}  ({len(picked)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
