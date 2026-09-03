"""사람이 확정한 물리 긴축으로 평가용 GT pose 를 만든다.

    python3 scripts/paper/pose_metric_closure_v1/build_reviewed_pose_gt.py

출력: data/pallet/results/paper_pose_metric_closure_v1/REVIEWED_POSE_GT.json

**원본 annotation 은 수정하지 않는다.**  사람 판정은 sidecar 에 있고 이 스크립트는
둘을 합쳐 새 파일 하나를 만든다.

왜 저장된 `pose_transform` 을 그대로 쓰지 않는가
    저장된 pose 는 `axis_assignment_confirmed = False` 상태에서 맞춰진 것이라 장단축이
    뒤바뀌어 있을 수 있다 — 319/319 가 미확인이었다.  그래서 사람이 확정한 치수 배정으로
    사람이 찍은 2D 키포인트에서 pose 를 다시 푼다.  입력은 GT 키포인트·intrinsics·
    확정된 물리 치수뿐이고 모델 예측은 들어가지 않는다.

    비교를 위해 저장된 pose 와의 차이도 함께 기록한다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
MANIFEST = OUT_DIR / "AXIS_REVIEW_MANIFEST.json"
LABELS = OUT_DIR / "AXIS_REVIEW_LABELS.json"
REVIEWED_GT = OUT_DIR / "REVIEWED_POSE_GT.json"

CF_WIDTH, CF_DEPTH = "CF_WIDTH", "CF_DEPTH"


def cuboid(across: float, height: float, along: float) -> np.ndarray:
    """camera-facing 0123 순서의 8 코너."""

    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
    return np.array([
        [-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
        [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb],
    ], dtype=np.float64)


def extents_for(long_axis: str, long_m: float, short_m: float,
                height_m: float) -> tuple[float, float, float]:
    if long_axis == CF_WIDTH:
        return (long_m, height_m, short_m)
    if long_axis == CF_DEPTH:
        return (short_m, height_m, long_m)
    raise ValueError(f"unexpected long_axis {long_axis!r}")


def geodesic_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cosine = np.clip((np.trace(b.T @ a) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main() -> int:
    if not (MANIFEST.exists() and LABELS.exists()):
        print("manifest or labels missing")
        return 1
    frames = json.loads(MANIFEST.read_text())["frames_list"]
    labels = json.loads(LABELS.read_text())["frames"]

    built: dict[str, dict] = {}
    skipped: list[dict] = []
    stored_deltas: list[float] = []

    for frame in frames:
        entry = labels.get(frame["frame_id"])
        if entry is None or entry.get("status") != "CONFIRMED":
            skipped.append({"frame_id": frame["frame_id"],
                            "reason": "no confirmed human axis"})
            continue
        payload = json.loads((REPO_ROOT / frame["annotation"]).read_text())
        obj = payload["objects"][0]
        raw = payload.get("camera_data", {}).get("intrinsics")
        if not raw:
            skipped.append({"frame_id": frame["frame_id"], "reason": "no intrinsics"})
            continue
        camera = np.array([[raw["fx"], 0.0, raw["cx"]],
                           [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]], np.float64)

        points = np.array([p if p else [np.nan, np.nan]
                           for p in frame["keypoints_xy"]], np.float64)[:8]
        usable = np.isfinite(points).all(axis=1)
        if usable.sum() < 6:
            skipped.append({"frame_id": frame["frame_id"],
                            "reason": f"only {int(usable.sum())} usable keypoints"})
            continue

        dims = obj.get("physical_dimensions_m") or {}
        height_m = float(dims.get("y", 0.11))
        extents = extents_for(entry["long_axis"], frame["physical_long_m"],
                              frame["physical_short_m"], height_m)
        model = cuboid(*extents)

        ok, rvec, tvec = cv2.solvePnP(model[usable], points[usable], camera, None,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            skipped.append({"frame_id": frame["frame_id"], "reason": "SQPnP failed"})
            continue
        rvec, tvec = cv2.solvePnPRefineLM(model[usable], points[usable], camera,
                                          None, rvec, tvec)
        rotation, _ = cv2.Rodrigues(rvec)
        translation = tvec.reshape(-1)

        projected, _ = cv2.projectPoints(model, rvec, tvec, camera, None)
        residual = float(np.linalg.norm(
            projected.reshape(-1, 2)[usable] - points[usable], axis=1).mean())

        stored_delta = None
        stored = obj.get("pose_transform")
        if stored:
            matrix = np.asarray(stored, dtype=np.float64)
            if matrix.shape == (4, 4):
                stored_delta = geodesic_degrees(rotation, matrix[:3, :3])
                stored_deltas.append(stored_delta)

        built[frame["frame_id"]] = {
            "frame_id": frame["frame_id"],
            "session_id": frame["session_id"],
            "paper_domain": frame["paper_domain"],
            "object_type": frame["object_type"],
            "t_gt": translation.tolist(),
            "R_gt_representative": rotation.tolist(),
            "long_axis": entry["long_axis"],
            "short_axis": entry["short_axis"],
            "symmetry_group_yaw_degrees": [0, 180],
            "physical_dimensions_m": {"across": extents[0], "height": extents[1],
                                      "along": extents[2]},
            "physical_long_m": frame["physical_long_m"],
            "physical_short_m": frame["physical_short_m"],
            "human_axis_confirmed": True,
            "fit_residual_px": residual,
            "usable_keypoints": int(usable.sum()),
            "rotation_delta_vs_stored_pose_deg": stored_delta,
        }

    residuals = [r["fit_residual_px"] for r in built.values()]
    report = {
        "schema_version": "reviewed_pose_gt_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_annotations_modified": False,
        "pose_source":
            "re-solved from the human-annotated 2D keypoints with the human-confirmed "
            "physical dimension assignment. The stored pose_transform was fitted while "
            "axis_assignment_confirmed was False on all 319 frames, so it may carry the "
            "wrong long/short assignment.",
        "inputs": {
            "annotations": "data/evaluation/pallet_eval_v1 (read only)",
            "registry": "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json (read only)",
            "human_review": str(LABELS.relative_to(REPO_ROOT)),
        },
        "model_predictions_used": False,
        "symmetry_group_yaw_degrees": [0, 180],
        "symmetry_note":
            "orientation is defined as a 180-degree equivalence class, so the human was "
            "never asked to resolve the front/back sign. 90 degrees is NOT in the group.",
        "total_candidates": len(frames),
        "built": len(built),
        "skipped": len(skipped),
        "skipped_detail": skipped[:80],
        "fit_residual_px": {
            "median": float(np.median(residuals)) if residuals else None,
            "p90": float(np.percentile(residuals, 90)) if residuals else None,
            "max": float(np.max(residuals)) if residuals else None,
        },
        "rotation_delta_vs_stored_pose_deg": {
            "n": len(stored_deltas),
            "median": float(np.median(stored_deltas)) if stored_deltas else None,
            "above_45_deg": int(sum(1 for d in stored_deltas if d > 45.0)),
            "meaning": "how far the stored (unverified) pose sits from the reviewed one. "
                       "Values near 90 are frames whose stored axis assignment was wrong.",
        },
        "frames": built,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWED_GT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"built    {len(built)} / {len(frames)}")
    print(f"skipped  {len(skipped)}")
    if residuals:
        print(f"fit residual px  median {np.median(residuals):.2f}  "
              f"p90 {np.percentile(residuals, 90):.2f}  max {np.max(residuals):.2f}")
    if stored_deltas:
        print(f"rotation delta vs stored pose  median {np.median(stored_deltas):.2f} deg  "
              f">45 deg: {sum(1 for d in stored_deltas if d > 45.0)}")
    print(f"wrote {REVIEWED_GT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
