"""GT_AXIS_RESOLUTION_LOCK 그대로 6D GT 를 만든다.

    python3 scripts/paper/pose_metric_closure_v1/build_geometry_resolved_pose_gt.py \
        --pose-object-contract data/pallet/results/paper_pose_metric_closure_v1/POSE_EVAL_OBJECT_CONTRACT.json

출력: data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json

**모델 예측을 일절 읽지 않는다.**  입력은 사람 어노 키포인트 · intrinsics · 등록된
물리 치수 · object type 뿐이다.  체크포인트를 열지 않고 추론하지 않는다.

축 결정은 lock 의 규칙 그대로 — 두 parity 를 각각 SQPnP + RefineLM 으로 풀고 GT
키포인트에 대한 재투영 잔차가 작은 쪽을 고른다.  180도 부호는 등가류로 남긴다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_DIR = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
MANIFEST = OUT_DIR / "AXIS_REVIEW_MANIFEST.json"
LOCK = OUT_DIR / "GT_AXIS_RESOLUTION_LOCK.json"
OUT = OUT_DIR / "GEOMETRY_RESOLVED_POSE_GT.json"

CF_WIDTH, CF_DEPTH = "CF_WIDTH", "CF_DEPTH"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cuboid(across: float, height: float, along: float) -> np.ndarray:
    """camera-facing 0123 순서의 8 코너."""

    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
    return np.array([
        [-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
        [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb],
    ], dtype=np.float64)


def solve(model: np.ndarray, points: np.ndarray, camera: np.ndarray,
          usable: np.ndarray):
    """lock 이 지정한 solver — SQPnP 뒤 RefineLM, 두 가설에 동일 적용."""

    ok, rvec, tvec = cv2.solvePnP(model[usable], points[usable], camera, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(model[usable], points[usable], camera, None,
                                      rvec, tvec)
    projected, _ = cv2.projectPoints(model, rvec, tvec, camera, None)
    residual = float(np.linalg.norm(
        projected.reshape(-1, 2)[usable] - points[usable], axis=1).mean())
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, tvec.reshape(-1), residual


def elevation_deg(rotation: np.ndarray, translation: np.ndarray) -> float:
    """팔레트 상면 법선과 시선 사이 각으로 본 앙각.  진단용이며 선택에 쓰이지 않는다."""

    normal = rotation @ np.array([0.0, -1.0, 0.0])
    look = translation / max(float(np.linalg.norm(translation)), 1e-9)
    return float(abs(90.0 - np.degrees(np.arccos(np.clip(abs(normal @ look), -1, 1)))))


def main() -> int:
    from pose_evaluation_paths import build_argument_parser, load_pose_object_contract, object_spec

    args = build_argument_parser(__doc__).parse_args()
    contract = load_pose_object_contract(args.pose_object_contract)
    lock = json.loads(LOCK.read_text())
    if lock.get("status") != "FROZEN":
        raise SystemExit("GT_AXIS_RESOLUTION_LOCK is not FROZEN — refusing to build")
    quality_px = float(lock["quality_condition"]["threshold_px"])

    frames = json.loads(MANIFEST.read_text())["frames_list"]
    built: dict[str, dict] = {}
    unresolved: list[dict] = []

    for frame in frames:
        payload = json.loads((REPO_ROOT / frame["annotation"]).read_text())
        obj = payload["objects"][0]
        raw = payload.get("camera_data", {}).get("intrinsics")
        if not raw:
            unresolved.append({"frame_id": frame["frame_id"], "reason": "no intrinsics"})
            continue
        camera = np.array([[raw["fx"], 0.0, raw["cx"]],
                           [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]], np.float64)
        spec = object_spec(contract, frame["object_type"])
        long_m, short_m, height_m = spec["long_m"], spec["short_m"], spec["height_m"]

        points = np.array([p if p else [np.nan, np.nan]
                           for p in frame["keypoints_xy"]], np.float64)[:8]
        usable = np.isfinite(points).all(axis=1)
        if usable.sum() < 6:
            unresolved.append({"frame_id": frame["frame_id"],
                               "reason": f"only {int(usable.sum())} usable corners"})
            continue

        # hypothesis A = camera-facing WIDTH is long ; B = camera-facing DEPTH is long
        solved = {}
        for name, (across, along) in ((CF_WIDTH, (long_m, short_m)),
                                      (CF_DEPTH, (short_m, long_m))):
            result = solve(cuboid(across, height_m, along), points, camera, usable)
            if result is None:
                break
            solved[name] = result
        if len(solved) != 2:
            unresolved.append({"frame_id": frame["frame_id"], "reason": "SQPnP failed"})
            continue

        chosen = min(solved, key=lambda k: solved[k][2])
        other = CF_DEPTH if chosen == CF_WIDTH else CF_WIDTH
        rotation, translation, residual = solved[chosen]
        alternative = solved[other][2]
        across, along = ((long_m, short_m) if chosen == CF_WIDTH
                         else (short_m, long_m))
        stored = obj.get("reproj_error_px")

        built[frame["frame_id"]] = {
            "frame_id": frame["frame_id"],
            "session_id": frame["session_id"],
            "paper_domain": frame["paper_domain"],
            "object_type": frame["object_type"],
            "axis_resolution_source": "manual_gt_keypoints_plus_known_geometry",
            "physical_long_axis": chosen,
            "axis_sign": "180_DEG_EQUIVALENCE_CLASS",
            "R_gt_representative": rotation.tolist(),
            "t_gt": translation.tolist(),
            "hypothesis_A_reproj_px": solved[CF_WIDTH][2],
            "hypothesis_B_reproj_px": solved[CF_DEPTH][2],
            "chosen_reproj_px": residual,
            "alternative_reproj_px": alternative,
            "resolution_margin_px": alternative - residual,
            "resolution_ratio": alternative / max(residual, 1e-9),
            "resolved": True,
            "physical_dimensions_m": {"across": across, "height": height_m,
                                      "along": along},
            "usable_corners": int(usable.sum()),
            "stored_annotation_reproj_px": stored,
            "meets_annotation_quality_bar": (
                None if not isinstance(stored, (int, float)) else stored < quality_px),
            "elevation_deg": elevation_deg(rotation, translation),
        }

    for entry in unresolved:
        entry["resolved"] = False

    report = {
        "schema_version": "geometry_resolved_pose_gt_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "GT_AXIS_RESOLUTION_LOCK.json",
        "rule_sha256": sha256_file(LOCK),
        "pose_object_contract": str(
            Path(args.pose_object_contract).resolve().relative_to(REPO_ROOT)),
        "pose_object_contract_sha256": contract["contract_sha256"],
        "model_predictions_used": False,
        "source_annotations_modified": False,
        "axis_resolution_source": "manual_gt_keypoints_plus_known_geometry",
        "symmetry_group_yaw_degrees": [0, 180],
        "total": len(frames),
        "resolved": len(built),
        "unresolved": len(unresolved),
        "unresolved_detail": unresolved,
        "frames": built,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    residuals = np.array([v["chosen_reproj_px"] for v in built.values()])
    margins = np.array([v["resolution_margin_px"] for v in built.values()])
    print(f"resolved   {len(built)} / {len(frames)}")
    print(f"unresolved {len(unresolved)}")
    print(f"chosen reproj px   median {np.median(residuals):.2f}  "
          f"p90 {np.percentile(residuals, 90):.2f}  p95 {np.percentile(residuals, 95):.2f}  "
          f"max {residuals.max():.2f}")
    print(f"margin px          median {np.median(margins):.2f}  "
          f"min {margins.min():.2f}")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
