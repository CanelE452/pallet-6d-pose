"""§12 geometry convention gate — line 과 YOLO 가 같은 코너 색인을 쓰는가.

    python3 scripts/paper/fast6d_screen_v1b/check_geometry_parity.py \
        --output-dir data/pallet/results/paper_fast6d_screen_v1b

line branch 는 12 개 구조 edge 를 camera-facing 0123 위상으로 예측하고,
`mh_theta.theta_residual` 은 그 edge 를 **투영된 model point 색인**으로 짝짓는다.
그래서 YOLO 쪽 model point 순서가 line 쪽과 index-wise 로 같아야 한다.

세 가지를 확인한다.

    P1  두 model point 생성기가 좌표까지 동일한가 (해석적)
    P2  GT pose 로 투영한 model point 가 수동 어노 키포인트와 index-wise 로 맞는가
    P3  0~3 이 실제로 near face 인가 (camera z 비교)

object type 별로 따로 낸다.  wood 가 통과 못 하면 wood 는 line arm 에서 뺀다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CLOSURE = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
sys.path.insert(0, str(REPO_ROOT / "scripts/paper/pose_metric_closure_v1"))
sys.path.insert(0, str(REPO_ROOT / "scripts/annotate"))
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    audit = out_dir / "audit"
    audit.mkdir(parents=True, exist_ok=True)

    from symmetry_aware_pose_metrics import cuboid_model_points
    from annotate_pnp import make_pallet_keypoints_3d_diagram

    lock = json.loads((out_dir / "FAST_6D_SCREEN_V1B_LOCK.json").read_text())
    gate = lock["geometry_convention_gate"]
    threshold = float(gate["gate_threshold_px"])

    report = {"schema_version": "fast6d_v1b_geometry_gate_v1",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "gate_threshold_px": threshold}

    # P1 — 두 생성기가 같은 점을 같은 순서로 내는가
    p1 = {}
    for name, (across, height, along) in (
            ("plastic_standard_110x130x11", (1.10, 0.11, 1.30)),
            ("wood_small_80x59x14", (0.59, 0.14, 0.80)),
            ("swapped_axis_case", (1.30, 0.11, 1.10))):
        a = cuboid_model_points((across, height, along))
        b = make_pallet_keypoints_3d_diagram(width=across, depth=along, height=height)[:8]
        p1[name] = {"max_abs_difference_m": float(np.abs(a - b).max()),
                    "identical": bool(np.array_equal(a, b))}
    report["P1_model_point_generators_identical"] = p1

    # P2 / P3 — GT pose 투영이 수동 어노와 index-wise 로 맞는가
    gt_all = json.loads((CLOSURE / "GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]
    manifest = {f["frame_id"]: f for f in
                json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]}

    # 대조군: 90도 재라벨 순열 (near/far 유지, 좌우 회전) 과 near/far 뒤집기
    ROT90 = [1, 5, 6, 2, 0, 4, 7, 3]
    SWAP_NEAR_FAR = [4, 5, 6, 7, 0, 1, 2, 3]

    per_type: dict[str, dict] = {}
    for frame_id, truth in gt_all.items():
        frame = manifest[frame_id]
        object_type = frame["object_type"]
        annotation = json.loads((REPO_ROOT / frame["annotation"]).read_text())
        raw = annotation["camera_data"]["intrinsics"]
        camera = np.array([[raw["fx"], 0, raw["cx"]], [0, raw["fy"], raw["cy"]],
                           [0, 0, 1]], np.float64)
        dims = truth["physical_dimensions_m"]
        model = cuboid_model_points((dims["across"], dims["height"], dims["along"]))
        R = np.asarray(truth["R_gt_representative"], np.float64)
        t = np.asarray(truth["t_gt"], np.float64)
        camera_points = model @ R.T + t
        z = camera_points[:, 2]
        projected = np.stack([camera[0, 0] * camera_points[:, 0] / z + camera[0, 2],
                              camera[1, 1] * camera_points[:, 1] / z + camera[1, 2]], 1)
        manual = np.asarray(frame["keypoints_xy"], np.float64)[:8]
        if not np.isfinite(manual).all():
            continue
        bucket = per_type.setdefault(object_type, {"identity": [], "rot90": [],
                                                   "swap": [], "near_first": [],
                                                   "frames": 0})
        bucket["frames"] += 1
        bucket["identity"].append(float(np.median(np.linalg.norm(projected - manual, axis=1))))
        bucket["rot90"].append(float(np.median(np.linalg.norm(projected[ROT90] - manual, axis=1))))
        bucket["swap"].append(float(np.median(np.linalg.norm(projected[SWAP_NEAR_FAR] - manual, axis=1))))
        bucket["near_first"].append(bool(z[:4].mean() < z[4:].mean()))

    p2 = {}
    for object_type, bucket in sorted(per_type.items()):
        identity = np.array(bucket["identity"])
        p2[object_type] = {
            "frames": bucket["frames"],
            "identity_median_px": float(np.median(identity)),
            "identity_p90_px": float(np.percentile(identity, 90)),
            "control_rot90_median_px": float(np.median(bucket["rot90"])),
            "control_near_far_swap_median_px": float(np.median(bucket["swap"])),
            "P3_fraction_0123_is_near_face": float(np.mean(bucket["near_first"])),
            "PASS": bool(np.median(identity) < threshold
                         and np.median(identity) < np.median(bucket["rot90"])
                         and np.median(identity) < np.median(bucket["swap"])),
        }
    report["P2_index_wise_projection_parity"] = p2

    statuses = {}
    for object_type, block in p2.items():
        statuses[object_type] = "OK" if block["PASS"] else "BLOCKED_GEOMETRY_CONVENTION"
    report["object_status"] = statuses
    report["WOOD_LINE_STATUS"] = statuses.get("wood_small_80x59x14", "MISSING")
    report["PLASTIC_LINE_STATUS"] = statuses.get("plastic_standard_110x130x11", "MISSING")
    report["line_population"] = ("ALL" if all(v == "OK" for v in statuses.values())
                                 else "PLASTIC_ONLY" if statuses.get(
                                     "plastic_standard_110x130x11") == "OK" else "NONE")

    (audit / "GEOMETRY_CONVENTION_GATE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
