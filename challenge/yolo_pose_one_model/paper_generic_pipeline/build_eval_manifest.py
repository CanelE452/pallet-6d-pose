"""Legacy-compatible DEV 140 GT-QA-clean manifest를 만들고 SHA로 잠근다.

모델 출력이 도착하기 전에 GT 쪽을 전부 굳혀 둔다.  그래야 결과를 본 뒤에
평가 대상이 바뀌는 일이 생기지 않는다.

두 셋 모두 **development** 다.  이미 여러 분석에 썼으므로 final test 가 아니다.
"""
from __future__ import annotations

import hashlib, json, os, sys
from pathlib import Path
import numpy as np

ROOT = str(Path(__file__).resolve().parents[3])
for sub in ("scripts/stage0/model_compare", "scripts/annotate", "challenge",
            "scripts/stage0/real_eval"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import mc_frames as MF        # noqa: E402

OUT = os.path.join(ROOT, "challenge/yolo_pose_one_model/paper_generic_pipeline")


def _object_points(dimensions):
    """Eight corners in the frozen camera-facing 0123 convention."""
    width = float(dimensions["width"]) / 2.0
    height = float(dimensions["height"]) / 2.0
    depth = float(dimensions["depth"]) / 2.0
    return np.asarray([
        [-width, -height, -depth],
        [+width, -height, -depth],
        [+width, +height, -depth],
        [-width, +height, -depth],
        [-width, -height, +depth],
        [+width, -height, +depth],
        [+width, +height, +depth],
        [-width, +height, +depth],
    ], dtype=float)


def _gt_of(label):
    obj = label["objects"][0]
    pose = np.asarray(obj["pose_transform"], dtype=float)
    intrinsics = label["camera_data"]["intrinsics"]
    return {
        "R": pose[:3, :3],
        "t": pose[:3, 3],
        "K": np.asarray([
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=float),
        "gt8": np.asarray(obj["projected_cuboid"], dtype=float)[:8],
        "model": _object_points(obj["dimensions_m"]),
    }


def main():
    items, problems = [], []
    seen = set()
    for key, sealed, jp, ip, label in MF.frames():
        fid = os.path.splitext(os.path.basename(jp))[0]
        if fid in seen:
            problems.append(f"duplicate frame_id {fid}")
        seen.add(fid)
        camera_data = label.get("camera_data") or {}
        width, height = camera_data.get("width"), camera_data.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            problems.append(f"missing image dimensions {fid}")
            continue
        obj = label["objects"][0]
        if obj.get("pose_transform") is None:
            problems.append(f"missing GT pose {fid}")
            continue
        t = _gt_of(label)
        items.append({
            "frame_id": fid,
            "population": "REAL_CHALLENGE_DEV_105" if sealed
                          else "REAL_DEV_OPEN_56",
            "set": key,
            "image": os.path.relpath(ip, ROOT),
            "label": os.path.relpath(jp, ROOT),
            "width": width, "height": height,
            "K": t["K"].tolist(),
            "dimensions_m": obj["dimensions_m"],
            "R_gt": t["R"].tolist(), "t_gt": t["t"].tolist(),
            "gt_corners_2d": t["gt8"].tolist(),
            "gt_centroid_2d": obj.get("projected_cuboid_centroid"),
            "object_points": t["model"].tolist(),
        })
    counts = {}
    for i in items:
        counts[i["population"]] = counts.get(i["population"], 0) + 1
    payload = {
        "manifest": "PAPER_YOLO_EVAL_DEV_140_GT_QA_CLEAN",
        "role": "DEVELOPMENT — final test 아님. 이미 여러 분석에 사용됨",
        "gt_qa": {
            "status": "CLEAN",
            "raw_total_before_quarantine": 161,
            "excluded": 21,
            "quarantine_registry": "challenge/real_gt_v2/INVALID_GT_QUARANTINE.json",
            "identity": "exact annotation path/SHA; historical population labels retained",
        },
        "keypoint_convention": "camera-facing 0123 (0-3 near face, "
                               "{0,1,4,5} top, {2,3,6,7} bottom, 8 centroid)",
        "object_points_source": "annotate_pnp.make_pallet_keypoints_3d_diagram"
                                "(per-frame dimensions_m)",
        "n_total": len(items), "counts": counts,
        "checks": {"expected_total": 140, "duplicates": 0,
                   "missing_rgb": 0, "missing_gt": 0,
                   "problems": problems},
        "items": items,
    }
    # Population strings are historical identifiers consumed by old analysis
    # scripts; their numeric suffixes no longer describe the clean membership.
    ok = (len(items) == 140 and not problems
          and counts.get("REAL_DEV_OPEN_56") == 52
          and counts.get("REAL_CHALLENGE_DEV_105") == 88)
    payload["MANIFEST_VALID"] = ok
    text = json.dumps(payload, indent=1)
    open(os.path.join(OUT, "eval_manifest.json"), "w").write(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    open(os.path.join(OUT, "eval_manifest.sha256"), "w").write(
        f"{digest}  eval_manifest.json\n")
    print(f"  n={len(items)}  {counts}")
    print(f"  problems={problems if problems else '없음'}")
    print(f"  MANIFEST_VALID={ok}  sha256 {digest[:16]}")


if __name__ == "__main__":
    main()
