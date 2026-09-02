"""§13 — adaptation pool 에서 복원이 실제로 얼마나 일어나는지 GT 없이 센다.

이 단계에서 GT 를 쓰지 않는다.  복원이 **가능한지** 와 **얼마나 움직이는지** 만 본다.
후보가 거의 없으면 V4 는 애초에 검정력이 없다 — 그걸 먼저 확인한다.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v2"))

from keypoint_scores import per_keypoint_scores  # noqa: E402
from geometry_repair import (  # noqa: E402
    HIGH_CONFIDENCE, REPAIR_CANDIDATE, UNRELIABLE, REPAIR_OK,
    N_CORNERS, repair_keypoints,
)

V1_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v1"
V4_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v4"
CACHE = V1_RESULTS / "teacher_cache/R0_TEACHER_CACHE.json"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
OUT = V4_RESULTS / "V4_REPAIR_POOL_AUDIT.json"

POOL_OBJECT_TYPE = "plastic_standard_110x130x11"
BOX_CONF = 0.85
KP_FLOOR = 0.5
KP_HIGH_CONF = 0.95
TAU = 0.05
AMBIGUITY_Q = 0.75


def pool_dimensions() -> dict:
    for entry in json.loads(REGISTRY.read_text())["objects"]:
        if entry["object_type"] == POOL_OBJECT_TYPE:
            dims = entry["physical_dimensions_m"]
            return {axis: float(dims[axis]) for axis in ("x", "y", "z")}
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {POOL_OBJECT_TYPE}")


def main() -> int:
    cache = json.loads(CACHE.read_text())
    dimensions = pool_dimensions()

    per_condition: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    displacement: dict[str, list[float]] = collections.defaultdict(list)
    displacement_normalised: dict[str, list[float]] = collections.defaultdict(list)
    anchors_available: dict[str, list[int]] = collections.defaultdict(list)
    frames = 0

    for index, entry in enumerate(cache["entries"]):
        top = entry.get("top1")
        if not top or float(top["box_conf"]) < BOX_CONF:
            continue
        frames += 1
        condition = entry.get("paper_condition") or "unknown"
        flip = entry.get("flip_top1") or {}
        keypoints = np.asarray(top["keypoints_xy"], dtype=float)
        confidence = np.asarray(top["keypoints_conf"], dtype=float)
        camera = np.asarray(entry["camera_matrix"], dtype=float)
        scores = per_keypoint_scores(
            keypoints, confidence, camera, dimensions,
            flip_keypoints_2d=(np.asarray(flip["keypoints_xy"], dtype=float)
                               if flip else None),
            flip_conf=(np.asarray(flip["keypoints_conf"], dtype=float)
                       if flip else None),
            kp_conf_threshold=KP_FLOOR, remove_threshold=TAU,
            flip_threshold=TAU, ambiguity_threshold=AMBIGUITY_Q)
        repair = repair_keypoints(
            keypoints, confidence, camera, dimensions, scores,
            (float(entry["image_width"]), float(entry["image_height"])),
            kp_high_conf=KP_HIGH_CONF, kp_floor=KP_FLOOR, tau_reproj=TAU)

        counter = per_condition[condition]
        counter["frames"] += 1
        counter["ambiguous_frames"] += int(repair["ambiguous_view"])
        anchors_available[condition].append(repair["n_anchor"])
        for corner in range(N_CORNERS):
            counter[f"tier_{repair['tier'][corner]}"] += 1
            status = repair["repair_status"][corner]
            if status is not None:
                counter[f"repair_{status}"] += 1
            if status == REPAIR_OK:
                displacement[condition].append(repair["displacement_px"][corner])
                value = repair["displacement_normalised"][corner]
                if value is not None and np.isfinite(value):
                    displacement_normalised[condition].append(float(value))
        if (index + 1) % 200 == 0:
            print(f"  {index + 1}/{len(cache['entries'])}", flush=True)

    def summarise(values) -> dict:
        array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if array.size == 0:
            return {"n": 0, "median": None, "p90": None, "max": None}
        return {"n": int(array.size), "median": float(np.median(array)),
                "p90": float(np.percentile(array, 90)), "max": float(array.max())}

    report = {
        "schema_version": "v4_repair_pool_audit_v1",
        "gt_used": False,
        "thresholds": {"box_conf": BOX_CONF, "kp_floor": KP_FLOOR,
                       "kp_high_conf": KP_HIGH_CONF, "tau": TAU,
                       "ambiguity_q": AMBIGUITY_Q, "min_anchors": 6},
        "box_accepted_frames": frames,
        "by_condition": {
            condition: {
                **dict(counter),
                "anchors": summarise(anchors_available[condition]),
                "displacement_px": summarise(displacement[condition]),
                "displacement_normalised": summarise(
                    displacement_normalised[condition]),
            }
            for condition, counter in sorted(per_condition.items())
        },
    }
    V4_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"\nbox-accepted frames {frames}")
    print(f"{'condition':11} {'frames':>7} {'high':>6} {'cand':>6} {'low':>5} "
          f"{'repaired':>9} {'no_anch':>8} {'no_hyp':>7} {'disagree':>9} {'ambig':>6}")
    print("-" * 88)
    for condition, block in report["by_condition"].items():
        print(f"{condition:11} {block['frames']:7d} "
              f"{block.get('tier_HIGH_CONFIDENCE', 0):6d} "
              f"{block.get('tier_REPAIR_CANDIDATE', 0):6d} "
              f"{block.get('tier_UNRELIABLE', 0):5d} "
              f"{block.get('repair_REPAIRED', 0):9d} "
              f"{block.get('repair_NO_ANCHORS', 0):8d} "
              f"{block.get('repair_NO_VALID_HYPOTHESIS', 0):7d} "
              f"{block.get('repair_HYPOTHESIS_DISAGREE', 0):9d} "
              f"{block.get('repair_AMBIGUOUS_VIEW', 0):6d}")
    print()
    for condition, block in report["by_condition"].items():
        d = block["displacement_px"]
        n = block["displacement_normalised"]
        print(f"{condition:11} 복원 이동량  n={d['n']:4d}  "
              f"median {d['median'] if d['median'] is None else round(d['median'], 2)} px"
              f"  p90 {d['p90'] if d['p90'] is None else round(d['p90'], 2)} px"
              f"  정규화 median "
              f"{n['median'] if n['median'] is None else round(n['median'], 4)}")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
