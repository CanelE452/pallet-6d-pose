"""GATE 0 — depth validity 와 pallet ROI depth support 를 센다.

    python3 scripts/self_training_yolo/depth_corrected/audit_depth_support.py \
        --output-dir data/pallet/results/paper_depth_selftrain_v1/gate0

출력  DEPTH_SUPPORT_PER_FRAME.csv · DEPTH_SUPPORT_SUMMARY.json

ROI 는 기존 frozen R0 prediction cache 의 bbox 와 8 코너 convex hull 에서만
가져온다.  teacher 의 keypoint 정확도를 여기서 평가하지 않는다.
평가 GT 는 어느 경로에서도 읽지 않는다.

valid depth 정의는 저장소가 이미 쓰는 것만 따른다 — 0 은 무효(sample_depth 가
`d > 0.05` 로 거른다).  새로운 far clipping 임계값을 만들지 않는다.  다만 dtype
최대값에 포화된 픽셀은 **따로 세어서 보고**한다(무효인지 측정인지 저장소가
말해주지 않으므로 양쪽 수치를 다 남긴다).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHES = [
    "data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json",
    "data/pallet/results/paper_selftrain_site_v1/preflight/SITE_A_TEACHER_CACHE.json",
]
NIGHT = {"capturenight01", "capturenight02", "capturenight03",
         "capturenight04", "capturenight10"}
# lock 에 적힌 support 기준.  결과를 보고 바꾸지 않는다.
MIN_ABSOLUTE_PIXELS = 200
MIN_ROI_FRACTION = 0.05
KP_CONF = 0.5   # PSEUDOLABEL_FILTER_LOCK 의 keypoint validity.  새로 정하지 않는다.


def depth_path_for(rgb_relative: str) -> Path:
    path = REPO_ROOT / rgb_relative
    return path.parent.parent / "depth" / f"{path.stem}.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 두 캐시를 합치되 같은 이미지는 한 번만 (SITE_A 는 day 전량, MAIN 은 night 포함)
    entries: dict[str, dict] = {}
    for relative in CACHES:
        payload = json.loads((REPO_ROOT / relative).read_text())
        for entry in payload["entries"]:
            entries.setdefault(entry["image_path"], entry)
    print(f"ROI 를 줄 수 있는 프레임 {len(entries)} (두 R0 캐시 합집합)")

    rows = []
    for image_path, entry in sorted(entries.items()):
        depth_file = depth_path_for(image_path)
        if not depth_file.exists():
            continue
        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        dtype_max = np.iinfo(depth.dtype).max
        positive = depth > 0
        saturated = depth == dtype_max
        valid = positive & ~saturated          # 보수적 정의
        valid_loose = positive                 # 저장소 소비자 정의 (0 만 무효)

        record = {
            "image_path": image_path,
            "source_recording": entry["capture_session"],
            "lighting": "night" if entry["capture_session"] in NIGHT else "day",
            "full_valid_ratio_strict": float(valid.mean()),
            "full_valid_ratio_loose": float(valid_loose.mean()),
            "zero_ratio": float((depth == 0).mean()),
            "saturated_ratio": float(saturated.mean()),
            "has_roi": False,
            "bbox_valid_ratio": None, "hull_valid_ratio": None,
            "roi_valid_pixels": None, "roi_pixels": None,
            "largest_component_ratio": None, "usable_support": None,
        }

        top = entry.get("top1")
        if top is not None:
            height, width = depth.shape
            x1, y1, x2, y2 = [int(round(v)) for v in top["box_xyxy"]]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, width), min(y2, height)
            if x2 > x1 and y2 > y1:
                record["has_roi"] = True
                box_mask = np.zeros(depth.shape, np.uint8)
                box_mask[y1:y2, x1:x2] = 1
                record["bbox_valid_ratio"] = float(valid[y1:y2, x1:x2].mean())

                keypoints = np.asarray(top["keypoints_xy"], np.float32)[:8]
                confidences = np.nan_to_num(
                    np.asarray(top["keypoints_conf"], float)[:8], nan=0.0)
                usable = keypoints[confidences >= KP_CONF]
                hull_mask = box_mask
                if len(usable) >= 3:
                    hull = cv2.convexHull(usable.reshape(-1, 1, 2))
                    hull_mask = np.zeros(depth.shape, np.uint8)
                    cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 1)
                roi = hull_mask.astype(bool)
                roi_pixels = int(roi.sum())
                roi_valid = valid & roi
                record["hull_valid_ratio"] = (float(roi_valid.sum() / roi_pixels)
                                              if roi_pixels else 0.0)
                record["roi_pixels"] = roi_pixels
                record["roi_valid_pixels"] = int(roi_valid.sum())

                count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    roi_valid.astype(np.uint8), 8)
                largest = (int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0)
                record["largest_component_ratio"] = (largest / roi_pixels
                                                     if roi_pixels else 0.0)
                record["usable_support"] = bool(
                    record["roi_valid_pixels"]
                    >= max(MIN_ABSOLUTE_PIXELS, MIN_ROI_FRACTION * roi_pixels))
        rows.append(record)
        if len(rows) % 500 == 0:
            print(f"  {len(rows)}/{len(entries)}", flush=True)

    with (out_dir / "DEPTH_SUPPORT_PER_FRAME.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def summarize(subset):
        with_roi = [r for r in subset if r["has_roi"]]
        block = {
            "frames": len(subset),
            "frames_with_roi": len(with_roi),
            "full_valid_strict_median": float(np.median(
                [r["full_valid_ratio_strict"] for r in subset])),
            "full_valid_strict_p10": float(np.percentile(
                [r["full_valid_ratio_strict"] for r in subset], 10)),
            "full_valid_loose_median": float(np.median(
                [r["full_valid_ratio_loose"] for r in subset])),
            "saturated_median": float(np.median([r["saturated_ratio"] for r in subset])),
            "zero_median": float(np.median([r["zero_ratio"] for r in subset])),
        }
        if with_roi:
            block |= {
                "roi_valid_median": float(np.median(
                    [r["hull_valid_ratio"] for r in with_roi])),
                "roi_valid_p10": float(np.percentile(
                    [r["hull_valid_ratio"] for r in with_roi], 10)),
                "roi_valid_pixels_median": float(np.median(
                    [r["roi_valid_pixels"] for r in with_roi])),
                "largest_component_median": float(np.median(
                    [r["largest_component_ratio"] for r in with_roi])),
                "usable_support_rate": float(np.mean(
                    [r["usable_support"] for r in with_roi])),
            }
        return block

    summary = {"ALL": summarize(rows)}
    for lighting in ("day", "night"):
        subset = [r for r in rows if r["lighting"] == lighting]
        if subset:
            summary[lighting.upper()] = summarize(subset)
    for recording in sorted({r["source_recording"] for r in rows}):
        summary[recording] = summarize(
            [r for r in rows if r["source_recording"] == recording])

    (out_dir / "DEPTH_SUPPORT_SUMMARY.json").write_text(json.dumps({
        "schema_version": "depth_support_summary_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gt_used": False,
        "roi_source": "existing frozen R0 prediction caches, bbox and 8-corner convex hull",
        "teacher_keypoint_accuracy_not_evaluated": True,
        "valid_definitions": {
            "strict": "depth > 0 and depth != dtype max",
            "loose": "depth > 0, which is what the repository's own sample_depth uses",
            "why_two": ("no acquisition record says whether the dtype maximum is an "
                        "invalid code or a real far reading, so both are reported and "
                        "no clipping rule is invented here"),
        },
        "usable_support_definition": {
            "rule": "roi valid pixels >= max(200, 5% of roi pixels)",
            "declared_in": "METHOD_INTENT_LOCK.json, before results",
            "means": "enough depth samples to attempt a geometric fit, not that a fit would be accurate",
        },
        "summary": summary,
    }, indent=2) + "\n")

    print(f"\n{'group':22}{'frames':>8}{'ROI':>7}{'full valid':>12}{'ROI valid':>11}"
          f"{'ROI px':>9}{'usable':>9}")
    print("-" * 78)
    for name in ["ALL", "DAY", "NIGHT"] + sorted({r["source_recording"] for r in rows}):
        block = summary.get(name)
        if not block:
            continue
        print(f"{name:22}{block['frames']:8d}{block['frames_with_roi']:7d}"
              f"{block['full_valid_strict_median']:12.3f}"
              f"{block.get('roi_valid_median', float('nan')):11.3f}"
              f"{block.get('roi_valid_pixels_median', float('nan')):9.0f}"
              f"{block.get('usable_support_rate', float('nan')):9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
