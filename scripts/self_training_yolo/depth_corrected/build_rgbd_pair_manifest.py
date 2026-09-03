"""GATE 0 — RGB 와 depth 가 어떤 근거로 짝지어지는지 복원한다.

    python3 scripts/self_training_yolo/depth_corrected/build_rgbd_pair_manifest.py \
        --output-dir data/pallet/results/paper_depth_selftrain_v1/gate0

출력  RGBD_PAIR_MANIFEST.csv · RGBD_PAIR_SUMMARY.json

근거 우선순위대로 찾는다.

    1  명시적 timestamp / metadata association
    2  동일 acquisition timestamp (RGB 와 depth 각각의 시각)
    3  저장소에 이미 있는 pairing contract
    4  파일 이름 stem 일치

stem 이 같다는 것만으로 **동기화를 확정하지 않는다**.  stem 이 유일한 근거이면
timestamp delta 는 측정 불가로 남긴다 — 0 으로 적지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "data/pallet/raw_data"
RECORDINGS = [
    ("day", "outside/capturepallet01"), ("day", "outside/capturepallet10"),
    ("day", "outside/capturepallet11"),
    ("night", "night/capturenight01"), ("night", "night/capturenight02"),
    ("night", "night/capturenight03"), ("night", "night/capturenight04"),
    ("night", "night/capturenight10"),
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# 저장소가 이미 갖고 있는 pairing 계약 (근거 3)
EXISTING_CONTRACT = {
    "file": "challenge/scripts/live/run_live_io.py",
    "symbol": "load_seq",
    "behaviour": "pairs rgb/<stem>.png with depth/<stem>.png for this exact layout",
    "strength": ("this is a repository convention, not an acquisition record. It shows "
                 "the project treats same-stem files as a pair; it does not show that "
                 "the two frames were captured at the same instant."),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = [], {}
    for lighting, relative in RECORDINGS:
        base = RAW / relative
        rgb = {p.stem: p for p in (base / "rgb").iterdir()
               if p.suffix.lower() in IMAGE_SUFFIXES}
        depth = {p.stem: p for p in (base / "depth").iterdir()
                 if p.suffix.lower() in IMAGE_SUFFIXES}

        # 근거 1·2 를 줄 만한 파일이 이 촬영에 있는가
        side_files = [p.name for p in base.iterdir() if p.is_file()]
        has_association_metadata = any(
            token in name.lower() for name in side_files
            for token in ("timestamp", "frames", "manifest", "session", "assoc", "sync"))

        paired = sorted(set(rgb) & set(depth))
        rgb_only = sorted(set(rgb) - set(depth))
        depth_only = sorted(set(depth) - set(rgb))

        method = ("EXPLICIT_METADATA_ASSOCIATION" if has_association_metadata
                  else "EXACT_FILENAME_STEM")
        for stem in paired:
            rows.append({
                "source_recording": base.name,
                "lighting": lighting,
                "rgb_path": str(rgb[stem].relative_to(REPO_ROOT)),
                "depth_path": str(depth[stem].relative_to(REPO_ROOT)),
                "rgb_timestamp": stem,
                # depth 파일은 독립적인 시각을 갖고 있지 않다.  같은 stem 을 공유할 뿐이다.
                "depth_timestamp": stem,
                "timestamp_delta_ms": "UNMEASURABLE",
                "pairing_method": method,
                "paired": True,
            })
        for stem in rgb_only:
            rows.append({
                "source_recording": base.name, "lighting": lighting,
                "rgb_path": str(rgb[stem].relative_to(REPO_ROOT)), "depth_path": None,
                "rgb_timestamp": stem, "depth_timestamp": None,
                "timestamp_delta_ms": None, "pairing_method": method, "paired": False,
            })

        summary[base.name] = {
            "lighting": lighting,
            "rgb": len(rgb), "depth": len(depth),
            "paired": len(paired),
            "rgb_without_depth": len(rgb_only),
            "depth_without_rgb": len(depth_only),
            "pairing_rate": len(paired) / len(rgb) if rgb else 0.0,
            "pairing_method": method,
            "side_files": side_files,
            "independent_depth_timestamp_exists": False,
            "timestamp_delta_median_ms": None,
            "timestamp_delta_p90_ms": None,
            "timestamp_delta_max_ms": None,
        }
        print(f"  {base.name:20} rgb {len(rgb):5d}  paired {len(paired):5d}  "
              f"rate {summary[base.name]['pairing_rate']:.3f}  method {method}")

    report = {
        "schema_version": "rgbd_pair_summary_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gt_used": False,
        "evidence_search_order": [
            "explicit timestamp/metadata association",
            "independent acquisition timestamps for rgb and depth",
            "existing pairing contract in the repository",
            "exact filename stem",
        ],
        "existing_contract_found": EXISTING_CONTRACT,
        "explicit_association_metadata_found": False,
        "independent_depth_timestamps_found": False,
        "why_delta_is_unmeasurable": (
            "each frame carries one nanosecond stem that names both the rgb and the "
            "depth file. There is no second, depth-side timestamp to difference "
            "against, so the rgb-to-depth capture delta cannot be measured from what "
            "is stored. Reporting it as 0 ms would be circular."),
        "recordings": summary,
        "overall": {
            "rgb": sum(v["rgb"] for v in summary.values()),
            "paired": sum(v["paired"] for v in summary.values()),
            "pairing_rate": (sum(v["paired"] for v in summary.values())
                             / max(sum(v["rgb"] for v in summary.values()), 1)),
        },
    }
    (out_dir / "RGBD_PAIR_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out_dir / "RGBD_PAIR_MANIFEST.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\noverall pairing rate {report['overall']['pairing_rate']:.4f}  "
          f"({report['overall']['paired']}/{report['overall']['rgb']})")
    print("timestamp delta: UNMEASURABLE — 독립적인 depth 시각이 저장돼 있지 않다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
