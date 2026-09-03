"""GATE 0 — RGB-D 센서 계약 감사.  inventory · depth scale · calibration.

    python3 scripts/self_training_yolo/depth_corrected/audit_rgbd_contract.py \
        --output-dir data/pallet/results/paper_depth_selftrain_v1/gate0

출력  RGBD_RECORDING_INVENTORY.{csv,json} · DEPTH_SCALE_AUDIT.json ·
      RGBD_CALIBRATION_AUDIT.json

"RealSense 니까 mm" 로 정하지 않는다.  scale 과 alignment 는 저장소 안에서
근거를 찾고, 못 찾으면 UNKNOWN 으로 남긴다.  파일 이름·해상도 일치는 근거가
아니다.

평가 GT 를 읽지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "data/pallet/raw_data"

RECORDINGS = [
    ("day", "outside/capturepallet01"), ("day", "outside/capturepallet10"),
    ("day", "outside/capturepallet11"),
    ("night", "night/capturenight01"), ("night", "night/capturenight02"),
    ("night", "night/capturenight03"), ("night", "night/capturenight04"),
    ("night", "night/capturenight10"),
]
SAMPLE_FRAMES = 24     # scale 통계용 표본 (전수 해싱은 support 단계에서)

# 저장된 depth 를 실제로 해석하는 저장소 코드.  여기가 유일한 scale 근거다.
CONSUMER_CONTRACT = {
    "file": "challenge/scripts/live/run_live_io.py",
    "symbol": "NpDepthFrame",
    "docstring": "numpy uint16(mm) depth -> RealSense depth_frame 인터페이스 흉내",
    "conversion": "get_distance returns d[y, x] / 1000.0",
    "loader": "load_seq reads the same rgb/ + depth/ + cam_K.txt layout as these recordings",
    "corroboration": "scripts/evaluation/classify_daytime_visibility.py sets DEPTH_SCALE_MM = 0.001",
    "what_it_is_not": ("this is a consumer-side contract in this repository, not a "
                       "depth_scale declared by the acquisition software; no metadata "
                       "file in these recordings states a scale"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory, scale_audit, calibration = [], {}, {}
    for lighting, relative in RECORDINGS:
        base = RAW / relative
        rgb_dir, depth_dir = base / "rgb", base / "depth"
        rgb = sorted(p for p in rgb_dir.iterdir()) if rgb_dir.is_dir() else []
        depth = sorted(p for p in depth_dir.iterdir()) if depth_dir.is_dir() else []
        cam_k = base / "cam_K.txt"

        rgb_sample = cv2.imread(str(rgb[0])) if rgb else None
        depth_sample = cv2.imread(str(depth[0]), cv2.IMREAD_UNCHANGED) if depth else None

        # 이 촬영에 있는 메타 파일 전부 (없으면 없다고 적는다)
        meta_files = sorted(p.name for p in base.iterdir() if p.is_file())

        entry = {
            "source_recording": base.name,
            "lighting": lighting,
            "rgb_root": str(rgb_dir.relative_to(REPO_ROOT)),
            "depth_root": str(depth_dir.relative_to(REPO_ROOT)) if depth_dir.is_dir() else None,
            "rgb_count": len(rgb),
            "depth_count": len(depth),
            "rgb_resolution": (f"{rgb_sample.shape[1]}x{rgb_sample.shape[0]}"
                               if rgb_sample is not None else None),
            "depth_resolution": (f"{depth_sample.shape[1]}x{depth_sample.shape[0]}"
                                 if depth_sample is not None else None),
            "rgb_dtype": str(rgb_sample.dtype) if rgb_sample is not None else None,
            "depth_dtype": str(depth_sample.dtype) if depth_sample is not None else None,
            "camera_metadata_path": (str(cam_k.relative_to(REPO_ROOT))
                                     if cam_k.exists() else None),
            "session_metadata_path": None,
            "metadata_files_present": meta_files,
            "color_intrinsics_available": cam_k.exists(),
            "depth_intrinsics_available": False,
            "extrinsics_available": False,
            "depth_scale_available": False,
            "depth_scale_value": None,
            "depth_scale_source": None,
            "alignment_status": "ALIGNMENT_UNKNOWN",
            "timestamp_source": "filename stem, nanosecond integer",
        }

        # ── calibration
        colour_k = None
        if cam_k.exists():
            colour_k = np.loadtxt(cam_k)
            entry["depth_scale_available"] = True
            entry["depth_scale_value"] = 0.001
            entry["depth_scale_source"] = "repository consumer contract (see DEPTH_SCALE_AUDIT)"
        calibration[base.name] = {
            "color_K_status": "PRESENT" if cam_k.exists() else "MISSING",
            "color_K": colour_k.tolist() if colour_k is not None else None,
            "color_K_sha256": sha256_file(cam_k) if cam_k.exists() else None,
            "depth_K_status": "MISSING",
            "extrinsics_status": "MISSING",
            "aligned_depth_status": "UNPROVEN",
            "calibration_source": (str(cam_k.relative_to(REPO_ROOT))
                                   if cam_k.exists() else None),
            "note": ("only one 3x3 intrinsic file exists per recording and it carries no "
                     "stream label, no distortion and no depth-to-color transform. If the "
                     "stored depth is not already aligned to colour, nothing in the "
                     "repository could align it."),
        }

        # ── depth 값 분포 (표본)
        if depth:
            positions = np.linspace(0, len(depth) - 1,
                                    min(SAMPLE_FRAMES, len(depth))).round().astype(int)
            zero_fraction, saturated_fraction, pooled = [], [], []
            dtype_max = None
            for index in positions:
                image = cv2.imread(str(depth[index]), cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue
                dtype_max = np.iinfo(image.dtype).max
                zero_fraction.append(float((image == 0).mean()))
                saturated_fraction.append(float((image == dtype_max).mean()))
                values = image[(image > 0) & (image < dtype_max)]
                if values.size:
                    pooled.append(values.astype(np.float64))
            pooled_values = np.concatenate(pooled) if pooled else np.array([])
            scale_audit[base.name] = {
                "lighting": lighting,
                "raw_dtype": entry["depth_dtype"],
                "raw_dtype_max": int(dtype_max) if dtype_max else None,
                "frames_sampled": int(len(positions)),
                "zero_fraction_mean": float(np.mean(zero_fraction)) if zero_fraction else None,
                "saturated_at_dtype_max_fraction_mean": (
                    float(np.mean(saturated_fraction)) if saturated_fraction else None),
                "raw_excluding_zero_and_saturated": {
                    "min": float(pooled_values.min()) if pooled_values.size else None,
                    "p10": float(np.percentile(pooled_values, 10)) if pooled_values.size else None,
                    "median": float(np.median(pooled_values)) if pooled_values.size else None,
                    "p90": float(np.percentile(pooled_values, 90)) if pooled_values.size else None,
                    "max": float(pooled_values.max()) if pooled_values.size else None,
                },
                "metres_if_scale_is_0_001": {
                    "p10": float(np.percentile(pooled_values, 10) / 1000) if pooled_values.size else None,
                    "median": float(np.median(pooled_values) / 1000) if pooled_values.size else None,
                    "p90": float(np.percentile(pooled_values, 90) / 1000) if pooled_values.size else None,
                },
                "depth_scale": 0.001,
                "metric_conversion_confidence": "CONSUMER_CONTRACT_ONLY",
            }
        inventory.append(entry)
        print(f"  {base.name:20} rgb {len(rgb):5d}  depth {len(depth):5d}  "
              f"K {'yes' if cam_k.exists() else 'NO':3}  meta {meta_files}")

    # ── cam_K 가 촬영마다 같은지 (다르면 하나로 덮어쓰지 않는다)
    by_hash: dict[str, list[str]] = {}
    for name, block in calibration.items():
        by_hash.setdefault(block["color_K_sha256"] or "none", []).append(name)

    scale_report = {
        "schema_version": "depth_scale_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gt_used": False,
        "scale_was_assumed_from_sensor_folklore": False,
        "evidence": CONSUMER_CONTRACT,
        "no_acquisition_metadata_declares_scale": True,
        "saturation_warning": ("a value equal to the dtype maximum is reported separately. "
                               "No consumer in this repository special-cases it, so it "
                               "would convert to an implausible distance if treated as a "
                               "measurement. This audit does not invent a clipping rule; "
                               "it reports the fraction."),
        "recordings": scale_audit,
    }
    (out_dir / "DEPTH_SCALE_AUDIT.json").write_text(json.dumps(scale_report, indent=2) + "\n")

    calibration_report = {
        "schema_version": "rgbd_calibration_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gt_used": False,
        "intrinsics_groups": by_hash,
        "intrinsics_differ_between_recordings": len(by_hash) > 1,
        "recordings": calibration,
        "verdict_inputs": {
            "depth_intrinsics_found_anywhere": False,
            "depth_to_color_extrinsics_found_anywhere": False,
            "aligned_to_color_declared_anywhere": False,
            "searched": ["recording folders", "scripts/", "challenge/", "config/",
                         "data/pallet/raw_data/"],
            "closest_related_code": ("challenge/scripts/live/run_live.py and "
                                     "scripts/dope/run_dope_live.py call "
                                     "rs.align(rs.stream.color), but only on a LIVE "
                                     "pipeline; the offline --seq path just reads the "
                                     "stored PNG and samples it at colour pixel "
                                     "coordinates, which assumes alignment rather than "
                                     "proving it"),
        },
    }
    (out_dir / "RGBD_CALIBRATION_AUDIT.json").write_text(
        json.dumps(calibration_report, indent=2) + "\n")

    (out_dir / "RGBD_RECORDING_INVENTORY.json").write_text(json.dumps({
        "schema_version": "rgbd_recording_inventory_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gt_used": False,
        "recordings": inventory,
    }, indent=2) + "\n")
    fields = [k for k in inventory[0] if k != "metadata_files_present"]
    with (out_dir / "RGBD_RECORDING_INVENTORY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in inventory:
            writer.writerow({k: entry[k] for k in fields})

    print(f"\nintrinsics 그룹 {len(by_hash)}")
    for digest, names in by_hash.items():
        print(f"  {digest[:10]}  {names}")
    print(f"\n{'recording':20}{'zero%':>8}{'sat%':>8}{'raw p50':>10}{'m @0.001':>10}")
    print("-" * 58)
    for name, block in scale_audit.items():
        print(f"{name:20}{block['zero_fraction_mean'] * 100:8.2f}"
              f"{block['saturated_at_dtype_max_fraction_mean'] * 100:8.2f}"
              f"{block['raw_excluding_zero_and_saturated']['median']:10.0f}"
              f"{block['metres_if_scale_is_0_001']['median']:10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
