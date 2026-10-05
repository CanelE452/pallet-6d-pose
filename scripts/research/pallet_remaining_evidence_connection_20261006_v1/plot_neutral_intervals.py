"""Scientific figures from the completed descriptive command panel and raw PNGs."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--handoff", type=Path, required=True)
    args = parser.parse_args(); start = time.perf_counter()
    doc = args.root / "_docs/experiments/pallet_remaining_evidence_connection_20261006_v1/neutral_intervals"
    data = args.root / "data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/neutral_intervals"
    result = json.loads((data / "NEUTRAL_COMMAND_RESULT.json").read_text())
    inventory = json.loads(gzip.decompress((doc / "inputs/STOP_INTERVALS_INVENTORY.json.gz").read_bytes()).decode("utf-8-sig"))
    images = doc / "images"; images.mkdir(parents=True, exist_ok=True)
    provenance = []
    keys = [("x_population_std_m", 100, "x population SD (cm)"),
            ("z_population_std_m", 100, "z population SD (cm)"),
            ("yaw_population_std_360_unwrapped_deg", 1, "yaw population SD (deg)")]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, layout="constrained")
    for ax, (key, scale, label) in zip(axes, keys):
        for method, color in [("Base", "#26708e"), ("N3", "#b74630")]:
            ax.plot(range(1, 33), [item["methods"][method][key] * scale for item in result["intervals"]],
                    marker="o", markersize=4, color=color, label=method, linewidth=1)
        ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(loc="upper right")
    axes[-1].set_xticks(range(1, 33)); axes[-1].set_xlabel("All supplied neutral-command intervals, chronological order (1-32)")
    fig.suptitle("Observed output spread during neutral CAN commands\nRelative stationarity unconfirmed; not a stationary-noise or accuracy result", fontsize=14)
    target = images / "all_32_command_spreads.png"; fig.savefig(target, dpi=130); plt.close(fig)
    for page in range(4):
        fig, axes = plt.subplots(8, 3, figsize=(13, 20), layout="constrained")
        for row, candidate in enumerate(inventory["machine_proposed_intervals"][page*8:(page+1)*8]):
            for col, image in enumerate(candidate["representative_raw_images"]):
                source = args.handoff / "stops" / image["image_relative_path"]
                content = source.read_bytes()
                assert hashlib.sha256(content).hexdigest() == image["image_sha256"]
                bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
                assert bgr is not None and list(bgr.shape[:2]) == [480, 640]
                assert hashlib.sha256(bgr.tobytes()).hexdigest() == image["decoded_bgr_sha256"]
                ax = axes[row, col]; ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); ax.set_axis_off()
                ax.set_title(f"{page*8+row+1:02d} / {image['role']} / {image['frame_id']}", fontsize=9)
                provenance.append({"source_path_in_zip": "stops/" + image["image_relative_path"],
                                   "frame_id": image["frame_id"], "file_sha256": image["image_sha256"],
                                   "decoded_bgr_sha256": image["decoded_bgr_sha256"], "interval_id": candidate["interval_id"],
                                   "role": image["role"], "figure": f"images/raw_candidate_preview_{page+1}.png"})
        fig.suptitle(f"Raw candidate previews {page*8+1}-{page*8+8}: start / middle / end\nNo model overlays; snapshots do not establish whole-interval stationarity", fontsize=13)
        target = images / f"raw_candidate_preview_{page+1}.png"; fig.savefig(target, dpi=100); plt.close(fig)
    (doc / "FIGURE_PROVENANCE.json").write_text(json.dumps({"source_result_sha256": hashlib.sha256((data / "NEUTRAL_COMMAND_RESULT.json").read_bytes()).hexdigest(),
                    "raw_images_verified": len(provenance), "raw_image_sources": provenance,
                    "output_figures": [{"path": str(p.relative_to(doc)), "bytes": p.stat().st_size,
                                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(images.glob("*.png"))],
                    "transform": "Scientific figure composition/resizing of original unannotated source pixels; no points, detections or machine labels added",
                    "human_review_created": False, "cpu_wall_seconds": time.perf_counter()-start}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "raw_png_file_and_pixel_hashes": len(provenance), "figures": 5, "cpu_wall_seconds": time.perf_counter()-start}))


if __name__ == "__main__":
    main()
