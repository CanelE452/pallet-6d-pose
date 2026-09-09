#!/usr/bin/env python3
"""Fixed classical candidate/oracle diagnostic on the existing real DEV manifest.

No training, tuning, semantic association, or deployed accuracy measurement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                  (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
CONFIG = {
    "input": "Full original image converted to grayscale; no crop, resize, padding or GT in detector",
    "canny": {"threshold1": 50, "threshold2": 150, "apertureSize": 3, "L2gradient": False},
    "HoughLinesP": {"rho": 1, "theta_deg": 1, "threshold": 30, "minLineLength": 15, "maxLineGap": 5},
    "LSD": {"refine": "LSD_REFINE_STD", "other_parameters": "OpenCV defaults; no extra filtering"},
    "oracle": {"angle_max_deg": 5, "full_gt_endpoint_distance_max_diagonal": .01,
               "visible_gt_projected_overlap_min": .25,
               "support": "Valid nondegenerate GT segment intersects original-image rectangle [0,w-1] x [0,h-1]",
               "distance": "Mean absolute perpendicular distances of BOTH original full GT endpoints to infinite candidate line",
               "overlap": "Project candidate endpoints onto GT unit direction; intersect this interval with the GT segment clipped to the image; divide overlap by clipped GT length",
               "association": "Each role independently asks if ANY candidate passes all gates; candidates may match multiple roles; GT selects representative match"},
    "panel_selection": "One SHA256-smallest frame ID per sorted real group, fixed independently of outputs",
}


def clipped_gt_interval(segment, width, height):
    p0, p1 = np.asarray(segment, float)
    delta = p1 - p0
    low, high = 0., 1.
    for p, q in ((-delta[0], p0[0]), (delta[0], width - 1 - p0[0]),
                 (-delta[1], p0[1]), (delta[1], height - 1 - p0[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        elif p < 0:
            low = max(low, q / p)
        else:
            high = min(high, q / p)
        if low > high:
            return None
    return low, high


def oracle_matches(candidates, points, valid, width, height):
    rows = []
    truth = np.asarray(points, dtype=float)[EDGES]
    candidate_directions = candidates[:, 1] - candidates[:, 0]
    candidate_lengths = np.linalg.norm(candidate_directions, axis=1)
    candidate_unit = candidate_directions / np.maximum(candidate_lengths[:, None], 1e-12)
    candidate_normal = np.stack([-candidate_unit[:, 1], candidate_unit[:, 0]], axis=1)
    diagonal = float(np.hypot(width, height))
    for role, (a, b) in enumerate(EDGES):
        segment = truth[role]
        length = float(np.linalg.norm(segment[1] - segment[0]))
        clip = clipped_gt_interval(segment, width, height)
        supported = bool(valid[a] and valid[b] and np.isfinite(segment).all()
                         and length >= 2 and clip is not None and (clip[1] - clip[0]) * length > 1e-6)
        row = {"role": role, "supported": supported, "matched": False, "matching_candidate_count": 0,
               "best_candidate_index": None, "best_angle_deg": None,
               "best_distance_px": None, "best_overlap_fraction": None,
               "nearest_angle_distance_candidate_index": None,
               "gt_full_length_px": length, "gt_clipped_length_px": None}
        if supported:
            unit = (segment[1] - segment[0]) / length
            visible_low, visible_high = clip[0] * length, clip[1] * length
            visible_length = visible_high - visible_low
            row["gt_clipped_length_px"] = visible_length
            if len(candidates):
                angle = np.rad2deg(np.arccos(np.clip(np.abs(candidate_unit @ unit), 0, 1)))
                distance = np.abs(((segment[None] - candidates[:, :1]) * candidate_normal[:, None]).sum(axis=2)).mean(axis=1)
                projected = ((candidates - segment[0]) * unit).sum(axis=2)
                overlap = np.maximum(0, np.minimum(projected.max(axis=1), visible_high)
                                     - np.maximum(projected.min(axis=1), visible_low)) / visible_length
                passing = (angle <= 5) & (distance <= .01 * diagonal) & (overlap >= .25) & (candidate_lengths > 1e-8)
                indices = np.flatnonzero(passing)
                score = (angle / 5) ** 2 + (distance / (.01 * diagonal)) ** 2
                row["nearest_angle_distance_candidate_index"] = int(np.argmin(score))
                if len(indices):
                    best = int(indices[np.argmin(score[indices])])
                    row.update(matched=True, matching_candidate_count=int(len(indices)),
                               best_candidate_index=best, best_angle_deg=float(angle[best]),
                               best_distance_px=float(distance[best]), best_overlap_fraction=float(overlap[best]))
        rows.append(row)
    return rows


def summarize(rows, frames):
    result = {}
    for method in ("HoughLinesP", "LSD"):
        selected = [r for r in rows if r["method"] == method and r["supported"]]
        counts = np.array([r[f"{method}_candidate_count"] for r in frames])
        matched = sum(r["matched"] for r in selected)
        result[method] = {"supported_roles": len(selected), "matched_roles": matched,
                          "oracle_role_coverage": matched / len(selected) if selected else None,
                          "candidate_count": {"total": int(counts.sum()), "median": float(np.median(counts)),
                                              "mean": float(np.mean(counts)), "p90": float(np.percentile(counts, 90))}}
    return result


def panel(path, rgb, canny, candidates, points, roles, frame_id):
    height, width = rgb.shape[:2]
    truth = np.asarray(points)[EDGES]
    supported = np.array([r["supported"] for r in roles])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    fig.subplots_adjust(left=.025, right=.975, top=.89, bottom=.065, hspace=.20, wspace=.09)
    fig.suptitle(frame_id, fontsize=14, weight="bold", y=.982)
    fig.text(.5, .944, "Fixed classical candidates on original RGB | GT used only for oracle matching", ha="center", fontsize=11)
    for ax in axes.flat:
        ax.imshow(rgb)
        ax.set_xlim(-.5, width - .5); ax.set_ylim(height - .5, -.5)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].add_collection(LineCollection(truth[supported], colors="#39ff73", linewidths=1.5))
    for row in roles:
        if row["supported"]:
            midpoint = truth[row["role"]].mean(axis=0)
            axes[0, 0].text(*midpoint, str(row["role"]), color="black", fontsize=8,
                            bbox={"facecolor": "#39ff73", "edgecolor": "none", "pad": .4})
    axes[0, 0].set_title("Original + GT cuboid segments (green, role IDs)")
    axes[0, 1].imshow(canny, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Canny: 50 / 150, original image")
    axes[1, 0].add_collection(LineCollection(candidates, colors="#ffab45", linewidths=.8, alpha=.9))
    axes[1, 0].set_title(f"All HoughLinesP candidates: {len(candidates)} (orange)")
    axes[1, 1].add_collection(LineCollection(truth[supported], colors="#39ff73", linewidths=1.8))
    selected = sorted({r["best_candidate_index"] for r in roles if r["matched"]})
    axes[1, 1].add_collection(LineCollection(candidates[selected], colors="#ff44dd", linewidths=2.1))
    matches = sum(r["matched"] for r in roles)
    axes[1, 1].set_title(f"GT-oracle-selected candidates (magenta): {matches}/{supported.sum()} roles")
    fig.text(.5, .026, "Oracle coverage is candidate availability, not semantic detection accuracy. Hidden cuboid lines may lack visible edges; background lines can match accidentally.", ha="center", fontsize=9)
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    out = run_dir / "diagnosis/classical"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir(exist_ok=True)
    (out / "panels").mkdir(exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["populations"]["real_dev"]
    selected = [min([r for r in records if r["group"] == group],
                    key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())["id"]
                for group in sorted({r["group"] for r in records})]
    cv2.setNumThreads(1)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    all_rows, frames, panels = [], [], []
    for record in records:
        bgr = cv2.imread(record["image"])
        if bgr is None:
            raise ValueError(record["image"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if (width, height) != (record["width"], record["height"]):
            raise ValueError("Manifest/image dimension mismatch")
        canny = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=False)
        hough = cv2.HoughLinesP(canny, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=5)
        detected_lsd = lsd.detect(gray)[0]
        candidates = {"HoughLinesP": np.asarray(hough, float).reshape(-1, 2, 2) if hough is not None else np.empty((0, 2, 2)),
                      "LSD": np.asarray(detected_lsd, float).reshape(-1, 2, 2) if detected_lsd is not None else np.empty((0, 2, 2))}
        frame = {"id": record["id"], "group": record["group"], "width": width, "height": height,
                 "canny_edge_pixel_fraction": float(np.mean(canny > 0))}
        matched = {}
        for method, lines in candidates.items():
            frame[f"{method}_candidate_count"] = len(lines)
            matched[method] = oracle_matches(lines, record["gt_points"], record["gt_valid"], width, height)
            for row in matched[method]:
                all_rows.append(dict(id=record["id"], group=record["group"], method=method, **row))
        frames.append(frame)
        key = hashlib.sha256(record["id"].encode()).hexdigest()[:12]
        candidate_path = out / "candidates" / f"{key}.npz"
        np.savez_compressed(candidate_path, **candidates, gt_points=record["gt_points"], gt_valid=record["gt_valid"])
        frame["candidate_artifact"] = str(candidate_path.relative_to(run_dir))
        if record["id"] in selected:
            panel_path = out / "panels" / f"{record['group']}__{key}.png"
            panel(panel_path, rgb, canny, candidates["HoughLinesP"], record["gt_points"], matched["HoughLinesP"], record["id"])
            panels.append({"id": record["id"], "group": record["group"], "path": str(panel_path.relative_to(run_dir))})
        print(f"{record['id']}: Hough {len(candidates['HoughLinesP'])}, LSD {len(candidates['LSD'])}", flush=True)
    report = {"status": "COMPLETE", "config": CONFIG, "opencv_version": cv2.__version__,
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "population": "existing real_dev", "n_frames": len(records),
              "overall": summarize(all_rows, frames),
              "by_group": {group: summarize([r for r in all_rows if r["group"] == group],
                                            [f for f in frames if f["group"] == group]) for group in sorted({r["group"] for r in records})},
              "by_role": {str(role): {method: {"supported": sum(r["supported"] for r in all_rows if r["role"] == role and r["method"] == method),
                                               "matched": sum(r["matched"] for r in all_rows if r["role"] == role and r["method"] == method)}
                                      for method in candidates} for role in range(12)},
              "panels": panels, "frames": frames,
              "interpretation_limits": [
                  "GT-oracle candidate availability is not deployed semantic detection precision or model accuracy.",
                  "There is no learned role association, and one candidate can cover multiple GT roles.",
                  "Background lines can match a GT line accidentally within the tolerance.",
                  "Cuboid supporting lines can be occluded or geometrically virtual, with no physical visible edge.",
                  "Failure of these fixed classical settings does not prove that the image lacks usable information.",
                  "LSD uses its native default scale/refinement and no Hough-specific length filter; compare as separate candidate generators.",
                  "Original full GT endpoints determine line distance; overlap uses only the GT segment clipped into the original image.",
              ]}
    (out / "RESULTS.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (out / "PER_ROLE.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    text = ["# Fixed classical edge-candidate diagnostic", "", "This is GT-oracle candidate coverage, not semantic detection accuracy.", "",
            "| Method | Available GT roles | Oracle matched | Coverage | Median candidates/frame |", "|---|---:|---:|---:|---:|"]
    for method, result in report["overall"].items():
        text.append(f"| {method} | {result['supported_roles']} | {result['matched_roles']} | {result['oracle_role_coverage']:.1%} | {result['candidate_count']['median']:.1f} |")
    text += ["", "Canny 50/150; HoughLinesP rho=1px, theta=1deg, threshold=30, minimum segment length=15px, maximum gap=5px. No tuning.",
             "Oracle gate: angle <=5deg; mean full-GT-endpoint distance <=1% image diagonal; candidate projection overlaps >=25% of the image-clipped GT segment.", ""]
    text.extend(f"- {note}" for note in report["interpretation_limits"])
    text += ["", "Predetermined panels (one SHA256-smallest ID per session):", ""]
    text.extend(f"- [{item['id']}]({Path(item['path']).relative_to('diagnosis/classical').as_posix()})" for item in panels)
    (out / "REPORT.md").write_text("\n".join(text) + "\n")
    print(json.dumps(report["overall"], indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run(parser.parse_args().run_dir)
