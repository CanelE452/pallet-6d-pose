"""TEMPORAL PILOT — 이웃 RGB 프레임의 증거로 teacher 좌표를 고친다.

    python3 scripts/self_training_yolo/temporal_refine/refine_temporal_keypoints.py \
        --output-dir data/pallet/results/paper_temporal_selftrain_v1/pilot

출력  TEMPORAL_REFINEMENT_PER_FRAME.json · TEMPORAL_REFINEMENT_SUMMARY.json

★ 이 파일은 정답을 보지 않는다.  `refine_center` 는 GT 인자를 받지 않고, 이
모듈은 어노테이션 경로를 열지 않는다.  평가는 전적으로 별도 스크립트의 일이다.

RGB · cam_K · frozen R0 예측 · 등록된 plastic 치수 · lock 만 읽는다.
depth 를 읽지 않는다.  flow 파라미터·tracklet 길이는 lock 값 그대로다.
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
N_CORNERS = 8


def cuboid(across: float, height: float, along: float) -> np.ndarray:
    ha, hh, hb = across / 2, height / 2, along / 2
    return np.array([[-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
                     [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb]],
                    dtype=np.float64)


def transport(point, frames_grey, flow_params, fb_max):
    """이웃 좌표를 중앙 프레임까지 연속 hop 으로 옮긴다.  실패하면 None."""

    current = np.array([[point]], np.float32)
    for source, target in zip(frames_grey[:-1], frames_grey[1:]):
        forward, status, _ = cv2.calcOpticalFlowPyrLK(source, target, current, None,
                                                      **flow_params)
        if status is None or not status.all():
            return None
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(target, source, forward, None,
                                                         **flow_params)
        if status_b is None or not status_b.all():
            return None
        if float(np.linalg.norm(backward - current)) > fb_max:
            return None
        height, width = target.shape
        x, y = forward[0, 0]
        if not (0 <= x < width and 0 <= y < height):
            return None
        current = forward
    return current[0, 0].astype(np.float64)


def solve(model, points, camera, usable):
    ok, rvec, tvec = cv2.solvePnP(model[usable], points[usable], camera, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(model[usable], points[usable], camera, None,
                                      rvec, tvec)
    projected, _ = cv2.projectPoints(model, rvec, tvec, camera, None)
    projected = projected.reshape(-1, 2)
    residual = float(np.median(np.linalg.norm(projected[usable] - points[usable], axis=1)))
    if not np.isfinite(projected).all():
        return None
    return projected, residual


def refine_center(center_rgb: str, neighbour_rgbs: list[str], predictions: dict,
                  camera: np.ndarray, dimensions: dict, config: dict) -> dict:
    """중앙 프레임 하나를 고친다.  **정답 인자가 없다.**"""

    kp_conf_min = config["kp_conf_min"]
    fb_max = config["fb_max"]
    flow_params = config["flow_params"]
    min_observations = config["min_observations"]

    center = predictions.get(center_rgb, {}).get("top1")
    if center is None:
        return {"status": "NO_CENTER_DETECTION"}
    center_kp = np.asarray(center["keypoints_xy"], np.float64)[:N_CORNERS]
    center_conf = np.nan_to_num(
        np.asarray(center["keypoints_conf"], float)[:N_CORNERS], nan=0.0)

    grey_cache: dict[str, np.ndarray] = {}

    def grey(path: str):
        if path not in grey_cache:
            image = cv2.imread(str(REPO_ROOT / path))
            grey_cache[path] = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                if image is not None else None)
        return grey_cache[path]

    # 이웃을 중앙 기준 거리 순으로 정렬해 hop 경로를 만든다
    ordered = sorted(neighbour_rgbs, key=lambda p: int(Path(p).stem))
    center_position = sorted(ordered + [center_rgb],
                             key=lambda p: int(Path(p).stem)).index(center_rgb)
    sequence = sorted(ordered + [center_rgb], key=lambda p: int(Path(p).stem))

    observations = [[] for _ in range(N_CORNERS)]
    for index in range(N_CORNERS):
        if center_conf[index] >= kp_conf_min and np.isfinite(center_kp[index]).all():
            observations[index].append(center_kp[index])

    rejected_hops = 0
    attempted_hops = 0
    for position, path in enumerate(sequence):
        if position == center_position:
            continue
        neighbour = predictions.get(path, {}).get("top1")
        if neighbour is None:
            continue
        kp = np.asarray(neighbour["keypoints_xy"], np.float64)[:N_CORNERS]
        conf = np.nan_to_num(np.asarray(neighbour["keypoints_conf"], float)[:N_CORNERS],
                             nan=0.0)
        step = 1 if position < center_position else -1
        chain = [sequence[i] for i in range(position, center_position + step, step)]
        greys = [grey(p) for p in chain]
        if any(g is None for g in greys):
            continue
        for index in range(N_CORNERS):
            if conf[index] < kp_conf_min or not np.isfinite(kp[index]).all():
                continue
            attempted_hops += 1
            moved = transport(kp[index], greys, flow_params, fb_max)
            if moved is None:
                rejected_hops += 1
                continue
            observations[index].append(moved)

    counts = [len(o) for o in observations]
    consensus = np.full((N_CORNERS, 2), np.nan)
    for index, obs in enumerate(observations):
        if len(obs) >= min_observations:
            arr = np.asarray(obs)
            consensus[index] = [float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))]

    result = {
        "status": "OK",
        "raw_teacher": center_kp.tolist(),
        "raw_teacher_conf": center_conf.tolist(),
        "observation_counts": counts,
        "attempted_hops": attempted_hops,
        "rejected_hops": rejected_hops,
        "temporal_only": consensus.tolist(),
        "temporal_resolved_corners": int(np.isfinite(consensus).all(axis=1).sum()),
        "temporal_geometry": None,
        "geometry_status": None,
        "selected_hypothesis": None,
    }
    usable = np.isfinite(consensus).all(axis=1)
    if usable.sum() < 6:
        result["geometry_status"] = "TOO_FEW_TEMPORAL_CORNERS"
        return result

    long_m = max(dimensions["x"], dimensions["z"])
    short_m = min(dimensions["x"], dimensions["z"])
    height_m = dimensions["y"]
    best = None
    for name, (across, along) in (("CF_WIDTH", (long_m, short_m)),
                                  ("CF_DEPTH", (short_m, long_m))):
        fit = solve(cuboid(across, height_m, along), consensus, camera, usable)
        if fit and (best is None or fit[1] < best[1][1]):
            best = (name, fit)
    if best is None:
        result["geometry_status"] = "PNP_FAILED"
        return result
    name, (projected, residual) = best
    result["temporal_geometry"] = projected[:N_CORNERS].tolist()
    result["geometry_status"] = "OK"
    result["selected_hypothesis"] = name
    result["geometry_reproj_residual_px"] = residual
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    lock = json.loads((out_dir.parent / "TEMPORAL_METHOD_LOCK.json").read_text())
    flow = lock["optical_flow"]
    config = {
        "kp_conf_min": float(lock["keypoint_validity"]["kp_conf_min"]),
        "fb_max": float(flow["forward_backward_max_error_px"]),
        "min_observations": int(lock["algorithm"]["min_observations"]),
        "flow_params": {
            "winSize": tuple(flow["win_size"]),
            "maxLevel": int(flow["max_level"]),
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         int(flow["criteria_count"]), float(flow["criteria_epsilon"])),
        },
    }
    dimensions = lock["object_geometry"]["dimensions_m"]
    print(f"flow {config['flow_params']['winSize']} lvl {config['flow_params']['maxLevel']} "
          f"fb {config['fb_max']}  min obs {config['min_observations']}")

    predictions = json.loads(
        (out_dir / "R0_TEMPORAL_TEACHER_CACHE.json").read_text())["predictions"]
    rows = [r for r in csv.DictReader((out_dir / "TEMPORAL_PILOT_POPULATION.csv").open())
            if r["eligible"] == "True"]
    print(f"centres {len(rows)}")

    camera_cache: dict[str, np.ndarray] = {}
    results = {}
    for order, row in enumerate(rows, start=1):
        session = row["source_recording"]
        if session not in camera_cache:
            path = REPO_ROOT / row["center_rgb"]
            camera_cache[session] = np.loadtxt(path.parent.parent / "cam_K.txt")
        outcome = refine_center(
            row["center_rgb"],
            [p for p in row["neighbor_rgb_paths"].split("|") if p],
            predictions, camera_cache[session], dimensions, config)
        outcome["source_recording"] = session
        outcome["center_rgb"] = row["center_rgb"]
        outcome["gt_annotation_path"] = row["gt_annotation_path"]   # 경로만, 읽지 않는다
        results[row["center_frame_id"]] = outcome
        if order % 20 == 0:
            print(f"  {order}/{len(rows)}", flush=True)

    (out_dir / "TEMPORAL_REFINEMENT_PER_FRAME.json").write_text(json.dumps({
        "schema_version": "temporal_refinement_per_frame_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_read": False,
        "depth_read": False,
        "flow": flow,
        "frames": results,
    }, indent=2) + "\n")

    ok = [r for r in results.values() if r["status"] == "OK"]
    geometry_ok = [r for r in ok if r["geometry_status"] == "OK"]
    counts = np.array([c for r in ok for c in r["observation_counts"]])
    attempted = sum(r["attempted_hops"] for r in ok)
    rejected = sum(r["rejected_hops"] for r in ok)
    summary = {
        "schema_version": "temporal_refinement_summary_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "centres": len(results),
        "with_center_detection": len(ok),
        "tracklet_length": lock["tracklet"]["length"],
        "observations_per_corner_median": float(np.median(counts)) if counts.size else None,
        "corners_with_at_least_min_observations": float(
            np.mean(counts >= config["min_observations"])) if counts.size else None,
        "fb_rejection_rate": rejected / attempted if attempted else None,
        "attempted_hops": attempted, "rejected_hops": rejected,
        "temporal_resolved_corners_median": float(np.median(
            [r["temporal_resolved_corners"] for r in ok])) if ok else None,
        "geometry_ok": len(geometry_ok),
        "geometry_coverage": len(geometry_ok) / len(ok) if ok else 0.0,
        "geometry_status_counts": {
            status: sum(1 for r in ok if r["geometry_status"] == status)
            for status in {r["geometry_status"] for r in ok}},
    }
    (out_dir / "TEMPORAL_REFINEMENT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"\ncentres {summary['centres']}  detected {summary['with_center_detection']}")
    print(f"observations per corner median {summary['observations_per_corner_median']}")
    print(f"corners with >= {config['min_observations']} observations "
          f"{summary['corners_with_at_least_min_observations']:.3f}")
    print(f"FB rejection rate {summary['fb_rejection_rate']:.4f}")
    print(f"geometry coverage {summary['geometry_coverage']:.3f}  "
          f"{summary['geometry_status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
