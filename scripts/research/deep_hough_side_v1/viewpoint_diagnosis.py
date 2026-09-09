#!/usr/bin/env python3
"""CPU-only descriptive viewpoint coverage audit of the frozen DHT experiment.

Projected face-area ratios are image-space proxies, not calibrated camera angles.
No PnP fit, GT modification, feature inference, training or threshold selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import targets as T

ROOT = Path(__file__).resolve().parents[3]
FACES = {"front": (0, 3, 2, 1), **T.SIDE_FACES, "top": (0, 1, 5, 4)}
SIDE_BINS = (0, .1, .3, 1., float("inf"))
TOP_BINS = (0, 1., 2., 4., 8., float("inf"))
CELL_BINS = (0, 1., 2., 4., float("inf"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def q(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"),
                    np.percentile(values, [0, 10, 25, 50, 75, 90, 100]).tolist()))


def proxies(record):
    points = np.asarray(record["gt_points"], float)
    if not np.asarray(record["gt_valid"]).all() or not np.isfinite(points).all():
        raise ValueError(f"Incomplete cuboid for proxy: {record['id']}")
    areas = {}
    for name, indices in FACES.items():
        p = points[list(indices)]
        areas[name] = float(.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                                      - p[:, 1] * np.roll(p[:, 0], -1)))
    front = abs(areas["front"])
    if front < 1:
        raise ValueError(f"Degenerate front polygon: {record['id']}")
    segments = points[np.asarray(T.SIDE_EDGES)]
    delta = segments[:, 1] - segments[:, 0]
    lengths = np.linalg.norm(delta, axis=-1)
    grid_lengths = np.linalg.norm(delta * [50 / (record["width"] + 200),
                                          50 / (record["height"] + 200)], axis=-1)
    front_width = (np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[3])) / 2
    return {
        "id": record["id"], "population": record["population"], "group": record.get("group", ""),
        "side_exposure": max(-areas["left"], -areas["right"], 0) / front,
        "top_exposure": max(-areas["top"], 0) / front,
        "front_area_px2": front, "front_width_fraction": float(front_width / record["width"]),
        "height_length_px_median": float(np.median(lengths[:4])),
        "depth_length_px_median": float(np.median(lengths[4:])),
        "height_length_cells_median": float(np.median(grid_lengths[:4])),
        "depth_length_cells_median": float(np.median(grid_lengths[4:])),
        "camera_elevation_deg_metadata": record.get("conditions", {}).get("camera_elevation_deg"),
        "camera_azimuth_deg_metadata": record.get("conditions", {}).get("camera_azimuth_deg"),
        "length_px": lengths.tolist(), "length_cells": grid_lengths.tolist(),
    }


def summarize_frames(rows):
    out = {"n_frames": len(rows)}
    for key in ("side_exposure", "top_exposure", "front_width_fraction", "height_length_cells_median", "depth_length_cells_median"):
        out[key] = q([r[key] for r in rows])
    for arm in ("Direct", "DHT"):
        for key in ("all_distance_px", "all_angle_deg", "height_distance_px", "depth_distance_px"):
            values = [r[f"{arm}_{key}"] for r in rows]
            out[f"{arm}_{key}"] = {"frame_mean": float(np.mean(values)), "frame_median": float(np.median(values))} if values else None
    changes = [r["DHT_minus_Direct_all_distance_px"] for r in rows]
    out["DHT_minus_Direct_frame_mean_px"] = float(np.mean(changes)) if changes else None
    out["n_frames_DHT_distance_lower"] = sum(x < 0 for x in changes)
    return out


def bin_label(lo, hi):
    return f"[{lo:g},{hi:g})"


def correlate(rows, x, y):
    xx, yy = np.asarray([r[x] for r in rows]), np.asarray([r[y] for r in rows])
    if len(rows) < 3 or np.ptp(xx) == 0 or np.ptp(yy) == 0:
        return {"n": len(rows), "spearman_rho": None}
    return {"n": len(rows), "spearman_rho": float(spearmanr(xx, yy).statistic)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/pallet/results/deep_hough_side_v1")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg = json.loads((run / "CONFIG.json").read_text())
    source = Path(cfg["source_cache"]) / "manifest.json"
    manifest = json.loads(source.read_text())
    out = run / "diagnosis_viewpoint"
    out.mkdir(parents=True, exist_ok=True)
    records = [r for rows in manifest["populations"].values() for r in rows]
    geometry = {r["id"]: proxies(r) for r in records}
    errors = defaultdict(list)
    with (run / "PER_ROLE.csv").open() as stream:
        for row in csv.DictReader(stream):
            if int(row["seed"]) not in cfg["seeds"]:
                raise ValueError("Unexpected seed in errors")
            errors[(row["id"], row["arm"], int(row["role"]))].append(
                (int(row["seed"]), float(row["angle_deg"]), float(row["distance_px"])))
    line_rows = []
    for (identity, arm, role), values in errors.items():
        if {v[0] for v in values} != set(cfg["seeds"]) or len(values) != len(cfg["seeds"]):
            raise ValueError("Incomplete or duplicate role seeds")
        g = geometry[identity]
        line_rows.append({"id": identity, "population": g["population"], "group": g["group"],
                          "arm": arm, "role": role, "length_px": g["length_px"][role],
                          "length_cells": g["length_cells"][role],
                          "angle_deg_seed_mean": float(np.mean([v[1] for v in values])),
                          "distance_px_seed_mean": float(np.mean([v[2] for v in values]))})
    by_frame_arm = defaultdict(list)
    for r in line_rows:
        by_frame_arm[(r["id"], r["arm"])].append(r)
    frames = []
    for g in geometry.values():
        row = {k: v for k, v in g.items() if k not in ("length_px", "length_cells")}
        for arm in ("Direct", "DHT"):
            these = by_frame_arm[(g["id"], arm)]
            for subset, role_filter in (("all", lambda r: True), ("height", lambda r: r < 4), ("depth", lambda r: r >= 4)):
                selected = [r for r in these if role_filter(r["role"])]
                for metric in ("angle_deg", "distance_px"):
                    row[f"{arm}_{subset}_{metric}"] = float(np.mean([r[f"{metric}_seed_mean"] for r in selected]))
        row["DHT_minus_Direct_all_distance_px"] = row["DHT_all_distance_px"] - row["Direct_all_distance_px"]
        frames.append(row)
    train = [r for r in frames if r["population"] == "synth_train"]
    real = [r for r in frames if r["population"] == "real_dev"]
    ts = np.asarray([r["side_exposure"] for r in train])
    tt = np.asarray([r["top_exposure"] for r in train])
    tw = np.asarray([r["front_width_fraction"] for r in train])
    neighborhoods = []
    for r in real:
        ds = np.abs(ts - r["side_exposure"])
        dt = np.abs(np.log((1 + tt) / (1 + r["top_exposure"])))
        dw = np.abs(np.log(tw / r["front_width_fraction"]))
        near = (ds <= .1) & (dt <= np.log(1.25))
        broad = (ds <= .2) & (dt <= np.log(1.5))
        standardized = np.hypot(ds / .1, dt / np.log(1.25))
        nearest = np.argsort(standardized)[:5]
        r["nearby_train_pose_count"] = int(near.sum())
        r["nearby_train_pose_scale_count"] = int((near & (dw <= np.log(1.25))).sum())
        r["broad_nearby_train_pose_count"] = int(broad.sum())
        r["nearest_train_proxy_distance"] = float(standardized.min())
        neighborhoods.append({"id": r["id"], "group": r["group"],
                              **{k: r[k] for k in ("side_exposure", "top_exposure", "front_width_fraction",
                                                   "nearby_train_pose_count", "nearby_train_pose_scale_count",
                                                   "broad_nearby_train_pose_count", "nearest_train_proxy_distance")},
                              "five_nearest_training_examples": [{"id": train[i]["id"], "side_exposure": train[i]["side_exposure"],
                                                                   "top_exposure": train[i]["top_exposure"], "proxy_distance": float(standardized[i])} for i in nearest]})
    conditions = {
        "all": lambda r: True,
        "small_side_le_0.1": lambda r: r["side_exposure"] <= .1,
        "small_top_le_1": lambda r: r["top_exposure"] <= 1,
        "larger_top_gt_1": lambda r: r["top_exposure"] > 1,
        "small_side_and_small_top": lambda r: r["side_exposure"] <= .1 and r["top_exposure"] <= 1,
        "small_side_large_top": lambda r: r["side_exposure"] <= .1 and r["top_exposure"] > 1,
        "larger_side_gt_0.1": lambda r: r["side_exposure"] > .1,
    }
    populations = {}
    for pop in manifest["populations"]:
        these = [r for r in frames if r["population"] == pop]
        populations[pop] = {name: summarize_frames([r for r in these if pred(r)]) for name, pred in conditions.items()}
    bins = []
    for pop in manifest["populations"]:
        these = [r for r in frames if r["population"] == pop]
        for axis, edges in (("side_exposure", SIDE_BINS), ("top_exposure", TOP_BINS)):
            for lo, hi in zip(edges[:-1], edges[1:]):
                selected = [r for r in these if lo <= r[axis] < hi]
                stats = summarize_frames(selected)
                bins.append({"population": pop, "axis": axis, "bin": bin_label(lo, hi), "n_frames": len(selected),
                             "Direct_frame_mean_distance_px": stats["Direct_all_distance_px"]["frame_mean"] if selected else None,
                             "DHT_frame_mean_distance_px": stats["DHT_all_distance_px"]["frame_mean"] if selected else None,
                             "DHT_frame_mean_angle_deg": stats["DHT_all_angle_deg"]["frame_mean"] if selected else None,
                             "DHT_minus_Direct_frame_mean_px": stats["DHT_minus_Direct_frame_mean_px"]})
    length_bins = []
    for pop in manifest["populations"]:
        for arm in ("Direct", "DHT"):
            for lo, hi in zip(CELL_BINS[:-1], CELL_BINS[1:]):
                selected = [r for r in line_rows if r["population"] == pop and r["arm"] == arm and lo <= r["length_cells"] < hi]
                length_bins.append({"population": pop, "arm": arm, "length_cells_bin": bin_label(lo, hi),
                                    "n_unique_lines": len(selected), "n_frames": len({r["id"] for r in selected}),
                                    "angle_deg_seed_mean_median": float(np.median([r["angle_deg_seed_mean"] for r in selected])) if selected else None,
                                    "distance_px_seed_mean_median": float(np.median([r["distance_px_seed_mean"] for r in selected])) if selected else None})
    correlations = {x: correlate(real, x, "DHT_all_distance_px") for x in
                    ("side_exposure", "top_exposure", "front_width_fraction", "height_length_cells_median",
                     "depth_length_cells_median", "nearby_train_pose_count", "nearest_train_proxy_distance")}
    correlations["scope"] = "Descriptive Spearman correlations across 52 frames after role/seed averaging; no independent-line p-values or causal interpretation."
    elevations = [r["camera_elevation_deg_metadata"] for r in train]
    world_elevation, metadata_difference = [], []
    for record in manifest["populations"]["synth_train"]:
        annotation = json.loads(Path(record["annotation"]).read_text())
        center_world = np.asarray(annotation["objects"][0]["cuboid"], float)[:8].mean(0)
        camera_world = np.asarray(annotation["camera_data"]["location_worldframe"], float)
        delta = camera_world - center_world
        value = float(np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2]))))
        world_elevation.append(value)
        metadata_difference.append(value - record["conditions"]["camera_elevation_deg"])
    camera = {"train_elevation_deg": q(elevations),
              "train_elevation_counts": {f"below_{cut}_deg": sum(v < cut for v in elevations) for cut in (5, 10, 15, 20, 30)},
              "train_world_azimuth_deg": q([r["camera_azimuth_deg_metadata"] for r in train]),
              "real_angle_metadata_available": sum(r["camera_elevation_deg_metadata"] is not None for r in real),
              "world_geometry_crosscheck": {"n_frames": len(world_elevation),
                                            "definition": "atan2(camera_world.z-world_cuboid_centroid.z, norm(camera_world.xy-world_cuboid_centroid.xy)); camera elevation relative to cuboid center and horizontal world plane",
                                            "angle_deg": q(world_elevation),
                                            "max_abs_difference_from_metadata_deg": float(np.abs(metadata_difference).max()),
                                            "n_below_15_deg": sum(v < 15 for v in world_elevation)},
              "interpretation": "Synthetic renderer metadata corroborated against world cuboid/camera geometry. World azimuth is not relative frontality. Real angle not estimated from 2D proxy or fitted PnP."}
    report = {
        "schema": "deep_hough_side_viewpoint_diagnosis_v1",
        "input_sha256": {"manifest": sha(source), "PER_ROLE.csv": sha(run / "PER_ROLE.csv"), "CONFIG.json": sha(run / "CONFIG.json")},
        "proxy_definitions": {"side_exposure": "max(-signed_area(left),-signed_area(right),0)/abs(front_area)",
                              "top_exposure": "max(-signed_area(top),0)/abs(front_area)",
                              "front_width_fraction": "mean projected front upper/lower width divided by original image width",
                              "scope": "Projected geometric face-area ratios; shape, perspective and annotation affect them. Not physical visibility, yaw or elevation angles."},
        "bin_rule": "Fixed descriptive thresholds: side<=0.1; top<=1 means projected top no larger than front. Bins are not tuned to errors or used for model selection.",
        "matching_rule": {"near": "abs(side_train-side_real)<=0.1 and abs(log((1+top_train)/(1+top_real)))<=log(1.25)",
                          "near_with_scale": "near plus abs(log(front_width_fraction_train/front_width_fraction_real))<=log(1.25)",
                          "broad": "side tolerance0.2; transformed top ratio factor1.5",
                          "distance": "Euclidean norm of side diff/0.1 and log1p-top diff/log(1.25); dimensionless proxy distance, not degrees"},
        "populations": populations, "synthetic_camera_metadata": camera,
        "real_groups": {g: summarize_frames([r for r in real if r["group"] == g]) for g in sorted({r["group"] for r in real})},
        "real_neighborhoods": {"near_count": q([r["nearby_train_pose_count"] for r in real]),
                               "n_with_zero_near": sum(r["nearby_train_pose_count"] == 0 for r in real),
                               "n_with_zero_broad": sum(r["broad_nearby_train_pose_count"] == 0 for r in real),
                               "n_with_zero_near_pose_scale": sum(r["nearby_train_pose_scale_count"] == 0 for r in real)},
        "real_correlations": correlations,
        "target_selection": {"trained_side_edges": T.SIDE_EDGES, "left_new_roles": [1,3,4,7], "right_new_roles": [0,2,5,6],
                             "excluded_width_edges": [[0,1],[2,3],[4,5],[6,7]],
                             "ui": "Overview draws all8lines; role dropdown shows one role's sensitivity/parameter panel. Both side planes are output, full12cuboid is not.",
                             "frontal_geometry": "Front upper/lower width edges are excluded while front verticals are included. Depth connectors foreshorten near a frontal low view, reducing line evidence. A calibrated known-dimension front face with sufficient correctly corresponding corners can support pose/PnP plus hidden-edge projection, requiring ambiguity checks; one isolated visible corner cannot determine all hidden geometry."},
        "limitations": ["Post-hoc descriptive coverage/error association is not a data-augmentation or causality experiment.",
                        "Low top exposure and low side exposure are distinct; side exposure alone must not be called low elevation.",
                        "Synthetic elevation below15deg absent; real elevation not metrically estimated.",
                        "Near proxy counts do not match material, lighting, object scale unless stated, occlusion or image appearance.",
                        "Inherited backbone-pretraining overlap unverified; existing real DEV52, no independent final test.",
                        "Feature padding/resolution not ablated, so its effect cannot be measured from these outputs."]}
    write_json(out / "VIEWPOINT_DIAGNOSIS.json", report)
    write_json(out / "REAL_NEAREST_TRAIN.json", neighborhoods)
    columns = list(frames[0]) + [k for k in real[0] if k not in frames[0]]
    write_csv(out / "PER_FRAME.csv", [{k: r.get(k) for k in columns} for r in frames])
    write_csv(out / "BY_PROXY_BIN.csv", bins)
    write_csv(out / "BY_LINE_LENGTH.csv", length_bins)
    low, high = populations["real_dev"]["small_top_le_1"], populations["real_dev"]["larger_top_gt_1"]
    train_low = populations["synth_train"]["small_top_le_1"]["n_frames"]
    root_note = [
        "# 정면처럼 보이는 영상의 실패와 데이터 범위 진단", "",
        "이번 모델은 좌우 측면을 모두 출력합니다. 왼쪽4선+오른쪽4선이며, 갤러리의 overview는8선을 모두 그리고 dropdown 아래는 선택한1선의 상세 지도입니다. "
        "다만 ‘측면’을 앞에서 뒤로 이어지는 깊이 방향 면으로 정의했습니다. 앞면의 긴 위·아래 선(0–1,2–3)과 뒷면의 긴 위·아래 선(4–5,6–7)은 이번8head에 포함되지 않습니다. 전체cuboid12선을 예측한 실험이 아닙니다.", "",
        "## 어느 조건이 적은가", "",
        "투영된 측면/앞면 면적 비율≤0.1을 측면 노출이 작은 조건, 윗면/앞면 면적 비율≤1을 윗면 노출이 작은 조건으로 셌습니다. "
        "둘은 별개이며 이 비율을 실제 yaw·고도 각도로 바꾸지 않았습니다.", "",
        "| 조건 | 합성 학습2048 | 실사DEV52 |", "|---|---:|---:|",
        f"| 측면 노출 비율≤0.1 | {populations['synth_train']['small_side_le_0.1']['n_frames']} | {populations['real_dev']['small_side_le_0.1']['n_frames']} |",
        f"| 윗면 노출 비율≤1 | {train_low} | {low['n_frames']} |",
        f"| 두 조건 모두 | {populations['synth_train']['small_side_and_small_top']['n_frames']} | {populations['real_dev']['small_side_and_small_top']['n_frames']} |", "",
        f"윗면 노출이 작은 조건은 학습의 {100*train_low/len(train):.2f}%지만 실사의 {100*low['n_frames']/len(real):.1f}%입니다. "
        "합성 고도 metadata는 최소15.0001°·중앙33.8754°·최대60.1825°이며 15° 미만은0장입니다. "
        f"실제 world camera와 cuboid 중심으로 계산한 고도와 최대 {camera['world_geometry_crosscheck']['max_abs_difference_from_metadata_deg']:.8f}° 차이로 일치합니다. "
        "실사에는 이 고도 metadata가 없어서 실사 각도를 확정하지 않았습니다. 합성 world azimuth가0–360°에 걸쳐 있다는 사실만으로 물체 기준 정면 시점이 충분하다고 판단하지 않습니다.", "",
        "## 같은 고정 모델의 실제 오차", "",
        f"윗면 노출이 작은 실사{low['n_frames']}장의 DHT 프레임 평균 위치 오차는 {low['DHT_all_distance_px']['frame_mean']:.2f}px, "
        f"나머지{high['n_frames']}장은 {high['DHT_all_distance_px']['frame_mean']:.2f}px입니다. "
        "여기서 각 프레임의 지원 선을 평균하고3seed를 평균했습니다. 앞선 보고서의 선별 중앙값7.57px와 다른 통계입니다.", "",
        "**윗면 노출이 작은22장은 outdoor22와 정확히 같습니다.** 따라서 시점 부족과 야외 배경·조명·적재물 효과를 이 데이터만으로 분리할 수 없습니다. "
        f"측면 노출만 작고 윗면은 충분한26장은 DHT 평균 {populations['real_dev']['small_side_large_top']['DHT_all_distance_px']['frame_mean']:.2f}px로, "
        f"Direct보다 {-populations['real_dev']['small_side_large_top']['DHT_minus_Direct_frame_mean_px']:.2f}px 낮습니다. 정면처럼 보인다는 이유 하나로 항상 실패하는 것은 아닙니다.", "",
        "## 학습에서 비슷한 투영 형태를 실제로 찾아보기", "",
        "각 실사마다 `측면 비율 차이≤0.1`과 `1+윗면 비율의 비≤1.25배`를 동시에 만족하는 학습 이미지를 셌습니다. "
        f"{report['real_neighborhoods']['n_with_zero_near']}/52장은0장이고 모두 outdoor입니다. 범위를 측면0.2·윗면1.5배로 넓혀도 "
        f"{report['real_neighborhoods']['n_with_zero_broad']}/52장은0장입니다. 영상 크기 조건까지 넣으면 "
        f"{report['real_neighborhoods']['n_with_zero_near_pose_scale']}/52장은0장입니다. "
        "이것은 고정한2D기하 허용범위의 개수이며 재질·배경·가림까지 비슷한 사진의 개수는 아닙니다. 실사별 개수와 가장 가까운 학습5개 ID는 REAL_NEAREST_TRAIN.json에 있습니다.", "",
        "## 짧아진 선과 출력 대상도 영향을 준다", "",
        "현재 입력에서 GT 선15/414개(7프레임)는50×50 특징의1셀보다 짧습니다. 이15선은 DHT의3seed평균 각도 오차 중앙값24.10°·위치87.44px였고, "
        "4셀 이상158선은2.95°·8.37px였습니다. 짧은 선에는 깊이8개와 높이7개가 섞여 있습니다. 길이·시점·촬영 그룹이 함께 변하므로 이 비교도 단독 원인 실험은 아닙니다.", "",
        "정면에 가까운 낮은 시점에서는 깊이 방향 선이 짧아지거나 겹치는 반면 앞면의 긴 가로 경계가 더 직접적인 단서가 될 수 있습니다. "
        "이번 학습에서 제외한 앞면 위·아래 선을 포함하는 구조와, 보이는 앞면의 여러 코너·알려진 팔레트 치수·보정된 카메라로 pose를 구하고 가려진 경계를 투영하는 구조를 비교할 수 있습니다. "
        "후자는 대응점과 평면 pose의 모호성을 확인해야 하며 한 코너만으로 나머지 구조가 정해지는 것은 아닙니다.", "",
        "실험으로 확인된 것은 데이터 범위 부족과 큰 오차가 함께 나타난다는 점입니다. 낮은 시점 합성 추가, 전체12선 또는 앞면 지도 추가, 해상도/패딩 변경의 효과는 아직 재학습 비교로 확인하지 않았습니다.", "",
        "[분포·오차 그림](coverage_and_errors.png) · [전체 JSON](VIEWPOINT_DIAGNOSIS.json) · [프레임별 CSV](PER_FRAME.csv) · [선 길이별 CSV](BY_LINE_LENGTH.csv)",
    ]
    (out / "VIEWPOINT_FINDINGS.md").write_text("\n".join(root_note) + "\n")
    colors = {"eval_cad": "#3574bb", "eval_noapril": "#28a477", "eval_outside": "#d64f64"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax = axes[0, 0]
    ax.scatter(ts, tt, s=8, color="#a2a7ad", alpha=.3, label="Synthetic train (2048)")
    for group, color in colors.items():
        r = [x for x in real if x["group"] == group]
        ax.scatter([x["side_exposure"] for x in r], [x["top_exposure"] for x in r], s=38, edgecolors="black", linewidths=.3, color=color, label=group)
    ax.axhline(1, color="#777", linestyle="--"); ax.axvline(.1, color="#777", linestyle="--")
    ax.set(xlabel="Projected side area / front area", ylabel="Projected top area / front area", title="GT geometry coverage (ratios, not angles)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for group, color in colors.items():
        r = [x for x in real if x["group"] == group]
        ax.scatter([x["top_exposure"] for x in r], [x["DHT_all_distance_px"] for x in r], color=color, label=group)
    ax.axvline(1, color="#777", linestyle="--")
    ax.set(xlabel="Projected top / front area", ylabel="DHT mean line distance per frame (px)", title="Real DEV52: errors averaged over roles and 3 seeds")
    ax = axes[1, 0]
    ax.hist(elevations, bins=np.arange(0, 66, 5), color="#6f7d8f", edgecolor="white")
    ax.axvspan(0, 15, color="#e88e8e", alpha=.2)
    ax.set(xlabel="Renderer camera elevation metadata (degrees)", ylabel="Synthetic training frames", title="No synthetic training views below 15 degrees")
    ax = axes[1, 1]
    for group, color in colors.items():
        r = [x for x in real if x["group"] == group]
        ax.scatter([x["nearby_train_pose_count"] for x in r], [x["DHT_minus_Direct_all_distance_px"] for x in r], color=color)
    ax.axhline(0, color="#777", linestyle="--")
    ax.set(xlabel="Synthetic train neighbors under fixed 2D proxy tolerances", ylabel="DHT - Direct mean line error (px)", title="Coverage association does not establish cause")
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle("Pallet side-line viewpoint diagnosis: fixed models, no retraining", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out / "coverage_and_errors.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"camera": camera, "counts": {p: {k: v["n_frames"] for k,v in populations[p].items()} for p in ("synth_train","real_dev")},
                      "real_neighbors": report["real_neighborhoods"], "real_conditions": populations["real_dev"], "correlations": correlations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
