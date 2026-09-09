"""Fixed-model inference sensitivity to input padding; no training or selection.

Three float32-resize treatments keep geometry fixed: reflect101, black, and
ImageNet-mean constant padding. An additional canonical uint8-resize reflect
control checks the inherited feature cache and measures resize-rounding drift.
The models were trained with canonical reflect padding. This is an inference
perturbation, not a fair comparison of models trained with different padding.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

import runner as R


MODES = ("reflect_float32", "black_float32", "imagenet_mean_float32")
MEAN_BGR = (103.53, 116.28, 123.675)


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def scalar_stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"n": 0}
    return {"n": int(values.size), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "mean": float(np.mean(values)),
            "max": float(values.max())}


def prepared_modes(record, pad):
    canonical, _, _ = R.C.prepare(record, pad)
    original = cv2.imread(record["image"])
    if original is None or original.shape[:2] != (record["height"], record["width"]):
        raise ValueError(f"Image dimensions changed: {record['id']}")
    original = original.astype(np.float32)
    prepared = {"canonical_reflect_uint8": canonical}
    for mode in MODES:
        if mode == "reflect_float32":
            canvas = cv2.copyMakeBorder(original, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        else:
            fill = (0., 0., 0.) if mode == "black_float32" else MEAN_BGR
            canvas = cv2.copyMakeBorder(original, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
        rgb = cv2.cvtColor(cv2.resize(canvas, (400, 400)), cv2.COLOR_BGR2RGB)
        prepared[mode] = ((rgb / 255 - R.C.MEAN) / R.C.STD).transpose(2, 0, 1)
    return prepared


def line_motion(reference, variant, gt_points):
    """Compare lines on GT-endpoint projections onto the reference prediction."""
    points = np.asarray(gt_points)[np.asarray(R.T.SIDE_EDGES)]
    direction = reference[:, 1] - reference[:, 0]
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True).clip(1e-12)
    normal = np.stack([-direction[:, 1], direction[:, 0]], -1)
    residual = ((points - reference[:, :1]) * normal[:, None]).sum(-1)
    projected = points - residual[..., None] * normal[:, None]
    other = variant[:, 1] - variant[:, 0]
    other /= np.linalg.norm(other, axis=-1, keepdims=True).clip(1e-12)
    other_normal = np.stack([-other[:, 1], other[:, 0]], -1)
    motion = np.abs(((projected - variant[:, :1]) * other_normal[:, None]).sum(-1)).mean(-1)
    angle = np.rad2deg(np.arccos(np.abs((direction * other).sum(-1)).clip(0, 1)))
    return motion, angle


def border_sensitivity(run_dir):
    manifest = json.loads((run_dir / "visualization_manifest.json").read_text())
    rows = []
    for sample in manifest["samples"]:
        image = cv2.imread(sample["image"])
        h, w = image.shape[:2]
        pad = sample["pad_px"]
        # Treat each feature value as uniform within its grid cell. In padded
        # image pixel-center coordinates, a cell spans [j*w/50-.5,(j+1)*w/50-.5].
        xlo, xhi = np.arange(50) * w / 50 - .5, np.arange(1, 51) * w / 50 - .5
        ylo, yhi = np.arange(50) * h / 50 - .5, np.arange(1, 51) * h / 50 - .5
        xfrac = np.maximum(0, np.minimum(xhi, w-pad-.5) - np.maximum(xlo, pad-.5)) / (w/50)
        yfrac = np.maximum(0, np.minimum(yhi, h-pad-.5) - np.maximum(ylo, pad-.5)) / (h/50)
        roi_fraction = yfrac[:, None] * xfrac[None]
        xcenters = (np.arange(50)+.5) * w/50 - .5
        ycenters = (np.arange(50)+.5) * h/50 - .5
        center_roi = ((ycenters >= pad-.5) & (ycenters < h-pad-.5))[:, None] & (
            (xcenters >= pad-.5) & (xcenters < w-pad-.5))[None]
        with np.load(run_dir / sample["artifact"]) as artifact:
            maps = artifact["feature_sensitivity"].astype(float)
            support = artifact["support"].astype(bool)
        for role, values in enumerate(maps):
            total = values.sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError("Cannot normalize empty or invalid feature sensitivity")
            rows.append(dict(id=sample["id"], population=sample["population"], role=role,
                             supported=bool(support[role]),
                             border_fraction_fractional_cells=float((values*(1-roi_fraction)).sum()/total),
                             border_fraction_feature_centers=float(values[~center_roi].sum()/total),
                             border_area_fraction=float((1-roi_fraction).mean())))
    summary = {}
    for population in sorted({r["population"] for r in rows}):
        these = [r for r in rows if r["population"] == population and r["supported"]]
        summary[population] = {
            "n_frames": len({r["id"] for r in these}), "n_supported_maps": len(these),
            "border_fraction_fractional_cells": scalar_stats([r["border_fraction_fractional_cells"] for r in these]),
            "border_fraction_feature_centers": scalar_stats([r["border_fraction_feature_centers"] for r in these]),
            "border_area_fraction": float(np.mean([r["border_area_fraction"] for r in these])),
        }
    return {"rows": rows, "by_population": summary,
            "scope": "Existing predetermined12 gallery images, DHTseed1; supported role maps only in summaries. This is normalized absolute-gradient-times-feature sensitivity, not attention or causal input-pixel importance.",
            "approximation": "Fractional-cell estimate assumes each feature-map value is constant within its cell and weights its overlap with the original unpadded ROI. Binary alternative classifies feature centers. Receptive fields extend beyond individual cells, so neither estimates exact padding-pixel contribution."}


def original_batch_control(data, models, lattice, cfg, saved, probe_indices, mismatches, batch):
    """Recover original manifest batches and inspect any near-tied probe peaks."""
    report = {"reference_error_max_abs": 0., "reference_rows": 0, "changed_reference_rows": 0, "mismatches": [], "peak_controls": []}
    for arm, model in models.items():
        for population in ("real_dev", "synth_test"):
            result = R.evaluate(data, model, data.populations[population], cfg, lattice, arm, 1)
            for row in result:
                old = saved[(row["id"], row["role"], arm)]
                error = max(abs(row[metric]-float(old[metric])) for metric in ("angle_deg", "distance_px"))
                report["reference_rows"] += 1
                report["reference_error_max_abs"] = max(report["reference_error_max_abs"], error)
                report["changed_reference_rows"] += int(error > 1e-6)
                if error > 1e-6:
                    report["mismatches"].append(dict(id=row["id"], arm=arm, population=population, role=row["role"], maximum_error_difference=error))
    for mismatch in mismatches:
        pi = next(j for j, i in enumerate(probe_indices) if data.records[i]["id"] == mismatch["id"])
        index = probe_indices[pi]
        original_indices = data.populations[data.records[index]["population"]]
        oi = original_indices.index(index)
        original_batch = original_indices[oi//cfg["batch"]*cfg["batch"]:(oi//cfg["batch"]+1)*cfg["batch"]]
        probe_batch = probe_indices[pi//batch*batch:(pi//batch+1)*batch]
        a = models[mismatch["arm"]](data.batch(original_batch)[0])[original_batch.index(index), mismatch["role"]]
        b = models[mismatch["arm"]](data.batch(probe_batch)[0])[probe_batch.index(index), mismatch["role"]]
        a = a.masked_fill(~lattice.valid.flatten(), -1e9)
        b = b.masked_fill(~lattice.valid.flatten(), -1e9)
        av, ai = a.topk(2)
        bv, bi = b.topk(2)
        report["peak_controls"].append({
            **mismatch, "original_batch_size": len(original_batch), "probe_batch_size": len(probe_batch),
            "maximum_logit_difference": float((a-b).abs().max()),
            "original_top2_indices": ai.cpu().tolist(), "probe_top2_indices": bi.cpu().tolist(),
            "original_top2_logits": av.cpu().tolist(), "probe_top2_logits": bv.cpu().tolist(),
            "original_top2_gap": float(av[0]-av[1]), "probe_top2_gap": float(bv[0]-bv[1]),
        })
    report["PASS"] = report["changed_reference_rows"] == 0
    return report


@torch.no_grad()
def run(run_dir, batch=12):
    started = time.monotonic()
    cfg = json.loads((run_dir / "CONFIG.json").read_text())
    out = run_dir / "diagnosis_padding"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = Path(__file__).resolve().parent
    protected = [run_dir / name for name in ("CONFIG.json", "RESULTS.json", "PER_ROLE.csv")]
    protected += [core / name for name in ("runner.py", "network.py", "dht.py", "targets.py")]
    protected += [run_dir / "checkpoints" / f"{arm}_seed1_final.pth" for arm in ("Direct", "DHT")]
    protected_before = {str(p): R.C.sha(p) for p in protected}
    if R.C.sha(Path(cfg["backbone"])) != cfg["backbone_sha256"]:
        raise ValueError("Backbone weights differ from the original experiment")
    dh = R.C.load_dh()
    lattice = R.SparseDHT().to(device)
    data = R.Data(Path(cfg["source_cache"]), device)
    models = {arm: R.load_model(run_dir, arm, 1, lattice, dh, device)
              for arm in ("Direct", "DHT")}
    backbone = R.C.load_backbone(cfg["backbone"], device)
    synthetic = sorted(data.populations["synth_test"], key=lambda i: hashlib.sha256(
        ("dht-padding-probe-v1:"+data.records[i]["id"]).encode()).hexdigest())[:64]
    indices = list(data.populations["real_dev"]) + synthetic
    records = [data.records[i] for i in indices]
    write(out / "SELECTION.json", {
        "real_dev": [r["id"] for r in records if r["population"] == "real_dev"],
        "synth_test": [r["id"] for r in records if r["population"] == "synth_test"],
        "selection": "All52 real_dev + first64 synth_test by SHA256('dht-padding-probe-v1:'+id); independent of predictions",
    })
    predictions = {mode: {arm: [] for arm in models} for mode in ("cached_reflect", "canonical_reflect_uint8", *MODES)}
    parity = {"feature_max_abs": 0., "feature_abs_sum": 0., "feature_elements": 0, "feature_unequal": 0,
              "float_resize_feature_max_abs": 0., "models": {arm: {"logit_max_abs": 0., "decoded_bins_changed": 0} for arm in models}}
    for start in range(0, len(indices), batch):
        subset = indices[start:start+batch]
        prepared = [prepared_modes(data.records[i], cfg["pad_px"]) for i in subset]
        cached, _, _, _ = data.batch(subset)
        canonical_features = None
        cached_scores = {arm: model(cached) for arm, model in models.items()}
        for arm, scores in cached_scores.items():
            t, r = R.decode(scores, lattice)
            predictions["cached_reflect"][arm].extend(zip(t.cpu().numpy(), r.cpu().numpy()))
        for mode in ("canonical_reflect_uint8", *MODES):
            images = torch.from_numpy(np.stack([p[mode] for p in prepared])).to(device)
            features = backbone(images).half().float()
            if not torch.isfinite(features).all():
                raise ValueError("Nonfinite features in padding perturbation")
            if mode == "canonical_reflect_uint8":
                canonical_features = features
                diff = (features-cached).abs()
                parity["feature_max_abs"] = max(parity["feature_max_abs"], float(diff.max()))
                parity["feature_abs_sum"] += float(diff.double().sum())
                parity["feature_elements"] += diff.numel()
                parity["feature_unequal"] += int(torch.count_nonzero(diff))
            elif mode == "reflect_float32":
                parity["float_resize_feature_max_abs"] = max(parity["float_resize_feature_max_abs"], float((features-canonical_features).abs().max()))
            for arm, model in models.items():
                scores = model(features)
                theta, rho = R.decode(scores, lattice)
                predictions[mode][arm].extend(zip(theta.cpu().numpy(), rho.cpu().numpy()))
                if mode == "canonical_reflect_uint8":
                    reference_theta, reference_rho = R.decode(cached_scores[arm], lattice)
                    current = parity["models"][arm]
                    current["logit_max_abs"] = max(current["logit_max_abs"], float((scores-cached_scores[arm]).abs().max()))
                    current["decoded_bins_changed"] += int(torch.count_nonzero((theta != reference_theta) | (rho != reference_rho)))
        R.log(f'Padding probe extracted/evaluated {start+len(subset)}/{len(indices)} images x4 preprocessing modes')
    parity["feature_mean_abs"] = parity.pop("feature_abs_sum")/parity["feature_elements"]
    parity["feature_unequal_fraction"] = parity["feature_unequal"]/parity["feature_elements"]
    parity["feature_bit_exact"] = parity["feature_unequal"] == 0
    rows, frame_rows = [], []
    saved = {}
    with (run_dir / "PER_ROLE.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["seed"] == "1":
                saved[(row["id"], int(row["role"]), row["arm"])] = row
    parity["saved_error_max_abs"] = 0.
    parity["real_saved_error_max_abs"] = 0.
    parity["saved_error_mismatches"] = []
    for index, record in enumerate(records):
        support = data.support[indices[index]].cpu().numpy()
        for arm in models:
            line_maps = {mode: R.T.line_pixels(*predictions[mode][arm][index], record["width"], record["height"])
                         for mode in predictions}
            cached_errors = R.T.pixel_errors(line_maps["cached_reflect"], record["gt_points"])
            for mode, lines in line_maps.items():
                angle, distance = R.T.pixel_errors(lines, record["gt_points"])
                motion, angle_motion = line_motion(line_maps["reflect_float32"], lines, record["gt_points"])
                these = []
                for role in np.flatnonzero(support):
                    old = saved[(record["id"], int(role), arm)]
                    if mode == "cached_reflect":
                        difference = max(abs(float(angle[role])-float(old["angle_deg"])), abs(float(distance[role])-float(old["distance_px"])))
                        parity["saved_error_max_abs"] = max(parity["saved_error_max_abs"], difference)
                        if record["population"] == "real_dev":
                            parity["real_saved_error_max_abs"] = max(parity["real_saved_error_max_abs"], difference)
                        if difference > 1e-6:
                            parity["saved_error_mismatches"].append(dict(id=record["id"], arm=arm, role=int(role),
                                angle=float(angle[role]), old_angle=float(old["angle_deg"]), distance=float(distance[role]), old_distance=float(old["distance_px"])))
                    row = dict(id=record["id"], population=record["population"], group=record.get("group", ""),
                               arm=arm, mode=mode, role=int(role), angle_deg=float(angle[role]), distance_px=float(distance[role]),
                               line_movement_from_float_reflect_px=float(motion[role]),
                               angle_movement_from_float_reflect_deg=float(angle_motion[role]),
                               error_change_from_cached_reflect_px=float(distance[role]-cached_errors[1][role]))
                    rows.append(row)
                    these.append(row)
                frame_rows.append(dict(id=record["id"], population=record["population"], group=record.get("group", ""), arm=arm, mode=mode,
                                       n_roles=len(these), **{key: float(np.mean([r[key] for r in these])) for key in
                                       ("distance_px", "angle_deg", "line_movement_from_float_reflect_px", "angle_movement_from_float_reflect_deg", "error_change_from_cached_reflect_px")}))
    if parity["saved_error_max_abs"] > 1e-6:
        parity["original_batch_control"] = original_batch_control(data, models, lattice, cfg, saved, indices,
                                                                  parity["saved_error_mismatches"], batch)
        parity["primary_real_and_selected_DHT_PARITY_PASS"] = all(
            p["arm"] == "Direct" and p["id"].startswith("paper_") for p in parity["saved_error_mismatches"])
        if not parity["primary_real_and_selected_DHT_PARITY_PASS"]:
            write(out / "PARITY_FAILURE.json", parity)
            raise ValueError("Primary real or selected DHT cached-input parity failed")
    summary = {}
    for population in ("real_dev", "synth_test"):
        summary[population] = {}
        for arm in models:
            summary[population][arm] = {}
            for mode in predictions:
                these = [r for r in rows if r["population"] == population and r["arm"] == arm and r["mode"] == mode]
                frames = [r for r in frame_rows if r["population"] == population and r["arm"] == arm and r["mode"] == mode]
                summary[population][arm][mode] = {"n_frames": len(frames), "n_roles": len(these),
                    **{metric: scalar_stats([r[metric] for r in these]) for metric in ("angle_deg", "distance_px")},
                    "frame_mean_error": scalar_stats([r["distance_px"] for r in frames]),
                    "frame_mean_line_movement_from_float_reflect_px": scalar_stats([r["line_movement_from_float_reflect_px"] for r in frames]),
                    "frame_mean_error_change_from_cached_reflect_px": scalar_stats([r["error_change_from_cached_reflect_px"] for r in frames])}
    by_group = {}
    for group in sorted({r["group"] for r in rows if r["population"] == "real_dev"}):
        by_group[group] = {}
        for arm in models:
            by_group[group][arm] = {}
            for mode in predictions:
                these = [r for r in rows if r["group"] == group and r["arm"] == arm and r["mode"] == mode]
                frames = [r for r in frame_rows if r["group"] == group and r["arm"] == arm and r["mode"] == mode]
                by_group[group][arm][mode] = {"n_frames": len(frames), "n_roles": len(these),
                    **{metric: scalar_stats([r[metric] for r in these]) for metric in ("angle_deg", "distance_px")},
                    "frame_mean_error": scalar_stats([r["distance_px"] for r in frames])}
    sensitivity = border_sensitivity(run_dir)
    protected_after = {str(p): R.C.sha(p) for p in protected}
    if protected_before != protected_after:
        raise ValueError("Protected training/results files changed during padding probe")
    report = {
        "schema": "deep_hough_padding_inference_probe_v1", "seed": 1, "training_padding": "uint8 BORDER_REFLECT_101 100px",
        "scope": "Frozen-model inference perturbation. Both models were trained with reflection. Not a matched padding-training comparison, model selection, or independent real final test.",
        "selection": "All52 existing real_dev and fixed SHA-first64 synth_test. See SELECTION.json.",
        "image_geometry": "Same100pxpadding,400x400resize, frozen128x50x50VGG, correctedtargets, and checkpoints in every mode; no GT used in model inputs.",
        "pixel_preprocessing": "Primary3modes pad and resize float32 BGR, then RGB/255 and same ImageNet mean/std. Additional canonical uint8 reflect control matches C.prepare before FP16-feature-then-FP32 quantization.",
        "mean_padding_BGR": MEAN_BGR, "normalization_RGB_mean": R.C.MEAN.tolist(), "normalization_RGB_std": R.C.STD.tolist(),
        "reflect_reference_policy": "Originalcachedreflect is retained for previous-result reproducibility; padding-effect comparison usesfloat32reflect to isolate changes fromfloat32black/mean. Canonical re-extraction drift and floatresize drift are both reported.",
        "line_movement_definition": "Project GT segment endpoints onto float32-reflect predicted line, then average their perpendicular distances to perturbed predicted line; compare original-image normal angles as well. GT provides evaluation anchors only.",
        "cache_parity": parity, "population_summary": summary, "real_by_group": by_group,
        "frame_rows": frame_rows, "border_feature_sensitivity": sensitivity,
        "protected_files_unchanged": True, "protected_sha256": protected_before,
        "probe_script_sha256": R.C.sha(Path(__file__)), "device": str(device), "elapsed_seconds": time.monotonic()-started,
    }
    write(out / "PAD_PROBE.json", report)
    with (out / "PER_ROLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    notes = ["# 반사 패딩을 빈 패딩으로 바꾼 추론 진단", "",
        "반사 패딩으로 이미 학습한 Direct와 DHT의 seed 1을 고정한 채 입력 테두리만 바꿨습니다. "
        "세 패딩으로 각각 다시 학습한 공정한 비교가 아니므로, 이 결과로 어느 패딩의 학습 성능이 더 좋다고 결론낼 수 없습니다.", "",
        "실사 개발셋 전체 52장과 사전에 SHA 순서로 고른 합성 평가 64장입니다. 패딩 크기·리사이즈·정답 좌표·모델 가중치는 같습니다. "
        "주 비교 세 조건은 모두 float32로 패딩·리사이즈하고 같은 정규화를 적용했습니다. 평균색은 BGR=(103.53,116.28,123.675)로 정규화 후 거의 0입니다.", "",
        "| 자료 | 모델 | 패딩 | 위치 중앙값 px | 위치 P90 px | 방향 중앙값 ° | 방향 P90 ° | 프레임 평균 위치오차 px |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    labels = {"reflect_float32": "반사", "black_float32": "검정", "imagenet_mean_float32": "ImageNet 평균색"}
    for population in summary:
        for arm in models:
            for mode in MODES:
                s = summary[population][arm][mode]
                notes.append(f"| {population} | {arm} | {labels[mode]} | {s['distance_px']['median']:.2f} | {s['distance_px']['p90']:.2f} | {s['angle_deg']['median']:.2f} | {s['angle_deg']['p90']:.2f} | {s['frame_mean_error']['mean']:.2f} |")
    notes += ["", "이 표는 단일 seed에서 각 조건의 평가 역할을 모은 통계입니다. 이전 보고서의 3개 seed 평균과 직접 비교하지 않습니다.", "",
        f"캐시 재현 확인: uint8 반사 입력을 다시 추출한 특징의 최대 절대차 {parity['feature_max_abs']:.8g}, "
        f"원소 불일치 비율 {parity['feature_unequal_fraction']:.8g}; 실사 원본 캐시 입력의 저장된 선 오차 재현 최대차 {parity['real_saved_error_max_abs']:.8g}입니다. "
        "float32 반사와 기존 uint8 반사를 별도 보관해, 리사이즈 반올림 차이를 패딩 효과로 오해하지 않도록 했습니다.", "",
        "| 실사 DHT 그룹 | 패딩 | 위치 중앙값 px | 위치 P90 px | 프레임 평균 위치오차 px |", "|---|---|---:|---:|---:|"]
    for group, group_data in by_group.items():
        for mode in MODES:
            s = group_data["DHT"][mode]
            notes.append(f"| {group} | {labels[mode]} | {s['distance_px']['median']:.2f} | {s['distance_px']['p90']:.2f} | {s['frame_mean_error']['mean']:.2f} |")
    if parity["saved_error_mismatches"]:
        control = parity["original_batch_control"]
        notes += ["", f"재현 대조: probe에서 기존 저장 예측과 달라진 역할은 {len(parity['saved_error_mismatches'])}개입니다. "
                  f"원래 모집단 순서·batch 12로 별도 검사한 {control['reference_rows']:,}개 역할 중에서도 "
                  f"{control['changed_reference_rows']}개가 달랐습니다. "
                  "차이는 합성 Direct의 최고 점수가 매우 가까운 후보에서 발생했고, 실사 및 이 probe의 DHT 예측은 저장 결과와 일치합니다. "
                  "최고 2개 logit 간격과 배치 간 logit 차이를 JSON에 보관했으며 원인을 배치 순서로 단정하지 않습니다. "
                  "모든 패딩 조건은 같은 실행 환경·배치로 비교했고, 이 재현 차이를 패딩 효과로 해석하지 않습니다."]
    notes += ["", "선이 얼마나 바뀌었는지도 같은 정답 끝점을 반사 조건의 예측 선에 투영한 위치에서 측정했습니다. "
              "이는 새 선까지의 수직거리이며 단순한 오차 증감과는 다른 값입니다.", ""]
    for arm in models:
        for mode in MODES[1:]:
            s = summary["real_dev"][arm][mode]["frame_mean_line_movement_from_float_reflect_px"]
            notes.append(f"- 실사 {arm}, {labels[mode]}: 프레임 평균 선 이동량의 중앙값 {s['median']:.2f}px, P90 {s['p90']:.2f}px.")
    notes += ["", "## 이미 저장된 특징 민감도에서 테두리 비율", "",
              "사전에 선택한 갤러리 12장의 DHT 민감도 맵을 사용했습니다. 각 특징 셀 안에서 값이 균일하다고 근사해 "
              "원본 영역과 겹치는 면적을 계산했습니다. 특징 중심이 테두리에 속하는지만 세는 결과도 JSON에 함께 있습니다.", "",
              "| 표본 | 장수 | 유효 역할 맵 | 테두리 민감도 비율 중앙값 | 평균 | 테두리 면적 비율 |", "|---|---:|---:|---:|---:|---:|"]
    for population, s in sensitivity["by_population"].items():
        v = s["border_fraction_fractional_cells"]
        notes.append(f"| {population} | {s['n_frames']} | {s['n_supported_maps']} | {v['median']*100:.1f}% | {v['mean']*100:.1f}% | {s['border_area_fraction']*100:.1f}% |")
    notes += ["", "민감도는 `|gradient × feature|`의 상대적 분포입니다. Attention이나 원본 패딩 픽셀의 인과적 기여도가 아닙니다. "
              "각 특징의 수용영역은 셀 밖까지 퍼지므로 이 비율로 '테두리 픽셀에 의존한 정확한 비중'을 계산할 수 없습니다. "
              "조건 변경으로 오차가 바뀌는 것은 입력 의존성을 보여주지만, 반사된 물체 복사본의 혼동 때문인지 경계 불연속이나 "
              "학습·추론 분포 차이 때문인지는 이 비교 하나로 분리되지 않습니다.", "",
              "[전체 수치와 프레임 이동](PAD_PROBE.json) · [역할별 CSV](PER_ROLE.csv) · [표본 목록](SELECTION.json)", ""]
    (out / "PAD_FINDINGS.md").write_text("\n".join(notes))
    R.log(f'Padding probe completed: {out}; elapsed {report["elapsed_seconds"]:.1f}s')
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.batch)
