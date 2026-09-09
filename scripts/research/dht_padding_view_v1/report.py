#!/usr/bin/env python3
"""Build the offline padding/data-mix experiment plan and verified results.

Main results remain NOT_RUN until completed main runs pass identity, final-step,
and artifact-hash checks. Isolated smoke metrics are never pooled into them.
Open the local dashboard by default; --no-open writes files only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import itertools
import json
import math
import shlex
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AXES = {
    "elevation_bin": ["elev_lt5", "elev_5_15", "elev_15_30", "elev_ge30", "unknown"],
    "side_bin": ["side_le0p1", "side_0p1_0p3", "side_gt0p3", "unknown"],
    "top_bin": ["top_le1", "top_1_4", "top_gt4", "unknown"],
    "size_bin": ["width_lt0p25", "width_0p25_0p5", "width_0p5_1", "width_ge1", "unknown"],
    "yaw_bin": ["yaw_lt5", "yaw_5_15", "yaw_15_30", "yaw_ge30", "unknown"],
}
LABELS = {
    "elev_lt5": "고도 <5°", "elev_5_15": "고도 5–15°", "elev_15_30": "고도 15–30°",
    "elev_ge30": "고도 ≥30°", "unknown": "정보 없음",
    "side_le0p1": "측면 비율 ≤0.1", "side_0p1_0p3": "측면 비율 0.1–0.3", "side_gt0p3": "측면 비율 >0.3",
    "top_le1": "윗면 비율 ≤1", "top_1_4": "윗면 비율 1–4", "top_gt4": "윗면 비율 >4",
    "width_lt0p25": "영상 폭의 <25%", "width_0p25_0p5": "영상 폭의 25–50%",
    "width_0p5_1": "영상 폭의 50–100%", "width_ge1": "영상 폭의 ≥100%",
    "yaw_lt5": "정면 yaw <5°", "yaw_5_15": "정면 yaw 5–15°",
    "yaw_15_30": "정면 yaw 15–30°", "yaw_ge30": "정면 yaw ≥30°",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def open_dashboard(path):
    """Reuse the user's project-level local-browser launcher without training imports."""
    source = HERE.parent / "deep_hough_side_v1/visualize.py"
    spec = importlib.util.spec_from_file_location("_dht_local_gallery_launcher", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.open_gallery(Path(path))


def combinations(cfg):
    training = cfg["training"]
    return itertools.product(training["regimes"], cfg["conditions"], training["seeds"])


def bin_value(cohort, axis):
    value = cohort.get(axis) or "unknown"
    if value not in AXES[axis]:
        raise ValueError(f"Unknown {axis} label: {value}")
    return value


def coverage(count):
    return "ABSENT" if count == 0 else "SMALL" if count < 20 else "PRESENT"


def derived_cohort(record):
    """Add a report-only true yaw bin; never modify frozen source records."""
    cohort = dict(record.get("cohort", {}))
    yaw = cohort.get("frontal_yaw_deg_world")
    if yaw is None or not math.isfinite(float(yaw)):
        cohort["yaw_bin"] = "unknown"
    else:
        yaw = abs(float(yaw))
        cohort["yaw_bin"] = "yaw_lt5" if yaw < 5 else "yaw_5_15" if yaw < 15 else "yaw_15_30" if yaw < 30 else "yaw_ge30"
    return cohort


def cohort_counts(manifest):
    records = manifest["records"]
    groups = {f"train / {name}": indices for name, indices in manifest["train_regimes"].items()}
    groups.update({f"{'pool' if name.endswith('_train') else 'eval'} / {name}": indices
                   for name, indices in manifest["populations"].items() if name != "synth_train"})
    output = {}
    for name, indices in groups.items():
        rows = [{**records[index], "cohort": derived_cohort(records[index])} for index in indices]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate frame IDs in cohort: {name}")
        bins = {axis: Counter(bin_value(row.get("cohort", {}), axis) for row in rows) for axis in AXES}
        cells = []
        for elevation, side in itertools.product(AXES["elevation_bin"], AXES["side_bin"]):
            members = [row for row in rows if bin_value(row.get("cohort", {}), "elevation_bin") == elevation
                       and bin_value(row.get("cohort", {}), "side_bin") == side]
            cells.append({"elevation_bin": elevation, "side_bin": side, "n_frames": len(members),
                          "coverage_status": coverage(len(members))})
        top_cells = []
        for top, side in itertools.product(AXES["top_bin"], AXES["side_bin"]):
            count = sum(bin_value(row.get("cohort", {}), "top_bin") == top
                        and bin_value(row.get("cohort", {}), "side_bin") == side for row in rows)
            top_cells.append({"top_bin": top, "side_bin": side, "n_frames": count,
                              "coverage_status": coverage(count)})
        output[name] = {"n_frames": len(rows), "elevation_x_side": cells, "top_x_side": top_cells,
                        "bins": {axis: {value: bins[axis][value] for value in AXES[axis]} for axis in AXES}}
    return output


def scalar_summary(rows, cfg):
    if not rows:
        return {"n_frames": 0, "n_roles": 0, "angle_deg": None, "distance_px": None, "success_fraction": None}
    result = {"n_frames": len({row["id"] for row in rows}), "n_roles": len(rows)}
    for metric in ("angle_deg", "distance_px"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {"median": float(np.median(values)), "p90": float(np.percentile(values, 90))}
    evaluation = cfg["evaluation"]
    result["success_fraction"] = float(np.mean([
        row["angle_deg"] <= evaluation["success_angle_deg"]
        and row["distance_px"] <= evaluation["success_distance_px"] for row in rows]))
    return result


def read_verified_run(run_dir, cfg, regime, condition, seed, manifest=None):
    directory = run_dir / "runs" / regime / condition / f"seed_{seed}"
    result = {"regime": regime, "condition": condition, "seed": seed,
              "path": str(directory.relative_to(run_dir)), "status": "NOT_RUN", "errors": []}
    completion_path = directory / "COMPLETION.json"
    if not completion_path.exists():
        if any((directory / name).exists() for name in ("INITIALIZATION.json", "history.json", "checkpoint_latest.pth", "checkpoint_final.pth", "metrics.csv")):
            result["status"] = "INCOMPLETE"
        return result, []
    try:
        completion = read(completion_path)
        cache_path = run_dir / "cache" / condition / "CACHE.json"
        cache = read(cache_path)
        old = HERE.parent / "deep_hough_side_v1"
        core = {"train_eval.py": sha(HERE / "train_eval.py"),
                **{f"reused/{name}": sha(old / name) for name in ("dht.py", "network.py", "targets.py")}}
        checks = {
            "PASS": completion.get("PASS") is True,
            "training_complete": completion.get("training_complete") is True,
            "evaluation_complete": completion.get("evaluation_complete") is True,
            "stage": completion.get("stage") == cfg["stage"],
            "identity": (completion.get("regime"), completion.get("condition"), completion.get("seed")) == (regime, condition, seed),
            "final_step": completion.get("expected_steps") == completion.get("final_step") == completion.get("history_final_step") == cfg["training"]["steps"],
            "main_budget": cfg["stage"] != "main" or cfg["training"]["steps"] == 6000,
            "config_sha256": completion.get("config_sha256") == sha(run_dir / "CONFIG.json"),
            "manifest_sha256": completion.get("manifest_sha256") == sha(run_dir / "manifest.json"),
            "metrics_csv_sha256": completion.get("metrics_csv_sha256") == sha(directory / "metrics.csv"),
            "checkpoint_sha256": completion.get("checkpoint_sha256") == sha(directory / "checkpoint_final.pth"),
            "core_code_sha256": completion.get("core_code_sha256") == core,
            "cache_metadata_sha256": completion.get("cache_metadata_sha256") == sha(cache_path),
            "cache_signature_sha256": completion.get("cache_signature_sha256") == hashlib.sha256(
                json.dumps(cache["signature"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }
        result["verification"] = checks
        if not all(checks.values()):
            raise ValueError("Failed completion checks: " + ", ".join(key for key, passed in checks.items() if not passed))
        manifest = manifest if manifest is not None else read(run_dir / "manifest.json")
        cohorts_by_id = {record["id"]: derived_cohort(record) for record in manifest["records"]}
        rows, seen = [], set()
        with (directory / "metrics.csv").open(newline="") as stream:
            for raw in csv.DictReader(stream):
                row = dict(raw)
                row["seed"], row["role"] = int(raw["seed"]), int(raw["role"])
                if (row["regime"], row["condition"], row["seed"]) != (regime, condition, seed):
                    raise ValueError("CSV run identity mismatch")
                if row["population"] not in cfg["evaluation"]["populations"] or not 0 <= row["role"] < 8:
                    raise ValueError("Unexpected CSV population or role")
                for key in ("angle_deg", "distance_px", "frame_diagonal"):
                    row[key] = float(raw[key])
                    if not math.isfinite(row[key]) or row[key] < 0:
                        raise ValueError(f"Invalid CSV metric: {key}")
                row["camera_facing"] = str(raw["camera_facing"]).lower() in ("true", "1")
                for axis in AXES:
                    row[axis] = bin_value(cohorts_by_id[row["id"]] if axis == "yaw_bin" else raw, axis)
                identity = row["population"], row["id"], row["role"]
                if identity in seen:
                    raise ValueError("Duplicate CSV frame-role row")
                seen.add(identity)
                rows.append(row)
        if not rows:
            raise ValueError("Completed run contains no metric rows")
        expected_populations = set(cfg["evaluation"]["populations"])
        if set(completion.get("evaluated_populations", {})) != expected_populations:
            raise ValueError("Completion population set differs from configuration")
        for population in expected_populations:
            count = sum(row["population"] == population for row in rows)
            if completion["evaluated_populations"][population]["n_roles"] != count:
                raise ValueError(f"CSV role count differs from completion: {population}")
        result.update(status="COMPLETE", final_step=completion["final_step"],
                      completion_sha256=sha(completion_path), n_metric_rows=len(rows))
        return result, rows
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status="INVALID", errors=[str(exc)])
        return result, []


def stratified_summary(subset, cfg):
    result = scalar_summary(subset, cfg)
    result["height"] = scalar_summary([row for row in subset if row["role"] < 4], cfg)
    result["depth"] = scalar_summary([row for row in subset if row["role"] >= 4], cfg)
    result["camera_facing"] = scalar_summary([row for row in subset if row["camera_facing"]], cfg)
    result["by_view"] = {
        axis: {value: scalar_summary([row for row in subset if row[axis] == value], cfg) for value in AXES[axis]}
        for axis in AXES}
    return result


def run_summaries(rows, cfg):
    output = {}
    for population in cfg["evaluation"]["populations"]:
        subset = [row for row in rows if row["population"] == population]
        result = stratified_summary(subset, cfg)
        result["by_group"] = {group: stratified_summary([row for row in subset if row["group"] == group], cfg)
                              for group in sorted({row["group"] for row in subset})}
        output[population] = result
    return output


def across_seeds(runs, cfg):
    grouped = defaultdict(list)
    for run in runs:
        for population, values in run.get("populations", {}).items():
            grouped[(run["regime"], run["condition"], population)].append((run["seed"], values))
    result = []
    for (regime, condition, population), entries in grouped.items():
        item = {"regime": regime, "condition": condition, "population": population,
                "seeds": sorted(seed for seed, _ in entries), "expected_seeds": cfg["training"]["seeds"],
                "aggregation": "Mean and [min,max] of per-seed statistics, not pooled-line metrics or confidence intervals.",
                "metrics": {}}
        for metric, statistic in [(m, s) for m in ("angle_deg", "distance_px") for s in ("median", "p90")]:
            values = [p[metric][statistic] for _, p in entries if p[metric] is not None]
            item["metrics"][f"{metric}_{statistic}"] = {"mean": float(np.mean(values)), "min": min(values), "max": max(values)} if values else None
        values = [p["success_fraction"] for _, p in entries if p["success_fraction"] is not None]
        item["metrics"]["success_fraction"] = {"mean": float(np.mean(values)), "min": min(values), "max": max(values)} if values else None
        result.append(item)
    return result


def paired_padding(runs, raw_rows):
    """Compare the same image-role predictions against reflect100, by seed."""
    output = []
    completed = {(run["regime"], run["condition"], run["seed"]) for run in runs if run["status"] == "COMPLETE"}
    for regime, condition, seed in sorted(completed):
        reference = (regime, "reflect100", seed)
        if condition == "reflect100" or reference not in completed:
            continue
        a = {(r["population"], r["id"], r["role"]): r for r in raw_rows[reference]}
        b = {(r["population"], r["id"], r["role"]): r for r in raw_rows[(regime, condition, seed)]}
        common = sorted(a.keys() & b.keys())
        for population in sorted({key[0] for key in common}):
            keys = [key for key in common if key[0] == population]
            shifts = [b[key]["distance_px"] - a[key]["distance_px"] for key in keys]
            output.append({"regime": regime, "condition": condition, "reference": "reflect100", "seed": seed,
                           "population": population, "n_matched_frames": len({key[1] for key in keys}),
                           "n_matched_roles": len(keys), "n_unmatched_roles": len((a.keys() ^ b.keys()) & {key for key in a.keys() | b.keys() if key[0] == population}),
                           "distance_delta_median_px": float(np.median(shifts)),
                           "distance_delta_mean_px": float(np.mean(shifts)),
                           "distance_improved_fraction": float(np.mean(np.asarray(shifts) < 0))})
    return output


def paired_regimes(runs, raw_rows):
    """Matched evaluation rows for low_balanced minus base within each padding."""
    output = []
    completed = {(run["regime"], run["condition"], run["seed"]) for run in runs if run["status"] == "COMPLETE"}
    for regime, condition, seed in sorted(completed):
        if regime != "low_balanced" or ("base", condition, seed) not in completed:
            continue
        a = {(r["population"], r["id"], r["role"]): r for r in raw_rows[("base", condition, seed)]}
        b = {(r["population"], r["id"], r["role"]): r for r in raw_rows[(regime, condition, seed)]}
        common = sorted(a.keys() & b.keys())
        for population in sorted({key[0] for key in common}):
            keys = [key for key in common if key[0] == population]
            shifts = [b[key]["distance_px"] - a[key]["distance_px"] for key in keys]
            output.append({"condition": condition, "seed": seed, "population": population,
                           "reference": "base", "comparison": "low_balanced",
                           "n_matched_frames": len({key[1] for key in keys}), "n_matched_roles": len(keys),
                           "n_unmatched_roles": sum(key[0] == population for key in a.keys() ^ b.keys()),
                           "distance_delta_median_px": float(np.median(shifts)),
                           "distance_delta_mean_px": float(np.mean(shifts))})
    return output


def build_matrix(run_dir):
    cfg, manifest = read(run_dir / "CONFIG.json"), read(run_dir / "manifest.json")
    if cfg["stage"] not in ("main", "smoke"):
        raise ValueError("CONFIG stage must be main or smoke")
    runs, raw = [], {}
    for regime, condition, seed in combinations(cfg):
        run, rows = read_verified_run(run_dir, cfg, regime, condition, seed, manifest)
        if rows and cfg["stage"] == "main":
            run["populations"] = run_summaries(rows, cfg)
            raw[(regime, condition, seed)] = rows
        runs.append(run)
    completed = sum(run["status"] == "COMPLETE" for run in runs)
    main_completed = completed if cfg["stage"] == "main" else 0
    invalid = sum(run["status"] == "INVALID" for run in runs)
    incomplete = sum(run["status"] == "INCOMPLETE" for run in runs)
    main_status = ("NOT_RUN" if cfg["stage"] != "main" else "FAILED" if invalid else
                   "INCOMPLETE" if incomplete and main_completed == 0 else
                   "NOT_RUN" if main_completed == 0 else "COMPLETE" if main_completed == len(runs) else "PARTIAL")
    smoke = {"status": "NOT_RUN", "complete_runs": 0, "expected_runs": 0, "runs": []}
    smoke_dir = run_dir / "smoke"
    if (smoke_dir / "CONFIG.json").exists() and (smoke_dir / "manifest.json").exists():
        smoke_cfg = read(smoke_dir / "CONFIG.json")
        if smoke_cfg.get("stage") != "smoke":
            smoke["status"] = "INVALID_STAGE"
        else:
            smoke_manifest = read(smoke_dir / "manifest.json")
            smoke_runs = [read_verified_run(smoke_dir, smoke_cfg, *combo, smoke_manifest)[0] for combo in combinations(smoke_cfg)]
            n_done = sum(run["status"] == "COMPLETE" for run in smoke_runs)
            smoke = {"status": "COMPLETE" if n_done == len(smoke_runs) else "NOT_RUN" if n_done == 0 else "PARTIAL",
                     "complete_runs": n_done, "expected_runs": len(smoke_runs), "runs": smoke_runs,
                     "steps_per_run": smoke_cfg["training"]["steps"], "scope": "Implementation smoke only; no main metrics."}
    if cfg["stage"] == "smoke":
        smoke = {"status": "COMPLETE" if completed == len(runs) else "PARTIAL" if completed else "NOT_RUN",
                 "complete_runs": completed, "expected_runs": len(runs), "runs": runs,
                 "steps_per_run": cfg["training"]["steps"], "scope": "Implementation smoke only; no main metrics."}
    caches = {}
    for condition in cfg["conditions"]:
        path = run_dir / "cache" / condition / "CACHE.json"
        caches[condition] = {"metadata_present": path.is_file(), "path": str(path.relative_to(run_dir))}
        if path.is_file():
            caches[condition]["metadata"] = read(path)
    python = str(Path(sys.executable).resolve())
    base_command = [python, str(HERE / "driver.py"), "--run-dir", str(run_dir)]
    original = set(manifest["train_regimes"]["base"])
    composition = {}
    for regime, indices in manifest["train_regimes"].items():
        retained = len(original & set(indices))
        composition[regime] = {"n_frames": len(indices), "original_train_retained": retained,
                               "original_train_pool": len(original),
                               "retained_original_fraction": retained / len(original) if original else None,
                               "source_families": dict(Counter(manifest["records"][i].get("source_family", "unknown") for i in indices))}
    examples = []
    example_audit = run_dir / "DATA_VISUAL_QA.json"
    if example_audit.exists():
        for sample in read(example_audit).get("samples", []):
            path = Path(sample["overlay"])
            if not path.is_absolute():
                path = run_dir / path
            if not path.is_file():
                raise ValueError(f"Missing audited data example: {path}")
            examples.append({"case": sample["case"], "id": sample["id"],
                             "overlay": path.relative_to(run_dir).as_posix(),
                             "notes": sample.get("visual_notes", []), "cohort": sample.get("cohort", {})})
    return {
        "schema": "dht_padding_view_matrix_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha(run_dir / "CONFIG.json"), "manifest_sha256": sha(run_dir / "manifest.json"),
        "stage": cfg["stage"], "config": cfg, "main_status": main_status,
        "completion": {"expected_runs": len(runs), "complete_runs": completed,
                       "complete_main_runs": main_completed, "invalid_runs": invalid, "incomplete_runs": incomplete},
        "runs": runs, "cohorts": cohort_counts(manifest), "training_composition": composition,
        "cache": caches, "smoke": smoke,
        "paired_padding": paired_padding(runs, raw) if cfg["stage"] == "main" else [],
        "paired_regimes": paired_regimes(runs, raw) if cfg["stage"] == "main" else [],
        "seed_summary": across_seeds(runs, cfg) if cfg["stage"] == "main" else [],
        "evaluation_groups": {population: sorted({manifest["records"][i].get("group", "unknown") for i in manifest["populations"][population]})
                              for population in cfg["evaluation"]["populations"]},
        "evaluation_group_counts": {population: dict(Counter(manifest["records"][i].get("group", "unknown") for i in manifest["populations"][population]))
                                    for population in cfg["evaluation"]["populations"]},
        "yaw_bin_provenance": "Report-only bins of abs(frontal_yaw_deg_world), joined to CSV by frame id; missing real metadata stays unknown. Frozen manifest/config/cache unchanged.",
        "data_examples": examples,
        "commands": {"plan": shlex.join(base_command), "execute": shlex.join(base_command + ["--execute"]),
                     "execute_no_open": shlex.join(base_command + ["--execute", "--no-open"])},
        "scope": "Same frozen DHT8; padding comparison within a regime; base versus low_balanced is a data-mix comparison, not an isolated causal viewpoint change.",
        "metrics_scope": "Verified final main runs only. Per-seed supported-role distributions; smoke metrics excluded. Existing real_dev is not a new independent test.",
    }


def make_html(matrix):
    data = json.dumps(matrix, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DHT 패딩 × 학습 데이터 구성 실험</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#17263b;background:#f3f6fa;color-scheme:light}body{max-width:1450px;margin:auto;padding:24px}h1{font-size:27px;margin:0 0 12px}h2{font-size:19px;margin:0 0 12px}p{line-height:1.65;margin:8px 0}.muted{font-size:14px;color:#526279}.card{padding:22px;background:white;border:1px solid #dce4ee;border-radius:12px;margin:0 0 18px}.badge{display:inline-block;padding:5px 9px;border-radius:5px;font-weight:700;font-size:12px;background:#e9eef5;color:#41516a}.COMPLETE,.PRESENT{background:#e1f3e9;color:#21623e}.INVALID,.ABSENT,.INVALID_STAGE{background:#fbe6e7;color:#a32935}.SMALL,.PARTIAL,.INCOMPLETE{background:#fff0cb;color:#775116}.stats{display:flex;gap:14px;flex-wrap:wrap}.stat{padding:16px;background:#f5f8fc;border-radius:8px;min-width:150px}.stat strong{display:block;font-size:25px;margin-top:7px}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e1e7ef;padding:11px;text-align:left;vertical-align:top;white-space:nowrap}th{background:#f7f9fc}td .badge{margin-top:4px}.cellcount{font-size:22px;font-weight:700}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}label{font-size:13px;font-weight:650;display:flex;flex-direction:column;gap:5px}select{padding:8px;border:1px solid #bcc9da;border-radius:5px;font:inherit;background:white;max-width:85vw}pre{padding:13px;background:#eef3f9;border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#125cbc}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.empty{padding:20px;border:1px dashed #bdc9da;color:#526279;border-radius:8px}details summary{cursor:pointer;margin:12px 0}button{padding:7px 12px;background:#e9eff8;color:#233b62;border:1px solid #bcc9da;border-radius:5px;cursor:pointer}@media(max-width:750px){body{padding:12px}.columns{grid-template-columns:1fr}.card{padding:16px}}
</style><body>
<section class="card"><h1>DHT 패딩 × 학습 데이터 구성 실험</h1><p><span id="main-status" class="badge"></span> <span id="status-text"></span></p>
<p>같은 DHT8 구조에서 패딩 3종과 학습 데이터 구성 2종을 비교합니다. 패딩 방식은 각 구성 안에서 학습부터 평가까지 동일하게 적용합니다. 같은 구성·seed의 패딩 조건끼리 초기 가중치와 학습 순서를 맞추고, 평가 이미지는 모든 조건에 동일하게 사용합니다.</p>
<p class="muted">패딩 폭 100px, 입력 400×400, 특징 50×50, 측면 역할 8개를 고정합니다. 이번 실행표에는 패딩 폭 변경·ROI·12개 역할로의 확장을 섞지 않습니다.</p>
<p class="muted">base와 low_balanced의 차이는 렌더링 데이터 풀을 포함한 <strong>학습 데이터 구성 차이</strong>입니다. 관점 하나만 바꾼 인과 실험으로 해석하지 않습니다. 고도는 메타데이터에서 얻은 각도이며, 측면·윗면 노출 비율은 영상 기하 지표입니다. 실제 영상에 고도 메타데이터가 없으면 ‘정보 없음’으로 표시합니다.</p>
<div class="stats"><div class="stat">예정 본학습 실행<strong id="expected"></strong></div><div class="stat">검증된 본학습 완료<strong id="completed"></strong></div><div class="stat">실행당 업데이트<strong id="steps"></strong></div><div class="stat">별도 smoke 확인<strong id="smoke"></strong></div></div></section>
<section class="card"><h2>실험 실행표</h2><p class="muted">완료는 최종 step, 설정·manifest, checkpoint·CSV 해시와 평가 모집단을 확인한 경우에만 표시합니다. NOT_RUN은 본학습 결과 없음, INCOMPLETE는 불완전한 실행, INVALID는 검증 실패입니다.</p><div id="run-matrix" class="tablewrap"></div><details><summary>검증 실패 또는 미완료 내역</summary><div id="run-errors"></div></details></section>
<section class="card"><h2>데이터 범위와 빈 구간</h2><div id="training-composition"></div><p class="muted">두 구성의 학습 장수는 동일합니다. low_balanced는 기존 학습 데이터의 절반을 유지하고 저고도 합성을 추가하므로, 관점 외에도 렌더러·배경·재질 분포가 달라질 수 있습니다.</p><p class="muted">셀의 숫자는 서로 다른 이미지 수입니다. ABSENT=0장, SMALL=1–19장, PRESENT=20장 이상입니다. 20장은 데이터 분포를 보기 위한 표시 기준이며 성능 통과 기준이 아닙니다. 같은 분류 기준을 학습·평가에 적용합니다.</p><div class="controls"><label>데이터 집합<select id="cohort"></select></label></div><p id="cohort-total"></p><h3>실제 고도 메타데이터 × 측면 노출</h3><div id="coverage" class="tablewrap"></div><h3>정면에서 벗어난 실제 yaw 각도</h3><p class="muted">기존 world 기하의 frontal_yaw_deg_world로 계산한 분류입니다. 작은 yaw와 낮은 카메라 고도는 별개의 조건입니다. 실사는 yaw 메타데이터도 없어 ‘정보 없음’으로 남깁니다.</p><div id="yaw-counts"></div><h3>영상의 윗면 노출 × 측면 노출</h3><p class="muted">아래 표의 두 축은 투영 면적에서 계산한 지표입니다. 실사의 고도 각도를 대신 추정한 값으로 해석하지 않습니다.</p><div id="top-coverage" class="tablewrap"></div><div class="columns"><div><h3>윗면 노출 비율</h3><div id="top-counts"></div></div><div><h3>팔레트 크기</h3><div id="size-counts"></div></div></div></section>
<section class="card" id="examples-card"><h2>실제 데이터와 정답 예시 — 모델 예측 아님</h2><p class="muted">저고도·정면/비스듬한 방향·작은/큰 팔레트 예시입니다. 초록은 학습할 측면 정답 8개, 주황은 이번에 제외한 폭 방향 정답 4개입니다. 가림과 잘림을 포함한 구조적 정답이며, 물리적 엣지 가시성 정답은 아닙니다.</p><div class="controls"><label>데이터 예시<select id="data-example-select"></select></label></div><a id="data-example-link" target="_blank"><img id="data-example" style="width:100%;height:auto" alt="데이터 이미지와 구조적 정답 예시, 모델 예측 아님"></a><p id="data-example-notes" class="muted"></p><p><a href="DATA_VISUAL_QA.json">데이터 시각 검사 기록</a></p></section>
<section class="card"><h2>고정 평가 지표</h2><p id="success-definition"></p><p class="muted">위치 오차는 정답 선분 양 끝점에서 예측 무한직선까지의 수직거리 평균(px), 방향 오차는 무방향 직선 사이의 작은 각도입니다. 전체 측면 8개, 짧은 높이선 4개, 긴 깊이선 4개, 카메라를 향한 측면을 구분합니다. 가려진 직육면체 경계가 포함되며 물리적으로 보이는 엣지만의 평가가 아닙니다. 고정 최종 checkpoint를 평가하고 실사 DEV로 선택하지 않습니다.</p>
<div class="controls"><label>평가 모집단<select id="population"></select></label><label>촬영·데이터 그룹<select id="dataset-group"></select></label><label>선 부분집합<select id="subset"><option value="all">전체 측면</option><option value="height">짧은 높이선</option><option value="depth">긴 깊이선</option><option value="camera_facing">카메라를 향한 측면</option></select></label><label>관점 분류<select id="view-axis"><option value="all">전체 관점</option><option value="elevation_bin">고도</option><option value="yaw_bin">정면 yaw</option><option value="side_bin">측면 노출</option><option value="top_bin">윗면 노출</option><option value="size_bin">팔레트 크기</option></select></label></div><div id="metrics"></div><div id="seed-summary"></div><p class="muted">수치는 검증된 본학습 실행의 seed별 통계입니다. Smoke 수치는 포함하지 않습니다. Seed 요약은 각 seed 통계의 평균 [최솟값, 최댓값]이며 선을 합친 pooled 통계나 신뢰구간이 아닙니다. 관점별 표를 선택하면 전체 측면 선을 기준으로 분리하며, 빈 집합은 정확도 수치를 만들지 않습니다. 기존에 학습된 모델의 추론 패딩만 바꾼 진단은 이번 새 학습 실행표의 결과에 포함하지 않습니다.</p><div id="paired"></div><div id="paired-regimes"></div></section>
<section class="card"><h2>실행 명령</h2><p>현재 계획과 저장된 결과만 갱신:</p><pre id="plan-command"></pre><p>준비된 설정으로 본학습을 실행할 때:</p><pre id="execute-command"></pre><p class="muted">기본 driver는 계획만 작성합니다. --execute를 직접 지정해야 캐시 준비와 본학습이 실행됩니다. 완료 후 HTML을 기본 브라우저로 한 번 엽니다. 자동 열기를 끄려면 --no-open을 추가합니다.</p></section>
<section class="card"><h2>준비 상태와 원본 파일</h2><div id="cache-status"></div><p id="smoke-detail"></p><p><a href="CONFIG.json">설정</a> · <a href="manifest.json">이미지 분리·코호트</a> · <a href="MATRIX.json">실행표와 집계 JSON</a> · <a href="DATA_AUDIT.json">데이터 감사</a></p><details><summary>전체 계획 JSON</summary><pre id="raw"></pre></details></section>
<script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),L=__LABELS__,axes=__AXES__;
const $=id=>document.getElementById(id),esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const badge=status=>`<span class="badge ${esc(status)}">${esc(status)}</span>`;
const table=(headers,rows)=>`<div class="tablewrap"><table><thead><tr>${headers.map(x=>`<th>${esc(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const num=(v,d=2)=>v===null||v===undefined?'—':Number(v).toFixed(d);
$('main-status').textContent=D.main_status;$('main-status').className='badge '+D.main_status;
$('status-text').textContent=D.main_status==='NOT_RUN'?'본학습 결과는 아직 없습니다. 준비·smoke 검증을 본학습 완료로 표시하지 않습니다.':D.main_status==='COMPLETE'?'모든 본학습 실행의 완료 검증을 통과했습니다.':D.main_status==='FAILED'?'본학습 실행의 완료 검증이 실패했습니다. 아래 내역을 확인하세요.':D.main_status==='INCOMPLETE'?'시작된 본학습 실행이 있지만 아직 완료되지 않았습니다.':'일부 본학습 실행만 완료되었습니다. 전체 비교는 아직 끝나지 않았습니다.';
$('expected').textContent=D.stage==='main'?D.completion.expected_runs:'smoke 전용';$('completed').textContent=D.completion.complete_main_runs;$('steps').textContent=D.config.training.steps.toLocaleString();$('smoke').textContent=`${D.smoke.complete_runs}/${D.smoke.expected_runs}`;
const seeds=D.config.training.seeds,rows=[];
for(const regime of D.config.training.regimes)for(const [condition,padding] of Object.entries(D.config.conditions)){
 const cells=seeds.map(seed=>{const r=D.runs.find(x=>x.regime===regime&&x.condition===condition&&x.seed===seed);return badge(r.status)+(r.status==='COMPLETE'?`<br><a href="${esc(r.path)}/COMPLETION.json">완료 기록</a>`:'');});
 rows.push([esc(regime),esc(condition),esc(`${padding.mode} / ${padding.pad_px}px`),...cells]);
}
$('run-matrix').innerHTML=table(['학습 구성','패딩','입력 처리',...seeds.map(s=>'seed '+s)],rows);
$('run-errors').innerHTML=D.runs.filter(r=>r.errors.length||r.status==='INCOMPLETE').map(r=>`<p>${esc(`${r.regime}/${r.condition}/seed_${r.seed}`)}: ${esc(r.errors.join('; ')||'최종 완료 기록 없음')}</p>`).join('')||'<p class="muted">검증 실패 또는 불완전한 실행 기록이 없습니다.</p>';
for(const name of Object.keys(D.cohorts))$('cohort').add(new Option(name,name));
$('training-composition').innerHTML=table(['학습 구성','총 이미지','기존 train 유지','기존 train 유지율','소스별 이미지'],Object.entries(D.training_composition).map(([name,c])=>[esc(name),String(c.n_frames),`${c.original_train_retained} / ${c.original_train_pool}`,c.retained_original_fraction===null?'—':num(c.retained_original_fraction*100,0)+'%',esc(Object.entries(c.source_families).map(([source,n])=>`${source}: ${n}`).join(' / '))]));
function coverage(){const group=D.cohorts[$('cohort').value];$('cohort-total').textContent=`${group.n_frames.toLocaleString()}장`;
 for(const [axis,key,id,title] of [['elevation_bin','elevation_x_side','coverage','고도 / 측면 노출'],['top_bin','top_x_side','top-coverage','윗면 / 측면 노출']]){
  const rows=axes[axis].map(e=>[esc(L[e]||e),...axes.side_bin.map(s=>{const c=group[key].find(x=>x[axis]===e&&x.side_bin===s);return `<span class="cellcount">${c.n_frames}</span><br>${badge(c.coverage_status)}`;}),String(group.bins[axis][e])]);
  rows.push(['전체',...axes.side_bin.map(s=>String(group.bins.side_bin[s])),String(group.n_frames)]);
  $(id).innerHTML=table([title,...axes.side_bin.map(s=>L[s]||s),'합계'],rows);
 }
 for(const [axis,id] of [['top_bin','top-counts'],['size_bin','size-counts']])$(id).innerHTML=table(['분류','이미지 수'],Object.entries(group.bins[axis]).map(([label,n])=>[esc(L[label]||label),String(n)]));
 $('yaw-counts').innerHTML=table(['yaw 분류','이미지 수','데이터 범위'],Object.entries(group.bins.yaw_bin).map(([label,n])=>[esc(L[label]||label),String(n),badge(n===0?'ABSENT':n<20?'SMALL':'PRESENT')]));
}
$('cohort').addEventListener('change',coverage);coverage();
if(D.data_examples.length){D.data_examples.forEach((s,i)=>$('data-example-select').add(new Option(`${s.case} / ${s.id}`,String(i))));function example(){const s=D.data_examples[Number($('data-example-select').value)];$('data-example').src=s.overlay;$('data-example-link').href=s.overlay;$('data-example-notes').textContent=Array.isArray(s.notes)?s.notes.join(' '):String(s.notes);}$('data-example-select').addEventListener('change',example);example();}else{$('examples-card').hidden=true;}
for(const population of D.config.evaluation.populations)$('population').add(new Option(population,population));
if(D.config.evaluation.populations.includes('real_dev'))$('population').value='real_dev';
const E=D.config.evaluation;$('success-definition').textContent=`성공률: 방향 오차 ≤${E.success_angle_deg}°와 위치 오차 ≤${E.success_distance_px}px를 동시에 만족한 선의 비율. 각도·위치 중앙값과 P90도 함께 기록합니다.`;
function groupOptions(){const select=$('dataset-group'),pop=$('population').value;select.replaceChildren(new Option('전체 그룹','all'));for(const g of D.evaluation_groups[pop]||[])select.add(new Option(`${g} (${D.evaluation_group_counts[pop][g]}장)`,g));}groupOptions();
function metrics(){const population=$('population').value,axis=$('view-axis').value,subset=$('subset').value,datasetGroup=$('dataset-group').value,rows=[],seedGroups=new Map();
 $('subset').disabled=axis!=='all';
 for(const run of D.runs){if(!run.populations)continue;let p=run.populations[population];if(!p)continue;if(datasetGroup!=='all')p=p.by_group[datasetGroup];if(!p)continue;
 const groups=axis==='all'?[[subset,subset==='all'?p:p[subset]]]:Object.entries(p.by_view[axis]);
 for(const [label,s] of groups){rows.push([esc(run.regime),esc(run.condition),String(run.seed),esc(L[label]||label),String(s.n_frames),String(s.n_roles),num(s.angle_deg?.median),num(s.angle_deg?.p90),num(s.distance_px?.median),num(s.distance_px?.p90),s.success_fraction===null?'—':num(s.success_fraction*100,1)+'%']);const key=JSON.stringify([run.regime,run.condition,label]);if(!seedGroups.has(key))seedGroups.set(key,[]);seedGroups.get(key).push({seed:run.seed,s});}}
 $('metrics').innerHTML=rows.length?table(['학습 구성','패딩','seed','부분집합','이미지','선','각도 중앙값 °','각도 P90 °','위치 중앙값 px','위치 P90 px','성공률'],rows):`<div class="empty">${esc(D.main_status)} — 선택한 집합에 검증된 본학습 결과가 없습니다. 여기에 임의 수치나 smoke 정확도를 채우지 않습니다.</div>`;
 const aggregate=values=>{values=values.filter(v=>v!==null&&v!==undefined);return values.length?`${num(values.reduce((a,b)=>a+b,0)/values.length)} [${num(Math.min(...values))}, ${num(Math.max(...values))}]`:'—';};
 $('seed-summary').innerHTML=seedGroups.size?'<h3>Seed 통계의 평균 [최솟값, 최댓값]</h3>'+table(['학습 구성','패딩','부분집합','완료 seed','각도 중앙값 °','각도 P90 °','위치 중앙값 px','위치 P90 px','성공률 %'],[...seedGroups].map(([key,entries])=>{const [regime,condition,label]=JSON.parse(key);return [esc(regime),esc(condition),esc(L[label]||label),`${entries.length}/${D.config.training.seeds.length}`,aggregate(entries.map(e=>e.s.angle_deg?.median)),aggregate(entries.map(e=>e.s.angle_deg?.p90)),aggregate(entries.map(e=>e.s.distance_px?.median)),aggregate(entries.map(e=>e.s.distance_px?.p90)),aggregate(entries.map(e=>e.s.success_fraction===null?null:e.s.success_fraction*100))];})):'';
 const pairs=datasetGroup==='all'?D.paired_padding.filter(p=>p.population===population):[];
 $('paired').innerHTML=pairs.length?'<h3>같은 이미지·역할의 패딩 비교</h3><p class="muted">같은 학습 구성·seed에서 reflect100 대비 위치 오차 차이입니다. 음수면 비교 패딩의 오차가 더 작습니다.</p>'+table(['학습 구성','패딩','seed','일치 이미지','일치 선','누락 선','오차 차이 중앙값 px','오차 차이 평균 px'],pairs.map(p=>[esc(p.regime),esc(p.condition),String(p.seed),String(p.n_matched_frames),String(p.n_matched_roles),String(p.n_unmatched_roles),num(p.distance_delta_median_px),num(p.distance_delta_mean_px)])):'';
 const mixture=datasetGroup==='all'?D.paired_regimes.filter(p=>p.population===population):[];
 $('paired-regimes').innerHTML=mixture.length?'<h3>같은 패딩에서 학습 데이터 구성 비교</h3><p class="muted">같은 평가 이미지·역할·seed에서 low_balanced − base 위치 오차입니다. 관점만의 인과 효과가 아닙니다.</p>'+table(['패딩','seed','일치 이미지','일치 선','누락 선','오차 차이 중앙값 px','오차 차이 평균 px'],mixture.map(p=>[esc(p.condition),String(p.seed),String(p.n_matched_frames),String(p.n_matched_roles),String(p.n_unmatched_roles),num(p.distance_delta_median_px),num(p.distance_delta_mean_px)])):'';
}
$('population').addEventListener('change',()=>{groupOptions();metrics();});for(const id of ['dataset-group','subset','view-axis'])$(id).addEventListener('change',metrics);metrics();
$('plan-command').textContent=D.commands.plan;$('execute-command').textContent=D.commands.execute;
$('cache-status').innerHTML=table(['패딩','캐시 메타데이터'],Object.entries(D.cache).map(([name,value])=>[esc(name),value.metadata_present?`<a href="${esc(value.path)}">CACHE.json 있음</a>`:'아직 없음']));
$('smoke-detail').textContent=`별도 smoke: ${D.smoke.status}, ${D.smoke.complete_runs}/${D.smoke.expected_runs} 실행 완료${D.smoke.steps_per_run?' · 실행당 '+D.smoke.steps_per_run+' step':''}. 본학습의 성능 결과에 포함하지 않습니다.`;
$('raw').textContent=JSON.stringify(D,null,2);
</script></body></html>"""
    return page.replace("__DATA__", data).replace("__LABELS__", json.dumps(LABELS, ensure_ascii=False)).replace("__AXES__", json.dumps(AXES))


def generate(run_dir):
    run_dir = Path(run_dir).resolve()
    matrix = build_matrix(run_dir)
    write(run_dir / "MATRIX.json", matrix)
    dashboard = run_dir / "experiment_dashboard.html"
    dashboard.write_text(make_html(matrix), encoding="utf-8")
    print(f"{matrix['main_status']}: {matrix['completion']['complete_main_runs']} verified main runs; {dashboard}", flush=True)
    return dashboard, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/pallet/results/dht_padding_view_v1")
    parser.add_argument("--no-open", action="store_true", help="Write the dashboard without opening the local browser")
    args = parser.parse_args()
    dashboard, _ = generate(args.run_dir)
    if not args.no_open:
        open_dashboard(dashboard)


if __name__ == "__main__":
    main()
