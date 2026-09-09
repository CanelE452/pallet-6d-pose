"""Matched, descriptive analysis of the completed fixed Direct/DHT experiment.

This file never trains a model or selects a checkpoint. Bootstrap intervals
describe paired frame means in these recorded evaluation populations, treating
the recorded seeds as fixed; they do not establish independent real-world
generalization or account for correlation within a recording session.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ARMS = ("Direct", "DHT")
METRICS = ("angle_deg", "distance_px", "distance_diagonal")
SUBSETS = {
    "all_eight": lambda pair: True,
    "height_four": lambda pair: pair["role"] < 4,
    "depth_four": lambda pair: pair["role"] >= 4,
    "camera_facing": lambda pair: pair["camera_facing"],
}
SUBSET_NAMES = {
    "all_eight": "측면 외곽 전체 8개",
    "height_four": "높이 방향 4개 (앞·뒤 포함)",
    "depth_four": "깊이 방향 4개",
    "camera_facing": "카메라를 향한 측면의 경계",
}
POPULATION_NAMES = {
    "synth_train": "합성 학습",
    "synth_val": "합성 검증",
    "synth_test": "동일 생성 계열 합성 평가",
    "cross_v4": "다른 합성 평가",
    "real_dev": "실사 개발셋",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched_rows(path, seeds):
    """Require complete Direct/DHT pairs and identical support across seeds."""
    indexed = {arm: {} for arm in ARMS}
    with Path(path).open(newline="") as handle:
        for raw in csv.DictReader(handle):
            arm = raw["arm"]
            if arm not in ARMS:
                raise ValueError(f"Unknown experiment arm: {arm}")
            facing = raw["camera_facing"].lower()
            if facing not in ("true", "false", "1", "0"):
                raise ValueError("camera_facing must be a boolean")
            row = dict(
                id=raw["id"], population=raw["population"], group=raw["group"],
                role=int(raw["role"]), seed=int(raw["seed"]),
                camera_facing=facing in ("true", "1"),
                **{metric: float(raw[metric]) for metric in METRICS},
            )
            if row["seed"] not in seeds or not 0 <= row["role"] < 8:
                raise ValueError("Unexpected seed or role in saved evaluation")
            if any(not np.isfinite(row[m]) or row[m] < 0 for m in METRICS):
                raise ValueError("Evaluation errors must be finite and nonnegative")
            key = (row["population"], row["id"], row["role"], row["seed"])
            if key in indexed[arm]:
                raise ValueError(f"Duplicate {arm} evaluation row: {key}")
            indexed[arm][key] = row
    if not indexed["Direct"] or indexed["Direct"].keys() != indexed["DHT"].keys():
        raise ValueError("Direct/DHT saved evaluation pairs are missing or asymmetric")
    seed_coverage = defaultdict(set)
    pairs = []
    for key, direct in indexed["Direct"].items():
        dht = indexed["DHT"][key]
        for meta in ("group", "camera_facing"):
            if direct[meta] != dht[meta]:
                raise ValueError(f"Pair metadata mismatch: {key} / {meta}")
        pair = {k: direct[k] for k in ("id", "population", "group", "role", "seed", "camera_facing")}
        for arm, row in (("Direct", direct), ("DHT", dht)):
            pair[arm] = {metric: row[metric] for metric in METRICS}
        pairs.append(pair)
        seed_coverage[key[:3]].add(key[3])
    if any(coverage != set(seeds) for coverage in seed_coverage.values()):
        raise ValueError("Every scored frame/role must occur for every configured seed")
    return pairs


def seed_statistics(pairs, seeds):
    """Compute each seed's line statistics first, then average those statistics."""
    result = {
        "n_frames": len({p["id"] for p in pairs}),
        "n_unique_frame_roles": len({(p["id"], p["role"]) for p in pairs}),
        "n_matched_seed_frame_roles": len(pairs),
    }
    if not pairs:
        return result
    for arm in ARMS:
        per_seed = {}
        for seed in seeds:
            these = [p for p in pairs if p["seed"] == seed]
            if not these:
                raise ValueError("Subset unexpectedly lacks a configured seed")
            per_seed[str(seed)] = {"n_roles": len(these)}
            for metric in METRICS:
                values = np.asarray([p[arm][metric] for p in these])
                per_seed[str(seed)][metric] = {
                    "median": float(np.median(values)),
                    "p90": float(np.percentile(values, 90)),
                    "mean": float(np.mean(values)),
                }
            per_seed[str(seed)]["success_5deg_8px"] = float(np.mean([
                p[arm]["angle_deg"] <= 5 and p[arm]["distance_px"] <= 8 for p in these
            ]))
        averaged = {}
        for metric in METRICS:
            averaged[metric] = {}
            for stat in ("median", "p90", "mean"):
                values = [per_seed[str(seed)][metric][stat] for seed in seeds]
                averaged[metric][f"{stat}_mean_across_seeds"] = float(np.mean(values))
                averaged[metric][f"{stat}_range_across_seeds"] = [min(values), max(values)]
        averaged["success_5deg_8px_mean_across_seeds"] = float(np.mean([
            per_seed[str(seed)]["success_5deg_8px"] for seed in seeds
        ]))
        result[arm] = {"per_seed": per_seed, "seed_summary": averaged}
    return result


def paired_frame_bootstrap(pairs, seeds, repetitions=2000, random_seed=42):
    """Average roles within a frame, then fixed seeds, then bootstrap frames.

    Direct and DHT always share the same sampled frame indices. Different
    roles or seeds of a frame are never treated as independent observations.
    """
    if not pairs:
        return {"n_frames": 0}
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[(pair["id"], pair["seed"])].append(pair)
    frame_ids = sorted({p["id"] for p in pairs})
    rng = np.random.default_rng(random_seed)
    sampled = rng.integers(0, len(frame_ids), size=(repetitions, len(frame_ids)))
    result = {"n_frames": len(frame_ids), "bootstrap_repetitions": repetitions,
              "bootstrap_seed": random_seed, "frame_ids": frame_ids}
    for metric in METRICS:
        values = np.asarray([
            [[np.mean([p[arm][metric] for p in grouped[(frame, seed)]])
              for arm in ARMS] for seed in seeds] for frame in frame_ids
        ])  # frame, seed, arm
        seed_averaged = values.mean(axis=1)
        delta = seed_averaged[:, 1] - seed_averaged[:, 0]
        simulated = delta[sampled].mean(axis=1)
        result[metric] = {
            "Direct_frame_mean": float(seed_averaged[:, 0].mean()),
            "DHT_frame_mean": float(seed_averaged[:, 1].mean()),
            "DHT_minus_Direct_frame_mean": float(delta.mean()),
            "DHT_minus_Direct_bootstrap_95pct_interval": np.quantile(simulated, [.025, .975]).tolist(),
            "DHT_minus_Direct_frame_median": float(np.median(delta)),
            "fraction_frames_DHT_lower": float(np.mean(delta < 0)),
            "DHT_minus_Direct_frame_means": delta.tolist(),
            "DHT_minus_Direct_frame_mean_per_seed": {
                str(seed): float((values[:, i, 1] - values[:, i, 0]).mean())
                for i, seed in enumerate(seeds)
            },
        }
    return result


def subset_report(pairs, seeds):
    report = seed_statistics(pairs, seeds)
    report["paired_frame_bootstrap"] = paired_frame_bootstrap(pairs, seeds)
    return report


def verify_saved_summary(result, populations, seeds):
    """Prevent silently analyzing a CSV from another run than RESULTS.json."""
    for population, report in populations.items():
        for arm in ARMS:
            for seed in seeds:
                saved = result["by_run"][f"{arm}_seed{seed}"][population]
                computed = report[arm]["per_seed"][str(seed)]
                if saved["n_roles"] != computed["n_roles"]:
                    raise ValueError("CSV and RESULTS.json disagree on scored row counts")
                for metric in METRICS:
                    for stat in ("median", "p90", "mean"):
                        if not np.isclose(saved[metric][stat], computed[metric][stat], atol=1e-8, rtol=1e-8):
                            raise ValueError(f"CSV/JSON mismatch: {population}/{arm}/{seed}/{metric}/{stat}")


def value(report, arm, metric="distance_px", stat="median"):
    return report[arm]["seed_summary"][metric][f"{stat}_mean_across_seeds"]


def verdict(real):
    shifts = [value(real, "DHT", "distance_px", stat) - value(real, "Direct", "distance_px", stat)
              for stat in ("median", "p90")]
    paired = real["paired_frame_bootstrap"]["distance_px"]
    interval = paired["DHT_minus_Direct_bootstrap_95pct_interval"]
    if max(shifts) < 0 and interval[1] < 0:
        return "REAL_DEV_DISTANCE_IMPROVEMENT", "이 실사 개발셋에서는 DHT의 선 위치 중앙값·큰 오차·프레임 평균 오차가 함께 감소했습니다."
    if min(shifts) > 0 and interval[0] > 0:
        return "REAL_DEV_DISTANCE_WORSENED", "이 실사 개발셋에서는 DHT의 선 위치 중앙값·큰 오차·프레임 평균 오차가 함께 증가했습니다."
    return "REAL_DEV_DISTANCE_MIXED", "이 실사 개발셋에서는 지표별 결과가 섞여 있어 DHT의 위치 정확도가 일관되게 개선되었다고 판단하지 않습니다."


def location_angle_diagnosis(pairs, manifest, edges, seeds):
    """Use D=max(|midpoint offset|, length/2*|sin(angle error)|).

    When D is strictly larger than the angular term, its value is the absolute
    midpoint offset. Otherwise the offset is only bounded above by D; we do not
    recover or report it as an exact value for those lines.
    """
    records = {r["id"]: r for rows in manifest["populations"].values() for r in rows}
    derived = []
    for pair in pairs:
        if pair["population"] != "real_dev":
            continue
        points = np.asarray(records[pair["id"]]["gt_points"])[np.asarray(edges[pair["role"]])]
        length = float(np.linalg.norm(points[1] - points[0]))
        for arm in ARMS:
            angular_term = length / 2 * abs(np.sin(np.deg2rad(pair[arm]["angle_deg"])))
            distance = pair[arm]["distance_px"]
            if angular_term > distance + 1e-4:
                raise ValueError("Endpoint-distance geometry identity violated by saved metrics")
            derived.append(dict(arm=arm, seed=pair["seed"], role=pair["role"],
                                angular_term_px=angular_term, distance_px=distance,
                                position_dominates=distance > angular_term + 1e-4))
    report = {
        "identity": "mean absolute GT-endpoint distance = max(absolute GT-midpoint distance to predicted line, GT-segment length / 2 * abs(sin(undirected angle error)))",
        "position_dominates": "D > angular_term + 1e-4px. In that subset D equals the absolute midpoint offset; outside it the offset is only bounded above by D.",
        "scope": "Descriptive decomposition of measured error; does not identify the cause of a learned model's failure.",
    }
    for arm in ARMS:
        per_seed = {}
        for seed in seeds:
            these = [d for d in derived if d["arm"] == arm and d["seed"] == seed]
            failures = [d for d in these if d["distance_px"] > 8]
            per_seed[str(seed)] = {
                "n_roles": len(these),
                "position_dominates_fraction": float(np.mean([d["position_dominates"] for d in these])),
                "angular_term_median_px": float(np.median([d["angular_term_px"] for d in these])),
                "n_distance_above_8px": len(failures),
                "position_dominates_fraction_among_distance_above_8px": (
                    float(np.mean([d["position_dominates"] for d in failures])) if failures else None),
            }
        report[arm] = {
            "per_seed": per_seed,
            "position_dominates_fraction_mean_across_seeds": float(np.mean([
                v["position_dominates_fraction"] for v in per_seed.values()
            ])),
            "angular_term_median_px_mean_across_seeds": float(np.mean([
                v["angular_term_median_px"] for v in per_seed.values()
            ])),
        }
    return report


def learning_curves(out, seeds):
    """Plot recorded window-average training losses without inferring convergence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    colors, styles = {"Direct": "#08799b", "DHT": "#ba3089"}, ("-", "--", ":", "-.")
    summary = {}
    for arm in ARMS:
        for i, seed in enumerate(seeds):
            name = f"{arm}_seed{seed}"
            history = read_json(out / f"{name}_history.json")
            steps = np.asarray([v["step"] for v in history])
            loss = np.asarray([v["loss_mean"] for v in history])
            if len(history) < 2 or not np.isfinite(loss).all() or not np.all(np.diff(steps) > 0):
                raise ValueError(f"Invalid saved training history: {name}")
            for ax, keep in ((axes[0], np.ones(len(steps), bool)), (axes[1], steps >= 500)):
                ax.plot(steps[keep], loss[keep], label=f"{arm} seed {seed}", color=colors[arm],
                        ls=styles[i % len(styles)], lw=1.8, marker="o", markersize=3)
            summary[name] = {
                "last_step": int(steps[-1]), "final_window_mean_loss": float(loss[-1]),
                "previous_window_mean_loss": float(loss[-2]),
                "final_minus_previous_window_loss": float(loss[-1] - loss[-2]),
            }
    for ax, title in zip(axes, ("All recorded steps", "From step 500: window means")):
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Soft cross-entropy loss")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Matched synthetic-only Direct / DHT side-line training", weight="bold")
    fig.text(.5, .022, "Step 1 is one batch; later points average losses since the preceding log (normally 500 steps).\n"
             "Same targets / loss for both models. These are training curves, not evidence of convergence or real-world generalization.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=.065, right=.985, bottom=.19, top=.83, wspace=.22)
    fig.savefig(out / "learning_curves.png", dpi=160, facecolor="white")
    plt.close(fig)
    return {"curves": summary, "aggregation": "Step1 single batch; subsequent values average preceding logged window, normally500steps.",
            "scope": "Training losses only; residual decline does not establish convergence or its absence."}


def verify_completion(out, cfg, populations, learning):
    """Inspect saved model metadata on CPU and preserve final-step evidence."""
    import torch

    config_hash = sha(out / "CONFIG.json")
    by_run = {}
    here = Path(__file__).resolve().parent
    core_scripts = ("runner.py", "network.py", "dht.py", "targets.py")
    current_core = {name: sha(here / name) for name in core_scripts}
    for arm in ARMS:
        for seed in cfg["seeds"]:
            name = f"{arm}_seed{seed}"
            path = out / "checkpoints" / f"{name}_final.pth"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if (checkpoint["arm"] != arm or checkpoint["seed"] != seed
                    or checkpoint["steps"] != cfg["steps"] or checkpoint["config_sha256"] != config_hash
                    or checkpoint["config"] != cfg):
                raise ValueError(f"Checkpoint final-step/config evidence failed: {name}")
            if learning["curves"][name]["last_step"] != cfg["steps"]:
                raise ValueError(f"Training history lacks final configured step: {name}")
            saved_core = {script: checkpoint.get("script_sha256", {}).get(script) for script in core_scripts}
            if saved_core != current_core:
                raise ValueError(f"Core training source hashes changed since checkpoint: {name}")
            if not checkpoint["model"] or any(not torch.isfinite(v).all() for v in checkpoint["model"].values()):
                raise ValueError(f"Nonfinite or empty checkpoint model weights: {name}")
            by_run[name] = {
                "arm": arm, "seed": seed, "actual_final_step": checkpoint["steps"],
                "history_final_step": learning["curves"][name]["last_step"],
                "checkpoint": str(path.relative_to(out)), "checkpoint_sha256": sha(path),
                "config_sha256": config_hash, "core_training_script_sha256": saved_core,
                "evaluated_roles_by_population": {
                    pop: report[arm]["per_seed"][str(seed)]["n_roles"] for pop, report in populations.items()
                },
            }
    return {
        "schema": "deep_hough_side_v1_final_step_verification", "PASS": True,
        "scope": "All configured fixed-step training runs, finite checkpoint weights, unchanged core training source, and CSV/RESULTS numeric consistency. Gallery rendering/visual QA is separate.",
        "expected_steps": cfg["steps"], "expected_runs": len(ARMS) * len(cfg["seeds"]),
        "config_sha256": config_hash, "by_run": by_run,
        "evaluation_csv_rows": sum(p["n_matched_seed_frame_roles"] for p in populations.values()) * len(ARMS),
        "matched_direct_dht_pairs": sum(p["n_matched_seed_frame_roles"] for p in populations.values()),
        "results_sha256": sha(out / "RESULTS.json"), "per_role_csv_sha256": sha(out / "PER_ROLE.csv"),
    }


def markdown_report(out, analysis, result, manifest, target_audit, checks, sanity):
    cfg, seeds = result["config"], result["config"]["seeds"]
    final_windows_decreased = sum(v["final_minus_previous_window_loss"] < 0
                                  for v in analysis["learning_curves"]["curves"].values())
    real = analysis["matched"]["real_dev"]["subsets"]["all_eight"]
    paired = real["paired_frame_bootstrap"]["distance_px"]
    ci = paired["DHT_minus_Direct_bootstrap_95pct_interval"]
    counts = {p: len(rows) for p, rows in manifest["populations"].items()}
    exceptions = []
    outdoor = analysis["matched"]["real_dev"]["by_group"].get("eval_outside")
    if outdoor:
        change = outdoor["paired_frame_bootstrap"]["distance_px"]["DHT_minus_Direct_frame_mean"]
        if change > 0:
            exceptions.append(f"야외 {outdoor['n_frames']}장의 프레임 평균 위치 오차는 {change:.2f}px 증가했습니다")
    facing = analysis["matched"]["real_dev"]["subsets"]["camera_facing"]
    if facing["n_frames"]:
        change = facing["paired_frame_bootstrap"]["distance_px"]["DHT_minus_Direct_frame_mean"]
        if change > 0:
            exceptions.append(f"카메라를 향한 측면 경계의 프레임 평균도 {change:.2f}px 증가했습니다")
    caveat = "다만 " + "; ".join(exceptions) + ". 전체 중앙값 개선이 모든 조건의 개선을 뜻하지 않습니다." if exceptions else ""
    lines = [
        "# 팔레트 측면 외곽선: Direct Hough와 Deep Hough Transform", "",
        analysis["conclusion_ko"],
        caveat,
        "실사 평가 자료는 기존 개발셋입니다. 아래 결과로 새로운 촬영 환경에서의 최종 일반화 성능을 확정하지 않습니다.", "",
        "[시각화 갤러리](deep_hough_gallery.html) · [전체 분석 JSON](ANALYSIS.json) · [개별 선 오차 CSV](PER_ROLE.csv)", "",
        "| 평가 자료 | 장수 | Direct 위치 중앙값 px | DHT 위치 중앙값 px | Direct 위치 P90 px | DHT 위치 P90 px | Direct 방향 중앙값 ° | DHT 방향 중앙값 ° |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for population in ("synth_test", "cross_v4", "real_dev"):
        report = analysis["population_summary"][population]
        lines.append(
            f"| {POPULATION_NAMES[population]} | {report['n_frames']} | "
            f"{value(report, 'Direct'):.2f} | {value(report, 'DHT'):.2f} | "
            f"{value(report, 'Direct', stat='p90'):.2f} | {value(report, 'DHT', stat='p90'):.2f} | "
            f"{value(report, 'Direct', 'angle_deg'):.2f} | {value(report, 'DHT', 'angle_deg'):.2f} |"
        )
    lines += [
        "", f"표의 값은 {len(seeds)}개 seed 각각에서 선 전체의 중앙값 또는 90백분위를 계산한 뒤 평균한 값입니다. "
        "모든 실행을 합친 중앙값이나 신뢰구간이 아닙니다. 위치 오차는 정답 선분의 양 끝점에서 예측 무한직선까지의 평균 수직거리이며 원본 이미지 px 단위입니다.", "",
        f"실사에서 seed를 먼저 평균한 프레임별 평균 위치 오차는 Direct {paired['Direct_frame_mean']:.2f}px, "
        f"DHT {paired['DHT_frame_mean']:.2f}px입니다. 대응 차이(DHT−Direct)는 {paired['DHT_minus_Direct_frame_mean']:+.2f}px, "
        f"프레임 재표집 95% 구간은 [{ci[0]:+.2f}, {ci[1]:+.2f}]px입니다. 음수일수록 DHT가 좋습니다.",
        "재표집은 같은 프레임의 두 모델을 묶고, 역할 평균→seed 평균→프레임 재표집 순서로 2,000회(seed 42) 수행했습니다. "
        "이 구간은 기록된 seed·개발 프레임에 대한 기술적 비교이며 촬영 세션 내 상관이나 새로운 학습 seed의 불확실성을 반영하지 않습니다.", "",
        "## 어떤 측면 선에서 달라졌는가", "",
        "| 실사 선 집합 | 프레임 | 고유 선 수 | Direct 위치 중앙값 px | DHT 위치 중앙값 px | Direct 방향 중앙값 ° | DHT 방향 중앙값 ° | 프레임 평균 위치 차이 px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in analysis["matched"]["real_dev"]["subsets"].items():
        if not report["n_frames"]:
            continue
        difference = report["paired_frame_bootstrap"]["distance_px"]["DHT_minus_Direct_frame_mean"]
        lines.append(
            f"| {SUBSET_NAMES[name]} | {report['n_frames']} | {report['n_unique_frame_roles']} | "
            f"{value(report, 'Direct'):.2f} | {value(report, 'DHT'):.2f} | "
            f"{value(report, 'Direct', 'angle_deg'):.2f} | {value(report, 'DHT', 'angle_deg'):.2f} | {difference:+.2f} |"
        )
    lines += [
        "", "높이 방향은 앞·뒤의 짧은 높이선 4개이며, 깊이 방향은 좌우 측면의 위·아래 선 4개입니다. "
        "카메라를 향한 면은 정답 cuboid의 면 방향으로 분류한 보조 집합입니다. 실제로 보이는 재료 경계인지, 적재물에 가려졌는지를 판별한 정답은 아닙니다.", "",
        "| 실사 촬영 그룹 | 장수 | Direct 위치 중앙값 px | DHT 위치 중앙값 px | 프레임 평균 위치 차이 px | 재표집 95% 구간 px |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, report in analysis["matched"]["real_dev"]["by_group"].items():
        difference = report["paired_frame_bootstrap"]["distance_px"]
        lower, upper = difference["DHT_minus_Direct_bootstrap_95pct_interval"]
        lines.append(
            f"| {group} | {report['n_frames']} | {value(report, 'Direct'):.2f} | {value(report, 'DHT'):.2f} | "
            f"{difference['DHT_minus_Direct_frame_mean']:+.2f} | [{lower:+.2f}, {upper:+.2f}] |"
        )
    lines += [
        "", "합성 평가의 동일한 역할별·그룹별·프레임 대응 분석과 각 seed 수치는 [ANALYSIS.json](ANALYSIS.json)에 있습니다. "
        "집합별 결과는 진단용이며 여러 비교를 수행한 뒤의 유의성 검정으로 해석하지 않습니다.", "",
        "위치와 방향의 기여도도 분리했습니다. 정답 선분 길이를 L, 방향 오차를 α라 하면 끝점 거리 오차 D는 "
        "`max(정답 중점의 수직 위치 오차, L/2 × |sin α|)`와 같습니다. "
        f"실사에서 위치 항이 방향 항보다 큰 비율은 Direct {analysis['location_angle_diagnosis']['Direct']['position_dominates_fraction_mean_across_seeds']*100:.1f}%, "
        f"DHT {analysis['location_angle_diagnosis']['DHT']['position_dominates_fraction_mean_across_seeds']*100:.1f}%입니다. "
        "방향 항이 큰 경우 중점 위치 오차의 정확한 값은 이 CSV만으로 복원하지 않았습니다. 이 분해는 오차의 형태를 설명하며 실패의 원인 자체를 입증하지 않습니다.", "",
        "## 시각화에서 확인한 사례", "",
        "아래는 사전에 정한 갤러리 표본 중 직접 확인한 두 사례입니다. 이 두 장이 전체 성공·실패 비율을 대표한다는 뜻은 아닙니다. "
        "둘 다 왼쪽 위 측면 경계(role 4)를 비교합니다.", "",
        "- [기본 실사 사례](visualizations/006_eval_cad__1778653055734035712/role_04.png): "
        "Direct는 9.77°·27.46px, DHT는 0.73°·8.32px입니다. DHT의 선이 왼쪽 위 경계 방향에 가깝고, "
        "특징 민감도는 그 경계와 양 끝 모서리 주변에 나타납니다. Hough 확률도 정답 가까이에 모여 있습니다.",
        "- [방향 오류가 남은 야외 사례](visualizations/011_eval_outside__1778651530691638016/role_04.png): "
        "Direct는 36.64°·48.35px, DHT는 23.49°·16.98px입니다. DHT 선이 팔레트 왼쪽 끝점 근처를 지나지만 "
        "정답 방향과는 크게 다릅니다. 민감도가 끝점 주변뿐 아니라 예측 선을 따라 배경에도 나타나고, "
        "Hough 확률은 여러 각도·위치 후보를 연결하는 길쭉한 띠 모양입니다.", "",
        "짧게 투영된 경계에서 한 끝점 주변 정보만 강하면 여러 방향이 비슷한 지지를 받을 수 있고, "
        "선 위 배경 특징이 후보 구분을 어렵게 할 수 있다는 가설을 세울 수 있습니다. "
        "그러나 이 지도는 국소 특징 민감도이며 인과적 근거가 아닙니다. 배경을 바꾸거나 가리는 대조 실험과 "
        "투영된 경계 길이별 평가로 이 가설을 별도로 검증해야 합니다.", "",
        "## 학습과 정답", "",
        f"합성 {counts['synth_train']:,}장으로만 학습했습니다. 검증 {counts['synth_val']:,}장, "
        f"동일 생성 계열 평가 {counts['synth_test']:,}장, 다른 합성 {counts['cross_v4']:,}장, 실사 개발셋 {counts['real_dev']:,}장을 평가했습니다. "
        f"두 구조 모두 seed {', '.join(map(str, seeds))}, {cfg['steps']:,} step, batch {cfg['batch']}, "
        f"AdamW(lr={cfg['lr']}, weight_decay={cfg['weight_decay']})로 학습하고 마지막 체크포인트를 사용했습니다. "
        "실사 이미지는 학습, 임계값 조정, 체크포인트 선택에 사용하지 않았습니다.", "",
        "두 모델은 같은 고정 합성 사전학습 DOPE VGG 특징(128×50×50), 같은 입력·배치 순서·8개 정답 역할·Hough 격자·soft CE 손실을 사용합니다. "
        "Direct는 역할별 전역 attention으로 만든 descriptor가 후보 선에 점수를 줍니다. DHT는 XY 좌표와 이미지 특징을 합친 뒤 "
        "1×1 및 두 3×3 합성곱으로 16채널 특징을 만들고, 후보 선을 따라 그 특징을 집계합니다. "
        "이후 선 좌표·집계 질량 정보를 더해 파라미터 공간의 이웃 선을 합성곱으로 비교하고 8개 역할의 점수를 냅니다. "
        "구조와 파라미터 수가 달라 하나의 요소만 바꾼 실험은 아닙니다.", "",
        "[원 논문](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540239.pdf)의 특징 집계 원리를 적용한 과제용 DHT입니다. "
        "원 논문의 다중 해상도 ResNet-FPN 모델을 그대로 재현하지 않았습니다. 이 구현은 각 픽셀의 rho 투표를 이웃 2개 bin에 선형 분배하고, "
        "집계 질량으로 나눈 평균을 사용합니다. θ 경계를 넘을 때 rho 부호가 뒤집히도록 합성곱 패딩을 처리합니다.", "",
        "정답은 좌우 측면을 둘러싼 구조적 cuboid 직선 8개이며 숨겨진 선도 포함합니다. 화면에 나타나는 재료 엣지의 직접 주석은 아닙니다. "
        "유효한 두 정답 끝점, 원본 화면과 교차하는 선분, 길이 2px 이상인 역할만 평가합니다. "
        "예측은 무한직선이며 정확한 유한 선분 끝점 추정까지 평가한 결과는 아닙니다.", "",
        "입력은 반사 패딩 100px 후 400×400으로 변환합니다. 좌표는 OpenCV resize의 반 픽셀 위치와 VGG stride 8의 "
        "첫 수용영역 중심 3.5를 반영해 `(원본좌표 + 100 + 0.5) × 50 / 패딩영상크기 − 0.5`로 맞췄습니다. "
        "기존 캐시의 연속 0…50 정답 좌표만 보정했으며 원본 정답점이나 특징 캐시는 수정하지 않았습니다. "
        "이전 12개 선·3,000 step 실험과는 역할 수·학습량·좌표계가 달라 그 수치를 직접 개선율의 기준으로 사용하지 않습니다.", "",
        "![학습 손실 곡선](learning_curves.png)", "",
        "첫 점은 첫 batch 손실이며 이후 점은 직전 기록 이후(통상 500 step)의 평균 손실입니다. "
        "색으로 구조, 선 모양으로 seed를 구분했습니다. 학습 손실 곡선이며 검증 손실이나 실사 성능 곡선이 아닙니다. "
        f"마지막 구간 평균 손실은 {len(ARMS)*len(seeds)}개 실행 중 {final_windows_decreased}개에서 직전 구간보다 낮았습니다. "
        "추가 학습의 영향을 배제할 수 없고, 이 곡선만으로 충분히 수렴했다고 판단하지 않습니다.", "",
        f"완료 검증에서 {len(ARMS)*len(seeds)}개 체크포인트와 각 학습 이력의 마지막 step이 모두 {cfg['steps']:,}임을 확인했습니다. "
        "체크포인트 SHA·설정 SHA·학습 코드 SHA·평가 행 수는 [COMPLETION.json](COMPLETION.json)에 저장했습니다.", "",
        "## 검증과 해석 범위", "",
    ]
    if checks:
        oracle = checks["target_peak_oracle"]["real_dev"]
        lines.append(
            f"θ {cfg['theta_bins']}개(1° 간격), ρ 0.5 특징 셀 간격의 동일 격자를 사용했습니다. "
            f"실사 정답 분포에서 가장 높은 bin을 직접 고르는 검증은 위치 중앙값 {oracle['distance_px']['median']:.2f}px, "
            f"P90 {oracle['distance_px']['p90']:.2f}px, 방향 중앙값 {oracle['angle_deg']['median']:.2f}°입니다. "
            "정답을 사용한 격자 진단이므로 학습 모델의 실사 성능이 아닙니다."
        )
        lines += ["", f"시작 시 두 모델의 순전파·역전파·유한 손실을 확인했습니다. "
                  f"모델 전체 파라미터 수는 Direct {checks['Direct']['parameters']:,}, DHT {checks['DHT']['parameters']:,}개입니다. "
                  "DHT 연산자는 별도로 CPU/CUDA 좌표·정규화·gradient·θ 경계 패딩 검증을 통과했습니다."]
    if sanity:
        lines += ["", f"본학습과 별도로 합성 32장 DHT sanity를 수행했습니다. 저장된 마지막 위치 중앙값은 "
                  f"{sanity['distance_px']['median']:.2f}px, 방향 중앙값은 {sanity['angle_deg']['median']:.2f}°입니다. "
                  "sanity 수치를 실사 전이 성능으로 사용하지 않습니다."]
    audit = manifest.get("audit", {})
    lines += [
        "", f"선정 이미지의 SHA 중복 없음: {audit.get('image_sha256_disjoint')}; 생성 묶음 단위 분리: {audit.get('group_disjoint')}. "
        f"이번 원본 재검증에서 이미지·주석 {analysis['source_revalidation']['file_count']:,}개 파일의 SHA가 manifest와 모두 일치했고, "
        "학습 인덱스에 실사 개발셋이 포함되지 않음을 확인했습니다. "
        "같은 자산이 여러 합성 분할에 등장하므로 미사용 mesh 일반화를 보장하지 않습니다. 백본의 과거 합성 사전학습 원본 목록이 없어 "
        "그 자료와 현재 합성 평가셋의 전체 중복 여부는 검증하지 못했습니다.", "",
        f"보정한 좌표의 원본 왕복 최대 위치 오차는 {target_audit.get('roundtrip_max_distance_px', float('nan')):.6f}px입니다. "
        "기하 감사의 카메라 방향 분류는 이용 가능한 합성 3D 정보와 일치했습니다. 실사에는 동일한 3D 가시성 검증 자료가 없습니다.", "",
        "[기하 감사](target_audit/TARGET_AUDIT.json) · [학습 전 점검](CHECKS.json) · [32장 sanity](SANITY.json) · "
        "[실행 설정](CONFIG.json) · [출처 기록](PROVENANCE.json) · [원본 파일 재검증](SOURCE_REVALIDATION.json)", "",
        "갤러리는 사전에 고정한 SHA 순서와 첫 seed로 선택한 사례입니다. 열 지도는 예측 최고점 logit에 대한 입력 VGG 특징의 "
        "`|gradient × feature|`를 채널별로 더한 민감도입니다. Attention이나 인과적 픽셀 중요도와 같은 의미로 해석할 수 없습니다. "
        "역할별로 밝기를 정규화하므로 그림 사이 절대 크기를 비교하지 않습니다. Hough 열 지도는 역할별 실제 softmax 확률입니다.", "",
    ]
    return "\n".join(lines)


def analyze(out):
    out = Path(out).resolve()
    result = read_json(out / "RESULTS.json")
    cfg = result["config"]
    if cfg != read_json(out / "CONFIG.json"):
        raise ValueError("RESULTS.json and CONFIG.json describe different experiments")
    seeds = cfg["seeds"]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Configured seeds must be unique")
    pairs = matched_rows(out / "PER_ROLE.csv", seeds)
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["population"]].append(pair)
    populations = {population: seed_statistics(these, seeds) for population, these in grouped.items()}
    verify_saved_summary(result, populations, seeds)
    matched = {}
    for population in ("real_dev", "synth_test"):
        these = grouped[population]
        matched[population] = {
            "subsets": {name: subset_report([p for p in these if predicate(p)], seeds)
                        for name, predicate in SUBSETS.items()},
            "by_group": {group: subset_report([p for p in these if p["group"] == group], seeds)
                         for group in sorted({p["group"] for p in these})},
            "by_role": {str(role): subset_report([p for p in these if p["role"] == role], seeds)
                        for role in range(8)},
        }
    status, conclusion = verdict(matched["real_dev"]["subsets"]["all_eight"])
    source = Path(cfg["source_cache"])
    manifest = read_json(source / "manifest.json")
    target_audit = read_json(out / "target_audit/TARGET_AUDIT.json")
    source_revalidation = read_json(out / "SOURCE_REVALIDATION.json")
    if not source_revalidation["PASS"] or source_revalidation["manifest_sha256"] != cfg["source_manifest_sha256"]:
        raise ValueError("Source revalidation did not pass or refers to a different manifest")
    checks = read_json(out / "CHECKS.json") if (out / "CHECKS.json").exists() else {}
    sanity = read_json(out / "SANITY.json") if (out / "SANITY.json").exists() else {}
    diagnosis = location_angle_diagnosis(pairs, manifest, cfg["edges"], seeds)
    learning = learning_curves(out, seeds)
    completion = verify_completion(out, cfg, populations, learning)
    analysis = {
        "schema": "deep_hough_side_v1_matched_analysis",
        "config": cfg,
        "input_sha256": {name: sha(out / name) for name in ("RESULTS.json", "PER_ROLE.csv", "CONFIG.json")},
        "aggregation": "Within each seed, pool supported role errors and compute median/p90/mean; average these statistics across configured seeds.",
        "paired_bootstrap": "Exact pair by population/frame id/role/seed. Average supported roles within each frame, then average fixed seeds per frame, then resample paired frames 2000 times using RNG seed42. Negative DHT-minus-Direct favors DHT.",
        "bootstrap_scope": "Descriptive conditional intervals for recorded frames and fixed seeds. No capture-session dependence model, seed-population uncertainty, multiple-comparison correction, or independent final-test claim.",
        "role_subsets": {"height_four": [0, 1, 2, 3], "depth_four": [4, 5, 6, 7],
                         "camera_facing": "GT projected side-face orientation; not physical visibility"},
        "population_summary": populations,
        "matched": matched,
        "location_angle_diagnosis": diagnosis,
        "learning_curves": learning,
        "source_revalidation": {
            key: source_revalidation[key] for key in ("PASS", "file_count", "record_count", "manifest_sha256",
                                                      "all_source_hashes_match", "no_real_index_in_training_population")
        },
        "completion": {"PASS": completion["PASS"], "expected_steps": completion["expected_steps"],
                       "expected_runs": completion["expected_runs"], "evaluation_csv_rows": completion["evaluation_csv_rows"]},
        "verdict": status,
        "conclusion_ko": conclusion,
    }
    write_json(out / "ANALYSIS.json", analysis)
    write_json(out / "COMPLETION.json", completion)
    (out / "REPORT.md").write_text(markdown_report(out, analysis, result, manifest, target_audit, checks, sanity))
    print(json.dumps({"verdict": status, "conclusion_ko": conclusion,
                      "real_all_eight": matched["real_dev"]["subsets"]["all_eight"]["paired_frame_bootstrap"]["distance_px"]},
                     ensure_ascii=False, indent=2))
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    analyze(parser.parse_args().run_dir)
