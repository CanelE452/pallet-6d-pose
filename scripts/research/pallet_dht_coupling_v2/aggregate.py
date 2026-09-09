"""Aggregate actual full-model DEV evaluations with matched-seed/session inference.

No prediction, training, final-holdout access, or selection is performed here.
The old tested statistic primitives are reused, not the old frozen-branch study.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIMITIVES = ROOT / "scripts/research/pallet_line_pose_v1/aggregate_results.py"
spec = importlib.util.spec_from_file_location("paper_paired_statistic_primitives", PRIMITIVES)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
NEW_ARMS = ("balanced", "pcgrad", "balanced_pcgrad", "incidence")
REFERENCE_ARMS = ("point_only", "hough_features", "hough_joint")
ARMS = REFERENCE_ARMS + NEW_ARMS
SEEDS = (1, 2, 3)
PRIMARY_REFERENCES = ("hough_joint", "point_only")
FAMILY_SIZE, FAMILY_ALPHA, SESSION_RESAMPLES = 48, .05, 100000
V1_ROOT = ROOT / "data/pallet/results/pallet_dht_joint_v1"
METRICS, KEYPOINT_METRICS, POSE_METRICS = S.METRICS, S.KEYPOINT_METRICS, S.POSE_METRICS
read, sha, write, bound, checked = S.read, S.sha, S.write, S.bound, S.checked
POSE = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
R0_2D = ROOT / "data/pallet/results/paper_eval_v1/arms/R0.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nullable_stats(values):
    if any(value is None for value in values):
        return dict(mean=None, std=None, seeds=values, status="MISSING_EVALUABLE_OBSERVATIONS",
                    std_definition="sample standard deviation across three trained seeds, ddof=1")
    return S.mean_std(values)


def validate_cell(run_dir, arm, seed):
    label = f"{arm}_seed{seed}"
    directory = run_dir / "evaluation" / label
    result, done, protocol = (read(directory / name) for name in ("RESULTS.json", "COMPLETION.json", "EVALUATION_PROTOCOL.json"))
    require(result.get("complete") and result.get("PASS") and done.get("complete") and done.get("PASS"), f"Incomplete {label}")
    require(result["arm"] == arm and result["seed"] == seed == done["seed"], "Run identity mismatch")
    require(result["population"] == dict(positive=319, negative=2689, role="DEV", held_out_final=False), "DEV population changed")
    for payload in (result, done):
        require(payload.get("actual_positive_forwards") == 319 and payload.get("actual_negative_forwards") == 2689
                and payload.get("baseline_candidate_copying") is False, "Actual new full-model forwards are required")
    checked(directory / "EVALUATION_PROTOCOL.json", done["evaluation_protocol_sha256"])
    checked(result["checkpoint"], done["checkpoint_sha256"])
    for path, digest in done["output_sha256"].items():
        checked(directory / path, digest)
    for path, digest in done["source_sha256"].items():
        checked(path, digest)
    require(protocol["protocol"]["sha256"] == sha(run_dir / "TRAIN_PROTOCOL.json"), "Training protocol changed")
    training = protocol["completed_training"]
    require(training["main_training_complete"], "Completed main training is required")
    config = read(training["cell_config"])
    trained = read(training["completion"])
    checked(training["cell_config"], training["cell_config_file_sha256"])
    checked(training["completion"], training["completion_sha256"])
    require(trained["stage"] == "main" and trained["complete"] and trained["PASS"], "Smoke cannot enter main aggregation")
    checked(Path(training["completion"]).with_name("history.json"), trained["history_sha256"])
    checked(Path(training["completion"]).with_name("BATCH_TRACE.jsonl"), trained["batch_trace_sha256"])
    require(trained["checkpoint_sha256"] == result["checkpoint_sha256"], "Training/evaluation checkpoint differs")
    return directory, result, trained, config


def interval(draws, lower, alpha=.05):
    draws = np.asarray(draws, dtype=np.float64)
    require(draws.ndim == 1 and len(draws) and np.isfinite(draws).all(), "Invalid bootstrap draws")
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return dict(low=float(low), high=float(high), nominal_coverage=1-alpha,
        excludes_zero=bool(low > 0 or high < 0),
        confirmed_benefit=bool(high < 0 if lower else low > 0),
        bootstrap_fraction_better=float(np.mean(draws < 0 if lower else draws > 0)),
        interpretation="Conditional percentile-bootstrap approximation; bootstrap fraction is not a p-value.")


def paired_metric(left, right, metric, session_resamples=SESSION_RESAMPLES, frame_resamples=10000):
    """Same paired trained seeds and same resampled IDs; never treat seeds as frames."""
    require(len(left) == len(right) == 3, "Exactly three matched trained seeds are required")
    stores = left + right
    keys = sorted(set.intersection(*(set(store) for store in stores)))
    if metric in KEYPOINT_METRICS:
        keys = [key for key in keys if all(len(store[key]["errors"]) for store in stores)]
    if not keys:
        absent = dict(low=None, high=None, confirmed_benefit=False)
        return dict(difference=None, paired_frames=0, paired_sessions=0, status="NO_COMMON_OBSERVATIONS",
                    frame_level=absent, session_cluster=absent, session_simultaneous=absent)
    for key in keys:
        require(len({store[key]["session_id"] for store in stores}) == 1, "Paired session assignments differ")
    sessions = sorted({left[0][key]["session_id"] for key in keys})
    require(all(sessions), "Missing session ID")
    seed = 20260902 if metric in KEYPOINT_METRICS else 20260903
    result, observed = {}, None
    for scheme, count in (("frame_level", frame_resamples), ("session_cluster", session_resamples)):
        lookup = ({key: i for i, key in enumerate(keys)} if scheme == "frame_level" else
                  {key: sessions.index(left[0][key]["session_id"]) for key in keys})
        n = len(keys) if scheme == "frame_level" else len(sessions)
        series = [S.prepare_series(store, keys, metric, lookup) for store in stores]
        if observed is None:
            ones = np.ones((1, n), dtype=np.int32)
            statistics = [S.weighted_statistic(values, groups, ones, metric, diameters)[0]
                          for values, groups, diameters in series]
            observed = float(np.mean(statistics[:3]) - np.mean(statistics[3:]))
        # Separate fixed streams keep session draws independent of the number of
        # optional descriptive frame draws. Every candidate uses the same stream.
        rng = np.random.default_rng(seed + (1 if scheme == "session_cluster" else 0))
        draws = np.empty(count, dtype=np.float64)
        for start in range(0, count, 128):
            size = min(128, count-start)
            weights = rng.multinomial(n, np.full(n, 1/n), size=size).astype(np.int32)
            statistics = np.stack([S.weighted_statistic(values, groups, weights, metric, diameters)
                                   for values, groups, diameters in series])
            draws[start:start+size] = statistics[:3].mean(0) - statistics[3:].mean(0)
        result[scheme] = interval(draws, metric in S.LOWER)
        result[scheme]["resamples"] = count
        if scheme == "session_cluster":
            result["session_simultaneous"] = interval(draws, metric in S.LOWER, FAMILY_ALPHA/FAMILY_SIZE)
            result["session_simultaneous"].update(method="bonferroni_percentile", family_size=FAMILY_SIZE,
                family_alpha=FAMILY_ALPHA, resamples=count, adjusted_alpha=FAMILY_ALPHA/FAMILY_SIZE)
    return dict(difference=observed, better="lower" if metric in S.LOWER else "higher",
        paired_frames=len(keys), paired_sessions=len(sessions),
        per_seed_available_frames_left=[len(store) for store in left],
        per_seed_available_frames_right=[len(store) for store in right],
        frames_not_used_left=[len(store)-len(keys) for store in left],
        frames_not_used_right=[len(store)-len(keys) for store in right],
        seed=seed, **result,
        estimand="Mean of three matched seed-specific evaluator statistic differences on common observed frames; cluster resampling preserves seed pairing.")


def reference_decision(summary, arm, reference):
    candidate, control = summary["arms"][arm], summary["arms"][reference]
    comparison = summary["comparisons"][f"{arm}_vs_{reference}"]["metrics"]
    criteria = {}
    for name in METRICS:
        a, b = candidate["metrics"][name]["mean"], control["metrics"][name]["mean"]
        gain = a is not None and b is not None and (a < b if name in S.LOWER else a > b)
        ci = comparison[name]["session_simultaneous"]["confirmed_benefit"]
        criteria[name] = dict(seed_mean_improved=bool(gain), simultaneous_session_ci_supports_improvement=bool(ci),
            descriptive_95_session_ci_supports_improvement=bool(comparison[name]["session_cluster"]["confirmed_benefit"]),
            confirmed=bool(gain and ci))
    rows = {(r["arm"], r["seed"]): r for r in summary["runs"]}
    kp_coverage = all(rows[arm, seed]["two_d"]["keypoint_matched_frame_count_iou50"] >=
                      rows[reference, seed]["two_d"]["keypoint_matched_frame_count_iou50"] for seed in SEEDS)
    pose_coverage = all(rows[arm, seed]["coverage"] >= rows[reference, seed]["coverage"] for seed in SEEDS)
    kp = all(criteria[m]["confirmed"] for m in KEYPOINT_METRICS) and kp_coverage
    pose = all(criteria[m]["confirmed"] for m in POSE_METRICS) and pose_coverage
    return dict(reference=reference, metric_criteria=criteria, keypoint_gain_confirmed=bool(kp),
                pose_gain_confirmed=bool(pose), overall_accuracy_improved=bool(kp and pose),
                keypoint_coverage_preserved=kp_coverage, pose_coverage_preserved=pose_coverage)


def make_verdict(summary):
    candidates = {}
    for arm in NEW_ARMS:
        references = {reference: reference_decision(summary, arm, reference) for reference in PRIMARY_REFERENCES}
        candidates[arm] = dict(references=references,
            meets_both_reference_criteria=all(row["overall_accuracy_improved"] for row in references.values()))
    any_gain = any(row["meets_both_reference_criteria"] for row in candidates.values())
    headline = ("사전지정한 결합 방법 중 양쪽 대조군 대비 동시 개선 기준을 충족한 방법이 있습니다."
                if any_gain else "점·선 결합 학습의 두 대조군 대비 사전지정 종합 개선 기준은 확인되지 않았습니다.")
    reasons = [f"{arm}: 기존 joint와 점 전용 양쪽 대비 6지표·동시 구간·coverage 기준 " +
               ("충족" if row["meets_both_reference_criteria"] else "미충족") for arm, row in candidates.items()]
    return dict(schema="pallet_dht_coupling_verdict_v2", complete=True, PASS=True,
        PASS_semantics="Artifact and statistic integrity; not runtime parity or model improvement.",
        overall_accuracy_improved=any_gain, candidates=candidates, headline_ko=headline, reasons=reasons,
        new_arms=list(NEW_ARMS), primary_references=list(PRIMARY_REFERENCES), runs=summary["runs"],
        selected_winning_arm=None, real_model_selection_performed=False,
        multiplicity=dict(family_size=FAMILY_SIZE, family_alpha=FAMILY_ALPHA, method="bonferroni_percentile", session_draws=SESSION_RESAMPLES),
        discord_lines_ko=[headline, "4방법×2대조군×6지표의 48비교에 동시 구간을 적용했습니다. 실사 DEV는 재사용 데이터입니다."],
        decision="Each candidate/reference claim requires all six full-population seed means and Bonferroni session intervals to improve, with coverage preserved for every paired seed. All candidates remain reported; no real-data winner selection.")


def verify_reference_manifest(run_dir, protocol=None):
    reference = protocol.get("reuse_control_manifest", run_dir / "REUSED_CONTROLS.json") if protocol else run_dir / "REUSED_CONTROLS.json"
    path = Path(reference["path"] if isinstance(reference, dict) else reference)
    if isinstance(reference, dict):
        checked(path, reference["sha256"])
    value = read(path)
    require(value.get("schema") == "pallet_dht_coupling_reused_controls_v2" and value.get("complete") and value.get("PASS"),
            "Audited reused-control manifest is required")
    require(value["n_control_cells"] == len(value["controls"]) == 9, "Exactly nine reused controls required")
    require({(c["arm"], c["seed"]) for c in value["controls"]} == {(a, s) for a in REFERENCE_ARMS for s in SEEDS},
            "Control identities differ")
    for key, digest in value["input_sha256"].items():
        checked(key, digest)
    for control in value["controls"]:
        require(control["reused"] is True and control["new_v2_accuracy_forwards"] == 0, "Reused runs cannot be labeled new inference")
        for artifact in [control["checkpoint"], *control["artifacts"].values()]:
            checked(artifact["path"], artifact["sha256"])
    if protocol:
        for arm in REFERENCE_ARMS:
            require(protocol["references"][arm]["arm"] == arm
                    and Path(protocol["references"][arm]["run_dir"]).resolve() == Path(value["control_run_dir"]).resolve(),
                    "Frozen control reference path differs")
        if protocol.get("reuse_control_manifest_sha256"):
            checked(path, protocol["reuse_control_manifest_sha256"])
    return value


def view_diagnostics(kp_stores, pose_stores, source_root):
    strata_path = source_root / "VIEW_STRATA.json"
    strata = read(strata_path)
    require(strata["frozen_before_new_real_predictions"] and strata["model_predictions_read"] == 0,
            "Existing predeclared view strata required")
    for path, digest in strata["source_sha256"].items():
        checked(path, digest)
    require(len(strata["records"]) == 319, "View strata denominator differs")
    rows = []
    for axis in ("elevation_bin", "frontness_bin"):
        for group in sorted({r[axis] for r in strata["records"]}):
            members = [r for r in strata["records"] if r[axis] == group]
            kp_keys = {r["frame_id"] for r in members}
            pose_keys = {r["pose_frame_id"] for r in members}
            for arm in ARMS:
                per_seed = []
                for index, seed in enumerate(SEEDS):
                    kp = [v for key, v in kp_stores[arm][index].items() if key in kp_keys]
                    pose = [v for key, v in pose_stores[arm][index].items() if key in pose_keys]
                    require(len(kp) == len(members), "View/2D join dropped frames")
                    per_seed.append(dict(seed=seed, matched_frames=sum(len(r["errors"]) > 0 for r in kp),
                        pose_frames=len(pose), metrics={**S.keypoint_summary(kp), **S.pose_summary(pose)}))
                rows.append(dict(arm=arm, axis=axis, group=group, n_frames=len(members),
                    n_sessions=len({r["session_id"] for r in members}),
                    metrics={m: nullable_stats([r["metrics"].get(m) for r in per_seed]) for m in METRICS}, per_seed=per_seed))
    return dict(schema="pallet_dht_coupling_view_diagnostics_v2", complete=True, rows=rows,
        strata=bound(strata_path), counts=strata["counts"], primary_verdict_changed=False,
        meaning="Predeclared v1 reconstructed elevation and 2D face-area proxies; descriptive only, not calibrated yaw or physical visibility. No subgroup hypothesis testing or real selection.")


def aggregate(run_dir):
    run_dir = Path(run_dir).resolve()
    require((run_dir / "PURPOSE.md").is_file(), "Purpose-declared v2 output root required")
    protocol = read(run_dir / "TRAIN_PROTOCOL.json")
    require(protocol["schema"] == "pallet_dht_coupling_protocol_v2" and protocol["arms"] == list(NEW_ARMS), "Frozen v2 protocol differs")
    expected_statistics = dict(family_size=48, family_alpha=.05, session_resamples=100000,
                              frame_resamples=10000, method="bonferroni_percentile", primary_references=list(PRIMARY_REFERENCES))
    require(all(protocol["statistics"].get(k) == v for k, v in expected_statistics.items()), "Registered multiplicity/draw budget differs")
    controls = verify_reference_manifest(run_dir, protocol)
    control_root = Path(controls["control_run_dir"])
    baseline = dict(two_d=read(R0_2D)["metrics"]["box_and_keypoint_2d"],
        main_6d=read(POSE / "POSE_EVALUATION_R0.json")["paths"]["MAIN"]["ALL"],
        coverage=read(POSE / "POSE_EVALUATION_R0.json")["paths"]["MAIN"]["coverage"],
        role="Historical R0 reference only; primary references are reused same-budget joint and point_only.")
    r0_kp = S.load_keypoint_rows(R0_2D.with_name("R0_per_frame.csv"))
    kp_stores, pose_stores, runs, sources, budgets = {}, {}, [], {}, []
    traces = {seed: [] for seed in SEEDS}
    for arm in ARMS:
        kp_stores[arm], pose_stores[arm] = [], []
        root = control_root if arm in REFERENCE_ARMS else run_dir
        for seed in SEEDS:
            directory, result, trained, config = validate_cell(root, arm, seed)
            label = f"{arm}_seed{seed}"
            require(trained["optimizer_steps"] == 6998 and trained["epochs_completed"] == 2, "Same fixed final budget required")
            traces[seed].append(trained["batch_trace_sha256"])
            require(trained["batch_trace_sha256"] == controls["original_trace_sha256_by_seed"][str(seed)],
                    "New/control actual augmented batch traces differ")
            budgets.append({key: config[key] for key in ("epochs", "batch", "lr", "optimizer", "amp", "train_frames", "val_frames", "training_recipe")})
            kp = S.load_keypoint_rows(directory / "PAPER_2D_per_frame.csv")
            pose = {row["frame_id"]: row for row in read(directory / "POSE_PER_FRAME_BY_ARM.json")["per_frame"][label]}
            require(set(kp) == set(r0_kp) and all(kp[k]["session_id"] == r0_kp[k]["session_id"] for k in kp),
                    "319 canonical frame/session identities differ")
            for fields, expected in ((S.keypoint_summary(list(kp.values())), result["two_d"]),
                                     (S.pose_summary(list(pose.values())), result["main_6d"])):
                for metric in METRICS:
                    if metric in fields:
                        if fields[metric] is None:
                            require(expected.get(metric) is None, "Missing metric was fabricated")
                        else:
                            S.close(fields[metric], expected[metric], label+"/"+metric)
            kp_stores[arm].append(kp)
            pose_stores[arm].append(pose)
            sessions = sorted({row["session_id"] for row in kp.values()})
            require(len(sessions) == 13, "Expected13 capture sessions")
            per_session = {sid: dict(two_d=S.keypoint_summary([r for r in kp.values() if r["session_id"] == sid]),
                main_6d=S.pose_summary([r for r in pose.values() if r["session_id"] == sid])) for sid in sessions}
            gradient = None
            if arm in NEW_ARMS:
                gradient_path = root / "runs" / label / "GRADIENT_AUDIT.json"
                gradient = read(gradient_path)
                require(gradient.get("complete") and gradient["arm"] == arm, "Actual gradient diagnostic audit incomplete")
            runs.append(dict(arm=arm, seed=seed, evaluation_arm=label, run_dir=str(root), reused=arm in REFERENCE_ARMS,
                two_d=result["two_d"], main_6d=result["main_6d"], coverage=result["main_6d_coverage"],
                negative=result["negative"], per_session=per_session, training=trained,
                gradient_audit=bound(gradient_path) if gradient is not None else None))
            sources[label] = {name: bound(directory / name) for name in ("RESULTS.json", "COMPLETION.json", "EVALUATION_PROTOCOL.json",
                "PAPER_2D_per_frame.csv", "POSE_PER_FRAME_BY_ARM.json", "NEGATIVE_OUTCOMES.json", "LINE_EVIDENCE.json")}
    require(all(value == budgets[0] for value in budgets), "New/reused runs differ in source data, optimizer, budget or augmentation recipe")
    families = {}
    for arm in ARMS:
        rows = [r for r in runs if r["arm"] == arm]
        families[arm] = dict(reused=arm in REFERENCE_ARMS,
            metrics={m: nullable_stats([r["two_d" if m in KEYPOINT_METRICS else "main_6d"].get(m) for r in rows]) for m in METRICS},
            pose_coverage=nullable_stats([r["coverage"] for r in rows]),
            keypoint_matched_frames=nullable_stats([r["two_d"]["keypoint_matched_frame_count_iou50"] for r in rows]),
            box_ap50=nullable_stats([r["two_d"]["box_ap50"] for r in rows]),
            box_ap50_95=nullable_stats([r["two_d"]["box_ap50_95"] for r in rows]),
            negative={t: nullable_stats([r["negative"]["by_threshold"][t]["fraction_frames_with_detection"] for r in rows])
                      for t in ("0.001", "0.25", "0.5", "0.85")},
            per_session={sid: dict(metrics={m: nullable_stats([r["per_session"][sid]["two_d" if m in KEYPOINT_METRICS else "main_6d"].get(m) for r in rows])
                                           for m in METRICS}) for sid in rows[0]["per_session"]})
    comparisons = {}
    for arm in NEW_ARMS:
        for reference in PRIMARY_REFERENCES:
            row = dict(left=arm, right=reference, primary_family_member=True, metrics={})
            for metric in METRICS:
                stores = kp_stores if metric in KEYPOINT_METRICS else pose_stores
                row["metrics"][metric] = paired_metric(stores[arm], stores[reference], metric)
                print(f"100k paired session bootstrap {arm} vs {reference}: {metric}", flush=True)
            comparisons[f"{arm}_vs_{reference}"] = row
    views = view_diagnostics(kp_stores, pose_stores, control_root)
    runtime = read(run_dir / "RUNTIME.json") if (run_dir / "RUNTIME.json").exists() else dict(complete=False, status="NOT_MEASURED")
    summary = dict(schema="pallet_dht_coupling_summary_v2", complete=True, PASS=True, new_arms=list(NEW_ARMS),
        primary_references=list(PRIMARY_REFERENCES), population=dict(positive=319, negative=2689, sessions=13, role="DEV", held_out_final=False),
        baseline=baseline, runs=runs, arms=families, comparisons=comparisons, runtime=runtime, training_budget=budgets[0],
        n_completed_evaluations=12, n_verified_reference_evaluations=9,
        identity=dict(actual_new_positive_forwards=319*12, actual_new_negative_forwards=2689*12,
            reused_reference_evaluations=9, reused_reference_new_forwards=0, baseline_candidate_copying=False,
            matched_augmented_batch_traces=True, trace_sha256_by_seed=traces),
        methods={**expected_statistics, "ddof": 1, "seed_aggregation": "mean of three paired evaluator statistics",
            "primary_ci": "session_simultaneous", "descriptive_ci": "session_cluster",
            "interpretation": "Bonferroni percentile intervals approximate simultaneous coverage conditional on13 reused capture sessions; no fabricated p-values and no independent-final claim."},
        limitations=["실사319 positive와2689 negative는 반복 사용한 DEV이며 독립 final을 평가하지 않았습니다.",
            "기존9개 대조군은 검증 후 재사용했습니다. 새12개 모델만 이번에 실제 추론했습니다.",
            "48개 사전지정 지표 비교에 Bonferroni percentile 구간을 적용합니다. 13세션에 조건부인 bootstrap 근사이며 확정적 보장은 아닙니다.",
            "관측 가능한 공통 프레임의 대응 차이와 각 모델 전체 매칭 모집단 평균은 다릅니다. Coverage를 별도로 유지해야 합니다.",
            "실사에서 우승 방법이나 체크포인트를 선택하지 않습니다. 네 후보를 모두 보고합니다.",
            "6D GT와 앙각은 기존 기하 재구성입니다. 면적비는 yaw 또는 물리 가시성 실측이 아닙니다.",
            "실제 선 증거는 attention, 인과 설명 또는 보정된 정답 확률이 아닙니다. Gradient 충돌 감소도 성능 향상의 증명은 아닙니다.",
            "시간 수집 완료와 엄격한 예측 parity는 별개입니다. 실패가 있으면 숨기지 않고 참고 시간으로 표시합니다."],
        sources=sources, source_bindings=dict(training_protocol=bound(run_dir / "TRAIN_PROTOCOL.json"),
            reused_controls=bound(run_dir / "REUSED_CONTROLS.json"), statistic_primitives=bound(PRIMITIVES), aggregate=bound(__file__)))
    verdict = make_verdict(summary)
    for name, value in (("SUMMARY.json", summary), ("VERDICT.json", verdict), ("VIEW_DIAGNOSIS.json", views)):
        write(run_dir / name, value)
    input_hashes = {str(run_dir / "TRAIN_PROTOCOL.json"): sha(run_dir / "TRAIN_PROTOCOL.json"),
                   str(run_dir / "REUSED_CONTROLS.json"): sha(run_dir / "REUSED_CONTROLS.json"), str(PRIMITIVES): sha(PRIMITIVES)}
    for artifacts in sources.values():
        input_hashes.update({a["path"]: a["sha256"] for a in artifacts.values()})
    write(run_dir / "AGGREGATE_COMPLETE.json", dict(complete=True, PASS=True, n_completed_evaluations=12,
        n_verified_reference_evaluations=9, n_gallery_model_runs=21, overall_accuracy_improved=verdict["overall_accuracy_improved"],
        input_sha256=input_hashes, source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        output_sha256={str(run_dir / n): sha(run_dir / n) for n in ("SUMMARY.json", "VERDICT.json", "VIEW_DIAGNOSIS.json")}))
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    answer = aggregate(parser.parse_args().run_dir)
    print(json.dumps({k: answer[k] for k in ("complete", "PASS", "overall_accuracy_improved", "headline_ko")}, ensure_ascii=False))
