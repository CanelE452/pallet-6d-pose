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
ARMS = ("point_only", "hough_features", "hough_joint")
SEEDS = (1, 2, 3)
PRIMARY, CONTROL = "hough_joint", "point_only"
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


def compare(stores, left, right, metric):
    # Missing predictions remain in coverage; absent paired metrics are explicit.
    keysets = [set(rows) for rows in stores[left] + stores[right]]
    common = set.intersection(*keysets)
    if metric in KEYPOINT_METRICS:
        common = {key for key in common if all(len(rows[key]["errors"]) for rows in stores[left] + stores[right])}
    if not common:
        return dict(difference=None, paired_frames=0, paired_sessions=0, status="NO_COMMON_OBSERVATIONS",
                    frame_level=dict(low=None, high=None, confirmed_benefit=False),
                    session_cluster=dict(low=None, high=None, confirmed_benefit=False))
    return S.paired_metric(stores[left], stores[right], metric, resamples=10000)


def make_verdict(summary):
    primary, control = summary["arms"][PRIMARY], summary["arms"][CONTROL]
    comparison = summary["comparisons"][f"{PRIMARY}_vs_{CONTROL}"]["metrics"]
    criteria = {}
    for name in METRICS:
        value, reference = primary["metrics"][name]["mean"], control["metrics"][name]["mean"]
        gain = value is not None and reference is not None and (value < reference if name in S.LOWER else value > reference)
        ci = comparison[name]["session_cluster"]["confirmed_benefit"]
        criteria[name] = dict(seed_mean_improved=bool(gain), session_ci_supports_improvement=bool(ci), confirmed=bool(gain and ci))
    paired = {seed: {row["arm"]: row for row in summary["runs"] if row["seed"] == seed} for seed in SEEDS}
    kp_coverage = all(rows[PRIMARY]["two_d"]["keypoint_matched_frame_count_iou50"] >= rows[CONTROL]["two_d"]["keypoint_matched_frame_count_iou50"] for rows in paired.values())
    pose_coverage = all(rows[PRIMARY]["coverage"] >= rows[CONTROL]["coverage"] for rows in paired.values())
    kp = all(criteria[name]["confirmed"] for name in KEYPOINT_METRICS) and kp_coverage
    pose = all(criteria[name]["confirmed"] for name in POSE_METRICS) and pose_coverage
    overall = bool(kp and pose)
    headline = ("실제 DHT 공동학습이 같은 예산의 점 모델 대비 사전등록한 2D·6D 개선 기준을 충족했습니다."
                if overall else "실제 DHT 공동학습의 같은 예산 점 모델 대비 종합 우월성은 확인되지 않았습니다.")
    reasons = [f"{name}: 3 seed 평균 {'개선' if c['seed_mean_improved'] else '개선 미확인'}, 세션 95% CI {'지지' if c['session_ci_supports_improvement'] else '미지지'}" for name, c in criteria.items()]
    if not kp_coverage:
        reasons.append("하나 이상의 seed에서 점 평가 매칭 coverage가 같은 seed 대조군보다 낮습니다.")
    if not pose_coverage:
        reasons.append("하나 이상의 seed에서 MAIN 6D coverage가 같은 seed 대조군보다 낮습니다.")
    return dict(schema="pallet_dht_joint_verdict_v1", complete=True, PASS=True,
        PASS_semantics="Execution and artifact/statistic integrity; does not mean accuracy improvement.",
        primary_arm=PRIMARY, primary_reference=CONTROL, overall_accuracy_improved=overall,
        keypoint_gain_confirmed=bool(kp), pose_gain_confirmed=bool(pose),
        keypoint_coverage_preserved=kp_coverage, pose_coverage_preserved=pose_coverage,
        metric_criteria=criteria, headline_ko=headline, reasons=reasons, runs=summary["runs"],
        discord_lines_ko=[headline, f"2D 개선 기준={kp}; 6D 개선 기준={pose}. 동일 예산 3 seeds와 13세션 대응 bootstrap.",
                          "실사 319 positive와 2,689 negative 모두 새 모델로 추론한 재사용 DEV 결과입니다."],
        decision="Both 2D and all four MAIN6D metrics must improve in full-population seed mean and paired session CI, with matching/pose coverage preserved separately for every paired seed. Negative FP/AP are separate outcomes, not an implicit guarantee.")


def aggregate(run_dir):
    run_dir = Path(run_dir).resolve()
    require((run_dir / "PURPOSE.md").is_file(), "Purpose-declared output root required")
    protocol = read(run_dir / "TRAIN_PROTOCOL.json")
    baseline = dict(two_d=read(R0_2D)["metrics"]["box_and_keypoint_2d"],
        main_6d=read(POSE / "POSE_EVALUATION_R0.json")["paths"]["MAIN"]["ALL"],
        coverage=read(POSE / "POSE_EVALUATION_R0.json")["paths"]["MAIN"]["coverage"],
        role="historical R0 reference; primary reference is matched-budget point_only")
    r0_kp = S.load_keypoint_rows(R0_2D.with_name("R0_per_frame.csv"))
    r0_pose = {row["frame_id"]: row for row in read(POSE / "POSE_PER_FRAME_BY_ARM.json")["per_frame"]["R0"]}
    kp_stores, pose_stores = {"R0": [r0_kp] * 3}, {"R0": [r0_pose] * 3}
    runs, sources, traces, budgets = [], {}, {}, []
    for arm in ARMS:
        kp_stores[arm], pose_stores[arm] = [], []
        for seed in SEEDS:
            directory, result, trained, config = validate_cell(run_dir, arm, seed)
            label = f"{arm}_seed{seed}"
            traces.setdefault(seed, []).append(trained["batch_trace_sha256"])
            budgets.append({key: config[key] for key in ("epochs", "batch", "lr", "optimizer", "amp", "train_frames", "val_frames", "training_recipe")})
            kp = S.load_keypoint_rows(directory / "PAPER_2D_per_frame.csv")
            pp = {row["frame_id"]: row for row in read(directory / "POSE_PER_FRAME_BY_ARM.json")["per_frame"][label]}
            require(set(kp) == set(r0_kp), "All 319 positive frame rows must be retained")
            require(all(kp[key]["session_id"] == r0_kp[key]["session_id"] for key in kp), "Session assignments changed")
            for fields, expected in [(S.keypoint_summary(list(kp.values())), result["two_d"]), (S.pose_summary(list(pp.values())), result["main_6d"])]:
                for metric in METRICS:
                    if metric in fields:
                        if fields[metric] is None:
                            require(expected.get(metric) is None, "Missing metric was fabricated")
                        else:
                            S.close(fields[metric], expected[metric], label + "/" + metric)
            kp_stores[arm].append(kp)
            pose_stores[arm].append(pp)
            sessions = sorted({row["session_id"] for row in kp.values()})
            require(len(sessions) == 13, "Expected 13 canonical capture sessions")
            per_session = {sid: dict(two_d=S.keypoint_summary([r for r in kp.values() if r["session_id"] == sid]),
                                     main_6d=S.pose_summary([r for r in pp.values() if r["session_id"] == sid])) for sid in sessions}
            runs.append(dict(arm=arm, seed=seed, evaluation_arm=label, two_d=result["two_d"], main_6d=result["main_6d"],
                coverage=result["main_6d_coverage"], negative=result["negative"], per_session=per_session, training=trained))
            sources[label] = {name: bound(directory / name) for name in ("RESULTS.json", "COMPLETION.json", "EVALUATION_PROTOCOL.json", "PAPER_2D_per_frame.csv", "POSE_PER_FRAME_BY_ARM.json", "NEGATIVE_OUTCOMES.json", "LINE_EVIDENCE.json")}
    require(all(value == budgets[0] for value in budgets), "Arms do not share training data/optimizer/budget/augmentation recipe")
    require(all(len(set(values)) == 1 for values in traces.values()), "Paired seeds did not receive identical augmented batches across arms")
    families = {}
    for arm in ARMS:
        rows = [row for row in runs if row["arm"] == arm]
        families[arm] = dict(metrics={name: nullable_stats([r["two_d" if name in KEYPOINT_METRICS else "main_6d"].get(name) for r in rows]) for name in METRICS},
            pose_coverage=nullable_stats([r["coverage"] for r in rows]),
            keypoint_matched_frames=nullable_stats([r["two_d"]["keypoint_matched_frame_count_iou50"] for r in rows]),
            box_ap50=nullable_stats([r["two_d"]["box_ap50"] for r in rows]),
            box_ap50_95=nullable_stats([r["two_d"]["box_ap50_95"] for r in rows]),
            negative={threshold: nullable_stats([r["negative"]["by_threshold"][threshold]["fraction_frames_with_detection"] for r in rows]) for threshold in ("0.001", "0.25", "0.5", "0.85")},
            per_session={sid: dict(metrics={name: nullable_stats([r["per_session"][sid]["two_d" if name in KEYPOINT_METRICS else "main_6d"].get(name) for r in rows]) for name in METRICS}) for sid in rows[0]["per_session"]})
    comparisons = {}
    for left, right in [(PRIMARY, CONTROL), ("hough_features", CONTROL), (PRIMARY, "hough_features"), *[(arm, "R0") for arm in ARMS]]:
        comparisons[f"{left}_vs_{right}"] = dict(left=left, right=right, metrics={})
        for name in METRICS:
            comparisons[f"{left}_vs_{right}"]["metrics"][name] = compare(kp_stores if name in KEYPOINT_METRICS else pose_stores, left, right, name)
            print(f"10k paired bootstrap {left} vs {right}: {name}", flush=True)
    runtime = read(run_dir / "RUNTIME.json") if (run_dir / "RUNTIME.json").exists() else dict(complete=False, status="NOT_MEASURED")
    limits = ["실사 319 positive / 2,689 negative는 재사용 DEV입니다. 독립 final 데이터는 접근하지 않았습니다.",
        "주 비교는 같은 예산 point_only입니다. R0는 추가 학습 전의 과거 기준이며 DHT 효과와 추가 학습 효과를 분리해야 합니다.",
        "각 metric의 같은 관측 가능 프레임에서 3 seed 통계 차이를 먼저 평균하고, 같은 세션 bootstrap draw를 적용합니다. Coverage는 별도 전체 모집단 지표입니다.",
        "13개 세션의 조건부 95% 구간이며 반복 비교 전체를 교정하지 않습니다. 미래 학습 seed 모집단의 불확실성을 모두 반영하지 않습니다.",
        "6D GT는 기존 geometry reconstruction 계약입니다. 직접 측정한 물리 위치 정확도라고 해석하지 않습니다.",
        "라인 sigmoid/역투표 지도는 실제 학습된 선 증거입니다. Attention, 인과적 중요도, 보정된 정답 확률 또는 물리적 가시성을 뜻하지 않습니다.",
        "검출 AP와 negative FP는 실제 새로운 forward로 측정하며 보존을 가정하지 않습니다. 속도는 별도 실제 통제 측정이 있을 때만 비교합니다."]
    summary = dict(schema="pallet_dht_joint_summary_v1", complete=True, PASS=True, primary_arm=PRIMARY, primary_reference=CONTROL,
        population=dict(positive=319, negative=2689, sessions=13, role="DEV", held_out_final=False), baseline=baseline,
        runs=runs, arms=families, comparisons=comparisons, runtime=runtime, training_budget=budgets[0],
        identity=dict(actual_positive_forwards=319 * 9, actual_negative_forwards=2689 * 9, baseline_candidate_copying=False,
                      matched_augmented_batch_traces=True, trace_sha256_by_seed=traces),
        methods=dict(resamples=10000, primary_ci="session_cluster", seed_aggregation="mean of paired seed statistics, same frame/session draws", ddof=1,
                     paired_coverage="intersection of available frames across compared seeds; full-population coverage separately reported"),
        limitations=limits, sources=sources, source_bindings={"training_protocol": bound(run_dir / "TRAIN_PROTOCOL.json"), "statistic_primitives": bound(PRIMITIVES), "aggregate": bound(__file__)})
    verdict = make_verdict(summary)
    write(run_dir / "SUMMARY.json", summary)
    write(run_dir / "VERDICT.json", verdict)
    write(run_dir / "AGGREGATE_COMPLETE.json", dict(complete=True, PASS=True, n_completed_evaluations=9,
        primary_arm=PRIMARY, primary_reference=CONTROL, overall_accuracy_improved=verdict["overall_accuracy_improved"],
        input_sha256={str(run_dir / "TRAIN_PROTOCOL.json"): sha(run_dir / "TRAIN_PROTOCOL.json"), str(PRIMITIVES): sha(PRIMITIVES)},
        output_sha256={str(run_dir / name): sha(run_dir / name) for name in ("SUMMARY.json", "VERDICT.json")},
        source_sha256={str(Path(__file__)): sha(__file__)}))
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    answer = aggregate(args.run_dir)
    print(json.dumps({key: answer[key] for key in ("complete", "PASS", "overall_accuracy_improved", "headline_ko")}, ensure_ascii=False))
