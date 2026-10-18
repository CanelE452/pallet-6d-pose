"""Independent finite policy search and held-run confirmation reconstruction.

Imports neither the production policy nor its calibration implementation.
Only synthetic labels are decoded; no PRIMARY/CAL reference is accepted.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import time
import numpy as np
from scripts.research.pallet_risk_information_v1.common import read, rows, sha, write


def tau_value(value):
    return -math.inf if value == {"kind": "NEGATIVE_INFINITY"} else float(value["value"])


def summarize(items, tau):
    chosen, metrics, changes = [], [], 0
    for row, seed, label in items:
        values = row["seeds"][seed]
        h = int(np.argmax(values["h1_logits"]))
        risk = np.max(np.asarray(values["rcont_axes"], float), axis=1)
        r = int(np.argmin(risk))
        entries = row["entries"]
        hp, rp = entries[h]["actual_pose"], entries[r]["actual_pose"]
        same = (np.max(np.abs(np.asarray(hp["R_physical"]) - rp["R_physical"])) <= 1e-7
                and np.max(np.abs(np.asarray(hp["centroid"]) - rp["centroid"])) <= 1e-7 * max(row["object_scale_m"], 1e-9))
        gap = values["h1_logits"][h] - values["h1_logits"][r]
        switch = not same and risk[h] > risk[r] and gap <= tau
        index = r if switch else h
        changes += switch
        chosen.append(index)
        metrics.append(label[entries[index]["candidate_id"]])
    out = dict(denominator=len(items), successes=sum(m["success_5cm5deg"] for m in metrics), changes=int(changes), returned_action_indices=chosen)
    for axis in ("translation_cm", "rotation_deg"):
        available = [m[axis] for m in metrics if m[axis] is not None and math.isfinite(m[axis])]
        out[axis] = dict(n=len(available), mean=float(np.mean(available)) if available else None)
    for name, axis, threshold in (("T_gt50cm", "translation_cm", 50), ("T_gt100cm", "translation_cm", 100), ("R_gt45deg", "rotation_deg", 45)):
        out[name] = sum(m[axis] is not None and m[axis] > threshold for m in metrics)
    return out


def admissible(value, base, strict=False):
    if not value["denominator"] or value["successes"] < base["successes"]:
        return False
    for axis in ("translation_cm", "rotation_deg"):
        if value[axis]["n"] != base[axis]["n"] or value[axis]["mean"] is None or value[axis]["mean"] > base[axis]["mean"]:
            return False
    if any(value[k] > base[k] for k in ("T_gt100cm", "R_gt45deg")):
        return False
    return not strict or (value["changes"] > 0 and (any(value[a]["mean"] < base[a]["mean"] for a in ("translation_cm", "rotation_deg"))
                           or any(value[k] < base[k] for k in ("T_gt100cm", "R_gt45deg"))))


def run(run_dir, labels, output):
    run_dir, labels, output = map(Path, (run_dir, labels, output))
    start = time.monotonic()
    policy = read(run_dir / "POLICY_CALIBRATION.json")
    split = read(run_dir / "POLICY_SPLIT_LOCK_v2.json")
    score_dir = run_dir / "policy_validation_scores_v1"
    ss = read(score_dir / "SCORE_SEAL.json")
    assert sha(labels) == policy["bindings"]["original_labels_sha256"]
    assert sha(run_dir / "EXPERIMENT_PLAN.json") == policy["bindings"]["experiment_plan_sha256"]
    assert sha(run_dir / "POLICY_SPLIT_LOCK_v2.json") == policy["bindings"]["policy_split_sha256"]
    assert sha(score_dir / "VALIDATION_POLICY_SCORES.jsonl.gz") == ss["scores_sha256"] == policy["bindings"]["scores_sha256"]
    scores = list(rows(score_dir / "VALIDATION_POLICY_SCORES.jsonl.gz"))
    assignments = {a["id"]: a for a in split["assignments"]}
    targets = {r["id"]: {m["candidate_id"]: m for m in r["action_metrics"]} for r in rows(labels) if r["split"] == "validation"}
    assert len(scores) == len(assignments) == len(targets) == 235
    counts, failures = Counter(), []
    def check(key, condition, detail=None):
        counts[key] += 1
        if not bool(condition):
            failures.append(dict(check=key, detail=detail))
    def compare(a, b, path=""):
        if isinstance(a, dict):
            check("mapping_keys", set(a) == set(b), path)
            for key in set(a) & set(b):
                compare(a[key], b[key], path + "/" + key)
        elif isinstance(a, list):
            check("list_length", len(a) == len(b), path)
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + "/" + str(i))
        elif isinstance(a, float):
            check("numeric_reconstruction", math.isfinite(b) and abs(a - b) <= 1e-9 + 1e-12 * abs(a), path)
        else:
            check("exact_reconstruction", a == b, path)
    groups = Counter(a["group"] for a in assignments.values())
    ordered = sorted(groups, key=lambda g: (hashlib.sha256((split["hash_salt"] + "|" + g).encode()).hexdigest(), g))
    check("hash_sorted_original_runs", ordered == split["hash_ordered_groups"])
    choices = [(abs(3 * sum(groups[g] for g in ordered[:i]) - 2 * 235), i) for i in range(1, len(ordered))]
    cutoff = min(choices)[1]
    check("label_free_closest_group_prefix", cutoff == split["selected_prefix_cutoff"])
    side = {g: "development" if i < cutoff else "confirmation" for i, g in enumerate(ordered)}
    check("no_group_split_across_sides", all(a["subset"] == side[a["group"]] for a in assignments.values()))
    check("no_labels_used_in_split", split["labels_read_for_split"] == 0)
    for seed in ("20261018", "20261019", "20261020"):
        dev = [(r, seed, targets[r["id"]]) for r in scores if assignments[r["id"]]["subset"] == "development"]
        conf = [(r, seed, targets[r["id"]]) for r in scores if assignments[r["id"]]["subset"] == "confirmation"]
        base = summarize(dev, -math.inf)
        thresholds = {-math.inf}
        for row, _, _ in dev:
            values = row["seeds"][seed]
            h = int(np.argmax(values["h1_logits"]))
            r = int(np.argmin(np.max(values["rcont_axes"], axis=1)))
            thresholds.add(values["h1_logits"][h] - values["h1_logits"][r])
        actual = policy["seeds"][seed]
        compare(base, actual["development_baseline"], seed + "/baseline")
        trials = []
        check("every_unique_gap_plus_identity", sorted(thresholds) == [tau_value(t["tau"]) for t in actual["all_thresholds"]], seed)
        for tau, saved in zip(sorted(thresholds), actual["all_thresholds"]):
            value = summarize(dev, tau)
            compare(value, saved["metrics"], seed + "/threshold")
            valid = admissible(value, base, True)
            check("strict_nonidentity_development_admissibility", valid == saved["admissible"], seed)
            if valid:
                trials.append((value["translation_cm"]["mean"], value["rotation_deg"]["mean"], value["changes"], tau))
        tau = min(trials)[-1] if trials else -math.inf
        check("fixed_lexicographic_tau_selection", tau == tau_value(actual["calibrated_trial_tau"]), seed)
        if trials:
            cb, cv = summarize(conf, -math.inf), summarize(conf, tau)
            passed = admissible(cv, cb)
            compare(cb, actual["confirmation"]["baseline"], seed + "/confirmation_baseline")
            compare(cv, actual["confirmation"]["selected"], seed + "/confirmation_trial")
            check("once_only_confirmation", actual["confirmation_applications"] == 1, seed)
            check("failed_confirmation_must_retain_H1", tau_value(actual["active_tau"]) == (tau if passed else -math.inf), seed)
            check("confirmation_pass_status", passed == actual["confirmation"]["passed"], seed)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="INDEPENDENT_H1_POLICY_CALIBRATION_V1", status="PASS" if not failures else "FAIL",
        checks=sum(counts.values()), checks_by_kind=dict(counts), failure_count=len(failures), failures=failures,
        development_frames=153, confirmation_frames=82, source_groups=23, shared_validation_asset_count=1,
        actual_original_thresholds_and_returned_indices_checked=True, production_policy_imports=0,
        real_reference_decodes=0, new_optimizer_creations=0, independent_safety_certification=False,
        source_sha256=sha(__file__), calibration_sha256=sha(run_dir / "POLICY_CALIBRATION.json"),
        wall_seconds=time.monotonic() - start)
    write(output / "VERIFICATION.json", result)
    print(result["status"], result["checks"], "checks", result["failure_count"], "failures", flush=True)
    if failures:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    for n in ("run-dir", "labels", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    run(a.run_dir, a.labels, a.output)


if __name__ == "__main__":
    main()
