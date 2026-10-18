"""Reference-only evaluation of all sealed frozen-bank outputs.

This module is never imported by predictors or training. Production metrics and
cluster bootstrap deliberately retain the previous campaign's exact arithmetic;
an independent audit must separately verify these reported numbers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
from pathlib import Path
import time

import numpy as np

from scripts.research.pallet_boundary_action_repair_v1.action_data import actions, same_pose
from scripts.research.pallet_statewise_end_to_end_recovery_v1.evaluator import _safe_metric, _status
from scripts.research.pallet_accuracy_closeout_20261017_v1.evaluation import aggregate, paired
from .common import read, rows, save_rows, sha, write, csv_write, measured
from .registry import BASELINES, COMPARATORS, POPULATIONS, EXPECTED_N, arms, contrasts, protocol


def source(pose, feature):
    if not pose.get("available"):
        return "NO_POSE"
    scale = feature.get("object_scale_m", 1.)
    if same_pose(pose, feature.get("fallback"), scale):
        return "N0"
    if same_pose(pose, feature.get("e0_regenerated"), scale):
        return "E0"
    return "OTHER"


def continuous(records, before, after, *, only_both_failure=False):
    selected = [r for r in records if not only_both_failure or
                (not r["metrics"][before]["success"] and
                 not r["metrics"][after]["success"])]
    result = {"denominator": len(selected)}
    for axis in ("translation_cm", "rotation_deg"):
        values = [r["metrics"][after][axis] - r["metrics"][before][axis]
                  for r in selected if r["metrics"][after][axis] is not None
                  and r["metrics"][before][axis] is not None]
        result[axis] = dict(common_available_n=len(values),
            sum=float(sum(values)), mean=float(np.mean(values)) if values else None,
            improved_n=sum(x < 0 for x in values), worsened_n=sum(x > 0 for x in values),
            equal_n=sum(x == 0 for x in values),
            minimum=float(min(values)) if values else None,
            maximum=float(max(values)) if values else None)
    return result


def verify_frozen(run_dir):
    manifest = read(run_dir / "PREDICTION_MANIFEST.json")
    if manifest["status"] != "ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING":
        raise ValueError("All registered outputs must be fixed before reference access")
    if manifest["experiment_plan_sha256"] != sha(run_dir / "EXPERIMENT_PLAN.json"):
        raise ValueError("Preregistered plan changed")
    if manifest.get("real_reference_inputs") is not False:
        raise ValueError("Prediction reference contract missing")
    path = run_dir / "PREDICTIONS.jsonl.gz"
    if manifest["artifacts"][path.name] != sha(path):
        raise ValueError("Sealed predictions changed")
    values = list(rows(path))
    expected = list(arms(manifest["visual_executed"]))
    if manifest["arms"] != expected or len(values) != 238:
        raise ValueError("Registered arm or full denominator mismatch")
    if [r["id"] for r in values] != manifest["ordered_ids"]:
        raise ValueError("Original row order changed")
    if len({r["id"] for r in values}) != len(values):
        raise ValueError("Duplicate real identity")
    if any(set(r["predictions"]) != set(expected) for r in values):
        raise ValueError("Missing registered output")
    for binding in manifest.get("model_and_input_bindings", []):
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Bound model/input bytes changed: " + binding["path"])
    return values, manifest


def run(run_dir, reference_dir, features, output):
    run_dir, reference_dir, output = map(Path, (run_dir, reference_dir, output))
    started = time.monotonic()
    ledger, manifest = verify_frozen(run_dir)
    feature_rows = list(rows(features))
    if manifest["real_features_sha256"] != sha(features):
        raise ValueError("Common real feature bank changed")
    by_feature = {r["id"]: r for r in feature_rows}
    if set(by_feature) != {r["id"] for r in ledger}:
        raise ValueError("Feature/reference-free metadata join changed")
    refpath = reference_dir / "FIXED_REFERENCES.json"
    refseal = read(reference_dir / "SCORE_SEAL.json")
    if sha(refpath) != refseal["artifacts"][refpath.name]:
        raise ValueError("Original reference seal does not match")
    output.mkdir(parents=True, exist_ok=False)
    # No real reference has been decoded above this line.
    references = read(refpath)["references"]
    write(output / "REFERENCE_OPEN_RECEIPT.json", dict(epoch=time.time(),
        prediction_manifest_sha256=sha(run_dir / "PREDICTION_MANIFEST.json"),
        reference_sha256=sha(refpath), source_sha256=sha(__file__)))
    registered = manifest["arms"]
    scored, coverage, all_failure_detail = [], [], []
    for row in ledger:
        feature = by_feature[row["id"]]
        ref = references[row["partition"]][row["id"]]
        for key in ("partition", "session", "cluster_id"):
            if row[key] != feature[key]:
                raise ValueError("Identity join mismatch: " + key)
        full_actions, pair_actions = actions(feature), actions(feature, pair=True)
        full_metrics = [_safe_metric(c["actual_pose"], ref) for c in full_actions]
        pair_metrics = [_safe_metric(c["actual_pose"], ref) for c in pair_actions]
        full_has = any(m["success"] for m in full_metrics)
        pair_has = any(m["success"] for m in pair_metrics)
        value = {key: row[key] for key in ("id", "partition", "session", "cluster_id")}
        value.update(poses={a: row["predictions"][a]["actual_pose"] for a in registered},
            metrics={a: _safe_metric(row["predictions"][a]["actual_pose"], ref)
                     for a in registered},
            status={a: _status(row["predictions"][a]) for a in registered},
            action_sources={a: source(row["predictions"][a]["actual_pose"], feature)
                            for a in registered},
            full_bank_has_success=full_has, pair_bank_has_success=pair_has,
            full_action_count=len(full_actions), pair_action_count=len(pair_actions))
        # Every new output must be one unchanged actual action, without a new pose.
        for arm in registered:
            if arm in BASELINES:
                continue
            pose = value["poses"][arm]
            if pose.get("available") and not any(
                    pose == c["actual_pose"] for c in full_actions):
                raise ValueError("Returned pose was not an exact original action: " + arm)
            if not full_has and value["metrics"][arm]["success"]:
                raise ValueError("Unchanged all-failure bank cannot yield new success")
        scored.append(value)
        coverage.append(dict(id=row["id"], partition=row["partition"],
            full_action_count=len(full_actions), pair_action_count=len(pair_actions),
            full_has_success=full_has, pair_has_success=pair_has,
            full_success_action_ids=[c["candidate_id"] for c, m in zip(full_actions, full_metrics)
                                     if m["success"]],
            pair_success_action_ids=[c["candidate_id"] for c, m in zip(pair_actions, pair_metrics)
                                     if m["success"]]))
    primary = [r for r in scored if r["partition"] != "CAL"]
    if sum(r["full_bank_has_success"] for r in primary) != 151:
        raise ValueError("Fixed full bank/reference no longer has the original151 successes")
    if sum(r["pair_bank_has_success"] for r in primary) != 133:
        raise ValueError("Fixed pair bank/reference no longer has the original133 successes")
    subsets = {p: [r for r in scored if r["partition"] in parts]
               for p, parts in POPULATIONS.items()}
    if {p: len(v) for p, v in subsets.items()} != EXPECTED_N:
        raise ValueError("Full reporting populations changed")
    summaries = {p: aggregate(subset, registered) for p, subset in subsets.items()}
    for methods in summaries.values():
        for value in methods.values():
            for axis in ("translation_cm", "rotation_deg"):
                sd = value[axis]["sample_std"]
                value[axis]["variance_ddof1"] = sd ** 2 if sd is not None else None
    pairs, transitions, source_counts, source_transitions = {}, [], [], []
    all_failure_summary, three_way, selection_diagnosis = [], [], []
    for population, subset in subsets.items():
        pairs[population] = {}
        for before, after in contrasts(manifest["visual_executed"]):
            value = paired(subset, before, after)
            value["both_failure_continuous"] = continuous(
                subset, before, after, only_both_failure=True)
            pairs[population][before + "__TO__" + after] = value
            for r in subset:
                mb, ma = r["metrics"][before], r["metrics"][after]
                group = ("BOTH_SUCCESS" if mb["success"] and ma["success"] else
                         "RECOVERY" if ma["success"] else "DAMAGE" if mb["success"]
                         else "BOTH_FAILURE")
                transitions.append(dict(population=population, before=before, after=after,
                    transition=group, id=r["id"],
                    before_translation_cm=mb["translation_cm"],
                    after_translation_cm=ma["translation_cm"],
                    before_rotation_deg=mb["rotation_deg"],
                    after_rotation_deg=ma["rotation_deg"]))
        for arm in registered:
            counts = Counter(r["action_sources"][arm] for r in subset)
            for role in ("N0", "E0", "OTHER", "NO_POSE"):
                source_counts.append(dict(population=population, arm=arm,
                    source=role, n=counts[role], denominator=len(subset)))
            for before in COMPARATORS:
                for group in ("BOTH_SUCCESS", "RECOVERY", "DAMAGE", "BOTH_FAILURE"):
                    chosen = []
                    for r in subset:
                        b, a = r["metrics"][before]["success"], r["metrics"][arm]["success"]
                        transition = "BOTH_SUCCESS" if b and a else "RECOVERY" if a else "DAMAGE" if b else "BOTH_FAILURE"
                        if transition == group:
                            chosen.append(r)
                    cc = Counter(r["action_sources"][arm] for r in chosen)
                    for role in ("N0", "E0", "OTHER", "NO_POSE"):
                        source_transitions.append(dict(population=population, arm=arm,
                            before=before, transition=group, source=role, n=cc[role],
                            transition_denominator=len(chosen)))
            for kind, flag in (("FULL_FIXED_ACTION_BANK", "full_bank_has_success"),
                               ("PAIR_FIXED_ACTION_BANK", "pair_bank_has_success")):
                chosen = [r for r in subset if not r[flag]]
                for before in COMPARATORS:
                    stats = continuous(chosen, before, arm)
                    result = dict(population=population, fixed_set=kind, before=before,
                        arm=arm, denominator=len(chosen))
                    for axis in ("translation_cm", "rotation_deg"):
                        result.update({axis + "/" + k: v for k, v in stats[axis].items()})
                    all_failure_summary.append(result)
                    for r in chosen:
                        mb, ma = r["metrics"][before], r["metrics"][arm]
                        all_failure_detail.append(dict(population=population,
                            fixed_set=kind, before=before, arm=arm, id=r["id"],
                            source=r["action_sources"][arm],
                            before_available=mb["pose_available"], after_available=ma["pose_available"],
                            before_translation_cm=mb["translation_cm"],
                            after_translation_cm=ma["translation_cm"],
                            delta_translation_cm=(ma["translation_cm"] - mb["translation_cm"])
                                if ma["translation_cm"] is not None and mb["translation_cm"] is not None else None,
                            before_rotation_deg=mb["rotation_deg"], after_rotation_deg=ma["rotation_deg"],
                            delta_rotation_deg=(ma["rotation_deg"] - mb["rotation_deg"])
                                if ma["rotation_deg"] is not None and mb["rotation_deg"] is not None else None,
                            T_gt50cm=ma["translation_cm"] is not None and ma["translation_cm"] > 50,
                            T_gt100cm=ma["translation_cm"] is not None and ma["translation_cm"] > 100,
                            R_gt45deg=ma["rotation_deg"] is not None and ma["rotation_deg"] > 45))
            for n0, e0, new in itertools.product((False, True), repeat=3):
                ids = [r["id"] for r in subset if
                    (r["metrics"]["N0"]["success"], r["metrics"]["E0"]["success"],
                     r["metrics"][arm]["success"]) == (n0, e0, new)]
                three_way.append(dict(population=population, arm=arm, N0_success=n0,
                    E0_success=e0, new_success=new, n=len(ids), denominator=len(subset),
                    ids="|".join(ids)))
            is_pair = "_PAIR_" in arm
            bank_flag = "pair_bank_has_success" if is_pair else "full_bank_has_success"
            selection_diagnosis.append(dict(population=population, arm=arm,
                bank="PAIR" if is_pair else "FULL", denominator=len(subset),
                successful_action_exists_n=sum(r[bank_flag] for r in subset),
                selected_success_n=sum(r["metrics"][arm]["success"] for r in subset),
                selection_loss_n=sum(r[bank_flag] and not r["metrics"][arm]["success"] for r in subset),
                all_actions_failure_n=sum(not r[bank_flag] for r in subset)))
        print("EVALUATED_POPULATION", population, len(subset), flush=True)
    save_rows(output / "SCORED_ROWS.jsonl.gz", scored)
    save_rows(output / "CANDIDATE_COVERAGE.jsonl.gz", coverage)
    table = []
    for population, methods in summaries.items():
        for arm, value in methods.items():
            entry = dict(population=population, arm=arm, denominator=value["denominator"],
                successes=value["successes"], success_rate=value["success_rate"], no_pose=value["no_pose"],
                T_gt50cm=value["T_gt50cm"], T_gt100cm=value["T_gt100cm"], R_gt45deg=value["R_gt45deg"])
            for axis in ("translation_cm", "rotation_deg"):
                entry.update({axis + "/" + key: val for key, val in value[axis].items()
                              if not isinstance(val, (dict, list))})
            table.append(entry)
    csv_write(output / "COMPARISON_ALL_ARMS.csv", table)
    csv_write(output / "ROW_METRICS.csv.gz", (dict(
        id=r["id"], partition=r["partition"], session=r["session"], cluster_id=r["cluster_id"], arm=arm,
        **r["metrics"][arm], source=r["action_sources"][arm],
        full_bank_has_success=r["full_bank_has_success"], pair_bank_has_success=r["pair_bank_has_success"],
        fallback=r["status"][arm].get("fallback"), status=r["status"][arm].get("status"))
        for r in scored for arm in registered))
    csv_write(output / "PAIRED_TRANSITIONS.csv.gz", transitions)
    csv_write(output / "ACTION_SOURCE_COUNTS.csv", source_counts)
    csv_write(output / "GAIN_DAMAGE_ACTUAL_SOURCE.csv", source_transitions)
    csv_write(output / "ALL_FAILURE_RISK.csv", all_failure_summary)
    csv_write(output / "ALL_FAILURE_RISK_ROWS.csv.gz", all_failure_detail)
    csv_write(output / "THREE_WAY_SUCCESS.csv", three_way)
    csv_write(output / "SELECTION_DIAGNOSIS.csv", selection_diagnosis)
    write(output / "EVALUATION.json", dict(schema="FROZEN_BANK_RISK_EVALUATION_V1",
        arms=registered, protocol=protocol(), populations=summaries, paired=pairs,
        original_full_bank_primary_success_possible=151,
        original_full_bank_primary_all_failure=41,
        original_pair_bank_primary_success_possible=133,
        original_pair_bank_primary_all_failure=59, independent_real_test_n=0))
    write(output / "EVALUATION_SEAL.json", dict(status="COMPLETE",
        prediction_manifest_sha256=sha(run_dir / "PREDICTION_MANIFEST.json"),
        reference_sha256=sha(refpath), evaluation_source_sha256=sha(__file__),
        registry_source_sha256=sha(Path(__file__).with_name("registry.py")),
        artifacts={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        wall_seconds=time.monotonic() - started, all_failures_retained=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with measured(args.run_dir / "RUN_LOG.jsonl", "POSTSEAL_REAL_EVALUATION"):
        run(args.run_dir, args.reference_dir, args.features, args.output)


if __name__ == "__main__":
    main()
