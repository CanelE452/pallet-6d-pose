"""Independent final contract audit for the bounded v2 experiment."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import immutable_json, read_json, sha256


REQUIRED = ("REPORT_KO.md", "PROTOCOL_LOCK.json", "FAILURE_MECHANISM_TESTS.json",
            "ORACLE_DECOMPOSITION.json", "INIT_PARITY.json", "SYNTH_PER_SEED.json",
            "SYNTH_PER_FRAME.csv", "RESULT_SUMMARY.json", "CODE_DIFF_SUMMARY.md")


def run(repo: Path, docs: Path, raw: Path, output: Path):
    failures = []; rows = []
    direct_config = read_json(repo / "scripts/research/pallet_symmetry_dht_local_v2/configs/direct.json")
    hough_config = read_json(repo / "scripts/research/pallet_symmetry_dht_local_v2/configs/hough.json")
    if direct_config["loss_weights"] != hough_config["loss_weights"]: failures.append("D/H loss terms differ")
    if {**direct_config["model"], "arm": None} != {**hough_config["model"], "arm": None}: failures.append("D/H non-arm model config differs")
    for seed in (1, 2, 3):
        starts = {arm: read_json(raw / f"heads/{arm}_seed{seed}/START.json") for arm in ("direct", "hough")}
        completions = {arm: read_json(raw / f"heads/{arm}_seed{seed}/COMPLETION.json") for arm in ("direct", "hough")}
        checks = {"exact_2000": all(x["steps"] == 2000 for x in completions.values()),
                  "complete": all(x["complete"] for x in completions.values()),
                  "cuda0": all(x["device_actual"] == "cuda:0" for x in starts.values()),
                  "same_manifest": starts["direct"]["manifest_sha256"] == starts["hough"]["manifest_sha256"],
                  "same_order": starts["direct"]["minibatch_order_sha256"] == starts["hough"]["minibatch_order_sha256"],
                  "same_common": starts["direct"]["common_parameter_sha256"] == starts["hough"]["common_parameter_sha256"],
                  "same_utility": starts["direct"]["initial_utility_head_sha256"] == starts["hough"]["initial_utility_head_sha256"],
                  "checkpoint_hashes": all(sha256(x["checkpoint"]) == x["checkpoint_sha256"] for x in completions.values())}
        if not all(checks.values()): failures.append(f"seed{seed} training contract")
        rows.append({"seed": seed, "checks": checks, "PASS": all(checks.values())})
    for name in ("point", "direct_seed1", "direct_seed2", "direct_seed3", "hough_seed1", "hough_seed2", "hough_seed3"):
        prediction = read_json(raw / f"predictions/{name}_synth_val.json")
        evaluation = read_json(raw / f"evaluations/{name}_synth_val.json")
        if prediction.get("GT_opened") is not False or len(prediction["records"]) != 512: failures.append(f"{name} GT-free prediction")
        if evaluation["n_frames"] != 512: failures.append(f"{name} evaluation population")
    summary = read_json(docs / "RESULT_SUMMARY.json")
    if summary["scientific_verdict"] != "DHT_LOCAL_TRACK_CLOSED" or summary["real_DEV_executed"] or summary["FINAL_opened"]:
        failures.append("termination contract")
    missing = [name for name in REQUIRED if not (docs / name).is_file()]
    if missing: failures.append(f"missing required docs: {missing}")
    branch = subprocess.run(("git", "branch", "--show-current"), cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    if branch != "main": failures.append(f"branch is {branch}")
    value = {"schema": "symdht_local_final_audit_v2", "PASS": not failures,
             "failures": failures, "branch": branch, "D_H_loss_terms_identical": direct_config["loss_weights"] == hough_config["loss_weights"],
             "required_documents": list(REQUIRED), "missing_documents": missing, "training_rows": rows,
             "prediction_count": 7, "evaluation_count": 7, "synth_val_frames_each": 512,
             "scientific_verdict": summary["scientific_verdict"], "real_DEV_executed": summary["real_DEV_executed"],
             "FINAL_opened": summary["FINAL_opened"], "C4_status": summary["C4_status"]}
    immutable_json(output, value); return value


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--docs", type=Path, required=True); parser.add_argument("--raw", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); value = run(args.repo.resolve(), args.docs.resolve(), args.raw.resolve(), args.output.resolve()); print("PASS" if value["PASS"] else "FAIL")
    if not value["PASS"]: raise SystemExit(1)


if __name__ == "__main__": main()
