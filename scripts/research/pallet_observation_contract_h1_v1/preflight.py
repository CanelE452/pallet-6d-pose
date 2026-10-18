"""Byte-bind unchanged inputs and prior complete replay, without scoring RGB/GT."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import time
from scripts.research.pallet_risk_information_v1.common import read, sha, write


def run(run_dir, risk_run, repo):
    started = time.monotonic()
    run_dir, risk_run, repo = map(Path, (run_dir, risk_run, repo))
    previous = risk_run / "independent_p0_v1"
    inventory = read(previous / "INPUTS_PRIVATE.json")
    proof = read(previous / "VERIFICATION.json")
    assert proof["status"] == "PASS" and proof["failure_count"] == 0
    assert proof["private_input_manifest_sha256"] == sha(previous / "INPUTS_PRIVATE.json")
    assert proof["source_sha256"] == sha(repo / "scripts/research/pallet_risk_information_v1/independent_checks.py")
    checks = []
    for key, entry in inventory["inputs"].items():
        path = Path(entry["path"])
        digest = sha(path)  # References are bytes only, never JSON-decoded here.
        assert digest == entry["sha256"], key
        checks.append(dict(key=key, **entry, current_byte_check=True))
    pm = read(risk_run / "PREDICTION_MANIFEST.json")
    assert sha(risk_run / "PREDICTIONS.jsonl.gz") == pm["artifacts"]["PREDICTIONS.jsonl.gz"]
    extra = []
    for seed in (20261018, 20261019, 20261020):
        name = "R_CONT_F0_s" + str(seed)
        path = risk_run / "training" / name / "best.pt"
        receipt = read(path.with_name("RUN_RECEIPT.json"))
        assert receipt["status"] == "COMPLETE" and receipt["checkpoint_sha256"] == sha(path)
        binding = [x for x in pm["model_and_input_bindings"] if x["path"] == str(path)]
        assert len(binding) == 1 and binding[0]["sha256"] == sha(path)
        extra.append(dict(key="checkpoint/" + name, path=str(path), sha256=sha(path),
                          original_receipt_sha256=sha(path.with_name("RUN_RECEIPT.json"))))
    replay = dict(status="REUSED_COMPLETE_REPLAY_WITH_ALL_BOUND_BYTES_RECHECKED",
                 proof_sha256=sha(previous / "VERIFICATION.json"),
                 original_checks=proof["checks"],
                 checkpoint_replayed_rows=proof["checkpoint_replayed_rows"],
                 maximum_logit_absolute_difference=proof["maximum_logit_absolute_difference"],
                 fresh_H1_replay_here=False,
                 R_CONT_prediction_manifest_sha256=sha(risk_run / "PREDICTION_MANIFEST.json"),
                 archived_prediction_sha256=sha(risk_run / "PREDICTIONS.jsonl.gz"))
    paths = [
        "_docs/experiments/pallet_risk_information_v1/DECISION_KO.md",
        "_docs/experiments/pallet_risk_information_v1/B_REFERENCE_RESULTS.md",
        "_docs/experiments/pallet_risk_information_v1/MODEL_AND_SUPERVISION.md",
        "_docs/experiments/pallet_risk_information_v1/COMPARISON_ALL_ARMS.csv",
        "_docs/experiments/pallet_risk_information_v1/SELECTION_DIAGNOSIS.csv",
        "_docs/experiments/pallet_boundary_action_repair_v1/METHOD_L.md",
        "scripts/research/pallet_boundary_action_repair_v1/action_data.py",
        "scripts/research/pallet_boundary_action_repair_v1/action_models.py",
        "scripts/research/pallet_boundary_action_repair_v1/train_actions.py",
        "scripts/research/pallet_statewise_end_to_end_recovery_v1/solver.py",
        "scripts/research/pallet_statewise_end_to_end_recovery_v1/boundary_model.py",
        "scripts/research/pallet_risk_information_v1/risk_models.py",
        "scripts/research/pallet_risk_information_v1/supervision.py",
        "scripts/research/pallet_risk_information_v1/train.py",
    ]
    sources = {p: sha(repo / p) for p in paths}
    bundle = run_dir / "instruction_bundle"
    package = read(bundle / "PACKAGE_MANIFEST.json")
    for entry in package["files"]:
        assert sha(bundle / entry["name"]) == entry["sha256"]
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    value = dict(schema="OBSERVATION_H1_INPUT_BINDING_V1", status="PASS",
                 base_main=base, existing_inputs=checks, additional_checkpoints=extra,
                 original_source_sha256=sources, prior_replay=replay,
                 instruction_sha256=sha(bundle / "CLI_OBSERVATION_CONTRACT_AND_H1_GUARD_20261018_V1.md"),
                 reference_content_decodes=0, new_fits=0, new_pose_solves=0,
                 wall_seconds=time.monotonic() - started)
    write(run_dir / "INPUT_MANIFEST.json", value)
    print("P0_PASS", len(checks), "unchanged inputs; 3 R_CONT checkpoints; reference decode0", flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ("run-dir", "risk-run", "repo"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.run_dir, a.risk_run, a.repo)


if __name__ == "__main__":
    main()
