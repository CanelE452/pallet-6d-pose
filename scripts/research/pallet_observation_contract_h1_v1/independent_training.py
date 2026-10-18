"""Independent no-B initialization/order/checkpoint/attempt accounting checks."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from scripts.research.pallet_boundary_action_repair_v1 import action_models as M
from scripts.research.pallet_risk_information_v1.common import read, rows, sha, write


def initial_hash(seed):
    torch.manual_seed(seed)
    model = M.make_model("MLP")
    payload = b"".join(name.encode() + value.detach().numpy().tobytes()
                       for name, value in model.state_dict().items())
    return hashlib.sha256(payload).hexdigest()


def run(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    start = time.monotonic()
    torch.set_num_threads(1)
    plan_hash = sha(run_dir / "EXPERIMENT_PLAN.json")
    manifest = read(run_dir / "NO_B_TRAINING_MANIFEST.json")
    attempts = list(rows(run_dir / "GLOBAL_FIT_ATTEMPTS.jsonl"))
    counts, failures = Counter(), []
    def check(key, condition, detail=None):
        counts[key] += 1
        if not bool(condition):
            failures.append(dict(check=key, detail=detail))
    check("three_original_scientific_seeds", {r["seed"] for r in manifest["runs"]} == {20261018, 20261019, 20261020} and len(manifest["runs"]) == 3)
    check("all_optimizer_attempts_counted_under9", len(attempts) <= 9 and manifest["actual_new_optimizer_attempts"] == len(attempts))
    check("attempt_indices_never_rewritten", [r["attempt"] for r in attempts] == list(range(1, len(attempts) + 1)))
    complete = set()
    for run in manifest["runs"]:
        seed = run["seed"]
        folder = run_dir / "training" / ("H1_NO_B_FEATURE_s" + str(seed))
        receipt = read(folder / "RUN_RECEIPT.json")
        history = read(folder / "HISTORY.json")
        best, last = [torch.load(folder / n, map_location="cpu") for n in ("best.pt", "last.pt")]
        for name, key in (("best.pt", "checkpoint_sha256"), ("last.pt", "last_checkpoint_sha256"), ("HISTORY.json", "history_sha256")):
            check("actual_saved_artifact_sha", sha(folder / name) == receipt[key], dict(seed=seed, name=name))
        check("initialization_recipe_exact", initial_hash(seed) == receipt["initial_state_sha256"] == best["initial_state_sha256"], seed)
        rng = np.random.default_rng(seed)
        for index, epoch in enumerate(history):
            expected = rng.permutation(768)
            check("original_epoch_order_exact", hashlib.sha256(expected.astype("<i8").tobytes()).hexdigest() == epoch["training_order_sha256"], dict(seed=seed, epoch=index + 1))
            check("original_epoch_denominators", epoch["epoch"] == index + 1 and epoch["train"]["frames"] == 768 and epoch["validation"]["frames"] == 235, dict(seed=seed, epoch=index + 1))
        losses = [e["validation"]["loss"] for e in history]
        first_min = min(range(len(losses)), key=lambda i: losses[i])
        check("first_strict_validation_BCE_minimum", best["epoch"] == receipt["best_epoch"] == first_min + 1 and receipt["best_validation_loss"] == losses[first_min], seed)
        check("last_not_silently_substituted_for_best", last["epoch"] == receipt["epochs"] == len(history), seed)
        current, patience, expected_stop = float("inf"), 0, None
        for i, loss in enumerate(losses):
            if loss < current:
                current, patience = loss, 0
            else:
                patience += 1
            if patience == 5:
                expected_stop = i + 1
                break
        check("same50epoch_patience5_rule", len(history) == (expected_stop if expected_stop is not None else 50), seed)
        check("actual_original24steps_per_epoch", receipt["optimizer_steps"] == 24 * len(history), seed)
        check("unchanged_fixed_plan", receipt["bindings"]["experiment_plan_sha256"] == plan_hash, seed)
        check("unchanged188architecture", best["input_dim"] == 188 and receipt["parameters"] == 16321 and best["architecture"] == "MLP" and best["objective"] == "H1", seed)
        complete.add(receipt["attempt"])
    failed = []
    for folder in sorted((run_dir / "failed_attempts").glob("*")):
        path = folder / "FAILED_TECHNICAL.json"
        if path.is_file():
            value = read(path)
            attempt = value["attempt"]["attempt"]
            failed.append(attempt)
            check("failed_optimizer_attempt_not_erased", value["optimizer_created"] is True and attempt not in complete, attempt)
    check("every_attempt_has_one_terminal_outcome", complete | set(failed) == {r["attempt"] for r in attempts} and complete.isdisjoint(failed) and len(failed) == len(set(failed)))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="INDEPENDENT_NO_B_TRAINING_ACCOUNTING_V1", status="PASS" if not failures else "FAIL",
                  checks=sum(counts.values()), checks_by_kind=dict(counts), failure_count=len(failures), failures=failures,
                  actual_optimizer_attempts=len(attempts), complete_scientific_fits=len(complete), failed_optimizer_attempts=len(failed),
                  failed_attempt_ids=failed, complete_attempt_ids=sorted(complete), global_cap=9,
                  historical_initial_weight_bytes_available=False,
                  initialization_scope="Exact reconstruction of frozen original source recipe; original archived initial bytes were not saved",
                  new_optimizer_creations_by_this_audit=0, real_reference_decodes=0,
                  source_sha256=sha(__file__), manifest_sha256=sha(run_dir / "NO_B_TRAINING_MANIFEST.json"),
                  wall_seconds=time.monotonic() - start)
    write(output / "VERIFICATION.json", result)
    print(result["status"], result["checks"], "checks", result["failure_count"], "failures", flush=True)
    if failures:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.run_dir, a.output)


if __name__ == "__main__":
    main()
