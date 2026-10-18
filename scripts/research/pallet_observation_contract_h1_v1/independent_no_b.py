"""Reference-free independent mask closure and actual trained-head mutation audit."""
from __future__ import annotations
import argparse
from copy import deepcopy
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from scripts.research.pallet_boundary_action_repair_v1 import action_data as original
from scripts.research.pallet_boundary_action_repair_v1.action_models import make_model
from scripts.research.pallet_risk_information_v1.independent_checks import independent_actions
from scripts.research.pallet_risk_information_v1.common import read, rows, write, sha
from . import no_b

REMOVED = (2, 3, 4, 10, 13, 14, 15, 23, 25, 28, 29, 30)
MASK = tuple(i + offset for offset in (0, 31, 62, 93, 124, 155) for i in REMOVED)
KEPT = tuple(i for i in range(188) if i not in MASK)


def independent_input(frame, normalizer):
    entries = independent_actions(frame)
    local = []
    for entry in entries:
        raw = entry["features"]
        row = np.zeros(62, np.float32)
        for index in range(31):
            if index in REMOVED:
                continue
            row[index + 31] = raw[index + 31]
            if raw[index + 31] and normalizer["valid_train_counts"][index]:
                row[index] = (raw[index] - normalizer["mean"][index]) / normalizer["scale"][index]
        local.append(row)
    out = np.zeros((len(entries), 188), np.float32)
    if entries:
        matrix = np.asarray(local, dtype=np.float32)
        out[:, :62] = matrix
        # Float32 summation/division preserves the original pooling precision.
        out[:, 62:124] = np.sum(matrix, axis=0, dtype=np.float32) / np.float32(len(entries))
        out[:, 124:186] = np.max(matrix, axis=0)
        out[:, 186] = np.log1p(len(entries))
        out[:, 187] = [float("N0" in set(e["source_baseline_aliases"])) for e in entries]
    return out, entries


def run(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    start = time.monotonic()
    torch.set_num_threads(1)
    manifest = read(run_dir / "INPUT_MANIFEST.json")
    files = {r["key"]: r for r in manifest["existing_inputs"]}
    norm_path = Path(files["normalizer"]["path"])
    assert sha(norm_path) == files["normalizer"]["sha256"]
    normalizer = read(norm_path)
    models = []
    for seed in (20261018, 20261019, 20261020):
        p = run_dir / "training" / ("H1_NO_B_FEATURE_s" + str(seed)) / "best.pt"
        receipt = read(p.with_name("RUN_RECEIPT.json"))
        assert receipt["status"] == "COMPLETE" and sha(p) == receipt["checkpoint_sha256"]
        cp = torch.load(p, map_location="cpu")
        model = make_model("MLP")
        model.load_state_dict(cp["model_state_dict"])
        models.append(model.eval())
    counts, failures = Counter(), []
    def check(key, value, detail=None):
        counts[key] += 1
        if not bool(value):
            failures.append(dict(check=key, detail=detail))
    actions = 0
    for key, expected_n in (("synthetic_features", 1003), ("real_features", 238)):
        entry = files[key]
        assert sha(entry["path"]) == entry["sha256"]
        n = 0
        for frame in rows(entry["path"]):
            n += 1
            x, entries = no_b.inputs(frame, normalizer)
            independent, independent_entries = independent_input(frame, normalizer)
            check("independent188_input_bitwise", np.array_equal(x, independent), frame["id"])
            check("independent_action_order", [e["candidate_id"] for e in entries] ==
                  [e["candidate_id"] for e in independent_entries], frame["id"])
            check("independent_actual_poses", [e["actual_pose"] for e in entries] ==
                  [e["actual_pose"] for e in independent_entries], frame["id"])
            old, old_entries = original.inputs(frame, normalizer, "H1")
            check("all116_retained_slots_bitwise", np.array_equal(old[:, KEPT], x[:, KEPT]), frame["id"])
            check("all72_removed_slots_zero", bool(np.all(x[:, MASK] == 0)), frame["id"])
            check("original_action_count_order", [e["candidate_id"] for e in entries] ==
                  [e["candidate_id"] for e in old_entries], frame["id"])
            altered = deepcopy(frame)
            for ci, candidate in enumerate(altered["candidates"]):
                for j in REMOVED:
                    candidate["features"][j] = float((ci + 1) * (j + 1) + 10000)
                    candidate["features"][31 + j] = 1. - candidate["features"][31 + j]
            changed, changed_entries = no_b.inputs(altered, normalizer)
            check("removed_value_validity_mutation_input_invariance", np.array_equal(x, changed), frame["id"])
            check("mutation_action_pose_order_invariance", [e["actual_pose"] for e in entries] ==
                  [e["actual_pose"] for e in changed_entries], frame["id"])
            with torch.inference_mode():
                for i, model in enumerate(models):
                    before = model(torch.from_numpy(x)).numpy()
                    after = model(torch.from_numpy(changed)).numpy()
                    check("actual_trained_head_logit_invariance/" + str(i), np.array_equal(before, after), frame["id"])
            actions += len(entries)
        check("full_frame_denominator/" + key, n == expected_n, n)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="INDEPENDENT_NO_B_CLOSURE_V1", status="PASS" if not failures else "FAIL",
                  checks=sum(counts.values()), checks_by_kind=dict(counts), failures=failures,
                  failure_count=len(failures), full_frames=1241, actual_actions=actions,
                  independent_builder_imports_production_no_b_input=False,
                  source_sha256=sha(__file__), producer_source_sha256=sha(Path(no_b.__file__)),
                  reference_content_decodes=0, newly_trained_head_count=3,
                  physical_ownership_certification=False, complete_B_free_producer=False,
                  wall_seconds=time.monotonic() - start)
    write(output / "VERIFICATION.json", result)
    print(result["status"], result["checks"], "checks", len(failures), "failures", flush=True)
    if failures:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    run(a.run_dir, a.output)


if __name__ == "__main__":
    main()
