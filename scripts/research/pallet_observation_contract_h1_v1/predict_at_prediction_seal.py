"""Fresh-process reference-denied selection from the unchanged real bank.

Archived H1/R_CONT scores are reused only with exact source, checkpoint and
physical action-order bindings. No pose solver or image producer is imported.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from scripts.research.pallet_boundary_action_repair_v1 import action_data as D
from scripts.research.pallet_boundary_action_repair_v1 import action_models as M
from scripts.research.pallet_risk_information_v1.common import read, rows, save_rows, sha, write
from scripts.research.pallet_risk_information_v1.prediction_guard import PredictionGuard
from . import no_b
from .policy import select_guarded, decode_tau
from .registry import BASELINES, SEEDS, AUGMENTATION_GROUPS, arms


def bind_scores(frame, entries, archived):
    """Bind numerical scores to the exact already-existing physical order."""
    ids = [c["candidate_id"] for c in entries]
    hs, rs = {}, {}
    for seed in SEEDS:
        h = archived["D1_H1_MLP_s" + str(seed)]
        r = archived["R_CONT_F0_s" + str(seed)]
        if [c["candidate_id"] for c in h["actions"]] != ids:
            raise ValueError("Original H1 cached action order differs")
        if [c["candidate_id"] for c in r["actions"]] != ids:
            raise ValueError("Original R_CONT cached action order differs")
        logits = np.asarray(h["logits"], dtype=float)
        axes = np.asarray(r["model_outputs"], dtype=float)
        if entries:
            if logits.shape != (len(entries),) or axes.shape != (len(entries), 2):
                raise ValueError("Archived per-action score shape mismatch")
            hi, ri = int(np.argmax(logits)), int(np.argmin(np.max(axes, axis=1)))
            if h["actual_pose"] != entries[hi]["actual_pose"]:
                raise ValueError("Archived H1 returned pose/score mismatch")
            if r["actual_pose"] != entries[ri]["actual_pose"]:
                raise ValueError("Archived R_CONT returned pose/score mismatch")
        hs[seed], rs[seed] = logits, axes
    return hs, rs


def guarded_entry(frame, entries, h_logits, r_axes, tau, policy_status, diagnostic=False):
    decision = select_guarded(entries, h_logits, r_axes, tau, frame["object_scale_m"])
    selected = decision["selected_index"]
    if selected is None:
        return dict(actual_pose=dict(available=False), prediction_status="H1_NO_POSE_PRESERVED",
                    action_role="NO_POSE", fallback=True, technical_failure=False,
                    selected_candidate_id=None, guard_decision=decision,
                    policy_status=policy_status, diagnostic_trial=diagnostic)
    c = entries[selected]
    aliases = sorted(D.source_aliases(c))
    role = "N0" if "N0" in aliases else "E0" if "E0" in aliases else "OTHER"
    return dict(actual_pose=deepcopy(c["actual_pose"]),
                prediction_status="GUARDED_R_CONT_SWITCH" if decision["changed"] else "ORIGINAL_H1_RETAINED",
                action_role=role, fallback=not decision["changed"], technical_failure=False,
                selected_candidate_id=c["candidate_id"], selected_source_aliases=aliases,
                selected_candidate_aliases=c["candidate_aliases"], guard_decision=decision,
                policy_status=policy_status, diagnostic_trial=diagnostic,
                adoption_scope="UNADOPTED_CALIBRATION_TRIAL" if diagnostic else "FROZEN_INTERNAL_CHECK_POLICY",
                calibrated_probability=False)


def run(a):
    start, cpu = time.monotonic(), time.process_time()
    root = a.run_dir
    if (root / "PREDICTION_MANIFEST.json").exists():
        raise FileExistsError("All registered outputs are immutable once sealed")
    plan_path = root / "EXPERIMENT_PLAN.json"
    policy_path = root / "POLICY_CALIBRATION.json"
    aug_path = root / "AUGMENTATION_GATE.json"
    checkpoints = [("H1_NO_B_FEATURE_s" + str(s), root / "training" /
                    ("H1_NO_B_FEATURE_s" + str(s)) / "best.pt") for s in SEEDS]
    for group in AUGMENTATION_GROUPS:
        for seed in SEEDS:
            cp = root / "training" / (group + "_s" + str(seed)) / "best.pt"
            if cp.exists():
                checkpoints.append((group + "_s" + str(seed), cp))
    allowed = [plan_path, policy_path, aug_path, root / "INPUT_MANIFEST.json",
               root / "POLICY_CONFIRMATION.json", root / "POLICY_SPLIT_LOCK_v2.json",
               a.features, a.feature_seal, a.normalizer,
               a.risk_run / "PREDICTIONS.jsonl.gz", a.risk_run / "PREDICTION_MANIFEST.json"]
    for seed in SEEDS:
        allowed.append(a.d1_run / "action_training_v1" / ("D1_H1_MLP_s" + str(seed)) / "best.pt")
        allowed.append(a.risk_run / "training" / ("R_CONT_F0_s" + str(seed)) / "best.pt")
    for _, cp in checkpoints:
        allowed.extend((cp, cp.with_name("RUN_RECEIPT.json")))
    guard = PredictionGuard(allowed_files=allowed,
        allowed_read_roots=(Path(sys.prefix), Path("/usr/lib"), Path(__file__).resolve().parents[2]),
        output_root=root,
        denied_files=(a.reference_dir / "FIXED_REFERENCES.json",),
        denied_roots=(a.reference_dir, a.reference_dir.parent / "score_v1",
                      root / "evaluation_v1", a.risk_run / "evaluation_v1",
                      root / "b_query_v1", root / "coverage_v1"))
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with guard:
        plan = read(plan_path)
        if sha(plan_path) != a.plan_sha256 or plan["status"] != "FROZEN_BEFORE_FIRST_OPTIMIZER_UPDATE":
            raise ValueError("Preregistered plan changed")
        inputs = read(root / "INPUT_MANIFEST.json")
        pm = read(a.risk_run / "PREDICTION_MANIFEST.json")
        if sha(a.risk_run / "PREDICTION_MANIFEST.json") != inputs["prior_replay"]["R_CONT_prediction_manifest_sha256"]:
            raise ValueError("Archived score/model provenance changed")
        if sha(a.risk_run / "PREDICTIONS.jsonl.gz") != pm["artifacts"]["PREDICTIONS.jsonl.gz"]:
            raise ValueError("Archived baseline outputs changed")
        D.bound(a.features, a.feature_seal)
        norm = read(a.normalizer)
        if sha(a.normalizer) != next(e["sha256"] for e in inputs["existing_inputs"] if e["key"] == "normalizer"):
            raise ValueError("Original shared train normalizer changed")
        policies = read(policy_path)
        confirmation = read(root / "POLICY_CONFIRMATION.json")
        if policies["bindings"]["experiment_plan_sha256"] != sha(plan_path):
            raise ValueError("Policy was not selected under this fixed plan")
        if policies["bindings"]["policy_split_sha256"] != sha(root / "POLICY_SPLIT_LOCK_v2.json"):
            raise ValueError("Label-free policy grouping split changed")
        if confirmation["calibration_sha256"] != sha(policy_path) or confirmation["retuning_after_confirmation"] != 0:
            raise ValueError("Once-only internal confirmation binding changed")
        known = {e["key"]: e for e in inputs["existing_inputs"] + inputs["additional_checkpoints"]}
        for seed in SEEDS:
            for kind, group in (("H1", "D1_H1_MLP"), ("R_CONT", "R_CONT_F0")):
                cp = policies["bindings"]["checkpoints"][str(seed)][kind]
                expected = known["checkpoint/" + group + "_s" + str(seed)]
                if cp["path"] != expected["path"] or cp["sha256"] != expected["sha256"] or sha(cp["path"]) != cp["sha256"]:
                    raise ValueError("Policy scores refer to a different archived checkpoint")
        aug = read(aug_path)
        if aug["status"] not in ("EXECUTED", "NOT_RUN"):
            raise ValueError("Conditional augmentation must have an actual terminal gate")
        augmentation = aug["status"] == "EXECUTED"
        expected = arms(augmentation)
        expected_models = [x for x in expected if x.startswith("H1_NO_B_FEATURE_") or x.startswith("AUG_")]
        if set(n for n, _ in checkpoints) != set(expected_models):
            raise ValueError("Missing or unregistered actual trained model")
        models = []
        for name, cp in checkpoints:
            receipt = read(cp.with_name("RUN_RECEIPT.json"))
            if receipt["status"] != "COMPLETE" or receipt["checkpoint_sha256"] != sha(cp):
                raise ValueError("New checkpoint not complete or bound")
            checkpoint = torch.load(cp, map_location="cpu")
            if checkpoint["bindings"]["normalizer_sha256"] != sha(a.normalizer):
                raise ValueError("New model changed normalizer")
            if checkpoint["bindings"]["experiment_plan_sha256"] != sha(plan_path):
                raise ValueError("New model was not fitted under this fixed plan")
            if name.startswith("H1_NO_B_FEATURE") and checkpoint["bindings"]["source_hashes"]["new"]["no_b.py"] != sha(Path(no_b.__file__)):
                raise ValueError("Prediction mask differs from the actual trained mask")
            model = M.make_model("MLP")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            models.append((name, model))
        frames = list(rows(a.features))
        old = {r["id"]: r for r in rows(a.risk_run / "PREDICTIONS.jsonl.gz")}
        if len(frames) != 238 or set(old) != {f["id"] for f in frames}:
            raise ValueError("Complete common238 frames required")
        output, times = [], {n: [] for n, _ in models}
        policy_times = []
        for frame in frames:
            D.validate_frame(frame)
            identity = {k: frame[k] for k in ("id", "partition", "session", "cluster_id")}
            prior = old[frame["id"]]
            if any(prior[k] != v for k, v in identity.items()):
                raise ValueError("Original identity join changed")
            predictions = {n: deepcopy(prior["predictions"][n]) for n in BASELINES}
            entries = D.actions(frame)
            hs, rs = bind_scores(frame, entries, predictions)
            for name, model in models:
                tick = time.monotonic()
                predictions[name] = no_b.infer(frame, model, norm) if name.startswith("H1_NO_B_FEATURE") else M.infer(frame, model, norm, "H1")
                times[name].append(time.monotonic() - tick)
            tick = time.monotonic()
            for seed in SEEDS:
                policy = policies["seeds"][str(seed)]
                for group, key, diagnostic in (("H1_GUARDED_SWITCH", "active_tau", False),
                                               ("H1_GUARD_CALIBRATED_TRIAL", "calibrated_trial_tau", True)):
                    name = group + "_s" + str(seed)
                    predictions[name] = guarded_entry(frame, entries, hs[seed], rs[seed],
                        decode_tau(policy[key]), policy["status"], diagnostic)
            policy_times.append(time.monotonic() - tick)
            if set(predictions) != set(expected):
                raise ValueError("A registered output disappeared")
            for n in expected:
                if n in BASELINES:
                    continue
                pose = predictions[n]["actual_pose"]
                if pose.get("available") and not any(pose == e["actual_pose"] for e in entries):
                    raise ValueError("New output is not an exact original physical action")
            output.append(dict(**identity, object_scale_m=frame["object_scale_m"],
                               predictions=predictions, reference_inputs=False))
        save_rows(root / "PREDICTIONS.jsonl.gz", output)
        bindings = [dict(path=str(p), sha256=sha(p)) for p in allowed]
    write(root / "PREDICTION_GUARD.json", guard.receipt())
    write(root / "PREDICTION_MANIFEST.json", dict(
        schema="OBSERVATION_H1_ALL_OUTPUT_SEAL_V1",
        status="ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING",
        experiment_plan_sha256=sha(plan_path), real_reference_inputs=False,
        real_features_sha256=sha(a.features), augmentation_executed=augmentation,
        arms=list(expected), ordered_ids=[r["id"] for r in output],
        artifacts={"PREDICTIONS.jsonl.gz": sha(root / "PREDICTIONS.jsonl.gz")},
        model_and_input_bindings=bindings, prediction_guard_sha256=sha(root / "PREDICTION_GUARD.json"),
        source_sha256={n: sha(Path(__file__).with_name(n)) for n in ("predict.py", "no_b.py", "policy.py", "registry.py")},
        new_PRIMARY_candidates=0, new_PRIMARY_pose_solves=0,
        archived_baselines="REUSED_BOUND_ACTUAL_SCORES_AND_POSES; original producer cost not zero",
        cached_head_host_seconds={n: dict(rows=len(v), sum=sum(v), mean=float(np.mean(v))) for n, v in times.items()},
        cached_policy_host_seconds=sum(policy_times),
        timing_scope="Cached feature construction + tiny CPU head/decision; excludes original D1 producer, not deployment FPS",
        wall_seconds=time.monotonic() - start, CPU_seconds=time.process_time() - cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def main():
    p = argparse.ArgumentParser()
    for name in ("run-dir", "risk-run", "d1-run", "features", "feature-seal", "normalizer", "reference-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
