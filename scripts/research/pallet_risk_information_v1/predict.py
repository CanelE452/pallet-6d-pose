"""Fresh-process, reference-denied inference on the unchanged real action bank."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from scripts.research.pallet_boundary_action_repair_v1 import action_data as OLD
from .common import read, rows, save_rows, sha, write, measured
from .feature_support import normalized_extra
from .prediction_guard import PredictionGuard
from .registry import BASELINES, BASIC_GROUPS, VISUAL_GROUPS, SEEDS, arms
from .risk_models import RiskModel, choose
from .visual_patches import PatchProvider, CHUNK_ACTIONS


def output_for(frame, model, f0, entries, extra, provider, device):
    if not entries:
        return dict(actual_pose=dict(available=False), prediction_status="NO_POSE",
            action_role="NO_POSE", selected_candidate_id=None, selected_source_aliases=[],
            fallback=False, technical_failure=False, actions=[], model_outputs=[])
    f0 = torch.as_tensor(f0, dtype=torch.float32, device=device)
    ext = torch.as_tensor(extra, dtype=torch.float32, device=device) if model.extra_dim else None
    with torch.inference_mode():
        if model.visual:
            parts = []
            for first in range(0, len(entries), CHUNK_ACTIONS):
                last = min(first + CHUNK_ACTIONS, len(entries))
                patches, valid = provider(frame, entries, range(first, last))
                parts.append(model(f0[first:last], ext[first:last],
                    torch.as_tensor(patches, dtype=torch.float32, device=device),
                    torch.as_tensor(valid, dtype=torch.float32, device=device)))
            output = torch.cat(parts).cpu().numpy()
        else:
            output = model(f0, ext).cpu().numpy()
    selected = choose(output, model.arm)
    action = entries[selected]
    aliases = sorted(OLD.source_aliases(action))
    role = "N0" if "N0" in aliases else "E0" if "E0" in aliases else "OTHER"
    scalar = float(np.max(output[selected])) if model.arm == "R_CONT_F0" else float(output[selected])
    return dict(actual_pose=deepcopy(action["actual_pose"]),
        prediction_status="ACTUAL_FROZEN_BANK_ACTION_SELECTED",
        selected_candidate_id=action["candidate_id"], selected_source_aliases=aliases,
        selected_candidate_aliases=action["candidate_aliases"], action_role=role,
        fallback=False, technical_failure=False, selected_score=scalar,
        score_semantics="minimize maximum predicted log-axis error" if model.arm == "R_CONT_F0"
                        else "maximize uncalibrated within-bank rank score",
        calibrated_probability=False, model_outputs=output.tolist(),
        actions=[dict(candidate_id=c["candidate_id"], action_role=c["action_role"],
            source_aliases=sorted(OLD.source_aliases(c)), candidate_aliases=c["candidate_aliases"])
                 for c in entries], actual_action_count=len(entries))


def run(args):
    run_dir, old_run = args.run_dir, args.old_run
    started, cpu = time.monotonic(), time.process_time()
    if (run_dir / "PREDICTION_MANIFEST.json").exists():
        raise FileExistsError("Existing predictions must remain sealed")
    plan_path = run_dir / "EXPERIMENT_PLAN.json"
    gate_path = run_dir / "V_GATE.json"
    # Enumerate paths without decoding any input before the guard is active.
    possible_checkpoints = [(group + "_s" + str(seed),
                            run_dir / "training" / (group + "_s" + str(seed)) / "best.pt")
                           for group in BASIC_GROUPS + VISUAL_GROUPS for seed in SEEDS]
    exact_files = [plan_path, gate_path, args.features, args.feature_seal,
        args.f0_normalizer, args.extra_features, args.extra_seal, args.extra_normalizer,
        old_run / "PREDICTIONS.jsonl.gz", old_run / "PREDICTION_SEAL.json"]
    for _, checkpoint in possible_checkpoints:
        if checkpoint.is_file():
            exact_files.extend((checkpoint, checkpoint.with_name("RUN_RECEIPT.json")))
    if args.visual_inputs is not None and args.visual_cache is not None:
        exact_files.extend((args.visual_inputs, args.visual_cache / "CACHE_SEAL.json"))
        exact_files.extend(p for p in args.visual_cache.iterdir()
                           if p.suffix in (".json", ".npy"))
    refs = args.reference_dir
    denied_files = [refs / "FIXED_REFERENCES.json", refs.parent / "score_v1" / "FIXED_REFERENCES.json"]
    denied_roots = [refs, refs.parent / "score_v1", old_run / "evaluation_v1",
                    run_dir / "evaluation_v1", run_dir / "b_reference_support_v1",
                    run_dir / "independent_evaluation_v1"]
    guard = PredictionGuard(allowed_files=exact_files,
        allowed_read_roots=(Path(sys.prefix), Path("/usr/lib"), Path(__file__).resolve().parents[2]),
        output_root=run_dir, denied_files=denied_files, denied_roots=denied_roots)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with guard:
        # This gate uses only preregistered synthetic-validation/input/budget
        # evidence; real-reference files remain forbidden throughout inference.
        gate = read(gate_path)
        visual = bool(gate["run_visual"])
        groups = BASIC_GROUPS + (VISUAL_GROUPS if visual else ())
        checkpoints = [(name, path) for name, path in possible_checkpoints
                       if any(name == group + "_s" + str(seed)
                              for group in groups for seed in SEEDS)]
        if visual and not all((args.visual_inputs, args.visual_cache)):
            raise ValueError("Executed visual gate requires complete sealed visual inputs/cache")
        plan = read(plan_path)
        if plan["status"] != "FROZEN_BEFORE_FIRST_OPTIMIZER_UPDATE":
            raise ValueError("Original experiment plan is not fixed")
        OLD.bound(args.features, args.feature_seal)
        OLD.bound(args.extra_features, args.extra_seal)
        f0_normalizer = read(args.f0_normalizer)
        extra_normalizer = read(args.extra_normalizer)
        frames = list(rows(args.features))
        extras = {r["id"]: r for r in rows(args.extra_features)}
        if len(frames) != 238 or set(extras) != {r["id"] for r in frames}:
            raise ValueError("Full common238 inputs required")
        old_seal = read(old_run / "PREDICTION_SEAL.json")
        if sha(old_run / "PREDICTIONS.jsonl.gz") != old_seal["predictions_sha256"]:
            raise ValueError("Archived comparator predictions changed")
        archived = {r["id"]: r for r in rows(old_run / "PREDICTIONS.jsonl.gz")}
        if visual:
            cache_seal = read(args.visual_cache / "CACHE_SEAL.json")
            if (cache_seal["status"] != "COMPLETE"
                    or cache_seal["domain"] != "real"
                    or cache_seal["input_sha256"] != sha(args.visual_inputs)
                    or cache_seal["features_sha256"] != sha(args.features)
                    or len(cache_seal["records"]) != 238):
                raise ValueError("Complete exact real visual cache binding required")
            if {r["id"] for r in cache_seal["records"]} != {r["id"] for r in frames}:
                raise ValueError("Visual cache omitted an original real frame")
            for record in cache_seal["records"]:
                if sha(args.visual_cache / record["array_file"]) != record["array_sha256"]:
                    raise ValueError("Visual patch array changed")
                metadata_path = (args.visual_cache / record["array_file"]).with_suffix(".json")
                if sha(metadata_path) != record["metadata_sha256"]:
                    raise ValueError("Visual patch metadata changed")
        provider = PatchProvider(args.visual_inputs, args.visual_cache) if visual else None
        models = []
        for name, checkpoint_path in checkpoints:
            receipt = read(checkpoint_path.with_name("RUN_RECEIPT.json"))
            if receipt["status"] != "COMPLETE" or receipt["checkpoint_sha256"] != sha(checkpoint_path):
                raise ValueError("Registered trained checkpoint not complete/bound")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if name != checkpoint["arm"] + "_s" + str(checkpoint["seed"]):
                raise ValueError("Checkpoint arm/seed mismatch")
            if checkpoint["bindings"]["normalizer_sha256"] != sha(args.f0_normalizer):
                raise ValueError("Original shared D1 F0 normalization changed")
            if checkpoint["extra_dim"] and checkpoint["bindings"]["extra_normalizer_sha256"] != sha(args.extra_normalizer):
                raise ValueError("F1 train-only normalization changed")
            device = "cuda" if checkpoint["arm"].startswith("V_") else "cpu"
            model = RiskModel(checkpoint["arm"], checkpoint["extra_dim"])
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(device).eval()
            models.append((name, model, device))
        output = []
        head_times = {name: [] for name, _, _ in models}
        shared_feature_times = []
        for frame in frames:
            feature_start = time.monotonic()
            OLD.validate_frame(frame)
            f0, entries = OLD.inputs(frame, f0_normalizer, "H1")
            extra = normalized_extra(frame, extras[frame["id"]], extra_normalizer)
            shared_feature_times.append(time.monotonic() - feature_start)
            old = archived[frame["id"]]
            identity = {k: frame[k] for k in ("id", "partition", "session", "cluster_id")}
            if any(old[k] != identity[k] for k in identity):
                raise ValueError("Archived comparator identity mismatch")
            predictions = {a: deepcopy(old["predictions"][a]) for a in BASELINES}
            for name, model, device in models:
                head_start = time.monotonic()
                predictions[name] = output_for(frame, model, f0, entries, extra, provider, device)
                # output_for returns CPU arrays, completing visual CUDA work.
                head_times[name].append(time.monotonic() - head_start)
            if set(predictions) != set(arms(visual)):
                raise ValueError("Incomplete registered arm output")
            output.append(dict(**identity, object_scale_m=frame["object_scale_m"],
                predictions=predictions, reference_inputs=False))
        save_rows(run_dir / "PREDICTIONS.jsonl.gz", output)
        bindings = [dict(path=str(p), sha256=sha(p)) for p in exact_files]
        if visual:
            torch.cuda.synchronize()
    write(run_dir / "PREDICTION_GUARD.json", guard.receipt())
    write(run_dir / "PREDICTION_MANIFEST.json", dict(
        status="ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING",
        before_scoring=True, real_reference_inputs=False, visual_executed=visual,
        experiment_plan_sha256=sha(plan_path), V_gate_sha256=sha(gate_path),
        real_features_sha256=sha(args.features), arms=list(arms(visual)),
        ordered_ids=[r["id"] for r in output], model_and_input_bindings=bindings,
        source_sha256={n: sha(Path(__file__).with_name(n))
            for n in ("predict.py", "risk_models.py", "feature_support.py", "visual_patches.py", "prediction_guard.py", "registry.py")},
        prediction_guard_sha256=sha(run_dir / "PREDICTION_GUARD.json"),
        artifacts={"PREDICTIONS.jsonl.gz": sha(run_dir / "PREDICTIONS.jsonl.gz")},
        no_new_candidate_generation=True, no_new_pose_solve=True,
        head_forward_host_timings={name: dict(rows=len(values), sum_seconds=sum(values),
            mean_seconds=float(np.mean(values)), P90_seconds=float(np.percentile(values, 90)),
            max_seconds=max(values), includes="tensor transfer, actual head and visual cache reads when used; output returned to CPU",
            excludes="initial detector/N3/SubPix/state/pose producer; shared features and checkpoint loading")
            for name, values in head_times.items()},
        shared_normalized_feature_host_seconds=sum(shared_feature_times),
        archived_baseline_forward_status="REUSED_SAVED_OUTPUTS; no new forward here; original cache production cost is not zero",
        timing_limitation="Common sequential cached campaign; not independent deployment latency or Jetson FPS",
        wall_seconds=time.monotonic() - started, CPU_seconds=time.process_time() - cpu,
        GPU_activity_seconds=None,
        GPU_activity_note="CUDA inference, if run, bounded above by whole process wall; not exact GPU time",
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def main():
    parser = argparse.ArgumentParser()
    for name in ("run-dir", "old-run", "features", "feature-seal", "f0-normalizer",
                 "extra-features", "extra-seal", "extra-normalizer", "reference-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--visual-inputs", type=Path)
    parser.add_argument("--visual-cache", type=Path)
    args = parser.parse_args()
    with measured(args.run_dir / "RUN_LOG.jsonl", "REFERENCE_DENIED_REAL_PREDICTION"):
        run(args)


if __name__ == "__main__":
    main()
