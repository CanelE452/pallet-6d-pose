"""Evaluate only the final matched pose-target checkpoint, without GT selection."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time
import numpy as np
import torch
import cv2
sys.dont_write_bytecode = True
from .baseline import BASELINE_ROOT
from scripts.research.pallet_joint_action_handoff_20261006_v1.a_common import POSE, read, sha, write, hash_value
from scripts.research.pallet_joint_action_handoff_20261006_v1.a_data import Data, network_bank
from scripts.research.pallet_joint_action_handoff_20261006_v1.scorer import JointActionScorer, decode_bank
from .training import (ROOT, DOC, OLD_DOC, ReadOnlyBanks, numeric_contract,
                       forward_bank, locked_paths, frozen_input_contract)

_spec = importlib.util.spec_from_file_location("pose_target_existing_eval_math",
    BASELINE_ROOT / "scripts/research/pallet_dim_conditioned_p_v1/eval_math.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def finite_json(value):
    if isinstance(value, dict): return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [finite_json(v) for v in value]
    if hasattr(value, "tolist"): return finite_json(value.tolist())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def indexed(rows):
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "Duplicate stored row IDs"
    return {row["id"]: row for row in rows}


def legacy_rows(split):
    base = read(OLD_DOC / f"results/A_{split}_BASELINES.json")
    rows = {name: indexed(values) for name, values in base["rows"].items()}
    for seed in (1, 2, 3):
        payload = read(OLD_DOC / f"results/A_{split}_FIT_GEO_seed{seed}.json")
        rows[f"FIT_GEO_J_seed{seed}"] = indexed(payload["rows"]["GEO_J"])
    oracle = indexed([row for row in read(OLD_DOC / f"results/A_{split}_ORACLE.json")["rows"] if row["arm"] == "GEO"])
    for values in rows.values(): assert set(values) == set(rows["RAW"])
    assert set(oracle) == set(rows["RAW"])
    return rows, oracle


def real_banks(directory, frames):
    path = Path(directory) / "sampling_masks/REAL_DEV_generated_banks.npz"
    receipt = read(OLD_DOC / "results/A_SAMPLING_MASK_RECEIPT.json")
    expected = next(v for v in receipt["external_mask_files"] if Path(v["path"]).name == path.name)
    assert sha(path) == expected["sha256"] and path.stat().st_size == expected["bytes"]
    saved = np.load(path, allow_pickle=False)
    ids = saved["ids"].astype(str).tolist()
    assert len(ids) == len(set(ids)) == 319
    lookup = {fid: i for i, fid in enumerate(ids)}
    assert set(lookup) == {f["id"] for f in frames}
    result = []
    for frame in frames:
        i = lookup[frame["id"]]
        count = int(saved["counts"][i])
        points = np.array(saved["points"][i, :count], copy=True)
        q = frame["q"] if frame["q"] is not None else np.full((9, 2), np.nan)
        assert np.array_equal(points[0], q, equal_nan=True), "Stored real bank NoOp differs from frozen detected object"
        result.append(dict(points=points, hypotheses=["NoOp"] + [None] * (count - 1)))
    return result, dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
        generating_hypothesis_labels="NOT_RETAINED_IN_PRIOR_REAL_CACHE; final F hypothesis is evaluated")


def prediction_packet(data, frames, banks, source, device="cuda"):
    """Only original inference inputs. No GT, poses, permutations or oracle cost."""
    if source:
        rows = [f["cache_row"] for f in frames]
        batch = data.batch(rows, device=device, supervision=False)
    else:
        batch = data.real_batch(frames, device=device)
    assert set(batch) == {"p3", "p4", "points", "boxes", "point_valid", "input_shape", "context"}
    maximum = max(2, max(len(b["points"]) for b in banks))
    q, action_valid = [], []
    for i, (frame, bank) in enumerate(zip(frames, banks)):
        points = network_bank(bank, frame, batch["points"][i].cpu().numpy())
        padded = np.repeat(points[:1], maximum, axis=0)
        padded[:len(points)] = points
        q.append(padded)
        action_valid.append(np.arange(maximum) < len(points))
    return batch, torch.from_numpy(np.stack(q)).to(device), torch.from_numpy(np.stack(action_valid)).to(device)


def select_native(output, banks, frames):
    """The sole inference decision is original hard J; native centre stays RAW."""
    _, chosen = decode_bank(output, "J")
    result = []
    for index, bank, frame in zip(chosen.cpu().numpy(), banks, frames):
        index = int(index)
        assert 0 <= index < len(bank["points"])
        q = np.array(bank["points"][index], copy=True)
        q[8:] = frame["q"][8:]
        result.append((q, index, bank["hypotheses"][index]))
    return result


def scored(frame, points, target, index, generation, seed):
    """GT is accessed after the prediction has been fixed, for metrics only."""
    if points is None: points = np.full((9, 2), np.nan)
    detected = frame["q"] is not None and np.isfinite(frame["q"][:8]).any()
    corner = M.measure(points, target["gt"], target["valid"], target["permutations"],
                       frame["raw_hw"], target["matched"] and detected, detected)
    corner.update(id=frame["id"], session=frame["session"])
    begin = time.monotonic()
    counts = {name: 0 for name in ("solvePnP", "solvePnPGeneric", "solvePnPRefineLM")}
    original = {name: getattr(cv2, name) for name in counts}
    for name in counts:
        def counted(*args, _name=name, **kwargs):
            counts[_name] += 1
            return original[_name](*args, **kwargs)
        setattr(cv2, name, counted)
    try:
        pose = POSE.infer(points, frame["K"], frame["xyz"], frame["source"])
    finally:
        for name, function in original.items(): setattr(cv2, name, function)
    seconds = time.monotonic() - begin
    metric = POSE.metric((frame["id"], pose, frame["truth"]))
    return dict(id=frame["id"], session=frame["session"], method=f"POSE_TARGET_GEO_J_seed{seed}",
        corner=corner, pose=metric, selected_index=index,
        final_hypothesis=pose.get("selected_hypothesis"), generating_hypothesis=generation,
        reference_distance_m=float(np.linalg.norm(frame["truth"]["t"])), native_points=points,
        F_attempt=True, F_complete=True, F_available=pose["available"], F_seconds=seconds,
        PnP_counts=counts)


def summarize(rows):
    successful = [r["pose"] for r in rows if r["pose"]["available"]]
    result = dict(corner=M.summary([r["corner"] for r in rows]),
        pose=dict(total_frames=len(rows), available=len(successful), failures=len(rows) - len(successful),
                  coverage=len(successful) / len(rows)),
        NoOp=sum(r["selected_index"] == 0 for r in rows),
        selected_action_distribution={str(i): sum(r["selected_index"] == i for r in rows)
                                      for i in sorted(set(r["selected_index"] for r in rows))})
    for key in ("translation_cm", "rotation_deg", "ADDsym_m"):
        values = [r[key] for r in successful]
        result["pose"][key] = dict(median=float(np.median(values)) if values else None,
                                  P90=float(np.quantile(values, .9)) if values else None)
    return result


@torch.no_grad()
def evaluate(data, banks, args):
    fit_path = DOC / f"fits/POSE_TARGET_GEO_seed{args.seed}.json"
    fit = read(fit_path)
    assert fit["complete"] and fit["updates"] == 6000 and fit["final_checkpoint_only"]
    ckpath = Path(fit["checkpoint_path"])
    assert sha(ckpath) == fit["checkpoint_sha256"]
    checkpoint = torch.load(ckpath, map_location="cpu", weights_only=False)
    assert checkpoint["complete"] and checkpoint["step"] == 6000 and checkpoint["binding"] == fit["binding"]
    frozen_inputs = frozen_input_contract(data)
    head = JointActionScorer(5, **checkpoint["config"]).cuda().eval()
    head.load_state_dict(checkpoint["model_state_dict"])
    head.requires_grad_(False)
    targets_path = data.root / "data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json"
    targets = read(targets_path)
    sets = [("SYNTH_HELDOUT", [data.source_frame(row) for row in data.eval_rows])]
    if args.split in ("all", "real"): sets.append(("REAL_DEV", data.real_frames()))
    if args.split == "real": sets = [v for v in sets if v[0] == "REAL_DEV"]
    results = {}
    for split, frames in sets:
        expected = 1985 if split == "SYNTH_HELDOUT" else 319
        assert len(frames) == expected
        rows_path = DOC / f"results/{split}_POSE_TARGET_GEO_J_seed{args.seed}.json"
        old_methods, oracle = legacy_rows(split)
        assert {f["id"] for f in frames} == set(old_methods["RAW"])
        if split == "SYNTH_HELDOUT":
            prepared = [banks.get(f["cache_row"]) for f in frames]
            bank_artifact = dict(path=str(banks.directory / "source_banks.npy"), sha256=banks.bank_sha256)
        else:
            prepared, bank_artifact = real_banks(args.bank_cache, frames)
        binding = hash_value(dict(protocol=sha(DOC / "PROTOCOL.json"), code=sha(Path(__file__)),
            checkpoint=fit["checkpoint_sha256"], bank=bank_artifact, target=sha(targets_path)))
        external = Path(args.cost_cache) / "evaluations" / split / f"seed{args.seed}"
        completion = external / "COMPLETE.json"
        if rows_path.exists():
            prior = read(rows_path)
            assert prior["binding"] == binding and prior["full_denominator"] == expected
            assert len(prior["rows"]) == expected and prior["complete"]
            completed = read(completion)
            assert completed["binding"] == binding and completed["result_sha256"] == sha(rows_path)
            manifest_path = external / "ROW_MANIFEST.json"
            assert sha(manifest_path) == completed["row_manifest_sha256"]
            for artifact in read(manifest_path)["rows"]:
                path = external / artifact["file"]
                assert path.stat().st_size == artifact["bytes"] and sha(path) == artifact["sha256"]
            results[split] = prior
            continue
        rows, batch_calls, examples = [], 0, 0
        row_artifacts = []
        started = time.monotonic()
        journal = DOC / f"results/{split}_POSE_TARGET_GEO_J_seed{args.seed}_ATTEMPTS.json"
        assert not journal.exists(), "Interrupted evaluation cannot be silently rerun; preserve and account attempts first"
        assert not completion.exists() and not (external / "ROW_BINDING.json").exists(), "Prior incomplete evaluation cache must not be silently overwritten"
        external.mkdir(parents=True, exist_ok=True)
        write(external / "ROW_BINDING.json", dict(binding=binding, full_denominator=expected,
            checkpoint_sha256=fit["checkpoint_sha256"], code_sha256=sha(Path(__file__))))
        for start in range(0, len(frames), 16 if split == "SYNTH_HELDOUT" else 1):
            selected_frames = frames[start:start + (16 if split == "SYNTH_HELDOUT" else 1)]
            selected_banks = prepared[start:start + len(selected_frames)]
            run_network = not (split == "REAL_DEV" and
                (selected_frames[0]["q"] is None or len(selected_banks[0]["points"]) == 1))
            write(journal, dict(status="ATTEMPT", binding=binding, full_denominator=expected,
                checkpoint=fit["checkpoint_sha256"], attempted_frame_offset=start,
                attempted_batch_ids=[f["id"] for f in selected_frames],
                completed_row_count=len(rows), attempted_batch_calls=batch_calls + int(run_network),
                completed_batch_calls=batch_calls, completed_examples=examples, external_rows=str(external)))
            if not run_network:
                choices = [(selected_frames[0]["q"], 0, "NoOp")]
            else:
                batch, candidates, valid = prediction_packet(data, selected_frames, selected_banks, split == "SYNTH_HELDOUT")
                output = forward_bank(head, batch, candidates, valid)
                batch_calls += 1
                examples += len(selected_frames)
                choices = select_native(output, selected_banks, selected_frames)
            for frame, bank, (q, index, generation) in zip(selected_frames, selected_banks, choices):
                write(journal, dict(status="FINAL_F_ATTEMPT", binding=binding, full_denominator=expected,
                    frame_id=frame["id"], selected_index=index,
                    completed_row_count=len(rows), attempted_final_F_calls=len(rows) + 1,
                    completed_final_F_calls=len(rows), completed_batch_calls=batch_calls,
                    completed_examples=examples, external_rows=str(external)))
                row = scored(frame, q, targets[frame["id"]], index, generation, args.seed)
                o = oracle[frame["id"]]
                row.update(action_count=len(bank["points"]), oracle_selected_index=o["index"],
                    oracle_ADDsym_m=o["oracle_ADDsym_m"],
                    oracle_gap_m=row["pose"]["ADDsym_m"] - o["oracle_ADDsym_m"]
                    if row["pose"]["available"] and o["oracle_ADDsym_m"] is not None else None,
                    RAW_final_hypothesis=old_methods["RAW"][frame["id"]]["final_hypothesis"],
                    oracle_final_hypothesis=o["final_hypothesis"])
                if row["oracle_gap_m"] is not None:
                    assert row["oracle_gap_m"] >= -1e-7, "Chosen same-bank F beats cached oracle beyond fixed numerical tolerance"
                row = finite_json(row)
                row_path = external / f"row_{len(rows):06d}.json"
                assert not row_path.exists(), "An actual F row must never be overwritten or silently recomputed"
                write(row_path, dict(binding=binding, row=row))
                row_artifacts.append(dict(id=row["id"], file=row_path.name,
                    sha256=sha(row_path), bytes=row_path.stat().st_size))
                rows.append(row)
            write(journal, dict(status="BATCH_COMPLETE", binding=binding, full_denominator=expected,
                completed_row_count=len(rows), completed_batch_calls=batch_calls,
                completed_examples=examples, completed_final_F_calls=len(rows), external_rows=str(external)))
            if start % 320 == 0: print("POSE_TARGET_EVAL", args.seed, split, len(rows), expected, flush=True)
        assert len(rows) == expected and len(indexed(rows)) == expected
        execution = dict(new_refiner_batches=batch_calls, new_refiner_examples=examples,
            new_backbone_calls=0, new_bank_generation=0, new_final_F_attempts=len(rows),
            new_final_F_completed=sum(r["F_complete"] for r in rows),
            final_F_available=sum(r["F_available"] for r in rows), optimizer_updates=0,
            PnP_counts={k: sum(r["PnP_counts"][k] for r in rows) for k in ("solvePnP", "solvePnPGeneric", "solvePnPRefineLM")},
            seconds_wall=time.monotonic() - started, final_F_seconds=sum(r["F_seconds"] for r in rows))
        result = dict(schema="pose_target_final_checkpoint_evaluation_v1", status="DONE", complete=True,
            seed=args.seed, split=split, rows=rows, summary=summarize(rows), full_denominator=expected,
            checkpoint_sha256=fit["checkpoint_sha256"], bank_binding=banks.binding, bank_artifact=bank_artifact,
            target_sha256=sha(targets_path), code_sha256=sha(Path(__file__)), binding=binding,
            frozen_feature_and_dimension_inputs=frozen_inputs,
            GT_inference_access=False, trained=True, readout="unchanged J", execution=execution,
            same_immutable_baselines=list(old_methods), reference="Repeated-use DEV, not independent physical 6D metrology")
        write(rows_path, finite_json(result))
        manifest_path = external / "ROW_MANIFEST.json"
        write(manifest_path, dict(binding=binding, rows=row_artifacts, full_denominator=expected))
        write(completion, dict(binding=binding, result_sha256=sha(rows_path),
            row_manifest_sha256=sha(manifest_path), full_denominator=expected, execution=execution))
        write(journal, dict(status="DONE", binding=binding, full_denominator=expected,
            execution=execution, result_sha256=sha(rows_path), external_completion_path=str(completion),
            external_completion_sha256=sha(completion)))
        results[split] = result
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bank-cache", type=Path, required=True)
    parser.add_argument("--cost-cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--split", choices=("all", "synthetic", "real"), default="all")
    args = parser.parse_args(argv)
    assert (DOC / "PROTOCOL.json").exists()
    locked_paths(args)
    numeric_contract()
    cv2.setNumThreads(1)
    data = Data(args.source_root)
    banks = ReadOnlyBanks(data, args.bank_cache)
    result = evaluate(data, banks, args)
    print(json.dumps({k: dict(status=v["status"],frames=v["full_denominator"],execution=v["execution"]) for k,v in result.items()}))
    return result


if __name__ == "__main__":
    main()
