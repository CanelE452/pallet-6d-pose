"""Final hard-J development evaluation and a fixed TRAIN cost-table probe.

Only the two final quick checkpoints are used. Inference never receives GT,
oracle costs or F-availability masks. Prior helpers are called as pure readers
or computations; their destinations and entrypoints remain unchanged.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import cv2
import numpy as np
import torch

sys.dont_write_bytecode = True
from .common import (ROOT, DOC, OUTPUT, COST, BANKS, OLD_DOC, HARD_DOC, Data,
                     Banks, read, write, sha, hash_value, numeric_contract,
                     forward_bank, action_scores, verify_derived)
from scripts.research.pallet_pose_target_6d_20261006_v1 import evaluation as E
from scripts.research.pallet_joint_action_handoff_20261006_v1.scorer import JointActionScorer

METHODS = ("SOFT6D", "EXPECT6D")
SPLITS = (("SYNTH_HELDOUT", 1985), ("REAL_DEV", 319))
PNP = ("solvePnP", "solvePnPGeneric", "solvePnPRefineLM")


def clean(value):
    return E.finite_json(value)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_paths(source_root, bank_cache, cost_cache, output_cache):
    protocol = read(DOC / "PROTOCOL.json")
    prior = read(HARD_DOC / "PROTOCOL.json")
    for key, actual in (("source_root", source_root), ("candidate_bank_cache", bank_cache),
                        ("pose_cost_cache", cost_cache)):
        assert Path(actual).resolve() == Path(prior[key]).resolve(), key
        if key in protocol:
            assert Path(actual).resolve() == Path(protocol[key]).resolve(), key
    output = Path(output_cache).resolve()
    assert output not in (Path(bank_cache).resolve(), Path(cost_cache).resolve())
    if "output_cache" in protocol:
        assert output == Path(protocol["output_cache"]).resolve()
    assert read(DOC / "LOSS_SETUP.json")["status"] == "PASS"
    return protocol, read(DOC / "LOSS_SETUP.json")


def fit_receipt(method, output_cache):
    assert method in METHODS
    fit = read(DOC / "TRAIN_RECEIPTS.json")["methods"][method]
    assert fit["status"] == "DONE" and fit["complete"]
    assert fit["updates"] == 6000 and fit["exposures"] == 96000
    path = Path(fit["checkpoint_path"])
    assert path.resolve().is_relative_to(Path(output_cache).resolve())
    assert sha(path) == fit["checkpoint_sha256"]
    old = read(OLD_DOC / "A_fits/GEO_seed1.json")
    assert fit["initial_state_sha256"] == old["first_step"]["initial_state_sha256"]
    assert fit["order_sha256"] == old["order_sha256"]
    return fit


def load_head(fit):
    checkpoint = torch.load(fit["checkpoint_path"], map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 6000 and checkpoint["config"] == fit["config"]
    if "binding" in fit:
        assert checkpoint["binding"] == fit["binding"]
    head = JointActionScorer(5, **checkpoint["config"]).cuda().eval()
    head.load_state_dict(checkpoint["model_state_dict"])
    assert sum(p.numel() for p in head.parameters()) == 20259
    head.requires_grad_(False)
    return head


def original_ids(split):
    registry = read(OLD_DOC / "results/A_ID_MANIFEST.json")["IDs"]
    return registry["synthetic_evaluation" if split == "SYNTH_HELDOUT" else "real_evaluation"]


def split_paths(split, method, output_cache):
    name = f"{split}_{method}_seed1"
    return (DOC / "results" / (name + ".jsonl"),
            DOC / "results" / (name + "_EXECUTION.json"),
            Path(output_cache) / "evaluations" / (name + "_ATTEMPT.json"))


def reuse_evaluation(split, method, output_cache, fit):
    rows_path, receipt_path, guard_path = split_paths(split, method, output_cache)
    if not receipt_path.exists():
        assert not rows_path.exists() and not rows_path.with_suffix(".jsonl.pending").exists()
        assert not guard_path.exists(), "Incomplete evaluation is preserved; no hidden retry"
        return None
    result = read(receipt_path)
    assert result["status"] == "DONE" and result["complete"]
    assert result["checkpoint_sha256"] == fit["checkpoint_sha256"]
    assert result["code_sha256"] == sha(Path(__file__))
    assert result["protocol_sha256"] == sha(DOC / "PROTOCOL.json")
    assert result["setup_sha256"] == sha(DOC / "LOSS_SETUP.json")
    assert result["rows_artifact"]["sha256"] == sha(rows_path)
    assert result["rows_artifact"]["bytes"] == rows_path.stat().st_size
    rows = read_jsonl(rows_path)
    ids = [r["id"] for r in rows]
    assert ids == original_ids(split) and len(ids) == len(set(ids)) == result["full_denominator"]
    guard = read(guard_path)
    assert guard["status"] == "DONE" and guard["binding"] == result["binding"]
    assert guard["execution"] == result["execution"] and guard["rows_sha256"] == sha(rows_path)
    return result


@torch.no_grad()
def evaluate(method, source_root=ROOT, bank_cache=BANKS, cost_cache=COST, output_cache=OUTPUT):
    _, setup = check_paths(source_root, bank_cache, cost_cache, output_cache)
    fit = fit_receipt(method, output_cache)
    results = {split: reuse_evaluation(split, method, output_cache, fit) for split, _ in SPLITS}
    if all(value is not None for value in results.values()):
        return results
    verify_derived(setup)
    numeric_contract()
    cv2.setNumThreads(1)
    data = Data(source_root)
    banks = Banks(data, bank_cache, setup_report=setup)
    head = load_head(fit)
    target_path = data.root / "data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json"
    targets = read(target_path)
    for split, denominator in SPLITS:
        if results[split] is not None:
            continue
        frames = ([data.source_frame(row) for row in data.eval_rows]
                  if split == "SYNTH_HELDOUT" else data.real_frames())
        ids = [frame["id"] for frame in frames]
        assert ids == original_ids(split) and len(ids) == len(set(ids)) == denominator
        base = read(OLD_DOC / f"results/A_{split}_BASELINES.json")
        raw = E.indexed(base["rows"]["RAW"])
        oracle = E.indexed([r for r in read(OLD_DOC / f"results/A_{split}_ORACLE.json")["rows"] if r["arm"] == "GEO"])
        assert set(ids) == set(raw) == set(oracle) and sha(target_path) == base["target_sha256"]
        if split == "SYNTH_HELDOUT":
            prepared = [banks.get(frame["cache_row"], frame) for frame in frames]
            bank_artifact = dict(path=str(Path(bank_cache) / "source_banks.npy"), sha256=banks.bank_sha256)
        else:
            prepared, bank_artifact = E.real_banks(bank_cache, frames)
        assert banks.binding == base["bank_binding"]
        binding = hash_value(dict(method=method, checkpoint=fit["checkpoint_sha256"],
            protocol=sha(DOC / "PROTOCOL.json"), setup=sha(DOC / "LOSS_SETUP.json"),
            code=sha(Path(__file__)), bank=bank_artifact, target=sha(target_path)))
        rows_path, receipt_path, guard_path = split_paths(split, method, output_cache)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        pending = rows_path.with_suffix(".jsonl.pending")
        rows, batches, examples, attempted_F = [], 0, 0, 0
        began = time.monotonic()
        ledger = dict(status="ATTEMPT", binding=binding, method=method, split=split,
            full_denominator=denominator, attempted_F=0, completed_F=0,
            attempted_refiner_batches=0, completed_refiner_batches=0, completed_refiner_examples=0,
            reservation_scope="Pre-call ledger entries are declared attempts; interrupted call completion is unknown and never silently replayed")
        write(guard_path, ledger)
        try:
            with pending.open("x") as stream:
                size = 16 if split == "SYNTH_HELDOUT" else 1
                for start in range(0, denominator, size):
                    group, group_banks = frames[start:start + size], prepared[start:start + size]
                    run_network = not (split == "REAL_DEV" and (group[0]["q"] is None or len(group_banks[0]["points"]) == 1))
                    ledger.update(status="FORWARD_ATTEMPT", batch_ids=[f["id"] for f in group],
                                  attempted_refiner_batches=batches + int(run_network))
                    write(guard_path, ledger)
                    if run_network:
                        batch, candidates, valid = E.prediction_packet(data, group, group_banks, split == "SYNTH_HELDOUT")
                        output = forward_bank(head, batch, candidates, valid)
                        choices = E.select_native(output, group_banks, group)
                        batches += 1
                        examples += len(group)
                    else:
                        choices = [(group[0]["q"], 0, "NoOp")]
                    ledger.update(completed_refiner_batches=batches, completed_refiner_examples=examples)
                    for frame, bank, (points, index, generation) in zip(group, group_banks, choices):
                        attempted_F += 1
                        assert attempted_F <= denominator
                        ledger.update(status="FINAL_F_ATTEMPT", frame_id=frame["id"],
                                      attempted_F=attempted_F, completed_F=len(rows))
                        write(guard_path, ledger)
                        row = E.scored(frame, points, targets[frame["id"]], index, generation, 1)
                        row["method"] = f"{method}_GEO_J_seed1"
                        bound = oracle[frame["id"]]
                        assert len(bank["points"]) == bound["actions"]
                        assert row["session"] == raw[row["id"]]["session"]
                        row.update(action_count=len(bank["points"]), oracle_selected_index=bound["index"],
                            oracle_ADDsym_m=bound["oracle_ADDsym_m"],
                            oracle_gap_m=row["pose"]["ADDsym_m"] - bound["oracle_ADDsym_m"]
                            if row["pose"]["available"] and bound["oracle_ADDsym_m"] is not None else None,
                            RAW_final_hypothesis=raw[row["id"]]["final_hypothesis"],
                            oracle_final_hypothesis=bound["final_hypothesis"])
                        if row["oracle_gap_m"] is not None:
                            assert row["oracle_gap_m"] >= -1e-7, "Same-bank F violates cached oracle bound"
                        row = clean(row)
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        rows.append(row)
                    ledger.update(status="BATCH_COMPLETE", completed_F=len(rows))
                    write(guard_path, ledger)
                    if start % 320 == 0:
                        print("QUICK_EVAL", method, split, len(rows), denominator, flush=True)
            assert [r["id"] for r in rows] == ids and attempted_F == denominator
            pending.replace(rows_path)
            execution = dict(new_refiner_batches=batches, new_refiner_examples=examples,
                new_backbone_calls=0, new_bank_generation=0, new_TRAIN_candidate_F=0,
                new_final_F_attempts=attempted_F, new_final_F_completed=sum(r["F_complete"] for r in rows),
                final_F_available=sum(r["F_available"] for r in rows), optimizer_updates=0,
                PnP_counts={k: sum(r["PnP_counts"][k] for r in rows) for k in PNP},
                seconds_wall=time.monotonic() - began, final_F_seconds=sum(r["F_seconds"] for r in rows),
                time_scope="Final evaluation loop wall including GPU/CPU/I/O; no deployment latency benchmark")
            result = dict(schema="quick_pose_loss_final_evaluation_v1", status="DONE", complete=True,
                method=method, seed=1, split=split, full_denominator=denominator,
                checkpoint_sha256=fit["checkpoint_sha256"], bank_binding=banks.binding, bank_artifact=bank_artifact,
                target_sha256=sha(target_path), code_sha256=sha(Path(__file__)),
                protocol_sha256=sha(DOC / "PROTOCOL.json"), setup_sha256=sha(DOC / "LOSS_SETUP.json"),
                binding=binding, GT_inference_access=False, readout="unchanged J",
                rowfile_sha256=sha(rows_path),
                rows_artifact=dict(path=str(rows_path), sha256=sha(rows_path), bytes=rows_path.stat().st_size),
                summary=E.summarize(rows), execution=execution,
                reference="Repeated-use DEV; REAL same2D/dimensions reconstruction is not independent physical metrology")
            write(receipt_path, result)
            ledger.update(status="DONE", execution=execution, rows_sha256=sha(rows_path), completed_F=denominator)
            write(guard_path, ledger)
            results[split] = result
        except Exception as error:
            ledger.update(status="RUN_FAILED", error=type(error).__name__, reason=str(error),
                          attempted_F=attempted_F, completed_F=len(rows), quiet_retry=0)
            write(guard_path, ledger)
            raise
    return results


def probe_row(frame, source_row, index, probability, costs, available, relative, scale):
    """Read costs after prediction; +inf truth and its finite proxy stay distinct."""
    active = len(probability)
    assert np.isfinite(probability).all() and np.all(probability >= 0)
    assert np.isclose(probability.sum(), 1., rtol=1e-6, atol=1e-7)
    assert np.isfinite(relative[:active]).all()
    c, valid = np.asarray(costs[:active], float), np.asarray(available[:active], bool)
    good = valid & np.isfinite(c)
    assert good.any()
    best = float(c[good].min())
    failed_mass = float(probability[~good].sum())
    expected_is_infinite = failed_mass > 0
    expected = None if expected_is_infinite else float(np.sum(probability[good] * c[good]))
    selected = float(c[index]) if good[index] else None
    entropy = float(-np.sum(probability[probability > 0] * np.log(probability[probability > 0])))
    return dict(id=frame["id"], session=frame["session"], source_cache_row=int(source_row),
        selected_index=int(index), action_count=active, NoOp=index == 0,
        oracle_index=int(np.flatnonzero(c == best)[0]), oracle_exact=index == int(np.flatnonzero(c == best)[0]),
        selected_ADDsym_m=selected, selected_F_available=bool(good[index]), oracle_ADDsym_m=best,
        oracle_gap_m=selected - best if selected is not None else None,
        expected_ADDsym_m=expected, expected_ADDsym_infinite=expected_is_infinite,
        failed_action_probability_mass=failed_mass,
        expected_finite_proxy_ADDsym_m=best + scale * float(np.sum(probability * relative[:active])),
        max_probability=float(probability.max()), entropy=entropy,
        metric_source="immutable TRAIN final-F cost table; no new pose inference")


@torch.no_grad()
def probe(method, source_root=ROOT, bank_cache=BANKS, cost_cache=COST, output_cache=OUTPUT):
    _, setup = check_paths(source_root, bank_cache, cost_cache, output_cache)
    fit = fit_receipt(method, output_cache)
    path = DOC / f"TRAIN_PROBE_{method}_seed1.json"
    guard_path = Path(output_cache) / "probes" / f"{method}_seed1_ATTEMPT.json"
    binding = hash_value(dict(method=method, checkpoint=fit["checkpoint_sha256"],
        protocol=sha(DOC / "PROTOCOL.json"), setup=sha(DOC / "LOSS_SETUP.json"), code=sha(Path(__file__))))
    if path.exists():
        result = read(path)
        assert result["status"] == "DONE" and result["complete"] and result["binding"] == binding
        assert len(result["rows"]) == 256
        guard = read(guard_path)
        assert guard["status"] == "DONE" and guard["result_sha256"] == sha(path)
        return result
    assert not guard_path.exists(), "Incomplete TRAIN probe preserved; no repeated forward"
    verify_derived(setup)
    selected_rows = np.load(setup["paths"]["calibration_rows"], mmap_mode="r")
    assert selected_rows.shape == (1024,) and len(set(selected_rows.tolist())) == 1024
    selected_rows = selected_rows[:256]
    numeric_contract()
    data = Data(source_root)
    assert np.isin(selected_rows, data.train_rows).all()
    frames = [data.source_frame(int(row)) for row in selected_rows]
    assert [f["id"] for f in frames] == setup["calibration"]["ids"][:256]
    banks = Banks(data, bank_cache, setup_report=setup)
    head = load_head(fit)
    cost = np.load(Path(cost_cache) / "cost_ADDsym_m.npy", mmap_mode="r")
    available = np.load(Path(cost_cache) / "F_available.npy", mmap_mode="r")
    relative = np.load(setup["paths"]["r_effective"], mmap_mode="r")
    rows, batches, examples = [], 0, 0
    began = time.monotonic()
    ledger = dict(status="ATTEMPT", binding=binding, completed_batches=0, completed_examples=0,
                  new_F_calls=0, new_PnP_calls=0, quiet_retry=0)
    write(guard_path, ledger)
    try:
        for start in range(0, 256, 16):
            group = frames[start:start + 16]
            prepared = [banks.get(frame["cache_row"], frame) for frame in group]
            ledger.update(status="FORWARD_ATTEMPT", attempted_batch=batches + 1)
            write(guard_path, ledger)
            packet, q, valid = E.prediction_packet(data, group, prepared, True)
            output = forward_bank(head, packet, q, valid)
            choices = E.select_native(output, prepared, group)
            probabilities = action_scores(output).softmax(-1).cpu().numpy()
            batches += 1
            examples += len(group)
            for frame, bank, (_, index, _) , p in zip(group, prepared, choices, probabilities):
                count = len(bank["points"])
                assert index == int(p.argmax()) and np.all(p[count:] == 0)
                source_row = frame["cache_row"]
                rows.append(probe_row(frame, source_row, index, p[:count].astype(float),
                    cost[source_row], available[source_row], relative[source_row], setup["scale_m"]))
            ledger.update(status="BATCH_COMPLETE", completed_batches=batches, completed_examples=examples)
            write(guard_path, ledger)
        assert len(rows) == examples == 256 and batches == 16
        gaps = [r["oracle_gap_m"] for r in rows if r["oracle_gap_m"] is not None]
        assert all(g >= -1e-7 for g in gaps)
        result = dict(schema="quick_pose_loss_fixed_TRAIN_probe_v1", status="DONE", complete=True,
            method=method, seed=1, binding=binding, checkpoint_sha256=fit["checkpoint_sha256"],
            code_sha256=sha(Path(__file__)), rows=rows,
            summary=dict(frames=256, NoOp=sum(r["NoOp"] for r in rows),
                oracle_exact=sum(r["oracle_exact"] for r in rows), selected_F_failures=sum(not r["selected_F_available"] for r in rows),
                mean_gap_m=float(np.mean(gaps)) if gaps else None,
                expected_infinite_frames=sum(r["expected_ADDsym_infinite"] for r in rows)),
            execution=dict(new_refiner_batches=16, new_refiner_examples=256, new_F_calls=0,
                PnP_counts={k: 0 for k in PNP}, new_backbone_calls=0, new_bank_generation=0,
                optimizer_updates=0, seconds_wall=time.monotonic() - began),
            GT_inference_access=False,
            scope="Final6000 fixed TRAIN probe: first256 of prespecified SHA-sorted1024 calibration TRAIN IDs; no validation/model/temperature selection",
            failure_proxy="Separate finite r_fail proxy; expected actual cached ADD is infinite when positive probability reaches an F-failed action")
        write(path, clean(result))
        ledger.update(status="DONE", result_sha256=sha(path))
        write(guard_path, ledger)
        return result
    except Exception as error:
        ledger.update(status="RUN_FAILED", error=type(error).__name__, reason=str(error),
                      completed_batches=batches, completed_examples=examples)
        write(guard_path, ledger)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", "--arm", dest="method", required=True, choices=METHODS)
    parser.add_argument("--stage", choices=("evaluate", "probe", "all"), default="all")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--bank-cache", type=Path, default=BANKS)
    parser.add_argument("--cost-cache", type=Path, default=COST)
    parser.add_argument("--output-cache", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    arguments = dict(source_root=args.source_root, bank_cache=args.bank_cache,
                     cost_cache=args.cost_cache, output_cache=args.output_cache)
    result = {}
    if args.stage in ("evaluate", "all"):
        result["evaluation"] = evaluate(args.method, **arguments)
    if args.stage in ("probe", "all"):
        result["probe"] = probe(args.method, **arguments)
    print(json.dumps(dict(method=args.method, stage=args.stage, status="DONE")))
    return result


if __name__ == "__main__":
    main()
